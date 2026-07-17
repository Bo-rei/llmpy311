#!/usr/bin/env python3
"""Frozen/CE/SupCon MiniLM 表征对照实验（v21）。

本模块只服务于“冻结 MiniLM 是否合理”这一表示层问题，不修改 v19/v20 的
检测器和结果。默认命令只做输入审计；只有显式 ``--execute`` 才会训练 CE 或
SupCon。这样在模型权重、文本划分或依赖缺失时，实验会留下可恢复的 manifest，
而不会产生看似完整但不可审计的数字。

固定边界协议：L2-normalized representation、per-cluster diagonal covariance、
mean+std radius、threshold=1。``--tune-threshold`` 只在 validation 选择阈值，
test 只做一次最终评价。smoke 结果带有 ``smoke=true``，不能直接进入论文主表。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from tools.experiments.cluster_separability.analysis import _json as _read_json, _load_cache
from tools.experiments.cluster_separability.protocol import compute_binary_oos_metrics
from tools.experiments.cluster_separability.v20_analysis import V19_ROOT

V21_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v21"
MODEL_ROOT = PROJECT_ROOT.parent / "assets" / "models" / "all-MiniLM-L6-v2"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
METHODS = ("frozen", "ce", "supcon")
K_VALUES = (1, 2)


def _safe(value: Any) -> Any:
    """把 numpy/Path/NaN 转成稳定 JSON 标量，避免 manifest 写入失败。"""
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(_safe(row) for row in rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit(dataset: str, kir: int, seed: int) -> Path:
    """返回 v19 固定 K=1 单元，作为数据/cache provenance 锚点。"""
    return V19_ROOT / "fixed" / dataset / f"kir{kir}_seed{seed}" / "euclidean" / "k1"


def _selected_k(dataset: str, kir: int, seed: int) -> int | None:
    """读取 v19 的 dataset-level selected K；缺失时返回 None，不猜测。"""
    path = V19_ROOT / "selected_k_summary.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    match = frame[(frame["dataset"] == dataset) & (frame["kir"] == kir) &
                  (frame["data_seed"] == seed) & (frame["distance"] == "mahalanobis_diag")]
    if len(match) != 1:
        return None
    value = int(match.iloc[0]["selected_k"])
    return value if value >= 1 else None


def _preflight_one(dataset: str, kir: int, seed: int, model_path: Path) -> dict[str, Any]:
    """不导入模型、不训练，只检查本次表征对照的所有输入。"""
    result: dict[str, Any] = {"dataset": dataset, "kir": kir, "data_seed": seed, "ready": False, "reasons": []}
    anchor = _unit(dataset, kir, seed)
    manifest_path = anchor / "run_manifest.json"
    if not manifest_path.is_file():
        result["reasons"].append(f"missing_v19_manifest:{manifest_path}")
        return result
    manifest = _read_json(manifest_path)
    data_root = Path(manifest.get("data_root", ""))
    result["data_root"] = str(data_root)
    if not data_root.is_dir():
        result["reasons"].append(f"missing_data_root:{data_root}")
    split_info: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_path = data_root / "gate" / f"{split}.json"
        cache_info = manifest.get("embedding_cache", {}).get(split, {})
        cache_path = V19_ROOT / "embedding_cache" / dataset / f"kir{kir}_seed{seed}" / f"{split}_{str(cache_info.get('cache_key', ''))[:16]}.npz"
        exists = split_path.is_file() and cache_path.is_file()
        split_info[split] = {"text_json": str(split_path), "embedding_cache": str(cache_path), "ready": exists}
        if not exists:
            result["reasons"].append(f"missing_{split}_input")
    required_model = [model_path / "config.json", model_path / "tokenizer.json"]
    # Sentence-transformer 的 model.safetensors 或 pytorch_model.bin 任一存在即可。
    weight_paths = [model_path / name for name in ("model.safetensors", "pytorch_model.bin") if (model_path / name).is_file()]
    if not weight_paths:
        required_model.append(model_path / "model.safetensors")
    for path in required_model:
        if not path.is_file():
            result["reasons"].append(f"missing_model_file:{path}")
    result["splits"] = split_info
    result["model_path"] = str(model_path)
    result["model_files"] = {p.name: _sha256(p) for p in [*required_model, *weight_paths] if p.is_file()}
    result["ready"] = not result["reasons"]
    return result


def build_preflight(output_root: Path, datasets: Sequence[str] = DATASETS, kir: int = 50, seed: int = 42,
                    model_path: Path = MODEL_ROOT) -> dict[str, Any]:
    """生成完整预检报告；此函数永远不启动训练。"""
    checks = [_preflight_one(dataset, kir, seed, model_path) for dataset in datasets]
    selected = {dataset: _selected_k(dataset, kir, seed) for dataset in datasets}
    payload = {
        "schema_version": 1,
        "experiment_family": "cluster_separability_v21_representation_adaptation",
        "v19_frozen": True,
        "protocol": {
            "representations": METHODS,
            "selected_k_from_v19": selected,
            "fixed_boundary": {"k": K_VALUES, "distance": "mahalanobis_diag", "l2_normalize": True,
                               "covariance_scope": "per_cluster", "radius_method": "mean_std", "radius_lambda": 1.0,
                               "threshold": 1.0},
            "selection_split": "validation",
            "oos_positive_class": True,
            "test_used_for_training_or_selection": False,
        },
        "checks": checks,
        "ready_count": sum(bool(x["ready"]) for x in checks),
        "requested_count": len(checks),
    }
    _write_json(output_root / "preflight.json", payload)
    return payload


def _load_inputs(dataset: str, kir: int, seed: int) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    anchor = _unit(dataset, kir, seed)
    manifest = _read_json(anchor / "run_manifest.json")
    data_root = Path(manifest["data_root"])
    rows = {split: _read_json(data_root / "gate" / f"{split}.json") for split in ("train", "val", "test")}
    embeddings = {split: _load_cache(V19_ROOT, dataset, kir, seed, split, manifest) for split in rows}
    return manifest, rows, embeddings


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _sample_rows(rows: Sequence[Mapping[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    """固定抽样，默认保留全部 Known train；smoke 用 limit 限制 CPU 训练成本。"""
    records = [dict(row) for row in rows]
    if limit is None or len(records) <= limit:
        return records
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=int(limit), replace=False))
    return [records[int(i)] for i in indices]


def _sample_rows_with_indices(rows: Sequence[Mapping[str, Any]], limit: int | None, seed: int,
                              stratify_intent: bool = False) -> tuple[list[dict[str, Any]], np.ndarray]:
    """返回行和原始位置，保证 Frozen 与适配表示使用完全相同的 smoke 子集。"""
    records = [dict(row) for row in rows]
    if limit is None or len(records) <= limit:
        indices = np.arange(len(records), dtype=np.int64)
    elif stratify_intent:
        # smoke 训练至少保留每个已知 intent 一个样本，避免“随机 128 条恰好
        # 没覆盖多数类别”把表示对照退化成缺类检测器。
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(records):
            groups.setdefault(str(row.get("intent", "")), []).append(index)
        if len(groups) > int(limit):
            raise ValueError("stratified smoke limit must cover all intents")
        rng = np.random.default_rng(seed)
        selected = [rng.choice(values) for _, values in sorted(groups.items())]
        remaining = np.setdiff1d(np.arange(len(records)), np.asarray(selected, dtype=np.int64), assume_unique=False)
        extra = int(limit) - len(selected)
        if extra:
            selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())
        indices = np.sort(np.asarray(selected, dtype=np.int64))
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(records), size=int(limit), replace=False)).astype(np.int64)
    return [records[int(i)] for i in indices], indices


def _mean_pool(hidden: Any, mask: Any) -> Any:
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def _supervised_contrastive_loss(features: Any, labels: Any, temperature: float = 0.07) -> Any:
    """标准 SupCon loss；没有正样本的 singleton 行会被安全跳过。"""
    import torch
    z = torch.nn.functional.normalize(features, dim=-1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    same = labels[:, None].eq(labels[None, :])
    valid = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = same & valid
    exp_logits = torch.exp(logits) * valid
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    count = positives.sum(dim=1)
    usable = count > 0
    if not torch.any(usable):
        return features.sum() * 0.0
    return -(log_prob * positives).sum(dim=1)[usable].div(count[usable]).mean()


def _encode_batch(model: Any, tokenizer: Any, texts: Sequence[str], device: Any, batch_size: int) -> np.ndarray:
    import torch
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(list(texts[start:start + batch_size]), padding=True, truncation=True,
                              max_length=256, return_tensors="pt").to(device)
            pooled = _mean_pool(model(**batch).last_hidden_state, batch["attention_mask"])
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, int(model.config.hidden_size)), dtype=np.float32)


def _train_representation(method: str, model_path: Path, train_rows: Sequence[Mapping[str, Any]], output_dir: Path,
                          seed: int, epochs: int, batch_size: int, max_train_samples: int | None) -> dict[str, Any]:
    """训练 CE/SupCon encoder；Frozen 不进入此函数。"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    _set_seed(seed)
    rows = _sample_rows(train_rows, max_train_samples, seed)
    intents = sorted({str(row["intent"]) for row in train_rows})
    label_map = {name: index for index, name in enumerate(intents)}
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    head = torch.nn.Linear(int(encoder.config.hidden_size), len(intents)).to(device) if method == "ce" else None
    parameters = list(encoder.parameters()) + (list(head.parameters()) if head is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=2e-5)
    encoder.train()
    losses: list[float] = []
    for _epoch in range(int(epochs)):
        order = np.random.default_rng(seed + _epoch).permutation(len(rows))
        for start in range(0, len(order), batch_size):
            selected = [rows[int(i)] for i in order[start:start + batch_size]]
            batch = tokenizer([str(row["text"]) for row in selected], padding=True, truncation=True,
                              max_length=256, return_tensors="pt").to(device)
            labels = torch.tensor([label_map[str(row["intent"])] for row in selected], dtype=torch.long, device=device)
            pooled = _mean_pool(encoder(**batch).last_hidden_state, batch["attention_mask"])
            loss = torch.nn.functional.cross_entropy(head(pooled), labels) if method == "ce" else _supervised_contrastive_loss(pooled, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "label_map": label_map}, output_dir / "encoder.pt")
    return {"method": method, "train_sample_count": len(rows), "epochs": epochs, "batch_size": batch_size,
            "loss_mean": float(np.mean(losses)) if losses else None, "device": str(device),
            "checkpoint": str(output_dir / "encoder.pt"), "label_count": len(intents)}


def _threshold_from_validation(labels: np.ndarray, scores: np.ndarray, guard: float = 0.80) -> tuple[float, dict[str, Any]]:
    """只用 validation 选择 operating point，避免 test threshold leakage。"""
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    choices: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = compute_binary_oos_metrics(labels, scores, float(threshold))
        if metrics["id_recall"] >= guard:
            choices.append((float(threshold), metrics))
    if not choices:
        return 1.0, {"selection_status": "no_candidate_meets_guard", "guard": guard}
    selected = max(choices, key=lambda item: (item[1]["oos_f1"], -item[1]["fpr95"], item[0]))
    return selected[0], {"selection_status": "selected", "guard": guard, "validation": selected[1], "candidate_count": len(choices)}


def _evaluate_representation(embeddings: dict[str, np.ndarray], rows: dict[str, list[dict[str, Any]]], method: str,
                             k: int, seed: int, tune_threshold: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    """在同一 representation 上执行固定 K 的 Gate；不改历史 detector。"""
    train_intents = np.asarray([str(row["intent"]) for row in rows["train"]], dtype=object)
    detector = MultiSphereOOSDetector(center_mode="class_centroid_mixture", subcenters_per_intent=k,
                                      radius_method="mean_std", radius_lambda=1.0,
                                      distance_metric="mahalanobis_diag", covariance_eps=1e-6,
                                      l2_normalize=True, random_state=42)
    detector.fit(embeddings["train"], train_intents)
    val_scores = detector.predict_with_scores(embeddings["val"])["score"].astype(float)
    test_scores = detector.predict_with_scores(embeddings["test"])["score"].astype(float)
    val_labels = np.asarray([int(row["label"]) for row in rows["val"]], dtype=np.int64)
    test_labels = np.asarray([int(row["label"]) for row in rows["test"]], dtype=np.int64)
    threshold, selection = _threshold_from_validation(val_labels, val_scores) if tune_threshold else (1.0, {"selection_status": "fixed", "threshold": 1.0})
    metrics = compute_binary_oos_metrics(test_labels, test_scores, threshold)
    metrics.update({"method": method, "k": k, "data_seed": seed, "threshold": threshold,
                    "validation_oos_f1": selection.get("validation", {}).get("oos_f1"),
                    "representation_dim": int(embeddings["train"].shape[1]), "test_count": len(test_labels)})
    records = pd.DataFrame({"true_binary_label": test_labels, "score": test_scores,
                            "prediction": (test_scores > threshold).astype(int),
                            "method": method, "k": k})
    return {**metrics, "threshold_selection": selection}, records


def run_smoke(output_root: Path = V21_ROOT, dataset: str = "clinc150", kir: int = 50, seed: int = 42,
              model_path: Path = MODEL_ROOT, epochs: int = 1, batch_size: int = 16,
              max_train_samples: int | None = 128, max_eval_samples: int | None = 256,
              tune_threshold: bool = False, smoke: bool = True) -> dict[str, Any]:
    """运行单个 dataset/seed 对照；正式运行使用完整 split 并单独标记。"""
    preflight = _preflight_one(dataset, kir, seed, model_path)
    phase = "smoke" if smoke else "formal"
    root = output_root / "representation_adaptation" / phase / dataset / f"kir{kir}_seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    if not preflight["ready"]:
        payload = {"status": "blocked_missing_inputs", "preflight": preflight, "smoke": smoke}
        _write_json(root / "run_manifest.json", payload)
        return payload
    manifest, rows, frozen_embeddings = _load_inputs(dataset, kir, seed)
    selected_k = _selected_k(dataset, kir, seed)
    run_k_values = tuple(sorted(set(K_VALUES + ((selected_k,) if selected_k else ()))))
    run_rows: list[dict[str, Any]] = []
    smoke_rows: dict[str, list[dict[str, Any]]] = {}
    smoke_indices: dict[str, np.ndarray] = {}
    for index, split in enumerate(("train", "val", "test")):
        limit = max_train_samples if split == "train" else max_eval_samples
        smoke_rows[split], smoke_indices[split] = _sample_rows_with_indices(
            rows[split], limit, seed + index, stratify_intent=(split == "train")
        )

    for method in METHODS:
        method_root = root / method
        eval_rows = smoke_rows
        if method == "frozen":
            embeddings = {split: frozen_embeddings[split][smoke_indices[split]] for split in smoke_rows}
            training = {"status": "reused_v19_embedding_cache", "cache_source": str(V19_ROOT / "embedding_cache")}
        else:
            training = _train_representation(method, model_path, rows["train"], method_root / "checkpoint", seed,
                                             epochs, batch_size, max_train_samples)
            import torch
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
            state = torch.load(method_root / "checkpoint" / "encoder.pt", map_location="cpu", weights_only=True)
            encoder.load_state_dict(state["encoder"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder.to(device)
            # Gate 训练/评价必须对应同一批 smoke 行；这样 Frozen 与两个适配方法
            # 的差异只来自 representation，而不是抽样数量。
            embeddings = {split: _encode_batch(encoder, tokenizer, [str(row["text"]) for row in smoke_rows[split]], device, batch_size)
                          for split in smoke_rows}
        for k in run_k_values:
            metrics, scores = _evaluate_representation(embeddings, eval_rows, method, k, seed, tune_threshold)
            unit = method_root / f"k{k}"
            unit.mkdir(parents=True, exist_ok=True)
            scores.to_parquet(unit / "scores.parquet", index=False)
            _write_json(unit / "eval_results.json", metrics)
            _write_json(unit / "threshold_selection.json", metrics.pop("threshold_selection"))
            run_rows.append(metrics)
        _write_json(method_root / "training_manifest.json", training)
    _write_csv(root / "representation_adaptation_by_k.csv", run_rows)
    payload = {"status": "completed", "smoke": smoke, "dataset": dataset, "kir": kir, "data_seed": seed,
               "methods": METHODS, "k": run_k_values, "selected_k": selected_k, "rows": len(run_rows), "output_root": str(root),
               "v19_frozen": True, "training_limits": {"epochs": epochs, "max_train_samples": max_train_samples,
                                                          "max_eval_samples": max_eval_samples},
               "preflight": preflight, "python": platform.python_version(),
               "protocol": {"distance": "mahalanobis_diag", "l2_normalize": True,
                            "covariance_scope": "per_cluster", "radius_method": "mean_std",
                            "radius_lambda": 1.0, "fixed_threshold": 1.0,
                            "validation_tuning": "optional_threshold_only"}}
    _write_json(root / "run_manifest.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v21 Frozen/CE/SupCon MiniLM adaptation audit")
    parser.add_argument("--output-root", type=Path, default=V21_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_ROOT)
    parser.add_argument("--dataset", choices=DATASETS, default="clinc150")
    parser.add_argument("--kir", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight", action="store_true", help="只审计输入，不加载模型、不训练")
    parser.add_argument("--execute", action="store_true", help="显式启动训练")
    parser.add_argument("--formal", action="store_true", help="使用完整 train/val/test，运行正式单元")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.execute or args.preflight:
        payload = build_preflight(args.output_root, (args.dataset,), args.kir, args.seed, args.model_path)
    else:
        if args.formal:
            payload = run_smoke(args.output_root, args.dataset, args.kir, args.seed, args.model_path,
                                args.epochs, args.batch_size, None, None, smoke=False)
        else:
            payload = run_smoke(args.output_root, args.dataset, args.kir, args.seed, args.model_path,
                                args.epochs, args.batch_size, args.max_train_samples, args.max_eval_samples,
                                smoke=True)
    print(json.dumps(_safe(payload), ensure_ascii=False, indent=2))
    return 0 if payload.get("status", "completed") != "blocked_missing_inputs" else 2


if __name__ == "__main__":
    raise SystemExit(main())
