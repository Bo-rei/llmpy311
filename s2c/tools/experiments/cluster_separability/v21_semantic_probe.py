#!/usr/bin/env python3
"""冻结 MiniLM 语义空间的第一阶段诊断。

本脚本只读取 v19 已冻结的 gate split、embedding cache 和 Gate score，不重新编码
文本，也不修改 v19/v20 结果。它把论文第 3.2 节的两个前提拆成可审计指标：

* 局部邻域是否与 Known intent 标签一致（Purity、类内/类间距离、margin）；
* 近 OOS 的错误来自表示碰撞，还是来自局部边界过覆盖。

中间层表示不在 v19 cache 中，因此运行时会额外写出 ``layer_preflight.json``，
明确记录阻断原因；不得用最终句向量冒充 layer2/4/6 表示。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.cluster_separability.analysis import (  # noqa: E402
    _centroids,
    _l2,
    _load_cache,
    semantic_distance,
    validation_bucket_thresholds,
)
from tools.experiments.cluster_separability.protocol import compute_binary_oos_metrics  # noqa: E402

V19_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v19"
V21_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v21"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
SEEDS = (13, 42, 87)
KIRS = (50,)
PURITY_KS = (5, 10, 20)
PAIR_SAMPLE_COUNT = 100_000


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(_json_safe(row) for row in rows)


def _unit(root: Path, phase: str, dataset: str, seed: int, k: int) -> Path:
    return root / phase / dataset / f"kir50_seed{seed}" / "mahalanobis_diag" / f"k{k}"


def _load_split(unit: Path, v19_root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    manifest = _json(unit / "run_manifest.json")
    data_root = Path(manifest["data_root"])
    rows = {split: _json(data_root / "gate" / f"{split}.json") for split in ("train", "val", "test")}
    embeddings = {
        split: _load_cache(v19_root, manifest["config"]["dataset"], 50, int(manifest["config"]["data_seed"]), split, manifest)
        for split in rows
    }
    for split in rows:
        if len(rows[split]) != len(embeddings[split]):
            raise ValueError(f"{split} rows/cache mismatch for {unit}: {len(rows[split])} != {len(embeddings[split])}")
    return manifest, rows, embeddings


def _known_mask(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label"]) == 0 for row in rows], dtype=bool)


def _intent_labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([str(row["intent"]) for row in rows], dtype=object)


def _cosine_distance_matrix(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return 1.0 - _l2(query) @ _l2(reference).T


def _purity_rows(
    train_rows: Sequence[Mapping[str, Any]],
    train_x: np.ndarray,
    query_rows: Sequence[Mapping[str, Any]],
    query_x: np.ndarray,
    *,
    split: str,
    dataset: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """计算 leave-one-out train purity，及 val/test Known 对 train 的邻域纯度。"""

    train_labels = _intent_labels(train_rows)
    query_labels = _intent_labels(query_rows)
    train_known = _known_mask(train_rows)
    query_known = _known_mask(query_rows)
    reference_x = _l2(train_x[train_known])
    reference_labels = train_labels[train_known]
    if split == "train":
        # 训练样本的最近邻必须排除自身，否则 Purity@k 会被恒等的 self-match 虚高。
        distances = _cosine_distance_matrix(query_x[query_known], reference_x)
        distances[np.arange(len(distances)), np.arange(len(distances))] = np.inf
        query_labels_known = query_labels[query_known]
    else:
        distances = _cosine_distance_matrix(query_x[query_known], reference_x)
        query_labels_known = query_labels[query_known]
    order = np.argsort(distances, axis=1)[:, : max(PURITY_KS)]
    nearest = reference_labels[order]
    per_intent_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for k in PURITY_KS:
        purity = np.mean(nearest[:, :k] == query_labels_known[:, None], axis=1)
        for intent in sorted(set(query_labels_known)):
            values = purity[query_labels_known == intent]
            per_intent_rows.append({
                "dataset": dataset, "kir": 50, "data_seed": seed, "split": split,
                "k": k, "intent": intent, "sample_count": len(values),
                "purity_mean": float(np.mean(values)),
            })
        intent_means = np.asarray([
            float(np.mean(purity[query_labels_known == intent])) for intent in sorted(set(query_labels_known))
        ])
        summary_rows.append({
            "dataset": dataset, "kir": 50, "data_seed": seed, "split": split,
            "k": k, "sample_count": len(purity), "macro_purity": float(np.mean(purity)),
            "bottom10_intent_purity": float(np.quantile(intent_means, 0.10)),
            "intent_count": len(intent_means),
        })
    # 只导出 Known query 的跨意图邻居计数，避免 OOS 标签污染“意图邻域”诊断。
    confusion = Counter()
    top_k = nearest[:, :10]
    for true_label, neighbors in zip(query_labels_known, top_k):
        for neighbor in neighbors:
            if neighbor != true_label:
                confusion[(str(true_label), str(neighbor))] += 1
    return per_intent_rows, summary_rows, [
        {"dataset": dataset, "kir": 50, "data_seed": seed, "split": split,
         "true_intent": a, "neighbor_intent": b, "neighbor_count": count}
        for (a, b), count in sorted(confusion.items())
    ]


def _distance_geometry(
    rows: Sequence[Mapping[str, Any]], x: np.ndarray, *, dataset: str, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """用固定随机种子抽样 same/different intent pair，避免构造超大距离矩阵。"""

    labels = _intent_labels(rows)
    known = _known_mask(rows)
    labels = labels[known]
    x = _l2(x[known])
    rng = np.random.default_rng(seed + 1009)
    intents = sorted(set(labels))
    intra_values: list[float] = []
    per_intent: list[dict[str, Any]] = []
    for intent in intents:
        indices = np.flatnonzero(labels == intent)
        if len(indices) < 2:
            continue
        count = min(PAIR_SAMPLE_COUNT // max(len(intents), 1), len(indices) * (len(indices) - 1) // 2)
        left = rng.choice(indices, size=max(count, 1), replace=True)
        right = rng.choice(indices, size=max(count, 1), replace=True)
        valid = left != right
        if not np.any(valid):
            continue
        values = 1.0 - np.sum(x[left[valid]] * x[right[valid]], axis=1)
        intra_values.extend(values.tolist())
        per_intent.append({
            "dataset": dataset, "kir": 50, "data_seed": seed, "intent": intent,
            "intra_distance": float(np.mean(values)), "pair_count": int(len(values)),
        })
    left = rng.integers(0, len(labels), size=PAIR_SAMPLE_COUNT)
    right = rng.integers(0, len(labels), size=PAIR_SAMPLE_COUNT)
    valid = labels[left] != labels[right]
    inter_values = 1.0 - np.sum(x[left[valid]] * x[right[valid]], axis=1)
    intra = float(np.mean(intra_values)) if intra_values else math.nan
    inter = float(np.mean(inter_values)) if len(inter_values) else math.nan
    summary = {
        "dataset": dataset, "kir": 50, "data_seed": seed,
        "known_train_count": len(labels), "intra_pair_count": len(intra_values),
        "inter_pair_count": int(len(inter_values)), "intra_distance": intra,
        "inter_distance": inter,
        "relative_separation": (inter - intra) / inter if math.isfinite(intra) and inter else math.nan,
    }
    return summary, per_intent


def _intent_centers(rows: Sequence[Mapping[str, Any]], x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels = _intent_labels(rows)
    known = _known_mask(rows)
    intents = sorted(set(labels[known]))
    centers = np.stack([np.mean(_l2(x[known][labels[known] == intent]), axis=0) for intent in intents])
    return _l2(centers), intents


def _margin_rows(
    rows: Sequence[Mapping[str, Any]], x: np.ndarray, train_rows: Sequence[Mapping[str, Any]], train_x: np.ndarray,
    *, split: str, dataset: str, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    centers, intents = _intent_centers(train_rows, train_x)
    labels = _intent_labels(rows)
    known = _known_mask(rows)
    distances = 1.0 - _l2(x) @ centers.T
    order = np.argsort(distances, axis=1)
    nearest = order[:, 0]
    second = order[:, 1] if len(intents) > 1 else order[:, 0]
    margin = distances[np.arange(len(x)), second] - distances[np.arange(len(x)), nearest]
    known_rows = []
    for idx in np.flatnonzero(known):
        true_index = intents.index(str(labels[idx]))
        true_dist = distances[idx, true_index]
        other = np.delete(distances[idx], true_index)
        known_rows.append({
            "dataset": dataset, "kir": 50, "data_seed": seed, "split": split,
            "intent": str(labels[idx]), "sample_index": int(idx),
            "true_center_distance": float(true_dist),
            "second_center_distance": float(np.min(other)) if len(other) else math.nan,
            "true_intent_margin": float(np.min(other) - true_dist) if len(other) else math.nan,
            "nearest_intent": intents[int(nearest[idx])],
            "nearest_correct": int(intents[int(nearest[idx])] == str(labels[idx])),
        })
    summary = {
        "dataset": dataset, "kir": 50, "data_seed": seed, "split": split,
        "known_sample_count": len(known_rows),
        "nearest_center_accuracy": float(np.mean([r["nearest_correct"] for r in known_rows])) if known_rows else math.nan,
        "margin_mean": float(np.mean([r["true_intent_margin"] for r in known_rows])) if known_rows else math.nan,
        "margin_median": float(np.median([r["true_intent_margin"] for r in known_rows])) if known_rows else math.nan,
        "margin_bottom10": float(np.quantile([r["true_intent_margin"] for r in known_rows], 0.10)) if known_rows else math.nan,
    }
    return summary, known_rows


def _select_threshold(val_labels: np.ndarray, val_scores: np.ndarray) -> float:
    """仅用 validation 选置信度阈值，保留至少 80% Known recall。"""

    candidates = np.unique(np.quantile(val_scores, np.linspace(0.01, 0.99, 99)))
    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        metrics = compute_binary_oos_metrics(val_labels, val_scores, float(threshold))
        if metrics["id_recall"] >= 0.80:
            scored.append((metrics["oos_f1"], -metrics["fpr95"] if math.isfinite(metrics["fpr95"]) else -1.0, float(threshold)))
    if scored:
        return max(scored)[2]
    return float(np.quantile(val_scores[val_labels == 0], 0.95))


def _probe_rows(
    train_rows: Sequence[Mapping[str, Any]], train_x: np.ndarray,
    val_rows: Sequence[Mapping[str, Any]], val_x: np.ndarray,
    test_rows: Sequence[Mapping[str, Any]], test_x: np.ndarray,
    *, dataset: str, seed: int,
) -> list[dict[str, Any]]:
    train_known = _known_mask(train_rows)
    y_train = _intent_labels(train_rows)[train_known]
    x_train = _l2(train_x[train_known])
    probes: list[tuple[str, Any]] = [
        ("linear_probe", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)),
        ("knn", KNeighborsClassifier(n_neighbors=10, weights="distance", metric="cosine", algorithm="brute")),
    ]
    rows: list[dict[str, Any]] = []
    val_label = np.asarray([int(row["label"]) for row in val_rows], dtype=np.int64)
    test_label = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    for name, model in probes:
        model.fit(x_train, y_train)
        val_pred = model.predict(_l2(val_x))
        test_pred = model.predict(_l2(test_x))
        val_proba = model.predict_proba(_l2(val_x))
        test_proba = model.predict_proba(_l2(test_x))
        val_score = 1.0 - np.max(val_proba, axis=1)
        test_score = 1.0 - np.max(test_proba, axis=1)
        threshold = _select_threshold(val_label, val_score)
        for split, labels, pred, scores in (("val", val_label, val_pred, val_score), ("test", test_label, test_pred, test_score)):
            known = labels == 0
            oos_metrics = compute_binary_oos_metrics(labels, scores, threshold)
            rows.append({
                "dataset": dataset, "kir": 50, "data_seed": seed, "probe": name,
                "split": split, "known_count": int(np.sum(known)),
                "known_macro_f1": float(f1_score(_intent_labels(val_rows if split == "val" else test_rows)[known], pred[known], average="macro")) if np.any(known) else math.nan,
                "known_accuracy": float(accuracy_score(_intent_labels(val_rows if split == "val" else test_rows)[known], pred[known])) if np.any(known) else math.nan,
                "threshold_from_val": threshold, **oos_metrics,
            })
    return rows


def _semantic_boundary_rows(
    dataset: str, seed: int, k: int, unit: Path, train_rows: Sequence[Mapping[str, Any]], train_x: np.ndarray,
    val_rows: Sequence[Mapping[str, Any]], val_x: np.ndarray, test_rows: Sequence[Mapping[str, Any]], test_x: np.ndarray,
) -> list[dict[str, Any]]:
    scores = pd.read_parquet(unit / "scores.parquet")
    if len(scores) != len(test_rows) or not np.array_equal(scores["true_binary_label"].to_numpy(), np.asarray([int(r["label"]) for r in test_rows])):
        raise ValueError(f"score/test alignment mismatch for {unit}")
    centers, intents = _intent_centers(train_rows, train_x)
    val_labels = np.asarray([int(r["label"]) for r in val_rows])
    test_labels = np.asarray([int(r["label"]) for r in test_rows])
    val_dist = semantic_distance(val_x, centers)
    test_dist = semantic_distance(test_x, centers)
    q20, q80 = validation_bucket_thresholds(val_labels, val_dist)
    buckets = np.where(test_dist <= q20, "near", np.where(test_dist <= q80, "medium", "far"))
    # t_c 只由 Known validation 到其真实单中心的 cosine similarity 得到。
    train_labels = _intent_labels(train_rows)
    val_intents = _intent_labels(val_rows)
    thresholds: dict[str, float] = {}
    val_cos = _l2(val_x) @ centers.T
    for index, intent in enumerate(intents):
        values = val_cos[(val_labels == 0) & (val_intents == intent), index]
        thresholds[intent] = float(np.quantile(values, 0.05)) if len(values) else math.nan
    test_cos = _l2(test_x) @ centers.T
    nearest = np.argmax(test_cos, axis=1)
    nearest_intent = np.asarray([intents[i] for i in nearest], dtype=object)
    support_threshold = np.asarray([thresholds[str(intent)] for intent in nearest_intent])
    collision = (test_labels == 1) & (test_cos[np.arange(len(test_x)), nearest] >= support_threshold)
    gate_accept = scores["score"].to_numpy(dtype=float) <= 1.0
    false_accept = (test_labels == 1) & gate_accept
    boundary_overcoverage = (test_labels == 1) & ~collision & gate_accept
    rows: list[dict[str, Any]] = []
    for bucket in ("near", "medium", "far", "all"):
        mask = (test_labels == 1) if bucket == "all" else ((test_labels == 1) & (buckets == bucket))
        if not np.any(mask):
            continue
        rows.append({
            "dataset": dataset, "kir": 50, "data_seed": seed, "k_gate": k,
            "bucket": bucket, "oos_count": int(np.sum(mask)),
            "representation_collision_rate": float(np.mean(collision[mask])),
            "boundary_overcoverage_rate": float(np.mean(boundary_overcoverage[mask])),
            "false_accept_rate": float(np.mean(false_accept[mask])),
            "false_accept_count": int(np.sum(false_accept & mask)),
            "false_accept_representation_collision_rate": float(np.mean(collision[false_accept & mask])) if np.any(false_accept & mask) else math.nan,
            "false_accept_boundary_overcoverage_rate": float(np.mean(boundary_overcoverage[false_accept & mask])) if np.any(false_accept & mask) else math.nan,
            "correct_rejection_rate": float(np.sum((test_labels == 1) & ~gate_accept & mask) / np.sum(mask)),
            "mean_nearest_cosine": float(np.mean(test_cos[np.arange(len(test_x)), nearest][mask])),
            "validation_q20": q20, "validation_q80": q80,
        })
    return rows


def _effective_rank(x: np.ndarray) -> float:
    centered = x - np.mean(x, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    weights = singular**2
    weights = weights / np.maximum(np.sum(weights), 1e-12)
    return float(np.exp(-np.sum(weights * np.log(np.maximum(weights, 1e-12)))))


def run_semantic_probe(
    *, v19_root: Path = V19_ROOT, output_root: Path = V21_ROOT,
    datasets: Sequence[str] = DATASETS, seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """运行 3 个数据集、KIR=50、3 个 seed 的不训练语义诊断。"""

    output_root.mkdir(parents=True, exist_ok=True)
    purity_intent: list[dict[str, Any]] = []
    purity_summary: list[dict[str, Any]] = []
    purity_confusion: list[dict[str, Any]] = []
    geometry_summary: list[dict[str, Any]] = []
    geometry_intent: list[dict[str, Any]] = []
    margin_summary: list[dict[str, Any]] = []
    margin_detail: list[dict[str, Any]] = []
    probe_metrics: list[dict[str, Any]] = []
    boundary_decomposition: list[dict[str, Any]] = []
    layer_entries: list[dict[str, Any]] = []
    completed = 0
    for dataset in datasets:
        for seed in seeds:
            k1 = _unit(v19_root, "fixed", dataset, seed, 1)
            if not (k1 / "run_manifest.json").is_file():
                layer_entries.append({"dataset": dataset, "data_seed": seed, "status": "blocked_missing_v19_unit", "unit": str(k1)})
                continue
            manifest, rows, embeddings = _load_split(k1, v19_root)
            for split in ("train", "val", "test"):
                intent, summary, confusion = _purity_rows(rows["train"], embeddings["train"], rows[split], embeddings[split], split=split, dataset=dataset, seed=seed)
                purity_intent.extend(intent); purity_summary.extend(summary); purity_confusion.extend(confusion)
            geometry, geometry_by_intent = _distance_geometry(rows["train"], embeddings["train"], dataset=dataset, seed=seed)
            geometry_summary.append(geometry); geometry_intent.extend(geometry_by_intent)
            for split in ("val", "test"):
                summary, detail = _margin_rows(rows[split], embeddings[split], rows["train"], embeddings["train"], split=split, dataset=dataset, seed=seed)
                margin_summary.append(summary); margin_detail.extend(detail)
            probe_metrics.extend(_probe_rows(rows["train"], embeddings["train"], rows["val"], embeddings["val"], rows["test"], embeddings["test"], dataset=dataset, seed=seed))
            selected_path = V19_ROOT / "selected_k_summary.csv"
            selected_frame = pd.read_csv(selected_path)
            selected = selected_frame[(selected_frame.dataset == dataset) & (selected_frame.kir == 50) & (selected_frame.data_seed == seed) & (selected_frame.distance == "mahalanobis_diag")]
            ks = [1, 2]
            if len(selected) == 1:
                selected_k = int(selected.iloc[0].selected_k)
                if selected_k not in ks:
                    ks.append(selected_k)
            for k in ks:
                phase = "fixed" if k in (1, 2) else "tuned"
                unit = _unit(v19_root, phase, dataset, seed, k)
                if (unit / "scores.parquet").is_file():
                    boundary_decomposition.extend(_semantic_boundary_rows(dataset, seed, k, unit, rows["train"], embeddings["train"], rows["val"], embeddings["val"], rows["test"], embeddings["test"]))
            layer_entries.append({
                "dataset": dataset, "kir": 50, "data_seed": seed,
                "status": "blocked_intermediate_layers_not_cached",
                "final_embedding_shape": list(embeddings["train"].shape),
                "required_layers": [2, 4, 6],
                "reason": "v19 cache stores only final 384-d SentenceTransformer embeddings; no token-level hidden states or layer cache is available",
            })
            completed += 1
    _write_csv(output_root / "minilm_intent_neighborhood_purity.csv", purity_summary)
    _write_csv(output_root / "minilm_intent_neighborhood_purity_by_intent.csv", purity_intent)
    _write_csv(output_root / "minilm_cross_intent_neighbor_confusion.csv", purity_confusion)
    _write_csv(output_root / "minilm_intent_geometry.csv", geometry_summary)
    _write_csv(output_root / "minilm_intent_geometry_by_intent.csv", geometry_intent)
    _write_csv(output_root / "minilm_intent_margin.csv", margin_summary)
    _write_csv(output_root / "minilm_intent_margin_by_sample.csv", margin_detail)
    _write_csv(output_root / "minilm_probe_metrics.csv", probe_metrics)
    _write_csv(output_root / "near_oos_representation_boundary_decomposition.csv", boundary_decomposition)
    _write_json(output_root / "layer_preflight.json", {
        "status": "blocked",
        "required_representations": ["layer2", "layer4", "layer6", "final_sentence_embedding"],
        "final_sentence_embedding": "available",
        "intermediate_layers": "blocked",
        "entries": layer_entries,
        "instruction": "重新编码时必须从 MiniLM Transformer hidden states 提取层级表示；禁止用 final embedding 代替中间层。",
    })
    _write_json(output_root / "v21_semantic_probe_manifest.json", {
        "experiment": "v21_minilm_semantic_probe",
        "v19_root": str(v19_root.resolve()), "output_root": str(output_root.resolve()),
        "datasets": list(datasets), "kir": 50, "seeds": list(seeds),
        "completed_units": completed, "expected_units": len(datasets) * len(seeds),
        "embedding_source": "v19 embedding_cache final SentenceTransformer embedding",
        "intermediate_layer_status": "blocked",
        "historical_artifacts_overwritten": False,
    })
    return {"completed_units": completed, "expected_units": len(datasets) * len(seeds), "probe_rows": len(probe_metrics), "boundary_rows": len(boundary_decomposition)}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v21 frozen MiniLM semantic probes")
    parser.add_argument("--v19-root", type=Path, default=V19_ROOT)
    parser.add_argument("--output-root", type=Path, default=V21_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    print(json.dumps(run_semantic_probe(v19_root=args.v19_root, output_root=args.output_root, datasets=args.datasets, seeds=args.seeds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
