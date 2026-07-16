#!/usr/bin/env python3
"""运行受控的 MiniLM Gate-only OOS Baseline。

MSP、Energy 和 Entropy 共用同一个“冻结 MiniLM embedding + 线性已知意图分类头”，
它们只更换 OOS score，避免将分类器差异误当作 scoring 方法差异。kNN 和 LOF
使用同一份 L2 MiniLM 表示。所有超参和阈值只用 validation，test 在选择完成后
评估一次。该脚本只评价 Known/OOS Gate，不产生 Known macro-F1 或端到端 accuracy。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from .protocol import compute_binary_oos_metrics
from .runner import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    PATHS,
    _dataset_root,
    _fixed_k1_guard,
    _git_revision,
    _json_load,
    _load_or_encode,
    _model_fingerprint,
    _oos_source,
    _sample_id,
    _sha256_file,
    _sha256_payload,
    _write_json,
)

METHODS = ("msp", "energy", "entropy", "knn", "lof")
GRID_DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
GRID_KIRS = (25, 50, 75)
GRID_SEEDS = (13, 42, 87)
C_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
KNN_NEIGHBORS = (5, 10, 20)
LOF_NEIGHBORS = (10, 20, 50)
REQUIRED_OUTPUTS = (
    "eval_results.json",
    "run_manifest.json",
    "threshold_selection.json",
    "scores.parquet",
    "timing.json",
)
REQUIRED_SCORE_COLUMNS = {
    "sample_id",
    "true_binary_label",
    "true_intent",
    "oos_source",
    "score",
    "prediction",
    "nearest_intent",
    "nearest_cluster",
    "distance",
    "radius",
    "coverage_count",
}
REQUIRED_METRICS = {
    "oos_precision",
    "oos_recall",
    "oos_f1",
    "id_recall",
    "oos_rejection",
    "auroc",
    "aupr_oos",
    "fpr95",
}


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _linear_oos_scores(
    classifier: LogisticRegression,
    embeddings: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """从同一线性分类头生成不同 OOS score，并统一为“越大越 OOS”。

    ``decision_function`` 在二分类时只返回一列 margin，因此显式还原为对称的
    两列 logits，使 Energy 的定义不依赖 sklearn 的返回形状。
    """

    probabilities = classifier.predict_proba(embeddings)
    logits = np.asarray(classifier.decision_function(embeddings), dtype=np.float64)
    if logits.ndim == 1:
        logits = np.column_stack((-0.5 * logits, 0.5 * logits))
    if method == "msp":
        scores = 1.0 - np.max(probabilities, axis=1)
    elif method == "entropy":
        clipped = np.clip(probabilities, 1e-12, 1.0)
        scores = -np.sum(clipped * np.log(clipped), axis=1) / math.log(probabilities.shape[1])
    elif method == "energy":
        # 不对 validation/test 分别做 min-max 归一化。该操作会使分数依赖
        # 当前 split 的样本构成，导致 validation 上选出的阈值无法直接用于 test。
        scores = -logsumexp(logits, axis=1)
    else:
        raise ValueError(f"unsupported linear baseline: {method}")
    predicted = classifier.classes_[np.argmax(probabilities, axis=1)]
    return np.asarray(scores, dtype=np.float64), np.asarray(predicted, dtype=np.int64)


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    if unique.size == 0:
        raise ValueError("cannot select a threshold from empty scores")
    return np.concatenate(([np.nextafter(unique[0], -np.inf)], unique))


def select_operating_point(
    labels: np.ndarray,
    scores: np.ndarray,
    id_recall_guard: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """在 validation 上选运行点，并完整保存所有阈值候选以便审计。

    满足 ID-recall guard 时优先 OOS F1；若一个阈值都不满足，则优先恢复
    ID recall，并通过 ``guard_violation`` 显式暴露失败，不为了得到数字而放宽协议。
    """

    candidates = [
        {
            "threshold": float(threshold),
            "metrics": compute_binary_oos_metrics(labels, scores, float(threshold)),
        }
        for threshold in _threshold_candidates(scores)
    ]
    eligible = [row for row in candidates if row["metrics"]["id_recall"] >= id_recall_guard]
    pool = eligible or candidates

    def rank(row: Mapping[str, Any]) -> tuple[float, ...]:
        metrics = row["metrics"]
        fpr95 = metrics["fpr95"]
        inverse_fpr95 = -float(fpr95) if math.isfinite(float(fpr95)) else -math.inf
        if eligible:
            return (
                float(metrics["oos_f1"]),
                inverse_fpr95,
                float(metrics["id_recall"]),
                float(row["threshold"]),
            )
        return (
            float(metrics["id_recall"]),
            float(metrics["oos_f1"]),
            inverse_fpr95,
            float(row["threshold"]),
        )

    selected = dict(max(pool, key=rank))
    selected["guard_violation"] = not bool(eligible)
    return selected, candidates


def _fit_linear_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_known_x: np.ndarray,
    val_known_y: np.ndarray,
) -> tuple[LogisticRegression, dict[str, Any]]:
    """只用 Known train 拟合分类头，用 Known validation macro-F1 选正则强度。

    OOS validation 不参与分类头 C 的选择，它只用于之后的 OOS 阈值选择。
    这样把“Known 分类头质量”和“OOS operating point”两层超参分开。
    """

    candidates: list[dict[str, Any]] = []
    fitted: dict[float, LogisticRegression] = {}
    for c_value in C_CANDIDATES:
        classifier = LogisticRegression(
            C=float(c_value),
            class_weight=None,
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )
        classifier.fit(train_x, train_y)
        predicted = classifier.predict(val_known_x)
        score = float(f1_score(val_known_y, predicted, average="macro", zero_division=0))
        fitted[float(c_value)] = classifier
        candidates.append({"C": float(c_value), "known_validation_macro_f1": score})
    selected = max(candidates, key=lambda row: (row["known_validation_macro_f1"], -row["C"]))
    return fitted[selected["C"]], {"selected": selected, "candidates": candidates}


def _score_neighbor_method(
    method: str,
    neighbor_count: int,
    train_x: np.ndarray,
    query_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回高值为 OOS 的距离分数，以及可解释的最近训练样本索引。

    LOF 的 ``score_samples`` 越大越像训练分布，所以这里取负号后再交给统一指标函数。
    K 大于训练样本数时会安全截断，但 manifest 仍保留候选设置以便复现。
    """

    effective = min(int(neighbor_count), max(1, train_x.shape[0] - 1))
    nearest = NearestNeighbors(n_neighbors=effective, metric="euclidean", n_jobs=1)
    nearest.fit(train_x)
    if method == "knn":
        distances, indices = nearest.kneighbors(query_x)
        return np.mean(distances, axis=1), indices[:, 0]
    if method != "lof":
        raise ValueError(f"unsupported neighbor baseline: {method}")
    detector = LocalOutlierFactor(n_neighbors=effective, novelty=True, n_jobs=1)
    detector.fit(train_x)
    _, indices = nearest.kneighbors(query_x, n_neighbors=1)
    return -detector.score_samples(query_x), indices[:, 0]


def _score_frame(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    threshold: float,
    nearest_intents: Sequence[str | None],
    known_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    records = []
    for index, (row, score, nearest_intent) in enumerate(zip(rows, scores, nearest_intents)):
        records.append(
            {
                "sample_id": _sample_id(row, "test", index),
                "true_binary_label": int(row["label"]),
                "true_intent": str(row["intent"]),
                "oos_source": _oos_source(row, known_manifest),
                "source_split": str(row.get("source_split", row.get("split", "test"))),
                "score": float(score),
                "prediction": int(float(score) > float(threshold)),
                "nearest_intent": nearest_intent,
                "nearest_cluster": None,
                "distance": float(score),
                "radius": float(threshold),
                "coverage_count": None,
            }
        )
    return pd.DataFrame.from_records(records)


def _unit_dir(output_root: Path, dataset: str, kir: int, seed: int, method: str) -> Path:
    return output_root / dataset / f"kir{kir}_seed{seed}" / method


def _required_inputs(data_root: Path) -> list[Path]:
    return [
        data_root / "KNOWN_INTENTS.json",
        data_root / "MANIFEST.json",
        *(data_root / "gate" / f"{split}.json" for split in ("train", "val", "test")),
    ]


def _unit_config(args: argparse.Namespace) -> dict[str, Any]:
    """返回参与 resume 判定的完整实验配置。

    这里故意保持与首批五个 CLINC 单元相同的字段，避免仅因增加网格调度器
    就改变方法协议或使已有完整结果失效。
    """

    method = str(args.method)
    is_linear = method in {"msp", "energy", "entropy"}
    is_neighbor = method in {"knn", "lof"}
    return {
        "dataset": args.dataset,
        "kir": int(args.kir),
        "data_seed": int(args.data_seed),
        "method": method,
        "preprocessing": "raw_minilm" if is_linear else "l2_normalized_minilm",
        "linear_C_candidates": list(C_CANDIDATES) if is_linear else None,
        "neighbor_candidates": (
            list(KNN_NEIGHBORS if method == "knn" else LOF_NEIGHBORS)
            if is_neighbor
            else None
        ),
        "energy_temperature": 1.0 if method == "energy" else None,
        "split_minmax_normalization": False,
    }


def _completion_error(
    unit_dir: Path,
    data_root: Path,
    required_inputs: Sequence[Path],
    args: argparse.Namespace,
) -> str | None:
    """审计一个单元是否可安全复用；完整时返回 ``None``。

    ``--resume`` 不能只检查目录或单个指标文件。这里同时核对完成标记、配置、
    数据和编码器指纹，并读取 Parquet schema 与行数，以免中断写入或换数据后
    静默复用旧分数。
    """

    missing_outputs = [name for name in REQUIRED_OUTPUTS if not (unit_dir / name).is_file()]
    if missing_outputs:
        return "missing_outputs:" + ",".join(missing_outputs)
    missing_inputs = [path for path in required_inputs if not path.is_file()]
    if missing_inputs:
        return "missing_inputs:" + ",".join(str(path) for path in missing_inputs)
    try:
        manifest = _json_load(unit_dir / "run_manifest.json")
        if manifest.get("status") != "complete":
            return "manifest_not_complete"
        config = _unit_config(args)
        if manifest.get("config") != config or manifest.get("config_hash") != _sha256_payload(config):
            return "config_mismatch"
        expected_hashes = {
            str(path.relative_to(data_root)): _sha256_file(path) for path in required_inputs
        }
        if manifest.get("input_hashes") != expected_hashes:
            return "input_hash_mismatch"
        if manifest.get("model") != _model_fingerprint(Path(args.encoder_path)):
            return "model_mismatch"

        results = _json_load(unit_dir / "eval_results.json")
        if results.get("protocol") != "minilm_gate_baselines_v19":
            return "protocol_mismatch"
        if results.get("method") != args.method:
            return "method_mismatch"
        for split in ("validation", "test"):
            if not REQUIRED_METRICS.issubset((results.get(split) or {}).keys()):
                return f"missing_{split}_metrics"

        threshold = _json_load(unit_dir / "threshold_selection.json")
        if threshold.get("test_used_for_selection") is not False:
            return "test_selection_violation"
        if "threshold" not in (threshold.get("selected_operating_point") or {}):
            return "missing_selected_threshold"
        timing = _json_load(unit_dir / "timing.json")
        if "total_seconds" not in timing:
            return "missing_timing"

        scores = pd.read_parquet(unit_dir / "scores.parquet")
        if not REQUIRED_SCORE_COLUMNS.issubset(scores.columns):
            return "scores_schema_mismatch"
        expected_test_rows = len(_json_load(data_root / "gate" / "test.json"))
        if len(scores) != expected_test_rows:
            return "scores_row_count_mismatch"
        if scores["sample_id"].isna().any() or scores["sample_id"].duplicated().any():
            return "scores_sample_id_invalid"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return "unreadable_artifact"
    return None


def _can_resume(
    unit_dir: Path,
    data_root: Path,
    required_inputs: Sequence[Path],
    args: argparse.Namespace,
) -> bool:
    return _completion_error(unit_dir, data_root, required_inputs, args) is None


def _grid_arguments(args: argparse.Namespace) -> list[argparse.Namespace]:
    units: list[argparse.Namespace] = []
    for dataset in GRID_DATASETS:
        for kir in GRID_KIRS:
            for seed in GRID_SEEDS:
                for method in METHODS:
                    unit = copy.copy(args)
                    unit.dataset = dataset
                    unit.kir = kir
                    unit.data_seed = seed
                    unit.method = method
                    unit.grid = False
                    units.append(unit)
    return units


def _unit_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "kir": int(args.kir),
        "data_seed": int(args.data_seed),
        "method": args.method,
        "output_dir": str(
            _unit_dir(Path(args.output_root), args.dataset, args.kir, args.data_seed, args.method)
        ),
    }


def _write_missing_audit(
    units: Sequence[argparse.Namespace],
    failures: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    """写出 canonical 135 单元的完整性审计，便于中断后精确续跑。"""

    missing_units: list[dict[str, Any]] = []
    for unit in units:
        data_root = _dataset_root(Path(unit.data_root), unit.dataset, unit.kir, unit.data_seed)
        required = _required_inputs(data_root)
        unit_dir = _unit_dir(
            Path(unit.output_root), unit.dataset, unit.kir, unit.data_seed, unit.method
        )
        error = _completion_error(unit_dir, data_root, required, unit)
        if error is not None:
            missing_units.append({**_unit_identity(unit), "reason": error})

    payload = {
        "protocol": "minilm_gate_baselines_v19",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_unit_count": len(units),
        "complete_unit_count": len(units) - len(missing_units),
        "missing_unit_count": len(missing_units),
        "failed_unit_count": len(failures),
        "missing_units": missing_units,
        "failures": list(failures),
    }
    output_root = Path(units[0].output_root) if units else DEFAULT_OUTPUT_ROOT / "baselines"
    audit_path = output_root / "baseline_missing_cells.json"
    _write_json(audit_path, payload)
    return audit_path, payload


def _dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    data_root = _dataset_root(Path(args.data_root), args.dataset, args.kir, args.data_seed)
    return {
        "dry_run": True,
        **_unit_identity(args),
        "data_root": str(data_root),
        "required_inputs": [
            {"path": str(path), "exists": path.is_file()} for path in _required_inputs(data_root)
        ],
        "would_write": list(REQUIRED_OUTPUTS),
        "test_used_for_selection": False,
    }


def run_unit(args: argparse.Namespace) -> Path:
    """运行一个 dataset/KIR/seed/method 的 Gate-only Baseline 单元。"""

    started = time.perf_counter()
    data_root = _dataset_root(Path(args.data_root), args.dataset, args.kir, args.data_seed)
    required = _required_inputs(data_root)
    known_path, manifest_path = required[:2]
    split_paths = {split: data_root / "gate" / f"{split}.json" for split in ("train", "val", "test")}
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing experiment inputs: " + ", ".join(missing))
    unit_dir = _unit_dir(Path(args.output_root), args.dataset, args.kir, args.data_seed, args.method)
    if getattr(args, "resume", False) and _can_resume(unit_dir, data_root, required, args):
        return unit_dir

    known_manifest = _json_load(known_path)
    rows = {split: _json_load(path) for split, path in split_paths.items()}
    if any(int(row["label"]) != 0 for row in rows["train"]):
        raise ValueError("Gate train must contain known samples only")

    from sentence_transformers import SentenceTransformer

    model_path = Path(args.encoder_path)
    model_fingerprint = _model_fingerprint(model_path)
    # Baseline 的分类头、kNN 和 LOF 都在 CPU 上运行；显式把仅用于 cache miss
    # 兜底编码的 MiniLM 放在 CPU，避免已命中 embedding cache 时仍无谓占用 GPU。
    encoder = SentenceTransformer(str(model_path), device="cpu")
    embeddings: dict[str, np.ndarray] = {}
    cache_metadata: dict[str, Any] = {}
    encoding_started = time.perf_counter()
    for split in ("train", "val", "test"):
        embeddings[split], cache_metadata[split] = _load_or_encode(
            rows=rows[split],
            split_path=split_paths[split],
            split=split,
            encoder=encoder,
            model_fingerprint=model_fingerprint,
            cache_root=Path(args.cache_root),
            dataset=args.dataset,
            kir=args.kir,
            seed=args.data_seed,
            batch_size=args.batch_size,
        )
    encoding_seconds = time.perf_counter() - encoding_started

    labels = {
        split: np.asarray([int(row["label"]) for row in rows[split]], dtype=np.int64)
        for split in ("val", "test")
    }
    train_intents = np.asarray([str(row["intent"]) for row in rows["train"]], dtype=object)
    intent_names = sorted(set(train_intents.tolist()))
    intent_to_id = {intent: index for index, intent in enumerate(intent_names)}
    train_y = np.asarray([intent_to_id[intent] for intent in train_intents], dtype=np.int64)
    guard, fixed_k1_recall = _fixed_k1_guard(
        embeddings["train"], train_intents, rows["val"], embeddings["val"], "euclidean"
    )

    fit_started = time.perf_counter()
    if args.method in {"msp", "energy", "entropy"}:
        # 线性分类头只在 known train 上拟合；validation 中的 known 样本
        # 仅用于选 C，不会把 OOS 伪造成额外分类类别。
        known_mask = labels["val"] == 0
        val_known_y = np.asarray(
            [intent_to_id[str(row["intent"])] for row in rows["val"] if int(row["label"]) == 0],
            dtype=np.int64,
        )
        classifier, classifier_selection = _fit_linear_classifier(
            embeddings["train"], train_y, embeddings["val"][known_mask], val_known_y
        )
        val_scores, _ = _linear_oos_scores(classifier, embeddings["val"], args.method)
        operating_point, threshold_candidates = select_operating_point(labels["val"], val_scores, guard)
        hyperparameter_selection = classifier_selection
        preprocessing = "raw_minilm"
        fit_and_validation_seconds = time.perf_counter() - fit_started
        test_started = time.perf_counter()
        test_scores, test_predicted = _linear_oos_scores(
            classifier, embeddings["test"], args.method
        )
        nearest_intents = [intent_names[int(index)] for index in test_predicted]
        test_scoring_seconds = time.perf_counter() - test_started
    else:
        # 邻域方法的距离对向量模长敏感，因此按协议先做 L2 归一化。
        # 对这些 embedding，Euclidean 和 cosine 的排序等价，不把它们包装成两个独立 Baseline。
        train_x = _l2_normalize(embeddings["train"])
        val_x = _l2_normalize(embeddings["val"])
        test_x = _l2_normalize(embeddings["test"])
        neighbor_values = KNN_NEIGHBORS if args.method == "knn" else LOF_NEIGHBORS
        candidate_runs = []
        validation_scores = {}
        for neighbor_count in neighbor_values:
            val_scores_i, _ = _score_neighbor_method(
                args.method, neighbor_count, train_x, val_x
            )
            selected_i, thresholds_i = select_operating_point(labels["val"], val_scores_i, guard)
            candidate_runs.append(
                {
                    "n_neighbors": int(neighbor_count),
                    "selected_operating_point": selected_i,
                    "threshold_candidates": thresholds_i,
                }
            )
            validation_scores[int(neighbor_count)] = val_scores_i

        eligible_runs = [
            row for row in candidate_runs if not row["selected_operating_point"]["guard_violation"]
        ]
        pool = eligible_runs or candidate_runs
        selected_run = max(
            pool,
            key=lambda row: (
                row["selected_operating_point"]["metrics"]["oos_f1"],
                -row["selected_operating_point"]["metrics"]["fpr95"],
                row["selected_operating_point"]["metrics"]["id_recall"],
                -row["n_neighbors"],
            ),
        )
        selected_neighbors = int(selected_run["n_neighbors"])
        val_scores = validation_scores[selected_neighbors]
        operating_point = selected_run["selected_operating_point"]
        threshold_candidates = selected_run["threshold_candidates"]
        hyperparameter_selection = {
            "selected": {"n_neighbors": selected_neighbors},
            "candidates": candidate_runs,
        }
        preprocessing = "l2_normalized_minilm"
        fit_and_validation_seconds = time.perf_counter() - fit_started
        test_started = time.perf_counter()
        test_scores, test_nearest = _score_neighbor_method(
            args.method, selected_neighbors, train_x, test_x
        )
        nearest_intents = [str(train_intents[int(index)]) for index in test_nearest]
        test_scoring_seconds = time.perf_counter() - test_started

    # operating point 到此已完全冻结。test 只用该阈值计算一次指标，
    # 后续不根据 test 结果调整邻居数、C 或 threshold。
    threshold = float(operating_point["threshold"])
    test_metrics = compute_binary_oos_metrics(labels["test"], test_scores, threshold)
    val_metrics = compute_binary_oos_metrics(labels["val"], val_scores, threshold)
    scores = _score_frame(rows["test"], test_scores, threshold, nearest_intents, known_manifest)

    unit_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(unit_dir / "scores.parquet", index=False)
    timing = {
        "embedding_load_or_encode_seconds": encoding_seconds,
        "fit_and_validation_seconds": fit_and_validation_seconds,
        "test_scoring_seconds": test_scoring_seconds,
        "total_seconds": time.perf_counter() - started,
        "test_samples_per_second": len(rows["test"]) / max(test_scoring_seconds, 1e-12),
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    _write_json(unit_dir / "timing.json", timing)
    _write_json(
        unit_dir / "eval_results.json",
        {
            "protocol": "minilm_gate_baselines_v19",
            "method": args.method,
            "validation": val_metrics,
            "test": test_metrics,
            "timing": timing,
        },
    )
    _write_json(
        unit_dir / "threshold_selection.json",
        {
            "selection_split": "gate/val.json",
            "test_used_for_selection": False,
            "id_recall_guard": guard,
            "fixed_k1_validation_id_recall": fixed_k1_recall,
            "selected_operating_point": operating_point,
            "threshold_candidates": threshold_candidates,
            "hyperparameter_selection": hyperparameter_selection,
        },
    )
    config = _unit_config(args)
    if config["preprocessing"] != preprocessing:
        raise RuntimeError("internal preprocessing/config mismatch")
    # manifest 最后写入，并且只有所有产物落盘后才标记 complete；中断留下的
    # 半成品目录不会通过下一次 ``--resume`` 的完整性审计。
    _write_json(
        unit_dir / "run_manifest.json",
        {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "project_root": str(PROJECT_ROOT),
            "output_dir": str(unit_dir),
            "config": config,
            "config_hash": _sha256_payload(config),
            "data_root": str(data_root),
            "input_hashes": {str(path.relative_to(data_root)): _sha256_file(path) for path in required},
            "model": model_fingerprint,
            "embedding_cache": cache_metadata,
            "timing": timing,
            "historical_artifacts_overwritten": False,
        },
    )
    return unit_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("clinc150", "banking77_oos", "stackoverflow"), default="clinc150")
    parser.add_argument("--kir", type=int, choices=(25, 50, 75), default=50)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--encoder-path", type=Path, default=PATHS.minilm)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "embedding_cache")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "baselines")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run the canonical 3 datasets x 3 KIR x 3 seeds x 5 methods grid.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.grid:
        units = _grid_arguments(args)
        if args.dry_run:
            payloads = [_dry_run_payload(unit) for unit in units]
            print(json.dumps({"unit_count": len(payloads), "units": payloads}, indent=2))
            return

        failures: list[dict[str, Any]] = []
        for index, unit in enumerate(units, start=1):
            try:
                output_dir = run_unit(unit)
                print(f"[{index}/{len(units)}] {output_dir}", flush=True)
            except Exception as exc:  # noqa: BLE001 - 网格必须保留其余单元的审计机会。
                failure = {**_unit_identity(unit), "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(f"[{index}/{len(units)}] FAILED {failure['error']}", file=sys.stderr, flush=True)

        audit_path, audit = _write_missing_audit(units, failures)
        print(
            json.dumps(
                {
                    "status": "complete" if audit["missing_unit_count"] == 0 else "incomplete",
                    "expected_unit_count": audit["expected_unit_count"],
                    "complete_unit_count": audit["complete_unit_count"],
                    "missing_unit_count": audit["missing_unit_count"],
                    "audit_path": str(audit_path),
                },
                indent=2,
            )
        )
        if audit["missing_unit_count"]:
            raise SystemExit(1)
        return

    if args.method is None:
        raise ValueError("--method is required unless --grid is used")
    if args.dry_run:
        print(json.dumps(_dry_run_payload(args), indent=2))
        return
    output_dir = run_unit(args)
    print(json.dumps({"status": "complete", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
