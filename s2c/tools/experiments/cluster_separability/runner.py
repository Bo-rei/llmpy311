#!/usr/bin/env python3
"""运行受控的 MiniLM 多簇/OOS 可分性实验。

该 Runner 不重写历史 ``MultiSphereOOSDetector``，只在其外层补充统一协议、
embedding 缓存、标准 OOS 指标和可审计产物，从而保证新消融与历史 K=2
结果共享同一检测器语义。

``fixed`` 固定边界参数，只回答“改变 K 本身是否有效”；``tuned`` 允许每个 K
在 validation 上独立选边界，回答“部署时该 K 的最佳性能”。test 在两种设置中
都不参与 K、阈值或边界选择。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gate.multi_sphere_oos_detector import MultiSphereOOSDetector, SphereConfig
from src.runtime import WorkspacePaths
from .protocol import (
    compute_binary_oos_metrics,
    compute_cluster_quality_metrics,
    compute_coverage_counts,
    gold_sample_kind,
    select_boundary,
)

PATHS = WorkspacePaths.discover(PROJECT_ROOT)
DEFAULT_DATA_ROOT = PATHS.prepared_data_root / "multidataset" / "v19"
DEFAULT_OUTPUT_ROOT = (
    PATHS.artifact_root / "outputs" / "experiments" / "cluster_separability_v19"
)
SUPPORTED_EXECUTION_PHASES = {"smoke", "fixed", "tuned"}
PHASES = ("smoke", "fixed", "tuned", "baselines", "analysis")
DISTANCES = ("euclidean", "mahalanobis_diag")
LAMBDA_CANDIDATES = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50)
GAMMA_CANDIDATES: tuple[float | None, ...] = (None, 0.98, 0.95, 0.92)
GRID_DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
GRID_KIRS = (25, 50, 75)
GRID_SEEDS = (13, 42, 87)
GRID_K = (1, 2, 3, 4, 5)
_ENCODER_CACHE: dict[str, Any] = {}


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _dataset_root(base: Path, dataset: str, kir: int, data_seed: int) -> Path:
    return base / dataset / f"kir{int(kir)}_seed{int(data_seed)}"


def _unit_dir(output_root: Path, phase: str, dataset: str, kir: int, seed: int, distance: str, k: int) -> Path:
    return (
        output_root
        / phase
        / dataset
        / f"kir{int(kir)}_seed{int(seed)}"
        / distance
        / f"k{int(k)}"
    )


def _model_fingerprint(model_path: Path) -> dict[str, Any]:
    files = [
        model_path / "config.json",
        model_path / "config_sentence_transformers.json",
        model_path / "modules.json",
    ]
    return {
        "path": str(model_path.resolve()),
        "files": {item.name: _sha256_file(item) for item in files if item.is_file()},
    }


def _load_encoder(model_path: Path) -> Any:
    key = str(model_path.resolve())
    if key not in _ENCODER_CACHE:
        from sentence_transformers import SentenceTransformer

        # MiniLM 只负责一次性编码且 embedding 会落入可复用 cache。显式放在 CPU
        # 可避免几何网格长期占用训练 GPU；设备不进入方法协议，也不改变缓存 key。
        _ENCODER_CACHE[key] = SentenceTransformer(key, device="cpu")
    return _ENCODER_CACHE[key]


def _cache_key(dataset: str, kir: int, seed: int, split: str, data_hash: str, model_hash: str) -> str:
    return _sha256_payload(
        {
            "dataset": dataset,
            "kir": int(kir),
            "seed": int(seed),
            "split": split,
            "data_hash": data_hash,
            "model_hash": model_hash,
        }
    )


def _load_or_encode(
    *,
    rows: Sequence[Mapping[str, Any]],
    split_path: Path,
    split: str,
    encoder: Any,
    model_fingerprint: Mapping[str, Any],
    cache_root: Path,
    dataset: str,
    kir: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """按“数据内容 + encoder 指纹”复用 embedding，并返回完整 provenance。

    缓存键不只包含 dataset/KIR/seed：分割文件或模型配置一旦变化，hash 会随之
    变化，旧 embedding 不会被静默复用。缓存保存 encoder 的原始输出；L2 处理
    由 detector 按固定协议执行，不在此处改写语义空间。
    """
    data_hash = _sha256_file(split_path)
    model_hash = _sha256_payload(model_fingerprint)
    key = _cache_key(dataset, kir, seed, split, data_hash, model_hash)
    cache_dir = cache_root / dataset / f"kir{int(kir)}_seed{int(seed)}"
    cache_path = cache_dir / f"{split}_{key[:16]}.npz"
    metadata_path = cache_dir / f"{split}_{key[:16]}.json"
    expected = {
        "cache_key": key,
        "dataset": dataset,
        "kir": int(kir),
        "data_seed": int(seed),
        "split": split,
        "data_hash": data_hash,
        "model_hash": model_hash,
        "sample_count": len(rows),
    }
    if cache_path.is_file() and metadata_path.is_file():
        metadata = _json_load(metadata_path)
        if all(metadata.get(field) == value for field, value in expected.items()):
            with np.load(cache_path, allow_pickle=False) as payload:
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            if embeddings.shape[0] != len(rows):
                raise RuntimeError(f"embedding cache row mismatch: {cache_path}")
            metadata["cache_hit"] = True
            return embeddings, metadata

    texts = [str(row["text"]) for row in rows]
    embeddings = np.asarray(
        encoder.encode(texts, batch_size=int(batch_size), show_progress_bar=False),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise RuntimeError(f"encoder returned invalid shape {embeddings.shape} for {len(rows)} rows")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    metadata = {
        **expected,
        "embedding_shape": list(embeddings.shape),
        "embedding_hash": hashlib.sha256(embeddings.tobytes(order="C")).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_hit": False,
    }
    _write_json(metadata_path, metadata)
    return embeddings, metadata


def _build_detector(k_gate: int, distance: str, radius_lambda: float = 1.0, margin_gamma: float | None = None) -> MultiSphereOOSDetector:
    """使用论文协议冻结的参数构造 Gate，不在网格中引入额外自由度。"""

    return MultiSphereOOSDetector(
        n_clusters=None,
        radius_quantile=0.90,
        radius_method="mean_std",
        radius_lambda=float(radius_lambda),
        center_mode="class_centroid_mixture",
        distance_metric=distance,
        margin_gamma=margin_gamma,
        # 对角马氏距离使用“每个局部簇单独估计”的对角协方差。
        # 这一定义与现有 detector 和历史结果链保持一致，不是全局协方差。
        covariance_eps=1e-6,
        l2_normalize=True,
        subcenters_per_intent=int(k_gate),
        random_state=42,
    )


def _all_sphere_distances(detector: MultiSphereOOSDetector, embeddings: np.ndarray) -> np.ndarray:
    x = detector._normalize_embeddings(np.asarray(embeddings))
    columns: list[np.ndarray] = []
    for sphere in detector.spheres:
        diff = x - sphere.center
        if detector.distance_metric == "mahalanobis_diag":
            if sphere.inv_diag_cov is None:
                raise RuntimeError("mahalanobis sphere has no inverse diagonal covariance")
            column = np.sqrt(np.sum((diff**2) * sphere.inv_diag_cov, axis=1))
        else:
            column = np.linalg.norm(diff, axis=1)
        columns.append(np.asarray(column, dtype=np.float64))
    if not columns:
        raise RuntimeError("fitted detector contains no spheres")
    return np.stack(columns, axis=1)


def _oos_source(row: Mapping[str, Any], known_manifest: Mapping[str, Any]) -> str:
    if int(row["label"]) == 0:
        return "known"
    source = str(row.get("source_split", row.get("split", ""))).lower()
    intent = str(row.get("intent", ""))
    if source.startswith(("id-oos", "ood-oos")):
        return "provided_oos"
    if source.startswith(("oos_", "clinc_oos")) or intent.lower() == "oos":
        return "native_oos"
    if source.startswith("heldout_oos"):
        return "heldout_unknown"
    if intent in {str(value) for value in known_manifest.get("unknown_intents", [])}:
        return "heldout_unknown"
    return "native_or_provided_oos"


def _sample_id(row: Mapping[str, Any], split: str, index: int) -> str:
    if row.get("source_id") is not None:
        return f"{split}:{row['source_id']}"
    payload = {
        "split": split,
        "index": int(index),
        "text": str(row.get("text", "")),
        "intent": str(row.get("intent", "")),
        "source_split": str(row.get("source_split", "")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _score_split(
    detector: MultiSphereOOSDetector,
    rows: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    known_manifest: Mapping[str, Any],
    split: str,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """在一个固定分割上产生指标和样本级审计表。

    ``score`` 是距离与局部半径的比值，大于 1 表示落在最近接受区域之外。
    ``coverage_count`` 则使用全部球计算，且按不同 intent 去重，两者不可互相替代。
    """

    gate = detector.predict_with_scores(embeddings)
    distances = _all_sphere_distances(detector, embeddings)
    radii = np.asarray([sphere.radius for sphere in detector.spheres], dtype=np.float64)
    sphere_intents = [str(sphere.intent_name) for sphere in detector.spheres]
    coverage = compute_coverage_counts(distances, radii, sphere_intents)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray(gate["score"], dtype=np.float64)
    prediction = (scores > float(threshold)).astype(np.int64)
    if detector.margin_gamma is not None:
        prediction = np.where(np.asarray(gate["margin_ok"], dtype=bool), prediction, 1)
    metrics = compute_binary_oos_metrics(labels, scores, threshold)
    # 公共指标函数只能看到标量 score，不知道 margin 的二次拒绝规则。
    # 启用 margin 时，因此用 detector 的最终 prediction 重算 confusion 类指标；
    # AUROC/AUPR/FPR95 仍由连续 score 计算，不伪造一个带 margin 的连续排序分数。
    if detector.margin_gamma is not None:
        tp = int(np.sum((prediction == 1) & (labels == 1)))
        fp = int(np.sum((prediction == 1) & (labels == 0)))
        fn = int(np.sum((prediction == 0) & (labels == 1)))
        known_n = int(np.sum(labels == 0))
        oos_n = int(np.sum(labels == 1))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / oos_n if oos_n else math.nan
        metrics.update(
            {
                "oos_precision": precision,
                "oos_recall": recall,
                "oos_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "id_recall": (known_n - fp) / known_n if known_n else math.nan,
                "oos_rejection": recall,
            }
        )

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sphere_index = int(gate["nearest_cluster"][index])
        sphere: SphereConfig = detector.spheres[sphere_index]
        source = _oos_source(row, known_manifest)
        gold_kind = gold_sample_kind(
            {
                **dict(row),
                "oos_source": source,
                "true_binary_label": int(row["label"]),
            },
            known_manifest,
        )
        requested_k = int(detector.subcenters_per_intent)
        effective_k = len(detector.intent_to_clusters.get(str(row["intent"]), [])) if int(row["label"]) == 0 else None
        records.append(
            {
                "sample_id": _sample_id(row, split, index),
                "true_binary_label": int(row["label"]),
                "true_intent": str(row["intent"]),
                "oos_source": source,
                "gold_sample_kind": gold_kind,
                "source_split": str(row.get("source_split", row.get("split", split))),
                "score": float(scores[index]),
                "prediction": int(prediction[index]),
                "nearest_intent": sphere.intent_name,
                "nearest_cluster": int(sphere.cluster_id),
                "distance": float(gate["distance"][index]),
                "radius": float(gate["radius"][index]),
                "coverage_count": int(coverage[index]),
                "requested_k": requested_k,
                "effective_k": effective_k,
            }
        )
    frame = pd.DataFrame.from_records(records)
    metrics["sample_count"] = int(len(rows))
    metrics["known_count"] = int(np.sum(labels == 0))
    metrics["oos_count"] = int(np.sum(labels == 1))
    metrics["known_multi_coverage_rate"] = float(np.mean(coverage[labels == 0] >= 2)) if np.any(labels == 0) else math.nan
    metrics["oos_multi_coverage_rate"] = float(np.mean(coverage[labels == 1] >= 2)) if np.any(labels == 1) else math.nan
    return metrics, frame


def _cluster_rows(detector: MultiSphereOOSDetector, train_embeddings: np.ndarray, train_intents: np.ndarray) -> list[dict[str, Any]]:
    """按 intent 导出局部簇质量，保留 requested K 与 effective K 的区别。

    当某个 intent 的训练样本少于 K 时，detector 会安全截断子中心数。导出两个
    K 可以避免将“请求 K=5”误报成“所有 intent 都拟合了 5 个簇”。
    """

    if detector._train_cluster_labels is None:
        raise RuntimeError("detector does not expose fitted train cluster labels")
    normalized = detector._normalize_embeddings(train_embeddings)
    global_labels = np.asarray(detector._train_cluster_labels, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for intent in sorted(detector.intent_to_clusters):
        mask = train_intents == intent
        intent_global = global_labels[mask]
        cluster_ids = detector.intent_to_clusters[intent]
        remap = {cluster_id: local_id for local_id, cluster_id in enumerate(cluster_ids)}
        local_labels = np.asarray([remap[int(value)] for value in intent_global], dtype=np.int64)
        quality = compute_cluster_quality_metrics(normalized[mask], local_labels)
        intent_spheres = [sphere for sphere in detector.spheres if sphere.intent_name == intent]
        rows.append(
            {
                "intent": intent,
                "support": int(np.sum(mask)),
                "requested_k": int(detector.subcenters_per_intent),
                "effective_k": len(cluster_ids),
                "minimum_radius": min((float(s.radius) for s in intent_spheres), default=math.nan),
                "maximum_radius": max((float(s.radius) for s in intent_spheres), default=math.nan),
                **quality,
            }
        )
    return rows


def _tune_detector(
    detector: MultiSphereOOSDetector,
    val_rows: Sequence[Mapping[str, Any]],
    val_embeddings: np.ndarray,
    guard: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """在不重复 KMeans 的前提下，仅用 validation 选择半径与 margin。

    中心、簇归属和马氏协方差在 detector.fit 后均已固定，因此候选组合只需
    重算半径和决策，无需重复聚类。这既减少计算，也避免每个边界候选意外
    得到不同的 KMeans 局部最优解。
    """

    labels = np.asarray([int(row["label"]) for row in val_rows], dtype=np.int64)
    all_distances = _all_sphere_distances(detector, val_embeddings)
    nearest_index = np.argmin(all_distances, axis=1)
    nearest_distance = all_distances[np.arange(all_distances.shape[0]), nearest_index]
    if all_distances.shape[1] > 1:
        nearest_two = np.partition(all_distances, 1, axis=1)[:, :2]
        first_distance = np.min(nearest_two, axis=1)
        second_distance = np.max(nearest_two, axis=1)
    else:
        first_distance = nearest_distance
        second_distance = np.full_like(nearest_distance, np.inf)

    candidates: list[dict[str, Any]] = []
    for radius_lambda in LAMBDA_CANDIDATES:
        detector.radius_lambda = float(radius_lambda)
        # ``fit`` 已在 detector 内部缓存 L2-normalized train embedding。半径必须
        # 基于该缓存重算；如果此处传入 raw embedding，会把归一化中心与原始尺度半径混用。
        detector._compute_radii()
        radii = np.asarray([sphere.radius for sphere in detector.spheres], dtype=np.float64)
        scores = nearest_distance / np.maximum(radii[nearest_index], 1e-12)
        for margin_gamma in GAMMA_CANDIDATES:
            prediction = scores > 1.0
            if margin_gamma is not None:
                prediction |= first_distance >= float(margin_gamma) * second_distance
            metrics = compute_binary_oos_metrics(labels, scores, 1.0)
            known = labels == 0
            oos = labels == 1
            tp = int(np.sum(prediction & oos))
            fp = int(np.sum(prediction & known))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / int(np.sum(oos)) if np.any(oos) else math.nan
            metrics.update(
                {
                    "oos_precision": precision,
                    "oos_recall": recall,
                    "oos_f1": 2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0,
                    "id_recall": float(np.mean(~prediction[known])) if np.any(known) else math.nan,
                    "oos_rejection": recall,
                }
            )
            candidates.append(
                {
                    "radius_lambda": float(radius_lambda),
                    "margin_gamma": margin_gamma,
                    "metrics": metrics,
                }
            )
    selected = select_boundary(candidates, float(guard))
    detector.radius_lambda = float(selected["radius_lambda"])
    detector.margin_gamma = selected["margin_gamma"]
    detector._compute_radii()
    return selected, candidates


def _fixed_k1_guard(
    train_embeddings: np.ndarray,
    train_intents: np.ndarray,
    val_rows: Sequence[Mapping[str, Any]],
    val_embeddings: np.ndarray,
    distance: str,
) -> tuple[float, float]:
    """以同距离的 fixed K=1 作为 ID recall 守门基准。

    guard 取 ``max(0.80, K1 recall - 0.01)``，允许 1 个百分点的波动，但防止通过
    大量误拒 Known 来换取表面上更高的 OOS F1。
    """

    detector = _build_detector(1, distance)
    detector.fit(train_embeddings, train_intents)
    labels = np.asarray([int(row["label"]) for row in val_rows], dtype=np.int64)
    scores = detector.predict_with_scores(val_embeddings)["score"]
    fixed_recall = float(compute_binary_oos_metrics(labels, scores, 1.0)["id_recall"])
    return max(0.80, fixed_recall - 0.01), fixed_recall


def _dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    data_root = _dataset_root(Path(args.data_root), args.dataset, args.kir, args.data_seed)
    unit_dir = _unit_dir(Path(args.output_root), args.phase, args.dataset, args.kir, args.data_seed, args.distance, args.k_gate)
    required = [data_root / "KNOWN_INTENTS.json", data_root / "MANIFEST.json"] + [
        data_root / "gate" / f"{split}.json" for split in ("train", "val", "test")
    ]
    return {
        "dry_run": True,
        "phase": args.phase,
        "execution_supported": args.phase in SUPPORTED_EXECUTION_PHASES,
        "dataset": args.dataset,
        "kir": int(args.kir),
        "data_seed": int(args.data_seed),
        "k_gate": int(args.k_gate),
        "kmeans_seed": 42,
        "distance": args.distance,
        "data_root": str(data_root),
        "output_dir": str(unit_dir),
        "required_inputs": [{"path": str(path), "exists": path.is_file()} for path in required],
        "would_write": [
            "eval_results.json",
            "run_manifest.json",
            "threshold_selection.json",
            "scores.parquet",
            "validation_scores.parquet",
            "cluster_metrics.csv",
            "detector.json",
        ],
    }


def _unit_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase": args.phase,
        "dataset": args.dataset,
        "kir": int(args.kir),
        "data_seed": int(args.data_seed),
        "k_gate": int(args.k_gate),
        "kmeans_seed": 42,
        "distance": args.distance,
        "center_mode": "class_centroid_mixture",
        "radius_method": "mean_std",
        "l2_normalize": True,
        "covariance_scope": "per_cluster",
        "covariance_eps": 1e-6,
    }


def _can_resume(
    unit_dir: Path,
    data_root: Path,
    required_inputs: Sequence[Path],
    args: argparse.Namespace,
) -> bool:
    """只在产物完整、配置一致且所有输入 hash 匹配时允许 resume。

    单独存在 ``eval_results.json`` 不足以证明单元完成；中断写入、更换分割
    或改变 K/距离后都必须重跑，不得被 ``--resume`` 静默跳过。
    """

    required_outputs = (
        "eval_results.json",
        "run_manifest.json",
        "threshold_selection.json",
        "scores.parquet",
        "validation_scores.parquet",
        "cluster_metrics.csv",
        "detector.json",
    )
    if not all((unit_dir / name).is_file() for name in required_outputs):
        return False
    try:
        manifest = _json_load(unit_dir / "run_manifest.json")
        if manifest.get("status") != "complete":
            return False
        if manifest.get("config_hash") != _sha256_payload(_unit_config(args)):
            return False
        expected_hashes = {
            str(path.relative_to(data_root)): _sha256_file(path)
            for path in required_inputs
        }
        return manifest.get("input_hashes") == expected_hashes
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run_unit(args: argparse.Namespace) -> Path:
    """运行一个完整、可独立复现的 dataset/KIR/seed/distance/K 实验单元。"""

    if args.phase not in SUPPORTED_EXECUTION_PHASES:
        raise NotImplementedError(f"phase {args.phase!r} currently supports --dry-run only")
    started = time.perf_counter()
    data_root = _dataset_root(Path(args.data_root), args.dataset, args.kir, args.data_seed)
    known_path = data_root / "KNOWN_INTENTS.json"
    manifest_path = data_root / "MANIFEST.json"
    split_paths = {split: data_root / "gate" / f"{split}.json" for split in ("train", "val", "test")}
    required = [known_path, manifest_path, *split_paths.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing experiment inputs: " + ", ".join(missing))
    unit_dir = _unit_dir(
        Path(args.output_root), args.phase, args.dataset, args.kir,
        args.data_seed, args.distance, args.k_gate,
    )
    if getattr(args, "resume", False) and _can_resume(unit_dir, data_root, required, args):
        return unit_dir

    known_manifest = _json_load(known_path)
    data_manifest = _json_load(manifest_path)
    rows = {split: _json_load(path) for split, path in split_paths.items()}
    if any(int(row["label"]) != 0 for row in rows["train"]):
        raise ValueError("Gate train must contain known samples only")

    model_path = Path(args.encoder_path)
    model_fingerprint = _model_fingerprint(model_path)
    encoder = _load_encoder(model_path)
    cache_root = Path(args.output_root) / "embedding_cache"
    embeddings: dict[str, np.ndarray] = {}
    cache_metadata: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        embeddings[split], cache_metadata[split] = _load_or_encode(
            rows=rows[split],
            split_path=split_paths[split],
            split=split,
            encoder=encoder,
            model_fingerprint=model_fingerprint,
            cache_root=cache_root,
            dataset=args.dataset,
            kir=args.kir,
            seed=args.data_seed,
            batch_size=args.batch_size,
        )

    train_intents = np.asarray([str(row["intent"]) for row in rows["train"]], dtype=object)
    detector = _build_detector(args.k_gate, args.distance)
    detector.fit(embeddings["train"], train_intents)

    threshold_selection: dict[str, Any]
    if args.phase == "tuned":
        guard, fixed_k1_recall = _fixed_k1_guard(
            embeddings["train"], train_intents, rows["val"], embeddings["val"], args.distance
        )
        selected, candidates = _tune_detector(
            detector, rows["val"], embeddings["val"], guard
        )
        threshold_selection = {
            "type": "validation_boundary_selection",
            "selection_split": "gate/val.json",
            "test_used_for_selection": False,
            "id_recall_guard": guard,
            "fixed_k1_validation_id_recall": fixed_k1_recall,
            "selected": selected,
            "candidates": candidates,
        }
    else:
        threshold_selection = {
            "type": "fixed_boundary",
            "selection_split": None,
            "test_used_for_selection": False,
            "radius_lambda": 1.0,
            "margin_gamma": None,
            "decision_threshold": 1.0,
        }

    # 先完成 validation 选择，再对 test 做一次固定决策的评估。
    # test score 不回流到 detector、K 或阈值选择。
    val_metrics, validation_scores = _score_split(
        detector, rows["val"], embeddings["val"], known_manifest, "val", 1.0
    )
    test_metrics, scores = _score_split(
        detector, rows["test"], embeddings["test"], known_manifest, "test", 1.0
    )
    cluster_rows = _cluster_rows(detector, embeddings["train"], train_intents)

    unit_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(unit_dir / "scores.parquet", index=False)
    validation_scores.to_parquet(unit_dir / "validation_scores.parquet", index=False)
    with (unit_dir / "cluster_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(cluster_rows[0].keys()) if cluster_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(_json_safe(cluster_rows))

    detector.save(unit_dir / "detector.json")
    eval_results = {
        "protocol": "cluster_separability_v19",
        "phase": args.phase,
        "validation": val_metrics,
        "test": test_metrics,
        "boundary": {
            "radius_method": detector.radius_method,
            "radius_lambda": detector.radius_lambda,
            "margin_gamma": detector.margin_gamma,
            "decision_threshold": 1.0,
        },
        "sphere_count": len(detector.spheres),
        "intent_count": len(detector.intent_to_clusters),
    }
    _write_json(unit_dir / "eval_results.json", eval_results)
    _write_json(unit_dir / "threshold_selection.json", threshold_selection)

    input_hashes = {str(path.relative_to(data_root)): _sha256_file(path) for path in required}
    config = _unit_config(args)
    # manifest 最后写入，且只有到达此处才标记 complete。
    # 因此中途崩溃留下的部分文件不会被 resume 误认为成功单元。
    run_manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(unit_dir),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "config": config,
        "config_hash": _sha256_payload(config),
        "data_root": str(data_root),
        "data_manifest_hash": _sha256_file(manifest_path),
        "input_hashes": input_hashes,
        "model": model_fingerprint,
        "embedding_cache": cache_metadata,
        "requested_k": int(args.k_gate),
        "effective_k_by_intent": {
            intent: len(clusters) for intent, clusters in detector.intent_to_clusters.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
        "historical_artifacts_overwritten": False,
        "data_manifest": {
            "dataset": data_manifest.get("dataset"),
            "kir": data_manifest.get("kir"),
            "seed": data_manifest.get("seed"),
        },
    }
    _write_json(unit_dir / "run_manifest.json", run_manifest)
    return unit_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="smoke")
    parser.add_argument("--dataset", choices=("clinc150", "banking77_oos", "stackoverflow"), default="clinc150")
    parser.add_argument("--kir", type=int, choices=(25, 50, 75), default=50)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--k-gate", type=int, default=1)
    parser.add_argument("--distance", choices=DISTANCES, default="euclidean")
    parser.add_argument("--encoder-path", type=Path, default=PATHS.minilm)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run the canonical 3x3x5x2x3 grid for fixed or tuned phase.",
    )
    return parser


def _grid_arguments(args: argparse.Namespace) -> list[argparse.Namespace]:
    units = []
    for dataset in GRID_DATASETS:
        for kir in GRID_KIRS:
            for seed in GRID_SEEDS:
                for distance in DISTANCES:
                    for k_gate in GRID_K:
                        unit = copy.copy(args)
                        unit.dataset = dataset
                        unit.kir = kir
                        unit.data_seed = seed
                        unit.distance = distance
                        unit.k_gate = k_gate
                        unit.grid = False
                        units.append(unit)
    return units


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.k_gate < 1:
        raise ValueError("--k-gate must be >= 1")
    if args.grid:
        if args.phase not in {"fixed", "tuned"}:
            raise ValueError("--grid requires --phase fixed or tuned")
        units = _grid_arguments(args)
        if args.dry_run:
            payloads = [_dry_run_payload(unit) for unit in units]
            print(json.dumps({"unit_count": len(payloads), "units": payloads}, indent=2))
            return
        completed = []
        for index, unit in enumerate(units, start=1):
            output = run_unit(unit)
            completed.append(str(output))
            print(f"[{index}/{len(units)}] {output}", flush=True)
        print(json.dumps({"status": "complete", "unit_count": len(completed)}, indent=2))
        return
    if args.dry_run:
        print(json.dumps(_json_safe(_dry_run_payload(args)), indent=2, ensure_ascii=False))
        return
    output_dir = run_unit(args)
    print(json.dumps({"status": "complete", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
