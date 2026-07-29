#!/usr/bin/env python3
"""KMeans 多簇与随机分簇的受控对照。

随机对照保持每个意图的子簇数量和 KMeans 的簇大小多重集合一致，只改变样本
归属关系。这样可以把“增加球体数量/局部覆盖”与“语义聚类归属”分开。实验
直接复用 v19 embedding cache，不重新编码文本；所有结果写入 v20。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from tools.experiments.cluster_separability.analysis import _json as _read_json, _load_cache
from tools.experiments.cluster_separability.protocol import (
    compute_binary_oos_metrics,
    compute_coverage_counts,
    gold_sample_kind,
)
from tools.experiments.cluster_separability.runner import (
    _all_sphere_distances,
    _build_detector,
    _oos_source,
    _sample_id,
)

V19_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v19"
V20_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v20"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
SEEDS = (13, 42, 87)
DISTANCES = ("euclidean", "mahalanobis_diag")
REPEATS = (1, 2, 3, 4, 5)
_INPUT_CACHE: dict[tuple[str, int], tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]] = {}
_REFERENCE_SIZE_CACHE: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _unit(root: Path, phase: str, dataset: str, kir: int, seed: int, distance: str, k: int) -> Path:
    return root / phase / dataset / f"kir{kir}_seed{seed}" / distance / f"k{k}"


def _selected_k(v19_root: Path, dataset: str, seed: int, distance: str) -> int:
    rows = pd.read_csv(v19_root / "selected_k_summary.csv")
    match = rows[
        (rows["dataset"] == dataset)
        & (rows["kir"] == 50)
        & (rows["data_seed"] == seed)
        & (rows["distance"] == distance)
    ]
    if len(match) != 1:
        raise ValueError(f"selected K is not unique for {dataset}/{seed}/{distance}")
    return int(match.iloc[0]["selected_k"])


class _StaticCenters:
    """给现有 detector 提供 KMeans 同形状的静态中心容器。"""

    def __init__(self, centers: np.ndarray, labels: np.ndarray) -> None:
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_clusters = int(len(centers))


def _random_partition_detector(
    train_embeddings: np.ndarray,
    train_intents: np.ndarray,
    k: int,
    distance: str,
    repeat: int,
    reference_sizes: dict[str, np.ndarray],
) -> tuple[MultiSphereOOSDetector, list[dict[str, Any]]]:
    """拟合保持 KMeans 簇大小的随机分区 detector。"""

    detector = _build_detector(k, distance)
    normalized = detector._normalize_embeddings(np.asarray(train_embeddings))
    intents = np.asarray(train_intents, dtype=object)
    global_labels = np.full(len(intents), -1, dtype=np.int64)
    centers: list[np.ndarray] = []
    detector.intent_to_clusters = {}
    detector.cluster_to_intent = {}
    detector.intent_to_cluster = {}
    next_id = 0
    rng = np.random.default_rng(1000 + int(repeat))
    quality_rows: list[dict[str, Any]] = []

    for intent in sorted(np.unique(intents).tolist()):
        indices = np.where(intents == intent)[0]
        points = normalized[indices]
        effective_k = min(int(k), max(1, len(points)))
        # 仅使用 KMeans 的簇大小，不复用其样本归属；大小在同一 dataset/seed/K
        # 下固定，跨五个 random repeat 缓存后再复用，避免把 KMeans 计算误算进
        # 随机对照的随机性。
        sizes = reference_sizes[str(intent)]
        shuffled = rng.permutation(indices)
        assigned: list[np.ndarray] = []
        cursor = 0
        cluster_ids: list[int] = []
        local_labels = np.empty(len(indices), dtype=np.int64)
        for local_id, size in enumerate(sizes.tolist()):
            member_indices = shuffled[cursor : cursor + int(size)]
            cursor += int(size)
            member_mask = np.isin(indices, member_indices)
            local_labels[member_mask] = local_id
            center = normalized[member_indices].mean(axis=0)
            cluster_ids.append(next_id)
            detector.cluster_to_intent[next_id] = str(intent)
            centers.append(center)
            next_id += 1
        global_labels[indices] = np.asarray([cluster_ids[int(local)] for local in local_labels], dtype=np.int64)
        detector.intent_to_clusters[str(intent)] = cluster_ids
        detector.intent_to_cluster[str(intent)] = cluster_ids[0]
        quality_rows.append(
            {
                "intent": str(intent),
                "support": int(len(indices)),
                "requested_k": int(k),
                "effective_k": int(effective_k),
                "minimum_cluster_ratio": float(np.min(sizes) / max(len(indices), 1)),
                "cluster_sizes": "|".join(str(int(value)) for value in sizes),
            }
        )

    center_array = np.asarray(centers, dtype=np.float64)
    detector.n_clusters = int(len(center_array))
    detector.kmeans = _StaticCenters(center_array, global_labels)
    detector._train_embeddings = normalized
    detector._train_cluster_labels = global_labels
    detector._compute_radii(normalized, global_labels)
    detector.fitted = True
    return detector, quality_rows


def _load_inputs(v19_root: Path, dataset: str, seed: int) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    cache_key = (dataset, int(seed))
    if cache_key in _INPUT_CACHE:
        return _INPUT_CACHE[cache_key]
    reference = _unit(v19_root, "fixed", dataset, 50, seed, "euclidean", 1)
    manifest = _read_json(reference / "run_manifest.json")
    data_root = Path(manifest["data_root"])
    rows = {split: _read_json(data_root / "gate" / f"{split}.json") for split in ("train", "val", "test")}
    embeddings = {
        split: _load_cache(v19_root, dataset, 50, seed, split, manifest) for split in rows
    }
    known_manifest = _read_json(data_root / "KNOWN_INTENTS.json")
    _INPUT_CACHE[cache_key] = (known_manifest, rows, embeddings)
    return _INPUT_CACHE[cache_key]


def _score_split_fast(
    detector: MultiSphereOOSDetector,
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    known_manifest: dict[str, Any],
    split: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """向量化 random-control scoring，避免逐样本 Python 循环成为瓶颈。"""

    distances = _all_sphere_distances(detector, embeddings)
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(rows)), nearest]
    radii = np.asarray([sphere.radius for sphere in detector.spheres], dtype=np.float64)
    intents = [str(sphere.intent_name) for sphere in detector.spheres]
    scores = nearest_distance / np.maximum(radii[nearest], 1e-12)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    prediction = (scores > 1.0).astype(np.int64)
    metrics = compute_binary_oos_metrics(labels, scores, 1.0)
    coverage = compute_coverage_counts(distances, radii, intents)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source = _oos_source(row, known_manifest)
        gold_kind = gold_sample_kind(
            {**dict(row), "oos_source": source, "true_binary_label": int(row["label"])},
            known_manifest,
        )
        sphere = detector.spheres[int(nearest[index])]
        records.append(
            {
                "sample_id": _sample_id(row, split, index),
                "true_binary_label": int(labels[index]),
                "true_intent": str(row["intent"]),
                "oos_source": source,
                "gold_sample_kind": gold_kind,
                "source_split": str(row.get("source_split", row.get("split", split))),
                "score": float(scores[index]),
                "prediction": int(prediction[index]),
                "nearest_intent": str(sphere.intent_name),
                "nearest_cluster": int(sphere.cluster_id),
                "distance": float(nearest_distance[index]),
                "radius": float(radii[nearest[index]]),
                "coverage_count": int(coverage[index]),
            }
        )
    metrics.update(
        {
            "sample_count": int(len(rows)),
            "known_count": int(np.sum(labels == 0)),
            "oos_count": int(np.sum(labels == 1)),
            "known_multi_coverage_rate": float(np.mean(coverage[labels == 0] >= 2)),
            "oos_multi_coverage_rate": float(np.mean(coverage[labels == 1] >= 2)),
        }
    )
    return metrics, pd.DataFrame.from_records(records)


def _run_one(v19_root: Path, output_root: Path, dataset: str, seed: int, distance: str, k: int, repeat: int) -> dict[str, Any]:
    existing_unit = output_root / "random_partition" / dataset / f"kir50_seed{seed}" / distance / f"k{k}" / f"repeat{repeat}"
    existing_eval = existing_unit / "eval_results.json"
    existing_metrics = existing_unit / "cluster_metrics.csv"
    if existing_eval.is_file() and (existing_unit / "scores.parquet").is_file() and existing_metrics.is_file():
        payload = _read_json(existing_eval)
        quality = pd.read_csv(existing_metrics)
        test = payload["test"]
        return {
            "dataset": dataset,
            "kir": 50,
            "data_seed": seed,
            "distance": distance,
            "k_gate": k,
            "random_repeat": repeat,
            "test_oos_f1": test["oos_f1"],
            "test_id_recall": test["id_recall"],
            "test_auroc": test["auroc"],
            "test_fpr95": test["fpr95"],
            "known_multi_coverage_rate": test["known_multi_coverage_rate"],
            "oos_multi_coverage_rate": test["oos_multi_coverage_rate"],
            "minimum_cluster_ratio": float(quality["minimum_cluster_ratio"].min()),
        }
    known_manifest, rows, embeddings = _load_inputs(v19_root, dataset, seed)
    train_intents = np.asarray([str(row["intent"]) for row in rows["train"]], dtype=object)
    cache_key = (dataset, int(seed), int(k))
    if cache_key not in _REFERENCE_SIZE_CACHE:
        reference_sizes: dict[str, np.ndarray] = {}
        normalized = _build_detector(k, distance)._normalize_embeddings(embeddings["train"])
        for intent in sorted(np.unique(train_intents).tolist()):
            points = normalized[train_intents == intent]
            effective_k = min(int(k), max(1, len(points)))
            if effective_k == 1:
                sizes = np.asarray([len(points)], dtype=np.int64)
            else:
                reference = KMeans(n_clusters=effective_k, random_state=42, n_init=10).fit(points)
                sizes = np.sort(np.bincount(reference.labels_, minlength=effective_k))[::-1]
            reference_sizes[str(intent)] = sizes
        _REFERENCE_SIZE_CACHE[cache_key] = reference_sizes
    detector, quality = _random_partition_detector(
        embeddings["train"], train_intents, k, distance, repeat, _REFERENCE_SIZE_CACHE[cache_key]
    )
    val_metrics, val_scores = _score_split_fast(detector, rows["val"], embeddings["val"], known_manifest, "val")
    test_metrics, test_scores = _score_split_fast(detector, rows["test"], embeddings["test"], known_manifest, "test")
    unit = output_root / "random_partition" / dataset / f"kir50_seed{seed}" / distance / f"k{k}" / f"repeat{repeat}"
    unit.mkdir(parents=True, exist_ok=True)
    val_scores.to_parquet(unit / "validation_scores.parquet", index=False)
    test_scores.to_parquet(unit / "scores.parquet", index=False)
    _write_csv(unit / "cluster_metrics.csv", quality)
    _write_json(unit / "eval_results.json", {"protocol": "cluster_separability_v20_random_partition", "validation": val_metrics, "test": test_metrics})
    _write_json(unit / "run_manifest.json", {
        "status": "complete",
        "source_v19_root": str(v19_root),
        "dataset": dataset,
        "kir": 50,
        "data_seed": seed,
        "distance": distance,
        "k_gate": k,
        "random_repeat": repeat,
        "partition_seed": 1000 + repeat,
        "partition_size_source": "same_per_intent_KMeans_cluster_size_multiset",
        "kmeans_random_state_for_size_reference": 42,
        "radius_lambda": 1.0,
        "decision_threshold": 1.0,
        "v19_frozen": True,
    })
    return {
        "dataset": dataset,
        "kir": 50,
        "data_seed": seed,
        "distance": distance,
        "k_gate": k,
        "random_repeat": repeat,
        "test_oos_f1": test_metrics["oos_f1"],
        "test_id_recall": test_metrics["id_recall"],
        "test_auroc": test_metrics["auroc"],
        "test_fpr95": test_metrics["fpr95"],
        "known_multi_coverage_rate": test_metrics["known_multi_coverage_rate"],
        "oos_multi_coverage_rate": test_metrics["oos_multi_coverage_rate"],
        "minimum_cluster_ratio": min(float(row["minimum_cluster_ratio"]) for row in quality),
    }


def run(v19_root: Path = V19_ROOT, v20_root: Path = V20_ROOT) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for distance in DISTANCES:
                selected = _selected_k(v19_root, dataset, seed, distance)
                requested = sorted({2, selected})
                for k in requested:
                    if k <= 1:
                        skipped.append({"dataset": dataset, "data_seed": seed, "distance": distance, "selected_k": selected, "reason": "selected_k_is_1"})
                        continue
                    for repeat in (1, 2, 3, 4, 5):
                        rows.append(_run_one(v19_root, v20_root, dataset, seed, distance, k, repeat))
    _write_csv(v20_root / "random_partition_by_repeat.csv", rows)
    kmeans_rows: list[dict[str, Any]] = []
    for key, group in pd.DataFrame(rows).groupby(["dataset", "data_seed", "distance", "k_gate"]):
        dataset, seed, distance, k = key
        fixed = _read_json(_unit(v19_root, "fixed", dataset, 50, seed, distance, int(k)) / "eval_results.json")
        random_f1 = group["test_oos_f1"].astype(float)
        random_id = group["test_id_recall"].astype(float)
        kmeans_rows.append({
            "dataset": dataset,
            "kir": 50,
            "data_seed": int(seed),
            "distance": distance,
            "k_gate": int(k),
            "random_repeat_count": len(group),
            "kmeans_test_oos_f1": float(fixed["test"]["oos_f1"]),
            "random_test_oos_f1_mean": float(random_f1.mean()),
            "random_test_oos_f1_std": float(random_f1.std(ddof=1)),
            "delta_semantic_kmeans_minus_random": float(fixed["test"]["oos_f1"] - random_f1.mean()),
            "random_test_id_recall_mean": float(random_id.mean()),
            "kmeans_test_id_recall": float(fixed["test"]["id_recall"]),
            "random_test_auroc_mean": float(group["test_auroc"].mean()),
            "random_test_fpr95_mean": float(group["test_fpr95"].mean()),
            "random_minimum_cluster_ratio_mean": float(group["minimum_cluster_ratio"].mean()),
        })
    _write_csv(v20_root / "random_partition_summary.csv", kmeans_rows)
    manifest = {
        "protocol": "cluster_separability_v20_random_partition",
        "source_v19_root": str(v19_root),
        "output_root": str(v20_root),
        "completed_random_units": len(rows),
        "summary_rows": len(kmeans_rows),
        "skipped": skipped,
        "v19_frozen": True,
    }
    _write_json(v20_root / "random_partition_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KMeans vs random partition control")
    parser.add_argument("--v19-root", default=str(V19_ROOT))
    parser.add_argument("--v20-root", default=str(V20_ROOT))
    args = parser.parse_args()
    print(json.dumps(run(Path(args.v19_root), Path(args.v20_root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
