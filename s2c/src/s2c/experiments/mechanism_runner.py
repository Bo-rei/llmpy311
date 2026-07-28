"""Independent E3 mechanism diagnostics for the active protocol.

E3 reuses the frozen E2 embeddings and data views but writes to a separate
artifact root.  The only test-set operation is the declared partition-control
Gate evaluation; stability and reliability features use train/calibration only.
No function in this module can resume or overwrite an E2 run directory.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from s2c.data.hashing import atomic_write_json, atomic_write_text, sha256_file, sha256_json
from s2c.data.manifests import dataset_manifest_path, read_json
from s2c.evaluation.metrics import compute_binary_oos_metrics
from s2c.experiments.partitions import (
    PartitionResult,
    build_partition,
    fit_injected_detector,
    normalize_for_detector,
    partition_sizes,
)
from s2c.gate.view_loader import GateViews, load_gate_views
from s2c.runtime.paths import ProtocolV2Paths
from s2c.tracking.run_manifest import atomic_run_directory

E3_ROOT_NAME = "e3_mechanisms"
PARTITIONS = ("kmeans", "random_balanced")
DIAGNOSTIC_PARTITION_SEEDS = (0, 1, 2, 3, 4, 42, 87, 100, 123, 20260725)


@dataclass(frozen=True)
class E3Bundle:
    """Frozen arrays and views loaded from the already completed E2 cache."""

    dataset: str
    seed: int
    kir: float
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    views: GateViews
    e2_manifest: dict[str, Any]


@dataclass(frozen=True)
class PartitionControlSpec:
    dataset: str
    kir: float
    seed: int
    k: int
    distance: str
    partition: str
    partition_seed: int = 42

    @property
    def run_id(self) -> str:
        return (
            "protocol_v2_textoir_v1__e3_partition_control__"
            f"{self.dataset}__kir_{self.kir:.2f}__seed_{self.seed}__k_{self.k}__"
            f"dist_{self.distance}__partition_{self.partition}__pseed_{self.partition_seed}"
        )


@dataclass(frozen=True)
class DiagnosticGroup:
    dataset: str
    kir: float
    seed: int
    k: int

    @property
    def group_id(self) -> str:
        return f"{self.dataset}__kir_{self.kir:.2f}__seed_{self.seed}__k_{self.k}"


def e3_root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / E3_ROOT_NAME


def _atomic_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _e2_run_dir(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float, distance: str) -> Path:
    name = (
        f"protocol_v2_textoir_v1__{dataset}__kir_{kir:.2f}__seed_{seed}__"
        f"repr_frozen_minilm__k_1__dist_{distance}__boundary_mean_std"
    )
    path = paths.run_root / "e2_gate_core_dense" / name
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(f"E2 reference run is missing: {path}")
    return path


def _cache_path(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float, split: str, cache_key_sha: str) -> Path:
    prefix = cache_key_sha[:20]
    if split == "canonical":
        return paths.embedding_cache_root / "canonical" / dataset / f"canonical_{prefix}.npz"
    return paths.embedding_cache_root / dataset / f"seed_{seed}" / f"kir_{kir:.2f}" / f"{split}_{prefix}.npz"


def _load_cached_array(
    paths: ProtocolV2Paths,
    dataset: str,
    seed: int,
    kir: float,
    split: str,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an existing E2 cache; never encode or silently rebuild it."""

    key_sha = str(metadata.get("cache_key_sha256", ""))
    if not key_sha:
        raise ValueError(f"E2 cache metadata has no cache key: dataset={dataset}, seed={seed}, split={split}")
    cache_path = _cache_path(paths, dataset, seed, kir, split, key_sha)
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"E2 embedding cache is missing: {cache_path}")
    actual_metadata = read_json(metadata_path)
    if actual_metadata.get("cache_key_sha256") != key_sha:
        raise ValueError(f"E2 cache key mismatch: {cache_path}")
    if actual_metadata.get("embedding_sha256") != metadata.get("embedding_sha256"):
        raise ValueError(f"E2 embedding hash mismatch: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as payload:
        values = np.asarray(payload["embeddings"], dtype=np.float32)
    if list(values.shape) != list(metadata.get("embedding_shape", values.shape)):
        raise ValueError(f"E2 embedding shape mismatch: {cache_path}")
    return values, actual_metadata


def _load_canonical_calibration(
    paths: ProtocolV2Paths,
    dataset: str,
    seed: int,
    kir: float,
    canonical_metadata: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
) -> np.ndarray:
    """Project calibration sample IDs into the frozen canonical cache."""

    cache_key_sha = str(canonical_metadata["cache_key_sha256"])
    cache_path = _cache_path(paths, dataset, seed, kir, "canonical", cache_key_sha)
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Canonical E2 cache is missing: {cache_path}")
    actual_metadata = read_json(metadata_path)
    if actual_metadata.get("embedding_sha256") != canonical_metadata.get("embedding_sha256"):
        raise ValueError(f"Canonical E2 embedding hash mismatch: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as payload:
        values = np.asarray(payload["embeddings"], dtype=np.float32)
        sample_ids = np.asarray(payload["sample_ids"], dtype="U64")
    index = {str(sample_id): i for i, sample_id in enumerate(sample_ids.tolist())}
    try:
        indices = [index[str(row["sample_id"])] for row in calibration_rows]
    except KeyError as exc:
        raise KeyError(f"Calibration sample missing from canonical E2 cache: {exc.args[0]}") from exc
    return np.ascontiguousarray(values[np.asarray(indices, dtype=np.int64)])


def load_e2_bundle(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> E3Bundle:
    """Load one dataset/KIR/seed without any textoir or re-encoding fallback."""

    e2_dir = _e2_run_dir(paths, dataset, seed, kir, "euclidean")
    e2_manifest = read_json(e2_dir / "manifest.json")
    if e2_manifest.get("protocol_version") != paths.dataset_version:
        raise ValueError(f"E2 protocol mismatch: {e2_dir}")
    views = load_gate_views(paths, dataset, seed, kir)
    cache = e2_manifest.get("embedding_cache", {})
    train, _ = _load_cached_array(paths, dataset, seed, kir, "train_known", cache["train"])
    test, _ = _load_cached_array(paths, dataset, seed, kir, "test_combined", cache["test"])
    calibration = _load_canonical_calibration(
        paths,
        dataset,
        seed,
        kir,
        cache["canonical"],
        views.calibration,
    )
    if train.shape[0] != len(views.train) or test.shape[0] != len(views.test):
        raise ValueError(f"E2 cache/view count mismatch: dataset={dataset}, seed={seed}, KIR={kir}")
    return E3Bundle(dataset, seed, kir, train, calibration, test, views, e2_manifest)


def _metrics_breakdown(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float = 1.0) -> dict[str, dict[str, float]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    result = {"combined": compute_binary_oos_metrics(labels, scores, threshold)}
    known = np.flatnonzero(labels == 0)
    for source in ("heldout_intent", "native"):
        oos = np.asarray([i for i, row in enumerate(rows) if row.get("oos_source") == source], dtype=np.int64)
        if oos.size:
            selected = np.concatenate((known, oos))
            result[source] = compute_binary_oos_metrics(labels[selected], scores[selected], threshold)
    return result


def _config_payload(spec: PartitionControlSpec) -> dict[str, Any]:
    return {
        "stage": "E3-A",
        "protocol_version": "protocol_v2_textoir_v1",
        "dataset": spec.dataset,
        "kir": spec.kir,
        "seed": spec.seed,
        "k_gate": spec.k,
        "distance": spec.distance,
        "partition": spec.partition,
        "partition_seed": spec.partition_seed,
        "representation": "frozen_minilm",
        "boundary": "mean_std",
        "radius_lambda": 1.0,
        "score_direction": "higher_is_more_oos",
        "selection": "fixed_boundary_known_only_calibration",
    }


def _control_run_dir(paths: ProtocolV2Paths, spec: PartitionControlSpec) -> Path:
    return e3_root(paths) / "runs" / spec.run_id


def _require_e3_provenance(paths: ProtocolV2Paths) -> tuple[str, str]:
    snapshot = e3_root(paths) / "E3_PROVENANCE_SNAPSHOT.json"
    patch = e3_root(paths) / "E3_CODE_SNAPSHOT.patch"
    if not snapshot.is_file() or not patch.is_file():
        raise RuntimeError("E3 provenance is not frozen; run E3-0 before formal experiments")
    return sha256_file(snapshot), sha256_file(patch)


def _e2_closeout_manifest(paths: ProtocolV2Paths) -> Path:
    return e3_root(paths).parent / "summaries" / "e2_closeout" / "E2_closeout_manifest.json"


def _write_control_run(paths: ProtocolV2Paths, spec: PartitionControlSpec, bundle: E3Bundle) -> Path:
    run_dir = _control_run_dir(paths, spec)
    config = _config_payload(spec)
    config_hash = sha256_json(config)
    if run_dir.exists():
        existing = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").is_file() else {}
        if existing.get("config_hash") == config_hash and existing.get("status") == "complete":
            return run_dir
        raise RuntimeError(f"Refusing to overwrite E3 run: {run_dir}")
    normalized_train = normalize_for_detector(bundle.train)
    train_intents = np.asarray([str(row["intent"]) for row in bundle.views.train], dtype=object)
    partition_result = build_partition(normalized_train, train_intents, spec.k, spec.partition, spec.partition_seed)
    detector = fit_injected_detector(
        normalized_train,
        train_intents,
        partition_result,
        distance=spec.distance,
        radius_lambda=1.0,
        random_state=spec.partition_seed,
    )
    test_scores_started = time.perf_counter()
    test_output = detector.predict_with_scores(bundle.test)
    elapsed = time.perf_counter() - test_scores_started
    scores = np.asarray(test_output["score"], dtype=np.float64)
    metrics = _metrics_breakdown(bundle.views.test, scores)
    metrics["combined"].update(
        {
            "scoring_seconds": elapsed,
            "samples_per_second": len(bundle.views.test) / elapsed if elapsed else float("inf"),
            "effective_cluster_count": partition_result.cluster_count,
            "minimum_cluster_size": int(partition_sizes(partition_result).min()),
        }
    )
    e2_manifest = bundle.e2_manifest
    e3_snapshot_sha256, e3_patch_sha256 = _require_e3_provenance(paths)
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_text(temporary / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=True))
        atomic_write_json(temporary / "metrics.json", {"combined": metrics["combined"], "oos_breakdown": metrics})
        atomic_write_json(
            temporary / "partition.json",
            {
                "partition": spec.partition,
                "partition_seed": spec.partition_seed,
                "cluster_count": partition_result.cluster_count,
                "cluster_sizes": partition_sizes(partition_result).tolist(),
            },
        )
        atomic_write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "stage": "E3-A",
                "run_id": spec.run_id,
                "protocol_version": paths.dataset_version,
                "config": config,
                "config_hash": config_hash,
                "e2_reference_run_id": e2_manifest.get("run_id"),
                "e2_reference_manifest_sha256": sha256_file(_e2_run_dir(paths, spec.dataset, spec.seed, spec.kir, "euclidean") / "manifest.json"),
                "e3_provenance_snapshot_sha256": e3_snapshot_sha256,
                "e3_code_snapshot_sha256": e3_patch_sha256,
                "canonical_manifest_sha256": e2_manifest.get("canonical_manifest_sha256"),
                "registry_sha256": e2_manifest.get("registry_sha256"),
                "embedding_cache": e2_manifest.get("embedding_cache"),
                "test_used_for_selection": False,
                "historical_artifacts_overwritten": False,
                "generated_at": datetime.now(UTC).isoformat(),
            },
        )
    return run_dir


def partition_control_specs(config_path: Path) -> list[PartitionControlSpec]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [
        PartitionControlSpec(dataset, float(kir), int(seed), int(k), str(distance), str(partition))
        for dataset in payload["datasets"]
        for kir in payload["kirs"]
        for seed in payload["seeds"]
        for k in payload["k_values"]
        for partition in payload["partitions"]
        for distance in payload["distances"]
    ]


def _state_path(paths: ProtocolV2Paths, name: str) -> Path:
    return e3_root(paths) / "state" / f"{name}.state.json"


def run_partition_control(
    paths: ProtocolV2Paths,
    specs: list[PartitionControlSpec],
    *,
    dry_run: bool,
    resume: bool,
    state_name: str = "e3_partition_control",
) -> tuple[int, list[dict[str, str]]]:
    root = e3_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plans" / "e3_partition_control.plan.json"
    _atomic_json(plan_path, {"stage": "E3-A", "run_count": len(specs), "runs": [_config_payload(s) for s in specs]})
    state_path = _state_path(paths, state_name)
    completed = 0
    failed: list[dict[str, str]] = []
    if not dry_run:
        _require_e3_provenance(paths)
    bundles: dict[tuple[str, int, float], E3Bundle] = {}
    for spec in specs:
        run_dir = _control_run_dir(paths, spec)
        config_hash = sha256_json(_config_payload(spec))
        if resume and (run_dir / "manifest.json").is_file():
            existing = read_json(run_dir / "manifest.json")
            if existing.get("status") == "complete" and existing.get("config_hash") == config_hash:
                completed += 1
                continue
        if dry_run:
            completed += 1
            continue
        try:
            key = (spec.dataset, spec.seed, spec.kir)
            if key not in bundles:
                bundles[key] = load_e2_bundle(paths, *key)
            _write_control_run(paths, spec, bundles[key])
            completed += 1
        except Exception as exc:  # Keep independent cells resumable.
            failed.append({"run_id": spec.run_id, "error_type": type(exc).__name__, "error": str(exc)})
        _atomic_json(
            state_path,
            {
                "stage": "E3-A",
                "planned": len(specs),
                "completed": completed,
                "failed": failed,
                "last_run_id": spec.run_id,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
    return completed, failed


def _tiny_thresholds(intent_count: int) -> tuple[int, int, int]:
    return (3, 5, max(5, int(math.ceil(0.05 * intent_count))))


def _pairwise_ari(label_sets: list[np.ndarray]) -> tuple[float, float, float]:
    values = [
        adjusted_rand_score(left, right)
        for index, left in enumerate(label_sets)
        for right in label_sets[index + 1 :]
    ]
    if not values:
        return math.nan, math.nan, math.nan
    return float(np.mean(values)), float(np.median(values)), float(np.min(values))


def _centroid_drift(results: list[PartitionResult]) -> float:
    if len(results) < 2:
        return math.nan
    displacements: list[float] = []
    reference = results[0]
    for current in results[1:]:
        for intent, ref_ids in reference.intent_to_clusters.items():
            cur_ids = current.intent_to_clusters[intent]
            ref = reference.centers[list(ref_ids)]
            cur = current.centers[list(cur_ids)]
            cost = np.linalg.norm(ref[:, None, :] - cur[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            displacements.extend(cost[rows, cols].tolist())
    return float(np.mean(displacements)) if displacements else math.nan


def _silhouette(points: np.ndarray, labels: np.ndarray, seed: int) -> tuple[float, float, float, str | None]:
    if np.unique(labels).size < 2:
        return math.nan, math.nan, math.nan, "fewer_than_two_clusters"
    max_points = min(points.shape[0], 2000)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(points.shape[0], size=max_points, replace=False)) if max_points < points.shape[0] else np.arange(points.shape[0])
    try:
        values = silhouette_samples(points[indices], labels[indices], metric="euclidean")
    except ValueError as exc:
        return math.nan, math.nan, math.nan, str(exc)
    return float(np.mean(values)), float(np.min(values)), float(np.quantile(values, 0.10)), None


def _diagnostic_row(
    bundle: E3Bundle,
    partition_result: PartitionResult,
    partition: str,
    partition_seed: int,
    distance: str,
    stable_labels: list[np.ndarray],
    stable_results: list[PartitionResult],
    k1_features: dict[str, dict[str, float]],
) -> dict[str, Any]:
    train = normalize_for_detector(bundle.train)
    intents = np.asarray([str(row["intent"]) for row in bundle.views.train], dtype=object)
    detector = fit_injected_detector(train, intents, partition_result, distance=distance, radius_lambda=1.0, random_state=partition_seed)
    calibration_output = detector.predict_with_scores(bundle.calibration)
    calibration_pred = np.asarray(calibration_output["score"], dtype=float) > 1.0
    sizes = partition_sizes(partition_result)
    # The percentage threshold is defined per intent, not by the entire
    # known-training table.  Global counts remain useful for the run-level
    # audit, while intent-level features below use the exact local count.
    tiny3, tiny5, tiny5pct = _tiny_thresholds(train.shape[0])
    radii = np.asarray([sphere.radius for sphere in detector.spheres], dtype=float)
    silhouette_mean, silhouette_min, silhouette_p10, silhouette_reason = _silhouette(train, partition_result.labels, partition_seed)
    ari_mean, ari_median, ari_min = _pairwise_ari(stable_labels)
    row: dict[str, Any] = {
        "dataset": bundle.dataset,
        "kir": bundle.kir,
        "seed": bundle.seed,
        "k": max(len(cluster_ids) for cluster_ids in partition_result.intent_to_clusters.values()),
        "partition": partition,
        "partition_seed": partition_seed,
        "distance": distance,
        "intent_count": len(partition_result.intent_to_clusters),
        "cluster_count": partition_result.cluster_count,
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "cluster_size_cv": float(np.std(sizes) / np.mean(sizes)) if sizes.size and np.mean(sizes) else math.nan,
        "tiny_cluster_count_lt3": int(np.sum(sizes < tiny3)),
        "tiny_cluster_count_lt5": int(np.sum(sizes < tiny5)),
        "tiny_cluster_count_lt5pct": int(np.sum(sizes < tiny5pct)),
        "tiny_cluster_ratio": float(np.mean(sizes < tiny5pct)) if sizes.size else math.nan,
        "silhouette_mean": silhouette_mean,
        "silhouette_min": silhouette_min,
        "silhouette_p10": silhouette_p10,
        "silhouette_unavailable_reason": silhouette_reason,
        "pairwise_ari_mean": ari_mean,
        "pairwise_ari_median": ari_median,
        "pairwise_ari_min": ari_min,
        "centroid_drift": _centroid_drift(stable_results),
        "radius_mean": float(np.mean(radii)) if radii.size else math.nan,
        "radius_std": float(np.std(radii)) if radii.size else math.nan,
        "radius_cv": float(np.std(radii) / np.mean(radii)) if radii.size and np.mean(radii) else math.nan,
        "calibration_coverage": float(np.mean(~calibration_pred)) if calibration_pred.size else math.nan,
        "calibration_false_rejection": float(np.mean(calibration_pred)) if calibration_pred.size else math.nan,
        "coverage_sample_count": int(calibration_pred.size),
    }
    # Per-intent known-only features are intentionally kept in the same row
    # family; the summary module expands them into the public feature table.
    intent_rows: list[dict[str, Any]] = []
    for intent in sorted(partition_result.intent_to_clusters):
        mask = intents == intent
        intent_sizes = np.asarray([np.sum(partition_result.labels[mask] == cid) for cid in partition_result.intent_to_clusters[intent]], dtype=int)
        calibration_mask = np.asarray([str(row["intent"]) == intent for row in bundle.views.calibration], dtype=bool)
        intent_rows.append(
            {
                "intent": intent,
                "intent_train_count": int(mask.sum()),
                "min_cluster_size": int(intent_sizes.min()),
                "min_cluster_ratio": float(intent_sizes.min() / mask.sum()),
                "tiny_cluster_ratio": float(np.mean(intent_sizes < max(5, int(math.ceil(0.05 * mask.sum()))))),
                "calibration_coverage": float(np.mean(~calibration_pred[calibration_mask])) if calibration_mask.any() else math.nan,
                "calibration_false_rejection": float(np.mean(calibration_pred[calibration_mask])) if calibration_mask.any() else math.nan,
                "radius_mean": float(np.mean([detector.spheres[cid].radius for cid in partition_result.intent_to_clusters[intent]])),
                "radius_cv": float(np.std([detector.spheres[cid].radius for cid in partition_result.intent_to_clusters[intent]]) / np.mean([detector.spheres[cid].radius for cid in partition_result.intent_to_clusters[intent]])) if len(intent_sizes) > 1 else 0.0,
                "cluster_size_cv": float(np.std(intent_sizes) / np.mean(intent_sizes)) if len(intent_sizes) > 1 else 0.0,
                "dominant_cluster_ratio": float(intent_sizes.max() / mask.sum()),
            }
        )
        centers = partition_result.centers[list(partition_result.intent_to_clusters[intent])]
        assigned = partition_result.labels[mask]
        cluster_ids = partition_result.intent_to_clusters[intent]
        local_wcss = float(
            sum(
                np.sum((train[mask][assigned == cluster_id] - partition_result.centers[cluster_id]) ** 2)
                for cluster_id in cluster_ids
            )
        )
        class_center = train[mask].mean(axis=0)
        class_wcss = float(np.sum((train[mask] - class_center) ** 2))
        if len(cluster_ids) > 1:
            pairwise_centers = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            nonzero = pairwise_centers[np.triu_indices(len(cluster_ids), k=1)]
            centroid_separation = float(nonzero.min()) if nonzero.size else math.nan
        else:
            centroid_separation = math.nan
        radius_mean = intent_rows[-1]["radius_mean"]
        intent_rows[-1].update(
            {
                "compactness_gain": float(1.0 - local_wcss / max(class_wcss, 1e-12)),
                "centroid_separation": centroid_separation,
                "separation_radius_ratio": centroid_separation / max(float(radius_mean), 1e-12)
                if np.isfinite(centroid_separation)
                else math.nan,
                "k1_calibration_coverage": k1_features[intent]["calibration_coverage"],
                "k1_calibration_false_rejection": k1_features[intent]["calibration_false_rejection"],
                "coverage_drop_vs_k1": intent_rows[-1]["calibration_coverage"] - k1_features[intent]["calibration_coverage"],
                "false_rejection_increase_vs_k1": intent_rows[-1]["calibration_false_rejection"]
                - k1_features[intent]["calibration_false_rejection"],
            }
        )
    row["intent_features"] = intent_rows
    return row


def diagnostic_groups(config_path: Path) -> list[DiagnosticGroup]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [
        DiagnosticGroup(str(dataset), float(kir), int(seed), int(k))
        for dataset in payload["datasets"]
        for kir in payload["kirs"]
        for seed in payload["seeds"]
        for k in payload["k_values"]
    ]


def diagnostic_partition_seeds(config_path: Path) -> tuple[int, ...]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = tuple(int(seed) for seed in payload.get("partition_seeds", DIAGNOSTIC_PARTITION_SEEDS))
    if not seeds:
        raise ValueError("E3 diagnostics require at least one partition seed")
    return seeds


def _known_only_baseline_features(bundle: E3Bundle, distance: str) -> dict[str, dict[str, float]]:
    """Compute the K=1 calibration reference without reading any OOS rows."""

    train = normalize_for_detector(bundle.train)
    intents = np.asarray([str(row["intent"]) for row in bundle.views.train], dtype=object)
    baseline_partition = build_partition(train, intents, 1, "kmeans", 42)
    detector = fit_injected_detector(train, intents, baseline_partition, distance=distance, radius_lambda=1.0)
    calibration_scores = np.asarray(detector.predict_with_scores(bundle.calibration)["score"], dtype=float)
    rejected = calibration_scores > 1.0
    features: dict[str, dict[str, float]] = {}
    for intent in sorted(baseline_partition.intent_to_clusters):
        mask = np.asarray([str(row["intent"]) == intent for row in bundle.views.calibration], dtype=bool)
        features[intent] = {
            "calibration_coverage": float(np.mean(~rejected[mask])) if mask.any() else math.nan,
            "calibration_false_rejection": float(np.mean(rejected[mask])) if mask.any() else math.nan,
        }
    return features


def run_cluster_diagnostics(
    paths: ProtocolV2Paths,
    groups: list[DiagnosticGroup],
    *,
    partition_seeds: tuple[int, ...] = DIAGNOSTIC_PARTITION_SEEDS,
    resume: bool = True,
    state_name: str = "e3_cluster_diagnostics",
) -> tuple[int, list[dict[str, str]]]:
    root = e3_root(paths) / "diagnostics" / "groups"
    root.mkdir(parents=True, exist_ok=True)
    _require_e3_provenance(paths)
    state_path = _state_path(paths, state_name)
    completed = 0
    failed: list[dict[str, str]] = []
    bundles: dict[tuple[str, int, float], E3Bundle] = {}
    for group in groups:
        output_path = root / f"{group.group_id}.json"
        if resume and output_path.is_file():
            completed += 1
            continue
        try:
            key = (group.dataset, group.seed, group.kir)
            if key not in bundles:
                bundles[key] = load_e2_bundle(paths, *key)
            bundle = bundles[key]
            train = normalize_for_detector(bundle.train)
            intents = np.asarray([str(row["intent"]) for row in bundle.views.train], dtype=object)
            k1_by_distance = {
                distance: _known_only_baseline_features(bundle, distance)
                for distance in ("euclidean", "mahalanobis_diag")
            }
            rows: list[dict[str, Any]] = []
            for partition in PARTITIONS:
                partitions = [build_partition(train, intents, group.k, partition, seed) for seed in partition_seeds]
                labels = [p.labels for p in partitions]
                for partition_seed, result in zip(partition_seeds, partitions, strict=True):
                    for distance in ("euclidean", "mahalanobis_diag"):
                        rows.append(
                            _diagnostic_row(
                                bundle,
                                result,
                                partition,
                                partition_seed,
                                distance,
                                labels,
                                partitions,
                                k1_by_distance[distance],
                            )
                        )
            _atomic_json(output_path, {"group": group.__dict__, "rows": rows})
            completed += 1
        except Exception as exc:
            failed.append({"group_id": group.group_id, "error_type": type(exc).__name__, "error": str(exc)})
        _atomic_json(state_path, {"stage": "E3-B/C", "planned": len(groups), "completed": completed, "failed": failed, "last_group": group.group_id, "updated_at": datetime.now(UTC).isoformat()})
    return completed, failed


def _git_patch() -> bytes:
    repo_root = Path(
        subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
    )
    tracked = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=repo_root, capture_output=True, check=True
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    chunks = [tracked]
    for path in untracked:
        if path.startswith("artifacts/") or path.startswith("s2c/data/"):
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "/dev/null", path],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if result.stdout:
            chunks.append(result.stdout)
    return b"\n".join(chunks)


def freeze_provenance(paths: ProtocolV2Paths, config_paths: Iterable[Path]) -> Path:
    root = e3_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    patch_path = root / "E3_CODE_SNAPSHOT.patch"
    patch = _git_patch()
    atomic_write_text(patch_path, patch.decode("utf-8", errors="replace"))
    canonical = {d: sha256_file(dataset_manifest_path(paths.manifest_root, d)) for d in ("clinc150", "banking77", "stackoverflow")}
    registry_files = sorted(paths.registry_root.glob("*/seed_*/kir_*.json"))
    registry_tree = sha256_json([(str(p.relative_to(paths.registry_root)), sha256_file(p)) for p in registry_files])
    e2_closeout = _e2_closeout_manifest(paths)
    snapshot = {
        "schema_version": "s2c.e3_provenance_snapshot.v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "protocol_version": paths.dataset_version,
        "stage": "E3",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()),
        "git_diff_sha256": sha256_file(patch_path),
        "config_sha256": {str(path): sha256_file(path) for path in config_paths},
        "e2_closeout_manifest_sha256": sha256_file(e2_closeout) if e2_closeout.is_file() else None,
        "canonical_manifest_sha256": canonical,
        "registry_tree_sha256": registry_tree,
        "registry_count": len(registry_files),
        "encoder": "all-MiniLM-L6-v2",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": __import__("sklearn").__version__,
        "bootstrap_rng_seed": 20260725,
        "partition_implementations_sha256": {
            "partitions.py": sha256_file(Path(__file__).with_name("partitions.py")),
            "mechanism_runner.py": sha256_file(Path(__file__)),
        },
    }
    _atomic_json(root / "E3_PROVENANCE_SNAPSHOT.json", snapshot)
    return root / "E3_PROVENANCE_SNAPSHOT.json"


def _main_partition(args: argparse.Namespace) -> int:
    paths = ProtocolV2Paths.discover()
    specs = partition_control_specs(args.config)
    if args.dataset:
        specs = [s for s in specs if s.dataset in args.dataset]
    if args.seed:
        specs = [s for s in specs if s.seed in args.seed]
    if args.kir:
        specs = [s for s in specs if s.kir in {round(v, 2) for v in args.kir}]
    if args.partition:
        specs = [s for s in specs if s.partition in args.partition]
    if args.distance:
        specs = [s for s in specs if s.distance in args.distance]
    if args.freeze_provenance:
        provenance_configs = [args.config, *args.provenance_config]
        freeze_provenance(paths, provenance_configs)
    completed, failed = run_partition_control(paths, specs, dry_run=args.dry_run, resume=args.resume)
    print(json.dumps({"planned": len(specs), "completed": completed, "failed": failed}, ensure_ascii=False))
    return 0 if not failed else 1


def _main_diagnostics(args: argparse.Namespace) -> int:
    paths = ProtocolV2Paths.discover()
    groups = diagnostic_groups(args.config)
    if args.dataset:
        groups = [g for g in groups if g.dataset in args.dataset]
    if args.seed:
        groups = [g for g in groups if g.seed in args.seed]
    completed, failed = run_cluster_diagnostics(
        paths,
        groups,
        partition_seeds=diagnostic_partition_seeds(args.config),
        resume=args.resume,
    )
    print(json.dumps({"planned": len(groups), "completed": completed, "failed": failed}, ensure_ascii=False))
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    partition = sub.add_parser("partition-control")
    partition.add_argument("--config", type=Path, required=True)
    partition.add_argument("--dry-run", action="store_true")
    partition.add_argument("--resume", action="store_true")
    partition.add_argument("--freeze-provenance", action="store_true")
    partition.add_argument("--provenance-config", type=Path, action="append", default=[])
    partition.add_argument("--dataset", action="append")
    partition.add_argument("--seed", type=int, action="append")
    partition.add_argument("--kir", type=float, action="append")
    partition.add_argument("--partition", action="append", choices=PARTITIONS)
    partition.add_argument("--distance", action="append", choices=("euclidean", "mahalanobis_diag"))
    partition.set_defaults(func=_main_partition)
    diagnostics = sub.add_parser("cluster-diagnostics")
    diagnostics.add_argument("--config", type=Path, required=True)
    diagnostics.add_argument("--resume", action="store_true")
    diagnostics.add_argument("--dataset", action="append")
    diagnostics.add_argument("--seed", type=int, action="append")
    diagnostics.set_defaults(func=_main_diagnostics)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
