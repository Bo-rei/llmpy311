"""Run a small, protocol-aligned MOGB baseline comparison.

The runner consumes the active protocol_v2 views and frozen MiniLM cache.  It
does not train BERT and it never regenerates a Known/OOS split.  The official
MOGB training path is intentionally a separate preflight because the upstream
repository has legacy dependencies and a different data contract.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_json
from protocol_v2.data.manifests import dataset_manifest_path, read_json, view_manifest_path
from protocol_v2.data.registry import registry_path
from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.experiments.matrix import GateRunSpec
from protocol_v2.experiments.mogb import (
    AdaptiveGranularBallClusterer,
    MOGBBoundary,
    balls_to_rows,
    make_mogb_boundaries,
    score_mogb_boundaries,
)
from protocol_v2.experiments.partitions import build_partition, fit_injected_detector, normalize_for_detector
from protocol_v2.gate.view_loader import GateViews, load_gate_views
from protocol_v2.runtime.paths import ProtocolV2Paths
from protocol_v2.tracking.provenance import file_hashes
from protocol_v2.tracking.run_manifest import atomic_run_directory, environment_snapshot

from protocol_v2.experiments.runner import (
    _canonical_embedding_cache,
    _embedding_cache,
    _model_fingerprint,
    _model_path,
)


METHODS = (
    "single_centroid",
    "random_partition",
    "fixed_k2",
    "mogb_minilm",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
    "mogb_official_reproduction",
)


def _load_cached_inputs(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> tuple[GateViews, np.ndarray, np.ndarray, dict[str, Any]]:
    paths.require_experiment_admission(dataset)
    views = load_gate_views(paths, dataset, seed, kir)
    registry = read_json(registry_path(paths, dataset, seed, kir))
    canonical_manifest = dataset_manifest_path(paths.manifest_root, dataset)
    input_hashes = file_hashes(
        {
            "registry": registry_path(paths, dataset, seed, kir),
            "canonical_manifest": canonical_manifest,
            "view_manifest": view_manifest_path(paths.manifest_root, dataset, seed, kir),
            "s2c_export_manifest": views.export_root / "export_manifest.json",
        }
    )
    model_path = _model_path(paths, "all-MiniLM-L6-v2")
    model = _model_fingerprint(model_path)
    # A MOGB fair run must consume the already materialized cache.  Passing no
    # encoder makes a cache miss fail loudly instead of silently encoding a new
    # corpus during a baseline comparison.
    try:
        canonical = _canonical_embedding_cache(paths, dataset, model, None, 128)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            f"Missing frozen MiniLM canonical cache for {dataset}; refusing implicit encoding"
        ) from exc
    spec = GateRunSpec(
        experiment_name="gate_core_dense",
        dataset=dataset,
        kir=kir,
        seed=seed,
        k_gate=2,
        distance="euclidean",
        representation="frozen_minilm",
        boundary="mean_std",
        radius_lambda=1.0,
        encoder_name="all-MiniLM-L6-v2",
        encoder_device="cpu",
        protocol_version=paths.dataset_version,
    )
    train, _ = _embedding_cache(
        paths, spec, "train_known", views.train, registry["registry_sha256"], input_hashes["canonical_manifest"], model, canonical
    )
    test, _ = _embedding_cache(
        paths, spec, "test_combined", views.test, registry["registry_sha256"], input_hashes["canonical_manifest"], model, canonical
    )
    return views, train, test, {
        "registry_sha256": registry["registry_sha256"],
        "canonical_manifest_sha256": input_hashes["canonical_manifest"],
        "view_manifest_sha256": input_hashes["view_manifest"],
        "export_manifest_sha256": input_hashes["s2c_export_manifest"],
        "model": model,
        "canonical_embedding_sha256": canonical.metadata.get("embedding_sha256"),
    }


def _open_metrics(rows: list[dict[str, Any]], output: dict[str, np.ndarray]) -> dict[str, float]:
    gold_binary = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    binary = compute_binary_oos_metrics(gold_binary, output["score"], threshold=1.0)
    gold = np.asarray(["oos" if int(row["label"]) else str(row["intent"]) for row in rows], dtype=object)
    predicted = np.asarray(output["predicted_label"], dtype=object)
    known_labels = sorted({str(row["intent"]) for row in rows if int(row["label"]) == 0})
    all_labels = known_labels + ["oos"]
    return {
        **binary,
        "accuracy": float(accuracy_score(gold, predicted)),
        "f1_all": float(f1_score(gold, predicted, labels=all_labels, average="macro", zero_division=0)),
        "f1_k": float(f1_score(gold, predicted, labels=known_labels, average="macro", zero_division=0)),
        "f1_u": float(f1_score(gold == "oos", predicted == "oos", average="binary", zero_division=0)),
    }


def _detector_output(detector: Any, train: np.ndarray, test: np.ndarray, rows: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    raw = detector.predict_with_scores(test)
    labels = np.asarray([detector.spheres[int(index)].intent_name for index in raw["nearest_cluster"]], dtype=object)
    labels[raw["pred"].astype(bool)] = "oos"
    output = {
        "score": np.asarray(raw["score"], dtype=np.float64),
        "predicted_oos": np.asarray(raw["pred"], dtype=np.int64),
        "nearest_ball": np.asarray(raw["nearest_cluster"], dtype=np.int64),
        "distance": np.asarray(raw["distance"], dtype=np.float64),
        "radius": np.asarray(raw["radius"], dtype=np.float64),
        "predicted_label": labels,
        "accepted_ball_count": np.asarray(raw["accepted_sphere_count"], dtype=np.int64),
    }
    balls = [
        MOGBBoundary(
            int(sphere.cluster_id),
            np.asarray(sphere.center, dtype=np.float64),
            float(sphere.radius),
            str(sphere.intent_name),
            np.flatnonzero(np.asarray(detector._train_cluster_labels) == int(sphere.cluster_id)),
            sphere.inv_diag_cov,
        )
        for sphere in detector.spheres
    ]
    return output, [
        {
            "ball_id": ball.ball_id,
            "parent_id": None,
            "depth": 0,
            "majority_label": ball.label,
            "purity": 1.0,
            "sample_count": int(ball.sample_indices.size),
            "radius": ball.radius,
            "selected": True,
            "stop_reason": "fixed_partition",
        }
        for ball in balls
    ]


def _run_method(method: str, train: np.ndarray, train_rows: list[dict[str, Any]], test: np.ndarray, test_rows: list[dict[str, Any]], seed: int) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    train_intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    normalized_train = normalize_for_detector(train)
    normalized_test = normalize_for_detector(test)
    if method == "mogb_official_reproduction":
        raise RuntimeError("official_source_unrunnable: legacy repository requires missing utils and old BERT/TextOIR contract")
    if method in {"single_centroid", "fixed_k2"}:
        from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector

        detector = MultiSphereOOSDetector(
            radius_method="mean_std",
            radius_lambda=1.0,
            center_mode="class_centroid_mixture",
            distance_metric="euclidean",
            l2_normalize=True,
            subcenters_per_intent=1 if method == "single_centroid" else 2,
            random_state=42,
        )
        detector.fit(train, train_intents)
        output, balls = _detector_output(detector, train, test, test_rows)
        return output, balls, {"partition": "ours_fixed_k", "boundary": "ours_mean_std", "cluster_count": len(balls)}
    if method == "random_partition":
        partition = build_partition(train, train_intents, 2, "random_balanced", seed)
        detector = fit_injected_detector(train, train_intents, partition, distance="euclidean", radius_lambda=1.0)
        output, balls = _detector_output(detector, train, test, test_rows)
        return output, balls, {"partition": "random_balanced", "boundary": "ours_mean_std", "cluster_count": len(balls)}

    clusterer = AdaptiveGranularBallClusterer(seed=seed)
    clusterer.fit(normalized_train, train_intents)
    if method == "mogb_minilm":
        boundaries = make_mogb_boundaries(clusterer, boundary="mean", distance="euclidean")
        output = score_mogb_boundaries(normalized_test, boundaries, distance="euclidean")
        return output, balls_to_rows(clusterer), {"partition": "mogb_adaptive", "boundary": "mogb_mean_euclidean", "cluster_count": len(boundaries), "ball_statistics": clusterer.ball_statistics()}
    if method == "mogb_partition_ours_boundary":
        boundaries = make_mogb_boundaries(clusterer, boundary="mean_std", distance="mahalanobis_diag")
        output = score_mogb_boundaries(normalized_test, boundaries, distance="mahalanobis_diag")
        return output, balls_to_rows(clusterer), {"partition": "mogb_adaptive", "boundary": "ours_diag_mean_std", "cluster_count": len(boundaries), "ball_statistics": clusterer.ball_statistics()}
    if method == "ours_partition_mogb_boundary":
        from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector

        detector = MultiSphereOOSDetector(
            radius_method="mean_std",
            radius_lambda=1.0,
            center_mode="class_centroid_mixture",
            distance_metric="euclidean",
            l2_normalize=True,
            subcenters_per_intent=2,
            random_state=42,
        )
        detector.fit(train, train_intents)
        boundaries: list[MOGBBoundary] = []
        for sphere in detector.spheres:
            indices = np.flatnonzero(detector._train_cluster_labels == sphere.cluster_id)
            points = detector._train_embeddings[indices]
            radius = float(np.linalg.norm(points - sphere.center, axis=1).mean())
            boundaries.append(MOGBBoundary(sphere.cluster_id, sphere.center, max(radius, 1e-12), str(sphere.intent_name), indices))
        output = score_mogb_boundaries(normalized_test, boundaries, distance="euclidean")
        return output, [{"ball_id": ball.ball_id, "parent_id": None, "depth": 0, "majority_label": ball.label, "purity": 1.0, "sample_count": int(ball.sample_indices.size), "radius": ball.radius, "selected": True, "stop_reason": "ours_fixed_k"} for ball in boundaries], {"partition": "ours_fixed_k2", "boundary": "mogb_mean_euclidean", "cluster_count": len(boundaries)}
    raise ValueError(f"Unknown MOGB method: {method}")


def _write_predictions(path: Path, rows: list[dict[str, Any]], output: dict[str, np.ndarray], method: str, dataset: str, kir: float, seed: int) -> None:
    fields = ["sample_id", "gold_intent", "gold_is_oos", "predicted_label", "predicted_is_oos", "nearest_ball", "distance", "radius", "normalized_score", "accepted_ball_count", "method", "dataset", "kir", "seed"]
    lines = ["\t".join(fields)]
    for index, row in enumerate(rows):
        values = [
            row["sample_id"], row["intent"], int(row["label"]), str(output["predicted_label"][index]), int(output["predicted_oos"][index]),
            int(output["nearest_ball"][index]), float(output["distance"][index]), float(output["radius"][index]), float(output["score"][index]),
            int(output["accepted_ball_count"][index]), method, dataset, kir, seed,
        ]
        lines.append("\t".join(str(value) for value in values))
    atomic_write_text(path, "\n".join(lines) + "\n")


def run_one(paths: ProtocolV2Paths, dataset: str, kir: float, seed: int, method: str, output_root: Path, overwrite: bool = False) -> Path:
    views, train, test, inputs = _load_cached_inputs(paths, dataset, seed, kir)
    run_dir = output_root / dataset / f"kir_{kir:.2f}" / f"seed_{seed}" / method
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        return run_dir
    if run_dir.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite incomplete MOGB run: {run_dir}")
    if overwrite and run_dir.exists():
        expected_root = paths.run_root / "mogb_baseline_v1"
        if expected_root not in run_dir.parents:
            raise RuntimeError(f"Refusing overwrite outside MOGB artifact root: {run_dir}")
        shutil.rmtree(run_dir)
    config = {"protocol_version": paths.dataset_version, "dataset": dataset, "kir": kir, "seed": seed, "method": method, "representation": "frozen_minilm", "test_used_for_selection": False}
    config_hash = sha256_json(config)
    started = time.perf_counter()
    output, ball_rows, method_details = _run_method(method, train, views.train, test, views.test, seed)
    metrics = _open_metrics(views.test, output)
    selected_ball_rows = [row for row in ball_rows if bool(row.get("selected", True))]
    metrics.update({"scoring_seconds": time.perf_counter() - started, "samples_per_second": len(test) / max(time.perf_counter() - started, 1e-12), "effective_cluster_count": method_details["cluster_count"], "minimum_cluster_size": min(int(row["sample_count"]) for row in selected_ball_rows)})
    oos_mask = np.asarray([int(row["label"]) for row in views.test], dtype=bool)
    ball_rows = [dict(row) for row in ball_rows]
    for ball in ball_rows:
        assigned = output["nearest_ball"] == int(ball["ball_id"])
        ball["test_oos_assignment_count"] = int(np.sum(assigned & oos_mask))
        ball["test_oos_false_accept_count"] = int(np.sum(assigned & oos_mask & (output["predicted_oos"] == 0)))
        ball["test_known_false_reject_count"] = int(np.sum(assigned & ~oos_mask & (output["predicted_oos"] == 1)))
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_json(temporary / "config.json", config)
        atomic_write_json(temporary / "metrics.json", metrics)
        atomic_write_json(temporary / "ball_statistics.json", method_details.get("ball_statistics", {"selected_balls": len(ball_rows), "total_balls": len(ball_rows)}))
        atomic_write_json(temporary / "inputs.json", inputs)
        atomic_write_json(temporary / "method_details.json", method_details)
        atomic_write_text(temporary / "balls.jsonl", "\n".join(json.dumps(row, sort_keys=True) for row in ball_rows) + "\n")
        _write_predictions(temporary / "predictions.tsv", views.test, output, method, dataset, kir, seed)
        atomic_write_json(temporary / "environment.json", environment_snapshot(paths.project_root))
        atomic_write_json(temporary / "manifest.json", {"status": "complete", "run_id": f"mogb__{dataset}__kir_{kir:.2f}__seed_{seed}__{method}", "config_hash": config_hash, "input_hashes": inputs, "test_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started, "source": "MOGB baseline integration; not an original s2c method"})
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="stackoverflow")
    parser.add_argument("--kir", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=METHODS, default="mogb_minilm")
    parser.add_argument("--device", default="cpu", help="Recorded only; fair mode consumes cached embeddings")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Skip a complete cell; incomplete cells are never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Validate the requested input cell without writing a result")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    output_root = args.output_dir or paths.run_root / "mogb_baseline_v1"
    if args.dry_run:
        views, train, test, inputs = _load_cached_inputs(paths, args.dataset, args.seed, args.kir)
        print(json.dumps({"status": "ready", "dataset": args.dataset, "kir": args.kir, "seed": args.seed, "method": args.method, "train_rows": len(train), "test_rows": len(test), "inputs": inputs}, sort_keys=True))
        return 0
    result = run_one(paths, args.dataset, args.kir, args.seed, args.method, output_root, args.overwrite)
    print(json.dumps({"status": "complete", "run_dir": str(result), "method": args.method, "device": args.device, "resume": bool(args.resume)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
