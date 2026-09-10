#!/usr/bin/env python3
"""Search CLINC H1 Trainable Gate geometry and confirm locked full pipelines.

The search reuses completed Known-only checkpoints.  It selects representation,
centers, covariance rule, radius rule, lambda, threshold and acceptance mode on
validation OOS F1, then evaluates only the locked per-seed choices on test.
Raw embeddings and predictions remain private and are not written to results.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from tools.eval.run_historical_trainable_full_pipeline import (
    DEFAULT_H1_ROOT,
    MODEL_ROOT,
    _RacalGateEncoder,
    _make_pipeline,
    compute_metrics,
)
from tools.eval.run_historical_trainable_parameter_full_pipeline import (
    _metrics_from_output,
    _read_views,
    _vectorized_output,
    write_csv,
)

NAME = "historical_trainable_gate_search"
DATASET = "clinc150"
SEEDS = (13, 42, 87)
PAPER_OOS_F1 = 91.96
LAMBDA_VALUES = (.5, .75, 1., 1.25, 1.5, 2., 2.5, 3.)
THRESHOLDS = tuple(round(value / 100, 2) for value in range(70, 131, 2))
MODES = ("nearest_sphere", "normalized_union")
REPRESENTATIONS = {
    "existing_last2": DEFAULT_H1_ROOT / DATASET / "kir50_seed{seed}" / "trainable_k1" / "checkpoint.pt",
    "last2_long": ROOT.parent / "artifacts/s2c/runs/historical_trainable_checkpoint_selection/last2_long/clinc150/kir50_seed{seed}/trainable_k1/checkpoint.pt",
    "projection_only": ROOT.parent / "artifacts/s2c/runs/historical_trainable_checkpoint_selection/projection_only/clinc150/kir50_seed{seed}/trainable_k1/checkpoint.pt",
    "lora_long": ROOT.parent / "artifacts/s2c/runs/historical_trainable_checkpoint_selection/lora_long/clinc150/kir50_seed{seed}/trainable_k1/checkpoint.pt",
}
GEOMETRIES = (
    ("k1_local", 1, None, "local"),
    ("k2_local", 2, None, "local"),
    ("k3_local", 3, None, "local"),
    ("k5_local", 5, None, "local"),
    ("adaptive25_k2_local", 1, ("mean_distance", .25, 2), "local"),
    ("adaptive50_k2_local", 1, ("mean_distance", .50, 2), "local"),
    ("k1_shrink_intent25", 1, None, "shrink_intent25"),
    ("k1_shrink_global25", 1, None, "shrink_global25"),
    ("k2_shrink_intent25", 2, None, "shrink_intent25"),
    ("adaptive25_shrink_intent25", 1, ("mean_distance", .25, 2), "shrink_intent25"),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_path(name: str, seed: int) -> Path:
    path = Path(str(REPRESENTATIONS[name]).format(seed=seed))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def encode_checkpoint(path: Path, views: dict[str, list[dict[str, Any]]], device: torch.device, splits: tuple[str, ...]) -> dict[str, np.ndarray]:
    encoder = _RacalGateEncoder(MODEL_ROOT / "all-MiniLM-L6-v2", path, device)
    values = {split: encoder.encode([str(row["text"]) for row in views[split]], batch_size=256) for split in splits}
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values


def _rank_intents(train_values: np.ndarray, train_rows: list[dict[str, Any]]) -> list[str]:
    intents = sorted({str(row["intent"]) for row in train_rows})
    scores = []
    values = np.asarray(train_values, dtype=np.float64)
    for intent in intents:
        points = values[np.asarray([str(row["intent"]) == intent for row in train_rows])]
        center = points.mean(axis=0)
        scores.append((float(np.linalg.norm(points - center, axis=1).mean()), intent))
    return [intent for _, intent in sorted(scores, reverse=True)]


def _geometry_detector(train_values: np.ndarray, train_rows: list[dict[str, Any]], geometry: tuple[Any, ...]) -> MultiSphereOOSDetector:
    name, fixed_k, adaptive, covariance = geometry
    overrides: dict[str, int] = {}
    if adaptive is not None:
        _, fraction, split_k = adaptive
        ranked = _rank_intents(train_values, train_rows)
        for intent in ranked[: max(1, int(np.ceil(len(ranked) * fraction)))]:
            overrides[intent] = int(split_k)
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=int(fixed_k),
        subcenters_overrides=overrides,
        radius_method="mean_std",
        radius_lambda=1.,
        distance_metric="mahalanobis_diag",
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
        acceptance_mode="nearest_sphere",
    )
    detector.fit(train_values, np.asarray([str(row["intent"]) for row in train_rows], dtype=object))
    if covariance != "local":
        _apply_covariance_shrinkage(detector, train_values, train_rows, covariance)
    return detector


def _apply_covariance_shrinkage(detector: MultiSphereOOSDetector, train_values: np.ndarray, train_rows: list[dict[str, Any]], mode: str) -> None:
    values = detector._normalize_embeddings(np.asarray(train_values, dtype=np.float64))
    labels = np.asarray(detector._train_cluster_labels)
    intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    global_center = values.mean(axis=0)
    global_var = np.var(values - global_center, axis=0)
    alpha = .25
    intent_vars: dict[str, np.ndarray] = {}
    for intent in sorted(set(intents.tolist())):
        points = values[intents == intent]
        intent_vars[intent] = np.var(points - points.mean(axis=0), axis=0)
    for sphere in detector.spheres:
        points = values[labels == sphere.cluster_id]
        local_var = np.var(points - sphere.center, axis=0)
        target = global_var if mode == "shrink_global25" else intent_vars[str(sphere.intent_name)]
        sphere.inv_diag_cov = 1. / (np.clip((1 - alpha) * local_var + alpha * target, 1e-6, None))


def _set_radius(detector: MultiSphereOOSDetector, train_values: np.ndarray, radius_lambda: float, rule: str) -> None:
    values = detector._normalize_embeddings(np.asarray(train_values, dtype=np.float64))
    labels = np.asarray(detector._train_cluster_labels)
    for sphere in detector.spheres:
        points = values[labels == sphere.cluster_id]
        diff = points - sphere.center
        distances = np.sqrt(np.sum((diff ** 2) * sphere.inv_diag_cov, axis=1))
        sphere.radius = float(np.quantile(distances, .95)) if rule == "quantile95" else float(distances.mean() + radius_lambda * distances.std())


def _detector_signature(detector: MultiSphereOOSDetector) -> dict[str, Any]:
    return {
        "radius_method": detector.radius_method,
        "radius_lambda": detector.radius_lambda,
        "distance_metric": detector.distance_metric,
        "acceptance_mode": detector.acceptance_mode,
        "spheres": [
            {"cluster_id": int(s.cluster_id), "intent": str(s.intent_name), "center": np.asarray(s.center).tolist(), "radius": float(s.radius), "inv_diag_cov": np.asarray(s.inv_diag_cov).tolist()}
            for s in detector.spheres
        ],
    }


def _candidate_metrics(detector: MultiSphereOOSDetector, values: np.ndarray, rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    output = _vectorized_output(detector, values)
    metrics = _metrics_from_output(detector, output, rows, threshold)
    return {"oos_f1": float(metrics["f1_u"]), "known_f1": float(metrics["f1_k"]),
            "accuracy": float(metrics["accuracy"]), "known_recall": float(metrics["known_recall"]),
            "false_accept_rate": float(metrics["false_accept_rate"]), "auroc": float(metrics["auroc"]),
            "aupr_oos": float(metrics["aupr_oos"])}


def _select(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(min(rows, key=lambda row: (-float(row["oos_f1"]), int(row["center_count"]), float(row["radius_lambda"]), abs(float(row["threshold"]) - 1), row["geometry"])))


def _pipeline_confirm(detector: MultiSphereOOSDetector, checkpoint: Path, seed: int, device: torch.device, views: dict[str, list[dict[str, Any]]], threshold: float) -> dict[str, Any]:
    from tools.eval.run_historical_trainable_full_pipeline import _detector_state
    with tempfile.TemporaryDirectory(prefix="s2c_gate_search_pipeline_") as temporary:
        detector_path = Path(temporary) / "detector.json"
        state = _detector_state(_detector_signature(detector))
        # The evaluator applies a fixed score=1 boundary.  A validation score
        # threshold is therefore serialized as an equivalent radius scale.
        for sphere in state["spheres"]:
            sphere["radius"] = float(sphere["radius"]) * float(threshold)
        detector_path.write_text(json.dumps(state), encoding="utf-8")
        pipeline = _make_pipeline(DATASET, seed, device, detector_path, DEFAULT_H1_ROOT)
        pipeline.gate_encoder = _RacalGateEncoder(MODEL_ROOT / "all-MiniLM-L6-v2", checkpoint, device)
        predictions = pipeline.predict_batch([str(row["text"]) for row in views["test"]], batch_size=128)
        metrics, stages = compute_metrics(views["test"], predictions)
        del pipeline, predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {**metrics, **{f"stage_count_{key}": int(value) for key, value in stages.items()}}


def run(device: torch.device, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    validation_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for seed in SEEDS:
            views = _read_views(DATASET, seed)
            values = encode_checkpoint(checkpoint_path(representation, seed), views, device, ("train", "val"))
            cell: list[dict[str, Any]] = []
            for geometry in GEOMETRIES:
                detector = _geometry_detector(values["train"], views["train"], geometry)
                for rule in ("mean_std", "quantile95"):
                    lambdas = LAMBDA_VALUES if rule == "mean_std" else (1.,)
                    for radius_lambda in lambdas:
                        _set_radius(detector, values["train"], radius_lambda, rule)
                        for mode in MODES:
                            detector.acceptance_mode = mode
                            output = _vectorized_output(detector, values["val"])
                            for threshold in THRESHOLDS:
                                raw_metrics = _metrics_from_output(detector, output, views["val"], threshold)
                                metrics = {"oos_f1": float(raw_metrics["f1_u"]), "known_f1": float(raw_metrics["f1_k"]),
                                           "accuracy": float(raw_metrics["accuracy"]), "known_recall": float(raw_metrics["known_recall"]),
                                           "false_accept_rate": float(raw_metrics["false_accept_rate"]), "auroc": float(raw_metrics["auroc"]),
                                           "aupr_oos": float(raw_metrics["aupr_oos"])}
                                row = {"representation": representation, "seed": seed, "geometry": geometry[0], "center_count": len(detector.spheres), "covariance": geometry[3], "radius_rule": rule, "radius_lambda": radius_lambda, "acceptance_mode": mode, "threshold": threshold, **metrics, "test_used_for_selection": False}
                                cell.append(row)
            best = _select(cell)
            selections.append(best)
            validation_rows.extend(cell)
            print(f"LOCK CANDIDATE {representation} seed={seed}: {best['geometry']} val_oos={best['oos_f1']:.6f}", flush=True)
            del values
            if device.type == "cuda":
                torch.cuda.empty_cache()
    by_seed = {seed: _select([row for row in selections if int(row["seed"]) == seed]) for seed in SEEDS}
    locks = [{**by_seed[seed], "selection_objective": "validation_oos_f1_only", "test_metrics_computed": False} for seed in SEEDS]
    (output_root / "selection_lock.json").write_text(json.dumps({"choices": locks, "test_metrics_computed": False}, indent=2), encoding="utf-8")
    write_csv(output_root / "validation_candidates.csv", validation_rows)
    test_rows: list[dict[str, Any]] = []
    for lock in locks:
        seed = int(lock["seed"]); representation = str(lock["representation"]); views = _read_views(DATASET, seed)
        values = encode_checkpoint(checkpoint_path(representation, seed), views, device, ("train", "test"))
        geometry = next(item for item in GEOMETRIES if item[0] == lock["geometry"])
        detector = _geometry_detector(values["train"], views["train"], geometry)
        _set_radius(detector, values["train"], float(lock["radius_lambda"]), str(lock["radius_rule"]))
        detector.acceptance_mode = str(lock["acceptance_mode"])
        gate = _candidate_metrics(detector, values["test"], views["test"], float(lock["threshold"]))
        pipeline = _pipeline_confirm(detector, checkpoint_path(representation, seed), seed, device, views, float(lock["threshold"]))
        test_rows.append({**lock, "test_used_for_selection": False, **{f"gate_{key}": value for key, value in gate.items()}, **{f"pipeline_{key}": value for key, value in pipeline.items()}})
        del values
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(output_root / "selected_test.csv", test_rows)
    (output_root / "selection_lock.json").write_text(json.dumps({"choices": locks, "test_metrics_computed": True}, indent=2) + "\n", encoding="utf-8")
    manifest = {"experiment": NAME, "protocol": "historical_v19_paper_main", "evidence_level": "H1 controlled", "dataset": DATASET, "kir": .50, "seeds": list(SEEDS), "representations": list(REPRESENTATIONS), "geometry_count": len(GEOMETRIES), "validation_only_selection": True, "test_used_for_selection": False, "device": str(device), "completed_units": len(test_rows), "status": "complete"}
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/analysis" / NAME)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    run(torch.device(args.device), args.output_root.resolve())


if __name__ == "__main__":
    main()
