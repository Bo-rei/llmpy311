#!/usr/bin/env python3
"""Search known-validation per-sphere calibration for the H1 Trainable Gate."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from protocol_v2.experiments.racal_v1.representation import choose_device  # noqa: E402
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector, SphereConfig  # noqa: E402
from tools.eval.run_historical_trainable_extended_boundary_search import (  # noqa: E402
    _distance_matrix,
    _score_from_distances,
)
from tools.eval.run_historical_trainable_parameter_full_pipeline import (  # noqa: E402
    DATASETS,
    DEFAULT_H1_ROOT,
    PAPER,
    SEEDS,
    _encode_cell,
    _fit_detector,
    _metrics_from_output,
    _run_pipeline,
    compute_metrics,
    read_json,
    write_csv,
)


OUTPUT_ROOT = ROOT / "results" / "analysis" / "historical_trainable_per_sphere_calibration_search"
K_VALUES = (1, 2, 3)
LAMBDA_VALUES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00)
CALIBRATION_QUANTILES = (0.80, 0.90, 0.95, 0.975, 0.99)
GLOBAL_THRESHOLDS = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30)
ACCEPTANCE_MODES = ("nearest_sphere", "normalized_union")
BASELINE = (1, 1.0, 1.0, "none", None, "nearest_sphere")
METRICS = ("oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos")


def _read_views(dataset: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    root = ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19" / dataset / f"kir50_seed{seed}"
    known = {str(value) for value in read_json(root / "KNOWN_INTENTS.json")["known_intents"]}
    views = {}
    for split in ("train", "val", "test"):
        rows = []
        for raw in read_json(root / "gate" / f"{split}.json"):
            row = dict(raw)
            row["text"] = str(row["text"])
            row["intent"] = str(row["intent"])
            row["label"] = int(row.get("label", 0 if row["intent"] in known else 1))
            rows.append(row)
        views[split] = rows
    return views


def _config_key(row: Mapping[str, Any]) -> tuple[int, float, float, str, float | None, str]:
    q = None if str(row["calibration_mode"]) == "none" else float(row["calibration_quantile"])
    return int(row["k"]), float(row["radius_lambda"]), float(row["global_threshold"]), str(row["calibration_mode"]), q, str(row["acceptance_mode"])


def _score_output(
    base_output: Mapping[str, np.ndarray],
    sphere_thresholds: np.ndarray,
    global_threshold: float,
) -> dict[str, np.ndarray]:
    selected = np.asarray(base_output["nearest_cluster"], dtype=np.int64)
    effective = np.asarray(base_output["score"], dtype=np.float64) / np.clip(sphere_thresholds[selected], 1e-12, None)
    return {
        "score": effective,
        "nearest_cluster": selected,
        "distance": np.asarray(base_output["distance"], dtype=np.float64),
        "global_threshold": np.asarray([float(global_threshold)], dtype=np.float64),
    }


def _calibration_thresholds(
    base_output: Mapping[str, np.ndarray],
    val_rows: Sequence[Mapping[str, Any]],
    sphere_count: int,
    quantile: float,
) -> np.ndarray:
    labels = np.asarray([int(row["label"]) for row in val_rows], dtype=np.int64)
    selected = np.asarray(base_output["nearest_cluster"], dtype=np.int64)
    scores = np.asarray(base_output["score"], dtype=np.float64)
    result = np.ones((sphere_count,), dtype=np.float64)
    for sphere_id in range(sphere_count):
        values = scores[(labels == 0) & (selected == sphere_id)]
        if len(values):
            result[sphere_id] = max(float(np.quantile(values, quantile)), 1e-6)
    return result


def _row(dataset: str, seed: int, split: str, k: int, radius_lambda: float, global_threshold: float, mode: str, calibration_mode: str, calibration_quantile: float | None, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset, "seed": int(seed), "split": split, "k": int(k), "radius_lambda": float(radius_lambda),
        "global_threshold": float(global_threshold), "acceptance_mode": mode, "calibration_mode": calibration_mode,
        "calibration_quantile": "none" if calibration_quantile is None else float(calibration_quantile),
        "oos_f1": float(metrics["f1_u"]), "known_f1": float(metrics["f1_k"]), "accuracy": float(metrics["accuracy"]),
        "known_recall": float(metrics["known_recall"]), "false_accept_rate": float(metrics["false_accept_rate"]),
        "false_reject_rate": float(metrics["false_reject_rate"]), "auroc": float(metrics["auroc"]), "aupr_oos": float(metrics["aupr_oos"]),
    }


def _search_cell(dataset: str, seed: int, views: Mapping[str, list[dict[str, Any]]], values: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        detector = _fit_detector(values["train"], views["train"], k, 1.0, "nearest_sphere")
        distances = {split: _distance_matrix(detector, values[split]) for split in ("val", "test")}
        for radius_lambda in LAMBDA_VALUES:
            detector.radius_lambda = float(radius_lambda)
            detector._compute_radii()
            for mode in ACCEPTANCE_MODES:
                detector.acceptance_mode = mode
                base_outputs = {split: _score_from_distances(distances[split], detector, mode) for split in ("val", "test")}
                candidates: list[tuple[str, float | None, np.ndarray]] = [("none", None, np.ones(len(detector.spheres), dtype=np.float64))]
                for quantile in CALIBRATION_QUANTILES:
                    candidates.append(("per_sphere_known_quantile", quantile, _calibration_thresholds(base_outputs["val"], views["val"], len(detector.spheres), quantile)))
                for calibration_mode, quantile, sphere_thresholds in candidates:
                    for global_threshold in GLOBAL_THRESHOLDS:
                        for split in ("val", "test"):
                            output = _score_output(base_outputs[split], sphere_thresholds, global_threshold)
                            metrics = _metrics_from_output(detector, output, views[split], global_threshold)
                            rows.append(_row(dataset, seed, split, k, radius_lambda, global_threshold, mode, calibration_mode, quantile, metrics))
    return rows


def _select(rows: Sequence[Mapping[str, Any]], dataset: str, seed: int) -> dict[str, Any]:
    val = [row for row in rows if str(row["dataset"]) == dataset and int(row["seed"]) == seed and str(row["split"]) == "val"]
    baseline = next(row for row in val if _config_key(row) == BASELINE)
    eligible = [row for row in val if float(row["known_f1"]) >= float(baseline["known_f1"]) - 0.01 and float(row["accuracy"]) >= float(baseline["accuracy"]) - 0.01]
    if not eligible:
        raise ValueError(f"no eligible per-sphere calibration candidate: {dataset}/seed{seed}")
    selected = max(eligible, key=lambda row: (float(row["oos_f1"]), float(row["known_f1"]), float(row["accuracy"]), 1 if row["calibration_mode"] == "none" else 0, -int(row["k"]), -abs(float(row["radius_lambda"]) - 1), -abs(float(row["global_threshold"]) - 1)))
    return {**dict(selected), "selection_scope": "per_seed_validation", "test_used_for_selection": False, "selection_guard": "same-seed Known F1 and Accuracy within 1 pp of uncalibrated K=1 baseline"}


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row.get("selection_scope", "per_seed_validation")))].append(row)
    output = []
    for (dataset, scope), group in sorted(grouped.items()):
        item: dict[str, Any] = {"dataset": dataset, "selection_scope": scope, "seed_count": len(group), "selected_configurations": "; ".join(f"seed{row['seed']}:K={row['k']},lambda={row['radius_lambda']},t={row['global_threshold']},{row['acceptance_mode']},{row['calibration_mode']},{row['calibration_quantile']}" for row in group)}
        for metric in ("oos_f1", "f1_all", "known_macro_f1", "overall_accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "router_error_rate", "expert_error_rate"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = float(np.mean(values)); item[f"{metric}_std"] = float(np.std(values))
        item["paper_oos_f1"] = PAPER[dataset]["oos_f1"]
        item["delta_oos_f1_pp"] = float(item["oos_f1_mean"] * 100 - PAPER[dataset]["oos_f1"])
        item["beats_paper_oos_f1_mean"] = bool(item["delta_oos_f1_pp"] > 0)
        output.append(item)
    return output


def _pipeline_detector(base: MultiSphereOOSDetector, sphere_thresholds: np.ndarray, global_threshold: float) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture", subcenters_per_intent=base.subcenters_per_intent,
        radius_method=base.radius_method, radius_lambda=base.radius_lambda, distance_metric=base.distance_metric,
        covariance_eps=base.covariance_eps, l2_normalize=base.l2_normalize, random_state=base.random_state,
        acceptance_mode=base.acceptance_mode,
    )
    detector.n_clusters = base.n_clusters
    detector.intent_to_cluster = dict(base.intent_to_cluster)
    detector.intent_to_clusters = {str(key): [int(value) for value in values] for key, values in base.intent_to_clusters.items()}
    detector.cluster_to_intent = {int(key): str(value) for key, value in base.cluster_to_intent.items()}
    detector.spheres = [
        SphereConfig(
            center=np.asarray(sphere.center, dtype=float),
            radius=float(sphere.radius) * float(sphere_thresholds[int(sphere.cluster_id)]) * float(global_threshold),
            cluster_id=int(sphere.cluster_id), intent_name=sphere.intent_name,
            inv_diag_cov=None if sphere.inv_diag_cov is None else np.asarray(sphere.inv_diag_cov, dtype=float),
        )
        for sphere in base.spheres
    ]
    detector.fitted = True
    return detector


def run(datasets: Sequence[str], seeds: Sequence[int], device: torch.device, h1_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.time(); all_rows: list[dict[str, Any]] = []; selected: list[dict[str, Any]] = []; selected_test: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="s2c_per_sphere_pipeline_") as temporary:
        temporary_root = Path(temporary)
        for dataset in datasets:
            dataset_rows: list[dict[str, Any]] = []
            cache: dict[int, tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]] = {}
            for seed in seeds:
                views, values = _encode_cell(dataset, int(seed), device, h1_root)
                dataset_rows.extend(_search_cell(dataset, int(seed), views, values)); cache[int(seed)] = (views, values)
            all_rows.extend(dataset_rows)
            choices = [_select(dataset_rows, dataset, int(seed)) for seed in seeds]
            selected.extend(choices)
            for choice in choices:
                seed = int(choice["seed"]); views, values = cache[seed]
                config = _config_key(choice); base = _fit_detector(values["train"], views["train"], config[0], config[1], config[5])
                base_output = _score_from_distances(_distance_matrix(base, values["val"]), base, config[5])
                quantile = None if config[3] == "none" else float(config[4])
                sphere_thresholds = np.ones(len(base.spheres), dtype=np.float64) if quantile is None else _calibration_thresholds(base_output, views["val"], len(base.spheres), quantile)
                direct_detector = _pipeline_detector(base, sphere_thresholds, config[2])
                predictions, metadata = _run_pipeline(dataset, seed, device, direct_detector, 1.0, h1_root, temporary_root)
                metrics, stages = compute_metrics(views["test"], predictions)
                test_gate = next(row for row in dataset_rows if str(row["split"]) == "test" and int(row["seed"]) == seed and _config_key(row) == config)
                if abs(float(metrics["oos_f1"]) - float(test_gate["oos_f1"])) > 1e-10:
                    raise AssertionError(f"Gate/full pipeline OOS mismatch: {dataset}/seed{seed}")
                result = {"dataset": dataset, "seed": seed, "k": config[0], "radius_lambda": config[1], "global_threshold": config[2], "acceptance_mode": config[5], "calibration_mode": config[3], "calibration_quantile": "none" if quantile is None else quantile, "selection_scope": "per_seed_validation", "test_used_for_selection": False, "direct_pipeline_verified": True, "direct_prediction_count": int(metadata["prediction_count"]), **{key: float(metrics[key]) for key in ("oos_f1", "f1_all", "known_macro_f1", "overall_accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "router_error_rate", "expert_error_rate")}, "paper_oos_f1": PAPER[dataset]["oos_f1"], "delta_oos_f1_pp": float(metrics["oos_f1"] * 100 - PAPER[dataset]["oos_f1"]), "beats_paper_oos_f1": bool(metrics["oos_f1"] * 100 > PAPER[dataset]["oos_f1"])}
                result.update({f"stage_count_{key}": int(value) for key, value in stages.items()}); selected_test.append(result)
                del predictions, direct_detector, base, views, values
                if device.type == "cuda": torch.cuda.empty_cache()
            cache.clear()
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "gate_workpoints.csv", all_rows); write_csv(output_root / "selected_validation_per_seed.csv", selected); write_csv(output_root / "selected_test_per_seed.csv", selected_test); write_csv(output_root / "selected_test_summary.csv", _summary(selected_test))
    manifest = {"schema_version": "s2c.historical_trainable_per_sphere_calibration_search.v1", "stage": "historical_trainable_per_sphere_calibration_search_v1", "protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert", "datasets": list(datasets), "seeds": [int(seed) for seed in seeds], "kir": 0.50, "representation": "existing_trainable_last2_minilm_plus_projection_checkpoint", "selection_scope": "per_seed_validation", "selection_split": "validation", "selection_guard": "same-seed Known F1 and Accuracy within 1 pp of uncalibrated K=1 baseline", "test_used_for_selection": False, "oos_used_for_training": False, "downstream_router_expert_fixed": True, "direct_pipeline_verified": True, "device": str(device), "paper_reference": "fulltex.tex KIR=.50 Ours row", "paper_oos_f1": PAPER, "candidate_count_per_seed_dataset": len(K_VALUES) * len(LAMBDA_VALUES) * len(ACCEPTANCE_MODES) * (1 + len(CALIBRATION_QUANTILES)) * len(GLOBAL_THRESHOLDS) * 2, "completed_gate_rows": len(all_rows), "completed_selected_direct_units": len(selected_test), "paper_beating_datasets": sorted({str(row["dataset"]) for row in selected_test if bool(row["beats_paper_oos_f1"])}), "elapsed_seconds": time.time() - started}
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--dataset", choices=DATASETS, action="append"); parser.add_argument("--seed", choices=SEEDS, type=int, action="append"); parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); parser.add_argument("--h1-root", type=Path, default=DEFAULT_H1_ROOT); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args()
    device = choose_device(args.device); print(json.dumps(run(tuple(args.dataset or DATASETS), tuple(args.seed or SEEDS), device, args.h1_root.resolve(), args.output_root.resolve()), ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
