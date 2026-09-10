#!/usr/bin/env python3
"""Evaluate historical H1 Trainable-MiniLM Gate work-points in the cascade.

The Gate work-points are selected on validation data only.  Test rows are used
for confirmation, and the existing Router/Expert checkpoints are kept fixed.
The downstream predictions are replayed once with a permissive Gate so every
candidate can be evaluated without writing raw prediction artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from protocol_v2.experiments.racal_v1.boundary import detector_signature  # noqa: E402
from protocol_v2.experiments.racal_v1.representation import choose_device  # noqa: E402
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector  # noqa: E402
import tools.eval.run_historical_trainable_full_pipeline as pipeline_eval  # noqa: E402
from tools.eval.run_historical_trainable_full_pipeline import (  # noqa: E402
    _make_pipeline,
    _RacalGateEncoder,
    compute_metrics,
    read_json,
    write_csv,
)


DATA_ROOT = ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19"
MODEL_ROOT = ROOT.parent / "assets" / "models"
DEFAULT_H1_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_protocol_v2" / "minilm_k1"
OUTPUT_ROOT = ROOT / "results" / "analysis" / "historical_trainable_parameter_full_pipeline"
KIR = 0.50
OOS_ONLY_SELECTION = False
EXTERNAL_OOS_TARGET_SELECTION = False

DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
SEEDS = (13, 42, 87)
K_VALUES = (1, 2, 3)
LAMBDA_VALUES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
THRESHOLD_VALUES = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
ACCEPTANCE_MODES = ("nearest_sphere", "normalized_union")
OOS_K_VALUES = (1, 2, 3, 5)
OOS_LAMBDA_VALUES = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50)
OOS_THRESHOLD_VALUES = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
PAPER = {
    "clinc150": {"known_f1": 79.95, "oos_f1": 91.96, "accuracy": 86.78},
    "stackoverflow": {"known_f1": 75.48, "oos_f1": 89.71, "accuracy": 85.54},
    "banking77_oos": {"known_f1": 74.90, "oos_f1": 88.23, "accuracy": 78.98},
}
PAPER_BY_KIR = {
    ("clinc150", 0.25): {"known_f1": 71.75, "oos_f1": 95.01, "accuracy": 90.45},
    ("clinc150", 0.50): {"known_f1": 79.95, "oos_f1": 91.96, "accuracy": 86.78},
    ("clinc150", 0.75): {"known_f1": 66.32, "oos_f1": 87.10, "accuracy": 79.83},
    ("stackoverflow", 0.25): {"known_f1": 73.61, "oos_f1": 94.47, "accuracy": 91.04},
    ("stackoverflow", 0.50): {"known_f1": 75.48, "oos_f1": 89.71, "accuracy": 85.54},
    ("stackoverflow", 0.75): {"known_f1": 80.18, "oos_f1": 75.57, "accuracy": 81.32},
    ("banking77_oos", 0.25): {"known_f1": 79.95, "oos_f1": 93.99, "accuracy": 89.07},
    ("banking77_oos", 0.50): {"known_f1": 74.90, "oos_f1": 88.23, "accuracy": 78.98},
    ("banking77_oos", 0.75): {"known_f1": 70.28, "oos_f1": 86.49, "accuracy": 77.84},
}
EXTERNAL_OOS_F1_BY_KIR = {
    ("clinc150", 0.25): 93.56,
    ("clinc150", 0.50): 90.10,
    ("clinc150", 0.75): 86.00,
    ("stackoverflow", 0.25): 92.65,
    ("stackoverflow", 0.50): 88.86,
    ("stackoverflow", 0.75): 74.55,
    ("banking77_oos", 0.25): 86.57,
    ("banking77_oos", 0.50): 79.93,
    ("banking77_oos", 0.75): 69.37,
}
METRICS = (
    "oos_f1",
    "known_f1",
    "accuracy",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
    "auroc",
    "aupr_oos",
)
FULL_METRICS = (
    "oos_f1",
    "f1_all",
    "known_macro_f1",
    "overall_accuracy",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
    "router_error_rate",
    "expert_error_rate",
)


def _fit_detector(
    train_values: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    k: int,
    radius_lambda: float,
    acceptance_mode: str,
) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=int(k),
        radius_method="mean_std",
        radius_lambda=float(radius_lambda),
        distance_metric="mahalanobis_diag",
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
        acceptance_mode=acceptance_mode,
    )
    detector.fit(train_values, np.asarray([str(row["intent"]) for row in train_rows], dtype=object))
    return detector


def _vectorized_output(
    detector: MultiSphereOOSDetector,
    embeddings: np.ndarray,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    values = np.asarray(embeddings, dtype=np.float64)
    if detector.l2_normalize:
        values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    centers = np.asarray([sphere.center for sphere in detector.spheres], dtype=np.float64)
    radii = np.asarray([sphere.radius for sphere in detector.spheres], dtype=np.float64)
    inv_cov = np.asarray([sphere.inv_diag_cov for sphere in detector.spheres], dtype=np.float64)
    score_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    distance_chunks: list[np.ndarray] = []
    for start in range(0, len(values), chunk_size):
        batch = values[start : start + chunk_size]
        diff = batch[:, None, :] - centers[None, :, :]
        distances = np.sqrt(np.sum(np.square(diff) * inv_cov[None, :, :], axis=2))
        raw_indices = np.argmin(distances, axis=1)
        if detector.acceptance_mode == "normalized_union":
            ratios = distances / np.clip(radii[None, :], 1e-12, None)
            indices = np.argmin(ratios, axis=1)
            scores = ratios[np.arange(len(batch)), indices]
            selected_distances = distances[np.arange(len(batch)), indices]
        else:
            indices = raw_indices
            selected_distances = distances[np.arange(len(batch)), indices]
            scores = selected_distances / np.clip(radii[indices], 1e-12, None)
        score_chunks.append(scores)
        index_chunks.append(indices)
        distance_chunks.append(selected_distances)
    return {
        "score": np.concatenate(score_chunks),
        "nearest_cluster": np.concatenate(index_chunks),
        "distance": np.concatenate(distance_chunks),
    }


def _metrics_from_output(
    detector: MultiSphereOOSDetector,
    output: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    predicted_oos = np.asarray(output["score"] > float(threshold), dtype=np.int64)
    known_mask = labels == 0
    oos_mask = labels == 1
    known_intents = sorted({str(row["intent"]) for row in rows if int(row["label"]) == 0})
    truth = [str(row["intent"]) if not is_oos else "__oos__" for row, is_oos in zip(rows, labels, strict=True)]
    predicted = [
        "__oos__" if is_oos else str(detector.cluster_to_intent.get(int(output["nearest_cluster"][index]), "__unknown__"))
        for index, is_oos in enumerate(predicted_oos)
    ]
    return {
        "f1_u": float(f1_score(labels, predicted_oos, pos_label=1, zero_division=0)),
        "f1_k": float(f1_score(truth, predicted, labels=known_intents, average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(truth, dtype=object) == np.asarray(predicted, dtype=object))),
        "known_recall": float(np.mean(predicted_oos[known_mask] == 0)),
        "false_accept_rate": float(np.mean(predicted_oos[oos_mask] == 0)),
        "false_reject_rate": float(np.mean(predicted_oos[known_mask] == 1)),
        "auroc": float(roc_auc_score(labels, output["score"])),
        "aupr_oos": float(average_precision_score(labels, output["score"])),
    }


def _read_views(dataset: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    root = DATA_ROOT / dataset / f"kir{int(round(KIR * 100)):02d}_seed{seed}"
    known_payload = read_json(root / "KNOWN_INTENTS.json")
    known = {str(value) for value in known_payload["known_intents"]}
    views: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        raw_rows = read_json(root / "gate" / f"{split}.json")
        rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            row = dict(raw)
            row["text"] = str(row["text"])
            row["intent"] = str(row["intent"])
            row["label"] = int(row.get("label", 0 if row["intent"] in known else 1))
            rows.append(row)
        if not rows:
            raise ValueError(f"empty {dataset}/seed{seed}/{split}")
        views[split] = rows
    return views


def _checkpoint_path(h1_root: Path, dataset: str, seed: int) -> Path:
    path = h1_root / dataset / f"kir{int(round(KIR * 100)):02d}_seed{seed}" / "trainable_k1" / "checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _encode_cell(
    dataset: str,
    seed: int,
    device: torch.device,
    h1_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    views = _read_views(dataset, seed)
    encoder = _RacalGateEncoder(
        MODEL_ROOT / "all-MiniLM-L6-v2",
        _checkpoint_path(h1_root, dataset, seed),
        device,
    )
    values = {
        split: np.asarray(
            encoder.encode([str(row["text"]) for row in rows], batch_size=256),
            dtype=np.float32,
        )
        for split, rows in views.items()
    }
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return views, values


def _candidate_configs() -> Sequence[tuple[int, float, float, str]]:
    k_values = OOS_K_VALUES if OOS_ONLY_SELECTION else K_VALUES
    lambda_values = OOS_LAMBDA_VALUES if OOS_ONLY_SELECTION else LAMBDA_VALUES
    threshold_values = OOS_THRESHOLD_VALUES if OOS_ONLY_SELECTION else THRESHOLD_VALUES
    return [
        (k, radius_lambda, threshold, mode)
        for k in k_values
        for radius_lambda in lambda_values
        for threshold in threshold_values
        for mode in ACCEPTANCE_MODES
    ]


def _key(k: int, radius_lambda: float, threshold: float, mode: str) -> tuple[int, float, float, str]:
    return (int(k), round(float(radius_lambda), 8), round(float(threshold), 8), str(mode))


def _gate_row(
    dataset: str,
    seed: int,
    split: str,
    k: int,
    radius_lambda: float,
    threshold: float,
    mode: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "split": split,
        "k": int(k),
        "radius_lambda": float(radius_lambda),
        "threshold": float(threshold),
        "acceptance_mode": mode,
        "oos_f1": float(metrics["f1_u"]),
        "known_f1": float(metrics["f1_k"]),
        "accuracy": float(metrics["accuracy"]),
        "known_recall": float(metrics["known_recall"]),
        "false_accept_rate": float(metrics["false_accept_rate"]),
        "false_reject_rate": float(metrics["false_reject_rate"]),
        "auroc": float(metrics["auroc"]),
        "aupr_oos": float(metrics["aupr_oos"]),
    }


def _search_cell(
    dataset: str,
    seed: int,
    views: Mapping[str, list[dict[str, Any]]],
    values: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[tuple[int, float, str], dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    outputs: dict[tuple[int, float, str], dict[str, np.ndarray]] = {}
    k_values = OOS_K_VALUES if OOS_ONLY_SELECTION else K_VALUES
    lambda_values = OOS_LAMBDA_VALUES if OOS_ONLY_SELECTION else LAMBDA_VALUES
    threshold_values = OOS_THRESHOLD_VALUES if OOS_ONLY_SELECTION else THRESHOLD_VALUES
    for k in k_values:
        detector = _fit_detector(values["train"], views["train"], k, 1.0, "nearest_sphere")
        for radius_lambda in lambda_values:
            detector.radius_lambda = float(radius_lambda)
            detector._compute_radii()
            for mode in ACCEPTANCE_MODES:
                detector.acceptance_mode = mode
                cache_key = (int(k), round(float(radius_lambda), 8), mode)
                outputs[cache_key] = {
                    split: _vectorized_output(detector, values[split])
                    for split in ("val", "test")
                }
                for threshold in threshold_values:
                    for split in ("val", "test"):
                        metrics = _metrics_from_output(
                            detector,
                            outputs[cache_key][split],
                            views[split],
                            threshold=float(threshold),
                        )
                        rows.append(
                            _gate_row(
                                dataset,
                                seed,
                                split,
                                k,
                                radius_lambda,
                                threshold,
                                mode,
                                metrics,
                            )
                        )
    return rows, outputs


def _is_config(row: Mapping[str, Any], config: tuple[int, float, float, str]) -> bool:
    k, radius_lambda, threshold, mode = config
    return (
        int(row["k"]) == int(k)
        and np.isclose(float(row["radius_lambda"]), float(radius_lambda))
        and np.isclose(float(row["threshold"]), float(threshold))
        and str(row["acceptance_mode"]) == str(mode)
    )


def _aggregate(rows: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["split"]) == split:
            config = _key(int(row["k"]), float(row["radius_lambda"]), float(row["threshold"]), str(row["acceptance_mode"]))
            grouped[config].append(row)
    output: list[dict[str, Any]] = []
    for (k, radius_lambda, threshold, mode), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "dataset": str(group[0]["dataset"]),
            "split": split,
            "k": k,
            "radius_lambda": radius_lambda,
            "threshold": threshold,
            "acceptance_mode": mode,
            "seed_count": len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
            item[f"{metric}_min"] = float(np.min(values))
        output.append(item)
    return output


def _annotate_validation_guard(
    aggregate: list[dict[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    baseline_rows = {
        int(row["seed"]): row
        for row in validation_rows
        if _is_config(row, _baseline_config())
    }
    baseline = next(row for row in aggregate if _is_config(row, _baseline_config()))
    for row in aggregate:
        config = (
            int(row["k"]),
            float(row["radius_lambda"]),
            float(row["threshold"]),
            str(row["acceptance_mode"]),
        )
        candidate_rows = {
            int(item["seed"]): item
            for item in validation_rows
            if _is_config(item, config)
        }
        mean_guard = (
            float(row["known_f1_mean"]) >= float(baseline["known_f1_mean"]) - 0.01
            and float(row["accuracy_mean"]) >= float(baseline["accuracy_mean"]) - 0.01
        )
        per_seed_guard = all(
            seed in candidate_rows
            and float(candidate_rows[seed]["known_f1"]) >= float(baseline_rows[seed]["known_f1"]) - 0.01
            and float(candidate_rows[seed]["accuracy"]) >= float(baseline_rows[seed]["accuracy"]) - 0.01
            for seed in baseline_rows
        )
        row["eligible_per_seed_guard"] = bool(per_seed_guard and mean_guard)
        row["eligible_mean_guard"] = bool(mean_guard)
    return aggregate


def _baseline_config() -> tuple[int, float, float, str]:
    return (1, 1.0, 1.0, "nearest_sphere")


def _select(
    validation_rows: Sequence[Mapping[str, Any]],
    dataset: str,
) -> tuple[dict[str, Any], str]:
    aggregate = _aggregate(validation_rows, "val")
    if EXTERNAL_OOS_TARGET_SELECTION:
        target = EXTERNAL_OOS_F1_BY_KIR[(dataset, float(KIR))] / 100.0
        eligible = [row for row in aggregate if float(row["oos_f1_mean"]) >= target]
        if not eligible:
            raise ValueError(
                f"no validation candidate reaches external OOS-F1 target {target:.4f} for {dataset}"
            )
        selected = max(
            eligible,
            key=lambda row: (
                float(row["accuracy_mean"]),
                float(row["known_recall_mean"]),
                float(row["oos_f1_mean"]),
                -int(row["k"]),
                -abs(float(row["radius_lambda"]) - 1.0),
                -abs(float(row["threshold"]) - 1.0),
            ),
        )
        selected = dict(selected)
        selected["eligible_per_seed_guard"] = True
        selected["eligible_mean_guard"] = True
        paper = PAPER_BY_KIR[(dataset, float(KIR))]
        selected.update(
            {
                "paper_known_f1": paper["known_f1"],
                "paper_oos_f1": paper["oos_f1"],
                "paper_accuracy": paper["accuracy"],
                "external_oos_f1_target": EXTERNAL_OOS_F1_BY_KIR[(dataset, float(KIR))],
                "baseline_validation_known_f1_mean": None,
                "baseline_validation_accuracy_mean": None,
                "baseline_validation_oos_f1_mean": None,
                "selection_guard": "validation_oos_f1_at_external_baseline_max_accuracy_then_known_recall",
                "selection_uses_validation_oos_labels": True,
                "test_used_for_selection": False,
            }
        )
        return selected, selected["selection_guard"]
    if OOS_ONLY_SELECTION:
        selected = max(
            aggregate,
            key=lambda row: (
                float(row["oos_f1_mean"]),
                -float(row["false_accept_rate_mean"]),
                -int(row["k"]),
                -abs(float(row["radius_lambda"]) - 1.0),
                -abs(float(row["threshold"]) - 1.0),
            ),
        )
        selected = dict(selected)
        selected["eligible_per_seed_guard"] = True
        selected["eligible_mean_guard"] = True
        paper = PAPER_BY_KIR[(dataset, float(KIR))]
        selected.update(
            {
                "paper_known_f1": paper["known_f1"],
                "paper_oos_f1": paper["oos_f1"],
                "paper_accuracy": paper["accuracy"],
                "baseline_validation_known_f1_mean": None,
                "baseline_validation_accuracy_mean": None,
                "baseline_validation_oos_f1_mean": None,
                "selection_guard": "none_oos_f1_only",
                "selection_uses_validation_oos_labels": True,
                "test_used_for_selection": False,
            }
        )
        return selected, "none_oos_f1_only"
    baseline = next(row for row in aggregate if _is_config(row, _baseline_config()))
    by_seed = {
        int(row["seed"]): row
        for row in validation_rows
        if _is_config(row, _baseline_config())
    }
    annotated: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for row in aggregate:
        candidates = [
            item
            for item in validation_rows
            if int(item["seed"]) in by_seed and _is_config(item, (int(row["k"]), float(row["radius_lambda"]), float(row["threshold"]), str(row["acceptance_mode"])))
        ]
        candidate_by_seed = {int(item["seed"]): item for item in candidates}
        per_seed_guard = all(
            seed in candidate_by_seed
            and float(candidate_by_seed[seed]["known_f1"]) >= float(by_seed[seed]["known_f1"]) - 0.01
            and float(candidate_by_seed[seed]["accuracy"]) >= float(by_seed[seed]["accuracy"]) - 0.01
            for seed in by_seed
        )
        mean_guard = (
            float(row["known_f1_mean"]) >= float(baseline["known_f1_mean"]) - 0.01
            and float(row["accuracy_mean"]) >= float(baseline["accuracy_mean"]) - 0.01
        )
        row = dict(row)
        row["eligible_per_seed_guard"] = bool(per_seed_guard and mean_guard)
        row["eligible_mean_guard"] = bool(mean_guard)
        annotated.append(row)
        if per_seed_guard and mean_guard:
            eligible.append(row)
    guard_mode = "per_seed_and_mean_1pp"
    if not eligible:
        eligible = [row for row in annotated if bool(row.get("eligible_mean_guard", False))]
        guard_mode = "mean_1pp_fallback"
    if not eligible:
        raise ValueError(f"no validation candidate satisfies Known F1/Accuracy guard for {dataset}")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["oos_f1_mean"]),
            float(row["known_f1_mean"]),
            float(row["accuracy_mean"]),
            1 if str(row["acceptance_mode"]) == "nearest_sphere" else 0,
            -int(row["k"]),
            -abs(float(row["radius_lambda"]) - 1.0),
            -abs(float(row["threshold"]) - 1.0),
        ),
    )
    selected.update(
        {
            "paper_known_f1": PAPER_BY_KIR[(dataset, float(KIR))]["known_f1"],
            "paper_oos_f1": PAPER_BY_KIR[(dataset, float(KIR))]["oos_f1"],
            "paper_accuracy": PAPER_BY_KIR[(dataset, float(KIR))]["accuracy"],
            "baseline_validation_known_f1_mean": float(baseline["known_f1_mean"]),
            "baseline_validation_accuracy_mean": float(baseline["accuracy_mean"]),
            "baseline_validation_oos_f1_mean": float(baseline["oos_f1_mean"]),
            "selection_guard": guard_mode,
            "selection_uses_validation_oos_labels": True,
            "test_used_for_selection": False,
        }
    )
    return selected, guard_mode


def _detector_state(signature: Mapping[str, Any]) -> dict[str, Any]:
    spheres = sorted(signature["spheres"], key=lambda item: int(item["cluster_id"]))
    intent_to_cluster: dict[str, int] = {}
    intent_to_clusters: dict[str, list[int]] = defaultdict(list)
    cluster_to_intent: dict[str, str] = {}
    for sphere in spheres:
        intent = str(sphere["intent"])
        cluster_id = int(sphere["cluster_id"])
        intent_to_cluster.setdefault(intent, cluster_id)
        intent_to_clusters[intent].append(cluster_id)
        cluster_to_intent[str(cluster_id)] = intent
    return {
        "n_clusters": len(spheres),
        "radius_quantile": 0.95,
        "radius_method": str(signature["radius_method"]),
        "radius_lambda": float(signature["radius_lambda"]),
        "center_mode": "class_centroid_mixture",
        "distance_metric": str(signature["distance_metric"]),
        "margin_gamma": None,
        "covariance_eps": 1e-6,
        "l2_normalize": True,
        "subcenters_per_intent": max((len(value) for value in intent_to_clusters.values()), default=1),
        "subcenters_overrides": {},
        "random_state": 42,
        "acceptance_mode": str(signature["acceptance_mode"]),
        "intent_to_cluster": intent_to_cluster,
        "intent_to_clusters": dict(intent_to_clusters),
        "cluster_to_intent": cluster_to_intent,
        "spheres": [
            {
                "center": sphere["center"],
                "radius": float(sphere["radius"]),
                "cluster_id": int(sphere["cluster_id"]),
                "intent_name": str(sphere["intent"]),
                "inv_diag_cov": sphere["inv_diag_cov"],
            }
            for sphere in spheres
        ],
    }


def _write_detector(path: Path, detector: MultiSphereOOSDetector, radius_scale: float) -> None:
    signature = detector_signature(detector)
    signature["spheres"] = [
        {**sphere, "radius": float(sphere["radius"]) * float(radius_scale)}
        for sphere in signature["spheres"]
    ]
    state = _detector_state(signature)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _run_pipeline(
    dataset: str,
    seed: int,
    device: torch.device,
    detector: MultiSphereOOSDetector,
    radius_scale: float,
    h1_root: Path,
    temporary_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detector_path = temporary_root / f"{dataset}_seed{seed}_{radius_scale:.8f}.json"
    _write_detector(detector_path, detector, radius_scale)
    pipeline = _make_pipeline(dataset, seed, device, detector_path, h1_root)
    views = _read_views(dataset, seed)
    predictions = pipeline.predict_batch([str(row["text"]) for row in views["test"]], batch_size=128)
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gate_scores = np.asarray([float(item.get("gate_score", 0.0)) for item in predictions], dtype=np.float64)
    gate_preds = np.asarray([int(bool(item.get("is_oos", item.get("gate_pred", 1)))) for item in predictions], dtype=np.int64)
    return predictions, {
        "detector_path": str(detector_path),
        "prediction_count": len(predictions),
        "gate_oos_count": int(np.sum(gate_preds)),
        "gate_score_min": float(np.min(gate_scores)),
        "gate_score_max": float(np.max(gate_scores)),
    }


def _pass_through_predictions(
    dataset: str,
    seed: int,
    device: torch.device,
    detector: MultiSphereOOSDetector,
    test_values: np.ndarray,
    h1_root: Path,
    temporary_root: Path,
) -> list[dict[str, Any]]:
    raw_output = _vectorized_output(detector, test_values)
    radius_scale = float(max(1_000_000.0, 2.0 * np.max(raw_output["score"]) + 1.0))
    predictions, _ = _run_pipeline(dataset, seed, device, detector, radius_scale, h1_root, temporary_root)
    if len(predictions) != len(test_values):
        raise AssertionError(f"pass-through prediction count mismatch: {dataset}/seed{seed}")
    downstream = []
    for index, prediction in enumerate(predictions):
        if prediction.get("domain") is None or prediction.get("intent") is None:
            raise AssertionError(
                f"pass-through downstream prediction missing at {dataset}/seed{seed}/{index}: "
                f"gate_pred={prediction.get('gate_pred')}, gate_score={prediction.get('gate_score')}, "
                f"radius={prediction.get('gate_radius')}"
            )
        downstream.append(
            {
                "domain": str(prediction["domain"]),
                "intent": str(prediction["intent"]),
            }
        )
    return downstream


def _compose_predictions(
    rows: Sequence[Mapping[str, Any]],
    output: Mapping[str, np.ndarray],
    threshold: float,
    downstream: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pred_oos = np.asarray(output["score"] > float(threshold), dtype=np.int64)
    predictions: list[dict[str, Any]] = []
    for index, is_oos in enumerate(pred_oos):
        prediction: dict[str, Any] = {
            "is_oos": bool(is_oos),
            "gate_pred": int(is_oos),
            "fast_gate_pred": int(is_oos),
            "gate_score": float(output["score"][index]),
        }
        if not is_oos:
            prediction.update(downstream[index])
        predictions.append(prediction)
    return predictions


def _full_row(
    gate_row: Mapping[str, Any],
    metrics: Mapping[str, Any],
    stages: Mapping[str, Any],
    mode: str,
    direct_verified: bool,
) -> dict[str, Any]:
    row = {key: gate_row[key] for key in ("dataset", "seed", "k", "radius_lambda", "threshold", "acceptance_mode")}
    row.update(
        {
            "split": "test",
            "full_pipeline_evaluation_mode": mode,
            "direct_pipeline_verified": bool(direct_verified),
            "oos_f1": float(metrics["oos_f1"]),
            "f1_all": float(metrics["f1_all"]),
            "known_macro_f1": float(metrics["known_macro_f1"]),
            "overall_accuracy": float(metrics["overall_accuracy"]),
            "known_recall": float(metrics["known_recall"]),
            "false_accept_rate": float(metrics["false_accept_rate"]),
            "false_reject_rate": float(metrics["false_reject_rate"]),
            "router_error_rate": float(metrics["router_error_rate"]),
            "expert_error_rate": float(metrics["expert_error_rate"]),
        }
    )
    for key, value in stages.items():
        row[f"stage_count_{key}"] = int(value)
    return row


def _summary_rows(rows: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["dataset"]),
                *_key(int(row["k"]), float(row["radius_lambda"]), float(row["threshold"]), str(row["acceptance_mode"])),
            )
        ].append(row)
    result: list[dict[str, Any]] = []
    for (dataset, k, radius_lambda, threshold, mode), group in grouped.items():
        item: dict[str, Any] = {
            "dataset": dataset,
            "configuration_type": kind,
            "k": k,
            "radius_lambda": radius_lambda,
            "threshold": threshold,
            "acceptance_mode": mode,
            "seed_count": len(group),
        }
        for metric in FULL_METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
        paper = PAPER_BY_KIR[(dataset, float(KIR))]
        item["paper_oos_f1"] = float(paper["oos_f1"])
        item["delta_oos_f1_pp"] = float(item["oos_f1_mean"] * 100.0 - paper["oos_f1"])
        item["beats_paper_oos_f1_mean"] = bool(item["delta_oos_f1_pp"] > 0.0)
        result.append(item)
    return sorted(result, key=lambda row: (str(row["dataset"]), -float(row["oos_f1_mean"])))


def run(
    datasets: Sequence[str],
    seeds: Sequence[int],
    device: torch.device,
    h1_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    pipeline_eval.KIR = float(KIR)
    pipeline_eval.CASCADE_ROOT = (
        ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cascade_full"
        / f"gpu_kir{int(round(KIR * 100)):02d}"
    ).resolve()
    pipeline_eval.DATA_ROOT = DATA_ROOT.resolve()
    started = time.time()
    direct_verification_tolerance = (
        5e-3
        if EXTERNAL_OOS_TARGET_SELECTION
        else 1e-3
        if OOS_ONLY_SELECTION
        else 1e-10
    )
    output_root.mkdir(parents=True, exist_ok=True)
    all_gate_rows: list[dict[str, Any]] = []
    all_val_aggregate: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    all_full_rows: list[dict[str, Any]] = []
    selected_full_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="s2c_h1_parameter_pipeline_") as temporary:
        temporary_root = Path(temporary)
        for dataset in datasets:
            cell_cache: dict[int, dict[str, Any]] = {}
            dataset_gate_rows: list[dict[str, Any]] = []
            for seed in seeds:
                views, values = _encode_cell(dataset, seed, device, h1_root)
                gate_rows, outputs = _search_cell(dataset, seed, views, values)
                dataset_gate_rows.extend(gate_rows)
                cell_cache[int(seed)] = {"views": views, "values": values, "outputs": outputs}
            selected, guard_mode = _select(dataset_gate_rows, dataset)
            selected_rows.append(selected)
            all_gate_rows.extend(dataset_gate_rows)
            val_aggregate = _annotate_validation_guard(
                _aggregate(dataset_gate_rows, "val"),
                [row for row in dataset_gate_rows if str(row["split"]) == "val"],
            )
            for row in val_aggregate:
                row["selected"] = _is_config(row, (int(selected["k"]), float(selected["radius_lambda"]), float(selected["threshold"]), str(selected["acceptance_mode"])))
                row["paper_oos_f1"] = PAPER_BY_KIR[(dataset, float(KIR))]["oos_f1"]
                row["selection_guard_mode"] = guard_mode
            all_val_aggregate.extend(val_aggregate)

            for seed in seeds:
                cell = cell_cache[int(seed)]
                views = cell["views"]
                values = cell["values"]
                baseline_detector = _fit_detector(values["train"], views["train"], 1, 1.0, "nearest_sphere")
                downstream = _pass_through_predictions(
                    dataset,
                    int(seed),
                    device,
                    baseline_detector,
                    values["test"],
                    h1_root,
                    temporary_root,
                )
                test_rows = [row for row in dataset_gate_rows if int(row["seed"]) == int(seed) and str(row["split"]) == "test"]
                selected_config = (int(selected["k"]), float(selected["radius_lambda"]), float(selected["threshold"]), str(selected["acceptance_mode"]))
                for gate_row in test_rows:
                    config = (int(gate_row["k"]), float(gate_row["radius_lambda"]), float(gate_row["threshold"]), str(gate_row["acceptance_mode"]))
                    output_key = (int(gate_row["k"]), round(float(gate_row["radius_lambda"]), 8), str(gate_row["acceptance_mode"]))
                    output = cell["outputs"][output_key]["test"]
                    candidate_predictions = _compose_predictions(views["test"], output, float(gate_row["threshold"]), downstream)
                    metrics, stages = compute_metrics(views["test"], candidate_predictions)
                    full_row = _full_row(gate_row, metrics, stages, "derived_from_fixed_downstream_replay", False)
                    full_row["selected_by_validation"] = bool(_is_config(gate_row, selected_config))
                    full_row["paper_known_f1"] = PAPER_BY_KIR[(dataset, float(KIR))]["known_f1"]
                    full_row["paper_accuracy"] = PAPER_BY_KIR[(dataset, float(KIR))]["accuracy"]
                    all_full_rows.append(full_row)
                selected_detector = _fit_detector(
                    values["train"],
                    views["train"],
                    int(selected["k"]),
                    float(selected["radius_lambda"]),
                    str(selected["acceptance_mode"]),
                )
                selected_output = cell["outputs"][(int(selected["k"]), round(float(selected["radius_lambda"]), 8), str(selected["acceptance_mode"]))]["test"]
                derived_row = next(
                    row
                    for row in all_full_rows
                    if str(row["dataset"]) == dataset
                    and int(row["seed"]) == int(seed)
                    and _is_config(row, selected_config)
                )
                direct_predictions, direct_meta = _run_pipeline(
                    dataset,
                    int(seed),
                    device,
                    selected_detector,
                    float(selected["threshold"]),
                    h1_root,
                    temporary_root,
                )
                direct_metrics, direct_stages = compute_metrics(views["test"], direct_predictions)
                direct_gate = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "split": "test",
                    "k": int(selected["k"]),
                    "radius_lambda": float(selected["radius_lambda"]),
                    "threshold": float(selected["threshold"]),
                    "acceptance_mode": str(selected["acceptance_mode"]),
                }
                direct_full_row = _full_row(direct_gate, direct_metrics, direct_stages, "direct_pipeline", True)
                direct_full_row["selected_by_validation"] = True
                direct_full_row["paper_known_f1"] = PAPER_BY_KIR[(dataset, float(KIR))]["known_f1"]
                direct_full_row["paper_accuracy"] = PAPER_BY_KIR[(dataset, float(KIR))]["accuracy"]
                selected_full_rows.append(direct_full_row)
                compare = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "configuration": f"K={selected['k']},lambda={selected['radius_lambda']},threshold={selected['threshold']},{selected['acceptance_mode']}",
                    "direct_prediction_count": int(direct_meta["prediction_count"]),
                }
                for metric in FULL_METRICS:
                    delta = abs(float(direct_metrics[metric]) - float(derived_row[metric]))
                    compare[f"{metric}_abs_delta"] = delta
                    if delta > direct_verification_tolerance:
                        derived_gate = np.asarray(
                            _compose_predictions(
                                views["test"],
                                selected_output,
                                float(selected["threshold"]),
                                downstream,
                            ),
                            dtype=object,
                        )
                        direct_gate = np.asarray(
                            [int(bool(item.get("is_oos", item.get("gate_pred", 1)))) for item in direct_predictions],
                            dtype=np.int64,
                        )
                        derived_gate_values = np.asarray(
                            [int(bool(item["is_oos"])) for item in derived_gate],
                            dtype=np.int64,
                        )
                        raise AssertionError(
                            f"derived/direct mismatch {dataset}/seed{seed}/{metric}: {delta}; "
                            f"gate_mismatches={int(np.sum(direct_gate != derived_gate_values))}; "
                            f"direct_oos={int(np.sum(direct_gate))}; derived_oos={int(np.sum(derived_gate_values))}; "
                            f"config={selected_config}; direct_gate_meta={direct_meta}"
                        )
                compare["max_abs_delta"] = max(float(compare[f"{metric}_abs_delta"]) for metric in FULL_METRICS)
                verification_rows.append(compare)
                del selected_output, selected_detector, baseline_detector
                del direct_predictions, downstream
            for value in cell_cache.values():
                del value["views"], value["values"], value["outputs"]
            if device.type == "cuda":
                torch.cuda.empty_cache()

    candidate_summary = _summary_rows(all_full_rows, "all_test_candidates")
    selected_summary = _summary_rows(selected_full_rows, "selected_test_direct")
    paper_beating = [
        row for row in candidate_summary if bool(row["beats_paper_oos_f1_mean"])
    ]
    best_test_by_dataset = [
        max(
            (row for row in candidate_summary if str(row["dataset"]) == dataset),
            key=lambda row: float(row["oos_f1_mean"]),
        )
        for dataset in datasets
    ]
    write_csv(output_root / "gate_workpoints.csv", all_gate_rows)
    write_csv(output_root / "validation_aggregate.csv", all_val_aggregate)
    write_csv(output_root / "selected_validation.csv", selected_rows)
    write_csv(output_root / "full_pipeline_candidates_test.csv", all_full_rows)
    write_csv(output_root / "full_pipeline_selected_per_seed.csv", selected_full_rows)
    write_csv(output_root / "full_pipeline_candidate_summary.csv", candidate_summary)
    write_csv(output_root / "full_pipeline_selected_summary.csv", selected_summary)
    write_csv(output_root / "paper_beating_candidates.csv", paper_beating or [{"dataset": "none", "note": "no aggregate candidate exceeded paper OOS F1"}])
    write_csv(output_root / "best_test_candidate_by_dataset.csv", best_test_by_dataset)
    write_csv(output_root / "direct_vs_derived_verification.csv", verification_rows)
    manifest = {
        "schema_version": "s2c.historical_trainable_parameter_full_pipeline.v1",
        "stage": "historical_trainable_parameter_full_pipeline_v1",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert",
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "kir": float(KIR),
        "gate_representation": "existing_trainable_last2_minilm_plus_projection_checkpoint",
        "gate_distance": "mahalanobis_diag",
        "gate_radius_method": "mean_std",
        "candidate_k_values": list(OOS_K_VALUES if OOS_ONLY_SELECTION else K_VALUES),
        "candidate_lambda_values": list(OOS_LAMBDA_VALUES if OOS_ONLY_SELECTION else LAMBDA_VALUES),
        "candidate_threshold_values": list(OOS_THRESHOLD_VALUES if OOS_ONLY_SELECTION else THRESHOLD_VALUES),
        "acceptance_modes": list(ACCEPTANCE_MODES),
        "candidate_count_per_seed_dataset": len(_candidate_configs()),
        "selection_split": "validation",
        "selection_uses_validation_oos_labels": True,
        "selection_guard": selected_rows[0].get("selection_guard", "") if selected_rows else "",
        "selection_objective": (
            "validation_oos_f1_at_external_baseline_max_accuracy_then_known_recall"
            if EXTERNAL_OOS_TARGET_SELECTION
            else "validation_oos_f1_only"
            if OOS_ONLY_SELECTION
            else "validation_oos_f1_with_known_f1_accuracy_guard"
        ),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "downstream_router_expert_fixed": True,
        "full_pipeline_candidate_mode": "derived_from_one_permissive_gate_downstream_replay",
        "selected_direct_pipeline_verified": True,
        "raw_predictions_written": False,
        "device": str(device),
        "h1_trainable_artifact_root": str(h1_root.resolve()),
        "paper_reference": f"fulltex.tex KIR={KIR:.2f} Ours row",
        "paper_oos_f1": {dataset: PAPER_BY_KIR[(dataset, float(KIR))]["oos_f1"] for dataset in datasets},
        "completed_gate_rows": len(all_gate_rows),
        "completed_full_candidate_rows": len(all_full_rows),
        "completed_selected_direct_rows": len(selected_full_rows),
        "paper_beating_candidate_count": len(paper_beating),
        "paper_beating_datasets": sorted({str(row["dataset"]) for row in paper_beating}),
        "direct_vs_derived_max_abs_delta": max(float(row["max_abs_delta"]) for row in verification_rows),
        "direct_vs_derived_tolerance": direct_verification_tolerance,
        "elapsed_seconds": time.time() - started,
    }
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    global KIR, OOS_ONLY_SELECTION, EXTERNAL_OOS_TARGET_SELECTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--seed", choices=SEEDS, type=int, action="append")
    parser.add_argument("--kir", choices=(0.25, 0.50, 0.75), type=float, default=0.50)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--oos-only", action="store_true", help="Select candidates by validation OOS F1 only; allow Known metrics to fall.")
    selection_group.add_argument("--oos-sota-target", action="store_true", help="Require validation OOS F1 to reach the external baseline target, then maximize Accuracy and Known Recall.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--h1-root", type=Path, default=DEFAULT_H1_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    KIR = float(args.kir)
    EXTERNAL_OOS_TARGET_SELECTION = bool(args.oos_sota_target)
    OOS_ONLY_SELECTION = bool(args.oos_only or args.oos_sota_target)
    device = choose_device(args.device)
    datasets = tuple(args.dataset or DATASETS)
    seeds = tuple(args.seed or SEEDS)
    print(json.dumps(run(datasets, seeds, device, args.h1_root.resolve(), args.output_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
