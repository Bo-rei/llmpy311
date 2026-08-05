"""E2-compatible K=1 boundary and metric functions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score

from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector


def fit_k1_detector(train: np.ndarray, rows: Sequence[Mapping[str, Any]], distance: str) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=1,
        radius_method="mean_std",
        radius_lambda=1.0,
        distance_metric=distance,
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
        acceptance_mode="nearest_sphere",
    )
    detector.fit(np.asarray(train), np.asarray([str(row["intent"]) for row in rows], dtype=object))
    return detector


def detector_signature(detector: MultiSphereOOSDetector) -> dict[str, Any]:
    spheres = []
    for sphere in sorted(detector.spheres, key=lambda value: int(value.cluster_id)):
        spheres.append({
            "cluster_id": int(sphere.cluster_id),
            "intent": sphere.intent_name,
            "center": np.asarray(sphere.center, dtype=float).tolist(),
            "radius": float(sphere.radius),
            "inv_diag_cov": None if sphere.inv_diag_cov is None else np.asarray(sphere.inv_diag_cov, dtype=float).tolist(),
        })
    return {"distance_metric": detector.distance_metric, "radius_method": detector.radius_method, "radius_lambda": detector.radius_lambda, "acceptance_mode": detector.acceptance_mode, "spheres": spheres}


def _open_label(detector: MultiSphereOOSDetector, output: Mapping[str, Any], index: int) -> str:
    if int(output["pred"][index]) == 1:
        return "__oos__"
    cluster = int(output["nearest_cluster"][index])
    return str(detector.cluster_to_intent.get(cluster, "__unknown__"))


def evaluate_open(
    detector: MultiSphereOOSDetector,
    embeddings: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    threshold: float = 1.0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    output = detector.predict_with_scores(np.asarray(embeddings))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    metrics = compute_binary_oos_metrics(labels, output["score"], threshold)
    known_intents = sorted({str(row["intent"]) for row in rows if int(row["label"]) == 0})
    truth = [str(row["intent"]) if int(row["label"]) == 0 else "__oos__" for row in rows]
    predicted = [_open_label(detector, output, i) for i in range(len(rows))]
    all_labels = [*known_intents, "__oos__"]
    metrics.update({
        "f1_all": float(f1_score(truth, predicted, labels=all_labels, average="macro", zero_division=0)),
        "f1_u": float(f1_score(truth, predicted, labels=["__oos__"], average="macro", zero_division=0)),
        "f1_k": float(f1_score(truth, predicted, labels=known_intents, average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(truth, dtype=object) == np.asarray(predicted, dtype=object))),
        "known_recall": float(metrics["id_recall"]),
    })
    predictions = []
    for index, row in enumerate(rows):
        predictions.append({
            "sample_id": row["sample_id"],
            "gold_intent": row["intent"],
            "gold_is_oos": int(row["label"]),
            "predicted_is_oos": int(output["pred"][index]),
            "predicted_intent": predicted[index],
            "nearest_cluster": int(output["nearest_cluster"][index]),
            "distance": float(output["distance"][index]),
            "radius": float(output["radius"][index]),
            "oos_score": float(output["score"][index]),
            "oos_source": row.get("oos_source"),
        })
    return metrics, predictions


def evaluate_known_calibration(detector: MultiSphereOOSDetector, embeddings: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics, predictions = evaluate_open(detector, embeddings, rows, threshold=1.0)
    accepted = np.asarray([int(row["predicted_is_oos"]) == 0 for row in predictions], dtype=bool)
    return {
        "known_recall": float(np.mean(accepted)) if len(accepted) else math.nan,
        "known_f1": float(metrics["f1_k"]),
        "false_reject_rate": float(1.0 - np.mean(accepted)) if len(accepted) else math.nan,
    }


def load_reference_metrics(reference_dir: Path) -> dict[str, Any]:
    return json.loads((reference_dir / "metrics.json").read_text(encoding="utf-8"))["combined"]


def load_reference_predictions(reference_dir: Path) -> list[dict[str, Any]]:
    path = reference_dir / "predictions" / "test.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare_reference(
    reference_dir: Path,
    metrics: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_metrics = load_reference_metrics(reference_dir)
    metric_names = ("oos_f1", "id_recall", "auroc", "aupr_oos", "false_accept_rate", "false_reject_rate")
    deltas = {name: abs(float(metrics[name]) - float(reference_metrics[name])) for name in metric_names}
    reference_predictions = load_reference_predictions(reference_dir)
    if len(reference_predictions) != len(predictions):
        raise AssertionError("RACAL/E2 prediction row count differs")
    mismatch = 0
    score_max_delta = 0.0
    for left, right in zip(reference_predictions, predictions, strict=True):
        if str(left["sample_id"]) != str(right["sample_id"]):
            mismatch += 1
        if int(left["predicted_is_oos"]) != int(right["predicted_is_oos"]):
            mismatch += 1
        score_max_delta = max(score_max_delta, abs(float(left["oos_score"]) - float(right["oos_score"])))
    return {
        "reference_dir": str(reference_dir),
        "prediction_mismatch_count": mismatch,
        "sample_id_mismatch_count": sum(str(left["sample_id"]) != str(right["sample_id"]) for left, right in zip(reference_predictions, predictions, strict=True)),
        "prediction_score_max_abs_delta": score_max_delta,
        "metric_abs_delta": deltas,
        "metric_max_abs_delta": max(deltas.values()),
        "within_tolerance": mismatch == 0 and score_max_delta <= 1e-10 and max(deltas.values()) <= 1e-10,
    }
