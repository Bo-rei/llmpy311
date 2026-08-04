"""Competition-Calibrated Support Gate (CCSG).

CCSG keeps the active protocol's centres, diagonal Mahalanobis geometry and
mean-plus-standard-deviation radii.  It changes only the final aggregation:
subcentres form one class-level support score instead of independent union
acceptors, and the winning class must also beat its runner-up by a calibrated
margin.  All calibration helpers are Known-only and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, roc_curve

from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector


EPS = 1e-8
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TARGET_FALSE_REJECTION = 0.05


@dataclass(frozen=True)
class SupportMarginFeatures:
    """Continuous class-support and competition features for a batch."""

    support: np.ndarray
    margin: np.ndarray
    raw_score: np.ndarray
    raw_intent: np.ndarray
    top_intent: np.ndarray
    runner_up: np.ndarray


@dataclass(frozen=True)
class Calibration:
    """Known-only operating point for one CCSG ablation."""

    method: str
    target_false_rejection: float
    threshold: float
    support_mean: float
    support_std: float
    margin_mean: float
    margin_std: float
    support_threshold: float | None
    margin_threshold: float | None
    calibration_false_rejection: float


def _upper_order_statistic(values: np.ndarray, target_false_rejection: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("CCSG calibration requires finite non-empty Known scores")
    if not 0.0 <= target_false_rejection < 1.0:
        raise ValueError("target_false_rejection must be in [0, 1)")
    rank = min(int(math.ceil((values.size + 1) * (1.0 - target_false_rejection))), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


def _intent_spheres(detector: MultiSphereOOSDetector) -> dict[str, list[Any]]:
    by_intent: dict[str, list[Any]] = {}
    for sphere in detector.spheres:
        by_intent.setdefault(str(sphere.intent_name), []).append(sphere)
    if not by_intent:
        raise ValueError("CCSG requires a fitted detector with labelled spheres")
    return by_intent


def _sphere_weights(detector: MultiSphereOOSDetector) -> dict[int, float]:
    labels = np.asarray(detector._train_cluster_labels, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("CCSG requires non-empty detector training labels")
    counts = {int(cluster_id): int(np.sum(labels == int(cluster_id))) for cluster_id in range(detector.n_clusters)}
    by_intent: dict[str, int] = {}
    for sphere in detector.spheres:
        by_intent[str(sphere.intent_name)] = by_intent.get(str(sphere.intent_name), 0) + counts.get(int(sphere.cluster_id), 0)
    result: dict[int, float] = {}
    for sphere in detector.spheres:
        total = max(by_intent.get(str(sphere.intent_name), 0), 1)
        result[int(sphere.cluster_id)] = max(counts.get(int(sphere.cluster_id), 0), 1) / total
    return result


def _logsumexp(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    pivot = float(np.max(array))
    return float(pivot + np.log(np.exp(array - pivot).sum()))


def support_margin_features(
    detector: MultiSphereOOSDetector,
    embeddings: np.ndarray,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> SupportMarginFeatures:
    """Compute one support score per intent and the top-two class margin.

    The raw score is the minimum normalized distance over all spheres and is
    retained only for the current union baseline.  CCSG never uses that raw
    minimum as its acceptance rule.
    """

    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("CCSG temperature must be finite and positive")
    values = detector._normalize_embeddings(np.asarray(embeddings))
    by_intent = _intent_spheres(detector)
    weights = _sphere_weights(detector)
    supports: list[dict[str, float]] = []
    raw_scores: list[float] = []
    raw_intents: list[str] = []
    top: list[str] = []
    runner: list[str] = []
    margins: list[float] = []
    for value in values:
        intent_scores: dict[str, float] = {}
        ratios: list[float] = []
        ratio_intents: list[str] = []
        for intent, spheres in by_intent.items():
            terms: list[float] = []
            for sphere in spheres:
                distance = detector._distance(value, sphere)
                radius = max(float(sphere.radius), EPS)
                ratio = distance / radius
                ratios.append(ratio)
                ratio_intents.append(intent)
                terms.append(math.log(max(weights[int(sphere.cluster_id)], EPS)) - (ratio * ratio) / (2.0 * temperature))
            intent_scores[intent] = _logsumexp(terms)
        ranked = sorted(intent_scores.items(), key=lambda pair: (-pair[1], pair[0]))
        best, second = ranked[0], ranked[1] if len(ranked) > 1 else ("__none__", -math.inf)
        top.append(best[0])
        runner.append(second[0])
        supports.append(intent_scores)
        margins.append(float(best[1] - second[1]))
        raw_index = int(np.argmin(np.asarray(ratios, dtype=np.float64)))
        raw_scores.append(float(ratios[raw_index]))
        raw_intents.append(ratio_intents[raw_index])
    support_values = np.asarray([max(item.values()) for item in supports], dtype=np.float64)
    return SupportMarginFeatures(
        support=support_values,
        margin=np.asarray(margins, dtype=np.float64),
        raw_score=np.asarray(raw_scores, dtype=np.float64),
        raw_intent=np.asarray(raw_intents, dtype=object),
        top_intent=np.asarray(top, dtype=object),
        runner_up=np.asarray(runner, dtype=object),
    )


def _standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - float(mean)) / max(float(std), EPS)


def calibrate(
    method: str,
    features: SupportMarginFeatures,
    *,
    target_false_rejection: float = DEFAULT_TARGET_FALSE_REJECTION,
) -> Calibration:
    """Fit one threshold from Known calibration only.

    ``current_*`` and ``mixture_support`` use one absolute support score;
    ``margin_only`` isolates the margin; ``ccsg`` uses a joint max of
    standardized support and margin nonconformities.  The independent-AND
    variant is deliberately retained only as an ablation.
    """

    support_nonconf = -np.asarray(features.support, dtype=np.float64)
    margin_nonconf = -np.asarray(features.margin, dtype=np.float64)
    support_mean = float(np.mean(support_nonconf))
    support_std = float(np.std(support_nonconf))
    margin_mean = float(np.mean(margin_nonconf))
    margin_std = float(np.std(margin_nonconf))
    if method in {"current_k1", "current_k2_union", "mixture_support"}:
        source = features.raw_score if method.startswith("current_") else support_nonconf
        threshold = _upper_order_statistic(source, target_false_rejection)
        rejected = source > threshold
        return Calibration(method, target_false_rejection, threshold, support_mean, support_std, margin_mean, margin_std, None, None, float(np.mean(rejected)))
    if method == "margin_only":
        threshold = _upper_order_statistic(margin_nonconf, target_false_rejection)
        rejected = margin_nonconf > threshold
        return Calibration(method, target_false_rejection, threshold, support_mean, support_std, margin_mean, margin_std, None, threshold, float(np.mean(rejected)))
    if method == "ccsg_independent":
        support_threshold = _upper_order_statistic(support_nonconf, target_false_rejection)
        margin_threshold = _upper_order_statistic(margin_nonconf, target_false_rejection)
        rejected = (support_nonconf > support_threshold) | (margin_nonconf > margin_threshold)
        return Calibration(method, target_false_rejection, math.nan, support_mean, support_std, margin_mean, margin_std, support_threshold, margin_threshold, float(np.mean(rejected)))
    if method == "ccsg":
        joint = np.maximum(_standardize(support_nonconf, support_mean, support_std), _standardize(margin_nonconf, margin_mean, margin_std))
        threshold = _upper_order_statistic(joint, target_false_rejection)
        return Calibration(method, target_false_rejection, threshold, support_mean, support_std, margin_mean, margin_std, None, None, float(np.mean(joint > threshold)))
    raise ValueError(f"Unknown CCSG calibration method: {method}")


def apply_calibration(features: SupportMarginFeatures, calibration: Calibration) -> dict[str, np.ndarray]:
    """Apply a frozen calibration and return OOS score/prediction arrays."""

    method = calibration.method
    support_nonconf = -np.asarray(features.support, dtype=np.float64)
    margin_nonconf = -np.asarray(features.margin, dtype=np.float64)
    if method in {"current_k1", "current_k2_union"}:
        scores = np.asarray(features.raw_score, dtype=np.float64)
        predicted_oos = scores > calibration.threshold
    elif method == "mixture_support":
        scores = support_nonconf
        predicted_oos = scores > calibration.threshold
    elif method == "margin_only":
        scores = margin_nonconf
        predicted_oos = scores > calibration.threshold
    elif method == "ccsg_independent":
        z_support = _standardize(support_nonconf, calibration.support_mean, calibration.support_std)
        z_margin = _standardize(margin_nonconf, calibration.margin_mean, calibration.margin_std)
        scores = np.maximum(z_support, z_margin)
        predicted_oos = (support_nonconf > float(calibration.support_threshold)) | (margin_nonconf > float(calibration.margin_threshold))
    elif method == "ccsg":
        z_support = _standardize(support_nonconf, calibration.support_mean, calibration.support_std)
        z_margin = _standardize(margin_nonconf, calibration.margin_mean, calibration.margin_std)
        scores = np.maximum(z_support, z_margin)
        predicted_oos = scores > calibration.threshold
    else:
        raise ValueError(f"Unknown CCSG calibration method: {method}")
    return {
        "score": np.asarray(scores, dtype=np.float64),
        "pred": np.asarray(predicted_oos, dtype=np.int64),
        "top_intent": np.asarray(features.top_intent, dtype=object),
        "runner_up": np.asarray(features.runner_up, dtype=object),
        "support": np.asarray(features.support, dtype=np.float64),
        "margin": np.asarray(features.margin, dtype=np.float64),
        "raw_score": np.asarray(features.raw_score, dtype=np.float64),
        "raw_intent": np.asarray(features.raw_intent, dtype=object),
        "prediction_intent": np.asarray(
            features.raw_intent if method in {"current_k1", "current_k2_union"} else features.top_intent,
            dtype=object,
        ),
    }


def open_metrics(rows: Sequence[Mapping[str, Any]], output: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Evaluate binary OOS and open-intent classification metrics."""

    labels = np.asarray([0 if str(row.get("oos_source", "known")) == "known" else 1 for row in rows], dtype=np.int64)
    scores = np.asarray(output["score"], dtype=np.float64)
    predicted_oos = np.asarray(output["pred"], dtype=np.int64)
    if labels.size != scores.size or labels.size != predicted_oos.size:
        raise ValueError("CCSG outputs must align with test rows")
    known = labels == 0
    oos = labels == 1
    tp = int(np.sum((predicted_oos == 1) & oos))
    fp = int(np.sum((predicted_oos == 1) & known))
    fn = int(np.sum((predicted_oos == 0) & oos))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    if np.unique(labels).size == 2:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        valid_fpr = fpr[tpr >= 0.95]
        auroc = float(roc_auc_score(labels, scores))
        aupr = float(average_precision_score(labels, scores))
        fpr95 = float(valid_fpr.min()) if valid_fpr.size else math.nan
    else:
        auroc = aupr = fpr95 = math.nan
    binary = {
        "oos_precision": float(precision),
        "oos_recall": float(recall),
        "oos_f1": float(f1),
        "oos_rejection": float(recall),
        "auroc": auroc,
        "aupr_oos": aupr,
        "fpr95": fpr95,
        "id_recall": float(np.mean(predicted_oos[known] == 0)) if np.any(known) else math.nan,
        "false_reject_rate": float(np.mean(predicted_oos[known] == 1)) if np.any(known) else math.nan,
        "false_accept_rate": float(np.mean(predicted_oos[oos] == 0)) if np.any(oos) else math.nan,
    }
    known_intents = sorted({str(row["intent"]) for row, label in zip(rows, labels) if label == 0})
    oos_label = "__oos__"
    gold = [str(row["intent"]) if label == 0 else oos_label for row, label in zip(rows, labels)]
    predicted = [oos_label if int(flag) else str(intent) for flag, intent in zip(predicted_oos, output["prediction_intent"])]
    all_labels = known_intents + [oos_label]
    return {
        **binary,
        "known_recall": binary["id_recall"],
        "known_macro_f1": float(f1_score(gold, predicted, labels=known_intents, average="macro", zero_division=0)),
        "f1_all": float(f1_score(gold, predicted, labels=all_labels, average="macro", zero_division=0)),
        "f1_k": float(f1_score(gold, predicted, labels=known_intents, average="macro", zero_division=0)),
        "f1_u": float(binary["oos_f1"]),
        "accuracy": float(accuracy_score(gold, predicted)),
    }


__all__ = [
    "Calibration",
    "SupportMarginFeatures",
    "apply_calibration",
    "calibrate",
    "open_metrics",
    "support_margin_features",
]
