"""Union-risk calibrated subcenter selection for the active protocol.

This module deliberately keeps the existing ``MultiSphereOOSDetector``
geometry.  URCSG only decides whether one intent may use more than one
subcenter.  All selection statistics are computed from proper-train and
known-calibration data; test rows are accepted by callers only after the
selection maps have been frozen.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector


EPSILON_COVERAGE = 0.02
RHO_UNION_RISK = 0.05
WILSON_Z = 1.959963984540054


@dataclass(frozen=True)
class RiskEstimate:
    """Known-only incremental union-risk estimate for one intent/candidate."""

    intent: str
    candidate_k: int
    eligible_episode_count: int
    pseudo_oos_count: int
    incremental_accept_count: int
    delta_union_risk: float
    union_risk_ucb95: float
    shuffled_delta_union_risk: float
    shuffled_union_risk_ucb95: float
    known_coverage_k: float
    known_coverage_k1: float
    coverage_delta: float
    q95_normalized_score_k: float
    q95_normalized_score_k1: float
    feasible: bool
    shuffled_feasible: bool


def wilson_upper(successes: int, total: int, z: float = WILSON_Z) -> float:
    """Return a finite-sample Wilson upper confidence bound."""

    successes = int(successes)
    total = int(total)
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total")
    if total == 0:
        return math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return float(min(1.0, (center + spread) / denominator))


def fit_detector(
    train_embeddings: np.ndarray,
    train_intents: np.ndarray,
    *,
    distance: str,
    overrides: Mapping[str, int] | None = None,
    seed: int = 42,
) -> MultiSphereOOSDetector:
    """Fit the active detector with K=1 by default and target overrides."""

    detector = MultiSphereOOSDetector(
        radius_method="mean_std",
        radius_lambda=1.0,
        center_mode="class_centroid_mixture",
        distance_metric=str(distance),
        covariance_eps=1e-6,
        l2_normalize=True,
        subcenters_per_intent=1,
        subcenters_overrides={str(k): int(v) for k, v in (overrides or {}).items()},
        random_state=int(seed),
        acceptance_mode="nearest_sphere",
    )
    detector.fit(np.asarray(train_embeddings), np.asarray(train_intents, dtype=object))
    return detector


def _sphere_output(
    detector: MultiSphereOOSDetector,
    embeddings: np.ndarray,
    *,
    excluded_intent: str | None = None,
) -> dict[str, np.ndarray]:
    """Score with an optional leave-one-intent-out sphere mask.

    The raw-nearest-sphere acceptance semantics match the frozen E2 contract.
    The mask is used only for a calibration episode and never changes a final
    test detector.
    """

    values = detector._normalize_embeddings(np.asarray(embeddings))
    spheres = [
        sphere
        for sphere in detector.spheres
        if excluded_intent is None or str(sphere.intent_name) != str(excluded_intent)
    ]
    if not spheres:
        raise ValueError("Leave-one-intent-out episode removed every sphere")
    preds: list[int] = []
    scores: list[float] = []
    nearest: list[int] = []
    distances: list[float] = []
    radii: list[float] = []
    accepted_counts: list[int] = []
    for value in values:
        raw_distances = np.asarray([detector._distance(value, sphere) for sphere in spheres], dtype=np.float64)
        selected = int(np.argmin(raw_distances))
        distance = float(raw_distances[selected])
        radius = max(float(spheres[selected].radius), 1e-12)
        ratios = raw_distances / np.maximum(np.asarray([sphere.radius for sphere in spheres], dtype=np.float64), 1e-12)
        accepted = bool(distance <= radius)
        preds.append(0 if accepted else 1)
        scores.append(float(distance / radius))
        # ``MultiSphereOOSDetector`` exposes globally assigned cluster IDs, but
        # a leave-one-intent-out episode filters the sphere list.  Return the
        # position in the filtered list so downstream label lookup remains
        # correct even when cluster IDs are non-contiguous.
        nearest.append(selected)
        distances.append(distance)
        radii.append(radius)
        accepted_counts.append(int(np.sum(ratios <= 1.0)))
    return {
        "pred": np.asarray(preds, dtype=np.int64),
        "score": np.asarray(scores, dtype=np.float64),
        "nearest_cluster": np.asarray(nearest, dtype=np.int64),
        "distance": np.asarray(distances, dtype=np.float64),
        "radius": np.asarray(radii, dtype=np.float64),
        "accepted_sphere_count": np.asarray(accepted_counts, dtype=np.int64),
    }


def _binary_labels(rows: Iterable[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [0 if str(row.get("oos_source", "known")) == "known" else 1 for row in rows],
        dtype=np.int64,
    )


def open_metrics(
    rows: list[Mapping[str, Any]],
    output: Mapping[str, np.ndarray],
    detector: MultiSphereOOSDetector,
    *,
    threshold: float = 1.0,
) -> dict[str, float]:
    """Compute the common Gate and open-intent metrics for final test rows."""

    labels = _binary_labels(rows)
    scores = np.asarray(output["score"], dtype=np.float64)
    predicted_oos = (scores > float(threshold)).astype(np.int64)
    binary = compute_binary_oos_metrics(labels, scores, threshold=float(threshold))
    known_intents = sorted({str(row["intent"]) for row, label in zip(rows, labels) if label == 0})
    oos_label = "__oos__"
    gold = [str(row["intent"]) if label == 0 else oos_label for row, label in zip(rows, labels)]
    predicted: list[str] = []
    for is_oos, cluster in zip(predicted_oos, np.asarray(output["nearest_cluster"], dtype=np.int64)):
        if int(is_oos):
            predicted.append(oos_label)
        else:
            predicted.append(str(detector.spheres[int(cluster)].intent_name))
    all_labels = known_intents + [oos_label]
    known_gold = [str(row["intent"]) for row, label in zip(rows, labels) if label == 0]
    known_pred = [value for value, label in zip(predicted, labels) if label == 0]
    return {
        **binary,
        "known_recall": float(binary["id_recall"]),
        "known_macro_f1": float(f1_score(known_gold, known_pred, labels=known_intents, average="macro", zero_division=0)),
        "f1_all": float(f1_score(gold, predicted, labels=all_labels, average="macro", zero_division=0)),
        "f1_k": float(f1_score(gold, predicted, labels=known_intents, average="macro", zero_division=0)),
        "f1_u": float(binary["oos_f1"]),
        "accuracy": float(accuracy_score(gold, predicted)),
    }


def _bootstrap_upper_lower(values: np.ndarray, *, seed: int, repeats: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, values.size, size=(int(repeats), values.size))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_summary(deltas: Iterable[float], *, seed: int = 20260725) -> dict[str, float | int]:
    """Summarize paired differences with deterministic bootstrap and W/T/L."""

    values = np.asarray(list(deltas), dtype=np.float64)
    low, high = _bootstrap_upper_lower(values, seed=seed)
    wins = int(np.sum(values > 1e-12))
    ties = int(np.sum(np.abs(values) <= 1e-12))
    losses = int(np.sum(values < -1e-12))
    return {
        "n": int(values.size),
        "mean_delta": float(np.mean(values)) if values.size else math.nan,
        "std_delta": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "ci95_low": low,
        "ci95_high": high,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _episode_estimate(
    *,
    intent: str,
    candidate_k: int,
    baseline_detector: MultiSphereOOSDetector,
    candidate_detector: MultiSphereOOSDetector,
    calibration_embeddings: np.ndarray,
    calibration_intents: np.ndarray,
    known_intents: list[str],
    seed: int,
) -> tuple[int, int, int, int, int]:
    """Return actual and shuffled successes/counts without refitting episodes."""

    eligible = [h for h in known_intents if h != str(intent) and np.any(calibration_intents == h)]
    if len(known_intents) - 1 < 2:
        return 0, 0, 0, 0, 0
    actual_success = actual_total = 0
    episode_values: list[np.ndarray] = []
    for held_out in eligible:
        values = calibration_embeddings[calibration_intents == held_out]
        episode_values.append(values)
        baseline = _sphere_output(baseline_detector, values, excluded_intent=held_out)
        candidate = _sphere_output(candidate_detector, values, excluded_intent=held_out)
        incremental = (candidate["pred"] == 0) & (baseline["pred"] == 1)
        actual_success += int(np.sum(incremental))
        actual_total += int(values.shape[0])
    if not episode_values:
        return len(eligible), actual_total, actual_success, 0, 0
    intent_seed = int.from_bytes(hashlib.sha256(str(intent).encode("utf-8")).digest()[:4], "big")
    rng = np.random.default_rng(int(seed) + 1009 * int(candidate_k) + 7919 * intent_seed)
    order = rng.permutation(len(episode_values))
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    shuffled_success = shuffled_total = 0
    for index, held_out in enumerate(eligible):
        values = episode_values[int(order[index])]
        baseline = _sphere_output(baseline_detector, values, excluded_intent=held_out)
        candidate = _sphere_output(candidate_detector, values, excluded_intent=held_out)
        incremental = (candidate["pred"] == 0) & (baseline["pred"] == 1)
        shuffled_success += int(np.sum(incremental))
        shuffled_total += int(values.shape[0])
    return len(eligible), actual_total, actual_success, shuffled_total, shuffled_success


def estimate_target_risk(
    *,
    intent: str,
    candidate_k: int,
    baseline_detector: MultiSphereOOSDetector,
    candidate_detector: MultiSphereOOSDetector,
    calibration_embeddings: np.ndarray,
    calibration_intents: np.ndarray,
    known_intents: list[str],
    seed: int,
    epsilon_coverage: float = EPSILON_COVERAGE,
    rho_union_risk: float = RHO_UNION_RISK,
) -> RiskEstimate:
    """Estimate target-specific marginal union risk and feasibility."""

    own = np.asarray(calibration_embeddings)[np.asarray(calibration_intents) == str(intent)]
    base_output = _sphere_output(baseline_detector, own)
    candidate_output = _sphere_output(candidate_detector, own)
    coverage_k1 = float(np.mean(base_output["pred"] == 0)) if own.shape[0] else math.nan
    coverage_k = float(np.mean(candidate_output["pred"] == 0)) if own.shape[0] else math.nan
    q95_k1 = float(np.quantile(base_output["score"], 0.95)) if own.shape[0] else math.nan
    q95_k = float(np.quantile(candidate_output["score"], 0.95)) if own.shape[0] else math.nan
    eligible, total, success, shuffled_total, shuffled_success = _episode_estimate(
        intent=str(intent),
        candidate_k=int(candidate_k),
        baseline_detector=baseline_detector,
        candidate_detector=candidate_detector,
        calibration_embeddings=np.asarray(calibration_embeddings),
        calibration_intents=np.asarray(calibration_intents, dtype=object),
        known_intents=list(known_intents),
        seed=int(seed),
    )
    delta = float(success / total) if total else 0.0
    shuffled_delta = float(shuffled_success / shuffled_total) if shuffled_total else 0.0
    ucb = 0.0 if int(candidate_k) == 1 else wilson_upper(success, total)
    shuffled_ucb = 0.0 if int(candidate_k) == 1 else wilson_upper(shuffled_success, shuffled_total)
    feasible = bool(
        np.isfinite(coverage_k1)
        and np.isfinite(coverage_k)
        and coverage_k >= coverage_k1 - float(epsilon_coverage)
        and ucb <= float(rho_union_risk)
    )
    shuffled_feasible = bool(
        np.isfinite(coverage_k1)
        and np.isfinite(coverage_k)
        and coverage_k >= coverage_k1 - float(epsilon_coverage)
        and shuffled_ucb <= float(rho_union_risk)
    )
    return RiskEstimate(
        intent=str(intent),
        candidate_k=int(candidate_k),
        eligible_episode_count=int(eligible),
        pseudo_oos_count=int(total),
        incremental_accept_count=int(success),
        delta_union_risk=delta,
        union_risk_ucb95=float(ucb),
        shuffled_delta_union_risk=shuffled_delta,
        shuffled_union_risk_ucb95=float(shuffled_ucb),
        known_coverage_k=coverage_k,
        known_coverage_k1=coverage_k1,
        coverage_delta=float(coverage_k - coverage_k1),
        q95_normalized_score_k=q95_k,
        q95_normalized_score_k1=q95_k1,
        feasible=feasible,
        shuffled_feasible=shuffled_feasible,
    )


def select_k(
    estimates: Iterable[RiskEstimate],
    *,
    strategy: str,
    shuffled: bool = False,
) -> RiskEstimate:
    """Select one candidate using a pre-registered Known-only strategy."""

    values = list(estimates)
    if not values:
        raise ValueError("URCSG requires at least one candidate estimate")
    feasible = [item for item in values if (item.shuffled_feasible if shuffled else item.feasible)]
    if not feasible:
        feasible = [item for item in values if item.candidate_k == 1]
    if not feasible:
        raise RuntimeError("K=1 must be present as the safe URCSG fallback")
    if strategy == "urcsg_largest_feasible":
        return max(feasible, key=lambda item: item.candidate_k)
    if strategy == "urcsg_min_q95":
        return min(feasible, key=lambda item: (item.q95_normalized_score_k, item.candidate_k))
    raise ValueError(f"Unknown URCSG strategy: {strategy}")


def estimate_target_risk_rows(**kwargs: Any) -> list[dict[str, Any]]:
    """Serialize candidate estimates for CSV/manifest consumers."""

    estimates: list[RiskEstimate] = kwargs.pop("estimates")
    primary = kwargs.pop("primary")
    largest = kwargs.pop("largest")
    shuffled_primary = kwargs.pop("shuffled_primary")
    shuffled_largest = kwargs.pop("shuffled_largest")
    rows: list[dict[str, Any]] = []
    for item in estimates:
        rows.append(
            {
                "intent": item.intent,
                "candidate_k": item.candidate_k,
                "eligible_episode_count": item.eligible_episode_count,
                "pseudo_oos_count": item.pseudo_oos_count,
                "incremental_accept_count": item.incremental_accept_count,
                "delta_union_risk": item.delta_union_risk,
                "union_risk_ucb95": item.union_risk_ucb95,
                "shuffled_delta_union_risk": item.shuffled_delta_union_risk,
                "shuffled_union_risk_ucb95": item.shuffled_union_risk_ucb95,
                "known_coverage": item.known_coverage_k,
                "known_coverage_k1": item.known_coverage_k1,
                "coverage_delta": item.coverage_delta,
                "q95_normalized_score": item.q95_normalized_score_k,
                "q95_normalized_score_k1": item.q95_normalized_score_k1,
                "feasible": item.feasible,
                "shuffled_feasible": item.shuffled_feasible,
                # The generic fields are the primary (min-q95) selector
                # contract.  Strategy-specific fields remain below for the
                # registered ablation and shuffled negative control.
                "selected_k": primary.candidate_k,
                "selection_reason": "min_q95_among_feasible",
                "ineligible": bool(item.eligible_episode_count < 1 or item.pseudo_oos_count == 0),
                "skip_reason": (
                    "fewer_than_two_remaining_known_intents_or_empty_calibration"
                    if item.eligible_episode_count < 1 or item.pseudo_oos_count == 0
                    else ""
                ),
                "selected_k_primary": primary.candidate_k,
                "selected_k_largest": largest.candidate_k,
                "selected_k_shuffled_primary": shuffled_primary.candidate_k,
                "selected_k_shuffled_largest": shuffled_largest.candidate_k,
                "selection_reason_primary": "min_q95_among_feasible",
                "selection_reason_largest": "largest_feasible",
            }
        )
    return rows
