from __future__ import annotations

import numpy as np

from protocol_v2.experiments.urcsg import (
    estimate_target_risk,
    fit_detector,
    select_k,
    wilson_upper,
)


def _toy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train = np.asarray(
        [
            [0.0, 0.0], [0.1, 0.0],
            [5.0, 0.0], [5.1, 0.0],
            [0.0, 5.0], [0.0, 5.1],
        ], dtype=np.float64
    )
    intents = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    calibration = train + 0.01
    return train, intents, calibration, intents.copy(), ["a", "b", "c"]


def test_wilson_upper_is_finite_and_conservative():
    assert wilson_upper(0, 10) > 0.0
    assert 0.99 < wilson_upper(10, 10) <= 1.0


def test_urcsg_estimate_has_k1_safe_fallback_and_finite_scores():
    train, intents, calibration, calibration_intents, known = _toy_data()
    baseline = fit_detector(train, intents, distance="euclidean", seed=42)
    candidate = fit_detector(train, intents, distance="euclidean", overrides={"a": 2}, seed=42)
    estimate = estimate_target_risk(
        intent="a",
        candidate_k=2,
        baseline_detector=baseline,
        candidate_detector=candidate,
        calibration_embeddings=calibration,
        calibration_intents=calibration_intents,
        known_intents=known,
        seed=42,
    )
    assert estimate.eligible_episode_count == 2
    assert np.isfinite(estimate.union_risk_ucb95)
    selected = select_k(
        [
            estimate,
            estimate.__class__(
                **{**estimate.__dict__, "candidate_k": 1, "feasible": True, "shuffled_feasible": True}
            ),
        ],
        strategy="urcsg_largest_feasible",
    )
    assert selected.candidate_k in {1, 2}


def test_target_candidate_does_not_change_other_intent_sphere_count():
    train, intents, _, _, _ = _toy_data()
    baseline = fit_detector(train, intents, distance="euclidean", seed=13)
    candidate = fit_detector(train, intents, distance="euclidean", overrides={"a": 2}, seed=13)
    baseline_counts = {name: len(baseline.intent_to_clusters[name]) for name in baseline.intent_to_clusters}
    candidate_counts = {name: len(candidate.intent_to_clusters[name]) for name in candidate.intent_to_clusters}
    assert candidate_counts["a"] == 2
    assert candidate_counts["b"] == baseline_counts["b"] == 1
    assert candidate_counts["c"] == baseline_counts["c"] == 1
