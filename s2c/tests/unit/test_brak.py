from __future__ import annotations

import numpy as np

from protocol_v2.experiments.brak import evaluate_intent_candidates, selected_partition


def _points(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    first = rng.normal(0.0, 0.15, size=(24, 4))
    second = rng.normal(2.0, 0.15, size=(24, 4))
    return np.vstack([first, second]), np.asarray(["a"] * 24 + ["b"] * 24, dtype=object)


def test_brak_selection_is_reproducible_and_has_known_only_contract() -> None:
    points, _ = _points()
    calibration = points + 0.01
    one = evaluate_intent_candidates(
        "a",
        points[:24],
        calibration[:24],
        calibration[24:],
        seed=42,
        bootstrap_repeats=2,
    )
    two = evaluate_intent_candidates(
        "a",
        points[:24],
        calibration[:24],
        calibration[24:],
        seed=42,
        bootstrap_repeats=2,
    )
    assert one.selected_k == two.selected_k
    assert [row.objective for row in one.candidates] == [row.objective for row in two.candidates]
    assert all(row.proper_recall >= 0.0 and row.proper_recall <= 1.0 for row in one.candidates)
    assert all(row.safe for row in one.candidates if row.proper_recall >= one.candidates[0].proper_recall - 0.02)


def test_selected_partition_preserves_intents_and_uses_local_labels() -> None:
    points, intents = _points()
    selections = {}
    for intent in ("a", "b"):
        mask = intents == intent
        target = points[mask]
        other = points[~mask]
        selections[intent] = evaluate_intent_candidates(
            intent,
            target,
            target,
            other,
            seed=42,
            bootstrap_repeats=1,
        )
    labels, centers, cluster_to_intent, intent_to_clusters = selected_partition(points, intents, selections)
    assert labels.shape == (points.shape[0],)
    assert centers.shape[0] == len(cluster_to_intent)
    assert set(cluster_to_intent.values()) == {"a", "b"}
    assert set(intent_to_clusters) == {"a", "b"}
    assert np.all(np.isin(labels, list(cluster_to_intent)))


def test_candidate_k_is_bounded_by_available_samples() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    result = evaluate_intent_candidates("a", points, points, np.empty((0, 2)), max_k=5, bootstrap_repeats=1)
    assert [candidate.k for candidate in result.candidates] == [1, 2, 3]

