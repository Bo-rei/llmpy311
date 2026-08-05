from __future__ import annotations

import numpy as np

from protocol_v2.experiments.joint_adaptive_v1.runner import (
    _candidate_intents,
    _initial_centers,
    _split_candidate,
)


def _rows(labels: list[str]) -> list[dict[str, object]]:
    return [{"sample_id": str(i), "intent": label, "label": 0} for i, label in enumerate(labels)]


def test_pca_split_is_deterministic_and_balanced() -> None:
    values = np.vstack([np.linspace(-1, 1, 20)[:, None], np.zeros((20, 1))]).astype(np.float32)
    rows = _rows(["a"] * 40)
    left, right = _split_candidate(values, rows, "a", 10, 42)  # type: ignore[misc]
    assert len(left) == 20
    assert len(right) == 20
    left_again, right_again = _split_candidate(values, rows, "a", 10, 42)  # type: ignore[misc]
    assert np.array_equal(left, left_again)
    assert np.array_equal(right, right_again)


def test_candidate_order_uses_train_residual_only() -> None:
    rng = np.random.default_rng(4)
    values = np.vstack([
        rng.normal(0.0, 0.01, size=(40, 4)),
        rng.normal(0.0, 0.20, size=(40, 4)),
    ]).astype(np.float32)
    rows = _rows(["tight"] * 40 + ["wide"] * 40)
    candidates = _candidate_intents(values, rows, 2, 10)
    assert candidates
    assert candidates[0]["intent"] == "wide"


def test_initial_centers_are_one_per_intent() -> None:
    values = np.eye(6, dtype=np.float32)
    rows = _rows(["b", "a", "b", "a", "b", "a"])
    state = _initial_centers(values, rows)
    assert state.center_intents == ("a", "b")
    assert state.centers.shape == (2, 6)
