from __future__ import annotations

import numpy as np

from protocol_v2.experiments.joint_adaptive_contract_repair_v1 import _fit_fixed_parent, _fit_children_with_fixed_parent, _scores
from protocol_v2.experiments.joint_adaptive_v1.runner import CenterState


def _rows():
    return [
        {"intent": "a", "sample_id": "a0", "label": 0},
        {"intent": "a", "sample_id": "a1", "label": 0},
        {"intent": "b", "sample_id": "b0", "label": 0},
        {"intent": "b", "sample_id": "b1", "label": 0},
    ]


def test_parent_boundary_parameters_are_not_refit_for_children():
    rows = _rows()
    values = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    state = CenterState(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), ("a", "b"), np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), ("a", "b"), ())
    parent = _fit_fixed_parent(values, rows, state, 1.0)
    child_state = CenterState(np.asarray([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]], dtype=np.float32), ("a", "a", "b"), state.parent_centers, state.parent_intents, ())
    child, _ = _fit_children_with_fixed_parent(values, rows, child_state, parent, 1.0)
    np.testing.assert_array_equal(child.parent_centers, parent.parent_centers)
    np.testing.assert_array_equal(child.parent_radii, parent.parent_radii)
    np.testing.assert_array_equal(child.parent_inv_diag_cov, parent.parent_inv_diag_cov)


def test_guarded_score_is_distinct_from_unconstrained_score():
    rows = _rows()
    values = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    state = CenterState(values[[0, 2]], ("a", "b"), values[[0, 2]], ("a", "b"), ())
    parent = _fit_fixed_parent(values, rows, state, 1.0)
    child, _ = _fit_children_with_fixed_parent(values, rows, state, parent, 1.0)
    output = _scores(child, np.asarray([[0.0, 1.0]], dtype=np.float32), 0.1)
    assert output["score"].shape == (1,)
    assert output["unconstrained_score"].shape == (1,)
    assert np.all(np.isfinite(output["score"]))


def test_child_diagnostics_report_load_and_separation():
    rows = _rows()
    values = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    parent_state = CenterState(values[[0, 2]], ("a", "b"), values[[0, 2]], ("a", "b"), ())
    parent = _fit_fixed_parent(values, rows, parent_state, 1.0)
    child_state = CenterState(np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32), ("a", "a", "b"), parent_state.parent_centers, parent_state.parent_intents, ())
    _, diagnostics = _fit_children_with_fixed_parent(values, rows, child_state, parent, 1.0)
    assert diagnostics["minimum_child_load"] >= 0
    assert diagnostics["minimum_child_separation"] >= 0
