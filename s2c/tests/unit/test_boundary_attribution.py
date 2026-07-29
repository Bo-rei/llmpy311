from __future__ import annotations

import numpy as np

from protocol_v2.experiments.boundary_attribution import (
    EXPECTED_UNITS,
    BoundaryModel,
    _sphere_variances,
    build_plan,
    fit_boundary,
    score_boundary,
)


def test_attribution_plan_is_pre_registered_and_unique() -> None:
    plan = build_plan()
    assert len(plan) == EXPECTED_UNITS == 60
    assert len({spec.run_id for spec in plan}) == EXPECTED_UNITS
    assert sum(spec.phase == "A_covariance" for spec in plan) == 33
    assert sum(spec.phase == "B_score" for spec in plan) == 9
    assert sum(spec.phase == "C_radius" for spec in plan) == 9
    assert sum(spec.phase == "D_final_k1" for spec in plan) == 9


def test_shared_intent_covariance_is_equal_within_intent() -> None:
    train = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [1.0, 0.0],
            [1.4, 0.2],
            [0.0, 3.0],
            [0.1, 3.8],
        ],
        dtype=np.float64,
    )
    assignments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    centers = np.asarray([[0.1, 0.05], [1.2, 0.1], [0.05, 3.4]], dtype=np.float64)
    variances, inverse = _sphere_variances(
        train,
        assignments,
        centers,
        ("a", "a", "b"),
        "shared_intent_diag",
        1e-6,
    )
    assert variances is not None and inverse is not None
    assert np.array_equal(variances[0], variances[1])
    assert not np.array_equal(variances[0], variances[2])
    assert np.allclose(inverse, 1.0 / variances)


def test_per_cluster_covariance_can_differ() -> None:
    train = np.asarray([[0.0], [0.2], [1.0], [2.0]], dtype=np.float64)
    assignments = np.asarray([0, 0, 1, 1], dtype=np.int64)
    centers = np.asarray([[0.1], [1.5]], dtype=np.float64)
    variances, _ = _sphere_variances(
        train,
        assignments,
        centers,
        ("a", "a"),
        "per_cluster_diag",
        1e-6,
    )
    assert variances is not None
    assert variances[0, 0] < variances[1, 0]


def test_raw_nearest_and_normalized_min_can_select_different_spheres() -> None:
    model = BoundaryModel(
        centers=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        intents=("a", "a"),
        assignments=np.asarray([0, 1]),
        inverse_variances=None,
        variances=None,
        radii=np.asarray([0.05, 0.50]),
        cluster_sizes=np.asarray([1, 1]),
        covariance_scope="euclidean",
        radius_rule="mean_std",
    )
    probe = np.asarray([[0.8, 0.2]], dtype=np.float64)
    raw = score_boundary(probe, model, "raw_distance_nearest")
    normalized = score_boundary(probe, model, "normalized_score_min")
    assert int(raw["selected_sphere"][0]) == 0
    assert int(normalized["selected_sphere"][0]) == 1
    assert float(raw["score"][0]) > float(normalized["score"][0])


def test_quantile_radius_uses_known_train_distances_only() -> None:
    train = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.20],
            [0.92, -0.39],
            [-1.0, 0.0],
            [-0.98, -0.20],
            [-0.92, 0.39],
        ],
        dtype=np.float64,
    )
    intents = ["a", "a", "a", "b", "b", "b"]
    model = fit_boundary(
        train,
        intents,
        k=1,
        covariance_scope="euclidean",
        radius_rule="quantile_95",
        covariance_eps=1e-6,
        radius_lambda=1.0,
        quantile=0.95,
    )
    normalized = train / np.linalg.norm(train, axis=1, keepdims=True)
    for sphere_id, intent in enumerate(model.intents):
        points = normalized[np.asarray(intents) == intent]
        expected = np.quantile(np.linalg.norm(points - model.centers[sphere_id], axis=1), 0.95)
        assert np.isclose(model.radii[sphere_id], expected)
