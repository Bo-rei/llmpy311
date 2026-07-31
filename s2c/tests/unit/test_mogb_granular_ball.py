import numpy as np

from protocol_v2.experiments.mogb import (
    AdaptiveGranularBallClusterer,
    _nearest_center_assignment,
    make_mogb_boundaries,
    score_mogb_boundaries,
)


def test_memory_efficient_assignment_matches_broadcast_euclidean_reference() -> None:
    rng = np.random.RandomState(20260731)
    points = rng.normal(size=(250, 64))
    centers = rng.normal(size=(7, 64))
    reference = np.argmin(np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2), axis=1)
    assert np.array_equal(_nearest_center_assignment(points, centers), reference)


def test_mogb_ball_center_radius_and_determinism() -> None:
    points = np.asarray([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0], [5.2, 0.0]])
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    first = AdaptiveGranularBallClusterer(
        purity_get_ball=1.0,
        purity_select_ball=1.0,
        min_ball_get_ball=1,
        min_ball_select_ball=1,
        seed=42,
    ).fit(points, labels)
    second = AdaptiveGranularBallClusterer(
        purity_get_ball=1.0,
        purity_select_ball=1.0,
        min_ball_get_ball=1,
        min_ball_select_ball=1,
        seed=42,
    ).fit(points, labels)
    assert [ball.sample_indices.tolist() for ball in first.selected_balls] == [ball.sample_indices.tolist() for ball in second.selected_balls]
    assert len(first.selected_balls) == 2
    ball = next(ball for ball in first.selected_balls if ball.majority_label == "a")
    assert np.allclose(ball.centroid, [0.1, 0.0])
    assert np.isclose(ball.radius, 0.1)


def test_mogb_nearest_ball_rejects_outside_point() -> None:
    points = np.asarray([[0.0], [0.2], [5.0], [5.2]])
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    clusterer = AdaptiveGranularBallClusterer(
        purity_get_ball=1.0,
        purity_select_ball=1.0,
        min_ball_get_ball=1,
        min_ball_select_ball=1,
        seed=0,
    ).fit(points, labels)
    boundaries = make_mogb_boundaries(clusterer)
    output = score_mogb_boundaries(np.asarray([[0.1], [10.0]]), boundaries)
    assert output["predicted_label"].tolist() == ["a", "oos"]
    assert output["predicted_oos"].tolist() == [0, 1]


def test_mogb_stops_on_pure_or_tiny_ball() -> None:
    points = np.asarray([[0.0], [0.1], [0.2]])
    labels = np.asarray(["a", "a", "b"], dtype=object)
    clusterer = AdaptiveGranularBallClusterer(
        purity_get_ball=1.0,
        purity_select_ball=0.5,
        min_ball_get_ball=10,
        min_ball_select_ball=1,
        seed=0,
    ).fit(points, labels)
    assert clusterer.selected_balls
    assert all(ball.stop_reason == "minimum_samples" for ball in clusterer.selected_balls)
