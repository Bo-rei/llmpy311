from __future__ import annotations

import numpy as np
import pytest

from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from protocol_v2.experiments.partitions import (
    build_partition,
    fit_injected_detector,
    normalize_for_detector,
    partition_sizes,
)


def test_random_balanced_partition_sizes_differ_by_at_most_one() -> None:
    embeddings = np.arange(18, dtype=np.float64).reshape(9, 2)
    intents = np.array(["a"] * 5 + ["b"] * 4, dtype=object)
    result = build_partition(embeddings, intents, 3, "random_balanced", 42)
    for intent in ("a", "b"):
        cluster_ids = result.intent_to_clusters[intent]
        sizes = [int(np.sum(result.labels == cluster_id)) for cluster_id in cluster_ids]
        assert max(sizes) - min(sizes) <= 1


def test_partition_seed_is_reproducible() -> None:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(20, 4))
    intents = np.array(["a"] * 10 + ["b"] * 10, dtype=object)
    left = build_partition(embeddings, intents, 2, "random_balanced", 42)
    right = build_partition(embeddings, intents, 2, "random_balanced", 42)
    assert np.array_equal(left.labels, right.labels)
    assert np.array_equal(left.centers, right.centers)


def test_kmeans_seed_42_injected_detector_matches_active_detector() -> None:
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(24, 8))
    intents = np.array(["a"] * 8 + ["b"] * 8 + ["c"] * 8, dtype=object)
    partition = build_partition(normalize_for_detector(embeddings), intents, 2, "kmeans", 42)
    injected = fit_injected_detector(
        embeddings,
        intents,
        partition,
        distance="euclidean",
        radius_lambda=1.0,
        random_state=42,
    )
    reference = MultiSphereOOSDetector(
        radius_method="mean_std",
        radius_lambda=1.0,
        center_mode="class_centroid_mixture",
        distance_metric="euclidean",
        covariance_eps=1e-6,
        l2_normalize=True,
        subcenters_per_intent=2,
        random_state=42,
    )
    reference.fit(embeddings, intents)
    assert np.array_equal(injected._train_cluster_labels, reference._train_cluster_labels)
    assert np.allclose(
        [sphere.center for sphere in injected.spheres],
        [sphere.center for sphere in reference.spheres],
        atol=1e-12,
    )
    assert np.allclose(
        [sphere.radius for sphere in injected.spheres],
        [sphere.radius for sphere in reference.spheres],
        atol=1e-12,
    )
    probe = rng.normal(size=(7, 8))
    left = injected.predict_with_scores(probe)
    right = reference.predict_with_scores(probe)
    assert np.allclose(left["score"], right["score"], atol=1e-12)
    assert np.array_equal(left["pred"], right["pred"])


def test_partition_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown E3 partition"):
        build_partition(np.ones((2, 2)), np.array(["a", "a"]), 2, "bad", 42)


def test_partition_sizes_are_stable() -> None:
    result = build_partition(
        np.arange(24, dtype=np.float64).reshape(12, 2),
        np.array(["a"] * 6 + ["b"] * 6, dtype=object),
        2,
        "random_balanced",
        42,
    )
    assert np.array_equal(partition_sizes(result), np.array([3, 3, 3, 3]))


def test_normalized_union_accepts_a_non_nearest_sphere() -> None:
    """The opt-in union contract must not depend on raw center proximity."""

    nearest_contract = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        radius_method="mean_std",
        acceptance_mode="nearest_sphere",
    )
    union_contract = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        radius_method="mean_std",
        acceptance_mode="normalized_union",
    )
    # A query is closer to sphere 0 in raw distance, but is inside sphere 1.
    # This is exactly the case that raw-nearest d/r scoring mishandles.
    from protocol_v2.gate.multi_sphere_oos_detector import SphereConfig

    spheres = [
        SphereConfig(center=np.asarray([0.0, 0.0]), radius=0.1, cluster_id=0, intent_name="a"),
        SphereConfig(center=np.asarray([0.3, 0.0]), radius=0.5, cluster_id=1, intent_name="a"),
    ]
    for detector in (nearest_contract, union_contract):
        detector.spheres = spheres
        detector.fitted = True

    probe = np.asarray([[0.05, 0.0]])
    nearest = nearest_contract.predict_with_scores(probe)
    union = union_contract.predict_with_scores(probe)
    assert nearest["pred"].tolist() == [0]
    assert union["pred"].tolist() == [0]

    # Move the probe just outside the raw-nearest tiny sphere but inside the
    # wider, farther sphere.
    probe = np.asarray([[0.12, 0.0]])
    nearest = nearest_contract.predict_with_scores(probe)
    union = union_contract.predict_with_scores(probe)
    assert nearest["pred"].tolist() == [1]
    assert union["pred"].tolist() == [0]
    assert union["score"][0] <= 1.0
    assert union["nearest_cluster"].tolist() == [1]


def test_detector_defaults_to_historical_nearest_sphere_contract() -> None:
    detector = MultiSphereOOSDetector()
    assert detector.acceptance_mode == "nearest_sphere"
