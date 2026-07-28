from __future__ import annotations

import numpy as np
import pytest

from gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from s2c.experiments.partitions import (
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


def test_kmeans_seed_42_injected_detector_matches_legacy_detector() -> None:
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
    legacy = MultiSphereOOSDetector(
        radius_method="mean_std",
        radius_lambda=1.0,
        center_mode="class_centroid_mixture",
        distance_metric="euclidean",
        covariance_eps=1e-6,
        l2_normalize=True,
        subcenters_per_intent=2,
        random_state=42,
    )
    legacy.fit(embeddings, intents)
    assert np.array_equal(injected._train_cluster_labels, legacy._train_cluster_labels)
    assert np.allclose(
        [sphere.center for sphere in injected.spheres],
        [sphere.center for sphere in legacy.spheres],
        atol=1e-12,
    )
    assert np.allclose(
        [sphere.radius for sphere in injected.spheres],
        [sphere.radius for sphere in legacy.spheres],
        atol=1e-12,
    )
    probe = rng.normal(size=(7, 8))
    left = injected.predict_with_scores(probe)
    right = legacy.predict_with_scores(probe)
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
