"""Regression tests for the StackOverflow multi-sphere contract boundary."""

from __future__ import annotations

import numpy as np

from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector, SphereConfig


def test_nearest_sphere_and_normalized_union_are_explicitly_distinct() -> None:
    spheres = [
        SphereConfig(center=np.asarray([0.0, 0.0]), radius=0.1, cluster_id=0, intent_name="known"),
        SphereConfig(center=np.asarray([0.3, 0.0]), radius=0.5, cluster_id=1, intent_name="known"),
    ]
    query = np.asarray([[0.12, 0.0]])
    nearest = MultiSphereOOSDetector(acceptance_mode="nearest_sphere")
    union = MultiSphereOOSDetector(acceptance_mode="normalized_union")
    for detector in (nearest, union):
        detector.spheres = spheres
        detector.fitted = True

    nearest_output = nearest.predict_with_scores(query)
    union_output = union.predict_with_scores(query)

    assert nearest_output["pred"].tolist() == [1]
    assert union_output["pred"].tolist() == [0]
    assert union_output["score"][0] <= 1.0


def test_stackoverflow_debug_keeps_row_order_in_partitioned_detector() -> None:
    embeddings = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=np.float64
    )
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=2,
        random_state=42,
        l2_normalize=False,
    )
    detector.fit(embeddings, np.asarray(["a", "a", "b", "b"], dtype=object))
    output = detector.predict_with_scores(embeddings)
    assert output["score"].shape == (4,)
    assert output["pred"].shape == (4,)
    assert detector._train_cluster_labels.shape == (4,)
