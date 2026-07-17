from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.experiments.cluster_separability.v20_random_partition import (
    _random_partition_detector,
)


def test_random_partition_preserves_reference_cluster_sizes():
    """随机对照只打乱归属，不改变每个 intent 的容量多重集合。"""

    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
        ],
        dtype=np.float32,
    )
    intents = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    reference_sizes = {"a": np.asarray([2, 1]), "b": np.asarray([2, 1])}

    detector, quality = _random_partition_detector(
        embeddings, intents, 2, "euclidean", 1, reference_sizes
    )

    assert len(detector.spheres) == 4
    assert sorted(row["cluster_sizes"] for row in quality) == ["2|1", "2|1"]
    for intent, cluster_ids in detector.intent_to_clusters.items():
        observed = [int(np.sum(detector._train_cluster_labels == cluster_id)) for cluster_id in cluster_ids]
        assert sorted(observed) == [1, 2]


def test_random_partition_repeat_changes_assignment_but_not_protocol_shape():
    """不同 repeat 改变随机归属，同时保持中心/半径数量和输出结构不变。"""

    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(20, 4)).astype(np.float32)
    intents = np.asarray(["a"] * 10 + ["b"] * 10, dtype=object)
    sizes = {"a": np.asarray([5, 5]), "b": np.asarray([5, 5])}

    first, _ = _random_partition_detector(embeddings, intents, 2, "mahalanobis_diag", 1, sizes)
    second, _ = _random_partition_detector(embeddings, intents, 2, "mahalanobis_diag", 2, sizes)

    assert len(first.spheres) == len(second.spheres) == 4
    assert first._train_cluster_labels.shape == second._train_cluster_labels.shape
    assert not np.array_equal(first._train_cluster_labels, second._train_cluster_labels)
