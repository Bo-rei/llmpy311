"""Injectable within-intent partitions for the E3 mechanism study.

The E2 detector remains the source of truth for distance, radius and score
semantics.  This module only decides which training point belongs to which
local center, then injects those labels into a detector instance.  Keeping the
partition boundary here makes the KMeans/random control explicit and prevents
E3 from silently changing the historical detector implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class PartitionResult:
    """A deterministic global partition of one known-intent training table."""

    labels: np.ndarray
    centers: np.ndarray
    cluster_to_intent: dict[int, str]
    intent_to_clusters: dict[str, tuple[int, ...]]

    @property
    def cluster_count(self) -> int:
        return int(self.centers.shape[0])


def normalize_for_detector(embeddings: np.ndarray) -> np.ndarray:
    """Match the legacy detector's explicit L2 normalization step."""

    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


class _StaticCenters:
    """Minimal sklearn-compatible center holder used by the legacy detector."""

    def __init__(self, centers: np.ndarray, labels: np.ndarray) -> None:
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_clusters = int(centers.shape[0])


def _balanced_local_labels(count: int, clusters: int, seed: int) -> np.ndarray:
    """Return labels whose group sizes differ by at most one."""

    actual_clusters = min(max(1, int(clusters)), count)
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    labels = np.empty(count, dtype=np.int64)
    labels[order] = np.arange(count, dtype=np.int64) % actual_clusters
    return labels


def build_partition(
    embeddings: np.ndarray,
    intents: np.ndarray,
    clusters_per_intent: int,
    partition: str,
    partition_seed: int,
) -> PartitionResult:
    """Build a KMeans or size-matched random partition within each intent.

    ``kmeans`` deliberately matches the legacy detector: sorted intent order,
    sklearn KMeans, ``n_init=10`` and the supplied random seed.  The random
    control never reads text or test labels; it only permutes rows belonging to
    the same known intent.
    """

    points = np.asarray(embeddings, dtype=np.float64)
    labels_in = np.asarray(intents, dtype=object).reshape(-1)
    if points.ndim != 2 or points.shape[0] != labels_in.shape[0] or not points.shape[0]:
        raise ValueError("Partition requires a non-empty 2-D embedding table aligned with intents")
    if clusters_per_intent < 1:
        raise ValueError(f"clusters_per_intent must be positive: {clusters_per_intent}")
    if partition not in {"kmeans", "random_balanced"}:
        raise ValueError(f"Unknown E3 partition: {partition}")

    global_labels = np.empty(points.shape[0], dtype=np.int64)
    centers: list[np.ndarray] = []
    cluster_to_intent: dict[int, str] = {}
    intent_to_clusters: dict[str, tuple[int, ...]] = {}
    next_cluster = 0
    for intent in sorted(str(value) for value in np.unique(labels_in)):
        indices = np.flatnonzero(labels_in.astype(str) == intent)
        local_points = points[indices]
        local_k = min(int(clusters_per_intent), int(local_points.shape[0]))
        if local_k == 1:
            local_labels = np.zeros(local_points.shape[0], dtype=np.int64)
            local_centers = local_points.mean(axis=0, keepdims=True)
        elif partition == "kmeans":
            model = KMeans(n_clusters=local_k, random_state=partition_seed, n_init=10)
            local_labels = model.fit_predict(local_points).astype(np.int64)
            local_centers = np.asarray(model.cluster_centers_, dtype=np.float64)
        else:
            # Offset the seed by the intent position so two intents never share
            # a hidden RNG stream merely because their names sort similarly.
            local_seed = int(partition_seed) + next_cluster * 1009
            local_labels = _balanced_local_labels(local_points.shape[0], local_k, local_seed)
            local_centers = np.asarray(
                [local_points[local_labels == index].mean(axis=0) for index in range(local_k)],
                dtype=np.float64,
            )
        assigned = tuple(range(next_cluster, next_cluster + local_k))
        intent_to_clusters[intent] = assigned
        for local_index, global_index in enumerate(assigned):
            cluster_to_intent[global_index] = intent
            centers.append(local_centers[local_index])
        global_labels[indices] = np.asarray([assigned[int(value)] for value in local_labels], dtype=np.int64)
        next_cluster += local_k

    return PartitionResult(
        labels=global_labels,
        centers=np.asarray(centers, dtype=np.float64),
        cluster_to_intent=cluster_to_intent,
        intent_to_clusters=intent_to_clusters,
    )


def fit_injected_detector(
    embeddings: np.ndarray,
    intents: np.ndarray,
    partition_result: PartitionResult,
    *,
    distance: str,
    radius_lambda: float = 1.0,
    random_state: int = 42,
) -> Any:
    """Fit the established detector with externally supplied cluster labels."""

    from gate.multi_sphere_oos_detector import MultiSphereOOSDetector

    detector = MultiSphereOOSDetector(
        radius_method="mean_std",
        radius_lambda=radius_lambda,
        center_mode="class_centroid_mixture",
        distance_metric=distance,
        covariance_eps=1e-6,
        l2_normalize=True,
        subcenters_per_intent=1,
        random_state=random_state,
    )
    normalized = detector._normalize_embeddings(np.asarray(embeddings, dtype=np.float64))
    labels = np.asarray(intents, dtype=object).reshape(-1)
    if normalized.shape[0] != partition_result.labels.shape[0] or labels.shape[0] != normalized.shape[0]:
        raise ValueError("Injected partition and embedding rows are not aligned")
    detector._train_embeddings = normalized
    detector._train_cluster_labels = np.asarray(partition_result.labels, dtype=np.int64)
    detector.n_clusters = partition_result.cluster_count
    detector.cluster_to_intent = dict(partition_result.cluster_to_intent)
    detector.intent_to_clusters = {
        intent: list(cluster_ids) for intent, cluster_ids in partition_result.intent_to_clusters.items()
    }
    detector.intent_to_cluster = {
        intent: int(cluster_ids[0]) for intent, cluster_ids in partition_result.intent_to_clusters.items()
    }
    detector.kmeans = _StaticCenters(partition_result.centers, detector._train_cluster_labels)
    detector._compute_radii(detector._train_embeddings, detector._train_cluster_labels)
    detector.fitted = True
    return detector


def partition_sizes(result: PartitionResult) -> np.ndarray:
    """Return global cluster sizes in stable cluster-id order."""

    return np.bincount(result.labels, minlength=result.cluster_count).astype(np.int64)
