"""Known-only local-support scores and split-conformal calibration for CLMSG.

Milestones 1--3 intentionally stop at the non-parametric support model.  This
module does not contain KMeans, granular balls, local PCA, or label entropy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


DistanceMetric = Literal["cosine", "normalized_euclidean"]


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a finite row-normalized float64 table."""

    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise ValueError("Embeddings must be a non-empty two-dimensional table")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Cannot normalize a zero-norm embedding")
    return np.ascontiguousarray(vectors / norms, dtype=np.float64)


def normalized_distances(
    query: np.ndarray,
    support: np.ndarray,
    metric: DistanceMetric,
) -> np.ndarray:
    """Pairwise distance for already L2-normalized vectors."""

    similarity = np.clip(np.asarray(query) @ np.asarray(support).T, -1.0, 1.0)
    cosine = np.maximum(1.0 - similarity, 0.0)
    if metric == "cosine":
        return cosine
    if metric == "normalized_euclidean":
        return np.sqrt(np.maximum(2.0 * cosine, 0.0))
    raise ValueError(f"Unsupported CLMSG distance: {metric}")


def _same_intent_local_scales(
    support: np.ndarray,
    labels: np.ndarray,
    *,
    k_scale: int,
    metric: DistanceMetric,
    eps: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute the kth same-intent neighbour distance, excluding self."""

    if k_scale < 1:
        raise ValueError("k_scale must be positive")
    scales = np.full(support.shape[0], eps, dtype=np.float64)
    effective_k: dict[str, int] = {}
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        if indices.size <= 1:
            effective_k[str(label)] = 0
            continue
        selected = support[indices]
        distance = normalized_distances(selected, selected, metric)
        np.fill_diagonal(distance, np.inf)
        kth = min(k_scale, indices.size - 1)
        scales[indices] = np.partition(distance, kth - 1, axis=1)[:, kth - 1]
        effective_k[str(label)] = int(kth)
    return np.maximum(scales, eps), effective_k


@dataclass(frozen=True)
class SupportScores:
    """The two Milestone-2 scores and their predicted support labels."""

    knn_score: np.ndarray
    knn_support_index: np.ndarray
    local_scale_score: np.ndarray
    local_scale_support_index: np.ndarray
    class_local_scale_score: np.ndarray
    class_local_scale_support_index: np.ndarray

    def score_for_mode(self, mode: str, gamma: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Return a prescribed global, class-conditional, or hybrid support score."""

        if mode == "global_knn":
            return self.local_scale_score, self.local_scale_support_index
        if mode == "class_conditional_knn":
            return self.class_local_scale_score, self.class_local_scale_support_index
        if mode == "hybrid_knn":
            if not 0.0 <= gamma <= 1.0:
                raise ValueError("hybrid gamma must lie in [0, 1]")
            score = gamma * self.class_local_scale_score + (1.0 - gamma) * self.local_scale_score
            # The candidate intent is determined by the ordinary nearest
            # support point, never by the test label. The class-conditional
            # support index therefore owns the open-intent label.
            return score, self.class_local_scale_support_index
        raise ValueError(f"Unsupported CLMSG support mode: {mode}")


class LocalSupportModel:
    """Exact chunked KNN and support-point local-scale scorer."""

    def __init__(
        self,
        *,
        metric: DistanceMetric = "cosine",
        k_neighbors: int = 10,
        k_scale: int = 10,
        eps: float = 1e-8,
        chunk_size: int = 256,
    ) -> None:
        if k_neighbors < 1 or k_scale < 1 or chunk_size < 1 or eps <= 0:
            raise ValueError("CLMSG k values, chunk size, and epsilon must be positive")
        if metric not in {"cosine", "normalized_euclidean"}:
            raise ValueError(f"Unsupported CLMSG distance: {metric}")
        self.metric = metric
        self.k_neighbors = int(k_neighbors)
        self.k_scale = int(k_scale)
        self.eps = float(eps)
        self.chunk_size = int(chunk_size)
        self.support: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.local_scales: np.ndarray | None = None
        self.effective_scale_k: dict[str, int] = {}

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "LocalSupportModel":
        target = np.asarray(labels, dtype=object).reshape(-1)
        support = l2_normalize(embeddings)
        if target.size != support.shape[0] or not target.size:
            raise ValueError("Support labels must align with support embeddings")
        scales, effective = _same_intent_local_scales(
            support,
            target,
            k_scale=self.k_scale,
            metric=self.metric,
            eps=self.eps,
        )
        self.support = support
        self.labels = target
        self.local_scales = scales
        self.effective_scale_k = effective
        return self

    def score(self, embeddings: np.ndarray) -> SupportScores:
        if self.support is None or self.labels is None or self.local_scales is None:
            raise RuntimeError("LocalSupportModel must be fitted before scoring")
        query = l2_normalize(embeddings)
        count = query.shape[0]
        kth = min(self.k_neighbors, self.support.shape[0])
        knn_score = np.empty(count, dtype=np.float64)
        knn_index = np.empty(count, dtype=np.int64)
        local_score = np.empty(count, dtype=np.float64)
        local_index = np.empty(count, dtype=np.int64)
        class_score = np.empty(count, dtype=np.float64)
        class_index = np.empty(count, dtype=np.int64)
        for start in range(0, count, self.chunk_size):
            stop = min(start + self.chunk_size, count)
            distance = normalized_distances(query[start:stop], self.support, self.metric)
            knn_score[start:stop] = np.partition(distance, kth - 1, axis=1)[:, kth - 1]
            knn_index[start:stop] = np.argmin(distance, axis=1)
            normalized = distance / self.local_scales[None, :]
            chosen = np.argmin(normalized, axis=1)
            local_index[start:stop] = chosen
            local_score[start:stop] = normalized[np.arange(stop - start), chosen]
            candidate_labels = self.labels[knn_index[start:stop]]
            for label in sorted(set(candidate_labels.tolist())):
                query_mask = candidate_labels == label
                support_indices = np.flatnonzero(self.labels == label)
                restricted = normalized[query_mask][:, support_indices]
                restricted_choice = np.argmin(restricted, axis=1)
                absolute_choice = support_indices[restricted_choice]
                block_rows = np.flatnonzero(query_mask) + start
                class_index[block_rows] = absolute_choice
                class_score[block_rows] = restricted[np.arange(block_rows.size), restricted_choice]
        if not all(
            np.isfinite(values).all() for values in (knn_score, local_score, class_score)
        ):
            raise RuntimeError("CLMSG produced a non-finite support score")
        return SupportScores(knn_score, knn_index, local_score, local_index, class_score, class_index)

    def label_for_indices(self, indices: np.ndarray) -> np.ndarray:
        if self.labels is None:
            raise RuntimeError("LocalSupportModel must be fitted before label lookup")
        return np.asarray(self.labels[np.asarray(indices, dtype=np.int64)], dtype=object)

    def support_statistics(self) -> dict[str, object]:
        if self.labels is None or self.local_scales is None:
            raise RuntimeError("LocalSupportModel must be fitted before diagnostics")
        per_intent: dict[str, dict[str, float | int]] = {}
        for label in sorted(set(self.labels.tolist())):
            selected = self.local_scales[self.labels == label]
            per_intent[str(label)] = {
                "sample_count": int(selected.size),
                "effective_k_scale": int(self.effective_scale_k[str(label)]),
                "mean_local_scale": float(selected.mean()),
                "median_local_scale": float(np.median(selected)),
                "min_local_scale": float(selected.min()),
                "max_local_scale": float(selected.max()),
            }
        return {
            "support_count": int(self.labels.size),
            "intent_count": len(per_intent),
            "distance": self.metric,
            "k_neighbors": self.k_neighbors,
            "k_scale": self.k_scale,
            "epsilon": self.eps,
            "mean_local_scale": float(self.local_scales.mean()),
            "median_local_scale": float(np.median(self.local_scales)),
            "per_intent": per_intent,
        }


def known_order_statistic(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    """Known-only upper order statistic used by the KNN control."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Known calibration scores must be finite and non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = min(int(math.ceil((values.size + 1) * (1.0 - alpha))), values.size)
    return float(np.partition(values, rank - 1)[rank - 1]), rank


def split_conformal_p_values(calibration_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    """Return finite-sample p-values with the required greater-or-equal tie rule."""

    calibration = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    query = np.asarray(query_scores, dtype=np.float64).reshape(-1)
    if not calibration.size or not np.isfinite(calibration).all() or not np.isfinite(query).all():
        raise ValueError("Conformal scores must be finite and calibration must be non-empty")
    ordered = np.sort(calibration)
    greater_or_equal = ordered.size - np.searchsorted(ordered, query, side="left")
    result = (1.0 + greater_or_equal.astype(np.float64)) / (ordered.size + 1.0)
    if np.any(result <= 0.0) or np.any(result > 1.0):
        raise RuntimeError("Split-conformal p-value escaped (0, 1]")
    return result
