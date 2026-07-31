"""Small, auditable MOGB-compatible components for protocol_v2.

This module is deliberately separate from the frozen E2 detector.  It keeps the
official MOGB ideas that matter for a baseline (recursive purity-driven balls,
mean Euclidean radii, and nearest-ball open inference) while exposing a typed
adapter for the current protocol.  It is not presented as a reimplementation of
the authors' training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class GranularBall:
    """A leaf in the MOGB recursive partition tree."""

    ball_id: int
    parent_id: int | None
    depth: int
    sample_indices: np.ndarray
    centroid: np.ndarray
    radius: float
    majority_label: str
    purity: float
    sample_count: int
    label_histogram: dict[str, int]
    is_selected: bool = False
    split_reason: str | None = None
    stop_reason: str | None = None


@dataclass(frozen=True)
class MOGBBoundary:
    """A selected ball plus the statistics required by a boundary adapter."""

    ball_id: int
    center: np.ndarray
    radius: float
    label: str
    sample_indices: np.ndarray
    inv_diag_cov: np.ndarray | None = None


def _validate_table(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=object).reshape(-1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[0] != target.size:
        raise ValueError("MOGB partition requires a non-empty aligned feature/label table")
    if not np.isfinite(values).all():
        raise ValueError("MOGB features contain NaN or infinite values")
    return values, target.astype(str)


def _ball_statistics(indices: np.ndarray, values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float, str, dict[str, int], float]:
    points = values[indices]
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    histogram: dict[str, int] = {}
    for label in labels[indices].tolist():
        histogram[str(label)] = histogram.get(str(label), 0) + 1
    majority = max(sorted(histogram), key=lambda label: histogram[label])
    purity = histogram[majority] / float(indices.size)
    # The paper/code path used by MOGB computes the *mean* radius.  Keeping it
    # here makes that contract explicit; the max-radius helper is not used.
    return centroid, float(distances.mean()), majority, histogram, float(purity)


def _nearest_center_assignment(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return Euclidean nearest centers without a samples×centers×dims tensor."""

    values = np.asarray(points, dtype=np.float64)
    seeds = np.asarray(centers, dtype=np.float64)
    squared = (
        np.einsum("ij,ij->i", values, values)[:, None]
        + np.einsum("ij,ij->i", seeds, seeds)[None, :]
        - 2.0 * values @ seeds.T
    )
    # Roundoff can make a mathematically zero distance slightly negative.  The
    # clamp preserves Euclidean ordering while avoiding invalid square roots;
    # argmin on squared distances is identical to argmin on distances.
    np.maximum(squared, 0.0, out=squared)
    return np.argmin(squared, axis=1)


class AdaptiveGranularBallClusterer:
    """Deterministic-compatible recursive granular-ball partitioner.

    The official code stores balls in string-keyed dictionaries and chooses one
    random point per other label as a split seed.  This implementation preserves
    that split rule but stores integer-indexed tree nodes, which makes provenance
    and unit testing possible without changing the research protocol.
    """

    def __init__(
        self,
        *,
        purity_train: float = 0.90,
        purity_get_ball: float = 1.00,
        purity_select_ball: float = 0.90,
        min_ball_train: int = 10,
        min_ball_get_ball: int = 5,
        min_ball_select_ball: int = 10,
        seed: int = 0,
        max_leaf_balls: int = 10000,
    ) -> None:
        self.purity_train = float(purity_train)
        self.purity_get_ball = float(purity_get_ball)
        self.purity_select_ball = float(purity_select_ball)
        self.min_ball_train = int(min_ball_train)
        self.min_ball_get_ball = int(min_ball_get_ball)
        self.min_ball_select_ball = int(min_ball_select_ball)
        self.seed = int(seed)
        self.max_leaf_balls = int(max_leaf_balls)
        self.balls: list[GranularBall] = []
        self.selected_balls: list[GranularBall] = []
        self.sample_to_ball: np.ndarray | None = None
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def _make_ball(self, ball_id: int, parent_id: int | None, depth: int, indices: np.ndarray) -> GranularBall:
        assert self._features is not None and self._labels is not None
        center, radius, majority, histogram, purity = _ball_statistics(indices, self._features, self._labels)
        return GranularBall(
            ball_id=ball_id,
            parent_id=parent_id,
            depth=depth,
            sample_indices=np.asarray(indices, dtype=np.int64),
            centroid=center,
            radius=radius,
            majority_label=majority,
            purity=purity,
            sample_count=int(indices.size),
            label_histogram=histogram,
        )

    def _split_indices(self, indices: np.ndarray, rng: np.random.RandomState) -> list[np.ndarray]:
        assert self._features is not None and self._labels is not None
        labels = self._labels[indices]
        unique = sorted(set(labels.tolist()))
        if len(unique) <= 1:
            return [indices]
        # Match cluster3.py: one seed from the current label and one seed from
        # each other label, then assign every point to the nearest seed.
        majority = max(unique, key=lambda value: int(np.sum(labels == value)))
        ordered = [majority] + [value for value in unique if value != majority]
        seeds = []
        for label in ordered:
            candidates = indices[labels == label]
            seeds.append(int(candidates[rng.randint(0, len(candidates))]))
        centers = self._features[np.asarray(seeds, dtype=np.int64)]
        assignment = _nearest_center_assignment(self._features[indices], centers)
        children = [indices[assignment == child] for child in range(len(seeds))]
        children = [child for child in children if child.size]
        if len(children) <= 1 or sum(child.size for child in children) != indices.size:
            return [indices]
        return children

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "AdaptiveGranularBallClusterer":
        values, target = _validate_table(features, labels)
        self._features, self._labels = values, target
        self.balls = []
        self.selected_balls = []
        self.sample_to_ball = np.full(values.shape[0], -1, dtype=np.int64)
        rng = np.random.RandomState(self.seed)
        pending: list[tuple[int | None, int, np.ndarray]] = [(None, 0, np.arange(values.shape[0], dtype=np.int64))]
        next_id = 0
        while pending:
            parent_id, depth, indices = pending.pop(0)
            ball = self._make_ball(next_id, parent_id, depth, indices)
            next_id += 1
            can_split = (
                ball.purity < self.purity_get_ball
                and ball.sample_count > self.min_ball_get_ball
                and len(ball.label_histogram) > 1
            )
            if can_split:
                children = self._split_indices(indices, rng)
            else:
                children = [indices]
            if len(children) > 1:
                ball.split_reason = "purity_below_threshold"
                self.balls.append(ball)
                for child in children:
                    pending.append((ball.ball_id, depth + 1, child))
                if len(pending) + len(self.balls) > self.max_leaf_balls * 4:
                    raise RuntimeError("MOGB partition exceeded safety node limit")
                continue
            if ball.purity >= self.purity_get_ball:
                ball.stop_reason = "purity_reached"
            elif ball.sample_count <= self.min_ball_get_ball:
                ball.stop_reason = "minimum_samples"
            else:
                ball.stop_reason = "unsplittable"
            self.balls.append(ball)
            if sum(item.stop_reason is not None for item in self.balls) > self.max_leaf_balls:
                raise RuntimeError("MOGB partition exceeded max_leaf_balls")

        leaves = [ball for ball in self.balls if ball.stop_reason is not None]
        for ball in leaves:
            if ball.sample_count >= self.min_ball_select_ball and ball.purity >= self.purity_select_ball:
                ball.is_selected = True
                self.selected_balls.append(ball)
                self.sample_to_ball[ball.sample_indices] = ball.ball_id
        if not self.selected_balls:
            raise RuntimeError("MOGB produced no selected granular balls")
        return self

    @property
    def total_balls(self) -> int:
        return len(self.balls)

    def ball_statistics(self) -> dict[str, Any]:
        selected = self.selected_balls
        counts = [ball.sample_count for ball in selected]
        radii = [ball.radius for ball in selected]
        purities = [ball.purity for ball in selected]
        per_intent: dict[str, int] = {}
        for ball in selected:
            per_intent[ball.majority_label] = per_intent.get(ball.majority_label, 0) + 1
        return {
            "total_balls": self.total_balls,
            "selected_balls": len(selected),
            "filtered_balls": len([ball for ball in self.balls if ball.stop_reason is not None]) - len(selected),
            "balls_per_intent": per_intent,
            "mean_balls_per_intent": float(np.mean(list(per_intent.values()))) if per_intent else 0.0,
            "median_balls_per_intent": float(np.median(list(per_intent.values()))) if per_intent else 0.0,
            "max_balls_per_intent": int(max(per_intent.values())) if per_intent else 0,
            "min_balls_per_intent": int(min(per_intent.values())) if per_intent else 0,
            "mean_purity": float(np.mean(purities)) if purities else 0.0,
            "mean_radius": float(np.mean(radii)) if radii else 0.0,
            "mean_samples_per_ball": float(np.mean(counts)) if counts else 0.0,
            "max_tree_depth": max((ball.depth for ball in self.balls), default=0),
        }


def make_mogb_boundaries(
    clusterer: AdaptiveGranularBallClusterer,
    *,
    boundary: str = "mean",
    distance: str = "euclidean",
) -> list[MOGBBoundary]:
    """Convert selected balls to MOGB mean-radius or ours-style boundaries."""
    if boundary not in {"mean", "mean_std"}:
        raise ValueError(f"Unknown MOGB boundary adapter: {boundary}")
    result: list[MOGBBoundary] = []
    assert clusterer._features is not None
    for ball in clusterer.selected_balls:
        points = clusterer._features[ball.sample_indices]
        variance = np.var(points - ball.centroid, axis=0) + 1e-6
        inv_diag_cov = 1.0 / variance
        diff = points - ball.centroid
        if distance == "euclidean":
            distances = np.linalg.norm(diff, axis=1)
        elif distance == "mahalanobis_diag":
            distances = np.sqrt(np.sum(diff * diff * inv_diag_cov, axis=1))
        else:
            raise ValueError(f"Unknown MOGB boundary distance: {distance}")
        radius = float(distances.mean())
        if boundary == "mean_std":
            radius += float(distances.std())
        result.append(
            MOGBBoundary(
                ball.ball_id,
                ball.centroid,
                max(radius, 1e-12),
                ball.majority_label,
                ball.sample_indices,
                inv_diag_cov,
            )
        )
    return result


def score_mogb_boundaries(features: np.ndarray, boundaries: Iterable[MOGBBoundary], *, distance: str = "euclidean") -> dict[str, np.ndarray]:
    """Score samples with MOGB's nearest-ball open-set rule."""
    values = np.asarray(features, dtype=np.float64)
    balls = list(boundaries)
    if not balls:
        raise ValueError("At least one selected MOGB ball is required")
    distances = []
    for ball in balls:
        diff = values - ball.center
        if distance == "mahalanobis_diag":
            if ball.inv_diag_cov is None:
                raise ValueError("Missing diagonal covariance for MOGB boundary")
            distances.append(np.sqrt(np.sum(diff * diff * ball.inv_diag_cov, axis=1)))
        elif distance == "euclidean":
            distances.append(np.linalg.norm(diff, axis=1))
        else:
            raise ValueError(f"Unknown MOGB distance: {distance}")
    matrix = np.stack(distances, axis=1)
    nearest = np.argmin(matrix, axis=1)
    nearest_distance = matrix[np.arange(values.shape[0]), nearest]
    radii = np.asarray([ball.radius for ball in balls], dtype=np.float64)
    nearest_radius = radii[nearest]
    score = nearest_distance / np.maximum(nearest_radius, 1e-12)
    predicted_oos = (nearest_distance >= nearest_radius).astype(np.int64)
    predicted_label = np.asarray([balls[index].label for index in nearest], dtype=object)
    predicted_label[predicted_oos == 1] = "oos"
    return {
        "score": score,
        "predicted_oos": predicted_oos,
        "nearest_ball": np.asarray([balls[index].ball_id for index in nearest], dtype=np.int64),
        "distance": nearest_distance,
        "radius": nearest_radius,
        "predicted_label": predicted_label,
        "accepted_ball_count": np.sum(matrix < radii[None, :], axis=1).astype(np.int64),
    }


def balls_to_rows(clusterer: AdaptiveGranularBallClusterer) -> list[dict[str, Any]]:
    return [
        {
            "ball_id": ball.ball_id,
            "parent_id": ball.parent_id,
            "depth": ball.depth,
            "majority_label": ball.majority_label,
            "purity": ball.purity,
            "sample_count": ball.sample_count,
            "radius": ball.radius,
            "selected": ball.is_selected,
            "stop_reason": ball.stop_reason,
            "split_reason": ball.split_reason,
        }
        for ball in clusterer.balls
    ]
