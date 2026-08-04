"""Deterministic local PCA partitioning and bootstrap stability."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score


def _normalise_sign(axis: np.ndarray) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    pivot = int(np.argmax(np.abs(values)))
    if values[pivot] < 0:
        values = -values
    return values


def pca_median_split(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Split one local cluster along a deterministic PCA axis.

    A sorted balanced fallback is used for degenerate covariance or a tied
    projection.  The returned labels are always 0/1 and are independent of
    sklearn's global KMeans initialisation.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("PCA split requires at least two rows")
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    degenerate = components.size == 0 or not np.isfinite(components).all() or float(singular[0] if singular.size else 0.0) <= 1e-12
    if degenerate:
        axis = np.zeros(values.shape[1], dtype=np.float64)
        axis[0] = 1.0
        projections = values[:, 0]
        fallback = True
    else:
        axis = _normalise_sign(components[0])
        projections = centered @ axis
        fallback = False
    order = np.argsort(projections, kind="mergesort")
    split = int(values.shape[0] // 2)
    labels = np.empty(values.shape[0], dtype=np.int64)
    labels[order[:split]] = 0
    labels[order[split:]] = 1
    return labels, axis, {
        "axis": axis.tolist(),
        "median_projection": float(np.median(projections)),
        "degenerate_fallback": bool(fallback),
        "split_index": split,
    }


def apply_split_rule(points: np.ndarray, axis: np.ndarray, median_projection: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    projection = centered @ _normalise_sign(np.asarray(axis, dtype=np.float64))
    labels = (projection >= float(median_projection)).astype(np.int64)
    if np.unique(labels).size < 2:
        order = np.argsort(projection, kind="mergesort")
        labels = np.zeros(values.shape[0], dtype=np.int64)
        labels[order[values.shape[0] // 2 :]] = 1
    return labels


def bootstrap_split_stability(
    points: np.ndarray,
    *,
    seed: int,
    repeats: int = 20,
) -> dict[str, float | int]:
    """Compare bootstrap-refit assignments against the full deterministic split."""

    values = np.asarray(points, dtype=np.float64)
    full, _, _ = pca_median_split(values)
    if values.shape[0] < 4:
        return {"repeats": int(repeats), "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    rng = np.random.default_rng(int(seed))
    scores: list[float] = []
    for repeat in range(int(repeats)):
        sample = rng.integers(0, values.shape[0], size=values.shape[0])
        boot_labels, axis, info = pca_median_split(values[sample])
        projected = apply_split_rule(values, axis, float(info["median_projection"]))
        # Side labels are arbitrary; align them before ARI for readable side
        # diagnostics even though ARI itself is permutation invariant.
        score = float(adjusted_rand_score(full, projected))
        scores.append(score)
    array = np.asarray(scores, dtype=np.float64)
    return {
        "repeats": int(repeats),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def sse(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return 0.0
    center = values.mean(axis=0, keepdims=True)
    return float(np.square(values - center).sum())

