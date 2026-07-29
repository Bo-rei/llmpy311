"""Known-only radius estimators for protocol_v2 multi-sphere experiments.

The active detector owns center fitting and distance geometry.  This module
only replaces the acceptance radius after fitting, so each boundary method is
compared on identical centers, clusters, embeddings and registries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


QUANTILES = {
    "quantile_90": 0.90,
    "quantile_95": 0.95,
    "quantile_975": 0.975,
}
SUPPORTED_BOUNDARIES = frozenset({"mean_std", *QUANTILES, "median_mad", "known_conformal"})


@dataclass(frozen=True)
class BoundarySelection:
    """A fixed known-only operating point; scores above threshold mean OOS."""

    method: str
    threshold: float
    details: dict[str, Any]


def _cluster_distances(detector: Any, sphere: Any) -> np.ndarray:
    labels = np.asarray(detector._train_cluster_labels, dtype=np.int64)
    points = np.asarray(detector._train_embeddings, dtype=np.float32)[labels == int(sphere.cluster_id)]
    return np.asarray([detector._distance(point, sphere) for point in points], dtype=np.float64)


def apply_radius_estimator(detector: Any, method: str) -> BoundarySelection:
    """Apply a deterministic radius rule using only known training embeddings."""
    if method not in SUPPORTED_BOUNDARIES:
        raise ValueError(f"Unsupported protocol_v2 boundary method: {method}")
    if method in {"mean_std", "known_conformal"}:
        return BoundarySelection(method=method, threshold=1.0, details={"radius_source": "legacy_mean_std"})

    radii: list[float] = []
    for sphere in detector.spheres:
        distances = _cluster_distances(detector, sphere)
        if method in QUANTILES:
            radius = float(np.quantile(distances, QUANTILES[method]))
        else:
            median = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median)))
            # 1.4826 makes MAD comparable to a Gaussian standard deviation;
            # 1.645 targets a known-only 95% acceptance rule without OOS tuning.
            radius = median + 1.645 * 1.4826 * mad
        sphere.radius = max(radius, 1e-12)
        radii.append(float(sphere.radius))
    return BoundarySelection(method=method, threshold=1.0, details={"radii": radii})


def known_conformal_threshold(scores: np.ndarray, alpha: float = 0.05) -> BoundarySelection:
    """Choose a finite-sample upper score quantile from known calibration only."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Known conformal calibration requires finite known scores")
    rank = min(int(np.ceil((values.size + 1) * (1.0 - alpha))), values.size)
    threshold = float(np.partition(values, rank - 1)[rank - 1])
    return BoundarySelection(
        method="known_conformal",
        threshold=threshold,
        details={"alpha": alpha, "known_calibration_count": int(values.size), "order_statistic_rank": rank},
    )
