"""Shrinkage diagonal covariance and parent/child radius fitting."""

from __future__ import annotations

import numpy as np

from .contracts import AdaptiveConfig, CenterSpec


def _variance(points: np.ndarray, center: np.ndarray, epsilon: float) -> np.ndarray:
    diff = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    return np.var(diff, axis=0) + float(epsilon)


def fit_center(
    points: np.ndarray,
    *,
    intent: str,
    local_id: int,
    sample_indices: np.ndarray,
    class_variance: np.ndarray,
    rho: float,
    config: AdaptiveConfig,
    stability: float = 1.0,
    parent_local_id: int | None = None,
    birth_round: int = 0,
) -> CenterSpec:
    values = np.asarray(points, dtype=np.float64)
    center = values.mean(axis=0)
    local_variance = _variance(values, center, config.covariance_epsilon)
    variance = float(rho) * local_variance + (1.0 - float(rho)) * np.asarray(class_variance, dtype=np.float64)
    variance = np.maximum(variance, float(config.covariance_epsilon))
    inv = 1.0 / variance
    distances = np.sqrt(np.sum(np.square(values - center) * inv, axis=1))
    radius = float(np.mean(distances) + config.radius_lambda * np.std(distances))
    radius = max(radius, float(np.sqrt(config.covariance_epsilon)))
    return CenterSpec(
        intent=str(intent),
        local_id=int(local_id),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        center=np.asarray(center, dtype=np.float64),
        radius=radius,
        inv_diag_cov=np.asarray(inv, dtype=np.float64),
        stability=float(stability),
        parent_local_id=parent_local_id,
        birth_round=int(birth_round),
    )


def fit_parent(points: np.ndarray, *, intent: str, config: AdaptiveConfig) -> CenterSpec:
    values = np.asarray(points, dtype=np.float64)
    center = values.mean(axis=0)
    class_variance = _variance(values, center, config.covariance_epsilon)
    return fit_center(
        values,
        intent=intent,
        local_id=0,
        sample_indices=np.arange(values.shape[0], dtype=np.int64),
        class_variance=class_variance,
        rho=1.0,
        config=config,
        stability=1.0,
    )


def distance(values: np.ndarray, center: CenterSpec) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    diff = x - center.center
    return np.sqrt(np.sum(np.square(diff) * center.inv_diag_cov, axis=1))


def score(values: np.ndarray, center: CenterSpec) -> np.ndarray:
    return distance(values, center) / max(float(center.radius), 1e-12)

