"""Known-only split--merge adaptive-K prototype.

This module is deliberately a research skeleton, not a replacement for the
frozen E2/E3 detector.  It starts each intent with one cluster and accepts a
binary split only when every pre-registered safety gate passes.  The
cross-intent acceptance signal must be supplied by a caller that owns the
full Known-only calibration geometry; when it is omitted the candidate is
rejected rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class AdaptiveSplitMergeConfig:
    tau_compact: float = 0.10
    n_min: int = 5
    tau_stability: float = 0.80
    epsilon: float = 0.02
    complexity_penalty: float = 0.01
    max_k: int = 5
    bootstrap_repeats: int = 5


@dataclass(frozen=True)
class SplitMetrics:
    compactness_gain: float
    min_child_size: int
    stability: float
    cross_intent_acceptance_increase: float | None
    complexity_adjusted_gain: float


@dataclass(frozen=True)
class SplitDecision:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: SplitMetrics


@dataclass(frozen=True)
class AdaptiveSplitMergeResult:
    labels: np.ndarray
    centers: np.ndarray
    decisions: tuple[SplitDecision, ...]


def _sse(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    center = np.mean(points, axis=0)
    return float(np.square(points - center).sum())


def compactness_gain(parent: np.ndarray, children: Iterable[np.ndarray]) -> float:
    """Return relative SSE reduction, safely handling a zero-SSE parent."""

    parent_sse = _sse(np.asarray(parent, dtype=float))
    child_sse = sum(_sse(np.asarray(child, dtype=float)) for child in children)
    if parent_sse <= 1e-12:
        return 0.0
    return float((parent_sse - child_sse) / parent_sse)


def merge_small_clusters(features: np.ndarray, labels: np.ndarray, n_min: int) -> np.ndarray:
    """Merge undersized clusters into their nearest sufficiently large center."""

    values = np.asarray(features, dtype=float)
    current = np.asarray(labels, dtype=np.int64).copy()
    if values.ndim != 2 or len(values) != len(current):
        raise ValueError("features and labels must be aligned 2-D arrays")
    if n_min < 1:
        raise ValueError("n_min must be positive")
    while True:
        ids, counts = np.unique(current, return_counts=True)
        small = [int(cluster) for cluster, count in zip(ids, counts) if int(count) < n_min]
        large = [int(cluster) for cluster, count in zip(ids, counts) if int(count) >= n_min]
        if not small or not large or len(ids) == 1:
            return current
        centers = {int(cluster): values[current == cluster].mean(axis=0) for cluster in ids}
        for cluster in sorted(small):
            target = min(large, key=lambda candidate: (float(np.linalg.norm(centers[cluster] - centers[candidate])), candidate))
            current[current == cluster] = target


def bootstrap_stability(features: np.ndarray, seed: int, repeats: int = 5) -> float:
    """Estimate split stability by ARI on repeated bootstrap-fitted KMeans."""

    values = np.asarray(features, dtype=float)
    if len(values) < 4:
        return 0.0
    rng = np.random.default_rng(seed)
    assignments: list[np.ndarray] = []
    for repeat in range(max(1, int(repeats))):
        sample = rng.integers(0, len(values), size=len(values))
        model = KMeans(n_clusters=2, n_init=10, random_state=seed + repeat)
        model.fit(values[sample])
        assignments.append(model.predict(values))
    scores = [adjusted_rand_score(assignments[0], assignment) for assignment in assignments[1:]]
    return float(np.mean(scores)) if scores else 1.0


def evaluate_split(metrics: SplitMetrics, config: AdaptiveSplitMergeConfig) -> SplitDecision:
    """Apply all split gates; missing cross-intent evidence rejects safely."""

    reasons: list[str] = []
    if metrics.compactness_gain <= config.tau_compact:
        reasons.append("compactness_gain_below_threshold")
    if metrics.min_child_size < config.n_min:
        reasons.append("child_below_min_size")
    if metrics.stability < config.tau_stability:
        reasons.append("stability_below_threshold")
    if metrics.cross_intent_acceptance_increase is None:
        reasons.append("cross_intent_acceptance_not_measured")
    elif metrics.cross_intent_acceptance_increase > config.epsilon:
        reasons.append("cross_intent_acceptance_increase_above_epsilon")
    if metrics.complexity_adjusted_gain <= 0.0:
        reasons.append("complexity_adjusted_gain_non_positive")
    return SplitDecision(accepted=not reasons, reasons=tuple(reasons), metrics=metrics)


def fit_split_merge(
    features: np.ndarray,
    *,
    seed: int,
    config: AdaptiveSplitMergeConfig | None = None,
    cross_intent_acceptance_increase: float | None = None,
) -> AdaptiveSplitMergeResult:
    """Run a bounded, Known-only split--merge prototype on one intent."""

    cfg = config or AdaptiveSplitMergeConfig()
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("features must be a non-empty 2-D array")
    if cfg.max_k < 1:
        raise ValueError("max_k must be positive")
    labels = np.zeros(values.shape[0], dtype=np.int64)
    decisions: list[SplitDecision] = []
    next_label = 1
    while len(np.unique(labels)) < cfg.max_k:
        ids, counts = np.unique(labels, return_counts=True)
        parent = int(ids[int(np.argmax(counts))])
        parent_indices = np.flatnonzero(labels == parent)
        if len(parent_indices) < 2 * cfg.n_min:
            break
        model = KMeans(n_clusters=2, n_init=10, random_state=seed + len(decisions))
        proposed = model.fit_predict(values[parent_indices]).astype(np.int64)
        proposed = merge_small_clusters(values[parent_indices], proposed, cfg.n_min)
        child_ids = sorted(int(value) for value in np.unique(proposed))
        if len(child_ids) < 2:
            break
        children = [values[parent_indices[proposed == child]] for child in child_ids]
        gain = compactness_gain(values[parent_indices], children)
        stability = bootstrap_stability(values[parent_indices], seed + len(decisions), cfg.bootstrap_repeats)
        adjusted_gain = gain - cfg.complexity_penalty * len(child_ids)
        split_metrics = SplitMetrics(
            compactness_gain=gain,
            min_child_size=min(len(child) for child in children),
            stability=stability,
            cross_intent_acceptance_increase=cross_intent_acceptance_increase,
            complexity_adjusted_gain=adjusted_gain,
        )
        decision = evaluate_split(split_metrics, cfg)
        decisions.append(decision)
        if not decision.accepted:
            break
        # Keep global cluster ids stable and deterministic.
        for child_index, child in enumerate(child_ids):
            target = parent if child_index == 0 else next_label
            if child_index > 0:
                next_label += 1
            labels[parent_indices[proposed == child]] = target
    ids = sorted(int(value) for value in np.unique(labels))
    centers = np.asarray([values[labels == cluster].mean(axis=0) for cluster in ids], dtype=float)
    remap = {cluster: index for index, cluster in enumerate(ids)}
    labels = np.asarray([remap[int(value)] for value in labels], dtype=np.int64)
    return AdaptiveSplitMergeResult(labels=labels, centers=centers, decisions=tuple(decisions))


__all__ = [
    "AdaptiveSplitMergeConfig",
    "AdaptiveSplitMergeResult",
    "SplitDecision",
    "SplitMetrics",
    "bootstrap_stability",
    "compactness_gain",
    "evaluate_split",
    "fit_split_merge",
    "merge_small_clusters",
]
