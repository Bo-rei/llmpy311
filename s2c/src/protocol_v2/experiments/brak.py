"""Known-only Boundary-Risk-Aware Adaptive K (BRAK) selection.

BRAK is deliberately a selection layer around the frozen protocol_v2 Gate.  It
does not change the detector, add pseudo-OOS examples, or inspect test labels.
For each intent it evaluates K=1..K_max on proper-train/calibration data and
then chooses the smallest K that has a material known-only risk improvement
without exceeding the K=1 calibration-recall safety budget.

The module keeps candidate diagnostics separate from final test evaluation so
that a caller can prove that K selection was made before test inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class LocalCandidate:
    """A per-intent candidate partition and its known-only boundary statistics."""

    intent: str
    k: int
    labels: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    inv_diag_covariances: tuple[np.ndarray | None, ...]
    train_indices: np.ndarray
    proper_recall: float
    self_rejection: float
    cross_intent_leakage: float
    union_overlap: float
    instability: float
    complexity: float
    objective: float
    safe: bool
    improvement_over_k1: float


@dataclass(frozen=True)
class BRAKSelection:
    """Selected candidate and all candidate diagnostics for one intent."""

    intent: str
    selected_k: int
    selected: LocalCandidate
    candidates: tuple[LocalCandidate, ...]
    delta: float
    min_improvement: float


def _validate_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0]:
        raise ValueError("BRAK requires a non-empty 2-D point matrix")
    if not np.isfinite(values).all():
        raise ValueError("BRAK points contain NaN or infinite values")
    return values


def _distances(
    points: np.ndarray,
    centers: np.ndarray,
    inv_diag_covariances: tuple[np.ndarray | None, ...],
    distance: str,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    result = np.empty((values.shape[0], centers.shape[0]), dtype=np.float64)
    for index, center in enumerate(centers):
        diff = values - center
        if distance == "euclidean":
            result[:, index] = np.linalg.norm(diff, axis=1)
        elif distance == "mahalanobis_diag":
            inv = inv_diag_covariances[index]
            if inv is None:
                raise ValueError("Missing diagonal covariance for Mahalanobis candidate")
            result[:, index] = np.sqrt(np.sum((diff**2) * inv, axis=1))
        else:
            raise ValueError(f"Unsupported BRAK distance: {distance}")
    return result


def _fit_partition(
    points: np.ndarray,
    k: int,
    seed: int,
    distance: str,
    covariance_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray | None, ...]]:
    values = _validate_points(points)
    actual_k = min(max(1, int(k)), values.shape[0])
    if actual_k == 1:
        labels = np.zeros(values.shape[0], dtype=np.int64)
        centers = values.mean(axis=0, keepdims=True)
    else:
        model = KMeans(n_clusters=actual_k, n_init=10, random_state=int(seed))
        labels = model.fit_predict(values).astype(np.int64)
        centers = np.asarray(model.cluster_centers_, dtype=np.float64)

    inv_covariances: list[np.ndarray | None] = []
    radii: list[float] = []
    for cluster_id in range(actual_k):
        cluster_points = values[labels == cluster_id]
        if cluster_points.shape[0] == 0:
            raise RuntimeError(f"BRAK produced an empty cluster: k={k}, cluster={cluster_id}")
        diff = cluster_points - centers[cluster_id]
        if distance == "euclidean":
            inv = None
            cluster_distances = np.linalg.norm(diff, axis=1)
        elif distance == "mahalanobis_diag":
            variance = np.var(diff, axis=0) + float(covariance_eps)
            inv = 1.0 / variance
            cluster_distances = np.sqrt(np.sum((diff**2) * inv, axis=1))
        else:
            raise ValueError(f"Unsupported BRAK distance: {distance}")
        inv_covariances.append(inv)
        radii.append(float(np.mean(cluster_distances) + np.std(cluster_distances)))
    return labels, centers, np.maximum(np.asarray(radii, dtype=np.float64), 1e-12), tuple(inv_covariances)


def _bootstrap_instability(
    points: np.ndarray,
    base_labels: np.ndarray,
    k: int,
    seed: int,
    repeats: int,
) -> float:
    """Return one minus the mean ARI under proper-train bootstrap refits."""

    if k <= 1 or repeats <= 0 or points.shape[0] < k:
        return 0.0
    rng = np.random.default_rng(int(seed))
    aris: list[float] = []
    for _ in range(int(repeats)):
        sample_indices = rng.integers(0, points.shape[0], size=points.shape[0])
        sampled = points[sample_indices]
        # Duplicated bootstrap rows can make a requested K infeasible.  In that
        # case the candidate is maximally unstable rather than silently dropped.
        if np.unique(sampled, axis=0).shape[0] < k:
            aris.append(0.0)
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(0, 2**31 - 1)))
        model.fit(sampled)
        prediction = model.predict(points)
        aris.append(float(adjusted_rand_score(base_labels, prediction)))
    return float(1.0 - np.mean(aris)) if aris else 0.0


def _candidate_risk(
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray | None, ...]],
    proper_points: np.ndarray,
    calibration_target: np.ndarray,
    calibration_other: np.ndarray,
    base_radius: float,
    base_recall: float,
    k: int,
    *,
    seed: int,
    distance: str,
    bootstrap_repeats: int,
    alpha: float,
    beta: float,
    gamma: float,
    eta: float,
    delta: float,
    min_improvement: float,
    base_labels: np.ndarray,
) -> tuple[float, float, float, float, float, float, bool, float]:
    labels, centers, radii, inv_covariances = candidate
    target_distances = _distances(calibration_target, centers, inv_covariances, distance)
    other_distances = _distances(calibration_other, centers, inv_covariances, distance)
    target_accepted = np.any(target_distances <= radii[None, :], axis=1) if calibration_target.size else np.asarray([], dtype=bool)
    other_accepted = np.any(other_distances <= radii[None, :], axis=1) if calibration_other.size else np.asarray([], dtype=bool)
    target_recall = float(np.mean(target_accepted)) if target_accepted.size else 1.0
    self_rejection = 1.0 - target_recall
    cross_leakage = float(np.mean(other_accepted)) if other_accepted.size else 0.0

    if target_distances.size:
        overlap_rate = float(np.mean(np.sum(target_distances <= radii[None, :], axis=1) > 1))
    else:
        overlap_rate = 0.0
    # A bounded, known-only proxy for extra union volume.  It compares the
    # average candidate radius with the K=1 radius, so K=1 has zero expansion.
    average_radius_ratio = float(np.mean(radii) / max(float(base_radius), 1e-12))
    radius_expansion = float(max(0.0, average_radius_ratio - 1.0) / max(1.0, average_radius_ratio))
    union_overlap = float(0.5 * overlap_rate + 0.5 * radius_expansion)
    instability = _bootstrap_instability(proper_points, base_labels if k == 1 else labels, k, seed, bootstrap_repeats)
    complexity = float(max(0, k - 1))
    objective = float(self_rejection + alpha * cross_leakage + beta * union_overlap + gamma * instability + eta * complexity)
    safe = bool(target_recall >= float(base_recall) - float(delta))
    improvement = float(0.0)  # filled by evaluate_intent_candidates after K=1 is known
    return target_recall, self_rejection, cross_leakage, union_overlap, instability, objective, safe, improvement


def evaluate_intent_candidates(
    intent: str,
    proper_points: np.ndarray,
    calibration_target: np.ndarray,
    calibration_other: np.ndarray,
    *,
    max_k: int = 5,
    seed: int = 42,
    distance: str = "mahalanobis_diag",
    covariance_eps: float = 1e-6,
    bootstrap_repeats: int = 5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
    eta: float = 0.01,
    delta: float = 0.02,
    min_improvement: float = 0.01,
) -> BRAKSelection:
    """Evaluate K candidates for one intent using proper-train/calibration only."""

    points = _validate_points(proper_points)
    target = _validate_points(calibration_target) if np.asarray(calibration_target).size else np.empty((0, points.shape[1]))
    other = _validate_points(calibration_other) if np.asarray(calibration_other).size else np.empty((0, points.shape[1]))
    if max_k < 1:
        raise ValueError("max_k must be positive")
    candidate_data: list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray | None, ...]]] = []
    for k in range(1, min(int(max_k), points.shape[0]) + 1):
        candidate_data.append(_fit_partition(points, k, int(seed), distance, covariance_eps))
    base_labels, base_centers, base_radii, base_inv = candidate_data[0]
    base_distances = _distances(target, base_centers, base_inv, distance)
    base_recall = float(np.mean(np.any(base_distances <= base_radii[None, :], axis=1))) if target.size else 1.0
    base_radius = float(base_radii[0])
    rows: list[LocalCandidate] = []
    base_objective: float | None = None
    for k, data in enumerate(candidate_data, start=1):
        values = _candidate_risk(
            data,
            points,
            target,
            other,
            base_radius,
            base_recall,
            k,
            seed=int(seed) + 100003 * k,
            distance=distance,
            bootstrap_repeats=bootstrap_repeats,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            eta=eta,
            delta=delta,
            min_improvement=min_improvement,
            base_labels=base_labels,
        )
        target_recall, self_rejection, cross_leakage, union_overlap, instability, objective, safe, _ = values
        if base_objective is None:
            base_objective = objective
        improvement = float(base_objective - objective)
        labels, centers, radii, inv_cov = data
        rows.append(
            LocalCandidate(
                intent=str(intent),
                k=int(k),
                labels=labels,
                centers=centers,
                radii=radii,
                inv_diag_covariances=inv_cov,
                train_indices=np.arange(points.shape[0], dtype=np.int64),
                proper_recall=float(target_recall),
                self_rejection=float(self_rejection),
                cross_intent_leakage=float(cross_leakage),
                union_overlap=float(union_overlap),
                instability=float(instability),
                complexity=float(max(0, k - 1)),
                objective=float(objective),
                safe=bool(safe),
                improvement_over_k1=improvement,
            )
        )

    selected = rows[0]
    # Minimum sufficient K: do not choose a more complex candidate unless it is
    # safe and improves J over K=1 by the preregistered margin.
    for candidate in rows[1:]:
        if candidate.safe and candidate.improvement_over_k1 >= float(min_improvement):
            selected = candidate
            break
    return BRAKSelection(
        intent=str(intent),
        selected_k=int(selected.k),
        selected=selected,
        candidates=tuple(rows),
        delta=float(delta),
        min_improvement=float(min_improvement),
    )


def selection_rows(selection: BRAKSelection) -> list[dict[str, Any]]:
    """Serialize one selection without embeddings or test-derived values."""

    return [
        {
            "intent": candidate.intent,
            "candidate_k": candidate.k,
            "selected_k": selection.selected_k,
            "proper_recall": candidate.proper_recall,
            "self_rejection": candidate.self_rejection,
            "cross_intent_leakage": candidate.cross_intent_leakage,
            "union_overlap": candidate.union_overlap,
            "instability": candidate.instability,
            "complexity": candidate.complexity,
            "objective": candidate.objective,
            "safe": candidate.safe,
            "improvement_over_k1": candidate.improvement_over_k1,
            "delta": selection.delta,
            "min_improvement": selection.min_improvement,
            "selection_source": "proper_train_and_known_calibration_only",
        }
        for candidate in selection.candidates
    ]


def selected_partition(
    points: np.ndarray,
    intents: np.ndarray,
    selections: dict[str, BRAKSelection],
) -> tuple[np.ndarray, np.ndarray, dict[int, str], dict[str, tuple[int, ...]]]:
    """Convert per-intent selected local labels to active detector labels."""

    values = _validate_points(points)
    labels_in = np.asarray(intents, dtype=object).reshape(-1).astype(str)
    if values.shape[0] != labels_in.size:
        raise ValueError("BRAK selected partition inputs are misaligned")
    global_labels = np.empty(values.shape[0], dtype=np.int64)
    centers: list[np.ndarray] = []
    cluster_to_intent: dict[int, str] = {}
    intent_to_clusters: dict[str, tuple[int, ...]] = {}
    next_id = 0
    for intent in sorted(set(labels_in.tolist())):
        if intent not in selections:
            raise KeyError(f"Missing BRAK selection for intent: {intent}")
        candidate = selections[intent].selected
        indices = np.flatnonzero(labels_in == intent)
        if indices.size != candidate.labels.size:
            raise ValueError(f"BRAK candidate size mismatch for intent={intent}")
        assigned = tuple(range(next_id, next_id + candidate.k))
        intent_to_clusters[intent] = assigned
        for local_id, global_id in enumerate(assigned):
            cluster_to_intent[global_id] = intent
            centers.append(candidate.centers[local_id])
        global_labels[indices] = np.asarray([assigned[int(value)] for value in candidate.labels], dtype=np.int64)
        next_id += candidate.k
    return global_labels, np.asarray(centers, dtype=np.float64), cluster_to_intent, intent_to_clusters
