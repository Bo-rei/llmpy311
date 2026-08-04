"""Numerically stable class-level weighted evidence and parent guard."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp

from .contracts import CalibrationThresholds, CenterSpec
from .covariance import distance, score


@dataclass
class EvidenceOutput:
    energy: np.ndarray
    parent_score: np.ndarray
    gap: np.ndarray
    top_intent: np.ndarray
    second_intent: np.ndarray
    oos_score: np.ndarray
    predicted_oos: np.ndarray


class EvidenceModel:
    """A frozen structure with class evidence, parent boundary and margin."""

    def __init__(self, centers: dict[str, list[CenterSpec]], parents: dict[str, CenterSpec], thresholds: CalibrationThresholds | None = None):
        self.centers = centers
        self.parents = parents
        self.thresholds = thresholds

    def _class_energy(self, values: np.ndarray, intent: str) -> np.ndarray:
        points = np.asarray(values, dtype=np.float64)
        members = self.centers[str(intent)]
        counts = np.asarray([max(c.sample_count, 1) for c in members], dtype=np.float64)
        stability = np.asarray([max(c.stability, 1e-6) for c in members], dtype=np.float64)
        weights = counts * stability
        weights = weights / max(float(weights.sum()), 1e-12)
        terms = []
        for weight, center in zip(weights, members):
            q = np.square(distance(points, center) / max(float(center.radius), 1e-12))
            terms.append(np.log(max(float(weight), 1e-12)) - 0.5 * q)
        return -logsumexp(np.vstack(terms), axis=0)

    def raw(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(values, dtype=np.float64)
        intents = sorted(self.centers)
        energies = np.vstack([self._class_energy(points, intent) for intent in intents]).T
        order = np.argsort(energies, axis=1, kind="mergesort")
        top_idx = order[:, 0]
        second_idx = order[:, 1] if len(intents) > 1 else order[:, 0]
        top = energies[np.arange(points.shape[0]), top_idx]
        second = energies[np.arange(points.shape[0]), second_idx]
        gaps = second - top
        parent = np.vstack([score(points, self.parents[intent]) for intent in intents]).T
        parent_score = parent[np.arange(points.shape[0]), top_idx]
        return top, parent_score, gaps, np.asarray([intents[i] for i in top_idx], dtype=object), np.asarray([intents[i] for i in second_idx], dtype=object)

    def apply(self, values: np.ndarray) -> EvidenceOutput:
        energy, parent_score, gaps, top, second = self.raw(values)
        if self.thresholds is None:
            raise RuntimeError("Evidence thresholds are not fitted")
        thresholds = self.thresholds
        oos_score = np.maximum.reduce(
            [
                energy / max(thresholds.tau, 1e-12),
                parent_score / max(thresholds.tau_parent, 1e-12),
                thresholds.delta / np.maximum(gaps, 1e-12),
            ]
        )
        predicted_oos = (oos_score > 1.0).astype(np.int64)
        return EvidenceOutput(energy, parent_score, gaps, top, second, oos_score, predicted_oos)
