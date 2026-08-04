"""Data contracts for the isolated RC-AMBL experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AdaptiveConfig:
    """Frozen pilot defaults; values are selected from Known calibration only."""

    protocol_version: str = "protocol_v2_textoir_v1"
    representation: str = "frozen_minilm"
    distance: str = "mahalanobis_diag"
    radius_method: str = "mean_std"
    radius_lambda: float = 1.0
    covariance_epsilon: float = 1e-6
    rho_candidates: tuple[float, ...] = (0.25, 0.50, 0.75)
    target_false_rejection: float = 0.05
    target_margin_quantile: float = 0.05
    min_samples_absolute: int = 10
    min_samples_ratio: float = 0.05
    max_centers_per_intent: int = 4
    max_rounds: int = 3
    bootstrap_repeats: int = 20
    stability_threshold: float = 0.80
    max_known_recall_drop: float = 0.01
    max_ambiguity_increase: float = 0.0
    max_proxy_false_accept_increase: float = 0.0
    complexity_penalty: float = 0.01
    merge_overlap_ratio: float = 0.25
    seed: int = 42


@dataclass
class CenterSpec:
    intent: str
    local_id: int
    sample_indices: np.ndarray
    center: np.ndarray
    radius: float
    inv_diag_cov: np.ndarray
    stability: float = 1.0
    parent_local_id: int | None = None
    birth_round: int = 0
    active: bool = True

    @property
    def sample_count(self) -> int:
        return int(self.sample_indices.size)


@dataclass
class SplitOperation:
    round_index: int
    intent: str
    parent_local_id: int
    candidate_child_sizes: tuple[int, int]
    compactness_gain: float
    complexity_adjusted_gain: float
    stability_mean: float
    stability_median: float
    stability_min: float
    rho: float
    known_recall_delta: float
    ambiguity_delta: float
    proxy_false_accept_delta: float | None
    split_accepted: bool
    reject_reason: str | None = None
    merge_actions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CalibrationThresholds:
    tau: float
    tau_parent: float
    delta: float
    threshold_source: str
    n_threshold: int
    upper_rank: int
    lower_rank: int


@dataclass
class FitResult:
    centers: dict[str, list[CenterSpec]]
    parents: dict[str, CenterSpec]
    operations: list[SplitOperation]
    thresholds: CalibrationThresholds
    selection_audit: dict[str, Any]
    config: AdaptiveConfig

