"""Dual-representation fusion gate for v19 hierarchical open-intent pipeline.

Fuses MiniLM fast gate scores with SmolLM semantic scores via weighted
linear combination for improved known-intent accuracy without OOS regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DualRepresentationGateConfig:
    """Configuration for DualRepresentationGate."""

    alpha: float = 0.5
    beta: float = 0.5
    gate_threshold: float = 0.5


class DualRepresentationGate:
    """Fuse MiniLM fast gate and SmolLM semantic gate scores.

    Computes a weighted combination:
        fused_score = (alpha * minilm_score + beta * smollm_score) / (alpha + beta)

    Constraint: alpha + beta must equal 1.0 (enforced at construction and per-call).
    Use this gate when `dual_representation_enabled=true` in v19_gate.yaml.

    Args:
        config: Fusion gate configuration. alpha + beta must equal 1.0.

    Raises:
        ValueError: If alpha + beta != 1.0 (within tolerance 1e-6).
    """

    def __init__(self, config: Optional[DualRepresentationGateConfig] = None) -> None:
        if config is None:
            config = DualRepresentationGateConfig()
        self._validate_weights(config.alpha, config.beta)
        self.config = config

    @staticmethod
    def _validate_weights(alpha: float, beta: float) -> None:
        """Enforce alpha + beta == 1.0.

        Args:
            alpha: MiniLM weight.
            beta: SmolLM weight.

        Raises:
            ValueError: If weights do not sum to 1.0.
        """
        total = float(alpha) + float(beta)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"alpha + beta must equal 1.0, got alpha={alpha} + beta={beta} = {total}"
            )

    def fused_score(
        self,
        minilm_score: float,
        smollm_score: float,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> float:
        """Compute the fused gate score from two representations.

        Args:
            minilm_score: Fast MiniLM gate score (e.g., normalized distance).
            smollm_score: SmolLM semantic similarity score in [0, 1].
            alpha: Optional per-call override for MiniLM weight.
            beta: Optional per-call override for SmolLM weight.
                If either is provided, both must be provided and sum to 1.0.

        Returns:
            Fused score as float.

        Raises:
            ValueError: If only one of alpha/beta is provided, or if they
                do not sum to 1.0.
        """
        if alpha is None and beta is None:
            alpha = self.config.alpha
            beta = self.config.beta
        elif alpha is not None and beta is not None:
            self._validate_weights(alpha, beta)
        else:
            raise ValueError("Either both or neither of alpha and beta must be provided.")

        return float(alpha) * float(minilm_score) + float(beta) * float(smollm_score)

    def fused_score_batch(
        self,
        minilm_scores: List[float],
        smollm_scores: List[float],
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> List[float]:
        """Vectorized fused scores for a batch.

        Args:
            minilm_scores: List of fast gate scores.
            smollm_scores: List of semantic gate scores (same length).
            alpha: Optional per-call MiniLM weight override.
            beta: Optional per-call SmolLM weight override.

        Returns:
            List of fused scores.

        Raises:
            ValueError: If minilm_scores and smollm_scores have different lengths.
        """
        if len(minilm_scores) != len(smollm_scores):
            raise ValueError(
                f"minilm_scores ({len(minilm_scores)}) and smollm_scores "
                f"({len(smollm_scores)}) must have equal length."
            )

        if alpha is None and beta is None:
            alpha = self.config.alpha
            beta = self.config.beta
        elif alpha is not None and beta is not None:
            self._validate_weights(alpha, beta)
        else:
            raise ValueError("Either both or neither of alpha and beta must be provided.")

        return [
            float(alpha) * float(fast) + float(beta) * float(sem)
            for fast, sem in zip(minilm_scores, smollm_scores)
        ]

    def predict(
        self,
        minilm_score: float,
        smollm_score: float,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute fused score and binary ID/OOS prediction.

        Args:
            minilm_score: Fast gate score.
            smollm_score: Semantic gate score.
            alpha: Optional MiniLM weight override.
            beta: Optional SmolLM weight override.
            threshold: Decision threshold. Defaults to config.gate_threshold.

        Returns:
            Dictionary with fused_score, is_id, and threshold used.
        """
        tau = float(threshold) if threshold is not None else float(self.config.gate_threshold)
        score = self.fused_score(minilm_score, smollm_score, alpha=alpha, beta=beta)
        return {
            "fused_score": score,
            "is_id": bool(score >= tau),
            "threshold": tau,
            "alpha": float(alpha) if alpha is not None else float(self.config.alpha),
            "beta": float(beta) if beta is not None else float(self.config.beta),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate configuration."""
        return {
            "alpha": float(self.config.alpha),
            "beta": float(self.config.beta),
            "gate_threshold": float(self.config.gate_threshold),
        }
