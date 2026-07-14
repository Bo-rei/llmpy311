"""Shared configuration for the historical-best v19 prototype gate pipeline.

This module centralizes the exact pipeline settings that reproduced the
archived Prototype Gate baseline so the training and benchmark orchestration
paths can reuse the same values without drifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class HistoricalBestPipelineProfile:
    """Exact pipeline profile used as the canonical historical-best template."""

    name: str = "historical_best"
    gate_profile: str = "historical_best"
    gate_mode: str = "multisphere"
    gate_center_mode: str = "class_centroid_mixture"
    gate_distance_metric: str = "mahalanobis_diag"
    gate_l2_normalize: bool = True
    gate_subcenters_per_intent: int = 2
    gate_min_id_recall: float = 0.80
    semantic_gate_mode: str = "prototype"
    semantic_prompt_version: str = "ranking_v1"
    semantic_gate_threshold: float = 0.85
    semantic_uncertain_low: float = 0.98
    semantic_uncertain_high: float = 1.05
    semantic_top_k: int = 3
    semantic_tuning_mode: str = "fixed"
    prototype_centers_default: int = 1
    multi_proto_id_threshold: float = 0.5904965996742249
    multi_proto_threshold_mode: str = "fixed"
    router_lora_r: int = 32
    router_lora_alpha: int = 64
    expert_lora_r: int = 16
    expert_lora_alpha: int = 32
    semantic_verifier_lora_r: int = 32
    semantic_verifier_lora_alpha: int = 64
    eval_batch_size: int = 64
    eval_seed: int = 20260317

    def training_gate_defaults(self) -> Dict[str, object]:
        """Return the gate-stage defaults used by the multi-dataset trainer."""

        return {
            "gate_profile": self.gate_profile,
            "gate_center_mode": self.gate_center_mode,
            "gate_distance_metric": self.gate_distance_metric,
            "gate_l2_normalize": self.gate_l2_normalize,
            "gate_subcenters_per_intent": self.gate_subcenters_per_intent,
            "gate_min_id_recall": self.gate_min_id_recall,
        }

    def profile_sections(self) -> Dict[str, Dict[str, object]]:
        """Return the canonical profile as sectioned data/gate/router/expert/eval blocks."""

        return {
            "data": {
                "dataset": "CLINC150",
                "slug": "clinc150",
                "kir": 0.5,
                "seed": 42,
                "data_root": "data/multidataset/v19/clinc150/kir50_seed42",
                "gate_root": "data/multidataset/v19/clinc150/kir50_seed42/gate",
                "router_root": "data/multidataset/v19/clinc150/kir50_seed42/router",
                "experts_root": "data/multidataset/v19/clinc150/kir50_seed42/experts",
            },
            "gate": self.training_gate_defaults(),
            "router": {
                "model_family": "SmolLM-135M",
                "lora_r": self.router_lora_r,
                "lora_alpha": self.router_lora_alpha,
                "num_classes": 10,
            },
            "expert": {
                "model_family": "SmolLM-135M",
                "lora_r": self.expert_lora_r,
                "lora_alpha": self.expert_lora_alpha,
                "one_model_per_domain": True,
            },
            "eval": {
                "gate_mode": self.gate_mode,
                "semantic_gate_mode": self.semantic_gate_mode,
                "semantic_prompt_version": self.semantic_prompt_version,
                "semantic_gate_threshold": self.semantic_gate_threshold,
                "semantic_uncertain_low": self.semantic_uncertain_low,
                "semantic_uncertain_high": self.semantic_uncertain_high,
                "semantic_top_k": self.semantic_top_k,
                "semantic_tuning_mode": self.semantic_tuning_mode,
                "prototype_centers_default": self.prototype_centers_default,
                "multi_proto_id_threshold": self.multi_proto_id_threshold,
                "multi_proto_threshold_mode": self.multi_proto_threshold_mode,
                "semantic_verifier_lora_r": self.semantic_verifier_lora_r,
                "semantic_verifier_lora_alpha": self.semantic_verifier_lora_alpha,
                "batch_size": self.eval_batch_size,
                "seed": self.eval_seed,
            },
        }

    def evaluation_defaults(self) -> Dict[str, object]:
        """Return the evaluation defaults used by the benchmark runner."""

        return {
            "gate_mode": self.gate_mode,
            "semantic_gate_mode": self.semantic_gate_mode,
            "semantic_prompt_version": self.semantic_prompt_version,
            "semantic_gate_threshold": self.semantic_gate_threshold,
            "semantic_uncertain_low": self.semantic_uncertain_low,
            "semantic_uncertain_high": self.semantic_uncertain_high,
            "semantic_top_k": self.semantic_top_k,
            "semantic_tuning_mode": self.semantic_tuning_mode,
            "prototype_centers_default": self.prototype_centers_default,
            "multi_proto_id_threshold": self.multi_proto_id_threshold,
            "multi_proto_threshold_mode": self.multi_proto_threshold_mode,
            "semantic_verifier_lora_r": self.semantic_verifier_lora_r,
            "semantic_verifier_lora_alpha": self.semantic_verifier_lora_alpha,
        }

    def strict_replay_defaults(self) -> Dict[str, object]:
        """Return the frozen historical protocol required for strict replay."""

        return {
            "data_root": "data/v19",
            "known_intents_path": "data/v19/KNOWN_INTENTS.json",
            "gate_encoder_path": "all-MiniLM-L6-v2",
            "reference_eval_results": (
                "outputs/experiments/archive/sweeps/2026-03-23/"
                "pipeline_phase3_proto_eval/"
                "pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json"
            ),
            "frozen_eval_results": (
                "outputs/experiments/pipeline/frozen_prototype_gate/"
                "prototype_gate_frozen_2026-04-09/"
                "prototype_gate_pipeline_frozen/eval_results.json"
            ),
            "frozen_detector_path": (
                "outputs/experiments/archive/sweeps/2026-03-23/"
                "gate_l2_mix2_train/gate_l2_mix2_true_lambda_1p6/detector.json"
            ),
            "target_metrics": {
                "macro_f1": 0.8121241036585573,
                "overall_accuracy": 0.8677941443898891,
                "known_intent_accuracy": 0.799466429524233,
                "oos_f1": 0.9096192078299434,
                "gate_oos_rejection": 0.9150769230769231,
                "gate_id_recall": 0.8599377501111605,
            },
        }


HISTORICAL_BEST_PIPELINE = HistoricalBestPipelineProfile()
