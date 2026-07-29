from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval.eval_system_pipeline_v19 import (
    _apply_id_rescue_threshold,
    _evaluate,
    _tune_id_rescue_threshold_from_scored_predictions,
)
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector, SphereConfig
from legacy.pipeline.system_pipeline import HiLSAMoEV19Pipeline, PipelinePaths


def test_evaluate_exposes_primary_pipeline_metrics():
    records = [
        {"text": "k1", "intent": "intent_a", "domain": "banking", "label": 0},
        {"text": "k2", "intent": "intent_b", "domain": "banking", "label": 0},
        {"text": "o1", "intent": "oos", "domain": "unknown", "label": 1, "source_split": "ood-oos_test"},
        {"text": "o2", "intent": "oos", "domain": "unknown", "label": 1, "source_split": "ood-oos_test"},
    ]
    preds = [
        {
            "is_oos": False,
            "gate_pred": 0,
            "intent": "intent_a",
            "domain": "banking",
            "gate_score": 0.7,
            "gate_distance": 0.7,
            "gate_radius": 1.0,
            "domain_prob": 1.0,
            "intent_prob": 0.91,
        },
        {
            "is_oos": True,
            "gate_pred": 1,
            "intent": "__oos__",
            "domain": "",
            "gate_score": 1.1,
            "gate_distance": 1.1,
            "gate_radius": 1.0,
            "domain_prob": None,
            "intent_prob": None,
        },
        {
            "is_oos": True,
            "gate_pred": 1,
            "intent": "__oos__",
            "domain": "",
            "gate_score": 1.2,
            "gate_distance": 1.2,
            "gate_radius": 1.0,
            "domain_prob": None,
            "intent_prob": None,
        },
        {
            "is_oos": False,
            "gate_pred": 0,
            "intent": "intent_b",
            "domain": "banking",
            "gate_score": 0.8,
            "gate_distance": 0.8,
            "gate_radius": 1.0,
            "domain_prob": 1.0,
            "intent_prob": 0.75,
        },
    ]

    metrics = _evaluate(records, preds)

    assert metrics["overall_accuracy"] == 0.5
    assert metrics["known_accuracy"] == 0.5
    assert metrics["oos_accuracy"] == 0.5
    assert metrics["known_accuracy"] == metrics["known_intent_accuracy"]
    assert metrics["oos_accuracy"] == metrics["gate_oos_rejection"]
    assert metrics["known_macro_f1"] == 0.5
    assert metrics["primary_metrics"] == {
        "overall_accuracy": 0.5,
        "known_accuracy": 0.5,
        "oos_accuracy": 0.5,
        "macro_f1": metrics["macro_f1"],
        "known_macro_f1": 0.5,
        "oos_f1": metrics["oos_f1"],
    }


def test_evaluate_exposes_semantic_override_diagnostics():
    records = [
        {"text": "k1", "intent": "intent_a", "domain": "banking", "label": 0},
        {"text": "k2", "intent": "intent_b", "domain": "banking", "label": 0},
        {"text": "o1", "intent": "held_out_a", "domain": "unknown", "label": 1, "source_split": "heldout_oos_test"},
        {"text": "o2", "intent": "held_out_b", "domain": "unknown", "label": 1, "source_split": "heldout_oos_test"},
    ]
    preds = [
        {
            "is_oos": False,
            "gate_pred": 0,
            "fast_gate_pred": 1,
            "gate_stage": "semantic_gate",
            "intent": "intent_a",
            "domain": "banking",
            "gate_score": 0.7,
            "gate_distance": 0.7,
            "gate_radius": 1.0,
            "domain_prob": 1.0,
            "intent_prob": 0.91,
        },
        {
            "is_oos": True,
            "gate_pred": 1,
            "fast_gate_pred": 1,
            "gate_stage": "fast_gate",
            "intent": "__oos__",
            "domain": "",
            "gate_score": 1.1,
            "gate_distance": 1.1,
            "gate_radius": 1.0,
            "domain_prob": None,
            "intent_prob": None,
        },
        {
            "is_oos": False,
            "gate_pred": 0,
            "fast_gate_pred": 1,
            "gate_stage": "semantic_gate",
            "intent": "intent_a",
            "domain": "banking",
            "gate_score": 0.8,
            "gate_distance": 0.8,
            "gate_radius": 1.0,
            "domain_prob": 1.0,
            "intent_prob": 0.75,
        },
        {
            "is_oos": True,
            "gate_pred": 1,
            "fast_gate_pred": 0,
            "gate_stage": "semantic_gate",
            "intent": "__oos__",
            "domain": "",
            "gate_score": 1.3,
            "gate_distance": 1.3,
            "gate_radius": 1.0,
            "domain_prob": None,
            "intent_prob": None,
        },
    ]

    metrics = _evaluate(records, preds)

    assert metrics["fast_gate_metrics"]["gate_id_recall"] == 0.0
    assert metrics["fast_gate_metrics"]["gate_oos_rejection"] == 0.5
    assert metrics["post_semantic_metrics"]["gate_id_recall"] == metrics["gate_id_recall"]
    assert metrics["post_semantic_metrics"]["gate_oos_rejection"] == metrics["gate_oos_rejection"]
    assert metrics["semantic_override_delta"] == {
        "uncertain_count": 3,
        "changed_to_id": 2,
        "changed_to_oos": 1,
        "id_false_reject_before_semantic": 2,
        "id_false_reject_after_semantic": 1,
        "oos_false_accept_before_semantic": 1,
        "oos_false_accept_after_semantic": 1,
    }


def test_pipeline_applies_multisphere_radius_scale():
    pipeline = HiLSAMoEV19Pipeline(
        paths=PipelinePaths(
            model_path=Path("model"),
            gate_encoder_path=Path("gate_encoder"),
            gate_detector_path=Path("detector.json"),
            router_ckpt_path=Path("router.pt"),
            experts_root=Path("experts"),
            experts_data_root=Path("experts_data"),
            router_data_path=Path("router_train.json"),
            gate_train_path=Path("gate_train.json"),
        ),
        gate_radius_scale=0.975,
    )
    detector = MultiSphereOOSDetector()
    detector.spheres = [
        SphereConfig(center=[], radius=20.0, cluster_id=0),
        SphereConfig(center=[], radius=10.0, cluster_id=1),
    ]
    pipeline.gate_detector = detector

    pipeline._apply_gate_radius_scale()

    assert [sphere.radius for sphere in detector.spheres] == [19.5, 9.75]


def test_id_rescue_threshold_restores_high_confidence_oos_predictions():
    preds = [
        {
            "text": "known",
            "gate_pred": 1,
            "is_oos": True,
            "fast_gate_pred": 0,
            "intent": "__oos__",
            "domain": "",
            "rescue_score": 0.96,
            "rescue_domain": "stackoverflow",
            "rescue_domain_id": 0,
            "rescue_domain_prob": 1.0,
            "rescue_intent": "ajax",
            "rescue_intent_id": 3,
            "rescue_intent_prob": 0.96,
        },
        {
            "text": "unknown",
            "gate_pred": 1,
            "is_oos": True,
            "fast_gate_pred": 1,
            "intent": "__oos__",
            "domain": "",
            "rescue_score": 0.70,
            "rescue_domain": "stackoverflow",
            "rescue_domain_id": 0,
            "rescue_domain_prob": 1.0,
            "rescue_intent": "ajax",
            "rescue_intent_id": 3,
            "rescue_intent_prob": 0.70,
        },
    ]

    rescued = _apply_id_rescue_threshold(preds, threshold=0.9)

    assert rescued[0]["gate_pred"] == 0
    assert rescued[0]["is_oos"] is False
    assert rescued[0]["gate_stage"] == "id_rescue"
    assert rescued[0]["intent"] == "ajax"
    assert rescued[1]["gate_pred"] == 1
    assert rescued[1]["is_oos"] is True


def test_id_rescue_tuning_can_optimize_oos_f1():
    records = [
        {"text": "known", "intent": "ajax", "domain": "stackoverflow", "label": 0},
        {"text": "oos", "intent": "unknown", "domain": "unknown", "label": 1},
        {"text": "oos2", "intent": "unknown2", "domain": "unknown", "label": 1},
    ]
    preds = [
        {
            "gate_pred": 1,
            "is_oos": True,
            "fast_gate_pred": 0,
            "intent": "__oos__",
            "domain": "",
            "rescue_score": 0.95,
            "rescue_domain": "stackoverflow",
            "rescue_domain_id": 0,
            "rescue_domain_prob": 1.0,
            "rescue_intent": "ajax",
            "rescue_intent_id": 0,
            "rescue_intent_prob": 0.95,
        },
        {
            "gate_pred": 1,
            "is_oos": True,
            "fast_gate_pred": 1,
            "intent": "__oos__",
            "domain": "",
            "rescue_score": 0.80,
            "rescue_domain": "stackoverflow",
            "rescue_domain_id": 0,
            "rescue_domain_prob": 1.0,
            "rescue_intent": "ajax",
            "rescue_intent_id": 0,
            "rescue_intent_prob": 0.80,
        },
        {
            "gate_pred": 1,
            "is_oos": True,
            "fast_gate_pred": 1,
            "intent": "__oos__",
            "domain": "",
            "rescue_score": 0.60,
            "rescue_domain": "stackoverflow",
            "rescue_domain_id": 0,
            "rescue_domain_prob": 1.0,
            "rescue_intent": "ajax",
            "rescue_intent_id": 0,
            "rescue_intent_prob": 0.60,
        },
    ]

    tuning = _tune_id_rescue_threshold_from_scored_predictions(
        records=records,
        scored_predictions=preds,
        objective="val_oos_f1",
    )

    assert tuning["best_threshold"] > 0.8
    assert tuning["best_oos_f1"] == 1.0


def test_id_rescue_tuning_recall_guard_prefers_threshold_meeting_baseline_recall():
    records = [
        {"text": "known", "intent": "ajax", "domain": "stackoverflow", "label": 0},
        {"text": "oos", "intent": "unknown", "domain": "unknown", "label": 1},
        {"text": "oos2", "intent": "unknown2", "domain": "unknown", "label": 1},
    ]
    preds = [
        {
            "gate_pred": 1,
            "is_oos": True,
            "intent": "__oos__",
            "domain": "",
            "rescue_score": 0.95,
            "rescue_intent": "ajax",
            "rescue_intent_prob": 0.95,
        },
        {
            "gate_pred": 1,
            "is_oos": True,
            "intent": "__oos__",
            "domain": "",
            "rescue_score": 0.90,
            "rescue_intent": "ajax",
            "rescue_intent_prob": 0.90,
        },
        {
            "gate_pred": 1,
            "is_oos": True,
            "intent": "__oos__",
            "domain": "",
            "rescue_score": 0.60,
            "rescue_intent": "ajax",
            "rescue_intent_prob": 0.60,
        },
    ]

    tuning = _tune_id_rescue_threshold_from_scored_predictions(
        records=records,
        scored_predictions=preds,
        objective="val_oos_f1_recall_guard",
        min_oos_recall=1.0,
    )

    assert tuning["best_threshold"] > 0.90
    assert tuning["best_oos_recall"] == 1.0
