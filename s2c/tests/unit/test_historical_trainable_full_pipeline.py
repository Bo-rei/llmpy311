from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "eval" / "run_historical_trainable_full_pipeline.py"
SPEC = importlib.util.spec_from_file_location("historical_trainable_full_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compute_metrics_separates_gate_router_and_expert_errors() -> None:
    rows = [
        {"label": 0, "intent": "known_a", "domain": "domain_a"},
        {"label": 0, "intent": "known_b", "domain": "domain_a"},
        {"label": 1, "intent": "oos_a", "domain": "unknown"},
        {"label": 1, "intent": "oos_b", "domain": "unknown"},
    ]
    predictions = [
        {"is_oos": False, "gate_pred": 0, "gate_score": 0.2, "domain": "domain_a", "intent": "known_a"},
        {"is_oos": False, "gate_pred": 0, "gate_score": 0.3, "domain": "domain_a", "intent": "wrong"},
        {"is_oos": True, "gate_pred": 1, "gate_score": 1.4},
        {"is_oos": False, "gate_pred": 0, "gate_score": 0.8, "domain": "domain_a", "intent": "known_a"},
    ]
    metrics, stages = MODULE.compute_metrics(rows, predictions)
    assert stages == {
        "correct_oos_rejection": 1,
        "oos_accepted_by_gate": 1,
        "known_rejected_by_gate": 0,
        "known_wrong_domain": 0,
        "known_wrong_expert": 1,
        "correct_known_prediction": 1,
    }
    assert np.isclose(metrics["false_accept_rate"], 0.5)
    assert np.isclose(metrics["expert_error_rate"], 0.5)
    assert metrics["gate_accept_known_count"] == 2


def test_accepted_oos_is_a_false_negative_regardless_of_known_intent_assignment() -> None:
    rows = [
        {"label": 0, "intent": "known_a", "domain": "domain_a"},
        {"label": 0, "intent": "known_b", "domain": "domain_a"},
        {"label": 1, "intent": "held_out_a", "domain": "unknown"},
        {"label": 1, "intent": "held_out_b", "domain": "unknown"},
    ]
    predictions = [
        dict(is_oos=False, gate_score=0.2, intent="known_a", domain="domain_a"),
        dict(is_oos=False, gate_score=0.3, intent="known_b", domain="domain_a"),
        dict(is_oos=True, gate_score=1.4, intent="__oos__", domain="unknown"),
        dict(is_oos=False, gate_score=0.8, intent="known_a", domain="domain_a"),
    ]
    baseline, stages = MODULE.compute_metrics(rows, predictions)
    # TP=1, FP=0, FN=1: the leaked OOS is included in the denominator.
    assert np.isclose(baseline["oos_f1"], 2 / 3)
    assert stages["oos_accepted_by_gate"] == 1
    assert np.isclose(baseline["false_accept_rate"], 0.5)

    changed = [dict(p) for p in predictions]
    changed[3]["intent"] = "known_b"
    changed[0]["intent"] = "known_b"
    downstream_changed, _ = MODULE.compute_metrics(rows, changed)
    assert downstream_changed["oos_f1"] == baseline["oos_f1"]
    assert downstream_changed["overall_accuracy"] < baseline["overall_accuracy"]

    # A real downstream rejection must change the final decision, not just
    # substitute one Known class for another.
    changed[3].update(is_oos=True, intent="__oos__")
    rejected, stages = MODULE.compute_metrics(rows, changed)
    assert rejected["oos_f1"] == 1.0
    assert stages["oos_accepted_by_gate"] == 0


def test_detector_state_adapter_preserves_h1_signature_contract() -> None:
    signature = {
        "radius_method": "mean_std",
        "radius_lambda": 1.0,
        "distance_metric": "mahalanobis_diag",
        "acceptance_mode": "nearest_sphere",
        "spheres": [
            {
                "cluster_id": 0,
                "intent": "known",
                "center": [0.0, 1.0],
                "radius": 2.0,
                "inv_diag_cov": [3.0, 4.0],
            }
        ],
    }
    state = MODULE._detector_state(signature)
    assert state["n_clusters"] == 1
    assert state["intent_to_cluster"] == {"known": 0}
    assert state["cluster_to_intent"] == {"0": "known"}
    assert state["spheres"][0]["intent_name"] == "known"
    assert state["distance_metric"] == "mahalanobis_diag"


def test_kir_tag_changes_only_protocol_paths() -> None:
    original = MODULE.KIR
    try:
        MODULE.KIR = 0.25
        assert MODULE._kir_tag() == "kir25"
        assert MODULE._kir_tag(0.75) == "kir75"
    finally:
        MODULE.KIR = original
