from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval.eval_system_pipeline_v19 import _evaluate
from tools.analysis.operating_point_analysis_v19 import rank_operating_points


def test_evaluate_reports_oos_breakdown_by_source_group():
    records = [
        {
            "text": "known-a",
            "intent": "intent_a",
            "domain": "banking",
            "label": 0,
            "source_split": "test",
        },
        {
            "text": "id-oos-a",
            "intent": "held_out_a",
            "domain": "unknown",
            "label": 1,
            "source_split": "id-oos_test",
        },
        {
            "text": "ood-oos-a",
            "intent": "oos",
            "domain": "unknown",
            "label": 1,
            "source_split": "ood-oos_test",
        },
    ]
    preds = [
        {
            "gate_pred": 0,
            "is_oos": False,
            "intent": "intent_a",
            "domain": "banking",
            "domain_prob": 1.0,
            "intent_prob": 0.9,
            "gate_score": 0.1,
            "gate_distance": 0.1,
            "gate_radius": 1.0,
        },
        {
            "gate_pred": 0,
            "is_oos": False,
            "intent": "intent_a",
            "domain": "banking",
            "domain_prob": 1.0,
            "intent_prob": 0.8,
            "gate_score": 0.2,
            "gate_distance": 0.2,
            "gate_radius": 1.0,
        },
        {
            "gate_pred": 1,
            "is_oos": True,
            "intent": "__oos__",
            "domain": "__oos__",
            "domain_prob": None,
            "intent_prob": None,
            "gate_score": 0.9,
            "gate_distance": 0.9,
            "gate_radius": 1.0,
        },
    ]

    metrics = _evaluate(records, preds)

    assert metrics["oos_by_source"]["id_oos"]["count"] == 1
    assert metrics["oos_by_source"]["id_oos"]["gate_oos_rejection"] == 0.0
    assert metrics["oos_by_source"]["ood_oos"]["count"] == 1
    assert metrics["oos_by_source"]["ood_oos"]["gate_oos_rejection"] == 1.0


def test_rank_operating_points_filters_by_constraints_and_sorts_by_oos_f1(tmp_path: Path):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p3 = tmp_path / "c.json"

    payloads = [
        (
            p1,
            {
                "config": {"semantic_uncertain_low": 0.98, "semantic_uncertain_high": 1.05},
                "metrics": {
                    "known_intent_accuracy": 0.731,
                    "gate_id_recall": 0.821,
                    "gate_oos_rejection": 0.807,
                    "oos_f1": 0.866,
                    "macro_f1": 0.627,
                },
            },
        ),
        (
            p2,
            {
                "config": {"semantic_uncertain_low": 0.96, "semantic_uncertain_high": 1.04},
                "metrics": {
                    "known_intent_accuracy": 0.703,
                    "gate_id_recall": 0.785,
                    "gate_oos_rejection": 0.846,
                    "oos_f1": 0.883,
                    "macro_f1": 0.642,
                },
            },
        ),
        (
            p3,
            {
                "config": {"semantic_uncertain_low": 1.00, "semantic_uncertain_high": 1.08},
                "metrics": {
                    "known_intent_accuracy": 0.741,
                    "gate_id_recall": 0.838,
                    "gate_oos_rejection": 0.771,
                    "oos_f1": 0.846,
                    "macro_f1": 0.607,
                },
            },
        ),
    ]
    for path, payload in payloads:
        path.write_text(json.dumps(payload), encoding="utf-8")

    ranked = rank_operating_points(
        [p1, p2, p3],
        min_gate_id_recall=0.80,
        min_known_intent_accuracy=0.72,
        sort_by="oos_f1",
    )

    assert [row["path"] for row in ranked] == [str(p1), str(p3)]
    assert ranked[0]["oos_f1"] == 0.866
