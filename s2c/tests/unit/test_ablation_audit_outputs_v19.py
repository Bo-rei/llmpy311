from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis import audit_ablation_outputs_v19 as audit


def _write_prediction_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_recompute_metrics_uses_known_macro_f1_and_oos_binary_counts():
    rows = [
        audit.NormalizedPrediction(
            dataset="toy",
            variant="v",
            sample_id="0",
            text="a",
            gold_label="intent_a",
            pred_label="intent_a",
            gold_is_oos=False,
            pred_is_oos=False,
            gate_score=None,
            gate_decision="id",
            router_pred=None,
            expert_pred="intent_a",
            confidence_score=None,
            threshold=None,
            gate_model_path=None,
            router_model_path=None,
            expert_model_path=None,
            output_file_source="predictions.json",
        ),
        audit.NormalizedPrediction(
            dataset="toy",
            variant="v",
            sample_id="1",
            text="b",
            gold_label="intent_b",
            pred_label="__oos__",
            gold_is_oos=False,
            pred_is_oos=True,
            gate_score=None,
            gate_decision="oos",
            router_pred=None,
            expert_pred=None,
            confidence_score=None,
            threshold=None,
            gate_model_path=None,
            router_model_path=None,
            expert_model_path=None,
            output_file_source="predictions.json",
        ),
        audit.NormalizedPrediction(
            dataset="toy",
            variant="v",
            sample_id="2",
            text="c",
            gold_label="__oos__",
            pred_label="__oos__",
            gold_is_oos=True,
            pred_is_oos=True,
            gate_score=None,
            gate_decision="oos",
            router_pred=None,
            expert_pred=None,
            confidence_score=None,
            threshold=None,
            gate_model_path=None,
            router_model_path=None,
            expert_model_path=None,
            output_file_source="predictions.json",
        ),
    ]

    metrics, counts = audit.recompute_metrics(rows)

    assert metrics["known_f1"] == 0.5
    assert metrics["oos_f1"] == 2 / 3
    assert metrics["acc"] == 2 / 3
    assert counts["tp_oos"] == 1
    assert counts["fp_oos"] == 1
    assert counts["fn_oos"] == 0


def test_prediction_overlap_detects_identical_oos_decisions():
    left = [
        audit.NormalizedPrediction(
            dataset="toy",
            variant="Cascade-MiniLM",
            sample_id=str(idx),
            text=str(idx),
            gold_label="__oos__" if idx else "intent",
            pred_label="__oos__" if idx else "intent_a",
            gold_is_oos=bool(idx),
            pred_is_oos=bool(idx),
            gate_score=float(idx),
            gate_decision="oos" if idx else "id",
            router_pred=None,
            expert_pred=None,
            confidence_score=None,
            threshold=None,
            gate_model_path="minilm_gate",
            router_model_path=None,
            expert_model_path=None,
            output_file_source="left.json",
        )
        for idx in range(2)
    ]
    right = [
        row.with_variant(
            variant="Cascade-SmolLM",
            pred_label=row.pred_label,
            gate_model_path="smollm_gate",
            output_file_source="right.json",
        )
        for row in left
    ]

    overlap = audit.compare_prediction_overlap(left, right)

    assert overlap["same_binary_rate"] == 1.0
    assert overlap["same_pred_label_rate"] == 1.0
    assert overlap["tp_fp_fn_identical"] is True


def test_discover_prediction_groups_uses_banking_geometric_replacement(tmp_path: Path):
    root = tmp_path / "exp"
    prediction_path = (
        root
        / "banking77_oos"
        / "kir25_seed42"
        / "banking_wo_geometric_gate_expert_confidence"
        / "predictions.json"
    )
    _write_prediction_rows(prediction_path, [])

    groups = audit.discover_prediction_groups(root)

    assert groups[("BANKING77-OOS", "kir25_seed42", "w/o Gate")] == prediction_path
