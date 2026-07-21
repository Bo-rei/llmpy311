"""Cascade 汇总器的配对差值与错误聚合回归测试。"""

import json
from pathlib import Path

import pytest

from tools.analysis.export_cascade_repair_summary import (
    _error_summary_rows,
    _paired_rows,
    _seed_from_kir_seed,
    _summary_rows,
)


def _row(seed: int, gate: str, oos: float, known: float) -> dict[str, object]:
    return {
        "dataset": "banking77_oos",
        "kir_seed": f"kir50_seed{seed}",
        "gate": gate,
        "oos_f1": oos,
        "known_macro_f1": known,
        "overall_accuracy": 0.7,
        "id_recall": 0.8,
        "oos_false_accept_rate": 0.2,
        "known_false_reject_rate": 0.2,
        "router_error_rate": 0.0,
        "expert_error_rate": 0.1,
        "result_path": "unused",
        "result_sha256": "unused",
    }


def test_seed_parser_and_paired_delta_use_same_seed_reference() -> None:
    assert _seed_from_kir_seed("kir50_seed42") == 42
    rows = [
        _row(13, "frozen_k1", 0.50, 0.80),
        _row(13, "ce_recon_selected_k", 0.60, 0.75),
    ]

    paired = _paired_rows(rows)

    ce = next(row for row in paired if row["gate"] == "ce_recon_selected_k")
    assert ce["seed"] == 13
    assert ce["delta_vs_frozen_k1_oos_f1"] == pytest.approx(0.10)
    assert ce["delta_vs_frozen_k1_known_macro_f1"] == pytest.approx(-0.05)


def test_summary_rows_report_mean_and_std_of_seed_level_deltas() -> None:
    rows = [
        _row(13, "frozen_k1", 0.50, 0.80),
        _row(13, "ce_recon_selected_k", 0.60, 0.75),
        _row(42, "frozen_k1", 0.60, 0.70),
        _row(42, "ce_recon_selected_k", 0.70, 0.65),
    ]

    summary = _summary_rows(_paired_rows(rows))
    ce = next(row for row in summary if row["gate"] == "ce_recon_selected_k")
    assert ce["seed_count"] == 2
    assert ce["delta_vs_frozen_k1_oos_f1_mean"] == pytest.approx(0.10)
    assert ce["delta_vs_frozen_k1_known_macro_f1_mean"] == pytest.approx(-0.05)


def test_error_summary_keeps_dataset_gate_stage_groups(tmp_path: Path) -> None:
    result = tmp_path / "eval_results.json"
    result.write_text(
        json.dumps(
            {
                "cascade_error_decomposition_sample_level": {
                    "counts": {
                        "correct_oos_rejection": 3,
                        "oos_accepted_by_gate": 1,
                    },
                    "rates": {
                        "correct_oos_rejection": 0.75,
                        "oos_accepted_by_gate": 0.25,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    rows = [_row(13, "frozen_k1", 0.5, 0.8) | {"result_path": str(result)}]

    summary = _error_summary_rows(rows)

    assert {row["stage"] for row in summary} == {
        "correct_oos_rejection",
        "oos_accepted_by_gate",
    }
    accepted = next(row for row in summary if row["stage"] == "oos_accepted_by_gate")
    assert accepted["count_total"] == 1
    assert accepted["rate_mean"] == 0.25
