from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "tools/analysis/build_s2c_mogb_intent_structure_bridge_v1.py"
SPEC = importlib.util.spec_from_file_location("s2c_mogb_intent_bridge", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ball_group_contract() -> None:
    assert [MODULE.ball_group(value) for value in (0, 1, 2, 3, 8)] == ["0", "1", "2", "3+", "3+"]


def test_concentration_helpers() -> None:
    values = np.asarray([7.0, 2.0, 1.0, 0.0])
    assert MODULE.top_fraction_share(values, 0.25) == 0.7
    assert 0.0 < MODULE.gini(values) < 1.0


def test_bridge_recovers_known_and_oos_cost_deltas() -> None:
    base = {
        "dataset": "d",
        "kir": 0.5,
        "seed": 42,
        "intent": "a",
        "known_count": 10,
    }
    errors = pd.DataFrame(
        [
            {
                **base,
                "method": "trainable_k1",
                "known_false_reject": 1,
                "known_false_reject_rate": 0.1,
                "oos_false_accept": 3,
                "oos_false_accept_rate": 0.03,
            },
            {
                **base,
                "method": "mogb_minilm",
                "known_false_reject": 6,
                "known_false_reject_rate": 0.6,
                "oos_false_accept": 1,
                "oos_false_accept_rate": 0.01,
            },
        ]
    )
    balls = pd.DataFrame(
        [
            {
                "dataset": "d",
                "kir": 0.5,
                "seed": 42,
                "majority_label": "a",
                "ball_id": 1,
                "sample_count": 7,
                "radius": 0.5,
                "depth": 1,
                "tiny_support_lt_20": True,
            },
            {
                "dataset": "d",
                "kir": 0.5,
                "seed": 42,
                "majority_label": "a",
                "ball_id": 2,
                "sample_count": 3,
                "radius": 0.4,
                "depth": 2,
                "tiny_support_lt_20": True,
            },
        ]
    )
    bridge = MODULE.build_intent_bridge(errors, balls)
    row = bridge.iloc[0]
    assert row["known_recovery_count"] == 5
    assert row["extra_oos_accept_count"] == 2
    assert row["coverage_tradeoff_net_count"] == 3
    assert row["ball_count"] == 2


def test_frozen_inputs_have_expected_methods() -> None:
    errors, balls = MODULE.load_inputs()
    assert set(errors["method"]) == {"trainable_k1", "mogb_minilm"}
    assert set(balls["method"]) == {"mogb_minilm"}


def test_report_renderer_accepts_scalar_overall_summary() -> None:
    bridge = pd.DataFrame(
        {
            "intent": ["a"],
            "known_recovery_rate": [0.5],
            "extra_oos_accept_rate": [0.1],
            "missing_selected_ball": [False],
        }
    )
    group_summary = pd.DataFrame(
        {
            "dataset": ["d"],
            "kir": [0.5],
            "ball_count_group": ["1"],
            "intent_seed_rows": [1],
            "known_recovery_rate_mean": [0.5],
            "extra_oos_accept_rate_mean": [0.1],
            "coverage_tradeoff_net_count_mean": [1.0],
        }
    )
    dataset_summary = pd.DataFrame(
        {
            "dataset": ["d"],
            "kir": [0.5],
            "positive_recovery_intent_ratio_mean": [1.0],
            "missing_selected_ball_ratio_mean": [0.0],
        }
    )
    concentration = pd.DataFrame(
        {
            "dataset": ["d"],
            "kir": [0.5],
            "positive_recovery_intent_ratio": [1.0],
            "top10pct_recovery_share": [1.0],
            "top20pct_recovery_share": [1.0],
            "recovery_gini": [0.0],
        }
    )
    correlations = pd.DataFrame(
        {
            "outcome": ["known_recovery_rate"],
            "predictor": ["ball_count"],
            "spearman_rho": [0.0],
        }
    )
    report = MODULE.render_report(
        bridge,
        group_summary,
        dataset_summary,
        concentration,
        correlations,
        "abc",
    )
    assert "共桥接 `1` 条" in report
