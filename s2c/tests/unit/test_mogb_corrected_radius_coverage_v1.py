from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "analysis" / "run_mogb_corrected_radius_coverage_v1.py"
SPEC = importlib.util.spec_from_file_location("mogb_corrected_radius_coverage_v1", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stable_ratio_quantile_hits_requested_coverage() -> None:
    ratios = np.asarray([0.50, 0.75, 1.00, 1.25, 2.00], dtype=np.float64)
    threshold = MODULE.stable_ratio_quantile(ratios, 0.80)
    assert threshold == 1.25
    assert MODULE.compute_coverage(ratios, threshold) == 0.8


def test_compare_ball_rows_requires_same_identity_and_small_numeric_delta() -> None:
    reference = [
        {"ball_id": 0, "majority_label": 1, "sample_count": 10, "purity": 1.0, "radius": 2.0, "center_l2": 5.0},
        {"ball_id": 1, "majority_label": 2, "sample_count": 12, "purity": 0.9, "radius": 3.0, "center_l2": 6.0},
    ]
    observed = pd.DataFrame(
        [
            {"ball_id": 0, "majority_label": 1, "sample_count": 10, "purity": 1.0 + 1e-8, "radius": 2.0, "center_l2": 5.0},
            {"ball_id": 1, "majority_label": 2, "sample_count": 12, "purity": 0.9, "radius": 3.0 + 1e-8, "center_l2": 6.0},
        ]
    )
    result = MODULE.compare_ball_rows(reference, observed)
    assert result["passed"] is True
    assert result["reason"] == "ok"


def test_per_ball_attribution_counts_known_coverage_and_false_accepts() -> None:
    balls = pd.DataFrame(
        [
            {"ball_id": 0, "majority_label": 0, "sample_count": 20, "purity": 1.0, "radius": 1.0},
            {"ball_id": 1, "majority_label": 1, "sample_count": 15, "purity": 1.0, "radius": 2.0},
        ]
    )
    frame = MODULE.per_ball_attribution(
        workpoint="default_mean",
        nearest_ball=np.asarray([0, 0, 1, 1]),
        accepted=np.asarray([1, 0, 1, 1]),
        y_true=np.asarray([0, 2, 2, 1]),
        y_pred=np.asarray([0, 2, 1, 1]),
        dev_nearest_ball=np.asarray([0, 0, 1]),
        dev_distances=np.asarray([0.8, 1.2, 1.0]),
        dev_radii=np.asarray([1.0, 1.0, 2.0]),
        balls=balls,
        multiplier=1.0,
        unseen_token_id=2,
    )
    row0 = frame.loc[frame.ball_id == 0].iloc[0]
    row1 = frame.loc[frame.ball_id == 1].iloc[0]
    assert row0["dev_assigned_known_covered_count"] == 1
    assert math.isclose(row0["dev_assigned_known_coverage"], 0.5)
    assert row1["test_false_accept_count"] == 1
    assert row1["test_known_rejected_count"] == 0
