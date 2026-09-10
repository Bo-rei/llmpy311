from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "tools/analysis/build_s2c_mogb_operating_curve_attribution_v1.py"
SPEC = importlib.util.spec_from_file_location("s2c_mogb_operating_curve", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_threshold_replays_expected_oos_flags() -> None:
    y_oos = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.2, 0.9, 1.1, 2.0])
    metrics = MODULE.binary_metrics(y_oos, scores, threshold=1.0)
    assert metrics["known_recall"] == 1.0
    assert metrics["false_accept_rate"] == 0.0
    assert metrics["oos_f1"] == 1.0


def test_oracle_f1_is_not_below_fixed_threshold() -> None:
    y_oos = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.1, 0.4, 1.2, 0.8, 1.3, 1.8])
    fixed = MODULE.binary_metrics(y_oos, scores, threshold=1.0)["oos_f1"]
    oracle, threshold = MODULE.oracle_f1(y_oos, scores)
    assert np.isfinite(threshold)
    assert oracle >= fixed


def test_coverage_threshold_meets_requested_known_coverage() -> None:
    known_scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    threshold = MODULE.coverage_threshold(known_scores, 0.80)
    achieved = float(np.mean(known_scores <= threshold))
    assert achieved >= 0.80


def test_source_contract_has_90_existing_prediction_files() -> None:
    paths = MODULE.source_paths()
    assert len(paths) == 90
    assert all(path.is_file() for path in paths.values())
