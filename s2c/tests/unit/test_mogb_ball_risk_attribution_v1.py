from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "analysis" / "build_mogb_ball_risk_attribution_v1.py"
SPEC = importlib.util.spec_from_file_location("mogb_ball_risk_attribution_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gini_handles_uniform_and_concentrated_vectors() -> None:
    assert MODULE.gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert MODULE.gini([0, 0, 0, 4]) == pytest.approx(0.75)


def test_top_share_uses_at_least_one_ball() -> None:
    assert MODULE.top_share([1, 2, 7], 0.1) == pytest.approx(0.7)
    assert MODULE.top_share([1, 2, 7], 1.0) == pytest.approx(1.0)


def test_rank_correlation_is_tie_safe_and_reports_undefined() -> None:
    assert MODULE.rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert MODULE.rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert np.isnan(MODULE.rank_correlation([1, 1, 1], [1, 2, 3]))
