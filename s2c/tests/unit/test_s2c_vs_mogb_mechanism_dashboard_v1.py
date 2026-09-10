from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "tools/analysis/build_s2c_vs_mogb_mechanism_dashboard_v1.py"
SPEC = importlib.util.spec_from_file_location("s2c_vs_mogb_dashboard", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_oos_precision_is_recovered_from_f1_and_recall() -> None:
    precision = np.asarray([0.8, 0.6])
    recall = np.asarray([0.5, 0.9])
    f1 = 2 * precision * recall / (precision + recall)
    recovered = MODULE.oos_precision_from_f1_recall(f1, recall)
    np.testing.assert_allclose(recovered, precision, rtol=0, atol=1e-12)


def test_paired_ci_is_deterministic_and_contains_constant() -> None:
    first = MODULE.paired_ci(np.asarray([0.1, 0.1, 0.1, 0.1, 0.1]))
    second = MODULE.paired_ci(np.asarray([0.1, 0.1, 0.1, 0.1, 0.1]))
    assert first == second
    np.testing.assert_allclose(first, (0.1, 0.1), rtol=0, atol=1e-12)


def test_load_inputs_has_complete_paired_coverage() -> None:
    trainable, mogb = MODULE.load_inputs()
    assert len(trainable) == 45
    assert len(mogb) == 90
