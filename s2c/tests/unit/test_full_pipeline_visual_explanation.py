from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "analysis" / "build_full_pipeline_visual_explanation.py"
SPEC = importlib.util.spec_from_file_location("full_pipeline_visual_explanation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gate_f1_reconstructs_binary_gate_metric() -> None:
    # 80 known: 72 accepted; 100 OOS: 90 rejected.
    value = MODULE._gate_f1(80, 100, 72 / 80, 90 / 100)
    precision = 90 / (90 + 8)
    expected = 2 * precision * 0.90 / (precision + 0.90)
    assert np.isclose(value, expected)


def test_stage_and_method_contracts_are_explicit() -> None:
    assert MODULE.METHOD_LABELS["trainable_k1_current_h1"] == "Trainable H1 K=1"
    assert MODULE.STAGE_LABELS["known_wrong_domain"] == "Router"
    assert MODULE.STAGE_LABELS["known_wrong_expert"] == "Expert"
    assert len(MODULE.STAGES) == 6
