"""Regression coverage for the installed src-layout Gate runner."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_protocol_runner_builds_legacy_detector_through_public_gate_package(tmp_path: Path) -> None:
    """The protocol package must not rely on the checkout-only ``src.*`` namespace."""
    source_root = Path(__file__).resolve().parents[3] / "src"
    environment = {**os.environ, "PYTHONPATH": str(source_root)}
    code = """
from s2c.experiments.matrix import GateRunSpec
from s2c.experiments.runner import _build_detector

spec = GateRunSpec('unit', 'clinc150', 0.50, 42, 1, 'euclidean',
                   'frozen_minilm', 'mean_std', 1.0, 'all-MiniLM-L6-v2', 'cpu')
assert _build_detector(spec).subcenters_per_intent == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
