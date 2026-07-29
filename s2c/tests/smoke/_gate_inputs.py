"""Shared local-only smoke assertion for the active Gate input contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from protocol_v2.experiments.matrix import load_gate_matrix
from protocol_v2.experiments.runner import dry_run
from protocol_v2.runtime.paths import ProtocolV2Paths


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def assert_local_gate_inputs(dataset: str) -> None:
    paths = ProtocolV2Paths.discover(PROJECT_ROOT)
    specs = [
        spec
        for spec in load_gate_matrix(
            PROJECT_ROOT / "configs/experiments/protocol_v2_textoir_v1/smoke_gate.yaml"
        )
        if spec.dataset == dataset
    ]
    if not all(item["exists"] for spec in specs for item in dry_run(paths, spec)["required_inputs"]):
        pytest.skip("active TEXTOIR-compatible smoke exports are not materialized")
    assert len(specs) == 12
    for spec in specs:
        payload = dry_run(paths, spec)
        assert payload["uses_textoir_data"] is False
        assert all(item["exists"] for item in payload["required_inputs"])
