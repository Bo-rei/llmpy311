from __future__ import annotations

from pathlib import Path

from protocol_v2.experiments.racal_v1.stage2 import stage2_root
from protocol_v2.runtime.paths import ProtocolV2Paths


def test_stage2_root_is_separate_from_stage1_and_e2() -> None:
    paths = ProtocolV2Paths.discover(Path(__file__).resolve().parents[2])
    root = stage2_root(paths)
    assert root.name == "stage2_fixed_k2"
    assert root.parent.name == "racal_v1"
    assert root != paths.run_root / "e2_gate_core_dense"
    assert root != paths.run_root / "racal_v1"
