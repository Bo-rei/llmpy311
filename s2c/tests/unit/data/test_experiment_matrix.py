"""Regression tests for the declared protocol_v2 Gate execution matrices."""

from pathlib import Path

from protocol_v2.experiments.matrix import load_gate_matrix
from protocol_v2.experiments.runner import dry_run
from protocol_v2.runtime.paths import ProtocolV2Paths

from tests.fixtures.protocol_v2_helpers import make_paths


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_smoke_and_core_matrices_have_declared_unit_counts() -> None:
    root = PROJECT_ROOT / "configs/experiments/protocol_v2_textoir_v1"
    smoke = load_gate_matrix(root / "smoke_gate.yaml")
    core = load_gate_matrix(root / "gate_core_dense.yaml")
    assert len(smoke) == 36
    assert len(core) == 1650
    assert {spec.protocol_version for spec in smoke + core} == {"protocol_v2_textoir_v1"}
    assert len({spec.run_id for spec in core}) == len(core)


def test_dry_run_uses_only_local_protocol_v2_export_paths(tmp_path: Path) -> None:
    paths: ProtocolV2Paths = make_paths(tmp_path)
    spec = load_gate_matrix(PROJECT_ROOT / "configs/experiments/protocol_v2_textoir_v1/smoke_gate.yaml")[0]
    payload = dry_run(paths, spec)
    assert payload["uses_textoir_data"] is False
    assert "textoir" not in payload["runtime_data_root"]
    # The active protocol name contains "textoir" by design; the forbidden
    # runtime dependency is the external repository's corpus path.
    assert all("/textoir/data/" not in item["path"] for item in payload["required_inputs"])
