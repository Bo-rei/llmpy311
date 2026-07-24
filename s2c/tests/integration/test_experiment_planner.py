"""Integration coverage for protocol_v2 matrix planning and resume semantics."""

from pathlib import Path

from s2c.experiments import runner
from s2c.experiments.matrix import filter_gate_specs, load_gate_matrix
from s2c.experiments.registry import write_plan
from s2c.runtime.paths import ProtocolV2Paths


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_planner_writes_a_unique_declared_smoke_matrix(tmp_path: Path) -> None:
    specs = load_gate_matrix(PROJECT_ROOT / "configs/experiments/protocol_v2/smoke_gate.yaml")
    output = tmp_path / "smoke.json"
    write_plan(output, specs)
    text = output.read_text(encoding="utf-8")
    assert '"run_count":36' in text
    assert text.count('"run_id"') == 36


def test_shard_name_isolates_plan_and_state_files(tmp_path: Path, monkeypatch) -> None:
    """A matrix shard may not overwrite the unsharded resume state."""
    paths = ProtocolV2Paths(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        results_root=tmp_path / "results",
        legacy_root=tmp_path / "legacy",
        textoir_import_root=None,
    )
    config = PROJECT_ROOT / "configs/experiments/protocol_v2/smoke_gate.yaml"
    monkeypatch.setattr(runner.ProtocolV2Paths, "discover", classmethod(lambda cls: paths))
    monkeypatch.setattr(runner, "run_matrix", lambda *args, **kwargs: ([], []))

    assert runner.main(["--config", str(config), "--dataset", "clinc150", "--shard-name", "clinc150"]) == 0
    assert (paths.run_root / "plans" / "smoke_gate.clinc150.json").is_file()
    assert (paths.run_root / "plans" / "smoke_gate.clinc150.state.json").is_file()
    assert not (paths.run_root / "plans" / "smoke_gate.state.json").exists()


def test_boundary_matrix_expands_declared_methods() -> None:
    specs = load_gate_matrix(PROJECT_ROOT / "configs/experiments/protocol_v2/boundary_dense.yaml")
    assert len(specs) == 2160
    assert {spec.boundary for spec in specs} == {
        "mean_std",
        "quantile_90",
        "quantile_95",
        "quantile_975",
        "median_mad",
        "known_conformal",
    }


def test_matrix_filter_keeps_the_declared_official_smoke_shard() -> None:
    specs = load_gate_matrix(PROJECT_ROOT / "configs/experiments/protocol_v2/smoke_gate.yaml")
    selected = filter_gate_specs(specs, datasets=["clinc150"], seeds=[42], kirs=[0.50])
    assert len(selected) == 4
    assert {spec.dataset for spec in selected} == {"clinc150"}
    assert {spec.seed for spec in selected} == {42}
    assert {spec.kir for spec in selected} == {0.50}
