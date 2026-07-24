"""Protocol-v2 E4 adapter tests without models, network, or TEXTOIR runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from s2c.data import source_import
from s2c.data.canonicalize import build_canonical_dataset
from s2c.data.export_protocol import export_protocol
from s2c.data.registry import build_registry
from s2c.data.views import build_views
from s2c.experiments.external_baselines import (
    ExternalBaselineSpec,
    dry_run,
    load_external_baseline_matrix,
    method_availability,
    run_external_baseline,
)

from tests.fixtures.protocol_v2_helpers import make_paths, write_tsv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_runnable_snapshot(tmp_path: Path) -> Path:
    """Create enough Known examples for logistic, kNN and LOF smoke paths."""
    root = tmp_path / "textoir"
    labels = ("alpha", "beta", "gamma", "delta")
    for split in ("train", "dev", "test"):
        rows = [
            (f"{split} {label} request {repeat}", label)
            for label in labels
            for repeat in range(3)
        ]
        if split == "test":
            rows.append(("test native unknown request", "oos"))
        write_tsv(root / "data" / "oos" / f"{split}.tsv", rows)
    return root


def _prepared_paths(tmp_path: Path, monkeypatch) -> object:
    paths = make_paths(tmp_path)
    paths.experiment_admission_path.parent.mkdir(parents=True)
    paths.experiment_admission_path.write_text('{"status": "admitted"}\n', encoding="utf-8")
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, _write_runnable_snapshot(tmp_path), ["clinc150"])
    build_canonical_dataset(paths, "clinc150")
    build_registry(paths, "clinc150", 0, 0.5)
    build_views(paths, "clinc150", 0, 0.5)
    export_protocol(paths, ["clinc150"], [0.5], [0])
    return paths


class _FakeEncoder:
    """A deterministic local encoder substitute; no model download is possible."""

    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("alpha" in lowered),
                    float("beta" in lowered),
                    float("gamma" in lowered),
                    float("delta" in lowered or "unknown" in lowered),
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


def _spec(method: str) -> ExternalBaselineSpec:
    return ExternalBaselineSpec(
        experiment_name="external_baselines",
        dataset="clinc150",
        kir=0.5,
        seed=0,
        method=method,
        representation="frozen_minilm",
        encoder_name="fake-minilm",
        encoder_device="cpu",
    )


def _write_fake_model(paths) -> None:
    model = paths.project_root.parent / "assets" / "models" / "fake-minilm"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")


def test_external_matrix_has_one_fixed_registry_cell_per_method() -> None:
    specs = load_external_baseline_matrix(PROJECT_ROOT / "configs/experiments/protocol_v2/external_baselines.yaml")
    assert len(specs) == 27
    assert {(spec.dataset, spec.kir, spec.seed) for spec in specs} == {("clinc150", 0.5, 0)} | {
        ("banking77", 0.5, 0),
        ("stackoverflow", 0.5, 0),
    }
    assert {spec.method for spec in specs} == {"msp", "energy", "knn", "lof", "doc", "adb", "da_adb", "k_plus_1_way", "mogb"}


def test_external_method_contracts_do_not_fallback_to_fake_external_methods() -> None:
    assert method_availability("msp").adapter_kind == "native_frozen_minilm_control"
    assert method_availability("adb").state == "blocked"
    assert method_availability("mogb").state == "blocked"
    k_plus_one = method_availability("k_plus_1_way")
    assert k_plus_one.state == "unsupported"
    assert k_plus_one.uses_oos_for_training is False


def test_native_msp_uses_fixed_views_and_known_only_threshold(tmp_path: Path, monkeypatch) -> None:
    paths = _prepared_paths(tmp_path, monkeypatch)
    _write_fake_model(paths)

    run_dir, status = run_external_baseline(paths, _spec("msp"), encoder=_FakeEncoder())

    assert status == "complete"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    threshold = json.loads((run_dir / "threshold_selection.json").read_text(encoding="utf-8"))
    assert manifest["registry_export_contract"]["registry_sha256"]
    assert manifest["uses_oos_for_training"] is False
    assert manifest["uses_oos_for_calibration"] is False
    assert manifest["test_used_for_selection"] is False
    assert metrics["combined"]["oos_f1"] >= 0.0
    assert threshold["type"] == "known_only_conformal"
    assert (run_dir / "predictions" / "test.jsonl").is_file()

    resumed_dir, resumed_status = run_external_baseline(paths, _spec("msp"), encoder=_FakeEncoder())
    assert resumed_dir == run_dir
    assert resumed_status == "complete"


def test_dry_run_binds_native_control_to_s2c_export_manifest(tmp_path: Path, monkeypatch) -> None:
    paths = _prepared_paths(tmp_path, monkeypatch)
    _write_fake_model(paths)

    payload = dry_run(paths, _spec("knn"))

    assert payload["uses_textoir_data"] is False
    assert payload["method_availability"]["export_name"] == "s2c"
    assert payload["registry_export_contract"]["registry_sha256"]


def test_blocked_adapter_writes_auditable_non_metric_manifest(tmp_path: Path, monkeypatch) -> None:
    paths = _prepared_paths(tmp_path, monkeypatch)

    run_dir, status = run_external_baseline(paths, _spec("adb"))

    assert status == "blocked"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    blocked = json.loads((run_dir / "blocked.json").read_text(encoding="utf-8"))
    assert manifest["metrics_emitted"] is False
    assert manifest["registry_export_contract"]["registry_sha256"]
    assert manifest["uses_oos_for_training"] is False
    assert manifest["uses_oos_for_calibration"] is False
    assert blocked["status"] == "blocked"
    assert not (run_dir / "metrics.json").exists()
