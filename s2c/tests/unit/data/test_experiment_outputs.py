"""Tests for lightweight, immutable protocol_v2 run auditing utilities."""

from pathlib import Path

import numpy as np

from s2c.data import source_import
from s2c.data.canonicalize import build_canonical_dataset
from s2c.data.exporters._common import read_jsonl
from s2c.data.hashing import atomic_write_json, atomic_write_jsonl, sha256_json
from s2c.experiments.matrix import GateRunSpec
from s2c.experiments import runner
from s2c.experiments.runner import _config_payload, _run_paths
from s2c.experiments.summarize import summarize
from s2c.experiments.verify import verify

from tests.fixtures.protocol_v2_helpers import make_paths, make_textoir_snapshot


def _spec() -> GateRunSpec:
    return GateRunSpec(
        experiment_name="test_grid",
        dataset="clinc150",
        kir=0.5,
        seed=0,
        k_gate=1,
        distance="euclidean",
        representation="frozen_minilm",
        boundary="mean_std",
        radius_lambda=1.0,
        encoder_name="all-MiniLM-L6-v2",
        encoder_device="cpu",
    )


def test_summary_and_verify_accept_complete_immutable_run(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    spec = _spec()
    run_dir = _run_paths(paths, spec)
    run_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "status": "complete",
            "config_hash": sha256_json(_config_payload(spec)),
            "test_used_for_selection": False,
        },
    )
    atomic_write_json(run_dir / "metrics.json", {"combined": {"oos_f1": 0.5, "id_recall": 0.75}})
    atomic_write_jsonl(run_dir / "predictions" / "test.jsonl", [{"sample_id": "test"}])

    result = verify(paths, [spec], require_complete=True)
    assert result == {"planned": 1, "complete": 1, "missing": 0, "invalid": 0}
    output = tmp_path / "summary.csv"
    summary = summarize(paths, [spec], output)
    assert summary == {"planned": 1, "complete": 1}
    assert "oos_f1" in output.read_text(encoding="utf-8")


def test_matrix_reuses_one_encoder_for_matching_run_specs(tmp_path: Path, monkeypatch) -> None:
    paths = make_paths(tmp_path)
    paths.experiment_admission_path.parent.mkdir(parents=True)
    paths.experiment_admission_path.write_text('{"status": "admitted"}\n', encoding="utf-8")
    first = _spec()
    second = GateRunSpec(**{**first.__dict__, "k_gate": 2})
    loaded: list[tuple[str, str]] = []

    monkeypatch.setattr(runner, "_model_path", lambda *_: tmp_path / "model")
    monkeypatch.setattr(runner, "_model_fingerprint", lambda _: {"name": "model"})
    monkeypatch.setattr(
        runner,
        "_load_encoder",
        lambda _, device: loaded.append(("model", device)) or object(),
    )
    monkeypatch.setattr(
        runner,
        "_canonical_embedding_cache",
        lambda *_: runner.CanonicalEmbeddings(values=np.empty((0, 0)), index_by_sample_id={}, metadata={}),
    )
    monkeypatch.setattr(
        runner,
        "run_gate",
        lambda _paths, spec, **kwargs: _paths.run_root / spec.run_id,
    )

    completed, failed = runner.run_matrix(paths, [first, second], dry=False, resume=True, batch_size=8)
    assert len(completed) == 2
    assert failed == []
    assert loaded == [("model", "cpu")]


def test_canonical_embedding_cache_encodes_each_dataset_once(tmp_path: Path, monkeypatch) -> None:
    paths = make_paths(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, make_textoir_snapshot(tmp_path), ["clinc150"])
    build_canonical_dataset(paths, "clinc150")

    class FakeEncoder:
        calls = 0

        def encode(self, texts, **_kwargs):
            self.calls += 1
            return np.asarray([[float(index), 1.0] for index, _ in enumerate(texts)], dtype=np.float32)

    encoder = FakeEncoder()
    model = {"name": "fake"}
    first = runner._canonical_embedding_cache(paths, "clinc150", model, encoder, batch_size=8)
    second = runner._canonical_embedding_cache(paths, "clinc150", model, encoder, batch_size=8)
    rows = list(read_jsonl(paths.protocol_root / "clinc150" / "records.jsonl"))
    spec = _spec()
    values, metadata = runner._embedding_cache(
        paths,
        spec,
        "train_known",
        rows[:2],
        "registry",
        first.metadata["cache_key"]["canonical_manifest_sha256"],
        model,
        second,
    )
    assert encoder.calls == 1
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    assert values.shape == (2, 2)
    assert metadata["canonical_base_embedding_sha256"] == first.metadata["embedding_sha256"]
