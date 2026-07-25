"""Regression tests for the lightweight protocol implementation report."""

from __future__ import annotations

import csv
from dataclasses import replace
from io import StringIO
from pathlib import Path

import json

import tools.audit.generate_protocol_v2_implementation_report as report_module
from tools.audit.generate_protocol_v2_implementation_report import (
    _manifest_datasets,
    _remaining_gate_text,
    _source_rows,
    generate,
)

from tests.fixtures.protocol_v2_helpers import make_paths


def test_report_uses_only_materialized_version_datasets(tmp_path: Path) -> None:
    """A blocked schema dataset must not leak into an official-version report."""
    paths = replace(make_paths(tmp_path), dataset_version="protocol_v2_official_v1")
    for dataset in ("clinc150", "banking77"):
        root = paths.manifest_root / dataset
        root.mkdir(parents=True)
        (root / "SOURCE_MANIFEST.json").write_text("{}", encoding="utf-8")
        (root / "DATASET_MANIFEST.json").write_text("{}", encoding="utf-8")
    # StackOverflow is present in DATASET_SPECS but lacks official manifests.
    assert _manifest_datasets(paths) == ("banking77", "clinc150")


def test_source_report_keeps_license_entry_without_corpus_counts(tmp_path: Path) -> None:
    """Official provenance must retain its license hash even without row counts."""
    paths = make_paths(tmp_path)
    root = paths.manifest_root / "clinc150"
    root.mkdir(parents=True)
    (root / "SOURCE_MANIFEST.json").write_text(
        '{"source_name":"official","source_commit":"fixed",'
        '"source_relative_directory":"sources/official/fixed",'
        '"files":[{"relative_path":"LICENSE","role":"license","sha256":"abc",'
        '"byte_identical":true}]}',
        encoding="utf-8",
    )
    rows = _source_rows(paths, ("clinc150",))
    assert rows[0]["relative_path"] == "LICENSE"
    assert rows[0]["row_count"] == ""
    assert rows[0]["sha256"] == "abc"


def test_report_manifest_excludes_unmaterialized_schema_datasets(tmp_path: Path) -> None:
    """The audit manifest is scoped to evidence, not every known schema name."""
    paths = replace(make_paths(tmp_path), dataset_version="protocol_v2_official_v1")
    root = paths.manifest_root / "clinc150"
    root.mkdir(parents=True)
    (root / "SOURCE_MANIFEST.json").write_text(
        '{"source_name":"official","source_commit":"fixed",'
        '"source_relative_directory":"sources/official/fixed",'
        '"files":[{"relative_path":"records.json","role":"records",'
        '"split":"raw","row_count":1,"label_count":1,"size_bytes":1,'
        '"sha256":"abc","byte_identical":true}]}',
        encoding="utf-8",
    )
    (root / "DATASET_MANIFEST.json").write_text(
        '{"sample_count":1,"known_label_count":1,"native_oos_count":0,'
        '"exact_duplicate_count":0,"normalized_duplicate_count":0,'
        '"canonical_file_sha256":"def","source_manifest_sha256":"abc",'
        '"split_counts":{"train":1}}',
        encoding="utf-8",
    )
    output = tmp_path / "audit"
    generate(
        paths,
        output,
        test_summary="unit",
        runtime_status="unit",
        executed_commands=("pytest -q tests/unit/data",),
        moved_files=("results/old=>docs/archive/old",),
    )
    manifest = json.loads((output / "audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["materialized_datasets"] == ["clinc150"]
    assert set(manifest["inputs"]["source_manifest_sha256"]) == {"clinc150"}
    assert manifest["embedding_generation"] is False
    assert manifest["executed_commands"] == ["pytest -q tests/unit/data"]
    assert manifest["moved_files"] == ["results/old=>docs/archive/old"]
    blocker = (output / "blocker_report.md").read_text(encoding="utf-8")
    assert "active-protocol constraint report" in blocker
    assert "StackOverflow" in blocker
    requirement_matrix = list(
        csv.DictReader(StringIO((output / "requirement_matrix.csv").read_text(encoding="utf-8")))
    )
    # The active contract has no model evidence in this fixture, so E1 remains
    # in progress even though the source/canonical inventory is materialized.
    requirement_statuses = {row["requirement"]: row["status"] for row in requirement_matrix}
    assert requirement_statuses["E1 three-dataset 36-cell Gate smoke"] == "in_progress"
    assert requirement_statuses["E2 three-dataset 1,650-cell Gate grid"] == "in_progress"


def test_remaining_gate_records_local_only_stackoverflow_without_blocking() -> None:
    """The active snapshot permits local StackOverflow experiments, not release."""
    text = _remaining_gate_text(
        {
            "dataset_admission": {
                "clinc150": "admitted",
                "banking77": "admitted",
                "stackoverflow": "admitted_benchmark_local_only",
            }
        },
        ("banking77", "clinc150"),
        completed_runs=24,
    )

    assert "24 completed E1 smoke Gate run(s)" in text
    assert "StackOverflow" in text
    assert "must not be tracked in public Git" in text


def test_report_cli_defaults_to_active_textoir_version(monkeypatch, tmp_path: Path) -> None:
    """A bare audit command must use the sole active local benchmark version."""
    captured: dict[str, object] = {}
    discovered = make_paths(tmp_path)

    monkeypatch.setattr(
        report_module.ProtocolV2Paths,
        "discover",
        classmethod(lambda cls: discovered),
    )
    monkeypatch.setattr(
        report_module,
        "generate",
        lambda paths, output, test_summary, runtime_status, **kwargs: captured.update(
            {"paths": paths, "output": output, "test_summary": test_summary, "runtime_status": runtime_status}
        ),
    )

    assert report_module.main(["--output", str(tmp_path / "audit")]) == 0
    assert captured["paths"].dataset_version == "protocol_v2_textoir_v1"
