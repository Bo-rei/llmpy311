"""Regression coverage for the audited CLINC/Banking official source path."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from s2c.data.canonicalize import build_canonical_dataset
from s2c.data.official_import import import_official_dataset
from s2c.data.registry import build_registry
from s2c.data.views import build_views
from s2c.runtime.paths import ProtocolV2Paths

from tests.fixtures.protocol_v2_helpers import make_paths


def _official_paths(tmp_path: Path) -> ProtocolV2Paths:
    base = make_paths(tmp_path)
    return ProtocolV2Paths(
        project_root=base.project_root,
        data_root=base.data_root,
        artifacts_root=base.artifacts_root,
        results_root=base.results_root,
        legacy_root=base.legacy_root,
        textoir_import_root=None,
        dataset_version="protocol_v2_official_test",
    )


def _write_clinc_checkout(root: Path) -> None:
    values = {
        "train": [["alpha train", "alpha"], ["beta train", "beta"]],
        "val": [["alpha val", "alpha"], ["beta val", "beta"]],
        "test": [["alpha test", "alpha"], ["beta test", "beta"]],
        "oos_train": [["oos train", "oos"]],
        "oos_val": [["oos val", "oos"]],
        "oos_test": [["oos test", "oos"]],
    }
    (root / "data").mkdir(parents=True)
    (root / "data" / "data_full.json").write_text(json.dumps(values), encoding="utf-8")
    (root / "LICENSE").write_text("license", encoding="utf-8")


def _write_banking_snapshot(paths: ProtocolV2Paths) -> None:
    root = paths.data_root / "sources" / "official" / "fixture" / "banking77"
    root.mkdir(parents=True)
    with (root / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "category"])
        writer.writeheader()
        for label in ("alpha", "beta"):
            for index in range(4):
                writer.writerow({"text": f"{label} train {index}", "category": label})
    with (root / "test.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerow({"text": "alpha test", "category": "alpha"})
        writer.writerow({"text": "beta test", "category": "beta"})
    (root / "LICENSE").write_text("license", encoding="utf-8")
    manifest_root = paths.manifest_root / "banking77"
    manifest_root.mkdir(parents=True)
    (manifest_root / "SOURCE_MANIFEST.json").write_text(
        json.dumps(
            {
                "source_name": "official",
                "source_commit": "fixture",
                "source_format": "banking77_csv_v1",
                "source_relative_directory": "sources/official/fixture/banking77",
                "calibration_derivation": {
                    "algorithm": "stratified_sha256_rank_v1",
                    "source_split": "train",
                    "target_count": 2,
                    "salt": "fixture",
                },
                "files": [
                    {"relative_path": "train.csv", "role": "records", "split": "train"},
                    {"relative_path": "test.csv", "role": "records", "split": "test"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_official_clinc_keeps_raw_splits_and_native_oos(tmp_path: Path, monkeypatch) -> None:
    paths = _official_paths(tmp_path)
    checkout = tmp_path / "clinc"
    _write_clinc_checkout(checkout)
    monkeypatch.setattr("s2c.data.official_import._git_commit", lambda _: "828f8093932c8fe6ca7936c3d2e52903b1c523de")

    import_official_dataset(paths, "clinc150", checkout)
    manifest = build_canonical_dataset(paths, "clinc150")
    assert manifest["sample_count"] == 9
    assert manifest["split_counts"] == {
        "oos_test": 1,
        "oos_train": 1,
        "oos_val": 1,
        "test": 2,
        "train": 2,
        "val": 2,
    }
    assert manifest["native_oos_count"] == 3
    registry = build_registry(paths, "clinc150", seed=0, kir=0.5)
    view_manifest = build_views(paths, "clinc150", seed=0, kir=0.5)
    counts = {item["name"]: item["count"] for item in view_manifest["files"]}
    assert registry["known_count"] == 1
    assert counts["test_native_oos"] == 1


def test_banking_calibration_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    paths = _official_paths(tmp_path)
    _write_banking_snapshot(paths)
    first = build_canonical_dataset(paths, "banking77")
    second = build_canonical_dataset(paths, "banking77")
    assert first["canonical_file_sha256"] == second["canonical_file_sha256"]
    registry = build_registry(paths, "banking77", seed=0, kir=0.5)
    views = build_views(paths, "banking77", seed=0, kir=0.5)
    counts = {item["name"]: item["count"] for item in views["files"]}
    assert registry["known_count"] == 1
    assert counts["calibration_known"] == 1
    assert counts["train_known"] == 3
