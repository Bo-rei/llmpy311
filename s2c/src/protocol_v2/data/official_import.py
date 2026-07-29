"""Import verified official raw files without depending on them at runtime.

This module intentionally has a narrower job than :mod:`source_import`: it
copies the raw files selected by the three-way provenance decision.  It does
not convert their format, derive a development split, or contact a network.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protocol_v2.runtime.paths import ProtocolV2Paths

from .hashing import sha256_file
from .manifests import source_manifest_path, write_manifest


@dataclass(frozen=True)
class OfficialSource:
    """官方上游的不可变身份与重建数据所需的最小文件集合。"""

    dataset: str
    source_id: str
    repository: str
    revision: str
    source_format: str
    license_spdx: str
    files: tuple[tuple[str, str, str], ...]
    # (source-relative path, manifest role, raw split).  Metadata files use
    # ``role='metadata'`` and are kept for provenance but never parsed as rows.


OFFICIAL_SOURCES: dict[str, OfficialSource] = {
    "clinc150": OfficialSource(
        dataset="clinc150",
        source_id="clinc-oos-eval-828f8093932c8fe6ca7936c3d2e52903b1c523de",
        repository="https://github.com/clinc/oos-eval.git",
        revision="828f8093932c8fe6ca7936c3d2e52903b1c523de",
        source_format="clinc_data_full_json_v1",
        license_spdx="CC-BY-3.0",
        files=(
            ("data/data_full.json", "records", "raw"),
            ("LICENSE", "license", "metadata"),
        ),
    ),
    "banking77": OfficialSource(
        dataset="banking77",
        source_id="polyai-banking77-57ec275d8078af65b7731c2a98be812d844a6d6b",
        repository="https://github.com/PolyAI-LDN/task-specific-datasets.git",
        revision="57ec275d8078af65b7731c2a98be812d844a6d6b",
        source_format="banking77_csv_v1",
        license_spdx="CC-BY-4.0",
        files=(
            ("banking_data/train.csv", "records", "train"),
            ("banking_data/test.csv", "records", "test"),
            ("banking_data/categories.json", "metadata", "metadata"),
            ("LICENSE", "license", "metadata"),
        ),
    ),
}


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _copy_regular_file(source: Path, destination: Path) -> None:
    """逐字节复制，拒绝链接和同 inode，避免运行时暗中依赖外部 checkout。"""
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"Official source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if destination.is_symlink() or os.path.samefile(source, destination):
        raise RuntimeError(f"Official import created a forbidden link: {destination}")
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"Byte-identical copy check failed: {source}")


def _csv_counts(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"Unexpected Banking77 CSV header in {path}: {reader.fieldnames!r}")
        rows = list(reader)
    return len(rows), len({str(row["category"]) for row in rows})


def _clinc_counts(path: Path) -> tuple[int, int, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}
    if set(payload) != expected:
        raise ValueError(f"Unexpected CLINC data_full.json keys in {path}: {sorted(payload)}")
    split_counts = {name: len(rows) for name, rows in payload.items()}
    labels = {str(row[1]) for rows in payload.values() for row in rows if str(row[1]) != "oos"}
    return sum(split_counts.values()), len(labels), dict(sorted(split_counts.items()))


def _file_info(source: OfficialSource, target: Path, relative: str, role: str, split: str) -> dict[str, object]:
    copied = target / relative
    payload: dict[str, object] = {
        "relative_path": relative,
        "role": role,
        "split": split,
        "sha256": sha256_file(copied),
        "size_bytes": copied.stat().st_size,
        "byte_identical": True,
    }
    if source.dataset == "clinc150" and relative == "data/data_full.json":
        count, label_count, split_counts = _clinc_counts(copied)
        payload.update({"row_count": count, "label_count": label_count, "split_counts": split_counts})
    elif source.dataset == "banking77" and relative.endswith(".csv"):
        count, label_count = _csv_counts(copied)
        payload.update({"row_count": count, "label_count": label_count})
    return payload


def _source_manifest(paths: ProtocolV2Paths, source: OfficialSource, target: Path) -> dict[str, Any]:
    files = [_file_info(source, target, *item) for item in source.files]
    return {
        "schema_version": "protocol_v2.source_manifest.v2",
        "protocol_family": "protocol_v2",
        "protocol_version": paths.dataset_version,
        "dataset": source.dataset,
        "source_name": "official",
        "source_repository": source.repository,
        "source_commit": source.revision,
        "source_format": source.source_format,
        "source_relative_directory": f"sources/official/{source.source_id}/{source.dataset}",
        "license_status": "verified",
        "license_spdx": source.license_spdx,
        "license_file_sha256": next(item["sha256"] for item in files if item["role"] == "license"),
        "files": files,
        # Banking77 has no official validation split.  This declares the
        # independent derivation; it does not claim equivalence to TextOIR dev.
        "calibration_derivation": (
            {
                "algorithm": "stratified_sha256_rank_v1",
                "source_split": "train",
                "target_count": 1000,
                "salt": "banking77_official_calibration_v1",
            }
            if source.dataset == "banking77"
            else None
        ),
    }


def import_official_dataset(paths: ProtocolV2Paths, dataset: str, source_root: Path) -> dict[str, Any]:
    """Import one audited source with an explicit local checkout argument."""
    try:
        source = OFFICIAL_SOURCES[dataset]
    except KeyError as exc:
        raise ValueError(f"No verified official source is defined for {dataset!r}") from exc
    root = source_root.expanduser().resolve()
    if _git_commit(root) != source.revision:
        raise ValueError(f"Official checkout revision mismatch for {dataset}: expected {source.revision}")
    target = paths.data_root / "sources" / "official" / source.source_id / dataset
    expected = [(relative, root / relative) for relative, _, _ in source.files]
    if target.exists():
        for relative, original in expected:
            copied = target / relative
            if not copied.is_file() or copied.is_symlink() or os.path.samefile(original, copied):
                raise RuntimeError(f"Existing official snapshot is invalid: {copied}")
            if sha256_file(original) != sha256_file(copied):
                raise RuntimeError(f"Existing official snapshot differs from source: {copied}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{dataset}.", dir=target.parent))
        try:
            for relative, original in expected:
                _copy_regular_file(original, temporary / relative)
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    manifest = _source_manifest(paths, source, target)
    write_manifest(source_manifest_path(paths.manifest_root, dataset), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(OFFICIAL_SOURCES), action="append", required=True)
    parser.add_argument("--clinc-root", type=Path)
    parser.add_argument("--banking-root", type=Path)
    args = parser.parse_args(argv)
    roots = {"clinc150": args.clinc_root, "banking77": args.banking_root}
    paths = ProtocolV2Paths.discover()
    for dataset in args.dataset:
        root = roots[dataset]
        if root is None:
            parser.error(f"--{dataset.removesuffix('150') if dataset == 'clinc150' else 'banking'}-root is required for {dataset}")
        manifest = import_official_dataset(paths, dataset, root)
        print(f"imported {dataset}: {manifest['source_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
