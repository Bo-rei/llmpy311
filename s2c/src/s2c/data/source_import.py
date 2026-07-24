"""Byte-identical import of the fixed local TEXTOIR data snapshot."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from s2c.runtime.paths import ProtocolV2Paths

from .hashing import sha256_file
from .manifests import source_manifest_path, write_manifest
from .schema import DATASET_SPECS, get_dataset_spec


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _count_tsv(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["text", "label"]:
            raise ValueError(f"Expected text/label TSV header in {path}, got {reader.fieldnames!r}")
        rows = list(reader)
    return len(rows), len({row["label"] for row in rows})


def _source_files(textoir_root: Path, dataset: str) -> list[tuple[str, Path]]:
    spec = get_dataset_spec(dataset)
    directory = textoir_root / "data" / spec.textoir_directory
    files = [(f"{split}.tsv", directory / f"{split}.tsv") for split in spec.source_splits]
    missing = [path for _, path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing TEXTOIR {dataset} source file(s): {missing}")
    return files


def _verify_snapshot(target: Path, files: Iterable[tuple[str, Path]]) -> None:
    for relative, source in files:
        copied = target / relative
        if not copied.is_file() or copied.is_symlink():
            raise RuntimeError(f"Existing source snapshot is not a regular file: {copied}")
        if os.path.samefile(source, copied) or sha256_file(source) != sha256_file(copied):
            raise RuntimeError(f"Existing source snapshot differs from TEXTOIR source: {copied}")


def import_dataset(paths: ProtocolV2Paths, textoir_root: Path, dataset: str) -> dict[str, Any]:
    """Copy one dataset without text edits, hard links or symlinks."""
    spec = get_dataset_spec(dataset)
    commit = _git_commit(textoir_root)
    files = _source_files(textoir_root, dataset)
    target = paths.data_root / "sources" / "textoir" / commit / dataset
    if target.exists():
        _verify_snapshot(target, files)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{dataset}.", dir=target.parent))
        try:
            for relative, source in files:
                copied = temporary / relative
                shutil.copyfile(source, copied)
                if copied.is_symlink() or os.path.samefile(source, copied):
                    raise RuntimeError(f"Source import created a forbidden link: {copied}")
                if sha256_file(source) != sha256_file(copied):
                    raise RuntimeError(f"Byte-identical copy check failed for {dataset}/{relative}")
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    manifest_files: list[dict[str, object]] = []
    for relative, source in files:
        copied = target / relative
        row_count, label_count = _count_tsv(copied)
        manifest_files.append(
            {
                "relative_path": relative,
                "split": relative.removesuffix(".tsv"),
                "sha256": sha256_file(copied),
                "size_bytes": copied.stat().st_size,
                "row_count": row_count,
                "label_count": label_count,
                "encoding": "utf-8",
                "byte_identical": sha256_file(source) == sha256_file(copied),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "protocol_v2.source_manifest.v1",
        "protocol_version": paths.dataset_version,
        "dataset": dataset,
        "source_name": "textoir",
        "source_repository": "https://github.com/thuiar/TEXTOIR.git",
        "source_commit": commit,
        "original_directory": f"textoir/data/{spec.textoir_directory}",
        "imported_at": datetime.now(UTC).isoformat(),
        "import_command": "python -m s2c.data.import_textoir",
        "source_relative_directory": f"sources/textoir/{commit}/{dataset}",
        "license_provenance_note": (
            "Local data import only; see docs/audits/data_provenance for source and tracking policy."
        ),
        "files": manifest_files,
    }
    write_manifest(source_manifest_path(paths.manifest_root, dataset), manifest)
    return manifest


def import_textoir_snapshot(
    paths: ProtocolV2Paths,
    textoir_root: Path | None = None,
    datasets: Iterable[str] = DATASET_SPECS,
) -> dict[str, dict[str, Any]]:
    """Import all requested datasets from the only permitted external data source."""
    source = paths.require_textoir_import_root(textoir_root)
    return {dataset: import_dataset(paths, source, dataset) for dataset in datasets}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textoir-root", type=Path, help="Import-only TEXTOIR repository override.")
    parser.add_argument("--dataset", action="append", choices=sorted(DATASET_SPECS))
    args = parser.parse_args(argv)
    manifests = import_textoir_snapshot(
        ProtocolV2Paths.discover(), args.textoir_root, args.dataset or DATASET_SPECS.keys()
    )
    for dataset, manifest in manifests.items():
        print(f"imported {dataset}: {manifest['source_commit']} ({len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
