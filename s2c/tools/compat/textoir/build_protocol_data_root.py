#!/usr/bin/env python3
"""Materialize one protocol_v2 export as an isolated TextOIR data root.

The legacy TextOIR runner expects ``<data-root>/<dataset>/{train,dev,test}.tsv``.
This adapter copies the already audited protocol export into an artifact-local
directory, records source/destination hashes, and never reads ``textoir/data``.
It is intentionally a data adapter, not an experiment runner.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--runtime-dataset-name",
        default=None,
        help="Optional directory name expected by the legacy runner (for example banking or oos).",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-version", default="protocol_v2_textoir_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.export_dir.expanduser().resolve()
    destination = args.output_root.expanduser().resolve()
    source_manifest = source / "export_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Missing export_manifest.json: {source_manifest}")
    source_metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source_metadata.get("dataset") != args.dataset:
        raise ValueError(
            f"Export dataset {source_metadata.get('dataset')!r} does not match {args.dataset!r}"
        )
    runtime_dataset_name = args.runtime_dataset_name or args.dataset
    destination_dataset = destination / runtime_dataset_name
    destination_dataset.mkdir(parents=True, exist_ok=True)
    files = []
    for split in ("train", "dev", "test"):
        source_file = source / f"{split}.tsv"
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing source split: {source_file}")
        destination_file = destination_dataset / source_file.name
        shutil.copy2(source_file, destination_file)
        files.append(
            {
                "split": split,
                "source": str(source_file),
                "destination": str(destination_file),
                "source_sha256": sha256_file(source_file),
                "destination_sha256": sha256_file(destination_file),
                "size_bytes": destination_file.stat().st_size,
            }
        )
    if any(item["source_sha256"] != item["destination_sha256"] for item in files):
        raise RuntimeError("Protocol export copy changed at least one split SHA256")

    known_labels = source / "known_labels.json"
    if not known_labels.is_file():
        raise FileNotFoundError(f"Missing known_labels.json: {known_labels}")
    destination_known = destination_dataset / known_labels.name
    shutil.copy2(known_labels, destination_known)
    known_hash = {
        "source": str(known_labels),
        "destination": str(destination_known),
        "source_sha256": sha256_file(known_labels),
        "destination_sha256": sha256_file(destination_known),
    }
    if known_hash["source_sha256"] != known_hash["destination_sha256"]:
        raise RuntimeError("Protocol known-label copy changed SHA256")

    payload = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_version": args.protocol_version,
        "dataset": args.dataset,
        "runtime_dataset_name": runtime_dataset_name,
        "source_export_dir": str(source),
        "source_export_manifest_sha256": sha256_file(source_manifest),
        "source_export_metadata": source_metadata,
        "destination_root": str(destination),
        "files": files,
        "known_labels": known_hash,
    }
    write_json(destination / "PROTOCOL_DATA_ROOT_MANIFEST.json", payload)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
