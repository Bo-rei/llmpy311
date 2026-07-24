"""Shared read-only view loading and manifest helpers for all exporters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from s2c.runtime.paths import ProtocolV2Paths

from ..hashing import sha256_file, sha256_json
from ..manifests import dataset_manifest_path, read_json
from ..registry import registry_path
from ..schema import format_kir
from ..views import view_directory


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                yield value


def load_export_inputs(
    paths: ProtocolV2Paths, dataset: str, seed: int, kir: float
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    registry = read_json(registry_path(paths, dataset, seed, kir))
    view_root = view_directory(paths, dataset, seed, kir)
    views = {
        name: list(read_jsonl(view_root / f"{name}.jsonl"))
        for name in ("train_known", "calibration_known", "test_combined")
    }
    if not all((view_root / f"{name}.jsonl").is_file() for name in views):
        raise FileNotFoundError(f"Required views missing for dataset={dataset}, KIR={kir}, seed={seed}: {view_root}")
    dataset_manifest = read_json(dataset_manifest_path(paths.manifest_root, dataset))
    return registry, dataset_manifest, views


def export_directory(paths: ProtocolV2Paths, export_name: str, dataset: str, seed: int, kir: float) -> Path:
    return paths.export_root / export_name / dataset / f"seed_{seed}" / f"kir_{format_kir(kir)}"


def output_file_info(root: Path, path: Path, count: int | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if count is not None:
        row["count"] = count
    return row


def export_base_manifest(
    paths: ProtocolV2Paths,
    export_name: str,
    dataset: str,
    seed: int,
    kir: float,
    registry: dict[str, Any],
    dataset_manifest: dict[str, Any],
    output_root: Path,
    files: list[dict[str, object]],
    sample_ids: dict[str, list[str]],
    oos_mapping: str,
) -> dict[str, Any]:
    canonical_mapping = {
        "train_known": sample_ids["train"],
        "calibration_known": sample_ids.get("val", sample_ids.get("dev", [])),
        "test_combined": sample_ids["test"],
    }
    return {
        "schema_version": "protocol_v2.exports.v1",
        "protocol_version": paths.dataset_version,
        "export_name": export_name,
        "dataset": dataset,
        "seed": seed,
        "kir": kir,
        "registry_sha256": registry["registry_sha256"],
        "canonical_manifest_sha256": sha256_file(dataset_manifest_path(paths.manifest_root, dataset)),
        "output_relative_directory": output_root.relative_to(paths.data_root).as_posix(),
        "files": files,
        # Exporters may call the calibration split ``val`` or ``dev``. Hash a
        # protocol-level mapping so equality proves they consumed the same
        # canonical samples rather than merely sharing a file naming convention.
        "canonical_sample_id_mapping_sha256": sha256_json(canonical_mapping),
        "sample_id_map_file": "sample_ids.json",
        "oos_mapping_rule": oos_mapping,
    }
