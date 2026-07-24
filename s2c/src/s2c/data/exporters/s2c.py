"""Export protocol_v2 views in the legacy-shaped JSON format used by s2c Gate tools."""

from __future__ import annotations

from typing import Any

from s2c.runtime.paths import ProtocolV2Paths

from ..hashing import atomic_write_json
from ..manifests import export_manifest_path, write_manifest
from ._common import export_base_manifest, export_directory, load_export_inputs, output_file_info


def _gate_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    is_oos = split == "test" and str(row.get("evaluation_label")) == "oos"
    return {
        "sample_id": row["sample_id"],
        "text": row["text"],
        "intent": row["intent"],
        "label": int(is_oos),
        "is_oos": is_oos,
        "oos_source": row.get("oos_source", "known"),
        "source_split": row["original_split"],
        "dataset": row["dataset"],
    }


def export_s2c(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> dict[str, Any]:
    registry, dataset_manifest, views = load_export_inputs(paths, dataset, seed, kir)
    root = export_directory(paths, "s2c", dataset, seed, kir)
    gate = root / "gate"
    mapping = {"train": views["train_known"], "val": views["calibration_known"], "test": views["test_combined"]}
    files: list[dict[str, object]] = []
    sample_ids: dict[str, list[str]] = {}
    for split, rows in mapping.items():
        path = gate / f"{split}.json"
        payload = [_gate_row(row, split) for row in rows]
        atomic_write_json(path, payload)
        files.append(output_file_info(root, path, len(payload)))
        sample_ids[split] = [str(row["sample_id"]) for row in rows]
    atomic_write_json(root / "sample_ids.json", sample_ids)
    files.append(output_file_info(root, root / "sample_ids.json"))
    manifest = export_base_manifest(
        paths, "s2c", dataset, seed, kir, registry, dataset_manifest, root, files, sample_ids,
        "test_combined: evaluation_label=oos maps to label=1; known maps to label=0",
    )
    atomic_write_json(root / "export_manifest.json", manifest)
    write_manifest(export_manifest_path(paths.manifest_root, "s2c", dataset, seed, kir), manifest)
    return manifest

