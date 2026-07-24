"""Export a fixed protocol_v2 split in TEXTOIR TSV format without re-sampling labels."""

from __future__ import annotations

import csv
import io
from typing import Any

from s2c.runtime.paths import ProtocolV2Paths

from ..hashing import atomic_write_json, atomic_write_text
from ..manifests import export_manifest_path, write_manifest
from ._common import export_base_manifest, export_directory, load_export_inputs, output_file_info


def _tsv(rows: list[dict[str, Any]], *, oos_test: bool) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["text", "label"], delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        label = "oos" if oos_test and row.get("evaluation_label") == "oos" else str(row["intent"])
        writer.writerow({"text": row["text"], "label": label})
    return buffer.getvalue()


def export_textoir(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> dict[str, Any]:
    registry, dataset_manifest, views = load_export_inputs(paths, dataset, seed, kir)
    root = export_directory(paths, "textoir", dataset, seed, kir)
    mapping = {"train": (views["train_known"], False), "dev": (views["calibration_known"], False), "test": (views["test_combined"], True)}
    files: list[dict[str, object]] = []
    sample_ids: dict[str, list[str]] = {}
    for split, (rows, oos_test) in mapping.items():
        path = root / f"{split}.tsv"
        atomic_write_text(path, _tsv(rows, oos_test=oos_test))
        files.append(output_file_info(root, path, len(rows)))
        sample_ids[split] = [str(row["sample_id"]) for row in rows]
    label_map = {label: index for index, label in enumerate(registry["known_intents"])}
    atomic_write_json(root / "known_labels.json", list(registry["known_intents"]))
    atomic_write_json(root / "label_map.json", label_map)
    atomic_write_json(root / "sample_ids.json", sample_ids)
    for name in ("known_labels.json", "label_map.json", "sample_ids.json"):
        files.append(output_file_info(root, root / name))
    manifest = export_base_manifest(
        paths, "textoir", dataset, seed, kir, registry, dataset_manifest, root, files, sample_ids,
        "test_combined: all held-out and native OOS rows use literal label 'oos'",
    )
    atomic_write_json(root / "export_manifest.json", manifest)
    write_manifest(export_manifest_path(paths.manifest_root, "textoir", dataset, seed, kir), manifest)
    return manifest

