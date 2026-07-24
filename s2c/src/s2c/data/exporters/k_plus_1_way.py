"""Export the fixed known-label plus one OOS-class view without synthetic OOS training data."""

from __future__ import annotations

from typing import Any

from s2c.runtime.paths import ProtocolV2Paths

from ..hashing import atomic_write_json
from ..manifests import export_manifest_path, write_manifest
from ._common import export_base_manifest, export_directory, load_export_inputs, output_file_info


def _row(row: dict[str, Any], *, is_test: bool) -> dict[str, Any]:
    label = "__oos__" if is_test and row.get("evaluation_label") == "oos" else str(row["intent"])
    return {"sample_id": row["sample_id"], "text": row["text"], "label": label, "original_intent": row["intent"]}


def export_k_plus_1_way(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> dict[str, Any]:
    registry, dataset_manifest, views = load_export_inputs(paths, dataset, seed, kir)
    root = export_directory(paths, "k_plus_1_way", dataset, seed, kir)
    mapping = {"train": (views["train_known"], False), "dev": (views["calibration_known"], False), "test": (views["test_combined"], True)}
    files: list[dict[str, object]] = []
    sample_ids: dict[str, list[str]] = {}
    for split, (rows, is_test) in mapping.items():
        payload = [_row(row, is_test=is_test) for row in rows]
        path = root / f"{split}.json"
        atomic_write_json(path, payload)
        files.append(output_file_info(root, path, len(payload)))
        sample_ids[split] = [str(row["sample_id"]) for row in rows]
    labels = list(registry["known_intents"]) + ["__oos__"]
    atomic_write_json(root / "known_labels.json", list(registry["known_intents"]))
    atomic_write_json(root / "label_map.json", {label: index for index, label in enumerate(labels)})
    atomic_write_json(root / "sample_ids.json", sample_ids)
    for name in ("known_labels.json", "label_map.json", "sample_ids.json"):
        files.append(output_file_info(root, root / name))
    manifest = export_base_manifest(
        paths, "k_plus_1_way", dataset, seed, kir, registry, dataset_manifest, root, files, sample_ids,
        "train/dev contain Known classes only; test maps held-out and native OOS to reserved __oos__ class",
    )
    atomic_write_json(root / "export_manifest.json", manifest)
    write_manifest(export_manifest_path(paths.manifest_root, "k_plus_1_way", dataset, seed, kir), manifest)
    return manifest

