"""MOGB TSV export sourced only from protocol_v2 views and registries."""

from __future__ import annotations

from typing import Any

from protocol_v2.runtime.paths import ProtocolV2Paths

def export_mogb(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> dict[str, Any]:
    """Create the declared TSV contract; this is not a claim of MOGB reproduction."""
    # MOGB and TEXTOIR use the same two-column, fixed-split interchange here.
    # Copying is deliberately avoided: invoke the shared source-view logic, then
    # materialize under its own export root in a small compatibility wrapper.
    from ._common import export_base_manifest, export_directory, load_export_inputs, output_file_info
    from ..hashing import atomic_write_json, atomic_write_text
    from ..manifests import export_manifest_path, write_manifest
    from .textoir import _tsv

    registry, dataset_manifest, views = load_export_inputs(paths, dataset, seed, kir)
    root = export_directory(paths, "mogb", dataset, seed, kir)
    mapping = {
        "train": (views["train_known"], False),
        "dev": (views["calibration_known"], False),
        "test": (views["test_combined"], True),
    }
    files: list[dict[str, object]] = []
    sample_ids: dict[str, list[str]] = {}
    for split, (rows, oos_test) in mapping.items():
        path = root / f"{split}.tsv"
        atomic_write_text(path, _tsv(rows, oos_test=oos_test))
        files.append(output_file_info(root, path, len(rows)))
        sample_ids[split] = [str(row["sample_id"]) for row in rows]
    atomic_write_json(root / "known_labels.json", list(registry["known_intents"]))
    atomic_write_json(root / "label_map.json", {label: index for index, label in enumerate(registry["known_intents"])})
    atomic_write_json(root / "sample_ids.json", sample_ids)
    for name in ("known_labels.json", "label_map.json", "sample_ids.json"):
        files.append(output_file_info(root, root / name))
    manifest = export_base_manifest(
        paths, "mogb", dataset, seed, kir, registry, dataset_manifest, root, files, sample_ids,
        "test_combined: all held-out and native OOS rows use literal label 'oos'; no MOGB data transformation applied",
    )
    atomic_write_json(root / "export_manifest.json", manifest)
    write_manifest(export_manifest_path(paths.manifest_root, "mogb", dataset, seed, kir), manifest)
    return manifest
