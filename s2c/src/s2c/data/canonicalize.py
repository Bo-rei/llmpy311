"""Build immutable canonical records from an explicitly declared raw source."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from s2c.runtime.paths import ProtocolV2Paths

from .hashing import atomic_write_jsonl, sha256_file, sha256_json, sha256_text
from .manifests import calibration_derivation_path, dataset_manifest_path, read_json, source_manifest_path, write_manifest
from .schema import CANONICAL_SCHEMA_VERSION, DATASET_SPECS, get_dataset_spec


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _make_record(
    *,
    paths: ProtocolV2Paths,
    dataset: str,
    source_manifest: dict[str, Any],
    relative_path: str,
    source_row: int,
    original_split: str,
    view_role: str,
    text: str,
    intent: str,
) -> dict[str, object]:
    """生成以 provenance 为先的记录，绝不改写原始文本或标签。"""
    spec = get_dataset_spec(dataset)
    native_oos = spec.native_oos_label is not None and intent == spec.native_oos_label
    source_commit = str(source_manifest["source_commit"])
    sample_key = "|".join(
        (
            paths.dataset_version,
            dataset,
            source_commit,
            relative_path,
            original_split,
            str(source_row),
            text,
            intent,
        )
    )
    record: dict[str, object] = {
        "sample_id": sha256_text(sample_key),
        "dataset": dataset,
        "text": text,
        "intent": intent,
        # original_split is always the upstream spelling (for example CLINC
        # ``val`` and ``oos_test``); view_role is a separate protocol mapping.
        "original_split": original_split,
        "view_role": view_role,
        "native_oos": native_oos,
        "source_name": str(source_manifest["source_name"]),
        "source_commit": source_commit,
        "source_relative_path": str(source_manifest["source_relative_directory"]) + "/" + relative_path,
        "source_row": source_row,
        "text_sha256": sha256_text(text),
    }
    record["record_sha256"] = sha256_json(record)
    return record


def _textoir_records(
    paths: ProtocolV2Paths, dataset: str, source_manifest: dict[str, Any]
) -> list[dict[str, object]]:
    """Compatibility parser for the already-frozen TEXTOIR candidate only."""
    source_root = paths.data_root / str(source_manifest["source_relative_directory"])
    rows: list[dict[str, object]] = []
    for file_info in source_manifest["files"]:
        if not isinstance(file_info, dict):
            raise ValueError(f"Malformed source manifest for {dataset}")
        relative = str(file_info["relative_path"])
        path = source_root / relative
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["text", "label"]:
                raise ValueError(f"Unexpected source header for {dataset}: {path}")
            split = str(file_info["split"])
            view_role = {"train": "train", "dev": "calibration", "test": "test"}.get(split)
            if view_role is None:
                raise ValueError(f"Unsupported TEXTOIR split in {path}: {split}")
            for row_index, source_row in enumerate(reader, start=1):
                rows.append(
                    _make_record(
                        paths=paths,
                        dataset=dataset,
                        source_manifest=source_manifest,
                        relative_path=relative,
                        source_row=row_index,
                        original_split=split,
                        view_role=view_role,
                        text=str(source_row["text"]),
                        intent=str(source_row["label"]),
                    )
                )
    return rows


def _official_clinc_records(
    paths: ProtocolV2Paths, dataset: str, source_manifest: dict[str, Any]
) -> list[dict[str, object]]:
    source_root = paths.data_root / str(source_manifest["source_relative_directory"])
    info = next(item for item in source_manifest["files"] if item.get("role") == "records")
    relative = str(info["relative_path"])
    payload = json.loads((source_root / relative).read_text(encoding="utf-8"))
    roles = {
        "train": "train",
        "val": "calibration",
        "test": "test",
        # Native OOS train/validation rows are preserved in canonical data but
        # deliberately excluded from all Known-only views and model selection.
        "oos_train": "excluded_native_train",
        "oos_val": "excluded_native_calibration",
        "oos_test": "test",
    }
    expected = set(roles)
    if set(payload) != expected:
        raise ValueError(f"Unexpected CLINC split keys: {sorted(payload)}")
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test", "oos_train", "oos_val", "oos_test"):
        values = payload[split]
        if not isinstance(values, list):
            raise ValueError(f"CLINC split is not a list: {split}")
        for row_index, value in enumerate(values, start=1):
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"Malformed CLINC row at {split}[{row_index}]")
            rows.append(
                _make_record(
                    paths=paths,
                    dataset=dataset,
                    source_manifest=source_manifest,
                    relative_path=relative,
                    source_row=row_index,
                    original_split=split,
                    view_role=roles[split],
                    text=str(value[0]),
                    intent=str(value[1]),
                )
            )
    return rows


def _official_banking_records(
    paths: ProtocolV2Paths, dataset: str, source_manifest: dict[str, Any]
) -> list[dict[str, object]]:
    source_root = paths.data_root / str(source_manifest["source_relative_directory"])
    rows: list[dict[str, object]] = []
    for file_info in source_manifest["files"]:
        if file_info.get("role") != "records":
            continue
        relative, split = str(file_info["relative_path"]), str(file_info["split"])
        if split not in {"train", "test"}:
            raise ValueError(f"Unexpected Banking77 split: {split}")
        with (source_root / relative).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["text", "category"]:
                raise ValueError(f"Unexpected Banking77 header: {reader.fieldnames!r}")
            for source_row, value in enumerate(reader, start=2):
                rows.append(
                    _make_record(
                        paths=paths,
                        dataset=dataset,
                        source_manifest=source_manifest,
                        relative_path=relative,
                        source_row=source_row,
                        original_split=split,
                        view_role=split,
                        text=str(value["text"]),
                        intent=str(value["category"]),
                    )
                )
    return rows


def _derive_banking_calibration(
    paths: ProtocolV2Paths, dataset: str, records: list[dict[str, object]], source_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Mark a fixed, class-stratified calibration set from official train only.

    The source has no development split.  Selecting with per-class SHA256
    ranks makes the derivation independent of TextOIR and repeatable without
    altering raw text, labels, or original source splits.
    """
    rule = source_manifest.get("calibration_derivation")
    if not isinstance(rule, dict):
        raise ValueError("Official Banking77 source manifest lacks a calibration derivation rule")
    target_count, salt = int(rule["target_count"]), str(rule["salt"])
    candidates = [row for row in records if row["original_split"] == str(rule["source_split"])]
    if not 0 < target_count < len(candidates):
        raise ValueError("Invalid Banking77 calibration target")
    by_intent: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_intent[str(row["intent"])].append(row)
    total = len(candidates)
    base = {intent: target_count * len(rows) // total for intent, rows in by_intent.items()}
    remaining = target_count - sum(base.values())
    # Largest remainder is deterministic; lexical labels make exact ties explicit.
    fractions = sorted(
        ((target_count * len(rows) % total, intent) for intent, rows in by_intent.items()),
        key=lambda item: (-item[0], item[1]),
    )
    for _, intent in fractions[:remaining]:
        base[intent] += 1
    selected_ids: set[str] = set()
    per_intent: dict[str, int] = {}
    for intent, rows in sorted(by_intent.items()):
        ranked = sorted(rows, key=lambda row: sha256_text(f"{salt}|{row['sample_id']}"))
        chosen = ranked[: base[intent]]
        selected_ids.update(str(row["sample_id"]) for row in chosen)
        per_intent[intent] = len(chosen)
    if len(selected_ids) != target_count:
        raise RuntimeError("Banking77 calibration derivation selected an unexpected number of rows")
    for row in records:
        row["calibration_candidate"] = str(row["sample_id"]) in selected_ids
        row["record_sha256"] = sha256_json({key: value for key, value in row.items() if key != "record_sha256"})
    payload: dict[str, Any] = {
        "schema_version": "protocol_v2.calibration_derivation.v1",
        "protocol_version": paths.dataset_version,
        "dataset": dataset,
        "source_manifest_sha256": sha256_file(source_manifest_path(paths.manifest_root, dataset)),
        "algorithm": rule["algorithm"],
        "source_split": rule["source_split"],
        "target_count": target_count,
        "salt": salt,
        "selected_sample_ids": sorted(selected_ids),
        "per_intent_count": dict(sorted(per_intent.items())),
    }
    payload["derivation_sha256"] = sha256_json(payload)
    write_manifest(calibration_derivation_path(paths.manifest_root, dataset), payload)
    return payload


def _records(paths: ProtocolV2Paths, dataset: str) -> tuple[list[dict[str, object]], dict[str, Any], dict[str, Any] | None]:
    source_manifest = read_json(source_manifest_path(paths.manifest_root, dataset))
    source_format = str(source_manifest.get("source_format", "textoir_tsv_v1"))
    if source_format == "textoir_tsv_v1":
        return _textoir_records(paths, dataset, source_manifest), source_manifest, None
    if source_format == "clinc_data_full_json_v1":
        return _official_clinc_records(paths, dataset, source_manifest), source_manifest, None
    if source_format == "banking77_csv_v1":
        records = _official_banking_records(paths, dataset, source_manifest)
        derivation = _derive_banking_calibration(paths, dataset, records, source_manifest)
        return records, source_manifest, derivation
    raise ValueError(f"Unsupported source format for {dataset}: {source_format}")


def build_canonical_dataset(paths: ProtocolV2Paths, dataset: str) -> dict[str, Any]:
    """Write stable JSONL atomically; the same audited source yields the same bytes."""
    records, source_manifest, derivation = _records(paths, dataset)
    if len({str(record["sample_id"]) for record in records}) != len(records):
        raise ValueError(f"Canonical sample_id collision in {dataset}")
    output = paths.protocol_root / dataset / "records.jsonl"
    atomic_write_jsonl(output, records)
    exact_duplicates = sum(count - 1 for count in Counter(str(row["text"]) for row in records).values() if count > 1)
    normalized_duplicates = sum(
        count - 1 for count in Counter(_normalized_text(str(row["text"])) for row in records).values() if count > 1
    )
    observed_intents = {str(row["intent"]) for row in records if not bool(row["native_oos"])}
    declared_order = source_manifest.get("intent_universe_order")
    # Only the imported source manifest supplies this order.  It freezes
    # TEXTOIR's label array so registry generation does not need to read the
    # external repository again.  Legacy sources retain deterministic sorting.
    if isinstance(declared_order, list) and all(isinstance(value, str) for value in declared_order):
        known_intents = [str(value) for value in declared_order]
        if set(known_intents) != observed_intents or len(known_intents) != len(observed_intents):
            raise ValueError(f"TEXTOIR label universe does not match imported rows for {dataset}")
    else:
        known_intents = sorted(observed_intents)
    manifest: dict[str, Any] = {
        "schema_version": "protocol_v2.dataset_manifest.v2",
        "protocol_version": paths.dataset_version,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "dataset": dataset,
        "source_name": source_manifest["source_name"],
        "source_format": source_manifest.get("source_format", "textoir_tsv_v1"),
        "source_manifest_sha256": sha256_file(source_manifest_path(paths.manifest_root, dataset)),
        "canonical_relative_path": f"canonical/{paths.dataset_version}/{dataset}/records.jsonl",
        "canonical_file_sha256": sha256_file(output),
        "sample_count": len(records),
        "split_counts": dict(sorted(Counter(str(row["original_split"]) for row in records).items())),
        "view_role_counts": dict(sorted(Counter(str(row["view_role"]) for row in records).items())),
        "known_label_count": len(known_intents),
        "intent_universe": known_intents,
        "intent_universe_order_source": "textoir_benchmark_labels" if isinstance(declared_order, list) else "sorted_canonical_labels",
        "native_oos_count": sum(bool(row["native_oos"]) for row in records),
        "exact_duplicate_count": exact_duplicates,
        "normalized_duplicate_count": normalized_duplicates,
        "calibration_derivation_sha256": derivation["derivation_sha256"] if derivation else None,
        "builder_version": "protocol_v2.canonicalize.v2",
        "git_commit": _git_commit(paths.project_root),
    }
    write_manifest(dataset_manifest_path(paths.manifest_root, dataset), manifest)
    return manifest


def build_canonical(paths: ProtocolV2Paths, datasets: Iterable[str] = DATASET_SPECS) -> dict[str, dict[str, Any]]:
    return {dataset: build_canonical_dataset(paths, dataset) for dataset in datasets}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASET_SPECS))
    args = parser.parse_args(argv)
    manifests = build_canonical(ProtocolV2Paths.discover(), args.dataset or DATASET_SPECS.keys())
    for dataset, manifest in manifests.items():
        print(f"canonical {dataset}: {manifest['sample_count']} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
