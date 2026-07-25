"""Fail-closed validation for protocol_v2 data, registries, views and exports."""

from __future__ import annotations

import argparse
import json
from typing import Iterable

from s2c.runtime.paths import ProtocolV2Paths

from .exporters._common import read_jsonl
from .hashing import sha256_file
from .manifests import (
    calibration_derivation_path,
    dataset_manifest_path,
    read_json,
    source_manifest_path,
    view_manifest_path,
)
from .registry import registry_path, validate_registry
from .schema import ALL_REGISTRY_SEEDS, DATASET_SPECS, FORMAL_KIRS, format_kir
from .views import VIEW_NAMES, view_directory


def _required_export_names(paths: ProtocolV2Paths) -> tuple[str, ...]:
    """Keep historical audit validation compatible with its original exports."""
    base = ("s2c", "textoir", "mogb", "k_plus_1_way")
    return base + ("adb", "da_adb") if paths.dataset_version == "protocol_v2_textoir_v1" else base


def validate_source_snapshot(paths: ProtocolV2Paths, dataset: str) -> None:
    manifest_path = source_manifest_path(paths.manifest_root, dataset)
    manifest = read_json(manifest_path)
    source_root = paths.data_root / str(manifest["source_relative_directory"])
    for file_info in manifest["files"]:
        if not isinstance(file_info, dict):
            raise ValueError(f"Malformed source manifest for {dataset}")
        path = source_root / str(file_info["relative_path"])
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Imported source is missing or linked for dataset={dataset}: {path}")
        if sha256_file(path) != file_info["sha256"]:
            raise ValueError(f"Source SHA256 mismatch for dataset={dataset}: {path}")
    if manifest.get("source_name") == "official":
        license_files = [item for item in manifest["files"] if item.get("role") == "license"]
        if manifest.get("license_status") != "verified" or len(license_files) != 1:
            raise ValueError(f"Official source license provenance is incomplete for dataset={dataset}")
        if manifest.get("license_file_sha256") != license_files[0].get("sha256"):
            raise ValueError(f"Official license SHA256 mismatch for dataset={dataset}")
    if manifest.get("source_name") == "textoir":
        if manifest.get("source_format") != "textoir_tsv_v1":
            raise ValueError(f"Unexpected TEXTOIR source format for dataset={dataset}")
        labels = manifest.get("intent_universe_order")
        if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)):
            raise ValueError(f"TEXTOIR label-order manifest is invalid for dataset={dataset}")
        if dataset == "stackoverflow":
            expected = {
                "expected_samples": 20_000,
                "expected_labels": 20,
                "local_research_only": True,
                "redistribution_by_s2c": False,
                "per_row_attribution_complete": False,
            }
            for field, value in expected.items():
                if manifest.get(field) != value:
                    raise ValueError(f"StackOverflow local-benchmark policy mismatch: {field}")


def validate_canonical_dataset(paths: ProtocolV2Paths, dataset: str) -> None:
    manifest = read_json(dataset_manifest_path(paths.manifest_root, dataset))
    path = paths.data_root / str(manifest["canonical_relative_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Canonical data missing for dataset={dataset}: {path}")
    if sha256_file(path) != manifest["canonical_file_sha256"]:
        raise ValueError(f"Canonical SHA256 mismatch for dataset={dataset}: {path}")
    rows = list(read_jsonl(path))
    if len(rows) != manifest["sample_count"]:
        raise ValueError(f"Canonical row count mismatch for dataset={dataset}")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError(f"Canonical sample_id is not unique for dataset={dataset}")
    if dataset == "stackoverflow":
        if len(rows) != 20_000:
            raise ValueError("StackOverflow protocol_v2 must retain all 20,000 TEXTOIR rows")
        if len({str(row["intent"]) for row in rows}) != 20:
            raise ValueError("StackOverflow protocol_v2 must retain all 20 TEXTOIR labels")
        if manifest.get("native_oos_count") != 0:
            raise ValueError("StackOverflow protocol_v2 must not invent native OOS rows")
    # 这些固定计数只约束已核验的官方版本，避免小型 synthetic fixture 或历史
    # TEXTOIR candidate 被误判为官方 raw reconstruction。
    if manifest.get("source_name") == "official":
        expected = {
            "clinc150": {"sample_count": 23_700, "known_label_count": 150, "native_oos_count": 1_200},
            "banking77": {"sample_count": 13_083, "known_label_count": 77, "native_oos_count": 0},
        }.get(dataset)
        if expected is None:
            raise ValueError(f"Unadmitted official dataset in canonical validation: {dataset}")
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(f"Official canonical invariant failed for {dataset}: {field}")
        if dataset == "banking77":
            derivation = read_json(calibration_derivation_path(paths.manifest_root, dataset))
            selected = {str(value) for value in derivation["selected_sample_ids"]}
            marked = {str(row["sample_id"]) for row in rows if bool(row.get("calibration_candidate"))}
            if derivation.get("target_count") != 1_000 or selected != marked:
                raise ValueError("Banking77 calibration derivation is incomplete or inconsistent")


def validate_view(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> None:
    registry = read_json(registry_path(paths, dataset, seed, kir))
    validate_registry(registry)
    manifest_path = view_manifest_path(paths.manifest_root, dataset, seed, kir)
    manifest = read_json(manifest_path)
    root = view_directory(paths, dataset, seed, kir)
    counts: dict[str, int] = {}
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise ValueError(f"Malformed view manifest for dataset={dataset}, KIR={kir}, seed={seed}")
        name = str(item["name"])
        path = root / f"{name}.jsonl"
        rows = list(read_jsonl(path))
        if sha256_file(path) != item["sha256"] or len(rows) != item["count"]:
            raise ValueError(f"View manifest mismatch for {dataset}, KIR={kir}, seed={seed}, view={name}")
        counts[name] = len(rows)
        if name in {"train_known", "calibration_known"} and any(row.get("native_oos") for row in rows):
            raise ValueError(f"Native OOS leaked into {name}: {dataset}, KIR={kir}, seed={seed}")
        if name in {"train_known", "calibration_known"} and any(row["intent"] not in registry["known_intents"] for row in rows):
            raise ValueError(f"Held-out intent leaked into {name}: {dataset}, KIR={kir}, seed={seed}")
    if set(counts) != set(VIEW_NAMES):
        raise ValueError(f"Incomplete views for dataset={dataset}, KIR={kir}, seed={seed}")
    if counts["test_combined"] != counts["test_known"] + counts["test_heldout_oos"] + counts["test_native_oos"]:
        raise ValueError(f"Combined test count mismatch for dataset={dataset}, KIR={kir}, seed={seed}")
    known_train = set(str(row["sample_id"]) for row in read_jsonl(root / "train_known.jsonl"))
    calibration = set(str(row["sample_id"]) for row in read_jsonl(root / "calibration_known.jsonl"))
    test = set(str(row["sample_id"]) for row in read_jsonl(root / "test_combined.jsonl"))
    if known_train & calibration or known_train & test or calibration & test:
        raise ValueError(f"View sample overlap for dataset={dataset}, KIR={kir}, seed={seed}")


def validate_export(paths: ProtocolV2Paths, export_name: str, dataset: str, seed: int, kir: float) -> None:
    manifest_path = paths.manifest_root / dataset / "exports" / export_name / f"seed_{seed}" / f"kir_{format_kir(kir)}.json"
    manifest = read_json(manifest_path)
    root = paths.data_root / str(manifest["output_relative_directory"])
    for item in manifest["files"]:
        if not isinstance(item, dict):
            raise ValueError(f"Malformed export manifest for {export_name}")
        path = root / str(item["relative_path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Export SHA256 mismatch for {export_name}: {path}")
    if not (root / "sample_ids.json").is_file():
        raise FileNotFoundError(f"Export has no canonical sample-id map: {root}")


def validate_protocol(
    paths: ProtocolV2Paths,
    datasets: Iterable[str] = DATASET_SPECS,
    kirs: Iterable[float] = FORMAL_KIRS,
    seeds: Iterable[int] = ALL_REGISTRY_SEEDS,
    *,
    require_views: bool = False,
    require_exports: bool = False,
) -> dict[str, int]:
    datasets, kirs, seeds = tuple(datasets), tuple(kirs), tuple(seeds)
    for dataset in datasets:
        validate_source_snapshot(paths, dataset)
        validate_canonical_dataset(paths, dataset)
    registry_count = 0
    view_count = 0
    export_count = 0
    for dataset in datasets:
        for seed in seeds:
            for kir in kirs:
                registry = read_json(registry_path(paths, dataset, seed, kir))
                validate_registry(registry)
                registry_count += 1
                view_manifest = view_manifest_path(paths.manifest_root, dataset, seed, kir)
                if require_views or view_manifest.exists():
                    validate_view(paths, dataset, seed, kir)
                    view_count += 1
                if require_exports:
                    for name in _required_export_names(paths):
                        validate_export(paths, name, dataset, seed, kir)
                        export_count += 1
    return {"datasets": len(datasets), "registries": registry_count, "views": view_count, "exports": export_count}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASET_SPECS))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--kir", type=float, action="append")
    parser.add_argument("--require-views", action="store_true")
    parser.add_argument("--require-exports", action="store_true")
    args = parser.parse_args(argv)
    result = validate_protocol(
        ProtocolV2Paths.discover(),
        args.dataset or DATASET_SPECS.keys(),
        args.kir or FORMAL_KIRS,
        args.seed or ALL_REGISTRY_SEEDS,
        require_views=args.require_views,
        require_exports=args.require_exports,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
