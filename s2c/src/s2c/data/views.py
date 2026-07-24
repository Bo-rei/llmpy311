"""Materialize fixed Known-only calibration and open-set test views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from s2c.runtime.paths import ProtocolV2Paths

from .hashing import atomic_write_jsonl, sha256_file
from .manifests import calibration_derivation_path, dataset_manifest_path, read_json, view_manifest_path, write_manifest
from .registry import registry_path, validate_registry
from .schema import ALL_REGISTRY_SEEDS, DATASET_SPECS, FORMAL_KIRS, VIEW_SCHEMA_VERSION, format_kir


VIEW_NAMES = (
    "train_known",
    "calibration_known",
    "test_known",
    "test_heldout_oos",
    "test_native_oos",
    "test_combined",
)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Canonical JSONL row must be an object: {path}")
                yield value


def view_directory(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> Path:
    return paths.view_root / dataset / f"seed_{seed}" / f"kir_{format_kir(kir)}"


def _annotate(row: dict[str, Any], oos_source: str, evaluation_label: str) -> dict[str, Any]:
    return {**row, "oos_source": oos_source, "evaluation_label": evaluation_label}


def _view_role(row: dict[str, Any]) -> str:
    """优先使用显式 source-role；旧 candidate 行仅为兼容而回退推断。"""
    if "view_role" in row:
        return str(row["view_role"])
    return {"train": "train", "dev": "calibration", "test": "test"}.get(str(row["original_split"]), "excluded")


def build_views(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> dict[str, Any]:
    registry = read_json(registry_path(paths, dataset, seed, kir))
    validate_registry(registry)
    manifest = read_json(dataset_manifest_path(paths.manifest_root, dataset))
    canonical_path = paths.data_root / str(manifest["canonical_relative_path"])
    if not canonical_path.is_file():
        raise FileNotFoundError(f"Canonical data missing for dataset={dataset}: {canonical_path}")
    known = set(str(value) for value in registry["known_intents"])
    records = list(_read_jsonl(canonical_path))
    # Banking77 has no upstream dev split.  Its canonical build marks a stable
    # calibration candidate; it is removed from train without rewriting the
    # raw ``original_split='train'`` field.
    train_known = [
        row
        for row in records
        if _view_role(row) == "train"
        and not bool(row.get("calibration_candidate", False))
        and not row["native_oos"]
        and row["intent"] in known
    ]
    calibration_known = [
        row
        for row in records
        if (_view_role(row) == "calibration" or bool(row.get("calibration_candidate", False)))
        and not row["native_oos"]
        and row["intent"] in known
    ]
    test_known = [
        _annotate(row, "known", str(row["intent"]))
        for row in records
        if _view_role(row) == "test" and not row["native_oos"] and row["intent"] in known
    ]
    test_heldout = [
        _annotate(row, "heldout_intent", "oos")
        for row in records
        if _view_role(row) == "test" and not row["native_oos"] and row["intent"] not in known
    ]
    test_native = [
        _annotate(row, "native", "oos") for row in records if _view_role(row) == "test" and bool(row["native_oos"])
    ]
    views: dict[str, list[dict[str, Any]]] = {
        "train_known": train_known,
        "calibration_known": calibration_known,
        "test_known": test_known,
        "test_heldout_oos": test_heldout,
        "test_native_oos": test_native,
        "test_combined": test_known + test_heldout + test_native,
    }
    if any(bool(row["native_oos"]) for row in train_known + calibration_known):
        raise ValueError(f"Native OOS leaked into Known-only train/calibration for {dataset}/{seed}/{kir}")
    if set(map(lambda row: str(row["sample_id"]), train_known)) & set(map(lambda row: str(row["sample_id"]), calibration_known)):
        raise ValueError(f"Train/calibration sample overlap for {dataset}/{seed}/{kir}")
    output_dir = view_directory(paths, dataset, seed, kir)
    files: list[dict[str, object]] = []
    for name in VIEW_NAMES:
        path = output_dir / f"{name}.jsonl"
        atomic_write_jsonl(path, views[name])
        files.append({"name": name, "relative_path": f"views/{paths.dataset_version}/{dataset}/seed_{seed}/kir_{format_kir(kir)}/{name}.jsonl", "count": len(views[name]), "sha256": sha256_file(path)})
    local_manifest = {
        "schema_version": VIEW_SCHEMA_VERSION,
        "protocol_version": paths.dataset_version,
        "dataset": dataset,
        "seed": seed,
        "kir": kir,
        "registry_sha256": registry["registry_sha256"],
        "canonical_manifest_sha256": sha256_file(dataset_manifest_path(paths.manifest_root, dataset)),
        "calibration_derivation_sha256": (
            sha256_file(calibration_derivation_path(paths.manifest_root, dataset))
            if calibration_derivation_path(paths.manifest_root, dataset).is_file()
            else None
        ),
        "files": files,
    }
    write_manifest(output_dir / "VIEW_MANIFEST.json", local_manifest)
    write_manifest(view_manifest_path(paths.manifest_root, dataset, seed, kir), local_manifest)
    return local_manifest


def build_all_views(
    paths: ProtocolV2Paths,
    datasets: Iterable[str] = DATASET_SPECS,
    kirs: Iterable[float] = FORMAL_KIRS,
    seeds: Iterable[int] = ALL_REGISTRY_SEEDS,
) -> list[dict[str, Any]]:
    return [build_views(paths, dataset, seed, kir) for dataset in datasets for seed in seeds for kir in kirs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASET_SPECS))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--kir", type=float, action="append")
    args = parser.parse_args(argv)
    rows = build_all_views(
        ProtocolV2Paths.discover(),
        args.dataset or DATASET_SPECS.keys(),
        args.kir or FORMAL_KIRS,
        args.seed or ALL_REGISTRY_SEEDS,
    )
    print(f"built or verified {len(rows)} views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
