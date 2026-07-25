"""Ensure protocol_v2 corpora stay local while manifests remain traceable."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
ACTIVE_DATASET_VERSION = os.environ.get("S2C_DATASET_VERSION", "protocol_v2_textoir_v1")
FORBIDDEN_PREFIXES = ("sources/", "canonical/", "views/", "exports/", "cache/", "tmp/")
FORBIDDEN_SUFFIXES = (".jsonl", ".parquet", ".npz", ".pt")


def tracked_data_paths() -> list[Path]:
    git_root = Path(
        subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    project_prefix = PROJECT_ROOT.relative_to(git_root).as_posix() + "/"
    output = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return [Path(path.removeprefix(project_prefix)) for path in output if path.startswith(project_prefix + "data/")]


def validate_local_manifests() -> list[str]:
    errors: list[str] = []
    manifest_root = DATA_ROOT / "manifests" / ACTIVE_DATASET_VERSION
    source_root = DATA_ROOT / "sources" / "textoir"
    if source_root.exists():
        for dataset_dir in source_root.glob("*"):
            if not dataset_dir.is_dir():
                continue
            for dataset in dataset_dir.iterdir():
                if dataset.is_dir() and not (manifest_root / dataset.name / "SOURCE_MANIFEST.json").is_file():
                    errors.append(f"missing SOURCE_MANIFEST for imported data: {dataset.name}")
    for canonical in (DATA_ROOT / "canonical" / ACTIVE_DATASET_VERSION).glob("*") if (DATA_ROOT / "canonical" / ACTIVE_DATASET_VERSION).exists() else ():
        if canonical.is_dir() and not (manifest_root / canonical.name / "DATASET_MANIFEST.json").is_file():
            errors.append(f"missing DATASET_MANIFEST for canonical data: {canonical.name}")
    for path in (DATA_ROOT / "registries" / ACTIVE_DATASET_VERSION).rglob("*.json") if (DATA_ROOT / "registries" / ACTIVE_DATASET_VERSION).exists() else ():
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = str(payload.get("dataset", ""))
        if not (manifest_root / dataset / "SOURCE_MANIFEST.json").is_file():
            errors.append(f"registry points to missing source manifest: {path}")
        if not (manifest_root / dataset / "DATASET_MANIFEST.json").is_file():
            errors.append(f"registry points to missing canonical manifest: {path}")
    stack_manifest = manifest_root / "stackoverflow" / "SOURCE_MANIFEST.json"
    if stack_manifest.is_file():
        payload = json.loads(stack_manifest.read_text(encoding="utf-8"))
        if payload.get("redistribution_by_s2c") is not False:
            errors.append("StackOverflow manifest must forbid s2c redistribution")
        if payload.get("local_research_only") is not True:
            errors.append("StackOverflow manifest must be marked local-research-only")
    return errors


def main() -> int:
    errors: list[str] = []
    for relative in tracked_data_paths():
        rendered = relative.as_posix().removeprefix("data/")
        path = PROJECT_ROOT / relative
        if rendered.startswith(FORBIDDEN_PREFIXES) or path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"tracked local corpus or derived data is forbidden: {relative}")
        elif path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"tracked data file exceeds 5MB: {relative}")
    errors.extend(validate_local_manifests())
    if errors:
        print("data tracking check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("data tracking check: local corpora ignored; manifests and registries are coherent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
