"""Stable locations and JSON helpers for protocol_v2 provenance manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import atomic_write_json, sha256_file
from .schema import format_kir


def source_manifest_path(manifest_root: Path, dataset: str) -> Path:
    return manifest_root / dataset / "SOURCE_MANIFEST.json"


def dataset_manifest_path(manifest_root: Path, dataset: str) -> Path:
    return manifest_root / dataset / "DATASET_MANIFEST.json"


def calibration_derivation_path(manifest_root: Path, dataset: str) -> Path:
    """Location for a documented split derivation when upstream has no dev set."""
    return manifest_root / dataset / "CALIBRATION_DERIVATION.json"


def view_manifest_path(manifest_root: Path, dataset: str, seed: int, kir: float) -> Path:
    return manifest_root / dataset / "views" / f"seed_{seed}" / f"kir_{format_kir(kir)}.json"


def export_manifest_path(manifest_root: Path, export_name: str, dataset: str, seed: int, kir: float) -> Path:
    return manifest_root / dataset / "exports" / export_name / f"seed_{seed}" / f"kir_{format_kir(kir)}.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required protocol_v2 manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must be an object: {path}")
    return value


def write_manifest(path: Path, payload: dict[str, Any]) -> str:
    atomic_write_json(path, payload)
    return sha256_file(path)
