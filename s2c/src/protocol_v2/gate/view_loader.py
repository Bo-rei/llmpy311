"""Load the s2c export contract without touching legacy assets or TEXTOIR data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from protocol_v2.data.exporters._common import export_directory
from protocol_v2.runtime.paths import ProtocolV2Paths


@dataclass(frozen=True)
class GateViews:
    train: list[dict[str, Any]]
    calibration: list[dict[str, Any]]
    test: list[dict[str, Any]]
    export_root: Path


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing protocol_v2 Gate export: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Gate export must be a JSON list of objects: {path}")
    return payload


def load_gate_views(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> GateViews:
    root = export_directory(paths, "s2c", dataset, seed, kir)
    paths.reject_textoir_runtime_path(root)
    views = GateViews(_read(root / "gate" / "train.json"), _read(root / "gate" / "val.json"), _read(root / "gate" / "test.json"), root)
    if any(int(row["label"]) != 0 for row in views.train + views.calibration):
        raise ValueError(f"Known-only train/calibration violation: dataset={dataset}, KIR={kir}, seed={seed}")
    return views

