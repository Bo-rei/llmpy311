"""Completion checks for immutable protocol_v2 run directories."""

from __future__ import annotations

import json
from pathlib import Path


def completed_run(run_dir: Path, config_hash: str) -> bool:
    manifest = run_dir / "manifest.json"
    metrics = run_dir / "metrics.json"
    if not manifest.is_file() or not metrics.is_file():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return payload.get("status") == "complete" and payload.get("config_hash") == config_hash

