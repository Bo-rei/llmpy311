"""Dataset-profile loading for the canonical CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    dataset: str
    slug: str
    policy: dict[str, Any]
    stackoverflow_known_selection_strategy: str | None = None


def load_profile(path: Path, name: str) -> DatasetProfile:
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = payload.get("profiles", {}).get(name)
    if raw is None:
        raise KeyError(f"Unknown dataset profile: {name}")
    return DatasetProfile(
        name=name,
        dataset=raw["dataset"],
        slug=raw["slug"],
        policy=dict(raw.get("policy", {})),
        stackoverflow_known_selection_strategy=raw.get("stackoverflow_known_selection_strategy"),
    )
