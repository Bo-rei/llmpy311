"""Helpers for resolving frozen v19 component artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def _iter_existing(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        resolved = path.resolve()
        if resolved.exists():
            yield resolved


def resolve_frozen_router_ckpt(
    project_root: Path,
    requested_path: Optional[Path] = None,
    legacy_path: Optional[Path] = None,
) -> Path:
    """Resolve the frozen router checkpoint with the formal component path first."""
    candidate_paths = [
        project_root / "outputs/experiments/components/router/router_v19/best_model.pt",
    ]

    if requested_path is not None:
        candidate_paths.append(requested_path)

    if legacy_path is not None:
        candidate_paths.append(legacy_path)

    candidate_paths.append(project_root / "outputs/router_v19/best_model.pt")

    for resolved in _iter_existing(candidate_paths):
        return resolved

    tried = "\n- ".join(str(path.resolve()) for path in candidate_paths)
    raise FileNotFoundError(f"No usable router checkpoint found. Checked:\n- {tried}")


def resolve_frozen_experts_root(
    project_root: Path,
    requested_path: Optional[Path] = None,
    legacy_path: Optional[Path] = None,
) -> Path:
    """Resolve the frozen experts root with the formal component path first."""
    candidate_paths = [
        project_root / "outputs/experiments/components/experts/experts_v19",
    ]

    if requested_path is not None:
        candidate_paths.append(requested_path)

    if legacy_path is not None:
        candidate_paths.append(legacy_path)

    candidate_paths.append(project_root / "outputs/experts_v19")

    for resolved in _iter_existing(candidate_paths):
        return resolved

    tried = "\n- ".join(str(path.resolve()) for path in candidate_paths)
    raise FileNotFoundError(f"No usable experts root found. Checked:\n- {tried}")
