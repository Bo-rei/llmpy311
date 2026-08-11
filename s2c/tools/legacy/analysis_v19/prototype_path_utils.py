"""Shared helpers for resolving v19 prototype payload paths."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def _iter_existing(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        resolved = path.resolve()
        if resolved.exists():
            yield resolved


def resolve_multi_prototype_path(
    project_root: Path,
    requested_path: Optional[Path] = None,
) -> Path:
    """Resolve the historical multi-prototype payload with formal artifacts first.

    The repository has a legacy root alias that currently points at a smoke
    artifact. Historical reproduction should prefer the formal ablation payloads
    under `outputs/experiments/pipeline/ablations/...` instead.
    """

    candidate_paths = [
        project_root
        / "outputs/experiments/pipeline/ablations/stage3_2026-03-24_pipeline_ablation/02_gate_scoring/weak_gate/prototypes.json",
        project_root
        / "outputs/experiments/pipeline/ablations/stage3_2026-03-24_pipeline_ablation/02_gate_scoring/mid_gate/prototypes.json",
        project_root
        / "outputs/experiments/pipeline/ablations/stage3_2026-03-24_pipeline_ablation/02_gate_scoring/strong_gate/prototypes.json",
    ]

    if requested_path is not None:
        candidate_paths.append(requested_path)

    candidate_paths.append(project_root / "outputs/multi_prototypes_v19/prototypes.json")

    for resolved in _iter_existing(candidate_paths):
        return resolved

    tried = "\n- ".join(str(path.resolve()) for path in candidate_paths)
    raise FileNotFoundError(f"No usable multi-prototype payload found. Checked:\n- {tried}")
