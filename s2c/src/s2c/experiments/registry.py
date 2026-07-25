"""Sweep plan and state-file helpers."""

from __future__ import annotations

from pathlib import Path

from s2c.data.hashing import atomic_write_json

from .matrix import GateRunSpec


def write_plan(path: Path, specs: list[GateRunSpec]) -> None:
    atomic_write_json(
        path,
        {
            "run_count": len(specs),
            "runs": [
                {
                    "run_id": spec.run_id,
                    "protocol_version": spec.protocol_version,
                    "dataset": spec.dataset,
                    "kir": spec.kir,
                    "seed": spec.seed,
                    "k_gate": spec.k_gate,
                    "distance": spec.distance,
                }
                for spec in specs
            ],
        },
    )
