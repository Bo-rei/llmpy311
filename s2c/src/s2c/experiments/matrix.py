"""Declarative Gate matrix parsing with deterministic run identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from s2c.data.schema import format_kir


@dataclass(frozen=True)
class GateRunSpec:
    experiment_name: str
    dataset: str
    kir: float
    seed: int
    k_gate: int
    distance: str
    representation: str
    boundary: str
    radius_lambda: float
    encoder_name: str
    encoder_device: str
    protocol_version: str = "protocol_v2"

    @property
    def run_id(self) -> str:
        return "__".join(
            (
                self.protocol_version,
                self.dataset,
                f"kir_{format_kir(self.kir)}",
                f"seed_{self.seed}",
                f"repr_{self.representation}",
                f"k_{self.k_gate}",
                f"dist_{self.distance}",
                f"boundary_{self.boundary}",
            )
        )


def _required(payload: dict[str, Any], field: str) -> list[Any]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Experiment matrix requires non-empty list: {field}")
    return value


def load_gate_matrix(path: Path) -> list[GateRunSpec]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment configuration must be a mapping: {path}")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"Experiment defaults must be a mapping: {path}")
    name = str(payload.get("name", path.stem))
    protocol_version = str(payload.get("protocol_version", "protocol_v2"))
    boundaries = payload.get("boundary_methods", [defaults.get("boundary", "mean_std")])
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("Experiment boundary_methods must be a non-empty list when provided")
    result = [
        GateRunSpec(
            experiment_name=name,
            dataset=str(dataset),
            kir=float(kir),
            seed=int(seed),
            k_gate=int(k_gate),
            distance=str(distance),
            representation=str(defaults.get("representation", "frozen_minilm")),
            boundary=str(boundary),
            radius_lambda=float(defaults.get("radius_lambda", 1.0)),
            encoder_name=str(defaults.get("encoder_name", "all-MiniLM-L6-v2")),
            encoder_device=str(defaults.get("encoder_device", "cuda")),
            protocol_version=protocol_version,
        )
        for dataset in _required(payload, "datasets")
        for kir in _required(payload, "kirs")
        for seed in _required(payload, "seeds")
        for k_gate in _required(payload, "k_values")
        for distance in _required(payload, "distances")
        for boundary in boundaries
    ]
    if len({spec.run_id for spec in result}) != len(result):
        raise ValueError(f"Experiment configuration creates duplicate run ids: {path}")
    return result


def filter_gate_specs(
    specs: Iterable[GateRunSpec],
    *,
    datasets: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    kirs: Iterable[float] | None = None,
) -> list[GateRunSpec]:
    """Select an exact matrix shard without changing its run identifiers.

    Runner, verifier and summarizer must operate on the same declared subset.
    This keeps an admitted two-dataset smoke shard from being confused with a
    wider candidate config that intentionally still names blocked datasets.
    """
    selected_datasets = set(datasets or ())
    selected_seeds = set(seeds or ())
    selected_kirs = set(kirs or ())
    return [
        spec
        for spec in specs
        if (not selected_datasets or spec.dataset in selected_datasets)
        and (not selected_seeds or spec.seed in selected_seeds)
        and (not selected_kirs or spec.kir in selected_kirs)
    ]
