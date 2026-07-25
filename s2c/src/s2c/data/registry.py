"""Deterministic, shared Known-Intent-Ratio registries for protocol_v2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from s2c.runtime.paths import ProtocolV2Paths

from .hashing import sha256_file, sha256_json
from .manifests import dataset_manifest_path, read_json, write_manifest
from .schema import (
    ALL_REGISTRY_SEEDS,
    DATASET_SPECS,
    FORMAL_KIRS,
    NATIVE_OOS_LABEL,
    REGISTRY_SCHEMA_VERSION,
    format_kir,
)


def registry_path(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> Path:
    return paths.registry_root / dataset / f"seed_{seed}" / f"kir_{format_kir(kir)}.json"


def _registry_payload(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> dict[str, Any]:
    manifest_path = dataset_manifest_path(paths.manifest_root, dataset)
    canonical_manifest = read_json(manifest_path)
    # Keep the order recorded by canonicalization.  For the active snapshot it
    # is TEXTOIR's benchmark_labels order; sorting here would quietly change
    # the labels selected by a seed and invalidate fair baseline comparisons.
    intents = tuple(str(intent) for intent in canonical_manifest["intent_universe"])
    known_count = int(round(len(intents) * kir))
    if not 1 <= known_count < len(intents):
        raise ValueError(
            f"Invalid known_count for dataset={dataset}, KIR={kir}, seed={seed}: "
            f"{known_count} of {len(intents)}"
        )
    legacy_rng = np.random.RandomState(seed)
    known = tuple(str(value) for value in legacy_rng.choice(np.array(intents), size=known_count, replace=False))
    heldout = tuple(intent for intent in intents if intent not in set(known))
    payload: dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "protocol_version": paths.dataset_version,
        "dataset": dataset,
        "seed": seed,
        "requested_kir": kir,
        "known_count": known_count,
        "effective_kir": known_count / len(intents),
        "intent_universe": list(intents),
        "known_intents": list(known),
        "heldout_intents": list(heldout),
        "native_oos_label": NATIVE_OOS_LABEL if dataset == "clinc150" else None,
        "source_manifest_sha256": str(canonical_manifest["source_manifest_sha256"]),
        "canonical_manifest_sha256": sha256_file(manifest_path),
        "numpy_version": np.__version__,
        "selection_algorithm": "numpy.random.seed(seed); numpy.random.choice(intent_universe_in_manifest_order, round(n_labels * KIR), replace=False); MT19937-compatible RandomState",
    }
    payload["registry_sha256"] = sha256_json(payload)
    return payload


def validate_registry(payload: dict[str, Any]) -> None:
    universe = set(str(value) for value in payload["intent_universe"])
    known = set(str(value) for value in payload["known_intents"])
    heldout = set(str(value) for value in payload["heldout_intents"])
    if known & heldout or known | heldout != universe:
        raise ValueError(f"Registry partition invalid for {payload['dataset']} seed={payload['seed']}")
    if payload.get("native_oos_label") in universe:
        raise ValueError(f"Native OOS label leaked into intent universe: {payload['dataset']}")
    copy = dict(payload)
    expected = copy.pop("registry_sha256", None)
    if expected != sha256_json(copy):
        raise ValueError(f"Registry SHA256 mismatch for {payload['dataset']} seed={payload['seed']}")


def build_registry(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> dict[str, Any]:
    payload = _registry_payload(paths, dataset, seed, kir)
    validate_registry(payload)
    path = registry_path(paths, dataset, seed, kir)
    if path.exists():
        existing = read_json(path)
        if existing != payload:
            raise RuntimeError(f"Existing registry differs from deterministic payload: {path}")
        return existing
    write_manifest(path, payload)
    return payload


def build_registries(
    paths: ProtocolV2Paths,
    datasets: Iterable[str] = DATASET_SPECS,
    kirs: Iterable[float] = FORMAL_KIRS,
    seeds: Iterable[int] = ALL_REGISTRY_SEEDS,
) -> list[dict[str, Any]]:
    return [build_registry(paths, dataset, seed, kir) for dataset in datasets for seed in seeds for kir in kirs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASET_SPECS))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--kir", type=float, action="append")
    args = parser.parse_args(argv)
    rows = build_registries(
        ProtocolV2Paths.discover(),
        args.dataset or DATASET_SPECS.keys(),
        args.kir or FORMAL_KIRS,
        args.seed or ALL_REGISTRY_SEEDS,
    )
    print(f"built or verified {len(rows)} registries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
