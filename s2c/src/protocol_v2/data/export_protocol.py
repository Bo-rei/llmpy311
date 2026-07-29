"""Create all protocol_v2 method exports from canonical data, registry and views."""

from __future__ import annotations

import argparse
from typing import Any, Iterable

from protocol_v2.runtime.paths import ProtocolV2Paths

from .exporters import export_adb, export_da_adb, export_k_plus_1_way, export_mogb, export_s2c, export_textoir
from .schema import ALL_REGISTRY_SEEDS, DATASET_SPECS, FORMAL_KIRS


EXPORTERS = {
    "s2c": export_s2c,
    "textoir": export_textoir,
    "adb": export_adb,
    "da_adb": export_da_adb,
    "mogb": export_mogb,
    "k_plus_1_way": export_k_plus_1_way,
}


def export_protocol(
    paths: ProtocolV2Paths,
    datasets: Iterable[str] = DATASET_SPECS,
    kirs: Iterable[float] = FORMAL_KIRS,
    seeds: Iterable[int] = ALL_REGISTRY_SEEDS,
    names: Iterable[str] = EXPORTERS,
) -> list[dict[str, Any]]:
    return [
        EXPORTERS[name](paths, dataset, seed, kir)
        for name in names
        for dataset in datasets
        for seed in seeds
        for kir in kirs
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASET_SPECS))
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--kir", type=float, action="append")
    parser.add_argument("--export", dest="names", action="append", choices=sorted(EXPORTERS))
    args = parser.parse_args(argv)
    manifests = export_protocol(
        ProtocolV2Paths.discover(),
        args.dataset or DATASET_SPECS.keys(),
        args.kir or FORMAL_KIRS,
        args.seed or ALL_REGISTRY_SEEDS,
        args.names or EXPORTERS.keys(),
    )
    print(f"built or verified {len(manifests)} export manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
