"""Plan or run the OFAT frozen-MiniLM MOGB ablation sweep.

The sweep intentionally excludes the already completed ``mogb_baseline_v1``
reference cell (default partition + euclidean + mean).  That baseline is
recorded as metadata only; every planned cell here is new work under the
independent ``mogb_ablation_v1`` artifact root.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from protocol_v2.data.hashing import atomic_write_json
from protocol_v2.runtime.paths import ProtocolV2Paths

try:  # package import for tests
    from .run_mogb_ablation import _load_cached_inputs, load_config, run_one_loaded
except ImportError:  # direct script execution
    from run_mogb_ablation import _load_cached_inputs, load_config, run_one_loaded  # type: ignore[no-redef]


DEFAULT_DATASETS = ("clinc150", "banking77", "stackoverflow")
DEFAULT_KIRS = (0.25, 0.50, 0.75)
DEFAULT_SEEDS = (13, 42, 87, 100, 123)


def build_partition_variants() -> list[dict[str, Any]]:
    return [
        {"variant": "get_085", "purity_get_ball": 0.85},
        {"variant": "get_090", "purity_get_ball": 0.90},
        {"variant": "get_095", "purity_get_ball": 0.95},
        {"variant": "select_085", "purity_select_ball": 0.85},
        {"variant": "select_095", "purity_select_ball": 0.95},
        {"variant": "select_100", "purity_select_ball": 1.00},
        {"variant": "min_get_10", "min_ball_get_ball": 10},
        {"variant": "min_get_20", "min_ball_get_ball": 20},
        {"variant": "min_select_5", "min_ball_select_ball": 5},
        {"variant": "min_select_20", "min_ball_select_ball": 20},
    ]


def build_boundary_variants() -> list[dict[str, Any]]:
    return [
        {"variant": "default_mean_std", "distance": "euclidean", "boundary": "mean_std"},
        {"variant": "default_mahalanobis_mean", "distance": "mahalanobis_diag", "boundary": "mean"},
    ]


def build_specifications() -> list[dict[str, Any]]:
    specs = [
        {**item, "distance": "euclidean", "boundary": "mean"}
        for item in build_partition_variants()
    ]
    specs.extend(build_boundary_variants())
    return specs


def build_sweep_rows(
    datasets: tuple[str, ...],
    kirs: tuple[float, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    specs = build_specifications()
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for kir in kirs:
            for seed in seeds:
                for spec in specs:
                    rows.append(
                        {
                            "protocol_version": "protocol_v2_textoir_v1",
                            "dataset": dataset,
                            "kir": float(kir),
                            "seed": int(seed),
                            "variant": spec["variant"],
                            "distance": spec["distance"],
                            "boundary": spec["boundary"],
                            "purity_get_ball": spec.get("purity_get_ball"),
                            "purity_select_ball": spec.get("purity_select_ball"),
                            "min_ball_get_ball": spec.get("min_ball_get_ball"),
                            "min_ball_select_ball": spec.get("min_ball_select_ball"),
                            "test_used_for_selection": False,
                        }
                    )
    return rows


def _apply_sharding(rows: list[dict[str, Any]], shard_index: int | None, shard_count: int | None) -> list[dict[str, Any]]:
    if shard_index is None and shard_count is None:
        return rows
    if shard_index is None or shard_count is None:
        raise ValueError("Both --shard-index and --shard-count must be provided together")
    if shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")
    # Shard whole protocol cells, not individual variants.  Splitting variants
    # would make every worker reload the same immutable embedding cache and
    # defeat the shared-load execution contract.
    groups = list(_group_rows_by_cell(rows).values())
    selected = [group for index, group in enumerate(groups) if index % shard_count == shard_index]
    return [row for group in selected for row in group]


def _group_rows_by_cell(rows: list[dict[str, Any]]) -> dict[tuple[str, float, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), float(row["kir"]), int(row["seed"]))].append(row)
    return dict(grouped)


def _merge_config(base_config: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    config = dict(base_config)
    for key, value in row.items():
        if value is not None:
            config[key] = value
    if row.get("variant") is not None:
        config["output_variant"] = row["variant"]
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/mogb_ablation_v1.yaml"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--kirs", nargs="+", type=float, default=list(DEFAULT_KIRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    args = parser.parse_args(argv)

    paths = ProtocolV2Paths.discover()
    datasets = tuple(args.datasets)
    kirs = tuple(float(value) for value in args.kirs)
    seeds = tuple(int(value) for value in args.seeds)
    base_config = load_config(args.config)
    output_root = args.output_dir or paths.run_root / "mogb_ablation_v1"
    rows = build_sweep_rows(datasets, kirs, seeds)
    rows = _apply_sharding(rows, args.shard_index, args.shard_count)
    shard_suffix = (
        ""
        if args.shard_index is None
        else f"_shard_{args.shard_index}_of_{args.shard_count}"
    )
    grouped_rows = _group_rows_by_cell(rows)
    plan_path = output_root / "plans" / f"mogb_ablation_v1_plan{shard_suffix}.json"
    atomic_write_json(
        plan_path,
        {
            "protocol_version": paths.dataset_version,
            "output_root": str(output_root),
            "planned_units": len(rows),
            "rows": rows,
            "reference_reuse": {
                "variant": "reference_default_mean_euclidean",
                "source_root": str(paths.run_root / "mogb_baseline_v1"),
                "dataset_count": len(datasets),
                "kir_count": len(kirs),
                "seed_count": len(seeds),
                "planned_reference_cells_excluded": len(grouped_rows),
            },
            "test_used_for_selection": False,
        },
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "planned_units": len(rows),
                    "grouped_cells": len(grouped_rows),
                    "specifications_per_cell": len(build_specifications()),
                    "reference_cells_excluded": len(grouped_rows),
                    "plan": str(plan_path),
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                },
                sort_keys=True,
            )
        )
        return 0

    completed = 0
    skipped = 0
    failed: list[dict[str, Any]] = []
    for (dataset, kir, seed), cell_rows in grouped_rows.items():
        pending_rows: list[dict[str, Any]] = []
        for row in cell_rows:
            manifest = (
                output_root
                / str(row["variant"])
                / str(row["dataset"])
                / f"kir_{row['kir']:.2f}"
                / f"seed_{row['seed']}"
                / "manifest.json"
            )
            if args.resume and manifest.is_file() and not args.overwrite:
                skipped += 1
            else:
                pending_rows.append(row)
        if not pending_rows:
            continue
        try:
            views, train, test, inputs = _load_cached_inputs(paths, dataset, seed, kir)
        except RuntimeError as exc:
            for row in pending_rows:
                failed.append({**row, "error": f"{type(exc).__name__}: {exc}"})
            continue
        except Exception as exc:  # keep cells resumable and independently auditable
            for row in pending_rows:
                failed.append({**row, "error": f"{type(exc).__name__}: {exc}"})
            continue
        for row in pending_rows:
            config = _merge_config(base_config, row)
            try:
                run_one_loaded(paths, config, (views, train, test, inputs), output_dir=args.output_dir, overwrite=args.overwrite)
                completed += 1
            except RuntimeError as exc:
                failed.append({**row, "error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # keep cells resumable and independently auditable
                failed.append({**row, "error": f"{type(exc).__name__}: {exc}"})

    state_path = output_root / "plans" / f"mogb_ablation_v1_state{shard_suffix}.json"
    atomic_write_json(
        state_path,
        {
            "protocol_version": paths.dataset_version,
            "planned_units": len(rows),
            "grouped_cells": len(grouped_rows),
            "completed_this_invocation": completed,
            "skipped_complete": skipped,
            "failed": failed,
            "test_used_for_selection": False,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete" if not failed else "completed_with_failures",
                "planned_units": len(rows),
                "completed_this_invocation": completed,
                "skipped_complete": skipped,
                "failed": len(failed),
                "plan": str(plan_path),
                "state": str(state_path),
            },
            sort_keys=True,
        )
    )
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
