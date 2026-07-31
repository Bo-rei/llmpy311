"""Plan and run the fixed-K MOGB mean-radius ablation matrix."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from protocol_v2.data.hashing import atomic_write_json
from protocol_v2.runtime.paths import ProtocolV2Paths

try:  # package import for tests
    from .run_mogb_fixed_k_mean_ablation import (
        _load_cached_inputs,
        load_config,
        PARTITION_NAME,
        run_one_loaded,
    )
except ImportError:  # direct script execution
    from run_mogb_fixed_k_mean_ablation import (  # type: ignore[no-redef]
        _load_cached_inputs,
        load_config,
        PARTITION_NAME,
        run_one_loaded,
    )


DEFAULT_DATASETS = ("clinc150", "banking77", "stackoverflow")
DEFAULT_KIRS = (0.25, 0.50, 0.75)
DEFAULT_SEEDS = (13, 42, 87, 100, 123)
DEFAULT_NEW_K_VALUES = (1, 3, 4)


def build_rows(
    datasets: tuple[str, ...],
    kirs: tuple[float, ...],
    seeds: tuple[int, ...],
    k_values: tuple[int, ...] = DEFAULT_NEW_K_VALUES,
) -> list[dict[str, Any]]:
    return [
        {
            "protocol_version": "protocol_v2_textoir_v1",
            "dataset": dataset,
            "kir": float(kir),
            "seed": int(seed),
            "k": int(k),
            "partition": PARTITION_NAME,
            "partition_seed": 42,
            "distance": "euclidean",
            "boundary": "mean",
            "test_used_for_selection": False,
        }
        for dataset in datasets
        for kir in kirs
        for seed in seeds
        for k in k_values
    ]


def validate_reference(
    paths: ProtocolV2Paths,
    dataset: str,
    kir: float,
    seed: int,
    *,
    method: str,
    expected_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = (
        paths.run_root
        / "mogb_baseline_v1"
        / dataset
        / f"kir_{kir:.2f}"
        / f"seed_{seed}"
        / method
    )
    required = ("manifest.json", "config.json", "inputs.json", "metrics.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing K=2 reference files at {run_dir}: {missing}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    inputs = json.loads((run_dir / "inputs.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete K=2 reference: {run_dir}")
    if config.get("protocol_version") != paths.dataset_version:
        raise ValueError(f"Mixed protocol in K=2 reference: {run_dir}")
    if config.get("method") != method or bool(config.get("test_used_for_selection")):
        raise ValueError(f"Invalid K=2 reference contract: {run_dir}")
    if expected_inputs is not None and inputs != expected_inputs:
        raise ValueError(f"K=2 reference input hashes differ from fixed-K cell: {run_dir}")
    return {"run_dir": str(run_dir), "inputs": inputs, "run_id": manifest.get("run_id")}


def group_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, float, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), float(row["kir"]), int(row["seed"]))].append(row)
    return dict(groups)


def apply_shard(
    rows: list[dict[str, Any]], shard_index: int | None, shard_count: int | None
) -> list[dict[str, Any]]:
    if shard_index is None and shard_count is None:
        return rows
    if shard_index is None or shard_count is None:
        raise ValueError("Both --shard-index and --shard-count are required")
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("Invalid shard index/count")
    groups = list(group_rows(rows).values())
    return [row for index, group in enumerate(groups) if index % shard_count == shard_index for row in group]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/mogb_fixed_k_mean_ablation_v1.yaml"),
    )
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
    base = load_config(args.config)
    new_k_values = tuple(int(value) for value in base.get("new_k_values", ()))
    if not new_k_values or len(set(new_k_values)) != len(new_k_values):
        raise ValueError(f"Invalid new_k_values in {args.config}: {new_k_values}")
    reference = base.get("reference_k", {})
    if not isinstance(reference, dict):
        raise ValueError("reference_k must be a mapping")
    reference_k = int(reference.get("value", -1))
    reference_method = str(reference.get("source_method", ""))
    if reference_k in new_k_values or reference_k < 1 or not reference_method:
        raise ValueError(f"Invalid reference_k contract in {args.config}")
    root = args.output_dir or paths.run_root / "mogb_fixed_k_mean_ablation_v1"
    rows = build_rows(
        tuple(args.datasets), tuple(args.kirs), tuple(args.seeds), new_k_values
    )
    rows = apply_shard(rows, args.shard_index, args.shard_count)
    groups = group_rows(rows)
    suffix = "" if args.shard_index is None else f"_shard_{args.shard_index}_of_{args.shard_count}"
    plan_path = root / "plans" / f"fixed_k_mean_plan{suffix}.json"
    atomic_write_json(
        plan_path,
        {
            "protocol_version": paths.dataset_version,
            "planned_new_units": len(rows),
            "grouped_cells": len(groups),
            "rows": rows,
            "reference_reuse": {
                "k": reference_k,
                "method": reference_method,
                "source_root": str(paths.run_root / "mogb_baseline_v1"),
                "reference_units": len(groups),
            },
            "test_used_for_selection": False,
        },
    )
    if args.dry_run:
        for dataset, kir, seed in groups:
            validate_reference(
                paths, dataset, kir, seed, method=reference_method
            )
        print(
            json.dumps(
                {
                    "status": "planned",
                    "planned_new_units": len(rows),
                    "reference_units": len(groups),
                    "combined_units": len(rows) + len(groups),
                    "grouped_cells": len(groups),
                    "plan": str(plan_path),
                },
                sort_keys=True,
            )
        )
        return 0

    completed = skipped = 0
    failures: list[dict[str, Any]] = []
    validated_references: list[dict[str, Any]] = []
    for (dataset, kir, seed), cell_rows in groups.items():
        pending = []
        for row in cell_rows:
            manifest = (
                root
                / f"fixed_k{row['k']}"
                / dataset
                / f"kir_{kir:.2f}"
                / f"seed_{seed}"
                / "manifest.json"
            )
            if args.resume and manifest.is_file() and not args.overwrite:
                skipped += 1
            else:
                pending.append(row)
        existing_inputs: dict[str, Any] | None = None
        if not pending:
            existing_inputs = json.loads(
                (
                    root
                    / f"fixed_k{cell_rows[0]['k']}"
                    / dataset
                    / f"kir_{kir:.2f}"
                    / f"seed_{seed}"
                    / "inputs.json"
                ).read_text(encoding="utf-8")
            )
            validated_references.append(
                validate_reference(
                    paths,
                    dataset,
                    kir,
                    seed,
                    method=reference_method,
                    expected_inputs=existing_inputs,
                )
            )
            continue
        try:
            loaded = _load_cached_inputs(paths, dataset, seed, kir)
        except Exception as exc:  # keep protocol cells independently resumable
            failures.extend({**row, "error": f"{type(exc).__name__}: {exc}"} for row in pending)
            continue
        for row in pending:
            try:
                run_one_loaded(
                    paths,
                    {**base, **row},
                    loaded,
                    output_dir=args.output_dir,
                    overwrite=args.overwrite,
                )
                completed += 1
            except Exception as exc:  # record without fabricating a metric
                failures.append({**row, "error": f"{type(exc).__name__}: {exc}"})
        try:
            validated_references.append(
                validate_reference(
                    paths,
                    dataset,
                    kir,
                    seed,
                    method=reference_method,
                    expected_inputs=loaded[3],
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "seed": seed,
                    "k": reference_k,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    state_path = root / "plans" / f"fixed_k_mean_state{suffix}.json"
    atomic_write_json(
        state_path,
        {
            "protocol_version": paths.dataset_version,
            "planned_new_units": len(rows),
            "completed_this_invocation": completed,
            "skipped_complete": skipped,
            "validated_reference_units": len(validated_references),
            "validated_references": validated_references,
            "failed": failures,
            "test_used_for_selection": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete" if not failures else "completed_with_failures",
                "planned_new_units": len(rows),
                "completed_this_invocation": completed,
                "skipped_complete": skipped,
                "failed": len(failures),
                "plan": str(plan_path),
                "state": str(state_path),
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
