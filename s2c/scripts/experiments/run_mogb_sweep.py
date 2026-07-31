"""Plan or run the protocol-aligned MOGB baseline sweep.

The sweep is intentionally a thin orchestrator around ``run_mogb_fair``. It
does not create registries, encodings, or alternate splits. Existing complete
cells are skipped, so ``--resume`` is safe after an interrupted run. The
official legacy mode is opt-in because its preflight is currently blocked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from protocol_v2.data.hashing import atomic_write_json
from protocol_v2.runtime.paths import ProtocolV2Paths
from run_mogb_fair import METHODS, run_one


DEFAULT_DATASETS = ("clinc150", "banking77", "stackoverflow")
DEFAULT_KIRS = (0.25, 0.50, 0.75)
# protocol_v2_textoir_v1 materializes these five seeds. Do not silently use
# the legacy 0..4 seed list: a caller must pass it explicitly if desired.
DEFAULT_SEEDS = (13, 42, 87, 100, 123)
DEFAULT_METHODS = tuple(method for method in METHODS if method != "mogb_official_reproduction")


def _plan_rows(datasets: tuple[str, ...], kirs: tuple[float, ...], seeds: tuple[int, ...], methods: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "protocol_version": "protocol_v2_textoir_v1",
            "dataset": dataset,
            "kir": kir,
            "seed": seed,
            "method": method,
            "test_used_for_selection": False,
        }
        for dataset in datasets
        for kir in kirs
        for seed in seeds
        for method in methods
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--kirs", nargs="+", type=float, default=list(DEFAULT_KIRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--include-official", action="store_true", help="add the blocked legacy official mode to the plan")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu", help="Recorded only; fair mode consumes cached embeddings")
    parser.add_argument("--dry-run", action="store_true", help="write the deterministic plan without running cells")
    parser.add_argument("--resume", action="store_true", help="skip complete cells and continue incomplete cells")
    parser.add_argument("--overwrite", action="store_true", help="replace cells under the MOGB artifact root")
    args = parser.parse_args(argv)

    methods = tuple(args.methods)
    if args.include_official and "mogb_official_reproduction" not in methods:
        methods = methods + ("mogb_official_reproduction",)
    datasets = tuple(args.datasets)
    kirs = tuple(float(value) for value in args.kirs)
    seeds = tuple(int(value) for value in args.seeds)
    paths = ProtocolV2Paths.discover()
    output_root = args.output_dir or paths.run_root / "mogb_baseline_v1"
    plan = _plan_rows(datasets, kirs, seeds, methods)
    plan_path = output_root / "plans" / "mogb_sweep_plan.json"
    atomic_write_json(
        plan_path,
        {
            "protocol_version": paths.dataset_version,
            "output_root": str(output_root),
            "device": args.device,
            "resume": bool(args.resume),
            "overwrite": bool(args.overwrite),
            "planned_units": len(plan),
            "rows": plan,
        },
    )
    if args.dry_run:
        print(json.dumps({"status": "planned", "planned_units": len(plan), "plan": str(plan_path)}, sort_keys=True))
        return 0

    completed = 0
    skipped = 0
    failed: list[dict[str, Any]] = []
    for row in plan:
        run_dir = output_root / row["dataset"] / f"kir_{row['kir']:.2f}" / f"seed_{row['seed']}" / row["method"]
        manifest = run_dir / "manifest.json"
        if manifest.is_file() and not args.overwrite:
            skipped += 1
            continue
        try:
            run_one(paths, row["dataset"], row["kir"], row["seed"], row["method"], output_root, args.overwrite)
            completed += 1
        except Exception as exc:  # keep independent cells auditable and resumable
            failed.append({**row, "error": f"{type(exc).__name__}: {exc}"})

    state_path = output_root / "plans" / "mogb_sweep_state.json"
    atomic_write_json(
        state_path,
        {
            "protocol_version": paths.dataset_version,
            "planned_units": len(plan),
            "completed_this_invocation": completed,
            "skipped_complete": skipped,
            "failed": failed,
            "test_used_for_selection": False,
        },
    )
    print(json.dumps({"status": "complete" if not failed else "completed_with_failures", "planned_units": len(plan), "completed_this_invocation": completed, "skipped_complete": skipped, "failed": len(failed), "plan": str(plan_path), "state": str(state_path)}, sort_keys=True))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
