#!/usr/bin/env python3
"""Run RACAL-v1 stage 2: Trainable MiniLM K=1 versus fixed K=2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol_v2.data.hashing import atomic_write_json
from protocol_v2.experiments.racal_v1.stage2 import SEEDS, make_provenance, load_stage2_config, run_seed, stage2_root
from protocol_v2.runtime.paths import ProtocolV2Paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = ProtocolV2Paths.discover()
    config_path = args.config.resolve()
    config = load_stage2_config(config_path)
    paths.require_experiment_admission("stackoverflow")
    root = stage2_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    provenance_path = root / "RACAL_STAGE2_PROVENANCE.json"
    if not provenance_path.is_file():
        atomic_write_json(provenance_path, make_provenance(paths, config_path, config))
    selected = tuple(args.seeds or ((args.seed,) if args.seed is not None else SEEDS))
    if any(seed not in SEEDS for seed in selected):
        raise ValueError(f"Stage2 seeds must be drawn from {SEEDS}: {selected}")
    if args.dry_run:
        plan = {
            "status": "preflight_ok",
            "stage": "racal_v1_stage2_fixed_k2",
            "protocol_version": "protocol_v2_textoir_v1",
            "dataset": "stackoverflow",
            "kir": 0.50,
            "seeds": list(selected),
            "k_values": [1, 2],
            "distance": "mahalanobis_diag",
            "radius_method": "mean_std",
            "radius_lambda": 1.0,
            "threshold": 1.0,
            "partition_seed": 42,
            "test_used_for_selection": False,
            "oos_used_for_training": False,
            "planned_runs": len(selected),
        }
        atomic_write_json(root / "plans" / "stage2_plan.json", plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    results = [run_seed(paths, config, int(seed), resume=args.resume) for seed in selected]
    state = {"status": "complete", "stage": "racal_v1_stage2_fixed_k2", "seeds": list(selected), "completed": len(results), "test_used_for_selection": False}
    atomic_write_json(root / "state" / "stage2_state.json", state)
    print(json.dumps({"status": "complete", "completed": len(results), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
