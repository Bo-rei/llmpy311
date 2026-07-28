"""Create the R1_full plan without training or touching frozen experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def build_plan(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    datasets = list(config["datasets"])
    kirs = list(config["kirs"])
    seeds = list(config["seeds"])
    representations = list(config["representations"])
    k_values = list(config["k_values"])
    cell_units = len(datasets) * len(kirs) * len(seeds) * len(representations)
    gate_units = cell_units * len(k_values)
    return {
        "stage": "R1_full",
        "experiment_id": "r1_geometry_preserving_representation_full",
        "status": "plan_only_not_started",
        "protocol_version": config["protocol_version"],
        "datasets": datasets,
        "kirs": kirs,
        "seeds": seeds,
        "representations": representations,
        "k_values": k_values,
        "distance_by_dataset": config["distance_by_dataset"],
        "boundary": config["boundary"],
        "radius_lambda": config["radius_lambda"],
        "beta": config["beta"],
        "beta_source": config["beta_source"],
        "planned_cell_units": cell_units,
        "planned_gate_units": gate_units,
        "training_methods": ["ce_recon", "ce_recon_geometry"],
        "selection": "known_train_calibration_only; beta frozen from R1 pilot",
        "restrictions": [
            "no_oos_training",
            "no_test_based_checkpoint_or_beta_selection",
            "no_e2_e3_mutation",
            "no_adb_da_adb_mogb",
            "no_pipeline",
        ],
        "runs_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/protocol_v2_textoir_v1/r1_geometry_preserving_full.yaml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build_plan(args.config)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
