"""Declare the bounded R1 pilot without training or writing experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def build_plan(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    datasets = list(config["datasets"])
    seeds = list(config["seeds"])
    representations = list(config["representations"])
    k_values = list(config["k_values"])
    distances = list(config["distances"])
    units = len(datasets) * len(seeds) * len(representations) * len(k_values) * len(distances)
    return {
        "stage": "R1",
        "experiment_id": "r1_geometry_preserving_representation",
        "protocol_version": config["protocol_version"],
        "datasets": datasets,
        "kirs": [config["kir"]],
        "seeds": seeds,
        "representations": representations,
        "k_values": k_values,
        "distances": distances,
        "partition": "none",
        "boundary": config["boundary"],
        "planned_gate_units": units,
        "training_methods": ["ce_recon", "ce_recon_geometry"],
        "beta_selection": {
            "candidates": list(config["beta_candidates"]),
            "selection_seed": 42,
            "selection_data": "known_train_and_calibration_only",
        },
        "runs_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/protocol_v2_textoir_v1/r1_geometry_preserving.yaml"),
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
