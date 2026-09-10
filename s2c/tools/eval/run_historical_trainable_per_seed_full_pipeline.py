#!/usr/bin/env python3
"""Run per-seed validation-selected Trainable-Gate full-pipeline results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.eval.run_historical_trainable_parameter_full_pipeline import (  # noqa: E402
    DATASETS,
    DEFAULT_H1_ROOT,
    PAPER,
    SEEDS,
    _encode_cell,
    _fit_detector,
    _is_config,
    _run_pipeline,
    compute_metrics,
    write_csv,
)


SOURCE_ROOT = ROOT / "results" / "analysis" / "historical_trainable_parameter_full_pipeline"
OUTPUT_ROOT = ROOT / "results" / "analysis" / "historical_trainable_per_seed_full_pipeline"
BASELINE = (1, 1.0, 1.0, "nearest_sphere")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _key(row: Mapping[str, Any]) -> tuple[int, float, float, str]:
    return int(row["k"]), float(row["radius_lambda"]), float(row["threshold"]), str(row["acceptance_mode"])


def _select(rows: Sequence[Mapping[str, Any]], dataset: str, seed: int) -> dict[str, Any]:
    validation = [row for row in rows if str(row["dataset"]) == dataset and int(row["seed"]) == seed and str(row["split"]) == "val"]
    baseline = next(row for row in validation if _key(row) == BASELINE)
    eligible = [
        row for row in validation
        if float(row["known_f1"]) >= float(baseline["known_f1"]) - 0.01
        and float(row["accuracy"]) >= float(baseline["accuracy"]) - 0.01
    ]
    if not eligible:
        raise ValueError(f"no per-seed eligible candidate for {dataset}/seed{seed}")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["oos_f1"]),
            float(row["known_f1"]),
            float(row["accuracy"]),
            1 if str(row["acceptance_mode"]) == "nearest_sphere" else 0,
            -int(row["k"]),
            -abs(float(row["radius_lambda"]) - 1.0),
            -abs(float(row["threshold"]) - 1.0),
        ),
    )
    return {
        **dict(selected),
        "selection_scope": "per_seed_validation",
        "selection_uses_validation_oos_labels": True,
        "test_used_for_selection": False,
        "baseline_validation_oos_f1": float(baseline["oos_f1"]),
        "baseline_validation_known_f1": float(baseline["known_f1"]),
        "baseline_validation_accuracy": float(baseline["accuracy"]),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    result = []
    for dataset, group in sorted(grouped.items()):
        paper = PAPER[dataset]
        item: dict[str, Any] = {"dataset": dataset, "selection_scope": "per_seed_validation", "seed_count": len(group)}
        for metric in ("oos_f1", "f1_all", "known_macro_f1", "overall_accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "router_error_rate", "expert_error_rate"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
        item["paper_oos_f1"] = paper["oos_f1"]
        item["delta_oos_f1_pp"] = float(item["oos_f1_mean"] * 100.0 - paper["oos_f1"])
        item["beats_paper_oos_f1_mean"] = bool(item["delta_oos_f1_pp"] > 0.0)
        result.append(item)
    return result


def run(
    datasets: Sequence[str],
    seeds: Sequence[int],
    device: torch.device,
    h1_root: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    started = time.time()
    gate_rows = _read_csv(source_root / "gate_workpoints.csv")
    selected_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="s2c_per_seed_pipeline_") as temporary:
        temporary_root = Path(temporary)
        for dataset in datasets:
            for seed in seeds:
                selected = _select(gate_rows, dataset, int(seed))
                views, values = _encode_cell(dataset, int(seed), device, h1_root)
                config = _key(selected)
                detector = _fit_detector(values["train"], views["train"], config[0], config[1], config[3])
                predictions, metadata = _run_pipeline(
                    dataset,
                    int(seed),
                    device,
                    detector,
                    config[2],
                    h1_root,
                    temporary_root,
                )
                metrics, stages = compute_metrics(views["test"], predictions)
                gate_test = next(
                    row for row in gate_rows
                    if str(row["dataset"]) == dataset
                    and int(row["seed"]) == int(seed)
                    and str(row["split"]) == "test"
                    and _is_config(row, config)
                )
                gate_oos_delta = abs(float(gate_test["oos_f1"]) - float(metrics["oos_f1"]))
                if gate_oos_delta > 1e-10:
                    raise AssertionError(f"Gate/full pipeline OOS mismatch: {dataset}/seed{seed}: {gate_oos_delta}")
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "seed": int(seed),
                    "k": config[0],
                    "radius_lambda": config[1],
                    "threshold": config[2],
                    "acceptance_mode": config[3],
                    "selection_scope": "per_seed_validation",
                    "validation_oos_f1": float(selected["oos_f1"]),
                    "validation_known_f1": float(selected["known_f1"]),
                    "validation_accuracy": float(selected["accuracy"]),
                    "test_used_for_selection": False,
                    "direct_pipeline_verified": True,
                    "direct_prediction_count": int(metadata["prediction_count"]),
                    "paper_known_f1": PAPER[dataset]["known_f1"],
                    "paper_accuracy": PAPER[dataset]["accuracy"],
                    **{key: float(metrics[key]) for key in ("oos_f1", "f1_all", "known_macro_f1", "overall_accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "router_error_rate", "expert_error_rate")},
                    "paper_oos_f1": PAPER[dataset]["oos_f1"],
                    "delta_oos_f1_pp": float(metrics["oos_f1"] * 100.0 - PAPER[dataset]["oos_f1"]),
                    "beats_paper_oos_f1": bool(metrics["oos_f1"] * 100.0 > PAPER[dataset]["oos_f1"]),
                }
                row.update({f"stage_count_{key}": int(value) for key, value in stages.items()})
                selected_rows.append(row)
                del predictions, detector, views, values
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "per_seed.csv", selected_rows)
    write_csv(output_root / "summary.csv", _summary(selected_rows))
    write_csv(output_root / "paper_comparison.csv", _summary(selected_rows))
    manifest = {
        "schema_version": "s2c.historical_trainable_per_seed_full_pipeline.v1",
        "stage": "historical_trainable_per_seed_full_pipeline_v1",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert",
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "kir": 0.50,
        "selection_scope": "per_seed_validation",
        "selection_split": "validation",
        "selection_guard": "Known F1 and Accuracy within 1 pp of same-seed K=1 lambda=1 threshold=1 nearest_sphere baseline",
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "downstream_router_expert_fixed": True,
        "device": str(device),
        "source_gate_workpoints": str((source_root / "gate_workpoints.csv").resolve()),
        "h1_trainable_artifact_root": str(h1_root.resolve()),
        "paper_reference": "fulltex.tex KIR=.50 Ours row",
        "paper_oos_f1": PAPER,
        "completed_units": len(selected_rows),
        "direct_pipeline_verified_units": sum(bool(row["direct_pipeline_verified"]) for row in selected_rows),
        "paper_beating_datasets": sorted({str(row["dataset"]) for row in selected_rows if bool(row["beats_paper_oos_f1"])}),
        "elapsed_seconds": time.time() - started,
    }
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--seed", choices=SEEDS, type=int, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--h1-root", type=Path, default=DEFAULT_H1_ROOT)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    print(json.dumps(run(tuple(args.dataset or DATASETS), tuple(args.seed or SEEDS), device, args.h1_root.resolve(), args.source_root.resolve(), args.output_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
