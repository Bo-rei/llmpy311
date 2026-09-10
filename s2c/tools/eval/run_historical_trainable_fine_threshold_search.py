#!/usr/bin/env python3
"""Search fine-grained Gate thresholds for the historical H1 Trainable model."""

from __future__ import annotations

import argparse
import json
import sys
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
    _aggregate,
    _encode_cell,
    _fit_detector,
    _is_config,
    _metrics_from_output,
    write_csv,
)
from tools.eval.run_historical_trainable_extended_boundary_search import (  # noqa: E402
    _distance_matrix,
    _score_from_distances,
)


OUTPUT_ROOT = ROOT / "results" / "analysis" / "historical_trainable_fine_threshold_search"
K_VALUES = (1, 2, 3)
LAMBDA_VALUES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00)
THRESHOLD_VALUES = tuple(round(float(value), 2) for value in np.arange(0.80, 1.201, 0.01))
ACCEPTANCE_MODES = ("nearest_sphere", "normalized_union")
BASELINE = (1, 1.0, 1.0, "nearest_sphere")


def _key(row: Mapping[str, Any]) -> tuple[int, float, float, str]:
    return int(row["k"]), float(row["radius_lambda"]), float(row["threshold"]), str(row["acceptance_mode"])


def _row(dataset: str, seed: int, split: str, k: int, radius_lambda: float, threshold: float, mode: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed": int(seed),
        "split": split,
        "k": int(k),
        "radius_lambda": float(radius_lambda),
        "threshold": float(threshold),
        "acceptance_mode": str(mode),
        "oos_f1": float(metrics["f1_u"]),
        "known_f1": float(metrics["f1_k"]),
        "accuracy": float(metrics["accuracy"]),
        "known_recall": float(metrics["known_recall"]),
        "false_accept_rate": float(metrics["false_accept_rate"]),
        "false_reject_rate": float(metrics["false_reject_rate"]),
        "auroc": float(metrics["auroc"]),
        "aupr_oos": float(metrics["aupr_oos"]),
    }


def _search_cell(dataset: str, seed: int, views: Mapping[str, list[dict[str, Any]]], values: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        detector = _fit_detector(values["train"], views["train"], k, 1.0, "nearest_sphere")
        distances = {split: _distance_matrix(detector, values[split]) for split in ("val", "test")}
        for radius_lambda in LAMBDA_VALUES:
            detector.radius_lambda = float(radius_lambda)
            detector._compute_radii()
            for mode in ACCEPTANCE_MODES:
                detector.acceptance_mode = mode
                outputs = {split: _score_from_distances(distances[split], detector, mode) for split in ("val", "test")}
                for threshold in THRESHOLD_VALUES:
                    for split in ("val", "test"):
                        metrics = _metrics_from_output(detector, outputs[split], views[split], threshold)
                        rows.append(_row(dataset, seed, split, k, radius_lambda, threshold, mode, metrics))
    return rows


def _select_per_seed(rows: Sequence[Mapping[str, Any]], dataset: str, seed: int) -> dict[str, Any]:
    validation = [row for row in rows if str(row["dataset"]) == dataset and int(row["seed"]) == int(seed) and str(row["split"]) == "val"]
    baseline = next(row for row in validation if _key(row) == BASELINE)
    eligible = [row for row in validation if float(row["known_f1"]) >= float(baseline["known_f1"]) - 0.01 and float(row["accuracy"]) >= float(baseline["accuracy"]) - 0.01]
    return dict(
        max(
            eligible,
            key=lambda row: (
                float(row["oos_f1"]),
                float(row["known_f1"]),
                float(row["accuracy"]),
                1 if row["acceptance_mode"] == "nearest_sphere" else 0,
                -int(row["k"]),
                -abs(float(row["radius_lambda"]) - 1.0),
                -abs(float(row["threshold"]) - 1.0),
            ),
        )
    )


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["k"]), float(row["radius_lambda"]), float(row["threshold"]), str(row["acceptance_mode"]))].append(row)
    output = []
    for (dataset, k, radius_lambda, threshold, mode), group in grouped.items():
        item: dict[str, Any] = {"dataset": dataset, "k": k, "radius_lambda": radius_lambda, "threshold": threshold, "acceptance_mode": mode, "seed_count": len(group)}
        for metric in ("oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
        item["paper_oos_f1"] = PAPER[dataset]["oos_f1"]
        item["delta_oos_f1_pp"] = float(item["oos_f1_mean"] * 100.0 - PAPER[dataset]["oos_f1"])
        item["beats_paper_oos_f1_mean"] = bool(item["delta_oos_f1_pp"] > 0.0)
        output.append(item)
    return sorted(output, key=lambda row: (str(row["dataset"]), -float(row["oos_f1_mean"])))


def _dataset_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    output = []
    for dataset, group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "dataset": dataset,
            "selection_scope": "per_seed_validation",
            "seed_count": len(group),
            "selected_configurations": "; ".join(
                f"seed{row['seed']}:K={row['k']},lambda={row['radius_lambda']},threshold={row['threshold']},{row['acceptance_mode']}"
                for row in group
            ),
        }
        for metric in ("oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
        item["paper_oos_f1"] = PAPER[dataset]["oos_f1"]
        item["delta_oos_f1_pp"] = float(item["oos_f1_mean"] * 100.0 - PAPER[dataset]["oos_f1"])
        item["beats_paper_oos_f1_mean"] = bool(item["delta_oos_f1_pp"] > 0.0)
        output.append(item)
    return output


def run(datasets: Sequence[str], seeds: Sequence[int], device: torch.device, h1_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.time()
    all_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    per_seed_selected_test: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_rows: list[dict[str, Any]] = []
        for seed in seeds:
            views, values = _encode_cell(dataset, int(seed), device, h1_root)
            dataset_rows.extend(_search_cell(dataset, int(seed), views, values))
            del views, values
            if device.type == "cuda":
                torch.cuda.empty_cache()
        all_rows.extend(dataset_rows)
        selected_rows = [_select_per_seed(dataset_rows, dataset, int(seed)) for seed in seeds]
        for row in selected_rows:
            row["selection_scope"] = "per_seed_validation"
            row["test_used_for_selection"] = False
        selected.extend(selected_rows)
        # Gate-only test confirmation for the corresponding validation choice.
        for seed in seeds:
            choice = next(row for row in selected_rows if int(row["seed"]) == int(seed))
            test = next(row for row in dataset_rows if str(row["split"]) == "test" and int(row["seed"]) == int(seed) and _is_config(row, _key(choice)))
            per_seed_selected_test.append({**dict(test), "selection_scope": "per_seed_validation", "test_used_for_selection": False, "paper_oos_f1": PAPER[dataset]["oos_f1"], "delta_oos_f1_pp": float(float(test["oos_f1"]) * 100.0 - PAPER[dataset]["oos_f1"]), "beats_paper_oos_f1": bool(float(test["oos_f1"]) * 100.0 > PAPER[dataset]["oos_f1"])})
    test_summary = _dataset_summary(per_seed_selected_test)
    best_test = [max((row for row in _summary([item for item in all_rows if item["split"] == "test"]) if row["dataset"] == dataset), key=lambda row: float(row["oos_f1_mean"])) for dataset in datasets]
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "gate_workpoints.csv", all_rows)
    write_csv(output_root / "selected_validation_per_seed.csv", selected)
    write_csv(output_root / "selected_test_per_seed.csv", per_seed_selected_test)
    write_csv(output_root / "selected_test_summary.csv", test_summary)
    write_csv(output_root / "best_test_candidate_by_dataset.csv", best_test)
    manifest = {
        "schema_version": "s2c.historical_trainable_fine_threshold_search.v1",
        "stage": "historical_trainable_fine_threshold_search_v1",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_only_parameter_search",
        "datasets": list(datasets), "seeds": [int(seed) for seed in seeds], "kir": 0.50,
        "representation": "existing_trainable_last2_minilm_plus_projection_checkpoint",
        "distance_metric": "mahalanobis_diag", "radius_method": "mean_std",
        "k_values": list(K_VALUES), "radius_lambda_values": list(LAMBDA_VALUES), "threshold_values": list(THRESHOLD_VALUES), "acceptance_modes": list(ACCEPTANCE_MODES),
        "candidate_count_per_seed_dataset": len(K_VALUES) * len(LAMBDA_VALUES) * len(THRESHOLD_VALUES) * len(ACCEPTANCE_MODES),
        "selection_split": "validation", "selection_scope": "per_seed", "selection_guard": "same-seed Known F1 and Accuracy within 1 pp of K=1 lambda=1 threshold=1 nearest_sphere baseline", "test_used_for_selection": False, "oos_used_for_training": False, "full_pipeline_run": False,
        "device": str(device), "h1_trainable_artifact_root": str(h1_root.resolve()), "paper_reference": "fulltex.tex KIR=.50 Ours row", "paper_oos_f1": PAPER,
        "completed_gate_rows": len(all_rows), "completed_selected_test_units": len(per_seed_selected_test), "elapsed_seconds": time.time() - started,
    }
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--seed", choices=SEEDS, type=int, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--h1-root", type=Path, default=DEFAULT_H1_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    print(json.dumps(run(tuple(args.dataset or DATASETS), tuple(args.seed or SEEDS), device, args.h1_root.resolve(), args.output_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
