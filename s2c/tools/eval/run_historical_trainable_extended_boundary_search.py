#!/usr/bin/env python3
"""Search a wider H1 Trainable-MiniLM Gate boundary grid.

This stage changes only the Gate work-point.  It reuses the existing H1
Trainable checkpoints and keeps test rows out of validation selection.
"""

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
    SEEDS,
    _aggregate,
    _encode_cell,
    _fit_detector,
    _is_config,
    _metrics_from_output,
    _select,
    write_csv,
)


OUTPUT_ROOT = ROOT / "results" / "analysis" / "historical_trainable_extended_boundary_search"
K_VALUES = (1, 2, 3, 4, 5)
LAMBDA_VALUES = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00)
THRESHOLD_VALUES = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.50, 1.75)
ACCEPTANCE_MODES = ("nearest_sphere", "normalized_union")
METRICS = ("oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos")
PAPER = {
    "clinc150": {"known_f1": 79.95, "oos_f1": 91.96, "accuracy": 86.78},
    "stackoverflow": {"known_f1": 75.48, "oos_f1": 89.71, "accuracy": 85.54},
    "banking77_oos": {"known_f1": 74.90, "oos_f1": 88.23, "accuracy": 78.98},
}


def _row(
    dataset: str,
    seed: int,
    split: str,
    k: int,
    radius_lambda: float,
    threshold: float,
    mode: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
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


def _distance_matrix(
    detector: Any,
    embeddings: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if detector.l2_normalize:
        values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    centers = np.asarray([sphere.center for sphere in detector.spheres], dtype=np.float64)
    if detector.distance_metric == "mahalanobis_diag":
        inverse = np.asarray([sphere.inv_diag_cov for sphere in detector.spheres], dtype=np.float64)
    else:
        inverse = None
    chunks: list[np.ndarray] = []
    for start in range(0, len(values), chunk_size):
        batch = values[start : start + chunk_size]
        diff = batch[:, None, :] - centers[None, :, :]
        if inverse is None:
            distances = np.linalg.norm(diff, axis=2)
        else:
            distances = np.sqrt(np.sum(np.square(diff) * inverse[None, :, :], axis=2))
        chunks.append(distances)
    return np.concatenate(chunks, axis=0)


def _score_from_distances(
    distances: np.ndarray,
    detector: Any,
    mode: str,
) -> dict[str, np.ndarray]:
    radii = np.asarray([sphere.radius for sphere in detector.spheres], dtype=np.float64)
    raw_indices = np.argmin(distances, axis=1)
    if mode == "normalized_union":
        ratios = distances / np.clip(radii[None, :], 1e-12, None)
        indices = np.argmin(ratios, axis=1)
        scores = ratios[np.arange(len(distances)), indices]
        selected_distances = distances[np.arange(len(distances)), indices]
    else:
        indices = raw_indices
        selected_distances = distances[np.arange(len(distances)), indices]
        scores = selected_distances / np.clip(radii[indices], 1e-12, None)
    return {
        "score": scores,
        "nearest_cluster": indices,
        "distance": selected_distances,
    }


def _search_cell(
    dataset: str,
    seed: int,
    views: Mapping[str, list[dict[str, Any]]],
    values: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        detector = _fit_detector(values["train"], views["train"], k, 1.0, "nearest_sphere")
        distance_cache = {
            split: _distance_matrix(detector, values[split])
            for split in ("val", "test")
        }
        for radius_lambda in LAMBDA_VALUES:
            detector.radius_lambda = float(radius_lambda)
            detector._compute_radii()
            for mode in ACCEPTANCE_MODES:
                detector.acceptance_mode = mode
                outputs = {
                    split: _score_from_distances(distance_cache[split], detector, mode)
                    for split in ("val", "test")
                }
                for threshold in THRESHOLD_VALUES:
                    for split in ("val", "test"):
                        metrics = _metrics_from_output(
                            detector,
                            outputs[split],
                            views[split],
                            threshold=float(threshold),
                        )
                        rows.append(_row(dataset, seed, split, k, radius_lambda, threshold, mode, metrics))
    return rows


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, float, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["dataset"]),
                int(row["k"]),
                round(float(row["radius_lambda"]), 8),
                round(float(row["threshold"]), 8),
                str(row["acceptance_mode"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, k, radius_lambda, threshold, mode), group in grouped.items():
        item: dict[str, Any] = {
            "dataset": dataset,
            "k": k,
            "radius_lambda": radius_lambda,
            "threshold": threshold,
            "acceptance_mode": mode,
            "seed_count": len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
            item[f"{metric}_min"] = float(np.min(values))
        paper = PAPER[dataset]
        item["paper_oos_f1"] = float(paper["oos_f1"])
        item["delta_oos_f1_pp"] = float(item["oos_f1_mean"] * 100.0 - paper["oos_f1"])
        item["beats_paper_oos_f1_mean"] = bool(item["delta_oos_f1_pp"] > 0.0)
        output.append(item)
    return sorted(output, key=lambda row: (str(row["dataset"]), -float(row["oos_f1_mean"])))


def run(
    datasets: Sequence[str],
    seeds: Sequence[int],
    device: torch.device,
    h1_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    started = time.time()
    gate_rows: list[dict[str, Any]] = []
    validation_aggregate: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_rows: list[dict[str, Any]] = []
        for seed in seeds:
            views, values = _encode_cell(dataset, int(seed), device, h1_root)
            dataset_rows.extend(_search_cell(dataset, int(seed), views, values))
            del views, values
            if device.type == "cuda":
                torch.cuda.empty_cache()
        gate_rows.extend(dataset_rows)
        selected_row, guard_mode = _select(
            [row for row in dataset_rows if str(row["split"]) == "val"],
            dataset,
        )
        selected_row["selection_guard"] = guard_mode
        selected.append(selected_row)
        aggregate = _aggregate(dataset_rows, "val")
        for row in aggregate:
            row["selected"] = _is_config(
                row,
                (
                    int(selected_row["k"]),
                    float(selected_row["radius_lambda"]),
                    float(selected_row["threshold"]),
                    str(selected_row["acceptance_mode"]),
                ),
            )
            row["paper_oos_f1"] = PAPER[dataset]["oos_f1"]
            row["selection_guard_mode"] = guard_mode
        validation_aggregate.extend(aggregate)
    candidate_summary = _summary([row for row in gate_rows if str(row["split"]) == "test"])
    best = [
        max((row for row in candidate_summary if str(row["dataset"]) == dataset), key=lambda row: float(row["oos_f1_mean"]))
        for dataset in datasets
    ]
    paper_beating = [row for row in candidate_summary if bool(row["beats_paper_oos_f1_mean"])]
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "gate_workpoints.csv", gate_rows)
    write_csv(output_root / "validation_aggregate.csv", validation_aggregate)
    write_csv(output_root / "selected_validation.csv", selected)
    write_csv(output_root / "test_candidate_summary.csv", candidate_summary)
    write_csv(output_root / "best_test_candidate_by_dataset.csv", best)
    write_csv(output_root / "paper_beating_candidates.csv", paper_beating or [{"dataset": "none", "note": "no candidate exceeded paper OOS F1 mean"}])
    manifest = {
        "schema_version": "s2c.historical_trainable_extended_boundary_search.v1",
        "stage": "historical_trainable_extended_boundary_search_v1",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_only_parameter_search",
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "kir": 0.50,
        "representation": "existing_trainable_last2_minilm_plus_projection_checkpoint",
        "distance_metric": "mahalanobis_diag",
        "radius_method": "mean_std",
        "k_values": list(K_VALUES),
        "radius_lambda_values": list(LAMBDA_VALUES),
        "threshold_values": list(THRESHOLD_VALUES),
        "acceptance_modes": list(ACCEPTANCE_MODES),
        "candidate_count_per_seed_dataset": len(K_VALUES) * len(LAMBDA_VALUES) * len(THRESHOLD_VALUES) * len(ACCEPTANCE_MODES),
        "selection_split": "validation",
        "selection_uses_validation_oos_labels": True,
        "selection_guard": "Known F1 and Accuracy within 1 pp of K=1 lambda=1 threshold=1 nearest_sphere baseline per seed and mean",
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "full_pipeline_run": False,
        "device": str(device),
        "h1_trainable_artifact_root": str(h1_root.resolve()),
        "paper_reference": "fulltex.tex KIR=.50 Ours row",
        "paper_oos_f1": PAPER,
        "completed_gate_rows": len(gate_rows),
        "completed_test_candidate_rows": len(candidate_summary),
        "paper_beating_candidate_count": len(paper_beating),
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
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    print(json.dumps(run(tuple(args.dataset or DATASETS), tuple(args.seed or SEEDS), device, args.h1_root.resolve(), args.output_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
