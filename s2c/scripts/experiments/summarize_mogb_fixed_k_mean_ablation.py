"""Summarize the fixed-K frozen-MiniLM MOGB mean-radius ablation."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file, sha256_json
from protocol_v2.runtime.paths import ProtocolV2Paths


DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
K_VALUES = (1, 2, 3, 4)
NEW_K_VALUES = (1, 3, 4)
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_RESAMPLES = 10_000
PROTOCOL_VERSION = "protocol_v2_textoir_v1"
STAGE = "mogb_fixed_k_mean_ablation_v1"
REUSE_METHOD = "ours_partition_mogb_boundary"
ADAPTIVE_METHOD = "mogb_minilm"
SINGLE_METHOD = "single_centroid"
METRICS = (
    "oos_f1",
    "oos_precision",
    "oos_recall",
    "f1_all",
    "f1_u",
    "f1_k",
    "accuracy",
    "id_recall",
    "auroc",
    "aupr_oos",
    "fpr95",
    "false_accept_rate",
    "false_reject_rate",
    "effective_cluster_count",
    "minimum_cluster_size",
    "scoring_seconds",
    "samples_per_second",
)
COMPARISON_METRICS = (
    "oos_f1",
    "f1_all",
    "id_recall",
    "false_accept_rate",
    "false_reject_rate",
    "effective_cluster_count",
)
BALL_FIELDS = (
    "selected_balls",
    "total_balls",
    "filtered_balls",
    "mean_balls_per_intent",
    "min_balls_per_intent",
    "median_balls_per_intent",
    "max_balls_per_intent",
    "mean_ball_radius",
    "mean_ball_samples",
    "mean_ball_purity",
    "max_tree_depth",
)


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_metric_dict(metrics: dict[str, Any]) -> dict[str, float]:
    values = {metric: _safe_float(metrics.get(metric)) for metric in METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non_finite_metric")
    return values


def _bootstrap_ci(deltas: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    if deltas.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(deltas, size=(BOOTSTRAP_RESAMPLES, deltas.size), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def _fixed_run_dir(root: Path, dataset: str, kir: float, seed: int, k: int) -> Path:
    return root / f"fixed_k{k}" / dataset / f"kir_{kir:.2f}" / f"seed_{seed}"


def _baseline_run_dir(root: Path, dataset: str, kir: float, seed: int, method: str) -> Path:
    return root / dataset / f"kir_{kir:.2f}" / f"seed_{seed}" / method


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _ball_stats(ball_statistics: dict[str, Any], method_details: dict[str, Any]) -> dict[str, Any]:
    source = dict(ball_statistics)
    nested = method_details.get("ball_statistics")
    if isinstance(nested, dict):
        source = {**nested, **source}
    balls_per_intent = source.get("balls_per_intent")
    if isinstance(balls_per_intent, dict) and balls_per_intent:
        counts = [int(value) for value in balls_per_intent.values()]
    else:
        counts = []
    return {
        "selected_balls": int(source.get("selected_balls", method_details.get("cluster_count", 0))),
        "total_balls": int(source.get("total_balls", source.get("selected_balls", method_details.get("cluster_count", 0)))),
        "filtered_balls": int(source.get("filtered_balls", 0)),
        "mean_balls_per_intent": _safe_float(source.get("mean_balls_per_intent")),
        "min_balls_per_intent": int(source.get("min_balls_per_intent", min(counts))) if counts else math.nan,
        "median_balls_per_intent": _safe_float(source.get("median_balls_per_intent", statistics.median(counts))) if counts else math.nan,
        "max_balls_per_intent": int(source.get("max_balls_per_intent", max(counts))) if counts else math.nan,
        "mean_ball_radius": _safe_float(source.get("mean_radius")),
        "mean_ball_samples": _safe_float(source.get("mean_samples_per_ball")),
        "mean_ball_purity": _safe_float(source.get("mean_purity")),
        "max_tree_depth": _safe_float(source.get("max_tree_depth")),
    }


def _planned_fixed_cells() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                for k in K_VALUES:
                    rows.append(
                        {
                            "dataset": dataset,
                            "kir": float(kir),
                            "seed": int(seed),
                            "k": int(k),
                            "reused_reference": k == 2,
                        }
                    )
    return rows


def _planned_reference_cells(method: str) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "kir": float(kir),
            "seed": int(seed),
            "reference_method": method,
        }
        for dataset in DATASETS
        for kir in KIRS
        for seed in SEEDS
    ]


def _collect_fixed_row(
    *,
    fixed_root: Path,
    baseline_root: Path,
    cell: dict[str, Any],
    failures: list[dict[str, Any]],
    manifest_hashes: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset = str(cell["dataset"])
    kir = float(cell["kir"])
    seed = int(cell["seed"])
    k = int(cell["k"])
    reused_reference = bool(cell["reused_reference"])
    if reused_reference:
        run_dir = _baseline_run_dir(baseline_root, dataset, kir, seed, REUSE_METHOD)
        source_stage = "mogb_baseline_v1"
        source_method = REUSE_METHOD
    else:
        run_dir = _fixed_run_dir(fixed_root, dataset, kir, seed, k)
        source_stage = STAGE
        source_method = f"fixed_k{k}_mogb_mean"
    row: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "summary_stage": STAGE,
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "k": k,
        "representation": "frozen_minilm",
        "distance": "euclidean",
        "boundary": "mean",
        "acceptance": "nearest_ball",
        "test_used_for_selection": False,
        "source_stage": source_stage,
        "source_method": source_method,
        "source_run_dir": str(run_dir.relative_to(fixed_root.parent)),
        "source_root": "mogb_baseline_v1" if reused_reference else STAGE,
        "reused_reference": reused_reference,
    }
    required = ("manifest.json", "config.json", "metrics.json", "method_details.json", "ball_statistics.json")
    missing_files = [name for name in required if not (run_dir / name).is_file()]
    if missing_files:
        row["status"] = "missing"
        row["failure_reason"] = f"missing_required_files:{','.join(missing_files)}"
        failures.append(
            {
                "kind": "fixed_k",
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "k": k,
                "source_method": source_method,
                "reason": row["failure_reason"],
                "run_dir": str(run_dir),
            }
        )
        return row

    manifest_path = run_dir / "manifest.json"
    try:
        manifest = _load_json(manifest_path)
        config = _load_json(run_dir / "config.json")
        metrics = _load_json(run_dir / "metrics.json")
        method_details = _load_json(run_dir / "method_details.json")
        ball_statistics = _load_json(run_dir / "ball_statistics.json")
        metric_values = _finite_metric_dict(metrics)
        if manifest.get("status") != "complete":
            raise ValueError("manifest_not_complete")
        if manifest.get("test_used_for_selection") is not False:
            raise ValueError("test_selection_mismatch")
        if config.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("protocol_mismatch")
        if config.get("dataset") != dataset or int(config.get("seed")) != seed or float(config.get("kir")) != kir:
            raise ValueError("dataset_cell_mismatch")
        if reused_reference:
            if config.get("method") != REUSE_METHOD:
                raise ValueError("reference_method_mismatch")
            if method_details.get("partition") != "ours_fixed_k2":
                raise ValueError("reference_partition_mismatch")
            source_partition = "ours_fixed_k2"
            partition_seed = 42
        else:
            if int(config.get("k")) != k or int(manifest.get("k")) != k:
                raise ValueError("k_mismatch")
            if str(config.get("partition")) not in {
                "fixed_per_intent_kmeans",
                "per_intent_kmeans",
            }:
                raise ValueError("partition_mismatch")
            if str(config.get("distance")) != "euclidean" or str(config.get("boundary")) != "mean":
                raise ValueError("boundary_distance_mismatch")
            source_partition = str(config.get("partition"))
            partition_seed = int(config.get("partition_seed", 42))
        row.update(metric_values)
        row.update(_ball_stats(ball_statistics, method_details))
        row["status"] = "complete"
        row["failure_reason"] = ""
        row["partition"] = "per_intent_kmeans"
        row["source_partition"] = source_partition
        row["partition_seed"] = partition_seed
        row["cluster_partition_count"] = int(method_details.get("cluster_count", metric_values["effective_cluster_count"]))
        manifest_hashes.append(
            {
                "kind": "fixed_k",
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "k": k,
                "source_method": source_method,
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        return row
    except Exception as exc:
        reason = str(exc)
        row["status"] = "invalid"
        row["failure_reason"] = reason
        failures.append(
            {
                "kind": "fixed_k",
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "k": k,
                "source_method": source_method,
                "reason": reason,
                "run_dir": str(run_dir),
            }
        )
        return row


def _collect_reference_row(
    *,
    baseline_root: Path,
    cell: dict[str, Any],
    failures: list[dict[str, Any]],
    manifest_hashes: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset = str(cell["dataset"])
    kir = float(cell["kir"])
    seed = int(cell["seed"])
    method = str(cell["reference_method"])
    run_dir = _baseline_run_dir(baseline_root, dataset, kir, seed, method)
    row: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "reference_method": method,
        "source_stage": "mogb_baseline_v1",
        "source_run_dir": str(run_dir.relative_to(baseline_root.parent)),
        "source_root": "mogb_baseline_v1",
        "reused_reference": False,
        "test_used_for_selection": False,
    }
    required = ("manifest.json", "config.json", "metrics.json", "method_details.json", "ball_statistics.json")
    missing_files = [name for name in required if not (run_dir / name).is_file()]
    if missing_files:
        row["status"] = "missing"
        row["failure_reason"] = f"missing_required_files:{','.join(missing_files)}"
        failures.append(
            {
                "kind": "reference",
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "reference_method": method,
                "reason": row["failure_reason"],
                "run_dir": str(run_dir),
            }
        )
        return row

    manifest_path = run_dir / "manifest.json"
    try:
        manifest = _load_json(manifest_path)
        config = _load_json(run_dir / "config.json")
        metrics = _load_json(run_dir / "metrics.json")
        method_details = _load_json(run_dir / "method_details.json")
        ball_statistics = _load_json(run_dir / "ball_statistics.json")
        metric_values = _finite_metric_dict(metrics)
        if manifest.get("status") != "complete":
            raise ValueError("manifest_not_complete")
        if manifest.get("test_used_for_selection") is not False:
            raise ValueError("test_selection_mismatch")
        if config.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("protocol_mismatch")
        if config.get("dataset") != dataset or int(config.get("seed")) != seed or float(config.get("kir")) != kir:
            raise ValueError("dataset_cell_mismatch")
        if config.get("method") != method:
            raise ValueError("reference_method_mismatch")
        row.update(metric_values)
        row.update(_ball_stats(ball_statistics, method_details))
        row["status"] = "complete"
        row["failure_reason"] = ""
        row["partition"] = str(method_details.get("partition", config.get("method", method)))
        row["cluster_partition_count"] = int(method_details.get("cluster_count", metric_values["effective_cluster_count"]))
        manifest_hashes.append(
            {
                "kind": "reference",
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "reference_method": method,
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
        return row
    except Exception as exc:
        reason = str(exc)
        row["status"] = "invalid"
        row["failure_reason"] = reason
        failures.append(
            {
                "kind": "reference",
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "reference_method": method,
                "reason": reason,
                "run_dir": str(run_dir),
            }
        )
        return row


def _paired_rows(
    fixed_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    reference_name: str,
) -> list[dict[str, Any]]:
    complete_fixed = {
        (str(row["dataset"]), float(row["kir"]), int(row["seed"]), int(row["k"])): row
        for row in fixed_rows
        if row.get("status") == "complete"
    }
    complete_reference = {
        (str(row["dataset"]), float(row["kir"]), int(row["seed"])): row
        for row in reference_rows
        if row.get("status") == "complete"
    }
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for k in K_VALUES:
                matched = []
                for seed in SEEDS:
                    fixed = complete_fixed.get((dataset, float(kir), int(seed), int(k)))
                    reference = complete_reference.get((dataset, float(kir), int(seed)))
                    if fixed is None or reference is None:
                        continue
                    matched.append((fixed, reference))
                if not matched:
                    continue
                for metric in COMPARISON_METRICS:
                    candidate_values = np.asarray([float(fixed[metric]) for fixed, _ in matched], dtype=np.float64)
                    reference_values = np.asarray([float(reference[metric]) for _, reference in matched], dtype=np.float64)
                    deltas = candidate_values - reference_values
                    low, high = _bootstrap_ci(deltas)
                    std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0
                    nonzero = deltas[np.abs(deltas) > 1e-12]
                    output.append(
                        {
                            "dataset": dataset,
                            "kir": kir,
                            "k": k,
                            "reference": reference_name,
                            "metric": metric,
                            "n_pairs": int(deltas.size),
                            "mean_delta": float(np.mean(deltas)),
                            "median_delta": float(np.median(deltas)),
                            "std_delta": std,
                            "ci95_low": low,
                            "ci95_high": high,
                            "wins": int(np.sum(deltas > 1e-12)),
                            "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
                            "losses": int(np.sum(deltas < -1e-12)),
                            "paired_t_p": float(ttest_rel(candidate_values, reference_values).pvalue)
                            if deltas.size > 1
                            else math.nan,
                            "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if nonzero.size else 1.0,
                            "cohen_dz": float(np.mean(deltas) / std) if std > 0 else 0.0,
                        }
                    )
    return output


def summarize(fixed_root: Path, baseline_root: Path) -> dict[str, Any]:
    fixed_failures: list[dict[str, Any]] = []
    reference_failures: list[dict[str, Any]] = []
    fixed_manifest_hashes: list[dict[str, Any]] = []
    reference_manifest_hashes: list[dict[str, Any]] = []

    fixed_rows = [
        _collect_fixed_row(
            fixed_root=fixed_root,
            baseline_root=baseline_root,
            cell=cell,
            failures=fixed_failures,
            manifest_hashes=fixed_manifest_hashes,
        )
        for cell in _planned_fixed_cells()
    ]
    adaptive_rows = [
        _collect_reference_row(
            baseline_root=baseline_root,
            cell=cell,
            failures=reference_failures,
            manifest_hashes=reference_manifest_hashes,
        )
        for cell in _planned_reference_cells(ADAPTIVE_METHOD)
    ]
    single_rows = [
        _collect_reference_row(
            baseline_root=baseline_root,
            cell=cell,
            failures=reference_failures,
            manifest_hashes=reference_manifest_hashes,
        )
        for cell in _planned_reference_cells(SINGLE_METHOD)
    ]

    complete_adaptive = {
        (str(row["dataset"]), float(row["kir"]), int(row["seed"])): row
        for row in adaptive_rows
        if row.get("status") == "complete"
    }
    complete_single = {
        (str(row["dataset"]), float(row["kir"]), int(row["seed"])): row
        for row in single_rows
        if row.get("status") == "complete"
    }
    complete_fixed_rows = [row for row in fixed_rows if row.get("status") == "complete"]

    for row in complete_fixed_rows:
        adaptive = complete_adaptive.get((str(row["dataset"]), float(row["kir"]), int(row["seed"])))
        single = complete_single.get((str(row["dataset"]), float(row["kir"]), int(row["seed"])))
        if adaptive is not None:
            for metric in COMPARISON_METRICS:
                row[f"adaptive_{metric}"] = float(adaptive[metric])
                row[f"delta_vs_adaptive_{metric}"] = float(row[metric]) - float(adaptive[metric])
        if single is not None:
            for metric in COMPARISON_METRICS:
                row[f"single_{metric}"] = float(single[metric])
                row[f"delta_vs_single_{metric}"] = float(row[metric]) - float(single[metric])

    summary_root = fixed_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    fixed_fields = [
        "protocol_version",
        "summary_stage",
        "dataset",
        "kir",
        "seed",
        "k",
        "status",
        "failure_reason",
        "representation",
        "partition",
        "source_partition",
        "partition_seed",
        "cluster_partition_count",
        "distance",
        "boundary",
        "acceptance",
        "test_used_for_selection",
        "source_stage",
        "source_method",
        "source_run_dir",
        "source_root",
        "reused_reference",
        *METRICS,
        *BALL_FIELDS,
        *(f"adaptive_{metric}" for metric in COMPARISON_METRICS),
        *(f"delta_vs_adaptive_{metric}" for metric in COMPARISON_METRICS),
        *(f"single_{metric}" for metric in COMPARISON_METRICS),
        *(f"delta_vs_single_{metric}" for metric in COMPARISON_METRICS),
    ]
    atomic_write_text(
        summary_root / "all_fixed_k.csv",
        _csv_text(
            sorted(
                fixed_rows,
                key=lambda row: (str(row["dataset"]), float(row["kir"]), int(row["seed"]), int(row["k"])),
            ),
            fixed_fields,
        ),
    )
    atomic_write_text(
        summary_root / "all_runs.csv",
        _csv_text(
            sorted(
                fixed_rows,
                key=lambda row: (str(row["dataset"]), float(row["kir"]), int(row["seed"]), int(row["k"])),
            ),
            fixed_fields,
        ),
    )

    adaptive_fields = [
        "protocol_version",
        "dataset",
        "kir",
        "seed",
        "reference_method",
        "status",
        "failure_reason",
        "partition",
        "cluster_partition_count",
        "source_stage",
        "source_run_dir",
        "source_root",
        "reused_reference",
        "test_used_for_selection",
        *METRICS,
        *BALL_FIELDS,
    ]
    atomic_write_text(
        summary_root / "adaptive_reference.csv",
        _csv_text(
            sorted(adaptive_rows, key=lambda row: (str(row["dataset"]), float(row["kir"]), int(row["seed"]))),
            adaptive_fields,
        ),
    )

    dataset_kir_grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in complete_fixed_rows:
        dataset_kir_grouped[(str(row["dataset"]), float(row["kir"]), int(row["k"]))].append(row)
    dataset_kir_summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for k in K_VALUES:
                selected = dataset_kir_grouped.get((dataset, float(kir), int(k)), [])
                item: dict[str, Any] = {
                    "dataset": dataset,
                    "kir": kir,
                    "k": k,
                    "planned_seeds": len(SEEDS),
                    "complete_seeds": len(selected),
                    "missing_or_invalid_seeds": len(SEEDS) - len(selected),
                }
                if selected:
                    for metric in (*METRICS, *BALL_FIELDS):
                        values = [float(row[metric]) for row in selected if metric in row and math.isfinite(float(row[metric]))]
                        item[f"mean_{metric}"] = statistics.fmean(values) if values else math.nan
                        item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
                    for metric in COMPARISON_METRICS:
                        adaptive_deltas = [
                            float(row[f"delta_vs_adaptive_{metric}"])
                            for row in selected
                            if f"delta_vs_adaptive_{metric}" in row
                        ]
                        single_deltas = [
                            float(row[f"delta_vs_single_{metric}"])
                            for row in selected
                            if f"delta_vs_single_{metric}" in row
                        ]
                        item[f"mean_delta_vs_adaptive_{metric}"] = statistics.fmean(adaptive_deltas) if adaptive_deltas else math.nan
                        item[f"std_delta_vs_adaptive_{metric}"] = statistics.stdev(adaptive_deltas) if len(adaptive_deltas) > 1 else 0.0
                        item[f"mean_delta_vs_single_{metric}"] = statistics.fmean(single_deltas) if single_deltas else math.nan
                        item[f"std_delta_vs_single_{metric}"] = statistics.stdev(single_deltas) if len(single_deltas) > 1 else 0.0
                dataset_kir_summary.append(item)
    atomic_write_text(
        summary_root / "dataset_kir_summary.csv",
        _csv_text(
            dataset_kir_summary,
            list(dataset_kir_summary[0]) if dataset_kir_summary else ["dataset", "kir", "k"],
        ),
    )

    overall_grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in complete_fixed_rows:
        overall_grouped[int(row["k"])].append(row)
    overall_summary: list[dict[str, Any]] = []
    for k in K_VALUES:
        selected = overall_grouped.get(int(k), [])
        item: dict[str, Any] = {
            "k": k,
            "planned_cells": len(DATASETS) * len(KIRS) * len(SEEDS),
            "complete_cells": len(selected),
            "missing_or_invalid_cells": len(DATASETS) * len(KIRS) * len(SEEDS) - len(selected),
        }
        if selected:
            for metric in ("oos_f1", "f1_all", "id_recall", "false_accept_rate", "effective_cluster_count"):
                values = [float(row[metric]) for row in selected]
                item[f"mean_{metric}"] = statistics.fmean(values)
                item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
            item["mean_delta_vs_adaptive_oos_f1"] = statistics.fmean(
                float(row["delta_vs_adaptive_oos_f1"]) for row in selected if "delta_vs_adaptive_oos_f1" in row
            )
            item["mean_delta_vs_single_oos_f1"] = statistics.fmean(
                float(row["delta_vs_single_oos_f1"]) for row in selected if "delta_vs_single_oos_f1" in row
            )
        overall_summary.append(item)
    atomic_write_text(
        summary_root / "overall_summary.csv",
        _csv_text(overall_summary, list(overall_summary[0]) if overall_summary else ["k"]),
    )

    paired_vs_adaptive = _paired_rows(fixed_rows, adaptive_rows, reference_name="adaptive")
    paired_vs_single = _paired_rows(fixed_rows, single_rows, reference_name="single")
    paired_fields = list(paired_vs_adaptive[0]) if paired_vs_adaptive else ["dataset", "kir", "k", "reference", "metric"]
    atomic_write_text(summary_root / "paired_vs_adaptive.csv", _csv_text(paired_vs_adaptive, paired_fields))
    atomic_write_text(
        summary_root / "paired_vs_single.csv",
        _csv_text(
            paired_vs_single,
            list(paired_vs_single[0]) if paired_vs_single else ["dataset", "kir", "k", "reference", "metric"],
        ),
    )

    significance_rows = paired_vs_adaptive + paired_vs_single
    atomic_write_text(
        summary_root / "significance.csv",
        _csv_text(
            significance_rows,
            list(significance_rows[0]) if significance_rows else ["dataset", "kir", "k", "reference", "metric"],
        ),
    )

    failed_rows = sorted(
        fixed_failures + reference_failures,
        key=lambda row: (
            str(row["kind"]),
            str(row["dataset"]),
            float(row["kir"]),
            int(row["seed"]),
            int(row.get("k", -1)),
            str(row.get("reference_method", row.get("source_method", ""))),
        ),
    )
    atomic_write_text(
        summary_root / "failed_or_invalid_runs.csv",
        _csv_text(
            failed_rows,
            list(failed_rows[0])
            if failed_rows
            else ["kind", "dataset", "kir", "seed", "k", "source_method", "reference_method", "reason", "run_dir"],
        ),
    )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": STAGE,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "fixed_root": str(fixed_root),
        "baseline_root": str(baseline_root),
        "summary_root": str(summary_root),
        "planned_fixed_k_units": len(DATASETS) * len(KIRS) * len(SEEDS) * len(K_VALUES),
        "completed_fixed_k_units": len(complete_fixed_rows),
        "missing_fixed_k_units": sum(1 for row in fixed_rows if row.get("status") == "missing"),
        "invalid_fixed_k_units": sum(1 for row in fixed_rows if row.get("status") == "invalid"),
        "planned_new_fixed_k_units": len(DATASETS) * len(KIRS) * len(SEEDS) * len(NEW_K_VALUES),
        "planned_reused_k2_units": len(DATASETS) * len(KIRS) * len(SEEDS),
        "adaptive_reference_units": len(adaptive_rows),
        "adaptive_reference_complete_units": sum(1 for row in adaptive_rows if row.get("status") == "complete"),
        "single_reference_units": len(single_rows),
        "single_reference_complete_units": sum(1 for row in single_rows if row.get("status") == "complete"),
        "test_used_for_selection": False,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "comparison_metrics": list(COMPARISON_METRICS),
        "summary_files": {
            "all_fixed_k": "all_fixed_k.csv",
            "all_runs": "all_runs.csv",
            "adaptive_reference": "adaptive_reference.csv",
            "overall_summary": "overall_summary.csv",
            "dataset_kir_summary": "dataset_kir_summary.csv",
            "paired_vs_adaptive": "paired_vs_adaptive.csv",
            "paired_vs_single": "paired_vs_single.csv",
            "significance": "significance.csv",
            "failed_or_invalid_runs": "failed_or_invalid_runs.csv",
            "closeout": "MOGB_FIXED_K_MEAN_CLOSEOUT.md",
            "summary_manifest": "summary_manifest.json",
        },
        "source_contract": {
            "new_k_values": list(NEW_K_VALUES),
            "reused_k": {
                "k": 2,
                "method": REUSE_METHOD,
                "source_stage": "mogb_baseline_v1",
            },
            "adaptive_reference_method": ADAPTIVE_METHOD,
            "single_reference_method": SINGLE_METHOD,
            "test_used_for_selection": False,
        },
        "manifest_hashes": fixed_manifest_hashes,
        "reference_manifest_hashes": reference_manifest_hashes,
        "config_sha256": sha256_json(
            {
                "datasets": DATASETS,
                "kirs": KIRS,
                "seeds": SEEDS,
                "k_values": K_VALUES,
                "metrics": METRICS,
                "comparison_metrics": COMPARISON_METRICS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        ),
    }
    atomic_write_json(summary_root / "summary_manifest.json", manifest)

    overall_index = {int(row["k"]): row for row in overall_summary}
    closeout_lines = [
        "# MOGB fixed-K mean ablation closeout",
        "",
        f"- Generated: {manifest['generated_at']}",
        f"- Fixed-K contract: {manifest['completed_fixed_k_units']}/{manifest['planned_fixed_k_units']} completed rows across K=1..4 with K=2 reused from `{REUSE_METHOD}`.",
        f"- Adaptive reference: {manifest['adaptive_reference_complete_units']}/{manifest['adaptive_reference_units']} rows from `{ADAPTIVE_METHOD}`.",
        f"- Single reference: {manifest['single_reference_complete_units']}/{manifest['single_reference_units']} rows from `{SINGLE_METHOD}`.",
        "",
        "## Overall means",
        "",
        "| K | cells | mean_oos_f1 | mean_f1_all | mean_id_recall | mean_false_accept_rate | mean_delta_vs_adaptive_oos_f1 | mean_delta_vs_single_oos_f1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for k in K_VALUES:
        item = overall_index.get(k, {})
        closeout_lines.append(
            "| "
            + " | ".join(
                [
                    str(k),
                    str(item.get("complete_cells", 0)),
                    f"{_safe_float(item.get('mean_oos_f1')):.6f}" if "mean_oos_f1" in item else "nan",
                    f"{_safe_float(item.get('mean_f1_all')):.6f}" if "mean_f1_all" in item else "nan",
                    f"{_safe_float(item.get('mean_id_recall')):.6f}" if "mean_id_recall" in item else "nan",
                    f"{_safe_float(item.get('mean_false_accept_rate')):.6f}" if "mean_false_accept_rate" in item else "nan",
                    f"{_safe_float(item.get('mean_delta_vs_adaptive_oos_f1')):.6f}" if "mean_delta_vs_adaptive_oos_f1" in item else "nan",
                    f"{_safe_float(item.get('mean_delta_vs_single_oos_f1')):.6f}" if "mean_delta_vs_single_oos_f1" in item else "nan",
                ]
            )
            + " |"
        )
    closeout_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `all_fixed_k.csv`: full 180-cell fixed-K table with source path and reuse flags.",
            "- `adaptive_reference.csv`: paired adaptive baseline reference table.",
            "- `paired_vs_adaptive.csv`, `paired_vs_single.csv`, `significance.csv`: seed-paired effects with bootstrap confidence intervals and paired tests.",
            "- `failed_or_invalid_runs.csv`: missing/invalid source rows retained for audit.",
        ]
    )
    atomic_write_text(summary_root / "MOGB_FIXED_K_MEAN_CLOSEOUT.md", "\n".join(closeout_lines) + "\n")

    return {
        "summary_root": str(summary_root),
        "planned_fixed_k_units": manifest["planned_fixed_k_units"],
        "completed_fixed_k_units": manifest["completed_fixed_k_units"],
        "missing_fixed_k_units": manifest["missing_fixed_k_units"],
        "invalid_fixed_k_units": manifest["invalid_fixed_k_units"],
        "adaptive_reference_complete_units": manifest["adaptive_reference_complete_units"],
        "single_reference_complete_units": manifest["single_reference_complete_units"],
        "manifest_sha256": sha256_file(summary_root / "summary_manifest.json"),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    paths = ProtocolV2Paths.discover()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-root",
        type=Path,
        default=paths.run_root / STAGE,
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=paths.run_root / "mogb_baseline_v1",
    )
    args = parser.parse_args(argv)
    result = summarize(args.fixed_root, args.baseline_root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["missing_fixed_k_units"] == 0 and result["invalid_fixed_k_units"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
