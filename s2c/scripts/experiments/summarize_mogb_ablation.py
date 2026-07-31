"""Summarize the frozen-MiniLM MOGB OFAT ablation sweep."""

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

DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
REFERENCE_METHOD = "mogb_minilm"
BASELINE_METHOD = "single_centroid"
HYBRID_METHOD = "mogb_partition_ours_boundary"
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
DEFAULT_PARAMETERS = {
    "distance": "euclidean",
    "boundary": "mean",
    "purity_train": 0.90,
    "purity_get_ball": 1.00,
    "purity_select_ball": 0.90,
    "min_ball_train": 10,
    "min_ball_get_ball": 5,
    "min_ball_select_ball": 10,
}


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _bootstrap_ci(deltas: np.ndarray, *, seed: int = 20260730) -> tuple[float, float]:
    if deltas.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(deltas, size=(10_000, deltas.size), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _variant_dir(root: Path, variant: str, dataset: str, kir: float, seed: int) -> Path:
    return root / variant / dataset / f"kir_{kir:.2f}" / f"seed_{seed}"


def _baseline_dir(root: Path, dataset: str, kir: float, seed: int, method: str) -> Path:
    return root / dataset / f"kir_{kir:.2f}" / f"seed_{seed}" / method


def _variant_rows() -> list[dict[str, Any]]:
    return [
        {"variant": "get_085", "distance": "euclidean", "boundary": "mean", "purity_get_ball": 0.85},
        {"variant": "get_090", "distance": "euclidean", "boundary": "mean", "purity_get_ball": 0.90},
        {"variant": "get_095", "distance": "euclidean", "boundary": "mean", "purity_get_ball": 0.95},
        {"variant": "select_085", "distance": "euclidean", "boundary": "mean", "purity_select_ball": 0.85},
        {"variant": "select_095", "distance": "euclidean", "boundary": "mean", "purity_select_ball": 0.95},
        {"variant": "select_100", "distance": "euclidean", "boundary": "mean", "purity_select_ball": 1.00},
        {"variant": "min_get_10", "distance": "euclidean", "boundary": "mean", "min_ball_get_ball": 10},
        {"variant": "min_get_20", "distance": "euclidean", "boundary": "mean", "min_ball_get_ball": 20},
        {"variant": "min_select_5", "distance": "euclidean", "boundary": "mean", "min_ball_select_ball": 5},
        {"variant": "min_select_20", "distance": "euclidean", "boundary": "mean", "min_ball_select_ball": 20},
        {"variant": "default_mean_std", "distance": "euclidean", "boundary": "mean_std"},
        {"variant": "default_mahalanobis_mean", "distance": "mahalanobis_diag", "boundary": "mean"},
    ]


def summarize(ablation_root: Path, baseline_root: Path) -> dict[str, Any]:
    variants = _variant_rows()
    variant_names = tuple(str(spec["variant"]) for spec in variants)
    planned_units = len(DATASETS) * len(KIRS) * len(SEEDS) * len(variant_names)
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    manifest_hashes: list[dict[str, Any]] = []

    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                reference_cache: dict[str, dict[str, Any]] = {}
                for method in (REFERENCE_METHOD, BASELINE_METHOD, HYBRID_METHOD):
                    run_dir = _baseline_dir(baseline_root, dataset, kir, seed, method)
                    manifest_path = run_dir / "manifest.json"
                    metrics_path = run_dir / "metrics.json"
                    config_path = run_dir / "config.json"
                    if not (manifest_path.is_file() and metrics_path.is_file() and config_path.is_file()):
                        missing.append(
                            {
                                "kind": "baseline",
                                "dataset": dataset,
                                "kir": kir,
                                "seed": seed,
                                "variant": method,
                                "reason": "missing_required_files",
                                "run_dir": str(run_dir),
                            }
                        )
                        continue
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    if manifest.get("status") != "complete":
                        invalid.append(
                            {
                                "kind": "baseline",
                                "dataset": dataset,
                                "kir": kir,
                                "seed": seed,
                                "variant": method,
                                "reason": "manifest_not_complete",
                                "run_dir": str(run_dir),
                            }
                        )
                        continue
                    reference_cache[method] = {
                        "manifest": manifest,
                        "metrics": metrics,
                        "config": config,
                        "run_dir": run_dir,
                    }
                    baseline_rows.append(
                        {
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "method": method,
                            **{metric: _safe_float(metrics.get(metric)) for metric in METRICS},
                            "run_dir": str(run_dir),
                        }
                    )

                for spec in variants:
                    variant = str(spec["variant"])
                    run_dir = _variant_dir(ablation_root, variant, dataset, kir, seed)
                    manifest_path = run_dir / "manifest.json"
                    metrics_path = run_dir / "metrics.json"
                    config_path = run_dir / "config.json"
                    method_path = run_dir / "method_details.json"
                    ball_stats_path = run_dir / "ball_statistics.json"
                    if not all(path.is_file() for path in (manifest_path, metrics_path, config_path, method_path, ball_stats_path)):
                        missing.append(
                            {
                                "kind": "ablation",
                                "dataset": dataset,
                                "kir": kir,
                                "seed": seed,
                                "variant": variant,
                                "reason": "missing_required_files",
                                "run_dir": str(run_dir),
                            }
                        )
                        continue
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    json.loads(method_path.read_text(encoding="utf-8"))
                    ball_stats = json.loads(ball_stats_path.read_text(encoding="utf-8"))
                    if manifest.get("status") != "complete":
                        invalid.append(
                            {
                                "kind": "ablation",
                                "dataset": dataset,
                                "kir": kir,
                                "seed": seed,
                                "variant": variant,
                                "reason": "manifest_not_complete",
                                "run_dir": str(run_dir),
                            }
                        )
                        continue
                    if config.get("protocol_version") != "protocol_v2_textoir_v1":
                        invalid.append(
                            {
                                "kind": "ablation",
                                "dataset": dataset,
                                "kir": kir,
                                "seed": seed,
                                "variant": variant,
                                "reason": "protocol_mismatch",
                                "run_dir": str(run_dir),
                            }
                        )
                        continue
                    if str(config.get("output_variant")) != variant:
                        invalid.append(
                            {
                                "kind": "ablation",
                                "dataset": dataset,
                                "kir": kir,
                                "seed": seed,
                                "variant": variant,
                                "reason": f"output_variant_mismatch:{config.get('output_variant')}",
                                "run_dir": str(run_dir),
                            }
                        )
                        continue
                    expected_parameters = {**DEFAULT_PARAMETERS, **spec}
                    mismatched_parameters = {
                        key: {"expected": expected, "actual": config.get(key)}
                        for key, expected in expected_parameters.items()
                        if key != "variant" and config.get(key) != expected
                    }
                    if mismatched_parameters:
                        invalid.append(
                            {
                                "kind": "ablation",
                                "dataset": dataset,
                                "kir": kir,
                                "seed": seed,
                                "variant": variant,
                                "reason": f"parameter_mismatch:{json.dumps(mismatched_parameters, sort_keys=True)}",
                                "run_dir": str(run_dir),
                            }
                        )
                        continue
                    metric_values = {metric: _safe_float(metrics.get(metric)) for metric in METRICS}
                    if not all(math.isfinite(value) for value in metric_values.values()):
                        invalid.append(
                            {
                                "kind": "ablation",
                                "dataset": dataset,
                                "kir": kir,
                                "seed": seed,
                                "variant": variant,
                                "reason": "non_finite_metric",
                                "run_dir": str(run_dir),
                            }
                        )
                        continue
                    reference = reference_cache.get(REFERENCE_METHOD)
                    baseline = reference_cache.get(BASELINE_METHOD)
                    row = {
                        "protocol_version": "protocol_v2_textoir_v1",
                        "dataset": dataset,
                        "kir": kir,
                        "seed": seed,
                        "variant": variant,
                        "distance": str(config["distance"]),
                        "boundary": str(config["boundary"]),
                        "purity_get_ball": _safe_float(config.get("purity_get_ball")),
                        "purity_select_ball": _safe_float(config.get("purity_select_ball")),
                        "min_ball_get_ball": int(config.get("min_ball_get_ball")),
                        "min_ball_select_ball": int(config.get("min_ball_select_ball")),
                        **metric_values,
                        "selected_balls": int(ball_stats.get("selected_balls", metric_values["effective_cluster_count"])),
                        "filtered_balls": int(ball_stats.get("filtered_balls", 0)),
                        "mean_balls_per_intent": _safe_float(ball_stats.get("mean_balls_per_intent")),
                        "max_balls_per_intent": _safe_float(ball_stats.get("max_balls_per_intent")),
                        "mean_ball_radius": _safe_float(ball_stats.get("mean_radius")),
                        "mean_ball_samples": _safe_float(ball_stats.get("mean_samples_per_ball")),
                        "mean_ball_purity": _safe_float(ball_stats.get("mean_purity")),
                        "max_tree_depth": _safe_float(ball_stats.get("max_tree_depth")),
                        "run_dir": str(run_dir),
                    }
                    if reference is not None:
                        ref_metrics = reference["metrics"]
                        for metric in COMPARISON_METRICS:
                            row[f"reference_{metric}"] = _safe_float(ref_metrics.get(metric))
                            row[f"delta_vs_reference_{metric}"] = row[metric] - row[f"reference_{metric}"]
                    if baseline is not None:
                        base_metrics = baseline["metrics"]
                        for metric in COMPARISON_METRICS:
                            row[f"single_{metric}"] = _safe_float(base_metrics.get(metric))
                            row[f"delta_vs_single_{metric}"] = row[metric] - row[f"single_{metric}"]
                    rows.append(row)
                    manifest_hashes.append(
                        {
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "variant": variant,
                            "manifest_sha256": sha256_file(manifest_path),
                        }
                    )

    summary_root = ablation_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    all_fields = [
        "protocol_version",
        "dataset",
        "kir",
        "seed",
        "variant",
        "distance",
        "boundary",
        "purity_get_ball",
        "purity_select_ball",
        "min_ball_get_ball",
        "min_ball_select_ball",
        *METRICS,
        "selected_balls",
        "filtered_balls",
        "mean_balls_per_intent",
        "max_balls_per_intent",
        "mean_ball_radius",
        "mean_ball_samples",
        "mean_ball_purity",
        "max_tree_depth",
        *(f"reference_{metric}" for metric in COMPARISON_METRICS),
        *(f"delta_vs_reference_{metric}" for metric in COMPARISON_METRICS),
        *(f"single_{metric}" for metric in COMPARISON_METRICS),
        *(f"delta_vs_single_{metric}" for metric in COMPARISON_METRICS),
        "run_dir",
    ]
    atomic_write_text(summary_root / "all_runs.csv", _csv_text(sorted(rows, key=lambda row: (row["dataset"], float(row["kir"]), int(row["seed"]), row["variant"])), all_fields))
    baseline_fields = ["dataset", "kir", "seed", "method", *METRICS, "run_dir"]
    atomic_write_text(summary_root / "baseline_reference.csv", _csv_text(sorted(baseline_rows, key=lambda row: (row["dataset"], float(row["kir"]), int(row["seed"]), row["method"])), baseline_fields))

    boundary_components: list[dict[str, Any]] = []
    component_metadata = {
        REFERENCE_METHOD: ("euclidean_mean", "euclidean", "mean"),
        HYBRID_METHOD: ("mahalanobis_diag_mean_std", "mahalanobis_diag", "mean_std"),
    }
    for row in baseline_rows:
        method = str(row["method"])
        if method not in component_metadata:
            continue
        component, distance, boundary = component_metadata[method]
        boundary_components.append(
            {
                "dataset": row["dataset"],
                "kir": row["kir"],
                "seed": row["seed"],
                "component": component,
                "source": method,
                "distance": distance,
                "boundary": boundary,
                **{metric: row[metric] for metric in METRICS},
            }
        )
    for row in rows:
        if row["variant"] not in {"default_mean_std", "default_mahalanobis_mean"}:
            continue
        component = (
            "euclidean_mean_std"
            if row["variant"] == "default_mean_std"
            else "mahalanobis_diag_mean"
        )
        boundary_components.append(
            {
                "dataset": row["dataset"],
                "kir": row["kir"],
                "seed": row["seed"],
                "component": component,
                "source": row["variant"],
                "distance": row["distance"],
                "boundary": row["boundary"],
                **{metric: row[metric] for metric in METRICS},
            }
        )
    boundary_fields = ["dataset", "kir", "seed", "component", "source", "distance", "boundary", *METRICS]
    atomic_write_text(
        summary_root / "boundary_component_runs.csv",
        _csv_text(
            sorted(
                boundary_components,
                key=lambda row: (row["dataset"], float(row["kir"]), int(row["seed"]), row["component"]),
            ),
            boundary_fields,
        ),
    )
    boundary_grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in boundary_components:
        boundary_grouped[(str(row["dataset"]), float(row["kir"]), str(row["component"]))].append(row)
    boundary_summary: list[dict[str, Any]] = []
    for (dataset, kir, component), selected in sorted(boundary_grouped.items()):
        item: dict[str, Any] = {
            "dataset": dataset,
            "kir": kir,
            "component": component,
            "n_seeds": len(selected),
        }
        for metric in ("oos_f1", "f1_all", "id_recall", "false_accept_rate", "false_reject_rate"):
            values = [float(row[metric]) for row in selected]
            item[f"mean_{metric}"] = statistics.fmean(values)
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        boundary_summary.append(item)
    atomic_write_text(
        summary_root / "boundary_component_summary.csv",
        _csv_text(
            boundary_summary,
            list(boundary_summary[0]) if boundary_summary else ["dataset", "kir", "component", "n_seeds"],
        ),
    )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), float(row["kir"]), str(row["variant"]))].append(row)
    dataset_kir_summary: list[dict[str, Any]] = []
    for (dataset, kir, variant), selected in sorted(grouped.items()):
        item: dict[str, Any] = {"dataset": dataset, "kir": kir, "variant": variant, "n_seeds": len(selected)}
        for metric in (*METRICS, "selected_balls", "filtered_balls", "mean_balls_per_intent", "mean_ball_radius"):
            values = [float(row[metric]) for row in selected]
            item[f"mean_{metric}"] = statistics.fmean(values)
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for delta_metric in ("delta_vs_reference_oos_f1", "delta_vs_single_oos_f1", "delta_vs_reference_false_accept_rate", "delta_vs_single_false_accept_rate"):
            values = [float(row[delta_metric]) for row in selected if delta_metric in row]
            item[f"mean_{delta_metric}"] = statistics.fmean(values) if values else math.nan
            item[f"std_{delta_metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        dataset_kir_summary.append(item)
    atomic_write_text(summary_root / "dataset_kir_summary.csv", _csv_text(dataset_kir_summary, list(dataset_kir_summary[0]) if dataset_kir_summary else ["dataset", "kir", "variant", "n_seeds"]))

    overall_grouped = defaultdict(list)
    for row in rows:
        overall_grouped[str(row["variant"])].append(row)
    overall_summary: list[dict[str, Any]] = []
    for variant, selected in sorted(overall_grouped.items()):
        item = {"variant": variant, "n_cells": len(selected)}
        for metric in ("oos_f1", "f1_all", "id_recall", "false_accept_rate", "effective_cluster_count"):
            values = [float(row[metric]) for row in selected]
            item[f"mean_{metric}"] = statistics.fmean(values)
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        item["mean_delta_vs_reference_oos_f1"] = statistics.fmean(float(row["delta_vs_reference_oos_f1"]) for row in selected)
        item["mean_delta_vs_single_oos_f1"] = statistics.fmean(float(row["delta_vs_single_oos_f1"]) for row in selected)
        overall_summary.append(item)
    atomic_write_text(summary_root / "overall_summary.csv", _csv_text(overall_summary, list(overall_summary[0]) if overall_summary else ["variant", "n_cells"]))

    def _paired_rows(reference_field_prefix: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for dataset in DATASETS:
            for kir in KIRS:
                for variant in variant_names:
                    selected = [
                        row
                        for row in rows
                        if row["dataset"] == dataset and float(row["kir"]) == kir and row["variant"] == variant
                    ]
                    if not selected:
                        continue
                    for metric in COMPARISON_METRICS:
                        if reference_field_prefix == "reference":
                            reference_values = np.asarray([float(row[f"reference_{metric}"]) if f"reference_{metric}" in row else math.nan for row in selected], dtype=np.float64)
                        else:
                            reference_values = np.asarray([float(row[f"single_{metric}"]) if f"single_{metric}" in row else math.nan for row in selected], dtype=np.float64)
                        candidate_values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
                        keep = np.isfinite(candidate_values) & np.isfinite(reference_values)
                        candidate_values = candidate_values[keep]
                        reference_values = reference_values[keep]
                        if candidate_values.size == 0:
                            continue
                        deltas = candidate_values - reference_values
                        low, high = _bootstrap_ci(deltas)
                        std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0
                        nonzero = deltas[np.abs(deltas) > 1e-12]
                        output.append(
                            {
                                "dataset": dataset,
                                "kir": kir,
                                "variant": variant,
                                "reference": reference_field_prefix,
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

    paired_reference = _paired_rows("reference")
    paired_single = _paired_rows("single")
    atomic_write_text(summary_root / "paired_vs_reference.csv", _csv_text(paired_reference, list(paired_reference[0]) if paired_reference else ["dataset", "kir", "variant", "reference", "metric"]))
    atomic_write_text(summary_root / "paired_vs_single.csv", _csv_text(paired_single, list(paired_single[0]) if paired_single else ["dataset", "kir", "variant", "reference", "metric"]))
    significance_rows = paired_reference + paired_single
    atomic_write_text(
        summary_root / "significance_tests.csv",
        _csv_text(
            significance_rows,
            list(significance_rows[0])
            if significance_rows
            else ["dataset", "kir", "variant", "reference", "metric"],
        ),
    )

    tradeoff_rows = [
        {
            "dataset": row["dataset"],
            "kir": row["kir"],
            "seed": row["seed"],
            "variant": row["variant"],
            "delta_vs_reference_id_recall": row.get("delta_vs_reference_id_recall"),
            "delta_vs_reference_false_accept_rate": row.get("delta_vs_reference_false_accept_rate"),
            "delta_vs_single_id_recall": row.get("delta_vs_single_id_recall"),
            "delta_vs_single_false_accept_rate": row.get("delta_vs_single_false_accept_rate"),
            "selected_balls": row["selected_balls"],
            "mean_ball_radius": row["mean_ball_radius"],
        }
        for row in rows
    ]
    atomic_write_text(summary_root / "known_recall_false_accept_tradeoff.csv", _csv_text(tradeoff_rows, list(tradeoff_rows[0]) if tradeoff_rows else ["dataset", "kir", "seed", "variant"]))

    ball_rows = [
        {
            "dataset": row["dataset"],
            "kir": row["kir"],
            "seed": row["seed"],
            "variant": row["variant"],
            "selected_balls": row["selected_balls"],
            "filtered_balls": row["filtered_balls"],
            "mean_balls_per_intent": row["mean_balls_per_intent"],
            "max_balls_per_intent": row["max_balls_per_intent"],
            "mean_ball_radius": row["mean_ball_radius"],
            "mean_ball_samples": row["mean_ball_samples"],
            "mean_ball_purity": row["mean_ball_purity"],
            "max_tree_depth": row["max_tree_depth"],
            "effective_cluster_count": row["effective_cluster_count"],
            "minimum_cluster_size": row["minimum_cluster_size"],
        }
        for row in rows
    ]
    atomic_write_text(summary_root / "ball_diagnostics.csv", _csv_text(ball_rows, list(ball_rows[0]) if ball_rows else ["dataset", "kir", "seed", "variant"]))

    invalid_rows = sorted(missing + invalid, key=lambda row: (row["kind"], row["dataset"], float(row["kir"]), int(row["seed"]), row["variant"]))
    atomic_write_text(
        summary_root / "failed_or_invalid_runs.csv",
        _csv_text(invalid_rows, list(invalid_rows[0]) if invalid_rows else ["kind", "dataset", "kir", "seed", "variant", "reason", "run_dir"]),
    )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "protocol_version": "protocol_v2_textoir_v1",
        "stage": "mogb_ablation_v1_frozen_ofat",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "planned_units": planned_units,
        "completed_units": len(rows),
        "missing_units": len(missing),
        "invalid_units": len(invalid),
        "reference_method": REFERENCE_METHOD,
        "baseline_method": BASELINE_METHOD,
        "bootstrap": {"resamples": 10_000, "seed": 20260730},
        "summary_files": {
            "all_runs": "all_runs.csv",
            "baseline_reference": "baseline_reference.csv",
            "boundary_component_runs": "boundary_component_runs.csv",
            "boundary_component_summary": "boundary_component_summary.csv",
            "dataset_kir_summary": "dataset_kir_summary.csv",
            "overall_summary": "overall_summary.csv",
            "paired_vs_reference": "paired_vs_reference.csv",
            "paired_vs_single": "paired_vs_single.csv",
            "significance_tests": "significance_tests.csv",
            "known_recall_false_accept_tradeoff": "known_recall_false_accept_tradeoff.csv",
            "ball_diagnostics": "ball_diagnostics.csv",
            "failed_or_invalid_runs": "failed_or_invalid_runs.csv",
            "integrity_report": "MOGB_ABLATION_INTEGRITY.md",
            "closeout": "MOGB_ABLATION_CLOSEOUT.md",
        },
        "manifest_hashes": manifest_hashes,
    }
    atomic_write_json(summary_root / "MOGB_ABLATION_SUMMARY_MANIFEST.json", manifest)

    return {
        "summary_root": str(summary_root),
        "planned_units": planned_units,
        "completed_units": len(rows),
        "missing_units": len(missing),
        "invalid_units": len(invalid),
        "manifest_sha256": sha256_file(summary_root / "MOGB_ABLATION_SUMMARY_MANIFEST.json"),
        "config_sha256": sha256_json({"variants": variants, "metrics": METRICS, "comparison_metrics": COMPARISON_METRICS}),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_ablation_v1"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_baseline_v1"),
    )
    args = parser.parse_args(argv)
    summary = summarize(args.ablation_root, args.baseline_root)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["missing_units"] == 0 and summary["invalid_units"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
