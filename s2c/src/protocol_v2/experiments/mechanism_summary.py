"""Summaries for the independent protocol_v2 E3 mechanism study.

This module reads only completed E3 manifests/metrics and the immutable E2 K=1
reference runs.  It never selects a K from test data; ``oracle_test_best_k`` is
not generated here.  The association table is explicitly exploratory.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file
from protocol_v2.data.manifests import read_json
from protocol_v2.runtime.paths import ProtocolV2Paths

from .mechanism_runner import (
    E3_ROOT_NAME,
    PartitionControlSpec,
    _control_run_dir,
    _e2_run_dir,
    diagnostic_groups,
    diagnostic_partition_seeds,
    partition_control_specs,
)


METRICS = (
    "oos_f1",
    "id_recall",
    "auroc",
    "aupr_oos",
    "oos_precision",
    "oos_recall",
    "false_accept_rate",
    "false_reject_rate",
)
OOS_VIEWS = ("combined", "heldout_intent", "native")
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_RESAMPLES = 10_000


def _e3_root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / E3_ROOT_NAME


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _e2_metric(paths: ProtocolV2Paths, spec: PartitionControlSpec, source: str, metric: str) -> float:
    run = _e2_run_dir(paths, spec.dataset, spec.seed, spec.kir, spec.distance)
    payload = read_json(run / "metrics.json")
    value = payload.get("oos_breakdown", {}).get(source, {}).get(metric)
    return float(value) if value is not None and _finite(value) else math.nan


def _completed_control_rows(paths: ProtocolV2Paths, specs: list[PartitionControlSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        run_dir = _control_run_dir(paths, spec)
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.json"
        base = {
            "run_id": spec.run_id,
            "protocol_version": "protocol_v2_textoir_v1",
            "stage": "E3-A",
            "dataset": spec.dataset,
            "kir": f"{spec.kir:.2f}",
            "seed": spec.seed,
            "k": spec.k,
            "distance": spec.distance,
            "partition": spec.partition,
            "partition_seed": spec.partition_seed,
            "status": "missing",
            "run_relative_path": str(run_dir.relative_to(paths.artifacts_root)),
            "manifest_sha256": "",
        }
        if not manifest_path.is_file() or not metrics_path.is_file():
            rows.append(base)
            continue
        try:
            manifest = read_json(manifest_path)
            metrics = read_json(metrics_path)
            if (
                manifest.get("status") != "complete"
                or manifest.get("stage") != "E3-A"
                or manifest.get("protocol_version") != paths.dataset_version
            ):
                rows.append({**base, "status": "invalid"})
                continue
            for source in OOS_VIEWS:
                metric_values = metrics.get("oos_breakdown", {}).get(source, {})
                row = {**base, "oos_source": source, "status": "complete", "manifest_sha256": sha256_file(manifest_path)}
                row.update({metric: metric_values.get(metric, math.nan) for metric in METRICS})
                combined = metrics.get("combined", {})
                row.update(
                    {
                        "scoring_seconds": combined.get("scoring_seconds", math.nan),
                        "samples_per_second": combined.get("samples_per_second", math.nan),
                        "effective_cluster_count": combined.get("effective_cluster_count", math.nan),
                        "minimum_cluster_size": combined.get("minimum_cluster_size", math.nan),
                    }
                )
                rows.append(row)
        except (OSError, TypeError, ValueError, KeyError):
            rows.append({**base, "status": "invalid"})
    return rows


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_row(
    dataset: str,
    kir: str,
    k: int,
    distance: str,
    source: str,
    metric: str,
    comparison: str,
    values: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    low, high = _bootstrap_ci(values)
    mean = float(values.mean()) if values.size else math.nan
    median = float(np.median(values)) if values.size else math.nan
    scale = float(values.std(ddof=1)) if values.size > 1 else math.nan
    return {
        "dataset": dataset,
        "kir": kir,
        "k": k,
        "distance": distance,
        "oos_source": source,
        "metric": metric,
        "comparison": comparison,
        "n_pairs": int(values.size),
        "mean_delta": mean,
        "median_delta": median,
        "ci95_low": low,
        "ci95_high": high,
        "effect_size_mean_over_sd": mean / scale if _finite(scale) and scale > 0 else math.nan,
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.isclose(values, 0.0, atol=1e-12))),
        "losses": int(np.sum(values < -1e-12)),
        "bootstrap_rng_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }


def summarize_partition_control(paths: ProtocolV2Paths, specs: list[PartitionControlSpec]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _completed_control_rows(paths, specs)
    complete = [row for row in rows if row["status"] == "complete"]
    paired: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int, str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for row in complete:
        key = (row["dataset"], row["kir"], row["k"], row["distance"], row["oos_source"], row["partition"])
        grouped[key][int(row["seed"])] = {metric: float(row[metric]) for metric in METRICS}
    # Reconstruct the corresponding E2 K=1 values for every declared cell.
    e2_specs = {(spec.dataset, f"{spec.kir:.2f}", spec.seed, spec.distance): spec for spec in specs}
    for (dataset, kir, k, distance, source, partition), per_seed in grouped.items():
        deltas = {metric: [] for metric in METRICS}
        random_values = {metric: [] for metric in METRICS}
        kmeans_values = {metric: [] for metric in METRICS}
        for seed, values in per_seed.items():
            ref_spec = e2_specs[(dataset, kir, seed, distance)]
            for metric in METRICS:
                deltas[metric].append(values[metric] - _e2_metric(paths, ref_spec, source, metric))
                if partition == "random_balanced":
                    random_values[metric].append(values[metric])
                else:
                    kmeans_values[metric].append(values[metric])
        for metric, values in deltas.items():
            paired.append(_paired_row(dataset, kir, k, distance, source, metric, f"{partition}_vs_k1", np.asarray(values)))
    # KMeans-vs-random is a second paired comparison over identical cells.
    for key in sorted({key[:-1] for key in grouped}):
        dataset, kir, k, distance, source = key
        kmeans = grouped.get((*key, "kmeans"), {})
        random = grouped.get((*key, "random_balanced"), {})
        for metric in METRICS:
            common = sorted(set(kmeans) & set(random))
            values = np.asarray([kmeans[seed][metric] - random[seed][metric] for seed in common], dtype=float)
            paired.append(_paired_row(dataset, kir, k, distance, source, metric, "kmeans_vs_random_balanced", values))
    return rows, paired


def _flatten_diagnostics(paths: ProtocolV2Paths) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_root = _e3_root(paths) / "diagnostics" / "groups"
    rows: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for path in sorted(group_root.glob("*.json")):
        payload = read_json(path)
        group = payload.get("group", {})
        for row in payload.get("rows", []):
            base = {key: value for key, value in row.items() if key != "intent_features"}
            rows.append(base)
            for feature in row.get("intent_features", []):
                features.append(
                    {
                        "dataset": group.get("dataset"),
                        "kir": group.get("kir"),
                        "seed": group.get("seed"),
                        "k": group.get("k"),
                        "partition": row.get("partition"),
                        "partition_seed": row.get("partition_seed"),
                        "distance": row.get("distance"),
                        **feature,
                    }
                )
    return rows, features


def _association_rows(
    paths: ProtocolV2Paths,
    features: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    specs: list[PartitionControlSpec],
) -> list[dict[str, Any]]:
    """Join train/calibration reliability signals to E3-A deltas descriptively."""

    # Aggregate intent signals to a run-level value for the declared pseed=42
    # snapshot, then join the same dataset/KIR/seed/K/distance cell.
    selected: dict[tuple[str, str, int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in features:
        if int(row["partition_seed"]) == 42:
            key = (row["dataset"], f"{float(row['kir']):.2f}", int(row["seed"]), int(row["k"]), row["distance"], row["partition"])
            selected[key].append(row)
    control_index = {(row["dataset"], row["kir"], int(row["seed"]), int(row["k"]), row["distance"], row["partition"]): row for row in control_rows if row.get("oos_source") == "combined" and row.get("status") == "complete"}
    values: dict[tuple[str, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    feature_names = (
        "min_cluster_ratio",
        "tiny_cluster_ratio",
        "median_pairwise_ari",
        "silhouette_mean",
        "compactness_gain",
        "centroid_separation",
        "separation_radius_ratio",
        "radius_cv",
        "coverage_drop_vs_k1",
        "false_rejection_increase_vs_k1",
        "cluster_size_cv",
        "dominant_cluster_ratio",
    )
    spec_index = {
        (spec.dataset, f"{spec.kir:.2f}", spec.seed, spec.k, spec.distance, spec.partition): spec
        for spec in specs
    }
    for key, rows in selected.items():
        dataset, kir, seed, k, distance, partition = key
        control = control_index.get(key)
        if not control:
            continue
        ref_spec = spec_index[(dataset, kir, seed, k, distance, partition)]
        e2_ref = _e2_metric(paths, ref_spec, "combined", "oos_f1")
        delta = float(control["oos_f1"] - e2_ref)
        for feature_name in feature_names:
            feature_values = np.asarray([float(row.get(feature_name, math.nan)) for row in rows], dtype=float)
            if feature_values.size and np.isfinite(feature_values).any():
                values[(dataset, partition, distance, feature_name)].append((float(np.nanmean(feature_values)), delta))
    output: list[dict[str, Any]] = []
    for (dataset, partition, distance, feature), pairs in sorted(values.items()):
        x = np.asarray([pair[0] for pair in pairs], dtype=float)
        y = np.asarray([pair[1] for pair in pairs], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            rho, p_value = math.nan, math.nan
        else:
            result = spearmanr(x[valid], y[valid])
            rho, p_value = float(result.statistic), float(result.pvalue)
        output.append(
            {
                "dataset": dataset,
                "partition": partition,
                "distance": distance,
                "feature": feature,
                "n_cells": int(valid.sum()),
                "spearman_rho": rho,
                "p_value_exploratory": p_value,
                "analysis_scope": "exploratory_leave_dataset_uncontrolled",
            }
        )
    return output


def _decision_text(paths: ProtocolV2Paths, control_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# E3 dataset mechanism decision",
        "",
        "This is a mechanism diagnosis, not an adaptive-K method or a test-set selection rule.",
        "All categories are descriptive and use the declared E3-A/E3-B/C summaries.",
        "",
    ]
    for dataset in ("clinc150", "banking77", "stackoverflow"):
        dataset_rows = [row for row in control_rows if row.get("dataset") == dataset and row.get("oos_source") == "combined" and row.get("status") == "complete"]
        if not dataset_rows:
            category = "insufficient_evidence"
            reason = "No complete E3-A rows were found."
        else:
            deltas = []
            km_random = []
            for row in dataset_rows:
                ref = next(
                    (value for value in control_rows if value.get("dataset") == dataset and value.get("kir") == row.get("kir") and value.get("seed") == row.get("seed") and value.get("k") == row.get("k") and value.get("distance") == row.get("distance") and value.get("partition") == "random_balanced" and value.get("oos_source") == "combined" and value.get("status") == "complete"),
                    None,
                )
                if row.get("partition") == "kmeans" and ref:
                    km_random.append(float(row["oos_f1"]) - float(ref["oos_f1"]))
                if row.get("partition") == "kmeans":
                    # The paired CSV supplies the exact K=1 relation; this
                    # approximation is intentionally descriptive at closeout.
                    deltas.append(float(row["oos_f1"]))
            stability = [row for row in stability_rows if row.get("dataset") == dataset and row.get("partition") == "kmeans"]
            ari = np.asarray([float(row["pairwise_ari_median"]) for row in stability if _finite(row.get("pairwise_ari_median"))], dtype=float)
            tiny = np.asarray([float(row["tiny_cluster_ratio"]) for row in stability if _finite(row.get("tiny_cluster_ratio"))], dtype=float)
            mean_km_random = float(np.mean(km_random)) if km_random else math.nan
            median_ari = float(np.median(ari)) if ari.size else math.nan
            median_tiny = float(np.median(tiny)) if tiny.size else math.nan
            if _finite(mean_km_random) and mean_km_random > 0.01 and _finite(median_ari) and median_ari >= 0.7 and (_not_finite_or_small(median_tiny)):
                category = "real_multimodal_structure"
                reason = f"KMeans-random OOS F1 mean={mean_km_random:.4f}; median ARI={median_ari:.3f}; tiny ratio={median_tiny:.3f}."
            elif _finite(mean_km_random) and abs(mean_km_random) <= 0.01:
                category = "random_multisphere_effect"
                reason = f"KMeans and random-balanced are close (mean OOS F1 difference={mean_km_random:.4f})."
            elif _finite(median_ari) and median_ari < 0.5 and _finite(median_tiny) and median_tiny > 0.2:
                category = "fragmentation_failure"
                reason = f"Median ARI={median_ari:.3f} with median tiny-cluster ratio={median_tiny:.3f}."
            else:
                category = "boundary_union_failure"
                reason = f"Evidence is stable enough to separate partition effects, but does not support a semantic-cluster claim (KMeans-random mean={mean_km_random:.4f})."
        lines.extend([f"## {dataset}", "", f"**Category:** `{category}`", "", reason, ""])
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "* The E3 evidence does not define or validate a final adaptive-K policy.",
            "* Reliability associations are exploratory and must not be described as causal.",
            "* E4–E7 were not started by this study.",
        ]
    )
    return "\n".join(lines) + "\n"


def _not_finite_or_small(value: float) -> bool:
    return not _finite(value) or value < 0.2


def summarize_e3(paths: ProtocolV2Paths, partition_config: Path, diagnostic_config: Path) -> dict[str, Any]:
    specs = partition_control_specs(partition_config)
    control_rows, paired_rows = summarize_partition_control(paths, specs)
    stability_rows, reliability_features = _flatten_diagnostics(paths)
    reliability_association = _association_rows(paths, reliability_features, control_rows, specs)
    root = _e3_root(paths) / "summaries"
    _write_csv(root / "E3_partition_control_summary.csv", control_rows, sorted({key for row in control_rows for key in row}))
    _write_csv(root / "E3_partition_paired_effects.csv", paired_rows, sorted({key for row in paired_rows for key in row}))
    _write_csv(root / "E3_cluster_stability.csv", stability_rows, sorted({key for row in stability_rows for key in row}))
    stability_summary: list[dict[str, Any]] = []
    groups: dict[tuple[str, Any, Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        groups[(row["dataset"], row["kir"], row["k"], row["partition"], row["distance"])].append(row)
    numeric = ("pairwise_ari_mean", "pairwise_ari_median", "pairwise_ari_min", "centroid_drift", "tiny_cluster_ratio", "silhouette_mean", "silhouette_min", "silhouette_p10", "radius_cv", "calibration_coverage", "calibration_false_rejection", "cluster_size_cv")
    for key, group_rows in sorted(groups.items()):
        summary = dict(zip(("dataset", "kir", "k", "partition", "distance"), key, strict=True))
        summary["n_partition_seeds"] = len({row["partition_seed"] for row in group_rows})
        for field in numeric:
            values = np.asarray([float(row.get(field, math.nan)) for row in group_rows], dtype=float)
            summary[f"mean_{field}"] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
            summary[f"median_{field}"] = float(np.nanmedian(values)) if np.isfinite(values).any() else math.nan
        stability_summary.append(summary)
    _write_csv(root / "E3_cluster_stability_summary.csv", stability_summary, sorted({key for row in stability_summary for key in row}))
    tiny_rows = [
        {key: row.get(key) for key in ("dataset", "kir", "seed", "k", "partition", "partition_seed", "distance", "cluster_count", "min_cluster_size", "max_cluster_size", "tiny_cluster_count_lt3", "tiny_cluster_count_lt5", "tiny_cluster_count_lt5pct", "tiny_cluster_ratio", "cluster_size_cv")}
        for row in stability_rows
    ]
    _write_csv(root / "E3_tiny_cluster_analysis.csv", tiny_rows, sorted({key for row in tiny_rows for key in row}))
    coverage_rows = [
        {key: row.get(key) for key in ("dataset", "kir", "seed", "k", "partition", "partition_seed", "distance", "calibration_coverage", "calibration_false_rejection", "coverage_sample_count", "radius_mean", "radius_std", "radius_cv")}
        for row in stability_rows
    ]
    _write_csv(root / "E3_known_coverage_analysis.csv", coverage_rows, sorted({key for row in coverage_rows for key in row}))
    _write_csv(root / "E3_reliability_features.csv", reliability_features, sorted({key for row in reliability_features for key in row}))
    _write_csv(root / "E3_reliability_association.csv", reliability_association, sorted({key for row in reliability_association for key in row}))
    atomic_write_text(root / "E3_dataset_mechanism_decision.md", _decision_text(paths, control_rows, stability_rows))
    failed = [row for row in control_rows if row.get("status") != "complete"]
    _write_csv(root / "E3_failed_or_invalid_runs.csv", failed, sorted({key for row in failed for key in row}) if failed else ["run_id", "status"])
    planned = len(specs)
    complete = len({row["run_id"] for row in control_rows if row.get("status") == "complete"})
    diagnostics_expected = len(diagnostic_groups(diagnostic_config))
    diagnostics_complete = len(list((_e3_root(paths) / "diagnostics" / "groups").glob("*.json")))
    closeout = {
        "schema_version": "s2c.e3_closeout.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol_version": paths.dataset_version,
        "stage": "E3",
        "partition_control": {"planned": planned, "completed": complete, "failed_or_invalid": planned - complete},
        "diagnostics": {"planned_groups": diagnostics_expected, "completed_groups": diagnostics_complete, "partition_seeds": list(diagnostic_partition_seeds(diagnostic_config))},
        "e2_reference": "K=1 runs are referenced; no E2 run was written.",
        "test_set_selection": False,
        "formal_adaptive_k": False,
        "e4_to_e7_started": False,
        "summaries": sorted(path.name for path in root.iterdir() if path.is_file()),
    }
    atomic_write_json(root / "E3_closeout_manifest.json", closeout)
    report = "\n".join(
        [
            "# E3 integrity report",
            "",
            f"* Protocol: `{paths.dataset_version}`",
            f"* E3-A planned/completed/failed: `{planned}/{complete}/{planned - complete}`",
            f"* E3-B/C diagnostic groups planned/completed: `{diagnostics_expected}/{diagnostics_complete}`",
            "* K=1 is read-only E2 reference; E2 artifacts are not modified.",
            "* Stability and reliability features use train/calibration only.",
            "* No adaptive-K rule is declared; E4–E7 were not started.",
        ]
    ) + "\n"
    atomic_write_text(root / "E3_integrity_report.md", report)
    return closeout
