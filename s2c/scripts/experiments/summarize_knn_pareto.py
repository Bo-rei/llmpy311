"""Summarize the fixed KNN Known-only Pareto baseline across the active protocol."""

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
BASELINES = ("single_centroid", "fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary")
METRICS = (
    "oos_f1",
    "oos_precision",
    "oos_recall",
    "known_macro_f1",
    "known_recall",
    "f1_all",
    "accuracy",
    "auroc",
    "aupr_oos",
    "fpr95",
    "false_accept_rate",
    "false_reject_rate",
)


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _source_run(root: Path, dataset: str, kir: float, seed: int) -> tuple[Path, bool]:
    fresh = root / "knn_pareto_k10_alpha05" / dataset / f"kir_{kir:.2f}" / f"seed_{seed}"
    if fresh.is_dir():
        return fresh, False
    reused = root / "support_modes_v1" / dataset / f"kir_{kir:.2f}" / f"seed_{seed}"
    if dataset == "stackoverflow" and kir == 0.50 and seed in {13, 42, 87} and reused.is_dir():
        return reused, True
    return fresh, False


def _bootstrap_ci(deltas: np.ndarray, *, seed: int = 20260731) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(deltas, size=(10_000, deltas.size), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def summarize(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []
    manifest_hashes: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                run, reused = _source_run(root, dataset, kir, seed)
                required = ("manifest.json", "metrics.json", "baselines.json", "predictions.csv")
                absent = [name for name in required if not (run / name).is_file()]
                if absent:
                    missing.append(f"{dataset}|{kir:.2f}|{seed}|{','.join(absent)}")
                    continue
                manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
                if (
                    manifest.get("status") != "complete"
                    or manifest.get("test_used_for_selection") is not False
                    or "knn_only" not in metrics
                ):
                    invalid.append(str(run))
                    continue
                value = metrics["knn_only"]
                if not all(math.isfinite(float(value[key])) for key in METRICS):
                    invalid.append(str(run))
                    continue
                rows.append(
                    {
                        "protocol_version": "protocol_v2_textoir_v1",
                        "dataset": dataset,
                        "kir": kir,
                        "seed": seed,
                        "method": "knn_only",
                        "k_neighbors": 10,
                        "alpha": 0.05,
                        **{key: float(value[key]) for key in METRICS},
                        "reused_confirmation_cell": reused,
                        "source_run_dir": str(run),
                    }
                )
                payload = json.loads((run / "baselines.json").read_text(encoding="utf-8"))
                indexed = {item["method"]: item for item in payload["rows"]}
                for method in BASELINES:
                    item = indexed.get(method)
                    if item is None:
                        missing.append(f"{dataset}|{kir:.2f}|{seed}|baseline:{method}")
                        continue
                    baseline_rows.append(
                        {
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "method": method,
                            "oos_f1": float(item["oos_f1"]),
                            "known_recall": float(item["id_recall"]),
                            "f1_all": float(item["f1_all"]),
                            "accuracy": float(item["accuracy"]),
                        }
                    )
                manifest_hashes.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "seed": seed,
                        "sha256": sha256_file(run / "manifest.json"),
                    }
                )

    summary_root = root / "summary" / "knn_pareto_v1"
    fields = list(rows[0]) if rows else ["dataset", "kir", "seed", "method"]
    atomic_write_text(summary_root / "all_runs.csv", _csv_text(rows, fields))
    baseline_fields = list(baseline_rows[0]) if baseline_rows else ["dataset", "kir", "seed", "method"]
    atomic_write_text(summary_root / "baseline_reference.csv", _csv_text(baseline_rows, baseline_fields))

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), float(row["kir"]))].append(row)
    mean_std: list[dict[str, Any]] = []
    for (dataset, kir), selected in sorted(grouped.items()):
        item: dict[str, Any] = {"dataset": dataset, "kir": kir, "method": "knn_only", "n_seeds": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            item[f"mean_{metric}"] = statistics.fmean(values)
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        mean_std.append(item)
    mean_fields = list(mean_std[0]) if mean_std else ["dataset", "kir", "method", "n_seeds"]
    atomic_write_text(summary_root / "mean_std.csv", _csv_text(mean_std, mean_fields))

    by_cell = {(str(row["dataset"]), float(row["kir"]), int(row["seed"])): row for row in rows}
    baselines_by_cell = {
        (str(row["dataset"]), float(row["kir"]), int(row["seed"]), str(row["method"])): row
        for row in baseline_rows
    }
    paired: list[dict[str, Any]] = []
    for (dataset, kir, seed), row in sorted(by_cell.items()):
        for baseline in BASELINES:
            reference = baselines_by_cell[(dataset, kir, seed, baseline)]
            paired.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "seed": seed,
                    "baseline": baseline,
                    "knn_oos_f1": row["oos_f1"],
                    "baseline_oos_f1": reference["oos_f1"],
                    "delta_oos_f1": float(row["oos_f1"]) - float(reference["oos_f1"]),
                    "knn_known_recall": row["known_recall"],
                    "baseline_known_recall": reference["known_recall"],
                    "delta_known_recall": float(row["known_recall"])
                    - float(reference["known_recall"]),
                    "knn_f1_all": row["f1_all"],
                    "baseline_f1_all": reference["f1_all"],
                    "delta_f1_all": float(row["f1_all"]) - float(reference["f1_all"]),
                }
            )
    paired_fields = list(paired[0]) if paired else ["dataset", "kir", "seed", "baseline"]
    atomic_write_text(summary_root / "paired_effects.csv", _csv_text(paired, paired_fields))

    significance: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for baseline in BASELINES:
                selected = [
                    row
                    for row in paired
                    if row["dataset"] == dataset
                    and float(row["kir"]) == kir
                    and row["baseline"] == baseline
                ]
                deltas = np.asarray([float(row["delta_oos_f1"]) for row in selected], dtype=np.float64)
                knn_values = np.asarray([float(row["knn_oos_f1"]) for row in selected])
                reference_values = np.asarray([float(row["baseline_oos_f1"]) for row in selected])
                low, high = _bootstrap_ci(deltas)
                nonzero = deltas[np.abs(deltas) > 1e-12]
                wilcoxon_p = (
                    float(wilcoxon(nonzero, alternative="two-sided").pvalue) if nonzero.size else 1.0
                )
                std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0
                significance.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "baseline": baseline,
                        "n_pairs": deltas.size,
                        "mean_delta_oos_f1": float(np.mean(deltas)),
                        "std_delta_oos_f1": std,
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": int(np.sum(deltas > 1e-12)),
                        "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
                        "losses": int(np.sum(deltas < -1e-12)),
                        "paired_t_p": float(ttest_rel(knn_values, reference_values).pvalue),
                        "wilcoxon_p": wilcoxon_p,
                        "cohen_dz": float(np.mean(deltas) / std) if std > 0 else 0.0,
                    }
                )
    significance_fields = list(significance[0]) if significance else ["dataset", "kir", "baseline"]
    atomic_write_text(summary_root / "significance.csv", _csv_text(significance, significance_fields))

    project_root = ProtocolV2Paths.discover().project_root
    config = project_root / "configs" / "gates" / "knn_pareto_v1.yaml"
    runner = project_root / "scripts" / "experiments" / "run_clmsg.py"
    git_status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--short"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    integrity = {
        "status": "complete" if len(rows) == 45 and not missing and not invalid else "incomplete",
        "planned_cells": 45,
        "completed_cells": len(rows),
        "fresh_cells": sum(not bool(row["reused_confirmation_cell"]) for row in rows),
        "reused_cells": sum(bool(row["reused_confirmation_cell"]) for row in rows),
        "missing": missing,
        "invalid": invalid,
        "unique_cells": len(by_cell),
        "test_used_for_selection": False,
    }
    atomic_write_json(summary_root / "integrity.json", integrity)
    atomic_write_json(
        summary_root / "KNN_PARETO_PROVENANCE.json",
        {
            "stage": "knn_pareto_v1",
            "protocol_version": "protocol_v2_textoir_v1",
            "git_dirty": bool(git_status.strip()),
            "config_sha256": sha256_file(config),
            "runner_sha256": sha256_file(runner),
            "manifest_aggregate_sha256": sha256_json(manifest_hashes),
            "bootstrap_seed": 20260731,
            "bootstrap_resamples": 10_000,
            "test_used_for_selection": False,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    )

    single_stats = {
        (row["dataset"], float(row["kir"])): row
        for row in significance
        if row["baseline"] == "single_centroid"
    }
    report_rows = [
        (
            dataset,
            kir,
            next(row for row in mean_std if row["dataset"] == dataset and float(row["kir"]) == kir),
            single_stats[(dataset, kir)],
        )
        for dataset in DATASETS
        for kir in KIRS
    ]
    table = [
        "| Dataset | KIR | KNN OOS F1 mean±std | Δ vs Single | W/T/L |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, kir, aggregate, comparison in report_rows:
        table.append(
            f"| {dataset} | {kir:.2f} | {float(aggregate['mean_oos_f1']):.4f} ± "
            f"{float(aggregate['std_oos_f1']):.4f} | {float(comparison['mean_delta_oos_f1']):+.4f} | "
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} |"
        )
    atomic_write_text(
        summary_root / "KNN_PARETO_CLOSEOUT.md",
        "\n".join(
            [
                "# KNN Known-only Pareto baseline closeout",
                "",
                "Fixed contract: frozen MiniLM, cosine distance, k=10, Known-calibration alpha=0.05.",
                "No test OOS sample selected k, alpha, threshold, checkpoint, or method.",
                "",
                *table,
                "",
                "The table reports paired five-seed effects against the same-split Single-centroid baseline.",
                "KNN is retained as a full-protocol nonparametric baseline, not promoted to a new method.",
                "",
            ]
        ),
    )
    return integrity


def main() -> int:
    paths = ProtocolV2Paths.discover()
    result = summarize((paths.run_root / "clmsg_v1").resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
