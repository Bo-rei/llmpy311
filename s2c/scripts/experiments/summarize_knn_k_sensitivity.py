"""Summarize the predeclared KNN neighbour-count sensitivity grid."""

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
import yaml
from scipy.stats import ttest_rel, wilcoxon

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file, sha256_json
from protocol_v2.runtime.paths import ProtocolV2Paths


DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
K_VALUES = (5, 10, 20, 30)
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


def _source_run(root: Path, dataset: str, kir: float, seed: int, k: int) -> tuple[Path, bool]:
    variant = root / f"knn_pareto_k{k}_alpha05" / dataset / f"kir_{kir:.2f}" / f"seed_{seed}"
    if variant.is_dir():
        return variant, False
    reused = root / "support_modes_v1" / dataset / f"kir_{kir:.2f}" / f"seed_{seed}"
    if k == 10 and dataset == "stackoverflow" and kir == 0.50 and seed in {13, 42, 87}:
        return reused, True
    return variant, False


def _bootstrap_ci(deltas: np.ndarray, *, seed: int = 20260731) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(deltas, size=(10_000, deltas.size), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def _effect_row(
    *,
    dataset: str,
    kir: float,
    candidate_k: int,
    reference: str,
    deltas: np.ndarray,
    candidate: np.ndarray,
    reference_values: np.ndarray,
) -> dict[str, Any]:
    low, high = _bootstrap_ci(deltas)
    nonzero = deltas[np.abs(deltas) > 1e-12]
    std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0
    return {
        "dataset": dataset,
        "kir": kir,
        "candidate_k": candidate_k,
        "reference": reference,
        "n_pairs": int(deltas.size),
        "mean_delta_oos_f1": float(np.mean(deltas)),
        "median_delta_oos_f1": float(np.median(deltas)),
        "std_delta_oos_f1": std,
        "ci95_low": low,
        "ci95_high": high,
        "wins": int(np.sum(deltas > 1e-12)),
        "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
        "losses": int(np.sum(deltas < -1e-12)),
        "paired_t_p": float(ttest_rel(candidate, reference_values).pvalue),
        "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if nonzero.size else 1.0,
        "cohen_dz": float(np.mean(deltas) / std) if std > 0 else 0.0,
    }


def summarize(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []
    manifest_hashes: list[dict[str, Any]] = []
    for k in K_VALUES:
        for dataset in DATASETS:
            for kir in KIRS:
                for seed in SEEDS:
                    run, reused = _source_run(root, dataset, kir, seed, k)
                    required = ("manifest.json", "metrics.json", "baselines.json", "predictions.csv", "config.yaml")
                    absent = [name for name in required if not (run / name).is_file()]
                    if absent:
                        missing.append(f"k={k}|{dataset}|{kir:.2f}|{seed}|{','.join(absent)}")
                        continue
                    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
                    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
                    value = metrics.get("knn_only")
                    configured_k = int(config.get("k_neighbors", 10))
                    if (
                        manifest.get("status") != "complete"
                        or manifest.get("test_used_for_selection") is not False
                        or value is None
                        or configured_k != k
                        or config.get("protocol_version") != "protocol_v2_textoir_v1"
                    ):
                        invalid.append(str(run))
                        continue
                    if not all(math.isfinite(float(value[key])) for key in METRICS):
                        invalid.append(str(run))
                        continue
                    baselines = json.loads((run / "baselines.json").read_text(encoding="utf-8"))["rows"]
                    indexed_baselines = {item["method"]: item for item in baselines}
                    single = indexed_baselines.get("single_centroid")
                    if single is None or not math.isfinite(float(single["oos_f1"])):
                        invalid.append(str(run))
                        continue
                    rows.append(
                        {
                            "protocol_version": "protocol_v2_textoir_v1",
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "k_neighbors": k,
                            "alpha": 0.05,
                            **{metric: float(value[metric]) for metric in METRICS},
                            "single_centroid_oos_f1": float(single["oos_f1"]),
                            "single_centroid_known_recall": float(single["id_recall"]),
                            "reused_confirmation_cell": reused,
                            "source_run_dir": str(run),
                        }
                    )
                    if k == 5:
                        for method in BASELINES:
                            item = indexed_baselines.get(method)
                            if item is None:
                                missing.append(f"baseline={method}|{dataset}|{kir:.2f}|{seed}")
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
                        {"dataset": dataset, "kir": kir, "seed": seed, "k": k, "sha256": sha256_file(run / "manifest.json")}
                    )

    summary_root = root / "summary" / "knn_k_sensitivity_v1"
    fields = list(rows[0]) if rows else ["dataset", "kir", "seed", "k_neighbors"]
    atomic_write_text(summary_root / "all_runs.csv", _csv_text(rows, fields))
    baseline_fields = list(baseline_rows[0]) if baseline_rows else ["dataset", "kir", "seed", "method"]
    atomic_write_text(summary_root / "baseline_reference.csv", _csv_text(baseline_rows, baseline_fields))

    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), float(row["kir"]), int(row["k_neighbors"]))].append(row)
    mean_std: list[dict[str, Any]] = []
    for (dataset, kir, k), selected in sorted(grouped.items()):
        item: dict[str, Any] = {"dataset": dataset, "kir": kir, "k_neighbors": k, "n_seeds": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            item[f"mean_{metric}"] = statistics.fmean(values)
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        mean_std.append(item)
    mean_fields = list(mean_std[0]) if mean_std else ["dataset", "kir", "k_neighbors", "n_seeds"]
    atomic_write_text(summary_root / "mean_std.csv", _csv_text(mean_std, mean_fields))

    paired_single: list[dict[str, Any]] = []
    for row in rows:
        paired_single.append(
            {
                "dataset": row["dataset"],
                "kir": row["kir"],
                "seed": row["seed"],
                "k_neighbors": row["k_neighbors"],
                "knn_oos_f1": row["oos_f1"],
                "single_centroid_oos_f1": row["single_centroid_oos_f1"],
                "delta_oos_f1": float(row["oos_f1"]) - float(row["single_centroid_oos_f1"]),
                "knn_known_recall": row["known_recall"],
                "single_centroid_known_recall": row["single_centroid_known_recall"],
                "delta_known_recall": float(row["known_recall"]) - float(row["single_centroid_known_recall"]),
            }
        )
    paired_fields = list(paired_single[0]) if paired_single else ["dataset", "kir", "seed", "k_neighbors"]
    atomic_write_text(summary_root / "paired_vs_single.csv", _csv_text(paired_single, paired_fields))

    by_cell = {
        (str(row["dataset"]), float(row["kir"]), int(row["seed"]), int(row["k_neighbors"])): row
        for row in rows
    }
    paired_k10: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                reference = by_cell.get((dataset, kir, seed, 10))
                if reference is None:
                    continue
                for k in (5, 20, 30):
                    candidate = by_cell.get((dataset, kir, seed, k))
                    if candidate is None:
                        continue
                    paired_k10.append(
                        {
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "candidate_k": k,
                            "candidate_oos_f1": candidate["oos_f1"],
                            "k10_oos_f1": reference["oos_f1"],
                            "delta_oos_f1": float(candidate["oos_f1"]) - float(reference["oos_f1"]),
                            "candidate_known_recall": candidate["known_recall"],
                            "k10_known_recall": reference["known_recall"],
                            "delta_known_recall": float(candidate["known_recall"]) - float(reference["known_recall"]),
                        }
                    )
    paired_k_fields = list(paired_k10[0]) if paired_k10 else ["dataset", "kir", "seed", "candidate_k"]
    atomic_write_text(summary_root / "paired_vs_k10.csv", _csv_text(paired_k10, paired_k_fields))

    effects: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for k in K_VALUES:
                selected = [
                    row
                    for row in paired_single
                    if row["dataset"] == dataset and float(row["kir"]) == kir and int(row["k_neighbors"]) == k
                ]
                candidate = np.asarray([float(row["knn_oos_f1"]) for row in selected])
                reference = np.asarray([float(row["single_centroid_oos_f1"]) for row in selected])
                effects.append(
                    _effect_row(
                        dataset=dataset,
                        kir=kir,
                        candidate_k=k,
                        reference="single_centroid",
                        deltas=candidate - reference,
                        candidate=candidate,
                        reference_values=reference,
                    )
                )
            for k in (5, 20, 30):
                selected = [
                    row
                    for row in paired_k10
                    if row["dataset"] == dataset and float(row["kir"]) == kir and int(row["candidate_k"]) == k
                ]
                candidate = np.asarray([float(row["candidate_oos_f1"]) for row in selected])
                reference = np.asarray([float(row["k10_oos_f1"]) for row in selected])
                effects.append(
                    _effect_row(
                        dataset=dataset,
                        kir=kir,
                        candidate_k=k,
                        reference="k10",
                        deltas=candidate - reference,
                        candidate=candidate,
                        reference_values=reference,
                    )
                )
    effect_fields = list(effects[0]) if effects else ["dataset", "kir", "candidate_k", "reference"]
    atomic_write_text(summary_root / "paired_effects.csv", _csv_text(effects, effect_fields))
    significance = [
        {
            "dataset": row["dataset"],
            "kir": row["kir"],
            "comparison": row["reference"],
            "k_neighbors": row["candidate_k"],
            **{key: value for key, value in row.items() if key not in {"dataset", "kir", "reference", "candidate_k"}},
        }
        for row in effects
    ]
    atomic_write_text(summary_root / "significance.csv", _csv_text(significance, list(significance[0])))

    dataset_kir_summary: list[dict[str, Any]] = []
    for aggregate in mean_std:
        selected = [
            row
            for row in paired_single
            if row["dataset"] == aggregate["dataset"]
            and float(row["kir"]) == float(aggregate["kir"])
            and int(row["k_neighbors"]) == int(aggregate["k_neighbors"])
        ]
        deltas = np.asarray([float(row["delta_oos_f1"]) for row in selected])
        dataset_kir_summary.append(
            {
                "dataset": aggregate["dataset"],
                "kir": aggregate["kir"],
                "k_neighbors": aggregate["k_neighbors"],
                "n_seeds": aggregate["n_seeds"],
                "mean_oos_f1": aggregate["mean_oos_f1"],
                "std_oos_f1": aggregate["std_oos_f1"],
                "mean_known_recall": aggregate["mean_known_recall"],
                "std_known_recall": aggregate["std_known_recall"],
                "mean_accuracy": aggregate["mean_accuracy"],
                "mean_delta_oos_f1_vs_single": float(np.mean(deltas)),
                "wins": int(np.sum(deltas > 1e-12)),
                "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
                "losses": int(np.sum(deltas < -1e-12)),
            }
        )
    atomic_write_text(
        summary_root / "dataset_kir_summary.csv",
        _csv_text(dataset_kir_summary, list(dataset_kir_summary[0])),
    )

    descriptive_best: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            selected = [row for row in mean_std if row["dataset"] == dataset and float(row["kir"]) == kir]
            best = max(selected, key=lambda row: float(row["mean_oos_f1"]))
            descriptive_best.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "descriptive_oracle_k": best["k_neighbors"],
                    "mean_oos_f1": best["mean_oos_f1"],
                    "selection_allowed": False,
                }
            )
    atomic_write_text(
        summary_root / "descriptive_best_k.csv",
        _csv_text(descriptive_best, list(descriptive_best[0])),
    )

    project_root = ProtocolV2Paths.discover().project_root
    git_status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--short"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    integrity = {
        "status": "complete" if len(rows) == 180 and len(by_cell) == 180 and not missing and not invalid else "incomplete",
        "planned_cells": 180,
        "completed_cells": len(rows),
        "new_k_cells": sum(int(row["k_neighbors"]) != 10 for row in rows),
        "existing_k10_cells": sum(int(row["k_neighbors"]) == 10 for row in rows),
        "reused_k10_cells": sum(bool(row["reused_confirmation_cell"]) for row in rows),
        "unique_cells": len(by_cell),
        "missing": missing,
        "invalid": invalid,
        "test_used_for_selection": False,
    }
    atomic_write_json(summary_root / "integrity.json", integrity)
    atomic_write_json(
        summary_root / "KNN_K_SENSITIVITY_PROVENANCE.json",
        {
            "stage": "knn_k_sensitivity_v1",
            "protocol_version": "protocol_v2_textoir_v1",
            "git_dirty": bool(git_status.strip()),
            "runner_sha256": sha256_file(project_root / "scripts" / "experiments" / "run_clmsg.py"),
            "summarizer_sha256": sha256_file(Path(__file__).resolve()),
            "manifest_aggregate_sha256": sha256_json(manifest_hashes),
            "bootstrap_seed": 20260731,
            "bootstrap_resamples": 10_000,
            "test_used_for_selection": False,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    )

    overall: list[dict[str, Any]] = []
    for k in K_VALUES:
        selected = [row for row in paired_single if int(row["k_neighbors"]) == k]
        deltas = np.asarray([float(row["delta_oos_f1"]) for row in selected])
        oos = np.asarray([float(row["knn_oos_f1"]) for row in selected])
        references = np.asarray([float(row["single_centroid_oos_f1"]) for row in selected])
        low, high = _bootstrap_ci(deltas)
        nonzero = deltas[np.abs(deltas) > 1e-12]
        std_delta = float(np.std(deltas, ddof=1))
        overall.append(
            {
                "k_neighbors": k,
                "n_cells": len(selected),
                "mean_oos_f1": float(np.mean(oos)),
                "std_oos_f1": float(np.std(oos, ddof=1)),
                "mean_single_centroid_oos_f1": float(np.mean(references)),
                "mean_delta_oos_f1_vs_single": float(np.mean(deltas)),
                "std_delta_oos_f1_vs_single": std_delta,
                "ci95_low_delta_oos_f1_vs_single": low,
                "ci95_high_delta_oos_f1_vs_single": high,
                "wins": int(np.sum(deltas > 1e-12)),
                "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
                "losses": int(np.sum(deltas < -1e-12)),
                "paired_t_p": float(ttest_rel(oos, references).pvalue),
                "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if nonzero.size else 1.0,
                "cohen_dz": float(np.mean(deltas) / std_delta) if std_delta else 0.0,
            }
        )
    atomic_write_text(summary_root / "k_overview.csv", _csv_text(overall, list(overall[0])))
    table = [
        "| k | Mean OOS F1 | Delta vs Single | W/T/L |",
        "| ---: | ---: | ---: | ---: |",
        *[
            f"| {row['k_neighbors']} | {row['mean_oos_f1']:.4f} | "
            f"{row['mean_delta_oos_f1_vs_single']:+.4f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} |"
            for row in overall
        ],
    ]
    atomic_write_text(
        summary_root / "KNN_K_SENSITIVITY_CLOSEOUT.md",
        "\n".join(
            [
                "# KNN neighbour-count sensitivity closeout",
                "",
                "Fixed contract: frozen MiniLM, cosine distance, Known-only order-statistic calibration at alpha=0.05.",
                "All four k values are reported. Test OOS was not used to select k, alpha, threshold, or method.",
                "",
                *table,
                "",
                "The descriptive best-k file is an oracle sensitivity summary only and is not an authorized selection rule.",
                "This stage closes the ordinary KNN neighbour-count question; it does not promote a test-selected k.",
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
