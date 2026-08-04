#!/usr/bin/env python3
"""Summarize CCSG pilot cells and apply the pre-registered stop gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text  # noqa: E402


EXPERIMENT_ID = "ccsg_pilot_v1"
SINGLE = "current_k1"
CCSG_K1 = "ccsg_k1"
CCSG_K2 = "ccsg_k2"
CURRENT_K2 = "current_k2_union"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "\n")
        return
    fields = sorted({key for row in rows for key in row})
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    atomic_write_text(path, buffer.getvalue())


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _ci(values: list[float], seed: int = 20260725) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(10_000, array.size))
    means = array[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _load(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    metrics: list[dict[str, str]] = []
    calibrations: list[dict[str, str]] = []
    mechanisms: list[dict[str, str]] = []
    for metrics_path in sorted((root / "runs").glob("*/seed_*/metrics.csv")):
        metrics.extend(_read_csv(metrics_path))
        calibrations.extend(_read_csv(metrics_path.with_name("calibration.csv")))
        mechanisms.extend(_read_csv(metrics_path.with_name("mechanism.csv")))
    return metrics, calibrations, mechanisms


def _summary(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in metrics:
        groups.setdefault((row["dataset"], row["method"], row["k"]), []).append(row)
    for (dataset, method, k), values in sorted(groups.items()):
        for metric in ("oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate", "accuracy", "auroc", "aupr_oos"):
            numbers = [_float(row, metric) for row in values]
            low, high = _ci(numbers)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "k": int(k),
                    "metric": metric,
                    "n_seeds": len(numbers),
                    "mean": float(np.mean(numbers)),
                    "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return rows


def _paired(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    index = {(row["dataset"], int(row["seed"]), row["method"]): row for row in metrics}
    comparisons = ((CCSG_K2, CCSG_K1), (CCSG_K2, CURRENT_K2), (CCSG_K1, SINGLE))
    rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in metrics}):
        seeds = sorted({int(row["seed"]) for row in metrics if row["dataset"] == dataset})
        for left_method, right_method in comparisons:
            for metric in ("oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate"):
                deltas = [
                    _float(index[(dataset, seed, left_method)], metric) - _float(index[(dataset, seed, right_method)], metric)
                    for seed in seeds
                    if (dataset, seed, left_method) in index and (dataset, seed, right_method) in index
                ]
                values = np.asarray(deltas, dtype=np.float64)
                low, high = _ci(deltas)
                rows.append(
                    {
                        "dataset": dataset,
                        "left_method": left_method,
                        "right_method": right_method,
                        "metric": metric,
                        "n": len(deltas),
                        "mean_delta": float(np.mean(values)) if values.size else math.nan,
                        "std_delta": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": int(np.sum(values > 1e-12)),
                        "ties": int(np.sum(np.abs(values) <= 1e-12)),
                        "losses": int(np.sum(values < -1e-12)),
                    }
                )
    return rows


def _mean_delta(metrics: list[dict[str, str]], dataset: str, left: str, right: str, metric: str) -> tuple[float, int, int]:
    index = {(row["dataset"], int(row["seed"]), row["method"]): row for row in metrics}
    values = [
        _float(index[(dataset, seed, left)], metric) - _float(index[(dataset, seed, right)], metric)
        for seed in sorted({int(row["seed"]) for row in metrics if row["dataset"] == dataset})
        if (dataset, seed, left) in index and (dataset, seed, right) in index
    ]
    return (float(np.mean(values)) if values else math.nan, sum(value >= 0 for value in values), len(values))


def _decision(metrics: list[dict[str, str]]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for dataset in ("banking77", "stackoverflow", "clinc150"):
        k2_f1, k2_direction, n = _mean_delta(metrics, dataset, CCSG_K2, CCSG_K1, "f1_all")
        k2_oos, _, _ = _mean_delta(metrics, dataset, CCSG_K2, CCSG_K1, "oos_f1")
        k2_known, _, _ = _mean_delta(metrics, dataset, CCSG_K2, CCSG_K1, "known_recall")
        k2_far, _, _ = _mean_delta(metrics, dataset, CCSG_K2, CCSG_K1, "false_accept_rate")
        versus_union, _, _ = _mean_delta(metrics, dataset, CCSG_K2, CURRENT_K2, "f1_all")
        k1_gain, k1_direction, _ = _mean_delta(metrics, dataset, CCSG_K1, SINGLE, "f1_all")
        k1_oos, _, _ = _mean_delta(metrics, dataset, CCSG_K1, SINGLE, "oos_f1")
        k1_known, _, _ = _mean_delta(metrics, dataset, CCSG_K1, SINGLE, "known_recall")
        if dataset == "banking77":
            passed = bool((k2_oos >= 0.02 or k2_f1 >= 0.02) and k2_known >= -0.02 and k2_direction >= 2 and versus_union > 0)
        elif dataset == "stackoverflow":
            passed = bool(k2_far <= 0.01 and k2_f1 >= -0.005 and k2_known >= -0.02 and k2_direction >= 2)
        else:
            passed = bool(k2_f1 >= -0.005 and k2_known >= -0.01)
        gates[dataset] = {
            "pass": passed,
            "n_seeds": n,
            "ccsg_k2_minus_k1": {"oos_f1": k2_oos, "f1_all": k2_f1, "known_recall": k2_known, "false_accept_rate": k2_far},
            "ccsg_k2_minus_current_k2_union_f1_all": versus_union,
            "ccsg_k1_minus_current_k1": {"oos_f1": k1_oos, "f1_all": k1_gain, "known_recall": k1_known},
            "direction_nonnegative_f1_all": k2_direction,
        }
    k1_effective = all(
        gates[dataset]["ccsg_k1_minus_current_k1"]["f1_all"] >= 0.02
        and gates[dataset]["ccsg_k1_minus_current_k1"]["known_recall"] >= -0.02
        for dataset in ("banking77", "stackoverflow", "clinc150")
    )
    if all(item["pass"] for item in gates.values()):
        decision = "plan_ccsg_full_matrix"
        reason = None
    elif k1_effective:
        decision = "ccsg_single_support_gate_only"
        reason = "K2_did_not_pass_but_K1_joint_calibration_improved_all_datasets"
    else:
        decision = "stop_ccsg_pilot"
        reason = "one_or_more_pre_registered_dataset_gates_failed"
    return {
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "failure_reason": reason,
        "dataset_gates": gates,
        "test_used_for_selection": False,
        "selection_data": "known_calibration_only",
    }


def summarize(root: Path, results_root: Path) -> dict[str, Any]:
    metrics, calibrations, mechanisms = _load(root)
    out = results_root / "diagnostics" / "ccsg"
    out.mkdir(parents=True, exist_ok=True)
    summary = _summary(metrics)
    paired = _paired(metrics)
    decision = _decision(metrics)
    _write_csv(out / "pilot_summary.csv", summary)
    _write_csv(out / "mechanism_ablation.csv", summary)
    _write_csv(out / "k_effects.csv", paired)
    _write_csv(out / "calibration_summary.csv", calibrations)
    _write_csv(out / "sphere_statistics.csv", mechanisms)
    atomic_write_json(out / "decision.json", decision)
    closeout = (
        "# CCSG pilot closeout\n\n"
        f"- Experiment: `{EXPERIMENT_ID}`\n"
        f"- Decision: `{decision['decision']}`\n"
        f"- Test used for selection: `{decision['test_used_for_selection']}`\n"
        f"- Selection data: `{decision['selection_data']}`\n"
        "- CCSG is a scoring/calibration pilot; no encoder was trained and no prior E2/E3/URCSG run was modified.\n"
    )
    atomic_write_text(out / "CCSG_CLOSEOUT.md", closeout)
    return {"experiment_id": EXPERIMENT_ID, "metric_rows": len(metrics), "summary_rows": len(summary), "decision": decision}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args(argv)
    paths = __import__("protocol_v2.runtime.paths", fromlist=["ProtocolV2Paths"]).ProtocolV2Paths.discover()
    result = summarize(args.artifact_dir, args.results_dir or paths.results_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
