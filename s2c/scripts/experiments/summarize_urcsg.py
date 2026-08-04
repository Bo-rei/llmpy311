#!/usr/bin/env python3
"""Summarize URCSG pilot cells and apply the pre-registered decision gates."""

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
from protocol_v2.experiments.urcsg import paired_summary  # noqa: E402
from protocol_v2.runtime.paths import ProtocolV2Paths  # noqa: E402


EXPERIMENT_ID = "urcsg_pilot_v1"
PRIMARY = "URCSG-primary"
SINGLE = "single_centroid"


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
    writer = csv.DictWriter(
        buffer,
        fieldnames=fields,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    atomic_write_text(path, buffer.getvalue())


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _bootstrap_ci(values: list[float], seed: int = 20260725) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(10_000, len(array)))
    means = array[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _load_rows(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    metrics: list[dict[str, str]] = []
    selections: list[dict[str, str]] = []
    mechanisms: list[dict[str, Any]] = []
    for path in sorted((root / "runs").glob("*/seed_*/metrics.csv")):
        metrics.extend(_read_csv(path))
        selection_path = path.with_name("intent_selection.csv")
        selections.extend(_read_csv(selection_path))
        mechanism_path = path.with_name("mechanism.json")
        if mechanism_path.is_file():
            mechanisms.append(json.loads(mechanism_path.read_text(encoding="utf-8")))
    return metrics, selections, mechanisms


def _repair_selection_schema(root: Path) -> None:
    """Add the stable generic selection fields without rerunning a cell.

    This is a serialization migration for completed pilot artifacts.  It must
    not change any risk estimate or selected map; the strategy-specific fields
    remain the source of truth for audit comparisons.
    """

    for path in sorted((root / "runs").glob("*/seed_*/intent_selection.csv")):
        rows = _read_csv(path)
        if not rows:
            continue
        changed = False
        for row in rows:
            if "selected_k" not in row:
                row["selected_k"] = row.get("selected_k_primary", "")
                changed = True
            if "selection_reason" not in row:
                row["selection_reason"] = row.get("selection_reason_primary", "")
                changed = True
            if "ineligible" not in row:
                eligible = int(float(row.get("eligible_episode_count", "0") or 0))
                pseudo = int(float(row.get("pseudo_oos_count", "0") or 0))
                row["ineligible"] = str(eligible < 1 or pseudo == 0).lower()
                changed = True
            if "skip_reason" not in row:
                row["skip_reason"] = (
                    "fewer_than_two_remaining_known_intents_or_empty_calibration"
                    if row["ineligible"] == "true"
                    else ""
                )
                changed = True
        if changed:
            _write_csv(path, rows)


def _group_summary(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in metrics:
        if row.get("method") == "oracle_test_k":
            continue
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    result: list[dict[str, Any]] = []
    for (dataset, method), rows in sorted(groups.items()):
        for metric in ("oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate", "accuracy", "auroc", "aupr_oos"):
            values = [_float(row, metric) for row in rows if row.get(metric, "") != ""]
            low, high = _bootstrap_ci(values)
            result.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)) if values else math.nan,
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return result


def _paired(metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    index = {(r["dataset"], int(r["seed"]), r["method"]): r for r in metrics if r.get("method") != "oracle_test_k"}
    rows: list[dict[str, Any]] = []
    for dataset in sorted({r["dataset"] for r in metrics}):
        for method in ("fixed_k2", "fixed_k4", "BRAK", PRIMARY, "URCSG-largest-feasible", "URCSG-shuffled-primary"):
            for metric in ("oos_f1", "f1_all", "known_recall", "false_accept_rate"):
                deltas = []
                for seed in sorted({int(r["seed"]) for r in metrics if r["dataset"] == dataset}):
                    left = index.get((dataset, seed, method))
                    right = index.get((dataset, seed, SINGLE))
                    if left and right:
                        deltas.append(float(left[metric]) - float(right[metric]))
                rows.append({"dataset": dataset, "method": method, "metric": metric, **paired_summary(deltas)})
    return rows


def _selection_summary(selections: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in selections:
        grouped.setdefault((row["dataset"], row["intent"], int(row["candidate_k"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (dataset, intent, k), rows in sorted(grouped.items()):
        first = rows[0]
        result.append(
            {
                "dataset": dataset,
                "intent": intent,
                "candidate_k": k,
                "n_seeds": len(rows),
                "mean_delta_union_risk": float(np.mean([float(r["delta_union_risk"]) for r in rows])),
                "mean_union_risk_ucb95": float(np.mean([float(r["union_risk_ucb95"]) for r in rows])),
                "mean_coverage_delta": float(np.mean([float(r["coverage_delta"]) for r in rows])),
                "feasible_fraction": float(np.mean([str(r["feasible"]).lower() == "true" for r in rows])),
                "selected_k_primary_mode": first.get("selected_k_primary", ""),
                "selected_k_largest_mode": first.get("selected_k_largest", ""),
                "selected_k_shuffled_primary_mode": first.get("selected_k_shuffled_primary", ""),
                "selected_k_mode": first.get("selected_k", first.get("selected_k_primary", "")),
                "selected_k": first.get("selected_k", first.get("selected_k_primary", "")),
                "selection_reason": first.get("selection_reason", first.get("selection_reason_primary", "")),
                "ineligible_fraction": float(np.mean([str(r.get("ineligible", "false")).lower() == "true" for r in rows])),
            }
        )
    return result


def _decision(metrics: list[dict[str, str]], selections: list[dict[str, str]], mechanisms: list[dict[str, Any]]) -> dict[str, Any]:
    index = {(r["dataset"], int(r["seed"]), r["method"]): r for r in metrics}
    gates: dict[str, Any] = {}
    for dataset in ("banking77", "stackoverflow"):
        seeds = sorted({int(r["seed"]) for r in metrics if r["dataset"] == dataset})
        deltas = {metric: [] for metric in ("oos_f1", "f1_all", "known_recall", "false_accept_rate")}
        direction: list[bool] = []
        for seed in seeds:
            primary = index.get((dataset, seed, PRIMARY))
            single = index.get((dataset, seed, SINGLE))
            if not primary or not single:
                continue
            for metric in deltas:
                deltas[metric].append(float(primary[metric]) - float(single[metric]))
            direction.append(float(primary["f1_all"]) >= float(single["f1_all"]))
        selected = [int(r["selected_k_primary"]) for r in selections if r["dataset"] == dataset and r.get("candidate_k") == "1"]
        # Each intent has one row for candidate K=1 per seed; this is a stable
        # denominator for the pre-registered fraction of intents selecting K>1.
        gt1_fraction = float(np.mean(np.asarray(selected) > 1)) if selected else math.nan
        if dataset == "banking77":
            passed = bool(
                np.mean(deltas["oos_f1"]) >= 0.02
                and np.mean(deltas["f1_all"]) >= 0.01
                and np.mean(deltas["known_recall"]) >= -0.02
                and gt1_fraction >= 0.20
                and sum(direction) >= 2
            ) if deltas["oos_f1"] else False
        else:
            passed = bool(
                np.mean(deltas["f1_all"]) >= -0.005
                and np.mean(deltas["false_accept_rate"]) <= 0.01
                and gt1_fraction <= 0.20
                and sum(direction) >= 2
            ) if deltas["f1_all"] else False
        gates[dataset] = {
            "pass": passed,
            "mean_delta": {metric: float(np.mean(values)) if values else math.nan for metric, values in deltas.items()},
            "n_seed_deltas": len(deltas["f1_all"]),
            "direction_consistent_seed_count": sum(direction),
            "selected_k_gt1_fraction": gt1_fraction,
        }
    actual = [float(r["oos_f1"]) - float(index[(r["dataset"], int(r["seed"]), SINGLE)]["oos_f1"]) for r in metrics if r.get("method") == PRIMARY]
    shuffled = [float(r["oos_f1"]) - float(index[(r["dataset"], int(r["seed"]), SINGLE)]["oos_f1"]) for r in metrics if r.get("method") == "URCSG-shuffled-primary"]
    shuffled_weakens = bool(shuffled and actual and abs(float(np.mean(shuffled))) < abs(float(np.mean(actual))))
    if gates.get("banking77", {}).get("pass") and gates.get("stackoverflow", {}).get("pass") and shuffled_weakens:
        decision = "both_dataset_gates_pass_plan_full_matrix"
        failure_reason = None
    elif not gates.get("banking77", {}).get("pass"):
        decision = "stop"
        failure_reason = "banking77_gate_failed"
    elif not gates.get("stackoverflow", {}).get("pass"):
        decision = "stop"
        failure_reason = "stackoverflow_gate_failed"
    else:
        decision = "stop"
        failure_reason = "shuffled_control_did_not_weaken_advantage"
    return {"experiment_id": EXPERIMENT_ID, "decision": decision, "failure_reason": failure_reason, "dataset_gates": gates, "shuffled_control_weakens": shuffled_weakens, "test_used_for_selection": False}


def summarize(root: Path, results_root: Path) -> dict[str, Any]:
    _repair_selection_schema(root)
    metrics, selections, mechanisms = _load_rows(root)
    out = results_root / "diagnostics" / "urcsg"
    out.mkdir(parents=True, exist_ok=True)
    summary = _group_summary(metrics)
    paired = _paired(metrics)
    selection_summary = _selection_summary(selections)
    decision = _decision(metrics, selections, mechanisms)
    _write_csv(out / "pilot_summary.csv", summary)
    _write_csv(out / "intent_selection.csv", selection_summary)
    _write_csv(out / "mechanism_analysis.csv", paired)
    shuffled_rows = [row for row in paired if "shuffled" in row["method"]]
    _write_csv(out / "shuffled_control.csv", shuffled_rows)
    atomic_write_json(out / "decision.json", decision)
    # The registry points at lightweight artifact summaries for auditability;
    # the public-facing copies remain under results/diagnostics/urcsg.
    artifact_summary = root / "summaries"
    _write_csv(artifact_summary / "URCSG_PILOT_SUMMARY.csv", summary)
    _write_csv(artifact_summary / "URCSG_INTENT_SELECTION.csv", selection_summary)
    _write_csv(artifact_summary / "URCSG_MECHANISM_ANALYSIS.csv", paired)
    _write_csv(artifact_summary / "URCSG_SHUFFLED_CONTROL.csv", shuffled_rows)
    atomic_write_json(artifact_summary / "URCSG_DECISION.json", decision)
    return {"metrics_rows": len(metrics), "selection_rows": len(selections), **decision}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    artifact_dir = args.artifact_dir or paths.run_root / EXPERIMENT_ID
    results_dir = args.results_dir or paths.results_root
    result = summarize(artifact_dir, results_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
