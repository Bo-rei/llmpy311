#!/usr/bin/env python3
"""Audit adaptive-K evidence from the frozen E2 Gate runs.

This command is intentionally read-only with respect to ``../artifacts``.  It
recomputes the paper-facing metrics from the immutable per-sample predictions
and writes only compact diagnostics under ``results/diagnostics/adaptive_k``.
The chosen K values are descriptive test-set optima; they are never promoted
to a validation-time selection rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT.parent / "artifacts/s2c/runs/protocol_v2_textoir_v1/e2_gate_core_dense"
DEFAULT_OUTPUT = ROOT / "results/diagnostics/adaptive_k"


def _float(value: Any) -> float:
    return float(value)


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _metrics(rows: list[dict[str, Any]], known_labels: list[str]) -> dict[str, float]:
    labels = [*known_labels, "__OOS__"]
    confusion: Counter[tuple[str, str]] = Counter()
    for row in rows:
        gold = "__OOS__" if int(row["gold_is_oos"]) else str(row["gold_intent"])
        pred = "__OOS__" if int(row["predicted_is_oos"]) else str(row["nearest_known_intent"])
        confusion[(gold, pred)] += 1
    f1_by_label: dict[str, float] = {}
    for label in labels:
        tp = confusion[(label, label)]
        fp = sum(value for (gold, pred), value in confusion.items() if pred == label and gold != label)
        fn = sum(value for (gold, pred), value in confusion.items() if gold == label and pred != label)
        f1_by_label[label] = _f1(tp, fp, fn)
    known_rows = [row for row in rows if not int(row["gold_is_oos"])]
    accepted_known = sum(not int(row["predicted_is_oos"]) for row in known_rows)
    return {
        "oos_f1": f1_by_label["__OOS__"],
        "f1_all": sum(f1_by_label.values()) / len(labels) if labels else 0.0,
        "f1_known": sum(f1_by_label[label] for label in known_labels) / len(known_labels) if known_labels else 0.0,
        "accuracy": sum(
            ("__OOS__" if int(row["gold_is_oos"]) else str(row["gold_intent"]))
            == ("__OOS__" if int(row["predicted_is_oos"]) else str(row["nearest_known_intent"]))
            for row in rows
        ) / len(rows)
        if rows
        else 0.0,
        "known_recall": accepted_known / len(known_rows) if known_rows else 0.0,
    }


def _intent_rows(rows: list[dict[str, Any]], intent: str) -> dict[str, float]:
    known = [row for row in rows if not int(row["gold_is_oos"])]
    intent_known = [row for row in known if str(row["gold_intent"]) == intent]
    oos = [row for row in rows if int(row["gold_is_oos"])]
    accepted_intent = [
        row for row in rows if not int(row["predicted_is_oos"]) and str(row["nearest_known_intent"]) == intent
    ]
    tp = sum(
        not int(row["predicted_is_oos"]) and str(row["nearest_known_intent"]) == intent
        for row in intent_known
    )
    fp = sum(
        not int(row["predicted_is_oos"]) and str(row["nearest_known_intent"]) == intent
        for row in rows
        if int(row["gold_is_oos"]) or str(row["gold_intent"]) != intent
    )
    fn = len(intent_known) - tp
    # This binary per-intent score treats one Known intent as positive and
    # every other class/OOS sample as negative.  It is not global OOS F1;
    # keeping the definition explicit prevents it being mistaken for one.
    return {
        "intent_oos_f1": _f1(tp, fp, fn),
        "intent_known_recall": tp / len(intent_known) if intent_known else 0.0,
        "intent_false_reject_rate": sum(int(row["predicted_is_oos"]) for row in intent_known) / len(intent_known)
        if intent_known
        else 0.0,
        "intent_false_accept_count": sum(
            not int(row["predicted_is_oos"]) and str(row["nearest_known_intent"]) == intent for row in oos
        ),
        "intent_false_accept_rate": sum(
            not int(row["predicted_is_oos"]) and str(row["nearest_known_intent"]) == intent for row in oos
        )
        / len(oos)
        if oos
        else 0.0,
        "intent_f1_support": len(intent_known),
        "accepted_intent_count": len(accepted_intent),
    }


def _read_registry(dataset: str, kir: float, seed: int) -> dict[str, Any]:
    path = ROOT / "data/registries/protocol_v2_textoir_v1" / dataset / f"seed_{seed}" / f"kir_{kir:.2f}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _run_rows(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    registry = _read_registry(config["dataset"], float(config["kir"]), int(config["seed"]))
    rows = [json.loads(line) for line in (run_dir / "predictions/test.jsonl").read_text(encoding="utf-8").splitlines()]
    return config, rows, registry


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def audit(run_root: Path, output_root: Path) -> dict[str, Any]:
    run_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    run_records: list[dict[str, Any]] = []
    intent_records: list[dict[str, Any]] = []
    for index, run_dir in enumerate(run_dirs, start=1):
        config, rows, registry = _run_rows(run_dir)
        known = [str(value) for value in registry["known_intents"]]
        metrics = _metrics(rows, known)
        record = {
            "dataset": config["dataset"],
            "kir": f"{float(config['kir']):.2f}",
            "seed": int(config["seed"]),
            "k": int(config["k_gate"]),
            "distance": config["distance"],
            "oos_view": "combined",
            "run_id": manifest_run_id(run_dir),
            **metrics,
        }
        run_records.append(record)
        for intent in known:
            intent_records.append(
                {
                    "dataset": config["dataset"],
                    "kir": f"{float(config['kir']):.2f}",
                    "seed": int(config["seed"]),
                    "k": int(config["k_gate"]),
                    "distance": config["distance"],
                    "intent": intent,
                    "run_id": record["run_id"],
                    **_intent_rows(rows, intent),
                }
            )
        if index % 100 == 0:
            print(f"processed {index}/{len(run_dirs)} E2 runs", flush=True)

    # Each dataset/KIR/distance aggregate is deliberately descriptive.  The
    # best K is selected on the test metric only for sensitivity analysis.
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        groups[(record["dataset"], record["kir"], record["distance"])].append(record)
    dataset_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for (dataset, kir, distance), records in sorted(groups.items()):
        k_values = sorted({int(r["k"]) for r in records})
        means = {
            k: sum(float(r["oos_f1"]) for r in records if int(r["k"]) == k) /
            max(1, sum(1 for r in records if int(r["k"]) == k))
            for k in k_values
        }
        best_k = max(k_values, key=lambda k: (means[k], -k))
        k1 = means.get(1, float("nan"))
        best_records = [r for r in records if int(r["k"]) == best_k]
        dataset_rows.append(
            {
                "dataset": dataset,
                "kir": kir,
                "distance": distance,
                "oos_view": "combined",
                "best_k": best_k,
                "best_k_mean_oos_f1": means[best_k],
                "delta_oos_f1_vs_k1": means[best_k] - k1 if not math.isnan(k1) else "",
                "best_k_mean_f1_all": sum(float(r["f1_all"]) for r in best_records) / len(best_records),
                "delta_f1_all_vs_k1": _mean_metric_delta(records, best_k, 1, "f1_all"),
                "best_k_mean_known_recall": sum(float(r["known_recall"]) for r in best_records) / len(best_records),
                "delta_known_recall_vs_k1": _mean_metric_delta(records, best_k, 1, "known_recall"),
                "cross_seed_best_k_consistency": _best_k_consistency(records),
                "number_of_intents_benefiting_from_k_gt_1": _intent_benefit_count(intent_records, dataset, kir, distance),
                "number_of_intents_with_k_gt_1_in_at_least_3_seeds": _intent_stable_benefit_count(
                    intent_records, dataset, kir, distance
                ),
                "selection_note": "oracle test sensitivity only; not a validation-selected K",
            }
        )
        for seed in sorted({int(r["seed"]) for r in records}):
            seed_records = [r for r in records if int(r["seed"]) == seed]
            seed_means = {int(r["k"]): float(r["oos_f1"]) for r in seed_records}
            seed_best = max(seed_means, key=lambda k: (seed_means[k], -k))
            seed_rows.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "distance": distance,
                    "seed": seed,
                    "best_k": seed_best,
                    "best_k_oos_f1": seed_means[seed_best],
                    "k1_oos_f1": seed_means.get(1, ""),
                    "delta_oos_f1_vs_k1": seed_means[seed_best] - seed_means[1] if 1 in seed_means else "",
                    "selection_note": "test-set oracle for sensitivity analysis only",
                }
            )

    intent_groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in intent_records:
        intent_groups[(record["dataset"], record["kir"], record["distance"], record["intent"], record["seed"])].append(record)
    intent_rows: list[dict[str, Any]] = []
    for key, records in sorted(intent_groups.items()):
        dataset, kir, distance, intent, seed = key
        best = max(records, key=lambda r: (float(r["intent_oos_f1"]), -int(r["k"])))
        k1 = next((r for r in records if int(r["k"]) == 1), None)
        intent_rows.append(
            {
                "dataset": dataset,
                "kir": kir,
                "distance": distance,
                "intent": intent,
                "seed": seed,
                "best_k": int(best["k"]),
                "best_k_intent_oos_f1": float(best["intent_oos_f1"]),
                "delta_oos_f1_vs_k1": float(best["intent_oos_f1"]) - float(k1["intent_oos_f1"]) if k1 else "",
                "delta_intent_class_f1_vs_k1": _intent_metric_delta(records, best["k"], 1, "intent_oos_f1"),
                "delta_known_recall_vs_k1": float(best["intent_known_recall"]) - float(k1["intent_known_recall"]) if k1 else "",
                "best_k_intent_known_recall": float(best["intent_known_recall"]),
                "best_k_false_reject_rate": float(best["intent_false_reject_rate"]),
                "best_k_false_accept_count": int(best["intent_false_accept_count"]),
                "best_k_false_accept_rate": float(best["intent_false_accept_rate"]),
                "selection_metric_definition": "class F1 for one Known intent versus all remaining/OOS test samples; global F1-All is reported only at dataset level",
            }
        )

    _write_csv(output_root / "dataset_level.csv", dataset_rows, list(dataset_rows[0]) if dataset_rows else [])
    _write_csv(output_root / "intent_level.csv", intent_rows, list(intent_rows[0]) if intent_rows else [])
    _write_csv(output_root / "seed_stability.csv", seed_rows, list(seed_rows[0]) if seed_rows else [])
    summary = {
        "run_root": str(run_root),
        "run_count": len(run_records),
        "dataset_level_rows": len(dataset_rows),
        "intent_level_rows": len(intent_rows),
        "seed_stability_rows": len(seed_rows),
        "note": "All best-K fields are test-set oracle sensitivity summaries and must not select a formal method.",
    }
    (output_root / "audit_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def manifest_run_id(run_dir: Path) -> str:
    return run_dir.name


def _mean_metric_delta(records: list[dict[str, Any]], candidate: int, reference: int, metric: str) -> float | str:
    left = [float(r[metric]) for r in records if int(r["k"]) == candidate]
    right = [float(r[metric]) for r in records if int(r["k"]) == reference]
    return sum(left) / len(left) - sum(right) / len(right) if left and right else ""


def _intent_metric_delta(records: list[dict[str, Any]], candidate: Any, reference: int, metric: str) -> float | str:
    left = next((float(r[metric]) for r in records if int(r["k"]) == int(candidate)), None)
    right = next((float(r[metric]) for r in records if int(r["k"]) == reference), None)
    return left - right if left is not None and right is not None else ""


def _best_k_consistency(records: list[dict[str, Any]]) -> float:
    seeds = sorted({int(r["seed"]) for r in records})
    if not seeds:
        return float("nan")
    best = []
    for seed in seeds:
        values = {int(r["k"]): float(r["oos_f1"]) for r in records if int(r["seed"]) == seed}
        if values:
            best.append(max(values, key=lambda k: (values[k], -k)))
    if not best:
        return float("nan")
    return max(best.count(k) for k in set(best)) / len(best)


def _intent_benefit_count(records: list[dict[str, Any]], dataset: str, kir: str, distance: str) -> int:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["dataset"] == dataset and row["kir"] == kir and row["distance"] == distance:
            grouped[row["intent"]].append(row)
    count = 0
    for values in grouped.values():
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            by_seed[int(row["seed"])].append(row)
        benefits = []
        for seed_rows in by_seed.values():
            k1 = next((float(r["intent_oos_f1"]) for r in seed_rows if int(r["k"]) == 1), None)
            benefits.append(k1 is not None and any(int(r["k"]) > 1 and float(r["intent_oos_f1"]) > k1 for r in seed_rows))
        if any(benefits):
            count += 1
    return count


def _intent_stable_benefit_count(records: list[dict[str, Any]], dataset: str, kir: str, distance: str) -> int:
    """Count intents whose test-oracle best K is >1 on at least 3/5 seeds."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["dataset"] == dataset and row["kir"] == kir and row["distance"] == distance:
            grouped[row["intent"]].append(row)
    count = 0
    for values in grouped.values():
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            by_seed[int(row["seed"])].append(row)
        best_gt_one = 0
        for seed_rows in by_seed.values():
            best = max(seed_rows, key=lambda row: (float(row["intent_oos_f1"]), -int(row["k"])))
            best_gt_one += int(best["k"]) > 1
        if best_gt_one >= 3:
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = audit(args.run_root, args.output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
