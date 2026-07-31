"""Aggregate completed MOGB baseline run manifests into a lightweight CSV."""

from __future__ import annotations

import argparse
import math
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


FIELDS = ["dataset", "kir", "seed", "method", "oos_f1", "f1_all", "f1_u", "f1_k", "accuracy", "id_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos", "effective_cluster_count", "minimum_cluster_size", "run_dir"]


def _write_mean_std(rows: list[dict[str, object]], path: Path) -> None:
    metric_fields = [field for field in FIELDS if field not in {"dataset", "kir", "seed", "method", "run_dir"}]
    output_fields = ["dataset", "kir", "method", "n_seeds"]
    for field in metric_fields:
        output_fields.extend([f"{field}_mean", f"{field}_std"])
    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), float(row["kir"]), str(row["method"])), []).append(row)
    output: list[dict[str, object]] = []
    for (dataset, kir, method), group in sorted(grouped.items()):
        result: dict[str, object] = {"dataset": dataset, "kir": kir, "method": method, "n_seeds": len(group)}
        for field in metric_fields:
            values = np.asarray([float(item[field]) for item in group if item[field] is not None], dtype=np.float64)
            result[f"{field}_mean"] = float(values.mean()) if values.size else math.nan
            result[f"{field}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        output.append(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output)


def _write_significance(rows: list[dict[str, object]], path: Path) -> None:
    metrics = ["oos_f1", "f1_all", "id_recall", "false_accept_rate"]
    methods = sorted({str(row["method"]) for row in rows if str(row["method"]) != "fixed_k2"})
    index = {(str(row["dataset"]), float(row["kir"]), int(row["seed"]), str(row["method"])): row for row in rows}
    output = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for kir in sorted({float(row["kir"]) for row in rows}):
            for method in methods:
                paired = [
                    (index[(dataset, kir, seed, method)], index[(dataset, kir, seed, "fixed_k2")])
                    for seed in sorted({int(row["seed"]) for row in rows})
                    if (dataset, kir, seed, method) in index and (dataset, kir, seed, "fixed_k2") in index
                ]
                if not paired:
                    continue
                for metric in metrics:
                    candidate = np.asarray([float(left[metric]) for left, _ in paired], dtype=np.float64)
                    reference = np.asarray([float(right[metric]) for _, right in paired], dtype=np.float64)
                    delta = candidate - reference
                    ttest_p = float(ttest_rel(candidate, reference).pvalue) if len(delta) > 1 else math.nan
                    try:
                        wilcoxon_p = float(wilcoxon(delta).pvalue) if len(delta) > 1 and np.any(delta) else math.nan
                    except ValueError:
                        wilcoxon_p = math.nan
                    output.append({
                        "dataset": dataset,
                        "kir": kir,
                        "method": method,
                        "reference": "fixed_k2",
                        "metric": metric,
                        "n_pairs": len(delta),
                        "mean_delta": float(delta.mean()),
                        "std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                        "wins": int(np.sum(delta > 0)),
                        "ties": int(np.sum(delta == 0)),
                        "losses": int(np.sum(delta < 0)),
                        "paired_t_pvalue": ttest_p,
                        "wilcoxon_pvalue": wilcoxon_p,
                    })
    fields = ["dataset", "kir", "method", "reference", "metric", "n_pairs", "mean_delta", "std_delta", "wins", "ties", "losses", "paired_t_pvalue", "wilcoxon_pvalue"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def _write_per_intent(input_dir: Path, path: Path) -> None:
    fields = ["dataset", "kir", "seed", "method", "intent", "n_samples", "gold_oos", "oos_false_accept", "known_false_reject", "known_correct"]
    output: list[dict[str, object]] = []
    for manifest in sorted(input_dir.glob("**/manifest.json")):
        if manifest.name != "manifest.json":
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        config_path = manifest.parent / "config.json"
        prediction_path = manifest.parent / "predictions.tsv"
        if not config_path.is_file() or not prediction_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        groups: dict[str, dict[str, int]] = {}
        with prediction_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                intent = str(row["gold_intent"])
                item = groups.setdefault(intent, {"n_samples": 0, "gold_oos": 0, "oos_false_accept": 0, "known_false_reject": 0, "known_correct": 0})
                gold_oos = int(row["gold_is_oos"])
                predicted_oos = int(row["predicted_is_oos"])
                item["n_samples"] += 1
                item["gold_oos"] += gold_oos
                item["oos_false_accept"] += int(gold_oos and not predicted_oos)
                item["known_false_reject"] += int(not gold_oos and predicted_oos)
                item["known_correct"] += int(not gold_oos and row["predicted_label"] == intent)
        for intent, item in sorted(groups.items()):
            output.append({"dataset": config["dataset"], "kir": config["kir"], "seed": config["seed"], "method": config["method"], "intent": intent, **item})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_baseline_v1"))
    parser.add_argument("--output", type=Path, default=Path("../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_baseline_v1/summary/all_runs.csv"))
    args = parser.parse_args(argv)
    rows = []
    for manifest in sorted(args.input_dir.glob("**/manifest.json")):
        if manifest.name != "manifest.json":
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        metrics_path = manifest.parent / "metrics.json"
        config_path = manifest.parent / "config.json"
        if not metrics_path.is_file() or not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"dataset": config["dataset"], "kir": config["kir"], "seed": config["seed"], "method": config["method"], "run_dir": str(manifest.parent)}
        row.update({key: metrics.get(key) for key in FIELDS if key not in row})
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["dataset"], float(row["kir"]), int(row["seed"]), row["method"])))
    summary_dir = args.output.parent
    _write_mean_std(rows, summary_dir / "mean_std.csv")
    _write_significance(rows, summary_dir / "significance_tests.csv")
    _write_per_intent(args.input_dir, summary_dir / "per_intent_analysis.csv")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
