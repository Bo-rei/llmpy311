"""Aggregate RACAL-v1 stage-1 metrics into a light-weight CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("oos_f1", "oos_precision", "oos_recall", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos", "fpr95")


def summarize(root: Path, output_dir: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for method in ("frozen_k1", "trainable_k1"):
        for metrics_path in sorted((root / "runs" / method).glob("seed_*/metrics.json")):
            seed = int(metrics_path.parent.name.split("_", 1)[1])
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({"method": method, "seed": seed, **{key: metrics.get(key) for key in METRICS}, "checkpoint_sha256": metrics.get("checkpoint_sha256", ""), "test_used_for_selection": metrics.get("test_used_for_selection", False)})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "RACAL_V1_STAGE1.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["method", "seed", *METRICS, "checkpoint_sha256", "test_used_for_selection"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate: list[dict[str, object]] = []
    for method in ("frozen_k1", "trainable_k1"):
        values = [row for row in rows if row["method"] == method]
        if not values:
            continue
        record: dict[str, object] = {"method": method, "n_seeds": len(values)}
        for metric in METRICS:
            numbers = [float(row[metric]) for row in values if row[metric] is not None]
            record[f"{metric}_mean"] = sum(numbers) / len(numbers) if numbers else None
            record[f"{metric}_std"] = (sum((value - float(record[f"{metric}_mean"])) ** 2 for value in numbers) / len(numbers)) ** 0.5 if numbers else None
        aggregate.append(record)
    with (output_dir / "RACAL_V1_STAGE1_MEAN_STD.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in aggregate for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)
    return {"status": "complete", "rows": len(rows), "aggregate_rows": len(aggregate), "output_dir": str(output_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_root.resolve(), args.output_dir.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
