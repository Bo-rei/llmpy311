#!/usr/bin/env python3
"""Summarize RACAL-v1 stage-2 K=1/K=2 paired results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from protocol_v2.data.hashing import atomic_write_text
from protocol_v2.experiments.racal_v1.stage2 import SEEDS, stage2_root
from protocol_v2.runtime.paths import ProtocolV2Paths


METRICS = ("oos_f1", "oos_precision", "oos_recall", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos", "fpr95")


def _csv_text(rows: list[dict[str, object]]) -> str:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    from io import StringIO
    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def summarize(root: Path, output_dir: Path) -> dict[str, object]:
    per_seed: list[dict[str, object]] = []
    for seed in SEEDS:
        metrics_path = root / "runs" / f"seed_{seed}" / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row: dict[str, object] = {"dataset": "stackoverflow", "kir": 0.50, "seed": seed}
        for k in ("k1", "k2"):
            for metric in METRICS:
                row[f"{k}_{metric}"] = payload[k][metric]
        for metric in METRICS:
            row[f"k2_minus_k1_{metric}"] = payload["k2_minus_k1"][metric]
        row["checkpoint_sha256"] = json.loads((root / "runs" / f"seed_{seed}" / "run_manifest.json").read_text(encoding="utf-8"))["checkpoint"]["checkpoint_sha256"]
        per_seed.append(row)
    aggregate: list[dict[str, object]] = []
    for k in ("k1", "k2"):
        row: dict[str, object] = {"dataset": "stackoverflow", "kir": 0.50, "method": f"trainable_fixed_{k}", "n_seeds": len(per_seed)}
        for metric in METRICS:
            values = np.asarray([float(item[f"{k}_{metric}"]) for item in per_seed], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        aggregate.append(row)
    delta_row: dict[str, object] = {"dataset": "stackoverflow", "kir": 0.50, "method": "k2_minus_k1", "n_seeds": len(per_seed)}
    for metric in METRICS:
        values = np.asarray([float(item[f"k2_minus_k1_{metric}"]) for item in per_seed], dtype=float)
        delta_row[f"{metric}_mean"] = float(values.mean())
        delta_row[f"{metric}_std"] = float(values.std())
    aggregate.append(delta_row)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "RACAL_V1_STAGE2_PER_SEED.csv", _csv_text(per_seed))
    atomic_write_text(output_dir / "RACAL_V1_STAGE2_MEAN_STD.csv", _csv_text(aggregate))
    return {"status": "complete", "per_seed_rows": len(per_seed), "aggregate_rows": len(aggregate), "output_dir": str(output_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = ProtocolV2Paths.discover()
    root = (args.run_root or stage2_root(paths)).resolve()
    output_dir = (args.output_dir or paths.results_root / "diagnostics" / "racal_v1" / "stage2_fixed_k2").resolve()
    print(json.dumps(summarize(root, output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
