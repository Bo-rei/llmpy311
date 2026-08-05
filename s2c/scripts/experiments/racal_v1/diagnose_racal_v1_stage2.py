#!/usr/bin/env python3
"""Aggregate RACAL-v1 stage-2 sample and intent diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from protocol_v2.data.hashing import atomic_write_text
from protocol_v2.experiments.racal_v1.stage2 import SEEDS, stage2_root, _hash_sample_audit
from protocol_v2.runtime.paths import ProtocolV2Paths


def _csv_text(rows: list[dict[str, object]]) -> str:
    from io import StringIO
    fields = list(dict.fromkeys(key for row in rows for key in row))
    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def diagnose(root: Path, output_dir: Path) -> dict[str, object]:
    intent_rows: list[dict[str, object]] = []
    hashed_samples: list[dict[str, object]] = []
    category_counts: Counter[str] = Counter()
    for seed in SEEDS:
        run = root / "runs" / f"seed_{seed}"
        with (run / "intent_diagnostics.csv").open(encoding="utf-8", newline="") as handle:
            intent_rows.extend(dict(row) for row in csv.DictReader(handle))
        samples = [json.loads(line) for line in (run / "sample_audit.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in samples:
            for category in row["categories"]:
                category_counts[f"seed_{seed}:{category}"] += 1
        hashed_samples.extend(_hash_sample_audit(samples))
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "RACAL_V1_STAGE2_INTENT_DIAGNOSTICS.csv", _csv_text(intent_rows))
    atomic_write_text(output_dir / "RACAL_V1_STAGE2_SAMPLE_AUDIT_HASHES.csv", _csv_text(hashed_samples))
    atomic_write_text(output_dir / "RACAL_V1_STAGE2_SAMPLE_CATEGORY_COUNTS.csv", _csv_text([{"category": key, "count": value} for key, value in sorted(category_counts.items())]))
    return {"status": "complete", "intent_rows": len(intent_rows), "sample_rows": len(hashed_samples), "category_rows": len(category_counts), "output_dir": str(output_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = ProtocolV2Paths.discover()
    root = (args.run_root or stage2_root(paths)).resolve()
    output_dir = (args.output_dir or paths.results_root / "diagnostics" / "racal_v1" / "stage2_fixed_k2").resolve()
    print(json.dumps(diagnose(root, output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
