#!/usr/bin/env python3
"""Verify CCSG pilot coverage, provenance and Known-only calibration."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protocol_v2.data.hashing import atomic_write_json  # noqa: E402


EXPERIMENT_ID = "ccsg_pilot_v1"
EXPECTED = {(dataset, seed) for dataset in ("clinc150", "banking77", "stackoverflow") for seed in (13, 42, 87)}
EXPECTED_METHODS = {
    "current_k1",
    "current_k2_union",
    "mixture_support_k1",
    "mixture_support_k2",
    "margin_only_k1",
    "ccsg_k1",
    "ccsg_k2",
    "ccsg_independent_k2",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    for name in ("CCSG_PROVENANCE.json", "CCSG_INTEGRITY.json"):
        if not (root / name).is_file():
            errors.append(f"missing {name}")
    seen: set[tuple[str, int]] = set()
    metric_rows = 0
    for metrics_path in sorted((root / "runs").glob("*/seed_*/metrics.csv")):
        dataset = metrics_path.parent.parent.name
        seed = int(metrics_path.parent.name.split("_", 1)[1])
        seen.add((dataset, seed))
        rows = _read(metrics_path)
        metric_rows += len(rows)
        methods = {row.get("method") for row in rows}
        if methods != EXPECTED_METHODS or len(rows) != len(EXPECTED_METHODS):
            errors.append(f"{dataset}/seed_{seed}: method/row mismatch")
        required = ("oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate")
        for row in rows:
            if row.get("test_used_for_selection") != "False":
                errors.append(f"{dataset}/seed_{seed}/{row.get('method')}: test selection flag")
            for field in required:
                try:
                    value = float(row[field])
                    if value != value or abs(value) == float("inf"):
                        errors.append(f"{dataset}/seed_{seed}/{row.get('method')}: non-finite {field}")
                except (KeyError, ValueError):
                    errors.append(f"{dataset}/seed_{seed}/{row.get('method')}: missing {field}")
        calibration_path = metrics_path.with_name("calibration.csv")
        manifest_path = metrics_path.with_name("run_manifest.json")
        if not calibration_path.is_file() or not manifest_path.is_file():
            errors.append(f"{dataset}/seed_{seed}: missing calibration/manifest")
        else:
            calibration = _read(calibration_path)
            if len(calibration) != len(EXPECTED_METHODS):
                errors.append(f"{dataset}/seed_{seed}: calibration row mismatch")
            if any(row.get("test_used_for_selection") != "False" for row in calibration):
                errors.append(f"{dataset}/seed_{seed}: calibration used test")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("test_used_for_selection") is not False:
                errors.append(f"{dataset}/seed_{seed}: manifest selection contract")
            if "textoir" in json.dumps(manifest.get("inputs", {})).lower():
                errors.append(f"{dataset}/seed_{seed}: runtime input references textoir")
    missing = sorted(EXPECTED - seen)
    extra = sorted(seen - EXPECTED)
    if missing:
        errors.append(f"missing runs: {missing}")
    if extra:
        errors.append(f"unexpected runs: {extra}")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "expected_runs": len(EXPECTED),
        "completed_runs": len(seen & EXPECTED),
        "metrics_rows": metric_rows,
        "missing_runs": [list(item) for item in missing],
        "unexpected_runs": [list(item) for item in extra],
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }
    atomic_write_json(root / "CCSG_VERIFY.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(args.artifact_dir)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
