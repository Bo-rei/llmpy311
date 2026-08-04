#!/usr/bin/env python3
"""Verify URCSG pilot coverage, provenance and no-test-selection contract."""

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
from protocol_v2.runtime.paths import ProtocolV2Paths  # noqa: E402


EXPERIMENT_ID = "urcsg_pilot_v1"
EXPECTED = {(dataset, seed) for dataset in ("banking77", "stackoverflow") for seed in (13, 42, 87)}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    provenance = root / "URCSG_PROVENANCE.json"
    integrity = root / "URCSG_INTEGRITY.json"
    if not provenance.is_file():
        errors.append("missing URCSG_PROVENANCE.json")
    if not integrity.is_file():
        errors.append("missing URCSG_INTEGRITY.json")
    seen: set[tuple[str, int]] = set()
    run_rows = 0
    for metrics_path in sorted((root / "runs").glob("*/seed_*/metrics.csv")):
        dataset = metrics_path.parent.parent.name
        seed = int(metrics_path.parent.name.split("_", 1)[1])
        seen.add((dataset, seed))
        rows = _read_csv(metrics_path)
        run_rows += len(rows)
        if len(rows) != 9:
            errors.append(f"{dataset}/seed_{seed}: expected 9 metric rows, got {len(rows)}")
        if any(row.get("test_used_for_selection") == "True" and row.get("method") != "oracle_test_k" for row in rows):
            errors.append(f"{dataset}/seed_{seed}: test selection flag on non-oracle method")
        for required in ("oos_f1", "f1_all", "known_recall", "false_accept_rate"):
            for row in rows:
                try:
                    value = float(row[required])
                    if not (value == value and abs(value) != float("inf")):
                        errors.append(f"{dataset}/seed_{seed}: non-finite {required}")
                except (KeyError, ValueError):
                    errors.append(f"{dataset}/seed_{seed}: missing {required}")
        selection_path = metrics_path.with_name("intent_selection.csv")
        mechanism_path = metrics_path.with_name("mechanism.json")
        manifest_path = metrics_path.with_name("run_manifest.json")
        if not selection_path.is_file() or not mechanism_path.is_file() or not manifest_path.is_file():
            errors.append(f"{dataset}/seed_{seed}: missing run manifest/selection/mechanism")
        else:
            selection_rows = _read_csv(selection_path)
            if len(selection_rows) == 0:
                errors.append(f"{dataset}/seed_{seed}: empty intent selection")
            if any(row.get("test_used_for_selection") == "True" for row in selection_rows):
                errors.append(f"{dataset}/seed_{seed}: test selection in intent table")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("test_used_for_selection") is not False:
                errors.append(f"{dataset}/seed_{seed}: manifest test_used_for_selection is not false")
            inputs = manifest.get("inputs", {})
            if "textoir" in json.dumps(inputs).lower():
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
        "metrics_rows": run_rows,
        "missing_runs": [list(item) for item in missing],
        "unexpected_runs": [list(item) for item in extra],
        "errors": errors,
        "status": "pass" if not errors else "fail",
    }
    atomic_write_json(root / "URCSG_VERIFY.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    result = verify(args.artifact_dir or paths.run_root / EXPERIMENT_ID)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
