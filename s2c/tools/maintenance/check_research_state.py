#!/usr/bin/env python3
"""Validate the single research state/experiment ledger contract.

The checker is intentionally small: it prevents silent repetition of frozen
E2/E3 cells and catches a status document that claims completion without a
closeout artifact.  It does not inspect or rewrite experiment results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = ROOT / "docs" / "research"
STATUS_PATH = RESEARCH_ROOT / "RESEARCH_STATUS.md"
LEDGER_PATH = RESEARCH_ROOT / "EXPERIMENT_LEDGER.csv"
DECISION_PATH = RESEARCH_ROOT / "DECISION_LOG.md"
CLAIM_PATH = RESEARCH_ROOT / "PAPER_CLAIM_AUDIT.md"

FROZEN_REQUIRED = {
    "e2_gate_core_dense": (
        1650,
        ROOT.parent / "artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e2_closeout",
    ),
    "e3_mechanisms": (
        900,
        ROOT.parent / "artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/summaries",
    ),
}

R1_CLOSEOUT = ROOT.parent / "artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation/summaries/R1_CLOSEOUT.md"
R1_FULL_CLOSEOUT = ROOT.parent / "artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation_full/summaries/R1_FULL_CLOSEOUT.md"
R1_CONTRACT_REPAIR_CLOSEOUT = ROOT.parent / "artifacts/s2c/runs/protocol_v2_textoir_v1/r1_contract_repair_v1/R1_CONTRACT_REPAIR_CLOSEOUT.md"


def read_ledger(path: Path = LEDGER_PATH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(sorted(_norm(item) for item in value))
    text = str(value).strip().lower()
    parts = text.split("|")
    normalized: list[str] = []
    for part in parts:
        try:
            normalized.append(format(float(part), ".12g"))
        except ValueError:
            normalized.append(part)
    return "|".join(sorted(normalized))


def _plan_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict) and isinstance(payload.get("units"), list):
        return [dict(row) for row in payload["units"]]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported plan JSON shape: {path}")


def duplicate_plan_rows(plan_path: Path, ledger_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Return plan rows identical to a completed do_not_repeat ledger row."""

    dimensions = (
        "protocol_version",
        "datasets",
        "kirs",
        "seeds",
        "representations",
        "k_values",
        "distances",
        "partition",
        "boundary",
    )
    frozen = [row for row in ledger_rows if row.get("status") == "complete" and row.get("repeat_policy") == "do_not_repeat"]
    duplicates: list[dict[str, Any]] = []
    for candidate in _plan_rows(plan_path):
        for row in frozen:
            def value(item: Mapping[str, Any], field: str) -> Any:
                # Older ledger rows predate explicit partition/boundary columns;
                # their frozen E2/E3 protocols used the declared defaults.
                default = "none" if field == "partition" else "mean_std" if field == "boundary" else ""
                return item.get(field, default)

            if all(_norm(value(candidate, field)) == _norm(value(row, field)) for field in dimensions):
                duplicates.append({"plan": candidate, "ledger_experiment_id": row.get("experiment_id")})
                break
    return duplicates


def check_state() -> list[str]:
    errors: list[str] = []
    required = (STATUS_PATH, LEDGER_PATH, DECISION_PATH, CLAIM_PATH)
    errors.extend(f"missing_research_state:{path}" for path in required if not path.is_file())
    if errors:
        return errors
    ledger = read_ledger()
    if not ledger:
        errors.append("empty_experiment_ledger")
        return errors
    if any(not row.get("experiment_id") for row in ledger):
        errors.append("ledger_row_without_experiment_id")
    ids = [row["experiment_id"] for row in ledger]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_experiment_id")
    for experiment_id, (expected_units, summary_root) in FROZEN_REQUIRED.items():
        rows = [row for row in ledger if row.get("experiment_id") == experiment_id]
        if not rows:
            errors.append(f"missing_ledger_experiment:{experiment_id}")
            continue
        row = rows[0]
        if row.get("status") != "complete" or int(row.get("completed_units", "-1")) != expected_units:
            errors.append(f"frozen_experiment_not_complete:{experiment_id}")
        if not summary_root.is_dir():
            errors.append(f"missing_closeout:{experiment_id}:{summary_root}")
    r1_rows = [row for row in ledger if row.get("experiment_id") == "r1_geometry_preserving_representation"]
    if r1_rows and r1_rows[0].get("status") == "complete":
        if int(r1_rows[0].get("completed_units", "-1")) != 108:
            errors.append("r1_completed_units_mismatch")
        if not R1_CLOSEOUT.is_file():
            errors.append(f"missing_closeout:r1_geometry_preserving_representation:{R1_CLOSEOUT}")
    r1_full_rows = [row for row in ledger if row.get("experiment_id") == "r1_geometry_preserving_representation_full"]
    if r1_full_rows and r1_full_rows[0].get("status") == "complete":
        if int(r1_full_rows[0].get("completed_units", "-1")) != 270:
            errors.append("r1_full_completed_units_mismatch")
        if not R1_FULL_CLOSEOUT.is_file():
            errors.append(f"missing_closeout:r1_geometry_preserving_representation_full:{R1_FULL_CLOSEOUT}")
    repair_rows = [row for row in ledger if row.get("experiment_id") == "r1_contract_repair_v1"]
    if repair_rows and repair_rows[0].get("status") == "complete":
        if int(repair_rows[0].get("completed_units", "-1")) != 42:
            errors.append("r1_contract_repair_completed_units_mismatch")
        if not R1_CONTRACT_REPAIR_CLOSEOUT.is_file():
            errors.append(f"missing_closeout:r1_contract_repair_v1:{R1_CONTRACT_REPAIR_CLOSEOUT}")
    status = STATUS_PATH.read_text(encoding="utf-8")
    if status.count("## 当前唯一下一步") != 1:
        errors.append("status_must_have_one_current_next_step")
    if "protocol_v2_textoir_v1" not in status:
        errors.append("status_protocol_mismatch")
    active = [row for row in ledger if row.get("repeat_policy") == "active" and row.get("status") in {"planned", "running"}]
    if len(active) > 1:
        errors.append("multiple_active_experiments")
    if "R1_geometry_preserving_representation" not in status and active:
        errors.append("active_ledger_not_reflected_in_status")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="Plan JSON to check against frozen ledger rows")
    parser.add_argument("--allow-rerun", action="store_true")
    parser.add_argument("--rerun-reason")
    args = parser.parse_args(argv)
    errors = check_state()
    if args.plan:
        duplicates = duplicate_plan_rows(args.plan, read_ledger())
        if duplicates and not args.allow_rerun:
            print(json.dumps({"status": "duplicate_completed_experiment", "duplicates": duplicates}, ensure_ascii=False, indent=2))
            return 2
        if args.allow_rerun and not args.rerun_reason:
            errors.append("allow_rerun_requires_rerun_reason")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", "ledger_rows": len(read_ledger()), "plan_checked": bool(args.plan)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
