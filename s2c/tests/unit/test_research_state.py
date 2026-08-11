from __future__ import annotations

import json
from pathlib import Path

from tools.maintenance.check_research_state import duplicate_plan_rows, validate_ledger_schema


def _ledger_row(**overrides: str) -> dict[str, str]:
    row = {
        "experiment_id": "e2",
        "status": "complete",
        "repeat_policy": "do_not_repeat",
        "protocol_version": "protocol_v2_textoir_v1",
        "datasets": "clinc150|banking77|stackoverflow",
        "kirs": "25|50|75",
        "seeds": "13|42|87",
        "representations": "frozen",
        "k_values": "1|2|3|4|5",
        "distances": "euclidean|mahalanobis_diag",
        "partition": "",
        "boundary": "mean_std",
    }
    row.update(overrides)
    return row


def test_duplicate_plan_is_rejected_by_dimensions(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"units": [_ledger_row()]}), encoding="utf-8")
    assert len(duplicate_plan_rows(plan, [_ledger_row()])) == 1


def test_changed_method_is_not_duplicate(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    candidate = _ledger_row(representations="ce_recon_geometry")
    plan.write_text(json.dumps({"units": [candidate]}), encoding="utf-8")
    assert duplicate_plan_rows(plan, [_ledger_row()]) == []


def test_active_stage_can_be_named_without_r1_prefix(tmp_path: Path, monkeypatch) -> None:
    """The state checker must follow the ledger, not a historical R1 literal."""

    from tools.maintenance import check_research_state

    status = tmp_path / "RESEARCH_STATUS.md"
    status.write_text(
        "protocol_v2_textoir_v1\n## 当前唯一下一步\nmulticenter_boundary_attribution\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_research_state, "STATUS_PATH", status)
    monkeypatch.setattr(
        check_research_state,
        "read_ledger",
        lambda path=check_research_state.LEDGER_PATH: [
            {
                "experiment_id": "multicenter_boundary_attribution",
                "status": "planned",
                "repeat_policy": "active",
            }
        ],
    )
    monkeypatch.setattr(check_research_state, "FROZEN_REQUIRED", {})
    monkeypatch.setattr(check_research_state, "validate_ledger_schema", lambda: [])
    monkeypatch.setattr(check_research_state, "R1_CLOSEOUT", tmp_path / "unused")
    monkeypatch.setattr(check_research_state, "R1_FULL_CLOSEOUT", tmp_path / "unused")
    monkeypatch.setattr(check_research_state, "R1_CONTRACT_REPAIR_CLOSEOUT", tmp_path / "unused")
    assert check_research_state.check_state() == []


def test_ledger_schema_catches_missing_distances_column(tmp_path: Path) -> None:
    header = (
        "experiment_id,stage,protocol_version,experiment_family,research_question,hypothesis,datasets,kirs,seeds,"
        "representations,k_values,distances,status,planned_units,completed_units,failed_units,start_time,end_time,"
        "provenance_sha256,config_sha256,summary_path,main_result,decision,repeat_policy,supersedes,notes\n"
    )
    path = tmp_path / "ledger.csv"
    # The row has one field missing between k_values and status, which is the
    # exact failure mode that previously shifted all following columns.
    path.write_text(header + "e,stage,p,f,q,h,d,k,s,r,1|2,complete,2,2,0,2026,2026,sha,sha,summary,result,decision,do_not_repeat,,notes\n", encoding="utf-8")
    errors = validate_ledger_schema(path)
    assert "ledger_column_count:2:25" in errors
