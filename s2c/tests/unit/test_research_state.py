from __future__ import annotations

import json
from pathlib import Path

from tools.maintenance.check_research_state import duplicate_plan_rows


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
