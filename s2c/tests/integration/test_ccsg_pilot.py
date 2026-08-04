from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


def test_ccsg_pilot_dry_run_declares_registered_matrix():
    result = subprocess.run(
        [sys.executable, "scripts/experiments/run_ccsg_pilot.py", "--dry-run"],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["experiment_id"] == "ccsg_pilot_v1"
    assert payload["planned_cells"] == 9
    assert payload["planned_metric_rows"] == 72
    assert "ccsg_k2" in payload["methods"]
