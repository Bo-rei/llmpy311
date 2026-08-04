from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


def test_urcsg_pilot_dry_run_declares_exact_six_cells():
    result = subprocess.run(
        [sys.executable, "scripts/experiments/run_urcsg_pilot.py", "--dry-run"],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["experiment_id"] == "urcsg_pilot_v1"
    assert payload["planned_cells"] == 6
    assert payload["candidate_k"] == [1, 2, 3, 4, 5]
    assert payload["distance"] == "mahalanobis_diag"
