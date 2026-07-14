from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import cli
from src.runtime import RunnableArtifactUnavailable


@pytest.mark.parametrize("action", ["prepare", "train", "evaluate"])
@pytest.mark.parametrize("dataset", ["clinc150", "banking77_oos", "snips", "stackoverflow"])
def test_cli_dry_run_is_output_free_and_uses_module_execution(action: str, dataset: str, capsys: pytest.CaptureFixture[str]):
    before = sorted(PROJECT_ROOT.rglob("*.json"))

    assert cli.main([action, "--dataset", dataset, "--kir", "50", "--dry-run"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"][1] == "-m"
    assert payload["cwd"] == str(PROJECT_ROOT)
    assert sorted(PROJECT_ROOT.rglob("*.json")) == before


def test_cli_refuses_to_run_an_ablation_without_a_verified_component_bundle():
    with pytest.raises(RunnableArtifactUnavailable, match="results-only evidence"):
        cli.main(["ablate", "--dataset", "clinc150", "--kir", "50", "--dry-run"])
