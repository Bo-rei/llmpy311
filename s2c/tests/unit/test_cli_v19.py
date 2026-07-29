from __future__ import annotations

import json
from pathlib import Path

import pytest

from legacy import cli
from legacy.runtime import RunnableArtifactUnavailable

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
