from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import ArtifactRegistry, RunnableArtifactUnavailable, WorkspacePaths


def test_workspace_paths_are_independent_of_the_legacy_root_alias():
    paths = WorkspacePaths.discover(PROJECT_ROOT)

    assert paths.project_root == PROJECT_ROOT
    assert paths.workspace_root / "s2c" == PROJECT_ROOT
    assert paths.smollm135m.name == "smollm135m"


def test_registry_verifies_the_canonical_clinc_evaluation_anchor():
    registry = ArtifactRegistry.load(PROJECT_ROOT / "configs" / "artifacts.yaml")

    anchor = registry.verify_evidence("clinc150-kir50-frozen")

    assert anchor.status == "canonical_historical_evaluation_anchor"
    assert anchor.path.name == "eval_results.json"


def test_registry_verifies_every_registered_evidence_anchor():
    registry = ArtifactRegistry.load(PROJECT_ROOT / "configs" / "artifacts.yaml")

    assert len(registry.evidence_ids()) == 9
    for identifier in registry.evidence_ids():
        assert registry.verify_evidence(identifier).status in {
            "results_only",
            "canonical_historical_evaluation_anchor",
        }


def test_registry_refuses_to_treat_results_only_evidence_as_runnable():
    registry = ArtifactRegistry.load(PROJECT_ROOT / "configs" / "artifacts.yaml")

    with pytest.raises(RunnableArtifactUnavailable, match="results-only evidence"):
        registry.require_runnable("clinc150-kir50-frozen")
