from __future__ import annotations

from dataclasses import replace

import pytest

from s2c.runtime.paths import ProtocolV2Paths


def test_default_provenance_decision_uses_active_textoir_version() -> None:
    """Bare formal commands must resolve the sole active local benchmark version."""
    paths = ProtocolV2Paths.discover()
    assert paths.dataset_version == "protocol_v2_textoir_v1"
    assert paths.require_experiment_admission("clinc150")["status"] == "admitted"


def test_explicit_candidate_version_blocks_formal_execution() -> None:
    """Candidate TEXTOIR-derived data must not be resumable as a formal study."""
    paths = replace(ProtocolV2Paths.discover(), dataset_version="protocol_v2")
    with pytest.raises(RuntimeError, match="formal experiments are blocked"):
        paths.require_experiment_admission()


def test_active_admission_is_versioned_and_stackoverflow_is_local_only() -> None:
    """Local benchmark admission must not be mistaken for corpus redistribution."""
    active = ProtocolV2Paths.discover()

    assert active.require_experiment_admission("clinc150")["status"] == "admitted"
    assert active.require_experiment_admission("banking77")["status"] == "admitted"
    payload = active.require_experiment_admission("stackoverflow")
    assert payload["dataset_admission"]["stackoverflow"] == "admitted_benchmark_local_only"
    assert payload["dataset_policies"]["stackoverflow"]["redistribution_by_s2c"] is False
