from __future__ import annotations

from dataclasses import replace

import pytest

from s2c.runtime.paths import ProtocolV2Paths


def test_default_provenance_decision_uses_admitted_official_version() -> None:
    """Bare formal commands must resolve the admitted official version."""
    paths = ProtocolV2Paths.discover()
    assert paths.dataset_version == "protocol_v2_official_v1"
    assert paths.require_experiment_admission("clinc150")["status"] == "partially_admitted"


def test_explicit_candidate_version_blocks_formal_execution() -> None:
    """Candidate TEXTOIR-derived data must not be resumable as a formal study."""
    paths = replace(ProtocolV2Paths.discover(), dataset_version="protocol_v2")
    with pytest.raises(RuntimeError, match="formal experiments are blocked"):
        paths.require_experiment_admission()


def test_partial_admission_is_versioned_and_dataset_scoped() -> None:
    """Official reconstruction must not accidentally admit candidate or blocked datasets."""
    official = ProtocolV2Paths.discover()

    assert official.require_experiment_admission("clinc150")["status"] == "partially_admitted"
    assert official.require_experiment_admission("banking77")["status"] == "partially_admitted"
    with pytest.raises(RuntimeError, match="dataset=stackoverflow"):
        official.require_experiment_admission("stackoverflow")
