"""Policy regression tests for the local-only StackOverflow benchmark snapshot."""

from __future__ import annotations

from pathlib import Path

import pytest

from protocol_v2.data import source_import
from protocol_v2.data.canonicalize import build_canonical_dataset
from protocol_v2.data.manifests import read_json, source_manifest_path
from protocol_v2.data.validation import validate_canonical_dataset, validate_source_snapshot

from tests.fixtures.protocol_v2_helpers import make_paths, make_textoir_snapshot


def test_stackoverflow_snapshot_records_local_only_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local benchmark marker permits validation but does not permit release."""
    paths = make_paths(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, make_textoir_snapshot(tmp_path), ["stackoverflow"])
    manifest = read_json(source_manifest_path(paths.manifest_root, "stackoverflow"))

    assert manifest["local_research_only"] is True
    assert manifest["redistribution_by_s2c"] is False
    assert manifest["per_row_attribution_complete"] is False


def test_stackoverflow_validation_rejects_wrong_snapshot_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real protocol validates all 20,000 rows; a small fixture cannot claim it."""
    paths = make_paths(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, make_textoir_snapshot(tmp_path), ["stackoverflow"])
    build_canonical_dataset(paths, "stackoverflow")
    validate_source_snapshot(paths, "stackoverflow")
    with pytest.raises(ValueError, match="20,000"):
        validate_canonical_dataset(paths, "stackoverflow")
