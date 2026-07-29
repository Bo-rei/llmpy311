from pathlib import Path

import pytest

from protocol_v2.data import source_import
from protocol_v2.data.canonicalize import build_canonical_dataset
from protocol_v2.data.registry import build_registry
from protocol_v2.data.validation import validate_canonical_dataset, validate_source_snapshot

from tests.fixtures.protocol_v2_helpers import make_paths, make_textoir_snapshot


def test_manifest_validation_detects_source_and_canonical_changes(tmp_path: Path, monkeypatch) -> None:
    paths = make_paths(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, make_textoir_snapshot(tmp_path), ["clinc150"])
    build_canonical_dataset(paths, "clinc150")
    build_registry(paths, "clinc150", seed=0, kir=0.5)
    validate_source_snapshot(paths, "clinc150")
    validate_canonical_dataset(paths, "clinc150")
    (paths.protocol_root / "clinc150" / "records.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Canonical SHA256 mismatch"):
        validate_canonical_dataset(paths, "clinc150")

