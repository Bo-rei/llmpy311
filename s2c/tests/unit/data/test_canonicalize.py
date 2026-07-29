from pathlib import Path

from protocol_v2.data import source_import
from protocol_v2.data.canonicalize import build_canonical_dataset
from protocol_v2.data.hashing import sha256_file

from tests.fixtures.protocol_v2_helpers import make_paths, make_textoir_snapshot


def test_canonical_build_is_idempotent_and_retains_native_oos(tmp_path: Path, monkeypatch) -> None:
    paths = make_paths(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, make_textoir_snapshot(tmp_path), ["clinc150"])

    first = build_canonical_dataset(paths, "clinc150")
    canonical = paths.protocol_root / "clinc150" / "records.jsonl"
    first_sha = sha256_file(canonical)
    second = build_canonical_dataset(paths, "clinc150")

    assert first_sha == sha256_file(canonical) == first["canonical_file_sha256"] == second["canonical_file_sha256"]
    assert first["sample_count"] == 7
    assert first["native_oos_count"] == 1

