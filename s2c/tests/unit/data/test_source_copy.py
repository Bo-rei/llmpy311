from pathlib import Path

from protocol_v2.data import source_import
from protocol_v2.data.hashing import sha256_file
from protocol_v2.data.manifests import read_json, source_manifest_path

from tests.fixtures.protocol_v2_helpers import make_paths, make_textoir_snapshot


def test_source_import_copies_bytes_without_links(tmp_path: Path, monkeypatch) -> None:
    paths = make_paths(tmp_path)
    source = make_textoir_snapshot(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")

    source_import.import_textoir_snapshot(paths, source, ["clinc150"])

    copied = paths.data_root / "sources" / "textoir" / "snapshot" / "clinc150" / "train.tsv"
    assert copied.is_file() and not copied.is_symlink()
    assert sha256_file(copied) == sha256_file(source / "data" / "oos" / "train.tsv")
    manifest = read_json(source_manifest_path(paths.manifest_root, "clinc150"))
    assert all(item["byte_identical"] for item in manifest["files"])

