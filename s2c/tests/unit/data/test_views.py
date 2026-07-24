from pathlib import Path

from s2c.data import source_import
from s2c.data.canonicalize import build_canonical_dataset
from s2c.data.registry import build_registry
from s2c.data.views import build_views

from tests.fixtures.protocol_v2_helpers import make_paths, make_textoir_snapshot


def test_views_keep_calibration_known_only_and_preserve_native_oos(tmp_path: Path, monkeypatch) -> None:
    paths = make_paths(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, make_textoir_snapshot(tmp_path), ["clinc150"])
    build_canonical_dataset(paths, "clinc150")
    build_registry(paths, "clinc150", seed=0, kir=0.5)
    manifest = build_views(paths, "clinc150", seed=0, kir=0.5)

    counts = {item["name"]: item["count"] for item in manifest["files"]}
    assert counts["train_known"] == counts["calibration_known"] == counts["test_known"] == 1
    assert counts["test_heldout_oos"] == counts["test_native_oos"] == 1
    assert counts["test_combined"] == 3

