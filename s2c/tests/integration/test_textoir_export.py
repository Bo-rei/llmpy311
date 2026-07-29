from pathlib import Path

from protocol_v2.data import source_import
from protocol_v2.data.canonicalize import build_canonical_dataset
from protocol_v2.data.exporters import export_textoir
from protocol_v2.data.registry import build_registry
from protocol_v2.data.views import build_views

from tests.fixtures.protocol_v2_helpers import make_paths, make_textoir_snapshot


def test_textoir_export_never_requires_textoir_runtime_data(tmp_path: Path, monkeypatch) -> None:
    paths = make_paths(tmp_path)
    snapshot = make_textoir_snapshot(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, snapshot, ["clinc150"])
    build_canonical_dataset(paths, "clinc150")
    build_registry(paths, "clinc150", 0, 0.5)
    build_views(paths, "clinc150", 0, 0.5)
    snapshot.rename(snapshot.with_name("textoir.disabled"))
    manifest = export_textoir(paths, "clinc150", 0, 0.5)
    assert manifest["export_name"] == "textoir"

