from pathlib import Path

from s2c.data import source_import
from s2c.data.canonicalize import build_canonical
from s2c.data.export_protocol import export_protocol
from s2c.data.registry import build_registries
from s2c.data.validation import validate_protocol
from s2c.data.views import build_all_views

from tests.fixtures.protocol_v2_helpers import make_paths, make_textoir_snapshot


def test_protocol_build_from_import_to_export(tmp_path: Path, monkeypatch) -> None:
    paths = make_paths(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, make_textoir_snapshot(tmp_path), ["clinc150"])
    build_canonical(paths, ["clinc150"])
    build_registries(paths, ["clinc150"], [0.5], [0])
    build_all_views(paths, ["clinc150"], [0.5], [0])
    export_protocol(paths, ["clinc150"], [0.5], [0])

    result = validate_protocol(paths, ["clinc150"], [0.5], [0], require_views=True, require_exports=True)
    assert result == {"datasets": 1, "registries": 1, "views": 1, "exports": 6}
