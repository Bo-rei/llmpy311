from pathlib import Path

from s2c.data import source_import
from s2c.data.canonicalize import build_canonical_dataset
from s2c.data.exporters import export_adb, export_da_adb, export_k_plus_1_way, export_mogb, export_s2c, export_textoir
from s2c.data.registry import build_registry
from s2c.data.views import build_views

from tests.fixtures.protocol_v2_helpers import make_paths, make_textoir_snapshot


def test_all_exporters_share_the_same_registry_and_sample_mapping(tmp_path: Path, monkeypatch) -> None:
    paths = make_paths(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, make_textoir_snapshot(tmp_path), ["clinc150"])
    build_canonical_dataset(paths, "clinc150")
    registry = build_registry(paths, "clinc150", seed=0, kir=0.5)
    build_views(paths, "clinc150", seed=0, kir=0.5)
    manifests = [
        export_s2c(paths, "clinc150", 0, 0.5),
        export_textoir(paths, "clinc150", 0, 0.5),
        export_adb(paths, "clinc150", 0, 0.5),
        export_da_adb(paths, "clinc150", 0, 0.5),
        export_mogb(paths, "clinc150", 0, 0.5),
        export_k_plus_1_way(paths, "clinc150", 0, 0.5),
    ]
    assert {item["registry_sha256"] for item in manifests} == {registry["registry_sha256"]}
    assert len({item["canonical_sample_id_mapping_sha256"] for item in manifests}) == 1
