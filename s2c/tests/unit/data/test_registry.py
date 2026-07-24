from pathlib import Path

from s2c.data import source_import
from s2c.data.canonicalize import build_canonical_dataset
from s2c.data.registry import build_registry, validate_registry

from tests.fixtures.protocol_v2_helpers import make_paths, make_textoir_snapshot


def test_registry_is_deterministic_and_partitions_intents(tmp_path: Path, monkeypatch) -> None:
    paths = make_paths(tmp_path)
    monkeypatch.setattr(source_import, "_git_commit", lambda _: "snapshot")
    source_import.import_textoir_snapshot(paths, make_textoir_snapshot(tmp_path), ["clinc150"])
    build_canonical_dataset(paths, "clinc150")
    first = build_registry(paths, "clinc150", seed=0, kir=0.5)
    second = build_registry(paths, "clinc150", seed=0, kir=0.5)

    validate_registry(first)
    assert first == second
    assert first["known_count"] == 1
    assert set(first["known_intents"]).isdisjoint(first["heldout_intents"])

