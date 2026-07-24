"""Small self-contained TEXTOIR-shaped snapshots for protocol_v2 tests."""

from __future__ import annotations

from pathlib import Path

from s2c.runtime.paths import ProtocolV2Paths


def make_paths(tmp_path: Path) -> ProtocolV2Paths:
    project = tmp_path / "project" / "s2c"
    return ProtocolV2Paths(
        project_root=project,
        data_root=project / "data",
        artifacts_root=tmp_path / "artifacts" / "s2c",
        results_root=project / "results",
        legacy_root=tmp_path / "assets" / "datasets" / "s2c",
        textoir_import_root=None,
    )


def write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("text\tlabel\n" + "".join(f"{text}\t{label}\n" for text, label in rows), encoding="utf-8")


def make_textoir_snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "textoir"
    rows = {
        "oos": {
            "train": [("clinc train alpha", "alpha"), ("clinc train beta", "beta")],
            "dev": [("clinc dev alpha", "alpha"), ("clinc dev beta", "beta")],
            "test": [("clinc test alpha", "alpha"), ("clinc test beta", "beta"), ("clinc native oos", "oos")],
        },
        "banking": {
            "train": [("bank train alpha", "alpha"), ("bank train beta", "beta")],
            "dev": [("bank dev alpha", "alpha"), ("bank dev beta", "beta")],
            "test": [("bank test alpha", "alpha"), ("bank test beta", "beta")],
        },
        "stackoverflow": {
            "train": [("stack train alpha", "alpha"), ("stack train beta", "beta")],
            "dev": [("stack dev alpha", "alpha"), ("stack dev beta", "beta")],
            "test": [("stack test alpha", "alpha"), ("stack test beta", "beta")],
        },
    }
    for directory, split_rows in rows.items():
        for split, values in split_rows.items():
            write_tsv(root / "data" / directory / f"{split}.tsv", values)
    return root

