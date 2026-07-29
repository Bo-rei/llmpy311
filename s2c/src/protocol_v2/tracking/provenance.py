"""Small provenance helpers shared by protocol_v2 runners."""

from __future__ import annotations

from pathlib import Path

from protocol_v2.data.hashing import sha256_file


def file_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(paths.items())}

