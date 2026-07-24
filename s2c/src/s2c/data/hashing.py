"""Deterministic hashing and atomic writers used by all protocol_v2 builders."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(payload: object) -> str:
    return sha256_bytes(stable_json_bytes(payload))


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Mapping[str, object] | list[object]) -> None:
    _atomic_replace(path, stable_json_bytes(payload) + b"\n")


def atomic_write_text(path: Path, value: str) -> None:
    _atomic_replace(path, value.encode("utf-8"))


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    serialized = b"".join(stable_json_bytes(row) + b"\n" for row in rows)
    _atomic_replace(path, serialized)
