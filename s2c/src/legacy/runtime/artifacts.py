"""Verified result evidence and runnable-artifact availability."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class RunnableArtifactUnavailable(RuntimeError):
    """Raised when an execution requests evidence that is not runnable."""


@dataclass(frozen=True)
class EvidenceAnchor:
    identifier: str
    dataset: str
    kir: int
    path: Path
    sha256: str
    status: str


class ArtifactRegistry:
    """Loads verified evaluation evidence from the canonical config file."""

    def __init__(self, root: Path, payload: dict[str, Any]) -> None:
        self._root = root
        self._payload = payload

    @classmethod
    def load(cls, path: Path) -> "ArtifactRegistry":
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(path.parent.parent, payload)

    def evidence_anchor(self, identifier: str) -> EvidenceAnchor:
        for raw in self._payload.get("evidence_anchors", []):
            if raw["id"] == identifier:
                return EvidenceAnchor(
                    identifier=identifier,
                    dataset=raw["dataset"],
                    kir=raw["kir"],
                    path=(self._root / raw["path"]).resolve(),
                    sha256=raw["sha256"],
                    status=raw["reproducibility_status"],
                )
        raise KeyError(f"Unknown evidence anchor: {identifier}")

    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(raw["id"] for raw in self._payload.get("evidence_anchors", []))

    def verify_evidence(self, identifier: str) -> EvidenceAnchor:
        anchor = self.evidence_anchor(identifier)
        if not anchor.path.is_file():
            raise FileNotFoundError(f"Evidence anchor is missing: {anchor.path}")
        digest = hashlib.sha256(anchor.path.read_bytes()).hexdigest()
        if digest != anchor.sha256:
            raise RuntimeError(f"Evidence checksum mismatch for {identifier}")
        return anchor

    def require_runnable(self, identifier: str) -> Path:
        runnable = self._payload.get("runnable_components", {})
        path = runnable.get(identifier)
        if path is None:
            raise RunnableArtifactUnavailable(
                f"{identifier} is results-only evidence; no verified runnable component bundle is available"
            )
        resolved = (self._root / path).resolve()
        if not resolved.exists():
            raise RunnableArtifactUnavailable(f"Runnable component bundle is missing: {resolved}")
        return resolved
