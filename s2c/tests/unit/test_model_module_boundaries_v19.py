"""Regression tests for the public model-module import boundaries."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_router_package_exports_the_canonical_router():
    from legacy.router import QwenRouter
    from legacy.router import router_model

    assert QwenRouter is router_model.QwenRouter


def test_gate_default_backbone_uses_the_canonical_workspace_assets_path():
    from legacy.models.gate_svdd import DEFAULT_BACKBONE_PATH, SVDDGate

    assert DEFAULT_BACKBONE_PATH == str(
        PROJECT_ROOT.parent / "assets/models/smollm17b"
    )
    assert SVDDGate.__init__.__defaults__[0] == DEFAULT_BACKBONE_PATH
