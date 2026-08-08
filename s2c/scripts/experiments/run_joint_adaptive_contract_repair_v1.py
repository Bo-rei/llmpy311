#!/usr/bin/env python3
"""CLI for the isolated contract-repair adaptive multicenter pilot."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from protocol_v2.experiments.joint_adaptive_contract_repair_v1 import main


if __name__ == "__main__":
    raise SystemExit(main())
