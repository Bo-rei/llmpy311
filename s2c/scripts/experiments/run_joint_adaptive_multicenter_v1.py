#!/usr/bin/env python3
"""Run the bounded, training-participating adaptive multicenter pilot."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from protocol_v2.experiments.joint_adaptive_v1.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
