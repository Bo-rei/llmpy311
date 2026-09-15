"""Run the protocol-v2 Trainable MiniLM K sweep."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from protocol_v2.experiments.minilm_trainable_k_sweep_v1 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
