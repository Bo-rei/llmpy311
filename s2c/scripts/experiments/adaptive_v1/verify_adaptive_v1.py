#!/usr/bin/env python3
"""Verify RC-AMBL pilot coverage and selection contract."""

import argparse
import json
from pathlib import Path

from protocol_v2.experiments.adaptive_v1.summary import verify
from protocol_v2.runtime.paths import ProtocolV2Paths

parser = argparse.ArgumentParser()
parser.add_argument("--artifact-root", type=Path, default=None)
args = parser.parse_args()
root = args.artifact_root or (ProtocolV2Paths.discover().run_root / "adaptive_v1" / "contract_repair5")
result = verify(root)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if result["status"] == "pass" else 2)
