#!/usr/bin/env python3
"""Aggregate completed RC-AMBL pilot metrics."""

import argparse
import json
from pathlib import Path

from protocol_v2.experiments.adaptive_v1.summary import build_main_results, summarize
from protocol_v2.runtime.paths import ProtocolV2Paths

parser = argparse.ArgumentParser()
parser.add_argument("--artifact-root", type=Path, default=None)
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()
paths = ProtocolV2Paths.discover()
root = args.artifact_root or (paths.run_root / "adaptive_v1" / "contract_repair5")
output = args.output or (paths.results_root / "diagnostics" / "adaptive_v1")
rows = summarize(root, output)
main_rows = build_main_results(paths, root, output)
print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
print(json.dumps({"main_results_rows": len(main_rows), "main_results": str(output / "main_results.csv")}, ensure_ascii=False))
