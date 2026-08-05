#!/usr/bin/env python3
"""Verify RACAL-v1 stage-2 manifests and paired K=1/K=2 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol_v2.experiments.racal_v1.stage2 import SEEDS, stage2_root
from protocol_v2.runtime.paths import ProtocolV2Paths


def verify(root: Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    errors: list[str] = []
    provenance = root / "RACAL_STAGE2_PROVENANCE.json"
    if not provenance.is_file():
        errors.append(f"missing provenance: {provenance}")
    runs: list[dict[str, object]] = []
    for seed in seeds:
        run = root / "runs" / f"seed_{seed}"
        manifest_path = run / "run_manifest.json"
        metrics_path = run / "metrics.json"
        audit_path = run / "sample_audit.jsonl"
        intent_path = run / "intent_diagnostics.csv"
        for path in (manifest_path, metrics_path, audit_path, intent_path):
            if not path.is_file():
                errors.append(f"missing output: {path}")
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            errors.append(f"incomplete manifest: {manifest_path}")
        if manifest.get("test_used_for_selection") is not False:
            errors.append(f"test selection flag is not false: {manifest_path}")
        if manifest.get("k_values") != [1, 2]:
            errors.append(f"wrong k values: {manifest_path}")
        runs.append({"seed": seed, "status": manifest.get("status"), "config_hash": manifest.get("config_hash")})
    result = {"status": "pass" if not errors and len(runs) == len(seeds) else "fail", "expected_runs": len(seeds), "runs_seen": len(runs), "errors": errors, "runs": runs}
    (root / "RACAL_STAGE2_VERIFY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    paths = ProtocolV2Paths.discover()
    root = (args.run_root or stage2_root(paths)).resolve()
    result = verify(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
