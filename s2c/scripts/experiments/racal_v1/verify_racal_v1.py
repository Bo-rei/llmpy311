"""Verify RACAL-v1 stage-1 manifests without running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(root: Path, expected_seeds: tuple[int, ...] = (13, 42, 87)) -> dict[str, object]:
    errors: list[str] = []
    provenance = root / "RACAL_PROVENANCE.json"
    if not provenance.is_file():
        errors.append(f"missing provenance: {provenance}")
    runs: list[dict[str, object]] = []
    for method in ("frozen_k1", "trainable_k1"):
        for seed in expected_seeds:
            run = root / "runs" / method / f"seed_{seed}"
            manifest = run / ("run_manifest.json" if method == "frozen_k1" else "training_manifest.json")
            metrics = run / "metrics.json"
            predictions = run / "predictions.jsonl"
            if not manifest.is_file():
                errors.append(f"missing manifest: {manifest}")
                continue
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                errors.append(f"incomplete manifest: {manifest}")
            if payload.get("test_used_for_selection") is not False:
                errors.append(f"test selection flag is not false: {manifest}")
            if method == "trainable_k1" and not metrics.is_file():
                errors.append(f"missing trainable metrics: {metrics}")
            if not predictions.is_file():
                errors.append(f"missing predictions: {predictions}")
            runs.append({"method": method, "seed": seed, "status": payload.get("status")})
    result = {"status": "pass" if not errors else "fail", "expected_runs": len(expected_seeds) * 2, "runs_seen": len(runs), "errors": errors, "runs": runs}
    (root / "RACAL_VERIFY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
