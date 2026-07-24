"""Verify immutable protocol_v2 Gate-run outputs against a declared matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from s2c.data.hashing import sha256_json
from s2c.data.manifests import read_json
from s2c.runtime.paths import ProtocolV2Paths

from .matrix import GateRunSpec, filter_gate_specs, load_gate_matrix
from .runner import _config_payload, _run_paths


def verify(paths: ProtocolV2Paths, specs: list[GateRunSpec], *, require_complete: bool) -> dict[str, int]:
    complete = 0
    missing = 0
    invalid = 0
    for spec in specs:
        run_dir = _run_paths(paths, spec)
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "predictions" / "test.jsonl"
        if not manifest_path.is_file():
            missing += 1
            continue
        try:
            manifest = read_json(manifest_path)
            metrics = read_json(metrics_path)
            if (
                manifest.get("status") != "complete"
                or manifest.get("config_hash") != sha256_json(_config_payload(spec))
                or manifest.get("test_used_for_selection") is not False
                or not isinstance(metrics.get("combined"), dict)
                or not predictions_path.is_file()
            ):
                invalid += 1
                continue
            complete += 1
        except (OSError, ValueError, TypeError):
            invalid += 1
    if require_complete and (missing or invalid):
        raise SystemExit(f"Gate verification failed: complete={complete}, missing={missing}, invalid={invalid}")
    return {"planned": len(specs), "complete": complete, "missing": missing, "invalid": invalid}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--kir", type=float, action="append")
    args = parser.parse_args(argv)
    specs = filter_gate_specs(
        load_gate_matrix(args.config), datasets=args.dataset, seeds=args.seed, kirs=args.kir
    )
    result = verify(ProtocolV2Paths.discover(), specs, require_complete=args.require_complete)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
