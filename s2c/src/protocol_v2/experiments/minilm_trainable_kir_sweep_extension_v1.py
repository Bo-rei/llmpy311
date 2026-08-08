"""Seed-extension for the completed Known-only Trainable MiniLM KIR sweep.

This stage is intentionally isolated from ``minilm_trainable_kir_sweep_v1``:
it adds only seeds 100 and 123 so the Trainable/Fair-MOGB comparison can use
the same five seeds.  The audited base runner and training contract are reused
without changing the completed stage or its artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from protocol_v2.data.hashing import atomic_write_json, sha256_file, sha256_json
from protocol_v2.experiments import minilm_trainable_control_v1 as base
from protocol_v2.runtime.paths import ProtocolV2Paths


STAGE = "minilm_trainable_kir_sweep_extension_v1"
ALLOWED_KIRS = (0.25, 0.50, 0.75)
ALLOWED_SEEDS = (100, 123)


def load_config(path: Path) -> tuple[dict[str, Any], base.ControlConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Configuration must be a mapping: {path}")
    datasets = tuple(str(value).lower() for value in payload.get("datasets", ()))
    kirs = tuple(float(value) for value in payload.get("kirs", ()))
    seeds = tuple(int(value) for value in payload.get("seeds", ()))
    if not datasets or any(value not in base.ALLOWED_DATASETS for value in datasets):
        raise ValueError(f"Unsupported datasets: {datasets}")
    if kirs != ALLOWED_KIRS:
        raise ValueError(f"KIRs must be exactly {ALLOWED_KIRS}: {kirs}")
    if seeds != ALLOWED_SEEDS:
        raise ValueError(f"Seeds must be exactly {ALLOWED_SEEDS}: {seeds}")
    base_path = (path.parents[3] / str(payload.get("base_config", ""))).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base config is missing: {base_path}")
    base_config = base.load_config(base_path)
    normalized = {
        "stage": STAGE,
        "protocol_version": str(payload.get("protocol_version", "")),
        "datasets": datasets,
        "kirs": kirs,
        "seeds": seeds,
        "base_config": str(base_path),
        "base_config_sha256": sha256_file(base_path),
        "parent_stage": str(payload.get("parent_stage", "")),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    return normalized, base_config


def run_stage(paths: ProtocolV2Paths, config_path: Path, payload: Mapping[str, Any], base_config: base.ControlConfig, resume: bool, dry_run: bool) -> dict[str, Any]:
    if payload["protocol_version"] != "protocol_v2_textoir_v1":
        raise ValueError(f"Unsupported protocol: {payload['protocol_version']}")
    paths.require_experiment_admission()
    root = paths.run_root / STAGE
    root.mkdir(parents=True, exist_ok=True)
    provenance = root / "PROVENANCE.json"
    if not provenance.is_file():
        atomic_write_json(
            provenance,
            {
                "schema_version": "s2c.minilm_trainable_kir_sweep_extension_v1.provenance.v1",
                "stage": STAGE,
                "protocol_version": payload["protocol_version"],
                "config_path": str(config_path),
                "config_sha256": sha256_file(config_path),
                "config_hash": sha256_json(dict(payload)),
                "parent_stage": payload["parent_stage"],
                "base_config_sha256": payload["base_config_sha256"],
                "datasets": list(payload["datasets"]),
                "kirs": list(payload["kirs"]),
                "seeds": list(payload["seeds"]),
                "base_training_contract": base_config.as_dict(),
                "git": base.git_state(paths.project_root),
                "test_used_for_selection": False,
                "oos_used_for_training": False,
                "historical_artifacts_immutable": True,
            },
        )
    plans: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    previous_stage = base.STAGE
    try:
        for kir in payload["kirs"]:
            base.STAGE = f"{STAGE}/kir_{float(kir):.2f}"
            config = replace(base_config, kir=float(kir), datasets=tuple(payload["datasets"]), seeds=tuple(payload["seeds"]))
            item = base.run_stage(paths, config_path, config, payload["datasets"], payload["seeds"], resume, dry_run)
            if dry_run:
                plans.append(item)
            else:
                results.append(item)
    finally:
        base.STAGE = previous_stage
    state = {
        "stage": STAGE,
        "parent_stage": payload["parent_stage"],
        "datasets": list(payload["datasets"]),
        "kirs": list(payload["kirs"]),
        "seeds": list(payload["seeds"]),
        "planned_units": len(payload["datasets"]) * len(payload["kirs"]) * len(payload["seeds"]),
        "completed_units": sum(int(item.get("completed", 0)) for item in results),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    atomic_write_json(root / "state.json", state)
    return {"status": "dry_run" if dry_run else "complete", **state, "plans": plans, "results": results, "root": str(root)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Extend Trainable MiniLM KIR sweep with seeds 100 and 123")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = ProtocolV2Paths.discover()
    config_path = args.config.resolve()
    payload, base_config = load_config(config_path)
    result = run_stage(paths, config_path, payload, base_config, args.resume, args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
