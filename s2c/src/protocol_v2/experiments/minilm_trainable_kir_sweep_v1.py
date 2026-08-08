"""Known-only Trainable MiniLM KIR sweep.

This stage reuses the already audited K=1 training/evaluation implementation.
It changes only the KIR control variable and writes to a new run root.  No
historical stage is overwritten and no test OOS data is used for selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from protocol_v2.runtime.paths import ProtocolV2Paths
from protocol_v2.experiments import minilm_trainable_control_v1 as base
from protocol_v2.data.hashing import atomic_write_json, sha256_file, sha256_json


STAGE = "minilm_trainable_kir_sweep_v1"
ALLOWED_KIRS = (0.25, 0.50, 0.75)
ALLOWED_SEEDS = (13, 42, 87)


def _load_sweep(path: Path) -> tuple[dict[str, Any], base.ControlConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Configuration must be a mapping: {path}")
    kirs = tuple(float(value) for value in payload.get("kirs", ()))
    seeds = tuple(int(value) for value in payload.get("seeds", ()))
    datasets = tuple(str(value).lower() for value in payload.get("datasets", ()))
    if not kirs or any(value not in ALLOWED_KIRS for value in kirs):
        raise ValueError(f"KIRs must be drawn from {ALLOWED_KIRS}: {kirs}")
    if not seeds or any(value not in ALLOWED_SEEDS for value in seeds):
        raise ValueError(f"Seeds must be drawn from {ALLOWED_SEEDS}: {seeds}")
    if not datasets or any(value not in base.ALLOWED_DATASETS for value in datasets):
        raise ValueError(f"Unsupported datasets: {datasets}")
    base_path = (path.parents[3] / str(payload.get("base_config", ""))).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base Trainable MiniLM config is missing: {base_path}")
    base_config = base.load_config(base_path)
    return {
        "stage": STAGE,
        "protocol_version": str(payload.get("protocol_version", "")),
        "datasets": datasets,
        "kirs": kirs,
        "seeds": seeds,
        "base_config": str(base_path),
        "base_config_sha256": sha256_file(base_path),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }, base_config


def _stage_root(paths: ProtocolV2Paths, kir: float) -> Path:
    return paths.run_root / STAGE / f"kir_{kir:.2f}"


def _run_one(paths: ProtocolV2Paths, config_path: Path, base_config: base.ControlConfig, datasets: Sequence[str], seeds: Sequence[int], kir: float, resume: bool, dry_run: bool) -> dict[str, Any]:
    """Invoke the audited runner with an isolated stage namespace."""
    previous_stage = base.STAGE
    try:
        base.STAGE = f"{STAGE}/kir_{kir:.2f}"
        config = replace(base_config, kir=kir, datasets=tuple(datasets), seeds=tuple(seeds))
        return base.run_stage(paths, config_path, config, datasets, seeds, resume, dry_run)
    finally:
        base.STAGE = previous_stage


def _provenance(paths: ProtocolV2Paths, config_path: Path, payload: Mapping[str, Any], base_config: base.ControlConfig) -> dict[str, Any]:
    return {
        "schema_version": "s2c.minilm_trainable_kir_sweep_v1.provenance.v1",
        "stage": STAGE,
        "protocol_version": payload["protocol_version"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_hash": sha256_json(dict(payload)),
        "base_config_sha256": payload["base_config_sha256"],
        "datasets": list(payload["datasets"]),
        "kirs": list(payload["kirs"]),
        "seeds": list(payload["seeds"]),
        "base_training_contract": base_config.as_dict(),
        "git": base.git_state(paths.project_root),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "historical_artifacts_immutable": True,
    }


def run_sweep(paths: ProtocolV2Paths, config_path: Path, payload: Mapping[str, Any], base_config: base.ControlConfig, resume: bool, dry_run: bool) -> dict[str, Any]:
    paths.require_experiment_admission()
    root = paths.run_root / STAGE
    root.mkdir(parents=True, exist_ok=True)
    provenance_path = root / "PROVENANCE.json"
    if not provenance_path.is_file():
        atomic_write_json(provenance_path, _provenance(paths, config_path, payload, base_config))
    plans: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for kir in payload["kirs"]:
        item = _run_one(paths, config_path, base_config, payload["datasets"], payload["seeds"], float(kir), resume, dry_run)
        if dry_run:
            plans.append(item)
        else:
            results.append(item)
    state = {
        "stage": STAGE,
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Known-only Trainable MiniLM KIR sweep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    payload, base_config = _load_sweep(args.config.resolve())
    if payload["protocol_version"] != "protocol_v2_textoir_v1":
        raise ValueError(f"Unsupported protocol: {payload['protocol_version']}")
    result = run_sweep(paths, args.config.resolve(), payload, base_config, args.resume, args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
