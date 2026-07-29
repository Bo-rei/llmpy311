"""Resumable R1_full runner for geometry-preserving MiniLM Gate evidence.

This module is deliberately separate from the bounded R1 pilot.  It reuses
the pilot's training and Gate functions, reads only the frozen E2 embedding
cache, and writes to a new artifact root.  A cell is one dataset/KIR/seed/
representation combination; its two K values share the same representation
checkpoint and encoding.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from protocol_v2.data.hashing import atomic_write_json, sha256_file, sha256_json
from protocol_v2.data.manifests import dataset_manifest_path, read_json
from protocol_v2.experiments.geometry_preserving import (
    DEFAULT_MODEL,
    _safe,
    evaluate_gate,
    fixed_oos_buckets,
    geometry_metrics,
    load_bundle,
    train_representation,
    write_csv,
)
from protocol_v2.experiments.r1_runner import _encode_representation, _model_path
from protocol_v2.runtime.paths import ProtocolV2Paths


FULL_NAME = "r1_geometry_preserving_representation_full"
PILOT_NAME = "r1_geometry_preserving_representation"


def full_root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / FULL_NAME


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"R1_full config must be a mapping: {path}")
    return payload


def _registry_tree_hash(paths: ProtocolV2Paths) -> str:
    rows = [
        (str(path.relative_to(paths.registry_root)), sha256_file(path))
        for path in sorted(paths.registry_root.glob("*/seed_*/kir_*.json"))
    ]
    return sha256_json(rows)


def _git_patch(repo_root: Path, patch_path: Path) -> str:
    """Capture the current checkout identity without committing it."""

    tracked = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root.parent,
        capture_output=True,
        check=True,
    ).stdout
    chunks = [tracked]
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    for relative in untracked:
        if relative.startswith(("artifacts/", "assets/")):
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "/dev/null", relative],
            cwd=repo_root.parent,
            capture_output=True,
            check=False,
        )
        if result.stdout:
            chunks.append(result.stdout)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(b"\n".join(chunks))
    return sha256_file(patch_path)


def freeze_provenance(paths: ProtocolV2Paths, config_path: Path) -> dict[str, Any]:
    root = full_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_path = root / "R1_FULL_PROVENANCE_SNAPSHOT.json"
    patch_path = root / "R1_FULL_CODE_SNAPSHOT.patch"
    patch_sha = _git_patch(paths.project_root, patch_path)
    snapshot = {
        "schema_version": "s2c.r1_full_provenance.v1",
        "stage": "R1_full",
        "experiment": FULL_NAME,
        "protocol_version": paths.dataset_version,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=paths.project_root.parent, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "git_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=paths.project_root.parent, capture_output=True, text=True, check=True
        ).stdout.strip()),
        "code_patch_sha256": patch_sha,
        "code_patch": str(patch_path),
        "config_sha256": sha256_file(config_path),
        "e2_closeout_sha256": sha256_file(paths.run_root / "summaries/e2_closeout/E2_closeout_manifest.json"),
        "r1_pilot_closeout_sha256": sha256_file(
            paths.run_root / PILOT_NAME / "summaries/R1_CLOSEOUT.md"
        ),
        "canonical_manifest_sha256": {
            dataset: sha256_file(dataset_manifest_path(paths.manifest_root, dataset))
            for dataset in ("clinc150", "banking77", "stackoverflow")
        },
        "registry_tree_sha256": _registry_tree_hash(paths),
        "encoder": DEFAULT_MODEL,
        "python": platform.python_version(),
        "beta": 1.0,
        "beta_source": "R1 pilot Known-only global selection",
        "oos_used_for_selection": False,
        "test_used_for_selection": False,
    }
    if snapshot_path.is_file():
        existing = read_json(snapshot_path)
        if existing != snapshot:
            raise RuntimeError(f"Refusing to overwrite frozen R1_full provenance: {snapshot_path}")
    else:
        atomic_write_json(snapshot_path, snapshot)
    return snapshot


def require_provenance(paths: ProtocolV2Paths) -> dict[str, Any]:
    path = full_root(paths) / "R1_FULL_PROVENANCE_SNAPSHOT.json"
    if not path.is_file():
        raise RuntimeError("R1_full provenance is not frozen; run freeze first")
    return read_json(path)


def _cell_dir(paths: ProtocolV2Paths, dataset: str, kir: float, seed: int, representation: str) -> Path:
    return full_root(paths) / "gate_runs" / dataset / f"kir_{kir:.2f}" / f"seed_{seed}" / representation


def _checkpoint_dir(paths: ProtocolV2Paths, dataset: str, kir: float, seed: int, representation: str) -> Path:
    return full_root(paths) / "checkpoints" / dataset / f"kir_{kir:.2f}" / f"seed_{seed}" / representation


def _selected_distance(config: Mapping[str, Any], dataset: str) -> str:
    values = config.get("distance_by_dataset", {})
    distance = values.get(dataset)
    if not distance:
        raise ValueError(f"R1_full has no selected distance for dataset={dataset}")
    return str(distance)


def _train_cell(
    paths: ProtocolV2Paths,
    config: Mapping[str, Any],
    dataset: str,
    kir: float,
    seed: int,
    representation: str,
) -> dict[str, Any] | None:
    if representation == "frozen_minilm":
        return None
    bundle = load_bundle(paths, dataset, seed, kir)
    method = "ce_recon" if representation == "ce_recon" else "ce_recon_geometry"
    return train_representation(
        model_path=_model_path(paths, config),
        train_rows=bundle.views.train,
        calibration_rows=bundle.views.calibration,
        teacher_train=bundle.train,
        output_dir=_checkpoint_dir(paths, dataset, kir, seed, representation),
        method=method,
        seed=seed,
        beta=0.0 if method == "ce_recon" else float(config["beta"]),
        alpha=float(config["alpha"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        max_length=int(config["max_length"]),
    )


def _load_embeddings(
    paths: ProtocolV2Paths,
    config: Mapping[str, Any],
    bundle: Any,
    dataset: str,
    kir: float,
    seed: int,
    representation: str,
) -> dict[str, np.ndarray]:
    if representation == "frozen_minilm":
        return {"train": bundle.train, "calibration": bundle.calibration, "test": bundle.test}
    checkpoint = _checkpoint_dir(paths, dataset, kir, seed, representation) / "encoder.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"R1_full checkpoint missing: {checkpoint}")
    return _encode_representation(
        _model_path(paths, config),
        checkpoint,
        {"train": bundle.views.train, "calibration": bundle.views.calibration, "test": bundle.views.test},
        int(config["batch_size"]),
        int(config["max_length"]),
    )


def _run_cell(paths: ProtocolV2Paths, config: Mapping[str, Any], dataset: str, kir: float, seed: int, representation: str) -> Path:
    cell = _cell_dir(paths, dataset, kir, seed, representation)
    metrics_path = cell / "metrics.json"
    expected = {
        "protocol_version": paths.dataset_version,
        "stage": "R1_full",
        "experiment_id": FULL_NAME,
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "representation": representation,
        "distance": _selected_distance(config, dataset),
        "k_values": list(config["k_values"]),
    }
    if metrics_path.is_file():
        existing = read_json(metrics_path)
        if existing.get("status") == "complete" and existing.get("config") == expected:
            return metrics_path
        raise RuntimeError(f"Refusing to overwrite completed or incompatible R1_full cell: {cell}")

    started = time.perf_counter()
    bundle = load_bundle(paths, dataset, seed, kir)
    buckets, bucket_info = fixed_oos_buckets(
        bundle.train,
        bundle.test,
        bundle.views.test,
        [str(row["intent"]) for row in bundle.views.train],
    )
    training_manifest = _train_cell(paths, config, dataset, kir, seed, representation)
    embeddings = _load_embeddings(paths, config, bundle, dataset, kir, seed, representation)
    geometry = geometry_metrics(
        bundle.train,
        bundle.calibration,
        bundle.test,
        embeddings["train"],
        embeddings["calibration"],
        embeddings["test"],
        bundle.views.train,
        bundle.views.calibration,
        bundle.views.test,
        seed,
    )
    distance = _selected_distance(config, dataset)
    gate_rows: list[dict[str, Any]] = []
    for k in config["k_values"]:
        metrics = evaluate_gate(
            embeddings["train"],
            embeddings["calibration"],
            embeddings["test"],
            bundle.views.train,
            bundle.views.calibration,
            bundle.views.test,
            int(k),
            distance,
            buckets,
        )
        gate_rows.append({
            **metrics,
            **expected,
            "k": int(k),
            "boundary": str(config["boundary"]),
            "radius_lambda": float(config["radius_lambda"]),
            "bucket_q20": bucket_info.get("q20"),
            "bucket_q80": bucket_info.get("q80"),
            "bucket_definition": bucket_info.get("source"),
            "training_manifest": training_manifest,
            "geometry": geometry,
        })
    payload = {
        "status": "complete",
        "config": expected,
        "training_manifest": training_manifest,
        "geometry": geometry,
        "gate_rows": gate_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "e2_manifest": bundle.e2_manifest,
        "used_oos_for_training": False,
        "used_test_for_selection": False,
    }
    atomic_write_json(metrics_path, _safe(payload))
    return metrics_path


def build_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    cells = len(config["datasets"]) * len(config["kirs"]) * len(config["seeds"]) * len(config["representations"])
    return {
        "stage": "R1_full",
        "experiment_id": FULL_NAME,
        "protocol_version": config["protocol_version"],
        "planned_cell_units": cells,
        "planned_gate_units": cells * len(config["k_values"]),
        "runs_started": False,
        "config": dict(config),
    }


def run(paths: ProtocolV2Paths, config: Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    require_provenance(paths)
    plan = build_plan(config)
    atomic_write_json(full_root(paths) / "plans" / "R1_full_plan.json", plan)
    if dry_run:
        return {**plan, "dry_run": True}
    completed = 0
    failures: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for kir in config["kirs"]:
            for seed in config["seeds"]:
                for representation in config["representations"]:
                    try:
                        _run_cell(paths, config, str(dataset), float(kir), int(seed), str(representation))
                        completed += 1
                    except Exception as exc:  # preserve a resumable failure record
                        failures.append({
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "representation": representation,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
                        atomic_write_json(full_root(paths) / "failures" / f"{dataset}_kir_{float(kir):.2f}_seed_{seed}_{representation}.json", failures[-1])
    return {"planned_cell_units": plan["planned_cell_units"], "completed_cell_units": completed, "failed_cells": len(failures), "failures": failures}


def summarize(paths: ProtocolV2Paths, config: Mapping[str, Any]) -> dict[str, Any]:
    root = full_root(paths)
    cell_paths = sorted(root.glob("gate_runs/*/kir_*/seed_*/**/metrics.json"))
    training_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    failures = sorted(root.glob("failures/*.json"))
    for path in cell_paths:
        payload = read_json(path)
        if payload.get("status") != "complete":
            continue
        cell = payload["config"]
        manifest = payload.get("training_manifest")
        if manifest:
            training_rows.append({**manifest, **cell, "stage": "R1_full"})
        geometry_rows.append({**payload.get("geometry", {}), **cell, "stage": "R1_full"})
        for row in payload.get("gate_rows", []):
            flat = {key: value for key, value in row.items() if key not in {"geometry", "training_manifest"}}
            gate_rows.append(flat)
    summary_root = root / "summaries"
    write_csv(summary_root / "R1_full_training_summary.csv", training_rows)
    write_csv(summary_root / "R1_full_gate_summary.csv", gate_rows)
    write_csv(summary_root / "R1_full_geometry_analysis.csv", geometry_rows)
    write_csv(summary_root / "R1_full_failed_or_invalid_runs.csv", [read_json(path) for path in failures])
    expected_cells = int(build_plan(config)["planned_cell_units"])
    expected_gates = int(build_plan(config)["planned_gate_units"])
    integrity = {
        "protocol_version": paths.dataset_version,
        "planned_cell_units": expected_cells,
        "completed_cell_units": len(cell_paths),
        "planned_gate_units": expected_gates,
        "completed_gate_units": len(gate_rows),
        "failed_cells": len(failures),
        "invalid_metrics": sum(1 for row in gate_rows if not math.isfinite(float(row.get("oos_f1", math.nan)))),
        "e4_to_e7_started": False,
    }
    atomic_write_json(summary_root / "R1_full_integrity.json", integrity)
    return integrity


def verify(paths: ProtocolV2Paths, config: Mapping[str, Any]) -> dict[str, Any]:
    integrity = summarize(paths, config)
    expected = build_plan(config)
    ok = (
        integrity["completed_cell_units"] == expected["planned_cell_units"]
        and integrity["completed_gate_units"] == expected["planned_gate_units"]
        and integrity["failed_cells"] == 0
        and integrity["invalid_metrics"] == 0
    )
    return {**integrity, "status": "ok" if ok else "incomplete"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "plan", "run", "summarize", "verify"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/protocol_v2_textoir_v1/r1_geometry_preserving_full.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    config = load_config(args.config)
    if args.command == "freeze":
        print(json.dumps(_safe(freeze_provenance(paths, args.config)), ensure_ascii=False, indent=2))
        return 0
    if args.command == "plan":
        plan = build_plan(config)
        print(json.dumps(_safe(plan), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        result = run(paths, config, dry_run=args.dry_run)
    elif args.command == "summarize":
        result = summarize(paths, config)
    else:
        result = verify(paths, config)
    print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
    return 0 if result.get("status", "ok") in {"ok", "complete"} and int(result.get("failed_cells", 0)) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
