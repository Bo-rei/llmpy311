"""Trainable MiniLM representation with a controlled K=1..5 boundary sweep.

The encoder is trained once per dataset/seed by the existing Known-only
Trainable MiniLM K=1 contract.  The same checkpoint is then evaluated with
five fixed boundary geometries.  This isolates the effect of K without
retraining a different representation for every K.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from protocol_v2.data.hashing import atomic_write_json, sha256_file, sha256_json
from protocol_v2.experiments.racal_v1.boundary import (
    detector_signature,
    evaluate_open,
    fit_k1_detector,
)
from protocol_v2.experiments.racal_v1.representation import (
    RacalMiniLM,
    choose_device,
    encode_rows,
)
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from protocol_v2.gate.view_loader import load_gate_views
from protocol_v2.runtime.paths import ProtocolV2Paths
from protocol_v2.tracking.run_manifest import atomic_run_directory


STAGE = "minilm_trainable_k_sweep_v1"
PROTOCOL = "protocol_v2_textoir_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87, 100, 123)
K_VALUES = (1, 2, 3, 4, 5)
KIR = 0.50
DISTANCE = "mahalanobis_diag"
RADIUS_METHOD = "mean_std"
RADIUS_LAMBDA = 1.0
THRESHOLD = 1.0
PARTITION_SEED = 42
K1_REPLAY_TOLERANCE = 1e-6
PRIMARY_SOURCE = "minilm_trainable_kir_sweep_v1/kir_0.50"
EXTENSION_SOURCE = "minilm_trainable_kir_sweep_extension_v1/kir_0.50"


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        item = value.item()
        return item if not isinstance(item, float) or math.isfinite(item) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(str(row["sample_id"]) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=project_root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "base_commit": run("rev-parse", "HEAD"),
        "git_dirty": bool(run("status", "--short")),
        "status": run("status", "--short"),
    }


def _e2_manifest_path(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    name = (
        f"{PROTOCOL}__{dataset}__kir_0.50__seed_{seed}__"
        "repr_frozen_minilm__k_1__dist_euclidean__boundary_mean_std"
    )
    return paths.run_root / "e2_gate_core_dense" / name / "manifest.json"


def _load_views(paths: ProtocolV2Paths, dataset: str, seed: int) -> tuple[Any, dict[str, Any]]:
    manifest_path = _e2_manifest_path(paths, dataset, seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing E2 reference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != paths.dataset_version:
        raise ValueError(f"E2 protocol mismatch: {manifest_path}")

    views = load_gate_views(paths, dataset, seed, KIR)
    train_ids = {str(row["sample_id"]) for row in views.train}
    calibration_ids = {str(row["sample_id"]) for row in views.calibration}
    test_ids = {str(row["sample_id"]) for row in views.test}
    if train_ids & calibration_ids or train_ids & test_ids or calibration_ids & test_ids:
        raise ValueError(f"Split overlap for {dataset}/{seed}")
    if any(int(row["label"]) != 0 for row in views.train + views.calibration):
        raise ValueError(f"OOS row in Known train/calibration for {dataset}/{seed}")

    return views, {
        "manifest_path": str(manifest_path),
        "registry_sha256": manifest.get("registry_sha256"),
        "canonical_manifest_sha256": manifest.get("canonical_manifest_sha256"),
        "train_sample_ids_sha256": _rows_hash(views.train),
        "calibration_sample_ids_sha256": _rows_hash(views.calibration),
        "test_sample_ids_sha256": _rows_hash(views.test),
        "overlap_counts": {
            "train_calibration": 0,
            "train_test": 0,
            "calibration_test": 0,
        },
        "train_count": len(views.train),
        "calibration_count": len(views.calibration),
        "test_count": len(views.test),
        "test_known_count": int(sum(int(row["label"]) == 0 for row in views.test)),
        "test_oos_count": int(sum(int(row["label"]) == 1 for row in views.test)),
    }


def _source_stage(seed: int) -> str:
    if seed in (13, 42, 87):
        return PRIMARY_SOURCE
    return EXTENSION_SOURCE


def _source_run(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    return paths.run_root / _source_stage(seed) / "runs" / dataset / f"seed_{seed}"


def _load_checkpoint(
    paths: ProtocolV2Paths,
    config: Mapping[str, Any],
    dataset: str,
    seed: int,
    device: torch.device,
) -> tuple[RacalMiniLM, Any, dict[str, Any]]:
    run_dir = _source_run(paths, dataset, seed)
    manifest_path = run_dir / "training_manifest.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Trainable MiniLM checkpoint: {run_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "status": "complete",
        "protocol_version": PROTOCOL,
        "dataset": dataset,
        "kir": KIR,
        "seed": seed,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Trainable checkpoint contract mismatch for {manifest_path}: {key}")

    expected_hash = str(manifest.get("checkpoint_sha256", ""))
    actual_hash = sha256_file(checkpoint_path)
    if expected_hash != actual_hash:
        raise ValueError(f"Checkpoint hash changed: {checkpoint_path}")

    mode = str(manifest.get("freeze_report", {}).get("mode", ""))
    if mode != "last2_minilm_plus_projection":
        raise ValueError(f"Unexpected Trainable representation mode: {mode}")
    hidden_dim = int(manifest.get("projection_hidden_dim", config["projection_hidden_dim"]))
    model_path = (paths.project_root / str(config["model_path"])).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = RacalMiniLM(model_path, mode, hidden_dim).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()

    return model, tokenizer, {
        "stage": _source_stage(seed),
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_hash,
        "mode": mode,
        "projection_hidden_dim": hidden_dim,
    }


def _fit_detector(train: np.ndarray, rows: Sequence[Mapping[str, Any]], k: int) -> MultiSphereOOSDetector:
    if k == 1:
        return fit_k1_detector(train, rows, DISTANCE, radius_lambda=RADIUS_LAMBDA)
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=k,
        radius_method=RADIUS_METHOD,
        radius_lambda=RADIUS_LAMBDA,
        distance_metric=DISTANCE,
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=PARTITION_SEED,
        acceptance_mode="nearest_sphere",
    )
    detector.fit(np.asarray(train), np.asarray([str(row["intent"]) for row in rows], dtype=object))
    return detector


def _metrics_subset(metrics: Mapping[str, Any]) -> dict[str, float]:
    names = (
        "oos_f1",
        "oos_precision",
        "oos_recall",
        "f1_all",
        "f1_u",
        "f1_k",
        "accuracy",
        "known_recall",
        "false_accept_rate",
        "false_reject_rate",
        "auroc",
        "aupr_oos",
        "fpr95",
    )
    return {name: float(metrics[name]) for name in names}


def _run_dir(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    return paths.run_root / STAGE / "runs" / dataset / f"seed_{seed}"


def _run_unit(
    paths: ProtocolV2Paths,
    config: Mapping[str, Any],
    dataset: str,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    views, split = _load_views(paths, dataset, seed)
    device = choose_device(str(config.get("device", "auto")))
    model, tokenizer, checkpoint = _load_checkpoint(paths, config, dataset, seed, device)

    train_values = encode_rows(
        model, tokenizer, views.train, device, int(config["batch_size"]), int(config["max_length"])
    )
    calibration_values = encode_rows(
        model,
        tokenizer,
        views.calibration,
        device,
        int(config["batch_size"]),
        int(config["max_length"]),
    )
    test_values = encode_rows(
        model, tokenizer, views.test, device, int(config["batch_size"]), int(config["max_length"])
    )

    source_manifest = json.loads(Path(checkpoint["manifest_path"]).read_text(encoding="utf-8"))
    source_split = source_manifest.get("input", {}).get("split_validation", {})
    expected_input = {
        "train_sample_ids_sha256": split["train_sample_ids_sha256"],
        "calibration_sample_ids_sha256": split["calibration_sample_ids_sha256"],
        "test_sample_ids_sha256": split["test_sample_ids_sha256"],
        "registry_sha256": split["registry_sha256"],
        "canonical_manifest_sha256": split["canonical_manifest_sha256"],
    }
    mismatches = {}
    for key, expected in expected_input.items():
        source = source_split if key.endswith("sample_ids_sha256") else source_manifest.get("input", {})
        if source.get(key) != expected:
            mismatches[key] = {"source": source.get(key), "current": expected}
    if mismatches:
        raise ValueError(f"Trainable checkpoint input mismatch for {dataset}/{seed}: {mismatches}")

    config_payload = {
        "stage": STAGE,
        "protocol_version": PROTOCOL,
        "dataset": dataset,
        "kir": KIR,
        "seed": seed,
        "representation": "last2_minilm_plus_projection",
        "representation_checkpoint": checkpoint,
        "k_values": list(K_VALUES),
        "distance": DISTANCE,
        "radius_method": RADIUS_METHOD,
        "radius_lambda": RADIUS_LAMBDA,
        "threshold": THRESHOLD,
        "partition_seed": PARTITION_SEED,
        "input": split,
        "embedding_hashes": {
            "train": _array_hash(train_values),
            "calibration": _array_hash(calibration_values),
            "test": _array_hash(test_values),
        },
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    config_hash = sha256_json(config_payload)
    run_dir = _run_dir(paths, dataset, seed)
    existing_manifest = run_dir / "run_manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if resume and existing.get("config_hash") == config_hash and existing.get("status") == "complete":
            return {**existing, "run_dir": str(run_dir)}
        raise FileExistsError(f"K sweep run exists or has a different config: {run_dir}")

    results: dict[str, dict[str, float]] = {}
    calibration: dict[str, dict[str, float]] = {}
    signatures: dict[str, dict[str, Any]] = {}
    k1_replay: dict[str, float] = {}
    source_metrics_path = Path(checkpoint["manifest_path"]).parent / "metrics.json"
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))

    for k in K_VALUES:
        detector = _fit_detector(train_values, views.train, k)
        test_metrics, _ = evaluate_open(detector, test_values, views.test, THRESHOLD)
        calibration_metrics, _ = evaluate_open(detector, calibration_values, views.calibration, THRESHOLD)
        results[f"k_{k}"] = _metrics_subset(test_metrics)
        calibration[f"k_{k}"] = {
            "known_recall": float(calibration_metrics["known_recall"]),
            "false_reject_rate": float(calibration_metrics["false_reject_rate"]),
        }
        signatures[f"k_{k}"] = detector_signature(detector)
        if k == 1:
            k1_replay = {
                name: abs(float(test_metrics[name]) - float(source_metrics[name]))
                for name in ("oos_f1", "known_recall", "auroc", "aupr_oos")
                if name in source_metrics
            }
    if k1_replay and max(k1_replay.values()) > K1_REPLAY_TOLERANCE:
        raise RuntimeError(f"K=1 replay mismatch for {dataset}/{seed}: {k1_replay}")

    metrics = {
        "stage": STAGE,
        "dataset": dataset,
        "kir": KIR,
        "seed": seed,
        "representation": "last2_minilm_plus_projection",
        "distance": DISTANCE,
        "radius_method": RADIUS_METHOD,
        "radius_lambda": RADIUS_LAMBDA,
        "threshold": THRESHOLD,
        "results": results,
        "calibration": calibration,
        "k1_replay_max_abs_delta": max(k1_replay.values()) if k1_replay else None,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    started = time.time()
    manifest = {
        **config_payload,
        "config_hash": config_hash,
        "status": "complete",
        "stage": STAGE,
        "device": str(device),
        "detector_signatures": signatures,
        "metrics_path": str(run_dir / "metrics.json"),
        "elapsed_seconds": 0.0,
    }
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_json(temporary / "resolved_config.json", _safe(config_payload))
        atomic_write_json(temporary / "metrics.json", _safe(metrics))
        atomic_write_json(temporary / "detector_signatures.json", _safe(signatures))
        atomic_write_json(temporary / "run_manifest.json", _safe({**manifest, "elapsed_seconds": time.time() - started}))
    return {**manifest, "elapsed_seconds": time.time() - started, "run_dir": str(run_dir)}


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config must be a mapping: {path}")
    if str(payload.get("protocol_version")) != PROTOCOL:
        raise ValueError("Only protocol_v2_textoir_v1 is supported")
    if abs(float(payload.get("kir", -1)) - KIR) > 1e-12:
        raise ValueError("This sweep is fixed at KIR=0.50")
    if tuple(str(value).lower() for value in payload.get("datasets", [])) != DATASETS:
        raise ValueError(f"Datasets must be exactly {DATASETS}")
    if tuple(int(value) for value in payload.get("seeds", [])) != SEEDS:
        raise ValueError(f"Seeds must be exactly {SEEDS}")
    if tuple(int(value) for value in payload.get("k_values", [])) != K_VALUES:
        raise ValueError(f"K values must be exactly {K_VALUES}")
    if str(payload.get("distance")) != DISTANCE or str(payload.get("radius_method")) != RADIUS_METHOD:
        raise ValueError("Distance/radius contract mismatch")
    if float(payload.get("radius_lambda")) != RADIUS_LAMBDA or float(payload.get("threshold")) != THRESHOLD:
        raise ValueError("Radius/threshold contract mismatch")
    if bool(payload.get("test_used_for_selection", False)) or bool(payload.get("oos_used_for_training", False)):
        raise ValueError("Test/OOS use is forbidden")
    return {str(key): value for key, value in payload.items()}


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_summaries(paths: ProtocolV2Paths, unit_results: Sequence[Mapping[str, Any]]) -> None:
    stage = paths.run_root / STAGE
    per_seed: list[dict[str, Any]] = []
    for item in unit_results:
        metrics = json.loads((Path(item["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        for k in K_VALUES:
            row = {
                "dataset": metrics["dataset"],
                "kir": metrics["kir"],
                "seed": metrics["seed"],
                "k": k,
                "representation": metrics["representation"],
                "distance": metrics["distance"],
                **metrics["results"][f"k_{k}"],
            }
            per_seed.append(row)
    _atomic_csv(stage / "summary_per_seed.csv", per_seed)

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in per_seed:
        grouped.setdefault((str(row["dataset"]), int(row["k"])), []).append(row)
    summary: list[dict[str, Any]] = []
    metrics_names = ("oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos")
    for (dataset, k), rows in sorted(grouped.items()):
        output: dict[str, Any] = {"dataset": dataset, "kir": KIR, "k": k, "n_seeds": len(rows)}
        for name in metrics_names:
            values = [float(row[name]) for row in rows]
            output[f"{name}_mean"] = float(np.mean(values))
            output[f"{name}_std"] = float(np.std(values))
        summary.append(output)
    _atomic_csv(stage / "summary.csv", summary)


def _provenance(paths: ProtocolV2Paths, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "s2c.minilm_trainable_k_sweep_v1.provenance.v1",
        "stage": STAGE,
        "protocol_version": PROTOCOL,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config": dict(config),
        "source_stages": [PRIMARY_SOURCE, EXTENSION_SOURCE],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git": _git_state(paths.project_root),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "k_interpretation": "same trained representation evaluated with fixed K-dependent boundary geometry",
    }


def run_stage(
    paths: ProtocolV2Paths,
    config_path: Path,
    config: Mapping[str, Any],
    datasets: Sequence[str],
    seeds: Sequence[int],
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    paths.require_experiment_admission()
    for dataset in datasets:
        paths.require_experiment_admission(dataset)
    stage = paths.run_root / STAGE
    stage.mkdir(parents=True, exist_ok=True)
    provenance = stage / "PROVENANCE.json"
    if not provenance.is_file():
        atomic_write_json(provenance, _safe(_provenance(paths, config_path, config)))

    checks = []
    for dataset in datasets:
        for seed in seeds:
            views, split = _load_views(paths, dataset, seed)
            source = _source_run(paths, dataset, seed)
            if not (source / "training_manifest.json").is_file() or not (source / "checkpoint.pt").is_file():
                raise FileNotFoundError(f"Missing source checkpoint: {source}")
            checks.append({"dataset": dataset, "seed": seed, "split": split, "source_run": str(source), "train_count": len(views.train), "calibration_count": len(views.calibration), "test_count": len(views.test)})
    plan = {
        "stage": STAGE,
        "status": "preflight_ok",
        "datasets": list(datasets),
        "seeds": list(seeds),
        "k_values": list(K_VALUES),
        "planned_units": len(datasets) * len(seeds),
        "checks": checks,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    atomic_write_json(stage / "plans" / "stage_plan.json", _safe(plan))
    if dry_run:
        return plan

    total_units = len(datasets) * len(seeds)
    results = []
    for unit_index, (dataset, seed) in enumerate(
        ((dataset, seed) for dataset in datasets for seed in seeds),
        start=1,
    ):
        print(
            f"[trainable-k-sweep {unit_index}/{total_units}] {dataset}/seed_{seed}",
            flush=True,
        )
        results.append(_run_unit(paths, config, dataset, seed, resume))
    _write_summaries(paths, results)
    state = {
        "stage": STAGE,
        "status": "complete",
        "planned_units": len(results),
        "completed_units": len(results),
        "failed_units": 0,
        "k_values": list(K_VALUES),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    atomic_write_json(stage / "state.json", _safe(state))
    return {**state, "results": results, "root": str(stage)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Trainable MiniLM with K=1..5 boundary geometries")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if tuple(str(value).lower() for value in args.datasets) != DATASETS or tuple(args.seeds) != SEEDS:
        raise ValueError(f"Requested datasets/seeds must be exactly {DATASETS}/{SEEDS}")
    paths = ProtocolV2Paths.discover()
    result = run_stage(paths, args.config.resolve(), config, DATASETS, SEEDS, args.resume, args.dry_run)
    print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
