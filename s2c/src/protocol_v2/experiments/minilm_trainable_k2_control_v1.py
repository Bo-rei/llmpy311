"""Cross-dataset K=1 versus K=2 evaluation for frozen Trainable MiniLM checkpoints.

This is an evaluation-only control.  It does not retrain an encoder, choose K from
test data, or modify the historical RACAL/E2 artifacts.  CLINC150 and Banking77
reuse the checkpoints produced by ``minilm_trainable_control_v1``; StackOverflow's
already completed RACAL-v1 K=1/K=2 control remains the reference for that dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from protocol_v2.data.hashing import atomic_write_json, atomic_write_jsonl, sha256_file, sha256_json
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from protocol_v2.gate.view_loader import load_gate_views
from protocol_v2.experiments.racal_v1.boundary import detector_signature, evaluate_open, fit_k1_detector
from protocol_v2.experiments.racal_v1.representation import RacalMiniLM, choose_device, encode_rows
from protocol_v2.runtime.paths import ProtocolV2Paths
from protocol_v2.tracking.run_manifest import atomic_run_directory


STAGE = "minilm_trainable_k2_control_v1"
SOURCE_STAGE = "minilm_trainable_control_v1"
DATASETS = ("clinc150", "banking77")
SEEDS = (13, 42, 87)
KIR = 0.50
DISTANCE = "mahalanobis_diag"
RADIUS_METHOD = "mean_std"
RADIUS_LAMBDA = 1.0
THRESHOLD = 1.0
PARTITION_SEED = 42


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        number = value.item()
        return number if not isinstance(number, float) or math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["sample_id"]) for row in rows).encode("utf-8")).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=project_root, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    return {"base_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--short")), "status": run("status", "--short")}


def _stage_root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / STAGE


def _run_dir(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    return _stage_root(paths) / "runs" / dataset / f"seed_{seed}"


def _e2_manifest_path(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    name = (
        f"protocol_v2_textoir_v1__{dataset}__kir_0.50__seed_{seed}__"
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
        "overlap_counts": {"train_calibration": 0, "train_test": 0, "calibration_test": 0},
        "train_count": len(views.train),
        "calibration_count": len(views.calibration),
        "test_count": len(views.test),
        "test_known_count": int(sum(int(row["label"]) == 0 for row in views.test)),
        "test_oos_count": int(sum(int(row["label"]) == 1 for row in views.test)),
    }


def _source_run(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    return paths.run_root / SOURCE_STAGE / "runs" / dataset / f"seed_{seed}"


def _load_checkpoint(paths: ProtocolV2Paths, config: Mapping[str, Any], dataset: str, seed: int, device: torch.device) -> tuple[RacalMiniLM, Any, dict[str, Any]]:
    run_dir = _source_run(paths, dataset, seed)
    manifest_path = run_dir / "training_manifest.json"
    checkpoint = run_dir / "checkpoint.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Trainable K=1 checkpoint: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("test_used_for_selection") is not False:
        raise ValueError(f"Checkpoint is not a completed Known-only run: {manifest_path}")
    expected_hash = str(manifest.get("checkpoint_sha256", ""))
    actual_hash = sha256_file(checkpoint)
    if expected_hash != actual_hash:
        raise ValueError(f"Checkpoint hash changed: {checkpoint}")
    mode = str(manifest.get("freeze_report", {}).get("mode", "last2_minilm_plus_projection"))
    hidden_dim = int(manifest.get("projection_hidden_dim", config.get("projection_hidden_dim", 256)))
    model_path = (paths.project_root / str(config["model_path"])).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = RacalMiniLM(model_path, mode, hidden_dim).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer, {
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_hash,
        "mode": mode,
        "projection_hidden_dim": hidden_dim,
        "source_stage": SOURCE_STAGE,
    }


def _fit_detector(train: np.ndarray, rows: Sequence[Mapping[str, Any]], k: int) -> MultiSphereOOSDetector:
    if k == 1:
        return fit_k1_detector(train, rows, DISTANCE)
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


def _sphere_summary(detector: MultiSphereOOSDetector, train_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    labels = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    output: list[dict[str, Any]] = []
    for intent in sorted(set(labels.tolist())):
        ids = [int(value) for value in detector.intent_to_clusters[intent]]
        clusters: list[dict[str, Any]] = []
        for cluster_id in ids:
            mask = detector._train_cluster_labels == cluster_id
            sphere = next(item for item in detector.spheres if int(item.cluster_id) == cluster_id)
            points = detector._train_embeddings[mask]
            diff = points - sphere.center
            distances = np.sqrt(np.sum((diff**2) * sphere.inv_diag_cov, axis=1)) if sphere.inv_diag_cov is not None else np.linalg.norm(diff, axis=1)
            clusters.append({"cluster_id": cluster_id, "sample_count": int(mask.sum()), "radius": float(sphere.radius), "distance_variance": float(np.var(distances))})
        output.append({"intent": intent, "cluster_count": len(clusters), "clusters": clusters})
    return output


def _intent_diagnostics(
    detector_k1: MultiSphereOOSDetector,
    detector_k2: MultiSphereOOSDetector,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    pred_k1: Sequence[Mapping[str, Any]],
    pred_k2: Sequence[Mapping[str, Any]],
    dataset: str,
    seed: int,
) -> list[dict[str, Any]]:
    stats_k1 = {row["intent"]: row for row in _sphere_summary(detector_k1, train_rows)}
    stats_k2 = {row["intent"]: row for row in _sphere_summary(detector_k2, train_rows)}
    rows: list[dict[str, Any]] = []
    intents = sorted(stats_k1)
    for intent in intents:
        known_indices = [i for i, row in enumerate(test_rows) if int(row["label"]) == 0 and str(row["intent"]) == intent]
        oos_indices = [i for i, row in enumerate(test_rows) if int(row["label"]) == 1]
        k1_reject = sum(int(pred_k1[i]["predicted_is_oos"]) for i in known_indices)
        k2_reject = sum(int(pred_k2[i]["predicted_is_oos"]) for i in known_indices)
        new_oos = sum(int(pred_k1[i]["predicted_is_oos"]) == 1 and int(pred_k2[i]["predicted_is_oos"]) == 0 and str(pred_k2[i]["predicted_intent"]) == intent for i in oos_indices)
        k1_clusters = stats_k1[intent]["clusters"]
        k2_clusters = stats_k2[intent]["clusters"]
        rows.append({
            "dataset": dataset,
            "seed": seed,
            "intent": intent,
            "train_sample_count": int(sum(str(row["intent"]) == intent for row in train_rows)),
            "k1_cluster_count": len(k1_clusters),
            "k2_cluster_count": len(k2_clusters),
            "k1_cluster_sizes": "|".join(str(item["sample_count"]) for item in k1_clusters),
            "k2_cluster_sizes": "|".join(str(item["sample_count"]) for item in k2_clusters),
            "k1_radii": "|".join(f"{item['radius']:.8g}" for item in k1_clusters),
            "k2_radii": "|".join(f"{item['radius']:.8g}" for item in k2_clusters),
            "known_recall_k1": 1.0 - k1_reject / max(len(known_indices), 1),
            "known_recall_k2": 1.0 - k2_reject / max(len(known_indices), 1),
            "known_recall_delta": (k2_reject - k1_reject) * -1.0 / max(len(known_indices), 1),
            "newly_accepted_oos_count": new_oos,
        })
    return rows


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: _safe(row.get(field, "")) for field in fields} for row in rows)
    temporary.replace(path)


def _metrics_subset(metrics: Mapping[str, Any]) -> dict[str, float]:
    names = ("oos_f1", "oos_precision", "oos_recall", "f1_all", "f1_u", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos", "fpr95")
    return {name: float(metrics[name]) for name in names}


def _run_unit(paths: ProtocolV2Paths, config: Mapping[str, Any], dataset: str, seed: int, resume: bool) -> dict[str, Any]:
    views, split = _load_views(paths, dataset, seed)
    device = choose_device(str(config.get("device", "auto")))
    model, tokenizer, checkpoint = _load_checkpoint(paths, config, dataset, seed, device)
    train_values = encode_rows(model, tokenizer, views.train, device, int(config["batch_size"]), int(config["max_length"]))
    calibration_values = encode_rows(model, tokenizer, views.calibration, device, int(config["batch_size"]), int(config["max_length"]))
    test_values = encode_rows(model, tokenizer, views.test, device, int(config["batch_size"]), int(config["max_length"]))
    stage1_manifest = json.loads(Path(checkpoint["manifest_path"]).read_text(encoding="utf-8"))
    stage1_metrics = json.loads((Path(checkpoint["manifest_path"]).parent / "metrics.json").read_text(encoding="utf-8"))
    source_input = stage1_manifest.get("input", {})
    source_split = source_input.get("split_validation", {})
    expected_input = {
        "train_sample_ids_sha256": split["train_sample_ids_sha256"],
        "calibration_sample_ids_sha256": split["calibration_sample_ids_sha256"],
        "test_sample_ids_sha256": split["test_sample_ids_sha256"],
        "registry_sha256": split["registry_sha256"],
        "canonical_manifest_sha256": split["canonical_manifest_sha256"],
    }
    input_mismatches = {
        key: {"source": source_input.get(key), "current": value}
        for key, value in expected_input.items()
        if (source_split if key.endswith("sample_ids_sha256") else source_input).get(key) != value
    }
    if input_mismatches:
        raise ValueError(f"Trainable checkpoint input mismatch for {dataset}/{seed}: {input_mismatches}")
    config_payload = {
        "stage": STAGE,
        "protocol_version": "protocol_v2_textoir_v1",
        "dataset": dataset,
        "kir": KIR,
        "seed": seed,
        "representation": "trainable_minilm_k1_checkpoint_reused",
        "checkpoint": checkpoint,
        "distance": DISTANCE,
        "radius_method": RADIUS_METHOD,
        "radius_lambda": RADIUS_LAMBDA,
        "threshold": THRESHOLD,
        "partition_seed": PARTITION_SEED,
        "input": split,
        "embedding_hashes": {"train": _array_hash(train_values), "calibration": _array_hash(calibration_values), "test": _array_hash(test_values)},
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    config_hash = sha256_json(config_payload)
    run_dir = _run_dir(paths, dataset, seed)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume and existing.get("config_hash") == config_hash and existing.get("status") == "complete":
            return existing
        raise FileExistsError(f"K2 run exists or has a different config: {run_dir}")
    detector_k1 = _fit_detector(train_values, views.train, 1)
    detector_k2 = _fit_detector(train_values, views.train, 2)
    metrics_k1, predictions_k1 = evaluate_open(detector_k1, test_values, views.test, THRESHOLD)
    metrics_k2, predictions_k2 = evaluate_open(detector_k2, test_values, views.test, THRESHOLD)
    names = tuple(_metrics_subset(metrics_k1))
    delta = {name: float(metrics_k2[name]) - float(metrics_k1[name]) for name in names}
    replay_delta = {name: abs(float(stage1_metrics[name]) - float(metrics_k1[name])) for name in names if name in stage1_metrics}
    if replay_delta and max(replay_delta.values()) > 1e-10:
        raise RuntimeError(f"K=1 replay mismatch for {dataset}/{seed}: {max(replay_delta.values())}")
    diagnostics = _intent_diagnostics(detector_k1, detector_k2, views.train, views.test, predictions_k1, predictions_k2, dataset, seed)
    metrics = {
        "stage": STAGE,
        "dataset": dataset,
        "seed": seed,
        "k1": _metrics_subset(metrics_k1),
        "k2": _metrics_subset(metrics_k2),
        "k2_minus_k1": delta,
        "calibration_known_recall_k1": float(evaluate_open(detector_k1, calibration_values, views.calibration, THRESHOLD)[0]["known_recall"]),
        "calibration_known_recall_k2": float(evaluate_open(detector_k2, calibration_values, views.calibration, THRESHOLD)[0]["known_recall"]),
        "k1_replay_max_abs_delta": max(replay_delta.values()) if replay_delta else None,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    run_manifest = {**config_payload, "config_hash": config_hash, "status": "complete", "stage1_manifest": str(Path(checkpoint["manifest_path"])), "stage1_metrics": {"oos_f1": stage1_metrics.get("oos_f1"), "known_recall": stage1_metrics.get("known_recall")}, "intent_diagnostic_count": len(diagnostics), "elapsed_seconds": 0.0}
    started = time.time()
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_json(temporary / "resolved_config.json", _safe(config_payload))
        atomic_write_json(temporary / "metrics.json", _safe(metrics))
        atomic_write_json(temporary / "detector_signature_k1.json", _safe(detector_signature(detector_k1)))
        atomic_write_json(temporary / "detector_signature_k2.json", _safe(detector_signature(detector_k2)))
        atomic_write_jsonl(temporary / "predictions_k2.jsonl", _safe(predictions_k2))
        _atomic_csv(temporary / "intent_diagnostics.csv", diagnostics)
        atomic_write_json(temporary / "run_manifest.json", _safe({**run_manifest, "elapsed_seconds": time.time() - started}))
    return {**run_manifest, "elapsed_seconds": time.time() - started, "run_dir": str(run_dir)}


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config must be a mapping: {path}")
    if str(payload.get("protocol_version")) != "protocol_v2_textoir_v1":
        raise ValueError("Only protocol_v2_textoir_v1 is supported")
    if abs(float(payload.get("kir", -1)) - KIR) > 1e-12:
        raise ValueError("This control is fixed at KIR=0.50")
    if tuple(str(value).lower() for value in payload.get("datasets", [])) != DATASETS:
        raise ValueError(f"This new stage is restricted to datasets {DATASETS}")
    if tuple(int(value) for value in payload.get("seeds", [])) != SEEDS:
        raise ValueError(f"This new stage is restricted to seeds {SEEDS}")
    if str(payload.get("distance")) != DISTANCE or str(payload.get("radius_method")) != RADIUS_METHOD:
        raise ValueError("Distance/radius contract mismatch")
    if float(payload.get("radius_lambda")) != RADIUS_LAMBDA or float(payload.get("threshold")) != THRESHOLD:
        raise ValueError("Radius/threshold contract mismatch")
    if bool(payload.get("test_used_for_selection", False)) or bool(payload.get("oos_used_for_training", False)):
        raise ValueError("Test/OOS use is forbidden")
    return {str(key): value for key, value in payload.items()}


def _provenance(paths: ProtocolV2Paths, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "s2c.minilm_trainable_k2_control_v1.provenance.v1",
        "stage": STAGE,
        "source_stage": SOURCE_STAGE,
        "protocol_version": "protocol_v2_textoir_v1",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config": dict(config),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git": _git_state(paths.project_root),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "historical_artifacts_immutable": True,
    }


def run_stage(paths: ProtocolV2Paths, config_path: Path, config: Mapping[str, Any], datasets: Sequence[str], seeds: Sequence[int], resume: bool, dry_run: bool) -> dict[str, Any]:
    paths.require_experiment_admission()
    for dataset in datasets:
        paths.require_experiment_admission(dataset)
    stage = _stage_root(paths)
    stage.mkdir(parents=True, exist_ok=True)
    provenance = stage / "PROVENANCE.json"
    if not provenance.is_file():
        atomic_write_json(provenance, _safe(_provenance(paths, config_path, config)))
    checks = []
    for dataset in datasets:
        for seed in seeds:
            views, split = _load_views(paths, dataset, seed)
            checks.append({"dataset": dataset, "seed": seed, "split": split, "train_count": len(views.train), "calibration_count": len(views.calibration), "test_count": len(views.test)})
    plan = {"stage": STAGE, "status": "preflight_ok", "datasets": list(datasets), "seeds": list(seeds), "planned_units": len(datasets) * len(seeds), "checks": checks, "test_used_for_selection": False, "oos_used_for_training": False}
    atomic_write_json(stage / "plans" / "stage_plan.json", _safe(plan))
    if dry_run:
        return plan
    results = [_run_unit(paths, config, dataset, seed, resume) for dataset in datasets for seed in seeds]
    state = {"stage": STAGE, "status": "complete", "planned_units": len(results), "completed_units": len(results), "failed_units": 0, "test_used_for_selection": False, "oos_used_for_training": False}
    atomic_write_json(stage / "state.json", _safe(state))
    return {**state, "results": results, "root": str(stage)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Trainable MiniLM K=1 versus K=2 on CLINC150 and Banking77")
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
