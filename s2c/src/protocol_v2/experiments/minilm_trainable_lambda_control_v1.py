"""Known-only radius calibration for Trainable MiniLM K=1/K=2 controls.

This stage does not train an encoder. It reuses the completed Trainable MiniLM
checkpoints and measures whether the apparent performance gap is caused by the
radius contract. Candidate lambda values are selected from Known calibration
only; test OOS metrics are recorded only after the selection rule is fixed.
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

from protocol_v2.data.hashing import atomic_write_json, sha256_file, sha256_json
from protocol_v2.experiments.minilm_trainable_k2_control_v1 import _load_checkpoint, _load_views
from protocol_v2.experiments.racal_v1.stage2 import _load_stage1_model
from protocol_v2.experiments.racal_v1.boundary import evaluate_open
from protocol_v2.experiments.racal_v1.representation import choose_device, encode_rows
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from protocol_v2.runtime.paths import ProtocolV2Paths
from protocol_v2.tracking.run_manifest import atomic_run_directory


STAGE = "minilm_trainable_lambda_control_v1"
SOURCE_STAGE = "minilm_trainable_control_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87)
K_VALUES = (1, 2)
KIR = 0.50
DISTANCE = "mahalanobis_diag"
RADIUS_METHOD = "mean_std"
THRESHOLD = 1.0
PARTITION_SEED = 42
LAMBDA_VALUES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
KNOWN_CALIBRATION_FRR_LIMIT = 0.05


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


def _fit_detector(train_values: np.ndarray, train_rows: Sequence[Mapping[str, Any]], k: int) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=int(k),
        radius_method=RADIUS_METHOD,
        radius_lambda=1.0,
        distance_metric=DISTANCE,
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=PARTITION_SEED,
        acceptance_mode="nearest_sphere",
    )
    detector.fit(np.asarray(train_values), np.asarray([str(row["intent"]) for row in train_rows], dtype=object))
    return detector


def _load_source_checkpoint(
    paths: ProtocolV2Paths,
    config: Mapping[str, Any],
    dataset: str,
    seed: int,
    device: torch.device,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the immutable Trainable K=1 checkpoint for each dataset.

    CLINC150 and Banking77 were produced by the dedicated control stage.  The
    StackOverflow checkpoint predates that stage and lives in the RACAL
    stage-1 root, so it is loaded through the same verified loader rather than
    copying or retraining the model.
    """
    if dataset == "stackoverflow":
        model, tokenizer, metadata = _load_stage1_model(paths, seed, config, device)
        return model, tokenizer, {
            "manifest_path": str(metadata["manifest"].get("manifest_path", "")),
            "checkpoint_path": metadata["checkpoint"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "mode": metadata.get("mode"),
            "projection_hidden_dim": metadata.get("hidden_dim"),
            "source_stage": "racal_v1",
        }
    return _load_checkpoint(paths, config, dataset, seed, device)


def _calibration_metrics(detector: MultiSphereOOSDetector, values: np.ndarray) -> dict[str, float]:
    output = detector.predict_with_scores(np.asarray(values))
    false_reject_rate = float(np.mean(np.asarray(output["pred"], dtype=np.int64) == 1))
    return {"known_recall": 1.0 - false_reject_rate, "false_reject_rate": false_reject_rate}


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: _safe(row.get(field, "")) for field in fields} for row in rows)
    temporary.replace(path)


def _select_lambda(rows: Sequence[Mapping[str, Any]]) -> tuple[float, bool]:
    valid = [float(row["radius_lambda"]) for row in rows if float(row["calibration_false_reject_rate"]) <= KNOWN_CALIBRATION_FRR_LIMIT]
    if valid:
        return min(valid), True
    return max(LAMBDA_VALUES), False


def _run_unit(paths: ProtocolV2Paths, config: Mapping[str, Any], dataset: str, seed: int, resume: bool) -> dict[str, Any]:
    views, split = _load_views(paths, dataset, seed)
    device = choose_device(str(config.get("device", "auto")))
    model, tokenizer, checkpoint = _load_source_checkpoint(paths, config, dataset, seed, device)
    train_values = encode_rows(model, tokenizer, views.train, device, int(config["batch_size"]), int(config["max_length"]))
    calibration_values = encode_rows(model, tokenizer, views.calibration, device, int(config["batch_size"]), int(config["max_length"]))
    test_values = encode_rows(model, tokenizer, views.test, device, int(config["batch_size"]), int(config["max_length"]))
    detector_rows: dict[int, list[dict[str, Any]]] = {}
    selected: dict[int, float] = {}
    selection_valid: dict[int, bool] = {}
    selected_metrics: dict[int, dict[str, Any]] = {}
    for k in K_VALUES:
        detector = _fit_detector(train_values, views.train, k)
        rows: list[dict[str, Any]] = []
        for radius_lambda in LAMBDA_VALUES:
            detector.radius_lambda = float(radius_lambda)
            detector._compute_radii()
            calibration = _calibration_metrics(detector, calibration_values)
            test_metrics, _ = evaluate_open(detector, test_values, views.test, THRESHOLD)
            rows.append({
                "dataset": dataset,
                "kir": KIR,
                "seed": seed,
                "k": k,
                "radius_lambda": float(radius_lambda),
                "distance": DISTANCE,
                "radius_method": RADIUS_METHOD,
                "threshold": THRESHOLD,
                "calibration_known_recall": calibration["known_recall"],
                "calibration_false_reject_rate": calibration["false_reject_rate"],
                "test_used_for_selection": False,
                "oos_used_for_training": False,
                **{f"test_{key}": float(value) for key, value in test_metrics.items()},
            })
        chosen, valid = _select_lambda(rows)
        selected[k] = chosen
        selection_valid[k] = valid
        selected_metrics[k] = next(row for row in rows if np.isclose(row["radius_lambda"], chosen))
        for row in rows:
            row["known_only_selected_lambda"] = chosen
            row["known_only_selection_constraint_met"] = valid
        detector_rows[k] = rows
    flat_rows = [row for rows in detector_rows.values() for row in rows]
    config_payload = {
        "stage": STAGE,
        "protocol_version": "protocol_v2_textoir_v1",
        "dataset": dataset,
        "kir": KIR,
        "seed": seed,
        "representation": "trainable_minilm_last2_plus_projection_checkpoint_reused",
        "checkpoint": checkpoint,
        "k_values": list(K_VALUES),
        "lambda_values": list(LAMBDA_VALUES),
        "distance": DISTANCE,
        "radius_method": RADIUS_METHOD,
        "threshold": THRESHOLD,
        "calibration_frr_limit": KNOWN_CALIBRATION_FRR_LIMIT,
        "partition_seed": PARTITION_SEED,
        "input": split,
        "embedding_hashes": {"train": _array_hash(train_values), "calibration": _array_hash(calibration_values), "test": _array_hash(test_values)},
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    config_hash = sha256_json(config_payload)
    run_dir = _run_dir(paths, dataset, seed)
    existing_path = run_dir / "run_manifest.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if resume and existing.get("config_hash") == config_hash and existing.get("status") == "complete":
            return existing
        raise FileExistsError(f"Lambda run exists or has a different config: {run_dir}")
    started = time.time()
    run_manifest = {
        **config_payload,
        "config_hash": config_hash,
        "status": "complete",
        "selected_lambda": {str(k): selected[k] for k in K_VALUES},
        "selection_constraint_met": {str(k): selection_valid[k] for k in K_VALUES},
        "selected_test_metrics": {str(k): selected_metrics[k] for k in K_VALUES},
        "row_count": len(flat_rows),
    }
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_json(temporary / "resolved_config.json", _safe(config_payload))
        _atomic_csv(temporary / "lambda_metrics.csv", flat_rows)
        atomic_write_json(temporary / "run_manifest.json", _safe({**run_manifest, "elapsed_seconds": time.time() - started}))
    return {**run_manifest, "elapsed_seconds": time.time() - started, "run_dir": str(run_dir)}


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config must be a mapping: {path}")
    if str(payload.get("protocol_version")) != "protocol_v2_textoir_v1":
        raise ValueError("Only protocol_v2_textoir_v1 is supported")
    if tuple(str(value).lower() for value in payload.get("datasets", [])) != DATASETS:
        raise ValueError(f"This stage is restricted to datasets {DATASETS}")
    if tuple(int(value) for value in payload.get("seeds", [])) != SEEDS:
        raise ValueError(f"This stage is restricted to seeds {SEEDS}")
    if abs(float(payload.get("kir", -1)) - KIR) > 1e-12:
        raise ValueError("This stage is fixed at KIR=0.50")
    if bool(payload.get("test_used_for_selection", False)) or bool(payload.get("oos_used_for_training", False)):
        raise ValueError("Test/OOS use is forbidden")
    return {str(key): value for key, value in payload.items()}


def _provenance(paths: ProtocolV2Paths, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "s2c.minilm_trainable_lambda_control_v1.provenance.v1",
        "stage": STAGE,
        "source_stage": SOURCE_STAGE,
        "protocol_version": "protocol_v2_textoir_v1",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config": dict(config),
        "lambda_values": list(LAMBDA_VALUES),
        "selection_rule": f"minimum lambda with Known calibration false reject rate <= {KNOWN_CALIBRATION_FRR_LIMIT:.2f}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git": _git_state(paths.project_root),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "historical_artifacts_immutable": True,
    }


def run_stage(paths: ProtocolV2Paths, config_path: Path, config: Mapping[str, Any], resume: bool, dry_run: bool) -> dict[str, Any]:
    paths.require_experiment_admission()
    for dataset in DATASETS:
        paths.require_experiment_admission(dataset)
    stage = _stage_root(paths)
    stage.mkdir(parents=True, exist_ok=True)
    if not (stage / "PROVENANCE.json").is_file():
        atomic_write_json(stage / "PROVENANCE.json", _safe(_provenance(paths, config_path, config)))
    checks = []
    for dataset in DATASETS:
        for seed in SEEDS:
            views, split = _load_views(paths, dataset, seed)
            checks.append({"dataset": dataset, "seed": seed, "split": split, "train_count": len(views.train), "calibration_count": len(views.calibration), "test_count": len(views.test)})
    plan = {"stage": STAGE, "status": "preflight_ok", "datasets": list(DATASETS), "seeds": list(SEEDS), "k_values": list(K_VALUES), "lambda_values": list(LAMBDA_VALUES), "planned_units": len(DATASETS) * len(SEEDS) * len(K_VALUES) * len(LAMBDA_VALUES), "encoder_fits": 0, "checkpoint_selection": "known_calibration_only", "checks": checks, "test_used_for_selection": False, "oos_used_for_training": False}
    atomic_write_json(stage / "plans" / "stage_plan.json", _safe(plan))
    if dry_run:
        return plan
    results = [_run_unit(paths, config, dataset, seed, resume) for dataset in DATASETS for seed in SEEDS]
    state = {"stage": STAGE, "status": "complete", "planned_units": plan["planned_units"], "completed_units": plan["planned_units"], "failed_units": 0, "run_units": len(results), "test_used_for_selection": False, "oos_used_for_training": False}
    atomic_write_json(stage / "state.json", _safe(state))
    return {**state, "results": results, "root": str(stage)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Known-only lambda and K interaction control for Trainable MiniLM")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    paths = ProtocolV2Paths.discover()
    result = run_stage(paths, args.config.resolve(), config, args.resume, args.dry_run)
    print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
