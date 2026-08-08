"""Native OOS controls on the already trained MiniLM representation.

This stage is deliberately separate from ``native_baselines_v1``.  The latter
uses the frozen MiniLM cache; this stage reuses completed RACAL/trainable
checkpoints and changes only the detector family (MSP, Energy, kNN, LOF).
No encoder is trained here and no test label is used for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from protocol_v2.data.hashing import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file, sha256_json
from protocol_v2.data.manifests import read_json
from protocol_v2.experiments.external_baselines import (
    _breakdown,
    _known_only_threshold,
    _native_scores,
    _prediction_rows,
)
from protocol_v2.experiments.racal_v1.representation import RacalMiniLM, choose_device, encode_rows
from protocol_v2.gate.view_loader import GateViews, load_gate_views
from protocol_v2.runtime.paths import ProtocolV2Paths


STAGE = "native_baselines_trainable_v1"
METHODS = ("msp", "energy", "knn", "lof")
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)


def stage_root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / STAGE


def _model_path(paths: ProtocolV2Paths) -> Path:
    path = paths.project_root.parent / "assets" / "models" / "all-MiniLM-L6-v2"
    if not path.is_dir():
        raise FileNotFoundError(f"MiniLM model is unavailable: {path}")
    return path


def _checkpoint_path(paths: ProtocolV2Paths, dataset: str, kir: float, seed: int) -> Path:
    """Resolve parent/extension checkpoints without silently retraining."""

    kir_name = f"kir_{kir:.2f}"
    candidates = [
        paths.run_root / "minilm_trainable_kir_sweep_v1" / kir_name / "runs" / dataset / f"seed_{seed}" / "checkpoint.pt",
        paths.run_root / "minilm_trainable_kir_sweep_extension_v1" / kir_name / "runs" / dataset / f"seed_{seed}" / "checkpoint.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Completed Trainable MiniLM checkpoint is missing: {candidates}")


def _load_embeddings(
    paths: ProtocolV2Paths,
    views: GateViews,
    checkpoint: Path,
    device_name: str,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    model_path = _model_path(paths)
    manifest_path = checkpoint.parent / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest is missing next to checkpoint: {manifest_path}")
    training_manifest = read_json(manifest_path)
    hidden_dim = int(training_manifest.get("projection_hidden_dim", 256))
    requested = str(training_manifest.get("device", device_name))
    device = choose_device(device_name if device_name != "auto" else requested)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = RacalMiniLM(model_path, "last2_minilm_plus_projection", hidden_dim).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError(f"Trainable checkpoint has no model state: {checkpoint}")
    model.load_state_dict(state, strict=True)
    train = encode_rows(model, tokenizer, views.train, device, batch_size, max_length)
    calibration = encode_rows(model, tokenizer, views.calibration, device, batch_size, max_length)
    test = encode_rows(model, tokenizer, views.test, device, batch_size, max_length)
    checkpoint_hash = sha256_file(checkpoint)
    del tokenizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train, calibration, test, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "training_manifest_sha256": sha256_file(manifest_path),
        "device": str(device),
        "representation": "last2_minilm_plus_projection",
        "train_embedding_sha256": hashlib.sha256(np.ascontiguousarray(train).tobytes()).hexdigest(),
        "calibration_embedding_sha256": hashlib.sha256(np.ascontiguousarray(calibration).tobytes()).hexdigest(),
        "test_embedding_sha256": hashlib.sha256(np.ascontiguousarray(test).tobytes()).hexdigest(),
    }


def _run_dir(paths: ProtocolV2Paths, dataset: str, kir: float, seed: int, method: str) -> Path:
    return stage_root(paths) / "runs" / dataset / f"kir_{kir:.2f}" / f"seed_{seed}" / method


def _config(dataset: str, kir: float, seed: int, method: str) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "protocol_version": "protocol_v2_textoir_v1",
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "method": method,
        "representation": "trainable_minilm_last2_projection",
        "selection": "known_only_conformal_alpha_0.05",
        "uses_oos_for_training": False,
        "uses_oos_for_calibration": False,
        "test_used_for_selection": False,
    }


def run_combo(paths: ProtocolV2Paths, dataset: str, kir: float, seed: int, device: str, batch_size: int, max_length: int, resume: bool) -> dict[str, Any]:
    paths.require_experiment_admission(dataset)
    views = load_gate_views(paths, dataset, seed, kir)
    if any(int(row["label"]) != 0 for row in views.train + views.calibration):
        raise ValueError(f"Trainable native baseline violates Known-only train/calibration: {dataset}/{kir}/{seed}")
    checkpoint = _checkpoint_path(paths, dataset, kir, seed)
    train, calibration, test, embedding_info = _load_embeddings(paths, views, checkpoint, device, batch_size, max_length)
    train_labels = np.asarray([str(row["intent"]) for row in views.train], dtype=object)
    combo = {"dataset": dataset, "kir": kir, "seed": seed, "embedding": embedding_info}
    for method in METHODS:
        run_dir = _run_dir(paths, dataset, kir, seed, method)
        config = _config(dataset, kir, seed, method)
        config_hash = sha256_json(config)
        manifest_path = run_dir / "manifest.json"
        if resume and manifest_path.is_file():
            existing = read_json(manifest_path)
            if existing.get("config_hash") != config_hash:
                raise RuntimeError(f"Existing run has a different config: {run_dir}")
            combo.setdefault("statuses", {})[method] = existing.get("status", "unknown")
            continue
        if run_dir.exists():
            raise RuntimeError(f"Refusing to overwrite incomplete run: {run_dir}")
        started = time.perf_counter()
        calibration_scores, test_scores, _, test_nearest, method_details = _native_scores(
            type("Spec", (), {"method": method})(), train, calibration, test, train_labels
        )
        selection = _known_only_threshold(calibration_scores)
        metrics = _breakdown(views.test, test_scores, selection.threshold)
        run_id = f"{STAGE}__{dataset}__kir_{kir:.2f}__seed_{seed}__repr_trainable__baseline_{method}"
        config_payload = {**config, "run_id": run_id, "config_hash": config_hash}
        prediction_rows = _prediction_rows(views.test, test_scores, selection.threshold, test_nearest)
        with __import__("protocol_v2.tracking.run_manifest", fromlist=["atomic_run_directory"]).atomic_run_directory(run_dir) as temporary:
            atomic_write_text(temporary / "resolved_config.yaml", yaml.safe_dump(config_payload, sort_keys=True))
            atomic_write_json(temporary / "metrics.json", metrics)
            atomic_write_json(temporary / "threshold_selection.json", {
                "type": "known_only_conformal", "alpha": selection.alpha, "threshold": selection.threshold,
                "known_calibration_count": selection.calibration_count,
                "order_statistic_rank": selection.order_statistic_rank,
                "test_used_for_selection": False,
            })
            atomic_write_jsonl(temporary / "predictions" / "test.jsonl", prediction_rows)
            atomic_write_json(temporary / "manifest.json", {
                **config_payload,
                "status": "complete",
                "metrics_emitted": True,
                "method_details": method_details,
                "embedding_info": embedding_info,
                "elapsed_seconds": time.perf_counter() - started,
            })
        combo.setdefault("statuses", {})[method] = "complete"
    return combo


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def run_matrix(paths: ProtocolV2Paths, config: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    datasets = tuple(str(x) for x in config.get("datasets", DATASETS))
    kirs = tuple(float(x) for x in config.get("kirs", KIRS))
    seeds = tuple(int(x) for x in config.get("seeds", SEEDS))
    device = str(config.get("device", "auto"))
    batch_size = int(config.get("batch_size", 256))
    max_length = int(config.get("max_length", 256))
    summary: dict[str, Any] = {"stage": STAGE, "planned_combos": len(datasets) * len(kirs) * len(seeds), "planned_runs": len(datasets) * len(kirs) * len(seeds) * len(METHODS), "completed": [], "failed": []}
    for dataset in datasets:
        for kir in kirs:
            for seed in seeds:
                try:
                    summary["completed"].append(run_combo(paths, dataset, kir, seed, device, batch_size, max_length, resume))
                except Exception as exc:
                    summary["failed"].append({"dataset": dataset, "kir": kir, "seed": seed, "error_type": type(exc).__name__, "error": str(exc)})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        raise SystemExit("Refusing to run without explicit --execute")
    paths = ProtocolV2Paths.discover()
    summary = run_matrix(paths, load_config(args.config), resume=args.resume)
    summary_path = stage_root(paths) / "matrix_summary.json"
    atomic_write_json(summary_path, summary)
    print(json.dumps({"planned_runs": summary["planned_runs"], "completed_combos": len(summary["completed"]), "failed_combos": len(summary["failed"]), "summary": str(summary_path)}, ensure_ascii=False, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
