"""Cross-dataset Known-only Trainable MiniLM control.

This stage intentionally evaluates only K=1.  It reuses the RACAL training
objective and the protocol_v2 Gate evaluator, but writes to a new artifact
root so the original StackOverflow RACAL-v1 runs remain immutable.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from protocol_v2.data.hashing import atomic_write_json, atomic_write_jsonl, sha256_file, sha256_json
from protocol_v2.data.manifests import read_json
from protocol_v2.experiments.mechanism_runner import E3Bundle
from protocol_v2.gate.view_loader import load_gate_views
from protocol_v2.experiments.racal_v1.boundary import detector_signature, evaluate_open, fit_k1_detector
from protocol_v2.experiments.racal_v1.representation import RacalMiniLM, choose_device, encode_rows, set_seed
from protocol_v2.experiments.racal_v1.runner import _center_losses, _class_centers, _make_optimizer, _safe, _set_phase, _write_checkpoint
from protocol_v2.runtime.paths import ProtocolV2Paths


STAGE = "minilm_trainable_control_v1"
ALLOWED_DATASETS = ("clinc150", "banking77", "stackoverflow")
DEFAULT_SEEDS = (13, 42, 87)


@dataclass(frozen=True)
class ControlConfig:
    protocol_version: str
    model_path: str
    datasets: tuple[str, ...]
    kir: float
    seeds: tuple[int, ...]
    warmup_epochs: int
    finetune_epochs: int
    patience: int
    batch_size: int
    max_length: int
    projection_hidden_dim: int
    projection_lr: float
    backbone_lr: float
    temperature: float
    intra_weight: float
    inter_weight: float
    classification_weight: float
    inter_margin: float
    threshold: float
    device: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ControlConfig":
        required = ("protocol_version", "model_path", "datasets", "kir", "seeds")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        datasets = tuple(dict.fromkeys(str(value).lower() for value in payload["datasets"]))
        unknown = sorted(set(datasets) - set(ALLOWED_DATASETS))
        if unknown:
            raise ValueError(f"Unsupported datasets: {unknown}")
        kir = float(payload["kir"])
        if abs(kir - 0.50) > 1e-12:
            raise ValueError("This control is pre-registered for KIR=0.50 only")
        seeds = tuple(int(value) for value in payload["seeds"])
        if not seeds or any(value not in DEFAULT_SEEDS for value in seeds):
            raise ValueError(f"Seeds must be drawn from {DEFAULT_SEEDS}: {seeds}")
        return cls(
            protocol_version=str(payload["protocol_version"]),
            model_path=str(payload["model_path"]),
            datasets=datasets,
            kir=kir,
            seeds=seeds,
            warmup_epochs=int(payload.get("warmup_epochs", 1)),
            finetune_epochs=int(payload.get("finetune_epochs", 3)),
            patience=int(payload.get("patience", 1)),
            batch_size=int(payload.get("batch_size", 64)),
            max_length=int(payload.get("max_length", 256)),
            projection_hidden_dim=int(payload.get("projection_hidden_dim", 256)),
            projection_lr=float(payload.get("projection_lr", 2e-4)),
            backbone_lr=float(payload.get("backbone_lr", 2e-5)),
            temperature=float(payload.get("temperature", 0.07)),
            intra_weight=float(payload.get("intra_weight", 0.1)),
            inter_weight=float(payload.get("inter_weight", 0.1)),
            classification_weight=float(payload.get("classification_weight", 1.0)),
            inter_margin=float(payload.get("inter_margin", 0.20)),
            threshold=float(payload.get("threshold", 1.0)),
            device=str(payload.get("device", "auto")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": STAGE,
            "protocol_version": self.protocol_version,
            "model_path": self.model_path,
            "datasets": list(self.datasets),
            "kir": self.kir,
            "seeds": list(self.seeds),
            "warmup_epochs": self.warmup_epochs,
            "finetune_epochs": self.finetune_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "projection_hidden_dim": self.projection_hidden_dim,
            "projection_lr": self.projection_lr,
            "backbone_lr": self.backbone_lr,
            "temperature": self.temperature,
            "intra_weight": self.intra_weight,
            "inter_weight": self.inter_weight,
            "classification_weight": self.classification_weight,
            "inter_margin": self.inter_margin,
            "threshold": self.threshold,
            "device": self.device,
            "representation_mode": "last2_minilm_plus_projection",
            "selection": "known_calibration_only",
            "test_used_for_selection": False,
            "oos_used_for_training": False,
            "k_gate": 1,
            "distance": "mahalanobis_diag",
            "boundary": "mean_std",
        }


def load_config(path: Path) -> ControlConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config = ControlConfig.from_mapping(payload)
    if config.protocol_version != "protocol_v2_textoir_v1":
        raise ValueError(f"Unsupported protocol: {config.protocol_version}")
    return config


def root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / STAGE


def model_path(paths: ProtocolV2Paths, config: ControlConfig) -> Path:
    return (paths.project_root / config.model_path).resolve()


def rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["sample_id"]) for row in rows).encode("utf-8")).hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def model_hashes(path: Path) -> dict[str, str]:
    names = ("config.json", "tokenizer.json", "model.safetensors", "pytorch_model.bin")
    hashes = {name: sha256_file(path / name) for name in names if (path / name).is_file()}
    if "config.json" not in hashes or not ({"model.safetensors", "pytorch_model.bin"} & hashes.keys()):
        raise FileNotFoundError(f"Incomplete local MiniLM model: {path}")
    return hashes


def git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=project_root, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    return {"base_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--short")), "status": run("status", "--short")}


def validate_bundle(bundle: Any) -> dict[str, Any]:
    views = bundle.views
    train, calibration, test = views.train, views.calibration, views.test
    ids = {name: {str(row["sample_id"]) for row in rows} for name, rows in (("train", train), ("calibration", calibration), ("test", test))}
    overlaps = {"train_calibration": ids["train"] & ids["calibration"], "train_test": ids["train"] & ids["test"], "calibration_test": ids["calibration"] & ids["test"]}
    if any(overlaps.values()):
        raise ValueError(f"Split overlap for {bundle.dataset}/{bundle.seed}: {overlaps}")
    if any(int(row["label"]) != 0 for row in train + calibration):
        raise ValueError("Train and calibration must contain Known rows only")
    if len(bundle.train) != len(train) or len(bundle.calibration) != len(calibration) or len(bundle.test) != len(test):
        raise ValueError("Embedding/view count mismatch")
    return {
        "dataset": bundle.dataset,
        "seed": bundle.seed,
        "kir": bundle.kir,
        "train_count": len(train),
        "calibration_count": len(calibration),
        "test_count": len(test),
        "test_known_count": int(sum(int(row["label"]) == 0 for row in test)),
        "test_oos_count": int(sum(int(row["label"]) == 1 for row in test)),
        "train_sample_ids_sha256": rows_hash(train),
        "calibration_sample_ids_sha256": rows_hash(calibration),
        "test_sample_ids_sha256": rows_hash(test),
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }


def _atomic_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", delete=False) as handle:
        tmp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _safe(row.get(field, "")) for field in fields})
    tmp.replace(path)


def _run_dir(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    return root(paths) / "runs" / dataset / f"seed_{seed}"


def _load_control_bundle(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> E3Bundle:
    """Load views and the frozen E2 manifest without reading the large E2 arrays.

    Trainable controls re-encode the text, so the frozen E2 embedding cache is
    not an input to this stage.  We still require the exact E2 run manifest and
    current Gate views to bind the split/provenance contract.
    """
    name = (
        f"protocol_v2_textoir_v1__{dataset}__kir_{kir:.2f}__seed_{seed}__"
        "repr_frozen_minilm__k_1__dist_euclidean__boundary_mean_std"
    )
    e2_dir = paths.run_root / "e2_gate_core_dense" / name
    manifest_path = e2_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"E2 reference manifest is missing: {manifest_path}")
    e2_manifest = read_json(manifest_path)
    if e2_manifest.get("protocol_version") != paths.dataset_version:
        raise ValueError(f"E2 protocol mismatch: {manifest_path}")
    views = load_gate_views(paths, dataset, seed, kir)
    # Placeholder arrays preserve E3Bundle's shape contract; no frozen cache
    # values are consumed by this trainable representation stage.
    train = np.empty((len(views.train), 1), dtype=np.float32)
    calibration = np.empty((len(views.calibration), 1), dtype=np.float32)
    test = np.empty((len(views.test), 1), dtype=np.float32)
    return E3Bundle(dataset, seed, kir, train, calibration, test, views, e2_manifest)


def _bundle_payload(bundle: Any, validation: Mapping[str, Any]) -> dict[str, Any]:
    manifest = bundle.e2_manifest
    return {
        "dataset": bundle.dataset,
        "kir": bundle.kir,
        "seed": bundle.seed,
        "registry_sha256": manifest.get("registry_sha256"),
        "canonical_manifest_sha256": manifest.get("canonical_manifest_sha256"),
        "input_hashes": manifest.get("input_hashes", {}),
        "split_validation": dict(validation),
        "representation_input": "raw_protocol_view_text; frozen E2 arrays not read",
    }


def _checkpoint_score(model: RacalMiniLM, tokenizer: Any, bundle: Any, device: torch.device, config: ControlConfig) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    train_values = encode_rows(model, tokenizer, bundle.views.train, device, config.batch_size, config.max_length)
    calibration_values = encode_rows(model, tokenizer, bundle.views.calibration, device, config.batch_size, config.max_length)
    detector = fit_k1_detector(train_values, bundle.views.train, "mahalanobis_diag")
    metrics = evaluate_open(detector, calibration_values, bundle.views.calibration, config.threshold)[0]
    score = float(metrics["f1_k"] + 0.05 * metrics["known_recall"])
    return score, {key: float(value) for key, value in metrics.items() if isinstance(value, (float, int))}, train_values, calibration_values


def train_unit(paths: ProtocolV2Paths, config: ControlConfig, dataset: str, seed: int, resume: bool) -> dict[str, Any]:
    bundle = _load_control_bundle(paths, dataset, seed, config.kir)
    split_validation = validate_bundle(bundle)
    model_dir = model_path(paths, config)
    run_dir = _run_dir(paths, dataset, seed)
    payload = {**config.as_dict(), "dataset": dataset, "seed": seed, "input": _bundle_payload(bundle, split_validation), "model_files": model_hashes(model_dir)}
    config_hash = sha256_json(payload)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"Run exists; use --resume: {run_dir}")
        if existing.get("config_hash") != config_hash or existing.get("status") != "complete":
            raise RuntimeError(f"Refusing incompatible/incomplete run: {run_dir}")
        return existing

    started = time.time()
    set_seed(seed)
    device = choose_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = RacalMiniLM(model_dir, "last2_minilm_plus_projection", config.projection_hidden_dim).to(device)
    train_rows, calibration_rows = bundle.views.train, bundle.views.calibration
    label_names = sorted({str(row["intent"]) for row in train_rows})
    label_map = {name: index for index, name in enumerate(label_names)}
    targets = torch.as_tensor([label_map[str(row["intent"])] for row in train_rows], dtype=torch.long, device=device)
    indices = np.arange(len(train_rows), dtype=np.int64)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale = 0
    epoch_number = 0
    for phase, epochs in (("warmup", config.warmup_epochs), ("finetune", config.finetune_epochs)):
        _set_phase(model, phase)
        optimizer = _make_optimizer(model, config, phase)
        for _ in range(epochs):
            epoch_number += 1
            model.train()
            with torch.no_grad():
                refreshed = torch.as_tensor(encode_rows(model, tokenizer, train_rows, device, config.batch_size, config.max_length), device=device)
                centers_np, _, _ = _class_centers(refreshed.detach().cpu().numpy(), train_rows)
            centers = torch.as_tensor(centers_np, dtype=torch.float32, device=device)
            model.train()
            order = np.random.default_rng(seed + epoch_number * 7919).permutation(indices)
            losses: list[dict[str, float]] = []
            for start in range(0, len(order), config.batch_size):
                batch_indices = order[start : start + config.batch_size]
                batch_rows = [train_rows[int(index)] for index in batch_indices]
                tokens = tokenizer([str(row["text"]) for row in batch_rows], padding=True, truncation=True, max_length=config.max_length, return_tensors="pt").to(device)
                features = model(tokens)
                total, parts = _center_losses(features, targets[torch.as_tensor(batch_indices, dtype=torch.long, device=device)], centers, config.temperature, config.intra_weight, config.inter_weight, config.classification_weight, config.inter_margin)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(parts)
            score, cal_metrics, _, _ = _checkpoint_score(model, tokenizer, bundle, device, config)
            record = {"epoch": epoch_number, "phase": phase, "loss": float(np.mean([item["total"] for item in losses])), "classification_ce": float(np.mean([item["classification_ce"] for item in losses])), "intra_compactness": float(np.mean([item["intra_compactness"] for item in losses])), "inter_margin": float(np.mean([item["inter_margin"] for item in losses])), "selection_score": score, "calibration_f1_k": cal_metrics.get("f1_k"), "calibration_known_recall": cal_metrics.get("known_recall")}
            history.append(record)
            if score > best_score + 1e-12:
                best_score, best_epoch, stale = score, epoch_number, 0
                best_state = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "mode": model.mode, "label_map": label_map, "epoch": epoch_number, "freeze_report": model.freeze_report()}
            else:
                stale += 1
            if phase == "finetune" and stale > config.patience:
                break
        del optimizer
        if phase == "finetune" and stale > config.patience:
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for {dataset}/{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"
    _write_checkpoint(checkpoint, best_state)
    _atomic_tsv(run_dir / "training_history.tsv", history)
    model.load_state_dict(best_state["model"])
    model.eval()
    final_train = encode_rows(model, tokenizer, train_rows, device, config.batch_size, config.max_length)
    final_calibration = encode_rows(model, tokenizer, calibration_rows, device, config.batch_size, config.max_length)
    final_test = encode_rows(model, tokenizer, bundle.views.test, device, config.batch_size, config.max_length)
    detector = fit_k1_detector(final_train, train_rows, "mahalanobis_diag")
    metrics, predictions = evaluate_open(detector, final_test, bundle.views.test, config.threshold)
    checkpoint_hash = sha256_file(checkpoint)
    metrics.update({"stage": STAGE, "method": "trainable_k1", "dataset": dataset, "kir": config.kir, "seed": seed, "representation_mode": model.mode, "trainable_parameters": best_state["freeze_report"]["trainable_parameter_count"], "training_time_seconds": time.time() - started, "checkpoint_sha256": checkpoint_hash, "test_used_for_selection": False, "oos_used_for_training": False, "input_sample_order_hash": rows_hash(bundle.views.test), "train_embedding_sha256": array_hash(final_train), "calibration_embedding_sha256": array_hash(final_calibration), "test_embedding_sha256": array_hash(final_test), "split_validation": split_validation})
    training_manifest = {**payload, "config_hash": config_hash, "status": "complete", "device": str(device), "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash, "best_epoch": best_epoch, "selection_metric": "calibration_f1_k_plus_0.05_known_recall", "best_selection_score": best_score, "trainable_parameters": best_state["freeze_report"]["trainable_parameter_count"], "freeze_report": best_state["freeze_report"], "test_used_for_selection": False, "oos_used_for_training": False, "elapsed_seconds": time.time() - started}
    atomic_write_json(run_dir / "training_manifest.json", _safe(training_manifest))
    atomic_write_json(run_dir / "metrics.json", _safe(metrics))
    atomic_write_jsonl(run_dir / "predictions.jsonl", _safe(predictions))
    atomic_write_json(run_dir / "detector_signature.json", _safe(detector_signature(detector)))
    atomic_write_json(run_dir / "run_manifest.json", _safe({**training_manifest, "metrics": metrics, "run_dir": str(run_dir)}))
    return {**training_manifest, "metrics": metrics, "run_dir": str(run_dir)}


def make_provenance(paths: ProtocolV2Paths, config_path: Path, config: ControlConfig) -> dict[str, Any]:
    model_dir = model_path(paths, config)
    return {"schema_version": "s2c.minilm_trainable_control_v1.provenance.v1", "stage": STAGE, "protocol_version": config.protocol_version, "config_path": str(config_path), "config_sha256": sha256_file(config_path), "config_hash": sha256_json(config.as_dict()), "model_path": str(model_dir), "model_files": model_hashes(model_dir), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "git": git_state(paths.project_root), "datasets": list(config.datasets), "kir": config.kir, "seeds": list(config.seeds), "test_used_for_selection": False, "oos_used_for_training": False, "historical_artifacts_immutable": True}


def preflight(paths: ProtocolV2Paths, config_path: Path, config: ControlConfig) -> dict[str, Any]:
    paths.require_experiment_admission()
    checks = []
    for dataset in config.datasets:
        paths.require_experiment_admission(dataset)
        for seed in config.seeds:
            bundle = _load_control_bundle(paths, dataset, seed, config.kir)
            checks.append(validate_bundle(bundle))
    return {"status": "preflight_ok", "config": config.as_dict(), "model_path": str(model_path(paths, config)), "model_files": model_hashes(model_path(paths, config)), "checks": checks, "planned_units": len(config.datasets) * len(config.seeds), "test_used_for_selection": False, "oos_used_for_training": False}


def run_stage(paths: ProtocolV2Paths, config_path: Path, config: ControlConfig, datasets: Sequence[str], seeds: Sequence[int], resume: bool, dry_run: bool) -> dict[str, Any]:
    stage = root(paths)
    stage.mkdir(parents=True, exist_ok=True)
    provenance_path = stage / "PROVENANCE.json"
    if not provenance_path.is_file():
        atomic_write_json(provenance_path, _safe(make_provenance(paths, config_path, config)))
    if dry_run:
        plan = preflight(paths, config_path, config)
        atomic_write_json(stage / "plans" / "stage_plan.json", _safe(plan))
        return plan
    chosen_datasets = tuple(dict.fromkeys(str(value).lower() for value in datasets))
    chosen_seeds = tuple(int(value) for value in seeds)
    if any(value not in config.datasets for value in chosen_datasets) or any(value not in config.seeds for value in chosen_seeds):
        raise ValueError("Requested datasets/seeds must be declared in the configuration")
    results = [train_unit(paths, config, dataset, seed, resume) for dataset in chosen_datasets for seed in chosen_seeds]
    atomic_write_json(stage / "state.json", _safe({"stage": STAGE, "datasets": chosen_datasets, "seeds": chosen_seeds, "completed": len(results), "test_used_for_selection": False, "oos_used_for_training": False}))
    return {"status": "complete", "stage": STAGE, "completed": len(results), "results": results, "root": str(stage)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the cross-dataset Known-only Trainable MiniLM K=1 control")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    config = load_config(args.config)
    datasets = args.datasets or config.datasets
    seeds = args.seeds or config.seeds
    result = run_stage(paths, args.config.resolve(), config, datasets, seeds, args.resume, args.dry_run)
    print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
