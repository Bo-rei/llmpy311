"""RACAL-v1 first-stage runner.

Only two modes are intentionally implemented here:

* ``frozen_k1``: replay the immutable E2 K=1 contract;
* ``trainable_k1``: train the residual-adapted MiniLM and evaluate one-centre
  boundaries using the same Gate implementation.

Centre activation, fixed K=2, proxy-OOS and parent guards are deliberately not
part of this stage.  Keeping those out of the runner makes the first result a
clean representation-control experiment rather than a mixed-method pilot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from protocol_v2.data.hashing import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file
from protocol_v2.experiments.mechanism_runner import load_e2_bundle
from protocol_v2.runtime.paths import ProtocolV2Paths

from .boundary import compare_reference, detector_signature, evaluate_open, fit_k1_detector
from .contracts import (
    DATASET,
    KIR,
    STAGE,
    RacalConfig,
    array_hash,
    load_config,
    model_file_hashes,
    provenance_payload,
    reference_run_name,
    validate_bundle,
    rows_hash,
)
from .representation import RacalMiniLM, choose_device, encode_rows, set_seed


def stage_root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / STAGE


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


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "\n")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _safe(row.get(field, "")) for field in fields})
    temporary.replace(path)


def _load_or_fail_config(config_path: Path) -> RacalConfig:
    config = load_config(config_path)
    if config.protocol_version != "protocol_v2_textoir_v1":
        raise ValueError(f"RACAL only supports protocol_v2_textoir_v1: {config.protocol_version}")
    return config


def _model_path(paths: ProtocolV2Paths, config: RacalConfig) -> Path:
    return (paths.project_root / config.model_path).resolve()


def _reference_dir(paths: ProtocolV2Paths, seed: int) -> Path:
    return paths.run_root / "e2_gate_core_dense" / reference_run_name(seed)


def _input_payload(bundle: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    manifest = bundle.e2_manifest
    return {
        "protocol_version": manifest.get("protocol_version"),
        "dataset": bundle.dataset,
        "kir": bundle.kir,
        "seed": bundle.seed,
        "registry_sha256": manifest.get("registry_sha256"),
        "canonical_manifest_sha256": manifest.get("canonical_manifest_sha256"),
        "input_hashes": manifest.get("input_hashes", {}),
        "view_contract": contract,
        "train_embedding_sha256": array_hash(bundle.train),
        "calibration_embedding_sha256": array_hash(bundle.calibration),
        "test_embedding_sha256": array_hash(bundle.test),
        "train_sample_ids_sha256": rows_hash(bundle.views.train),
        "calibration_sample_ids_sha256": rows_hash(bundle.views.calibration),
        "test_sample_ids_sha256": rows_hash(bundle.views.test),
    }


def _ensure_run_reusable(run_dir: Path, config_hash: str, resume: bool) -> dict[str, Any] | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not resume:
        raise FileExistsError(f"RACAL run already exists; use --resume: {run_dir}")
    if existing.get("config_hash") != config_hash:
        raise RuntimeError(f"Refusing to reuse RACAL run with a different config: {run_dir}")
    if existing.get("status") != "complete":
        raise RuntimeError(f"Existing RACAL run is not complete: {run_dir}")
    return existing


def run_frozen_k1(paths: ProtocolV2Paths, config: RacalConfig, seed: int, resume: bool) -> dict[str, Any]:
    bundle = load_e2_bundle(paths, DATASET, seed, KIR)
    contract = validate_bundle(bundle)
    run_dir = stage_root(paths) / "runs" / "frozen_k1" / f"seed_{seed}"
    config_payload = {"stage": STAGE, "method": "frozen_k1", "seed": seed, **config.as_dict(), "input": _input_payload(bundle, contract)}
    config_hash = str(__import__("protocol_v2.data.hashing", fromlist=["sha256_json"]).sha256_json(config_payload))
    reusable = _ensure_run_reusable(run_dir, config_hash, resume)
    if reusable is not None:
        return reusable
    started = time.time()
    detector = fit_k1_detector(bundle.train, bundle.views.train, "mahalanobis_diag")
    metrics, predictions = evaluate_open(detector, bundle.test, bundle.views.test, config.threshold)
    comparison = compare_reference(_reference_dir(paths, seed), metrics, predictions)
    if not comparison["within_tolerance"]:
        raise RuntimeError(f"E2 K=1 exact replay failed for seed {seed}: {comparison}")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(run_dir / "predictions.jsonl", predictions)
    atomic_write_json(run_dir / "detector_signature.json", _safe(detector_signature(detector)))
    atomic_write_json(run_dir / "metrics.json", _safe(metrics))
    atomic_write_json(run_dir / "e2_replay.json", _safe(comparison))
    manifest = {
        **config_payload,
        "config_hash": config_hash,
        "status": "complete",
        "method": "frozen_k1",
        "reference_run": str(_reference_dir(paths, seed)),
        "e2_replay": comparison,
        "test_used_for_selection": False,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(run_dir / "run_manifest.json", _safe(manifest))
    return manifest


def _class_centers(values: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    names = sorted({str(row["intent"]) for row in rows})
    label_map = {name: index for index, name in enumerate(names)}
    labels = np.asarray([label_map[str(row["intent"])] for row in rows], dtype=np.int64)
    centers = np.asarray([values[labels == index].mean(axis=0) for index in range(len(names))], dtype=np.float32)
    centers /= np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    return centers, label_map, labels


def _center_losses(
    features: torch.Tensor,
    targets: torch.Tensor,
    centers: torch.Tensor,
    temperature: float,
    intra_weight: float,
    inter_weight: float,
    classification_weight: float,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    distances = torch.cdist(features, centers, p=2).pow(2)
    own = distances.gather(1, targets[:, None]).squeeze(1)
    masked = distances.clone()
    masked.scatter_(1, targets[:, None], float("inf"))
    other = masked.min(dim=1).values
    logits = -distances / max(float(temperature), 1e-6)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    intra = own.mean()
    inter = torch.relu(own - other + float(margin)).mean()
    total = classification_weight * ce + intra_weight * intra + inter_weight * inter
    return total, {"classification_ce": float(ce.detach().cpu()), "intra_compactness": float(intra.detach().cpu()), "inter_margin": float(inter.detach().cpu()), "total": float(total.detach().cpu())}


def _make_optimizer(model: RacalMiniLM, config: RacalConfig, phase: str) -> torch.optim.Optimizer:
    if phase == "warmup":
        params = [parameter for parameter in model.projection.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(params, lr=config.projection_lr)
    projection = [parameter for parameter in model.projection.parameters() if parameter.requires_grad]
    backbone = [parameter for name, parameter in model.named_parameters() if name.startswith("encoder.") and parameter.requires_grad]
    return torch.optim.AdamW([{"params": projection, "lr": config.projection_lr}, {"params": backbone, "lr": config.backbone_lr}])


def _set_phase(model: RacalMiniLM, phase: str) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.projection.parameters():
        parameter.requires_grad_(True)
    if phase == "finetune":
        layers = model.encoder.encoder.layer
        for block in layers[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)


def _select_checkpoint_score(model: RacalMiniLM, tokenizer: Any, bundle: Any, device: torch.device, config: RacalConfig) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    train_values = encode_rows(model, tokenizer, bundle.views.train, device, config.batch_size, config.max_length)
    calibration_values = encode_rows(model, tokenizer, bundle.views.calibration, device, config.batch_size, config.max_length)
    detector = fit_k1_detector(train_values, bundle.views.train, "mahalanobis_diag")
    known_metrics = evaluate_open(detector, calibration_values, bundle.views.calibration, config.threshold)[0]
    score = float(known_metrics["f1_k"] + 0.05 * known_metrics["known_recall"])
    return score, {key: float(value) for key, value in known_metrics.items() if isinstance(value, (float, int))}, train_values, calibration_values


def train_trainable_k1(paths: ProtocolV2Paths, config: RacalConfig, seed: int, resume: bool) -> dict[str, Any]:
    bundle = load_e2_bundle(paths, DATASET, seed, KIR)
    contract = validate_bundle(bundle)
    model_path = _model_path(paths, config)
    run_dir = stage_root(paths) / "runs" / "trainable_k1" / f"seed_{seed}"
    config_payload = {"stage": STAGE, "method": "trainable_k1", "seed": seed, **config.as_dict(), "input": _input_payload(bundle, contract), "model_files": model_file_hashes(model_path)}
    from protocol_v2.data.hashing import sha256_json
    config_hash = sha256_json(config_payload)
    reusable = _ensure_run_reusable(run_dir, config_hash, resume)
    if reusable is not None:
        return reusable
    started = time.time()
    set_seed(seed)
    device = choose_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = RacalMiniLM(model_path, "last2_minilm_plus_projection", config.projection_hidden_dim).to(device)
    train_rows, calibration_rows = bundle.views.train, bundle.views.calibration
    label_names = sorted({str(row["intent"]) for row in train_rows})
    label_map = {name: index for index, name in enumerate(label_names)}
    train_targets = torch.as_tensor([label_map[str(row["intent"])] for row in train_rows], dtype=torch.long, device=device)
    train_indices = np.arange(len(train_rows), dtype=np.int64)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    phases = [("warmup", config.warmup_epochs), ("finetune", config.finetune_epochs)]
    epoch_number = 0
    stale = 0
    for phase, epochs in phases:
        _set_phase(model, phase)
        optimizer = _make_optimizer(model, config, phase)
        for _ in range(epochs):
            epoch_number += 1
            model.train()
            with torch.no_grad():
                epoch_values = torch.as_tensor(encode_rows(model, tokenizer, train_rows, device, config.batch_size, config.max_length), device=device)
                centers_np, _, _ = _class_centers(epoch_values.detach().cpu().numpy(), train_rows)
            centers = torch.as_tensor(centers_np, dtype=torch.float32, device=device)
            # ``encode_rows`` switches the wrapper to eval mode for the full
            # centre refresh.  Restore train mode before computing gradients.
            model.train()
            order = np.random.default_rng(seed + epoch_number * 7919).permutation(train_indices)
            losses: list[dict[str, float]] = []
            for start in range(0, len(order), config.batch_size):
                batch_indices = order[start : start + config.batch_size]
                batch_rows = [train_rows[int(index)] for index in batch_indices]
                tokens = tokenizer([str(row["text"]) for row in batch_rows], padding=True, truncation=True, max_length=config.max_length, return_tensors="pt").to(device)
                target = train_targets[torch.as_tensor(batch_indices, dtype=torch.long, device=device)]
                features = model(tokens)
                total, parts = _center_losses(features, target, centers, config.temperature, config.intra_weight, config.inter_weight, config.classification_weight, config.inter_margin)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(parts)
            score, cal_metrics, _, _ = _select_checkpoint_score(model, tokenizer, bundle, device, config)
            row = {"epoch": epoch_number, "phase": phase, "loss": float(np.mean([item["total"] for item in losses])), "classification_ce": float(np.mean([item["classification_ce"] for item in losses])), "intra_compactness": float(np.mean([item["intra_compactness"] for item in losses])), "inter_margin": float(np.mean([item["inter_margin"] for item in losses])), "selection_score": score, "calibration_f1_k": cal_metrics.get("f1_k"), "calibration_known_recall": cal_metrics.get("known_recall")}
            history.append(row)
            if score > best_score + 1e-12:
                best_score = score
                best_epoch = epoch_number
                stale = 0
                best_state = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "mode": model.mode, "label_map": label_map, "epoch": epoch_number, "freeze_report": model.freeze_report()}
            else:
                stale += 1
            if phase == "finetune" and stale > config.patience:
                break
        del optimizer
        if phase == "finetune" and stale > config.patience:
            break
    if best_state is None:
        raise RuntimeError("RACAL trainable K=1 produced no checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"
    _write_checkpoint(checkpoint, best_state)
    _atomic_tsv(run_dir / "training_history.tsv", history)
    manifest = {**config_payload, "config_hash": config_hash, "status": "complete", "device": str(device), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "best_epoch": best_epoch, "selection_metric": "calibration_f1_k_plus_0.05_known_recall", "best_selection_score": best_score, "trainable_parameters": best_state["freeze_report"]["trainable_parameter_count"], "freeze_report": best_state["freeze_report"], "test_used_for_selection": False, "oos_used_for_training": False, "elapsed_seconds": time.time() - started}
    atomic_write_json(run_dir / "training_manifest.json", _safe(manifest))
    model.load_state_dict(best_state["model"])
    model.eval()
    final_train = encode_rows(model, tokenizer, train_rows, device, config.batch_size, config.max_length)
    final_calibration = encode_rows(model, tokenizer, calibration_rows, device, config.batch_size, config.max_length)
    final_test = encode_rows(model, tokenizer, bundle.views.test, device, config.batch_size, config.max_length)
    detector = fit_k1_detector(final_train, train_rows, "mahalanobis_diag")
    metrics, predictions = evaluate_open(detector, final_test, bundle.views.test, config.threshold)
    metrics.update({"stage": STAGE, "method": "trainable_k1", "dataset": DATASET, "kir": KIR, "seed": seed, "representation_mode": model.mode, "trainable_parameters": best_state["freeze_report"]["trainable_parameter_count"], "training_time_seconds": time.time() - started, "checkpoint_sha256": manifest["checkpoint_sha256"], "test_used_for_selection": False, "input_sample_order_hash": rows_hash(bundle.views.test), "train_embedding_sha256": array_hash(final_train), "calibration_embedding_sha256": array_hash(final_calibration), "test_embedding_sha256": array_hash(final_test)})
    atomic_write_json(run_dir / "metrics.json", _safe(metrics))
    atomic_write_jsonl(run_dir / "predictions.jsonl", predictions)
    atomic_write_json(run_dir / "detector_signature.json", _safe(detector_signature(detector)))
    return {**manifest, "metrics": metrics, "run_dir": str(run_dir)}


def make_provenance(paths: ProtocolV2Paths, config_path: Path, config: RacalConfig) -> dict[str, Any]:
    model_path = _model_path(paths, config)
    payload = provenance_payload(paths.project_root, config_path, config, model_path)
    payload["e2_reference_root"] = str(paths.run_root / "e2_gate_core_dense")
    payload["racal_artifact_root"] = str(stage_root(paths))
    payload["third_party_policy"] = "record_only; do not modify third_party/mogb_official"
    return payload


def preflight(paths: ProtocolV2Paths, config_path: Path, config: RacalConfig) -> dict[str, Any]:
    paths.require_experiment_admission(DATASET)
    model_path = _model_path(paths, config)
    files = model_file_hashes(model_path)
    checks = []
    for seed in config.seeds:
        bundle = load_e2_bundle(paths, DATASET, seed, KIR)
        checks.append({"seed": seed, **validate_bundle(bundle), "train_shape": list(bundle.train.shape), "calibration_shape": list(bundle.calibration.shape), "test_shape": list(bundle.test.shape), "reference_run": str(_reference_dir(paths, seed))})
    return {"status": "preflight_ok", "config": config.as_dict(), "model_path": str(model_path), "model_files": files, "checks": checks, "methods": ["frozen_k1", "trainable_k1"], "planned_gate_units": len(config.seeds) * 2, "test_used_for_selection": False}


def run_stage(paths: ProtocolV2Paths, config_path: Path, *, method: str, seeds: Sequence[int], resume: bool, dry_run: bool) -> dict[str, Any]:
    config = _load_or_fail_config(config_path)
    paths.require_experiment_admission(DATASET)
    root = stage_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    provenance_path = root / "RACAL_PROVENANCE.json"
    if not provenance_path.is_file():
        atomic_write_json(provenance_path, _safe(make_provenance(paths, config_path, config)))
    if dry_run:
        plan = preflight(paths, config_path, config)
        atomic_write_json(root / "plans" / "stage_plan.json", _safe(plan))
        return plan
    selected = tuple(int(seed) for seed in seeds)
    if any(seed not in config.seeds for seed in selected):
        raise ValueError(f"Requested seeds must be declared in config: {selected}")
    results = []
    for seed in selected:
        if method == "frozen_k1":
            results.append(run_frozen_k1(paths, config, seed, resume))
        elif method == "trainable_k1":
            results.append(train_trainable_k1(paths, config, seed, resume))
        else:
            raise ValueError("RACAL-v1 stage 1 supports only frozen_k1 or trainable_k1")
    atomic_write_json(root / "state" / f"{method}_state.json", _safe({"method": method, "seeds": selected, "completed": len(results), "test_used_for_selection": False}))
    return {"status": "complete", "method": method, "completed": len(results), "results": results, "root": str(root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RACAL-v1 first-stage controls")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("frozen_k1", "trainable_k1"), default="frozen_k1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    config = _load_or_fail_config(args.config)
    seeds = args.seeds or ((args.seed,) if args.seed is not None else config.seeds)
    result = run_stage(paths, args.config.resolve(), method=args.method, seeds=seeds, resume=args.resume, dry_run=args.dry_run)
    print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
