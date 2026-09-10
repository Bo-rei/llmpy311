#!/usr/bin/env python3
"""Run a protocol-bound Frozen/Trainable MiniLM K=1 control on v19 data.

This runner is deliberately separate from the active ``protocol_v2`` stages.
It consumes the archived v19 ``gate/{train,val,test}.json`` files directly,
keeps OOS out of representation training and checkpoint selection, and uses
the same K=1 detector/evaluator for the Frozen and Trainable rows.

The historical v19 Gate itself used a two-subcenter detector.  That method is
replayed by ``train_multisphere_corrected.py``; this file answers the narrower
question of whether a Trainable MiniLM representation changes the K=1 result
when the old data snapshot and split are held fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from protocol_v2.data.hashing import atomic_write_json, atomic_write_jsonl, sha256_file  # noqa: E402
from protocol_v2.experiments.racal_v1.boundary import detector_signature, evaluate_open, fit_k1_detector  # noqa: E402
from protocol_v2.experiments.racal_v1.representation import build_racal_model, choose_device, encode_rows, set_seed  # noqa: E402
from protocol_v2.experiments.racal_v1.runner import _center_losses, _class_centers  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    model_path: str
    dataset: str
    data_root: str
    seed: int
    projection_hidden_dim: int = 256
    warmup_epochs: int = 1
    finetune_epochs: int = 3
    patience: int = 1
    batch_size: int = 64
    max_length: int = 256
    projection_lr: float = 2e-4
    backbone_lr: float = 2e-5
    temperature: float = 0.07
    intra_weight: float = 0.1
    inter_weight: float = 0.1
    classification_weight: float = 1.0
    inter_margin: float = 0.20
    threshold: float = 1.0
    radius_lambda: float = 1.0
    representation_mode: str = "last2_minilm_plus_projection"
    selection_metric: str = "known_only"
    device: str = "auto"


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


def _row_id(row: Mapping[str, Any], split: str, index: int) -> str:
    payload = {
        "split": split,
        "index": int(index),
        "text": str(row.get("text", "")),
        "intent": str(row.get("intent", "")),
        "domain": str(row.get("domain", "")),
        "source_split": str(row.get("source_split", "")),
        "source_id": row.get("source_id"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_archive_views(data_root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    known_manifest = json.loads((data_root / "KNOWN_INTENTS.json").read_text(encoding="utf-8"))
    known = {str(value) for value in known_manifest["known_intents"]}
    views: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        rows = json.loads((data_root / "gate" / f"{split}.json").read_text(encoding="utf-8"))
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(rows):
            row = dict(raw)
            row["intent"] = str(row["intent"])
            row["text"] = str(row["text"])
            row["label"] = int(row.get("label", 0 if row["intent"] in known else 1))
            row["sample_id"] = _row_id(row, split, index)
            row["oos_source"] = "known" if row["label"] == 0 else str(row.get("split", "unknown"))
            normalized.append(row)
        views[split] = normalized
    snapshot = {
        "dataset": str(known_manifest.get("dataset", "")),
        "requested_kir": float(known_manifest.get("ratio")),
        "data_seed": int(known_manifest.get("seed")),
        "known_intents": sorted(known),
        "unknown_intents": sorted(str(value) for value in known_manifest.get("unknown_intents", [])),
        "selection_protocol": known_manifest.get("selection_protocol", {}),
        "split_counts": {
            split: {
                "total": len(rows),
                "known": int(sum(int(row["label"]) == 0 for row in rows)),
                "oos": int(sum(int(row["label"]) == 1 for row in rows)),
            }
            for split, rows in views.items()
        },
    }
    return snapshot, views


def _hash_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["sample_id"]) for row in rows).encode("utf-8")).hexdigest()


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _encode_frozen(model_path: Path, rows: Sequence[Mapping[str, Any]], device: torch.device, batch_size: int) -> np.ndarray:
    encoder = SentenceTransformer(str(model_path), device=str(device))
    return np.asarray(
        encoder.encode(
            [str(row["text"]) for row in rows],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ),
        dtype=np.float32,
    )


def _write_common(run_dir: Path, config: TrainConfig, snapshot: Mapping[str, Any], views: Mapping[str, Sequence[Mapping[str, Any]]], method: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "s2c.historical_protocol_v1.minilm_k1.v1",
        "method": method,
        "protocol_version": "historical_v19_archive",
        "dataset": config.dataset,
        "data_root": str(Path(config.data_root).resolve()),
        "data_snapshot": dict(snapshot),
        "split_sample_id_hashes": {name: _hash_rows(rows) for name, rows in views.items()},
        "model_path": str(Path(config.model_path).resolve()),
        "model_hashes": {
            name: sha256_file(Path(config.model_path) / name)
            for name in ("config.json", "tokenizer.json", "model.safetensors", "pytorch_model.bin")
            if (Path(config.model_path) / name).is_file()
        },
        "seed": int(config.seed),
        "requested_device": str(config.device),
        "threshold": float(config.threshold),
        "selection": "known_only_validation_for_checkpoint",
        "oos_used_for_training": False,
        "test_used_for_selection": False,
    }


def run_frozen(config: TrainConfig, snapshot: Mapping[str, Any], views: Mapping[str, list[dict[str, Any]]], run_dir: Path) -> dict[str, Any]:
    started = time.time()
    device = choose_device(config.device)
    values = {split: _encode_frozen(Path(config.model_path), rows, device, config.batch_size) for split, rows in views.items()}
    detector = fit_k1_detector(
        values["train"], views["train"], "mahalanobis_diag", radius_lambda=config.radius_lambda
    )
    metrics, predictions = evaluate_open(detector, values["test"], views["test"], config.threshold)
    payload = _write_common(run_dir, config, snapshot, views, "frozen_k1")
    payload.update({
        "device": str(device),
        "representation": "frozen_sentence_transformer_all-MiniLM-L6-v2",
        "distance": "mahalanobis_diag",
        "boundary": f"mean_std_lambda_{config.radius_lambda}",
        "embedding_hashes": {split: _hash_array(array) for split, array in values.items()},
        "metrics": _safe(metrics),
        "elapsed_seconds": time.time() - started,
        "status": "complete",
    })
    atomic_write_json(run_dir / "metrics.json", _safe(metrics))
    atomic_write_jsonl(run_dir / "predictions.jsonl", _safe(predictions))
    atomic_write_json(run_dir / "detector_signature.json", _safe(detector_signature(detector)))
    atomic_write_json(run_dir / "run_manifest.json", _safe(payload))
    return payload


def _make_optimizer(model: torch.nn.Module, config: TrainConfig, phase: str) -> torch.optim.Optimizer:
    projection = [p for p in model.projection.parameters() if p.requires_grad]
    if phase == "warmup":
        return torch.optim.AdamW(projection, lr=config.projection_lr)
    adapted = [p for name, p in model.named_parameters() if not name.startswith("projection.") and p.requires_grad]
    return torch.optim.AdamW([{"params": projection, "lr": config.projection_lr}, {"params": adapted, "lr": config.backbone_lr}])


def _set_phase(model: torch.nn.Module, phase: str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("projection."))
    if phase != "finetune":
        return
    if getattr(model, "mode", "") == "lora_minilm_plus_projection":
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
        return
    for block in model.encoder.encoder.layer[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)


def run_trainable(
    config: TrainConfig,
    snapshot: Mapping[str, Any],
    views: Mapping[str, list[dict[str, Any]]],
    run_dir: Path,
    method: str = "trainable_k1",
) -> dict[str, Any]:
    started = time.time()
    set_seed(config.seed)
    device = choose_device(config.device)
    model_path = Path(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = build_racal_model(model_path, config.representation_mode, config.projection_hidden_dim).to(device)
    train_rows = views["train"]
    if config.selection_metric not in {"known_only", "oos_f1"}:
        raise ValueError(f"Unsupported selection_metric: {config.selection_metric}")
    calibration_rows = (
        [row for row in views["val"] if int(row["label"]) == 0]
        if config.selection_metric == "known_only"
        else list(views["val"])
    )
    test_rows = views["test"]
    label_names = sorted({str(row["intent"]) for row in train_rows})
    label_map = {name: index for index, name in enumerate(label_names)}
    targets = torch.as_tensor([label_map[str(row["intent"])] for row in train_rows], dtype=torch.long, device=device)
    indices = np.arange(len(train_rows), dtype=np.int64)
    best_state: dict[str, Any] | None = None
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    epoch_number = 0
    for phase, epochs in (("warmup", config.warmup_epochs), ("finetune", config.finetune_epochs)):
        _set_phase(model, phase)
        optimizer = _make_optimizer(model, config, phase)
        for _ in range(epochs):
            epoch_number += 1
            model.train()
            with torch.no_grad():
                refreshed = encode_rows(model, tokenizer, train_rows, device, config.batch_size, config.max_length)
                centers_np, _, _ = _class_centers(refreshed, train_rows)
            centers = torch.as_tensor(centers_np, dtype=torch.float32, device=device)
            order = np.random.default_rng(config.seed + epoch_number * 7919).permutation(indices)
            losses: list[dict[str, float]] = []
            for start in range(0, len(order), config.batch_size):
                batch_indices = order[start : start + config.batch_size]
                tokens = tokenizer(
                    [str(train_rows[int(index)]["text"]) for index in batch_indices],
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                    return_tensors="pt",
                ).to(device)
                features = model(tokens)
                total, parts = _center_losses(
                    features,
                    targets[torch.as_tensor(batch_indices, dtype=torch.long, device=device)],
                    centers,
                    config.temperature,
                    config.intra_weight,
                    config.inter_weight,
                    config.classification_weight,
                    config.inter_margin,
                )
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(parts)
            model.eval()
            train_values = encode_rows(model, tokenizer, train_rows, device, config.batch_size, config.max_length)
            calibration_values = encode_rows(model, tokenizer, calibration_rows, device, config.batch_size, config.max_length)
            detector = fit_k1_detector(
                train_values, train_rows, "mahalanobis_diag", radius_lambda=config.radius_lambda
            )
            calibration_metrics, _ = evaluate_open(detector, calibration_values, calibration_rows, config.threshold)
            if config.selection_metric == "oos_f1":
                score = float(calibration_metrics["f1_u"] + 0.05 * calibration_metrics["known_recall"])
            else:
                score = float(calibration_metrics["f1_k"] + 0.05 * calibration_metrics["known_recall"])
            history.append({
                "epoch": epoch_number,
                "phase": phase,
                "loss": float(np.mean([item["total"] for item in losses])),
                "selection_score": score,
                "calibration_known_f1": float(calibration_metrics["f1_k"]),
                "calibration_known_recall": float(calibration_metrics["known_recall"]),
                "calibration_oos_f1": float(calibration_metrics["f1_u"]),
            })
            if score > best_score + 1e-12:
                best_score, best_epoch, stale = score, epoch_number, 0
                best_state = {
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "mode": model.mode,
                    "label_map": label_map,
                    "epoch": epoch_number,
                    "freeze_report": model.freeze_report(),
                }
            else:
                stale += 1
            if phase == "finetune" and stale > config.patience:
                break
        del optimizer
        if phase == "finetune" and stale > config.patience:
            break
    if best_state is None:
        raise RuntimeError(f"No trainable checkpoint produced for {config.dataset}")
    model.load_state_dict(best_state["model"])
    model.eval()
    final_train = encode_rows(model, tokenizer, train_rows, device, config.batch_size, config.max_length)
    final_calibration = encode_rows(model, tokenizer, calibration_rows, device, config.batch_size, config.max_length)
    final_test = encode_rows(model, tokenizer, test_rows, device, config.batch_size, config.max_length)
    detector = fit_k1_detector(
        final_train, train_rows, "mahalanobis_diag", radius_lambda=config.radius_lambda
    )
    metrics, predictions = evaluate_open(detector, final_test, test_rows, config.threshold)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"
    torch.save(best_state, checkpoint)
    atomic_write_json(run_dir / "training_history.json", _safe(history))
    atomic_write_json(run_dir / "metrics.json", _safe(metrics))
    atomic_write_jsonl(run_dir / "predictions.jsonl", _safe(predictions))
    atomic_write_json(run_dir / "detector_signature.json", _safe(detector_signature(detector)))
    payload = _write_common(run_dir, config, snapshot, views, method)
    payload.update({
        "device": str(device),
        "representation": config.representation_mode,
        "distance": "mahalanobis_diag",
        "boundary": f"mean_std_lambda_{config.radius_lambda}",
        "embedding_hashes": {
            "train": _hash_array(final_train),
            "calibration_known": _hash_array(final_calibration),
            "test": _hash_array(final_test),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "selection_metric": config.selection_metric,
        "trainable_parameters": int(best_state["freeze_report"]["trainable_parameter_count"]),
        "freeze_report": best_state["freeze_report"],
        "method": method,
        "metrics": _safe(metrics),
        "elapsed_seconds": time.time() - started,
        "status": "complete",
    })
    atomic_write_json(run_dir / "run_manifest.json", _safe(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=("frozen_k1", "trainable_k1", "lora_k1", "both"), default="both")
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--radius_lambda", type=float, default=1.0)
    parser.add_argument("--selection_metric", choices=("known_only", "oos_f1"), default="known_only")
    parser.add_argument(
        "--representation_mode",
        choices=("last2_minilm_plus_projection", "lora_minilm_plus_projection"),
        default="last2_minilm_plus_projection",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    snapshot, views = load_archive_views(args.data_root.resolve())
    representation_mode = (
        "lora_minilm_plus_projection"
        if args.method == "lora_k1"
        else str(args.representation_mode)
    )
    config = TrainConfig(
        model_path=str(args.model_path.resolve()),
        dataset=str(args.dataset),
        data_root=str(args.data_root.resolve()),
        seed=int(args.seed),
        warmup_epochs=int(args.warmup_epochs),
        finetune_epochs=int(args.finetune_epochs),
        patience=int(args.patience),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        radius_lambda=float(args.radius_lambda),
        representation_mode=representation_mode,
        selection_metric=str(args.selection_metric),
        device=str(args.device),
    )
    methods = ("frozen_k1", "trainable_k1") if args.method == "both" else (args.method,)
    outputs: dict[str, Any] = {"config": _safe(asdict(config)), "snapshot": _safe(snapshot), "runs": {}}
    for method in methods:
        run_dir = args.output_dir.resolve() / method
        if (run_dir / "run_manifest.json").is_file():
            raise FileExistsError(f"Run exists; choose a new output directory: {run_dir}")
        if method == "frozen_k1":
            result = run_frozen(config, snapshot, views, run_dir)
        else:
            result = run_trainable(config, snapshot, views, run_dir, method=method)
        outputs["runs"][method] = result
        print(json.dumps(_safe({"method": method, "metrics": result["metrics"], "run_dir": str(run_dir)}), ensure_ascii=False, indent=2))
    atomic_write_json(args.output_dir.resolve() / "summary.json", _safe(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
