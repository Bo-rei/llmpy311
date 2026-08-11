#!/usr/bin/env python3
"""Train and evaluate a protocol_v2 StackOverflow Cascade bridge.

This runner deliberately owns a new artifact root.  It does not reuse v19
Router/Expert checkpoints and it does not alter the frozen Gate runs.  The
StackOverflow snapshot has one declared domain, so the Router is an explicit
constant-routing stage and only the intent Expert is trained on
``train_known`` with ``calibration_known`` used for checkpoint selection.

The runner is intentionally small: it answers the first downstream question
for the existing Frozen K=1 and Trainable K=1 Gates before any wider Cascade
matrix is attempted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from legacy.router import SmolLMRouter  # noqa: E402
from protocol_v2.runtime.paths import ProtocolV2Paths  # noqa: E402


DATASET = "stackoverflow"
KIR = 0.50
DEFAULT_SEEDS = (13, 42, 87)
STAGE = "cascade_bridge_v1"
MODEL_PATH = PROJECT_ROOT.parent / "assets" / "models" / "smollm135m"
GATE_ROOT = (
    PROJECT_ROOT.parent
    / "artifacts"
    / "s2c"
    / "runs"
    / "protocol_v2_textoir_v1"
    / "racal_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rows_hash(rows: Iterable[dict[str, Any]]) -> str:
    payload = "\n".join(str(row["sample_id"]) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _IntentDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, label_map: dict[str, int], max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.label_map[row["intent"]], dtype=torch.long),
        }


def _load_views(paths: ProtocolV2Paths, seed: int) -> dict[str, list[dict[str, Any]]]:
    root = paths.view_root / DATASET / f"seed_{seed}" / f"kir_{KIR:.2f}"
    views = {
        "train": _rows(root / "train_known.jsonl"),
        "calibration": _rows(root / "calibration_known.jsonl"),
        "test": _rows(root / "test_combined.jsonl"),
    }
    if not all(views.values()):
        raise RuntimeError(f"Missing protocol_v2 views under {root}")
    return views


def _gate_path(variant: str, seed: int) -> Path:
    return GATE_ROOT / "runs" / variant / f"seed_{seed}" / "predictions.jsonl"


def _train_expert(
    views: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    seed: int,
    model_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = output_dir / "best_model.pt"
    label_names = sorted({str(row["intent"]) for row in views["train"]})
    label_map = {name: idx for idx, name in enumerate(label_names)}
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds = _IntentDataset(views["train"], tokenizer, label_map, max_length)
    cal_ds = _IntentDataset(views["calibration"], tokenizer, label_map, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    model = SmolLMRouter(
        model_path=str(model_path),
        num_classes=len(label_names),
        lora_r=16,
        lora_alpha=32,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_accuracy = -1.0

    def evaluate(loader: DataLoader) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(ids, mask)
                total_loss += float(criterion(logits, labels).item()) * len(labels)
                correct += int((logits.argmax(dim=-1) == labels).sum().item())
                total += len(labels)
        return total_loss / max(total, 1), correct / max(total, 1)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.item()) * len(labels)
            correct += int((logits.detach().argmax(dim=-1) == labels).sum().item())
            total += len(labels)
        cal_loss, cal_accuracy = evaluate(cal_loader)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(total, 1),
            "train_accuracy": correct / max(total, 1),
            "calibration_loss": cal_loss,
            "calibration_accuracy": cal_accuracy,
        }
        history.append(row)
        if cal_accuracy > best_accuracy:
            best_accuracy = cal_accuracy
            torch.save(model.state_dict(), checkpoint)

    manifest = {
        "stage": STAGE,
        "dataset": DATASET,
        "kir": KIR,
        "seed": seed,
        "model_path": str(model_path.resolve()),
        "label_names": label_names,
        "label_map": label_map,
        "train_count": len(views["train"]),
        "calibration_count": len(views["calibration"]),
        "train_ids_sha256": _rows_hash(views["train"]),
        "calibration_ids_sha256": _rows_hash(views["calibration"]),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "best_calibration_accuracy": best_accuracy,
        "checkpoint_sha256": _sha256(checkpoint),
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output_dir / "training_history.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row) + "\n")
    return manifest


def _expert_predict(
    rows: list[dict[str, Any]],
    checkpoint_dir: Path,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> dict[str, str]:
    manifest = json.loads((checkpoint_dir / "training_manifest.json").read_text(encoding="utf-8"))
    label_map = {str(k): int(v) for k, v in manifest["label_map"].items()}
    inverse = {value: key for key, value in label_map.items()}
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = SmolLMRouter(
        model_path=manifest["model_path"],
        num_classes=len(inverse),
        lora_r=16,
        lora_alpha=32,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location=device))
    model.eval()
    outputs: dict[str, str] = {}
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["text"] for row in batch_rows],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            predictions = logits.argmax(dim=-1).cpu().tolist()
        for row, prediction in zip(batch_rows, predictions):
            outputs[str(row["sample_id"])] = inverse[int(prediction)]
    return outputs


def _evaluate_gate_cascade(
    gate_variant: str,
    seed: int,
    test_rows: list[dict[str, Any]],
    expert_predictions: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate_rows = _rows(_gate_path(gate_variant, seed))
    gate_by_id = {str(row["sample_id"]): row for row in gate_rows}
    if set(gate_by_id) != {str(row["sample_id"]) for row in test_rows}:
        raise ValueError(f"Gate/test sample IDs differ for {gate_variant} seed={seed}")
    records: list[dict[str, Any]] = []
    for row in test_rows:
        gate = gate_by_id[str(row["sample_id"])]
        accepted = int(gate["predicted_is_oos"]) == 0
        predicted = expert_predictions[str(row["sample_id"])] if accepted else "__oos__"
        gold_oos = int(row.get("evaluation_label") == "oos" or row.get("oos_source") not in {None, "known"})
        gold = "__oos__" if gold_oos else str(row["intent"])
        records.append(
            {
                "sample_id": row["sample_id"],
                "gold_intent": gold,
                "gold_is_oos": gold_oos,
                "predicted_intent": predicted,
                "predicted_is_oos": int(predicted == "__oos__"),
                "gate_oos_score": float(gate["oos_score"]),
                "gate_variant": gate_variant,
                "seed": seed,
            }
        )
    gold_oos = np.asarray([r["gold_is_oos"] for r in records], dtype=int)
    pred_oos = np.asarray([r["predicted_is_oos"] for r in records], dtype=int)
    gold_labels = [r["gold_intent"] for r in records]
    pred_labels = [r["predicted_intent"] for r in records]
    known_mask = gold_oos == 0
    oos_mask = gold_oos == 1
    score = np.asarray([r["gate_oos_score"] for r in records], dtype=float)
    metrics = {
        "stage": STAGE,
        "dataset": DATASET,
        "kir": KIR,
        "seed": seed,
        "gate_variant": gate_variant,
        "expert_checkpoint_used": True,
        "oos_f1": f1_score(gold_oos, pred_oos, pos_label=1, zero_division=0),
        "f1_all": f1_score(gold_labels, pred_labels, average="macro", zero_division=0),
        "f1_k": f1_score([g for g, m in zip(gold_labels, known_mask) if m], [p for p, m in zip(pred_labels, known_mask) if m], average="macro", zero_division=0),
        "accuracy": accuracy_score(gold_labels, pred_labels),
        "known_recall": float(np.mean(pred_oos[known_mask] == 0)) if np.any(known_mask) else float("nan"),
        "false_accept_rate": float(np.mean(pred_oos[oos_mask] == 0)) if np.any(oos_mask) else float("nan"),
        "false_reject_rate": float(np.mean(pred_oos[known_mask] == 1)) if np.any(known_mask) else float("nan"),
        "auroc": roc_auc_score(gold_oos, score),
        "aupr_oos": average_precision_score(gold_oos, score),
        "expert_accuracy_given_gate_accept": float(
            np.mean(
                [
                    p == g
                    for p, g, accepted in zip(pred_labels, gold_labels, pred_oos == 0)
                    if accepted and g != "__oos__"
                ]
            )
        ),
        "count": len(records),
    }
    return metrics, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    paths = ProtocolV2Paths.discover(PROJECT_ROOT)
    paths.require_experiment_admission(DATASET)
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(MODEL_PATH)
    device = torch.device(args.device if args.device != "auto" and (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    output_root = args.output_root or paths.run_root / STAGE
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_metrics: list[dict[str, Any]] = []
    for seed in args.seeds:
        _set_seed(seed)
        views = _load_views(paths, seed)
        seed_root = output_root / f"seed_{seed}"
        checkpoint_dir = seed_root / "expert"
        training_manifest_path = checkpoint_dir / "training_manifest.json"
        if args.resume and training_manifest_path.is_file():
            expert_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
        else:
            expert_manifest = _train_expert(
                views, checkpoint_dir, seed, MODEL_PATH, args.epochs,
                args.batch_size, args.learning_rate, args.max_length, device,
            )
        expert_predictions = _expert_predict(views["test"], checkpoint_dir, device, args.batch_size * 2, args.max_length)
        for gate_variant in ("frozen_k1", "trainable_k1"):
            metrics, records = _evaluate_gate_cascade(gate_variant, seed, views["test"], expert_predictions)
            metrics["expert_checkpoint_sha256"] = expert_manifest["checkpoint_sha256"]
            metrics["train_ids_sha256"] = expert_manifest["train_ids_sha256"]
            metrics["calibration_ids_sha256"] = expert_manifest["calibration_ids_sha256"]
            metrics["test_ids_sha256"] = _rows_hash(views["test"])
            all_metrics.append(metrics)
            pred_path = seed_root / f"{gate_variant}_predictions.jsonl"
            with pred_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"stage": STAGE, "protocol": paths.dataset_version, "metrics": all_metrics, "elapsed_seconds": time.time() - started, "test_used_for_selection": False, "oos_used_for_training": False}, handle, indent=2)
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        if all_metrics:
            writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0]))
            writer.writeheader()
            writer.writerows(all_metrics)
    (output_root / "PROVENANCE.json").write_text(json.dumps({"stage": STAGE, "protocol": paths.dataset_version, "dataset": DATASET, "kir": KIR, "seeds": args.seeds, "model_path": str(MODEL_PATH.resolve()), "model_sha256": _sha256(MODEL_PATH / "model.safetensors"), "gate_sources": {variant: [str(_gate_path(variant, seed)) for seed in args.seeds] for variant in ("frozen_k1", "trainable_k1")}, "test_used_for_selection": False, "oos_used_for_training": False, "elapsed_seconds": time.time() - started}, indent=2), encoding="utf-8")
    print(json.dumps({"stage": STAGE, "completed": len(all_metrics), "output_root": str(output_root), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
