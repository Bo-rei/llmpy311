#!/usr/bin/env python3
"""Train and evaluate a single-stage SmolLM baseline on the v19 gate split.

The goal is to provide a homogeneous control for the follow-up ablation:
- no hierarchy
- one SmolLM backbone
- closed-set intent classification with validation-only OOS threshold tuning

This is the SmolLM counterpart to ``eval_single_model_ood_v19.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from legacy.router import SmolLMRouter  # noqa: E402
from legacy.runtime import WorkspacePaths  # noqa: E402
from tools.eval.eval_system_pipeline_v19 import OOS_LABEL, _evaluate  # noqa: E402

PATHS = WorkspacePaths.discover(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


class FlatIntentDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: Any,
        intent_to_id: Dict[str, int],
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.intent_to_id = intent_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        enc = self.tokenizer(
            str(row["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(
                int(self.intent_to_id[str(row["intent"])]),
                dtype=torch.long,
            ),
        }


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _split_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    known = [row for row in records if int(row["label"]) == 0]
    oos = [row for row in records if int(row["label"]) == 1]
    return known, oos


def _build_intent_mapping(known_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    intents = sorted({str(row["intent"]) for row in known_rows})
    return {intent: idx for idx, intent in enumerate(intents)}


def _build_intent_domain_map(known_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    intent_domain_counts: Dict[str, Dict[str, int]] = {}
    for row in known_rows:
        intent = str(row["intent"])
        domain = str(row["domain"])
        intent_domain_counts.setdefault(intent, {})
        intent_domain_counts[intent][domain] = intent_domain_counts[intent].get(domain, 0) + 1

    mapping: Dict[str, str] = {}
    for intent, domain_counts in intent_domain_counts.items():
        mapping[intent] = max(domain_counts.items(), key=lambda item: item[1])[0]
    return mapping


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * len(labels)
        preds = logits.argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += len(labels)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def _predict_logits(
    model: nn.Module,
    tokenizer: Any,
    rows: List[Dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()
    logits_all: List[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        enc = tokenizer(
            [str(row["text"]) for row in batch_rows],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        logits_all.append(logits.detach().cpu().numpy())
    if not logits_all:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(logits_all, axis=0)


def _compute_oos_score(
    probs: np.ndarray,
    logits: np.ndarray,
    strategy: str,
    temperature: float,
) -> np.ndarray:
    if strategy == "msp":
        return 1.0 - np.max(probs, axis=1)

    if strategy == "entropy":
        eps = 1e-12
        entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)
        norm = np.log(max(probs.shape[1], 2))
        return entropy / max(norm, eps)

    if strategy == "energy":
        t = max(float(temperature), 1e-6)
        energy = -t * logsumexp(logits / t, axis=1)
        lo, hi = float(np.min(energy)), float(np.max(energy))
        if hi <= lo:
            return np.zeros_like(energy)
        return (energy - lo) / (hi - lo)

    raise ValueError(f"Unsupported oos strategy: {strategy}")


def _select_threshold(
    rows: List[Dict[str, Any]],
    probs: np.ndarray,
    logits: np.ndarray,
    class_id_to_intent: Dict[int, str],
    intent_to_domain: Dict[str, str],
    strategy: str,
) -> Dict[str, Any]:
    candidate_thresholds = np.linspace(0.01, 0.99, 99)
    best_row: Dict[str, Any] | None = None
    search_rows: List[Dict[str, Any]] = []

    for tau in candidate_thresholds.tolist():
        preds = _build_predictions(
            rows=rows,
            probs=probs,
            logits=logits,
            class_id_to_intent=class_id_to_intent,
            intent_to_domain=intent_to_domain,
            strategy=strategy,
            threshold=float(tau),
        )
        metrics = _evaluate(rows, preds)
        row = {
            "threshold": float(tau),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
            "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
            "oos_f1": float(metrics.get("oos_f1", 0.0)),
            "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
            "gate_id_recall": float(metrics.get("gate_id_recall", 0.0)),
        }
        search_rows.append(row)
        if best_row is None or (
            row["macro_f1"] > best_row["macro_f1"]
            or (
                np.isclose(row["macro_f1"], best_row["macro_f1"])
                and row["overall_accuracy"] > best_row["overall_accuracy"]
            )
        ):
            best_row = row

    assert best_row is not None
    return {
        "best": best_row,
        "search": search_rows,
    }


def _build_predictions(
    rows: List[Dict[str, Any]],
    probs: np.ndarray,
    logits: np.ndarray,
    class_id_to_intent: Dict[int, str],
    intent_to_domain: Dict[str, str],
    strategy: str,
    threshold: float,
) -> List[Dict[str, Any]]:
    oos_score = _compute_oos_score(probs, logits, strategy=strategy, temperature=1.0)
    best_idx = np.argmax(probs, axis=1)
    best_prob = np.max(probs, axis=1)

    predictions: List[Dict[str, Any]] = []
    for idx in range(len(rows)):
        score = float(oos_score[idx])
        is_oos = bool(score >= threshold)
        intent_name = class_id_to_intent[int(best_idx[idx])]

        if is_oos:
            predictions.append(
                {
                    "text": str(rows[idx]["text"]),
                    "is_oos": True,
                    "gate_pred": 1,
                    "fast_gate_pred": 1,
                    "gate_score": score,
                    "gate_distance": score,
                    "gate_radius": float(threshold),
                    "gate_margin_ok": False,
                    "gate_nearest_cluster": -1,
                    "gate_nearest_intent": intent_name,
                    "gate_stage": f"single_stage_smollm_{strategy}",
                    "semantic_id_score": None,
                    "semantic_gate_decision": None,
                    "semantic_top_intent": None,
                    "semantic_top_domain": None,
                    "semantic_decision_score": None,
                    "semantic_mode": "none",
                    "semantic_policy": "threshold",
                    "semantic_verifier_score": None,
                    "final_gate_decision": "oos",
                    "domain_id": -1,
                    "domain": OOS_LABEL,
                    "domain_prob": float(1.0 - score),
                    "router_confidence": float(best_prob[idx]),
                    "intent_id": -1,
                    "intent": OOS_LABEL,
                    "intent_prob": 1.0,
                }
            )
            continue

        predictions.append(
            {
                "text": str(rows[idx]["text"]),
                "is_oos": False,
                "gate_pred": 0,
                "fast_gate_pred": 0,
                "gate_score": score,
                "gate_distance": score,
                "gate_radius": float(threshold),
                "gate_margin_ok": True,
                "gate_nearest_cluster": -1,
                "gate_nearest_intent": intent_name,
                "gate_stage": f"single_stage_smollm_{strategy}",
                "semantic_id_score": None,
                "semantic_gate_decision": None,
                "semantic_top_intent": None,
                "semantic_top_domain": None,
                "semantic_decision_score": None,
                "semantic_mode": "none",
                "semantic_policy": "threshold",
                "semantic_verifier_score": None,
                "final_gate_decision": "id",
                "domain_id": -1,
                "domain": intent_to_domain.get(intent_name, "unknown"),
                "domain_prob": float(best_prob[idx]),
                "router_confidence": float(best_prob[idx]),
                "intent_id": int(best_idx[idx]),
                "intent": intent_name,
                "intent_prob": float(best_prob[idx]),
            }
        )

    return predictions


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-stage SmolLM OOD baseline evaluator")
    parser.add_argument("--model_path", default=str(PATHS.smollm135m))
    parser.add_argument("--gate_train", default="data/v19/gate/train.json")
    parser.add_argument("--gate_val", default="data/v19/gate/val.json")
    parser.add_argument("--gate_test", default="data/v19/gate/test.json")
    parser.add_argument("--oos_strategy", choices=["msp", "energy", "entropy"], default="msp")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger_file = logging.FileHandler(output_dir / "train.log")
    logger_file.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(logger_file)

    _set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    train_rows = _load_json(Path(args.gate_train))
    val_rows = _load_json(Path(args.gate_val))
    test_rows = _load_json(Path(args.gate_test))

    known_train, _ = _split_records(train_rows)
    val_known, _ = _split_records(val_rows)
    intent_to_id = _build_intent_mapping(known_train)
    class_id_to_intent = {idx: intent for intent, idx in intent_to_id.items()}
    intent_to_domain = _build_intent_domain_map(known_train)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = FlatIntentDataset(known_train, tokenizer, intent_to_id, int(args.max_length))
    val_ds = FlatIntentDataset(val_known, tokenizer, intent_to_id, int(args.max_length))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size) * 2),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    model = SmolLMRouter(
        model_path=str(args.model_path),
        num_classes=len(intent_to_id),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
    ).to(device)
    if hasattr(model.base, "print_trainable_parameters"):
        model.base.print_trainable_parameters()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    total_steps = max(len(train_loader) * int(args.epochs), 1)
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history_rows: List[Dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item()) * len(labels)
            preds = logits.detach().argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += len(labels)

            if step % 50 == 0:
                LOGGER.info(
                    "Epoch %d step %d/%d  loss=%.4f  lr=%.2e",
                    epoch,
                    step,
                    len(train_loader),
                    float(loss.item()),
                    scheduler.get_last_lr()[0],
                )

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = _evaluate_classifier(model, val_loader, device)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
            }
        )
        LOGGER.info(
            "Epoch %d finished: train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

    history_path = output_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history_rows)

    torch.save(model.state_dict(), output_dir / "final_model.pt")

    val_logits = _predict_logits(
        model,
        tokenizer,
        val_rows,
        device=device,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).cpu().numpy()
    threshold_selection = _select_threshold(
        rows=val_rows,
        probs=val_probs,
        logits=val_logits,
        class_id_to_intent=class_id_to_intent,
        intent_to_domain=intent_to_domain,
        strategy=str(args.oos_strategy),
    )
    threshold = float(threshold_selection["best"]["threshold"])
    val_predictions = _build_predictions(
        rows=val_rows,
        probs=val_probs,
        logits=val_logits,
        class_id_to_intent=class_id_to_intent,
        intent_to_domain=intent_to_domain,
        strategy=str(args.oos_strategy),
        threshold=threshold,
    )
    val_metrics = _evaluate(val_rows, val_predictions)

    test_logits = _predict_logits(
        model,
        tokenizer,
        test_rows,
        device=device,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )
    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).cpu().numpy()
    test_predictions = _build_predictions(
        rows=test_rows,
        probs=test_probs,
        logits=test_logits,
        class_id_to_intent=class_id_to_intent,
        intent_to_domain=intent_to_domain,
        strategy=str(args.oos_strategy),
        threshold=threshold,
    )
    test_metrics = _evaluate(test_rows, test_predictions)

    eval_payload = {
        "config": vars(args),
        "single_stage_smollm": {
            "backbone": str(args.model_path),
            "classifier": "SmolLMRouter",
            "num_known_intents": len(intent_to_id),
            "oos_strategy": str(args.oos_strategy),
            "threshold": float(threshold),
            "threshold_source": {
                "type": "validation_macro_f1",
                "split": "gate_val",
                "objective": "macro_f1",
                "strategy": str(args.oos_strategy),
                "search_grid": {
                    "min": 0.01,
                    "max": 0.99,
                    "steps": 99,
                },
            },
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.lr),
                "warmup_ratio": float(args.warmup_ratio),
                "weight_decay": float(args.weight_decay),
                "max_length": int(args.max_length),
                "final_train_loss": float(history_rows[-1]["train_loss"]) if history_rows else None,
                "final_train_acc": float(history_rows[-1]["train_acc"]) if history_rows else None,
                "final_val_loss": float(history_rows[-1]["val_loss"]) if history_rows else None,
                "final_val_acc": float(history_rows[-1]["val_acc"]) if history_rows else None,
                "history_path": str(history_path),
                "checkpoint_path": str(output_dir / "final_model.pt"),
            },
            "validation_selection": {
                "metrics": val_metrics,
                "threshold": float(threshold),
            },
        },
        "protocol": {
            "test_used_for_tuning": False,
            "test_used_for_calibration": False,
            "mode": "evaluation_only",
        },
        "metrics": test_metrics,
    }

    merged_predictions: List[Dict[str, Any]] = []
    for row, pred in zip(test_rows, test_predictions):
        merged_predictions.append(
            {
                "text": row["text"],
                "true_intent": row["intent"],
                "true_domain": row["domain"],
                "true_gate_label": int(row["label"]),
                **pred,
            }
        )

    _write_json(output_dir / "predictions.json", merged_predictions)
    _write_json(output_dir / "eval_results.json", eval_payload)

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "baseline": "single_stage_smollm",
        "oos_strategy": str(args.oos_strategy),
        "threshold": float(threshold),
        "threshold_source": eval_payload["single_stage_smollm"]["threshold_source"],
        "training_history_path": str(history_path),
        "final_checkpoint_path": str(output_dir / "final_model.pt"),
        "metrics_preview": {
            "macro_f1": float(test_metrics.get("macro_f1", 0.0)),
            "overall_accuracy": float(test_metrics.get("overall_accuracy", 0.0)),
            "known_intent_accuracy": float(test_metrics.get("known_intent_accuracy", 0.0)),
            "gate_oos_rejection": float(test_metrics.get("gate_oos_rejection", 0.0)),
        },
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)

    print(json.dumps(run_manifest["metrics_preview"], ensure_ascii=False))


if __name__ == "__main__":
    main()
