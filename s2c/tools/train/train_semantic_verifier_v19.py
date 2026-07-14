#!/usr/bin/env python3
"""Train the gate prompt verifier for v19 with decoder-only LoRA.

The goal of this script is to fit a gate-specific SmolLM verifier that keeps
the prompt-based semantic gate behavior:

    prototype top-k -> prompt rerank -> ID/OOS decision

Training protocol:
- mine uncertain samples from gate train/val only
- build prototype top-k candidates from the router backbone space
- train on prompt-conditioned yes/no next-token likelihood
- never use test data for fitting or calibration
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gate.intent_prototype_matcher import IntentPrototypeMatcher  # noqa: E402
from src.gate.llm_semantic_verifier import LLMSemanticVerifier  # noqa: E402
from src.gate.multi_sphere_oos_detector import MultiSphereOOSDetector  # noqa: E402
from src.router.router_model import SmolLMRouter  # noqa: E402
from src.runtime import WorkspacePaths  # noqa: E402

PATHS = WorkspacePaths.discover(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


@dataclass
class PromptSample:
    """One candidate-wise prompt training example."""

    prompt: str
    label: int
    source: str
    query: str
    true_intent: str
    candidate_intent: str
    candidate_context: List[str]


class PromptVerifierDataset(Dataset):
    """Dataset for prompt-conditioned yes/no next-token training."""

    def __init__(
        self,
        samples: List[PromptSample],
        tokenizer: Any,
        yes_token_id: int,
        no_token_id: int,
        max_length: int,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.yes_token_id = int(yes_token_id)
        self.no_token_id = int(no_token_id)
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        encoded = self.tokenizer(
            sample.prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_id = self.yes_token_id if int(sample.label) == 1 else self.no_token_id
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_id": torch.tensor(target_id, dtype=torch.long),
            "label": torch.tensor(int(sample.label), dtype=torch.long),
        }


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_domain_mapping(records: List[Dict[str, Any]]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for row in records:
        mapping[int(row["label"])] = str(row["domain"])
    return dict(sorted(mapping.items(), key=lambda item: item[0]))


def _load_router_model(
    model_path: str,
    router_ckpt: Path,
    num_classes: int,
    lora_r: int,
    lora_alpha: int,
    device: torch.device,
) -> SmolLMRouter:
    state_dict = torch.load(router_ckpt, map_location="cpu")
    ckpt_num_classes = None
    for key, value in state_dict.items():
        if str(key).endswith("classifier.weight") and hasattr(value, "shape"):
            ckpt_num_classes = int(value.shape[0])
            break
    resolved_num_classes = int(ckpt_num_classes or num_classes)

    model = SmolLMRouter(
        model_path=model_path,
        num_classes=resolved_num_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _compute_gate_scores(
    records: List[Dict[str, Any]],
    gate_encoder_path: str,
    gate_detector_path: str,
    batch_size: int,
) -> np.ndarray:
    encoder = SentenceTransformer(gate_encoder_path)
    detector = MultiSphereOOSDetector()
    detector.load(gate_detector_path)

    texts = [str(row["text"]) for row in records]
    embeddings = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True)
    gate_out = detector.predict_with_scores(np.asarray(embeddings))
    return np.asarray(gate_out["score"], dtype=np.float32)


def _build_uncertain_pool(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    gate_scores_train: np.ndarray,
    gate_scores_val: np.ndarray,
    uncertain_low: float,
    uncertain_high: float,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    for row, score in zip(train_records, gate_scores_train.tolist()):
        if uncertain_low <= float(score) <= uncertain_high:
            samples.append(
                {
                    "text": str(row["text"]),
                    "intent": str(row["intent"]),
                    "domain": str(row["domain"]),
                    "label": int(row["label"]),
                    "gate_score": float(score),
                    "source": "train",
                }
            )

    for row, score in zip(val_records, gate_scores_val.tolist()):
        if uncertain_low <= float(score) <= uncertain_high:
            samples.append(
                {
                    "text": str(row["text"]),
                    "intent": str(row["intent"]),
                    "domain": str(row["domain"]),
                    "label": int(row["label"]),
                    "gate_score": float(score),
                    "source": "val",
                }
            )

    return samples


def _stratified_split(
    samples: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    id_samples = [sample for sample in samples if int(sample["label"]) == 0]
    oos_samples = [sample for sample in samples if int(sample["label"]) == 1]

    rng = random.Random(seed)
    rng.shuffle(id_samples)
    rng.shuffle(oos_samples)

    id_val_n = max(1, int(len(id_samples) * val_ratio)) if len(id_samples) > 1 else 0
    oos_val_n = max(1, int(len(oos_samples) * val_ratio)) if len(oos_samples) > 1 else 0

    val_samples = id_samples[:id_val_n] + oos_samples[:oos_val_n]
    train_samples = id_samples[id_val_n:] + oos_samples[oos_val_n:]

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError(
            "Uncertain pool split failed (empty train or val). "
            "Try widening uncertain interval."
        )

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def _binary_yes_no_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    yes_token_id: int,
    no_token_id: int,
) -> Dict[str, int]:
    """Map vocab-token predictions onto yes/no labels for verifier metrics."""
    del no_token_id
    pred_is_yes = preds == int(yes_token_id)
    label_is_yes = labels == 1

    correct = int((pred_is_yes == label_is_yes).sum().item())
    tp = int((pred_is_yes & label_is_yes).sum().item())
    fp = int((pred_is_yes & (~label_is_yes)).sum().item())
    fn = int(((~pred_is_yes) & label_is_yes).sum().item())
    return {
        "correct": correct,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _build_prompt_samples(
    samples: List[Dict[str, Any]],
    prototype_matcher: IntentPrototypeMatcher,
    semantic_top_k: int,
    prompt_version: str,
) -> List[PromptSample]:
    prompt_samples: List[PromptSample] = []
    for sample in samples:
        candidate_pack = prototype_matcher.top_k_intents(
            str(sample["text"]),
            top_k=int(semantic_top_k),
        )
        candidate_intents = [str(item) for item in candidate_pack["candidate_intents"]]
        true_intent = str(sample["intent"])
        is_known = int(sample["label"]) == 0

        if is_known and true_intent not in candidate_intents:
            candidate_intents = candidate_intents + [true_intent]

        for candidate_intent in candidate_intents:
            label = 1 if is_known and candidate_intent == true_intent else 0
            prompt = LLMSemanticVerifier.format_prompt(
                query=str(sample["text"]),
                intent_name=candidate_intent,
                candidate_intents=candidate_intents,
                prompt_version=prompt_version,
            )
            prompt_samples.append(
                PromptSample(
                    prompt=prompt,
                    label=label,
                    source=str(sample["source"]),
                    query=str(sample["text"]),
                    true_intent=true_intent,
                    candidate_intent=candidate_intent,
                    candidate_context=candidate_intents,
                )
            )

    return prompt_samples


@torch.no_grad()
def _evaluate_prompt_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_id = batch["target_id"].to(device)
        labels = batch["label"].to(device)

        outputs = model.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        loss = criterion(logits, target_id)

        preds = torch.argmax(logits, dim=-1)
        batch_metrics = _binary_yes_no_metrics(
            preds=preds,
            labels=labels,
            yes_token_id=int(loader.dataset.yes_token_id),
            no_token_id=int(loader.dataset.no_token_id),
        )
        total_loss += float(loss.item()) * int(labels.size(0))
        total += int(labels.size(0))
        correct += int(batch_metrics["correct"])
        tp += int(batch_metrics["tp"])
        fp += int(batch_metrics["fp"])
        fn += int(batch_metrics["fn"])

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))

    return {
        "loss": float(total_loss / max(total, 1)),
        "acc": float(correct / max(total, 1)),
        "yes_precision": precision,
        "yes_recall": recall,
        "yes_f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train v19 prompt semantic verifier")
    parser.add_argument("--model_path", default=str(PATHS.smollm135m))
    parser.add_argument("--router_ckpt", default=str(PATHS.artifact_root / "outputs/experiments/components/router/router_v19/best_model.pt"))
    parser.add_argument("--router_train", default=str(PATHS.prepared_data_root / "v19/router/train.json"))
    parser.add_argument("--gate_detector_path", default=str(PATHS.artifact_root / "outputs/core/gate_production/detector.json"))
    parser.add_argument("--gate_encoder_path", default=str(PATHS.minilm))
    parser.add_argument("--gate_train", default=str(PATHS.prepared_data_root / "v19/gate/train.json"))
    parser.add_argument("--gate_val", default=str(PATHS.prepared_data_root / "v19/gate/val.json"))
    parser.add_argument("--output_dir", default=str(PATHS.artifact_root / "outputs/prompt_semantic_verifier_v19"))
    parser.add_argument("--uncertain_low", type=float, default=0.98)
    parser.add_argument("--uncertain_high", type=float, default=1.05)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--semantic_top_k", type=int, default=3)
    parser.add_argument("--prompt_version", default="ranking_v1")
    parser.add_argument("--prototype_centers_default", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--gate_batch_size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(fh)

    LOGGER.info("Args: %s", vars(args))

    train_records = _load_json(Path(args.gate_train))
    val_records = _load_json(Path(args.gate_val))
    router_records = _load_json(Path(args.router_train))

    domain_id_to_name = _build_domain_mapping(router_records)
    device_name = str(args.device) if args.device is not None else _default_device()
    device = torch.device(device_name)
    LOGGER.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    router_model = _load_router_model(
        model_path=args.model_path,
        router_ckpt=Path(args.router_ckpt),
        num_classes=max(len(domain_id_to_name), 1),
        lora_r=max(int(args.lora_r), 1),
        lora_alpha=max(int(args.lora_alpha), 1),
        device=device,
    )

    prototype_matcher = IntentPrototypeMatcher(
        tokenizer=tokenizer,
        model=router_model,
        device=device,
    )
    prototype_matcher.build_prototypes(
        records=train_records,
        default_centers=int(args.prototype_centers_default),
        centers_overrides=None,
        random_state=int(args.seed),
    )

    LOGGER.info("Computing gate scores for uncertain mining...")
    train_scores = _compute_gate_scores(
        records=train_records,
        gate_encoder_path=args.gate_encoder_path,
        gate_detector_path=args.gate_detector_path,
        batch_size=args.gate_batch_size,
    )
    val_scores = _compute_gate_scores(
        records=val_records,
        gate_encoder_path=args.gate_encoder_path,
        gate_detector_path=args.gate_detector_path,
        batch_size=args.gate_batch_size,
    )

    uncertain_pool = _build_uncertain_pool(
        train_records=train_records,
        val_records=val_records,
        gate_scores_train=train_scores,
        gate_scores_val=val_scores,
        uncertain_low=float(args.uncertain_low),
        uncertain_high=float(args.uncertain_high),
    )

    if len(uncertain_pool) < 50:
        LOGGER.warning(
            "Uncertain pool too small (%d). Fallback to full train+val pool.",
            len(uncertain_pool),
        )
        uncertain_pool = [
            {
                "text": str(row["text"]),
                "intent": str(row["intent"]),
                "domain": str(row["domain"]),
                "label": int(row["label"]),
                "gate_score": float(score),
                "source": "train",
            }
            for row, score in zip(train_records, train_scores.tolist())
        ] + [
            {
                "text": str(row["text"]),
                "intent": str(row["intent"]),
                "domain": str(row["domain"]),
                "label": int(row["label"]),
                "gate_score": float(score),
                "source": "val",
            }
            for row, score in zip(val_records, val_scores.tolist())
        ]

    id_count = int(sum(int(sample["label"]) == 0 for sample in uncertain_pool))
    oos_count = int(sum(int(sample["label"]) == 1 for sample in uncertain_pool))
    LOGGER.info(
        "Verifier pool size=%d (ID=%d, OOS=%d)",
        len(uncertain_pool),
        id_count,
        oos_count,
    )

    if id_count == 0 or oos_count == 0:
        raise RuntimeError(
            "Verifier pool has single class only. Please adjust uncertain interval."
        )

    train_samples, val_samples = _stratified_split(
        uncertain_pool,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    train_prompt_samples = _build_prompt_samples(
        train_samples,
        prototype_matcher=prototype_matcher,
        semantic_top_k=int(args.semantic_top_k),
        prompt_version=str(args.prompt_version),
    )
    val_prompt_samples = _build_prompt_samples(
        val_samples,
        prototype_matcher=prototype_matcher,
        semantic_top_k=int(args.semantic_top_k),
        prompt_version=str(args.prompt_version),
    )

    _write_json(
        output_dir / "train_prompt_samples.json",
        [sample.__dict__ for sample in train_prompt_samples],
    )
    _write_json(
        output_dir / "val_prompt_samples.json",
        [sample.__dict__ for sample in val_prompt_samples],
    )

    yes_token_id = int(tokenizer(" yes", add_special_tokens=False)["input_ids"][0])
    no_token_id = int(tokenizer(" no", add_special_tokens=False)["input_ids"][0])

    train_ds = PromptVerifierDataset(
        train_prompt_samples,
        tokenizer,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        max_length=int(args.max_length),
    )
    val_ds = PromptVerifierDataset(
        val_prompt_samples,
        tokenizer,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        max_length=int(args.max_length),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(int(args.batch_size) * 2, 1),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )

    model = SmolLMRouter(
        model_path=args.model_path,
        num_classes=2,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
    ).to(device)
    model.train()

    # Logits are over tokenizer vocabulary, so class-weighted CE with 2 labels is invalid.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(len(train_loader) * int(args.epochs), 1)
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    metrics_csv = output_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_yes_f1"])

    best_val_yes_f1 = -1.0
    no_improve = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_id = batch["target_id"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            loss = criterion(logits, target_id)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits.detach(), dim=-1)
            running_loss += float(loss.item()) * int(target_id.size(0))
            running_correct += int((preds == target_id).sum().item())
            running_total += int(target_id.size(0))

            if step % 25 == 0:
                LOGGER.info(
                    "Epoch %d step %d/%d loss=%.4f lr=%.2e",
                    epoch,
                    step,
                    len(train_loader),
                    float(loss.item()),
                    scheduler.get_last_lr()[0],
                )

        train_loss = float(running_loss / max(running_total, 1))
        train_acc = float(running_correct / max(running_total, 1))
        val_metrics = _evaluate_prompt_model(model, val_loader, device)

        LOGGER.info(
            "Epoch %d train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_yes_f1=%.4f",
            epoch,
            train_loss,
            train_acc,
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["yes_f1"],
        )

        with open(metrics_csv, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_acc,
                    val_metrics["loss"],
                    val_metrics["acc"],
                    val_metrics["yes_f1"],
                ]
            )

        if val_metrics["yes_f1"] > best_val_yes_f1:
            best_val_yes_f1 = float(val_metrics["yes_f1"])
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            _write_json(
                output_dir / "best_metrics.json",
                {
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "pool_stats": {
                        "size": len(uncertain_pool),
                        "id": id_count,
                        "oos": oos_count,
                        "train_prompt_samples": len(train_prompt_samples),
                        "val_prompt_samples": len(val_prompt_samples),
                    },
                    "args": vars(args),
                },
            )
            LOGGER.info("New best val_yes_f1=%.4f", best_val_yes_f1)
        else:
            no_improve += 1
            if no_improve >= int(args.patience):
                LOGGER.info("Early stopping triggered.")
                break

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    _write_json(
        output_dir / "verifier_manifest.json",
        {
            "checkpoint": "best_model.pt",
            "prompt_version": str(args.prompt_version),
            "semantic_top_k": int(args.semantic_top_k),
            "label_schema": {"0": "no", "1": "yes"},
            "decision_score": "decoder_only_yes_probability",
            "token_ids": {
                "yes": int(yes_token_id),
                "no": int(no_token_id),
            },
            "uncertain_interval": {
                "low": float(args.uncertain_low),
                "high": float(args.uncertain_high),
            },
            "args": vars(args),
        },
    )

    LOGGER.info("Training done. Output -> %s", output_dir)


if __name__ == "__main__":
    main()
