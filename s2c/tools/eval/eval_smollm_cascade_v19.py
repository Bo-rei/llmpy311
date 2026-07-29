#!/usr/bin/env python3
"""Evaluate a SmolLM-only cascade baseline for v19 ablations.

The variant keeps the Gate -> Router -> Expert structure, but its OOS gate is
built from SmolLM hidden-state representations instead of the MiniLM encoder or
MiniLM detector used by the full pipeline and Cascade-MiniLM baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from legacy.router import SmolLMRouter  # noqa: E402
from tools.eval.eval_system_pipeline_v19 import OOS_LABEL, _evaluate  # noqa: E402
from tools.analysis.threshold_selection_v19 import select_main_table_constrained_threshold  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _known_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if int(row["label"]) == 0]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return payload


def _router_num_classes(router_train: Path) -> int:
    rows = _load_json(router_train)
    labels = sorted({int(row["label"]) for row in rows})
    if not labels:
        raise ValueError(f"Router training data is empty: {router_train}")
    return len(labels)


def _domain_mapping(router_train: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for row in _load_json(router_train):
        mapping[int(row["label"])] = str(row["domain"])
    return dict(sorted(mapping.items(), key=lambda item: item[0]))


@dataclass
class SmolLMPrototypeGate:
    centers: np.ndarray
    center_intents: List[str]
    threshold: float

    def predict(self, embeddings: np.ndarray) -> Dict[str, Any]:
        sims = embeddings @ self.centers.T
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(sims.shape[0]), best_idx]
        scores = 1.0 - best_sim
        pred = (scores > float(self.threshold)).astype(np.int64)
        return {
            "pred": pred,
            "score": scores.astype(np.float32),
            "distance": scores.astype(np.float32),
            "radius": np.full_like(scores, float(self.threshold), dtype=np.float32),
            "margin_ok": pred == 0,
            "nearest_cluster": best_idx.astype(np.int64),
            "nearest_intent": np.asarray([self.center_intents[int(idx)] for idx in best_idx], dtype=object),
        }


def _fit_smollm_gate_centers(
    embeddings: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    centers_per_intent: int,
) -> tuple[np.ndarray, List[str]]:
    known_items = [(idx, row) for idx, row in enumerate(rows) if int(row["label"]) == 0]
    indices_by_intent: Dict[str, List[int]] = {}
    for idx, row in known_items:
        indices_by_intent.setdefault(str(row["intent"]), []).append(idx)
    centers: List[np.ndarray] = []
    center_intents: List[str] = []
    for intent in sorted(indices_by_intent):
        vectors = embeddings[np.asarray(indices_by_intent[intent], dtype=np.int64)]
        n_centers = min(max(1, int(centers_per_intent)), vectors.shape[0])
        if n_centers == 1:
            intent_centers = np.mean(vectors, axis=0, keepdims=True)
        else:
            intent_centers = KMeans(n_clusters=n_centers, random_state=42, n_init=10).fit(vectors).cluster_centers_
        intent_centers = intent_centers / np.clip(np.linalg.norm(intent_centers, axis=1, keepdims=True), 1e-12, None)
        centers.append(intent_centers.astype(np.float32))
        center_intents.extend([intent] * int(intent_centers.shape[0]))
    if not centers:
        raise ValueError("Cannot build SmolLM gate without known training samples")
    return np.vstack(centers), center_intents


class SmolLMCascadeEvaluator:
    def __init__(
        self,
        *,
        model_path: Path,
        router_train: Path,
        router_ckpt: Path,
        experts_root: Path,
        experts_data_root: Path,
        device: torch.device,
        batch_size: int,
        max_length: int,
        router_lora_r: int,
        router_lora_alpha: int,
        expert_lora_r: int,
        expert_lora_alpha: int,
    ) -> None:
        self.model_path = model_path
        self.router_train = router_train
        self.router_ckpt = router_ckpt
        self.experts_root = experts_root
        self.experts_data_root = experts_data_root
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.router_lora_r = int(router_lora_r)
        self.router_lora_alpha = int(router_lora_alpha)
        self.expert_lora_r = int(expert_lora_r)
        self.expert_lora_alpha = int(expert_lora_alpha)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.router_model = SmolLMRouter(
            model_path=str(model_path),
            num_classes=_router_num_classes(router_train),
            lora_r=self.router_lora_r,
            lora_alpha=self.router_lora_alpha,
        ).to(device)
        self.router_model.load_state_dict(_load_state_dict(router_ckpt, device))
        self.router_model.eval()
        self.domain_id_to_name = _domain_mapping(router_train)
        self.expert_meta = self._index_experts()
        self.active_expert_domain: str | None = None
        self.active_expert_model: SmolLMRouter | None = None

    def _index_experts(self) -> Dict[str, Dict[str, Any]]:
        meta: Dict[str, Dict[str, Any]] = {}
        for domain_dir in sorted(self.experts_root.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            train_path = self.experts_data_root / domain / "train.json"
            ckpt_path = domain_dir / "best_model.pt"
            if not train_path.exists() or not ckpt_path.exists():
                continue
            label_to_intent = {
                int(row["label"]): str(row["intent"])
                for row in _load_json(train_path)
            }
            meta[domain] = {
                "checkpoint": ckpt_path,
                "num_classes": len(label_to_intent),
                "label_to_intent": label_to_intent,
            }
        if not meta:
            raise RuntimeError(f"No expert checkpoints discovered under {self.experts_root}")
        return meta

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = [str(text) for text in texts[start : start + self.batch_size]]
            enc = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            outputs = self.router_model.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
        if not all_embeddings:
            hidden_size = int(getattr(self.router_model.base.config, "hidden_size", 576))
            return np.zeros((0, hidden_size), dtype=np.float32)
        return np.vstack(all_embeddings)

    def build_gate(self, train_rows: Sequence[Dict[str, Any]], threshold: float, centers_per_intent: int) -> SmolLMPrototypeGate:
        embeddings = self.embed_texts([str(row["text"]) for row in train_rows])
        centers, center_intents = _fit_smollm_gate_centers(embeddings, train_rows, centers_per_intent)
        return SmolLMPrototypeGate(centers, center_intents, float(threshold))

    @torch.no_grad()
    def router_predict(self, texts: Sequence[str]) -> tuple[List[str], List[float]]:
        domains: List[str] = []
        probs_out: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = [str(text) for text in texts[start : start + self.batch_size]]
            enc = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            logits = self.router_model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            domains.extend(self.domain_id_to_name[int(idx)] for idx in pred.detach().cpu().tolist())
            probs_out.extend(float(value) for value in conf.detach().cpu().tolist())
        return domains, probs_out

    def _ensure_expert(self, domain: str) -> SmolLMRouter:
        if self.active_expert_domain == domain and self.active_expert_model is not None:
            return self.active_expert_model
        if domain not in self.expert_meta:
            raise KeyError(f"Expert domain not found: {domain}")
        info = self.expert_meta[domain]
        model = SmolLMRouter(
            model_path=str(self.model_path),
            num_classes=int(info["num_classes"]),
            lora_r=self.expert_lora_r,
            lora_alpha=self.expert_lora_alpha,
        ).to(self.device)
        model.load_state_dict(_load_state_dict(Path(info["checkpoint"]), self.device))
        model.eval()
        self.active_expert_domain = domain
        self.active_expert_model = model
        return model

    @torch.no_grad()
    def expert_predict(self, domain: str, texts: Sequence[str]) -> tuple[List[str], List[float]]:
        model = self._ensure_expert(domain)
        label_to_intent = self.expert_meta[domain]["label_to_intent"]
        intents: List[str] = []
        probs_out: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = [str(text) for text in texts[start : start + self.batch_size]]
            enc = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            logits = model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            intents.extend(str(label_to_intent[int(idx)]) for idx in pred.detach().cpu().tolist())
            probs_out.extend(float(value) for value in conf.detach().cpu().tolist())
        return intents, probs_out

    def predict(self, rows: Sequence[Dict[str, Any]], gate: SmolLMPrototypeGate) -> List[Dict[str, Any]]:
        texts = [str(row["text"]) for row in rows]
        embeddings = self.embed_texts(texts)
        gate_out = gate.predict(embeddings)
        domains, domain_probs = self.router_predict(texts)
        expert_intents = ["" for _ in rows]
        expert_probs = [0.0 for _ in rows]
        by_domain: Dict[str, List[int]] = {}
        for idx, domain in enumerate(domains):
            if int(gate_out["pred"][idx]) == 0:
                by_domain.setdefault(domain, []).append(idx)
        for domain, idxs in by_domain.items():
            intents, probs = self.expert_predict(domain, [texts[idx] for idx in idxs])
            for local_idx, row_idx in enumerate(idxs):
                expert_intents[row_idx] = intents[local_idx]
                expert_probs[row_idx] = probs[local_idx]
        predictions: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            is_oos = int(gate_out["pred"][idx]) == 1
            predictions.append(
                {
                    "text": str(row["text"]),
                    "is_oos": bool(is_oos),
                    "gate_pred": 1 if is_oos else 0,
                    "fast_gate_pred": 1 if is_oos else 0,
                    "gate_score": float(gate_out["score"][idx]),
                    "gate_distance": float(gate_out["distance"][idx]),
                    "gate_radius": float(gate_out["radius"][idx]),
                    "gate_margin_ok": bool(gate_out["margin_ok"][idx]),
                    "gate_nearest_cluster": int(gate_out["nearest_cluster"][idx]),
                    "gate_nearest_intent": str(gate_out["nearest_intent"][idx]),
                    "gate_stage": "cascade_smollm_gate",
                    "semantic_id_score": None,
                    "semantic_gate_decision": None,
                    "semantic_top_intent": None,
                    "semantic_top_domain": None,
                    "semantic_decision_score": None,
                    "semantic_mode": "none",
                    "semantic_policy": "threshold",
                    "semantic_verifier_score": None,
                    "final_gate_decision": "oos" if is_oos else "id",
                    "domain_id": -1 if is_oos else -1,
                    "domain": OOS_LABEL if is_oos else domains[idx],
                    "domain_prob": None if is_oos else float(domain_probs[idx]),
                    "intent_id": -1,
                    "intent": OOS_LABEL if is_oos else expert_intents[idx],
                    "intent_prob": 1.0 if is_oos else float(expert_probs[idx]),
                }
            )
        return predictions


def _gate_selection_score(rows: Sequence[Dict[str, Any]], gate_out: Dict[str, Any], objective: str) -> float:
    y_true = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray(gate_out["pred"], dtype=np.int64)
    if objective == "oos_f1":
        return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    if objective == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError(f"Unsupported threshold objective: {objective}")


def select_threshold(
    evaluator: SmolLMCascadeEvaluator,
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    *,
    thresholds: Sequence[float],
    centers_per_intent: int,
    objective: str,
    dataset_slug: str = "",
    kir_tag: str = "",
) -> tuple[SmolLMPrototypeGate, Dict[str, Any]]:
    train_embeddings = evaluator.embed_texts([str(row["text"]) for row in train_rows])
    centers, center_intents = _fit_smollm_gate_centers(train_embeddings, train_rows, centers_per_intent)
    val_texts = [str(row["text"]) for row in val_rows]
    val_embeddings = evaluator.embed_texts(val_texts)
    best: Dict[str, Any] | None = None
    sweep: List[Dict[str, Any]] = []
    for threshold in thresholds:
        gate = SmolLMPrototypeGate(centers, center_intents, float(threshold))
        predictions = evaluator.predict(val_rows, gate)
        metrics = _evaluate(val_rows, predictions)
        row = {
            "threshold": float(threshold),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
            "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
            "known_accuracy": float(metrics.get("known_intent_accuracy", metrics.get("known_accuracy", 0.0))),
            "oos_f1": float(metrics.get("oos_f1", 0.0)),
            "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
            "gate_id_recall": float(metrics.get("gate_id_recall", 0.0)),
        }
        if objective == "main_table_constrained_balanced":
            row["selection_score"] = 0.0
        else:
            gate_out = gate.predict(val_embeddings)
            row["selection_score"] = float(_gate_selection_score(val_rows, gate_out, objective))
        sweep.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selection_extra: Dict[str, Any] = {}
    if objective == "main_table_constrained_balanced":
        best, selection_extra = select_main_table_constrained_threshold(
            sweep,
            slug=str(dataset_slug),
            kir_tag=str(kir_tag),
        )
    return SmolLMPrototypeGate(centers, center_intents, float(best["threshold"])), {
        "best": best,
        "search": sweep,
        "threshold_source": "validation",
        "threshold_objective": selection_extra.get("threshold_objective", objective),
        **selection_extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolLM homogeneous cascade evaluator")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gate_train", required=True)
    parser.add_argument("--gate_val", required=True)
    parser.add_argument("--gate_test", required=True)
    parser.add_argument("--router_train", required=True)
    parser.add_argument("--router_ckpt", required=True)
    parser.add_argument("--experts_root", required=True)
    parser.add_argument("--experts_data_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--router_lora_r", type=int, default=32)
    parser.add_argument("--router_lora_alpha", type=int, default=64)
    parser.add_argument("--expert_lora_r", type=int, default=16)
    parser.add_argument("--expert_lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--threshold_min", type=float, default=0.01)
    parser.add_argument("--threshold_max", type=float, default=0.99)
    parser.add_argument("--threshold_steps", type=int, default=99)
    parser.add_argument(
        "--threshold_objective",
        choices=["macro_f1", "oos_f1", "main_table_constrained_balanced"],
        default="macro_f1",
    )
    parser.add_argument("--threshold_source", choices=["validation"], default="validation")
    parser.add_argument("--centers_per_intent", type=int, default=1)
    parser.add_argument("--dataset_slug", default="")
    parser.add_argument("--kir_tag", default="")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    _set_seed(int(args.seed))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_rows = _load_json(Path(args.gate_train))
    val_rows = _load_json(Path(args.gate_val))
    test_rows = _load_json(Path(args.gate_test))
    evaluator = SmolLMCascadeEvaluator(
        model_path=Path(args.model_path),
        router_train=Path(args.router_train),
        router_ckpt=Path(args.router_ckpt),
        experts_root=Path(args.experts_root),
        experts_data_root=Path(args.experts_data_root),
        device=device,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        router_lora_r=int(args.router_lora_r),
        router_lora_alpha=int(args.router_lora_alpha),
        expert_lora_r=int(args.expert_lora_r),
        expert_lora_alpha=int(args.expert_lora_alpha),
    )
    thresholds = np.linspace(float(args.threshold_min), float(args.threshold_max), int(args.threshold_steps)).tolist()
    gate, selection = select_threshold(
        evaluator,
        train_rows,
        val_rows,
        thresholds=thresholds,
        centers_per_intent=int(args.centers_per_intent),
        objective=str(args.threshold_objective),
        dataset_slug=str(args.dataset_slug),
        kir_tag=str(args.kir_tag),
    )
    predictions = evaluator.predict(test_rows, gate)
    metrics = _evaluate(test_rows, predictions)
    merged_predictions = [
        {
            "text": row["text"],
            "true_intent": row["intent"],
            "true_domain": row["domain"],
            "true_gate_label": int(row["label"]),
            **pred,
        }
        for row, pred in zip(test_rows, predictions)
    ]
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "predictions.json", merged_predictions)
    _write_json(
        output_dir / "eval_results.json",
        {
            "config": vars(args),
            "cascade_smollm": {
                "gate": "SmolLM_hidden_state_multi_centroid",
                "gate_model_path": str(args.model_path),
                "router": "SmolLMRouter",
                "router_ckpt": str(args.router_ckpt),
                "expert": "SmolLMRouter_per_domain",
                "experts_root": str(args.experts_root),
                "selected_threshold": float(gate.threshold),
            },
            "protocol": {
                "threshold_source": "validation",
                "test_used_for_tuning": False,
                "test_used_for_calibration": False,
            },
            "threshold_selection": selection,
            "metrics": metrics,
        },
    )
    with (output_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "selection_score",
                "macro_f1",
                "overall_accuracy",
                "known_intent_accuracy",
                "known_accuracy",
                "oos_f1",
                "gate_oos_rejection",
                "gate_id_recall",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in writer.fieldnames}
            for row in selection["search"]
        )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "baseline": "cascade_smollm",
        "gate_model_path": str(args.model_path),
        "gate_representation": "SmolLM_hidden_state_mean_pooling",
        "gate_detector_path": None,
        "gate_encoder_path": None,
        "router_ckpt": str(args.router_ckpt),
        "experts_root": str(args.experts_root),
        "threshold_source": "validation",
        "threshold_objective": str(args.threshold_objective),
        "selected_threshold": float(gate.threshold),
        "metrics_preview": {
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
            "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
            "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest["metrics_preview"], ensure_ascii=False))


if __name__ == "__main__":
    main()
