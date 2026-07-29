#!/usr/bin/env python3
"""Evaluate a MiniLM cascade baseline for v19 ablations.

This baseline keeps the Gate -> Router -> Expert structure intact:
- Gate: existing MiniLM multisphere detector
- Router: MiniLM embeddings + closed-set domain classifier
- Expert: one MiniLM closed-set intent classifier per predicted domain

It is not a single-stage global classifier and does not use MSP as the OOS
gate. OOS decisions come only from the loaded gate detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from tools.analysis.threshold_selection_v19 import select_main_table_constrained_threshold

OOS_LABEL = "__oos__"

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class ClosedSetHead:
    def __init__(self, labels: Sequence[str], classifier: LogisticRegression | None):
        unique = sorted(set(str(label) for label in labels))
        if not unique:
            raise ValueError("ClosedSetHead requires at least one label")
        self.labels = unique
        self.classifier = classifier
        self.constant_label = unique[0] if len(unique) == 1 else None

    @classmethod
    def fit(cls, x: np.ndarray, labels: Sequence[str], seed: int) -> "ClosedSetHead":
        unique = sorted(set(str(label) for label in labels))
        if len(unique) == 1:
            return cls(unique, None)
        label_to_id = {label: idx for idx, label in enumerate(unique)}
        y = np.asarray([label_to_id[str(label)] for label in labels], dtype=np.int64)
        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=int(seed),
            n_jobs=1,
        )
        clf.fit(x, y)
        return cls(unique, clf)

    def predict(self, x: np.ndarray) -> tuple[list[str], list[float]]:
        if self.constant_label is not None:
            return [self.constant_label for _ in range(x.shape[0])], [1.0 for _ in range(x.shape[0])]
        assert self.classifier is not None
        probs = self.classifier.predict_proba(x)
        best_idx = np.argmax(probs, axis=1)
        return [self.labels[int(idx)] for idx in best_idx], [float(probs[i, idx]) for i, idx in enumerate(best_idx)]


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _known_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if int(row["label"]) == 0]


def _encode(encoder: "SentenceTransformer", rows: Sequence[Dict[str, Any]], batch_size: int) -> np.ndarray:
    texts = [str(row["text"]) for row in rows]
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(encoder.encode(texts, batch_size=batch_size, show_progress_bar=False), dtype=np.float32)


def _load_gate(detector_path: Path) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector()
    detector.load(detector_path)
    return detector


def _gate_predict(detector: MultiSphereOOSDetector, x: np.ndarray) -> Dict[str, List[Any]]:
    out = detector.predict_with_scores(x)
    nearest_intents: List[str | None] = []
    for cluster in out.get("nearest_cluster", []):
        nearest_intents.append(detector.cluster_to_intent.get(int(cluster)))
    return {
        "pred": [int(x) for x in out["pred"]],
        "score": [float(x) for x in out["score"]],
        "distance": [float(x) for x in out["distance"]],
        "radius": [float(x) for x in out["radius"]],
        "margin_ok": [bool(x) for x in out.get("margin_ok", np.asarray(out["pred"]) == 0)],
        "nearest_cluster": [int(x) for x in out["nearest_cluster"]],
        "nearest_intent": nearest_intents,
    }


def _apply_score_threshold(gate_out: Dict[str, List[Any]], threshold: float) -> Dict[str, List[Any]]:
    adjusted = {key: list(value) for key, value in gate_out.items()}
    preds: List[int] = []
    for idx, score in enumerate(adjusted["score"]):
        margin_ok = bool(adjusted.get("margin_ok", [True] * len(adjusted["score"]))[idx])
        preds.append(0 if float(score) <= float(threshold) and margin_ok else 1)
    adjusted["pred"] = preds
    adjusted["selected_threshold"] = [float(threshold) for _ in preds]
    return adjusted


def _score_thresholds(
    *,
    rows: Sequence[Dict[str, Any]],
    base_gate_out: Dict[str, List[Any]],
    router_domains: Sequence[str],
    router_probs: Sequence[float],
    expert_intents: Sequence[str],
    expert_probs: Sequence[float],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    from tools.eval.eval_system_pipeline_v19 import _evaluate

    scored: List[Dict[str, Any]] = []
    for threshold in thresholds:
        gate_out = _apply_score_threshold(base_gate_out, float(threshold))
        predictions = _build_predictions(
            rows=rows,
            gate_out=gate_out,
            router_domains=router_domains,
            router_probs=router_probs,
            expert_intents=expert_intents,
            expert_probs=expert_probs,
        )
        metrics = _evaluate(rows, predictions)
        scored.append(
            {
                "threshold": float(threshold),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
                "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
                "known_accuracy": float(metrics.get("known_intent_accuracy", metrics.get("known_accuracy", 0.0))),
                "oos_f1": float(metrics.get("oos_f1", 0.0)),
                "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
                "gate_id_recall": float(metrics.get("gate_id_recall", 0.0)),
            }
        )
    return scored


def _train_heads(
    train_rows: Sequence[Dict[str, Any]],
    train_x: np.ndarray,
    seed: int,
) -> tuple[ClosedSetHead, Dict[str, ClosedSetHead]]:
    known = _known_rows(train_rows)
    if len(known) != train_x.shape[0]:
        raise ValueError("train_x must contain only known train rows")

    router = ClosedSetHead.fit(train_x, [str(row["domain"]) for row in known], seed)
    experts: Dict[str, ClosedSetHead] = {}
    domains = sorted({str(row["domain"]) for row in known})
    for domain in domains:
        idxs = [idx for idx, row in enumerate(known) if str(row["domain"]) == domain]
        domain_x = train_x[np.asarray(idxs, dtype=np.int64)]
        domain_labels = [str(known[idx]["intent"]) for idx in idxs]
        experts[domain] = ClosedSetHead.fit(domain_x, domain_labels, seed)
    return router, experts


def _predict_router_experts(
    router: ClosedSetHead,
    experts: Dict[str, ClosedSetHead],
    x: np.ndarray,
) -> tuple[list[str], list[float], list[str], list[float]]:
    domains, domain_probs = router.predict(x)
    intents: List[str] = []
    intent_probs: List[float] = []
    by_domain_indices: Dict[str, List[int]] = {}
    for idx, domain in enumerate(domains):
        by_domain_indices.setdefault(domain, []).append(idx)

    intents = ["" for _ in domains]
    intent_probs = [0.0 for _ in domains]
    for domain, idxs in by_domain_indices.items():
        if domain not in experts:
            raise ValueError(f"No expert trained for predicted domain: {domain}")
        domain_x = x[np.asarray(idxs, dtype=np.int64)]
        domain_intents, domain_intent_probs = experts[domain].predict(domain_x)
        for local_idx, row_idx in enumerate(idxs):
            intents[row_idx] = domain_intents[local_idx]
            intent_probs[row_idx] = domain_intent_probs[local_idx]
    return domains, domain_probs, intents, intent_probs


def _build_predictions(
    *,
    rows: Sequence[Dict[str, Any]],
    gate_out: Dict[str, Sequence[Any]],
    router_domains: Sequence[str],
    router_probs: Sequence[float],
    expert_intents: Sequence[str],
    expert_probs: Sequence[float],
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        is_oos = int(gate_out["pred"][idx]) == 1
        if is_oos:
            domain = OOS_LABEL
            intent = OOS_LABEL
            domain_prob = None
            intent_prob = None
            domain_id = -1
            intent_id = -1
        else:
            domain = str(router_domains[idx])
            intent = str(expert_intents[idx])
            domain_prob = float(router_probs[idx])
            intent_prob = float(expert_probs[idx])
            domain_id = -1
            intent_id = -1

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
                "gate_nearest_intent": gate_out["nearest_intent"][idx],
                "gate_stage": "cascade_minilm_gate",
                "semantic_id_score": None,
                "semantic_gate_decision": None,
                "semantic_top_intent": None,
                "semantic_top_domain": None,
                "semantic_decision_score": None,
                "semantic_mode": "none",
                "semantic_policy": "threshold",
                "semantic_verifier_score": None,
                "final_gate_decision": "oos" if is_oos else "id",
                "domain_id": domain_id,
                "domain": domain,
                "domain_prob": domain_prob,
                "intent_id": intent_id,
                "intent": intent,
                "intent_prob": intent_prob,
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniLM cascade baseline evaluator")
    parser.add_argument("--encoder_path", default="all-MiniLM-L6-v2")
    parser.add_argument("--gate_detector_path", required=True)
    parser.add_argument("--gate_train", default="data/v19/gate/train.json")
    parser.add_argument("--gate_val", default="data/v19/gate/val.json")
    parser.add_argument("--gate_test", default="data/v19/gate/test.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--threshold_min", type=float, default=0.2)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_steps", type=int, default=16)
    parser.add_argument("--dataset_slug", default="")
    parser.add_argument("--kir_tag", default="")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows_all = _load_json(Path(args.gate_train))
    val_rows = _load_json(Path(args.gate_val))
    test_rows = _load_json(Path(args.gate_test))
    train_rows = _known_rows(train_rows_all)

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(args.encoder_path))
    train_x = _encode(encoder, train_rows, int(args.batch_size))
    val_x = _encode(encoder, val_rows, int(args.batch_size))
    test_x = _encode(encoder, test_rows, int(args.batch_size))

    router, experts = _train_heads(train_rows, train_x, int(args.seed))
    detector = _load_gate(Path(args.gate_detector_path))
    val_gate_out = _gate_predict(detector, val_x)
    val_domains, val_domain_probs, val_intents, val_intent_probs = _predict_router_experts(router, experts, val_x)
    thresholds = np.linspace(float(args.threshold_min), float(args.threshold_max), int(args.threshold_steps)).tolist()
    sweep_rows = _score_thresholds(
        rows=val_rows,
        base_gate_out=val_gate_out,
        router_domains=val_domains,
        router_probs=val_domain_probs,
        expert_intents=val_intents,
        expert_probs=val_intent_probs,
        thresholds=thresholds,
    )
    selected_row, selection = select_main_table_constrained_threshold(
        sweep_rows,
        slug=str(args.dataset_slug),
        kir_tag=str(args.kir_tag),
    )
    selected_threshold = float(selected_row["threshold"])

    gate_out = _apply_score_threshold(_gate_predict(detector, test_x), selected_threshold)
    domains, domain_probs, intents, intent_probs = _predict_router_experts(router, experts, test_x)

    predictions = _build_predictions(
        rows=test_rows,
        gate_out=gate_out,
        router_domains=domains,
        router_probs=domain_probs,
        expert_intents=intents,
        expert_probs=intent_probs,
    )
    from tools.eval.eval_system_pipeline_v19 import _evaluate

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
    _write_json(output_dir / "predictions.json", merged_predictions)

    eval_payload = {
        "config": vars(args),
        "cascade_minilm": {
            "gate": "multisphere",
            "gate_detector_path": str(args.gate_detector_path),
            "router": "MiniLM_LogisticRegression_domain",
            "expert": "MiniLM_LogisticRegression_per_domain_intent",
            "num_domains": len(router.labels),
            "num_experts": len(experts),
            "encoder_path": str(args.encoder_path),
            "selected_threshold": selected_threshold,
        },
        "protocol": {
            "threshold_source": "validation",
            "test_used_for_tuning": False,
            "test_used_for_calibration": False,
            "mode": "validation_threshold_selection",
        },
        "threshold_selection": {
            "best": selected_row,
            "search": sweep_rows,
            **selection,
        },
        "metrics": metrics,
    }
    _write_json(output_dir / "eval_results.json", eval_payload)
    with (output_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "threshold",
            "macro_f1",
            "overall_accuracy",
            "known_intent_accuracy",
            "known_accuracy",
            "oos_f1",
            "gate_oos_rejection",
            "gate_id_recall",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in sweep_rows)

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "baseline": "cascade_minilm",
        "gate_detector_path": str(args.gate_detector_path),
        "encoder_path": str(args.encoder_path),
        "threshold_source": "validation",
        "threshold_objective": selection["threshold_objective"],
        "selected_threshold": selected_threshold,
        "metrics_preview": {
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
            "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
            "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
        },
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest["metrics_preview"], ensure_ascii=False))


if __name__ == "__main__":
    main()
