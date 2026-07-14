"""Evaluate single-stage MiniLM closed-set baseline with OOS rejection strategies.

This script is designed for pipeline-level ablation baselines:
- single-stage MiniLM + MSP
- single-stage MiniLM + Energy
- single-stage MiniLM + Entropy

Model family:
- SentenceTransformer embeddings (MiniLM)
- LogisticRegression closed-set classifier over known intents

OOS decision:
- MSP: 1 - max softmax prob
- Energy: -T * logsumexp(logits / T)
- Entropy: normalized entropy over class probabilities

Threshold is selected on validation split only (never test split).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import logsumexp
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval.eval_system_pipeline_v19 import OOS_LABEL, _evaluate


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _encode_texts(encoder: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    embeddings = encoder.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


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
        mapping[intent] = max(domain_counts.items(), key=lambda x: x[1])[0]
    return mapping


def _fit_closed_set_classifier(
    encoder: SentenceTransformer,
    known_train: List[Dict[str, Any]],
    intent_to_id: Dict[str, int],
    batch_size: int,
    random_seed: int,
    label_shuffle: bool,
) -> LogisticRegression:
    texts = [str(row["text"]) for row in known_train]
    labels = np.asarray([intent_to_id[str(row["intent"])] for row in known_train], dtype=np.int64)
    if label_shuffle:
        rng = np.random.default_rng(int(random_seed))
        labels = rng.permutation(labels)

    x = _encode_texts(encoder, texts, batch_size=batch_size)

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=int(random_seed),
        n_jobs=1,
    )
    clf.fit(x, labels)
    return clf


def _rows_hash(rows: List[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {
            "text": str(row.get("text", "")),
            "intent": str(row.get("intent", "")),
            "domain": str(row.get("domain", "")),
            "label": int(row.get("label", -1)),
        }
        digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _label_intents(rows: List[Dict[str, Any]], label: Optional[int] = None) -> List[str]:
    return sorted(
        {
            str(row["intent"])
            for row in rows
            if label is None or int(row.get("label", -1)) == int(label)
        }
    )


def _indices_for_rows(all_rows: List[Dict[str, Any]], selected_rows: List[Dict[str, Any]]) -> List[int]:
    selected_ids = {id(row) for row in selected_rows}
    return [idx for idx, row in enumerate(all_rows) if id(row) in selected_ids]


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
    scores: np.ndarray,
    rows: List[Dict[str, Any]],
    probs: np.ndarray,
    logits: np.ndarray,
    class_id_to_intent: Dict[int, str],
    intent_to_domain: Dict[str, str],
    strategy: str,
) -> float:
    candidate_thresholds = np.linspace(0.01, 0.99, 99)
    best_macro_f1 = -1.0
    best_tau = 0.5

    for tau in candidate_thresholds:
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
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_tau = float(tau)

    return best_tau


def _threshold_source_payload(
    *,
    mode: str,
    strategy: str,
    threshold: float,
    fixed_threshold: float,
    selection_split: str,
    selection_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if mode == "fixed":
        return {
            "type": "fixed",
            "threshold": float(fixed_threshold),
            "strategy": strategy,
            "selection_split": None,
            "selection_count": 0,
            "uses_validation_oos": False,
        }
    return {
        "type": "validation_macro_f1",
        "mode": mode,
        "split": selection_split,
        "objective": "macro_f1",
        "strategy": strategy,
        "selection_count": len(selection_rows),
        "selection_known_count": sum(1 for row in selection_rows if int(row["label"]) == 0),
        "selection_oos_count": sum(1 for row in selection_rows if int(row["label"]) == 1),
        "uses_validation_oos": any(int(row["label"]) == 1 for row in selection_rows),
        "search_grid": {
            "min": 0.01,
            "max": 0.99,
            "steps": 99,
        },
    }


def _select_threshold_for_mode(
    *,
    mode: str,
    fixed_threshold: float,
    val_rows: List[Dict[str, Any]],
    val_probs: np.ndarray,
    val_logits: np.ndarray,
    class_id_to_intent: Dict[int, str],
    intent_to_domain: Dict[str, str],
    strategy: str,
) -> Tuple[float, List[Dict[str, Any]], str]:
    if mode == "fixed":
        return float(fixed_threshold), [], "fixed"

    if mode == "val_tuned":
        selection_rows = val_rows
        selection_probs = val_probs
        selection_logits = val_logits
        selection_split = "gate_val"
    elif mode == "no_val_oos":
        selection_rows, _ = _split_records(val_rows)
        selection_indices = _indices_for_rows(val_rows, selection_rows)
        selection_probs = val_probs[selection_indices]
        selection_logits = val_logits[selection_indices]
        selection_split = "gate_val_known_only"
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")

    threshold = _select_threshold(
        scores=np.zeros(len(selection_rows), dtype=np.float32),
        rows=selection_rows,
        probs=selection_probs,
        logits=selection_logits,
        class_id_to_intent=class_id_to_intent,
        intent_to_domain=intent_to_domain,
        strategy=strategy,
    )
    return float(threshold), selection_rows, selection_split


def _audit_payload(
    *,
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    selection_rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    threshold_mode: str,
    threshold: float,
    val_oos_used: bool,
    label_shuffle: bool,
) -> Dict[str, Any]:
    known_breakdown = dict(metrics.get("cascade_error_breakdown", {}).get("known", {}))
    oos_breakdown = dict(metrics.get("cascade_error_breakdown", {}).get("oos", {}))
    val_oos_intents = set(_label_intents(val_rows, 1))
    test_oos_intents = set(_label_intents(test_rows, 1))
    return {
        "threshold_mode": threshold_mode,
        "threshold_value": float(threshold),
        "val_oos_used": bool(val_oos_used),
        "label_shuffle": bool(label_shuffle),
        "split_hash": {
            "gate_train": _rows_hash(train_rows),
            "gate_val": _rows_hash(val_rows),
            "gate_test": _rows_hash(test_rows),
            "threshold_selection": _rows_hash(selection_rows),
        },
        "counts": {
            "train_known": sum(1 for row in train_rows if int(row["label"]) == 0),
            "train_oos": sum(1 for row in train_rows if int(row["label"]) == 1),
            "val_known": sum(1 for row in val_rows if int(row["label"]) == 0),
            "val_oos": sum(1 for row in val_rows if int(row["label"]) == 1),
            "test_known": sum(1 for row in test_rows if int(row["label"]) == 0),
            "test_oos": sum(1 for row in test_rows if int(row["label"]) == 1),
            "threshold_selection": len(selection_rows),
        },
        "intent_space": {
            "train_known_intents": _label_intents(train_rows, 0),
            "val_known_intents": _label_intents(val_rows, 0),
            "val_oos_intents": _label_intents(val_rows, 1),
            "test_known_intents": _label_intents(test_rows, 0),
            "test_oos_intents": _label_intents(test_rows, 1),
            "val_test_oos_intent_overlap": sorted(val_oos_intents.intersection(test_oos_intents)),
        },
        "known_false_reject": int(known_breakdown.get("gate_false_reject", 0)),
        "oos_false_accept": int(oos_breakdown.get("gate_false_accept", 0)),
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
    for i in range(len(rows)):
        score = float(oos_score[i])
        is_oos = bool(score >= threshold)

        if is_oos:
            predictions.append(
                {
                    "text": str(rows[i]["text"]),
                    "is_oos": True,
                    "gate_pred": 1,
                    "fast_gate_pred": 1,
                    "gate_score": score,
                    "gate_distance": score,
                    "gate_radius": float(threshold),
                    "gate_margin_ok": False,
                    "gate_nearest_cluster": -1,
                    "gate_nearest_intent": None,
                    "gate_stage": f"single_stage_minilm_{strategy}",
                    "final_gate_decision": "oos",
                    "domain_id": -1,
                    "domain": OOS_LABEL,
                    "domain_prob": float(1.0 - score),
                    "intent_id": -1,
                    "intent": OOS_LABEL,
                    "intent_prob": 1.0,
                }
            )
            continue

        intent_name = class_id_to_intent[int(best_idx[i])]
        predictions.append(
            {
                "text": str(rows[i]["text"]),
                "is_oos": False,
                "gate_pred": 0,
                "fast_gate_pred": 0,
                "gate_score": score,
                "gate_distance": score,
                "gate_radius": float(threshold),
                "gate_margin_ok": True,
                "gate_nearest_cluster": -1,
                "gate_nearest_intent": intent_name,
                "gate_stage": f"single_stage_minilm_{strategy}",
                "final_gate_decision": "id",
                "domain_id": -1,
                "domain": intent_to_domain.get(intent_name, "unknown"),
                "domain_prob": float(best_prob[i]),
                "intent_id": int(best_idx[i]),
                "intent": intent_name,
                "intent_prob": float(best_prob[i]),
            }
        )

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-stage MiniLM OOD baseline evaluator")
    parser.add_argument("--encoder_path", default="all-MiniLM-L6-v2")
    parser.add_argument("--gate_train", default="data/v19/gate/train.json")
    parser.add_argument("--gate_val", default="data/v19/gate/val.json")
    parser.add_argument("--gate_test", default="data/v19/gate/test.json")
    parser.add_argument("--oos_strategy", choices=["msp", "energy", "entropy"], default="msp")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--threshold_mode", choices=["val_tuned", "fixed", "no_val_oos"], default="val_tuned")
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--label_shuffle", action="store_true")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(args.seed))

    train_rows = _load_json(Path(args.gate_train))
    val_rows = _load_json(Path(args.gate_val))
    test_rows = _load_json(Path(args.gate_test))

    known_train, _ = _split_records(train_rows)
    intent_to_id = _build_intent_mapping(known_train)
    class_id_to_intent = {idx: intent for intent, idx in intent_to_id.items()}
    intent_to_domain = _build_intent_domain_map(known_train)

    encoder = SentenceTransformer(str(args.encoder_path))
    clf = _fit_closed_set_classifier(
        encoder=encoder,
        known_train=known_train,
        intent_to_id=intent_to_id,
        batch_size=int(args.batch_size),
        random_seed=int(args.seed),
        label_shuffle=bool(args.label_shuffle),
    )

    val_texts = [str(row["text"]) for row in val_rows]
    val_x = _encode_texts(encoder, val_texts, batch_size=int(args.batch_size))
    val_logits = clf.decision_function(val_x)
    val_probs = clf.predict_proba(val_x)

    threshold, selection_rows, selection_split = _select_threshold_for_mode(
        mode=str(args.threshold_mode),
        fixed_threshold=float(args.fixed_threshold),
        val_rows=val_rows,
        val_probs=val_probs,
        val_logits=val_logits,
        class_id_to_intent=class_id_to_intent,
        intent_to_domain=intent_to_domain,
        strategy=str(args.oos_strategy),
    )
    val_predictions = _build_predictions(
        rows=val_rows,
        probs=val_probs,
        logits=val_logits,
        class_id_to_intent=class_id_to_intent,
        intent_to_domain=intent_to_domain,
        strategy=str(args.oos_strategy),
        threshold=float(threshold),
    )
    val_metrics = _evaluate(val_rows, val_predictions)

    test_texts = [str(row["text"]) for row in test_rows]
    test_x = _encode_texts(encoder, test_texts, batch_size=int(args.batch_size))
    test_logits = clf.decision_function(test_x)
    test_probs = clf.predict_proba(test_x)

    predictions = _build_predictions(
        rows=test_rows,
        probs=test_probs,
        logits=test_logits,
        class_id_to_intent=class_id_to_intent,
        intent_to_domain=intent_to_domain,
        strategy=str(args.oos_strategy),
        threshold=float(threshold),
    )

    metrics = _evaluate(test_rows, predictions)
    threshold_source = _threshold_source_payload(
        mode=str(args.threshold_mode),
        strategy=str(args.oos_strategy),
        threshold=float(threshold),
        fixed_threshold=float(args.fixed_threshold),
        selection_split=selection_split,
        selection_rows=selection_rows,
    )
    audit = _audit_payload(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        selection_rows=selection_rows,
        metrics=metrics,
        threshold_mode=str(args.threshold_mode),
        threshold=float(threshold),
        val_oos_used=bool(threshold_source["uses_validation_oos"]),
        label_shuffle=bool(args.label_shuffle),
    )

    eval_payload = {
        "config": vars(args),
        "single_stage_minilm": {
            "classifier": "logistic_regression",
            "feature_encoder": str(args.encoder_path),
            "num_known_intents": len(intent_to_id),
            "oos_strategy": str(args.oos_strategy),
            "threshold": float(threshold),
            "threshold_mode": str(args.threshold_mode),
            "threshold_source": threshold_source,
            "val_oos_used": bool(threshold_source["uses_validation_oos"]),
            "label_shuffle": bool(args.label_shuffle),
            "audit": audit,
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
        "metrics": metrics,
    }

    merged_predictions: List[Dict[str, Any]] = []
    for row, pred in zip(test_rows, predictions):
        merged_predictions.append(
            {
                "text": row["text"],
                "true_intent": row["intent"],
                "true_domain": row["domain"],
                "true_gate_label": int(row["label"]),
                **pred,
            }
        )

    with open(output_dir / "predictions.json", "w", encoding="utf-8") as file:
        json.dump(merged_predictions, file, indent=2, ensure_ascii=False)

    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as file:
        json.dump(eval_payload, file, indent=2, ensure_ascii=False)

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "baseline": "single_stage_minilm",
        "strategy": str(args.oos_strategy),
        "threshold": float(threshold),
        "threshold_mode": str(args.threshold_mode),
        "threshold_source": threshold_source,
        "val_oos_used": bool(threshold_source["uses_validation_oos"]),
        "label_shuffle": bool(args.label_shuffle),
        "audit": audit,
        "metrics_preview": {
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
            "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
            "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
        },
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as file:
        json.dump(run_manifest, file, indent=2, ensure_ascii=False)

    print(json.dumps(run_manifest["metrics_preview"], ensure_ascii=False))


if __name__ == "__main__":
    main()
