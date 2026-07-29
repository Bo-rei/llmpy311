"""Evaluate HiLSA-MoE v19 end-to-end pipeline on Gate test set.

Outputs:
- outputs/pipeline_v19/eval_results.json
- outputs/pipeline_v19/predictions.json

Metrics:
- gate_id_recall
- gate_oos_rejection
- overall_accuracy (ID intent correctness + OOS rejection)
- known_intent_accuracy (on true-ID subset)
- domain_accuracy (on true-ID subset)
- macro_f1 (labels: all known intents + __oos__)

Important:
- This script is evaluation-only. It performs NO threshold tuning,
  NO calibration fitting, and NO hyperparameter search on test data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from legacy.runtime import WorkspacePaths

PATHS = WorkspacePaths.discover(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

OOS_LABEL = "__oos__"


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _ece_binary(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Optional[float]:
    """Expected calibration error for binary outcomes.

    Args:
        confidences: Predicted confidence in [0, 1].
        labels: Binary correctness labels (0/1).
    """
    if confidences.size == 0:
        return None

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= left) & (confidences < right)
        else:
            mask = (confidences >= left) & (confidences <= right)

        if not np.any(mask):
            continue

        bin_conf = float(np.mean(confidences[mask]))
        bin_acc = float(np.mean(labels[mask]))
        ece += abs(bin_acc - bin_conf) * float(np.mean(mask))

    return float(ece)


def _brier_binary(confidences: np.ndarray, labels: np.ndarray) -> Optional[float]:
    if confidences.size == 0:
        return None
    return float(np.mean((confidences - labels) ** 2))


def _confidence_summary(values: np.ndarray) -> Dict[str, Optional[float]]:
    if values.size == 0:
        return {"mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_existing_any(paths: List[Optional[Path]], label: str) -> Path:
    """Resolve the first existing path from a prioritized list."""
    for candidate in paths:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    candidates = ", ".join(str(path.resolve()) for path in paths if path is not None)
    raise FileNotFoundError(f"Missing required {label}; tried: {candidates}")


def _normalize_semantic_gate_mode(mode: str) -> str:
    """Map archived semantic-gate aliases onto the current enum."""
    if mode == "verifier":
        return "llm_verifier"
    return mode


def _normalize_legacy_output_dir(output_dir: str) -> str:
    """Map historical `outputs/<name>` runs into the experiments tree."""
    path = Path(output_dir)
    if path.is_absolute():
        return output_dir

    parts = path.parts
    if len(parts) < 2 or parts[0] != "outputs" or parts[1] == "experiments":
        return output_dir

    return str(Path("outputs/experiments") / Path(*parts[1:]))


def _predict_with_gate_disabled(
    pipeline: Any,
    texts: List[str],
    gate_test_records: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 64,
    no_gate_mode: str = "disabled",
    router_confidence_threshold: float = 0.5,
    random_gate_prob: float = 0.5,
    random_seed: int = 20260324,
) -> List[Dict[str, Any]]:
    """Run Router/Expert on all samples and bypass gate decision.

    This is a pipeline-level ablation path used to evaluate "No Gate" where
    every sample is forced into closed-set Router -> Expert inference.
    
    Args:
        pipeline: The HiLSAMoEV19Pipeline instance.
        texts: List of input texts.
        gate_test_records: Optional list of ground truth records (needed for oracle_oos mode).
        batch_size: Batch size for inference.
        no_gate_mode: How to detect OOS:
            - 'disabled': No OOS detection (all marked as ID)
            - 'router_confidence': Use Router max_prob < threshold as OOS
            - 'intent_confidence': Use final Expert intent confidence < threshold as OOS
            - 'oracle_oos': Use ground truth labels
            - 'random': Random OOS decision by Bernoulli(random_gate_prob)
        router_confidence_threshold: Threshold for router_confidence mode.
        random_gate_prob: Probability of predicting OOS in random mode.
        random_seed: Random seed for reproducibility in random mode.
    
    Returns:
        List of prediction dictionaries.
    """
    if len(texts) == 0:
        return []

    gate_out = pipeline._gate_predict(texts)
    router_out = pipeline._router_predict(texts, batch_size=batch_size)
    predicted_domains = [
        pipeline.domain_id_to_name[int(domain_id)]
        for domain_id in router_out["domain_ids"]
    ]

    # Get Oracle OOS labels if needed
    oracle_oos_labels: Optional[List[bool]] = None
    if no_gate_mode == "oracle_oos" and gate_test_records is not None:
        oracle_oos_labels = [int(row["label"]) == 1 for row in gate_test_records]

    random_oos_labels: Optional[np.ndarray] = None
    if no_gate_mode == "random":
        random_gate_prob = float(np.clip(random_gate_prob, 0.0, 1.0))
        rng = np.random.default_rng(int(random_seed))
        random_oos_labels = rng.random(len(texts)) < random_gate_prob

    results: List[Dict[str, Any]] = []
    for idx, text in enumerate(texts):
        gate_nearest_intent = None
        if pipeline.gate_mode == "multisphere":
            gate_nearest_intent = pipeline.gate_detector.cluster_to_intent.get(
                int(gate_out["nearest_cluster"][idx])
            )
        else:
            gate_nearest_intent = str(gate_out["nearest_intent"][idx])

        # Determine OOS detection mode
        is_oos = False
        router_max_prob = float(router_out["domain_probs"][idx])
        if no_gate_mode == "router_confidence":
            # Use Router max probability threshold
            is_oos = router_max_prob < router_confidence_threshold
        elif no_gate_mode == "oracle_oos":
            # Use oracle labels
            if oracle_oos_labels is not None:
                is_oos = oracle_oos_labels[idx]
        elif no_gate_mode == "random":
            if random_oos_labels is not None:
                is_oos = bool(random_oos_labels[idx])
        # else: no_gate_mode == "disabled" -> is_oos stays False

        results.append(
            {
                "text": text,
                "is_oos": is_oos,
                "gate_pred": 1 if is_oos else 0,
                "fast_gate_pred": int(gate_out["pred"][idx]),
                "gate_score": float(1.0 - router_max_prob)
                if no_gate_mode == "router_confidence"
                else float(gate_out["score"][idx]),
                "gate_distance": float(1.0 - router_max_prob)
                if no_gate_mode == "router_confidence"
                else float(gate_out["distance"][idx]),
                "gate_radius": float(1.0 - router_confidence_threshold)
                if no_gate_mode == "router_confidence"
                else float(gate_out["radius"][idx]),
                "gate_margin_ok": bool(not is_oos)
                if no_gate_mode == "router_confidence"
                else bool(gate_out["margin_ok"][idx]),
                "gate_nearest_cluster": int(gate_out["nearest_cluster"][idx]),
                "gate_nearest_intent": gate_nearest_intent,
                "gate_stage": f"ablation_no_gate_{no_gate_mode}",
                "semantic_id_score": None,
                "semantic_gate_decision": None,
                "semantic_top_intent": None,
                "semantic_top_domain": None,
                "semantic_decision_score": None,
                "semantic_candidate_best_intent": None,
                "semantic_candidate_best_score": None,
                "semantic_candidate_runner_up_intent": None,
                "semantic_candidate_runner_up_score": None,
                "semantic_candidate_score_margin": None,
                "semantic_mode": pipeline.semantic_gate_mode,
                "semantic_policy": pipeline.semantic_decision_policy,
                "semantic_verifier_score": None,
                "final_gate_decision": "oos" if is_oos else "id",
                "domain_id": int(router_out["domain_ids"][idx]),
                "domain": predicted_domains[idx],
                "domain_prob": router_max_prob,
                "router_confidence_threshold": router_confidence_threshold if no_gate_mode == "router_confidence" else None,
                "random_gate_prob": random_gate_prob if no_gate_mode == "random" else None,
            }
        )

    domain_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, domain_name in enumerate(predicted_domains):
        # Skip OOS samples in Router path
        if not results[idx]["is_oos"]:
            domain_to_indices[domain_name].append(idx)

    for domain_name, indices in domain_to_indices.items():
        domain_texts = [texts[i] for i in indices]
        expert_out = pipeline._expert_predict_group(
            domain_name=domain_name,
            texts=domain_texts,
            batch_size=batch_size,
        )
        for local_idx, global_idx in enumerate(indices):
            results[global_idx]["intent_id"] = int(expert_out["intent_ids"][local_idx])
            results[global_idx]["intent"] = expert_out["intent_names"][local_idx]
            results[global_idx]["intent_prob"] = float(expert_out["intent_probs"][local_idx])
            results[global_idx]["no_gate_confidence"] = float(expert_out["intent_probs"][local_idx])

            if no_gate_mode == "intent_confidence":
                confidence = float(expert_out["intent_probs"][local_idx])
                is_oos = confidence < float(router_confidence_threshold)
                results[global_idx]["is_oos"] = bool(is_oos)
                results[global_idx]["gate_pred"] = 1 if is_oos else 0
                results[global_idx]["fast_gate_pred"] = 1 if is_oos else 0
                results[global_idx]["gate_score"] = float(1.0 - confidence)
                results[global_idx]["gate_distance"] = float(1.0 - confidence)
                results[global_idx]["gate_radius"] = float(max(1.0 - float(router_confidence_threshold), 0.0))
                results[global_idx]["gate_margin_ok"] = bool(not is_oos)
                results[global_idx]["gate_stage"] = "ablation_no_gate_intent_confidence"
                results[global_idx]["final_gate_decision"] = "oos" if is_oos else "id"
                results[global_idx]["router_confidence_threshold"] = float(router_confidence_threshold)
                if is_oos:
                    results[global_idx]["intent_id"] = -1
                    results[global_idx]["intent"] = "__oos__"
        
        # For OOS samples, set intent/domain to OOS marker
        for global_idx in range(len(results)):
            if results[global_idx]["is_oos"]:
                results[global_idx]["intent_id"] = -1
                results[global_idx]["intent"] = "__oos__"
                results[global_idx]["intent_prob"] = 1.0

    return results


def _tune_multi_proto_id_threshold(
    pipeline: Any,
    gate_val_records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Tune multi-prototype ID threshold on validation split only.

    The multi-prototype gate uses score_mode=top2_margin_conf by default.
    Its score range is typically far below 0.8, so fixed threshold can make
    all samples collapse to OOS. This helper searches threshold on validation
    split to maximize binary macro-F1 (ID vs OOS).
    """
    if len(gate_val_records) == 0:
        raise ValueError("gate_val_records is empty; cannot tune multi-prototype threshold")

    texts = [str(row["text"]) for row in gate_val_records]
    y_true_oos = np.asarray([int(row["label"]) for row in gate_val_records], dtype=np.int64)

    gate_out = pipeline._gate_predict(texts)
    sim_scores = 1.0 - np.asarray(gate_out["score"], dtype=np.float32)

    lo = float(np.min(sim_scores))
    hi = float(np.max(sim_scores))
    if hi <= lo:
        threshold = float(lo)
        y_pred = (sim_scores < threshold).astype(np.int64)
        macro = float(f1_score(y_true_oos, y_pred, average="macro", zero_division=0))
        return {
            "best_threshold": threshold,
            "best_macro_f1": macro,
            "search_min": lo,
            "search_max": hi,
        }

    candidates = np.linspace(lo, hi, num=201, dtype=np.float32)
    best_threshold = float(candidates[0])
    best_macro = -1.0

    for threshold in candidates.tolist():
        y_pred_oos = (sim_scores < float(threshold)).astype(np.int64)
        macro = float(f1_score(y_true_oos, y_pred_oos, average="macro", zero_division=0))
        if macro > best_macro:
            best_macro = macro
            best_threshold = float(threshold)

    return {
        "best_threshold": best_threshold,
        "best_macro_f1": float(best_macro),
        "search_min": lo,
        "search_max": hi,
    }


def _tune_semantic_threshold_from_scores(
    *,
    fast_gate_preds: np.ndarray,
    uncertain_indices: List[int],
    semantic_scores: np.ndarray,
    y_true_oos: np.ndarray,
) -> Dict[str, float]:
    """Tune semantic ID threshold on validation by binary macro-F1."""
    if y_true_oos.size == 0:
        raise ValueError("y_true_oos is empty; cannot tune semantic threshold")

    if len(uncertain_indices) != int(semantic_scores.shape[0]):
        raise ValueError("uncertain_indices and semantic_scores must align")

    if len(uncertain_indices) == 0:
        fixed_preds = np.asarray(fast_gate_preds, dtype=np.int64)
        macro = float(f1_score(y_true_oos, fixed_preds, average="macro", zero_division=0))
        return {
            "best_threshold": 0.5,
            "best_macro_f1": macro,
            "search_min": 0.5,
            "search_max": 0.5,
        }

    lo = float(np.min(semantic_scores))
    hi = float(np.max(semantic_scores))
    if hi <= lo:
        hi = lo

    candidates = np.linspace(lo, hi, num=201, dtype=np.float32) if hi > lo else np.asarray([lo], dtype=np.float32)
    best_threshold = float(candidates[0])
    best_macro = -1.0

    base_preds = np.asarray(fast_gate_preds, dtype=np.int64)
    uncertain_arr = np.asarray(uncertain_indices, dtype=np.int64)
    scores_arr = np.asarray(semantic_scores, dtype=np.float32)

    for threshold in candidates.tolist():
        y_pred_oos = base_preds.copy()
        y_pred_oos[uncertain_arr] = np.where(scores_arr >= float(threshold), 0, 1)
        macro = float(f1_score(y_true_oos, y_pred_oos, average="macro", zero_division=0))
        if macro > best_macro:
            best_macro = macro
            best_threshold = float(threshold)

    return {
        "best_threshold": best_threshold,
        "best_macro_f1": float(best_macro),
        "search_min": lo,
        "search_max": hi,
    }


def _tune_fusion_alpha_and_threshold(
    *,
    fast_gate_preds: np.ndarray,
    uncertain_indices: List[int],
    prototype_scores: np.ndarray,
    verifier_scores: np.ndarray,
    y_true_oos: np.ndarray,
) -> Dict[str, float]:
    """Tune fusion alpha and threshold on validation by binary macro-F1."""
    if len(uncertain_indices) != int(prototype_scores.shape[0]) or len(uncertain_indices) != int(verifier_scores.shape[0]):
        raise ValueError("Fusion tuning inputs must align on uncertain sample count")

    best_alpha = 0.7
    best_threshold = 0.5
    best_macro = -1.0
    best_search_min = 0.0
    best_search_max = 1.0

    for alpha in np.linspace(0.1, 0.9, num=9, dtype=np.float32).tolist():
        beta = float(1.0 - float(alpha))
        fused = np.asarray(
            [float(alpha) * float(p) + beta * float(v) for p, v in zip(prototype_scores.tolist(), verifier_scores.tolist())],
            dtype=np.float32,
        )
        tuning = _tune_semantic_threshold_from_scores(
            fast_gate_preds=fast_gate_preds,
            uncertain_indices=uncertain_indices,
            semantic_scores=fused,
            y_true_oos=y_true_oos,
        )
        if float(tuning["best_macro_f1"]) > best_macro:
            best_macro = float(tuning["best_macro_f1"])
            best_alpha = float(alpha)
            best_threshold = float(tuning["best_threshold"])
            best_search_min = float(tuning["search_min"])
            best_search_max = float(tuning["search_max"])

    return {
        "best_alpha": best_alpha,
        "best_beta": float(1.0 - best_alpha),
        "best_threshold": best_threshold,
        "best_macro_f1": best_macro,
        "search_min": best_search_min,
        "search_max": best_search_max,
    }


def build_baseline_comparison_table(baseline_path: str, candidate_path: str) -> Dict[str, Any]:
    """Build deterministic metric delta table between baseline and candidate eval files."""
    baseline = _load_json(Path(baseline_path))
    candidate = _load_json(Path(candidate_path))

    key_metrics = [
        "macro_f1",
        "overall_accuracy",
        "known_intent_accuracy",
        "gate_id_recall",
        "gate_oos_rejection",
        "oos_f1",
    ]

    baseline_metrics = baseline.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})

    metric_deltas: Dict[str, Dict[str, float]] = {}
    for metric_name in key_metrics:
        baseline_value = float(baseline_metrics.get(metric_name, 0.0))
        candidate_value = float(candidate_metrics.get(metric_name, 0.0))
        metric_deltas[metric_name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": candidate_value - baseline_value,
        }

    return {
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "key_metrics": key_metrics,
        "metric_deltas": metric_deltas,
    }


def _apply_id_rescue_threshold(
    scored_predictions: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Rescue high-confidence final-OOS predictions back to ID."""
    rescued: List[Dict[str, Any]] = []
    for pred in scored_predictions:
        row = dict(pred)
        score = row.get("rescue_score")
        if int(row.get("gate_pred", 1)) == 1 and score is not None and float(score) >= float(threshold):
            row["gate_pred"] = 0
            row["is_oos"] = False
            row["gate_stage"] = "id_rescue"
            row["final_gate_decision"] = "id"
            row["domain"] = row.get("rescue_domain", row.get("domain", ""))
            row["domain_id"] = row.get("rescue_domain_id")
            row["domain_prob"] = row.get("rescue_domain_prob")
            row["intent"] = row.get("rescue_intent", row.get("intent", OOS_LABEL))
            row["intent_id"] = row.get("rescue_intent_id")
            row["intent_prob"] = row.get("rescue_intent_prob")
        rescued.append(row)
    return rescued


def _tune_id_rescue_threshold_from_scored_predictions(
    *,
    records: List[Dict[str, Any]],
    scored_predictions: List[Dict[str, Any]],
    objective: str,
    min_oos_recall: Optional[float] = None,
) -> Dict[str, Any]:
    """Tune rescue threshold on validation predictions."""
    if objective not in {"val_oos_f1", "val_macro_f1", "val_oos_f1_recall_guard"}:
        raise ValueError(f"Unsupported ID rescue tuning objective: {objective}")

    scores = sorted(
        {
            float(row["rescue_score"])
            for row in scored_predictions
            if row.get("rescue_score") is not None
        }
    )
    if not scores:
        metrics = _evaluate(records, scored_predictions)
        return {
            "mode": objective,
            "best_threshold": 1.0,
            "best_oos_f1": float(metrics["oos_f1"]),
            "best_macro_f1": float(metrics["macro_f1"]),
            "candidate_count": 0,
        }

    candidates = sorted(set([0.0, 1.0] + scores))
    best: Optional[Dict[str, Any]] = None
    for threshold in candidates:
        rescued = _apply_id_rescue_threshold(scored_predictions, threshold=float(threshold))
        metrics = _evaluate(records, rescued)
        if objective == "val_macro_f1":
            objective_value = float(metrics["macro_f1"])
        else:
            objective_value = float(metrics["oos_f1"])
            guard = float(min_oos_recall) if min_oos_recall is not None else None
            if objective == "val_oos_f1_recall_guard" and guard is not None:
                if float(metrics["oos_recall"]) + 1e-12 < guard:
                    objective_value = -1.0 + float(metrics["oos_recall"])
        item = {
            "mode": objective,
            "best_threshold": float(threshold),
            "best_oos_f1": float(metrics["oos_f1"]),
            "best_macro_f1": float(metrics["macro_f1"]),
            "best_oos_precision": float(metrics["oos_precision"]),
            "best_oos_recall": float(metrics["oos_recall"]),
            "best_known_intent_accuracy": float(metrics["known_intent_accuracy"]),
            "best_gate_id_recall": float(metrics["gate_id_recall"]),
            "candidate_count": len(scores),
            "min_oos_recall": None if min_oos_recall is None else float(min_oos_recall),
            "_objective": objective_value,
        }
        if best is None or item["_objective"] > best["_objective"]:
            best = item

    assert best is not None
    best.pop("_objective", None)
    return best


def _score_id_rescue_candidates(
    *,
    pipeline: Any,
    texts: List[str],
    predictions: List[Dict[str, Any]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Annotate final-OOS predictions with closed-set expert rescue scores."""
    scored = [dict(row) for row in predictions]
    rescue_indices = [idx for idx, row in enumerate(scored) if int(row.get("gate_pred", 1)) == 1]
    if not rescue_indices:
        return scored

    rescue_texts = [texts[idx] for idx in rescue_indices]
    router_out = pipeline._router_predict(rescue_texts, batch_size=batch_size)
    predicted_domains = [
        pipeline.domain_id_to_name[int(domain_id)] for domain_id in router_out["domain_ids"]
    ]

    domain_to_local_indices: Dict[str, List[int]] = {}
    for local_idx, domain_name in enumerate(predicted_domains):
        domain_to_local_indices.setdefault(domain_name, []).append(local_idx)

    expert_by_local: Dict[int, Dict[str, Any]] = {}
    for domain_name, local_indices in domain_to_local_indices.items():
        group_texts = [rescue_texts[idx] for idx in local_indices]
        expert_out = pipeline._expert_predict_group(
            domain_name=domain_name,
            texts=group_texts,
            batch_size=batch_size,
        )
        for group_pos, local_idx in enumerate(local_indices):
            expert_by_local[local_idx] = {
                "intent_id": int(expert_out["intent_ids"][group_pos]),
                "intent": expert_out["intent_names"][group_pos],
                "intent_prob": float(expert_out["intent_probs"][group_pos]),
            }

    for local_idx, global_idx in enumerate(rescue_indices):
        expert = expert_by_local[local_idx]
        domain_prob = float(router_out["domain_probs"][local_idx])
        intent_prob = float(expert["intent_prob"])
        scored[global_idx]["rescue_domain_id"] = int(router_out["domain_ids"][local_idx])
        scored[global_idx]["rescue_domain"] = predicted_domains[local_idx]
        scored[global_idx]["rescue_domain_prob"] = domain_prob
        scored[global_idx]["rescue_intent_id"] = int(expert["intent_id"])
        scored[global_idx]["rescue_intent"] = expert["intent"]
        scored[global_idx]["rescue_intent_prob"] = intent_prob
        scored[global_idx]["rescue_score"] = float(domain_prob * intent_prob)

    return scored


def export_hard_intent_list(intent_decomp_path: str, reject_threshold: float) -> List[str]:
    """Extract hard intents with gate reject rate above threshold from decomposition files."""
    payload = _load_json(Path(intent_decomp_path))

    candidates = payload.get("all_intents_sorted_by_f1")
    if candidates is None:
        candidates = payload.get("bottom_10_by_f1", [])

    hard_intents: List[str] = []
    for row in candidates:
        reject_rate = float(row.get("gate_reject_rate", 0.0))
        intent_name = str(row.get("intent", "")).strip()
        if intent_name and reject_rate >= reject_threshold:
            hard_intents.append(intent_name)

    deduplicated = sorted(set(hard_intents))
    return deduplicated


def _evaluate(records: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(records) == len(preds)

    true_gate = np.array([int(row["label"]) for row in records])
    pred_gate = np.array([int(row["gate_pred"]) for row in preds])
    fast_pred_gate = np.array(
        [int(row.get("fast_gate_pred", row["gate_pred"])) for row in preds],
        dtype=np.int64,
    )

    id_mask = true_gate == 0
    oos_mask = true_gate == 1

    gate_id_recall = float(np.mean(pred_gate[id_mask] == 0)) if np.any(id_mask) else 0.0
    gate_oos_rejection = float(np.mean(pred_gate[oos_mask] == 1)) if np.any(oos_mask) else 0.0
    fast_gate_id_recall = float(np.mean(fast_pred_gate[id_mask] == 0)) if np.any(id_mask) else 0.0
    fast_gate_oos_rejection = float(np.mean(fast_pred_gate[oos_mask] == 1)) if np.any(oos_mask) else 0.0

    semantic_override_delta = {
        "uncertain_count": 0,
        "changed_to_id": 0,
        "changed_to_oos": 0,
        "id_false_reject_before_semantic": int(np.sum((true_gate == 0) & (fast_pred_gate == 1))),
        "id_false_reject_after_semantic": int(np.sum((true_gate == 0) & (pred_gate == 1))),
        "oos_false_accept_before_semantic": int(np.sum((true_gate == 1) & (fast_pred_gate == 0))),
        "oos_false_accept_after_semantic": int(np.sum((true_gate == 1) & (pred_gate == 0))),
    }
    for pred in preds:
        fast_pred = int(pred.get("fast_gate_pred", pred["gate_pred"]))
        final_pred = int(pred["gate_pred"])
        gate_stage = str(pred.get("gate_stage", "fast_gate"))
        if gate_stage != "fast_gate" or fast_pred != final_pred:
            semantic_override_delta["uncertain_count"] += 1
        if fast_pred == 1 and final_pred == 0:
            semantic_override_delta["changed_to_id"] += 1
        elif fast_pred == 0 and final_pred == 1:
            semantic_override_delta["changed_to_oos"] += 1

    true_labels: List[str] = []
    pred_labels: List[str] = []
    known_true_labels: List[str] = []
    known_pred_labels: List[str] = []

    known_intent_correct = 0
    known_domain_correct = 0
    known_total = int(np.sum(id_mask))

    known_pass_gate_total = 0
    known_intent_correct_given_gate_pass = 0
    known_domain_correct_given_gate_pass = 0

    gate_false_reject_id = 0
    gate_false_accept_oos = 0

    router_error_given_gate_pass = 0
    expert_error_given_router_correct = 0

    domain_conf_correct: List[int] = []
    domain_conf_values: List[float] = []
    intent_conf_correct: List[int] = []
    intent_conf_values: List[float] = []

    intent_id_total: Dict[str, int] = defaultdict(int)
    intent_id_gate_false_reject: Dict[str, int] = defaultdict(int)
    intent_oos_total: Dict[str, int] = defaultdict(int)
    intent_oos_gate_false_accept: Dict[str, int] = defaultdict(int)
    intent_gate_distance_sum: Dict[str, float] = defaultdict(float)
    intent_gate_radius_sum: Dict[str, float] = defaultdict(float)
    oos_source_total: Dict[str, int] = defaultdict(int)
    oos_source_false_accept: Dict[str, int] = defaultdict(int)

    gate_scores: List[float] = []
    gate_score_labels: List[int] = []

    overall_correct = 0

    for row, pred in zip(records, preds):
        true_is_oos = int(row["label"]) == 1
        pred_is_oos = bool(pred["is_oos"])
        gate_score = pred.get("gate_score")
        if gate_score is not None:
            gate_scores.append(float(gate_score))
            gate_score_labels.append(1 if true_is_oos else 0)

        if true_is_oos:
            true_oos_intent = str(row.get("intent", OOS_LABEL))
            source_split = str(row.get("source_split", row.get("split", ""))).lower()
            if source_split.startswith("id-oos"):
                oos_source_key = "id_oos"
            elif source_split.startswith("ood-oos"):
                oos_source_key = "ood_oos"
            elif source_split.startswith("heldout_oos"):
                oos_source_key = "heldout_oos"
            elif source_split.startswith("clinc_oos"):
                oos_source_key = "clinc_oos"
            elif source_split in {"oos_val", "oos_test", "oos_train"} or true_oos_intent == OOS_LABEL:
                oos_source_key = "native_oos"
            else:
                oos_source_key = "other_oos"
            oos_source_total[oos_source_key] += 1
            intent_oos_total[true_oos_intent] += 1
            true_labels.append(OOS_LABEL)
            pred_labels.append(OOS_LABEL if pred_is_oos else str(pred.get("intent", "")))
            if pred_is_oos:
                overall_correct += 1
            else:
                gate_false_accept_oos += 1
                oos_source_false_accept[oos_source_key] += 1
                intent_oos_gate_false_accept[true_oos_intent] += 1
            continue

        true_intent = str(row["intent"])
        true_domain = str(row["domain"])
        intent_id_total[true_intent] += 1

        gate_distance = pred.get("gate_distance")
        if gate_distance is not None:
            intent_gate_distance_sum[true_intent] += float(gate_distance)

        gate_radius = pred.get("gate_radius")
        if gate_radius is not None:
            intent_gate_radius_sum[true_intent] += float(gate_radius)

        true_labels.append(true_intent)
        pred_intent = str(pred.get("intent", OOS_LABEL)) if not pred_is_oos else OOS_LABEL
        pred_labels.append(pred_intent)
        known_true_labels.append(true_intent)
        known_pred_labels.append(pred_intent)

        if pred_is_oos:
            gate_false_reject_id += 1
            intent_id_gate_false_reject[true_intent] += 1
            continue

        known_pass_gate_total += 1

        if pred_intent == true_intent:
            known_intent_correct += 1
            overall_correct += 1
            known_intent_correct_given_gate_pass += 1

        pred_domain = str(pred.get("domain", ""))
        domain_is_correct = pred_domain == true_domain
        if domain_is_correct:
            known_domain_correct += 1
            known_domain_correct_given_gate_pass += 1
        else:
            router_error_given_gate_pass += 1

        if domain_is_correct and pred_intent != true_intent:
            expert_error_given_router_correct += 1

        domain_prob = pred.get("domain_prob")
        if domain_prob is not None:
            domain_conf_values.append(float(domain_prob))
            domain_conf_correct.append(int(domain_is_correct))

        intent_prob = pred.get("intent_prob")
        if intent_prob is not None:
            intent_conf_values.append(float(intent_prob))
            intent_conf_correct.append(int(pred_intent == true_intent))

    overall_accuracy = float(overall_correct / len(records)) if len(records) else 0.0
    known_intent_accuracy = float(known_intent_correct / known_total) if known_total else 0.0
    domain_accuracy = float(known_domain_correct / known_total) if known_total else 0.0
    intent_acc_given_gate_pass = _safe_div(known_intent_correct_given_gate_pass, known_pass_gate_total)
    domain_acc_given_gate_pass = _safe_div(known_domain_correct_given_gate_pass, known_pass_gate_total)
    id_throughput_after_gate = _safe_div(known_pass_gate_total, known_total)

    macro_f1 = float(f1_score(true_labels, pred_labels, average="macro"))
    known_label_space = sorted(intent_id_total.keys())
    known_macro_f1 = (
        float(
            f1_score(
                known_true_labels,
                known_pred_labels,
                labels=known_label_space,
                average="macro",
                zero_division=0,
            )
        )
        if known_true_labels
        else 0.0
    )

    oos_true_binary = (true_gate == 1).astype(np.int64)
    oos_pred_binary = np.array([int(bool(row["is_oos"])) for row in preds], dtype=np.int64)

    oos_precision, oos_recall, oos_f1, _ = precision_recall_fscore_support(
        oos_true_binary,
        oos_pred_binary,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    id_precision, id_recall, id_f1, _ = precision_recall_fscore_support(
        oos_true_binary,
        oos_pred_binary,
        average="binary",
        pos_label=0,
        zero_division=0,
    )

    domain_conf_array = np.asarray(domain_conf_values, dtype=np.float64)
    domain_correct_array = np.asarray(domain_conf_correct, dtype=np.int64)
    intent_conf_array = np.asarray(intent_conf_values, dtype=np.float64)
    intent_correct_array = np.asarray(intent_conf_correct, dtype=np.int64)

    calibration = {
        "domain": {
            "ece": _ece_binary(domain_conf_array, domain_correct_array),
            "brier": _brier_binary(domain_conf_array, domain_correct_array),
            "num_samples": int(domain_conf_array.size),
            "confidence_summary": _confidence_summary(domain_conf_array),
        },
        "intent": {
            "ece": _ece_binary(intent_conf_array, intent_correct_array),
            "brier": _brier_binary(intent_conf_array, intent_correct_array),
            "num_samples": int(intent_conf_array.size),
            "confidence_summary": _confidence_summary(intent_conf_array),
        },
        "gate": {
            "ece": None,
            "brier": None,
            "num_samples": len(records),
            "note": "gate confidence score is not exposed by current detector output",
        },
    }

    cascade_error_breakdown = {
        "known": {
            "total": known_total,
            "gate_false_reject": gate_false_reject_id,
            "gate_false_reject_rate": _safe_div(gate_false_reject_id, known_total),
            "passed_gate": known_pass_gate_total,
            "passed_gate_rate": id_throughput_after_gate,
            "router_error_given_gate_pass": router_error_given_gate_pass,
            "router_error_rate_given_gate_pass": _safe_div(
                router_error_given_gate_pass, known_pass_gate_total
            ),
            "expert_error_given_router_correct": expert_error_given_router_correct,
            "expert_error_rate_given_router_correct": _safe_div(
                expert_error_given_router_correct,
                known_domain_correct_given_gate_pass,
            ),
        },
        "oos": {
            "total": int(np.sum(oos_mask)),
            "gate_false_accept": gate_false_accept_oos,
            "gate_false_accept_rate": _safe_div(gate_false_accept_oos, int(np.sum(oos_mask))),
        },
    }

    open_set_metrics = {
        "auroc": None,
        "aupr_in": None,
        "aupr_out": None,
        "note": "requires continuous oos score; current output has binary gate_pred only",
    }
    if len(gate_scores) == len(records) and len(set(gate_score_labels)) > 1:
        y_true = np.asarray(gate_score_labels, dtype=np.int64)
        y_score_oos = np.asarray(gate_scores, dtype=np.float64)
        y_score_id = -y_score_oos
        open_set_metrics = {
            "auroc": float(roc_auc_score(y_true, y_score_oos)),
            "aupr_in": float(average_precision_score((1 - y_true), y_score_id)),
            "aupr_out": float(average_precision_score(y_true, y_score_oos)),
            "note": "computed from continuous gate_score=distance/radius",
        }

    id_intent_false_reject_stats: List[Dict[str, Any]] = []
    for intent_name, total in intent_id_total.items():
        false_reject = int(intent_id_gate_false_reject.get(intent_name, 0))
        id_intent_false_reject_stats.append(
            {
                "intent": intent_name,
                "support": int(total),
                "gate_false_reject": false_reject,
                "gate_false_reject_rate": _safe_div(false_reject, int(total)),
                "avg_distance": _safe_div(intent_gate_distance_sum.get(intent_name, 0.0), int(total)),
                "avg_radius": _safe_div(intent_gate_radius_sum.get(intent_name, 0.0), int(total)),
            }
        )
    id_intent_false_reject_stats.sort(
        key=lambda row: (row["gate_false_reject_rate"], row["support"]),
        reverse=True,
    )

    oos_intent_false_accept_stats: List[Dict[str, Any]] = []
    for intent_name, total in intent_oos_total.items():
        false_accept = int(intent_oos_gate_false_accept.get(intent_name, 0))
        oos_intent_false_accept_stats.append(
            {
                "intent": intent_name,
                "support": int(total),
                "gate_false_accept": false_accept,
                "gate_false_accept_rate": _safe_div(false_accept, int(total)),
            }
        )
    oos_intent_false_accept_stats.sort(
        key=lambda row: (row["gate_false_accept_rate"], row["support"]),
        reverse=True,
    )

    intent_diagnostics = {
        "known_id_gate_false_reject_by_intent": id_intent_false_reject_stats,
        "oos_gate_false_accept_by_intent": oos_intent_false_accept_stats,
    }

    oos_by_source: Dict[str, Dict[str, Any]] = {}
    for source_key, total in sorted(oos_source_total.items()):
        false_accept = int(oos_source_false_accept.get(source_key, 0))
        oos_by_source[source_key] = {
            "count": int(total),
            "gate_false_accept": false_accept,
            "gate_false_accept_rate": _safe_div(false_accept, int(total)),
            "gate_oos_rejection": _safe_div(int(total) - false_accept, int(total)),
        }

    primary_metrics = {
        "overall_accuracy": overall_accuracy,
        "known_accuracy": known_intent_accuracy,
        "oos_accuracy": gate_oos_rejection,
        "macro_f1": macro_f1,
        "known_macro_f1": known_macro_f1,
        "oos_f1": float(oos_f1),
    }

    return {
        "fast_gate_metrics": {
            "gate_id_recall": fast_gate_id_recall,
            "gate_oos_rejection": fast_gate_oos_rejection,
        },
        "post_semantic_metrics": {
            "gate_id_recall": gate_id_recall,
            "gate_oos_rejection": gate_oos_rejection,
        },
        "semantic_override_delta": semantic_override_delta,
        "gate_id_recall": gate_id_recall,
        "gate_oos_rejection": gate_oos_rejection,
        "oos_accuracy": gate_oos_rejection,
        "oos_precision": float(oos_precision),
        "oos_recall": float(oos_recall),
        "oos_f1": float(oos_f1),
        "id_precision": float(id_precision),
        "id_recall": float(id_recall),
        "id_f1": float(id_f1),
        "overall_accuracy": overall_accuracy,
        "known_accuracy": known_intent_accuracy,
        "known_intent_accuracy": known_intent_accuracy,
        "known_macro_f1": known_macro_f1,
        "domain_accuracy": domain_accuracy,
        "id_throughput_after_gate": id_throughput_after_gate,
        "intent_acc_given_gate_pass": intent_acc_given_gate_pass,
        "domain_acc_given_gate_pass": domain_acc_given_gate_pass,
        "macro_f1": macro_f1,
        "primary_metrics": primary_metrics,
        "cascade_error_breakdown": cascade_error_breakdown,
        "intent_diagnostics": intent_diagnostics,
        "oos_by_source": oos_by_source,
        "calibration": calibration,
        "open_set_metrics": open_set_metrics,
        "counts": {
            "total": len(records),
            "known_id": known_total,
            "oos": int(np.sum(oos_mask)),
        },
    }


def _cascade_error_stage(record: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    """为每条样本标记唯一 Cascade 错误阶段，便于归因而非只看总分。"""

    true_oos = int(record["label"]) == 1
    predicted_oos = bool(prediction.get("is_oos", prediction.get("gate_pred", 1)))
    if true_oos:
        return "correct_oos_rejection" if predicted_oos else "oos_accepted_by_gate"
    if predicted_oos:
        return "known_rejected_by_gate"
    if str(prediction.get("domain", "")) != str(record.get("domain", "")):
        return "known_wrong_domain"
    if str(prediction.get("intent", OOS_LABEL)) != str(record.get("intent", "")):
        return "known_wrong_expert"
    return "correct_known_prediction"


def _cascade_error_decomposition(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从样本级路径汇总计数和比例，不重新推理。"""

    stages = (
        "correct_oos_rejection",
        "oos_accepted_by_gate",
        "known_rejected_by_gate",
        "known_wrong_domain",
        "known_wrong_expert",
        "correct_known_prediction",
    )
    counts = {stage: sum(row.get("error_stage") == stage for row in predictions) for stage in stages}
    total = max(len(predictions), 1)
    return {"counts": counts, "rates": {stage: float(count / total) for stage, count in counts.items()}}


def _build_gate_diagnostics(
    metrics: Dict[str, Any],
    preds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build compact gate diagnostics for Phase 1 analysis export."""
    counts = metrics.get("counts", {})
    cascade = metrics.get("cascade_error_breakdown", {})
    known_cascade = cascade.get("known", {})
    oos_cascade = cascade.get("oos", {})

    fn_known = int(known_cascade.get("gate_false_reject", 0))
    fp_oos = int(oos_cascade.get("gate_false_accept", 0))
    total_known = int(counts.get("known_id", 0))
    total_oos = int(counts.get("oos", 0))

    gate_scores: List[float] = []
    gate_labels: List[int] = []
    for row in preds:
        score = row.get("gate_score")
        if score is None:
            continue
        gate_scores.append(float(score))
        gate_labels.append(int(row.get("true_gate_label", -1)))

    score_stats = {
        "available": len(gate_scores) > 0,
        "id": None,
        "oos": None,
        "recommended_uncertain_interval": None,
    }

    if len(gate_scores) > 0:
        scores = np.asarray(gate_scores, dtype=np.float64)
        labels = np.asarray(gate_labels, dtype=np.int64)
        id_scores = scores[labels == 0]
        oos_scores = scores[labels == 1]

        def _dist(arr: np.ndarray) -> Optional[Dict[str, float]]:
            if arr.size == 0:
                return None
            return {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
            }

        score_stats["id"] = _dist(id_scores)
        score_stats["oos"] = _dist(oos_scores)

        if id_scores.size > 0 and oos_scores.size > 0:
            overlap_low = float(np.percentile(id_scores, 90))
            overlap_high = float(np.percentile(oos_scores, 10))
            if overlap_low <= overlap_high:
                score_stats["recommended_uncertain_interval"] = {
                    "low": overlap_low,
                    "high": overlap_high,
                    "source": "id_p90_to_oos_p10",
                }

    return {
        "summary": {
            "fn_known": fn_known,
            "fp_oos": fp_oos,
            "true_known": max(total_known - fn_known, 0),
            "true_oos": max(total_oos - fp_oos, 0),
            "fn_known_rate": _safe_div(fn_known, total_known),
            "fp_oos_rate": _safe_div(fp_oos, total_oos),
            "gate_id_recall": float(metrics.get("gate_id_recall", 0.0)),
            "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
            "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
        },
        "top10_known_fn_intents": metrics.get("intent_diagnostics", {}).get(
            "known_id_gate_false_reject_by_intent", []
        )[:10],
        "score_distribution": score_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HiLSA-MoE v19 pipeline")
    parser.add_argument("--model_path", default=str(PATHS.smollm135m))
    parser.add_argument("--gate_encoder_path", default=str(PATHS.minilm))
    parser.add_argument(
        "--gate_encoder_checkpoint_path",
        default=None,
        help="可选的 AutoModel encoder.pt，用于 CE/SupCon/CE-Recon Gate 表示。",
    )
    parser.add_argument(
        "--gate_detector_path",
        default=str(PATHS.artifact_root / "outputs/gate_production/detector.json"),
    )
    parser.add_argument(
        "--router_ckpt",
        default=str(PATHS.artifact_root / "outputs/experiments/components/router/router_v19/best_model.pt"),
    )
    parser.add_argument(
        "--experts_root",
        default=str(PATHS.artifact_root / "outputs/experiments/components/experts/experts_v19"),
    )
    parser.add_argument("--experts_data_root", default=str(PATHS.prepared_data_root / "v19/experts"))
    parser.add_argument("--router_train", default=str(PATHS.prepared_data_root / "v19/router/train.json"))
    parser.add_argument("--gate_train", default=str(PATHS.prepared_data_root / "v19/gate/train.json"))
    parser.add_argument("--gate_val", default=str(PATHS.prepared_data_root / "v19/gate/val.json"))
    parser.add_argument("--gate_test", default=str(PATHS.prepared_data_root / "v19/gate/test.json"))
    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional dataset root override (expects gate/router/experts under this directory).",
    )
    parser.add_argument(
        "--data_root_scope",
        default="gate_only",
        choices=["gate_only", "all"],
        help="When data_root is set, override gate paths only (gate_only) or gate/router/experts paths (all).",
    )
    parser.add_argument(
        "--gate_mode",
        default="multisphere",
        choices=["multisphere", "multi_prototype", "linear_baseline"],
        help="Fast gate mode used before router/expert stages.",
    )
    parser.add_argument(
        "--gate_baseline_path",
        default=None,
        help="linear_baseline 模式使用的 validation-selected 线性 Gate pickle。",
    )
    parser.add_argument(
        "--gate_radius_scale",
        type=float,
        default=1.0,
        help="Eval-time scale applied to multisphere radii; values below 1.0 make OOS rejection stricter.",
    )
    parser.add_argument(
        "--multi_prototype_path",
        default=str(PATHS.artifact_root / "outputs/multi_prototypes_v19/prototypes.json"),
        help="Prototype payload path used when gate_mode=multi_prototype.",
    )
    parser.add_argument(
        "--multi_proto_id_threshold",
        type=float,
        default=0.80,
        help="ID accept threshold in similarity space for multi_prototype gate.",
    )
    parser.add_argument(
        "--multi_proto_threshold_mode",
        default="fixed",
        choices=["fixed", "val_macro_f1"],
        help="Threshold strategy for multi_prototype: fixed value or tune on gate_val split.",
    )
    parser.add_argument("--output_dir", default=str(PATHS.artifact_root / "outputs/pipeline_v19"))
    parser.add_argument(
        "--run_manifest_out",
        default=None,
        help="Optional path to write run manifest JSON. Defaults to <output_dir>/run_manifest.json.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--semantic_gate_enabled", action="store_true")
    parser.add_argument(
        "--semantic_gate_mode",
        default="prototype",
        choices=["none", "prototype", "verifier", "llm_verifier", "fusion"],
    )
    parser.add_argument("--semantic_uncertain_low", type=float, default=0.98)
    parser.add_argument("--semantic_uncertain_high", type=float, default=1.05)
    parser.add_argument("--semantic_gate_threshold", type=float, default=0.45)
    parser.add_argument("--semantic_top_k", type=int, default=3)
    parser.add_argument("--semantic_prompt_version", default="ranking_v1")
    parser.add_argument(
        "--semantic_tuning_mode",
        default="fixed",
        choices=["fixed", "val_macro_f1"],
    )
    parser.add_argument("--semantic_fusion_alpha", type=float, default=0.7)
    parser.add_argument("--semantic_fusion_beta", type=float, default=0.3)
    parser.add_argument(
        "--semantic_decision_policy",
        default="threshold",
        choices=["threshold", "two_stage_verifier"],
    )
    parser.add_argument("--semantic_low_conf_threshold", type=float, default=0.80)
    parser.add_argument("--semantic_high_conf_threshold", type=float, default=0.90)
    parser.add_argument("--semantic_verifier_threshold", type=float, default=0.50)
    parser.add_argument("--prompt_semantic_verifier_ckpt", default=None)
    parser.add_argument("--semantic_verifier_ckpt", default=None)
    parser.add_argument("--semantic_verifier_lora_r", type=int, default=16)
    parser.add_argument("--semantic_verifier_lora_alpha", type=int, default=32)
    parser.add_argument("--prototype_centers_default", type=int, default=1)
    parser.add_argument("--prototype_overrides_path", default=None)
    parser.add_argument(
        "--compare_baseline_eval",
        default=None,
        help="Optional path to baseline eval_results.json for delta table output.",
    )
    parser.add_argument(
        "--compare_candidate_eval",
        default=None,
        help="Optional path to candidate eval_results.json for delta table output.",
    )
    parser.add_argument(
        "--intent_decomposition_path",
        default=None,
        help="Optional path to intent_error_decomposition.json for hard intent export.",
    )
    parser.add_argument(
        "--hard_intent_threshold",
        type=float,
        default=0.25,
        help="Gate reject-rate threshold used when exporting hard intents.",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Run utility exports only (comparison / hard intents) without loading pipeline models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260317,
        help="Reserved for deterministic protocol tracking.",
    )
    parser.add_argument(
        "--ablation_no_gate",
        action="store_true",
        help="Pipeline ablation: bypass gate decisions and force all samples through Router->Expert.",
    )
    parser.add_argument(
        "--no_gate_mode",
        default="disabled",
           choices=["disabled", "router_confidence", "intent_confidence", "oracle_oos", "random"],
        help="How to detect OOS when Gate is disabled: 'disabled' (no OOS detection), "
               "'router_confidence' (use Router max prob threshold), "
               "'intent_confidence' (use Expert intent confidence threshold), 'oracle_oos' (use true labels), "
               "'random' (random Bernoulli gate).",
    )
    parser.add_argument(
        "--router_confidence_threshold",
        type=float,
        default=0.5,
        help="Threshold for Router confidence OOS detection. If Router max_prob < threshold, mark as OOS.",
    )
    parser.add_argument(
        "--router_confidence_threshold_source",
        default=None,
        help="Optional validation-only threshold source path for no_gate_confidence.",
    )
    parser.add_argument(
        "--random_gate_prob",
        type=float,
        default=0.5,
        help="P(OOS)=random_gate_prob when no_gate_mode=random.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=20260324,
        help="Random seed for no_gate_mode=random.",
    )
    parser.add_argument(
        "--oracle_oos_enabled",
        action="store_true",
        help="Use oracle OOS labels from ground truth (used with no_gate_mode=oracle_oos).",
    )
    parser.add_argument(
        "--export_gate_diagnostics",
        action="store_true",
        help="Export gate diagnostics and include it in eval_results.json.",
    )
    parser.add_argument(
        "--id_rescue_enabled",
        action="store_true",
        help="Post-process final OOS predictions with high-confidence Router/Expert ID rescue.",
    )
    parser.add_argument(
        "--id_rescue_threshold",
        type=float,
        default=0.95,
        help="Rescue score threshold when id_rescue_tuning_mode=fixed.",
    )
    parser.add_argument(
        "--id_rescue_tuning_mode",
        default="fixed",
        choices=["fixed", "val_oos_f1", "val_macro_f1", "val_oos_f1_recall_guard"],
        help="Validation objective for tuning the ID rescue threshold.",
    )
    parser.add_argument(
        "--id_rescue_min_oos_recall",
        type=float,
        default=None,
        help="Minimum validation OOS recall for id_rescue_tuning_mode=val_oos_f1_recall_guard.",
    )
    args = parser.parse_args()

    args.semantic_gate_mode = _normalize_semantic_gate_mode(str(args.semantic_gate_mode))

    if args.semantic_gate_mode == "none":
        args.semantic_gate_enabled = False

    if args.data_root:
        data_root = Path(args.data_root)
        args.gate_train = str(data_root / "gate" / "train.json")
        args.gate_val = str(data_root / "gate" / "val.json")
        args.gate_test = str(data_root / "gate" / "test.json")
        if args.data_root_scope == "all":
            args.router_train = str(data_root / "router" / "train.json")
            args.experts_data_root = str(data_root / "experts")

    args.output_dir = _normalize_legacy_output_dir(str(args.output_dir))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_verifier_supplied = bool(args.prompt_semantic_verifier_ckpt)
    prompt_verifier_used = bool(
        prompt_verifier_supplied and str(args.semantic_gate_mode) in {"llm_verifier", "fusion"}
    )
    if prompt_verifier_supplied and not prompt_verifier_used:
        LOGGER.warning(
            "prompt_semantic_verifier_ckpt was supplied but semantic_gate_mode=%s will not load it.",
            str(args.semantic_gate_mode),
        )

    if not args.skip_inference:
        from legacy.pipeline.system_pipeline import HiLSAMoEV19Pipeline, PipelinePaths  # noqa: E402

        prototype_overrides: Dict[str, int] = {}
        if args.prototype_overrides_path:
            overrides_payload = _load_json(Path(args.prototype_overrides_path))
            raw_overrides = overrides_payload.get("overrides", overrides_payload)
            prototype_overrides = {
                str(intent): int(value)
                for intent, value in raw_overrides.items()
            }

        paths = PipelinePaths(
            model_path=Path(args.model_path),
            gate_encoder_path=Path(args.gate_encoder_path),
            gate_detector_path=Path(args.gate_detector_path),
            router_ckpt_path=Path(args.router_ckpt),
            experts_root=Path(args.experts_root),
            experts_data_root=Path(args.experts_data_root),
            router_data_path=Path(args.router_train),
            gate_train_path=Path(args.gate_train),
            gate_encoder_checkpoint_path=Path(args.gate_encoder_checkpoint_path)
            if args.gate_encoder_checkpoint_path
            else None,
            gate_baseline_path=Path(args.gate_baseline_path) if args.gate_baseline_path else None,
            prompt_semantic_verifier_ckpt_path=Path(args.prompt_semantic_verifier_ckpt)
            if args.prompt_semantic_verifier_ckpt
            else None,
            semantic_verifier_ckpt_path=Path(args.semantic_verifier_ckpt)
            if args.semantic_verifier_ckpt
            else None,
            multi_prototype_path=Path(args.multi_prototype_path)
            if args.multi_prototype_path
            else None,
        )

        pipeline = HiLSAMoEV19Pipeline(
            paths=paths,
            device=args.device,
            semantic_gate_enabled=bool(args.semantic_gate_enabled),
            semantic_gate_mode=str(args.semantic_gate_mode),
            semantic_uncertain_low=float(args.semantic_uncertain_low),
            semantic_uncertain_high=float(args.semantic_uncertain_high),
            semantic_gate_threshold=float(args.semantic_gate_threshold),
            semantic_top_k=int(args.semantic_top_k),
            semantic_prompt_version=str(args.semantic_prompt_version),
            prototype_centers_default=int(args.prototype_centers_default),
            prototype_centers_overrides=prototype_overrides,
            semantic_fusion_alpha=float(args.semantic_fusion_alpha),
            semantic_fusion_beta=float(args.semantic_fusion_beta),
            semantic_decision_policy=str(args.semantic_decision_policy),
            semantic_low_conf_threshold=float(args.semantic_low_conf_threshold),
            semantic_high_conf_threshold=float(args.semantic_high_conf_threshold),
            semantic_verifier_threshold=float(args.semantic_verifier_threshold),
            semantic_verifier_lora_r=int(args.semantic_verifier_lora_r),
            semantic_verifier_lora_alpha=int(args.semantic_verifier_lora_alpha),
            gate_mode=str(args.gate_mode),
            multi_proto_id_threshold=float(args.multi_proto_id_threshold),
            gate_radius_scale=float(args.gate_radius_scale),
        )
        pipeline.load()

        threshold_tuning_report: Optional[Dict[str, Any]] = None
        semantic_tuning_report: Optional[Dict[str, Any]] = None
        if args.gate_mode == "multi_prototype" and args.multi_proto_threshold_mode == "val_macro_f1":
            gate_val_records = _load_json(Path(args.gate_val))
            tuning = _tune_multi_proto_id_threshold(
                pipeline=pipeline,
                gate_val_records=gate_val_records,
            )
            pipeline.multi_proto_id_threshold = float(tuning["best_threshold"])
            args.multi_proto_id_threshold = float(tuning["best_threshold"])
            threshold_tuning_report = {
                "mode": "val_macro_f1",
                "gate_val_path": str(Path(args.gate_val)),
                **tuning,
            }
            LOGGER.info(
                "Tuned multi-prototype threshold on validation: threshold=%.6f, macro_f1=%.6f, range=[%.6f, %.6f]",
                float(tuning["best_threshold"]),
                float(tuning["best_macro_f1"]),
                float(tuning["search_min"]),
                float(tuning["search_max"]),
            )

        if (
            bool(args.semantic_gate_enabled)
            and str(args.semantic_gate_mode) in {"prototype", "llm_verifier", "fusion"}
            and str(args.semantic_tuning_mode) == "val_macro_f1"
        ):
            gate_val_records = _load_json(Path(args.gate_val))
            val_texts = [str(row["text"]) for row in gate_val_records]
            y_true_oos = np.asarray([int(row["label"]) for row in gate_val_records], dtype=np.int64)
            gate_out = pipeline._gate_predict(val_texts)
            fast_gate_preds = np.asarray(gate_out["pred"], dtype=np.int64)
            uncertain_indices = pipeline.route_uncertain_gate_samples(np.asarray(gate_out["score"], dtype=np.float32))
            uncertain_texts = [val_texts[idx] for idx in uncertain_indices]
            semantic_out = pipeline.smollm_semantic_gate_verify(
                uncertain_texts,
                batch_size=args.batch_size,
            )

            if str(args.semantic_gate_mode) == "prototype":
                prototype_scores = np.asarray(
                    semantic_out["semantic_id_scores"],
                    dtype=np.float32,
                )
                tuning = _tune_semantic_threshold_from_scores(
                    fast_gate_preds=fast_gate_preds,
                    uncertain_indices=uncertain_indices,
                    semantic_scores=prototype_scores,
                    y_true_oos=y_true_oos,
                )
                pipeline.semantic_gate_threshold = float(tuning["best_threshold"])
                args.semantic_gate_threshold = float(tuning["best_threshold"])
                semantic_tuning_report = {
                    "mode": "val_macro_f1",
                    "semantic_mode": "prototype",
                    "gate_val_path": str(Path(args.gate_val)),
                    "uncertain_count": len(uncertain_indices),
                    **tuning,
                }
                LOGGER.info(
                    "Tuned semantic prototype threshold on validation: threshold=%.6f, macro_f1=%.6f",
                    float(tuning["best_threshold"]),
                    float(tuning["best_macro_f1"]),
                )
            elif str(args.semantic_gate_mode) == "llm_verifier":
                verifier_scores = np.asarray(
                    semantic_out["semantic_candidate_best_score"],
                    dtype=np.float32,
                )
                tuning = _tune_semantic_threshold_from_scores(
                    fast_gate_preds=fast_gate_preds,
                    uncertain_indices=uncertain_indices,
                    semantic_scores=verifier_scores,
                    y_true_oos=y_true_oos,
                )
                pipeline.semantic_gate_threshold = float(tuning["best_threshold"])
                args.semantic_gate_threshold = float(tuning["best_threshold"])
                semantic_tuning_report = {
                    "mode": "val_macro_f1",
                    "semantic_mode": "llm_verifier",
                    "gate_val_path": str(Path(args.gate_val)),
                    "uncertain_count": len(uncertain_indices),
                    **tuning,
                }
                LOGGER.info(
                    "Tuned semantic verifier threshold on validation: threshold=%.6f, macro_f1=%.6f",
                    float(tuning["best_threshold"]),
                    float(tuning["best_macro_f1"]),
                )
            else:
                prototype_scores = np.asarray(
                    semantic_out["semantic_id_scores"],
                    dtype=np.float32,
                )
                verifier_scores = np.asarray(
                    semantic_out["semantic_candidate_best_score"],
                    dtype=np.float32,
                )
                tuning = _tune_fusion_alpha_and_threshold(
                    fast_gate_preds=fast_gate_preds,
                    uncertain_indices=uncertain_indices,
                    prototype_scores=prototype_scores,
                    verifier_scores=verifier_scores,
                    y_true_oos=y_true_oos,
                )
                pipeline.semantic_fusion_alpha = float(tuning["best_alpha"])
                pipeline.semantic_fusion_beta = float(tuning["best_beta"])
                pipeline.semantic_gate_threshold = float(tuning["best_threshold"])
                args.semantic_fusion_alpha = float(tuning["best_alpha"])
                args.semantic_fusion_beta = float(tuning["best_beta"])
                args.semantic_gate_threshold = float(tuning["best_threshold"])
                semantic_tuning_report = {
                    "mode": "val_macro_f1",
                    "semantic_mode": "fusion",
                    "gate_val_path": str(Path(args.gate_val)),
                    "uncertain_count": len(uncertain_indices),
                    **tuning,
                }
                LOGGER.info(
                    "Tuned semantic fusion on validation: alpha=%.3f beta=%.3f threshold=%.6f macro_f1=%.6f",
                    float(tuning["best_alpha"]),
                    float(tuning["best_beta"]),
                    float(tuning["best_threshold"]),
                    float(tuning["best_macro_f1"]),
                )

        id_rescue_tuning_report: Optional[Dict[str, Any]] = None
        if bool(args.id_rescue_enabled) and str(args.id_rescue_tuning_mode) != "fixed":
            gate_val_records = _load_json(Path(args.gate_val))
            val_texts = [str(row["text"]) for row in gate_val_records]
            val_predictions = pipeline.predict_batch(val_texts, batch_size=args.batch_size)
            scored_val_predictions = _score_id_rescue_candidates(
                pipeline=pipeline,
                texts=val_texts,
                predictions=val_predictions,
                batch_size=args.batch_size,
            )
            rescue_tuning = _tune_id_rescue_threshold_from_scored_predictions(
                records=gate_val_records,
                scored_predictions=scored_val_predictions,
                objective=str(args.id_rescue_tuning_mode),
                min_oos_recall=args.id_rescue_min_oos_recall,
            )
            args.id_rescue_threshold = float(rescue_tuning["best_threshold"])
            id_rescue_tuning_report = {
                "gate_val_path": str(Path(args.gate_val)),
                **rescue_tuning,
            }
            LOGGER.info(
                "Tuned ID rescue threshold on validation: threshold=%.6f oos_f1=%.6f macro_f1=%.6f",
                float(rescue_tuning["best_threshold"]),
                float(rescue_tuning["best_oos_f1"]),
                float(rescue_tuning["best_macro_f1"]),
            )

        gate_test_records = _load_json(Path(args.gate_test))
        texts = [str(row["text"]) for row in gate_test_records]

        LOGGER.info("Running end-to-end inference on %d samples...", len(texts))
        LOGGER.info(
            "Protocol guard: test split is used for FINAL evaluation only; "
            "no tuning/calibration is performed in this script."
        )
        infer_start = time.perf_counter()
        if args.ablation_no_gate:
            predictions = _predict_with_gate_disabled(
                pipeline=pipeline,
                texts=texts,
                gate_test_records=gate_test_records,
                batch_size=args.batch_size,
                no_gate_mode=args.no_gate_mode,
                router_confidence_threshold=args.router_confidence_threshold,
                random_gate_prob=args.random_gate_prob,
                random_seed=args.random_seed,
            )
            LOGGER.info(
                "No-Gate ablation: mode=%s, router_threshold=%.2f",
                args.no_gate_mode,
                args.router_confidence_threshold,
            )
        else:
            predictions = pipeline.predict_batch(texts, batch_size=args.batch_size)
        if bool(args.id_rescue_enabled):
            scored_predictions = _score_id_rescue_candidates(
                pipeline=pipeline,
                texts=texts,
                predictions=predictions,
                batch_size=args.batch_size,
            )
            predictions = _apply_id_rescue_threshold(
                scored_predictions,
                threshold=float(args.id_rescue_threshold),
            )
        infer_elapsed = float(time.perf_counter() - infer_start)

        LOGGER.info("Computing metrics...")
        metrics = _evaluate(gate_test_records, predictions)
        metrics["latency"] = {
            "inference_seconds_total": infer_elapsed,
            "avg_ms_per_sample": float((infer_elapsed * 1000.0) / max(len(texts), 1)),
            "throughput_samples_per_sec": float(len(texts) / infer_elapsed) if infer_elapsed > 0 else 0.0,
        }

        results = {
            "config": vars(args),
            "protocol": {
                "test_used_for_tuning": False,
                "test_used_for_calibration": False,
                "mode": "evaluation_only",
            },
            "metrics": metrics,
        }

        if threshold_tuning_report is not None:
            results["threshold_tuning"] = threshold_tuning_report
        if semantic_tuning_report is not None:
            results["semantic_tuning"] = semantic_tuning_report
        if id_rescue_tuning_report is not None:
            results["id_rescue_tuning"] = id_rescue_tuning_report
        if args.ablation_no_gate and args.no_gate_mode in {"router_confidence", "intent_confidence"}:
            results["threshold_source"] = {
                "type": "validation_macro_f1",
                "split": "gate_val",
                "objective": "macro_f1",
                "source_path": str(args.router_confidence_threshold_source)
                if args.router_confidence_threshold_source
                else None,
                "threshold": float(args.router_confidence_threshold),
                "mode": str(args.no_gate_mode),
            }

        merged_predictions: List[Dict[str, Any]] = []
        for row, pred in zip(gate_test_records, predictions):
            merged = {
                "text": row["text"],
                "true_intent": row["intent"],
                "true_domain": row["domain"],
                "true_gate_label": int(row["label"]),
                **pred,
            }
            merged["error_stage"] = _cascade_error_stage(row, merged)
            merged_predictions.append(merged)

        results["cascade_error_decomposition_sample_level"] = _cascade_error_decomposition(
            merged_predictions
        )

        with open(output_dir / "predictions.json", "w", encoding="utf-8") as file:
            json.dump(merged_predictions, file, indent=2, ensure_ascii=False)

        if args.export_gate_diagnostics:
            gate_diagnostics = _build_gate_diagnostics(metrics=metrics, preds=merged_predictions)
            results["gate_diagnostics"] = gate_diagnostics
            with open(output_dir / "gate_diagnostics.json", "w", encoding="utf-8") as file:
                json.dump(gate_diagnostics, file, indent=2, ensure_ascii=False)

        with open(output_dir / "eval_results.json", "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

        run_manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "seed": int(args.seed),
            "data_root": str(args.data_root) if args.data_root else None,
            "gate_mode": str(args.gate_mode),
            "gate_radius_scale": float(args.gate_radius_scale),
            "semantic_gate_enabled": bool(args.semantic_gate_enabled),
            "semantic_gate_mode": str(args.semantic_gate_mode),
            "semantic_decision_policy": str(args.semantic_decision_policy),
            "id_rescue_enabled": bool(args.id_rescue_enabled),
            "id_rescue_threshold": float(args.id_rescue_threshold),
            "id_rescue_tuning_mode": str(args.id_rescue_tuning_mode),
            "id_rescue_min_oos_recall": args.id_rescue_min_oos_recall,
            "prompt_semantic_verifier_supplied": prompt_verifier_supplied,
            "prompt_semantic_verifier_used": prompt_verifier_used,
            "semantic_top_k": int(args.semantic_top_k),
            "semantic_prompt_version": str(args.semantic_prompt_version),
            "router_confidence_threshold": float(args.router_confidence_threshold)
            if args.ablation_no_gate and args.no_gate_mode == "router_confidence"
            else None,
            "router_confidence_threshold_source": str(args.router_confidence_threshold_source)
            if args.router_confidence_threshold_source
            else None,
            "gate_detector_path": str(args.gate_detector_path),
            "gate_encoder_path": str(args.gate_encoder_path),
            "gate_encoder_checkpoint_path": str(args.gate_encoder_checkpoint_path)
            if args.gate_encoder_checkpoint_path
            else None,
            "gate_baseline_path": str(args.gate_baseline_path) if args.gate_baseline_path else None,
            "router_ckpt": str(args.router_ckpt),
            "experts_root": str(args.experts_root),
            "gate_test": str(args.gate_test),
            "metrics_preview": {
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
                "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
                "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
            },
        }
        manifest_path = Path(args.run_manifest_out) if args.run_manifest_out else output_dir / "run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(run_manifest, file, indent=2, ensure_ascii=False)

    if args.compare_baseline_eval and args.compare_candidate_eval:
        comparison = build_baseline_comparison_table(
            baseline_path=args.compare_baseline_eval,
            candidate_path=args.compare_candidate_eval,
        )
        with open(output_dir / "baseline_comparison.json", "w", encoding="utf-8") as file:
            json.dump(comparison, file, indent=2, ensure_ascii=False)
        LOGGER.info("Wrote baseline comparison -> %s", output_dir / "baseline_comparison.json")

    if args.intent_decomposition_path:
        hard_intents = export_hard_intent_list(
            intent_decomp_path=args.intent_decomposition_path,
            reject_threshold=args.hard_intent_threshold,
        )
        hard_payload = {
            "source": args.intent_decomposition_path,
            "reject_threshold": float(args.hard_intent_threshold),
            "hard_intents": hard_intents,
            "count": len(hard_intents),
        }
        with open(output_dir / "hard_intents.json", "w", encoding="utf-8") as file:
            json.dump(hard_payload, file, indent=2, ensure_ascii=False)
        LOGGER.info("Wrote hard intents -> %s", output_dir / "hard_intents.json")

    if not args.skip_inference:
        LOGGER.info("Done. Results -> %s", output_dir / "eval_results.json")
        LOGGER.info("Metrics: %s", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
