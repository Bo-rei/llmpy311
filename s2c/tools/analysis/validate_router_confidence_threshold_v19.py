#!/usr/bin/env python3
"""Validate downstream-confidence OOS threshold on validation split only.

This script searches an Expert intent confidence threshold on the validation split
for the "no_gate_confidence" ablation. The selected threshold is written to
JSON/TXT and can be reused by the main structure/backbone ablation runner.

Important:
- validation-only tuning
- test split is never used for threshold selection
- the score is derived from Expert intent confidence, not the Gate detector
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval.eval_system_pipeline_v19 import OOS_LABEL, _evaluate  # noqa: E402
from src.runtime import WorkspacePaths  # noqa: E402
from tools.analysis.threshold_selection_v19 import (  # noqa: E402
    balanced_known_oos_score,
    select_main_table_constrained_threshold,
)

PATHS = WorkspacePaths.discover(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = (PROJECT_ROOT / p).resolve()
    if candidate.exists():
        return candidate
    return p


def _build_router_expert_predictions(
    pipeline: Any,
    texts: List[str],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Precompute Router + Expert outputs once for a validation batch."""
    router_out = pipeline._router_predict(texts, batch_size=batch_size)
    predicted_domains = [
        pipeline.domain_id_to_name[int(domain_id)]
        for domain_id in router_out["domain_ids"]
    ]
    router_conf = np.asarray(router_out["domain_probs"], dtype=np.float32)
    intent_conf = np.zeros(len(texts), dtype=np.float32)

    base_predictions: List[Dict[str, Any]] = []
    for idx, text in enumerate(texts):
        domain_id = int(router_out["domain_ids"][idx])
        domain_name = predicted_domains[idx]
        router_probability = float(router_conf[idx])
        base_predictions.append(
            {
                "text": text,
                "is_oos": False,
                "gate_pred": 0,
                "fast_gate_pred": 0,
                "gate_score": float(1.0 - router_probability),
                "gate_distance": float(1.0 - router_probability),
                "gate_radius": 1.0,
                "gate_margin_ok": True,
                "gate_nearest_cluster": -1,
                "gate_nearest_intent": None,
                "gate_stage": "ablation_no_gate_router_confidence",
                "semantic_id_score": None,
                "semantic_gate_decision": None,
                "semantic_top_intent": None,
                "semantic_top_domain": None,
                "semantic_decision_score": None,
                "semantic_mode": pipeline.semantic_gate_mode,
                "semantic_policy": pipeline.semantic_decision_policy,
                "semantic_verifier_score": None,
                "final_gate_decision": "id",
                "domain_id": domain_id,
                "domain": domain_name,
                "domain_prob": router_probability,
                "router_confidence": router_probability,
                "no_gate_confidence": 0.0,
                "intent_id": -1,
                "intent": OOS_LABEL,
                "intent_prob": 0.0,
            }
        )

    domain_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, domain_name in enumerate(predicted_domains):
        domain_to_indices[domain_name].append(idx)

    for domain_name, indices in domain_to_indices.items():
        domain_texts = [texts[i] for i in indices]
        expert_out = pipeline._expert_predict_group(
            domain_name=domain_name,
            texts=domain_texts,
            batch_size=batch_size,
        )
        for local_idx, global_idx in enumerate(indices):
            base_predictions[global_idx]["intent_id"] = int(expert_out["intent_ids"][local_idx])
            base_predictions[global_idx]["intent"] = str(expert_out["intent_names"][local_idx])
            intent_probability = float(expert_out["intent_probs"][local_idx])
            intent_conf[global_idx] = intent_probability
            base_predictions[global_idx]["intent_prob"] = intent_probability
            base_predictions[global_idx]["no_gate_confidence"] = intent_probability
            base_predictions[global_idx]["gate_nearest_intent"] = str(
                expert_out["intent_names"][local_idx]
            )

    return base_predictions, intent_conf, router_conf


def _materialize_predictions(
    base_predictions: List[Dict[str, Any]],
    no_gate_confidence: np.ndarray,
    threshold: float,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    gate_radius = float(max(1.0 - float(threshold), 0.0))

    for idx, base_pred in enumerate(base_predictions):
        confidence = float(no_gate_confidence[idx])
        is_oos = bool(confidence < float(threshold))
        pred = dict(base_pred)
        pred["is_oos"] = is_oos
        pred["gate_pred"] = 1 if is_oos else 0
        pred["fast_gate_pred"] = 1 if is_oos else 0
        pred["gate_radius"] = gate_radius
        pred["gate_margin_ok"] = not is_oos
        pred["gate_stage"] = "ablation_no_gate_intent_confidence"
        pred["final_gate_decision"] = "oos" if is_oos else "id"
        pred["router_confidence_threshold"] = float(threshold)
        pred["no_gate_confidence"] = confidence
        pred["gate_score"] = float(1.0 - confidence)
        pred["gate_distance"] = float(1.0 - confidence)
        if is_oos:
            pred["intent_id"] = -1
            pred["intent"] = OOS_LABEL
        predictions.append(pred)

    return predictions


def _selection_score(
    metrics: Dict[str, Any],
    objective: str,
) -> float:
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    overall_accuracy = float(metrics.get("overall_accuracy", 0.0))
    known_accuracy = float(metrics.get("known_intent_accuracy", 0.0))
    oos_f1 = float(metrics.get("oos_f1", 0.0))
    if objective == "macro_f1":
        return macro_f1
    if objective == "balanced":
        return 0.0 if (known_accuracy + oos_f1) == 0 else float(2 * known_accuracy * oos_f1 / (known_accuracy + oos_f1))
    if objective == "known_priority":
        return float(0.7 * known_accuracy + 0.3 * oos_f1)
    if objective == "main_table_constrained_balanced":
        return balanced_known_oos_score(metrics)
    raise ValueError(f"Unsupported threshold objective: {objective}")


def _search_threshold(
    pipeline: Any,
    val_records: List[Dict[str, Any]],
    batch_size: int,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
    confidence_source: str,
    threshold_objective: str,
    dataset_slug: str,
    kir_tag: str,
) -> Dict[str, Any]:
    val_texts = [str(row["text"]) for row in val_records]
    base_predictions, intent_confidence, router_confidence = _build_router_expert_predictions(
        pipeline=pipeline,
        texts=val_texts,
        batch_size=batch_size,
    )
    if confidence_source == "intent":
        no_gate_confidence = intent_confidence
    elif confidence_source == "router":
        no_gate_confidence = router_confidence
    else:
        raise ValueError(f"Unsupported confidence_source: {confidence_source}")

    thresholds = np.linspace(float(threshold_min), float(threshold_max), int(threshold_steps))
    search_rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None

    for threshold in thresholds.tolist():
        predictions = _materialize_predictions(
            base_predictions=base_predictions,
            no_gate_confidence=no_gate_confidence,
            threshold=float(threshold),
        )
        metrics = _evaluate(val_records, predictions)
        row = {
            "threshold": float(threshold),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
            "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
            "oos_f1": float(metrics.get("oos_f1", 0.0)),
            "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
            "gate_id_recall": float(metrics.get("gate_id_recall", 0.0)),
            "selection_score": float(_selection_score(metrics, threshold_objective)),
        }
        search_rows.append(row)

        if best_row is None or (
            row["selection_score"] > best_row["selection_score"]
            or (
                np.isclose(row["selection_score"], best_row["selection_score"])
                and row["overall_accuracy"] > best_row["overall_accuracy"]
            )
        ):
            best_row = row

    assert best_row is not None
    selection: Dict[str, Any] | None = None
    if threshold_objective == "main_table_constrained_balanced":
        best_row, selection = select_main_table_constrained_threshold(
            search_rows,
            slug=str(dataset_slug),
            kir_tag=str(kir_tag),
        )

    return {
        "best": best_row,
        "search": search_rows,
        "selection": selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate downstream confidence threshold on validation split only"
    )
    parser.add_argument("--model_path", default=str(PATHS.smollm135m))
    parser.add_argument("--gate_encoder_path", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--gate_detector_path",
        default="outputs/core/gate_production/detector.json",
    )
    parser.add_argument(
        "--router_ckpt",
        default="outputs/experiments/components/router/router_v19/best_model.pt",
    )
    parser.add_argument(
        "--experts_root",
        default="outputs/experiments/components/experts/experts_v19",
    )
    parser.add_argument("--experts_data_root", default="data/v19/experts")
    parser.add_argument("--router_train", default="data/v19/router/train.json")
    parser.add_argument("--gate_train", default="data/v19/gate/train.json")
    parser.add_argument("--gate_val", default="data/v19/gate/val.json")
    parser.add_argument("--gate_test", default="data/v19/gate/test.json")
    parser.add_argument("--data_root", default="data/v19")
    parser.add_argument("--data_root_scope", default="all")
    parser.add_argument("--gate_mode", default="multisphere")
    parser.add_argument("--multi_prototype_path", default="outputs/multi_prototypes_v19/prototypes.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold_min", type=float, default=0.2)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_steps", type=int, default=16)
    parser.add_argument("--confidence_source", choices=["intent", "router"], default="intent")
    parser.add_argument(
        "--threshold_objective",
        choices=["macro_f1", "balanced", "known_priority", "main_table_constrained_balanced"],
        default="macro_f1",
    )
    parser.add_argument("--dataset_slug", default="")
    parser.add_argument("--kir_tag", default="")
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_records = _load_json(_resolve(args.gate_val))
    LOGGER.info("Loaded %d validation records", len(val_records))

    from src.pipeline.system_pipeline import HiLSAMoEV19Pipeline, PipelinePaths  # noqa: E402

    paths = PipelinePaths(
        model_path=_resolve(args.model_path),
        gate_encoder_path=_resolve(args.gate_encoder_path),
        gate_detector_path=_resolve(args.gate_detector_path),
        router_ckpt_path=_resolve(args.router_ckpt),
        experts_root=_resolve(args.experts_root),
        experts_data_root=_resolve(args.experts_data_root),
        router_data_path=_resolve(args.router_train),
        gate_train_path=_resolve(args.gate_train),
        semantic_verifier_ckpt_path=None,
        multi_prototype_path=_resolve(args.multi_prototype_path),
    )

    pipeline = HiLSAMoEV19Pipeline(
        paths=paths,
        device=args.device,
        semantic_gate_enabled=False,
        gate_mode=str(args.gate_mode),
    )
    pipeline.load()

    search = _search_threshold(
        pipeline=pipeline,
        val_records=val_records,
        batch_size=int(args.batch_size),
        threshold_min=float(args.threshold_min),
        threshold_max=float(args.threshold_max),
        threshold_steps=int(args.threshold_steps),
        confidence_source=str(args.confidence_source),
        threshold_objective=str(args.threshold_objective),
        dataset_slug=str(args.dataset_slug),
        kir_tag=str(args.kir_tag),
    )
    best = search["best"]

    LOGGER.info(
        "Best intent-confidence threshold=%.6f macro_f1=%.6f overall_acc=%.6f",
        best["threshold"],
        best["macro_f1"],
        best["overall_accuracy"],
    )

    validation_payload = {
        "config": vars(args),
        "protocol": {
            "mode": "validation_only",
            "selection_split": "gate_val",
            "test_used_for_tuning": False,
            "test_used_for_calibration": False,
        },
        "threshold_source": {
            "type": f"validation_{args.threshold_objective}",
            "confidence_source": "expert_intent_probability" if str(args.confidence_source) == "intent" else "router_max_probability",
            "split": "gate_val",
            "objective": str(args.threshold_objective),
            "threshold_range": {
                "min": float(args.threshold_min),
                "max": float(args.threshold_max),
                "steps": int(args.threshold_steps),
            },
        },
        "best": best,
        "search": search["search"],
        "selection": search.get("selection"),
    }

    json_path = output_dir / "intent_confidence_threshold_validation.json"
    txt_path = output_dir / "intent_confidence_threshold_best.txt"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(validation_payload, file, indent=2, ensure_ascii=False)
    txt_path.write_text(f"{best['threshold']:.10f}\n", encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "validation_split": str(args.gate_val),
        "best_threshold": float(best["threshold"]),
        "metrics_preview": {
            "macro_f1": float(best["macro_f1"]),
            "overall_accuracy": float(best["overall_accuracy"]),
            "known_intent_accuracy": float(best["known_intent_accuracy"]),
            "gate_oos_rejection": float(best["gate_oos_rejection"]),
            "selection_score": float(best["selection_score"]),
        },
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(json.dumps(manifest["metrics_preview"], ensure_ascii=False))


if __name__ == "__main__":
    main()
