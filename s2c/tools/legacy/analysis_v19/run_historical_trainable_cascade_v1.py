#!/usr/bin/env python3
"""Evaluate the current Trainable K=1 Gate through the historical v19 downstream.

The Gate representation and detector come from
``run_trainable_minilm_historical_v1.py``.  Router and Expert checkpoints are
kept fixed to the audited H1 v19 downstream components.  This is an endpoint
bridge, not a retraining script: it answers whether the current Trainable
MiniLM Gate's OOS decision survives the old downstream Cascade contract.

The historical full-paper semantic/prototype Gate is intentionally disabled.
That missing H0 input is recorded in the output manifest instead of being
silently replaced by a different semantic module.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from protocol_v2.data.hashing import atomic_write_json
from protocol_v2.experiments.racal_v1.representation import RacalMiniLM
from tools.eval.eval_system_pipeline_v19 import _cascade_error_stage, _evaluate
from legacy.pipeline.system_pipeline import HiLSAMoEV19Pipeline, PipelinePaths


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_detector_signature(signature: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the compact historical Trainable signature into detector JSON."""

    spheres = []
    intent_to_cluster: dict[str, int] = {}
    cluster_to_intent: dict[str, str] = {}
    for item in sorted(signature["spheres"], key=lambda value: int(value["cluster_id"])):
        cluster_id = int(item["cluster_id"])
        intent = str(item["intent"])
        spheres.append(
            {
                "center": item["center"],
                "radius": float(item["radius"]),
                "cluster_id": cluster_id,
                "intent_name": intent,
                "inv_diag_cov": item.get("inv_diag_cov"),
            }
        )
        intent_to_cluster.setdefault(intent, cluster_id)
        cluster_to_intent[str(cluster_id)] = intent

    return {
        "n_clusters": len(spheres),
        "radius_quantile": 0.95,
        "radius_method": str(signature["radius_method"]),
        "radius_lambda": float(signature["radius_lambda"]),
        "center_mode": "class_centroid_mixture",
        "distance_metric": str(signature["distance_metric"]),
        "margin_gamma": None,
        "covariance_eps": 1e-6,
        "l2_normalize": True,
        "subcenters_per_intent": 1,
        "subcenters_overrides": {},
        "random_state": 42,
        "acceptance_mode": str(signature.get("acceptance_mode", "nearest_sphere")),
        "spheres": spheres,
        "intent_to_cluster": intent_to_cluster,
        "intent_to_clusters": {intent: [cluster] for intent, cluster in intent_to_cluster.items()},
        "cluster_to_intent": cluster_to_intent,
    }


class _TrainableEncoder:
    """Expose the current RACAL model through the historical ``encode`` API."""

    def __init__(self, model_path: Path, checkpoint_path: Path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_state = checkpoint["model"]
        projection_hidden_dim = int(model_state["projection.fc1.weight"].shape[0])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = RacalMiniLM(
            model_path,
            str(checkpoint["mode"]),
            projection_hidden_dim,
        )
        self.model.load_state_dict(model_state)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del show_progress_bar
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), int(batch_size)):
            batch = self.tokenizer(
                texts[start : start + int(batch_size)],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            chunks.append(self.model(batch).detach().cpu().numpy().astype(np.float32))
        if not chunks:
            return np.empty((0, 384), dtype=np.float32)
        return np.concatenate(chunks, axis=0)


def _load_trainable_predictions(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "predictions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    data_root = args.data_root.resolve()
    trainable_dir = args.trainable_run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_signature = _load_json(trainable_dir / "detector_signature.json")
    detector_state = serialize_detector_signature(gate_signature)
    detector_path = output_dir / "trainable_k1_detector.json"
    atomic_write_json(detector_path, detector_state)

    device = torch.device(args.device)
    paths = PipelinePaths(
        model_path=args.smollm_model.resolve(),
        gate_encoder_path=args.minilm_model.resolve(),
        gate_detector_path=detector_path,
        router_ckpt_path=args.router_ckpt.resolve(),
        experts_root=args.experts_root.resolve(),
        experts_data_root=(data_root / "experts").resolve(),
        router_data_path=(data_root / "router" / "train.json").resolve(),
        gate_train_path=(data_root / "gate" / "train.json").resolve(),
    )
    pipeline = HiLSAMoEV19Pipeline(
        paths=paths,
        device=str(device),
        semantic_gate_enabled=False,
        semantic_gate_mode="none",
        gate_mode="multisphere",
    )
    pipeline.load()
    pipeline.gate_encoder = _TrainableEncoder(
        args.minilm_model.resolve(),
        trainable_dir / "checkpoint.pt",
        device,
    )

    records = _load_json(data_root / "gate" / "test.json")
    texts = [str(row["text"]) for row in records]
    predictions = pipeline.predict_batch(texts, batch_size=int(args.batch_size))
    metrics = _evaluate(records, predictions)
    merged = []
    for record, prediction in zip(records, predictions):
        row = {
            "text": record["text"],
            "true_intent": record["intent"],
            "true_domain": record["domain"],
            "true_gate_label": int(record["label"]),
            **prediction,
        }
        row["error_stage"] = _cascade_error_stage(record, row)
        merged.append(row)

    trainable_predictions = _load_trainable_predictions(trainable_dir)
    if len(trainable_predictions) != len(merged):
        raise RuntimeError(
            f"Trainable Gate prediction length mismatch: {len(trainable_predictions)} vs {len(merged)}"
        )
    gate_match = [
        int(row["gate_pred"]) == int(saved["predicted_is_oos"])
        for row, saved in zip(merged, trainable_predictions)
    ]
    if not all(gate_match):
        raise RuntimeError(
            f"Converted Trainable Gate decisions differ on {len(gate_match) - sum(gate_match)} samples"
        )

    payload = {
        "schema_version": "s2c.historical_protocol_v1.trainable_cascade_v1",
        "status": "complete",
        "protocol": "H1 archived/prepared v19 data with fixed historical downstream components",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "data_root": str(data_root),
        "trainable_gate_run": str(trainable_dir),
        "representation": "last2_minilm_plus_projection",
        "gate_contract": {
            "distance_metric": gate_signature["distance_metric"],
            "radius_method": gate_signature["radius_method"],
            "radius_lambda": float(gate_signature["radius_lambda"]),
            "threshold": 1.0,
            "semantic_gate_enabled": False,
        },
        "downstream_contract": {
            "router_ckpt": str(args.router_ckpt.resolve()),
            "experts_root": str(args.experts_root.resolve()),
            "router_and_experts_fixed": True,
            "test_used_for_selection": False,
            "oos_used_for_training": False,
        },
        "gate_conversion_check": {
            "saved_gate_prediction_count": len(trainable_predictions),
            "matching_gate_decisions": int(sum(gate_match)),
            "all_gate_decisions_match": bool(all(gate_match)),
        },
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output_dir / "eval_results.json", payload)
    atomic_write_json(output_dir / "predictions.json", merged)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--trainable-run", required=True, type=Path)
    parser.add_argument("--minilm-model", required=True, type=Path)
    parser.add_argument("--smollm-model", required=True, type=Path)
    parser.add_argument("--router-ckpt", required=True, type=Path)
    parser.add_argument("--experts-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
