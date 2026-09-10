#!/usr/bin/env python3
"""Run current H1 Trainable MiniLM through the existing Router/Expert pipeline.

The runner is evaluation-only.  It reuses the completed H1 Trainable K=1
checkpoints and the existing v19 downstream Router/Expert components, then
writes aggregate metrics and error-stage counts only.  It does not train or
modify anything under ``../artifacts``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from protocol_v2.experiments.racal_v1.representation import build_racal_model


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_H1_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_protocol_v2" / "minilm_k1"
DEFAULT_FROZEN_ROOT = DEFAULT_H1_ROOT
KIR = 0.50
CASCADE_ROOT = ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cascade_full" / "gpu_kir50"
DATA_ROOT = ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19"
MODEL_ROOT = ROOT.parent / "assets" / "models"
OUTPUT_ROOT = ROOT / "results" / "analysis" / "historical_trainable_full_pipeline"

DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
SEEDS = (13, 42, 87)
STAGES = (
    "correct_oos_rejection",
    "oos_accepted_by_gate",
    "known_rejected_by_gate",
    "known_wrong_domain",
    "known_wrong_expert",
    "correct_known_prediction",
)


def _kir_tag(kir: float | None = None) -> str:
    value = KIR if kir is None else float(kir)
    return f"kir{int(round(value * 100)):02d}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _detector_state(signature: Mapping[str, Any]) -> dict[str, Any]:
    spheres = sorted(signature["spheres"], key=lambda item: int(item["cluster_id"]))
    intents = {str(item["intent"]): int(item["cluster_id"]) for item in spheres}
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
        "acceptance_mode": str(signature["acceptance_mode"]),
        "intent_to_cluster": intents,
        "intent_to_clusters": {key: [value] for key, value in intents.items()},
        "cluster_to_intent": {str(value): key for key, value in intents.items()},
        "spheres": [
            {
                "center": item["center"],
                "radius": float(item["radius"]),
                "cluster_id": int(item["cluster_id"]),
                "intent_name": str(item["intent"]),
                "inv_diag_cov": item["inv_diag_cov"],
            }
            for item in spheres
        ],
    }


class _RacalGateEncoder:
    """Expose the H1 Trainable checkpoint through the pipeline encode API."""

    def __init__(self, model_path: Path, checkpoint_path: Path, device: torch.device) -> None:
        from transformers import AutoTokenizer

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        hidden_dim = int(checkpoint["model"]["projection.fc1.weight"].shape[0])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = build_racal_model(model_path, str(checkpoint["mode"]), hidden_dim).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.device = device

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        from protocol_v2.experiments.racal_v1.representation import encode_rows

        rows = [{"text": str(text)} for text in texts]
        return encode_rows(self.model, self.tokenizer, rows, self.device, int(batch_size), 256)


def _load_component_plan(dataset: str, seed: int) -> dict[str, Any]:
    payload = read_json(CASCADE_ROOT / "component_plan.json")
    matches = [item for item in payload["plans"] if item["dataset"] == dataset and int(item["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"expected one downstream component plan for {dataset}/seed{seed}")
    return matches[0]


def _load_rows(dataset: str, seed: int) -> list[dict[str, Any]]:
    path = DATA_ROOT / dataset / f"{_kir_tag()}_seed{seed}" / "gate" / "test.json"
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty gate test: {path}")
    return [dict(row) for row in rows]


def _make_pipeline(
    dataset: str,
    seed: int,
    device: torch.device,
    detector_path: Path,
    h1_root: Path,
    gate_variant: str = "trainable_k1",
) -> Any:
    from legacy.pipeline.system_pipeline import HiLSAMoEV19Pipeline, PipelinePaths
    from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector

    data_root = DATA_ROOT / dataset / f"{_kir_tag()}_seed{seed}"
    component = _load_component_plan(dataset, seed)
    paths = PipelinePaths(
        model_path=MODEL_ROOT / "smollm135m",
        gate_encoder_path=MODEL_ROOT / "all-MiniLM-L6-v2",
        gate_detector_path=detector_path,
        router_ckpt_path=Path(component["router"]),
        experts_root=Path(component["experts"]),
        experts_data_root=data_root / "experts",
        router_data_path=data_root / "router" / "train.json",
        gate_train_path=data_root / "gate" / "train.json",
    )
    pipeline = HiLSAMoEV19Pipeline(
        paths,
        device=str(device),
        max_length=64,
        semantic_gate_enabled=False,
        gate_mode="multisphere",
    )
    if gate_variant == "frozen_k1":
        # Load the original frozen MiniLM and the same downstream components.
        pipeline._load_gate()
        pipeline._load_tokenizer()
        pipeline._load_router()
        pipeline._load_domain_mapping()
        pipeline._index_experts()
    elif gate_variant == "trainable_k1":
        # Load the downstream components without loading a second frozen Gate
        # encoder. The H1 Trainable encoder and detector are installed below.
        pipeline._load_tokenizer()
        pipeline._load_router()
        pipeline._load_domain_mapping()
        pipeline._index_experts()
        detector = MultiSphereOOSDetector()
        detector.load(detector_path)
        detector.cluster_to_intent = {int(key): value for key, value in detector.cluster_to_intent.items()}
        detector.intent_to_cluster = {str(key): int(value) for key, value in detector.intent_to_cluster.items()}
        detector.intent_to_clusters = {
            str(key): [int(value) for value in values]
            for key, values in detector.intent_to_clusters.items()
        }
        pipeline.gate_detector = detector
        pipeline.gate_encoder = _RacalGateEncoder(
            MODEL_ROOT / "all-MiniLM-L6-v2",
            h1_root / dataset / f"{_kir_tag()}_seed{seed}" / "trainable_k1" / "checkpoint.pt",
            device,
        )
    else:
        raise ValueError(f"unsupported gate_variant: {gate_variant}")
    return pipeline


def _error_stage(row: Mapping[str, Any], prediction: Mapping[str, Any]) -> str:
    true_oos = int(row["label"]) == 1
    predicted_oos = bool(prediction.get("is_oos", prediction.get("gate_pred", 1)))
    if true_oos:
        return "correct_oos_rejection" if predicted_oos else "oos_accepted_by_gate"
    if predicted_oos:
        return "known_rejected_by_gate"
    if str(prediction.get("domain", "")) != str(row.get("domain", "")):
        return "known_wrong_domain"
    if str(prediction.get("intent", "__missing__")) != str(row.get("intent", "")):
        return "known_wrong_expert"
    return "correct_known_prediction"


def compute_metrics(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    if len(rows) != len(predictions):
        raise ValueError("row/prediction length mismatch")
    true_oos = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    pred_oos = np.asarray([int(bool(prediction.get("is_oos", prediction.get("gate_pred", 1)))) for prediction in predictions], dtype=np.int64)
    known_mask = true_oos == 0
    oos_mask = true_oos == 1
    known_intents = sorted({str(row["intent"]) for row in rows if int(row["label"]) == 0})
    gold_labels = ["__oos__" if label else str(row["intent"]) for row, label in zip(rows, true_oos, strict=True)]
    predicted_labels = ["__oos__" if pred else str(prediction.get("intent", "__missing__")) for pred, prediction in zip(pred_oos, predictions, strict=True)]
    all_labels = [*known_intents, "__oos__"]
    gate_scores = np.asarray([float(prediction.get("gate_score", 0.0)) for prediction in predictions], dtype=np.float64)
    stage_counts = {stage: 0 for stage in STAGES}
    for row, prediction in zip(rows, predictions, strict=True):
        stage_counts[_error_stage(row, prediction)] += 1

    known_pass = known_mask & (pred_oos == 0)
    router_errors = sum(
        int(known_pass[index]) and str(predictions[index].get("domain", "")) != str(rows[index].get("domain", ""))
        for index in range(len(rows))
    )
    expert_errors = stage_counts["known_wrong_expert"]
    metrics = {
        "oos_f1": float(f1_score(true_oos, pred_oos, pos_label=1, zero_division=0)),
        "f1_all": float(f1_score(gold_labels, predicted_labels, labels=all_labels, average="macro", zero_division=0)),
        "known_macro_f1": float(f1_score([label for label, mask in zip(gold_labels, known_mask, strict=True) if mask], [label for label, mask in zip(predicted_labels, known_mask, strict=True) if mask], labels=known_intents, average="macro", zero_division=0)),
        "overall_accuracy": float(np.mean(np.asarray(gold_labels, dtype=object) == np.asarray(predicted_labels, dtype=object))),
        "known_recall": float(np.mean(pred_oos[known_mask] == 0)),
        "false_accept_rate": float(np.mean(pred_oos[oos_mask] == 0)),
        "false_reject_rate": float(np.mean(pred_oos[known_mask] == 1)),
        "auroc": float(roc_auc_score(true_oos, gate_scores)),
        "aupr_oos": float(average_precision_score(true_oos, gate_scores)),
        "router_error_rate": float(router_errors / max(int(np.sum(known_pass)), 1)),
        "expert_error_rate": float(expert_errors / max(int(np.sum(known_pass)), 1)),
        "gate_id_recall": float(np.mean(pred_oos[known_mask] == 0)),
        "gate_oos_rejection": float(np.mean(pred_oos[oos_mask] == 1)),
        "known_count": int(np.sum(known_mask)),
        "oos_count": int(np.sum(oos_mask)),
        "gate_accept_known_count": int(np.sum(known_pass)),
    }
    return metrics, stage_counts


def gate_metric_replay(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    true_oos = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    pred_oos = np.asarray([int(prediction.get("fast_gate_pred", prediction.get("gate_pred", 1))) for prediction in predictions], dtype=np.int64)
    scores = np.asarray([float(prediction.get("gate_score", 0.0)) for prediction in predictions], dtype=np.float64)
    return {
        "oos_f1": float(f1_score(true_oos, pred_oos, pos_label=1, zero_division=0)),
        "known_recall": float(np.mean(pred_oos[true_oos == 0] == 0)),
        "false_accept_rate": float(np.mean(pred_oos[true_oos == 1] == 0)),
        "auroc": float(roc_auc_score(true_oos, scores)),
    }


def run_cell(
    dataset: str,
    seed: int,
    device: torch.device,
    temporary_root: Path,
    h1_root: Path,
    frozen_root: Path,
    gate_variant: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    artifact_root = h1_root if gate_variant == "trainable_k1" else frozen_root
    artifact_dir = artifact_root / dataset / f"{_kir_tag()}_seed{seed}" / gate_variant
    signature_path = artifact_dir / "detector_signature.json"
    signature = read_json(signature_path)
    detector_path = temporary_root / f"{dataset}_seed{seed}_detector.json"
    detector_path.write_text(json.dumps(_detector_state(signature), ensure_ascii=False), encoding="utf-8")
    rows = _load_rows(dataset, seed)
    pipeline = _make_pipeline(dataset, seed, device, detector_path, h1_root, gate_variant)
    predictions = pipeline.predict_batch([str(row["text"]) for row in rows], batch_size=128)
    metrics, stage_counts = compute_metrics(rows, predictions)
    replay = gate_metric_replay(rows, predictions)
    stored = read_json(artifact_dir / "metrics.json")
    gate_deltas = {
        name: abs(float(replay[name]) - float(stored[name]))
        for name in ("oos_f1", "known_recall", "false_accept_rate", "auroc")
    }
    replay_max = max(gate_deltas.values())
    if replay_max > 1e-6:
        raise AssertionError(f"{gate_variant} Gate replay mismatch: {dataset}/seed{seed}: {gate_deltas}")
    row = {
        "stage": "historical_gate_ablation_full_pipeline_v1",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert",
        "dataset": dataset,
        "kir": float(KIR),
        "seed": seed,
        "gate": gate_variant,
        "representation": "frozen_minilm" if gate_variant == "frozen_k1" else "checkpoint_declared_trainable_representation",
        "downstream_source": f"cascade_full/gpu_{_kir_tag()} fixed Router and Expert components",
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "gate_replay_max_abs_delta": replay_max,
        **metrics,
        **{f"stage_count_{key}": int(value) for key, value in stage_counts.items()},
    }
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, [
        {
            "dataset": dataset,
            "seed": seed,
            "gate": gate_variant,
            "stage": stage,
            "count": int(stage_counts[stage]),
            "rate": float(stage_counts[stage] / max(len(rows), 1)),
        }
        for stage in STAGES
    ]


def main() -> int:
    import torch
    global KIR, CASCADE_ROOT, DATA_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--seed", choices=SEEDS, type=int, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--kir", type=float, choices=(0.25, 0.50, 0.75), default=0.50)
    parser.add_argument("--h1-root", type=Path, default=DEFAULT_H1_ROOT)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN_ROOT)
    parser.add_argument("--gate-variant", choices=("trainable_k1", "frozen_k1"), default="trainable_k1")
    parser.add_argument("--cascade-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    KIR = float(args.kir)
    CASCADE_ROOT = (args.cascade_root or (ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cascade_full" / f"gpu_{_kir_tag(KIR)}")).resolve()
    DATA_ROOT = (args.data_root or (ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19")).resolve()
    OUTPUT_ROOT = args.output_root.resolve()
    datasets = tuple(args.dataset or DATASETS)
    seeds = tuple(args.seed or SEEDS)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    h1_root = args.h1_root.resolve()
    frozen_root = args.frozen_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metric_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="s2c_trainable_full_pipeline_") as temporary:
        temporary_root = Path(temporary)
        for dataset in datasets:
            for seed in seeds:
                row, stages = run_cell(dataset, seed, device, temporary_root, h1_root, frozen_root, args.gate_variant)
                metric_rows.append(row)
                stage_rows.extend(stages)
    write_csv(output_root / "per_seed.csv", metric_rows)
    write_csv(output_root / "error_paths.csv", stage_rows)
    manifest = {
        "schema_version": "s2c.h1_trainable_full_pipeline.v1",
        "stage": "historical_gate_ablation_full_pipeline_v1",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert",
        "datasets": list(datasets),
        "seeds": list(seeds),
        "kir": float(KIR),
        "device": str(device),
        "completed_units": len(metric_rows),
        "downstream_artifact_root": str(CASCADE_ROOT),
        "h1_trainable_artifact_root": str(h1_root),
        "frozen_gate_artifact_root": str(frozen_root),
        "gate_variant": args.gate_variant,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "raw_predictions_written": False,
        "elapsed_seconds": time.time() - started,
    }
    (output_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": manifest["stage"], "completed_units": len(metric_rows), "output_root": str(output_root), "device": str(device), "h1_root": str(h1_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
