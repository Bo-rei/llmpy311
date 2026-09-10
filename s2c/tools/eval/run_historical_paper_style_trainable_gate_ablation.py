#!/usr/bin/env python3
"""Run the paper-style geometry with the current Trainable Gate and fixed H1 downstream."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tools.eval.run_historical_trainable_parameter_full_pipeline as parameter_eval
import tools.eval.run_historical_trainable_full_pipeline as pipeline_eval


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
SEED = 42
KIRS = (0.25, 0.50, 0.75)
H1_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_protocol_v2" / "minilm_k1"
OUTPUT_ROOT = ROOT / "results" / "analysis" / "historical_paper_style_trainable_gate_ablation"


def corrected_known(f1_all: float, oos_f1: float, known_count: int) -> float:
    return 100.0 * ((known_count + 1.0) * f1_all - oos_f1) / known_count


def run_cell(dataset: str, kir: float, device: torch.device, temporary_root: Path) -> dict[str, object]:
    parameter_eval.KIR = float(kir)
    pipeline_eval.KIR = float(kir)
    pipeline_eval.DATA_ROOT = parameter_eval.DATA_ROOT.resolve()
    pipeline_eval.CASCADE_ROOT = (
        ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cascade_full"
        / f"gpu_kir{int(kir * 100):02d}"
    ).resolve()
    views, values = parameter_eval._encode_cell(dataset, SEED, device, H1_ROOT)
    radius_lambda = 0.5 if dataset == "clinc150" else 1.0
    detector = parameter_eval._fit_detector(
        values["train"],
        views["train"],
        k=2,
        radius_lambda=radius_lambda,
        acceptance_mode="normalized_union",
    )
    predictions, meta = parameter_eval._run_pipeline(
        dataset,
        SEED,
        device,
        detector,
        radius_scale=1.0,
        h1_root=H1_ROOT,
        temporary_root=temporary_root,
    )
    metrics, stages = pipeline_eval.compute_metrics(views["test"], predictions)
    known_count = sum(int(row["label"]) == 0 for row in views["test"])
    known_intents = len({str(row["intent"]) for row in views["test"] if int(row["label"]) == 0})
    row = {
        "dataset": dataset,
        "kir": kir,
        "seed": SEED,
        "gate_variant": "trainable_gate_paper_geometry",
        "encoder": "last2_minilm_plus_residual_projection",
        "k_per_intent": 2,
        "radius_lambda": radius_lambda,
        "threshold": 1.0,
        "acceptance_mode": "normalized_union",
        "known_intent_count": known_intents,
        "known_f1": corrected_known(float(metrics["f1_all"]), float(metrics["oos_f1"]), known_intents),
        "f1_all": float(metrics["f1_all"]) * 100.0,
        "oos_f1": float(metrics["oos_f1"]) * 100.0,
        "accuracy": float(metrics["overall_accuracy"]) * 100.0,
        "known_recall": float(metrics["known_recall"]) * 100.0,
        "false_acceptance": float(metrics["false_accept_rate"]) * 100.0,
        "router_error": float(metrics["router_error_rate"]) * 100.0,
        "expert_error": float(metrics["expert_error_rate"]) * 100.0,
        "prediction_count": meta["prediction_count"],
        "device": str(device),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "stage_counts": json.dumps(stages, sort_keys=True),
    }
    del detector, predictions, views, values
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this paper-style ablation")
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="s2c_paper_style_trainable_gate_") as temporary:
        temporary_root = Path(temporary)
        for kir in KIRS:
            for dataset in DATASETS:
                rows.append(run_cell(dataset, kir, device, temporary_root))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "per_cell.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "s2c.historical_paper_style_trainable_gate_ablation.v1",
        "stage": "historical_paper_style_trainable_gate_ablation",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert",
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "seed": SEED,
        "gate": {
            "representation": "last2_minilm_plus_residual_projection",
            "k_per_intent": 2,
            "distance": "mahalanobis_diag",
            "radius": "mean+lambda*std",
            "lambda": {"clinc150": 0.5, "stackoverflow": 1.0, "banking77_oos": 1.0},
            "threshold": 1.0,
            "acceptance_mode": "normalized_union",
        },
        "downstream": "current H1 fixed Router/Expert components",
        "completed_units": len(rows),
        "device": "cuda",
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "report": "docs/analysis/historical_paper_style_trainable_gate_ablation_report.md",
        "per_cell": "results/analysis/historical_paper_style_trainable_gate_ablation/per_cell.csv",
    }
    (OUTPUT_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
