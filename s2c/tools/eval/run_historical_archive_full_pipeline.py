#!/usr/bin/env python3
"""Evaluate Frozen, partial-tuned and LoRA MiniLM Gates in one archive pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from legacy.pipeline.system_pipeline import HiLSAMoEV19Pipeline, PipelinePaths  # noqa: E402
from tools.eval.run_historical_trainable_full_pipeline import (  # noqa: E402
    _RacalGateEncoder,
    _detector_state,
    compute_metrics,
    gate_metric_replay,
    read_json,
)


DEFAULT_DATA_ROOT = ROOT.parent / "archives" / "submissions" / "s2c-submission" / "data"
DEFAULT_MODEL_ROOT = ROOT.parent / "assets" / "models"
DEFAULT_OUTPUT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_archive_full_pipeline_seed42"
DATASETS = ("clinc150", "stackoverflow", "banking77")
METHODS = ("frozen_k1", "partial_k1", "lora_k1")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(data_root: Path, dataset: str, seed: int) -> list[dict[str, Any]]:
    path = data_root / dataset / f"kir50_seed{seed}" / "gate" / "test.json"
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty archive Gate test: {path}")
    return [dict(row) for row in rows]


def _load_signature(gate_run: Path) -> dict[str, Any]:
    signature = read_json(gate_run / "detector_signature.json")
    if not isinstance(signature, dict) or not signature.get("spheres"):
        raise ValueError(f"invalid detector signature: {gate_run}")
    return signature


def _make_pipeline(
    *,
    dataset: str,
    seed: int,
    method: str,
    data_root: Path,
    model_root: Path,
    router_ckpt: Path,
    experts_root: Path,
    detector_path: Path,
    gate_run: Path,
    checkpoint_run: Path | None,
    device: torch.device,
) -> HiLSAMoEV19Pipeline:
    data_dir = data_root / dataset / f"kir50_seed{seed}"
    paths = PipelinePaths(
        model_path=model_root / "smollm135m",
        gate_encoder_path=model_root / "all-MiniLM-L6-v2",
        gate_detector_path=detector_path,
        router_ckpt_path=router_ckpt,
        experts_root=experts_root,
        experts_data_root=data_dir / "experts",
        router_data_path=data_dir / "router" / "train.json",
        gate_train_path=data_dir / "gate" / "train.json",
    )
    pipeline = HiLSAMoEV19Pipeline(
        paths,
        device=str(device),
        max_length=64,
        semantic_gate_enabled=False,
        gate_mode="multisphere",
    )
    pipeline.load()
    if method != "frozen_k1":
        checkpoint_path = gate_run / "checkpoint.pt"
        if checkpoint_run is not None:
            checkpoint_path = checkpoint_run / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        hidden_dim = int(checkpoint["model"]["projection.fc1.weight"].shape[0])
        pipeline.gate_encoder = _RacalGateEncoder(
            model_root / "all-MiniLM-L6-v2",
            checkpoint_path,
            device,
        )
        if int(hidden_dim) != 256:
            raise ValueError(f"unexpected projection hidden size for {method}: {hidden_dim}")
    return pipeline


def _run_method(
    *,
    dataset: str,
    seed: int,
    method: str,
    gate_run: Path,
    data_root: Path,
    model_root: Path,
    router_ckpt: Path,
    experts_root: Path,
    device: torch.device,
    checkpoint_run: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    signature = _load_signature(gate_run)
    rows = _load_rows(data_root, dataset, seed)
    with tempfile.TemporaryDirectory(prefix="s2c_archive_gate_") as temporary:
        detector_path = Path(temporary) / "detector.json"
        detector_path.write_text(
            json.dumps(_detector_state(signature), ensure_ascii=False),
            encoding="utf-8",
        )
        pipeline = _make_pipeline(
            dataset=dataset,
            seed=seed,
            method=method,
            data_root=data_root,
            model_root=model_root,
            router_ckpt=router_ckpt,
            experts_root=experts_root,
            detector_path=detector_path,
            gate_run=gate_run,
            checkpoint_run=checkpoint_run,
            device=device,
        )
        predictions = pipeline.predict_batch(
            [str(row["text"]) for row in rows],
            batch_size=128,
        )
    metrics, stage_counts = compute_metrics(rows, predictions)
    replay = gate_metric_replay(rows, predictions)
    gate_delta = max(
        abs(float(metrics[name]) - float(replay[name]))
        for name in ("oos_f1", "known_recall", "false_accept_rate")
    )
    if gate_delta > 1e-12:
        raise AssertionError(f"full pipeline changed Gate metrics for {dataset}/{method}: {gate_delta}")

    method_row = {
        "stage": "historical_archive_full_pipeline_v1",
        "protocol": "historical_v19_archive__gate_to_router_to_expert",
        "dataset": dataset,
        "kir": 0.50,
        "seed": seed,
        "method": method,
        "gate_run": str(gate_run),
        "router_ckpt": str(router_ckpt),
        "experts_root": str(experts_root),
        "oos_f1": float(metrics["oos_f1"]),
        "f1_all": float(metrics["f1_all"]),
        "known_macro_f1": float(metrics["known_macro_f1"]),
        "overall_accuracy": float(metrics["overall_accuracy"]),
        "known_recall": float(metrics["known_recall"]),
        "false_accept_rate": float(metrics["false_accept_rate"]),
        "false_reject_rate": float(metrics["false_reject_rate"]),
        "router_error_rate": float(metrics["router_error_rate"]),
        "expert_error_rate": float(metrics["expert_error_rate"]),
        "gate_metric_max_abs_delta": float(gate_delta),
        "known_count": int(metrics["known_count"]),
        "oos_count": int(metrics["oos_count"]),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    raw_rows = [
        {"row": row, "prediction": prediction, "method": method, "dataset": dataset, "seed": seed}
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    stage_rows = [
        {
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "stage": stage,
            "count": int(count),
            "rate": float(count / max(len(rows), 1)),
        }
        for stage, count in stage_counts.items()
    ]
    return method_row, raw_rows, stage_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--router-ckpt", type=Path, required=True)
    parser.add_argument("--experts-root", type=Path, required=True)
    parser.add_argument("--frozen-gate-run", type=Path, required=True)
    parser.add_argument("--partial-gate-run", type=Path, required=True)
    parser.add_argument("--lora-gate-run", type=Path, required=True)
    parser.add_argument("--partial-checkpoint-run", type=Path)
    parser.add_argument("--lora-checkpoint-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    output_root = args.output_root.resolve() / args.dataset / f"kir50_seed{args.seed}"
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    gate_runs = {
        "frozen_k1": args.frozen_gate_run.resolve(),
        "partial_k1": args.partial_gate_run.resolve(),
        "lora_k1": args.lora_gate_run.resolve(),
    }
    checkpoint_runs = {
        "frozen_k1": gate_runs["frozen_k1"],
        "partial_k1": (args.partial_checkpoint_run or args.partial_gate_run).resolve(),
        "lora_k1": (args.lora_checkpoint_run or args.lora_gate_run).resolve(),
    }
    started = time.time()
    metric_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for method in METHODS:
        row, raw_rows, stages = _run_method(
            dataset=args.dataset,
            seed=args.seed,
            method=method,
            gate_run=gate_runs[method],
            data_root=args.data_root.resolve(),
            model_root=args.model_root.resolve(),
            router_ckpt=args.router_ckpt.resolve(),
            experts_root=args.experts_root.resolve(),
            device=device,
            checkpoint_run=checkpoint_runs[method],
        )
        metric_rows.append(row)
        stage_rows.extend(stages)
        (output_root / f"predictions_{method}.json").write_text(
            json.dumps(raw_rows, ensure_ascii=False),
            encoding="utf-8",
        )
    _write_csv(output_root / "per_method.csv", metric_rows)
    _write_csv(output_root / "error_paths.csv", stage_rows)
    manifest = {
        "schema_version": "s2c.historical_archive_full_pipeline.v1",
        "protocol": "historical_v19_archive__gate_to_router_to_expert",
        "dataset": args.dataset,
        "seed": args.seed,
        "kir": 0.50,
        "data_root": str(args.data_root.resolve()),
        "model_root": str(args.model_root.resolve()),
        "router_ckpt": str(args.router_ckpt.resolve()),
        "experts_root": str(args.experts_root.resolve()),
        "gate_runs": {key: str(value) for key, value in gate_runs.items()},
        "checkpoint_runs": {key: str(value) for key, value in checkpoint_runs.items()},
        "methods": list(METHODS),
        "device": str(device),
        "semantic_gate_enabled": False,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "raw_predictions_written": True,
        "completed_methods": len(metric_rows),
        "elapsed_seconds": time.time() - started,
    }
    (output_root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(output_root), "completed_methods": len(metric_rows), "device": str(device)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
