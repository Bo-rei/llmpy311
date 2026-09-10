#!/usr/bin/env python3
"""Benchmark aggregate latency, parameters and peak CUDA memory for H1 Gate variants."""

from __future__ import annotations

import csv
import sys
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.eval import run_historical_trainable_full_pipeline as full

H1_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_protocol_v2" / "minilm_k1"
DATA_ROOT = ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19"
DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
SEED = 42
KIR = 0.50
OUTPUT = ROOT / "results" / "analysis" / "historical_deployment_benchmark"


def detector_path(dataset: str, variant: str, temporary: Path) -> Path:
    signature = json.loads(
        (H1_ROOT / dataset / "kir50_seed42" / variant / "detector_signature.json").read_text()
    )
    path = temporary / f"{dataset}_{variant}.json"
    path.write_text(json.dumps(full._detector_state(signature)))
    return path


def parameter_count(model: object) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def bench(
    predictor: Callable[[list[str], int], object],
    texts: list[str],
    batch_size: int,
    repeats: int = 20,
) -> tuple[float, int]:
    batch = texts[:batch_size]
    for _ in range(5):
        predictor(batch, batch_size)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        predictor(batch, batch_size)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / (repeats * len(batch)), int(torch.cuda.max_memory_allocated())


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    full.KIR = KIR
    full.DATA_ROOT = DATA_ROOT
    full.CASCADE_ROOT = ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cascade_full" / "gpu_kir50"
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="s2c_deployment_benchmark_") as temporary:
        temporary_root = Path(temporary)
        for dataset in DATASETS:
            views = full._load_rows(dataset, SEED)
            texts = [str(row["text"]) for row in views]
            for variant in ("frozen_k1", "trainable_k1"):
                detector = detector_path(dataset, variant, temporary_root)
                pipeline = full._make_pipeline(dataset, SEED, torch.device("cuda"), detector, H1_ROOT, variant)
                gate_model = pipeline.gate_encoder.model if variant == "trainable_k1" else pipeline.gate_encoder
                gate_params = parameter_count(gate_model)
                trainable_params = 0
                if variant == "trainable_k1":
                    manifest = json.loads(
                        (H1_ROOT / dataset / "kir50_seed42" / "trainable_k1" / "run_manifest.json").read_text()
                    )
                    trainable_params = int(manifest["trainable_parameters"])
                row = {
                    "dataset": dataset,
                    "kir": KIR,
                    "seed": SEED,
                    "gate": variant,
                    "gate_total_parameter_count": gate_params,
                    "gate_trainable_parameter_count": trainable_params,
                    "gate_batch1_ms_per_sample": None,
                    "gate_batch32_ms_per_sample": None,
                    "gate_batch1_samples_per_sec": None,
                    "gate_batch32_samples_per_sec": None,
                    "gate_batch1_peak_cuda_mb": None,
                    "gate_batch32_peak_cuda_mb": None,
                    "cascade_batch1_ms_per_sample": None,
                    "cascade_batch32_ms_per_sample": None,
                    "cascade_batch1_samples_per_sec": None,
                    "cascade_batch32_samples_per_sec": None,
                    "cascade_batch1_peak_cuda_mb": None,
                    "cascade_batch32_peak_cuda_mb": None,
                    "prediction_count": len(texts),
                    "device": "cuda",
                }
                def gate_predictor(batch: list[str], size: int) -> object:
                    del size
                    return pipeline._gate_predict(batch)

                def cascade_predictor(batch: list[str], size: int) -> object:
                    return pipeline.predict_batch(batch, batch_size=size)

                gate_b1, gate_mem_b1 = bench(gate_predictor, texts, 1)
                gate_b32, gate_mem_b32 = bench(gate_predictor, texts, min(32, len(texts)))
                cascade_b1, cascade_mem_b1 = bench(cascade_predictor, texts, 1)
                cascade_b32, cascade_mem_b32 = bench(cascade_predictor, texts, min(32, len(texts)))
                row["gate_batch1_ms_per_sample"] = gate_b1
                row["gate_batch32_ms_per_sample"] = gate_b32
                row["gate_batch1_samples_per_sec"] = 1000.0 / gate_b1
                row["gate_batch32_samples_per_sec"] = 1000.0 / gate_b32
                row["gate_batch1_peak_cuda_mb"] = gate_mem_b1 / (1024 * 1024)
                row["gate_batch32_peak_cuda_mb"] = gate_mem_b32 / (1024 * 1024)
                row["cascade_batch1_ms_per_sample"] = cascade_b1
                row["cascade_batch32_ms_per_sample"] = cascade_b32
                row["cascade_batch1_samples_per_sec"] = 1000.0 / cascade_b1
                row["cascade_batch32_samples_per_sec"] = 1000.0 / cascade_b32
                row["cascade_batch1_peak_cuda_mb"] = cascade_mem_b1 / (1024 * 1024)
                row["cascade_batch32_peak_cuda_mb"] = cascade_mem_b32 / (1024 * 1024)
                rows.append(row)
                del pipeline
                torch.cuda.empty_cache()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": "s2c.historical_deployment_benchmark.v2",
                "datasets": list(DATASETS),
                "kir": KIR,
                "seed": SEED,
                "variants": ["frozen_k1", "trainable_k1"],
                "device": "cuda",
                "raw_predictions_written": False,
                "measurement_scope": ["gate_only", "full_cascade"],
                "summary": "results/analysis/historical_deployment_benchmark/summary.csv",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"completed": len(rows), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
