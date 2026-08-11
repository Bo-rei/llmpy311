#!/usr/bin/env python3
"""Extend the verified protocol_v2 Cascade bridge to additional KIR values.

The runner trains a fresh Known-only SmolLM Expert per dataset/KIR/seed and
evaluates the immutable Frozen-K1 and Trainable-K1 Gate predictions from the
matching protocol view.  It deliberately owns a new artifact root and never
modifies the completed KIR=.50 bridge, E2, or MiniLM artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "scripts" / "experiments"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_protocol_v2_cascade_bridge_v1 as bridge  # noqa: E402


DATASETS = ("stackoverflow",)
DEFAULT_KIRS = (0.25, 0.75)
DEFAULT_SEEDS = (13, 42, 87)
STAGE = "cascade_bridge_kir_extension_v1"
PROTOCOL_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1"
E2_ROOT = PROTOCOL_ROOT / "e2_gate_core_dense"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_path(dataset: str, kir: float, variant: str, seed: int) -> Path:
    if variant == "trainable_k1":
        trainable_root = PROTOCOL_ROOT / "minilm_trainable_kir_sweep_v1" / f"kir_{kir:.2f}" / "runs"
        return trainable_root / dataset / f"seed_{seed}" / "predictions.jsonl"
    if variant == "frozen_k1":
        run = (
            f"protocol_v2_textoir_v1__{dataset}__kir_{kir:.2f}__seed_{seed}"
            "__repr_frozen_minilm__k_1__dist_mahalanobis_diag__boundary_mean_std"
        )
        return E2_ROOT / run / "predictions" / "test.jsonl"
    raise ValueError(f"Unknown Gate variant: {variant}")


def _configure(dataset: str, kir: float) -> None:
    bridge.DATASET = dataset
    bridge.KIR = kir
    bridge.STAGE = STAGE
    bridge._gate_path = lambda variant, seed: _gate_path(dataset, kir, variant, seed)


def _assert_gate_contract(dataset: str, kir: float, seed: int, test_rows: list[dict[str, Any]]) -> None:
    expected = {str(row["sample_id"]) for row in test_rows}
    for variant in ("frozen_k1", "trainable_k1"):
        path = _gate_path(dataset, kir, variant, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = bridge._rows(path)
        actual = {str(row["sample_id"]) for row in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(
                f"Gate/test contract mismatch: dataset={dataset} kir={kir} seed={seed} "
                f"variant={variant} rows={len(rows)} expected={len(expected)}"
            )


def _run_one_kir(
    paths: Any,
    dataset: str,
    kir: float,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    device: Any,
    output_root: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    _configure(dataset, kir)
    kir_root = output_root / dataset / f"kir_{kir:.2f}"
    kir_root.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    for seed in seeds:
        bridge._set_seed(seed)
        views = bridge._load_views(paths, seed)
        _assert_gate_contract(dataset, kir, seed, views["test"])
        seed_root = kir_root / f"seed_{seed}"
        checkpoint_dir = seed_root / "expert"
        training_manifest_path = checkpoint_dir / "training_manifest.json"
        if resume and training_manifest_path.is_file():
            expert_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
        else:
            expert_manifest = bridge._train_expert(
                views,
                checkpoint_dir,
                seed,
                bridge.MODEL_PATH,
                epochs,
                batch_size,
                learning_rate,
                max_length,
                device,
            )
        expert_predictions = bridge._expert_predict(
            views["test"], checkpoint_dir, device, batch_size * 2, max_length
        )
        for variant in ("frozen_k1", "trainable_k1"):
            source = _gate_path(dataset, kir, variant, seed)
            values, records = bridge._evaluate_gate_cascade(variant, seed, views["test"], expert_predictions)
            values.update(
                {
                    "stage": STAGE,
                    "dataset": dataset,
                    "kir": kir,
                    "expert_checkpoint_sha256": expert_manifest["checkpoint_sha256"],
                    "train_ids_sha256": expert_manifest["train_ids_sha256"],
                    "calibration_ids_sha256": expert_manifest["calibration_ids_sha256"],
                    "test_ids_sha256": bridge._rows_hash(views["test"]),
                    "gate_source": str(source.resolve()),
                    "test_used_for_selection": False,
                    "oos_used_for_training": False,
                }
            )
            metrics.append(values)
            pred_path = seed_root / f"{variant}_predictions.jsonl"
            with pred_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--kirs", nargs="+", type=float, default=list(DEFAULT_KIRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if any(kir not in {0.25, 0.50, 0.75} for kir in args.kirs):
        raise ValueError("This extension only accepts protocol KIR values 0.25, 0.50, 0.75")
    paths = bridge.ProtocolV2Paths.discover(PROJECT_ROOT)
    for dataset in args.datasets:
        paths.require_experiment_admission(dataset)
    output_root = args.output_root or paths.run_root / STAGE
    if args.dry_run:
        print(json.dumps({"stage": STAGE, "datasets": args.datasets, "kirs": args.kirs, "seeds": args.seeds, "planned": len(args.datasets) * len(args.kirs) * len(args.seeds) * 2}, indent=2))
        return
    device = bridge.torch.device(
        args.device
        if args.device != "auto" and (args.device != "cuda" or bridge.torch.cuda.is_available())
        else "cpu"
    )
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = []
    metrics_path = output_root / "metrics.json"
    if args.resume and metrics_path.is_file():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        all_metrics = list(existing.get("metrics", []))

    def merge_metrics(rows: list[dict[str, Any]]) -> None:
        by_key = {
            (str(row["dataset"]), float(row["kir"]), int(row["seed"]), str(row["gate_variant"])): row
            for row in all_metrics
        }
        for row in rows:
            key = (str(row["dataset"]), float(row["kir"]), int(row["seed"]), str(row["gate_variant"]))
            by_key[key] = row
        all_metrics[:] = [by_key[key] for key in sorted(by_key)]

    for dataset in args.datasets:
        for kir in args.kirs:
            merge_metrics(
                _run_one_kir(
                    paths,
                    dataset,
                    kir,
                    args.seeds,
                    args.epochs,
                    args.batch_size,
                    args.learning_rate,
                    args.max_length,
                    device,
                    output_root,
                    args.resume,
                )
            )
    resolved_kirs = sorted({float(row["kir"]) for row in all_metrics})
    resolved_seeds = sorted({int(row["seed"]) for row in all_metrics})
    payload = {
        "stage": STAGE,
        "protocol": paths.dataset_version,
        "datasets": args.datasets,
        "kirs": resolved_kirs,
        "seeds": resolved_seeds,
        "metrics": all_metrics,
        "elapsed_seconds": time.time() - started,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    (output_root / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    model_file = bridge.MODEL_PATH / "model.safetensors"
    (output_root / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "stage": STAGE,
                "protocol": paths.dataset_version,
                "datasets": args.datasets,
                "kirs": resolved_kirs,
                "seeds": resolved_seeds,
                "model_path": str(bridge.MODEL_PATH.resolve()),
                "model_sha256": _sha256(model_file),
                "test_used_for_selection": False,
                "oos_used_for_training": False,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"stage": STAGE, "completed": len(all_metrics), "output_root": str(output_root), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
