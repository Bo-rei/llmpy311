#!/usr/bin/env python3
"""Run the current-protocol Cascade bridge for CLINC150 and Banking77.

This is intentionally a thin orchestration layer around the already verified
StackOverflow bridge implementation.  It reuses the same Known-only Expert
training/evaluation contract and points the two Gate variants at immutable
current-protocol artifacts:

* Frozen K=1: E2 ``gate_core_dense`` test predictions;
* Trainable K=1: ``minilm_trainable_control_v1`` predictions.

The output root is independent from the completed StackOverflow bridge.  No
E2, R1, MOGB, or trainable-control artifact is modified.
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
if str(PROJECT_ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))

import run_protocol_v2_cascade_bridge_v1 as bridge  # noqa: E402


DATASETS = ("clinc150", "banking77")
KIR = 0.50
SEEDS = (13, 42, 87)
STAGE = "cascade_bridge_multidataset_v1"
PROTOCOL_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1"
E2_ROOT = PROTOCOL_ROOT / "e2_gate_core_dense"
TRAINABLE_ROOT = PROTOCOL_ROOT / "minilm_trainable_control_v1" / "runs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_path(dataset: str, variant: str, seed: int) -> Path:
    if variant == "trainable_k1":
        return TRAINABLE_ROOT / dataset / f"seed_{seed}" / "predictions.jsonl"
    if variant == "frozen_k1":
        run = (
            f"protocol_v2_textoir_v1__{dataset}__kir_0.50__seed_{seed}"
            "__repr_frozen_minilm__k_1__dist_mahalanobis_diag__boundary_mean_std"
        )
        return E2_ROOT / run / "predictions" / "test.jsonl"
    raise ValueError(f"Unknown Gate variant: {variant}")


def _configure_bridge(dataset: str) -> None:
    """Set module globals used by the verified bridge functions."""

    bridge.DATASET = dataset
    bridge.KIR = KIR
    bridge.STAGE = STAGE
    bridge._gate_path = lambda variant, seed: _gate_path(dataset, variant, seed)


def _assert_gate_contract(dataset: str, seed: int, test_rows: list[dict[str, Any]]) -> None:
    expected = {str(row["sample_id"]) for row in test_rows}
    for variant in ("frozen_k1", "trainable_k1"):
        path = _gate_path(dataset, variant, seed)
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = bridge._rows(path)
        actual = {str(row["sample_id"]) for row in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(
                f"Gate/test contract mismatch: dataset={dataset} seed={seed} "
                f"variant={variant} rows={len(rows)} expected={len(expected)} "
                f"missing={len(expected - actual)} extra={len(actual - expected)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    paths = bridge.ProtocolV2Paths.discover(PROJECT_ROOT)
    for dataset in args.datasets:
        paths.require_experiment_admission(dataset)
    device = bridge.torch.device(
        args.device
        if args.device != "auto" and (args.device != "cuda" or bridge.torch.cuda.is_available())
        else "cpu"
    )
    output_root = args.output_root or paths.run_root / STAGE
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_metrics: list[dict[str, Any]] = []
    provenance_sources: dict[str, dict[str, list[str]]] = {}

    for dataset in args.datasets:
        _configure_bridge(dataset)
        provenance_sources[dataset] = {variant: [] for variant in ("frozen_k1", "trainable_k1")}
        for seed in args.seeds:
            bridge._set_seed(seed)
            views = bridge._load_views(paths, seed)
            _assert_gate_contract(dataset, seed, views["test"])
            seed_root = output_root / dataset / f"seed_{seed}"
            checkpoint_dir = seed_root / "expert"
            training_manifest_path = checkpoint_dir / "training_manifest.json"
            if args.resume and training_manifest_path.is_file():
                expert_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
            else:
                expert_manifest = bridge._train_expert(
                    views,
                    checkpoint_dir,
                    seed,
                    bridge.MODEL_PATH,
                    args.epochs,
                    args.batch_size,
                    args.learning_rate,
                    args.max_length,
                    device,
                )
            expert_predictions = bridge._expert_predict(
                views["test"], checkpoint_dir, device, args.batch_size * 2, args.max_length
            )
            for gate_variant in ("frozen_k1", "trainable_k1"):
                source = _gate_path(dataset, gate_variant, seed)
                provenance_sources[dataset][gate_variant].append(str(source.resolve()))
                metrics, records = bridge._evaluate_gate_cascade(
                    gate_variant, seed, views["test"], expert_predictions
                )
                metrics.update(
                    {
                        "dataset": dataset,
                        "expert_checkpoint_sha256": expert_manifest["checkpoint_sha256"],
                        "train_ids_sha256": expert_manifest["train_ids_sha256"],
                        "calibration_ids_sha256": expert_manifest["calibration_ids_sha256"],
                        "test_ids_sha256": bridge._rows_hash(views["test"]),
                        "gate_source": str(source.resolve()),
                    }
                )
                all_metrics.append(metrics)
                pred_path = seed_root / f"{gate_variant}_predictions.jsonl"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                with pred_path.open("w", encoding="utf-8") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (output_root / "metrics.json").write_text(
        json.dumps(
            {
                "stage": STAGE,
                "protocol": paths.dataset_version,
                "datasets": list(args.datasets),
                "kir": KIR,
                "seeds": args.seeds,
                "metrics": all_metrics,
                "elapsed_seconds": time.time() - started,
                "test_used_for_selection": False,
                "oos_used_for_training": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        if all_metrics:
            writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0]))
            writer.writeheader()
            writer.writerows(all_metrics)
    model_file = bridge.MODEL_PATH / "model.safetensors"
    (output_root / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "stage": STAGE,
                "protocol": paths.dataset_version,
                "datasets": list(args.datasets),
                "kir": KIR,
                "seeds": args.seeds,
                "model_path": str(bridge.MODEL_PATH.resolve()),
                "model_sha256": _sha256(model_file),
                "gate_sources": provenance_sources,
                "test_used_for_selection": False,
                "oos_used_for_training": False,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": STAGE,
                "completed": len(all_metrics),
                "datasets": list(args.datasets),
                "output_root": str(output_root),
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
