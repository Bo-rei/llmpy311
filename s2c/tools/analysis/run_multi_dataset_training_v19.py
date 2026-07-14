#!/usr/bin/env python3
"""Orchestrate multi-dataset v19 training.

This runner is intentionally thin: it expands a dataset × KIR matrix and
invokes the existing stage-specific training scripts as subprocesses.

Default stage order:

1. Gate
2. Router
3. Experts
4. Semantic verifier

Skip / resume is conservative: a stage is skipped only when its expected
outputs already exist and the stage manifest says it completed successfully.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from tools.analysis.historical_best_pipeline_v19 import HISTORICAL_BEST_PIPELINE
from src.runtime import WorkspacePaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = WorkspacePaths.discover(PROJECT_ROOT)

GATE_SCRIPT = PROJECT_ROOT / "tools" / "gate" / "train_multisphere_corrected.py"
ROUTER_SCRIPT = PROJECT_ROOT / "tools" / "train" / "train_router_v19.py"
EXPERT_BATCH_SCRIPT = PROJECT_ROOT / "tools" / "train" / "train_all_experts_v19.py"
SNIPS_EXPERT_SCRIPT = PROJECT_ROOT / "tools" / "train" / "train_all_experts_snips_v19.py"
VERIFIER_SCRIPT = PROJECT_ROOT / "tools" / "train" / "train_semantic_verifier_v19.py"


LOGGER = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _slug(dataset: str) -> str:
    return dataset.strip().upper().replace("-", "_").lower()


def _kir_tag(kir: float) -> str:
    return f"kir{int(round(float(kir) * 100)):02d}"


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _run(cmd: Sequence[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Legacy trainers validate this env name strictly.
    env["CONDA_DEFAULT_ENV"] = "bo"
    subprocess.run(list(cmd), cwd=PROJECT_ROOT, check=True, env=env)


def _default_matrix(
    datasets: Sequence[str], kir_values: Sequence[float]
) -> List[Tuple[str, float]]:
    return [(str(dataset), float(kir)) for dataset in datasets for kir in kir_values]


def _expected_outputs(stage_root: Path, stage: str) -> List[Path]:
    if stage == "gate":
        return [
            stage_root / "corrected_multisphere_detector.json",
            stage_root / "corrected_multisphere_results.json",
        ]
    if stage == "router":
        return [stage_root / "best_model.pt", stage_root / "results.json"]
    if stage == "experts":
        return [stage_root / "all_results.json"]
    if stage == "verifier":
        return [
            stage_root / "best_model.pt",
            stage_root / "train_samples.json",
            stage_root / "val_samples.json",
        ]
    return []


def _is_stage_complete(stage_root: Path, stage: str) -> bool:
    expected = _expected_outputs(stage_root, stage)
    return bool(expected) and all(
        path.exists() and path.stat().st_size > 0 for path in expected
    )


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: List[str]
    output_dir: Path
    expected_outputs: List[Path]


def build_stage_specs(
    *,
    dataset: str,
    kir: float,
    seed: int,
    data_root_base: Path,
    artifact_root: Path,
    model_path: str,
    gate_encoder_path: str,
    gate_profile: str = HISTORICAL_BEST_PIPELINE.gate_profile,
    gate_center_mode: str = HISTORICAL_BEST_PIPELINE.gate_center_mode,
    gate_distance_metric: str = HISTORICAL_BEST_PIPELINE.gate_distance_metric,
    gate_l2_normalize: bool = HISTORICAL_BEST_PIPELINE.gate_l2_normalize,
    gate_subcenters_per_intent: int = HISTORICAL_BEST_PIPELINE.gate_subcenters_per_intent,
    gate_min_id_recall: float = HISTORICAL_BEST_PIPELINE.gate_min_id_recall,
    router_epochs: int = 10,
    expert_epochs: int = 15,
    verifier_epochs: int = 6,
    router_batch_size: int = 32,
    expert_batch_size: int = 32,
    verifier_batch_size: int = 16,
    verifier_lora_r: int = 32,
    verifier_lora_alpha: int = 64,
    gate_batch_size: int = 128,
    num_workers: int = 4,
    device: str = "cpu",
) -> Tuple[Path, List[StageSpec]]:
    dataset_slug = _slug(dataset)
    kir_tag = _kir_tag(kir)
    data_root = data_root_base / dataset_slug / f"{kir_tag}_seed{seed}"
    run_root = artifact_root / dataset_slug / f"{kir_tag}_seed{seed}"
    manifest = _load_json(data_root / "MANIFEST.json")
    domain_names = list(manifest.get("domains", []))
    if not domain_names:
        raise RuntimeError(
            f"Missing domains in manifest: {data_root / 'MANIFEST.json'}"
        )

    gate_out = run_root / "gate"
    router_out = run_root / "router"
    experts_out = run_root / "experts"
    verifier_out = run_root / "prompt_verifier"

    gate_cmd = [
        sys.executable,
        str(GATE_SCRIPT),
        "--data_root",
        str(data_root),
        "--model_path",
        gate_encoder_path,
        "--output_dir",
        str(gate_out),
        "--min_id_recall_constraint",
        str(gate_min_id_recall),
        "--center_mode",
        str(gate_center_mode),
        "--distance_metric",
        str(gate_distance_metric),
        "--subcenters_per_intent",
        str(gate_subcenters_per_intent),
    ]
    if gate_l2_normalize:
        gate_cmd.append("--l2_normalize")

    router_cmd = [
        sys.executable,
        str(ROUTER_SCRIPT),
        "--model_path",
        model_path,
        "--data_dir",
        str(data_root / "router"),
        "--output_dir",
        str(router_out),
        "--epochs",
        str(router_epochs),
        "--batch_size",
        str(router_batch_size),
        "--num_workers",
        str(num_workers),
        "--seed",
        str(seed),
    ]

    if dataset_slug == "snips":
        # SNIPS is single-domain and cannot use the CLINC-oriented domain whitelist.
        expert_cmd = [
            sys.executable,
            str(SNIPS_EXPERT_SCRIPT),
            "--model_path",
            model_path,
            "--data_dir",
            str(data_root / "experts"),
            "--output_dir",
            str(experts_out),
            "--epochs",
            str(expert_epochs),
            "--batch_size",
            str(expert_batch_size),
            "--num_workers",
            str(num_workers),
            "--seed",
            str(seed),
            "--skip_existing",
        ]
    else:
        expert_cmd = [
            sys.executable,
            str(EXPERT_BATCH_SCRIPT),
            "--model_path",
            model_path,
            "--data_dir",
            str(data_root / "experts"),
            "--output_dir",
            str(experts_out),
            "--domains",
            *domain_names,
            "--epochs",
            str(expert_epochs),
            "--batch_size",
            str(expert_batch_size),
            "--num_workers",
            str(num_workers),
            "--seed",
            str(seed),
            "--skip_existing",
        ]

    verifier_cmd = [
        sys.executable,
        str(VERIFIER_SCRIPT),
        "--model_path",
        model_path,
        "--router_ckpt",
        str(router_out / "best_model.pt"),
        "--router_train",
        str(data_root / "router" / "train.json"),
        "--gate_detector_path",
        str(gate_out / "corrected_multisphere_detector.json"),
        "--gate_encoder_path",
        gate_encoder_path,
        "--gate_train",
        str(data_root / "gate" / "train.json"),
        "--gate_val",
        str(data_root / "gate" / "val.json"),
        "--output_dir",
        str(verifier_out),
        "--epochs",
        str(verifier_epochs),
        "--batch_size",
        str(verifier_batch_size),
        "--lora_r",
        str(verifier_lora_r),
        "--lora_alpha",
        str(verifier_lora_alpha),
        "--gate_batch_size",
        str(gate_batch_size),
        "--num_workers",
        str(num_workers),
        "--seed",
        str(seed),
        "--device",
        str(device),
    ]

    return run_root, [
        StageSpec("gate", gate_cmd, gate_out, _expected_outputs(gate_out, "gate")),
        StageSpec(
            "router", router_cmd, router_out, _expected_outputs(router_out, "router")
        ),
        StageSpec(
            "experts",
            expert_cmd,
            experts_out,
            _expected_outputs(experts_out, "experts"),
        ),
        StageSpec(
            "verifier",
            verifier_cmd,
            verifier_out,
            _expected_outputs(verifier_out, "verifier"),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the multi-dataset v19 training pipeline"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["CLINC150", "BANKING77-OOS", "SNIPS"]
    )
    parser.add_argument(
        "--kir_values", nargs="+", type=float, default=[0.25, 0.5, 0.75]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_root_base",
        default=str(PATHS.prepared_data_root / "multidataset/v19"),
    )
    parser.add_argument(
        "--artifact_root",
        default=str(PATHS.artifact_root / "outputs/experiments/multi_dataset_v19"),
    )
    parser.add_argument("--model_path", default=str(PATHS.smollm135m))
    parser.add_argument("--gate_encoder_path", default=None)
    parser.add_argument(
        "--gate_profile",
        default="historical_best",
        choices=["historical_best", "multisphere_default"],
        help=(
            "Gate configuration profile. historical_best uses the legacy best "
            "mix2+l2 family; multisphere_default keeps the old centroid-only setup."
        ),
    )
    parser.add_argument(
        "--gate_center_mode",
        default=None,
        choices=["class_centroid", "class_centroid_mixture", "kmeans"],
        help="Optional explicit override for gate center mode.",
    )
    parser.add_argument(
        "--gate_distance_metric",
        default="mahalanobis_diag",
        choices=["euclidean", "mahalanobis_diag"],
        help="Distance metric passed to gate trainer.",
    )
    parser.add_argument(
        "--gate_l2_normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable L2 normalization for gate embeddings.",
    )
    parser.add_argument(
        "--gate_subcenters_per_intent",
        type=int,
        default=None,
        help="Number of subcenters per intent for gate training.",
    )
    parser.add_argument(
        "--gate_min_id_recall",
        type=float,
        default=None,
        help=(
            "Minimum ID recall guard for gate boundary selection. If omitted, "
            "defaults are profile-aware (historical_best=0.80, multisphere_default=0.85)."
        ),
    )
    parser.add_argument("--router_epochs", type=int, default=10)
    parser.add_argument("--expert_epochs", type=int, default=15)
    parser.add_argument("--verifier_epochs", type=int, default=6)
    parser.add_argument("--router_batch_size", type=int, default=32)
    parser.add_argument("--expert_batch_size", type=int, default=32)
    parser.add_argument("--verifier_batch_size", type=int, default=16)
    parser.add_argument("--router_lora_r", type=int, default=32)
    parser.add_argument("--router_lora_alpha", type=int, default=64)
    parser.add_argument("--verifier_lora_r", type=int, default=None)
    parser.add_argument("--verifier_lora_alpha", type=int, default=None)
    parser.add_argument("--gate_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    data_root_base = Path(args.data_root_base)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    device = str(args.device) if args.device is not None else _default_device()
    verifier_lora_r = (
        int(args.verifier_lora_r)
        if args.verifier_lora_r is not None
        else int(args.router_lora_r)
    )
    verifier_lora_alpha = (
        int(args.verifier_lora_alpha)
        if args.verifier_lora_alpha is not None
        else int(args.router_lora_alpha)
    )
    gate_encoder_path = (
        str(args.gate_encoder_path)
        if args.gate_encoder_path is not None and str(args.gate_encoder_path).strip() != ""
        else (
            HISTORICAL_BEST_PIPELINE.strict_replay_defaults()["gate_encoder_path"]
            if str(args.gate_profile) == HISTORICAL_BEST_PIPELINE.gate_profile
            else str(args.model_path)
        )
    )

    if str(args.gate_profile) == HISTORICAL_BEST_PIPELINE.gate_profile:
        default_gate_center_mode = HISTORICAL_BEST_PIPELINE.gate_center_mode
        default_gate_l2_normalize = HISTORICAL_BEST_PIPELINE.gate_l2_normalize
        default_gate_subcenters = HISTORICAL_BEST_PIPELINE.gate_subcenters_per_intent
    else:
        default_gate_center_mode = "class_centroid"
        default_gate_l2_normalize = False
        default_gate_subcenters = 1

    gate_center_mode = (
        str(args.gate_center_mode)
        if args.gate_center_mode is not None and str(args.gate_center_mode).strip() != ""
        else default_gate_center_mode
    )
    gate_l2_normalize = (
        bool(args.gate_l2_normalize)
        if args.gate_l2_normalize is not None
        else default_gate_l2_normalize
    )
    gate_subcenters_per_intent = (
        int(args.gate_subcenters_per_intent)
        if args.gate_subcenters_per_intent is not None
        else int(default_gate_subcenters)
    )
    gate_min_id_recall = (
        float(args.gate_min_id_recall)
        if args.gate_min_id_recall is not None
        else (
            HISTORICAL_BEST_PIPELINE.gate_min_id_recall
            if str(args.gate_profile) == HISTORICAL_BEST_PIPELINE.gate_profile
            else 0.85
        )
    )

    summary: List[Dict[str, Any]] = []

    for dataset, kir in _default_matrix(args.datasets, args.kir_values):
        run_root, stage_specs = build_stage_specs(
            dataset=dataset,
            kir=kir,
            seed=args.seed,
            data_root_base=data_root_base,
            artifact_root=artifact_root,
            model_path=args.model_path,
            gate_encoder_path=gate_encoder_path,
            gate_profile=str(args.gate_profile),
            gate_center_mode=gate_center_mode,
            gate_distance_metric=str(args.gate_distance_metric),
            gate_l2_normalize=gate_l2_normalize,
            gate_subcenters_per_intent=gate_subcenters_per_intent,
            gate_min_id_recall=gate_min_id_recall,
            router_epochs=args.router_epochs,
            expert_epochs=args.expert_epochs,
            verifier_epochs=args.verifier_epochs,
            router_batch_size=args.router_batch_size,
            expert_batch_size=args.expert_batch_size,
            verifier_batch_size=args.verifier_batch_size,
            verifier_lora_r=verifier_lora_r,
            verifier_lora_alpha=verifier_lora_alpha,
            gate_batch_size=args.gate_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        manifest_path = run_root / "train_manifest.json"
        run_record: Dict[str, Any] = {
            "pipeline_profile": HISTORICAL_BEST_PIPELINE.name,
            "dataset": dataset,
            "dataset_slug": _slug(dataset),
            "kir": float(kir),
            "seed": int(args.seed),
            "device": device,
            "gate_profile": str(args.gate_profile),
            "gate_config": {
                "gate_encoder_path": gate_encoder_path,
                "center_mode": gate_center_mode,
                "distance_metric": str(args.gate_distance_metric),
                "l2_normalize": bool(gate_l2_normalize),
                "subcenters_per_intent": int(gate_subcenters_per_intent),
                "min_id_recall_constraint": float(gate_min_id_recall),
            },
            "run_root": str(run_root),
            "manifest_path": str(manifest_path),
            "stages": [],
        }

        for stage in stage_specs:
            stage_manifest_path = stage.output_dir / "stage_manifest.json"
            manifest_complete = False
            if stage_manifest_path.exists():
                try:
                    stage_manifest = _load_json(stage_manifest_path)
                    manifest_complete = stage_manifest.get("status") == "completed"
                except json.JSONDecodeError:
                    manifest_complete = False

            should_skip = (
                manifest_complete
                and _is_stage_complete(stage.output_dir, stage.name)
            )
            if not args.force and should_skip:
                stage_status = "skipped"
            else:
                if stage.output_dir.exists() and not args.dry_run:
                    shutil.rmtree(stage.output_dir)
                stage_status = "dry_run" if args.dry_run else "running"
                if not args.dry_run:
                    try:
                        _run(stage.command)
                        stage_status = "completed"
                        _write_json(
                            stage_manifest_path,
                            {
                                "name": stage.name,
                                "status": stage_status,
                                "command": stage.command,
                                "output_dir": str(stage.output_dir),
                                "expected_outputs": [
                                    str(path) for path in stage.expected_outputs
                                ],
                            },
                        )
                    except subprocess.CalledProcessError as exc:
                        stage_status = f"failed(exit={exc.returncode})"
                        _write_json(
                            stage_manifest_path,
                            {
                                "name": stage.name,
                                "status": stage_status,
                                "command": stage.command,
                                "output_dir": str(stage.output_dir),
                                "expected_outputs": [
                                    str(path) for path in stage.expected_outputs
                                ],
                            },
                        )
                        run_record["stages"].append(
                            {
                                "name": stage.name,
                                "status": stage_status,
                                "command": stage.command,
                                "output_dir": str(stage.output_dir),
                                "expected_outputs": [
                                    str(path) for path in stage.expected_outputs
                                ],
                                "stage_manifest": str(stage_manifest_path),
                            }
                        )
                        _write_json(manifest_path, run_record)
                        summary.append(run_record)
                        if args.fail_fast:
                            raise
                        break

            run_record["stages"].append(
                {
                    "name": stage.name,
                    "status": stage_status,
                    "command": stage.command,
                    "output_dir": str(stage.output_dir),
                    "expected_outputs": [str(path) for path in stage.expected_outputs],
                    "stage_manifest": str(stage_manifest_path),
                }
            )

        _write_json(manifest_path, run_record)
        summary.append(run_record)

    summary_path = artifact_root / "training_summary.json"
    _write_json(summary_path, {"seed": args.seed, "device": device, "runs": summary})
    LOGGER.info("Wrote training summary -> %s", summary_path)


if __name__ == "__main__":
    main()
