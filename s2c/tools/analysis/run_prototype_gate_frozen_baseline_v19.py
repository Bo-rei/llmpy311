#!/usr/bin/env python3
"""Run the frozen historical Prototype Gate baseline for v19.

This wrapper exists to reproduce the archived best prototype result exactly:

    pipeline_v19_phase3_proto_d_narrow_t085_eval

It deliberately freezes the historical config family instead of using the later
presentation wrapper. The command line mirrors the archived evaluation config
and keeps the run self-describing through a small identity manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis.component_path_utils import (
    resolve_frozen_experts_root,
    resolve_frozen_router_ckpt,
)
from tools.analysis.prototype_path_utils import resolve_multi_prototype_path
from tools.analysis.historical_best_pipeline_v19 import HISTORICAL_BEST_PIPELINE


FROZEN_REFERENCE_EVAL = PROJECT_ROOT / (
    "outputs/experiments/archive/sweeps/2026-03-23/pipeline_phase3_proto_eval/"
    "pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json"
)
FROZEN_GATE_DETECTOR = PROJECT_ROOT / (
    "outputs/experiments/archive/sweeps/2026-03-23/gate_l2_mix2_train/"
    "gate_l2_mix2_true_lambda_1p6/detector.json"
)
FROZEN_ROUTER_CKPT = PROJECT_ROOT / "outputs/experiments/components/router/router_v19/best_model.pt"
FROZEN_EXPERTS_ROOT = PROJECT_ROOT / "outputs/experiments/components/experts/experts_v19"
# Keep the historical alias as a fallback, but prefer formal ablation payloads.
FROZEN_MULTI_PROTO_PATH = PROJECT_ROOT / "outputs/multi_prototypes_v19/prototypes.json"
FROZEN_OUTPUT_BASE = PROJECT_ROOT / "outputs/experiments/pipeline/frozen_prototype_gate"
FROZEN_OUTPUT_SUBDIR = "prototype_gate_pipeline_frozen"
FROZEN_MULTI_PROTO_ID_THRESHOLD = HISTORICAL_BEST_PIPELINE.multi_proto_id_threshold


def _default_device() -> Optional[str]:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else None
    except Exception:
        return None


def _run(cmd: List[str]) -> None:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def _resolve_existing(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing required {label}: {resolved}")
    return resolved


def _resolve_existing_any(paths: List[Optional[Path]], label: str) -> Path:
    for candidate in paths:
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    candidates = ", ".join(str(path.resolve()) for path in paths if path is not None)
    raise FileNotFoundError(f"Missing required {label}; tried: {candidates}")


def build_eval_command(
    *,
    seed: int,
    output_dir: Path,
    data_root: Path,
    gate_detector_path: Path,
    router_ckpt: Path,
    experts_root: Path,
    multi_prototype_path: Path,
    device: Optional[str],
    semantic_prompt_version: str,
    multi_proto_id_threshold: float,
    semantic_gate_threshold: float,
    semantic_uncertain_low: float,
    semantic_uncertain_high: float,
    prototype_centers_default: int,
    semantic_top_k: int,
    batch_size: int = 128,
) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools/eval/eval_system_pipeline_v19.py"),
        "--seed",
        str(seed),
        "--data_root",
        str(data_root),
        "--data_root_scope",
        "all",
        "--gate_detector_path",
        str(gate_detector_path),
        "--router_ckpt",
        str(router_ckpt),
        "--experts_root",
        str(experts_root),
        "--output_dir",
        str(output_dir),
        "--gate_mode",
        HISTORICAL_BEST_PIPELINE.gate_mode,
        "--multi_prototype_path",
        str(multi_prototype_path),
        "--multi_proto_threshold_mode",
        "fixed",
        "--multi_proto_id_threshold",
        str(multi_proto_id_threshold),
        "--semantic_gate_enabled",
        "--semantic_gate_mode",
        HISTORICAL_BEST_PIPELINE.semantic_gate_mode,
        "--semantic_prompt_version",
        str(semantic_prompt_version),
        "--semantic_gate_threshold",
        str(semantic_gate_threshold),
        "--semantic_uncertain_low",
        str(semantic_uncertain_low),
        "--semantic_uncertain_high",
        str(semantic_uncertain_high),
        "--prototype_centers_default",
        str(prototype_centers_default),
        "--semantic_top_k",
        str(semantic_top_k),
        "--semantic_decision_policy",
        "threshold",
        "--batch_size",
        str(batch_size),
    ]
    if device is not None:
        cmd.extend(["--device", str(device)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen historical Prototype Gate v19 baseline"
    )
    parser.add_argument("--exp_id", default="prototype_gate_frozen_2026-04-09")
    parser.add_argument("--seed", type=int, default=HISTORICAL_BEST_PIPELINE.eval_seed)
    parser.add_argument("--data_root", default="data/v19")
    parser.add_argument(
        "--gate_detector_path",
        default=str(FROZEN_GATE_DETECTOR),
        help="Frozen historical gate detector path.",
    )
    parser.add_argument(
        "--gate_detector_path_legacy",
        default="",
        help="Optional legacy detector path fallback for older snapshots.",
    )
    parser.add_argument(
        "--router_ckpt",
        default=str(FROZEN_ROUTER_CKPT),
        help="Frozen historical router checkpoint.",
    )
    parser.add_argument(
        "--router_ckpt_legacy",
        default="",
        help="Optional legacy router checkpoint fallback for older snapshots.",
    )
    parser.add_argument(
        "--experts_root",
        default=str(FROZEN_EXPERTS_ROOT),
        help="Frozen historical expert checkpoint root.",
    )
    parser.add_argument(
        "--experts_root_legacy",
        default="",
        help="Optional legacy experts root fallback for older snapshots.",
    )
    parser.add_argument(
        "--multi_prototype_path",
        default=str(FROZEN_MULTI_PROTO_PATH),
        help="Frozen multi-prototype payload path.",
    )
    parser.add_argument(
        "--multi_prototype_path_legacy",
        default="",
        help="Optional legacy multi-prototype payload fallback.",
    )
    parser.add_argument(
        "--semantic_prompt_version",
        default=HISTORICAL_BEST_PIPELINE.semantic_prompt_version,
        help="Historical prompt version used by the semantic gate.",
    )
    parser.add_argument(
        "--semantic_gate_threshold",
        type=float,
        default=HISTORICAL_BEST_PIPELINE.semantic_gate_threshold,
        help="Frozen semantic accept threshold from the archived best run.",
    )
    parser.add_argument(
        "--semantic_uncertain_low",
        type=float,
        default=HISTORICAL_BEST_PIPELINE.semantic_uncertain_low,
        help="Historical uncertain interval lower bound.",
    )
    parser.add_argument(
        "--semantic_uncertain_high",
        type=float,
        default=HISTORICAL_BEST_PIPELINE.semantic_uncertain_high,
        help="Historical uncertain interval upper bound.",
    )
    parser.add_argument(
        "--prototype_centers_default",
        type=int,
        default=HISTORICAL_BEST_PIPELINE.prototype_centers_default,
        help="Frozen prototype center count used by the archived best run.",
    )
    parser.add_argument(
        "--semantic_top_k",
        type=int,
        default=HISTORICAL_BEST_PIPELINE.semantic_top_k,
        help="Candidate top-k for the semantic gate prompt verifier.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=HISTORICAL_BEST_PIPELINE.eval_batch_size,
        help="Inference batch size for frozen baseline reproduction.",
    )
    parser.add_argument(
        "--multi_proto_id_threshold",
        type=float,
        default=FROZEN_MULTI_PROTO_ID_THRESHOLD,
        help="Frozen multi-prototype ID threshold from the archived best run.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override. Defaults to auto-detect CUDA when available.",
    )
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    device = args.device if args.device is not None else _default_device()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else FROZEN_OUTPUT_BASE / args.exp_id / FROZEN_OUTPUT_SUBDIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_gate_detector = _resolve_existing_any(
        [
            Path(args.gate_detector_path),
            Path(args.gate_detector_path_legacy) if args.gate_detector_path_legacy else None,
        ],
        "gate detector",
    )
    resolved_router_ckpt = resolve_frozen_router_ckpt(
        PROJECT_ROOT,
        requested_path=Path(args.router_ckpt),
        legacy_path=Path(args.router_ckpt_legacy) if args.router_ckpt_legacy else None,
    )
    resolved_experts_root = resolve_frozen_experts_root(
        PROJECT_ROOT,
        requested_path=Path(args.experts_root),
        legacy_path=Path(args.experts_root_legacy) if args.experts_root_legacy else None,
    )
    resolved_multi_proto_path = resolve_multi_prototype_path(
        PROJECT_ROOT,
        requested_path=Path(args.multi_prototype_path),
    )

    cmd = build_eval_command(
        seed=int(args.seed),
        output_dir=output_dir,
        data_root=Path(args.data_root),
        gate_detector_path=resolved_gate_detector,
        router_ckpt=resolved_router_ckpt,
        experts_root=resolved_experts_root,
        multi_prototype_path=resolved_multi_proto_path,
        device=device,
        semantic_prompt_version=str(args.semantic_prompt_version),
        multi_proto_id_threshold=float(args.multi_proto_id_threshold),
        semantic_gate_threshold=float(args.semantic_gate_threshold),
        semantic_uncertain_low=float(args.semantic_uncertain_low),
        semantic_uncertain_high=float(args.semantic_uncertain_high),
        prototype_centers_default=int(args.prototype_centers_default),
        semantic_top_k=int(args.semantic_top_k),
        batch_size=int(args.batch_size),
    )

    print("Running frozen Prototype Gate baseline...")
    print(f"Reference eval: {FROZEN_REFERENCE_EVAL}")
    print(" ".join(cmd))
    _run(cmd)

    manifest = {
        "canonical_pipeline_name": "prototype_gate_pipeline_frozen",
        "role": "historical_reproduction_baseline",
        "reference_eval_results": str(FROZEN_REFERENCE_EVAL),
        "config": {
            "data_root": str(args.data_root),
            "gate_mode": "multisphere",
            "multi_proto_threshold_mode": "fixed",
            "multi_proto_id_threshold": float(args.multi_proto_id_threshold),
            "semantic_gate_enabled": True,
            "semantic_gate_mode": "prototype",
            "semantic_decision_policy": "threshold",
            "semantic_gate_threshold": float(args.semantic_gate_threshold),
            "semantic_uncertain_low": float(args.semantic_uncertain_low),
            "semantic_uncertain_high": float(args.semantic_uncertain_high),
            "prototype_centers_default": int(args.prototype_centers_default),
            "semantic_top_k": int(args.semantic_top_k),
            "batch_size": int(args.batch_size),
            "semantic_prompt_version": str(args.semantic_prompt_version),
            "device": device,
            "seed": int(args.seed),
        },
        "paths": {
            "output_dir": str(output_dir),
            "eval_results": str(output_dir / "eval_results.json"),
            "predictions": str(output_dir / "predictions.json"),
            "run_manifest": str(output_dir / "run_manifest.json"),
        },
    }
    with open(output_dir / "prototype_baseline_identity.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(f"Done. Frozen baseline saved to: {output_dir}")


if __name__ == "__main__":
    main()
