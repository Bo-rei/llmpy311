#!/usr/bin/env python3
"""Orchestrate the multi-dataset v19 benchmark.

The script is intentionally conservative:

1. rebuild protocol artifacts for each dataset/KIR combination
2. optionally run the existing evaluation pipeline when checkpoints exist
3. write a compact aggregate summary for downstream analysis

The script does not train models by itself. It expects the user to place the
dataset-specific checkpoints under the conventional benchmark artifact root:

    <artifact_root>/<dataset_slug>/<kir_tag>/{gate,router,experts,prompt_verifier}/...

This keeps the orchestration layer simple while reusing the current project
training entrypoints unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.analysis.historical_best_pipeline_v19 import HISTORICAL_BEST_PIPELINE
from legacy.runtime import WorkspacePaths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = WorkspacePaths.discover(PROJECT_ROOT)
REBUILD_SCRIPT = PROJECT_ROOT / "scripts" / "data" / "active" / "rebuild_multi_dataset_v19.py"
EVAL_SCRIPT = PROJECT_ROOT / "tools" / "eval" / "eval_system_pipeline_v19.py"


def _run(cmd: List[str]) -> None:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def _kir_tag(kir: float) -> str:
    return f"kir{int(round(float(kir) * 100)):02d}"


def _slug(dataset: str) -> str:
    return dataset.strip().upper().replace("-", "_").lower()


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_gate_encoder_path(
    requested_gate_encoder_path: Optional[str],
    model_path: str,
) -> str:
    if requested_gate_encoder_path is not None and str(requested_gate_encoder_path).strip() != "":
        return str(requested_gate_encoder_path)
    return str(HISTORICAL_BEST_PIPELINE.strict_replay_defaults()["gate_encoder_path"])


def _apply_benchmark_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill historical-profile defaults without clobbering explicit overrides."""
    if str(args.gate_mode) != HISTORICAL_BEST_PIPELINE.gate_mode:
        return args

    defaults = {
        "semantic_gate_mode": HISTORICAL_BEST_PIPELINE.semantic_gate_mode,
        "semantic_prompt_version": HISTORICAL_BEST_PIPELINE.semantic_prompt_version,
        "semantic_gate_threshold": HISTORICAL_BEST_PIPELINE.semantic_gate_threshold,
        "semantic_uncertain_low": HISTORICAL_BEST_PIPELINE.semantic_uncertain_low,
        "semantic_uncertain_high": HISTORICAL_BEST_PIPELINE.semantic_uncertain_high,
        "semantic_top_k": HISTORICAL_BEST_PIPELINE.semantic_top_k,
        "semantic_tuning_mode": HISTORICAL_BEST_PIPELINE.semantic_tuning_mode,
        "prototype_centers_default": HISTORICAL_BEST_PIPELINE.prototype_centers_default,
        "multi_proto_id_threshold": HISTORICAL_BEST_PIPELINE.multi_proto_id_threshold,
        "multi_proto_threshold_mode": HISTORICAL_BEST_PIPELINE.multi_proto_threshold_mode,
        "semantic_verifier_lora_r": HISTORICAL_BEST_PIPELINE.semantic_verifier_lora_r,
        "semantic_verifier_lora_alpha": HISTORICAL_BEST_PIPELINE.semantic_verifier_lora_alpha,
    }
    for field_name, default_value in defaults.items():
        current_value = getattr(args, field_name, None)
        if current_value is None:
            setattr(args, field_name, default_value)
            continue
        if isinstance(current_value, str) and str(current_value).strip() == "":
            setattr(args, field_name, default_value)

    return args


def _resolve_effective_eval_config(dataset: str, args: argparse.Namespace) -> Dict[str, Any]:
    requested_profile = str(getattr(args, "eval_profile", "auto"))
    effective_profile = "historical_best" if requested_profile == "auto" else requested_profile

    config = {
        "eval_profile": effective_profile,
        "semantic_gate_enabled": True,
        "semantic_gate_mode": str(getattr(args, "semantic_gate_mode")),
        "semantic_prompt_version": str(getattr(args, "semantic_prompt_version")),
        "semantic_tuning_mode": str(getattr(args, "semantic_tuning_mode")),
        "semantic_gate_threshold": float(getattr(args, "semantic_gate_threshold")),
        "semantic_uncertain_low": float(getattr(args, "semantic_uncertain_low")),
        "semantic_uncertain_high": float(getattr(args, "semantic_uncertain_high")),
        "semantic_top_k": int(getattr(args, "semantic_top_k")),
        "prototype_centers_default": int(getattr(args, "prototype_centers_default")),
        "semantic_fusion_alpha": float(getattr(args, "semantic_fusion_alpha")),
        "semantic_fusion_beta": float(getattr(args, "semantic_fusion_beta")),
        "semantic_decision_policy": str(getattr(args, "semantic_decision_policy")),
        "semantic_low_conf_threshold": float(getattr(args, "semantic_low_conf_threshold")),
        "semantic_high_conf_threshold": float(getattr(args, "semantic_high_conf_threshold")),
        "semantic_verifier_threshold": float(getattr(args, "semantic_verifier_threshold")),
        "semantic_verifier_lora_r": int(getattr(args, "semantic_verifier_lora_r")),
        "semantic_verifier_lora_alpha": int(getattr(args, "semantic_verifier_lora_alpha")),
        "device": str(getattr(args, "device", None)) if getattr(args, "device", None) is not None else None,
        "gate_encoder_path": str(getattr(args, "gate_encoder_path", None)) if getattr(args, "gate_encoder_path", None) is not None else None,
        "gate_mode": str(getattr(args, "gate_mode", HISTORICAL_BEST_PIPELINE.gate_mode)),
        "gate_radius_scale": float(getattr(args, "gate_radius_scale", 1.0)),
        "multi_prototype_path": str(getattr(args, "multi_prototype_path", "outputs/multi_prototypes_v19/prototypes.json")),
        "multi_proto_id_threshold": float(getattr(args, "multi_proto_id_threshold", HISTORICAL_BEST_PIPELINE.multi_proto_id_threshold)),
        "multi_proto_threshold_mode": str(getattr(args, "multi_proto_threshold_mode", HISTORICAL_BEST_PIPELINE.multi_proto_threshold_mode)),
        "batch_size": int(getattr(args, "batch_size", 64)),
    }

    if effective_profile in {"verifier_retained", "stackoverflow20k"}:
        config["semantic_gate_mode"] = "llm_verifier"
        config["semantic_tuning_mode"] = "val_macro_f1"

    return config


def _maybe_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-dataset v19 benchmark")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CLINC150", "BANKING77-OOS", "SNIPS"],
    )
    parser.add_argument("--kir_values", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root_base", default=str(PATHS.prepared_data_root / "multidataset/v19"))
    parser.add_argument("--artifact_root", default=str(PATHS.artifact_root / "outputs/experiments/multi_dataset_v19"))
    parser.add_argument("--clinc_root", default=str(PATHS.source_data_root / "clinc150/data"))
    parser.add_argument("--banking_root", default=str(PATHS.source_data_root / "banking77_oos"))
    parser.add_argument("--snips_root", default=str(PATHS.source_data_root / "snips"))
    parser.add_argument("--stackoverflow_root", default=str(PATHS.source_data_root / "stackoverflow"))
    parser.add_argument(
        "--stackoverflow_known_selection_strategy",
        default="seeded_random",
        choices=["nested_prefix", "seeded_random"],
        help="Known-intent selection strategy for StackOverflow KIR rebuilds.",
    )
    parser.add_argument("--clinc_oos_root", default=str(PATHS.source_data_root / "clinc150/data"))
    parser.add_argument("--model_path", default=str(PATHS.smollm135m))
    parser.add_argument(
        "--gate_encoder_path",
        default=None,
        help=(
            "Optional gate encoder path. Defaults to the strict historical "
            "Gate encoder (all-MiniLM-L6-v2) when omitted."
        ),
    )
    parser.add_argument(
        "--semantic_gate_mode",
        default=HISTORICAL_BEST_PIPELINE.semantic_gate_mode,
        choices=["prototype", "llm_verifier", "fusion"],
    )
    parser.add_argument(
        "--semantic_prompt_version",
        default=HISTORICAL_BEST_PIPELINE.semantic_prompt_version,
    )
    parser.add_argument(
        "--semantic_tuning_mode",
        default=HISTORICAL_BEST_PIPELINE.semantic_tuning_mode,
        choices=["fixed", "val_macro_f1"],
    )
    parser.add_argument(
        "--semantic_gate_threshold",
        type=float,
        default=HISTORICAL_BEST_PIPELINE.semantic_gate_threshold,
    )
    parser.add_argument(
        "--semantic_uncertain_low",
        type=float,
        default=HISTORICAL_BEST_PIPELINE.semantic_uncertain_low,
    )
    parser.add_argument(
        "--semantic_uncertain_high",
        type=float,
        default=HISTORICAL_BEST_PIPELINE.semantic_uncertain_high,
    )
    parser.add_argument(
        "--semantic_top_k",
        type=int,
        default=HISTORICAL_BEST_PIPELINE.semantic_top_k,
    )
    parser.add_argument(
        "--prototype_centers_default",
        type=int,
        default=HISTORICAL_BEST_PIPELINE.prototype_centers_default,
    )
    parser.add_argument(
        "--gate_mode",
        default=HISTORICAL_BEST_PIPELINE.gate_mode,
        choices=["multisphere", "multi_prototype"],
    )
    parser.add_argument(
        "--gate_radius_scale",
        type=float,
        default=1.0,
        help="Eval-time scale applied to multisphere radii; values below 1.0 make OOS rejection stricter.",
    )
    parser.add_argument(
        "--multi_prototype_path",
        default="outputs/multi_prototypes_v19/prototypes.json",
    )
    parser.add_argument(
        "--multi_proto_id_threshold",
        type=float,
        default=HISTORICAL_BEST_PIPELINE.multi_proto_id_threshold,
    )
    parser.add_argument(
        "--multi_proto_threshold_mode",
        default=HISTORICAL_BEST_PIPELINE.multi_proto_threshold_mode,
        choices=["fixed", "val_macro_f1"],
    )
    parser.add_argument("--semantic_fusion_alpha", type=float, default=0.7)
    parser.add_argument("--semantic_fusion_beta", type=float, default=0.3)
    parser.add_argument("--semantic_decision_policy", default="threshold")
    parser.add_argument("--semantic_low_conf_threshold", type=float, default=0.80)
    parser.add_argument("--semantic_high_conf_threshold", type=float, default=0.90)
    parser.add_argument("--semantic_verifier_threshold", type=float, default=0.50)
    parser.add_argument(
        "--semantic_verifier_lora_r",
        type=int,
        default=HISTORICAL_BEST_PIPELINE.semantic_verifier_lora_r,
    )
    parser.add_argument(
        "--semantic_verifier_lora_alpha",
        type=int,
        default=HISTORICAL_BEST_PIPELINE.semantic_verifier_lora_alpha,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--eval_profile",
        default="auto",
        choices=["auto", "historical_best", "verifier_retained", "stackoverflow20k"],
        help=(
            "Evaluation profile. auto resolves to historical_best for every dataset; "
            "verifier_retained applies llm_verifier+val_macro_f1 uniformly; "
            "stackoverflow20k is a legacy compatibility alias."
        ),
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Only rebuild the datasets; do not run evaluation.",
    )
    parser.add_argument(
        "--export_gate_diagnostics",
        action="store_true",
        help="Forward gate diagnostics export to the eval script.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    LOGGER = logging.getLogger(__name__)

    data_root_base = Path(args.data_root_base)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    device = str(args.device) if args.device is not None else _default_device()
    gate_encoder_path = _resolve_gate_encoder_path(
        requested_gate_encoder_path=args.gate_encoder_path,
        model_path=str(args.model_path),
    )

    args = _apply_benchmark_profile_defaults(args)

    summary: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        dataset_key = dataset.strip().upper()
        dataset_slug = _slug(dataset)
        for kir in args.kir_values:
            kir = float(kir)
            kir_tag = _kir_tag(kir)
            data_root = data_root_base / dataset_slug / f"{kir_tag}_seed{int(args.seed)}"
            run_artifact_root = artifact_root / dataset_slug / f"{kir_tag}_seed{int(args.seed)}"
            run_artifact_root.mkdir(parents=True, exist_ok=True)

            rebuild_cmd = [
                sys.executable,
                str(REBUILD_SCRIPT),
                "--datasets",
                dataset_key,
                "--kir_values",
                str(kir),
                "--seed",
                str(args.seed),
                "--output_root",
                str(data_root_base),
                "--clinc_root",
                str(args.clinc_root),
                "--banking_root",
                str(args.banking_root),
                "--snips_root",
                str(args.snips_root),
                "--stackoverflow_root",
                str(args.stackoverflow_root),
                "--stackoverflow_known_selection_strategy",
                str(args.stackoverflow_known_selection_strategy),
                "--clinc_oos_root",
                str(args.clinc_oos_root),
            ]
            LOGGER.info("Rebuilding %s @ kir=%.2f -> %s", dataset_key, kir, data_root)
            _run(rebuild_cmd)

            effective_eval_config = _resolve_effective_eval_config(dataset_key, args)
            known_intents_manifest = _load_json(data_root / "KNOWN_INTENTS.json")
            selection_protocol = known_intents_manifest.get("selection_protocol", {})
            known_selection_strategy = str(
                selection_protocol.get("method", "unknown")
            ).replace("official_split_single_domain_", "").replace("_kir", "")

            run_item: Dict[str, Any] = {
                "dataset": dataset_key,
                "dataset_slug": dataset_slug,
                "kir": kir,
                "seed": int(args.seed),
                "data_root": str(data_root),
                "artifact_root": str(run_artifact_root),
                "rebuild_manifest": str(data_root / "MANIFEST.json"),
                "audit": str(data_root / "AUDIT.json"),
                "known_selection_strategy": known_selection_strategy,
                "effective_eval_config": effective_eval_config,
                "requested_eval_config": {
                    "device": str(device),
                    "gate_encoder_path": str(gate_encoder_path),
                    "gate_mode": str(args.gate_mode),
                    "gate_radius_scale": float(args.gate_radius_scale),
                    "multi_prototype_path": str(args.multi_prototype_path),
                    "multi_proto_id_threshold": float(args.multi_proto_id_threshold),
                    "multi_proto_threshold_mode": str(args.multi_proto_threshold_mode),
                    "semantic_gate_enabled": True,
                    "semantic_gate_mode": str(args.semantic_gate_mode),
                    "semantic_prompt_version": str(args.semantic_prompt_version),
                    "semantic_tuning_mode": str(args.semantic_tuning_mode),
                    "semantic_gate_threshold": float(args.semantic_gate_threshold),
                    "semantic_uncertain_low": float(args.semantic_uncertain_low),
                    "semantic_uncertain_high": float(args.semantic_uncertain_high),
                    "semantic_top_k": int(args.semantic_top_k),
                    "prototype_centers_default": int(args.prototype_centers_default),
                    "semantic_fusion_alpha": float(args.semantic_fusion_alpha),
                    "semantic_fusion_beta": float(args.semantic_fusion_beta),
                    "semantic_decision_policy": str(args.semantic_decision_policy),
                    "semantic_low_conf_threshold": float(args.semantic_low_conf_threshold),
                    "semantic_high_conf_threshold": float(args.semantic_high_conf_threshold),
                    "semantic_verifier_threshold": float(args.semantic_verifier_threshold),
                    "semantic_verifier_lora_r": int(args.semantic_verifier_lora_r),
                    "semantic_verifier_lora_alpha": int(args.semantic_verifier_lora_alpha),
                    "batch_size": int(args.batch_size),
                },
            }

            if args.skip_eval:
                run_item["status"] = "rebuild_only"
                summary.append(run_item)
                continue

            router_ckpt = run_artifact_root / "router" / "best_model.pt"
            experts_root = run_artifact_root / "experts"
            gate_detector_path = run_artifact_root / "gate" / "corrected_multisphere_detector.json"
            prompt_semantic_verifier_ckpt = run_artifact_root / "prompt_verifier" / "best_model.pt"
            eval_dir = run_artifact_root / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)

            run_item["checkpoint_paths"] = {
                "router_ckpt": str(router_ckpt),
                "experts_root": str(experts_root),
                "gate_detector_path": str(gate_detector_path),
                "prompt_semantic_verifier_ckpt": str(prompt_semantic_verifier_ckpt),
            }
            run_item["prompt_semantic_verifier_expected_used"] = bool(
                prompt_semantic_verifier_ckpt.exists()
                and str(effective_eval_config["semantic_gate_mode"]) in {"llm_verifier", "fusion"}
            )

            if not router_ckpt.exists() or not experts_root.exists() or not gate_detector_path.exists():
                run_item["status"] = "missing_checkpoints"
                summary.append(run_item)
                LOGGER.warning(
                    "Skipping eval for %s @ kir=%.2f because checkpoints are missing.",
                    dataset_key,
                    kir,
                )
                continue

            eval_cmd = [
                sys.executable,
                str(EVAL_SCRIPT),
                "--data_root",
                str(data_root),
                "--data_root_scope",
                "all",
                "--gate_detector_path",
                str(gate_detector_path),
                "--gate_encoder_path",
                str(gate_encoder_path),
                "--router_ckpt",
                str(router_ckpt),
                "--experts_root",
                str(experts_root),
                "--output_dir",
                str(eval_dir),
                "--batch_size",
                str(effective_eval_config["batch_size"]),
                "--semantic_gate_enabled",
                "--gate_mode",
                str(effective_eval_config["gate_mode"]),
                "--gate_radius_scale",
                str(effective_eval_config["gate_radius_scale"]),
                "--multi_prototype_path",
                str(effective_eval_config["multi_prototype_path"]),
                "--multi_proto_id_threshold",
                str(effective_eval_config["multi_proto_id_threshold"]),
                "--multi_proto_threshold_mode",
                str(effective_eval_config["multi_proto_threshold_mode"]),
                "--semantic_gate_mode",
                str(effective_eval_config["semantic_gate_mode"]),
                "--semantic_prompt_version",
                str(effective_eval_config["semantic_prompt_version"]),
                "--semantic_tuning_mode",
                str(effective_eval_config["semantic_tuning_mode"]),
                "--semantic_gate_threshold",
                str(effective_eval_config["semantic_gate_threshold"]),
                "--semantic_uncertain_low",
                str(effective_eval_config["semantic_uncertain_low"]),
                "--semantic_uncertain_high",
                str(effective_eval_config["semantic_uncertain_high"]),
                "--semantic_top_k",
                str(effective_eval_config["semantic_top_k"]),
                "--prototype_centers_default",
                str(effective_eval_config["prototype_centers_default"]),
                "--semantic_fusion_alpha",
                str(effective_eval_config["semantic_fusion_alpha"]),
                "--semantic_fusion_beta",
                str(effective_eval_config["semantic_fusion_beta"]),
                "--semantic_decision_policy",
                str(effective_eval_config["semantic_decision_policy"]),
                "--semantic_low_conf_threshold",
                str(effective_eval_config["semantic_low_conf_threshold"]),
                "--semantic_high_conf_threshold",
                str(effective_eval_config["semantic_high_conf_threshold"]),
                "--semantic_verifier_threshold",
                str(effective_eval_config["semantic_verifier_threshold"]),
                "--semantic_verifier_lora_r",
                str(effective_eval_config["semantic_verifier_lora_r"]),
                "--semantic_verifier_lora_alpha",
                str(effective_eval_config["semantic_verifier_lora_alpha"]),
                "--device",
                str(device),
            ]
            if prompt_semantic_verifier_ckpt.exists():
                eval_cmd.extend(["--prompt_semantic_verifier_ckpt", str(prompt_semantic_verifier_ckpt)])
            if args.export_gate_diagnostics:
                eval_cmd.append("--export_gate_diagnostics")

            LOGGER.info("Evaluating %s @ kir=%.2f", dataset_key, kir)
            _run(eval_cmd)

            eval_results_path = eval_dir / "eval_results.json"
            run_item["status"] = "evaluated" if eval_results_path.exists() else "eval_missing"
            run_item["eval_results_path"] = str(eval_results_path)
            if eval_results_path.exists():
                results = _load_json(eval_results_path)
                run_item["metrics"] = results.get("metrics", {})
                run_item["primary_metrics"] = results.get("metrics", {}).get("primary_metrics", {})
            summary.append(run_item)

    summary_payload = {
        "seed": int(args.seed),
        "device": device,
        "datasets": [str(dataset) for dataset in args.datasets],
        "kir_values": [float(value) for value in args.kir_values],
        "runs": summary,
    }
    summary_path = artifact_root / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False)
    LOGGER.info("Wrote benchmark summary -> %s", summary_path)


if __name__ == "__main__":
    main()
