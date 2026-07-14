#!/usr/bin/env python3
"""Run a temporary minimal ablation/debug set for v19.

This runner intentionally reuses already-trained component checkpoints and
only launches evaluation-time ablations:

- Full: multisphere gate + prompt semantic verifier
- w/o Gate: force Router/Expert path and disable OOS gate decisions
- w/o Verifier: fast multisphere gate only

It deliberately does not create a post-hoc "w/o Router" number. The previous
gate-nearest-intent postprocess preserved the full gate decision and collapsed
to the same result as Full on CLINC, so it is not a fair structural ablation.

This script is not the canonical paper-facing ablation entrypoint. Use
``tools/analysis/run_structure_backbone_ablation_v19.py`` for the official
latest-strongest structure/backbone suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import WorkspacePaths  # noqa: E402

PATHS = WorkspacePaths.discover(PROJECT_ROOT)

EVAL_SCRIPT = PROJECT_ROOT / "tools" / "eval" / "eval_system_pipeline_v19.py"
DEFAULT_VARIANTS = ("full", "wo_gate", "wo_verifier")
DEFAULT_DATASETS = ("clinc150", "banking77_oos", "stackoverflow")


@dataclass(frozen=True)
class DatasetRun:
    dataset: str
    slug: str
    data_root: Path
    artifact_root: Path
    gate_encoder_path: Path
    expected_gate_dim: int


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _run(cmd: List[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_status() -> Dict[str, Any]:
    try:
        import torch

        return {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    except Exception as exc:
        return {
            "available": False,
            "device_count": 0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "error": repr(exc),
        }


def _preflight_device(args: argparse.Namespace) -> None:
    if str(args.device).startswith("cuda") and not _cuda_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this process. "
            "Do not run long ablations on CPU; launch from a GPU-visible shell "
            "or pass --allow_cpu --device cpu for an explicit CPU fallback."
        )
    if str(args.device) == "cpu" and not bool(args.allow_cpu):
        raise RuntimeError(
            "Refusing to run long ablations on CPU by default. "
            "Use --allow_cpu only for a deliberate smoke/debug run."
        )


def _first_sphere_center_dim(detector_path: Path) -> int:
    detector = _load_json(detector_path)
    spheres = detector.get("spheres", [])
    if not spheres:
        raise ValueError(f"No spheres in detector: {detector_path}")
    center = spheres[0].get("center")
    if not isinstance(center, list) or not center:
        raise ValueError(f"Malformed first sphere center in detector: {detector_path}")
    return len(center)


def _router_num_classes(router_ckpt: Path) -> int:
    import torch

    state = torch.load(router_ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported router checkpoint format: {router_ckpt}")
    for key in ("classifier.weight", "model.classifier.weight", "router.classifier.weight"):
        value = state.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    for key, value in state.items():
        if key.endswith("classifier.weight") and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    raise ValueError(f"Could not find classifier.weight in router checkpoint: {router_ckpt}")


def _data_num_router_classes(data_root: Path) -> int:
    router_train = data_root / "router" / "train.json"
    records = _load_json(router_train)
    labels = {int(row["label"]) for row in records}
    return len(labels)


def _validate_run(run: DatasetRun, variants: Sequence[str]) -> Dict[str, Any]:
    required = {
        "data_gate_test": run.data_root / "gate" / "test.json",
        "data_router_train": run.data_root / "router" / "train.json",
        "gate_detector": run.artifact_root / "gate" / "corrected_multisphere_detector.json",
        "router_ckpt": run.artifact_root / "router" / "best_model.pt",
        "experts_manifest": run.artifact_root / "experts" / "stage_manifest.json",
    }
    if "full" in variants:
        required["prompt_verifier_ckpt"] = run.artifact_root / "prompt_verifier" / "best_model.pt"

    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{run.dataset} is missing required current-pipeline artifacts: {missing}"
        )

    detector_dim = _first_sphere_center_dim(required["gate_detector"])
    if detector_dim != run.expected_gate_dim:
        raise ValueError(
            f"{run.dataset} detector dim {detector_dim} does not match configured "
            f"gate encoder dim {run.expected_gate_dim}: {run.gate_encoder_path}"
        )

    router_classes = _router_num_classes(required["router_ckpt"])
    data_router_classes = _data_num_router_classes(run.data_root)
    if router_classes != data_router_classes:
        raise ValueError(
            f"{run.dataset} router class mismatch: checkpoint={router_classes}, "
            f"data={data_router_classes}. Artifact root is not aligned with data root."
        )

    return {
        "dataset": run.dataset,
        "slug": run.slug,
        "data_root": str(run.data_root),
        "artifact_root": str(run.artifact_root),
        "gate_encoder_path": str(run.gate_encoder_path),
        "gate_dim": detector_dim,
        "router_classes": router_classes,
    }


def _base_eval_cmd(run: DatasetRun, output_dir: Path, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        "--data_root",
        str(run.data_root),
        "--data_root_scope",
        "all",
        "--model_path",
        str(args.model_path),
        "--gate_encoder_path",
        str(run.gate_encoder_path),
        "--gate_detector_path",
        str(run.artifact_root / "gate" / "corrected_multisphere_detector.json"),
        "--router_ckpt",
        str(run.artifact_root / "router" / "best_model.pt"),
        "--experts_root",
        str(run.artifact_root / "experts"),
        "--prompt_semantic_verifier_ckpt",
        str(run.artifact_root / "prompt_verifier" / "best_model.pt"),
        "--output_dir",
        str(output_dir),
        "--batch_size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--gate_mode",
        "multisphere",
        "--semantic_prompt_version",
        str(args.semantic_prompt_version),
        "--semantic_top_k",
        str(args.semantic_top_k),
        "--semantic_verifier_lora_r",
        str(args.semantic_verifier_lora_r),
        "--semantic_verifier_lora_alpha",
        str(args.semantic_verifier_lora_alpha),
        "--export_gate_diagnostics",
    ]


def _run_variant(
    *,
    run: DatasetRun,
    variant: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if not bool(args.force) and (output_dir / "eval_results.json").exists():
        return
    cmd = _base_eval_cmd(run, output_dir, args)
    if variant == "full":
        cmd.extend(
            [
                "--semantic_gate_enabled",
                "--semantic_gate_mode",
                "llm_verifier",
                "--semantic_tuning_mode",
                "val_macro_f1",
            ]
        )
    elif variant == "wo_gate":
        cmd.extend(
            [
                "--ablation_no_gate",
                "--no_gate_mode",
                "disabled",
                "--semantic_gate_mode",
                "none",
            ]
        )
    elif variant == "wo_verifier":
        cmd.extend(["--semantic_gate_mode", "none"])
    else:
        raise ValueError(f"Unsupported subprocess variant: {variant}")
    _run(cmd)


def _metric_row(run: DatasetRun, variant: str, eval_path: Path) -> Dict[str, Any]:
    metrics = _load_json(eval_path)["metrics"]
    return {
        "dataset": run.dataset,
        "variant": variant,
        "eval_results": str(eval_path),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
        "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
        "known_macro_f1": float(metrics.get("known_macro_f1", 0.0)),
        "oos_f1": float(metrics.get("oos_f1", 0.0)),
        "gate_id_recall": float(metrics.get("gate_id_recall", 0.0)),
        "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
        "avg_ms_per_sample": float(
            metrics.get("latency", {}).get("avg_ms_per_sample", 0.0)
        ),
    }


def _write_summary(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    _write_json(output_root / "ablation_summary.json", {"runs": rows})
    if not rows:
        return
    csv_path = output_root / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal paper ablations for v19")
    parser.add_argument("--kir_tag", default="kir50_seed42")
    parser.add_argument("--output_root", default="outputs/experiments/paper_ablation_gpu_20260425")
    parser.add_argument("--model_path", default=str(PATHS.smollm135m))
    parser.add_argument("--gate_encoder_path", default="all-MiniLM-L6-v2")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=DEFAULT_DATASETS)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS), choices=DEFAULT_VARIANTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Explicitly allow CPU fallback for smoke/debug only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun variants even when eval_results.json already exists.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate dataset/artifact compatibility and exit without launching evaluations.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--semantic_prompt_version", default="ranking_v1")
    parser.add_argument("--semantic_top_k", type=int, default=3)
    parser.add_argument("--semantic_verifier_lora_r", type=int, default=32)
    parser.add_argument("--semantic_verifier_lora_alpha", type=int, default=64)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    run_by_slug = {
        "clinc150": DatasetRun(
            dataset="CLINC150",
            slug="clinc150",
            data_root=Path("data/multidataset/v19/clinc150") / args.kir_tag,
            artifact_root=Path("outputs/experiments/multi_dataset_v19_retrain_20260413/clinc150")
            / args.kir_tag,
            gate_encoder_path=Path(args.model_path),
            expected_gate_dim=576,
        ),
        "banking77_oos": DatasetRun(
            dataset="BANKING77-OOS",
            slug="banking77_oos",
            data_root=Path("data/multidataset/v19/banking77_oos") / args.kir_tag,
            artifact_root=Path("outputs/experiments/multi_dataset_v19/banking77_oos")
            / args.kir_tag,
            gate_encoder_path=Path(args.model_path),
            expected_gate_dim=576,
        ),
        "stackoverflow": DatasetRun(
            dataset="STACKOVERFLOW",
            slug="stackoverflow",
            data_root=Path("data/multidataset/v19/stackoverflow") / args.kir_tag,
            artifact_root=Path("outputs/experiments/stackoverflow20k_seeded_random_20260422/stackoverflow")
            / args.kir_tag,
            gate_encoder_path=Path(args.gate_encoder_path),
            expected_gate_dim=384,
        ),
    }
    runs = [run_by_slug[slug] for slug in args.datasets]

    validation = [_validate_run(run, args.variants) for run in runs]
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "cuda": _cuda_status(),
                    "datasets": validation,
                    "variants": args.variants,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    _preflight_device(args)

    rows: List[Dict[str, Any]] = []
    for run in runs:
        for variant in args.variants:
            variant_dir = output_root / run.slug / args.kir_tag / variant
            _run_variant(run=run, variant=variant, output_dir=variant_dir, args=args)
            rows.append(_metric_row(run, variant, variant_dir / "eval_results.json"))

    _write_summary(output_root, rows)
    print(
        json.dumps(
            {
                "summary": str(output_root / "ablation_summary.csv"),
                "runs": len(rows),
                "variants": args.variants,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
