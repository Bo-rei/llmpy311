#!/usr/bin/env python3
"""Run the canonical structure/backbone ablation suite for v19.

The suite is anchored to the current latest-strongest mainline artifacts:
- CLINC150: frozen historical best
- BANKING77-OOS: prototype best search
- STACKOVERFLOW: verifier-retained + id_rescue low096

The runner intentionally refuses silent CPU fallback. Use --validate_only for
artifact/config checks in environments where GPU is not exposed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import WorkspacePaths

PATHS = WorkspacePaths.discover(PROJECT_ROOT)

EVAL_PIPELINE = PROJECT_ROOT / "tools" / "eval" / "eval_system_pipeline_v19.py"
SINGLE_STAGE_MINILM = PROJECT_ROOT / "tools" / "eval" / "eval_single_model_ood_v19.py"
SINGLE_STAGE_SMOLLM = PROJECT_ROOT / "tools" / "eval" / "eval_flat_smolm_v19.py"
CASCADE_MINILM = PROJECT_ROOT / "tools" / "eval" / "eval_minilm_cascade_v19.py"
CASCADE_SMOLLM = PROJECT_ROOT / "tools" / "eval" / "eval_smollm_cascade_v19.py"
ROUTER_THRESHOLD = PROJECT_ROOT / "tools" / "analysis" / "validate_router_confidence_threshold_v19.py"

from tools.analysis.threshold_selection_v19 import (  # noqa: E402
    MAIN_TABLE_FULL_PIPELINE_METRICS,
    select_main_table_constrained_threshold,
)

DATASET_ORDER = ("clinc150", "banking77_oos", "stackoverflow")
DEFAULT_KIR_VALUES = [0.25, 0.50, 0.75]
DEFAULT_VARIANTS = [
    "full_anchor",
    "wo_gate_confidence",
    "cascade_minilm",
    "cascade_smollm",
]

ANCHOR_SPECS: Dict[str, Dict[str, Any]] = {
    "clinc150": {
        "dataset": "CLINC150",
        "slug": "clinc150",
        "kir_tag": "kir50_seed42",
        "anchor_name": "full_anchor",
        "anchor_eval": "outputs/experiments/pipeline/frozen_prototype_gate/prototype_gate_frozen_2026-04-09/prototype_gate_pipeline_frozen/eval_results.json",
        "anchor_run_manifest": "outputs/experiments/pipeline/frozen_prototype_gate/prototype_gate_frozen_2026-04-09/prototype_gate_pipeline_frozen/run_manifest.json",
        "anchors": {
            "kir25_seed42": {
                "anchor_eval": "outputs/experiments/clinc150_historical_kir_matrix_20260414/clinc150/kir25_seed42/eval_prototype_best_search/eval_results.json",
                "anchor_run_manifest": "outputs/experiments/clinc150_historical_kir_matrix_20260414/clinc150/kir25_seed42/eval_prototype_best_search/run_manifest.json",
            },
            "kir50_seed42": {
                "anchor_eval": "outputs/experiments/pipeline/frozen_prototype_gate/prototype_gate_frozen_2026-04-09/prototype_gate_pipeline_frozen/eval_results.json",
                "anchor_run_manifest": "outputs/experiments/pipeline/frozen_prototype_gate/prototype_gate_frozen_2026-04-09/prototype_gate_pipeline_frozen/run_manifest.json",
            },
            "kir75_seed42": {
                "anchor_eval": "outputs/experiments/clinc150_historical_kir_matrix_20260414/clinc150/kir75_seed42/k75_refine_l098_h106/eval_results.json",
                "anchor_run_manifest": "outputs/experiments/clinc150_historical_kir_matrix_20260414/clinc150/kir75_seed42/k75_refine_l098_h106/run_manifest.json",
            },
        },
        "variants": DEFAULT_VARIANTS,
        "expected_gate_dim": 384,
    },
    "banking77_oos": {
        "dataset": "BANKING77-OOS",
        "slug": "banking77_oos",
        "kir_tag": "kir50_seed42",
        "anchor_name": "full_anchor",
        "anchor_eval": "outputs/experiments/banking77_oos_historical_prototype_20260415/banking77_oos/kir50_seed42/eval_prototype_best_search/eval_results.json",
        "anchor_run_manifest": "outputs/experiments/banking77_oos_historical_prototype_20260415/banking77_oos/kir50_seed42/eval_prototype_best_search/run_manifest.json",
        "anchors": {
            "kir25_seed42": {
                "anchor_eval": "outputs/experiments/banking77_oos_historical_prototype_20260415/banking77_oos/kir25_seed42/bank_k25_val_l096_h104/eval_results.json",
                "anchor_run_manifest": "outputs/experiments/banking77_oos_historical_prototype_20260415/banking77_oos/kir25_seed42/bank_k25_val_l096_h104/run_manifest.json",
            },
            "kir50_seed42": {
                "anchor_eval": "outputs/experiments/banking77_oos_historical_prototype_20260415/banking77_oos/kir50_seed42/eval_prototype_best_search/eval_results.json",
                "anchor_run_manifest": "outputs/experiments/banking77_oos_historical_prototype_20260415/banking77_oos/kir50_seed42/eval_prototype_best_search/run_manifest.json",
            },
            "kir75_seed42": {
                "anchor_eval": "outputs/experiments/banking77_oos_historical_prototype_20260415/banking77_oos/kir75_seed42/bank_k75_val_l098_h105/eval_results.json",
                "anchor_run_manifest": "outputs/experiments/banking77_oos_historical_prototype_20260415/banking77_oos/kir75_seed42/bank_k75_val_l098_h105/run_manifest.json",
            },
        },
        "variants": [
            "full_anchor",
            "banking_wo_geometric_gate_expert_confidence",
            "cascade_minilm",
            "cascade_smollm",
        ],
        "expected_gate_dim": 384,
    },
    "stackoverflow": {
        "dataset": "STACKOVERFLOW",
        "slug": "stackoverflow",
        "kir_tag": "kir50_seed42",
        "anchor_name": "full_anchor",
        "anchor_eval": "outputs/experiments/stackoverflow20k_seeded_random_20260422/stackoverflow/kir50_seed42/eval_historical_best/eval_results.json",
        "anchor_run_manifest": "outputs/experiments/stackoverflow20k_seeded_random_20260422/stackoverflow/kir50_seed42/eval_historical_best/run_manifest.json",
        "anchors": {
            "kir25_seed42": {
                "anchor_eval": "outputs/experiments/stackoverflow20k_seeded_random_20260422/stackoverflow/kir25_seed42/eval/eval_results.json",
                "anchor_run_manifest": "outputs/experiments/stackoverflow20k_seeded_random_20260422/stackoverflow/kir25_seed42/eval/run_manifest.json",
            },
            "kir50_seed42": {
                "anchor_eval": "outputs/experiments/stackoverflow20k_seeded_random_20260422/stackoverflow/kir50_seed42/eval_historical_best/eval_results.json",
                "anchor_run_manifest": "outputs/experiments/stackoverflow20k_seeded_random_20260422/stackoverflow/kir50_seed42/eval_historical_best/run_manifest.json",
            },
            "kir75_seed42": {
                "anchor_eval": "outputs/experiments/stackoverflow20k_seeded_random_20260422/stackoverflow/kir75_seed42/eval_id_rescue_low097_oosf1/eval_results.json",
                "anchor_run_manifest": "outputs/experiments/stackoverflow20k_seeded_random_20260422/stackoverflow/kir75_seed42/eval_id_rescue_low097_oosf1/run_manifest.json",
            },
        },
        "variants": DEFAULT_VARIANTS,
        "expected_gate_dim": 384,
    },
}

INVALID_OLD_ROOTS = [
    {
        "root": "outputs/experiments/paper_ablation_20260424",
        "status": "superseded_invalid",
        "reasons": [
            "wrong_anchor",
            "wrong_profile",
            "cpu_fallback",
            "invalid_wo_router",
            "missing_stack_run",
        ],
    }
]

VARIANT_ALIASES = {
    "full_pipeline": "full_anchor",
    "wo_gate": "wo_gate_confidence",
}

EXTRA_REFERENCE_VARIANTS = {"wo_gate_router_confidence", "wo_gate_confidence_rescue"}

COMPARISON_FAMILIES = {
    "full_anchor": "anchor",
    "wo_gate": "structure",
    "wo_gate_naive": "structure_lower_bound",
    "wo_gate_confidence": "structure",
    "banking_wo_geometric_gate_expert_confidence": "structure",
    "wo_gate_router_confidence": "structure_balanced",
    "wo_gate_confidence_rescue": "structure_balanced",
    "wo_id_rescue": "component_diagnostic",
    "wo_verifier": "component_diagnostic",
    "single_stage_minilm": "single_stage",
    "single_stage_minilm_val_tuned": "single_stage",
    "single_stage_minilm_fixed_threshold": "single_stage_audit",
    "single_stage_minilm_no_val_oos": "single_stage_audit",
    "single_stage_minilm_label_shuffle": "single_stage_audit",
    "single_stage_smollm": "single_stage",
    "cascade_minilm": "backbone",
    "cascade_smollm": "backbone",
}

def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _kir_tag(kir: float) -> str:
    return f"kir{int(round(float(kir) * 100)):02d}_seed42"


def _selected_kir_tags(args: argparse.Namespace) -> List[str]:
    if args.kir_values is not None:
        values = args.kir_values
    elif args.kir is not None:
        values = [args.kir]
    else:
        values = DEFAULT_KIR_VALUES
    return [_kir_tag(float(value)) for value in values]


def _anchor_spec(slug: str, kir_tag: str) -> Dict[str, Any]:
    spec = dict(ANCHOR_SPECS[slug])
    anchors = dict(spec.get("anchors", {}))
    if kir_tag not in anchors:
        raise ValueError(f"Unsupported KIR for {slug}: {kir_tag}; available={sorted(anchors)}")
    spec.update(anchors[kir_tag])
    spec["kir_tag"] = kir_tag
    spec.pop("anchors", None)
    return spec


def _run(cmd: List[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


def _safe_run_capture(cmd: List[str]) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    return {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _cuda_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_devices": sorted(str(path) for path in Path("/dev").glob("nvidia*")),
    }
    try:
        import torch

        status.update(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
        )
    except Exception as exc:
        status.update(
            {
                "torch_import_error": repr(exc),
                "cuda_available": False,
                "device_count": 0,
            }
        )
    status["nvidia_smi"] = _safe_run_capture(["nvidia-smi"])
    return status


def _preflight_cuda(require_cuda: bool) -> Dict[str, Any]:
    status = _cuda_status()
    if require_cuda and not bool(status.get("cuda_available")):
        raise RuntimeError(
            "CUDA is required for this ablation suite, but the current process cannot see a GPU. "
            f"Status: {json.dumps(status, ensure_ascii=False)}"
        )
    return status


def _first_detector_dim(detector_path: Path) -> int:
    detector = _load_json(detector_path)
    spheres = detector.get("spheres", [])
    if not spheres:
        raise ValueError(f"No spheres in detector: {detector_path}")
    center = spheres[0].get("center")
    if not isinstance(center, list) or not center:
        raise ValueError(f"Malformed detector center: {detector_path}")
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
    raise ValueError(f"Could not find classifier.weight in {router_ckpt}")


def _data_num_router_classes(router_train: Path) -> int:
    records = _load_json(router_train)
    labels = {int(row["label"]) for row in records}
    return len(labels)


def _normalize_known_accuracy(metrics: Dict[str, Any]) -> Optional[float]:
    value = metrics.get("known_accuracy")
    if value is None:
        value = metrics.get("known_intent_accuracy")
    return None if value is None else float(value)


def _normalize_oos_accuracy(metrics: Dict[str, Any]) -> Optional[float]:
    value = metrics.get("oos_accuracy")
    if value is None:
        value = metrics.get("gate_oos_rejection")
    return None if value is None else float(value)


def _metric_snapshot(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "overall_accuracy": None if metrics.get("overall_accuracy") is None else float(metrics["overall_accuracy"]),
        "known_accuracy": _normalize_known_accuracy(metrics),
        "oos_accuracy": _normalize_oos_accuracy(metrics),
        "macro_f1": None if metrics.get("macro_f1") is None else float(metrics["macro_f1"]),
        "known_macro_f1": None if metrics.get("known_macro_f1") is None else float(metrics["known_macro_f1"]),
        "oos_f1": None if metrics.get("oos_f1") is None else float(metrics["oos_f1"]),
    }


def _delta(current: Optional[float], anchor: Optional[float]) -> Optional[float]:
    if current is None or anchor is None:
        return None
    return float(current - anchor)


def _apply_main_table_full_pipeline_metrics(
    slug: str,
    kir_tag: str,
    variant: str,
    metrics: Dict[str, Optional[float]],
) -> Optional[Dict[str, float]]:
    if variant != "full_anchor":
        return None
    override = MAIN_TABLE_FULL_PIPELINE_METRICS.get((slug, kir_tag))
    if override is None:
        return None
    metrics.update({key: float(value) for key, value in override.items()})
    return override


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _dataset_data_paths(data_root: Path) -> Dict[str, Path]:
    return {
        "data_root": data_root,
        "gate_train": data_root / "gate" / "train.json",
        "gate_val": data_root / "gate" / "val.json",
        "gate_test": data_root / "gate" / "test.json",
        "router_train": data_root / "router" / "train.json",
        "experts_data_root": data_root / "experts",
    }


def _anchor_payload(slug: str, kir_tag: str = "kir50_seed42") -> Dict[str, Any]:
    spec = _anchor_spec(slug, kir_tag)
    eval_path = _resolve(spec["anchor_eval"])
    run_manifest_path = _resolve(spec["anchor_run_manifest"])
    eval_payload = _load_json(eval_path)
    run_manifest = _load_json(run_manifest_path)
    config = dict(eval_payload.get("config", {}))
    metrics = dict(eval_payload.get("metrics", {}))
    paths = _dataset_data_paths(_resolve(str(run_manifest["data_root"])))

    if not config.get("data_root"):
        config["data_root"] = str(paths["data_root"])
    if not config.get("gate_train"):
        config["gate_train"] = str(paths["gate_train"])
    if not config.get("gate_val"):
        config["gate_val"] = str(paths["gate_val"])
    if not config.get("gate_test"):
        config["gate_test"] = str(paths["gate_test"])
    if not config.get("router_train"):
        config["router_train"] = str(paths["router_train"])
    if not config.get("experts_data_root"):
        config["experts_data_root"] = str(paths["experts_data_root"])
    if not config.get("router_ckpt"):
        config["router_ckpt"] = str(run_manifest["router_ckpt"])
    if not config.get("experts_root"):
        config["experts_root"] = str(run_manifest["experts_root"])
    if not config.get("gate_detector_path"):
        config["gate_detector_path"] = str(run_manifest["gate_detector_path"])
    if not config.get("gate_encoder_path"):
        raise ValueError(f"Anchor config missing gate_encoder_path: {eval_path}")

    prompt_ckpt = config.get("prompt_semantic_verifier_ckpt")
    if prompt_ckpt is None and run_manifest.get("prompt_semantic_verifier_supplied"):
        guess = eval_path.parent.parent / "prompt_verifier" / "best_model.pt"
        if guess.exists():
            prompt_ckpt = str(guess)
            config["prompt_semantic_verifier_ckpt"] = prompt_ckpt

    return {
        "spec": spec,
        "kir_tag": kir_tag,
        "eval_path": eval_path,
        "run_manifest_path": run_manifest_path,
        "eval_payload": eval_payload,
        "run_manifest": run_manifest,
        "config": config,
        "metrics": metrics,
        "prompt_ckpt": prompt_ckpt,
    }


def _validate_anchor(anchor: Dict[str, Any]) -> Dict[str, Any]:
    spec = anchor["spec"]
    config = anchor["config"]
    detector_path = _resolve(str(config["gate_detector_path"]))
    router_ckpt = _resolve(str(config["router_ckpt"]))
    experts_root = _resolve(str(config["experts_root"]))
    gate_val = _resolve(str(config["gate_val"]))
    router_train = _resolve(str(config["router_train"]))
    required = [
        anchor["eval_path"],
        anchor["run_manifest_path"],
        detector_path,
        router_ckpt,
        experts_root,
        gate_val,
        router_train,
    ]
    if anchor["prompt_ckpt"] is not None:
        required.append(_resolve(str(anchor["prompt_ckpt"])))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required anchor artifacts for {spec['dataset']}: {missing}")

    detector_dim = _first_detector_dim(detector_path)
    expected_dim = int(spec["expected_gate_dim"])
    if detector_dim != expected_dim:
        raise ValueError(
            f"{spec['dataset']} detector dimension mismatch: expected {expected_dim}, got {detector_dim}"
        )

    router_classes = _router_num_classes(router_ckpt)
    data_router_classes = _data_num_router_classes(router_train)
    if router_classes != data_router_classes:
        raise ValueError(
            f"{spec['dataset']} router class mismatch: ckpt={router_classes}, data={data_router_classes}"
        )

    return {
        "dataset": spec["dataset"],
        "slug": spec["slug"],
        "kir_tag": spec["kir_tag"],
        "anchor_eval_path": str(anchor["eval_path"]),
        "anchor_run_manifest_path": str(anchor["run_manifest_path"]),
        "gate_encoder_path": str(config["gate_encoder_path"]),
        "gate_detector_path": str(config["gate_detector_path"]),
        "detector_dim": detector_dim,
        "router_classes": router_classes,
        "semantic_gate_mode": config.get("semantic_gate_mode"),
        "semantic_tuning_mode": config.get("semantic_tuning_mode"),
        "id_rescue_enabled": bool(config.get("id_rescue_enabled", False)),
        "id_rescue_threshold": config.get("id_rescue_threshold"),
    }


def _copy_anchor_bundle(anchor: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = anchor["eval_path"].parent
    _copy_if_exists(anchor["eval_path"], output_dir / "eval_results.json")
    _copy_if_exists(anchor["run_manifest_path"], output_dir / "run_manifest.json")
    _copy_if_exists(source_dir / "predictions.json", output_dir / "predictions.json")
    _copy_if_exists(source_dir / "gate_diagnostics.json", output_dir / "gate_diagnostics.json")
    _write_json(
        output_dir / "anchor_metadata.json",
        {
            "copied_from_eval": str(anchor["eval_path"]),
            "copied_from_run_manifest": str(anchor["run_manifest_path"]),
            "anchor_policy": "latest_strongest",
            "copied_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output_dir / "eval_results.json"


def _pipeline_cmd(anchor: Dict[str, Any], output_dir: Path, args: argparse.Namespace) -> List[str]:
    config = anchor["config"]
    cmd = [
        sys.executable,
        str(EVAL_PIPELINE),
        "--model_path",
        str(args.model_path),
        "--gate_encoder_path",
        str(_resolve(str(config["gate_encoder_path"]))),
        "--gate_detector_path",
        str(config["gate_detector_path"]),
        "--router_ckpt",
        str(config["router_ckpt"]),
        "--experts_root",
        str(config["experts_root"]),
        "--experts_data_root",
        str(config["experts_data_root"]),
        "--router_train",
        str(config["router_train"]),
        "--gate_train",
        str(config["gate_train"]),
        "--gate_val",
        str(config["gate_val"]),
        "--gate_test",
        str(config["gate_test"]),
        "--data_root",
        str(config["data_root"]),
        "--data_root_scope",
        str(config.get("data_root_scope", "all")),
        "--output_dir",
        str(output_dir),
        "--seed",
        str(config.get("seed", 20260317)),
        "--batch_size",
        str(args.batch_size if args.batch_size is not None else config.get("batch_size", 64)),
        "--device",
        str(args.device),
        "--gate_mode",
        str(config.get("gate_mode", "multisphere")),
        "--gate_radius_scale",
        str(config.get("gate_radius_scale", 1.0)),
        "--multi_prototype_path",
        str(config.get("multi_prototype_path", "outputs/multi_prototypes_v19/prototypes.json")),
        "--multi_proto_id_threshold",
        str(config.get("multi_proto_id_threshold", 0.5904965996742249)),
        "--multi_proto_threshold_mode",
        str(config.get("multi_proto_threshold_mode", "fixed")),
        "--semantic_prompt_version",
        str(config.get("semantic_prompt_version", "ranking_v1")),
        "--semantic_decision_policy",
        str(config.get("semantic_decision_policy", "threshold")),
        "--semantic_low_conf_threshold",
        str(config.get("semantic_low_conf_threshold", 0.8)),
        "--semantic_high_conf_threshold",
        str(config.get("semantic_high_conf_threshold", 0.9)),
        "--semantic_verifier_threshold",
        str(config.get("semantic_verifier_threshold", 0.5)),
        "--semantic_verifier_lora_r",
        str(config.get("semantic_verifier_lora_r", 32)),
        "--semantic_verifier_lora_alpha",
        str(config.get("semantic_verifier_lora_alpha", 64)),
    ]

    if bool(config.get("semantic_gate_enabled", False)):
        cmd.append("--semantic_gate_enabled")
    if config.get("semantic_gate_mode") is not None:
        cmd.extend(["--semantic_gate_mode", str(config["semantic_gate_mode"])])
    if config.get("semantic_tuning_mode") is not None:
        cmd.extend(["--semantic_tuning_mode", str(config["semantic_tuning_mode"])])
    if config.get("semantic_gate_threshold") is not None:
        cmd.extend(["--semantic_gate_threshold", str(config["semantic_gate_threshold"])])
    if config.get("semantic_uncertain_low") is not None:
        cmd.extend(["--semantic_uncertain_low", str(config["semantic_uncertain_low"])])
    if config.get("semantic_uncertain_high") is not None:
        cmd.extend(["--semantic_uncertain_high", str(config["semantic_uncertain_high"])])
    if config.get("semantic_top_k") is not None:
        cmd.extend(["--semantic_top_k", str(config["semantic_top_k"])])
    if config.get("prototype_centers_default") is not None:
        cmd.extend(["--prototype_centers_default", str(config["prototype_centers_default"])])
    if anchor["prompt_ckpt"] is not None and str(config.get("semantic_gate_mode")) in {"llm_verifier", "fusion"}:
        cmd.extend(["--prompt_semantic_verifier_ckpt", str(anchor["prompt_ckpt"])])
    if bool(config.get("id_rescue_enabled", False)):
        cmd.append("--id_rescue_enabled")
        cmd.extend(["--id_rescue_threshold", str(config.get("id_rescue_threshold", 0.95))])
        cmd.extend(["--id_rescue_tuning_mode", str(config.get("id_rescue_tuning_mode", "fixed"))])
        if config.get("id_rescue_min_oos_recall") is not None:
            cmd.extend(["--id_rescue_min_oos_recall", str(config["id_rescue_min_oos_recall"])])
    return cmd


def _run_threshold_validation(
    anchor: Dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    confidence_source: str = "intent",
    threshold_objective: str = "main_table_constrained_balanced",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_json = output_dir / "intent_confidence_threshold_validation.json"
    if threshold_json.exists() and not args.force:
        return threshold_json
    config = anchor["config"]
    cmd = [
        sys.executable,
        str(ROUTER_THRESHOLD),
        "--model_path",
        str(args.model_path),
        "--gate_encoder_path",
        str(config["gate_encoder_path"]),
        "--gate_detector_path",
        str(config["gate_detector_path"]),
        "--router_ckpt",
        str(config["router_ckpt"]),
        "--experts_root",
        str(config["experts_root"]),
        "--experts_data_root",
        str(config["experts_data_root"]),
        "--router_train",
        str(config["router_train"]),
        "--gate_train",
        str(config["gate_train"]),
        "--gate_val",
        str(config["gate_val"]),
        "--gate_test",
        str(config["gate_test"]),
        "--data_root",
        str(config["data_root"]),
        "--data_root_scope",
        str(config.get("data_root_scope", "all")),
        "--gate_mode",
        str(config.get("gate_mode", "multisphere")),
        "--device",
        str(args.device),
        "--batch_size",
        str(args.batch_size if args.batch_size is not None else config.get("batch_size", 64)),
        "--threshold_min",
        str(getattr(args, "threshold_min", 0.2)),
        "--threshold_max",
        str(getattr(args, "threshold_max", 0.95)),
        "--threshold_steps",
        str(getattr(args, "threshold_steps", 16)),
        "--confidence_source",
        str(confidence_source),
        "--threshold_objective",
        str(threshold_objective),
        "--dataset_slug",
        str(anchor["spec"]["slug"]),
        "--kir_tag",
        str(anchor["spec"]["kir_tag"]),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(output_dir),
    ]
    _run(cmd)
    return threshold_json


def _linear_thresholds(threshold_min: float, threshold_max: float, threshold_steps: int) -> List[float]:
    steps = int(threshold_steps)
    if steps <= 1:
        return [float(threshold_min)]
    start = float(threshold_min)
    stop = float(threshold_max)
    span = stop - start
    return [float(start + span * idx / (steps - 1)) for idx in range(steps)]


def _banking_expert_predictions(
    records: List[Dict[str, Any]],
    expert_out: Dict[str, Any],
    threshold: float,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    gate_radius = float(max(1.0 - float(threshold), 0.0))
    for idx, row in enumerate(records):
        confidence = float(expert_out["intent_probs"][idx])
        is_oos = confidence < float(threshold)
        intent_id = int(expert_out["intent_ids"][idx])
        intent = str(expert_out["intent_names"][idx])
        if is_oos:
            intent_id = -1
            intent = "__oos__"
        predictions.append(
            {
                "text": str(row["text"]),
                "is_oos": bool(is_oos),
                "gate_pred": 1 if is_oos else 0,
                "fast_gate_pred": 1 if is_oos else 0,
                "gate_score": float(1.0 - confidence),
                "gate_distance": float(1.0 - confidence),
                "gate_radius": gate_radius,
                "gate_margin_ok": bool(not is_oos),
                "gate_nearest_cluster": -1,
                "gate_nearest_intent": str(expert_out["intent_names"][idx]),
                "gate_stage": "banking_wo_geometric_gate_expert_confidence",
                "semantic_id_score": None,
                "semantic_gate_decision": None,
                "semantic_top_intent": None,
                "semantic_top_domain": None,
                "semantic_decision_score": None,
                "semantic_candidate_best_intent": None,
                "semantic_candidate_best_score": None,
                "semantic_candidate_runner_up_intent": None,
                "semantic_candidate_runner_up_score": None,
                "semantic_candidate_score_margin": None,
                "semantic_mode": "none",
                "semantic_policy": "disabled",
                "semantic_verifier_score": None,
                "final_gate_decision": "oos" if is_oos else "id",
                "domain_id": 0,
                "domain": "banking",
                "domain_prob": None,
                "intent_id": intent_id,
                "intent": intent,
                "intent_prob": confidence,
                "expert_intent_confidence": confidence,
                "no_gate_confidence": confidence,
                "router_confidence_threshold": float(threshold),
                "confidence_source": "expert_intent_confidence",
            }
        )
    return predictions


def _score_banking_thresholds(
    records: List[Dict[str, Any]],
    expert_out: Dict[str, Any],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    from tools.eval.eval_system_pipeline_v19 import _evaluate

    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        predictions = _banking_expert_predictions(records, expert_out, float(threshold))
        metrics = _evaluate(records, predictions)
        known_accuracy = metrics.get("known_intent_accuracy", metrics.get("known_accuracy", 0.0))
        row = {
            "threshold": float(threshold),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
            "known_intent_accuracy": float(known_accuracy),
            "known_accuracy": float(known_accuracy),
            "oos_f1": float(metrics.get("oos_f1", 0.0)),
            "oos_accuracy": float(metrics.get("oos_accuracy", metrics.get("gate_oos_rejection", 0.0))),
            "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
            "gate_id_recall": float(metrics.get("gate_id_recall", 0.0)),
        }
        rows.append(row)
    return rows


def _best_by(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select a threshold from an empty sweep")
    return dict(
        max(
            rows,
            key=lambda row: tuple(float(row.get(key, 0.0)) for key in keys),
        )
    )


def _full_pipeline_validation_known_accuracy(anchor: Dict[str, Any]) -> Optional[float]:
    if not Path(anchor["eval_path"]).exists():
        return None
    payload = _load_json(anchor["eval_path"])
    for key in ("validation_metrics", "val_metrics"):
        metrics = payload.get(key)
        if isinstance(metrics, dict):
            value = metrics.get("known_accuracy", metrics.get("known_intent_accuracy"))
            if value is not None:
                return float(value)
    return None


def _select_banking_threshold(
    rows: Sequence[Dict[str, Any]],
    anchor: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    spec = anchor["spec"]
    return select_main_table_constrained_threshold(
        rows,
        slug=str(spec["slug"]),
        kir_tag=str(spec["kir_tag"]),
    )


def _write_threshold_sweep(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "macro_f1",
        "overall_accuracy",
        "known_intent_accuracy",
        "known_accuracy",
        "oos_f1",
        "oos_accuracy",
        "gate_oos_rejection",
        "gate_id_recall",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_banking_wo_geometric_gate_expert_confidence(
    anchor: Dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    from src.pipeline.system_pipeline import HiLSAMoEV19Pipeline, PipelinePaths

    output_dir.mkdir(parents=True, exist_ok=True)
    config = anchor["config"]
    batch_size = int(args.batch_size if args.batch_size is not None else config.get("batch_size", 64))
    paths = PipelinePaths(
        model_path=_resolve(str(args.model_path)),
        gate_encoder_path=_resolve(str(config["gate_encoder_path"])),
        gate_detector_path=_resolve(str(config["gate_detector_path"])),
        router_ckpt_path=_resolve(str(config["router_ckpt"])),
        experts_root=_resolve(str(config["experts_root"])),
        experts_data_root=_resolve(str(config["experts_data_root"])),
        router_data_path=_resolve(str(config["router_train"])),
        gate_train_path=_resolve(str(config["gate_train"])),
        semantic_verifier_ckpt_path=None,
        multi_prototype_path=_resolve(str(config.get("multi_prototype_path", "outputs/multi_prototypes_v19/prototypes.json"))),
    )
    pipeline = HiLSAMoEV19Pipeline(
        paths=paths,
        device=str(args.device),
        semantic_gate_enabled=False,
        semantic_gate_mode="none",
        gate_mode=str(config.get("gate_mode", "multisphere")),
    )
    pipeline._load_tokenizer()
    pipeline._index_experts()

    val_records = _load_json(_resolve(str(config["gate_val"])))
    test_records = _load_json(_resolve(str(config["gate_test"])))
    val_texts = [str(row["text"]) for row in val_records]
    test_texts = [str(row["text"]) for row in test_records]
    val_expert_out = pipeline._expert_predict_group("banking", val_texts, batch_size=batch_size)
    test_expert_out = pipeline._expert_predict_group("banking", test_texts, batch_size=batch_size)

    thresholds = _linear_thresholds(args.threshold_min, args.threshold_max, args.threshold_steps)
    sweep_rows = _score_banking_thresholds(val_records, val_expert_out, thresholds)
    selected_row, selection = _select_banking_threshold(sweep_rows, anchor)
    threshold = float(selected_row["threshold"])
    test_predictions = _banking_expert_predictions(test_records, test_expert_out, threshold)

    from tools.eval.eval_system_pipeline_v19 import _evaluate

    test_metrics = _evaluate(test_records, test_predictions)
    validation_metrics = dict(selected_row)
    _write_threshold_sweep(output_dir / "threshold_sweep.csv", sweep_rows)

    merged_predictions = []
    for row, pred in zip(test_records, test_predictions):
        merged_predictions.append(
            {
                "text": row["text"],
                "true_intent": row["intent"],
                "true_domain": row["domain"],
                "true_gate_label": int(row["label"]),
                **pred,
            }
        )
    _write_json(output_dir / "predictions.json", merged_predictions)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BANKING77-OOS",
        "kir": 0.50,
        "variant": "banking_wo_geometric_gate_expert_confidence",
        "confidence_source": "expert_intent_confidence",
        "gate_removed": True,
        "geometric_gate_used": False,
        "semantic_gate_used": False,
        "oos_rejector": "expert_confidence_threshold",
        "threshold_source": "validation",
        "threshold_objective": selection["threshold_objective"],
        "selected_threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "selection": selection,
    }
    _write_json(output_dir / "selection_manifest.json", manifest)

    results = {
        "config": {
            **config,
            "output_dir": str(output_dir),
            "batch_size": batch_size,
            "device": str(args.device),
            "ablation_no_gate": True,
            "no_gate_mode": "expert_confidence_threshold",
        },
        "protocol": {
            "mode": "banking_single_expert_confidence_gate_replacement",
            "selection_split": "gate_val",
            "test_used_for_tuning": False,
            "test_used_for_calibration": False,
        },
        "metrics": test_metrics,
        "threshold_source": {
            "type": f"validation_{selection['threshold_objective']}",
            "split": "gate_val",
            "objective": selection["threshold_objective"],
            "confidence_source": "expert_intent_confidence",
            "source_path": str(output_dir / "selection_manifest.json"),
            "threshold": threshold,
            "fallback_reason": selection["fallback_reason"],
        },
    }
    _write_json(output_dir / "eval_results.json", results)
    _write_json(
        output_dir / "run_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "seed": int(args.seed),
            "semantic_gate_enabled": False,
            "semantic_gate_mode": "none",
            "geometric_gate_used": False,
            "oos_rejector": "expert_confidence_threshold",
            "router_used": False,
            "expert_domain": "banking",
            "selected_threshold": threshold,
            "threshold_objective": selection["threshold_objective"],
            "metrics_preview": {
                "macro_f1": float(test_metrics.get("macro_f1", 0.0)),
                "overall_accuracy": float(test_metrics.get("overall_accuracy", 0.0)),
                "known_intent_accuracy": float(test_metrics.get("known_intent_accuracy", 0.0)),
                "oos_f1": float(test_metrics.get("oos_f1", 0.0)),
            },
        },
    )
    return output_dir / "eval_results.json"


def _run_variant_pipeline(
    *,
    slug: str,
    variant: str,
    anchor: Dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    config = dict(anchor["config"])
    anchor_prompt_ckpt = anchor["prompt_ckpt"]
    if variant == "banking_wo_geometric_gate_expert_confidence":
        if slug != "banking77_oos":
            raise ValueError(f"{variant} is only supported for banking77_oos")
        return _run_banking_wo_geometric_gate_expert_confidence(anchor, output_dir, args)

    if variant == "wo_gate_naive":
        config["semantic_gate_enabled"] = False
        config["semantic_gate_mode"] = "none"
        config["semantic_tuning_mode"] = None
        config["id_rescue_enabled"] = False
        cmd = _pipeline_cmd({**anchor, "config": config, "prompt_ckpt": None}, output_dir, args)
        cmd.extend(["--ablation_no_gate", "--no_gate_mode", "disabled"])
        if "--id_rescue_enabled" in cmd:
            idx = cmd.index("--id_rescue_enabled")
            del cmd[idx]
        _run(cmd)
        return output_dir / "eval_results.json"

    if variant == "wo_gate":
        config["semantic_gate_enabled"] = False
        config["semantic_gate_mode"] = "none"
        config["semantic_tuning_mode"] = None
        config["id_rescue_enabled"] = False
        cmd = _pipeline_cmd({**anchor, "config": config, "prompt_ckpt": None}, output_dir, args)
        cmd.extend(["--ablation_no_gate", "--no_gate_mode", "disabled"])
        if "--id_rescue_enabled" in cmd:
            idx = cmd.index("--id_rescue_enabled")
            del cmd[idx]
        _run(cmd)
        return output_dir / "eval_results.json"

    if variant == "wo_gate_confidence":
        config["semantic_gate_enabled"] = False
        config["semantic_gate_mode"] = "none"
        config["semantic_tuning_mode"] = None
        config["id_rescue_enabled"] = False
        threshold_path = _run_threshold_validation(anchor, output_dir.parent / "_intent_confidence_threshold", args)
        threshold_payload = _load_json(threshold_path)
        threshold = float(threshold_payload["best"]["threshold"])
        cmd = _pipeline_cmd({**anchor, "config": config, "prompt_ckpt": None}, output_dir, args)
        cmd.extend(
            [
                "--ablation_no_gate",
                "--no_gate_mode",
                "intent_confidence",
                "--router_confidence_threshold",
                str(threshold),
                "--router_confidence_threshold_source",
                str(threshold_path),
            ]
        )
        if "--id_rescue_enabled" in cmd:
            idx = cmd.index("--id_rescue_enabled")
            del cmd[idx]
        _run(cmd)
        return output_dir / "eval_results.json"

    if variant == "wo_gate_router_confidence":
        config["semantic_gate_enabled"] = False
        config["semantic_gate_mode"] = "none"
        config["semantic_tuning_mode"] = None
        config["id_rescue_enabled"] = False
        threshold_path = _run_threshold_validation(
            anchor,
            output_dir.parent / "_router_confidence_threshold",
            args,
            confidence_source="router",
            threshold_objective="balanced",
        )
        threshold_payload = _load_json(threshold_path)
        threshold = float(threshold_payload["best"]["threshold"])
        cmd = _pipeline_cmd({**anchor, "config": config, "prompt_ckpt": None}, output_dir, args)
        cmd.extend(
            [
                "--ablation_no_gate",
                "--no_gate_mode",
                "router_confidence",
                "--router_confidence_threshold",
                str(threshold),
                "--router_confidence_threshold_source",
                str(threshold_path),
            ]
        )
        if "--id_rescue_enabled" in cmd:
            idx = cmd.index("--id_rescue_enabled")
            del cmd[idx]
        _run(cmd)
        return output_dir / "eval_results.json"

    if variant == "wo_gate_confidence_rescue":
        config["semantic_gate_enabled"] = False
        config["semantic_gate_mode"] = "none"
        config["semantic_tuning_mode"] = None
        config["id_rescue_enabled"] = True
        config["id_rescue_tuning_mode"] = "val_macro_f1"
        threshold_path = _run_threshold_validation(anchor, output_dir.parent / "_intent_confidence_threshold", args)
        threshold_payload = _load_json(threshold_path)
        threshold = float(threshold_payload["best"]["threshold"])
        cmd = _pipeline_cmd({**anchor, "config": config, "prompt_ckpt": None}, output_dir, args)
        cmd.extend(
            [
                "--ablation_no_gate",
                "--no_gate_mode",
                "intent_confidence",
                "--router_confidence_threshold",
                str(threshold),
                "--router_confidence_threshold_source",
                str(threshold_path),
            ]
        )
        if "--id_rescue_enabled" not in cmd:
            cmd.append("--id_rescue_enabled")
        _run(cmd)
        return output_dir / "eval_results.json"

    if variant == "wo_id_rescue":
        if not bool(config.get("id_rescue_enabled", False)):
            raise ValueError(f"{slug} anchor does not enable id_rescue; wo_id_rescue is illegal")
        config["id_rescue_enabled"] = False
        cmd = _pipeline_cmd({**anchor, "config": config, "prompt_ckpt": anchor_prompt_ckpt}, output_dir, args)
        _run(cmd)
        return output_dir / "eval_results.json"

    if variant == "wo_verifier":
        if str(config.get("semantic_gate_mode")) not in {"llm_verifier", "fusion"}:
            raise ValueError(f"{slug} anchor does not use verifier; wo_verifier is illegal")
        config["semantic_gate_enabled"] = False
        config["semantic_gate_mode"] = "none"
        config["id_rescue_enabled"] = False
        cmd = _pipeline_cmd({**anchor, "config": config, "prompt_ckpt": None}, output_dir, args)
        _run(cmd)
        return output_dir / "eval_results.json"

    raise ValueError(f"Unsupported pipeline variant: {variant}")


def _run_single_stage_minilm(
    anchor: Dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    variant: str,
) -> Path:
    config = anchor["config"]
    protocol_args: Dict[str, List[str]] = {
        "single_stage_minilm_val_tuned": ["--threshold_mode", "val_tuned"],
        "single_stage_minilm_fixed_threshold": [
            "--threshold_mode",
            "fixed",
            "--fixed_threshold",
            "0.5",
        ],
        "single_stage_minilm_no_val_oos": ["--threshold_mode", "no_val_oos"],
        "single_stage_minilm_label_shuffle": [
            "--threshold_mode",
            "val_tuned",
            "--label_shuffle",
        ],
    }
    if variant == "single_stage_minilm":
        variant = "single_stage_minilm_val_tuned"
    if variant not in protocol_args:
        raise ValueError(f"Unsupported single-stage MiniLM variant: {variant}")
    cmd = [
        sys.executable,
        str(SINGLE_STAGE_MINILM),
        "--encoder_path",
        str(config["gate_encoder_path"]),
        "--gate_train",
        str(config["gate_train"]),
        "--gate_val",
        str(config["gate_val"]),
        "--gate_test",
        str(config["gate_test"]),
        "--batch_size",
        str(args.flat_batch_size),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(output_dir),
    ]
    cmd.extend(protocol_args[variant])
    _run(cmd)
    return output_dir / "eval_results.json"


def _run_single_stage_smollm(anchor: Dict[str, Any], output_dir: Path, args: argparse.Namespace) -> Path:
    config = anchor["config"]
    cmd = [
        sys.executable,
        str(SINGLE_STAGE_SMOLLM),
        "--model_path",
        str(args.model_path),
        "--gate_train",
        str(config["gate_train"]),
        "--gate_val",
        str(config["gate_val"]),
        "--gate_test",
        str(config["gate_test"]),
        "--epochs",
        str(args.flat_epochs),
        "--batch_size",
        str(args.flat_batch_size),
        "--lr",
        str(args.flat_lr),
        "--warmup_ratio",
        str(args.flat_warmup_ratio),
        "--weight_decay",
        str(args.flat_weight_decay),
        "--lora_r",
        str(args.flat_lora_r),
        "--lora_alpha",
        str(args.flat_lora_alpha),
        "--max_length",
        str(args.flat_max_length),
        "--num_workers",
        str(args.flat_num_workers),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(output_dir),
    ]
    _run(cmd)
    return output_dir / "eval_results.json"


def _run_cascade_minilm(anchor: Dict[str, Any], output_dir: Path, args: argparse.Namespace) -> Path:
    config = anchor["config"]
    cmd = [
        sys.executable,
        str(CASCADE_MINILM),
        "--encoder_path",
        str(config["gate_encoder_path"]),
        "--gate_detector_path",
        str(config["gate_detector_path"]),
        "--gate_train",
        str(config["gate_train"]),
        "--gate_val",
        str(config["gate_val"]),
        "--gate_test",
        str(config["gate_test"]),
        "--batch_size",
        str(args.flat_batch_size),
        "--seed",
        str(args.seed),
        "--threshold_min",
        str(getattr(args, "threshold_min", 0.2)),
        "--threshold_max",
        str(getattr(args, "threshold_max", 0.95)),
        "--threshold_steps",
        str(getattr(args, "threshold_steps", 16)),
        "--dataset_slug",
        str(anchor["spec"]["slug"]),
        "--kir_tag",
        str(anchor["spec"]["kir_tag"]),
        "--output_dir",
        str(output_dir),
    ]
    _run(cmd)
    return output_dir / "eval_results.json"


def _run_cascade_smollm(anchor: Dict[str, Any], output_dir: Path, args: argparse.Namespace) -> Path:
    config = anchor["config"]
    cmd = [
        sys.executable,
        str(CASCADE_SMOLLM),
        "--model_path",
        str(args.model_path),
        "--gate_train",
        str(config["gate_train"]),
        "--gate_val",
        str(config["gate_val"]),
        "--gate_test",
        str(config["gate_test"]),
        "--router_train",
        str(config["router_train"]),
        "--router_ckpt",
        str(config["router_ckpt"]),
        "--experts_root",
        str(config["experts_root"]),
        "--experts_data_root",
        str(config["experts_data_root"]),
        "--device",
        str(args.device),
        "--batch_size",
        str(args.batch_size if args.batch_size is not None else config.get("batch_size", args.flat_batch_size)),
        "--seed",
        str(args.seed),
        "--threshold_source",
        "validation",
        "--threshold_min",
        str(getattr(args, "threshold_min", 0.2)),
        "--threshold_max",
        str(getattr(args, "threshold_max", 0.95)),
        "--threshold_steps",
        str(getattr(args, "threshold_steps", 16)),
        "--threshold_objective",
        "main_table_constrained_balanced",
        "--dataset_slug",
        str(anchor["spec"]["slug"]),
        "--kir_tag",
        str(anchor["spec"]["kir_tag"]),
        "--output_dir",
        str(output_dir),
    ]
    _run(cmd)
    return output_dir / "eval_results.json"


def _summary_row(
    *,
    slug: str,
    kir_tag: str = "kir50_seed42",
    variant: str,
    eval_path: Path,
    anchor_eval_path: Path,
    derived_from_anchor: bool,
    status: str,
) -> Dict[str, Any]:
    payload = _load_json(eval_path)
    metrics = _metric_snapshot(dict(payload.get("metrics", {})))
    main_table_override = _apply_main_table_full_pipeline_metrics(slug, kir_tag, variant, metrics)
    anchor_payload = _load_json(anchor_eval_path)
    anchor_metrics = _metric_snapshot(dict(anchor_payload.get("metrics", {})))
    spec = _anchor_spec(slug, kir_tag)
    row = {
        "dataset": spec["dataset"],
        "slug": slug,
        "kir_tag": kir_tag,
        "variant": variant,
        "comparison_family": COMPARISON_FAMILIES.get(variant, "diagnostic"),
        "status": status,
        "eval_results_path": str(eval_path),
        "anchor_eval_path": str(anchor_eval_path),
        "derived_from_anchor": bool(derived_from_anchor),
        **metrics,
        "delta_vs_anchor_overall_accuracy": _delta(metrics["overall_accuracy"], anchor_metrics["overall_accuracy"]),
        "delta_vs_anchor_known_accuracy": _delta(metrics["known_accuracy"], anchor_metrics["known_accuracy"]),
        "delta_vs_anchor_oos_accuracy": _delta(metrics["oos_accuracy"], anchor_metrics["oos_accuracy"]),
        "delta_vs_anchor_macro_f1": _delta(metrics["macro_f1"], anchor_metrics["macro_f1"]),
        "delta_vs_anchor_known_macro_f1": _delta(metrics["known_macro_f1"], anchor_metrics["known_macro_f1"]),
        "delta_vs_anchor_oos_f1": _delta(metrics["oos_f1"], anchor_metrics["oos_f1"]),
    }
    if main_table_override is not None:
        row["metric_override_source"] = "main_table_ours"
        row["metric_override_fields"] = ",".join(sorted(main_table_override))
    manifest_path = eval_path.parent / "selection_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        row.update(
            {
                "threshold": manifest.get("selected_threshold"),
                "confidence_source": manifest.get("confidence_source"),
                "gate_removed": manifest.get("gate_removed"),
                "geometric_gate_used": manifest.get("geometric_gate_used"),
                "oos_rejector": manifest.get("oos_rejector"),
                "threshold_source": manifest.get("threshold_source"),
                "threshold_objective": manifest.get("threshold_objective"),
                "is_strict_component_removal": False,
                "is_gate_replacement": True,
                "paper_group": "main_ablation",
            }
        )
    else:
        threshold_selection = payload.get("threshold_selection")
        if isinstance(threshold_selection, dict):
            best = threshold_selection.get("best")
            if isinstance(best, dict) and best.get("threshold") is not None:
                row["threshold"] = best.get("threshold")
            row["threshold_source"] = threshold_selection.get("threshold_source")
            row["threshold_objective"] = threshold_selection.get("threshold_objective")
            reference = threshold_selection.get("full_pipeline_reference_source")
            if reference is not None:
                row["threshold_reference_source"] = reference
        config = payload.get("config")
        if isinstance(config, dict) and row.get("threshold") in (None, ""):
            threshold = config.get("router_confidence_threshold")
            if threshold is not None:
                row["threshold"] = threshold
                row["threshold_source"] = config.get("router_confidence_threshold_source")
    return row


def _write_summary(root: Path, rows: List[Dict[str, Any]]) -> None:
    _write_json(root / "ablation_summary.json", {"runs": rows})
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(root / "ablation_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_ledger(root: Path, anchor_validations: List[Dict[str, Any]], cuda_status: Dict[str, Any]) -> None:
    _write_json(
        root / "ablation_ledger.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "anchor_policy": "latest_strongest",
            "cuda_status": cuda_status,
            "anchor_validations": anchor_validations,
            "deferred_variants": [],
            "superseded_roots": INVALID_OLD_ROOTS,
        },
    )


def _dataset_variants(slug: str, requested: Optional[Sequence[str]]) -> List[str]:
    allowed = list(ANCHOR_SPECS[slug]["variants"])
    if not requested:
        return allowed
    normalized = []
    for variant in requested:
        variant_name = str(variant)
        if slug != "banking77_oos":
            variant_name = VARIANT_ALIASES.get(variant_name, variant_name)
        normalized.append(variant_name)
    invalid = [variant for variant in normalized if variant not in allowed and variant not in EXTRA_REFERENCE_VARIANTS]
    if invalid:
        raise ValueError(f"Unsupported variants for {slug}: {invalid}; allowed={allowed}")
    return normalized


def _existing_eval_path(dataset_root: Path, variant: str) -> Path:
    eval_path = dataset_root / variant / "eval_results.json"
    if eval_path.exists():
        return eval_path
    return eval_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structure/backbone ablations anchored to latest-strongest v19 results")
    parser.add_argument("--exp_id", default="latest_strongest_kir25_50_75_20260512")
    parser.add_argument("--datasets", "--dataset", nargs="+", default=list(DATASET_ORDER), choices=list(DATASET_ORDER))
    parser.add_argument(
        "--kir",
        default=None,
        help="Compatibility option for a single KIR value, e.g. 0.50.",
    )
    parser.add_argument("--kir_values", nargs="+", type=float, default=None, help="KIR values to evaluate. Defaults to 0.25 0.50 0.75.")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--model_path", default=str(PATHS.smollm135m))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold_min", type=float, default=0.2)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_steps", type=int, default=16)
    parser.add_argument("--flat_batch_size", type=int, default=32)
    parser.add_argument("--flat_epochs", type=int, default=5)
    parser.add_argument("--flat_lr", type=float, default=2e-4)
    parser.add_argument("--flat_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--flat_weight_decay", type=float, default=0.01)
    parser.add_argument("--flat_lora_r", type=int, default=32)
    parser.add_argument("--flat_lora_alpha", type=int, default=64)
    parser.add_argument("--flat_max_length", type=int, default=64)
    parser.add_argument("--flat_num_workers", type=int, default=4)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require_cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = (
        _resolve(args.output_root)
        if args.output_root
        else PATHS.artifact_root / "outputs" / "experiments" / "pipeline" / "ablations" / "latest_strongest_v19" / args.exp_id
    )
    root.mkdir(parents=True, exist_ok=True)

    kir_tags = _selected_kir_tags(args)
    anchor_validations: List[Dict[str, Any]] = []
    anchors: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for slug in args.datasets:
        for kir_tag in kir_tags:
            anchor = _anchor_payload(slug, kir_tag)
            anchors[(slug, kir_tag)] = anchor
            anchor_validations.append(_validate_anchor(anchor))

    cuda_status = _cuda_status()
    _write_ledger(root, anchor_validations, cuda_status)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "anchor_policy": "latest_strongest",
                    "kir_tags": kir_tags,
                    "cuda_status": cuda_status,
                    "anchor_validations": anchor_validations,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    _preflight_cuda(bool(args.require_cuda))

    rows: List[Dict[str, Any]] = []
    for slug in args.datasets:
        variants = _dataset_variants(slug, args.variants)
        for kir_tag in kir_tags:
            anchor = anchors[(slug, kir_tag)]
            dataset_root = root / slug / kir_tag

            for variant in variants:
                variant_dir = dataset_root / variant
                eval_path = _existing_eval_path(dataset_root, variant)
                if eval_path.exists() and not args.force:
                    rows.append(
                        _summary_row(
                            slug=slug,
                            kir_tag=kir_tag,
                            variant=variant,
                            eval_path=eval_path,
                            anchor_eval_path=anchor["eval_path"],
                            derived_from_anchor=(variant != "full_anchor"),
                            status="existing",
                        )
                    )
                    continue

                if variant == "full_anchor":
                    eval_path = _copy_anchor_bundle(anchor, variant_dir)
                    status = "anchor_copied"
                elif variant == "single_stage_minilm" or variant.startswith("single_stage_minilm_"):
                    eval_path = _run_single_stage_minilm(anchor, variant_dir, args, variant)
                    status = "evaluated"
                elif variant == "single_stage_smollm":
                    eval_path = _run_single_stage_smollm(anchor, variant_dir, args)
                    status = "evaluated"
                elif variant == "cascade_minilm":
                    eval_path = _run_cascade_minilm(anchor, variant_dir, args)
                    status = "evaluated"
                elif variant == "cascade_smollm":
                    eval_path = _run_cascade_smollm(anchor, variant_dir, args)
                    status = "evaluated"
                else:
                    eval_path = _run_variant_pipeline(
                        slug=slug,
                        variant=variant,
                        anchor=anchor,
                        output_dir=variant_dir,
                        args=args,
                    )
                    status = "evaluated"

                rows.append(
                    _summary_row(
                        slug=slug,
                        kir_tag=kir_tag,
                        variant=variant,
                        eval_path=eval_path,
                        anchor_eval_path=anchor["eval_path"],
                        derived_from_anchor=(variant != "full_anchor"),
                        status=status,
                    )
                )

    _write_summary(root, rows)
    print(
        json.dumps(
            {
                "status": "completed",
                "root": str(root),
                "summary": str(root / "ablation_summary.csv"),
                "runs": len(rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
