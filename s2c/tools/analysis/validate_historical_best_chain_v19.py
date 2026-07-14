#!/usr/bin/env python3
"""Validate and document the historical-best CLINC150@KIR50 mother chain.

This script is intentionally read-only with respect to model/data artifacts.
It inspects the existing repository state, compares the archived reference
result with the frozen reproduction result, and emits a machine-readable
report plus a human-readable markdown summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis.historical_best_pipeline_v19 import HISTORICAL_BEST_PIPELINE
REFERENCE_CONFIG_PATH = PROJECT_ROOT / "configs/v19/clinc150_historical_best_reference.json"
REFERENCE_EVAL_PATH = PROJECT_ROOT / (
    "outputs/experiments/archive/sweeps/2026-03-23/pipeline_phase3_proto_eval/"
    "pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json"
)
FROZEN_OUTPUT_DIR = PROJECT_ROOT / (
    "outputs/experiments/pipeline/frozen_prototype_gate/"
    "prototype_gate_frozen_2026-04-09/prototype_gate_pipeline_frozen"
)
FROZEN_EVAL_PATH = FROZEN_OUTPUT_DIR / "eval_results.json"
FROZEN_PREDICTIONS_PATH = FROZEN_OUTPUT_DIR / "predictions.json"
FROZEN_RUN_MANIFEST_PATH = FROZEN_OUTPUT_DIR / "run_manifest.json"
FROZEN_IDENTITY_PATH = FROZEN_OUTPUT_DIR / "prototype_baseline_identity.json"

DATA_ROOT = PROJECT_ROOT / "data/multidataset/v19/clinc150/kir50_seed42"
LEGACY_DATA_ROOT = PROJECT_ROOT / "data/v19"

CALL_CHAIN: List[Dict[str, Any]] = [
    {
        "stage": "data_rebuild",
        "owner": "scripts/data/active/rebuild_multi_dataset_v19.py",
        "role": "Create the CLINC150@KIR50 split and emit MANIFEST/AUDIT + gate/router/experts json files.",
        "fixed_vs_variable": "variable(dataset, kir, seed)",
        "inputs": [str(REFERENCE_CONFIG_PATH.relative_to(PROJECT_ROOT))],
        "outputs": [
            "data/multidataset/v19/clinc150/kir50_seed42/MANIFEST.json",
            "data/multidataset/v19/clinc150/kir50_seed42/AUDIT.json",
            "data/multidataset/v19/clinc150/kir50_seed42/gate/train.json",
            "data/multidataset/v19/clinc150/kir50_seed42/router/train.json",
            "data/multidataset/v19/clinc150/kir50_seed42/experts/*/*.json",
        ],
    },
    {
        "stage": "gate_train",
        "owner": "tools/gate/train_multisphere_corrected.py",
        "role": "Train the historical Gate detector that performs OOS rejection.",
        "fixed_vs_variable": "fixed(profile); variable(data only)",
        "inputs": ["data/multidataset/v19/clinc150/kir50_seed42/gate/*.json"],
        "outputs": [
            "outputs/experiments/archive/sweeps/2026-03-23/gate_l2_mix2_train/gate_l2_mix2_true_lambda_1p6/detector.json",
        ],
    },
    {
        "stage": "router_train",
        "owner": "tools/train/train_router_v19.py",
        "role": "Train the closed-set router used after gate acceptance.",
        "fixed_vs_variable": "fixed(profile); variable(data only)",
        "inputs": ["data/multidataset/v19/clinc150/kir50_seed42/router/*.json"],
        "outputs": [
            "outputs/experiments/components/router/router_v19/best_model.pt",
        ],
    },
    {
        "stage": "expert_train",
        "owner": "tools/train/train_all_experts_v19.py -> tools/train/train_expert_v19.py",
        "role": "Train one expert per domain for the known-intent branch.",
        "fixed_vs_variable": "fixed(profile); variable(data only)",
        "inputs": ["data/multidataset/v19/clinc150/kir50_seed42/experts/*/*.json"],
        "outputs": [
            "outputs/experiments/components/experts/experts_v19/*/best_model.pt",
        ],
    },
    {
        "stage": "semantic_verifier_train",
        "owner": "tools/train/train_semantic_verifier_v19.py",
        "role": "Train the semantic reranker/verifier used by the historical-best gate path.",
        "fixed_vs_variable": "fixed(profile); variable(data only)",
        "inputs": [
            "data/multidataset/v19/clinc150/kir50_seed42/gate/*.json",
            "outputs/experiments/components/router/router_v19/best_model.pt",
        ],
        "outputs": [
            "outputs/experiments/components/semantic/semantic_verifier_v19_u098_105/best_model.pt",
        ],
    },
    {
        "stage": "pipeline_assembly",
        "owner": "src/pipeline/system_pipeline.py",
        "role": "Assemble gate + router + experts into one inference pipeline.",
        "fixed_vs_variable": "fixed(profile); no data mutation",
        "inputs": [
            "outputs/experiments/archive/sweeps/2026-03-23/gate_l2_mix2_train/gate_l2_mix2_true_lambda_1p6/detector.json",
            "outputs/experiments/components/router/router_v19/best_model.pt",
            "outputs/experiments/components/experts/experts_v19/*/best_model.pt",
            "outputs/experiments/components/semantic/semantic_verifier_v19_u098_105/best_model.pt",
        ],
        "outputs": ["in-memory pipeline object"],
    },
    {
        "stage": "frozen_baseline_wrapper",
        "owner": "tools/analysis/run_prototype_gate_frozen_baseline_v19.py",
        "role": "Resolve frozen paths and launch the archived historical-best evaluation.",
        "fixed_vs_variable": "fixed(profile); runtime seed/device explicit",
        "inputs": [
            str(FROZEN_IDENTITY_PATH.relative_to(PROJECT_ROOT)),
            str(FROZEN_RUN_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        ],
        "outputs": [
            "outputs/experiments/pipeline/frozen_prototype_gate/prototype_gate_frozen_2026-04-09/prototype_gate_pipeline_frozen/eval_results.json",
            "outputs/experiments/pipeline/frozen_prototype_gate/prototype_gate_frozen_2026-04-09/prototype_gate_pipeline_frozen/predictions.json",
        ],
    },
    {
        "stage": "evaluation",
        "owner": "tools/eval/eval_system_pipeline_v19.py",
        "role": "Run explicit, no-search, no-tuning end-to-end evaluation.",
        "fixed_vs_variable": "fixed(profile); explicit config only",
        "inputs": [
            str(LEGACY_DATA_ROOT.relative_to(PROJECT_ROOT)),
            str(DATA_ROOT.relative_to(PROJECT_ROOT)),
        ],
        "outputs": [
            "outputs/experiments/archive/sweeps/2026-03-23/pipeline_phase3_proto_eval/pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json",
        ],
    },
]

SUPPORTING_CODE: List[Dict[str, str]] = [
    {
        "path": "tools/analysis/historical_best_pipeline_v19.py",
        "role": "Canonical profile object; holds the stable historical-best parameters.",
    },
    {
        "path": "tools/analysis/prototype_path_utils.py",
        "role": "Resolves prototype payloads for the frozen historical run.",
    },
    {
        "path": "tools/analysis/component_path_utils.py",
        "role": "Resolves frozen router and experts paths.",
    },
    {
        "path": "src/gate/multi_sphere_oos_detector.py",
        "role": "Gate detector implementation used by the historical-best path.",
    },
    {
        "path": "src/gate/intent_prototype_matcher.py",
        "role": "Prototype similarity scorer used by the semantic gate.",
    },
    {
        "path": "src/gate/llm_semantic_verifier.py",
        "role": "Semantic reranking/verifier module.",
    },
    {
        "path": "src/gate/multi_prototype_gate.py",
        "role": "Multi-prototype gate implementation.",
    },
    {
        "path": "src/models/architecture.py",
        "role": "Shared SmolLM router/expert architecture wrapper.",
    },
    {
        "path": "src/router/router_model.py",
        "role": "Router model wrapper used during verifier training and inference.",
    },
    {
        "path": "archive/reorg_2026-04-13/repro/historical_best_clinc150_kir50_py/pipeline_blueprint.py",
        "role": "Reverse-engineered pure-Python command blueprint for the historical-best chain.",
    },
    {
        "path": "archive/reorg_2026-04-13/repro/historical_best_clinc150_kir50_py/repro_entry.py",
        "role": "CLI entry to run the frozen historical reproduction or the CLINC150@KIR50 benchmark.",
    },
    {
        "path": "archive/reorg_2026-04-13/repro/file_index.py",
        "role": "Code index that records the historical-best CLINC150@KIR50 bundle.",
    },
]

PARAMETER_CHAIN: List[Dict[str, str]] = [
    {
        "dimension": "dataset",
        "value": "CLINC150",
        "classification": "fixed for historical-best reference; variable when extending to other datasets",
        "evidence": "configs/v19/clinc150_historical_best_reference.json",
    },
    {
        "dimension": "kir",
        "value": "0.50",
        "classification": "variable only at data rebuild time; no downstream stage may change it",
        "evidence": "data/multidataset/v19/clinc150/kir50_seed42/MANIFEST.json",
    },
    {
        "dimension": "dataset_seed",
        "value": "42",
        "classification": "fixed for the rebuilt CLINC150@KIR50 protocol",
        "evidence": "data/multidataset/v19/clinc150/kir50_seed42/MANIFEST.json",
    },
    {
        "dimension": "runtime_seed",
        "value": "20260317",
        "classification": "fixed for the frozen historical reproduction run",
        "evidence": "outputs/experiments/pipeline/frozen_prototype_gate/prototype_gate_frozen_2026-04-09/prototype_gate_pipeline_frozen/run_manifest.json",
    },
    {
        "dimension": "gate_profile",
        "value": "historical_best",
        "classification": "fixed profile constant",
        "evidence": "tools/analysis/historical_best_pipeline_v19.py",
    },
    {
        "dimension": "semantic_gate_mode",
        "value": "prototype",
        "classification": "fixed profile constant",
        "evidence": "tools/analysis/historical_best_pipeline_v19.py",
    },
    {
        "dimension": "semantic_gate_threshold",
        "value": "0.85",
        "classification": "fixed profile constant",
        "evidence": "tools/analysis/historical_best_pipeline_v19.py",
    },
    {
        "dimension": "multi_proto_id_threshold",
        "value": "0.5904965996742249",
        "classification": "fixed profile constant / frozen eval threshold",
        "evidence": "configs/v19/clinc150_historical_best_reference.json",
    },
    {
        "dimension": "router_lora",
        "value": "r=32, alpha=64",
        "classification": "fixed model configuration",
        "evidence": "configs/v19/clinc150_historical_best_reference.json",
    },
    {
        "dimension": "expert_lora",
        "value": "r=16, alpha=32",
        "classification": "fixed model configuration",
        "evidence": "configs/v19/clinc150_historical_best_reference.json",
    },
]


@dataclass(frozen=True)
class ValidationOutcome:
    all_required_files_present: bool
    missing: List[str]
    metric_deltas: Dict[str, float]


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return str(path.resolve())


def _exists(path: Path) -> bool:
    return path.resolve().exists()


def _metric_subset(metrics: Dict[str, Any]) -> Dict[str, float]:
    keys = [
        "macro_f1",
        "overall_accuracy",
        "known_intent_accuracy",
        "gate_id_recall",
        "gate_oos_rejection",
        "oos_f1",
        "domain_accuracy",
    ]
    subset: Dict[str, float] = {}
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            subset[key] = float(value)
    return subset


def _metric_deltas(reference: Dict[str, float], frozen: Dict[str, float]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for key in sorted(set(reference) & set(frozen)):
        deltas[key] = float(frozen[key] - reference[key])
    return deltas


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def _row(values: Sequence[Any]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [_row(headers), separator]
    lines.extend(_row(row) for row in rows)
    return "\n".join(lines)


def build_report() -> Dict[str, Any]:
    reference_config = _load_json(REFERENCE_CONFIG_PATH) if _exists(REFERENCE_CONFIG_PATH) else {}
    reference_eval = _load_json(REFERENCE_EVAL_PATH)
    frozen_eval = _load_json(FROZEN_EVAL_PATH)
    frozen_identity = _load_json(FROZEN_IDENTITY_PATH) if _exists(FROZEN_IDENTITY_PATH) else {}
    frozen_run_manifest = _load_json(FROZEN_RUN_MANIFEST_PATH) if _exists(FROZEN_RUN_MANIFEST_PATH) else {}

    required_paths = [
        REFERENCE_CONFIG_PATH,
        REFERENCE_EVAL_PATH,
        FROZEN_EVAL_PATH,
        FROZEN_PREDICTIONS_PATH,
        FROZEN_RUN_MANIFEST_PATH,
        FROZEN_IDENTITY_PATH,
        PROJECT_ROOT / "scripts/data/active/rebuild_multi_dataset_v19.py",
        PROJECT_ROOT / "tools/gate/train_multisphere_corrected.py",
        PROJECT_ROOT / "tools/train/train_router_v19.py",
        PROJECT_ROOT / "tools/train/train_all_experts_v19.py",
        PROJECT_ROOT / "tools/train/train_expert_v19.py",
        PROJECT_ROOT / "tools/train/train_semantic_verifier_v19.py",
        PROJECT_ROOT / "tools/analysis/historical_best_pipeline_v19.py",
        PROJECT_ROOT / "tools/analysis/run_prototype_gate_frozen_baseline_v19.py",
        PROJECT_ROOT / "tools/analysis/prototype_path_utils.py",
        PROJECT_ROOT / "tools/analysis/component_path_utils.py",
        PROJECT_ROOT / "tools/eval/eval_system_pipeline_v19.py",
        PROJECT_ROOT / "src/pipeline/system_pipeline.py",
        PROJECT_ROOT / "src/gate/multi_sphere_oos_detector.py",
        PROJECT_ROOT / "src/gate/intent_prototype_matcher.py",
        PROJECT_ROOT / "src/gate/llm_semantic_verifier.py",
        PROJECT_ROOT / "src/gate/multi_prototype_gate.py",
        PROJECT_ROOT / "src/models/architecture.py",
        PROJECT_ROOT / "src/router/router_model.py",
        PROJECT_ROOT / "archive/reorg_2026-04-13/repro/historical_best_clinc150_kir50_py/pipeline_blueprint.py",
        PROJECT_ROOT / "archive/reorg_2026-04-13/repro/historical_best_clinc150_kir50_py/repro_entry.py",
        PROJECT_ROOT / "archive/reorg_2026-04-13/repro/file_index.py",
        DATA_ROOT / "MANIFEST.json",
        DATA_ROOT / "AUDIT.json",
        DATA_ROOT / "gate/train.json",
        DATA_ROOT / "gate/val.json",
        DATA_ROOT / "gate/test.json",
        DATA_ROOT / "router/domain_map.json",
        DATA_ROOT / "router/train.json",
        DATA_ROOT / "router/val.json",
        DATA_ROOT / "router/test.json",
        LEGACY_DATA_ROOT / "OOS_STRATIFICATION.json",
        PROJECT_ROOT / "outputs/experiments/components/router/router_v19/best_model.pt",
        PROJECT_ROOT / "outputs/experiments/components/experts/experts_v19",
        PROJECT_ROOT / "outputs/experiments/components/semantic/semantic_verifier_v19_u098_105/best_model.pt",
    ]

    missing = [_rel(path) for path in required_paths if not _exists(path)]

    reference_metrics = _metric_subset(reference_eval.get("metrics", {}))
    frozen_metrics = _metric_subset(frozen_eval.get("metrics", {}))
    deltas = _metric_deltas(reference_metrics, frozen_metrics)

    outcome = ValidationOutcome(
        all_required_files_present=not missing,
        missing=missing,
        metric_deltas=deltas,
    )

    return {
        "title": "Historical Best CLINC150@KIR50 Mother Chain",
        "summary": {
            "dataset": "CLINC150",
            "kir": 0.5,
            "seed": 42,
            "profile": HISTORICAL_BEST_PIPELINE.name,
            "reference_eval": _rel(REFERENCE_EVAL_PATH),
            "frozen_eval": _rel(FROZEN_EVAL_PATH),
            "metrics_aligned": outcome.all_required_files_present and all(abs(v) <= 1e-9 for v in deltas.values()),
        },
        "profile_sections": reference_config.get("profile", HISTORICAL_BEST_PIPELINE.profile_sections()),
        "reference": {
            "config_path": _rel(REFERENCE_CONFIG_PATH),
            "metrics": reference_metrics,
        },
        "frozen": {
            "identity_manifest": _rel(FROZEN_IDENTITY_PATH),
            "run_manifest": _rel(FROZEN_RUN_MANIFEST_PATH),
            "metrics": frozen_metrics,
            "identity_excerpt": {
                "canonical_pipeline_name": frozen_identity.get("canonical_pipeline_name"),
                "role": frozen_identity.get("role"),
                "reference_eval_results": frozen_identity.get("reference_eval_results"),
            },
            "run_manifest_excerpt": {
                "seed": frozen_run_manifest.get("seed"),
                "data_root": frozen_run_manifest.get("data_root"),
                "semantic_gate_mode": frozen_run_manifest.get("semantic_gate_mode"),
                "semantic_gate_enabled": frozen_run_manifest.get("semantic_gate_enabled"),
            },
        },
        "call_chain": CALL_CHAIN,
        "supporting_code": SUPPORTING_CODE,
        "parameter_chain": PARAMETER_CHAIN,
        "artifacts": {
            "data_root": _rel(DATA_ROOT),
            "legacy_data_root": _rel(LEGACY_DATA_ROOT),
            "reference_eval": _rel(REFERENCE_EVAL_PATH),
            "frozen_eval": _rel(FROZEN_EVAL_PATH),
            "frozen_predictions": _rel(FROZEN_PREDICTIONS_PATH),
            "frozen_identity": _rel(FROZEN_IDENTITY_PATH),
            "frozen_run_manifest": _rel(FROZEN_RUN_MANIFEST_PATH),
            "gate_detector": reference_config.get("artifacts", {}).get("gate_detector_path"),
            "router_ckpt": reference_config.get("artifacts", {}).get("router_ckpt"),
            "experts_root": reference_config.get("artifacts", {}).get("experts_root"),
            "semantic_verifier_ckpt": reference_config.get("artifacts", {}).get("semantic_verifier_ckpt"),
            "multi_prototype_path": reference_config.get("artifacts", {}).get("multi_prototype_path"),
        },
        "validation": {
            "all_required_files_present": outcome.all_required_files_present,
            "missing": outcome.missing,
            "metric_deltas": outcome.metric_deltas,
        },
    }


def render_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    reference_metrics = report["reference"]["metrics"]
    frozen_metrics = report["frozen"]["metrics"]

    metric_rows = []
    for key in ["macro_f1", "overall_accuracy", "known_intent_accuracy", "gate_id_recall", "gate_oos_rejection", "oos_f1", "domain_accuracy"]:
        if key in reference_metrics or key in frozen_metrics:
            metric_rows.append(
                [
                    key,
                    reference_metrics.get(key, ""),
                    frozen_metrics.get(key, ""),
                    report["validation"]["metric_deltas"].get(key, ""),
                ]
            )

    call_rows = [
        [
            item["stage"],
            item["owner"],
            item["fixed_vs_variable"],
        ]
        for item in report["call_chain"]
    ]

    param_rows = [
        [
            item["dimension"],
            item["value"],
            item["classification"],
        ]
        for item in report["parameter_chain"]
    ]

    code_rows = [
        [item["path"], item["role"]]
        for item in report["supporting_code"]
    ]

    lines: List[str] = []
    lines.append("# Historical Best CLINC150@KIR50 Mother Chain")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Dataset: `{summary['dataset']}` | KIR: `{summary['kir']}` | Seed: `{summary['seed']}` | Profile: `{summary['profile']}`"
    )
    lines.append(f"- Reference eval: `{summary['reference_eval']}`")
    lines.append(f"- Frozen eval: `{summary['frozen_eval']}`")
    lines.append(f"- Validation complete: `{report['validation']['all_required_files_present']}`")
    if report["validation"]["missing"]:
        lines.append(f"- Missing files: `{len(report['validation']['missing'])}`")
    lines.append("")
    lines.append("## Metric Alignment")
    lines.append("")
    lines.append(_format_table(["metric", "reference", "frozen", "delta"], metric_rows))
    lines.append("")
    lines.append("## Canonical Call Chain")
    lines.append("")
    lines.append(_format_table(["stage", "active code", "controlled scope"], call_rows))
    lines.append("")
    lines.append("## Parameter Chain")
    lines.append("")
    lines.append(_format_table(["dimension", "canonical value", "classification"], param_rows))
    lines.append("")
    lines.append("## Supporting Code")
    lines.append("")
    lines.append(_format_table(["path", "role"], code_rows))
    lines.append("")
    lines.append("## Rules for Generalization")
    lines.append("")
    lines.append("- Dataset changes only at data rebuild time.")
    lines.append("- KIR changes only at data rebuild time.")
    lines.append("- The profile, stage order, and eval policy must stay fixed across all dataset/KIR experiments.")
    lines.append("- Downstream stages must never re-split data or retune thresholds on test data.")
    lines.append("")
    lines.append("## Reproduction Command")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 tools/analysis/validate_historical_best_chain_v19.py")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and document the historical-best CLINC150@KIR50 chain")
    parser.add_argument(
        "--output_dir",
        default="outputs/reports/historical_best_validation",
        help="Directory where the JSON and markdown reports will be written.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Maximum allowed absolute metric delta between archived and frozen evals.",
    )
    args = parser.parse_args()

    report = build_report()
    report["validation"]["tolerance"] = float(args.tolerance)
    report["validation"]["aligned"] = all(
        abs(delta) <= float(args.tolerance) for delta in report["validation"]["metric_deltas"].values()
    ) and report["validation"]["all_required_files_present"]
    report["summary"]["metrics_aligned"] = report["validation"]["aligned"]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "historical_best_mother_chain.json"
    md_path = output_dir / "historical_best_mother_chain.md"

    _write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"report_json: {json_path}")
    print(f"report_md: {md_path}")
    print(f"aligned: {report['validation']['aligned']}")
    if report["validation"]["missing"]:
        print("missing:")
        for item in report["validation"]["missing"]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
