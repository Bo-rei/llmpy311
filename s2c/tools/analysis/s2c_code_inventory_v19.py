#!/usr/bin/env python3
"""S2C code inventory and historical-best chain report.

This utility provides a machine-readable overview of the repository and a
conservative completeness check for the historical-best CLINC150@KIR50 chain.
It is intentionally lightweight so the report can be regenerated whenever the
repository evolves.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_BEST_CHAIN: List[str] = [
    "configs/v19/clinc150_historical_best_reference.json",
    "scripts/data/active/rebuild_multi_dataset_v19.py",
    "tools/gate/train_multisphere_corrected.py",
    "tools/train/train_router_v19.py",
    "tools/train/train_all_experts_v19.py",
    "tools/train/train_expert_v19.py",
    "tools/train/train_semantic_verifier_v19.py",
    "tools/analysis/historical_best_pipeline_v19.py",
    "tools/analysis/validate_historical_best_chain_v19.py",
    "tools/analysis/run_prototype_gate_frozen_baseline_v19.py",
    "tools/analysis/prototype_path_utils.py",
    "tools/analysis/component_path_utils.py",
    "tools/eval/eval_system_pipeline_v19.py",
    "src/legacy/pipeline/system_pipeline.py",
    "src/legacy/gate/multi_sphere_oos_detector.py",
    "src/legacy/gate/intent_prototype_matcher.py",
    "src/legacy/gate/llm_semantic_verifier.py",
    "src/legacy/gate/multi_prototype_gate.py",
    "src/legacy/models/architecture.py",
    "src/legacy/router/router_model.py",
    "archive/reorg_2026-04-13/repro/historical_best_clinc150_kir50_py/pipeline_blueprint.py",
    "archive/reorg_2026-04-13/repro/historical_best_clinc150_kir50_py/repro_entry.py",
    "archive/reorg_2026-04-13/repro/file_index.py",
]


@dataclass(frozen=True)
class FileMeta:
    """Metadata tags for one Python file."""

    relpath: str
    function_group: str
    dataset_scope: str
    lifecycle: str


def _function_group(relpath: str) -> str:
    lower = relpath.lower()
    if lower.startswith("scripts/data/"):
        return "data_preprocessing"
    if "/train/" in lower or lower.startswith("tools/train/") or lower.startswith("tools/gate/"):
        return "model_training"
    if "/eval/" in lower or lower.startswith("tools/eval/"):
        return "evaluation"
    if lower.startswith("tools/analysis/"):
        return "analysis_orchestration"
    if lower.startswith("src/legacy/pipeline/"):
        return "pipeline_assembly"
    if lower.startswith("src/"):
        return "core_model_logic"
    return "other"


def _dataset_scope(relpath: str) -> str:
    lower = relpath.lower()
    if "clinc" in lower:
        return "clinc150"
    if "banking" in lower:
        return "banking77_oos"
    if "snips" in lower:
        return "snips"
    return "shared"


def _lifecycle(relpath: str) -> str:
    lower = relpath.lower()
    if lower.startswith("tools/archive/") or lower.startswith("archive/"):
        return "archived"
    if lower.startswith("repro/") or "/repro/" in lower:
        return "repro"
    return "active"


def build_inventory(root: Path) -> List[FileMeta]:
    items: List[FileMeta] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        items.append(
            FileMeta(
                relpath=rel,
                function_group=_function_group(rel),
                dataset_scope=_dataset_scope(rel),
                lifecycle=_lifecycle(rel),
            )
        )
    return items


def summarize(items: List[FileMeta]) -> Dict[str, object]:
    by_function: Dict[str, int] = {}
    by_dataset: Dict[str, int] = {}
    by_lifecycle: Dict[str, int] = {}
    for item in items:
        by_function[item.function_group] = by_function.get(item.function_group, 0) + 1
        by_dataset[item.dataset_scope] = by_dataset.get(item.dataset_scope, 0) + 1
        by_lifecycle[item.lifecycle] = by_lifecycle.get(item.lifecycle, 0) + 1
    return {
        "total_py_files": len(items),
        "by_function": by_function,
        "by_dataset_scope": by_dataset,
        "by_lifecycle": by_lifecycle,
    }


def historical_best_chain_status(root: Path) -> Dict[str, object]:
    missing = [p for p in HISTORICAL_BEST_CHAIN if not (root / p).exists()]
    return {
        "required_count": len(HISTORICAL_BEST_CHAIN),
        "existing_count": len(HISTORICAL_BEST_CHAIN) - len(missing),
        "missing": missing,
        "complete": not missing,
    }


def archive_candidates(items: List[FileMeta]) -> List[str]:
    candidates: List[str] = []
    for item in items:
        lower = item.relpath.lower()
        if item.lifecycle != "active":
            continue
        if not lower.startswith("tools/"):
            continue
        if "legacy" in lower or "cleanup" in lower:
            candidates.append(item.relpath)
    return sorted(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the S2C code inventory report")
    parser.add_argument(
        "--output",
        default="outputs/reports/s2c_code_inventory_v19.json",
        help="Output report path relative to the project root.",
    )
    args = parser.parse_args()

    items = build_inventory(PROJECT_ROOT)
    report = {
        "summary": summarize(items),
        "historical_best_clinc150_kir50_chain": historical_best_chain_status(PROJECT_ROOT),
        "archive_candidates": archive_candidates(items),
        "files": [item.__dict__ for item in items],
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {output_path}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print("historical_best_complete:", report["historical_best_clinc150_kir50_chain"]["complete"])


if __name__ == "__main__":
    main()
