#!/usr/bin/env python3
"""Summarize v19 ablation outputs under an experiment root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

COMPARISON_FAMILIES = {
    "full_anchor": "anchor",
    "wo_gate": "structure",
    "wo_gate_naive": "structure_lower_bound",
    "wo_gate_confidence": "structure",
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
    "flat_minilm": "single_stage",
    "flat_smollm": "single_stage",
}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation eval_results files")
    parser.add_argument(
        "--root",
        default="outputs/experiments/pipeline/ablations/latest_strongest_v19/latest_strongest_kir50_20260425",
    )
    parser.add_argument(
        "--exclude_variants",
        nargs="*",
        default=["wo_router"],
        help="Variants to omit from paper-facing summaries.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    summary_json = root / "ablation_summary.json"
    if summary_json.exists():
        payload = _load_json(summary_json)
        rows = payload.get("runs", [])
        if rows:
            with open(root / "ablation_summary.csv", "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(
                json.dumps(
                    {"runs": len(rows), "summary": str(root / "ablation_summary.csv")},
                    ensure_ascii=False,
                )
            )
            return

    rows: List[Dict[str, Any]] = []
    for eval_path in sorted(root.glob("*/*/*/eval_results.json")):
        parts = eval_path.relative_to(root).parts
        dataset, kir_tag, variant = parts[0], parts[1], parts[2]
        if variant in set(args.exclude_variants):
            continue
        metrics = _load_json(eval_path).get("metrics", {})
        rows.append(
            {
                "dataset": dataset,
                "kir_tag": kir_tag,
                "variant": variant,
                "comparison_family": COMPARISON_FAMILIES.get(variant, "diagnostic"),
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
        )

    if not rows:
        print(json.dumps({"runs": 0, "root": str(root)}, ensure_ascii=False))
        return

    with open(root / "ablation_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(root / "ablation_summary.json", "w", encoding="utf-8") as file:
        json.dump({"runs": rows}, file, indent=2, ensure_ascii=False)

    print(json.dumps({"runs": len(rows), "summary": str(root / "ablation_summary.csv")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
