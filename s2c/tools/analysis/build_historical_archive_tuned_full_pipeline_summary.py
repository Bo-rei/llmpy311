#!/usr/bin/env python3
"""Publish the OOS-focused archive pipeline candidates and paper comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TUNED_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_archive_tuned_full_pipeline_seed42"
LORA_TUNED_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_archive_tuned_lora_extended_full_pipeline_seed42"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "analysis" / "historical_archive_tuned_full_pipeline"
PAPER = {
    "clinc150": {"known_f1": 79.95, "oos_f1": 91.96, "accuracy": 86.78},
    "stackoverflow": {"known_f1": 75.48, "oos_f1": 89.71, "accuracy": 85.54},
    "banking77": {"known_f1": 74.90, "oos_f1": 88.23, "accuracy": 78.98},
}
CONFIGS = {
    "clinc150": "Partial K=3, lambda=2.0, threshold=1.0, normalized_union",
    "stackoverflow": "Partial K=1, lambda=2.0, threshold=0.85, nearest_sphere",
    "banking77": "Partial K=2, lambda=1.0, threshold=0.95, nearest_sphere",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_rows(dataset: str, root: Path, method: str) -> tuple[dict[str, str], Path]:
    path = root / dataset / "kir50_seed42" / "per_method.csv"
    rows = [row for row in _read_rows(path) if row["method"] == method]
    if len(rows) != 1:
        raise ValueError(f"expected one {method} row in {path}")
    return rows[0], path


def build(output_root: Path) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for dataset in ("clinc150", "stackoverflow", "banking77"):
        row, source = _source_rows(dataset, TUNED_ROOT, "partial_k1")
        paper = PAPER[dataset]
        selected_rows.append(
            {
                "dataset": dataset,
                "method": "partial_k1",
                "configuration": CONFIGS[dataset],
                "known_f1": float(row["known_macro_f1"]) * 100.0,
                "oos_f1": float(row["oos_f1"]) * 100.0,
                "accuracy": float(row["overall_accuracy"]) * 100.0,
                "full_macro_f1": float(row["f1_all"]) * 100.0,
                "known_recall": float(row["known_recall"]) * 100.0,
                "false_accept_rate": float(row["false_accept_rate"]) * 100.0,
                "false_reject_rate": float(row["false_reject_rate"]) * 100.0,
                "expert_error_rate": float(row["expert_error_rate"]) * 100.0,
                "paper_known_f1": paper["known_f1"],
                "paper_oos_f1": paper["oos_f1"],
                "paper_accuracy": paper["accuracy"],
                "delta_known_f1_pp": float(row["known_macro_f1"]) * 100.0 - paper["known_f1"],
                "delta_oos_f1_pp": float(row["oos_f1"]) * 100.0 - paper["oos_f1"],
                "delta_accuracy_pp": float(row["overall_accuracy"]) * 100.0 - paper["accuracy"],
                "gate_metric_max_abs_delta": float(row["gate_metric_max_abs_delta"]),
                "source": str(source.resolve()),
            }
        )
        candidate_rows.append(selected_rows[-1])
    lora_row, lora_source = _source_rows("banking77", LORA_TUNED_ROOT, "lora_k1")
    candidate_rows.append(
        {
            "dataset": "banking77",
            "method": "lora_k1",
            "configuration": "LoRA K=4, lambda=2.0, threshold=0.90, nearest_sphere",
            "known_f1": float(lora_row["known_macro_f1"]) * 100.0,
            "oos_f1": float(lora_row["oos_f1"]) * 100.0,
            "accuracy": float(lora_row["overall_accuracy"]) * 100.0,
            "full_macro_f1": float(lora_row["f1_all"]) * 100.0,
            "known_recall": float(lora_row["known_recall"]) * 100.0,
            "false_accept_rate": float(lora_row["false_accept_rate"]) * 100.0,
            "false_reject_rate": float(lora_row["false_reject_rate"]) * 100.0,
            "expert_error_rate": float(lora_row["expert_error_rate"]) * 100.0,
            "paper_known_f1": PAPER["banking77"]["known_f1"],
            "paper_oos_f1": PAPER["banking77"]["oos_f1"],
            "paper_accuracy": PAPER["banking77"]["accuracy"],
            "delta_known_f1_pp": float(lora_row["known_macro_f1"]) * 100.0 - PAPER["banking77"]["known_f1"],
            "delta_oos_f1_pp": float(lora_row["oos_f1"]) * 100.0 - PAPER["banking77"]["oos_f1"],
            "delta_accuracy_pp": float(lora_row["overall_accuracy"]) * 100.0 - PAPER["banking77"]["accuracy"],
            "gate_metric_max_abs_delta": float(lora_row["gate_metric_max_abs_delta"]),
            "source": str(lora_source.resolve()),
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "comparison_to_paper.csv", selected_rows)
    _write_csv(output_root / "candidate_comparison.csv", candidate_rows)
    summary = {
        "schema_version": "s2c.historical_archive_tuned_full_pipeline.summary.v1",
        "protocol": "historical_v19_archive_development_tuning",
        "selection_split": "validation_with_oos_labels",
        "known_accuracy_guard": "validation known_f1 and accuracy within 1 pp of method baseline",
        "datasets": ["clinc150", "stackoverflow", "banking77"],
        "seed": 42,
        "kir": 0.50,
        "selected_method": "partial_k1",
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "gate_replay_max_abs_delta": max(float(row["gate_metric_max_abs_delta"]) for row in selected_rows),
        "selected_rows": len(selected_rows),
        "candidate_rows": len(candidate_rows),
        "paper_reference": "fulltex.tex KIR=.50 Ours row",
        "note": "Banking77 is standard archive data; paper historical Banking row is not a byte-identical task and is reported as a reference only.",
    }
    (output_root / "MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
