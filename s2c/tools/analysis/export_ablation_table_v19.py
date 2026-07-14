#!/usr/bin/env python3
"""Export the paper-facing v19 ablation table.

Only the main cascade ablation line is included:
full pipeline, w/o Gate, Cascade-MiniLM, and Cascade-SmolLM.
Single-stage baselines are intentionally excluded from this table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

MAIN_VARIANT_NAMES = {
    "full_anchor": "full_pipeline",
    "full_pipeline": "full_pipeline",
    "wo_gate_confidence": "wo_gate",
    "wo_gate": "wo_gate",
    "banking_wo_geometric_gate_expert_confidence": "w/o Geometric Gate",
    "cascade_minilm": "cascade_minilm",
    "cascade_smollm": "cascade_smollm",
}
MAIN_VARIANT_ORDER = {
    "full_pipeline": 0,
    "wo_gate": 1,
    "w/o Geometric Gate": 1,
    "cascade_minilm": 2,
    "cascade_smollm": 3,
}
AUDIT_VARIANT_NAMES = {
    "full_pipeline": "Full Pipeline",
    "wo_gate": "w/o Gate",
    "w/o Geometric Gate": "w/o Gate",
    "cascade_minilm": "Cascade-MiniLM",
    "cascade_smollm": "Cascade-SmolLM",
}
FIELDNAMES = ["dataset", "kir_tag", "variant", "overall_acc", "known_f1", "oos_f1", "threshold", "confidence_source"]


def _load_rows(root: Path) -> List[Dict[str, Any]]:
    summary_json = root / "ablation_summary.json"
    if summary_json.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        return list(payload.get("runs", []))

    summary_csv = root / "ablation_summary.csv"
    if summary_csv.exists():
        with summary_csv.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    raise FileNotFoundError(f"No ablation_summary.json or ablation_summary.csv found under {root}")


def _metric(row: Dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _load_audit_metrics(root: Path) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    path = root / "audit" / "ablation_metrics_recomputed.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        (str(row.get("dataset", "")), str(row.get("kir_tag", "")), str(row.get("variant", ""))): row
        for row in rows
    }


def _fill_from_audit(
    value: Any,
    audit_row: Dict[str, Any] | None,
    audit_field: str,
) -> Any:
    if value not in (None, ""):
        return value
    if not audit_row:
        return value
    return audit_row.get(audit_field, value)


def _parse_full_override(value: str) -> tuple[str, Dict[str, float]]:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 4:
        raise ValueError(
            "--full_override must use DATASET:KNOWN_F1:OOS_F1:ACC, "
            f"got {value!r}"
        )
    dataset, known_f1, oos_f1, overall_acc = parts
    return dataset, {
        "known_f1": float(known_f1),
        "oos_f1": float(oos_f1),
        "overall_acc": float(overall_acc),
    }


def export_table(
    root: Path,
    output_csv: Path,
    full_overrides: Dict[str, Dict[str, float]] | None = None,
) -> List[Dict[str, Any]]:
    # Kept for CLI/test backward compatibility, but intentionally ignored for
    # paper exports: all table values must come from real run summaries.
    full_overrides = full_overrides or {}
    loaded_rows = _load_rows(root)
    audit_metrics = _load_audit_metrics(root)
    banking_has_geometric_replacement = any(
        str(row.get("dataset", row.get("dataset_slug", ""))) == "BANKING77-OOS"
        and str(row.get("variant", "")) == "banking_wo_geometric_gate_expert_confidence"
        for row in loaded_rows
    )
    rows = []
    for row in loaded_rows:
        source_variant = str(row.get("variant", ""))
        dataset = row.get("dataset", row.get("dataset_slug", ""))
        if (
            banking_has_geometric_replacement
            and str(dataset) == "BANKING77-OOS"
            and source_variant == "wo_gate"
        ):
            continue
        paper_variant = MAIN_VARIANT_NAMES.get(source_variant)
        if paper_variant is None:
            continue
        overall_acc = _metric(row, "overall_accuracy", "overall_acc")
        known_f1 = _metric(row, "known_macro_f1", "known_f1")
        oos_f1 = _metric(row, "oos_f1")
        kir_tag = _metric(row, "kir_tag")
        audit_row = audit_metrics.get((str(dataset), str(kir_tag), AUDIT_VARIANT_NAMES.get(paper_variant, paper_variant)))
        overall_acc = _fill_from_audit(overall_acc, audit_row, "acc_raw")
        known_f1 = _fill_from_audit(known_f1, audit_row, "known_f1_raw")
        oos_f1 = _fill_from_audit(oos_f1, audit_row, "oos_f1_raw")
        rows.append(
            {
                "dataset": dataset,
                "kir_tag": kir_tag,
                "variant": paper_variant,
                "overall_acc": overall_acc,
                "known_f1": known_f1,
                "oos_f1": oos_f1,
                "macro_f1": _metric(row, "macro_f1"),
                "threshold": _metric(row, "threshold", "selected_threshold"),
                "confidence_source": _metric(row, "confidence_source"),
            }
        )

    rows.sort(
        key=lambda item: (
            str(item["dataset"]),
            str(item.get("kir_tag", "")),
            MAIN_VARIANT_ORDER.get(str(item["variant"]), 99),
        )
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDNAMES} for row in rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-facing v19 ablation table")
    parser.add_argument("root", help="Experiment root containing ablation_summary.json/csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <root>/paper_ablation_table.csv.",
    )
    parser.add_argument(
        "--full_override",
        action="append",
        default=[],
        help="Override a full_pipeline row as DATASET:KNOWN_F1:OOS_F1:ACC.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_csv = Path(args.output) if args.output else root / "paper_ablation_table.csv"
    overrides = dict(_parse_full_override(item) for item in args.full_override)
    rows = export_table(root, output_csv, full_overrides=overrides)
    print(json.dumps({"rows": len(rows), "output": str(output_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
