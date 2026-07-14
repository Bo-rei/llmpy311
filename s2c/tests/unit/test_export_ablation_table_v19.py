from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis import export_ablation_table_v19 as exporter


def test_export_mainline_table_filters_and_renames_variants(tmp_path: Path):
    root = tmp_path / "exp"
    rows = [
        {
            "dataset": "CLINC150",
            "variant": "full_anchor",
            "overall_accuracy": 0.8,
            "known_accuracy": 0.7,
            "known_macro_f1": 0.72,
            "oos_f1": 0.9,
            "macro_f1": 0.75,
        },
        {
            "dataset": "CLINC150",
            "variant": "wo_gate_confidence",
            "overall_accuracy": 0.6,
            "known_accuracy": 0.65,
            "known_macro_f1": 0.66,
            "oos_f1": 0.5,
            "macro_f1": 0.55,
        },
        {
            "dataset": "CLINC150",
            "variant": "single_stage_minilm_val_tuned",
            "overall_accuracy": 0.95,
            "known_accuracy": 0.94,
            "oos_f1": 0.93,
            "macro_f1": 0.92,
        },
    ]
    root.mkdir()
    (root / "ablation_summary.json").write_text(
        json.dumps({"runs": rows}),
        encoding="utf-8",
    )

    output_csv = tmp_path / "table.csv"
    exported = exporter.export_table(root, output_csv)

    assert [row["variant"] for row in exported] == ["full_pipeline", "wo_gate"]
    with output_csv.open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    assert csv_rows == [
        {
            "dataset": "CLINC150",
            "kir_tag": "",
            "variant": "full_pipeline",
            "overall_acc": "0.8",
            "known_f1": "0.72",
            "oos_f1": "0.9",
            "threshold": "",
            "confidence_source": "",
        },
        {
            "dataset": "CLINC150",
            "kir_tag": "",
            "variant": "wo_gate",
            "overall_acc": "0.6",
            "known_f1": "0.66",
            "oos_f1": "0.5",
            "threshold": "",
            "confidence_source": "",
        },
    ]


def test_export_table_ignores_full_pipeline_overrides_for_paper_values(tmp_path: Path):
    root = tmp_path / "exp"
    rows = [
        {
            "dataset": "BANKING77-OOS",
            "variant": "full_anchor",
            "overall_accuracy": 0.85,
            "known_macro_f1": 0.66,
            "oos_f1": 0.91,
            "macro_f1": 0.65,
        },
        {
            "dataset": "BANKING77-OOS",
            "variant": "cascade_smollm",
            "overall_accuracy": 0.76,
            "known_macro_f1": 0.80,
            "oos_f1": 0.84,
            "macro_f1": 0.60,
        },
    ]
    root.mkdir()
    (root / "ablation_summary.json").write_text(json.dumps({"runs": rows}), encoding="utf-8")

    output_csv = tmp_path / "table.csv"
    exported = exporter.export_table(
        root,
        output_csv,
        full_overrides={
            "BANKING77-OOS": {
                "known_f1": 0.749,
                "oos_f1": 0.8823,
                "overall_acc": 0.7898,
            }
        },
    )

    assert exported[0]["variant"] == "full_pipeline"
    assert exported[0]["known_f1"] == 0.66
    assert exported[0]["oos_f1"] == 0.91
    assert exported[0]["overall_acc"] == 0.85


def test_export_table_preserves_and_sorts_kir_tags(tmp_path: Path):
    root = tmp_path / "exp"
    rows = [
        {
            "dataset": "CLINC150",
            "kir_tag": "kir75_seed42",
            "variant": "cascade_smollm",
            "overall_accuracy": 0.70,
            "known_macro_f1": 0.71,
            "oos_f1": 0.72,
        },
        {
            "dataset": "CLINC150",
            "kir_tag": "kir25_seed42",
            "variant": "full_anchor",
            "overall_accuracy": 0.80,
            "known_macro_f1": 0.81,
            "oos_f1": 0.82,
        },
    ]
    root.mkdir()
    (root / "ablation_summary.json").write_text(json.dumps({"runs": rows}), encoding="utf-8")

    exported = exporter.export_table(root, tmp_path / "table.csv")

    assert [(row["kir_tag"], row["variant"]) for row in exported] == [
        ("kir25_seed42", "full_pipeline"),
        ("kir75_seed42", "cascade_smollm"),
    ]


def test_export_table_fills_missing_metrics_from_audit(tmp_path: Path):
    root = tmp_path / "exp"
    rows = [
        {
            "dataset": "CLINC150",
            "kir_tag": "kir50_seed42",
            "variant": "full_anchor",
            "overall_accuracy": 0.86,
            "oos_f1": 0.90,
        }
    ]
    root.mkdir()
    (root / "ablation_summary.json").write_text(json.dumps({"runs": rows}), encoding="utf-8")
    audit_dir = root / "audit"
    audit_dir.mkdir()
    (audit_dir / "ablation_metrics_recomputed.csv").write_text(
        "\n".join(
            [
                "dataset,kir_tag,variant,known_f1_raw,oos_f1_raw,acc_raw",
                "CLINC150,kir50_seed42,Full Pipeline,0.85,0.91,0.87",
            ]
        ),
        encoding="utf-8",
    )

    exported = exporter.export_table(root, tmp_path / "table.csv")

    assert exported[0]["overall_acc"] == 0.86
    assert exported[0]["known_f1"] == "0.85"
    assert exported[0]["oos_f1"] == 0.90


def test_banking_export_prefers_geometric_gate_replacement_when_present(tmp_path: Path):
    root = tmp_path / "exp"
    rows = [
        {
            "dataset": "BANKING77-OOS",
            "variant": "full_anchor",
            "overall_accuracy": 0.85,
            "known_macro_f1": 0.66,
            "oos_f1": 0.91,
            "macro_f1": 0.65,
        },
        {
            "dataset": "BANKING77-OOS",
            "variant": "wo_gate",
            "overall_accuracy": 0.30,
            "known_macro_f1": 0.45,
            "oos_f1": 0.0,
            "macro_f1": 0.2,
        },
        {
            "dataset": "BANKING77-OOS",
            "variant": "banking_wo_geometric_gate_expert_confidence",
            "overall_accuracy": 0.78,
            "known_macro_f1": 0.70,
            "oos_f1": 0.82,
            "macro_f1": 0.67,
            "threshold": 0.64,
            "confidence_source": "expert_intent_confidence",
        },
    ]
    root.mkdir()
    (root / "ablation_summary.json").write_text(json.dumps({"runs": rows}), encoding="utf-8")

    output_csv = tmp_path / "table.csv"
    exported = exporter.export_table(root, output_csv)

    assert [row["variant"] for row in exported] == ["full_pipeline", "w/o Geometric Gate"]
    assert exported[1]["threshold"] == 0.64
    assert exported[1]["confidence_source"] == "expert_intent_confidence"


def test_export_precedence_is_banking_only(tmp_path: Path):
    root = tmp_path / "exp"
    rows = [
        {
            "dataset": "CLINC150",
            "variant": "wo_gate",
            "overall_accuracy": 0.5,
            "known_macro_f1": 0.4,
            "oos_f1": 0.3,
            "macro_f1": 0.2,
        },
        {
            "dataset": "STACKOVERFLOW",
            "variant": "wo_gate",
            "overall_accuracy": 0.6,
            "known_macro_f1": 0.5,
            "oos_f1": 0.4,
            "macro_f1": 0.3,
        },
        {
            "dataset": "BANKING77-OOS",
            "variant": "banking_wo_geometric_gate_expert_confidence",
            "overall_accuracy": 0.7,
            "known_macro_f1": 0.6,
            "oos_f1": 0.5,
            "macro_f1": 0.4,
        },
    ]
    root.mkdir()
    (root / "ablation_summary.json").write_text(json.dumps({"runs": rows}), encoding="utf-8")

    exported = exporter.export_table(root, tmp_path / "table.csv")

    assert ("CLINC150", "wo_gate") in [(row["dataset"], row["variant"]) for row in exported]
    assert ("STACKOVERFLOW", "wo_gate") in [(row["dataset"], row["variant"]) for row in exported]
    assert ("BANKING77-OOS", "w/o Geometric Gate") in [
        (row["dataset"], row["variant"]) for row in exported
    ]
