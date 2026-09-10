#!/usr/bin/env python3
"""Export the completed archive full-pipeline comparison as small public tables.

The runner keeps checkpoints and raw predictions under ``../artifacts``.  This
builder only reads those outputs, verifies the nine seed-42 method rows, and
writes aggregate CSV/JSON files under ``results/analysis``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = (
    ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_archive_full_pipeline_seed42"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "analysis" / "historical_archive_full_pipeline"
DATASETS = ("clinc150", "stackoverflow", "banking77")
METHODS = ("frozen_k1", "partial_k1", "lora_k1")
METRIC_FIELDS = (
    "oos_f1",
    "false_accept_rate",
    "known_recall",
    "f1_all",
    "overall_accuracy",
    "known_macro_f1",
    "false_reject_rate",
    "router_error_rate",
    "expert_error_rate",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _gate_manifest_path(dataset: str, method: str) -> Path:
    if method == "frozen_k1":
        return (
            ROOT.parent
            / "artifacts"
            / "s2c"
            / "runs"
            / "historical_protocol_v1"
            / "minilm_k1"
            / dataset
            / "kir50_seed42"
            / method
            / "run_manifest.json"
        )
    if method == "partial_k1":
        if dataset != "banking77":
            return (
                ROOT.parent
                / "artifacts"
                / "s2c"
                / "runs"
                / "historical_protocol_v1"
                / "minilm_k1"
                / dataset
                / "kir50_seed42"
                / "trainable_k1"
                / "run_manifest.json"
            )
        return (
            ROOT.parent
            / "artifacts"
            / "s2c"
            / "runs"
            / "historical_archive_minilm_seed42"
            / dataset
            / "trainable_k1"
            / "run_manifest.json"
        )
    return (
        ROOT.parent
        / "artifacts"
        / "s2c"
        / "runs"
        / "historical_archive_minilm_seed42"
        / dataset
        / method
        / "run_manifest.json"
    )


def _downstream_training_device(dataset: str, artifact_root: Path) -> str:
    component_root = artifact_root.parent / "historical_archive_downstream_seed42" / dataset
    logs = [component_root / "router" / "train.log"]
    logs.extend(sorted((component_root / "experts").glob("*/train.log")))
    if not logs or any("Device: cuda" not in path.read_text(encoding="utf-8") for path in logs):
        return "not_verified"
    return "cuda"


def _parameter_row(dataset: str, method: str, artifact_root: Path) -> dict[str, Any]:
    manifest_path = _gate_manifest_path(dataset, method)
    manifest = _read_json(manifest_path)
    report = manifest.get("freeze_report", {})
    if method == "frozen_k1":
        representation = "frozen_sentence_transformer_all-MiniLM-L6-v2"
        base_count = 22_713_216
        base_trainable = 0
        lora_count = 0
        projection_count = 0
        trainable_count = 0
        gate_device = "not_recorded"
    else:
        representation = str(manifest.get("representation", ""))
        trainable_count = int(manifest.get("trainable_parameters", 0))
        projection_count = int(report.get("projection_parameter_count", 198_016)) if report else 198_016
        base_count = int(report.get("base_parameter_count", 22_713_216)) if report else 22_713_216
        lora_count = int(report.get("lora_parameter_count", 0)) if report else 0
        base_trainable = int(report.get("base_trainable_parameter_count", trainable_count - projection_count)) if report else trainable_count - projection_count
        gate_device = str(manifest.get("device") or manifest.get("requested_device") or "not_recorded")
    return {
        "dataset": dataset,
        "method": method,
        "representation": representation,
        "base_parameter_count": base_count,
        "base_trainable_parameter_count": base_trainable,
        "lora_parameter_count": lora_count,
        "projection_parameter_count": projection_count,
        "trainable_parameter_count": trainable_count,
        "gate_device": gate_device,
        "downstream_training_device": _downstream_training_device(dataset, artifact_root),
        "test_used_for_selection": bool(manifest.get("test_used_for_selection", False)),
        "oos_used_for_training": bool(manifest.get("oos_used_for_training", False)),
        "source_manifest": str(manifest_path.resolve()),
    }


def build(artifact_root: Path, output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    manifests: list[dict[str, Any]] = []
    for dataset in DATASETS:
        method_rows = _read_csv(artifact_root / dataset / "kir50_seed42" / "per_method.csv")
        if {row.get("method") for row in method_rows} != set(METHODS) or len(method_rows) != len(METHODS):
            raise ValueError(f"unexpected method rows for {dataset}: {method_rows}")
        dataset_rows = {row["method"]: row for row in method_rows}
        frozen = dataset_rows["frozen_k1"]
        partial = dataset_rows["partial_k1"]
        for method in METHODS:
            source = dataset_rows[method]
            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": int(source["seed"]),
                "kir": float(source["kir"]),
                "method": method,
            }
            for field in METRIC_FIELDS:
                row[field] = float(source[field])
            row.update(
                {
                    "delta_oos_f1_vs_frozen_pp": (float(source["oos_f1"]) - float(frozen["oos_f1"])) * 100.0,
                    "delta_false_accept_vs_frozen_pp": (float(source["false_accept_rate"]) - float(frozen["false_accept_rate"])) * 100.0,
                    "delta_oos_f1_vs_partial_pp": (float(source["oos_f1"]) - float(partial["oos_f1"])) * 100.0,
                    "delta_false_accept_vs_partial_pp": (float(source["false_accept_rate"]) - float(partial["false_accept_rate"])) * 100.0,
                    "gate_metric_max_abs_delta": float(source["gate_metric_max_abs_delta"]),
                    "known_count": int(source["known_count"]),
                    "oos_count": int(source["oos_count"]),
                    "test_used_for_selection": source["test_used_for_selection"],
                    "oos_used_for_training": source["oos_used_for_training"],
                }
            )
            rows.append(row)
        manifest = _read_json(artifact_root / dataset / "kir50_seed42" / "MANIFEST.json")
        manifests.append(manifest)

    parameter_rows = [
        _parameter_row(dataset, method, artifact_root)
        for dataset in DATASETS
        for method in METHODS
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "comparison.csv", rows)
    _write_csv(output_root / "parameter_contract.csv", parameter_rows)

    lora_rows = {row["dataset"]: row for row in rows if row["method"] == "lora_k1"}
    partial_rows = {row["dataset"]: row for row in rows if row["method"] == "partial_k1"}
    summary = {
        "schema_version": "s2c.historical_archive_full_pipeline.summary.v1",
        "protocol": "historical_v19_archive",
        "data_root": str((ROOT.parent / "archives" / "submissions" / "s2c-submission" / "data").resolve()),
        "artifact_root": str(artifact_root.resolve()),
        "datasets": list(DATASETS),
        "seed": 42,
        "kir": 0.50,
        "methods": list(METHODS),
        "completed_units": len(rows),
        "full_pipeline_device": sorted({str(manifest.get("device")) for manifest in manifests}),
        "gate_metric_max_abs_delta": max(float(row["gate_metric_max_abs_delta"]) for row in rows),
        "lora_oos_f1_not_below_partial_by_dataset": {
            dataset: float(lora_rows[dataset]["oos_f1"]) >= float(partial_rows[dataset]["oos_f1"])
            for dataset in DATASETS
        },
        "seed_expansion": "not_run",
        "seed_expansion_reason": "LoRA OOS F1 decreased versus partial on CLINC150 and StackOverflow seed42.",
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "source_manifests": [
            str((artifact_root / dataset / "kir50_seed42" / "MANIFEST.json").resolve())
            for dataset in DATASETS
        ],
    }
    (output_root / "MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    summary = build(args.artifact_root.resolve(), args.output_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
