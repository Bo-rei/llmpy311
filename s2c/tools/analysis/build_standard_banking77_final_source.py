"""Build the manifest-backed standard Banking77 final result source.

This exporter reads the already completed locked Ours evaluation and writes a
separate result namespace.  It deliberately does not inspect any historical
``banking77_oos`` result and does not select a configuration from test scores.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT.parent / "artifacts" / "s2c" / "analysis" / "banking_textoir_aligned"
OUT = ROOT / "results" / "final_paper_main" / "standard_banking77_textoir_aligned"
DATASET = "banking77"
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tag(kir: float, seed: int) -> str:
    return f"kir{round(kir * 100):02d}_seed{seed}"


def metric_percent(value: float) -> float:
    return float(value) * 100.0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=ART / "final_selection",
        help="locked Ours selection/evaluation root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT,
        help="result-source output directory",
    )
    args = parser.parse_args()
    selection_root = args.selection_root.resolve()
    output_root = args.output.resolve()
    from tools.eval.final_prediction_metrics import prediction_metrics

    data_manifest_path = ART / "MANIFEST.json"
    recipe_manifest_path = ART / "recipe_search_corrected" / "MANIFEST.json"
    selection_manifest_path = selection_root / "MANIFEST.json"
    components_manifest_path = ART / "components_manifest.json"
    for path in (data_manifest_path, recipe_manifest_path, selection_manifest_path, components_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    data_manifest = read(data_manifest_path)
    recipe_manifest = read(recipe_manifest_path)
    selection_manifest = read(selection_manifest_path)
    components_manifest = read(components_manifest_path)
    required_flags = {
        "test_used_for_selection": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
    }
    for name, manifest in (
        ("data", data_manifest),
        ("recipe", recipe_manifest),
        ("geometry", selection_manifest),
        ("components", components_manifest),
    ):
        for key, expected in required_flags.items():
            if manifest.get(key) is not expected:
                raise ValueError(f"{name} manifest violates {key}={expected}: {manifest.get(key)!r}")
    if data_manifest.get("dataset_variant") != "standard_77_intent":
        raise ValueError("primary source is not standard 77-intent Banking77")
    if data_manifest.get("protocol") != "protocol_v2_textoir_v1":
        raise ValueError("unexpected Banking77 protocol")
    selection_count = selection_manifest.get(
        "lock_count", selection_manifest.get("units")
    )
    if selection_manifest.get("status") != "locked" or selection_count != 9:
        raise ValueError("geometry selection is not a complete 9-cell lock")
    if components_manifest.get("status") != "complete":
        raise ValueError("downstream components are not complete")

    from sklearn.metrics import accuracy_score, f1_score

    per_seed: list[dict] = []
    lock_records: list[dict] = []
    dataset_audit: list[dict] = []
    source_hashes: dict[str, str] = {}
    for kir in KIRS:
        for seed in SEEDS:
            cell = tag(kir, seed)
            data = ART / "data" / DATASET / cell
            result_path = selection_root / "final" / DATASET / cell / "Ours.json"
            prediction_path = result_path.with_suffix(".npz")
            lock_path = selection_root / "locks" / f"{DATASET}_{cell}.json"
            for path in (data / "known_labels.json", data / "gate" / "test.json", result_path, prediction_path, lock_path):
                if not path.is_file():
                    raise FileNotFoundError(path)

            known = read(data / "known_labels.json")
            test = read(data / "gate" / "test.json")
            lock = read(lock_path)
            result = read(result_path)
            values = np.load(prediction_path, allow_pickle=False)
            y_true = values["y_true"].astype(str)
            y_pred = values["y_pred"].astype(str)
            expected = np.asarray(["__oos__" if int(row["label"]) else row["intent"] for row in test])
            if not np.array_equal(y_true, expected):
                raise ValueError(f"prediction/test alignment failed: {result_path}")
            if result.get("full_pipeline") is not True or result.get("method") != "Ours":
                raise ValueError(f"not a full-pipeline Ours result: {result_path}")
            for key, expected_value in required_flags.items():
                if lock.get(key) is not expected_value:
                    raise ValueError(f"lock violates {key}: {lock_path}")
            metrics = prediction_metrics(y_true, y_pred, known)
            for key in ("oos_f1", "known_f1", "accuracy"):
                if abs(float(metrics[key]) - float(result["metrics"][key])) > 1e-10:
                    raise ValueError(f"stored metric mismatch for {result_path}: {key}")
            truth_oos = y_true == "__oos__"
            pred_oos = y_pred == "__oos__"
            row = {
                "dataset": DATASET,
                "dataset_variant": "standard_77_intent",
                "protocol": "protocol_v2_textoir_v1",
                "method": "Ours",
                "kir": kir,
                "seed": seed,
                "oos_f1": metric_percent(metrics["oos_f1"]),
                "known_f1": metric_percent(metrics["known_f1"]),
                "accuracy": metric_percent(metrics["accuracy"]),
                "known_recall": metric_percent(metrics["known_recall"]),
                "false_accept_rate": metric_percent(metrics["false_accept_rate"]),
                "oos_precision": metric_percent(metrics["oos_precision"]),
                "oos_recall": metric_percent(metrics["oos_recall"]),
                "known_count": int(metrics["known_count"]),
                "oos_count": int(metrics["oos_count"]),
                "oos_false_accepts": int(metrics["oos_false_accepts"]),
                "known_false_rejects": int(metrics["known_false_rejects"]),
                "configuration_lock": str(lock_path),
                "result": str(result_path),
                "predictions": str(prediction_path),
                "test_used_for_selection": False,
            }
            per_seed.append(row)
            lock_records.append({"cell": cell, "lock": lock})
            dataset_audit.append(
                {
                    "cell": cell,
                    "known_labels": known,
                    "known_count": len(known),
                    "train_count": len(read(data / "gate" / "train.json")),
                    "dev_count": len(read(data / "gate" / "val.json")),
                    "test_count": len(test),
                    "test_known_count": int((~truth_oos).sum()),
                    "test_oos_count": int(truth_oos.sum()),
                    "predicted_oos_count": int(pred_oos.sum()),
                    "source_export": next(
                        unit["source_export"]
                        for unit in data_manifest["units"]
                        if unit["seed"] == seed and unit["kir"] == kir
                    ),
                }
            )
            for path in (result_path, prediction_path, lock_path):
                source_hashes[str(path)] = sha256(path)

    per_seed_fields = [
        "dataset", "dataset_variant", "protocol", "method", "kir", "seed",
        "oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate",
        "oos_precision", "oos_recall", "known_count", "oos_count",
        "oos_false_accepts", "known_false_rejects", "configuration_lock",
        "result", "predictions", "test_used_for_selection",
    ]
    write_csv(output_root / "per_seed.csv", per_seed, per_seed_fields)

    summary: list[dict] = []
    metrics = ("oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate", "oos_precision", "oos_recall")
    for kir in KIRS:
        group = [row for row in per_seed if row["kir"] == kir]
        if len(group) != 3:
            raise ValueError(f"expected three Ours seeds for KIR={kir}: {len(group)}")
        row = {"dataset": DATASET, "dataset_variant": "standard_77_intent", "protocol": "protocol_v2_textoir_v1", "method": "Ours", "kir": kir, "n": 3}
        for metric in metrics:
            values = np.asarray([item[metric] for item in group], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        row["selection"] = selection_manifest.get(
            "selection",
            selection_manifest.get(
                "threshold_selection",
                "Known-only selection; test deferred",
            ),
        )
        summary.append(row)

    summary_fields = ["dataset", "dataset_variant", "protocol", "method", "kir", "n"]
    for metric in metrics:
        summary_fields += [f"{metric}_mean", f"{metric}_std"]
    summary_fields += ["selection"]
    write_csv(output_root / "summary.csv", summary, summary_fields)

    reference_path = ROOT / "results" / "final_paper_main" / "banking_reported_references.csv"
    references: list[dict] = []
    if reference_path.is_file():
        with reference_path.open(newline="", encoding="utf-8") as stream:
            references = [row for row in csv.DictReader(stream) if row.get("dataset") == DATASET]
    comparison: list[dict] = []
    for row in summary:
        comparison.append(
            {
                "dataset": DATASET,
                "kir": row["kir"],
                "method": "Ours",
                "oos_f1": row["oos_f1_mean"],
                "known_f1": row["known_f1_mean"],
                "accuracy": row["accuracy_mean"],
                "source": str(output_root / "summary.csv"),
                "comparison": "three-seed final full-pipeline; TextOIR-aligned standard Banking77",
                "supervision": "Known-only train/dev",
                "backbone": "MiniLM Gate + SmolLM-135M downstream",
            }
        )
    for ref in references:
        comparison.append(
            {
                "dataset": DATASET,
                "kir": float(ref["kir"]),
                "method": ref["method"],
                "oos_f1": float(ref["oos_f1"]),
                "known_f1": float(ref["known_f1"]),
                "accuracy": float(ref["accuracy"]),
                "source": ref.get("source", ""),
                "comparison": ref.get("comparison", "reported reference"),
                "supervision": ref.get("supervision", "not audited"),
                "backbone": ref.get("backbone", "not audited"),
            }
        )
    strongest_by_kir = {}
    for kir in KIRS:
        baselines = [row for row in comparison if row["kir"] == kir and row["method"] != "Ours"]
        strongest_by_kir[kir] = max(baselines, key=lambda row: row["oos_f1"]) if baselines else None
    for row in comparison:
        strongest = strongest_by_kir.get(row["kir"])
        row["strongest_reported_baseline"] = strongest["method"] if strongest else ""
        row["delta_vs_strongest_reported_baseline_pp"] = (
            float(row["oos_f1"]) - float(strongest["oos_f1"]) if strongest else ""
        )
    comparison_fields = [
        "dataset", "kir", "method", "oos_f1", "known_f1", "accuracy", "source",
        "comparison", "supervision", "backbone", "strongest_reported_baseline",
        "delta_vs_strongest_reported_baseline_pp",
    ]
    write_csv(output_root / "comparison.csv", comparison, comparison_fields)

    lock_config = {
        "status": "locked",
        "dataset": DATASET,
        "dataset_variant": "standard_77_intent",
        "protocol": "protocol_v2_textoir_v1",
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "recipe_manifest": str(recipe_manifest_path),
        "geometry_manifest": str(selection_manifest_path),
        "components_manifest": str(components_manifest_path),
        "recipe_winners": {
            str(kir): read(ART / "recipe_search_corrected" / f"selection_kir{round(kir * 100):02d}.json")["winner"]
            for kir in KIRS
        },
        "geometry_locks": lock_records,
        "downstream": {
            "router": "single-domain constant router",
            "expert": "SmolLM-135M + LoRA, Known train/dev checkpoint selection, test deferred",
            "expert_lora_r": 16,
            "expert_lora_alpha": 32,
            "expert_epochs": 15,
            "expert_batch_size": 32,
            "expert_patience": 5,
        },
        "selection_flags": required_flags,
        "test_policy": "test predictions are final evaluation only; no test score enters selection",
    }
    write_json(output_root / "config_lock.json", lock_config)
    write_json(output_root / "dataset_audit.json", {"source": str(data_manifest_path), "cells": dataset_audit, "banking77_oos_excluded": True})

    lines = [
        "# Standard Banking77 final comparison",
        "",
        "This namespace is a manifest-backed standard `banking77` source for the current Ours result. It excludes `banking77_oos`.",
        "All Ours configurations were selected from Known train/dev only; test was read only for the final predictions stored here.",
        "",
        "| KIR | Ours OOS F1 | Strongest reported reference | Delta (pp) |",
        "|---:|---:|---|---:|",
    ]
    for row in summary:
        strongest = strongest_by_kir.get(row["kir"])
        baseline_name = strongest["method"] if strongest else "unavailable"
        baseline_value = strongest["oos_f1"] if strongest else None
        delta = row["oos_f1_mean"] - baseline_value if baseline_value is not None else None
        lines.append(
            f"| {row['kir']:.2f} | {row['oos_f1_mean']:.2f}±{row['oos_f1_std']:.2f} | "
            f"{baseline_name} ({baseline_value:.2f}) | {delta:+.2f} |"
        )
    lines += [
        "",
        "Reference rows are reported TextOIR values and are not shared-seed reruns in this namespace; numerical superiority is therefore not a substitute for a matched direct comparison.",
    ]
    (output_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for path in (data_manifest_path, recipe_manifest_path, selection_manifest_path, components_manifest_path):
        source_hashes[str(path)] = sha256(path)
    manifest = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_source": str(output_root),
        "dataset": DATASET,
        "dataset_variant": "standard_77_intent",
        "protocol": "protocol_v2_textoir_v1",
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "planned_units": 9,
        "completed_units": len(per_seed),
        "banking77_oos_excluded": True,
        "real_oos_used_for_training": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_used_for_selection": False,
        "test_read": "final evaluation only",
        "selection": "Known-only recipe and geometry selection; no test oracle",
        "metric_units": "percentage points in CSV; std uses population ddof=0",
        "artifacts": {
            "per_seed": str(output_root / "per_seed.csv"),
            "summary": str(output_root / "summary.csv"),
            "comparison": str(output_root / "comparison.csv"),
            "config_lock": str(output_root / "config_lock.json"),
            "dataset_audit": str(output_root / "dataset_audit.json"),
        },
        "source_hashes_sha256": source_hashes,
        "comparison_policy": "reported baselines are reference-only unless their split, seed, supervision and evaluator are matched",
    }
    write_json(output_root / "MANIFEST.json", manifest)
    print(json.dumps({"output": str(output_root), "completed_units": len(per_seed), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
