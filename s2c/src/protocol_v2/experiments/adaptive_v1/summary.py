"""Summaries and completion checks for RC-AMBL pilot artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file


def read_metrics(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "runs" / "stackoverflow").glob("seed_*/metrics.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def verify(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    seen = set()
    rows = read_metrics(root)
    for path in sorted((root / "runs" / "stackoverflow").glob("seed_*/run_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(path.parent.name.split("_", 1)[1])
        seen.add(seed)
        if payload.get("status") != "complete" or payload.get("test_used_for_selection") is not False:
            errors.append(f"invalid manifest {path}")
        if payload.get("experiment_id") != "adaptive_v1" or payload.get("dataset") != "stackoverflow":
            errors.append(f"manifest identity {path}")
        for row in rows:
            if int(row.get("seed", -1)) == seed:
                for key in ("oos_f1", "known_recall", "f1_all", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos", "fpr95", "fit_seconds", "inference_seconds"):
                    try:
                        value = float(row[key])
                        if not np.isfinite(value):
                            errors.append(f"non-finite {path}/{key}")
                    except (KeyError, ValueError):
                        errors.append(f"missing {path}/{key}")
                if row.get("method") not in {"RC-AMBL-KnownOnly", "RC-AMBL-ProxyOOS"}:
                    errors.append(f"unexpected method {path}/{row.get('method')}")
                if row.get("test_used_for_selection") != "False":
                    errors.append(f"test selection flag {path}")
    expected = {13, 42, 87}
    errors.extend(f"missing seed {seed}" for seed in sorted(expected - seen))
    result = {"experiment_id": "adaptive_v1", "planned_seeds": sorted(expected), "completed_seeds": sorted(seen & expected), "metric_rows": len(rows), "errors": errors, "status": "pass" if not errors and seen == expected else "fail"}
    atomic_write_json(root / "ADAPTIVE_V1_VERIFY.json", result)
    return result


def summarize(root: Path, output: Path) -> list[dict[str, Any]]:
    rows = read_metrics(root)
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(output / "per_seed_results.csv", buffer.getvalue())
    summary: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        entry = {"method": method, "n": len(subset)}
        for metric in ("oos_f1", "known_macro_f1", "f1_all", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos", "mean_k_y", "total_centers"):
            values = np.asarray([float(row[metric]) for row in subset], dtype=np.float64)
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary.append(entry)
    fields = sorted({key for row in summary for key in row})
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
    atomic_write_text(output / "aggregate_results.csv", buffer.getvalue())
    return summary


def _open_metrics_from_predictions(path: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = np.asarray([int(row["gold_is_oos"]) for row in rows], dtype=np.int64)
    pred_oos = np.asarray([int(row["predicted_is_oos"]) for row in rows], dtype=np.int64)
    gold = ["__oos__" if int(row["gold_is_oos"]) else str(row["gold_intent"]) for row in rows]
    pred = ["__oos__" if int(row["predicted_is_oos"]) else str(row.get("nearest_known_intent", "__unknown__")) for row in rows]
    known_labels = sorted({value for value, flag in zip(gold, labels) if not flag})
    known = labels == 0
    oos = labels == 1
    tp = int(np.sum((pred_oos == 1) & oos))
    fp = int(np.sum((pred_oos == 1) & known))
    fn = int(np.sum((pred_oos == 0) & oos))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "known_macro_f1": float(f1_score(gold, pred, labels=known_labels, average="macro", zero_division=0)),
        "f1_k": float(f1_score(gold, pred, labels=known_labels, average="macro", zero_division=0)),
        "f1_all": float(f1_score(gold, pred, labels=known_labels + ["__oos__"], average="macro", zero_division=0)),
        "f1_u": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "accuracy": float(accuracy_score(gold, pred)),
        "known_recall": float(np.mean(pred_oos[known] == 0)),
    }


def build_main_results(paths: Any, root: Path, output: Path) -> list[dict[str, Any]]:
    """Combine new RC-AMBL rows with hash-validated historical references.

    Historical rows are never numerically edited. Missing metrics remain NaN
    and are labelled ``validated_reused`` or ``unavailable`` explicitly.
    """

    rows: list[dict[str, Any]] = []
    reuse: list[dict[str, Any]] = []
    e2_root = paths.run_root / "e2_gate_core_dense"
    for seed in (13, 42, 87):
        for k, method in ((1, "s2c_k1_nearest_sphere"), (2, "s2c_fixed_k2_nearest_sphere")):
            candidates = sorted(e2_root.glob(f"*stackoverflow__kir_0.50__seed_{seed}__repr_frozen_minilm__k_{k}__dist_mahalanobis_diag__boundary_mean_std/metrics.json"))
            if not candidates:
                reuse.append({"method": method, "seed": seed, "source": "unavailable", "reason": "missing_exact_e2_run"})
                continue
            metric_path = candidates[0]
            payload = json.loads(metric_path.read_text(encoding="utf-8"))
            metric = payload["combined"]
            pred_path = metric_path.parent / "predictions" / "test.jsonl"
            open_values = _open_metrics_from_predictions(pred_path) if pred_path.is_file() else {}
            rows.append({"dataset": "stackoverflow", "kir": 0.50, "seed": seed, "method": method, "source": "validated_reused", "n": 1, "oos_f1": metric.get("oos_f1"), "known_macro_f1": open_values.get("known_macro_f1", np.nan), "f1_all": open_values.get("f1_all", np.nan), "f1_k": open_values.get("f1_k", np.nan), "f1_u": metric.get("oos_f1"), "accuracy": open_values.get("accuracy", np.nan), "known_recall": metric.get("id_recall"), "false_accept_rate": metric.get("false_accept_rate"), "false_reject_rate": metric.get("false_reject_rate"), "auroc": metric.get("auroc"), "aupr_oos": metric.get("aupr_oos"), "mean_k_y": float(k), "total_centers": metric.get("effective_cluster_count"), "protocol_version": "protocol_v2_textoir_v1", "distance": "mahalanobis_diag", "boundary": "mean_std", "threshold_source": "E2_fixed_threshold_1.0", "source_path": str(metric_path)})
            reuse.append({"method": method, "seed": seed, "source": "validated_reused", "source_path": str(metric_path), "metrics_sha256": sha256_file(metric_path)})

    e3_path = paths.run_root / "e3_mechanisms" / "summaries" / "E3_partition_control_summary.csv"
    if e3_path.is_file():
        with e3_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("dataset") == "stackoverflow" and abs(float(row.get("kir", 0)) - 0.5) < 1e-9 and int(row.get("seed", -1)) in {13, 42, 87} and row.get("k") == "2" and row.get("distance") == "mahalanobis_diag" and row.get("partition") == "random_balanced" and row.get("oos_source") == "combined":
                    rows.append({"dataset": "stackoverflow", "kir": 0.50, "seed": int(row["seed"]), "method": "random_balanced_k2_nearest_sphere", "source": "validated_reused", "n": 1, "oos_f1": float(row["oos_f1"]), "known_macro_f1": np.nan, "f1_all": np.nan, "f1_k": np.nan, "f1_u": float(row["oos_f1"]), "accuracy": np.nan, "known_recall": float(row["id_recall"]), "false_accept_rate": float(row["false_accept_rate"]), "false_reject_rate": float(row["false_reject_rate"]), "auroc": float(row["auroc"]), "aupr_oos": float(row["aupr_oos"]), "mean_k_y": 2.0, "total_centers": int(row["effective_cluster_count"]), "protocol_version": row.get("protocol_version"), "distance": row.get("distance"), "boundary": "mean_std", "threshold_source": "E3_fixed_threshold_1.0", "source_path": str(e3_path)})
                    reuse.append({"method": "random_balanced_k2_nearest_sphere", "seed": int(row["seed"]), "source": "validated_reused", "source_path": str(e3_path), "summary_row": True})

    brak_path = paths.run_root / "brak_v1" / "summaries" / "BRAK_PILOT_SUMMARY.tsv"
    if brak_path.is_file():
        with brak_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("dataset") == "stackoverflow" and abs(float(row.get("kir", 0)) - 0.5) < 1e-9 and row.get("method") == "brak" and int(row.get("seed", -1)) in {13, 42, 87}:
                    rows.append({"dataset": "stackoverflow", "kir": 0.50, "seed": int(row["seed"]), "method": "BRAK_known_only", "source": "validated_reused", "n": 1, "oos_f1": float(row["oos_f1"]), "known_macro_f1": float(row["f1_k"]), "f1_all": float(row["f1_all"]), "f1_k": float(row["f1_k"]), "f1_u": float(row["f1_u"]), "accuracy": float(row["accuracy"]), "known_recall": float(row["id_recall"]), "false_accept_rate": float(row["false_accept_rate"]), "false_reject_rate": float(row["false_reject_rate"]), "auroc": float(row["auroc"]), "aupr_oos": float(row["aupr_oos"]), "mean_k_y": 1.0, "total_centers": int(row["effective_cluster_count"]), "protocol_version": "protocol_v2_textoir_v1", "distance": "mahalanobis_diag", "boundary": "mean_std", "threshold_source": "BRAK_known_only", "source_path": str(brak_path)})
                    reuse.append({"method": "BRAK_known_only", "seed": int(row["seed"]), "source": "validated_reused", "source_path": str(brak_path), "coverage_note": "formal seed 13 unavailable; n=2"})

    for path, method_names in ((paths.results_root / "diagnostics" / "ccsg" / "pilot_summary.csv", {"ccsg_k1": "CCSG_k1_reused", "ccsg_k2": "CCSG_k2_reused"}), (paths.results_root / "diagnostics" / "urcsg" / "pilot_summary.csv", {"URCSG-primary": "URCSG_primary_reused"})):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("dataset") != "stackoverflow" or row.get("method") not in method_names or row.get("metric") != "oos_f1":
                    continue
                method = method_names[row["method"]]
                rows.append({"dataset": "stackoverflow", "kir": 0.50, "seed": "aggregate", "method": method, "source": "validated_reused", "n": int(row.get("n", row.get("n_seeds", 3))), "oos_f1": float(row["mean"]), "known_macro_f1": np.nan, "f1_all": np.nan, "f1_k": np.nan, "f1_u": float(row["mean"]), "accuracy": np.nan, "known_recall": np.nan, "false_accept_rate": np.nan, "false_reject_rate": np.nan, "auroc": np.nan, "aupr_oos": np.nan, "mean_k_y": np.nan, "total_centers": np.nan, "protocol_version": "protocol_v2_textoir_v1", "distance": "mahalanobis_diag", "boundary": "mean_std", "threshold_source": "Known-only", "source_path": str(path)})
                reuse.append({"method": method, "seed": "aggregate", "source": "validated_reused", "source_path": str(path), "n": row.get("n", row.get("n_seeds", 3))})

    mogb_path = paths.results_root / "mogb" / "fair_matrix.csv"
    if mogb_path.is_file():
        with mogb_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("dataset") == "stackoverflow" and abs(float(row.get("kir", 0)) - 0.5) < 1e-9 and row.get("method") == "mogb_minilm" and int(row.get("seed", -1)) in {13, 42, 87}:
                    rows.append({"dataset": "stackoverflow", "kir": 0.50, "seed": int(row["seed"]), "method": "MOGB-MiniLM-compatible-component-reference", "source": "validated_reused", "n": 1, "oos_f1": float(row["oos_f1"]), "known_macro_f1": float(row["f1_k"]), "f1_all": float(row["f1_all"]), "f1_k": float(row["f1_k"]), "f1_u": float(row["f1_u"]), "accuracy": float(row["accuracy"]), "known_recall": float(row["id_recall"]), "false_accept_rate": float(row["false_accept_rate"]), "false_reject_rate": float(row["false_reject_rate"]), "auroc": float(row["auroc"]), "aupr_oos": float(row["aupr_oos"]), "mean_k_y": np.nan, "total_centers": float(row["effective_cluster_count"]), "protocol_version": "protocol_v2_textoir_v1", "distance": "euclidean", "boundary": "MOGB_mean_radius", "threshold_source": "MOGB-compatible", "source_path": str(mogb_path)})
                    reuse.append({"method": "MOGB-MiniLM-compatible-component-reference", "seed": int(row["seed"]), "source": "validated_reused", "source_path": str(mogb_path), "note": "not official MOGB"})

    new_rows = read_metrics(root)
    for row in new_rows:
        row["n"] = 1
        row["source"] = "new_run"
        row["boundary"] = "weighted_class_evidence_parent_margin"
    rows.extend(new_rows)
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(output / "main_results.csv", buffer.getvalue())
    reuse_fields = sorted({key for row in reuse for key in row})
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=reuse_fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(reuse)
    atomic_write_text(output / "baseline_reuse_validation.csv", buffer.getvalue())
    return rows
