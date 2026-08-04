#!/usr/bin/env python3
"""Audit lambda selection and run a read-only protocol_v2 sensitivity sweep.

This stage deliberately reuses the frozen protocol_v2 embedding caches and never
writes under ``../artifacts``.  It refits only the small K=1/K=2 boundary
objects needed for the requested KIR=.50 diagnostic; no encoder is loaded and
no test score is used for selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from protocol_v2.data.manifests import dataset_manifest_path, read_json
from protocol_v2.data.registry import registry_path
from protocol_v2.data.schema import format_kir
from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from protocol_v2.runtime.paths import ProtocolV2Paths


DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87)
KIR = 0.50
DISTANCE = "mahalanobis_diag"
K_VALUES = (1, 2)
LAMBDA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
PAPER_DEFAULT_LAMBDA = {"clinc150": 0.5, "banking77": 1.0, "stackoverflow": 1.0}
PAPER_DEFAULT_K = 2
KNOWN_CALIBRATION_FRR_LIMIT = 0.05
STAGE = "lambda_leakage_audit_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                rows.append(value)
    return rows


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT.parent, text=True, capture_output=True, check=False
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 else None


def _load_views(paths: ProtocolV2Paths, dataset: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    root = paths.view_root / dataset / f"seed_{seed}" / f"kir_{format_kir(KIR)}"
    names = (
        "train_known",
        "calibration_known",
        "test_known",
        "test_heldout_oos",
        "test_native_oos",
        "test_combined",
    )
    views = {name: _read_jsonl(root / f"{name}.jsonl") for name in names}
    ids = {name: [str(row["sample_id"]) for row in rows] for name, rows in views.items()}
    if any(len(values) != len(set(values)) for values in ids.values()):
        raise ValueError(f"Duplicate sample IDs in views: {dataset}/seed_{seed}")
    if set(ids["train_known"]) & set(ids["calibration_known"]):
        raise ValueError(f"Train/calibration overlap: {dataset}/seed_{seed}")
    if set(ids["calibration_known"]) & set(ids["test_combined"]):
        raise ValueError(f"Calibration/test overlap: {dataset}/seed_{seed}")
    if set(ids["train_known"]) & set(ids["test_combined"]):
        raise ValueError(f"Train/test overlap: {dataset}/seed_{seed}")
    return views


def _find_cache(
    paths: ProtocolV2Paths,
    dataset: str,
    seed: int,
    split: str,
    rows: list[dict[str, Any]],
    registry_sha: str,
    canonical_sha: str,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    root = paths.embedding_cache_root / dataset / f"seed_{seed}" / f"kir_{format_kir(KIR)}"
    expected_ids_sha = _sha256_ids(str(row["sample_id"]) for row in rows)
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for metadata_path in root.glob(f"{split}_*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        key = metadata.get("cache_key", {})
        if key.get("split") != split or key.get("registry_sha256") != registry_sha:
            continue
        if key.get("canonical_manifest_sha256") != canonical_sha:
            continue
        if metadata.get("sample_ids_sha256") != expected_ids_sha:
            continue
        npz_path = metadata_path.with_suffix(".npz")
        if not npz_path.is_file() or int(metadata.get("sample_count", -1)) != len(rows):
            continue
        candidates.append((metadata_path.stat().st_mtime, npz_path, metadata))
    if not candidates:
        raise FileNotFoundError(
            f"No sample-id-aligned embedding cache: {dataset}/seed_{seed}/{split}; "
            f"expected_ids_sha256={expected_ids_sha}"
        )
    _, npz_path, metadata = max(candidates, key=lambda item: item[0])
    with np.load(npz_path, allow_pickle=False) as payload:
        values = np.asarray(payload["embeddings"], dtype=np.float32)
    if values.shape[0] != len(rows):
        raise ValueError(f"Cache row count mismatch: {npz_path}")
    if metadata.get("embedding_sha256") != hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest():
        raise ValueError(f"Cache content hash mismatch: {npz_path}")
    return values, metadata, npz_path


def _binary_labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([0 if str(row.get("oos_source", "known")) == "known" else 1 for row in rows], dtype=np.int64)


def _open_metrics(
    rows: list[dict[str, Any]], output: dict[str, np.ndarray], detector: MultiSphereOOSDetector
) -> dict[str, float]:
    labels = _binary_labels(rows)
    scores = np.asarray(output["score"], dtype=np.float64)
    predicted_oos = (scores > 1.0).astype(np.int64)
    binary = compute_binary_oos_metrics(labels, scores, threshold=1.0)
    known_intents = sorted({str(row["intent"]) for row, label in zip(rows, labels) if label == 0})
    oos_label = "__oos__"
    gold = [str(row["intent"]) if label == 0 else oos_label for row, label in zip(rows, labels)]
    predicted: list[str] = []
    for is_oos, cluster in zip(predicted_oos, output["nearest_cluster"]):
        if int(is_oos):
            predicted.append(oos_label)
        else:
            predicted.append(str(detector.spheres[int(cluster)].intent_name))
    all_labels = known_intents + [oos_label]
    return {
        **binary,
        "known_recall": float(binary["id_recall"]),
        "f1_all": float(f1_score(gold, predicted, labels=all_labels, average="macro", zero_division=0)),
        "f1_k": float(f1_score(gold, predicted, labels=known_intents, average="macro", zero_division=0)),
        "f1_u": float(binary["oos_f1"]),
        "accuracy": float(accuracy_score(gold, predicted)),
        "known_macro_f1": float(f1_score(
            [str(row["intent"]) for row, label in zip(rows, labels) if label == 0],
            [pred for pred, label in zip(predicted, labels) if label == 0],
            labels=known_intents,
            average="macro",
            zero_division=0,
        )),
    }


def _fit_detector(train: np.ndarray, intents: np.ndarray, k: int) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector(
        radius_method="mean_std",
        radius_lambda=1.0,
        center_mode="class_centroid_mixture",
        distance_metric=DISTANCE,
        covariance_eps=1e-6,
        l2_normalize=True,
        subcenters_per_intent=k,
        random_state=42,
    )
    detector.fit(train, intents)
    return detector


def _calibration_selection(
    detector: MultiSphereOOSDetector, calibration: np.ndarray
) -> tuple[float, bool, dict[float, float]]:
    rows: dict[float, float] = {}
    for value in LAMBDA_VALUES:
        detector.radius_lambda = float(value)
        detector._compute_radii()
        output = detector.predict_with_scores(calibration)
        rows[float(value)] = float(np.mean(np.asarray(output["pred"], dtype=np.int64) == 1))
    valid = [value for value, frr in rows.items() if frr <= KNOWN_CALIBRATION_FRR_LIMIT]
    if valid:
        return min(valid), True, rows
    return max(LAMBDA_VALUES), False, rows


def _e2_reference(paths: ProtocolV2Paths, dataset: str, seed: int, k: int) -> dict[str, Any] | None:
    run_id = (
        f"protocol_v2_textoir_v1__{dataset}__kir_0.50__seed_{seed}__repr_frozen_minilm__"
        f"k_{k}__dist_{DISTANCE}__boundary_mean_std"
    )
    path = paths.run_root / "e2_gate_core_dense" / run_id / "metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "metrics": payload["combined"], "run_id": run_id}


def _lambda_usage_rows() -> list[dict[str, Any]]:
    sources = [
        (
            "active_detector_geometry",
            "src/protocol_v2/gate/multi_sphere_oos_detector.py",
            "mean_std radius is mean distance plus lambda times distance std; lambda is a boundary input, not learned from test",
            "caller supplied",
            "fixed_without_selection",
            "known train only for radii",
        ),
        (
            "active_protocol_runner",
            "src/protocol_v2/experiments/runner.py",
            "radius_lambda comes from config; mean_std does not select lambda and test_used_for_selection=false",
            "1.0 default",
            "fixed_without_selection",
            "train_known for radius; no OOS selection",
        ),
        (
            "active_e2_config",
            "configs/experiments/protocol_v2_textoir_v1/gate_core_dense.yaml",
            "dense frozen-MiniLM sweep fixes radius_lambda=1.0",
            "1.0",
            "fixed_without_selection",
            "train_known only; test only evaluated",
        ),
        (
            "active_boundary_attribution_config",
            "configs/experiments/protocol_v2_textoir_v1/multicenter_boundary_attribution.yaml",
            "boundary-attribution controls fix radius_lambda=1.0; no lambda search is declared",
            "1.0",
            "fixed_without_selection",
            "Known train/calibration diagnostic contract; no test selection",
        ),
        (
            "paper_contract",
            "fulltex.tex",
            "paper settings state lambda=0.5 for CLINC150 and 1 for other datasets",
            "0.5/1.0 dataset-specific",
            "historical_fixed_contract",
            "paper text does not document a separate OOS selection split",
        ),
        (
            "historical_v19_tuned_runner",
            "tools/experiments/cluster_separability/runner.py",
            "LAMBDA_CANDIDATES loop selects a constrained candidate on val_data",
            "0.50..2.50",
            "validation_oos_selected",
            "legacy validation rows include OOS labels; not current protocol evidence",
        ),
        (
            "historical_corrected_runner",
            "tools/gate/train_multisphere_corrected.py",
            "tune_radius_on_val selects lambda and margin on validation",
            "0.50..2.50",
            "validation_oos_selected",
            "legacy result only; keep out of current primary table",
        ),
        (
            "historical_cascade_gate_preparation",
            "tools/eval/prepare_cascade_gates.py",
            "cascade preparation fixes radius_lambda=1.0 while selecting only other Known-validation settings",
            "1.0",
            "fixed_without_selection",
            "Known validation only; historical cascade contract",
        ),
        (
            "historical_true_lambda_artifact_reference",
            "tools/analysis/run_prototype_gate_frozen_baseline_v19.py",
            "legacy prototype points to a pre-existing true_lambda_1p6 artifact; this is not a new selection run",
            "1.6 artifact reference",
            "historical_artifact_reference",
            "selection provenance is historical and excluded from current primary evidence",
        ),
        (
            "r1_contract_repair",
            "configs/experiments/protocol_v2_textoir_v1/r1_contract_repair.yaml",
            "lambda fixed at 1.0; checkpoint selection uses Known calibration",
            "1.0",
            "known_only_selected_or_fixed",
            "no validation OOS in active R1 contract",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, relative, evidence, candidates, status, data_used in sources:
        path = PROJECT_ROOT / relative
        rows.append(
            {
                "source_id": name,
                "path": relative,
                "path_sha256": _sha256_file(path) if path.is_file() else None,
                "evidence": evidence,
                "lambda_candidates": candidates,
                "selection_status": status,
                "data_used": data_used,
                "test_used_for_selection": False,
                "current_primary_policy": status in {"fixed_without_selection", "known_only_selected_or_fixed"},
            }
        )
    return rows


def _split_audit(paths: ProtocolV2Paths) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for dataset in DATASETS:
        canonical_manifest = dataset_manifest_path(paths.manifest_root, dataset)
        canonical_sha = _sha256_file(canonical_manifest)
        for seed in SEEDS:
            views = _load_views(paths, dataset, seed)
            registry_file = registry_path(paths, dataset, seed, KIR)
            registry = read_json(registry_file)
            registry_sha = str(registry["registry_sha256"])
            train_ids = [str(row["sample_id"]) for row in views["train_known"]]
            calibration_ids = [str(row["sample_id"]) for row in views["calibration_known"]]
            test_known_ids = [str(row["sample_id"]) for row in views["test_known"]]
            test_oos_rows = views["test_heldout_oos"] + views["test_native_oos"]
            test_oos_ids = [str(row["sample_id"]) for row in test_oos_rows]
            checks = {
                "train_calibration_disjoint": not set(train_ids) & set(calibration_ids),
                "calibration_test_known_disjoint": not set(calibration_ids) & set(test_known_ids),
                "validation_oos_test_oos_disjoint": True,
                "validation_unknown_labels_test_unknown_labels_disjoint": True,
                "train_calibration_test_ids_unique": len(set(train_ids + calibration_ids + test_known_ids + test_oos_ids))
                == len(train_ids + calibration_ids + test_known_ids + test_oos_ids),
            }
            if not all(checks.values()):
                raise AssertionError(f"Split audit failed: {dataset}/seed_{seed}: {checks}")
            audit.append(
                {
                    "dataset": dataset,
                    "protocol_version": paths.dataset_version,
                    "kir": KIR,
                    "seed": seed,
                    "registry_sha256": registry_sha,
                    "canonical_manifest_sha256": canonical_sha,
                    "known_train_ids": train_ids,
                    "known_calibration_ids": calibration_ids,
                    "test_known_ids": test_known_ids,
                    "validation_oos_ids": [],
                    "test_oos_ids": test_oos_ids,
                    "validation_unknown_labels": [],
                    "test_unknown_labels": sorted({str(row["intent"]) for row in test_oos_rows}),
                    "counts": {
                        "known_train": len(train_ids),
                        "known_calibration": len(calibration_ids),
                        "test_known": len(test_known_ids),
                        "validation_oos": 0,
                        "test_oos": len(test_oos_ids),
                    },
                    "checks": checks,
                    "selection_policy": "known_only_calibration; no validation OOS is materialized in protocol_v2_textoir_v1",
                    "test_used_for_selection": False,
                }
            )
    return audit


def _run_sensitivity(paths: ProtocolV2Paths) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {"e2_lambda1_checks": [], "cache_inputs": [], "invalid_rows": []}
    for dataset in DATASETS:
        canonical_sha = _sha256_file(dataset_manifest_path(paths.manifest_root, dataset))
        paper_lambda = PAPER_DEFAULT_LAMBDA[dataset]
        for seed in SEEDS:
            views = _load_views(paths, dataset, seed)
            registry_file = registry_path(paths, dataset, seed, KIR)
            registry = read_json(registry_file)
            registry_sha = str(registry["registry_sha256"])
            train, train_meta, train_cache = _find_cache(
                paths, dataset, seed, "train_known", views["train_known"], registry_sha, canonical_sha
            )
            calibration, calibration_meta, calibration_cache = _find_cache(
                paths, dataset, seed, "calibration_known", views["calibration_known"], registry_sha, canonical_sha
            )
            test, test_meta, test_cache = _find_cache(
                paths, dataset, seed, "test_combined", views["test_combined"], registry_sha, canonical_sha
            )
            integrity["cache_inputs"].append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "train_cache": str(train_cache),
                    "calibration_cache": str(calibration_cache),
                    "test_cache": str(test_cache),
                    "train_embedding_sha256": train_meta["embedding_sha256"],
                    "calibration_embedding_sha256": calibration_meta["embedding_sha256"],
                    "test_embedding_sha256": test_meta["embedding_sha256"],
                }
            )
            train_intents = np.asarray([str(row["intent"]) for row in views["train_known"]], dtype=object)
            selection_data: dict[int, tuple[float, bool, dict[float, float]]] = {}
            detectors: dict[int, MultiSphereOOSDetector] = {}
            for k in K_VALUES:
                detector = _fit_detector(train, train_intents, k)
                selected_lambda, constraint_met, calibration_frr = _calibration_selection(detector, calibration)
                selection_data[k] = (selected_lambda, constraint_met, calibration_frr)
                detectors[k] = detector
            for source_k, k_label, alias_of in ((1, "k_1", None), (2, "k_2", None), (2, "paper_default_k", "k_2")):
                detector = detectors[source_k]
                selected_lambda, constraint_met, calibration_frr = selection_data[source_k]
                e2_reference = _e2_reference(paths, dataset, seed, source_k)
                for value in LAMBDA_VALUES:
                    detector.radius_lambda = float(value)
                    detector._compute_radii()
                    test_output = detector.predict_with_scores(test)
                    calibration_output = detector.predict_with_scores(calibration)
                    metrics = _open_metrics(views["test_combined"], test_output, detector)
                    calibration_frr_value = float(np.mean(np.asarray(calibration_output["pred"], dtype=np.int64) == 1))
                    e2_delta: dict[str, Any] = {}
                    if e2_reference is not None and abs(value - 1.0) < 1e-12:
                        expected = e2_reference["metrics"]
                        comparisons = [
                            abs(float(metrics["oos_f1"]) - float(expected["oos_f1"])),
                            abs(float(metrics["oos_precision"]) - float(expected["oos_precision"])),
                            abs(float(metrics["oos_recall"]) - float(expected["oos_recall"])),
                            abs(float(metrics["known_recall"]) - float(expected["id_recall"])),
                            abs(float(metrics["false_accept_rate"]) - float(expected["false_accept_rate"])),
                            abs(float(metrics["false_reject_rate"]) - float(expected["false_reject_rate"])),
                        ]
                        prediction_mismatch = None
                        prediction_path = Path(e2_reference["path"]).parent / "predictions" / "test.jsonl"
                        if prediction_path.is_file():
                            old = _read_jsonl(prediction_path)
                            prediction_mismatch = sum(
                                int(a["predicted_is_oos"]) != int(b)
                                for a, b in zip(old, test_output["pred"])
                            )
                        e2_delta = {
                            "max_abs_delta": max(comparisons),
                            "prediction_mismatch": prediction_mismatch,
                            "reference_run_id": e2_reference["run_id"],
                        }
                        integrity["e2_lambda1_checks"].append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "k": source_k,
                                **e2_delta,
                            }
                        )
                    cluster_labels = np.asarray(detector._train_cluster_labels, dtype=np.int64)
                    cluster_sizes = np.bincount(cluster_labels) if cluster_labels.size else np.asarray([], dtype=np.int64)
                    rows.append(
                        {
                            "stage": STAGE,
                            "protocol_version": paths.dataset_version,
                            "dataset": dataset,
                            "dataset_version": paths.dataset_version,
                            "kir": KIR,
                            "seed": seed,
                            "representation": "frozen_minilm",
                            "distance": DISTANCE,
                            "boundary": "mean_std",
                            "radius_method": "mean_std",
                            "threshold": 1.0,
                            "radius_lambda": value,
                            "paper_default_lambda": paper_lambda,
                            "k_label": k_label,
                            "k_gate": source_k,
                            "source_k": source_k,
                            "alias_of": alias_of or "",
                            "known_only_selected_lambda": selected_lambda,
                            "known_only_selection_constraint_met": constraint_met,
                            "calibration_false_reject_rate": calibration_frr_value,
                            "calibration_selected_false_reject_rate": calibration_frr[selected_lambda],
                            "calibration_known_count": len(calibration),
                            "test_count": len(views["test_combined"]),
                            **metrics,
                            "effective_cluster_count": len(detector.spheres),
                            "minimum_cluster_size": int(cluster_sizes.min()) if cluster_sizes.size else 0,
                            "e2_lambda1_max_abs_delta": e2_delta.get("max_abs_delta", ""),
                            "e2_lambda1_prediction_mismatch": e2_delta.get("prediction_mismatch", ""),
                            "selection_data": "known_calibration_only_for_diagnostic_selection; main sweep does not select lambda",
                            "test_used_for_selection": False,
                            "run_id": f"{STAGE}__{dataset}__kir_0.50__seed_{seed}__{k_label}__lambda_{value:.2f}",
                        }
                    )
    integrity["planned_rows"] = len(DATASETS) * len(SEEDS) * 3 * len(LAMBDA_VALUES)
    integrity["unique_fit_cells"] = len(DATASETS) * len(SEEDS) * len(K_VALUES)
    integrity["completed_rows"] = len(rows)
    integrity["failed_rows"] = len(integrity["invalid_rows"])
    integrity["distance"] = DISTANCE
    integrity["k_contract"] = {"k_1": 1, "k_2": 2, "paper_default_k": PAPER_DEFAULT_K, "paper_default_aliases_k_2": True}
    return rows, integrity


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _atomic_write_text(path, "")
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_adaptive_decision(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["k_label"] not in {"k_1", "k_2"}:
            continue
        if float(row["radius_lambda"]) != float(row["known_only_selected_lambda"]):
            continue
        grouped[(str(row["dataset"]), int(row["seed"]), str(row["k_label"]))] = row
    decisions: dict[str, Any] = {}
    for dataset in DATASETS:
        deltas = []
        selected = []
        for seed in SEEDS:
            one = grouped.get((dataset, seed, "k_1"))
            two = grouped.get((dataset, seed, "k_2"))
            if one is None or two is None:
                continue
            deltas.append(
                {
                    "seed": seed,
                    "oos_f1_delta_k2_minus_k1": float(two["oos_f1"]) - float(one["oos_f1"]),
                    "f1_all_delta_k2_minus_k1": float(two["f1_all"]) - float(one["f1_all"]),
                    "known_recall_delta_k2_minus_k1": float(two["known_recall"]) - float(one["known_recall"]),
                    "false_accept_delta_k2_minus_k1": float(two["false_accept_rate"]) - float(one["false_accept_rate"]),
                    "lambda_k1": float(one["known_only_selected_lambda"]),
                    "lambda_k2": float(two["known_only_selected_lambda"]),
                }
            )
            selected.extend([float(one["known_only_selected_lambda"]), float(two["known_only_selected_lambda"])])
        mean_f1_all = float(np.mean([item["f1_all_delta_k2_minus_k1"] for item in deltas])) if deltas else math.nan
        mean_oos = float(np.mean([item["oos_f1_delta_k2_minus_k1"] for item in deltas])) if deltas else math.nan
        mean_known = float(np.mean([item["known_recall_delta_k2_minus_k1"] for item in deltas])) if deltas else math.nan
        decisions[dataset] = {
            "status": "not_authorized_without_intent_level_known_only_evidence",
            "aggregate_known_only_selected_lambda_deltas": deltas,
            "mean_oos_f1_delta_k2_minus_k1": mean_oos,
            "mean_f1_all_delta_k2_minus_k1": mean_f1_all,
            "mean_known_recall_delta_k2_minus_k1": mean_known,
            "direction_consistent_oos_f1_seeds": int(sum(item["oos_f1_delta_k2_minus_k1"] >= 0 for item in deltas)),
            "known_only_selected_lambdas": sorted(set(selected)),
            "reason": "The existing per-intent evidence is test-oracle only; this stage does not promote it to adaptive-K selection evidence.",
        }
    _atomic_write_json(path, {"stage": STAGE, "adaptive_k_gate": decisions, "split_merge_pilot_authorized": False})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "diagnostics")
    args = parser.parse_args()
    paths = ProtocolV2Paths.discover(PROJECT_ROOT)
    paths.require_experiment_admission()
    if paths.dataset_version != "protocol_v2_textoir_v1":
        raise RuntimeError(f"Unexpected active dataset version: {paths.dataset_version}")
    leakage_root = args.output_root / "lambda_leakage"
    sensitivity_root = args.output_root / "lambda_sensitivity"
    split_audit = _split_audit(paths)
    _atomic_write_json(
        leakage_root / "data_split_audit.json",
        {
            "stage": STAGE,
            "protocol_version": paths.dataset_version,
            "kir": KIR,
            "datasets": DATASETS,
            "seeds": SEEDS,
            "validation_oos_available": False,
            "selection_policy": "known_only_calibration; no validation OOS is materialized",
            "test_used_for_selection": False,
            "audit_rows": split_audit,
            "lambda_usage": _lambda_usage_rows(),
            "hard_invariants": {
                "train_calibration_disjoint": True,
                "calibration_test_known_disjoint": True,
                "validation_oos_test_oos_disjoint": True,
                "validation_unknown_labels_test_unknown_labels_disjoint": True,
            },
            "git_commit": _git_value("-C", PROJECT_ROOT.parent, "rev-parse", "HEAD"),
        },
    )
    rows, integrity = _run_sensitivity(paths)
    _write_csv(sensitivity_root / "summary.csv", rows)
    _write_csv(sensitivity_root / "lambda_usage.csv", _lambda_usage_rows())
    _atomic_write_json(sensitivity_root / "integrity.json", integrity)
    _write_adaptive_decision(sensitivity_root / "adaptive_k_decision.json", rows)
    print(json.dumps({"split_audit_rows": len(split_audit), **integrity}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
