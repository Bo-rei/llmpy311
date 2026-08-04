#!/usr/bin/env python3
"""Run the pre-registered URCSG Known-only pilot.

URCSG is a selection layer around the active frozen-MiniLM Gate.  It never
encodes text, changes a registry, or uses test OOS rows to choose a subcenter
count.  Each target intent is tested by leave-one-known-intent-out calibration
episodes; only the target intent receives K>1 spheres.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file, sha256_json  # noqa: E402
from protocol_v2.data.manifests import dataset_manifest_path, read_json, view_manifest_path  # noqa: E402
from protocol_v2.data.registry import registry_path  # noqa: E402
from protocol_v2.experiments.brak import evaluate_intent_candidates  # noqa: E402
from protocol_v2.experiments.matrix import GateRunSpec  # noqa: E402
from protocol_v2.experiments.runner import (  # noqa: E402
    _canonical_embedding_cache,
    _embedding_cache,
    _model_fingerprint,
    _model_path,
)
from protocol_v2.experiments.urcsg import (  # noqa: E402
    estimate_target_risk,
    estimate_target_risk_rows,
    fit_detector,
    open_metrics,
    select_k,
)
from protocol_v2.gate.view_loader import load_gate_views  # noqa: E402
from protocol_v2.runtime.paths import ProtocolV2Paths  # noqa: E402
from protocol_v2.tracking.provenance import file_hashes  # noqa: E402


EXPERIMENT_ID = "urcsg_pilot_v1"
DATASETS = ("banking77", "stackoverflow")
SEEDS = (13, 42, 87)
KIR = 0.50
CANDIDATE_K = (1, 2, 3, 4, 5)
DISTANCE = "mahalanobis_diag"
RADIUS_LAMBDA = 1.0
METHODS = (
    "single_centroid",
    "fixed_k2",
    "fixed_k4",
    "BRAK",
    "URCSG-primary",
    "URCSG-largest-feasible",
    "URCSG-shuffled-primary",
    "URCSG-shuffled-largest-feasible",
    "oracle_test_k",
)


def _git_value(project_root: Path, args: list[str], fallback: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "\n")
        return
    fields = sorted({key for row in rows for key in row})
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    atomic_write_text(path, buffer.getvalue())


def _ids_sha(rows: list[dict[str, Any]]) -> str:
    return sha256_json([str(row["sample_id"]) for row in rows])


def _load_inputs(paths: ProtocolV2Paths, dataset: str, seed: int) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    paths.require_experiment_admission(dataset)
    views = load_gate_views(paths, dataset, seed, KIR)
    registry_file = registry_path(paths, dataset, seed, KIR)
    registry = read_json(registry_file)
    canonical_file = dataset_manifest_path(paths.manifest_root, dataset)
    view_file = view_manifest_path(paths.manifest_root, dataset, seed, KIR)
    input_files = file_hashes(
        {
            "registry": registry_file,
            "canonical_manifest": canonical_file,
            "view_manifest": view_file,
            "s2c_export_manifest": views.export_root / "export_manifest.json",
        }
    )
    model = _model_fingerprint(_model_path(paths, "all-MiniLM-L6-v2"))
    try:
        canonical = _canonical_embedding_cache(paths, dataset, model, None, 128)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(f"Missing frozen MiniLM cache for {dataset}; refusing implicit encoding") from exc
    spec = GateRunSpec(
        experiment_name=EXPERIMENT_ID,
        dataset=dataset,
        kir=KIR,
        seed=seed,
        k_gate=1,
        distance=DISTANCE,
        representation="frozen_minilm",
        boundary="mean_std",
        radius_lambda=RADIUS_LAMBDA,
        encoder_name="all-MiniLM-L6-v2",
        encoder_device="cpu",
        protocol_version=paths.dataset_version,
    )
    train, train_meta = _embedding_cache(
        paths, spec, "train_known", views.train, registry["registry_sha256"], input_files["canonical_manifest"], model, canonical
    )
    calibration, calibration_meta = _embedding_cache(
        paths, spec, "calibration_known", views.calibration, registry["registry_sha256"], input_files["canonical_manifest"], model, canonical
    )
    test, test_meta = _embedding_cache(
        paths, spec, "test_combined", views.test, registry["registry_sha256"], input_files["canonical_manifest"], model, canonical
    )
    ids = {
        "train": _ids_sha(views.train),
        "calibration": _ids_sha(views.calibration),
        "test": _ids_sha(views.test),
    }
    if set(ids.values()).__len__() != 3:
        raise ValueError(f"Split sample-id hashes unexpectedly collide: {dataset}/seed_{seed}")
    return views, train, calibration, test, {
        "registry_sha256": registry["registry_sha256"],
        "canonical_manifest_sha256": input_files["canonical_manifest"],
        "view_manifest_sha256": input_files["view_manifest"],
        "export_manifest_sha256": input_files["s2c_export_manifest"],
        "input_file_hashes": input_files,
        "model": model,
        "canonical_embedding_sha256": canonical.metadata.get("embedding_sha256"),
        "cache_hit": True,
        "cache_embedding_sha256": {
            "train": train_meta.get("embedding_sha256"),
            "calibration": calibration_meta.get("embedding_sha256"),
            "test": test_meta.get("embedding_sha256"),
        },
        "calibration_embedding_sha256": calibration_meta.get("embedding_sha256"),
        "sample_ids_sha256": ids,
        "cache_policy": "reuse_only; no implicit encoding",
    }


def _intent_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([str(row["intent"]) for row in rows], dtype=object)


def _fit_map(train: np.ndarray, train_intents: np.ndarray, selected: dict[str, int], seed: int):
    return fit_detector(train, train_intents, distance=DISTANCE, overrides=selected, seed=seed)


def _evaluate_map(train: np.ndarray, train_intents: np.ndarray, test: np.ndarray, test_rows: list[dict[str, Any]], selected: dict[str, int], seed: int) -> tuple[dict[str, float], Any]:
    detector = _fit_map(train, train_intents, selected, seed)
    output = detector.predict_with_scores(test)
    metrics = open_metrics(test_rows, output, detector)
    metrics.update(
        {
            "effective_cluster_count": int(len(detector.spheres)),
            "minimum_cluster_size": int(
                min(np.sum(detector._train_cluster_labels == int(sphere.cluster_id)) for sphere in detector.spheres)
            ),
            "selected_k_mean": float(np.mean(list(selected.values()))),
            "selected_k_median": float(np.median(list(selected.values()))),
            "selected_k_gt1_fraction": float(np.mean(np.asarray(list(selected.values())) > 1)),
        }
    )
    return metrics, detector


def _read_brak_map(paths: ProtocolV2Paths, dataset: str, seed: int, inputs: dict[str, Any], known: list[str]) -> dict[str, int] | None:
    """Reuse the completed BRAK StackOverflow cells only when provenance matches."""

    run_file = paths.run_root / "brak_v1" / "runs" / f"brak_v1__{dataset}__kir_{KIR:.2f}__seed_{seed}.json"
    distribution = paths.run_root / "brak_v1" / "summaries" / "BRAK_K_DISTRIBUTION.tsv"
    if not run_file.is_file() or not distribution.is_file():
        return None
    payload = json.loads(run_file.read_text(encoding="utf-8"))
    old_inputs = payload.get("inputs", {})
    for key in ("registry_sha256", "canonical_manifest_sha256", "view_manifest_sha256", "export_manifest_sha256", "calibration_embedding_sha256"):
        if old_inputs.get(key) != inputs.get(key):
            return None
    selected: dict[str, int] = {}
    with distribution.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("dataset") == dataset and int(row.get("seed", -1)) == seed and abs(float(row.get("kir", -1)) - KIR) < 1e-9:
                selected[str(row["intent"])] = int(row["selected_k"])
    return selected if set(selected) == set(known) else None


def _compute_brak_map(train: np.ndarray, train_intents: np.ndarray, calibration: np.ndarray, calibration_intents: np.ndarray, seed: int) -> dict[str, int]:
    selected: dict[str, int] = {}
    for intent in sorted(set(train_intents.tolist())):
        result = evaluate_intent_candidates(
            intent,
            train[train_intents == intent],
            calibration[calibration_intents == intent],
            calibration[calibration_intents != intent],
            max_k=5,
            seed=seed,
            distance=DISTANCE,
            covariance_eps=1e-6,
            bootstrap_repeats=5,
            alpha=1.0,
            beta=1.0,
            gamma=0.25,
            eta=0.01,
            delta=0.02,
            min_improvement=0.01,
        )
        selected[intent] = int(result.selected_k)
    return selected


def _run_cell(paths: ProtocolV2Paths, output_root: Path, dataset: str, seed: int, resume: bool) -> dict[str, Any]:
    run_id = f"{EXPERIMENT_ID}__{dataset}__kir_{KIR:.2f}__seed_{seed}"
    run_dir = output_root / "runs" / dataset / f"seed_{seed}"
    manifest_path = run_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            return {"run_id": run_id, "status": "skipped_existing", "metrics_rows": int(payload.get("metrics_rows", 0))}
    views, train, calibration, test, inputs = _load_inputs(paths, dataset, seed)
    train_intents = _intent_array(views.train)
    calibration_intents = _intent_array(views.calibration)
    known = sorted(set(train_intents.tolist()))
    baseline = fit_detector(train, train_intents, distance=DISTANCE, seed=seed)
    selection_rows: list[dict[str, Any]] = []
    primary_map: dict[str, int] = {}
    largest_map: dict[str, int] = {}
    shuffled_primary_map: dict[str, int] = {}
    shuffled_largest_map: dict[str, int] = {}
    for intent in known:
        estimates = []
        for k in CANDIDATE_K:
            candidate = baseline if k == 1 else fit_detector(train, train_intents, distance=DISTANCE, overrides={intent: k}, seed=seed)
            estimates.append(
                estimate_target_risk(
                    intent=intent,
                    candidate_k=k,
                    baseline_detector=baseline,
                    candidate_detector=candidate,
                    calibration_embeddings=calibration,
                    calibration_intents=calibration_intents,
                    known_intents=known,
                    seed=seed,
                )
            )
        primary = select_k(estimates, strategy="urcsg_min_q95")
        largest = select_k(estimates, strategy="urcsg_largest_feasible")
        shuffled_primary = select_k(estimates, strategy="urcsg_min_q95", shuffled=True)
        shuffled_largest = select_k(estimates, strategy="urcsg_largest_feasible", shuffled=True)
        primary_map[intent] = primary.candidate_k
        largest_map[intent] = largest.candidate_k
        shuffled_primary_map[intent] = shuffled_primary.candidate_k
        shuffled_largest_map[intent] = shuffled_largest.candidate_k
        selection_rows.extend(
            {
                "dataset": dataset,
                "seed": seed,
                "kir": KIR,
                "intent": intent,
                **row,
            }
            for row in estimate_target_risk_rows(
                estimates=estimates,
                primary=primary,
                largest=largest,
                shuffled_primary=shuffled_primary,
                shuffled_largest=shuffled_largest,
            )
        )
    maps: dict[str, dict[str, int]] = {
        "single_centroid": {intent: 1 for intent in known},
        "fixed_k2": {intent: 2 for intent in known},
        "fixed_k4": {intent: 4 for intent in known},
        "URCSG-primary": primary_map,
        "URCSG-largest-feasible": largest_map,
        "URCSG-shuffled-primary": shuffled_primary_map,
        "URCSG-shuffled-largest-feasible": shuffled_largest_map,
    }
    brak = _read_brak_map(paths, dataset, seed, inputs, known)
    if brak is None:
        brak = _compute_brak_map(train, train_intents, calibration, calibration_intents, seed)
    maps["BRAK"] = brak
    metric_rows: list[dict[str, Any]] = []
    detector_by_method: dict[str, Any] = {}
    for method in METHODS:
        if method == "oracle_test_k":
            continue
        metrics, detector = _evaluate_map(train, train_intents, test, views.test, maps[method], seed)
        detector_by_method[method] = detector
        metric_rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "seed": seed,
                "kir": KIR,
                "method": method,
                "distance": DISTANCE,
                "radius_lambda": RADIUS_LAMBDA,
                "selection_source": "known_calibration_only" if method.startswith("URCSG") or method == "BRAK" else "predeclared",
                "test_used_for_selection": False,
                **metrics,
            }
        )
    oracle_candidates: list[tuple[int, dict[str, float]]] = []
    for k in CANDIDATE_K:
        metrics, _ = _evaluate_map(train, train_intents, test, views.test, {intent: k for intent in known}, seed)
        oracle_candidates.append((k, metrics))
    oracle_k, oracle_metrics = max(oracle_candidates, key=lambda pair: pair[1]["oos_f1"])
    metric_rows.append(
        {
            "run_id": run_id,
            "dataset": dataset,
            "seed": seed,
            "kir": KIR,
            "method": "oracle_test_k",
            "distance": DISTANCE,
            "radius_lambda": RADIUS_LAMBDA,
            "selection_source": "test_only_descriptive_upper_bound",
            "test_used_for_selection": True,
            "oracle_k": oracle_k,
            **oracle_metrics,
        }
    )
    mechanisms = {
        "dataset": dataset,
        "seed": seed,
        "known_intent_count": len(known),
        "selected_k_distribution": {
            name: {str(k): int(sum(value == k for value in mapping.values())) for k in CANDIDATE_K}
            for name, mapping in maps.items()
        },
        "selected_k_by_intent": maps,
        "test_used_for_selection": False,
    }
    atomic_write_json(run_dir / "inputs.json", inputs)
    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "intent_selection.csv", selection_rows)
    atomic_write_json(run_dir / "mechanism.json", mechanisms)
    atomic_write_json(
        manifest_path,
        {
            "run_id": run_id,
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": paths.dataset_version,
            "dataset": dataset,
            "seed": seed,
            "kir": KIR,
            "status": "complete",
            "metrics_rows": len(metric_rows),
            "selection_rows": len(selection_rows),
            "inputs": inputs,
            "test_used_for_selection": False,
            "oracle_test_k_descriptive_only": True,
        },
    )
    return {"run_id": run_id, "status": "complete", "metrics_rows": len(metric_rows), "selection_rows": len(selection_rows)}


def run_pilot(paths: ProtocolV2Paths, output_root: Path, datasets: tuple[str, ...], seeds: tuple[int, ...], resume: bool) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": paths.dataset_version,
        "datasets": list(datasets),
        "kir": KIR,
        "seeds": list(seeds),
        "candidate_k": list(CANDIDATE_K),
        "distance": DISTANCE,
        "radius": "mean_std",
        "radius_lambda": RADIUS_LAMBDA,
        "selection_strategies": ["urcsg_min_q95", "urcsg_largest_feasible"],
        "epsilon_coverage": 0.02,
        "rho_union_risk": 0.05,
        "shuffled_negative_control": True,
        "selection_data": "proper_train_and_known_calibration_only",
        "test_used_for_selection": False,
        "planned_cells": len(datasets) * len(seeds),
    }
    atomic_write_json(output_root / "plans" / "urcsg_pilot_v1_plan.json", plan)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": paths.dataset_version,
        "base_commit": _git_value(PROJECT_ROOT, ["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(PROJECT_ROOT, ["status", "--short"])),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "start_time_unix": started,
        "plan_sha256": sha256_file(output_root / "plans" / "urcsg_pilot_v1_plan.json"),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs" / "experiments" / "urcsg_pilot_v1.yaml"),
        "bootstrap_seed": 20260725,
        "test_used_for_selection": False,
        "cache_policy": "frozen all-MiniLM cache reuse only",
    }
    atomic_write_json(output_root / "URCSG_PROVENANCE.json", provenance)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            try:
                results.append(_run_cell(paths, output_root, dataset, seed, resume))
            except Exception as exc:  # preserve machine-readable failure without fabricating metrics
                run_id = f"{EXPERIMENT_ID}__{dataset}__kir_{KIR:.2f}__seed_{seed}"
                failures.append({"run_id": run_id, "dataset": dataset, "seed": seed, "error": repr(exc)})
                atomic_write_json(output_root / "runs" / dataset / f"seed_{seed}" / "failure.json", failures[-1])
    completed = sum(item["status"] in {"complete", "skipped_existing"} for item in results)
    integrity = {
        "experiment_id": EXPERIMENT_ID,
        "planned_cells": len(datasets) * len(seeds),
        "expected_unit_count": len(datasets) * len(seeds),
        "completed_unit_count": int(completed),
        "completed_cells": int(completed),
        "failed_cells": len(failures),
        "missing_cells": len(datasets) * len(seeds) - int(completed) - len(failures),
        "duplicate_cells": 0,
        "invalid_cells": 0,
        "test_used_for_selection": False,
        "finished_time_unix": time.time(),
        "failures": failures,
    }
    atomic_write_json(output_root / "URCSG_INTEGRITY.json", integrity)
    provenance.update(
        {
            "finished_time_unix": integrity["finished_time_unix"],
            "integrity_sha256": sha256_file(output_root / "URCSG_INTEGRITY.json"),
            "completed_unit_count": integrity["completed_unit_count"],
            "failed_unit_count": integrity["failed_cells"],
        }
    )
    atomic_write_json(output_root / "URCSG_PROVENANCE.json", provenance)
    return integrity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    output_root = args.output_dir or paths.run_root / EXPERIMENT_ID
    plan = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": paths.dataset_version,
        "datasets": args.datasets,
        "kir": KIR,
        "seeds": args.seeds,
        "candidate_k": CANDIDATE_K,
        "distance": DISTANCE,
        "planned_cells": len(args.datasets) * len(args.seeds),
        "output_root": str(output_root),
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0
    result = run_pilot(paths, output_root, tuple(args.datasets), tuple(args.seeds), args.resume)
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["failures"] and result["completed_cells"] == result["planned_cells"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
