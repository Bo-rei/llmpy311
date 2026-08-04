#!/usr/bin/env python3
"""Run the registered CCSG Known-only scoring pilot.

The pilot reuses the frozen MiniLM cache and the active centre/radius fit.  It
does not train an encoder, rebuild a registry, or inspect test labels while
calibrating.  Each dataset/seed cell fits K=1 and K=2 once and evaluates the
registered support and margin ablations from the same fitted objects.
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
from protocol_v2.experiments.ccsg import (  # noqa: E402
    DEFAULT_TARGET_FALSE_REJECTION,
    apply_calibration,
    calibrate,
    open_metrics,
    support_margin_features,
)
from protocol_v2.experiments.matrix import GateRunSpec  # noqa: E402
from protocol_v2.experiments.runner import (  # noqa: E402
    _canonical_embedding_cache,
    _embedding_cache,
    _model_fingerprint,
    _model_path,
)
from protocol_v2.experiments.urcsg import fit_detector  # noqa: E402
from protocol_v2.gate.view_loader import load_gate_views  # noqa: E402
from protocol_v2.runtime.paths import ProtocolV2Paths  # noqa: E402
from protocol_v2.tracking.provenance import file_hashes  # noqa: E402


EXPERIMENT_ID = "ccsg_pilot_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87)
KIR = 0.50
DISTANCE = "mahalanobis_diag"
RADIUS_LAMBDA = 1.0
TEMPERATURE = 1.0
TARGET_FALSE_REJECTION = DEFAULT_TARGET_FALSE_REJECTION

# The first four are the contract baselines/controls; the remaining four
# isolate support aggregation, margin, and the joint CCSG decision.
METHODS: tuple[tuple[str, int, str], ...] = (
    ("current_k1", 1, "current_k1"),
    ("current_k2_union", 2, "current_k2_union"),
    ("mixture_support_k1", 1, "mixture_support"),
    ("mixture_support_k2", 2, "mixture_support"),
    ("margin_only_k1", 1, "margin_only"),
    ("ccsg_k1", 1, "ccsg"),
    ("ccsg_k2", 2, "ccsg"),
    ("ccsg_independent_k2", 2, "ccsg_independent"),
)


def _git_value(args: list[str], fallback: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "\n")
        return
    fields = sorted({key for row in rows for key in row})
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
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
    inputs = file_hashes(
        {
            "registry": registry_file,
            "canonical_manifest": canonical_file,
            "view_manifest": view_file,
            "s2c_export_manifest": views.export_root / "export_manifest.json",
        }
    )
    model = _model_fingerprint(_model_path(paths, "all-MiniLM-L6-v2"))
    # The pilot must reuse an existing cache; passing no encoder makes a cache
    # miss fail instead of silently encoding a new representation.
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
    train, train_meta = _embedding_cache(paths, spec, "train_known", views.train, registry["registry_sha256"], inputs["canonical_manifest"], model, canonical)
    calibration, calibration_meta = _embedding_cache(paths, spec, "calibration_known", views.calibration, registry["registry_sha256"], inputs["canonical_manifest"], model, canonical)
    test, test_meta = _embedding_cache(paths, spec, "test_combined", views.test, registry["registry_sha256"], inputs["canonical_manifest"], model, canonical)
    split_ids = {"train": _ids_sha(views.train), "calibration": _ids_sha(views.calibration), "test": _ids_sha(views.test)}
    if len(set(split_ids.values())) != 3:
        raise ValueError(f"Split sample-id hashes unexpectedly collide: {dataset}/seed_{seed}")
    calibration_ids = {str(row["sample_id"]) for row in views.calibration}
    test_ids = {str(row["sample_id"]) for row in views.test}
    if calibration_ids & test_ids:
        raise ValueError(f"Calibration/test sample overlap: {dataset}/seed_{seed}")
    return views, train, calibration, test, {
        "registry_sha256": registry["registry_sha256"],
        "canonical_manifest_sha256": inputs["canonical_manifest"],
        "view_manifest_sha256": inputs["view_manifest"],
        "export_manifest_sha256": inputs["s2c_export_manifest"],
        "input_file_hashes": inputs,
        "model": model,
        "cache_embedding_sha256": {
            "train": train_meta.get("embedding_sha256"),
            "calibration": calibration_meta.get("embedding_sha256"),
            "test": test_meta.get("embedding_sha256"),
        },
        "sample_ids_sha256": split_ids,
        "calibration_ids_disjoint_from_test": True,
        "test_used_for_selection": False,
        "cache_policy": "reuse_only_no_implicit_encoding",
    }


def _intent_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([str(row["intent"]) for row in rows], dtype=object)


def _run_cell(paths: ProtocolV2Paths, output_root: Path, dataset: str, seed: int, resume: bool) -> dict[str, Any]:
    run_id = f"{EXPERIMENT_ID}__{dataset}__kir_{KIR:.2f}__seed_{seed}"
    run_dir = output_root / "runs" / dataset / f"seed_{seed}"
    manifest_path = run_dir / "run_manifest.json"
    if resume and manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "complete":
        return {"run_id": run_id, "status": "skipped_existing", "metrics_rows": len(list((run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines())) - 1}
    views, train, calibration, test, inputs = _load_inputs(paths, dataset, seed)
    train_intents = _intent_array(views.train)
    detectors = {k: fit_detector(train, train_intents, distance=DISTANCE, overrides=None if k == 1 else {intent: k for intent in sorted(set(train_intents.tolist()))}, seed=seed) for k in (1, 2)}
    calibration_features = {k: support_margin_features(detector, calibration, temperature=TEMPERATURE) for k, detector in detectors.items()}
    test_features = {k: support_margin_features(detector, test, temperature=TEMPERATURE) for k, detector in detectors.items()}
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for method, k, calibration_method in METHODS:
        calibration_fit = calibrate(calibration_method, calibration_features[k], target_false_rejection=TARGET_FALSE_REJECTION)
        output = apply_calibration(test_features[k], calibration_fit)
        metrics = open_metrics(views.test, output)
        metric_rows.append(
            {
                "run_id": run_id,
                "experiment_id": EXPERIMENT_ID,
                "protocol_version": paths.dataset_version,
                "dataset": dataset,
                "kir": KIR,
                "seed": seed,
                "method": method,
                "k": k,
                "distance": DISTANCE,
                "radius_method": "mean_std",
                "radius_lambda": RADIUS_LAMBDA,
                "temperature": TEMPERATURE,
                "target_false_rejection": TARGET_FALSE_REJECTION,
                "threshold_source": "known_calibration_only",
                "test_used_for_selection": False,
                **metrics,
            }
        )
        calibration_rows.append(
            {
                "method": method,
                "k": k,
                "calibration_method": calibration_method,
                "threshold": calibration_fit.threshold,
                "support_threshold": calibration_fit.support_threshold,
                "margin_threshold": calibration_fit.margin_threshold,
                "support_mean": calibration_fit.support_mean,
                "support_std": calibration_fit.support_std,
                "margin_mean": calibration_fit.margin_mean,
                "margin_std": calibration_fit.margin_std,
                "target_false_rejection": TARGET_FALSE_REJECTION,
                "calibration_false_rejection": calibration_fit.calibration_false_rejection,
                "test_used_for_selection": False,
            }
        )
        for index, row in enumerate(views.test):
            prediction_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "method": method,
                    "seed": seed,
                    "gold_is_oos": int(row["label"]),
                    "gold_intent": row["intent"],
                    "oos_source": row.get("oos_source", "known"),
                    "predicted_is_oos": int(output["pred"][index]),
                    "prediction_intent": str(output["prediction_intent"][index]),
                    "runner_up": str(output["runner_up"][index]),
                    "oos_score": float(output["score"][index]),
                    "support": float(output["support"][index]),
                    "margin": float(output["margin"][index]),
                    "raw_score": float(output["raw_score"][index]),
                }
            )
    mechanism_rows = []
    for k, detector in detectors.items():
        for sphere in detector.spheres:
            size = int(np.sum(detector._train_cluster_labels == int(sphere.cluster_id)))
            mechanism_rows.append({"k": k, "cluster_id": int(sphere.cluster_id), "intent": str(sphere.intent_name), "sample_count": size, "radius": float(sphere.radius)})
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "inputs.json", inputs)
    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "calibration.csv", calibration_rows)
    _write_csv(run_dir / "predictions.csv", prediction_rows)
    _write_csv(run_dir / "mechanism.csv", mechanism_rows)
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
            "prediction_rows": len(prediction_rows),
            "inputs": inputs,
            "methods": [item[0] for item in METHODS],
            "test_used_for_selection": False,
        },
    )
    return {"run_id": run_id, "status": "complete", "metrics_rows": len(metric_rows), "prediction_rows": len(prediction_rows)}


def run_pilot(paths: ProtocolV2Paths, output_root: Path, datasets: tuple[str, ...], seeds: tuple[int, ...], resume: bool) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": paths.dataset_version,
        "datasets": list(datasets),
        "kir": KIR,
        "seeds": list(seeds),
        "k_values": [1, 2],
        "distance": DISTANCE,
        "radius_method": "mean_std",
        "radius_lambda": RADIUS_LAMBDA,
        "temperature": TEMPERATURE,
        "target_false_rejection": TARGET_FALSE_REJECTION,
        "methods": [item[0] for item in METHODS],
        "selection_data": "known_calibration_only",
        "test_used_for_selection": False,
        "planned_cells": len(datasets) * len(seeds),
        "planned_metric_rows": len(datasets) * len(seeds) * len(METHODS),
    }
    atomic_write_json(output_root / "plans" / "ccsg_pilot_v1_plan.json", plan)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": paths.dataset_version,
        "base_commit": _git_value(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(["status", "--short"])),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "start_time_unix": started,
        "plan_sha256": sha256_file(output_root / "plans" / "ccsg_pilot_v1_plan.json"),
        "config_sha256": sha256_file(PROJECT_ROOT / "configs" / "experiments" / "ccsg_pilot_v1.yaml"),
        "temperature": TEMPERATURE,
        "target_false_rejection": TARGET_FALSE_REJECTION,
        "test_used_for_selection": False,
        "cache_policy": "frozen all-MiniLM cache reuse only",
    }
    atomic_write_json(output_root / "CCSG_PROVENANCE.json", provenance)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            try:
                results.append(_run_cell(paths, output_root, dataset, seed, resume))
            except Exception as exc:
                run_id = f"{EXPERIMENT_ID}__{dataset}__kir_{KIR:.2f}__seed_{seed}"
                failure = {"run_id": run_id, "dataset": dataset, "seed": seed, "error": repr(exc)}
                failures.append(failure)
                atomic_write_json(output_root / "runs" / dataset / f"seed_{seed}" / "failure.json", failure)
    completed = sum(item["status"] in {"complete", "skipped_existing"} for item in results)
    integrity = {
        "experiment_id": EXPERIMENT_ID,
        "planned_cells": len(datasets) * len(seeds),
        "completed_cells": int(completed),
        "failed_cells": len(failures),
        "missing_cells": len(datasets) * len(seeds) - int(completed) - len(failures),
        "duplicate_cells": 0,
        "invalid_cells": 0,
        "test_used_for_selection": False,
        "finished_time_unix": time.time(),
        "failures": failures,
    }
    atomic_write_json(output_root / "CCSG_INTEGRITY.json", integrity)
    provenance.update(
        {
            "finished_time_unix": integrity["finished_time_unix"],
            "integrity_sha256": sha256_file(output_root / "CCSG_INTEGRITY.json"),
            "completed_cells": integrity["completed_cells"],
            "failed_cells": integrity["failed_cells"],
        }
    )
    atomic_write_json(output_root / "CCSG_PROVENANCE.json", provenance)
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
        "methods": [item[0] for item in METHODS],
        "planned_cells": len(args.datasets) * len(args.seeds),
        "planned_metric_rows": len(args.datasets) * len(args.seeds) * len(METHODS),
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
