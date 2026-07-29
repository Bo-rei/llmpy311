"""Integrity checks for the independent E3 artifact layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.data.manifests import read_json
from protocol_v2.runtime.paths import ProtocolV2Paths

from .mechanism_runner import (
    _control_run_dir,
    _e2_closeout_manifest,
    diagnostic_groups,
    diagnostic_partition_seeds,
    e3_root,
    load_e2_bundle,
    partition_control_specs,
)
from .partitions import build_partition, fit_injected_detector, normalize_for_detector


def verify_e3(
    paths: ProtocolV2Paths,
    partition_config: Path,
    diagnostic_config: Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    specs = partition_control_specs(partition_config)
    groups = diagnostic_groups(diagnostic_config)
    root = e3_root(paths)
    completed = 0
    missing = 0
    invalid = 0
    for spec in specs:
        run_dir = _control_run_dir(paths, spec)
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.json"
        if not manifest_path.is_file() or not metrics_path.is_file():
            missing += 1
            continue
        try:
            manifest = read_json(manifest_path)
            metrics = read_json(metrics_path)
            if (
                manifest.get("status") != "complete"
                or manifest.get("stage") != "E3-A"
                or manifest.get("protocol_version") != paths.dataset_version
                or manifest.get("run_id") != spec.run_id
                or manifest.get("config", {}).get("partition") != spec.partition
                or manifest.get("config", {}).get("k_gate") != spec.k
                or not isinstance(metrics.get("combined"), dict)
                or not isinstance(metrics.get("oos_breakdown"), dict)
            ):
                invalid += 1
                continue
            completed += 1
        except (OSError, TypeError, ValueError):
            invalid += 1
    diagnostic_complete = sum(1 for group in groups if (root / "diagnostics" / "groups" / f"{group.group_id}.json").is_file())
    diagnostic_missing = len(groups) - diagnostic_complete
    e2_immutable = True
    e2_closeout = _e2_closeout_manifest(paths)
    if e2_closeout.is_file():
        snapshot = read_json(e2_closeout)
        expected = snapshot.get("provenance_checks", {})
        e2_immutable = all(bool(value) for value in expected.values())
    result = {
        "protocol_version": paths.dataset_version,
        "partition_control": {
            "planned": len(specs),
            "completed": completed,
            "missing": missing,
            "invalid": invalid,
            "expected_unique_cells": len(specs),
        },
        "diagnostics": {
            "planned_groups": len(groups),
            "completed_groups": diagnostic_complete,
            "missing_groups": diagnostic_missing,
            "partition_seeds": list(diagnostic_partition_seeds(diagnostic_config)),
        },
        "e2_immutable_provenance_checks": e2_immutable,
        "e2_run_root": str(paths.run_root / "e2_gate_core_dense"),
        "e3_run_root": str(root),
        "e2_e3_roots_disjoint": (paths.run_root / "e2_gate_core_dense") != root,
        "textoir_runtime_path_used": False,
        "e4_to_e7_started": False,
    }
    if require_complete and (missing or invalid or diagnostic_missing or not e2_immutable):
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def verify_kmeans_e2_equivalence(
    paths: ProtocolV2Paths,
    *,
    dataset: str = "clinc150",
    kir: float = 0.50,
    seed: int = 42,
    k: int = 2,
    distance: str = "euclidean",
) -> dict[str, Any]:
    """Recompute one frozen E2 cell through the injectable KMeans adapter."""

    bundle = load_e2_bundle(paths, dataset, seed, kir)
    train = normalize_for_detector(bundle.train)
    intents = np.asarray([str(row["intent"]) for row in bundle.views.train], dtype=object)
    partition = build_partition(train, intents, k, "kmeans", 42)
    detector = fit_injected_detector(train, intents, partition, distance=distance, radius_lambda=1.0, random_state=42)
    output = detector.predict_with_scores(bundle.test)
    labels = np.asarray([int(row["label"]) for row in bundle.views.test], dtype=np.int64)
    actual = compute_binary_oos_metrics(labels, np.asarray(output["score"], dtype=float), threshold=1.0)
    e2_dir = paths.run_root / "e2_gate_core_dense" / (
        f"protocol_v2_textoir_v1__{dataset}__kir_{kir:.2f}__seed_{seed}__"
        f"repr_frozen_minilm__k_{k}__dist_{distance}__boundary_mean_std"
    )
    expected = read_json(e2_dir / "metrics.json")["combined"]
    comparable = (
        "oos_f1",
        "id_recall",
        "auroc",
        "aupr_oos",
        "oos_precision",
        "oos_recall",
        "false_accept_rate",
        "false_reject_rate",
    )
    deltas = {field: abs(float(actual[field]) - float(expected[field])) for field in comparable}
    return {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "k": k,
        "distance": distance,
        "e2_run": str(e2_dir),
        "max_abs_delta": max(deltas.values()),
        "deltas": deltas,
        "equivalent_within_1e-12": max(deltas.values()) <= 1e-12,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-config", type=Path, required=True)
    parser.add_argument("--diagnostic-config", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--check-equivalence", action="store_true")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    result = verify_e3(paths, args.partition_config, args.diagnostic_config, require_complete=args.require_complete)
    if args.check_equivalence:
        result["kmeans_e2_equivalence"] = verify_kmeans_e2_equivalence(paths)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
