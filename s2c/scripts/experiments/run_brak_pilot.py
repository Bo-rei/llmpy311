"""Run the registered StackOverflow BRAK known-only pilot.

The pilot uses the existing protocol_v2 canonical embedding cache and views.
It evaluates Fixed K=1..5 and a per-intent Boundary-Risk-Aware Adaptive K
selection on the same test set, but all BRAK decisions are made from proper
train and Known calibration rows.  No test score is read before the final
evaluation table is written.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402
from protocol_v2.data.manifests import dataset_manifest_path, read_json, view_manifest_path  # noqa: E402
from protocol_v2.data.registry import registry_path  # noqa: E402
from protocol_v2.experiments.brak import (  # noqa: E402
    BRAKSelection,
    evaluate_intent_candidates,
    selected_partition,
    selection_rows,
)
from protocol_v2.experiments.matrix import GateRunSpec  # noqa: E402
from protocol_v2.experiments.partitions import (  # noqa: E402
    PartitionResult,
    build_partition,
    fit_injected_detector,
    normalize_for_detector,
)
from protocol_v2.experiments.runner import (  # noqa: E402
    _canonical_embedding_cache,
    _embedding_cache,
    _model_fingerprint,
    _model_path,
)
from scripts.experiments.run_mogb_fair import _load_cached_inputs, _open_metrics  # noqa: E402
from protocol_v2.runtime.paths import ProtocolV2Paths  # noqa: E402
from protocol_v2.tracking.provenance import file_hashes  # noqa: E402


DEFAULT_SEEDS = (42, 87, 100)
DEFAULT_KS = (1, 2, 3, 4, 5)


def _git_value(project_root: Path, args: list[str], fallback: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def _load_inputs(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    views, train, test, base_inputs = _load_cached_inputs(paths, dataset, seed, kir)
    registry = read_json(registry_path(paths, dataset, seed, kir))
    model_path = _model_path(paths, "all-MiniLM-L6-v2")
    model = _model_fingerprint(model_path)
    canonical = _canonical_embedding_cache(paths, dataset, model, None, 128)
    spec = GateRunSpec(
        experiment_name="brak_v1",
        dataset=dataset,
        kir=kir,
        seed=seed,
        k_gate=1,
        distance="mahalanobis_diag",
        representation="frozen_minilm",
        boundary="mean_std",
        radius_lambda=1.0,
        encoder_name="all-MiniLM-L6-v2",
        encoder_device="cpu",
        protocol_version=paths.dataset_version,
    )
    canonical_sha = str(base_inputs["canonical_manifest_sha256"])
    input_hashes = file_hashes(
        {
            "registry": registry_path(paths, dataset, seed, kir),
            "canonical_manifest": dataset_manifest_path(paths.manifest_root, dataset),
            "view_manifest": view_manifest_path(paths.manifest_root, dataset, seed, kir),
            "s2c_export_manifest": views.export_root / "export_manifest.json",
        }
    )
    calibration, calibration_meta = _embedding_cache(
        paths,
        spec,
        "calibration_known",
        views.calibration,
        registry["registry_sha256"],
        canonical_sha,
        model,
        canonical,
    )
    inputs = {
        **base_inputs,
        "calibration_embedding_sha256": calibration_meta.get("embedding_sha256"),
        "calibration_sample_ids_sha256": calibration_meta.get("sample_ids_sha256"),
        "input_file_hashes": input_hashes,
        "cache_policy": "reuse_only; no implicit encoding",
    }
    return views, normalize_for_detector(train), normalize_for_detector(calibration), normalize_for_detector(test), inputs


def _evaluate_detector(detector: Any, test: np.ndarray, test_rows: list[dict[str, Any]]) -> dict[str, float]:
    output = detector.predict_with_scores(test)
    nearest_labels = np.asarray(
        [detector.spheres[int(index)].intent_name for index in output["nearest_cluster"]], dtype=object
    )
    predicted_labels = nearest_labels.copy()
    predicted_labels[output["pred"].astype(bool)] = "oos"
    metrics = _open_metrics(
        test_rows,
        {
            "score": np.asarray(output["score"], dtype=np.float64),
            "predicted_label": predicted_labels,
        },
    )
    metrics.update(
        {
            "effective_cluster_count": int(len(detector.spheres)),
            "minimum_cluster_size": int(
                min(
                    np.sum(detector._train_cluster_labels == int(sphere.cluster_id))
                    for sphere in detector.spheres
                )
            ),
        }
    )
    return metrics


def _fixed_detector(train: np.ndarray, train_rows: list[dict[str, Any]], k: int, seed: int) -> Any:
    intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    partition = build_partition(train, intents, k, "kmeans", seed)
    return fit_injected_detector(
        train,
        intents,
        partition,
        distance="mahalanobis_diag",
        radius_lambda=1.0,
        random_state=seed,
        acceptance_mode="nearest_sphere",
    )


def _selection_for_dataset(
    train: np.ndarray,
    train_rows: list[dict[str, Any]],
    calibration: np.ndarray,
    calibration_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, BRAKSelection], list[dict[str, Any]]]:
    train_intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    calibration_intents = np.asarray([str(row["intent"]) for row in calibration_rows], dtype=object)
    selections: dict[str, BRAKSelection] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    for intent in sorted(set(train_intents.tolist())):
        proper_points = train[train_intents == intent]
        target = calibration[calibration_intents == intent]
        other = calibration[calibration_intents != intent]
        selection = evaluate_intent_candidates(
            intent,
            proper_points,
            target,
            other,
            max_k=5,
            seed=seed,
            distance="mahalanobis_diag",
            covariance_eps=1e-6,
            bootstrap_repeats=5,
            alpha=1.0,
            beta=1.0,
            gamma=0.25,
            eta=0.01,
            delta=0.02,
            min_improvement=0.01,
        )
        selections[intent] = selection
        diagnostic_rows.extend(selection_rows(selection))
    return selections, diagnostic_rows


def _selected_detector(train: np.ndarray, train_rows: list[dict[str, Any]], selections: dict[str, BRAKSelection], seed: int) -> Any:
    intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    labels, centers, cluster_to_intent, intent_to_clusters = selected_partition(train, intents, selections)
    partition = PartitionResult(
        labels=labels,
        centers=centers,
        cluster_to_intent=cluster_to_intent,
        intent_to_clusters=intent_to_clusters,
    )
    return fit_injected_detector(
        train,
        intents,
        partition,
        distance="mahalanobis_diag",
        radius_lambda=1.0,
        random_state=seed,
        acceptance_mode="nearest_sphere",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "\n")
        return
    fields = sorted({key for row in rows for key in row})
    output = ["\t".join(fields)]
    for row in rows:
        output.append("\t".join(str(row.get(field, "")) for field in fields))
    atomic_write_text(path, "\n".join(output) + "\n")


def run_pilot(paths: ProtocolV2Paths, output_root: Path, datasets: tuple[str, ...], kir: float, seeds: tuple[int, ...]) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "experiment_id": "brak_v1_pilot",
        "protocol_version": paths.dataset_version,
        "datasets": list(datasets),
        "kir": float(kir),
        "seeds": list(seeds),
        "candidate_k": list(DEFAULT_KS),
        "distance": "mahalanobis_diag",
        "radius": "mean_std_lambda_1.0",
        "selection_data": "proper_train_and_known_calibration_only",
        "test_used_for_selection": False,
        "delta": 0.02,
        "min_improvement": 0.01,
    }
    atomic_write_json(output_root / "plans" / "brak_v1_plan.json", plan)
    summary_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            run_id = f"brak_v1__{dataset}__kir_{kir:.2f}__seed_{seed}"
            try:
                views, train, calibration, test, inputs = _load_inputs(paths, dataset, seed, kir)
                selections, selection_diag = _selection_for_dataset(train, views.train, calibration, views.calibration, seed)
                diagnostics.extend(
                    {
                        **row,
                        "run_id": run_id,
                        "dataset": dataset,
                        "kir": kir,
                        "seed": seed,
                    }
                    for row in selection_diag
                )
                fixed_metrics: dict[int, dict[str, float]] = {}
                for k in DEFAULT_KS:
                    detector = _fixed_detector(train, views.train, k, seed)
                    metrics = _evaluate_detector(detector, test, views.test)
                    fixed_metrics[k] = metrics
                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "method": f"fixed_k{k}",
                            "selection_source": "predeclared_fixed_k",
                            **metrics,
                        }
                    )
                adaptive_detector = _selected_detector(train, views.train, selections, seed)
                adaptive_metrics = _evaluate_detector(adaptive_detector, test, views.test)
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "kir": kir,
                        "seed": seed,
                        "method": "brak",
                        "selection_source": "proper_train_and_known_calibration_only",
                        "selected_k_mean": float(np.mean([selection.selected_k for selection in selections.values()])),
                        "selected_k_median": float(np.median([selection.selected_k for selection in selections.values()])),
                        **adaptive_metrics,
                    }
                )
                oracle_k = max(DEFAULT_KS, key=lambda candidate_k: fixed_metrics[candidate_k]["oos_f1"])
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": dataset,
                        "kir": kir,
                        "seed": seed,
                        "method": "oracle_test_best_k_upper_bound",
                        "selection_source": "test_only_analysis_upper_bound",
                        "oracle_k": oracle_k,
                        **fixed_metrics[oracle_k],
                    }
                )
                for intent, selection in selections.items():
                    distribution_rows.append(
                        {
                            "run_id": run_id,
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "intent": intent,
                            "selected_k": selection.selected_k,
                        }
                    )
                run_payload = {
                    "run_id": run_id,
                    "status": "complete",
                    "config": plan,
                    "inputs": inputs,
                    "selection_count": len(selections),
                    "selected_k_distribution": {
                        str(k): int(sum(selection.selected_k == k for selection in selections.values()))
                        for k in DEFAULT_KS
                    },
                    "test_used_for_selection": False,
                }
                atomic_write_json(output_root / "runs" / f"{run_id}.json", run_payload)
            except Exception as exc:  # preserve a machine-readable failure without fabricating a metric row
                failures.append({"run_id": run_id, "dataset": dataset, "kir": kir, "seed": seed, "error": repr(exc)})
    _write_csv(output_root / "summaries" / "BRAK_PILOT_SUMMARY.tsv", summary_rows)
    _write_csv(output_root / "summaries" / "BRAK_CANDIDATE_DIAGNOSTICS.tsv", diagnostics)
    _write_csv(output_root / "summaries" / "BRAK_K_DISTRIBUTION.tsv", distribution_rows)
    _write_csv(output_root / "summaries" / "BRAK_FAILED_RUNS.tsv", failures)
    provenance = {
        "experiment_id": "brak_v1_pilot",
        "protocol_version": paths.dataset_version,
        "base_commit": _git_value(paths.project_root, ["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(paths.project_root, ["status", "--short"])),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "plan_sha256": sha256_file(output_root / "plans" / "brak_v1_plan.json"),
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "test_used_for_selection": False,
        "selection_contract": "proper_train_and_known_calibration_only",
    }
    atomic_write_json(output_root / "BRAK_PROVENANCE.json", provenance)
    closeout = {
        "planned_cells": len(datasets) * len(seeds) * 7,
        "completed_cells": len(summary_rows),
        "failed_runs": len(failures),
        "selection_runs": len(datasets) * len(seeds) - len(failures),
        "note": "summary cell count includes fixed K=1..5, BRAK, and oracle test-best-K upper-bound rows",
    }
    atomic_write_json(output_root / "BRAK_INTEGRITY.json", closeout)
    return {**closeout, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["stackoverflow"])
    parser.add_argument("--kir", type=float, default=0.50)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    paths.require_experiment_admission()
    output_root = args.output_dir or paths.run_root / "brak_v1"
    if args.dry_run:
        print(json.dumps({"status": "ready", "datasets": args.datasets, "kir": args.kir, "seeds": args.seeds, "output_root": str(output_root)}, sort_keys=True))
        return 0
    result = run_pilot(paths, output_root, tuple(args.datasets), args.kir, tuple(args.seeds))
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
