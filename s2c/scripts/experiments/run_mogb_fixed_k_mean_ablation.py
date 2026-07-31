"""Run fixed per-intent KMeans with the frozen MOGB mean-radius boundary.

This stage changes only the number of per-intent KMeans centers. It reuses the
protocol_v2 registry, frozen MiniLM cache, L2 normalization, Euclidean distance,
mean-distance radius, nearest-ball inference, and evaluator used by the existing
``ours_partition_mogb_boundary`` reference. K=2 is therefore reused rather than
rerun in the formal sweep.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_json
from protocol_v2.experiments.mogb import MOGBBoundary, score_mogb_boundaries
from protocol_v2.experiments.partitions import normalize_for_detector
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from protocol_v2.runtime.paths import ProtocolV2Paths
from protocol_v2.tracking.run_manifest import atomic_run_directory, environment_snapshot

try:  # package import for tests
    from .run_mogb_fair import _load_cached_inputs, _open_metrics, _write_predictions
except ImportError:  # direct script execution
    from run_mogb_fair import _load_cached_inputs, _open_metrics, _write_predictions  # type: ignore[no-redef]


LoadedInputs = tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]
PARTITION_NAME = "per_intent_kmeans"


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return config


def run_fixed_k_mean(
    train: np.ndarray,
    train_rows: list[dict[str, Any]],
    test: np.ndarray,
    *,
    k: int,
    partition_seed: int = 42,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Fit fixed per-intent KMeans and score with MOGB's mean-radius rule."""

    if k < 1:
        raise ValueError("k must be positive")
    intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    detector = MultiSphereOOSDetector(
        radius_method="mean_std",
        radius_lambda=1.0,
        center_mode="class_centroid_mixture",
        distance_metric="euclidean",
        l2_normalize=True,
        subcenters_per_intent=int(k),
        random_state=int(partition_seed),
    )
    detector.fit(train, intents)
    boundaries: list[MOGBBoundary] = []
    ball_rows: list[dict[str, Any]] = []
    for sphere in detector.spheres:
        indices = np.flatnonzero(detector._train_cluster_labels == sphere.cluster_id)
        points = detector._train_embeddings[indices]
        distances = np.linalg.norm(points - sphere.center, axis=1)
        radius = max(float(distances.mean()), 1e-12)
        boundaries.append(
            MOGBBoundary(
                int(sphere.cluster_id),
                np.asarray(sphere.center, dtype=np.float64),
                radius,
                str(sphere.intent_name),
                indices,
            )
        )
        ball_rows.append(
            {
                "ball_id": int(sphere.cluster_id),
                "parent_id": None,
                "depth": 0,
                "majority_label": str(sphere.intent_name),
                "purity": 1.0,
                "sample_count": int(indices.size),
                "radius": radius,
                "selected": True,
                # Keep the existing K=2 reference metadata contract byte-for-byte
                # compatible; the explicit K remains in method_details/manifest.
                "stop_reason": "ours_fixed_k",
            }
        )
    output = score_mogb_boundaries(
        normalize_for_detector(test), boundaries, distance="euclidean"
    )
    return output, ball_rows, {
        "partition": PARTITION_NAME,
        "partition_seed": int(partition_seed),
        "k": int(k),
        "distance": "euclidean",
        "boundary": "mean",
        "acceptance": "nearest_ball",
        "cluster_count": len(boundaries),
    }


def _run_dir(root: Path, dataset: str, kir: float, seed: int, k: int) -> Path:
    return root / f"fixed_k{k}" / dataset / f"kir_{kir:.2f}" / f"seed_{seed}"


def _write_run(
    paths: ProtocolV2Paths,
    config: dict[str, Any],
    loaded: LoadedInputs,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    dataset = str(config["dataset"])
    kir = float(config["kir"])
    seed = int(config["seed"])
    k = int(config["k"])
    partition_seed = int(config.get("partition_seed", 42))
    root = output_dir or paths.run_root / "mogb_fixed_k_mean_ablation_v1"
    run_dir = _run_dir(root, dataset, kir, seed, k)
    if (run_dir / "manifest.json").is_file() and not overwrite:
        return run_dir
    if run_dir.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite incomplete fixed-K MOGB run: {run_dir}")
    if overwrite and run_dir.exists():
        if root not in run_dir.parents:
            raise RuntimeError(f"Refusing overwrite outside fixed-K MOGB root: {run_dir}")
        shutil.rmtree(run_dir)

    views, train, test, inputs = loaded
    resolved = {
        **config,
        "protocol_version": paths.dataset_version,
        "representation": "frozen_minilm",
        "partition": PARTITION_NAME,
        "partition_seed": partition_seed,
        "distance": "euclidean",
        "boundary": "mean",
        "acceptance": "nearest_ball",
        "test_used_for_selection": False,
    }
    started = time.perf_counter()
    output, ball_rows, details = run_fixed_k_mean(
        train, views.train, test, k=k, partition_seed=partition_seed
    )
    elapsed = time.perf_counter() - started
    metrics = _open_metrics(views.test, output)
    metrics.update(
        {
            "scoring_seconds": elapsed,
            "samples_per_second": len(test) / max(elapsed, 1e-12),
            "effective_cluster_count": int(details["cluster_count"]),
            "minimum_cluster_size": min(int(row["sample_count"]) for row in ball_rows),
        }
    )
    oos_mask = np.asarray([int(row["label"]) == 1 for row in views.test], dtype=bool)
    for ball in ball_rows:
        assigned = output["nearest_ball"] == int(ball["ball_id"])
        ball["test_oos_assignment_count"] = int(np.sum(assigned & oos_mask))
        ball["test_oos_false_accept_count"] = int(
            np.sum(assigned & oos_mask & (output["predicted_oos"] == 0))
        )
        ball["test_known_false_reject_count"] = int(
            np.sum(assigned & ~oos_mask & (output["predicted_oos"] == 1))
        )

    config_hash = sha256_json(resolved)
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_json(temporary / "config.json", resolved)
        atomic_write_json(temporary / "metrics.json", metrics)
        atomic_write_json(temporary / "inputs.json", inputs)
        atomic_write_json(temporary / "method_details.json", details)
        atomic_write_json(
            temporary / "ball_statistics.json",
            {
                "selected_balls": len(ball_rows),
                "total_balls": len(ball_rows),
                "mean_samples_per_ball": float(
                    np.mean([row["sample_count"] for row in ball_rows])
                ),
                "mean_radius": float(np.mean([row["radius"] for row in ball_rows])),
                "balls_per_intent": {
                    label: int(sum(row["majority_label"] == label for row in ball_rows))
                    for label in sorted({str(row["majority_label"]) for row in ball_rows})
                },
            },
        )
        atomic_write_text(
            temporary / "balls.jsonl",
            "\n".join(json.dumps(row, sort_keys=True) for row in ball_rows) + "\n",
        )
        _write_predictions(
            temporary / "predictions.tsv",
            views.test,
            output,
            f"fixed_k{k}_mogb_mean",
            dataset,
            kir,
            seed,
        )
        atomic_write_json(temporary / "environment.json", environment_snapshot(paths.project_root))
        atomic_write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "stage": "mogb_fixed_k_mean_ablation_v1",
                "run_id": f"mogb_fixed_k_mean__k{k}__{dataset}__kir_{kir:.2f}__seed_{seed}",
                "config_hash": config_hash,
                "input_hashes": inputs,
                "k": k,
                "partition_seed": partition_seed,
                "test_used_for_selection": False,
                "elapsed_seconds": elapsed,
                "source": "fixed-K partition with frozen MOGB mean-radius boundary",
            },
        )
    return run_dir


def run_one_loaded(
    paths: ProtocolV2Paths,
    config: dict[str, Any],
    loaded: LoadedInputs,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    return _write_run(paths, config, loaded, output_dir=output_dir, overwrite=overwrite)


def run_one(
    paths: ProtocolV2Paths,
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    loaded = _load_cached_inputs(
        paths, str(config["dataset"]), int(config["seed"]), float(config["kir"])
    )
    return _write_run(paths, config, loaded, output_dir=output_dir, overwrite=overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baselines/mogb_fixed_k_mean_ablation_v1.yaml"),
    )
    parser.add_argument("--dataset", default="stackoverflow")
    parser.add_argument("--kir", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProtocolV2Paths.discover()
    base = load_config(args.config)
    new_k_values = tuple(int(value) for value in base.get("new_k_values", ()))
    if not new_k_values or args.k not in new_k_values:
        raise ValueError(
            f"K={args.k} is not registered in {args.config}: {new_k_values}"
        )
    config = {
        **base,
        "dataset": args.dataset,
        "kir": float(args.kir),
        "seed": int(args.seed),
        "k": int(args.k),
    }
    if args.dry_run:
        loaded = _load_cached_inputs(paths, args.dataset, args.seed, args.kir)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "dataset": args.dataset,
                    "kir": args.kir,
                    "seed": args.seed,
                    "k": args.k,
                    "train_rows": len(loaded[1]),
                    "test_rows": len(loaded[2]),
                    "test_used_for_selection": False,
                },
                sort_keys=True,
            )
        )
        return 0
    result = run_one(paths, config, output_dir=args.output_dir, overwrite=args.overwrite)
    print(json.dumps({"status": "complete", "run_dir": str(result)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
