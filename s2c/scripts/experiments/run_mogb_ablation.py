"""Run protocol-aligned frozen-MiniLM MOGB ablations without touching E2/E3.

This runner reuses the same protocol_v2 split manifests, frozen MiniLM cache,
and evaluation contract as ``run_mogb_fair.py``.  It only varies the granular
ball partition thresholds and the frozen post-hoc boundary adapter.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_json
from protocol_v2.tracking.run_manifest import atomic_run_directory, environment_snapshot

try:  # package import for tests
    from .run_mogb_fair import (
        _load_cached_inputs,
        _open_metrics,
        _write_predictions,
    )
except ImportError:  # direct script execution
    from run_mogb_fair import (  # type: ignore[no-redef]
        _load_cached_inputs,
        _open_metrics,
        _write_predictions,
    )

from protocol_v2.experiments.mogb import (
    AdaptiveGranularBallClusterer,
    balls_to_rows,
    make_mogb_boundaries,
    score_mogb_boundaries,
)
from protocol_v2.experiments.partitions import normalize_for_detector
from protocol_v2.runtime.paths import ProtocolV2Paths


SAFE_SLUG = re.compile(r"[^a-z0-9._-]+")
DEFAULT_CLUSTERER = {
    "purity_train": 0.90,
    "purity_get_ball": 1.00,
    "purity_select_ball": 0.90,
    "min_ball_train": 10,
    "min_ball_get_ball": 5,
    "min_ball_select_ball": 10,
}
LoadedInputs = tuple[Any, Any, Any, dict[str, Any]]


def safe_variant_slug(value: str) -> str:
    slug = SAFE_SLUG.sub("-", value.strip().lower()).strip("-._")
    return slug or "default"


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def _ablation_variant_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "distance": str(config["distance"]),
        "boundary": str(config["boundary"]),
        "purity_train": float(config["purity_train"]),
        "purity_get_ball": float(config["purity_get_ball"]),
        "purity_select_ball": float(config["purity_select_ball"]),
        "min_ball_train": int(config["min_ball_train"]),
        "min_ball_get_ball": int(config["min_ball_get_ball"]),
        "min_ball_select_ball": int(config["min_ball_select_ball"]),
    }


def _clusterer_from_config(config: dict[str, Any], *, seed: int) -> AdaptiveGranularBallClusterer:
    return AdaptiveGranularBallClusterer(
        purity_train=float(config["purity_train"]),
        purity_get_ball=float(config["purity_get_ball"]),
        purity_select_ball=float(config["purity_select_ball"]),
        min_ball_train=int(config["min_ball_train"]),
        min_ball_get_ball=int(config["min_ball_get_ball"]),
        min_ball_select_ball=int(config["min_ball_select_ball"]),
        seed=seed,
    )


def _run_ablation(
    config: dict[str, Any],
    train: Any,
    train_rows: list[dict[str, Any]],
    test: Any,
    test_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    del test_rows
    normalized_train = normalize_for_detector(train)
    normalized_test = normalize_for_detector(test)
    clusterer = _clusterer_from_config(config, seed=seed)
    clusterer.fit(normalized_train, [str(row["intent"]) for row in train_rows])
    boundaries = make_mogb_boundaries(
        clusterer,
        boundary=str(config["boundary"]),
        distance=str(config["distance"]),
    )
    output = score_mogb_boundaries(normalized_test, boundaries, distance=str(config["distance"]))
    return (
        output,
        balls_to_rows(clusterer),
        {
            "partition": "mogb_adaptive",
            "boundary": f"{config['boundary']}_{config['distance']}",
            "cluster_count": len(boundaries),
            "ball_statistics": clusterer.ball_statistics(),
            "ablation_parameters": _ablation_variant_payload(config),
        },
    )


def _materialize_variant_config(
    base: dict[str, Any],
    *,
    dataset: str,
    kir: float,
    seed: int,
    variant: str | None = None,
    distance: str | None = None,
    boundary: str | None = None,
    purity_get_ball: float | None = None,
    purity_select_ball: float | None = None,
    min_ball_get_ball: int | None = None,
    min_ball_select_ball: int | None = None,
) -> dict[str, Any]:
    config = dict(base)
    config["dataset"] = dataset
    config["kir"] = float(kir)
    config["seed"] = int(seed)
    if variant is not None:
        config["output_variant"] = safe_variant_slug(variant)
    else:
        config["output_variant"] = safe_variant_slug(str(base.get("output_variant", "default")))
    if distance is not None:
        config["distance"] = distance
    if boundary is not None:
        config["boundary"] = boundary
    if purity_get_ball is not None:
        config["purity_get_ball"] = float(purity_get_ball)
    if purity_select_ball is not None:
        config["purity_select_ball"] = float(purity_select_ball)
    if min_ball_get_ball is not None:
        config["min_ball_get_ball"] = int(min_ball_get_ball)
    if min_ball_select_ball is not None:
        config["min_ball_select_ball"] = int(min_ball_select_ball)
    config["test_used_for_selection"] = False
    return config


def _run_root(paths: ProtocolV2Paths, output_dir: Path | None, variant: str) -> Path:
    base = output_dir or paths.run_root / "mogb_ablation_v1"
    return base / safe_variant_slug(variant)


def _write_run(
    paths: ProtocolV2Paths,
    config: dict[str, Any],
    views: Any,
    train: Any,
    test: Any,
    inputs: dict[str, Any],
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    dataset = str(config["dataset"])
    kir = float(config["kir"])
    seed = int(config["seed"])
    variant = safe_variant_slug(str(config["output_variant"]))
    root = _run_root(paths, output_dir, variant)
    run_dir = root / dataset / f"kir_{kir:.2f}" / f"seed_{seed}"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        return run_dir
    if run_dir.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite incomplete MOGB ablation run: {run_dir}")
    if overwrite and run_dir.exists():
        if root not in run_dir.parents:
            raise RuntimeError(f"Refusing overwrite outside MOGB ablation artifact root: {run_dir}")
        shutil.rmtree(run_dir)

    ablation_config = dict(config)
    ablation_config["output_variant"] = variant
    started = time.perf_counter()
    output, ball_rows, method_details = _run_ablation(
        ablation_config,
        train,
        views.train,
        test,
        views.test,
        seed=seed,
    )
    metrics = _open_metrics(views.test, output)
    selected_ball_rows = [row for row in ball_rows if bool(row.get("selected", True))]
    elapsed = time.perf_counter() - started
    metrics.update(
        {
            "scoring_seconds": elapsed,
            "samples_per_second": len(test) / max(elapsed, 1e-12),
            "effective_cluster_count": method_details["cluster_count"],
            "minimum_cluster_size": min(int(row["sample_count"]) for row in selected_ball_rows),
        }
    )
    oos_mask = [int(row["label"]) == 1 for row in views.test]
    for ball in ball_rows:
        assigned = output["nearest_ball"] == int(ball["ball_id"])
        ball["test_oos_assignment_count"] = int(sum(flag and assign for flag, assign in zip(oos_mask, assigned, strict=True)))
        ball["test_oos_false_accept_count"] = int(
            sum(flag and assign and int(pred) == 0 for flag, assign, pred in zip(oos_mask, assigned, output["predicted_oos"], strict=True))
        )
        ball["test_known_false_reject_count"] = int(
            sum((not flag) and assign and int(pred) == 1 for flag, assign, pred in zip(oos_mask, assigned, output["predicted_oos"], strict=True))
        )

    config_hash = sha256_json(ablation_config)
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_json(temporary / "config.json", ablation_config)
        atomic_write_json(temporary / "metrics.json", metrics)
        atomic_write_json(
            temporary / "ball_statistics.json",
            method_details.get("ball_statistics", {"selected_balls": len(ball_rows), "total_balls": len(ball_rows)}),
        )
        atomic_write_json(temporary / "inputs.json", inputs)
        atomic_write_json(temporary / "method_details.json", method_details)
        atomic_write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "run_id": f"mogb_ablation__{variant}__{dataset}__kir_{kir:.2f}__seed_{seed}",
                "config_hash": config_hash,
                "input_hashes": inputs,
                "test_used_for_selection": False,
                "elapsed_seconds": elapsed,
                "source": "MOGB frozen-MiniLM ablation integration",
                "variant": variant,
                "ablation_parameters": _ablation_variant_payload(ablation_config),
            },
        )
        atomic_write_json(temporary / "environment.json", environment_snapshot(paths.project_root))
        atomic_write_text(
            temporary / "balls.jsonl",
            "\n".join(json.dumps(row, sort_keys=True) for row in ball_rows) + "\n",
        )
        _write_predictions(temporary / "predictions.tsv", views.test, output, "mogb_ablation", dataset, kir, seed)
    return run_dir


def run_one(
    paths: ProtocolV2Paths,
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    dataset = str(config["dataset"])
    kir = float(config["kir"])
    seed = int(config["seed"])
    views, train, test, inputs = _load_cached_inputs(paths, dataset, seed, kir)
    return _write_run(
        paths,
        config,
        views,
        train,
        test,
        inputs,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def run_one_loaded(
    paths: ProtocolV2Paths,
    config: dict[str, Any],
    loaded: LoadedInputs,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    views, train, test, inputs = loaded
    return _write_run(
        paths,
        config,
        views,
        train,
        test,
        inputs,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/mogb_ablation_v1.yaml"))
    parser.add_argument("--dataset", default="stackoverflow")
    parser.add_argument("--kir", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant")
    parser.add_argument("--distance", choices=("euclidean", "mahalanobis_diag"))
    parser.add_argument("--boundary", choices=("mean", "mean_std"))
    parser.add_argument("--purity-get", dest="purity_get_ball", type=float)
    parser.add_argument("--purity-select", dest="purity_select_ball", type=float)
    parser.add_argument("--min-get", dest="min_ball_get_ball", type=int)
    parser.add_argument("--min-select", dest="min_ball_select_ball", type=int)
    parser.add_argument("--device", default="cpu", help="Recorded only; ablation consumes cached embeddings")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    base = load_config(args.config)
    config = _materialize_variant_config(
        base,
        dataset=args.dataset,
        kir=args.kir,
        seed=args.seed,
        variant=args.variant,
        distance=args.distance,
        boundary=args.boundary,
        purity_get_ball=args.purity_get_ball,
        purity_select_ball=args.purity_select_ball,
        min_ball_get_ball=args.min_ball_get_ball,
        min_ball_select_ball=args.min_ball_select_ball,
    )
    if args.dry_run:
        views, train, test, inputs = _load_cached_inputs(paths, args.dataset, args.seed, args.kir)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "dataset": args.dataset,
                    "kir": args.kir,
                    "seed": args.seed,
                    "variant": config["output_variant"],
                    "distance": config["distance"],
                    "boundary": config["boundary"],
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "inputs": inputs,
                    "test_used_for_selection": False,
                },
                sort_keys=True,
            )
        )
        return 0
    result = run_one(paths, config, output_dir=args.output_dir, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": "complete",
                "run_dir": str(result),
                "variant": config["output_variant"],
                "resume": bool(args.resume),
                "device": args.device,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
