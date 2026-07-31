"""Run CLMSG Milestones 1--3 on immutable protocol_v2 MiniLM caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, f1_score

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file, sha256_json
from protocol_v2.data.manifests import dataset_manifest_path, read_json, view_manifest_path
from protocol_v2.data.registry import registry_path
from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.experiments.matrix import GateRunSpec
from protocol_v2.experiments.runner import (
    _canonical_embedding_cache,
    _embedding_cache,
    _model_fingerprint,
    _model_path,
)
from protocol_v2.gate.clmsg import (
    LocalSupportModel,
    SupportScores,
    known_order_statistic,
    split_conformal_p_values,
)
from protocol_v2.gate.view_loader import GateViews, load_gate_views
from protocol_v2.runtime.paths import ProtocolV2Paths
from protocol_v2.tracking.provenance import file_hashes
from protocol_v2.tracking.run_manifest import atomic_run_directory, environment_snapshot


METHOD_KNN = "knn_only"
METHOD_LOCAL = "local_scale_knn"
METHOD_CONFORMAL = "local_scale_conformal"
SUPPORTED_METHODS = {METHOD_KNN, METHOD_LOCAL, METHOD_CONFORMAL}


def _csv_text(rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _ids_sha256(rows: list[dict[str, Any]]) -> str:
    value = "\n".join(str(row["sample_id"]) for row in rows)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_split_contract(views: GateViews) -> dict[str, Any]:
    if any(int(row["label"]) != 0 for row in views.train + views.calibration):
        raise ValueError("CLMSG proper-train and calibration must be Known-only")
    train_ids = {str(row["sample_id"]) for row in views.train}
    calibration_ids = {str(row["sample_id"]) for row in views.calibration}
    test_ids = {str(row["sample_id"]) for row in views.test}
    if len(train_ids) != len(views.train) or len(calibration_ids) != len(views.calibration):
        raise ValueError("CLMSG split contains duplicate sample IDs")
    if train_ids & calibration_ids or train_ids & test_ids or calibration_ids & test_ids:
        raise ValueError("CLMSG proper-train, calibration, and test sample IDs must be disjoint")
    train_intents = {str(row["intent"]) for row in views.train}
    calibration_intents = {str(row["intent"]) for row in views.calibration}
    if not calibration_intents <= train_intents:
        raise ValueError("Calibration contains an intent absent from proper-train")
    return {
        "proper_train_count": len(views.train),
        "calibration_count": len(views.calibration),
        "test_count": len(views.test),
        "proper_train_intent_count": len(train_intents),
        "proper_train_sample_ids_sha256": _ids_sha256(views.train),
        "calibration_sample_ids_sha256": _ids_sha256(views.calibration),
        "test_sample_ids_sha256": _ids_sha256(views.test),
        "sample_id_sets_disjoint": True,
        "train_and_calibration_known_only": True,
        "test_used_for_fit_or_calibration": False,
    }


def _load_cached_inputs(
    paths: ProtocolV2Paths,
    config: dict[str, Any],
    seed: int,
) -> tuple[GateViews, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dataset = str(config["dataset"])
    kir = float(config["kir"])
    paths.require_experiment_admission(dataset)
    views = load_gate_views(paths, dataset, seed, kir)
    split_contract = _validate_split_contract(views)
    registry_file = registry_path(paths, dataset, seed, kir)
    registry = read_json(registry_file)
    canonical_manifest = dataset_manifest_path(paths.manifest_root, dataset)
    current_view_manifest = view_manifest_path(paths.manifest_root, dataset, seed, kir)
    input_hashes = file_hashes(
        {
            "registry": registry_file,
            "canonical_manifest": canonical_manifest,
            "view_manifest": current_view_manifest,
            "s2c_export_manifest": views.export_root / "export_manifest.json",
        }
    )
    model_path = _model_path(paths, str(config["encoder_name"]))
    model = _model_fingerprint(model_path)
    try:
        canonical = _canonical_embedding_cache(paths, dataset, model, None, 128)
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            f"Missing frozen MiniLM canonical cache for {dataset}; CLMSG refuses implicit encoding"
        ) from exc
    spec = GateRunSpec(
        experiment_name="clmsg_v1",
        dataset=dataset,
        kir=kir,
        seed=seed,
        k_gate=1,
        distance="euclidean",
        representation="frozen_minilm",
        boundary="mean_std",
        radius_lambda=1.0,
        encoder_name=str(config["encoder_name"]),
        encoder_device="cpu",
        protocol_version=paths.dataset_version,
    )
    cached: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for cache_name, rows in (
        ("train_known", views.train),
        ("calibration_known", views.calibration),
        ("test_combined", views.test),
    ):
        arrays[cache_name], cached[cache_name] = _embedding_cache(
            paths,
            spec,
            cache_name,
            rows,
            registry["registry_sha256"],
            input_hashes["canonical_manifest"],
            model,
            canonical,
        )
    return views, arrays["train_known"], arrays["calibration_known"], arrays["test_combined"], {
        "registry_sha256": registry["registry_sha256"],
        "input_hashes": input_hashes,
        "model": model,
        "canonical_embedding_sha256": canonical.metadata["embedding_sha256"],
        "embedding_cache": cached,
        "split_contract": split_contract,
    }


def _open_metrics(
    rows: list[dict[str, Any]],
    oos_scores: np.ndarray,
    threshold: float,
    nearest_intents: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    binary_gold = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    predicted_oos = (np.asarray(oos_scores, dtype=np.float64) > threshold).astype(np.int64)
    binary = compute_binary_oos_metrics(binary_gold, oos_scores, threshold)
    gold = np.asarray(
        ["oos" if int(row["label"]) else str(row["intent"]) for row in rows], dtype=object
    )
    predicted = np.asarray(nearest_intents, dtype=object).copy()
    predicted[predicted_oos.astype(bool)] = "oos"
    known_labels = sorted({str(row["intent"]) for row in rows if int(row["label"]) == 0})
    all_labels = known_labels + ["oos"]
    known = binary_gold == 0
    oos = binary_gold == 1
    return {
        **binary,
        "known_recall": binary["id_recall"],
        "known_macro_f1": float(
            f1_score(gold, predicted, labels=known_labels, average="macro", zero_division=0)
        ),
        "f1_k": float(f1_score(gold, predicted, labels=known_labels, average="macro", zero_division=0)),
        "f1_u": float(f1_score(gold == "oos", predicted == "oos", zero_division=0)),
        "f1_all": float(f1_score(gold, predicted, labels=all_labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predicted)),
        "oos_to_known": int(np.sum(oos & (predicted_oos == 0))),
        "known_to_oos": int(np.sum(known & (predicted_oos == 1))),
        "known_to_wrong_known": int(np.sum(known & (predicted_oos == 0) & (predicted != gold))),
        "known_to_correct_known": int(np.sum(known & (predicted_oos == 0) & (predicted == gold))),
    }, predicted


def _method_payloads(
    model: LocalSupportModel,
    calibration: SupportScores,
    test: SupportScores,
    rows: list[dict[str, Any]],
    alphas: list[float],
    primary_alpha: float,
    support_modes: list[str],
    hybrid_gammas: list[float],
    enabled_methods: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    metrics: dict[str, dict[str, Any]] = {}
    predictions: list[dict[str, Any]] = []

    knn_threshold, knn_rank = known_order_statistic(calibration.knn_score, primary_alpha)
    definitions: list[tuple[str, np.ndarray, float, np.ndarray, np.ndarray | None, dict[str, Any]]] = []
    if METHOD_KNN in enabled_methods:
        definitions.append(
            (
                METHOD_KNN,
                test.knn_score,
                knn_threshold,
                model.label_for_indices(test.knn_support_index),
                None,
                {
                    "selection": "known_calibration_order_statistic",
                    "alpha": primary_alpha,
                    "order_statistic_rank": knn_rank,
                },
            )
        )

    mode_specs: list[tuple[str, float | None]] = []
    for mode in support_modes:
        if mode == "hybrid_knn":
            mode_specs.extend((mode, gamma) for gamma in hybrid_gammas)
        else:
            mode_specs.append((mode, None))
    for mode, gamma in mode_specs:
        effective_gamma = 0.5 if gamma is None else float(gamma)
        calibration_mode_score, _ = calibration.score_for_mode(mode, effective_gamma)
        test_mode_score, test_mode_index = test.score_for_mode(mode, effective_gamma)
        suffix = mode if gamma is None else f"{mode}_gamma_{gamma:g}"
        if METHOD_LOCAL in enabled_methods:
            definitions.append(
                (
                    f"{METHOD_LOCAL}__{suffix}",
                    test_mode_score,
                    1.0,
                    model.label_for_indices(test_mode_index),
                    None,
                    {
                        "selection": "natural_local_support_threshold",
                        "support_mode": mode,
                        "gamma": gamma,
                        "threshold": 1.0,
                    },
                )
            )
        if METHOD_CONFORMAL in enabled_methods:
            conformal_p = split_conformal_p_values(calibration_mode_score, test_mode_score)
            for alpha in alphas:
                definitions.append(
                    (
                        f"{METHOD_CONFORMAL}__{suffix}__alpha_{alpha:g}",
                        1.0 - conformal_p,
                        1.0 - alpha,
                        model.label_for_indices(test_mode_index),
                        conformal_p,
                        {
                            "selection": "global_split_conformal",
                            "support_mode": mode,
                            "gamma": gamma,
                            "alpha": alpha,
                            "calibration_count": int(calibration_mode_score.size),
                            "tie_rule": "greater_or_equal",
                        },
                    )
                )

    binary_gold = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    for method, scores, threshold, nearest, p_values, selection in definitions:
        result, predicted_labels = _open_metrics(rows, scores, threshold, nearest)
        result.update({"method": method, "threshold": float(threshold), "selection": selection})
        if p_values is not None:
            known = binary_gold == 0
            oos = binary_gold == 1
            alpha = float(selection["alpha"])
            result.update(
                {
                    "target_alpha": alpha,
                    "empirical_known_false_rejection": result["false_reject_rate"],
                    "coverage_error": float(result["false_reject_rate"] - alpha),
                    "mean_p_value_known": float(np.mean(p_values[known])),
                    "mean_p_value_oos": float(np.mean(p_values[oos])),
                }
            )
        else:
            result.update(
                {
                    "target_alpha": None,
                    "empirical_known_false_rejection": result["false_reject_rate"],
                    "coverage_error": None,
                    "mean_p_value_known": None,
                    "mean_p_value_oos": None,
                }
            )
        metrics[method] = result
        predicted_oos = np.asarray(scores) > threshold
        for index, row in enumerate(rows):
            predictions.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "gold_intent": str(row["intent"]),
                    "gold_is_oos": int(row["label"]),
                    "oos_source": str(row.get("oos_source", "known")),
                    "method": method,
                    "nearest_known_intent": str(nearest[index]),
                    "oos_score": float(scores[index]),
                    "conformal_p_value": "" if p_values is None else float(p_values[index]),
                    "threshold": float(threshold),
                    "predicted_is_oos": int(predicted_oos[index]),
                    "predicted_label": str(predicted_labels[index]),
                }
            )
    return metrics, predictions


def _per_intent_rows(
    test_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    support_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in predictions})
    known_intents = sorted(support_stats["per_intent"])
    for method in methods:
        selected = [row for row in predictions if row["method"] == method]
        for intent in known_intents:
            known_rows = [
                row for row in selected if int(row["gold_is_oos"]) == 0 and row["gold_intent"] == intent
            ]
            accepted_oos = [
                row
                for row in selected
                if int(row["gold_is_oos"]) == 1
                and int(row["predicted_is_oos"]) == 0
                and row["nearest_known_intent"] == intent
            ]
            rejected = sum(int(row["predicted_is_oos"]) for row in known_rows)
            stats = support_stats["per_intent"][intent]
            result.append(
                {
                    "method": method,
                    "intent": intent,
                    "num_train": stats["sample_count"],
                    "num_calibration": sum(
                        1 for row in test_rows[:0] if str(row.get("intent")) == intent
                    ),
                    "mean_local_scale": stats["mean_local_scale"],
                    "known_test_count": len(known_rows),
                    "known_false_reject_count": rejected,
                    "known_false_reject_rate": rejected / len(known_rows) if known_rows else 0.0,
                    "oos_false_accept_count": len(accepted_oos),
                }
            )
    return result


def _baseline_rows(paths: ProtocolV2Paths, dataset: str, kir: float, seed: int) -> tuple[list[dict[str, Any]], str | None]:
    summary = paths.run_root / "mogb_baseline_v1" / "summary" / "all_runs.csv"
    if not summary.is_file():
        return [], None
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["dataset"] == dataset
            and float(row["kir"]) == kir
            and int(row["seed"]) == seed
            and row["method"]
            in {"single_centroid", "fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary"}
        ]
    return rows, sha256_file(summary)


def _run_root(paths: ProtocolV2Paths, output_dir: Path | None) -> Path:
    canonical = (paths.run_root / "clmsg_v1").resolve()
    if output_dir is None:
        return canonical
    chosen = output_dir.expanduser().resolve()
    if chosen != canonical and canonical not in chosen.parents:
        raise ValueError(f"CLMSG output must stay below its isolated run root: {canonical}")
    return chosen


def run_seed(
    paths: ProtocolV2Paths,
    config: dict[str, Any],
    seed: int,
    *,
    output_dir: Path | None,
    resume: bool,
    overwrite: bool,
    explicit_cache: Path | None,
    explicit_manifest: Path | None,
    device: str,
    num_workers: int,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    if paths.dataset_version != config["protocol_version"]:
        raise ValueError("CLMSG config and active protocol version disagree")
    if not set(config["methods"]) <= SUPPORTED_METHODS:
        raise ValueError("Milestones 1--3 only support KNN, local-scale, and split conformal")
    if explicit_cache is not None and explicit_cache.expanduser().resolve() != paths.embedding_cache_root.resolve():
        raise ValueError("CLMSG must use the active protocol embedding cache")
    expected_manifest = view_manifest_path(
        paths.manifest_root, str(config["dataset"]), seed, float(config["kir"])
    ).resolve()
    if explicit_manifest is not None and explicit_manifest.expanduser().resolve() != expected_manifest:
        raise ValueError("CLMSG split manifest override does not match the fixed protocol view")
    if num_workers < 1:
        raise ValueError("num_workers must be positive")

    root = _run_root(paths, output_dir) / str(config.get("variant", "default"))
    run_dir = root / str(config["dataset"]) / f"kir_{float(config['kir']):.2f}" / f"seed_{seed}"
    resolved = {**config, "seed": seed, "device": device, "num_workers": num_workers}
    config_hash = sha256_json(resolved)
    manifest_path = run_dir / "manifest.json"
    if resume and manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("status") == "complete" and manifest.get("config_sha256") == config_hash:
            return run_dir, read_json(run_dir / "metrics.json")
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Existing CLMSG run is incomplete or incompatible: {run_dir}")
        shutil.rmtree(run_dir)

    started = time.perf_counter()
    views, train, calibration, test, provenance = _load_cached_inputs(paths, resolved, seed)
    labels = np.asarray([str(row["intent"]) for row in views.train], dtype=object)
    model = LocalSupportModel(
        metric=str(config["distance"]),
        k_neighbors=int(config["k_neighbors"]),
        k_scale=int(config["k_scale"]),
        eps=float(config["epsilon"]),
        chunk_size=int(config["chunk_size"]),
    ).fit(train, labels)
    score_started = time.perf_counter()
    calibration_scores = model.score(calibration)
    test_scores = model.score(test)
    scoring_seconds = time.perf_counter() - score_started
    metrics, predictions = _method_payloads(
        model,
        calibration_scores,
        test_scores,
        views.test,
        [float(value) for value in config["alphas"]],
        float(config["primary_alpha"]),
        [str(value) for value in config["support_modes"]],
        [float(value) for value in config["hybrid_gammas"]],
        {str(value) for value in config["methods"]},
    )
    support_stats = model.support_statistics()
    calibration_counts = {
        intent: sum(1 for row in views.calibration if str(row["intent"]) == intent)
        for intent in support_stats["per_intent"]
    }
    per_intent = _per_intent_rows(views.test, predictions, support_stats)
    for row in per_intent:
        row["num_calibration"] = calibration_counts[str(row["intent"])]
    baselines, baseline_sha = _baseline_rows(paths, str(config["dataset"]), float(config["kir"]), seed)
    elapsed = time.perf_counter() - started
    for item in metrics.values():
        item["scoring_seconds_shared"] = scoring_seconds
        item["samples_per_second_shared"] = len(views.test) / scoring_seconds if scoring_seconds else None

    with atomic_run_directory(run_dir) as temporary:
        atomic_write_text(
            temporary / "config.yaml", yaml.safe_dump(resolved, allow_unicode=True, sort_keys=True)
        )
        atomic_write_json(temporary / "metrics.json", metrics)
        atomic_write_json(temporary / "baselines.json", {"rows": baselines, "source_sha256": baseline_sha})
        atomic_write_text(
            temporary / "predictions.csv",
            _csv_text(
                predictions,
                [
                    "sample_id",
                    "gold_intent",
                    "gold_is_oos",
                    "oos_source",
                    "method",
                    "nearest_known_intent",
                    "oos_score",
                    "conformal_p_value",
                    "threshold",
                    "predicted_is_oos",
                    "predicted_label",
                ],
            ),
        )
        atomic_write_json(
            temporary / "calibration_manifest.json",
            {
                **provenance["split_contract"],
                "proper_train_source": config["proper_train_source"],
                "calibration_source": config["calibration_source"],
                "registry_sha256": provenance["registry_sha256"],
                "uses_oos": False,
                "test_used_for_selection": False,
            },
        )
        np.save(temporary / "calibration_scores.npy", calibration_scores.class_local_scale_score)
        np.save(temporary / "calibration_knn_scores.npy", calibration_scores.knn_score)
        np.savez_compressed(
            temporary / "calibration_scores_all.npz",
            global_knn=calibration_scores.local_scale_score,
            class_conditional_knn=calibration_scores.class_local_scale_score,
            **{
                f"hybrid_knn_gamma_{gamma:g}": calibration_scores.score_for_mode(
                    "hybrid_knn", gamma
                )[0]
                for gamma in [float(value) for value in config["hybrid_gammas"]]
            },
        )
        atomic_write_json(temporary / "support_statistics.json", support_stats)
        atomic_write_text(
            temporary / "per_intent_metrics.csv",
            _csv_text(
                per_intent,
                [
                    "method",
                    "intent",
                    "num_train",
                    "num_calibration",
                    "mean_local_scale",
                    "known_test_count",
                    "known_false_reject_count",
                    "known_false_reject_rate",
                    "oos_false_accept_count",
                ],
            ),
        )
        atomic_write_json(
            temporary / "runtime.json",
            {
                "elapsed_seconds": elapsed,
                "scoring_seconds": scoring_seconds,
                "device_argument": device,
                "actual_compute": "numpy_cpu_cached_embeddings",
                "num_workers_argument": num_workers,
            },
        )
        environment = {
            **environment_snapshot(paths.project_root),
            "numpy": np.__version__,
            "scikit_learn": sklearn_version,
        }
        atomic_write_text(temporary / "environment.txt", json.dumps(environment, sort_keys=True) + "\n")
        atomic_write_text(temporary / "git_commit.txt", str(environment["git_commit"]) + "\n")
        atomic_write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "stage": config["stage"],
                "protocol_version": config["protocol_version"],
                "run_id": f"clmsg_v1__{config['dataset']}__kir{float(config['kir']):.2f}__seed{seed}",
                "config_sha256": config_hash,
                "provenance": provenance,
                "baseline_summary_sha256": baseline_sha,
                "historical_artifacts_overwritten": False,
                "uses_oos_for_fit": False,
                "uses_oos_for_calibration": False,
                "test_used_for_selection": False,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
    return run_dir, metrics


def _dry_run(paths: ProtocolV2Paths, config: dict[str, Any], seed: int, output_dir: Path | None) -> dict[str, Any]:
    expected_manifest = view_manifest_path(
        paths.manifest_root, str(config["dataset"]), seed, float(config["kir"])
    )
    return {
        "stage": config["stage"],
        "protocol_version": paths.dataset_version,
        "dataset": config["dataset"],
        "kir": config["kir"],
        "seed": seed,
        "methods": config["methods"],
        "alphas": config["alphas"],
        "view_manifest": os.path.relpath(expected_manifest, paths.project_root),
        "output": os.path.relpath(
            _run_root(paths, output_dir)
            / str(config.get("variant", "default"))
            / str(config["dataset"])
            / f"kir_{float(config['kir']):.2f}"
            / f"seed_{seed}",
            paths.project_root,
        ),
        "implicit_encoding_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/gates/clmsg.yaml"))
    parser.add_argument("--dataset")
    parser.add_argument("--kir", type=float)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--k-neighbors", type=int)
    parser.add_argument("--primary-alpha", type=float)
    parser.add_argument("--variant")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"CLMSG config must be a mapping: {args.config}")
    if args.dataset is not None:
        config["dataset"] = args.dataset
    if args.kir is not None:
        config["kir"] = args.kir
    if args.k_neighbors is not None:
        config["k_neighbors"] = args.k_neighbors
    if args.primary_alpha is not None:
        config["primary_alpha"] = args.primary_alpha
    if args.variant is not None:
        config["variant"] = args.variant
    seeds = args.seed or [int(value) for value in config["seeds"]]
    paths = ProtocolV2Paths.discover()
    for seed in seeds:
        if args.dry_run:
            print(json.dumps(_dry_run(paths, config, seed, args.output_dir), sort_keys=True))
            continue
        run_dir, metrics = run_seed(
            paths,
            config,
            seed,
            output_dir=args.output_dir,
            resume=args.resume,
            overwrite=args.overwrite,
            explicit_cache=args.embedding_cache,
            explicit_manifest=args.split_manifest,
            device=args.device,
            num_workers=args.num_workers,
        )
        print(json.dumps({"seed": seed, "run_dir": str(run_dir), "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
