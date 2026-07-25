"""Resumable fixed-boundary MultiSphere Gate sweeps for protocol_v2.

This runner deliberately reuses the established MultiSphere detector.  It only
changes the data contract, provenance, cache key and output atomicity needed by
the new protocol.  It never reads TEXTOIR data, v19 prepared data or network
URLs at run time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from s2c.data.hashing import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file, sha256_json
from s2c.data.exporters._common import read_jsonl
from s2c.data.manifests import dataset_manifest_path, read_json, view_manifest_path
from s2c.data.registry import registry_path
from s2c.evaluation.metrics import compute_binary_oos_metrics
from s2c.gate.view_loader import GateViews, load_gate_views
from s2c.runtime.paths import ProtocolV2Paths
from s2c.tracking.provenance import file_hashes
from s2c.tracking.run_manifest import atomic_run_directory, environment_snapshot

from .boundaries import SUPPORTED_BOUNDARIES, apply_radius_estimator, known_conformal_threshold
from .matrix import GateRunSpec, filter_gate_specs, load_gate_matrix
from .registry import write_plan
from .resume import completed_run


def _model_path(paths: ProtocolV2Paths, name: str) -> Path:
    candidate = paths.project_root.parent / "assets" / "models" / name
    if not candidate.is_dir():
        raise FileNotFoundError(f"Local encoder is unavailable for protocol_v2: {candidate}")
    return candidate


def _model_fingerprint(path: Path) -> dict[str, Any]:
    files = {name: path / name for name in ("config.json", "modules.json", "config_sentence_transformers.json")}
    return {
        "name": path.name,
        "files": {name: sha256_file(file) for name, file in files.items() if file.is_file()},
    }


def _load_encoder(path: Path, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    # Do not call torch.cuda.is_available() at import time. The selected device
    # is explicit in the experiment config and failures remain visible.
    return SentenceTransformer(str(path), device=device)


@dataclass(frozen=True)
class CanonicalEmbeddings:
    """One frozen-encoder embedding table shared by every KIR view of a dataset."""

    values: np.ndarray
    index_by_sample_id: dict[str, int]
    metadata: dict[str, Any]


def _canonical_embedding_cache(
    paths: ProtocolV2Paths,
    dataset: str,
    model: dict[str, Any],
    encoder: Any,
    batch_size: int,
) -> CanonicalEmbeddings:
    """Encode every canonical row once, then serve KIR views by sample ID.

    KIR changes the accepted intent set, not a sentence's frozen MiniLM vector.
    Encoding the canonical corpus once prevents a dense KIR grid from silently
    repeating identical model inference for every registry.
    """
    manifest_path = dataset_manifest_path(paths.manifest_root, dataset)
    manifest = read_json(manifest_path)
    canonical_path = paths.data_root / str(manifest["canonical_relative_path"])
    rows = list(read_jsonl(canonical_path))
    sample_ids = np.asarray([str(row["sample_id"]) for row in rows], dtype="U64")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError(f"Canonical sample IDs are not unique: dataset={dataset}, path={canonical_path}")
    cache_key = {
        "canonical_manifest_sha256": sha256_file(manifest_path),
        "encoder": model,
        "preprocessing_sha256": sha256_json({"text_field": "text", "normalization": "detector_l2"}),
        "cache_scope": "canonical_dataset_once",
    }
    cache_name = sha256_json(cache_key)[:20]
    cache_dir = paths.embedding_cache_root / "canonical" / dataset
    cache_path = cache_dir / f"canonical_{cache_name}.npz"
    metadata_path = cache_dir / f"canonical_{cache_name}.json"
    expected_hash = sha256_json(cache_key)
    if cache_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("cache_key_sha256") == expected_hash:
            with np.load(cache_path, allow_pickle=False) as payload:
                values = np.asarray(payload["embeddings"], dtype=np.float32)
                cached_ids = np.asarray(payload["sample_ids"], dtype="U64")
            if values.shape[0] == len(sample_ids) and np.array_equal(cached_ids, sample_ids):
                return CanonicalEmbeddings(
                    values=values,
                    index_by_sample_id={sample_id: index for index, sample_id in enumerate(sample_ids.tolist())},
                    metadata={**metadata, "cache_hit": True},
                )
    values = np.asarray(
        encoder.encode([str(row["text"]) for row in rows], batch_size=batch_size, show_progress_bar=False),
        dtype=np.float32,
    )
    if values.ndim != 2 or values.shape[0] != len(rows):
        raise RuntimeError(f"Encoder returned invalid canonical shape {values.shape}: dataset={dataset}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, embeddings=values, sample_ids=sample_ids)
    os.replace(temporary, cache_path)
    metadata = {
        "cache_key": cache_key,
        "cache_key_sha256": expected_hash,
        "sample_count": len(rows),
        "embedding_shape": list(values.shape),
        "embedding_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids.tolist()).encode("utf-8")).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
        "cache_hit": False,
    }
    atomic_write_json(metadata_path, metadata)
    return CanonicalEmbeddings(
        values=values,
        index_by_sample_id={sample_id: index for index, sample_id in enumerate(sample_ids.tolist())},
        metadata=metadata,
    )


def _embedding_cache(
    paths: ProtocolV2Paths,
    spec: GateRunSpec,
    split: str,
    rows: list[dict[str, Any]],
    registry_sha: str,
    canonical_sha: str,
    model: dict[str, Any],
    canonical_embeddings: CanonicalEmbeddings,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_key = {
        "canonical_manifest_sha256": canonical_sha,
        "registry_sha256": registry_sha,
        "encoder": model,
        "preprocessing_sha256": sha256_json({"text_field": "text", "normalization": "detector_l2"}),
        "split": split,
    }
    cache_name = sha256_json(cache_key)[:20]
    cache_dir = paths.embedding_cache_root / spec.dataset / f"seed_{spec.seed}" / f"kir_{spec.kir:.2f}"
    cache_path = cache_dir / f"{split}_{cache_name}.npz"
    metadata_path = cache_dir / f"{split}_{cache_name}.json"
    expected_hash = sha256_json(cache_key)
    if cache_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("cache_key_sha256") == expected_hash:
            with np.load(cache_path, allow_pickle=False) as payload:
                values = np.asarray(payload["embeddings"], dtype=np.float32)
            if values.shape[0] == len(rows):
                return values, {**metadata, "cache_hit": True}
    try:
        indices = np.asarray(
            [canonical_embeddings.index_by_sample_id[str(row["sample_id"])] for row in rows], dtype=np.int64
        )
    except KeyError as exc:
        raise KeyError(
            f"View row is absent from canonical embedding cache: dataset={spec.dataset}, "
            f"KIR={spec.kir}, seed={spec.seed}, split={split}, sample_id={exc.args[0]}"
        ) from exc
    values = np.ascontiguousarray(canonical_embeddings.values[indices], dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, embeddings=values)
    os.replace(temporary, cache_path)
    metadata = {
        "cache_key": cache_key,
        "cache_key_sha256": expected_hash,
        "sample_count": len(rows),
        "embedding_shape": list(values.shape),
        "embedding_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "canonical_base_embedding_sha256": canonical_embeddings.metadata["embedding_sha256"],
        "canonical_base_cache_key_sha256": canonical_embeddings.metadata["cache_key_sha256"],
        "created_at": datetime.now(UTC).isoformat(),
        "cache_hit": False,
    }
    atomic_write_json(metadata_path, metadata)
    return values, metadata


def _build_detector(spec: GateRunSpec) -> Any:
    # The detector is legacy-tested algorithm code; protocol_v2 supplies a new
    # data boundary around it rather than creating a second implementation.
    # ``s2c`` is installed from ``src/``.  Import the legacy top-level package
    # directly so a CLI launched outside the checkout does not depend on the
    # checkout-only implicit ``src`` namespace.
    from gate.multi_sphere_oos_detector import MultiSphereOOSDetector

    return MultiSphereOOSDetector(
        radius_method="mean_std",
        radius_lambda=spec.radius_lambda,
        center_mode="class_centroid_mixture",
        distance_metric=spec.distance,
        covariance_eps=1e-6,
        l2_normalize=True,
        subcenters_per_intent=spec.k_gate,
        random_state=42,
    )


def _breakdown(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float = 1.0) -> dict[str, dict[str, float]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    result: dict[str, dict[str, float]] = {"combined": compute_binary_oos_metrics(labels, scores, threshold)}
    known_indices = np.flatnonzero(labels == 0)
    for source in ("heldout_intent", "native"):
        indices = np.array([index for index, row in enumerate(rows) if row.get("oos_source") == source], dtype=np.int64)
        if indices.size:
            selected = np.concatenate((known_indices, indices))
            result[source] = compute_binary_oos_metrics(labels[selected], scores[selected], threshold)
    return result


def _prediction_rows(
    rows: list[dict[str, Any]], output: dict[str, np.ndarray], detector: Any
) -> list[dict[str, Any]]:
    """Attach detector decisions to immutable test-row identifiers.

    ``nearest_cluster`` is an index in the detector's fitted sphere list.  The
    export therefore stores the corresponding known intent explicitly instead
    of forcing later analysis code to reload a detector checkpoint.  This is
    descriptive only: no test label participates in the prediction.
    """
    return [
        {
            "sample_id": row["sample_id"],
            "gold_is_oos": int(row["label"]),
            "gold_intent": row["intent"],
            "oos_source": row.get("oos_source", "known"),
            "oos_score": float(output["score"][index]),
            "predicted_is_oos": int(output["pred"][index]),
            "nearest_cluster": int(output["nearest_cluster"][index]),
            "nearest_known_intent": detector.spheres[int(output["nearest_cluster"][index])].intent_name,
            "distance": float(output["distance"][index]),
            "radius": float(output["radius"][index]),
        }
        for index, row in enumerate(rows)
    ]


def _config_payload(spec: GateRunSpec) -> dict[str, Any]:
    return {
        "protocol_version": spec.protocol_version,
        "dataset": spec.dataset,
        "kir": spec.kir,
        "seed": spec.seed,
        "representation": spec.representation,
        "k_gate": spec.k_gate,
        "distance": spec.distance,
        "boundary": spec.boundary,
        "radius_lambda": spec.radius_lambda,
        "encoder_name": spec.encoder_name,
        "encoder_device": spec.encoder_device,
        "oos_positive": True,
        "score_direction": "higher_is_more_oos",
        # Preserve the historical fixed-boundary payload exactly, so the
        # completed E1/E2 units remain resumable after adding E3 methods.
        "selection": "fixed_boundary_known_only_calibration"
        if spec.boundary == "mean_std"
        else "known_only_calibration",
    }


def _run_paths(paths: ProtocolV2Paths, spec: GateRunSpec) -> Path:
    return paths.run_root / spec.experiment_name / spec.run_id


def dry_run(paths: ProtocolV2Paths, spec: GateRunSpec) -> dict[str, Any]:
    if spec.protocol_version != paths.dataset_version:
        raise ValueError(
            f"Experiment config targets {spec.protocol_version}, but active data protocol is {paths.dataset_version}"
        )
    paths.reject_textoir_runtime_path(paths.data_root)
    root = paths.export_root / "s2c" / spec.dataset / f"seed_{spec.seed}" / f"kir_{spec.kir:.2f}"
    required = [root / "gate" / f"{name}.json" for name in ("train", "val", "test")]
    return {
        "run_id": spec.run_id,
        "run_dir": str(_run_paths(paths, spec)),
        "required_inputs": [{"path": str(path.relative_to(paths.project_root)), "exists": path.is_file()} for path in required],
        "runtime_data_root": str(paths.data_root.relative_to(paths.project_root)),
        "uses_textoir_data": False,
    }


def run_gate(
    paths: ProtocolV2Paths,
    spec: GateRunSpec,
    batch_size: int = 128,
    resume: bool = True,
    *,
    encoder: Any | None = None,
    model: dict[str, Any] | None = None,
    canonical_embeddings: CanonicalEmbeddings | None = None,
) -> Path:
    if spec.protocol_version != paths.dataset_version:
        raise ValueError(
            f"Experiment config targets {spec.protocol_version}, but active data protocol is {paths.dataset_version}"
        )
    if spec.representation != "frozen_minilm" or spec.boundary not in SUPPORTED_BOUNDARIES:
        raise NotImplementedError(f"Unsupported protocol_v2 Gate configuration: {spec.run_id}")
    paths.require_experiment_admission(spec.dataset)
    run_dir = _run_paths(paths, spec)
    config = _config_payload(spec)
    config_hash = sha256_json(config)
    if resume and completed_run(run_dir, config_hash):
        return run_dir
    if run_dir.exists():
        raise RuntimeError(f"Incomplete or incompatible run exists; refusing overwrite: {run_dir}")
    paths.reject_textoir_runtime_path(paths.data_root)
    views: GateViews = load_gate_views(paths, spec.dataset, spec.seed, spec.kir)
    registry = read_json(registry_path(paths, spec.dataset, spec.seed, spec.kir))
    canonical_manifest_path = dataset_manifest_path(paths.manifest_root, spec.dataset)
    input_hashes = file_hashes(
        {
            "registry": registry_path(paths, spec.dataset, spec.seed, spec.kir),
            "canonical_manifest": canonical_manifest_path,
            "view_manifest": view_manifest_path(paths.manifest_root, spec.dataset, spec.seed, spec.kir),
            "s2c_export_manifest": views.export_root / "export_manifest.json",
        }
    )
    model_path = _model_path(paths, spec.encoder_name)
    resolved_model = model or _model_fingerprint(model_path)
    resolved_encoder = encoder or _load_encoder(model_path, spec.encoder_device)
    resolved_canonical_embeddings = canonical_embeddings or _canonical_embedding_cache(
        paths, spec.dataset, resolved_model, resolved_encoder, batch_size
    )
    started = time.perf_counter()
    train_embeddings, train_cache = _embedding_cache(
        paths,
        spec,
        "train_known",
        views.train,
        registry["registry_sha256"],
        input_hashes["canonical_manifest"],
        resolved_model,
        resolved_canonical_embeddings,
    )
    test_embeddings, test_cache = _embedding_cache(
        paths,
        spec,
        "test_combined",
        views.test,
        registry["registry_sha256"],
        input_hashes["canonical_manifest"],
        resolved_model,
        resolved_canonical_embeddings,
    )
    calibration_embeddings: np.ndarray | None = None
    calibration_cache: dict[str, Any] | None = None
    if spec.boundary == "known_conformal":
        calibration_embeddings, calibration_cache = _embedding_cache(
            paths,
            spec,
            "calibration_known",
            views.calibration,
            registry["registry_sha256"],
            input_hashes["canonical_manifest"],
            resolved_model,
            resolved_canonical_embeddings,
        )
    detector = _build_detector(spec)
    detector.fit(train_embeddings, np.asarray([str(row["intent"]) for row in views.train], dtype=object))
    if any(int(row["label"]) != 0 for row in views.calibration):
        raise ValueError(f"Known-only calibration violation for {spec.run_id}")
    selection = apply_radius_estimator(detector, spec.boundary)
    if spec.boundary == "known_conformal":
        assert calibration_embeddings is not None
        calibration_scores = np.asarray(detector.predict_with_scores(calibration_embeddings)["score"], dtype=np.float64)
        selection = known_conformal_threshold(calibration_scores)
    score_started = time.perf_counter()
    output = detector.predict_with_scores(test_embeddings)
    scoring_seconds = time.perf_counter() - score_started
    output["pred"] = (np.asarray(output["score"], dtype=np.float64) > selection.threshold).astype(np.int64)
    metrics = _breakdown(views.test, np.asarray(output["score"], dtype=np.float64), selection.threshold)
    cluster_labels = np.asarray(detector._train_cluster_labels, dtype=np.int64)
    cluster_sizes = np.bincount(cluster_labels) if cluster_labels.size else np.array([], dtype=np.int64)
    main_metrics = metrics["combined"]
    main_metrics.update(
        {
            "scoring_seconds": scoring_seconds,
            "samples_per_second": len(views.test) / scoring_seconds if scoring_seconds else float("inf"),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "effective_cluster_count": len(detector.spheres),
            "minimum_cluster_size": int(cluster_sizes.min()) if cluster_sizes.size else 0,
        }
    )
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_text(temporary / "resolved_config.yaml", yaml.safe_dump(config, allow_unicode=True, sort_keys=True))
        atomic_write_json(temporary / "metrics.json", {"combined": main_metrics, "oos_breakdown": metrics})
        atomic_write_jsonl(
            temporary / "predictions" / "test.jsonl",
            _prediction_rows(views.test, output, detector),
        )
        atomic_write_json(
            temporary / "threshold_selection.json",
            {
                "type": "known_only_boundary",
                "boundary_method": selection.method,
                "radius_lambda": spec.radius_lambda,
                "threshold": selection.threshold,
                "details": selection.details,
                "test_used_for_selection": False,
            },
        )
        atomic_write_json(
            temporary / "environment.json",
            {**environment_snapshot(paths.project_root), "encoder": resolved_model, "encoder_device": spec.encoder_device},
        )
        (temporary / "logs").mkdir(parents=True, exist_ok=True)
        (temporary / "figures").mkdir(parents=True, exist_ok=True)
        (temporary / "logs" / "runner.txt").write_text("completed\n", encoding="utf-8")
        atomic_write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "protocol_version": spec.protocol_version,
                "run_id": spec.run_id,
                "config": config,
                "config_hash": config_hash,
                "input_hashes": input_hashes,
                "registry_sha256": registry["registry_sha256"],
                "canonical_manifest_sha256": input_hashes["canonical_manifest"],
                "embedding_cache": {
                    "canonical": resolved_canonical_embeddings.metadata,
                    "train": train_cache,
                    "test": test_cache,
                    "calibration": calibration_cache,
                },
                "elapsed_seconds": time.perf_counter() - started,
                "test_used_for_selection": False,
                "historical_artifacts_overwritten": False,
            },
        )
    return run_dir


def run_matrix(
    paths: ProtocolV2Paths,
    specs: Iterable[GateRunSpec],
    *,
    dry: bool,
    resume: bool,
    batch_size: int,
    state_path: Path | None = None,
) -> tuple[list[Path], list[dict[str, str]]]:
    spec_list = list(specs)
    if not dry:
        for dataset in {spec.dataset for spec in spec_list}:
            paths.require_experiment_admission(dataset)
    completed: list[Path] = []
    failed: list[dict[str, str]] = []
    loaded_encoders: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
    canonical_caches: dict[tuple[str, str, str], CanonicalEmbeddings] = {}

    def persist(last_run_id: str | None = None) -> None:
        if state_path is not None:
            atomic_write_json(
                state_path,
                {
                    "planned": len(spec_list),
                    "protocol_version": paths.dataset_version,
                    "completed": len(completed),
                    "failed": failed,
                    "last_run_id": last_run_id,
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )

    persist()
    for spec in spec_list:
        try:
            if dry:
                print(json.dumps(dry_run(paths, spec), sort_keys=True))
            else:
                if resume and completed_run(_run_paths(paths, spec), sha256_json(_config_payload(spec))):
                    completed.append(_run_paths(paths, spec))
                    continue
                key = (spec.encoder_name, spec.encoder_device)
                if key not in loaded_encoders:
                    path = _model_path(paths, spec.encoder_name)
                    loaded_encoders[key] = (_load_encoder(path, spec.encoder_device), _model_fingerprint(path))
                encoder, model = loaded_encoders[key]
                canonical_key = (spec.dataset, *key)
                if canonical_key not in canonical_caches:
                    canonical_caches[canonical_key] = _canonical_embedding_cache(
                        paths, spec.dataset, model, encoder, batch_size
                    )
                completed.append(
                    run_gate(
                        paths,
                        spec,
                        batch_size=batch_size,
                        resume=resume,
                        encoder=encoder,
                        model=model,
                        canonical_embeddings=canonical_caches[canonical_key],
                    )
                )
        except Exception as exc:  # Preserve other independent sweep cells.
            failed.append({"run_id": spec.run_id, "error_type": type(exc).__name__, "error": str(exc)})
        finally:
            persist(spec.run_id)
    return completed, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--kir", type=float, action="append")
    parser.add_argument(
        "--shard-name",
        help=(
            "Optional identifier for an independently resumable matrix shard. "
            "It changes only plan/state filenames; run IDs and output paths remain canonical."
        ),
    )
    args = parser.parse_args(argv)
    specs = filter_gate_specs(
        load_gate_matrix(args.config), datasets=args.dataset, seeds=args.seed, kirs=args.kir
    )
    paths = ProtocolV2Paths.discover()
    suffix = f".{args.shard_name}" if args.shard_name else ""
    plan = paths.run_root / "plans" / f"{args.config.stem}{suffix}.json"
    write_plan(plan, specs)
    state_path = paths.run_root / "plans" / f"{args.config.stem}{suffix}.state.json"
    completed, failed = run_matrix(
        paths,
        specs,
        dry=args.dry_run,
        resume=args.resume,
        batch_size=args.batch_size,
        state_path=state_path,
    )
    state = {"planned": len(specs), "completed": len(completed), "failed": failed}
    atomic_write_json(state_path, state)
    if failed:
        for item in failed:
            print(json.dumps(item, sort_keys=True), flush=True)
        return 1
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
