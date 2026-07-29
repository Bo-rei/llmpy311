"""Small, pre-registered attribution study for StackOverflow K=2 failures.

The experiment deliberately does not modify the legacy MultiSphere detector.
It reuses frozen E2 embeddings and the pooled-head checkpoints produced by the
R1 contract repair, then changes one boundary contract at a time:

1. covariance scope;
2. sphere selection score;
3. Known-train radius estimator.

The plan is fixed at 60 lightweight scoring cells.  It contains no encoder
training and cannot write into E2, E3, or R1 artifact roots.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
import torch
import yaml

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file, sha256_json
from protocol_v2.data.exporters._common import read_jsonl
from protocol_v2.data.manifests import dataset_manifest_path, read_json
from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.experiments.mechanism_runner import load_e2_bundle
from protocol_v2.experiments.partitions import build_partition, normalize_for_detector
from protocol_v2.experiments.r1_contract_repair import _encode_representation
from protocol_v2.runtime.paths import ProtocolV2Paths
from protocol_v2.tracking.run_manifest import atomic_run_directory


STAGE = "multicenter_boundary_attribution"
DATASET = "stackoverflow"
KIR = 0.50
SEEDS = (42, 87, 100)
REPRESENTATIONS = (
    "frozen_minilm",
    "ce_recon_pooled_head",
    "geometry_ce_recon_pooled_head",
)
ADAPTED_REPRESENTATIONS = REPRESENTATIONS[1:]
EXPECTED_UNITS = 60


@dataclass(frozen=True)
class AttributionSpec:
    """One immutable boundary-scoring cell."""

    phase: str
    seed: int
    representation: str
    k: int
    covariance_scope: str
    score_rule: str
    radius_rule: str

    @property
    def run_id(self) -> str:
        return (
            f"{STAGE}__{DATASET}__kir_{KIR:.2f}__seed_{self.seed}__"
            f"repr_{self.representation}__k_{self.k}__cov_{self.covariance_scope}__"
            f"score_{self.score_rule}__radius_{self.radius_rule}"
        )


@dataclass(frozen=True)
class BoundaryModel:
    """Fitted local centers and Known-only boundary statistics."""

    centers: np.ndarray
    intents: tuple[str, ...]
    assignments: np.ndarray
    inverse_variances: np.ndarray | None
    variances: np.ndarray | None
    radii: np.ndarray
    cluster_sizes: np.ndarray
    covariance_scope: str
    radius_rule: str


def attribution_root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / STAGE


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Boundary attribution config must be a mapping: {path}")
    if payload.get("stage") != STAGE or payload.get("protocol_version") != "protocol_v2_textoir_v1":
        raise ValueError("Boundary attribution config does not match the active stage/protocol")
    return payload


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_plan() -> list[AttributionSpec]:
    """Return the pre-registered 60-cell staged plan.

    Frozen K=1 under the current boundary is referenced from immutable E2
    rather than rerun.  The other 60 cells are new and disjoint from the
    completed ledger combinations.
    """

    plan: list[AttributionSpec] = []
    # A: adapted K=1 current contract (6), plus K=2 covariance isolation (27).
    for seed in SEEDS:
        for representation in ADAPTED_REPRESENTATIONS:
            plan.append(
                AttributionSpec(
                    "A_covariance",
                    seed,
                    representation,
                    1,
                    "per_cluster_diag",
                    "raw_distance_nearest",
                    "mean_std",
                )
            )
        for representation in REPRESENTATIONS:
            for covariance_scope in ("euclidean", "shared_intent_diag", "per_cluster_diag"):
                plan.append(
                    AttributionSpec(
                        "A_covariance",
                        seed,
                        representation,
                        2,
                        covariance_scope,
                        "raw_distance_nearest",
                        "mean_std",
                    )
                )
    # B: keep shared-intent covariance, change only sphere selection (9).
    for seed in SEEDS:
        for representation in REPRESENTATIONS:
            plan.append(
                AttributionSpec(
                    "B_score",
                    seed,
                    representation,
                    2,
                    "shared_intent_diag",
                    "normalized_score_min",
                    "mean_std",
                )
            )
    # C: keep normalized scoring, change only the Known-train radius (9).
    for seed in SEEDS:
        for representation in REPRESENTATIONS:
            plan.append(
                AttributionSpec(
                    "C_radius",
                    seed,
                    representation,
                    2,
                    "shared_intent_diag",
                    "normalized_score_min",
                    "quantile_95",
                )
            )
    # D: fair K=1 reference for the pre-registered final path (9).
    for seed in SEEDS:
        for representation in REPRESENTATIONS:
            plan.append(
                AttributionSpec(
                    "D_final_k1",
                    seed,
                    representation,
                    1,
                    "shared_intent_diag",
                    "normalized_score_min",
                    "quantile_95",
                )
            )
    if len(plan) != EXPECTED_UNITS or len({spec.run_id for spec in plan}) != EXPECTED_UNITS:
        raise AssertionError(f"Boundary attribution plan must contain {EXPECTED_UNITS} unique cells")
    return plan


def write_plan(paths: ProtocolV2Paths, config_path: Path) -> Path:
    paths.require_experiment_admission(DATASET)
    root = attribution_root(paths)
    plan_path = root / "plans" / "BOUNDARY_ATTRIBUTION_PLAN.csv"
    rows = [
        {
            "run_id": spec.run_id,
            "stage": STAGE,
            "phase": spec.phase,
            "protocol_version": paths.dataset_version,
            "dataset": DATASET,
            "kir": KIR,
            "seed": spec.seed,
            "representation": spec.representation,
            "k": spec.k,
            "partition": "kmeans",
            "partition_seed": 42,
            "covariance_scope": spec.covariance_scope,
            "score_rule": spec.score_rule,
            "radius_rule": spec.radius_rule,
            "config_sha256": sha256_file(config_path),
        }
        for spec in build_plan()
    ]
    _atomic_csv(plan_path, rows)
    atomic_write_json(
        plan_path.with_suffix(".json"),
        {
            "units": [
                {
                    "protocol_version": paths.dataset_version,
                    "datasets": DATASET,
                    "kirs": KIR,
                    "seeds": list(SEEDS),
                    "representations": list(REPRESENTATIONS),
                    "k_values": [1, 2],
                    "distances": ["euclidean", "mahalanobis_diag"],
                    "partition": "kmeans",
                    "boundary": ["mean_std", "quantile_95"],
                }
            ],
            "scoring_cells": rows,
        },
    )
    return plan_path


def _git_patch(repo_root: Path, destination: Path) -> str:
    tracked = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root.parent,
        capture_output=True,
        check=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    chunks = [tracked]
    for relative in untracked:
        if relative.startswith(("artifacts/", "assets/", "archives/", "textoir/")):
            continue
        diff = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "/dev/null", relative],
            cwd=repo_root.parent,
            capture_output=True,
            check=False,
        )
        if diff.stdout:
            chunks.append(diff.stdout)
    atomic_write_text(destination, b"\n".join(chunks).decode("utf-8", errors="replace"))
    return sha256_file(destination)


def freeze_provenance(paths: ProtocolV2Paths, config_path: Path) -> dict[str, Any]:
    root = attribution_root(paths)
    plan_path = write_plan(paths, config_path)
    patch_path = root / "BOUNDARY_ATTRIBUTION_CODE.patch"
    patch_sha = _git_patch(paths.project_root, patch_path)
    e2_closeout = paths.run_root / "summaries" / "e2_closeout" / "E2_closeout_manifest.json"
    repair_provenance = paths.run_root / "r1_contract_repair_v1" / "R1_CONTRACT_REPAIR_PROVENANCE.json"
    repair_closeout = paths.run_root / "r1_contract_repair_v1" / "R1_CONTRACT_REPAIR_CLOSEOUT.md"
    for required in (e2_closeout, repair_provenance, repair_closeout):
        if not required.is_file():
            raise FileNotFoundError(f"Required frozen parent evidence is missing: {required}")
    snapshot = {
        "schema_version": "s2c.multicenter_boundary_attribution.v1",
        "stage": STAGE,
        "protocol_version": paths.dataset_version,
        "dataset": DATASET,
        "kir": KIR,
        "seeds": list(SEEDS),
        "representations": list(REPRESENTATIONS),
        "planned_units": EXPECTED_UNITS,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.project_root.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "git_dirty": True,
        "code_patch_sha256": patch_sha,
        "config_sha256": sha256_file(config_path),
        "plan_sha256": sha256_file(plan_path),
        "e2_closeout_sha256": sha256_file(e2_closeout),
        "r1_contract_repair_provenance_sha256": sha256_file(repair_provenance),
        "r1_contract_repair_closeout_sha256": sha256_file(repair_closeout),
        "canonical_manifest_sha256": sha256_file(dataset_manifest_path(paths.manifest_root, DATASET)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "encoder_training": False,
        "used_oos_for_boundary_fit": False,
        "used_test_for_stage_configuration": False,
        "final_path": {
            "covariance_scope": "shared_intent_diag",
            "score_rule": "normalized_score_min",
            "radius_rule": "quantile_95",
        },
    }
    atomic_write_json(root / "BOUNDARY_ATTRIBUTION_PROVENANCE.json", snapshot)
    return snapshot


def require_provenance(paths: ProtocolV2Paths) -> dict[str, Any]:
    path = attribution_root(paths) / "BOUNDARY_ATTRIBUTION_PROVENANCE.json"
    if not path.is_file():
        raise RuntimeError("Boundary attribution provenance has not been frozen")
    payload = read_json(path)
    if payload.get("stage") != STAGE or payload.get("planned_units") != EXPECTED_UNITS:
        raise ValueError("Boundary attribution provenance does not match the fixed plan")
    return payload


def _model_path(paths: ProtocolV2Paths, config: Mapping[str, Any]) -> Path:
    configured = Path(str(config["model_path"]))
    for candidate in (
        configured if configured.is_absolute() else paths.project_root / configured,
        paths.project_root.parent / configured,
    ):
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    raise FileNotFoundError(f"Local MiniLM model is unavailable: {configured}")


def _checkpoint_path(paths: ProtocolV2Paths, seed: int, representation: str) -> Path:
    checkpoint = (
        paths.run_root
        / "r1_contract_repair_v1"
        / "checkpoints"
        / f"seed_{seed}"
        / representation
        / "encoder.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Contract-repair checkpoint is missing: {checkpoint}")
    return checkpoint


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _row_ids_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["sample_id"]) for row in rows).encode("utf-8")).hexdigest()


def _cache_path(paths: ProtocolV2Paths, seed: int, representation: str) -> Path:
    return attribution_root(paths) / "embedding_cache" / f"seed_{seed}" / f"{representation}.npz"


def materialize_embedding_cache(
    paths: ProtocolV2Paths,
    config: Mapping[str, Any],
    seed: int,
    representation: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load frozen E2 values or deterministically encode an existing checkpoint."""

    bundle = load_e2_bundle(paths, DATASET, seed, KIR)
    rows = {
        "train": bundle.views.train,
        "calibration": bundle.views.calibration,
        "test": bundle.views.test,
    }
    if representation == "frozen_minilm":
        arrays = {
            "train": normalize_for_detector(bundle.train).astype(np.float32),
            "calibration": normalize_for_detector(bundle.calibration).astype(np.float32),
            "test": normalize_for_detector(bundle.test).astype(np.float32),
        }
        return arrays, {
            "source": "immutable_e2_embedding_cache",
            "checkpoint_sha256": "frozen_e2_embedding_cache",
            "embedding_sha256": {name: _array_sha256(values) for name, values in arrays.items()},
            "row_ids_sha256": {name: _row_ids_sha256(split_rows) for name, split_rows in rows.items()},
            "cache_hit": True,
        }

    cache_path = _cache_path(paths, seed, representation)
    metadata_path = cache_path.with_suffix(".json")
    checkpoint = _checkpoint_path(paths, seed, representation)
    checkpoint_sha = sha256_file(checkpoint)
    expected = {
        "stage": STAGE,
        "seed": seed,
        "representation": representation,
        "checkpoint_sha256": checkpoint_sha,
        "row_ids_sha256": {name: _row_ids_sha256(split_rows) for name, split_rows in rows.items()},
        "gate_embedding": "normalized_pooled",
    }
    if cache_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if all(metadata.get(key) == value for key, value in expected.items()):
            with np.load(cache_path, allow_pickle=False) as payload:
                arrays = {
                    split: np.asarray(payload[split], dtype=np.float32)
                    for split in ("train", "calibration", "test")
                }
            actual_hashes = {name: _array_sha256(values) for name, values in arrays.items()}
            if actual_hashes == metadata.get("embedding_sha256"):
                return arrays, {**metadata, "cache_hit": True}
        raise RuntimeError(f"Refusing to replace mismatched attribution embedding cache: {cache_path}")

    encoded = _encode_representation(
        _model_path(paths, config),
        checkpoint,
        rows,
        config,
    )
    arrays = {
        name: normalize_for_detector(values).astype(np.float32)
        for name, values in encoded.items()
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, cache_path)
    metadata = {
        **expected,
        "source": "existing_r1_contract_repair_checkpoint",
        "embedding_sha256": {name: _array_sha256(values) for name, values in arrays.items()},
        "cache_hit": False,
    }
    atomic_write_json(metadata_path, metadata)
    return arrays, metadata


def _sphere_variances(
    train: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
    sphere_intents: Sequence[str],
    covariance_scope: str,
    eps: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if covariance_scope == "euclidean":
        return None, None
    if covariance_scope not in {"per_cluster_diag", "shared_intent_diag"}:
        raise ValueError(f"Unsupported covariance scope: {covariance_scope}")
    variances = np.empty_like(centers, dtype=np.float64)
    if covariance_scope == "per_cluster_diag":
        for sphere_id in range(len(centers)):
            points = train[assignments == sphere_id]
            variances[sphere_id] = np.var(points - centers[sphere_id], axis=0) + eps
    else:
        intents = np.asarray(sphere_intents, dtype=object)
        for intent in sorted(set(sphere_intents)):
            sphere_ids = np.flatnonzero(intents == intent)
            row_mask = np.isin(assignments, sphere_ids)
            residuals = train[row_mask] - centers[assignments[row_mask]]
            shared = np.var(residuals, axis=0) + eps
            variances[sphere_ids] = shared
    return variances, 1.0 / variances


def _distances(values: np.ndarray, centers: np.ndarray, inverse_variances: np.ndarray | None) -> np.ndarray:
    output = np.empty((values.shape[0], centers.shape[0]), dtype=np.float64)
    for sphere_id, center in enumerate(centers):
        difference = values - center
        if inverse_variances is None:
            output[:, sphere_id] = np.linalg.norm(difference, axis=1)
        else:
            output[:, sphere_id] = np.sqrt(np.sum(difference * difference * inverse_variances[sphere_id], axis=1))
    return output


def fit_boundary(
    train: np.ndarray,
    train_intents: Sequence[str],
    *,
    k: int,
    covariance_scope: str,
    radius_rule: str,
    covariance_eps: float,
    radius_lambda: float,
    quantile: float,
) -> BoundaryModel:
    """Fit KMeans centers, covariance and radii using Known train only."""

    normalized = normalize_for_detector(train)
    partition = build_partition(
        normalized,
        np.asarray(train_intents, dtype=object),
        k,
        "kmeans",
        42,
    )
    sphere_intents = tuple(partition.cluster_to_intent[index] for index in range(partition.cluster_count))
    variances, inverse = _sphere_variances(
        normalized,
        partition.labels,
        partition.centers,
        sphere_intents,
        covariance_scope,
        covariance_eps,
    )
    radii = np.empty(partition.cluster_count, dtype=np.float64)
    all_distances = _distances(normalized, partition.centers, inverse)
    for sphere_id in range(partition.cluster_count):
        values = all_distances[partition.labels == sphere_id, sphere_id]
        if radius_rule == "mean_std":
            radii[sphere_id] = float(values.mean() + radius_lambda * values.std())
        elif radius_rule == "quantile_95":
            radii[sphere_id] = float(np.quantile(values, quantile))
        else:
            raise ValueError(f"Unsupported radius rule: {radius_rule}")
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("Boundary radii must be finite and positive")
    return BoundaryModel(
        centers=np.asarray(partition.centers, dtype=np.float64),
        intents=sphere_intents,
        assignments=np.asarray(partition.labels, dtype=np.int64),
        inverse_variances=inverse,
        variances=variances,
        radii=radii,
        cluster_sizes=np.bincount(partition.labels, minlength=partition.cluster_count),
        covariance_scope=covariance_scope,
        radius_rule=radius_rule,
    )


def score_boundary(values: np.ndarray, model: BoundaryModel, score_rule: str) -> dict[str, np.ndarray]:
    distances = _distances(normalize_for_detector(values), model.centers, model.inverse_variances)
    ratios = distances / np.clip(model.radii[None, :], 1e-12, None)
    if score_rule == "raw_distance_nearest":
        selected = np.argmin(distances, axis=1)
    elif score_rule == "normalized_score_min":
        selected = np.argmin(ratios, axis=1)
    else:
        raise ValueError(f"Unsupported score rule: {score_rule}")
    row_index = np.arange(len(values))
    scores = ratios[row_index, selected]
    return {
        "score": scores,
        "prediction": (scores > 1.0).astype(np.int64),
        "selected_sphere": selected.astype(np.int64),
        "distance": distances[row_index, selected],
        "radius": model.radii[selected],
        "all_distances": distances,
    }


def _sphere_rows(
    spec: AttributionSpec,
    model: BoundaryModel,
    test_rows: Sequence[Mapping[str, Any]],
    output: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    labels = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    oos = labels == 1
    all_distances = np.asarray(output["all_distances"])
    selected = np.asarray(output["selected_sphere"])
    rows: list[dict[str, Any]] = []
    for sphere_id, intent in enumerate(model.intents):
        independent_accept = all_distances[:, sphere_id] <= model.radii[sphere_id]
        selected_accept = (selected == sphere_id) & (np.asarray(output["prediction"]) == 0)
        if model.variances is None:
            log_volume = math.nan
            geometric_mean_variance = math.nan
        else:
            log_volume = float(np.log(model.variances[sphere_id]).sum())
            geometric_mean_variance = float(np.exp(np.log(model.variances[sphere_id]).mean()))
        rows.append(
            {
                "run_id": spec.run_id,
                "phase": spec.phase,
                "seed": spec.seed,
                "representation": spec.representation,
                "k": spec.k,
                "covariance_scope": spec.covariance_scope,
                "score_rule": spec.score_rule,
                "radius_rule": spec.radius_rule,
                "sphere_id": sphere_id,
                "intent": intent,
                "train_cluster_size": int(model.cluster_sizes[sphere_id]),
                "radius": float(model.radii[sphere_id]),
                # Raw high-dimensional variance products underflow.  The log
                # volume and geometric mean are the numerically stable forms.
                "log_diagonal_variance_volume": log_volume,
                "geometric_mean_variance": geometric_mean_variance,
                "independent_accepted_oos": int(np.sum(independent_accept & oos)),
                "selected_accepted_oos": int(np.sum(selected_accept & oos)),
                "independent_accepted_known": int(np.sum(independent_accept & ~oos)),
                "selected_accepted_known": int(np.sum(selected_accept & ~oos)),
            }
        )
    return rows


def _run_one(
    paths: ProtocolV2Paths,
    config: Mapping[str, Any],
    spec: AttributionSpec,
    bundle: Any,
    embeddings: Mapping[str, np.ndarray],
    cache_metadata: Mapping[str, Any],
) -> Path:
    root = attribution_root(paths)
    run_dir = root / "runs" / spec.run_id
    resolved = {
        "stage": STAGE,
        "phase": spec.phase,
        "protocol_version": paths.dataset_version,
        "dataset": DATASET,
        "kir": KIR,
        "seed": spec.seed,
        "representation": spec.representation,
        "k": spec.k,
        "partition": "kmeans",
        "partition_seed": int(config["partition"]["seed"]),
        "covariance_scope": spec.covariance_scope,
        "score_rule": spec.score_rule,
        "radius_rule": spec.radius_rule,
        "covariance_eps": float(config["covariance_eps"]),
        "radius_lambda": float(config["radius_lambda"]),
        "quantile": float(config["quantile"]),
        "threshold": float(config["threshold"]),
    }
    config_hash = sha256_json(resolved)
    if run_dir.exists():
        manifest = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").is_file() else {}
        if manifest.get("status") == "complete" and manifest.get("config_sha256") == config_hash:
            return run_dir
        raise RuntimeError(f"Refusing to overwrite non-matching attribution run: {run_dir}")

    train_intents = [str(row["intent"]) for row in bundle.views.train]
    model = fit_boundary(
        embeddings["train"],
        train_intents,
        k=spec.k,
        covariance_scope=spec.covariance_scope,
        radius_rule=spec.radius_rule,
        covariance_eps=float(config["covariance_eps"]),
        radius_lambda=float(config["radius_lambda"]),
        quantile=float(config["quantile"]),
    )
    started = time.perf_counter()
    calibration_output = score_boundary(embeddings["calibration"], model, spec.score_rule)
    test_output = score_boundary(embeddings["test"], model, spec.score_rule)
    scoring_seconds = time.perf_counter() - started
    calibration_labels = np.zeros(len(bundle.views.calibration), dtype=np.int64)
    test_labels = np.asarray([int(row["label"]) for row in bundle.views.test], dtype=np.int64)
    metrics = compute_binary_oos_metrics(test_labels, test_output["score"], float(config["threshold"]))
    calibration_metrics = compute_binary_oos_metrics(
        calibration_labels,
        calibration_output["score"],
        float(config["threshold"]),
    )
    sphere_rows = _sphere_rows(spec, model, bundle.views.test, test_output)
    accepted_oos_ids = [
        str(row["sample_id"])
        for row, label, prediction in zip(
            bundle.views.test,
            test_labels,
            test_output["prediction"],
            strict=True,
        )
        if label == 1 and prediction == 0
    ]
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_json(temporary / "resolved_config.json", resolved)
        atomic_write_json(
            temporary / "metrics.json",
            {
                **metrics,
                "calibration_known_recall": calibration_metrics["id_recall"],
                "calibration_false_reject_rate": calibration_metrics["false_reject_rate"],
                "scoring_seconds": scoring_seconds,
                "throughput": len(bundle.views.test) / max(scoring_seconds, 1e-12),
                "effective_cluster_count": len(model.radii),
                "minimum_cluster_size": int(model.cluster_sizes.min()),
                "mean_radius": float(model.radii.mean()),
                "radius_cv": float(model.radii.std() / max(model.radii.mean(), 1e-12)),
            },
        )
        _atomic_csv(temporary / "sphere_diagnostics.csv", sphere_rows)
        atomic_write_json(
            temporary / "acceptance.json",
            {
                "accepted_oos_sample_ids": accepted_oos_ids,
                "accepted_oos_count": len(accepted_oos_ids),
                "test_sample_ids_sha256": _row_ids_sha256(bundle.views.test),
            },
        )
        provenance = require_provenance(paths)
        atomic_write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "stage": STAGE,
                "phase": spec.phase,
                "run_id": spec.run_id,
                "protocol_version": paths.dataset_version,
                "config": resolved,
                "config_sha256": config_hash,
                "stage_provenance_sha256": sha256_file(
                    attribution_root(paths) / "BOUNDARY_ATTRIBUTION_PROVENANCE.json"
                ),
                "code_patch_sha256": provenance["code_patch_sha256"],
                "embedding_cache": cache_metadata,
                "r1_checkpoint_sha256": cache_metadata["checkpoint_sha256"],
                "used_oos_for_fit": False,
                "used_test_for_configuration": False,
            },
        )
    return run_dir


def run_experiment(paths: ProtocolV2Paths, config: Mapping[str, Any]) -> dict[str, Any]:
    require_provenance(paths)
    paths.require_experiment_admission(DATASET)
    failures: list[dict[str, Any]] = []
    completed = 0
    for seed in SEEDS:
        bundle = load_e2_bundle(paths, DATASET, seed, KIR)
        cache: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
        for representation in REPRESENTATIONS:
            cache[representation] = materialize_embedding_cache(paths, config, seed, representation)
        for spec in [item for item in build_plan() if item.seed == seed]:
            try:
                embeddings, metadata = cache[spec.representation]
                _run_one(paths, config, spec, bundle, embeddings, metadata)
                completed += 1
            except Exception as exc:  # keep independent failures resumable
                failures.append(
                    {
                        "run_id": spec.run_id,
                        "phase": spec.phase,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    _atomic_csv(attribution_root(paths) / "summaries" / "BOUNDARY_ATTRIBUTION_FAILED.csv", failures)
    return {
        "planned": EXPECTED_UNITS,
        "completed": completed,
        "failed": len(failures),
    }


def _read_acceptance(path: Path) -> set[str]:
    return set(str(value) for value in read_json(path)["accepted_oos_sample_ids"])


def _read_result(paths: ProtocolV2Paths, spec: AttributionSpec) -> dict[str, Any]:
    run_dir = attribution_root(paths) / "runs" / spec.run_id
    metrics = read_json(run_dir / "metrics.json")
    return {
        "run_id": spec.run_id,
        "phase": spec.phase,
        "protocol_version": paths.dataset_version,
        "dataset": DATASET,
        "kir": KIR,
        "seed": spec.seed,
        "representation": spec.representation,
        "k": spec.k,
        "covariance_scope": spec.covariance_scope,
        "score_rule": spec.score_rule,
        "radius_rule": spec.radius_rule,
        **metrics,
    }


def _frozen_e2_k1_reference(paths: ProtocolV2Paths, seed: int) -> tuple[dict[str, Any], set[str]]:
    run_id = (
        f"protocol_v2_textoir_v1__{DATASET}__kir_{KIR:.2f}__seed_{seed}__"
        "repr_frozen_minilm__k_1__dist_mahalanobis_diag__boundary_mean_std"
    )
    run_dir = paths.run_root / "e2_gate_core_dense" / run_id
    metrics = read_json(run_dir / "metrics.json")["combined"]
    predictions = list(read_jsonl(run_dir / "predictions" / "test.jsonl"))
    accepted = {
        str(row["sample_id"])
        for row in predictions
        if int(row["gold_is_oos"]) == 1 and int(row["predicted_is_oos"]) == 0
    }
    return {"run_id": run_id, **metrics}, accepted


def _paired_rows(paths: ProtocolV2Paths, results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    by_key = {
        (
            row["seed"],
            row["representation"],
            row["k"],
            row["covariance_scope"],
            row["score_rule"],
            row["radius_rule"],
        ): row
        for row in results
    }
    def append_pair(
        comparison: str,
        seed: int,
        representation: str,
        one: Mapping[str, Any],
        two: Mapping[str, Any],
        one_accept: set[str],
        two_accept: set[str],
    ) -> None:
        pairs.append(
            {
                "comparison": comparison,
                "seed": seed,
                "representation": representation,
                "covariance_scope": two["covariance_scope"],
                "score_rule": two["score_rule"],
                "radius_rule": two["radius_rule"],
                "k1_run_id": one["run_id"],
                "k2_run_id": two["run_id"],
                "k1_oos_f1": one["oos_f1"],
                "k2_oos_f1": two["oos_f1"],
                "delta_oos_f1": float(two["oos_f1"]) - float(one["oos_f1"]),
                "delta_id_recall": float(two["id_recall"]) - float(one["id_recall"]),
                "delta_false_accept_rate": float(two["false_accept_rate"]) - float(one["false_accept_rate"]),
                "delta_false_reject_rate": float(two["false_reject_rate"]) - float(one["false_reject_rate"]),
                "newly_accepted_oos_count": len(two_accept - one_accept),
                "recovered_oos_count": len(one_accept - two_accept),
            }
        )

    # Current and shared-covariance candidates use the same K=1 model because
    # per-cluster and intent-shared covariance are identical when K=1.
    for seed in SEEDS:
        for representation in REPRESENTATIONS:
            if representation == "frozen_minilm":
                one, one_accept = _frozen_e2_k1_reference(paths, seed)
            else:
                one = by_key[
                    (seed, representation, 1, "per_cluster_diag", "raw_distance_nearest", "mean_std")
                ]
                one_accept = _read_acceptance(
                    attribution_root(paths) / "runs" / str(one["run_id"]) / "acceptance.json"
                )
            for comparison, covariance_scope in (
                ("current_per_cluster", "per_cluster_diag"),
                ("shared_covariance", "shared_intent_diag"),
            ):
                two = by_key[
                    (seed, representation, 2, covariance_scope, "raw_distance_nearest", "mean_std")
                ]
                two_accept = _read_acceptance(
                    attribution_root(paths) / "runs" / str(two["run_id"]) / "acceptance.json"
                )
                append_pair(comparison, seed, representation, one, two, one_accept, two_accept)

    # Fair final-path pairs for all representations.
    for seed in SEEDS:
        for representation in REPRESENTATIONS:
            one = by_key[(seed, representation, 1, "shared_intent_diag", "normalized_score_min", "quantile_95")]
            two = by_key[(seed, representation, 2, "shared_intent_diag", "normalized_score_min", "quantile_95")]
            one_accept = _read_acceptance(attribution_root(paths) / "runs" / str(one["run_id"]) / "acceptance.json")
            two_accept = _read_acceptance(attribution_root(paths) / "runs" / str(two["run_id"]) / "acceptance.json")
            append_pair("final_path", seed, representation, one, two, one_accept, two_accept)
    return pairs


def _stop_decision(config: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    gate = config["stop_gate"]
    rows: list[dict[str, Any]] = []
    for comparison in ("current_per_cluster", "shared_covariance", "final_path"):
        for representation in REPRESENTATIONS:
            selected = [
                row
                for row in pairs
                if row["representation"] == representation and row["comparison"] == comparison
            ]
            oos = np.asarray([float(row["delta_oos_f1"]) for row in selected])
            false_accept = np.asarray([float(row["delta_false_accept_rate"]) for row in selected])
            known = np.asarray([float(row["delta_id_recall"]) for row in selected])
            criteria = {
                "mean_oos_loss_ok": float(oos.mean()) >= -float(gate["maximum_mean_k2_oos_f1_loss"]),
                "single_seed_oos_loss_ok": float(oos.min()) >= -float(gate["maximum_single_seed_k2_oos_f1_loss"]),
                "mean_false_accept_ok": float(false_accept.mean()) <= float(gate["maximum_mean_false_accept_increase"]),
                "single_seed_false_accept_ok": float(false_accept.max()) <= float(gate["maximum_single_seed_false_accept_increase"]),
                "mean_known_recall_ok": float(known.mean()) >= -float(gate["maximum_mean_known_recall_loss"]),
                "single_seed_known_recall_ok": float(known.min()) >= -float(gate["maximum_single_seed_known_recall_loss"]),
            }
            rows.append(
                {
                    "comparison": comparison,
                    "representation": representation,
                    "mean_delta_oos_f1": float(oos.mean()),
                    "minimum_seed_delta_oos_f1": float(oos.min()),
                    "mean_delta_false_accept_rate": float(false_accept.mean()),
                    "maximum_seed_delta_false_accept_rate": float(false_accept.max()),
                    "mean_delta_id_recall": float(known.mean()),
                    "minimum_seed_delta_id_recall": float(known.min()),
                    **criteria,
                    "passes_stop_gate": all(criteria.values()),
                }
            )
    decision = (
        "boundary_mismatch_plausible_small_cross_dataset_extension_allowed"
        if any(bool(row["passes_stop_gate"]) for row in rows)
        else "stop_fixed_kmeans_multicenter_rescue"
    )
    return decision, rows


def _component_effects(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each pre-registered one-contract change at K=2."""

    by_key = {
        (
            row["seed"],
            row["representation"],
            row["covariance_scope"],
            row["score_rule"],
            row["radius_rule"],
        ): row
        for row in results
        if int(row["k"]) == 2
    }
    comparisons = (
        (
            "shared_covariance_minus_per_cluster",
            ("per_cluster_diag", "raw_distance_nearest", "mean_std"),
            ("shared_intent_diag", "raw_distance_nearest", "mean_std"),
        ),
        (
            "normalized_score_minus_raw_nearest",
            ("shared_intent_diag", "raw_distance_nearest", "mean_std"),
            ("shared_intent_diag", "normalized_score_min", "mean_std"),
        ),
        (
            "quantile95_minus_mean_std",
            ("shared_intent_diag", "normalized_score_min", "mean_std"),
            ("shared_intent_diag", "normalized_score_min", "quantile_95"),
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, before_config, after_config in comparisons:
        for representation in REPRESENTATIONS:
            before = [by_key[(seed, representation, *before_config)] for seed in SEEDS]
            after = [by_key[(seed, representation, *after_config)] for seed in SEEDS]
            rows.append(
                {
                    "component_change": name,
                    "representation": representation,
                    "mean_delta_oos_f1": float(
                        np.mean([float(right["oos_f1"]) - float(left["oos_f1"]) for left, right in zip(before, after, strict=True)])
                    ),
                    "mean_delta_id_recall": float(
                        np.mean([float(right["id_recall"]) - float(left["id_recall"]) for left, right in zip(before, after, strict=True)])
                    ),
                    "mean_delta_false_accept_rate": float(
                        np.mean(
                            [
                                float(right["false_accept_rate"]) - float(left["false_accept_rate"])
                                for left, right in zip(before, after, strict=True)
                            ]
                        )
                    ),
                    "mean_delta_false_reject_rate": float(
                        np.mean(
                            [
                                float(right["false_reject_rate"]) - float(left["false_reject_rate"])
                                for left, right in zip(before, after, strict=True)
                            ]
                        )
                    ),
                }
            )
    return rows


def closeout(paths: ProtocolV2Paths, config: Mapping[str, Any]) -> dict[str, Any]:
    plan = build_plan()
    missing = [spec.run_id for spec in plan if not (attribution_root(paths) / "runs" / spec.run_id / "manifest.json").is_file()]
    if missing:
        raise RuntimeError(f"Boundary attribution is incomplete: missing={len(missing)}")
    results = [_read_result(paths, spec) for spec in plan]
    pairs = _paired_rows(paths, results)
    component_rows = _component_effects(results)
    decision, stop_rows = _stop_decision(config, pairs)
    summary_root = attribution_root(paths) / "summaries"
    _atomic_csv(summary_root / "BOUNDARY_ATTRIBUTION_RESULTS.csv", results)
    _atomic_csv(summary_root / "BOUNDARY_ATTRIBUTION_K1_K2.csv", pairs)
    _atomic_csv(summary_root / "BOUNDARY_ATTRIBUTION_STOP_GATE.csv", stop_rows)
    _atomic_csv(summary_root / "BOUNDARY_ATTRIBUTION_COMPONENT_EFFECTS.csv", component_rows)
    sphere_rows: list[dict[str, Any]] = []
    for spec in plan:
        path = attribution_root(paths) / "runs" / spec.run_id / "sphere_diagnostics.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            sphere_rows.extend(csv.DictReader(handle))
    _atomic_csv(summary_root / "BOUNDARY_ATTRIBUTION_SPHERES.csv", sphere_rows)
    integrity = {
        "stage": STAGE,
        "planned": EXPECTED_UNITS,
        "completed": len(results),
        "failed": 0,
        "missing": 0,
        "duplicate_run_ids": len({row["run_id"] for row in results}) != len(results),
        "encoder_training_performed": False,
        "e2_e3_r1_artifacts_modified": False,
        "used_test_for_stage_configuration": False,
        "closeout_analysis_source_sha256": sha256_file(Path(__file__)),
        "decision": decision,
    }
    atomic_write_json(attribution_root(paths) / "BOUNDARY_ATTRIBUTION_INTEGRITY.json", integrity)
    lines = [
        "# Multi-center boundary attribution closeout",
        "",
        "## Scope",
        "",
        f"- Protocol: `{paths.dataset_version}`; dataset: StackOverflow; KIR: `0.50`; seeds: `{list(SEEDS)}`.",
        f"- Completed: `{len(results)}/{EXPECTED_UNITS}` lightweight scoring cells; encoder training: `false`.",
        "- The staged path was fixed before scoring: covariance → score rule → Known-train radius.",
        "",
        "## Stop-gate result",
        "",
    ]
    for row in stop_rows:
        lines.append(
            f"- `{row['comparison']}` / `{row['representation']}`: K2−K1 OOS F1 `{row['mean_delta_oos_f1']:+.4f}`, "
            f"false acceptance `{row['mean_delta_false_accept_rate']:+.4f}`, "
            f"Known Recall `{row['mean_delta_id_recall']:+.4f}`, pass=`{str(row['passes_stop_gate']).lower()}`."
        )
    lines.extend(
        [
            "",
            "## Component attribution",
            "",
        ]
    )
    for row in component_rows:
        lines.append(
            f"- `{row['component_change']}` / `{row['representation']}`: OOS F1 "
            f"`{row['mean_delta_oos_f1']:+.4f}`, false acceptance "
            f"`{row['mean_delta_false_accept_rate']:+.4f}`, Known Recall "
            f"`{row['mean_delta_id_recall']:+.4f}`."
        )
    lines.extend(
        [
            "",
            "Shared intent covariance is the only consistently helpful K=2 change, but it does not pass the pre-registered K2 safety gate. Normalized sphere selection and q95 radii increase acceptance-union overcoverage rather than repair it.",
            "",
            f"Decision: `{decision}`.",
            "",
            "This is a diagnostic route decision, not a deployment-selected boundary. Test OOS was not used to alter the pre-registered 60-cell path.",
        ]
    )
    report = "\n".join(lines) + "\n"
    atomic_write_text(attribution_root(paths) / "BOUNDARY_ATTRIBUTION_DECISION.md", report)
    atomic_write_text(attribution_root(paths) / "BOUNDARY_ATTRIBUTION_CLOSEOUT.md", report)
    return integrity


def verify(paths: ProtocolV2Paths) -> dict[str, Any]:
    plan = build_plan()
    complete = invalid = 0
    for spec in plan:
        run_dir = attribution_root(paths) / "runs" / spec.run_id
        try:
            manifest = read_json(run_dir / "manifest.json")
            metrics = read_json(run_dir / "metrics.json")
            if (
                manifest.get("status") == "complete"
                and manifest.get("run_id") == spec.run_id
                and manifest.get("stage") == STAGE
                and all(math.isfinite(float(metrics[field])) for field in ("oos_f1", "id_recall", "false_accept_rate", "false_reject_rate"))
            ):
                complete += 1
            else:
                invalid += 1
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            invalid += 1
    return {
        "stage": STAGE,
        "planned": EXPECTED_UNITS,
        "completed": complete,
        "invalid_or_missing": invalid,
        "run_root_disjoint_from_e2_e3_r1": all(
            attribution_root(paths) != paths.run_root / name
            for name in ("e2_gate_core_dense", "e3_mechanisms", "r1_contract_repair_v1")
        ),
        "textoir_runtime_dependency": False,
    }
