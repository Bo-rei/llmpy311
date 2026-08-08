"""Protocol-v2 adapters for controlled and external OOS baseline viability.

The goal of this module is deliberately narrow: every requested baseline is
bound to one protocol-v2 registry and its already-materialised views before
anything is executed.  Methods that can be evaluated locally (MSP, Energy,
kNN and LOF) are small, frozen-MiniLM Gate-only controls.  Methods whose
published implementation needs a separate upstream environment are *not*
silently approximated.  They publish an auditable blocked/unsupported manifest
instead of made-up metrics.

This is separate from :mod:`protocol_v2.experiments.runner`: it must not change the
completed E1/E2/E3 Gate sweep contract or reuse their run directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import yaml

from protocol_v2.data.hashing import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file, sha256_json
from protocol_v2.data.manifests import dataset_manifest_path, export_manifest_path, read_json, view_manifest_path
from protocol_v2.data.registry import registry_path
from protocol_v2.data.schema import format_kir
from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.gate.view_loader import GateViews, load_gate_views
from protocol_v2.runtime.paths import ProtocolV2Paths
from protocol_v2.tracking.provenance import file_hashes
from protocol_v2.tracking.run_manifest import atomic_run_directory, environment_snapshot


NativeMethod = Literal["msp", "energy", "knn", "lof"]
AvailabilityState = Literal["available", "blocked", "unsupported"]


@dataclass(frozen=True)
class ExternalBaselineSpec:
    """One fixed-registry E4 viability cell.

    The run identifier contains the method because external baselines do not
    share Gate geometry parameters.  It intentionally has no synthetic OOS
    training switch: the protocol exports train/dev as Known-only views.
    """

    experiment_name: str
    dataset: str
    kir: float
    seed: int
    method: str
    representation: str
    encoder_name: str
    encoder_device: str
    protocol_version: str = "protocol_v2"

    @property
    def run_id(self) -> str:
        return "__".join(
            (
                self.protocol_version,
                self.dataset,
                f"kir_{format_kir(self.kir)}",
                f"seed_{self.seed}",
                f"repr_{self.representation}",
                f"baseline_{self.method}",
            )
        )


@dataclass(frozen=True)
class MethodAvailability:
    """A reproducible explanation of why a method can or cannot run."""

    method: str
    state: AvailabilityState
    adapter_kind: str
    export_name: str
    reason: str | None
    required_packages: tuple[str, ...]
    missing_packages: tuple[str, ...]
    uses_known_only_train: bool
    uses_oos_for_training: bool
    uses_oos_for_calibration: bool
    requires_separate_environment: bool

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_packages"] = list(self.required_packages)
        payload["missing_packages"] = list(self.missing_packages)
        return payload


@dataclass(frozen=True)
class BaselineSelection:
    """Known-only score threshold selected before test evaluation."""

    threshold: float
    alpha: float
    calibration_count: int
    order_statistic_rank: int


NATIVE_METHODS: frozenset[str] = frozenset({"msp", "energy", "knn", "lof"})
EXTERNAL_METHODS: frozenset[str] = frozenset({"doc", "adb", "da_adb", "k_plus_1_way", "mogb"})
SUPPORTED_METHODS: frozenset[str] = NATIVE_METHODS | EXTERNAL_METHODS

# This is a contract table, not an implicit fallback chain.  In particular,
# MOGB's TSV export is sufficient for a later audited invocation but is not a
# claim that protocol_v2 reproduces MOGB's representation-learning algorithm.
METHOD_EXPORTS = {
    "msp": "s2c",
    "energy": "s2c",
    "knn": "s2c",
    "lof": "s2c",
    "doc": "textoir",
    "adb": "textoir",
    "da_adb": "textoir",
    "k_plus_1_way": "k_plus_1_way",
    "mogb": "mogb",
}


def _required(payload: dict[str, Any], field: str) -> list[Any]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"External baseline matrix requires non-empty list: {field}")
    return value


def load_external_baseline_matrix(path: Path) -> list[ExternalBaselineSpec]:
    """Parse the E4 declaration without deriving labels or data paths."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"External baseline configuration must be a mapping: {path}")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError(f"External baseline defaults must be a mapping: {path}")
    if defaults.get("source") != "protocol_v2_exports":
        raise ValueError("External baselines must declare defaults.source=protocol_v2_exports")
    name = str(payload.get("name", path.stem))
    methods = [str(method) for method in _required(payload, "methods")]
    unknown = sorted(set(methods) - SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unsupported external baseline method(s): {unknown}")
    specs = [
        ExternalBaselineSpec(
            experiment_name=name,
            dataset=str(dataset),
            kir=float(kir),
            seed=int(seed),
            method=method,
            representation=str(defaults.get("representation", "frozen_minilm")),
            encoder_name=str(defaults.get("encoder_name", "all-MiniLM-L6-v2")),
            encoder_device=str(defaults.get("encoder_device", "cuda")),
            protocol_version=str(payload.get("protocol_version", "protocol_v2")),
        )
        for dataset in _required(payload, "datasets")
        for kir in _required(payload, "kirs")
        for seed in _required(payload, "seeds")
        for method in methods
    ]
    if len({spec.run_id for spec in specs}) != len(specs):
        raise ValueError(f"External baseline configuration creates duplicate run ids: {path}")
    return specs


def _missing_packages(packages: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(package for package in packages if find_spec(package) is None)


def method_availability(method: str) -> MethodAvailability:
    """Describe a method without importing an upstream implementation.

    Importing TEXTOIR's old package into the current Python environment would
    make availability depend on import side effects and could silently bind to
    ``textoir/data``.  A separately audited environment is therefore an
    explicit prerequisite for DOC/ADB/DA-ADB.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported external baseline method: {method}")
    export_name = METHOD_EXPORTS[method]
    if method in NATIVE_METHODS:
        packages = ("numpy", "sklearn", "sentence_transformers")
        missing = _missing_packages(packages)
        return MethodAvailability(
            method=method,
            state="available" if not missing else "blocked",
            adapter_kind="native_frozen_minilm_control",
            export_name=export_name,
            reason=None if not missing else "missing local optional package(s): " + ", ".join(missing),
            required_packages=packages,
            missing_packages=missing,
            uses_known_only_train=True,
            uses_oos_for_training=False,
            uses_oos_for_calibration=False,
            requires_separate_environment=False,
        )
    if method == "k_plus_1_way":
        return MethodAvailability(
            method=method,
            state="unsupported",
            adapter_kind="fixed_split_export_only",
            export_name=export_name,
            reason=(
                "The protocol_v2 K+1 export deliberately has Known-only train/dev rows. "
                "No faithful K+1-way learner is available without introducing labelled or synthetic OOS training data."
            ),
            required_packages=(),
            missing_packages=(),
            uses_known_only_train=True,
            uses_oos_for_training=False,
            uses_oos_for_calibration=False,
            requires_separate_environment=True,
        )
    if method == "mogb":
        return MethodAvailability(
            method=method,
            state="blocked",
            adapter_kind="fixed_split_export_only",
            export_name=export_name,
            reason=(
                "The fixed MOGB TSV export exists, but an audited official MOGB environment and runner are not configured. "
                "A MiniLM post-processing substitute would not be an MOGB reproduction."
            ),
            required_packages=(),
            missing_packages=(),
            uses_known_only_train=True,
            uses_oos_for_training=False,
            uses_oos_for_calibration=False,
            requires_separate_environment=True,
        )
    return MethodAvailability(
        method=method,
        state="blocked",
        adapter_kind="textoir_export_only",
        export_name=export_name,
        reason=(
            "The fixed TEXTOIR TSV export exists, but this method requires a separately audited upstream-compatible "
            "environment and explicit prediction importer.  protocol_v2 will not import TEXTOIR in-process."
        ),
        required_packages=(),
        missing_packages=(),
        uses_known_only_train=True,
        uses_oos_for_training=False,
        uses_oos_for_calibration=False,
        requires_separate_environment=True,
    )


def _run_dir(paths: ProtocolV2Paths, spec: ExternalBaselineSpec) -> Path:
    # ``experiment_name`` is the isolated stage root.  Legacy callers retain
    # ``external_baselines``; new protocol stages can never collide with it.
    return paths.run_root / spec.experiment_name / spec.run_id


def _display_path(path: Path, project_root: Path) -> str:
    """Render a portable relative path even when artifacts are workspace siblings."""
    return os.path.relpath(path, project_root)


def _model_path(paths: ProtocolV2Paths, name: str) -> Path:
    candidate = paths.project_root.parent / "assets" / "models" / name
    if not candidate.is_dir():
        raise FileNotFoundError(f"Local encoder is unavailable for external baseline: {candidate}")
    return candidate


def _model_fingerprint(path: Path) -> dict[str, Any]:
    files = {name: path / name for name in ("config.json", "modules.json", "config_sentence_transformers.json")}
    return {"name": path.name, "files": {name: sha256_file(file) for name, file in files.items() if file.is_file()}}


def _load_encoder(path: Path, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    # Device selection is explicit in the config.  Do not probe CUDA globally:
    # that has caused native-runtime failures in several local environments.
    return SentenceTransformer(str(path), device=device)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, np.float32(1e-12))


def _score_linear(method: NativeMethod, classifier: Any, values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    probabilities = np.asarray(classifier.predict_proba(values), dtype=np.float64)
    logits = np.asarray(classifier.decision_function(values), dtype=np.float64)
    if logits.ndim == 1:
        logits = np.column_stack((-0.5 * logits, 0.5 * logits))
    if method == "msp":
        scores = 1.0 - np.max(probabilities, axis=1)
    elif method == "energy":
        maximum = np.max(logits, axis=1, keepdims=True)
        scores = -(maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1)))
    else:
        raise ValueError(f"Linear scorer does not support method={method}")
    indices = np.argmax(probabilities, axis=1)
    return np.asarray(scores, dtype=np.float64), [str(label) for label in classifier.classes_[indices]]


def _score_neighbour(method: NativeMethod, train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

    # 20 follows the historical controlled baseline but is clipped for small
    # protocol fixtures.  The effective value is exposed in the manifest.
    neighbours = min(20, max(1, train.shape[0] - 1))
    nearest = NearestNeighbors(n_neighbors=neighbours, metric="euclidean", n_jobs=1).fit(train)
    distances, indices = nearest.kneighbors(query)
    if method == "knn":
        return np.mean(distances, axis=1), indices[:, 0]
    if method != "lof":
        raise ValueError(f"Neighbour scorer does not support method={method}")
    detector = LocalOutlierFactor(n_neighbors=neighbours, novelty=True, n_jobs=1).fit(train)
    return -np.asarray(detector.score_samples(query), dtype=np.float64), indices[:, 0]


def _known_only_threshold(scores: np.ndarray, alpha: float = 0.05) -> BaselineSelection:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Known-only threshold selection requires finite calibration scores")
    rank = min(int(math.ceil((values.size + 1) * (1.0 - alpha))), values.size)
    threshold = float(np.partition(values, rank - 1)[rank - 1])
    return BaselineSelection(threshold, alpha, int(values.size), rank)


def _breakdown(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> dict[str, dict[str, float]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    result: dict[str, dict[str, float]] = {"combined": compute_binary_oos_metrics(labels, scores, threshold)}
    known_indices = np.flatnonzero(labels == 0)
    for source in ("heldout_intent", "native"):
        oos_indices = np.asarray(
            [index for index, row in enumerate(rows) if row.get("oos_source") == source], dtype=np.int64
        )
        if oos_indices.size:
            selected = np.concatenate((known_indices, oos_indices))
            result[source] = compute_binary_oos_metrics(labels[selected], scores[selected], threshold)
    return result


def _export_contract(paths: ProtocolV2Paths, spec: ExternalBaselineSpec, availability: MethodAvailability) -> dict[str, Any]:
    """Fail closed unless the method export proves it consumed this registry."""
    registry_path_value = registry_path(paths, spec.dataset, spec.seed, spec.kir)
    registry = read_json(registry_path_value)
    manifest_path = export_manifest_path(paths.manifest_root, availability.export_name, spec.dataset, spec.seed, spec.kir)
    export_manifest = read_json(manifest_path)
    if export_manifest.get("registry_sha256") != registry.get("registry_sha256"):
        raise ValueError(
            f"External adapter registry mismatch: method={spec.method}, dataset={spec.dataset}, "
            f"export={availability.export_name}"
        )
    if export_manifest.get("dataset") != spec.dataset or float(export_manifest.get("kir", -1)) != spec.kir:
        raise ValueError(f"External adapter export does not match run spec: {manifest_path}")
    return {
        "registry_path": registry_path_value.relative_to(paths.project_root).as_posix(),
        "registry_sha256": str(registry["registry_sha256"]),
        "export_manifest_path": manifest_path.relative_to(paths.project_root).as_posix(),
        "export_manifest_sha256": sha256_file(manifest_path),
        "canonical_sample_id_mapping_sha256": export_manifest["canonical_sample_id_mapping_sha256"],
    }


def _config_payload(spec: ExternalBaselineSpec, availability: MethodAvailability) -> dict[str, Any]:
    return {
        "experiment_name": spec.experiment_name,
        "protocol_version": spec.protocol_version,
        "dataset": spec.dataset,
        "kir": spec.kir,
        "seed": spec.seed,
        "method": spec.method,
        "representation": spec.representation,
        "encoder_name": spec.encoder_name,
        "encoder_device": spec.encoder_device,
        "adapter_kind": availability.adapter_kind,
        "export_name": availability.export_name,
        "oos_positive": True,
        "score_direction": "higher_is_more_oos",
        "selection": "known_only_conformal_alpha_0.05",
        "uses_oos_for_training": False,
        "uses_oos_for_calibration": False,
    }


def dry_run(paths: ProtocolV2Paths, spec: ExternalBaselineSpec) -> dict[str, Any]:
    """Return immutable dependencies and viability without running a model."""
    paths.reject_textoir_runtime_path(paths.data_root)
    availability = method_availability(spec.method)
    contract = _export_contract(paths, spec, availability)
    model_exists: bool | None = None
    model_relative_path: str | None = None
    if availability.state == "available":
        model_path = _model_path(paths, spec.encoder_name)
        model_exists = True
        model_relative_path = model_path.relative_to(paths.project_root.parent).as_posix()
    return {
        "run_id": spec.run_id,
        "run_dir": _display_path(_run_dir(paths, spec), paths.project_root),
        "method_availability": availability.as_json(),
        "registry_export_contract": contract,
        "local_encoder_available": model_exists,
        "local_encoder_path": model_relative_path,
        "uses_textoir_data": False,
    }


def _encode(encoder: Any, rows: list[dict[str, Any]], batch_size: int) -> np.ndarray:
    values = encoder.encode([str(row["text"]) for row in rows], batch_size=batch_size, show_progress_bar=False)
    vectors = _l2_normalize(np.asarray(values, dtype=np.float32))
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise RuntimeError(f"Frozen encoder returned invalid shape {vectors.shape}")
    return vectors


def _prediction_rows(
    rows: list[dict[str, Any]], scores: np.ndarray, threshold: float, nearest_intents: list[str]
) -> list[dict[str, Any]]:
    predictions = (np.asarray(scores, dtype=np.float64) > threshold).astype(np.int64)
    return [
        {
            "sample_id": str(row["sample_id"]),
            "gold_is_oos": int(row["label"]),
            "gold_intent": str(row["intent"]),
            "oos_source": str(row.get("oos_source", "known")),
            "oos_score": float(scores[index]),
            "predicted_is_oos": int(predictions[index]),
            "nearest_known_intent": nearest_intents[index],
        }
        for index, row in enumerate(rows)
    ]


def _native_scores(
    spec: ExternalBaselineSpec,
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    """Fit an intentionally small native control using Known training rows only."""
    method = spec.method
    if method in {"msp", "energy"}:
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
        classifier.fit(train, train_labels)
        calibration_scores, calibration_nearest = _score_linear(method, classifier, calibration)
        test_scores, test_nearest = _score_linear(method, classifier, test)
        return calibration_scores, test_scores, calibration_nearest, test_nearest, {
            "classifier": "sklearn.linear_model.LogisticRegression",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
        }
    calibration_scores, calibration_indices = _score_neighbour(method, train, calibration)
    test_scores, test_indices = _score_neighbour(method, train, test)
    labels = [str(label) for label in train_labels]
    return (
        calibration_scores,
        test_scores,
        [labels[int(index)] for index in calibration_indices],
        [labels[int(index)] for index in test_indices],
        {"neighbour_model": "sklearn.neighbors.LocalOutlierFactor" if method == "lof" else "sklearn.neighbors.NearestNeighbors", "n_neighbors": min(20, max(1, train.shape[0] - 1))},
    )


def _existing_status(run_dir: Path, config_hash: str) -> str | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("config_hash") != config_hash:
        return None
    return str(manifest.get("status"))


def run_external_baseline(
    paths: ProtocolV2Paths,
    spec: ExternalBaselineSpec,
    *,
    batch_size: int = 128,
    resume: bool = True,
    encoder: Any | None = None,
    precomputed: tuple[GateViews, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[Path, str]:
    """Execute a native control or write a non-metric blocked manifest.

    Returning ``blocked`` or ``unsupported`` is a successful viability result;
    it must not be collapsed into a fake zero-valued metric row.
    """
    paths.require_experiment_admission(spec.dataset)
    paths.reject_textoir_runtime_path(paths.data_root)
    availability = method_availability(spec.method)
    config = _config_payload(spec, availability)
    config_hash = sha256_json(config)
    run_dir = _run_dir(paths, spec)
    existing = _existing_status(run_dir, config_hash)
    if resume and existing is not None:
        return run_dir, existing
    if run_dir.exists():
        raise RuntimeError(f"Incomplete or incompatible external baseline run exists; refusing overwrite: {run_dir}")
    contract = _export_contract(paths, spec, availability)
    input_hashes = file_hashes(
        {
            "registry": registry_path(paths, spec.dataset, spec.seed, spec.kir),
            "canonical_manifest": dataset_manifest_path(paths.manifest_root, spec.dataset),
            "view_manifest": view_manifest_path(paths.manifest_root, spec.dataset, spec.seed, spec.kir),
            "export_manifest": export_manifest_path(
                paths.manifest_root, availability.export_name, spec.dataset, spec.seed, spec.kir
            ),
        }
    )
    if availability.state != "available":
        with atomic_run_directory(run_dir) as temporary:
            atomic_write_json(temporary / "manifest.json", {
                "status": availability.state,
                "run_id": spec.run_id,
                "config": config,
                "config_hash": config_hash,
                "method_availability": availability.as_json(),
                "registry_export_contract": contract,
                "input_hashes": input_hashes,
                "test_used_for_selection": False,
                "historical_artifacts_overwritten": False,
                "metrics_emitted": False,
                "uses_oos_for_training": availability.uses_oos_for_training,
                "uses_oos_for_calibration": availability.uses_oos_for_calibration,
                "created_at": datetime.now(UTC).isoformat(),
            })
            atomic_write_json(temporary / "blocked.json", {
                "status": availability.state,
                "method": spec.method,
                "reason": availability.reason,
                "next_required_action": (
                    "Create a separate audited external environment and import predictions using the fixed export manifest."
                ),
            })
        return run_dir, availability.state

    if spec.representation != "frozen_minilm":
        raise NotImplementedError(f"Native E4 controls only support frozen_minilm: {spec.run_id}")
    started = time.perf_counter()
    if precomputed is None:
        views: GateViews = load_gate_views(paths, spec.dataset, spec.seed, spec.kir)
    else:
        views = precomputed[0]
    if any(int(row["label"]) != 0 for row in views.train + views.calibration):
        raise ValueError(f"Known-only training/calibration violation: {spec.run_id}")
    model_path = _model_path(paths, spec.encoder_name)
    if precomputed is None:
        resolved_encoder = encoder or _load_encoder(model_path, spec.encoder_device)
        train = _encode(resolved_encoder, views.train, batch_size)
        calibration = _encode(resolved_encoder, views.calibration, batch_size)
        test = _encode(resolved_encoder, views.test, batch_size)
    else:
        _, train, calibration, test = precomputed
    train_labels = np.asarray([str(row["intent"]) for row in views.train], dtype=object)
    calibration_scores, test_scores, _calibration_nearest, test_nearest, method_details = _native_scores(
        spec, train, calibration, test, train_labels
    )
    selection = _known_only_threshold(calibration_scores)
    metrics = _breakdown(views.test, test_scores, selection.threshold)
    metrics["combined"].update(
        {
            "scoring_seconds": time.perf_counter() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
    )
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_text(temporary / "resolved_config.yaml", yaml.safe_dump(config, allow_unicode=True, sort_keys=True))
        atomic_write_json(temporary / "metrics.json", metrics)
        atomic_write_json(
            temporary / "threshold_selection.json",
            {
                "type": "known_only_conformal",
                "alpha": selection.alpha,
                "threshold": selection.threshold,
                "known_calibration_count": selection.calibration_count,
                "order_statistic_rank": selection.order_statistic_rank,
                "test_used_for_selection": False,
            },
        )
        atomic_write_jsonl(temporary / "predictions" / "test.jsonl", _prediction_rows(views.test, test_scores, selection.threshold, test_nearest))
        atomic_write_json(
            temporary / "environment.json",
            {
                **environment_snapshot(paths.project_root),
                "encoder": _model_fingerprint(model_path),
                "encoder_device": spec.encoder_device,
            },
        )
        atomic_write_json(
            temporary / "manifest.json",
            {
                "status": "complete",
                "run_id": spec.run_id,
                "config": config,
                "config_hash": config_hash,
                "method_availability": availability.as_json(),
                "method_details": method_details,
                "registry_export_contract": contract,
                "input_hashes": input_hashes,
                "elapsed_seconds": time.perf_counter() - started,
                "test_used_for_selection": False,
                "historical_artifacts_overwritten": False,
                "metrics_emitted": True,
                "uses_oos_for_training": False,
                "uses_oos_for_calibration": False,
                "embedding_sha256": {
                    "train": hashlib.sha256(train.tobytes(order="C")).hexdigest(),
                    "calibration": hashlib.sha256(calibration.tobytes(order="C")).hexdigest(),
                    "test": hashlib.sha256(test.tobytes(order="C")).hexdigest(),
                },
                "embedding_reused_within_matrix": precomputed is not None,
            },
        )
    return run_dir, "complete"


def run_matrix(
    paths: ProtocolV2Paths,
    specs: Iterable[ExternalBaselineSpec],
    *,
    execute: bool,
    resume: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Run an explicit E4 viability matrix without treating blocks as failures."""
    spec_list = list(specs)
    if execute:
        for dataset in {spec.dataset for spec in spec_list}:
            paths.require_experiment_admission(dataset)
    summary: dict[str, Any] = {"planned": 0, "complete": [], "blocked": [], "unsupported": [], "failed": [], "dry_run": []}
    cached_key: tuple[str, float, int, str, str] | None = None
    cached_payload: tuple[GateViews, np.ndarray, np.ndarray, np.ndarray] | None = None
    cached_encoder: Any | None = None
    for spec in spec_list:
        summary["planned"] += 1
        try:
            if not execute:
                summary["dry_run"].append(dry_run(paths, spec))
                continue
            precomputed = None
            if method_availability(spec.method).state == "available":
                cache_key = (spec.dataset, spec.kir, spec.seed, spec.encoder_name, spec.encoder_device)
                if cache_key != cached_key:
                    # Config generation keeps methods contiguous for one
                    # dataset/KIR/seed.  Reuse the expensive MiniLM encoding
                    # across MSP/Energy/kNN/LOF, then release it on the next
                    # key so a large sweep does not retain every embedding.
                    views = load_gate_views(paths, spec.dataset, spec.seed, spec.kir)
                    model_path = _model_path(paths, spec.encoder_name)
                    cached_encoder = _load_encoder(model_path, spec.encoder_device)
                    cached_payload = (
                        views,
                        _encode(cached_encoder, views.train, batch_size),
                        _encode(cached_encoder, views.calibration, batch_size),
                        _encode(cached_encoder, views.test, batch_size),
                    )
                    cached_key = cache_key
                precomputed = cached_payload
            run_dir, status = run_external_baseline(
                paths, spec, batch_size=batch_size, resume=resume, precomputed=precomputed
            )
            summary[status].append(run_dir.relative_to(paths.run_root).as_posix())
        except Exception as exc:  # One upstream adapter must not hide others.
            summary["failed"].append({"run_id": spec.run_id, "error_type": type(exc).__name__, "error": str(exc)})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly execute native controls or write blocked manifests")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--kir", type=float, action="append")
    parser.add_argument("--method", action="append")
    parser.add_argument("--smoke", action="store_true", help="Require the declared CLINC150/KIR0.50/seed0 viability cell")
    args = parser.parse_args(argv)
    specs = load_external_baseline_matrix(args.config)
    if args.smoke:
        specs = [spec for spec in specs if spec.dataset == "clinc150" and spec.kir == 0.5 and spec.seed == 0]
        if not specs:
            raise ValueError("--smoke found no clinc150/KIR0.50/seed0 cell in the matrix")
    specs = [
        spec
        for spec in specs
        if (not args.dataset or spec.dataset in args.dataset)
        and (not args.seed or spec.seed in args.seed)
        and (not args.kir or spec.kir in args.kir)
        and (not args.method or spec.method in args.method)
    ]
    paths = ProtocolV2Paths.discover()
    summary = run_matrix(paths, specs, execute=args.execute, resume=args.resume, batch_size=args.batch_size)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
