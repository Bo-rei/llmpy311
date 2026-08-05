"""RACAL-v1 stage 2: a pure Trainable-MiniLM K=1 versus fixed K=2 control.

This module deliberately contains no adaptive selection, proxy-OOS training,
threshold tuning, or risk gate.  It only reloads the immutable stage-1
checkpoint for a seed, re-encodes the fixed protocol views, fits the historical
nearest-sphere detector with one or two centres per intent, and records the
paired diagnostics needed to decide whether fixed K=2 is safe.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from transformers import AutoTokenizer

from protocol_v2.data.hashing import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file, sha256_json
from protocol_v2.experiments.mechanism_runner import E3Bundle, load_e2_bundle
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from protocol_v2.runtime.paths import ProtocolV2Paths
from protocol_v2.tracking.run_manifest import atomic_run_directory

from .boundary import evaluate_open, detector_signature
from .contracts import DATASET, KIR, rows_hash, validate_bundle
from .representation import RacalMiniLM, choose_device, encode_rows


STAGE2 = "racal_v1_stage2_fixed_k2"
STAGE1 = "racal_v1"
SEEDS = (13, 42, 87)
K_VALUES = (1, 2)
DISTANCE = "mahalanobis_diag"
RADIUS_METHOD = "mean_std"
RADIUS_LAMBDA = 1.0
THRESHOLD = 1.0
PARTITION_SEED = 42


def stage2_root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / STAGE1 / "stage2_fixed_k2"


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        number = value.item()
        return number if not isinstance(number, float) or math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    if not fields:
        atomic_write_text(path, "")
        return
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: _safe(row.get(field, "")) for field in fields} for row in rows)
    temporary.replace(path)


def _stage1_run_dir(paths: ProtocolV2Paths, seed: int) -> Path:
    return stage2_root(paths).parent / "runs" / "trainable_k1" / f"seed_{seed}"


def _fit_fixed_detector(train: np.ndarray, rows: Sequence[Mapping[str, Any]], k: int) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=int(k),
        radius_method=RADIUS_METHOD,
        radius_lambda=RADIUS_LAMBDA,
        distance_metric=DISTANCE,
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=PARTITION_SEED,
        acceptance_mode="nearest_sphere",
    )
    detector.fit(np.asarray(train), np.asarray([str(row["intent"]) for row in rows], dtype=object))
    return detector


def _load_stage1_model(paths: ProtocolV2Paths, seed: int, stage1_config: Mapping[str, Any], device: torch.device) -> tuple[RacalMiniLM, Any, dict[str, Any]]:
    run_dir = _stage1_run_dir(paths, seed)
    manifest_path = run_dir / "training_manifest.json"
    checkpoint = run_dir / "checkpoint.pt"
    if not manifest_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Missing stage-1 Trainable K=1 checkpoint for seed {seed}: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("test_used_for_selection") is not False:
        raise ValueError(f"Stage-1 checkpoint is not a completed Known-only run: {manifest_path}")
    expected_hash = str(manifest.get("checkpoint_sha256", ""))
    actual_hash = sha256_file(checkpoint)
    if expected_hash != actual_hash:
        raise ValueError(f"Stage-1 checkpoint hash changed for seed {seed}: {checkpoint}")
    mode = str(manifest.get("freeze_report", {}).get("mode", "last2_minilm_plus_projection"))
    hidden_dim = int(manifest.get("projection_hidden_dim", stage1_config.get("projection_hidden_dim", 256)))
    model_path = (paths.project_root / str(stage1_config["model_path"])).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = RacalMiniLM(model_path, mode, hidden_dim).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer, {"manifest": manifest, "checkpoint": str(checkpoint), "checkpoint_sha256": actual_hash, "mode": mode, "hidden_dim": hidden_dim}


def _verify_stage1_input(manifest: Mapping[str, Any], bundle: E3Bundle) -> dict[str, Any]:
    expected = manifest.get("input", {})
    actual = {
        "train_sample_ids_sha256": rows_hash(bundle.views.train),
        "calibration_sample_ids_sha256": rows_hash(bundle.views.calibration),
        "test_sample_ids_sha256": rows_hash(bundle.views.test),
        "registry_sha256": bundle.e2_manifest.get("registry_sha256"),
        "canonical_manifest_sha256": bundle.e2_manifest.get("canonical_manifest_sha256"),
    }
    mismatches = {
        key: {"expected": expected.get(key), "actual": value}
        for key, value in actual.items()
        if expected.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Stage-1 view/provenance mismatch: {mismatches}")
    return {"expected": {key: expected.get(key) for key in actual}, "actual": actual, "mismatch_count": 0}


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any], keys: Sequence[str]) -> dict[str, float]:
    return {key: float(right[key]) - float(left[key]) for key in keys}


def _selected_intent(detector: MultiSphereOOSDetector, prediction: Mapping[str, Any]) -> str:
    cluster = int(prediction["nearest_cluster"])
    return str(detector.cluster_to_intent.get(cluster, "__unknown__"))


def _enrich_predictions(detector: MultiSphereOOSDetector, predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in predictions:
        item = dict(row)
        item["selected_intent"] = _selected_intent(detector, row)
        item["selected_center"] = int(row["nearest_cluster"])
        item["normalized_score"] = float(row["oos_score"])
        enriched.append(item)
    return enriched


def _sample_audit(rows: Sequence[Mapping[str, Any]], pred_k1: Sequence[Mapping[str, Any]], pred_k2: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, left, right in zip(rows, pred_k1, pred_k2, strict=True):
        gold_is_oos = int(row["label"]) == 1
        k1_oos = int(left["predicted_is_oos"]) == 1
        k2_oos = int(right["predicted_is_oos"]) == 1
        categories: list[str] = []
        if gold_is_oos and not k1_oos and not k2_oos:
            categories.append("both_false_accept_oos")
        if gold_is_oos and k1_oos and not k2_oos:
            categories.append("k1_correct_reject_k2_wrong_accept_oos")
        if not gold_is_oos and k1_oos and not k2_oos:
            categories.append("k1_false_reject_k2_correct_known")
        if int(left["nearest_cluster"]) != int(right["nearest_cluster"]) or left["predicted_intent"] != right["predicted_intent"]:
            categories.append("k2_changed_center_or_intent")
        if not categories:
            categories.append("unchanged_or_other")
        output.append({
            "sample_id": row["sample_id"],
            "gold_intent": row["intent"],
            "gold_is_oos": int(row["label"]),
            "categories": categories,
            "k1_selected_intent": left["selected_intent"],
            "k2_selected_intent": right["selected_intent"],
            "k1_selected_center": int(left["selected_center"]),
            "k2_selected_center": int(right["selected_center"]),
            "k1_distance": float(left["distance"]),
            "k2_distance": float(right["distance"]),
            "k1_radius": float(left["radius"]),
            "k2_radius": float(right["radius"]),
            "k1_normalized_score": float(left["normalized_score"]),
            "k2_normalized_score": float(right["normalized_score"]),
            "k1_prediction": int(left["predicted_is_oos"]),
            "k2_prediction": int(right["predicted_is_oos"]),
            "k1_predicted_intent": left["predicted_intent"],
            "k2_predicted_intent": right["predicted_intent"],
        })
    return output


def _sphere_stats(detector: MultiSphereOOSDetector, train_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intents = sorted({str(row["intent"]) for row in train_rows})
    result: dict[str, Any] = {}
    for intent in intents:
        cluster_ids = [int(value) for value in detector.intent_to_clusters[intent]]
        intent_row: dict[str, Any] = {"cluster_ids": cluster_ids, "spheres": []}
        for cluster_id in cluster_ids:
            mask = detector._train_cluster_labels == cluster_id
            sphere = next(item for item in detector.spheres if int(item.cluster_id) == cluster_id)
            points = detector._train_embeddings[mask]
            diff = points - sphere.center
            if sphere.inv_diag_cov is not None:
                distances = np.sqrt(np.sum((diff ** 2) * sphere.inv_diag_cov, axis=1))
            else:
                distances = np.linalg.norm(diff, axis=1)
            intent_row["spheres"].append({
                "cluster_id": cluster_id,
                "sample_count": int(mask.sum()),
                "radius": float(sphere.radius),
                "distance_variance": float(np.var(distances)),
                "feature_variance_trace": float(np.var(points, axis=0).sum()),
                "mean_distance": float(np.mean(distances)),
            })
        result[intent] = intent_row
    return result


def _bootstrap_and_silhouette(points: np.ndarray, seed: int, repetitions: int) -> tuple[float | None, float | None, str | None]:
    if points.shape[0] < 4 or np.allclose(np.var(points, axis=0), 0.0):
        return None, None, "insufficient_or_constant_points"
    base = KMeans(n_clusters=2, random_state=PARTITION_SEED, n_init=10).fit(points)
    aris: list[float] = []
    rng = np.random.default_rng(seed + 17011)
    for _ in range(repetitions):
        indices = rng.integers(0, points.shape[0], size=points.shape[0])
        boot = KMeans(n_clusters=2, random_state=int(rng.integers(0, 2**31 - 1)), n_init=10).fit(points[indices])
        aris.append(float(adjusted_rand_score(base.labels_, boot.predict(points))))
    try:
        silhouette = float(silhouette_score(points, base.labels_))
    except ValueError as exc:
        silhouette = None
        reason = f"silhouette_undefined:{exc}"
    else:
        reason = None
    return float(np.mean(aris)) if aris else None, silhouette, reason


def _intent_diagnostics(
    detector_k1: MultiSphereOOSDetector,
    detector_k2: MultiSphereOOSDetector,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    pred_k1: Sequence[Mapping[str, Any]],
    pred_k2: Sequence[Mapping[str, Any]],
    seed: int,
    bootstrap_repetitions: int,
) -> list[dict[str, Any]]:
    train_labels = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    sphere_k1 = _sphere_stats(detector_k1, train_rows)
    sphere_k2 = _sphere_stats(detector_k2, train_rows)
    rows: list[dict[str, Any]] = []
    for intent in sorted(set(train_labels.tolist())):
        train_mask = train_labels == intent
        points = detector_k2._train_embeddings[train_mask]
        ari, silhouette, silhouette_reason = _bootstrap_and_silhouette(points, seed, bootstrap_repetitions)
        test_known = [i for i, row in enumerate(test_rows) if int(row["label"]) == 0 and str(row["intent"]) == intent]
        test_oos = [i for i, row in enumerate(test_rows) if int(row["label"]) == 1]
        k1_known_reject = sum(int(pred_k1[i]["predicted_is_oos"]) == 1 for i in test_known)
        k2_known_reject = sum(int(pred_k2[i]["predicted_is_oos"]) == 1 for i in test_known)
        newly_accepted_oos = sum(int(pred_k1[i]["predicted_is_oos"]) == 1 and int(pred_k2[i]["predicted_is_oos"]) == 0 and str(pred_k2[i]["selected_intent"]) == intent for i in test_oos)
        recovered_known = sum(int(pred_k1[i]["predicted_is_oos"]) == 1 and int(pred_k2[i]["predicted_is_oos"]) == 0 for i in test_known)
        k1_sphere = sphere_k1[intent]["spheres"][0]
        k2_spheres = sphere_k2[intent]["spheres"]
        rows.append({
            "dataset": DATASET,
            "kir": KIR,
            "seed": seed,
            "intent": intent,
            "train_sample_count": int(train_mask.sum()),
            "k1_radius": k1_sphere["radius"],
            "k1_distance_variance": k1_sphere["distance_variance"],
            "k2_cluster_count": len(k2_spheres),
            "k2_cluster_1_sample_count": k2_spheres[0]["sample_count"],
            "k2_cluster_2_sample_count": k2_spheres[1]["sample_count"],
            "k2_cluster_1_radius": k2_spheres[0]["radius"],
            "k2_cluster_2_radius": k2_spheres[1]["radius"],
            "k2_cluster_1_distance_variance": k2_spheres[0]["distance_variance"],
            "k2_cluster_2_distance_variance": k2_spheres[1]["distance_variance"],
            "bootstrap_ari_mean": ari,
            "silhouette": silhouette,
            "silhouette_reason": silhouette_reason or "defined",
            "known_reject_count_k1": k1_known_reject,
            "known_reject_count_k2": k2_known_reject,
            "known_recall_delta": -float(k2_known_reject - k1_known_reject) / max(len(test_known), 1),
            "newly_accepted_oos_count": newly_accepted_oos,
            "recovered_known_count": recovered_known,
            "net_benefit_recovered_known_minus_new_oos": recovered_known - newly_accepted_oos,
        })
    return rows


def _hash_sample_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        digest = hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest()
        output.append({
            "sample_id_sha256": digest,
            "gold_is_oos": row["gold_is_oos"],
            "categories": "|".join(row["categories"]),
            "k1_prediction": row["k1_prediction"],
            "k2_prediction": row["k2_prediction"],
            "k1_selected_intent": row["k1_selected_intent"],
            "k2_selected_intent": row["k2_selected_intent"],
        })
    return output


def _compare_stage1_metrics(stage1_metrics: Mapping[str, Any], current_metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos", "fpr95")
    deltas = {key: abs(float(stage1_metrics[key]) - float(current_metrics[key])) for key in keys}
    return {"metric_abs_delta": deltas, "metric_max_abs_delta": max(deltas.values()), "within_tolerance": max(deltas.values()) <= 1e-10}


def run_seed(paths: ProtocolV2Paths, config: Mapping[str, Any], seed: int, resume: bool = False) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"RACAL stage2 seeds are restricted to {SEEDS}")
    root = stage2_root(paths)
    bundle = load_e2_bundle(paths, DATASET, seed, KIR)
    view_contract = validate_bundle(bundle)
    stage1_config_path = (paths.project_root / str(config["stage1_config_path"])).resolve()
    stage1_config = yaml.safe_load(stage1_config_path.read_text(encoding="utf-8"))
    stage1_dir = _stage1_run_dir(paths, seed)
    stage1_manifest = json.loads((stage1_dir / "training_manifest.json").read_text(encoding="utf-8"))
    input_check = _verify_stage1_input(stage1_manifest, bundle)
    model, tokenizer, checkpoint_info = _load_stage1_model(paths, seed, stage1_config, choose_device(str(config.get("device", "auto"))))
    device = next(model.parameters()).device
    train_values = encode_rows(model, tokenizer, bundle.views.train, device, int(config["batch_size"]), int(config["max_length"]))
    calibration_values = encode_rows(model, tokenizer, bundle.views.calibration, device, int(config["batch_size"]), int(config["max_length"]))
    test_values = encode_rows(model, tokenizer, bundle.views.test, device, int(config["batch_size"]), int(config["max_length"]))
    stage1_metrics = json.loads((stage1_dir / "metrics.json").read_text(encoding="utf-8"))
    run_dir = root / "runs" / f"seed_{seed}"
    config_payload = {
        "stage": STAGE2,
        "protocol_version": "protocol_v2_textoir_v1",
        "dataset": DATASET,
        "kir": KIR,
        "seed": seed,
        "representation": "stage1_trainable_k1_checkpoint_reused",
        "checkpoint_sha256": checkpoint_info["checkpoint_sha256"],
        "distance": DISTANCE,
        "radius_method": RADIUS_METHOD,
        "radius_lambda": RADIUS_LAMBDA,
        "threshold": THRESHOLD,
        "partition_seed": PARTITION_SEED,
        "k_values": list(K_VALUES),
        "input": {"view_contract": view_contract, "stage1_input_check": input_check, "train_rows_hash": rows_hash(bundle.views.train), "calibration_rows_hash": rows_hash(bundle.views.calibration), "test_rows_hash": rows_hash(bundle.views.test)},
        "embedding_hashes": {"train": hashlib.sha256(np.ascontiguousarray(train_values).tobytes()).hexdigest(), "calibration": hashlib.sha256(np.ascontiguousarray(calibration_values).tobytes()).hexdigest(), "test": hashlib.sha256(np.ascontiguousarray(test_values).tobytes()).hexdigest()},
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    config_hash = sha256_json(config_payload)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume and existing.get("config_hash") == config_hash and existing.get("status") == "complete":
            return existing
        raise FileExistsError(f"Stage2 run already exists or has a different config: {run_dir}")
    detector_k1 = _fit_fixed_detector(train_values, bundle.views.train, 1)
    detector_k2 = _fit_fixed_detector(train_values, bundle.views.train, 2)
    metrics_k1, predictions_k1 = evaluate_open(detector_k1, test_values, bundle.views.test, THRESHOLD)
    metrics_k2, predictions_k2 = evaluate_open(detector_k2, test_values, bundle.views.test, THRESHOLD)
    predictions_k1 = _enrich_predictions(detector_k1, predictions_k1)
    predictions_k2 = _enrich_predictions(detector_k2, predictions_k2)
    stage1_replay = _compare_stage1_metrics(stage1_metrics, metrics_k1)
    if not stage1_replay["within_tolerance"]:
        raise RuntimeError(f"Stage2 re-encoded K=1 no longer matches stage1 metrics for seed {seed}: {stage1_replay}")
    audit = _sample_audit(bundle.views.test, predictions_k1, predictions_k2)
    intent_diagnostics = _intent_diagnostics(detector_k1, detector_k2, bundle.views.train, bundle.views.test, predictions_k1, predictions_k2, seed, int(config["bootstrap_repetitions"]))
    metric_keys = ("oos_f1", "oos_precision", "oos_recall", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos", "fpr95")
    delta = _metric_delta(metrics_k1, metrics_k2, metric_keys)
    metrics = {
        "k1": {key: float(metrics_k1[key]) for key in metric_keys},
        "k2": {key: float(metrics_k2[key]) for key in metric_keys},
        "k2_minus_k1": delta,
        "stage1_k1_replay": stage1_replay,
        "test_used_for_selection": False,
    }
    run_manifest = {**config_payload, "config_hash": config_hash, "status": "complete", "checkpoint": checkpoint_info, "stage1_manifest": str(stage1_dir / "training_manifest.json"), "stage1_replay": stage1_replay, "sample_audit_count": len(audit), "intent_diagnostic_count": len(intent_diagnostics), "bootstrap_repetitions": int(config["bootstrap_repetitions"]), "elapsed_seconds": 0.0}
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with atomic_run_directory(run_dir) as temporary:
        atomic_write_json(temporary / "resolved_config.json", _safe(config_payload))
        atomic_write_json(temporary / "metrics.json", _safe(metrics))
        atomic_write_json(temporary / "detector_signature_k1.json", _safe(detector_signature(detector_k1)))
        atomic_write_json(temporary / "detector_signature_k2.json", _safe(detector_signature(detector_k2)))
        atomic_write_jsonl(temporary / "predictions_k1.jsonl", predictions_k1)
        atomic_write_jsonl(temporary / "predictions_k2.jsonl", predictions_k2)
        atomic_write_jsonl(temporary / "sample_audit.jsonl", audit)
        _atomic_csv(temporary / "intent_diagnostics.csv", intent_diagnostics)
        atomic_write_json(temporary / "run_manifest.json", _safe({**run_manifest, "elapsed_seconds": time.time() - started}))
    return {**run_manifest, "elapsed_seconds": time.time() - started, "run_dir": str(run_dir)}


def load_stage2_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Stage2 config must be a mapping: {path}")
    if str(payload.get("protocol_version")) != "protocol_v2_textoir_v1" or str(payload.get("dataset", "")).lower() != DATASET or abs(float(payload.get("kir", -1)) - KIR) > 1e-12:
        raise ValueError("RACAL stage2 is restricted to protocol_v2_textoir_v1 StackOverflow KIR=0.50")
    if tuple(int(seed) for seed in payload.get("seeds", [])) != SEEDS:
        raise ValueError(f"RACAL stage2 must declare seeds {SEEDS}")
    if list(payload.get("k_values", [])) != list(K_VALUES):
        raise ValueError("RACAL stage2 must compare exactly K=1 and K=2")
    if str(payload.get("distance")) != DISTANCE or str(payload.get("radius_method")) != RADIUS_METHOD:
        raise ValueError("RACAL stage2 must use diagonal Mahalanobis and mean_std")
    if float(payload.get("radius_lambda")) != RADIUS_LAMBDA or float(payload.get("threshold")) != THRESHOLD:
        raise ValueError("RACAL stage2 radius/threshold contract is fixed")
    forbidden = ("proxy_oos", "risk_gate", "adaptive_k", "proxy_oos_training", "threshold_tuning")
    if any(bool(payload.get(key, False)) for key in forbidden):
        raise ValueError(f"Forbidden stage2 mechanism enabled: {forbidden}")
    required = ("stage1_config_path", "batch_size", "max_length", "bootstrap_repetitions")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Stage2 config is missing: {missing}")
    return {str(key): value for key, value in payload.items()}


def make_provenance(paths: ProtocolV2Paths, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    stage1_config_path = (paths.project_root / str(config["stage1_config_path"])).resolve()
    return {
        "schema_version": "s2c.racal_v1.stage2.provenance.v1",
        "stage": STAGE2,
        "protocol_version": "protocol_v2_textoir_v1",
        "dataset": DATASET,
        "kir": KIR,
        "seeds": list(SEEDS),
        "k_values": list(K_VALUES),
        "distance": DISTANCE,
        "radius_method": RADIUS_METHOD,
        "radius_lambda": RADIUS_LAMBDA,
        "threshold": THRESHOLD,
        "partition_seed": PARTITION_SEED,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "stage1_config_path": str(stage1_config_path),
        "stage1_config_sha256": sha256_file(stage1_config_path),
        "stage1_root": str(stage2_root(paths).parent),
        "historical_artifacts_immutable": True,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "third_party_policy": "record_only; do not modify third_party/mogb_official",
    }
