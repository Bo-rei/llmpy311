"""Executable StackOverflow RC-AMBL pilot runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file, sha256_json  # noqa: E402
from protocol_v2.data.manifests import dataset_manifest_path, read_json, view_manifest_path  # noqa: E402
from protocol_v2.data.registry import registry_path  # noqa: E402
from protocol_v2.experiments.matrix import GateRunSpec  # noqa: E402
from protocol_v2.experiments.runner import _canonical_embedding_cache, _embedding_cache, _model_fingerprint, _model_path  # noqa: E402
from protocol_v2.gate.view_loader import load_gate_views  # noqa: E402
from protocol_v2.runtime.paths import ProtocolV2Paths  # noqa: E402
from protocol_v2.tracking.provenance import file_hashes  # noqa: E402

from .calibration import ids_hash, split_calibration_rows  # noqa: E402
from .contracts import AdaptiveConfig  # noqa: E402
from .evidence import EvidenceModel  # noqa: E402
from .selection import fit_rc_ambl  # noqa: E402

DATASET = "stackoverflow"
KIR = 0.50
SEEDS = (13, 42, 87)
DISTANCE = "mahalanobis_diag"
RUN_REVISION = "contract_repair5"


def _hash_rows(rows: list[dict[str, Any]]) -> str:
    return sha256_json([str(row["sample_id"]) for row in rows])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    from io import StringIO
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(row for row in rows)
    atomic_write_text(path, output.getvalue())


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _load_inputs(paths: ProtocolV2Paths, seed: int):
    paths.require_experiment_admission(DATASET)
    views = load_gate_views(paths, DATASET, seed, KIR)
    registry_file = registry_path(paths, DATASET, seed, KIR)
    registry = read_json(registry_file)
    canonical_manifest = dataset_manifest_path(paths.manifest_root, DATASET)
    view_manifest = view_manifest_path(paths.manifest_root, DATASET, seed, KIR)
    model = _model_fingerprint(_model_path(paths, "all-MiniLM-L6-v2"))
    canonical = _canonical_embedding_cache(paths, DATASET, model, None, 128)
    spec = GateRunSpec("adaptive_v1", DATASET, KIR, seed, 1, DISTANCE, "frozen_minilm", "mean_std", 1.0, "all-MiniLM-L6-v2", "cpu", paths.dataset_version)
    train, train_meta = _embedding_cache(paths, spec, "train_known", views.train, registry["registry_sha256"], sha256_file(canonical_manifest), model, canonical)
    calibration, calibration_meta = _embedding_cache(paths, spec, "calibration_known", views.calibration, registry["registry_sha256"], sha256_file(canonical_manifest), model, canonical)
    test, test_meta = _embedding_cache(paths, spec, "test_combined", views.test, registry["registry_sha256"], sha256_file(canonical_manifest), model, canonical)
    # The persisted cache is the raw frozen MiniLM output.  The active E2
    # contract applies L2 normalization inside the detector, so RC-AMBL must
    # normalize the same three arrays before fitting covariance or evidence.
    def normalize(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return array / np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)
    train = normalize(train)
    calibration = normalize(calibration)
    test = normalize(test)
    select_rows, threshold_rows, split_audit = split_calibration_rows(views.calibration, seed)
    select_indices = [views.calibration.index(row) for row in select_rows]
    threshold_indices = [views.calibration.index(row) for row in threshold_rows]
    split_values = {"select": calibration[select_indices], "threshold": calibration[threshold_indices]}
    split_intents = {"select": np.asarray([str(row["intent"]) for row in select_rows], dtype=object), "threshold": np.asarray([str(row["intent"]) for row in threshold_rows], dtype=object)}
    hashes = file_hashes({"registry": registry_file, "canonical_manifest": canonical_manifest, "view_manifest": view_manifest, "export_manifest": views.export_root / "export_manifest.json"})
    input_meta = {"registry_sha256": registry["registry_sha256"], "canonical_manifest_sha256": hashes["canonical_manifest"], "view_manifest_sha256": hashes["view_manifest"], "export_manifest_sha256": hashes["export_manifest"], "cache_embedding_sha256": {"train": train_meta.get("embedding_sha256"), "calibration": calibration_meta.get("embedding_sha256"), "test": test_meta.get("embedding_sha256")}, "cache_sample_ids_sha256": {"train": train_meta.get("sample_ids_sha256"), "calibration": calibration_meta.get("sample_ids_sha256"), "test": test_meta.get("sample_ids_sha256")}, "split_ids_sha256": {"train": _hash_rows(views.train), "calibration": _hash_rows(views.calibration), "test": _hash_rows(views.test), "calibration_select": ids_hash(select_rows), "calibration_threshold": ids_hash(threshold_rows)}, "calibration_split_audit": split_audit, "test_used_for_selection": False, "cache_policy": "reuse_only_no_implicit_encoding", "model": model}
    train_intents = np.asarray([str(row["intent"]) for row in views.train], dtype=object)
    return views, train, train_intents, split_values, split_intents, test, input_meta


def _metrics(rows: list[dict[str, Any]], output: Any) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    predicted_oos = np.asarray(output.predicted_oos, dtype=np.int64)
    scores = np.asarray(output.oos_score, dtype=np.float64)
    known = labels == 0
    oos = labels == 1
    tp = int(np.sum((predicted_oos == 1) & oos))
    fp = int(np.sum((predicted_oos == 1) & known))
    fn = int(np.sum((predicted_oos == 0) & oos))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_u = 2 * precision * recall / max(precision + recall, 1e-12)
    gold = [str(row["intent"]) if not int(row["label"]) else "__oos__" for row in rows]
    predicted = ["__oos__" if int(flag) else str(intent) for flag, intent in zip(predicted_oos, output.top_intent)]
    known_intents = sorted({str(row["intent"]) for row in rows if int(row["label"]) == 0})
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        fpr95 = float(np.min(fpr[tpr >= 0.95])) if np.any(tpr >= 0.95) else 1.0
    except ValueError:
        fpr95 = 1.0
    return {"oos_precision": float(precision), "oos_recall": float(recall), "oos_f1": float(f1_u), "known_recall": float(np.mean(predicted_oos[known] == 0)), "known_macro_f1": float(f1_score(gold, predicted, labels=known_intents, average="macro", zero_division=0)), "f1_all": float(f1_score(gold, predicted, labels=known_intents + ["__oos__"], average="macro", zero_division=0)), "f1_k": float(f1_score(gold, predicted, labels=known_intents, average="macro", zero_division=0)), "f1_u": float(f1_u), "accuracy": float(accuracy_score(gold, predicted)), "false_accept_rate": float(np.mean(predicted_oos[oos] == 0)), "false_reject_rate": float(np.mean(predicted_oos[known] == 1)), "auroc": float(roc_auc_score(labels, scores)), "aupr_oos": float(average_precision_score(labels, scores)), "fpr95": fpr95}


def _proxy_rows(select_values: np.ndarray, select_intents: np.ndarray, train_intents: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    intents = sorted(set(train_intents.tolist()))
    rng = np.random.default_rng(np.random.SeedSequence([314159, int(seed)]))
    pairs = []
    for index in range(5):
        order = rng.permutation(len(intents))
        pairs.append((intents[int(order[2 * index])], intents[int(order[2 * index + 1])]))
    # The selected half of the Known calibration rows is used only as a
    # proxy evaluation pool; it never enters final threshold fitting.  The
    # five pairs are disjoint by construction and are recorded as episodes.
    proxy_mask = np.zeros(select_intents.shape[0], dtype=bool)
    episodes: list[dict[str, Any]] = []
    for i, pair in enumerate(pairs):
        mask = np.isin(select_intents.astype(str), np.asarray(pair, dtype=str))
        proxy_mask |= mask
        episodes.append({"episode": i, "hidden_intents": list(pair), "calibration_rows": int(np.sum(mask)), "train_hidden_rows_excluded": True})
    return np.asarray(select_values)[proxy_mask], np.asarray(select_intents)[proxy_mask], episodes


def run_cell(paths: ProtocolV2Paths, seed: int, output_root: Path, *, resume: bool = True) -> dict[str, Any]:
    run_dir = output_root / "runs" / DATASET / f"seed_{seed}"
    manifest_path = run_dir / "run_manifest.json"
    if resume and manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "complete":
        return {"seed": seed, "status": "skipped_existing"}
    started = time.time()
    views, train, train_intents, split_values, split_intents, test, inputs = _load_inputs(paths, seed)
    config = AdaptiveConfig(seed=seed)
    proxy_values, proxy_intents, proxy_episodes = _proxy_rows(split_values["select"], split_intents["select"], train_intents, seed)
    rows: list[dict[str, Any]] = []
    methods = []
    for mode in ("KnownOnly", "ProxyOOS"):
        fit = fit_rc_ambl(train, train_intents, split_values["select"], split_intents["select"], split_values["threshold"], mode=mode, config=config, proxy_values=proxy_values if mode == "ProxyOOS" else None, proxy_intents=proxy_intents if mode == "ProxyOOS" else None)
        model = EvidenceModel(fit.centers, fit.parents, fit.thresholds)
        inference_started = time.perf_counter()
        output = model.apply(test)
        inference_seconds = time.perf_counter() - inference_started
        metric = _metrics(views.test, output)
        metric.update({"experiment_id": "adaptive_v1", "run_id": f"adaptive_v1__{DATASET}__kir_0.50__seed_{seed}__{mode}", "protocol_version": paths.dataset_version, "dataset": DATASET, "kir": KIR, "seed": seed, "method": f"RC-AMBL-{mode}", "representation": config.representation, "distance": config.distance, "radius_method": config.radius_method, "radius_lambda": config.radius_lambda, "threshold_source": fit.thresholds.threshold_source, "tau": fit.thresholds.tau, "tau_parent": fit.thresholds.tau_parent, "delta": fit.thresholds.delta, "mean_k_y": float(np.mean([len(v) for v in fit.centers.values()])), "total_centers": int(sum(len(v) for v in fit.centers.values())), "accepted_splits": int(sum(op.split_accepted for op in fit.operations)), "rejected_splits": int(sum(not op.split_accepted for op in fit.operations)), "bootstrap_stability": float(np.mean([op.stability_median for op in fit.operations])) if fit.operations else 1.0, "fit_seconds": float(time.time() - started), "inference_seconds": float(inference_seconds), "trainable_parameters": 0, "test_used_for_selection": False, "source": "new_run"})
        rows.append(metric)
        methods.append((mode, fit, output))
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(run_dir / f"{mode}_config.json", {"config": config.__dict__, "mode": mode, "inputs": inputs})
        atomic_write_json(run_dir / f"{mode}_selection_audit.json", fit.selection_audit)
        atomic_write_json(run_dir / f"{mode}_thresholds.json", fit.thresholds.__dict__)
        atomic_write_json(run_dir / f"{mode}_proxy_episodes.json", proxy_episodes if mode == "ProxyOOS" else [])
        center_rows = []
        for intent, centers in fit.centers.items():
            for center in centers:
                center_rows.append({"intent": intent, "local_id": center.local_id, "sample_count": center.sample_count, "radius": center.radius, "stability": center.stability, "parent_local_id": center.parent_local_id, "birth_round": center.birth_round, "center": center.center.tolist(), "inv_diag_cov": center.inv_diag_cov.tolist()})
        atomic_write_json(run_dir / f"{mode}_centers.json", center_rows)
        _write_csv(run_dir / f"{mode}_operations.csv", [{"round": op.round_index, "intent": op.intent, "parent_local_id": op.parent_local_id, "child_sizes": list(op.candidate_child_sizes), "compactness_gain": op.compactness_gain, "complexity_adjusted_gain": op.complexity_adjusted_gain, "stability_median": op.stability_median, "rho": op.rho, "known_recall_delta": op.known_recall_delta, "ambiguity_delta": op.ambiguity_delta, "proxy_false_accept_delta": op.proxy_false_accept_delta, "split_accepted": op.split_accepted, "reject_reason": op.reject_reason} for op in fit.operations])
        pred_rows = []
        for i, row in enumerate(views.test):
            pred_rows.append({"sample_id_hash": hashlib.sha256(str(row["sample_id"]).encode()).hexdigest(), "gold_intent": row["intent"], "gold_is_oos": row["label"], "predicted_is_oos": int(output.predicted_oos[i]), "predicted_intent": str(output.top_intent[i]), "energy": float(output.energy[i]), "parent_score": float(output.parent_score[i]), "margin": float(output.gap[i]), "oos_score": float(output.oos_score[i])})
        _write_csv(run_dir / f"{mode}_predictions.csv", pred_rows)
    _write_csv(run_dir / "metrics.csv", rows)
    atomic_write_json(run_dir / "provenance.json", {"git_commit": _git(["rev-parse", "HEAD"]), "git_dirty": bool(_git(["status", "--porcelain"])), "inputs": inputs, "python": platform.python_version(), "numpy": np.__version__, "test_used_for_selection": False})
    atomic_write_json(manifest_path, {"run_id": f"adaptive_v1__{DATASET}__kir_0.50__seed_{seed}", "experiment_id": "adaptive_v1", "dataset": DATASET, "kir": KIR, "seed": seed, "status": "complete", "methods": [row["method"] for row in rows], "test_used_for_selection": False, "inputs": inputs})
    return {"seed": seed, "status": "complete", "metrics": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--kir", type=float, default=KIR)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate the fixed pilot inputs without writing runs")
    args = parser.parse_args(argv)
    if args.dataset != DATASET or abs(args.kir - KIR) > 1e-9:
        raise SystemExit("adaptive_v1 pilot is fixed to stackoverflow and KIR=0.50")
    paths = ProtocolV2Paths.discover()
    if args.dry_run:
        paths.require_experiment_admission(DATASET)
        views = load_gate_views(paths, DATASET, args.seed, KIR)
        registry_file = registry_path(paths, DATASET, args.seed, KIR)
        print(json.dumps({"status": "dry_run", "dataset": DATASET, "kir": KIR, "seed": args.seed, "train_rows": len(views.train), "calibration_rows": len(views.calibration), "test_rows": len(views.test), "registry": str(registry_file), "artifact_root": str(args.artifact_root or (paths.run_root / "adaptive_v1" / RUN_REVISION)), "writes": False}, ensure_ascii=False, sort_keys=True))
        return 0
    root = args.artifact_root or (paths.run_root / "adaptive_v1" / RUN_REVISION)
    result = run_cell(paths, args.seed, root, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
