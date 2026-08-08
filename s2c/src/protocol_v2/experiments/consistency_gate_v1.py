"""Known-only prediction-consistency gate on top of RACAL Trainable K=1.

This stage deliberately does not add centres or retrain the encoder.  It tests
whether a single-centroid representation can reject unstable evidence by
requiring agreement across deterministic surface normalization and fixed
Monte-Carlo dropout views.  All gate choices are calibrated on Known
calibration rows only; test OOS rows are read only for the final report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from protocol_v2.data.hashing import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file
from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.experiments.mechanism_runner import load_e2_bundle
from protocol_v2.experiments.racal_v1.boundary import evaluate_open, fit_k1_detector
from protocol_v2.experiments.racal_v1.contracts import array_hash, load_config, rows_hash
from protocol_v2.experiments.racal_v1.representation import RacalMiniLM, choose_device, encode_rows, set_seed
from protocol_v2.runtime.paths import ProtocolV2Paths


STAGE = "consistency_gate_v1"
DATASET = "stackoverflow"
KIR = 0.50
SEEDS = (13, 42, 87)
N_DROPOUT_VIEWS = 2
THRESHOLD = 1.0


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        item = value.item()
        return item if not isinstance(item, float) or math.isfinite(item) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _safe(row.get(field, "")) for field in fields})
    temporary.replace(path)


def _root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / STAGE


def _model_path(paths: ProtocolV2Paths, config: Mapping[str, Any]) -> Path:
    return (paths.project_root / str(config["model_path"])).resolve()


def _checkpoint_path(paths: ProtocolV2Paths, config: Mapping[str, Any], seed: int) -> Path:
    return Path(str(config["checkpoint_root"]).replace("{seed}", str(seed))).resolve()


def _normalise_surface(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _view_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "text": _normalise_surface(str(row["text"]))} for row in rows]


def _encode_dropout(model: RacalMiniLM, tokenizer: Any, rows: Sequence[Mapping[str, Any]], device: torch.device, batch_size: int, max_length: int, seed: int) -> np.ndarray:
    set_seed(seed)
    model.train()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            tokens = tokenizer([str(row["text"]) for row in batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            chunks.append(model(tokens).detach().cpu().numpy().astype(np.float32))
    model.eval()
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 384), dtype=np.float32)


def _intent_scores(detector: Any, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest intent, best normalized distance and second-best margin."""
    intents = sorted(detector.intent_to_clusters)
    ratios = np.zeros((len(embeddings), len(intents)), dtype=np.float64)
    for row_idx, embedding in enumerate(embeddings):
        for intent_idx, intent in enumerate(intents):
            sphere_ids = detector.intent_to_clusters[intent]
            ratios[row_idx, intent_idx] = min(
                detector._distance(embedding, detector.spheres[int(sphere_id)])
                / max(float(detector.spheres[int(sphere_id)].radius), 1e-12)
                for sphere_id in sphere_ids
            )
    order = np.argsort(ratios, axis=1)
    best = order[:, 0]
    second = order[:, 1] if ratios.shape[1] > 1 else best
    margins = ratios[np.arange(len(ratios)), second] - ratios[np.arange(len(ratios)), best]
    return np.asarray([intents[int(index)] for index in best], dtype=object), ratios[np.arange(len(ratios)), best], margins


def _view_stats(detector: Any, embeddings: np.ndarray) -> dict[str, np.ndarray]:
    output = detector.predict_with_scores(embeddings)
    intents, evidence, margin = _intent_scores(detector, embeddings)
    return {**output, "intent": intents, "evidence": evidence, "margin": margin}


def _conflict_count(base: Mapping[str, np.ndarray], views: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
    conflicts = np.zeros(len(base["pred"]), dtype=np.int64)
    for view in views:
        conflicts += (view["intent"] != base["intent"]) | (view["pred"] != 0) | (base["pred"] != 0)
    return conflicts


def _known_recall(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else math.nan


def _select_known_only(base: Mapping[str, np.ndarray], views: Sequence[Mapping[str, np.ndarray]], target_drop: float) -> dict[str, Any]:
    conflicts = _conflict_count(base, views)
    base_recall = _known_recall((base["pred"] == 0) & (base["score"] <= THRESHOLD))
    target = max(0.0, base_recall - float(target_drop))
    margins = np.asarray(base["margin"], dtype=np.float64)
    candidates = sorted({0.0, *[float(np.quantile(margins, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90)]})
    choices: list[tuple[int, float, float]] = []
    for allowed in range(len(views) + 1):
        for delta in candidates:
            accepted = (base["pred"] == 0) & (base["score"] <= THRESHOLD) & (conflicts <= allowed) & (base["margin"] >= delta)
            recall = _known_recall(accepted)
            if recall >= target - 1e-12:
                choices.append((allowed, delta, recall))
    if not choices:
        return {"allowed_conflicts": len(views), "margin_delta": 0.0, "base_recall": base_recall, "target_recall": target, "selected_known_recall": 0.0, "selection_fallback": True}
    # Minimise tolerated conflicts first, then maximise the evidence margin.
    choices.sort(key=lambda item: (item[0], -item[1]))
    allowed, delta, recall = choices[0]
    return {"allowed_conflicts": int(allowed), "margin_delta": float(delta), "base_recall": base_recall, "target_recall": target, "selected_known_recall": recall, "selection_fallback": False}


def _metrics(rows: Sequence[Mapping[str, Any]], accepted: np.ndarray, intents: np.ndarray, scores: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    binary = compute_binary_oos_metrics(labels, scores, 1.0)
    known_intents = sorted({str(row["intent"]) for row in rows if int(row["label"]) == 0})
    truth = [str(row["intent"]) if int(row["label"]) == 0 else "__oos__" for row in rows]
    predicted = [str(intents[i]) if bool(accepted[i]) else "__oos__" for i in range(len(rows))]
    all_labels = [*known_intents, "__oos__"]
    result = {
        **binary,
        "f1_all": float(f1_score(truth, predicted, labels=all_labels, average="macro", zero_division=0)),
        "f1_u": float(f1_score(truth, predicted, labels=["__oos__"], average="macro", zero_division=0)),
        "f1_k": float(f1_score(truth, predicted, labels=known_intents, average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(truth, dtype=object) == np.asarray(predicted, dtype=object))),
        "known_recall": float(np.mean(accepted[labels == 0])) if np.any(labels == 0) else math.nan,
        "false_accept_rate": float(np.mean(accepted[labels == 1])) if np.any(labels == 1) else math.nan,
        "false_reject_rate": float(1.0 - np.mean(accepted[labels == 0])) if np.any(labels == 0) else math.nan,
    }
    predictions = [{"sample_id": row["sample_id"], "gold_intent": row["intent"], "gold_is_oos": int(row["label"]), "predicted_intent": predicted[i], "accepted_known": int(accepted[i]), "evidence_score": float(scores[i]), "evidence_margin": float(0.0)} for i, row in enumerate(rows)]
    return result, predictions


def _evaluate_variant(rows: Sequence[Mapping[str, Any]], base: Mapping[str, np.ndarray], views: Sequence[Mapping[str, np.ndarray]], selection: Mapping[str, Any], variant: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conflicts = _conflict_count(base, views)
    delta = float(selection["margin_delta"])
    allowed = int(selection["allowed_conflicts"])
    view_max_score = np.max(np.asarray([view["score"] for view in views]), axis=0) if views else base["score"]
    if variant == "base_k1":
        accepted = (base["pred"] == 0) & (base["score"] <= THRESHOLD)
        scores = base["score"]
    elif variant == "consistency_strict":
        accepted = (base["pred"] == 0) & (base["score"] <= THRESHOLD) & (conflicts == 0)
        scores = np.maximum(base["score"], view_max_score) + (conflicts > 0).astype(float)
    elif variant == "evidence_margin":
        accepted = (base["pred"] == 0) & (base["score"] <= THRESHOLD) & (base["margin"] >= delta)
        scores = base["score"] + (base["margin"] < delta).astype(float)
    elif variant == "combined_selected":
        accepted = (base["pred"] == 0) & (base["score"] <= THRESHOLD) & (conflicts <= allowed) & (base["margin"] >= delta)
        scores = np.maximum(base["score"], view_max_score) + ((conflicts > allowed) | (base["margin"] < delta)).astype(float)
    else:
        raise ValueError(f"Unknown consistency variant: {variant}")
    metrics, predictions = _metrics(rows, accepted, base["intent"], scores)
    for index, item in enumerate(predictions):
        item.update({"variant": variant, "conflict_count": int(conflicts[index]), "base_margin": float(base["margin"][index]), "base_score": float(base["score"][index])})
    return metrics, predictions


def _load_model(paths: ProtocolV2Paths, config: Mapping[str, Any], seed: int, device: torch.device) -> tuple[RacalMiniLM, Any, dict[str, Any]]:
    model_path = _model_path(paths, config)
    model = RacalMiniLM(model_path, "last2_minilm_plus_projection", int(config.get("projection_hidden_dim", 256))).to(device)
    checkpoint = _checkpoint_path(paths, config, seed)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer, payload


def _run_seed(paths: ProtocolV2Paths, config: Mapping[str, Any], seed: int, root: Path) -> dict[str, Any]:
    started = time.time()
    set_seed(seed)
    bundle = load_e2_bundle(paths, DATASET, seed, KIR)
    model, tokenizer, checkpoint = _load_model(paths, config, seed, choose_device(str(config.get("device", "auto"))))
    device = next(model.parameters()).device
    train_rows, calibration_rows, test_rows = bundle.views.train, bundle.views.calibration, bundle.views.test
    batch_size = int(config.get("batch_size", 64))
    max_length = int(config.get("max_length", 256))
    train_values = encode_rows(model, tokenizer, train_rows, device, batch_size, max_length)
    calibration_values = encode_rows(model, tokenizer, calibration_rows, device, batch_size, max_length)
    test_values = encode_rows(model, tokenizer, test_rows, device, batch_size, max_length)
    detector = fit_k1_detector(train_values, train_rows, "mahalanobis_diag")
    calibration_views = [_view_stats(detector, calibration_values)]
    test_views = [_view_stats(detector, test_values)]
    for view_index in range(N_DROPOUT_VIEWS):
        calibration_views.append(_view_stats(detector, _encode_dropout(model, tokenizer, calibration_rows, device, batch_size, max_length, seed + 1009 + view_index)))
        test_views.append(_view_stats(detector, _encode_dropout(model, tokenizer, test_rows, device, batch_size, max_length, seed + 1009 + view_index)))
    surface_cal = encode_rows(model, tokenizer, _view_rows(calibration_rows), device, batch_size, max_length)
    surface_test = encode_rows(model, tokenizer, _view_rows(test_rows), device, batch_size, max_length)
    calibration_views.append(_view_stats(detector, surface_cal))
    test_views.append(_view_stats(detector, surface_test))
    base_cal, *calibration_other = calibration_views
    base_test, *test_other = test_views
    selection = _select_known_only(base_cal, calibration_other, float(config.get("max_known_recall_drop", 0.01)))
    variants = ["base_k1", "consistency_strict", "evidence_margin", "combined_selected"]
    variant_metrics: dict[str, Any] = {}
    variant_predictions: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        metrics, predictions = _evaluate_variant(test_rows, base_test, test_other, selection, variant)
        variant_metrics[variant] = metrics
        variant_predictions[variant] = predictions
    run_dir = root / "runs" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(paths, config, seed)
    atomic_write_json(run_dir / "metrics.json", _safe({"stage": STAGE, "dataset": DATASET, "kir": KIR, "seed": seed, "selection": selection, "variants": variant_metrics, "n_views": len(test_views), "checkpoint_sha256": sha256_file(checkpoint_path), "checkpoint_path": str(checkpoint_path), "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)), "elapsed_seconds": time.time() - started, "test_used_for_selection": False, "oos_used_for_training": False, "train_ids_sha256": rows_hash(train_rows), "calibration_ids_sha256": rows_hash(calibration_rows), "test_ids_sha256": rows_hash(test_rows), "train_embedding_sha256": array_hash(train_values), "calibration_embedding_sha256": array_hash(calibration_values), "test_embedding_sha256": array_hash(test_values)}))
    atomic_write_jsonl(run_dir / "predictions.jsonl", [item for variant in variants for item in variant_predictions[variant]])
    _atomic_csv(run_dir / "variant_metrics.csv", [{"seed": seed, "variant": variant, **metrics} for variant, metrics in variant_metrics.items()])
    atomic_write_json(run_dir / "selection.json", _safe(selection))
    atomic_write_json(run_dir / "run_manifest.json", _safe({"stage": STAGE, "dataset": DATASET, "kir": KIR, "seed": seed, "status": "complete", "checkpoint_path": str(_checkpoint_path(paths, config, seed)), "checkpoint_metadata": checkpoint.get("freeze_report", {}), "selection": selection, "n_views": len(test_views), "test_used_for_selection": False, "oos_used_for_training": False, "elapsed_seconds": time.time() - started}))
    return {"seed": seed, "status": "complete", "selection": selection, "variants": variant_metrics, "run_dir": str(run_dir)}


def _provenance(paths: ProtocolV2Paths, config_path: Path, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    source_files = [paths.project_root / "src/protocol_v2/experiments/consistency_gate_v1.py", paths.project_root / "scripts/experiments/run_consistency_gate_v1.py", config_path]
    source_manifest = {str(path.relative_to(paths.project_root)): sha256_file(path) for path in source_files}
    patch = root / "CONSISTENCY_GATE_CODE.patch"
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=paths.project_root.parent, check=True, capture_output=True).stdout.decode("utf-8", errors="replace")
    atomic_write_text(patch, diff + "\n# untracked source hashes\n" + json.dumps(source_manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return {"schema_version": "s2c.consistency_gate_v1.provenance.v1", "stage": STAGE, "protocol_version": paths.dataset_version, "dataset": DATASET, "kir": KIR, "seeds": list(SEEDS), "base_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.project_root.parent, check=True, capture_output=True, text=True).stdout.strip(), "git_dirty": bool(subprocess.run(["git", "status", "--short"], cwd=paths.project_root.parent, check=True, capture_output=True, text=True).stdout.strip()), "source_files": source_manifest, "code_patch_sha256": sha256_file(patch), "config_sha256": sha256_file(config_path), "checkpoint_root": str(config["checkpoint_root"]), "views": ["original_eval", "mc_dropout_0", "mc_dropout_1", "surface_normalized"], "selection": "Known calibration only; target known recall drop <= configured tolerance", "test_used_for_selection": False, "oos_used_for_training": False, "created_at": time.time()}


def _verify(paths: ProtocolV2Paths, config_path: Path, root: Path) -> dict[str, Any]:
    errors: list[str] = []
    provenance_path = root / "CONSISTENCY_GATE_PROVENANCE.json"
    if not provenance_path.is_file():
        errors.append("missing provenance")
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("config_sha256") != sha256_file(config_path):
            errors.append("config hash mismatch")
        if provenance.get("code_patch_sha256") != sha256_file(root / "CONSISTENCY_GATE_CODE.patch"):
            errors.append("code patch hash mismatch")
    for seed in SEEDS:
        if not (root / "runs" / f"seed_{seed}" / "run_manifest.json").is_file():
            errors.append(f"missing seed {seed}")
    return {"stage": STAGE, "status": "complete" if not errors else "failed", "planned": len(SEEDS), "completed": sum((root / "runs" / f"seed_{seed}" / "run_manifest.json").is_file() for seed in SEEDS), "errors": errors}


def _validate_config(config: Mapping[str, Any]) -> None:
    if str(config.get("stage")) != STAGE:
        raise ValueError(f"Expected stage={STAGE}")
    if str(config.get("protocol_version")) != "protocol_v2_textoir_v1":
        raise ValueError("Consistency gate requires protocol_v2_textoir_v1")
    if str(config.get("dataset", "")).lower() != DATASET or abs(float(config.get("kir", -1.0)) - KIR) > 1e-12:
        raise ValueError("Consistency gate pilot is restricted to StackOverflow KIR=0.50")
    if tuple(int(seed) for seed in config.get("seeds", [])) != SEEDS:
        raise ValueError(f"Consistency gate must declare seeds {SEEDS}")
    if not config.get("test_used_for_selection") is False or not config.get("oos_used_for_training") is False:
        raise ValueError("Consistency gate cannot use test OOS for selection or training")


def _summarize(root: Path) -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        payload = json.loads((root / "runs" / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
        for variant, metrics in payload["variants"].items():
            rows.append({"seed": seed, "variant": variant, **metrics})
    variants = sorted({str(row["variant"]) for row in rows})
    aggregate = []
    for variant in variants:
        subset = [row for row in rows if row["variant"] == variant]
        aggregate.append({"variant": variant, "seed_count": len(subset), **{f"{key}_mean": float(np.mean([float(row[key]) for row in subset])) for key in ("oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos")}, **{f"{key}_std": float(np.std([float(row[key]) for row in subset], ddof=0)) for key in ("oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos")}})
    _atomic_csv(root / "CONSISTENCY_GATE_SUMMARY.csv", aggregate)
    return {"stage": STAGE, "seed_count": len(SEEDS), "aggregate": aggregate}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Known-only consistency/evidence conflict Gate pilot")
    parser.add_argument("command", choices=("run", "summarize", "verify"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    config = json.loads(json.dumps(__import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))))
    _validate_config(config)
    paths.require_experiment_admission(DATASET)
    root = _root(paths)
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        if not (root / "CONSISTENCY_GATE_PROVENANCE.json").is_file():
            atomic_write_json(root / "CONSISTENCY_GATE_PROVENANCE.json", _safe(_provenance(paths, args.config.resolve(), config, root)))
        seeds = [args.seed] if args.seed is not None else list(SEEDS)
        results = []
        for seed in seeds:
            run_manifest = root / "runs" / f"seed_{seed}" / "run_manifest.json"
            if args.resume and run_manifest.is_file():
                results.append(json.loads(run_manifest.read_text(encoding="utf-8")))
            else:
                results.append(_run_seed(paths, config, int(seed), root))
        atomic_write_json(root / "CONSISTENCY_GATE_RESULTS.json", _safe({"stage": STAGE, "status": "complete", "results": results}))
        print(json.dumps(_safe({"stage": STAGE, "status": "complete", "completed": len(results), "root": root}), ensure_ascii=False, indent=2))
    elif args.command == "summarize":
        print(json.dumps(_safe(_summarize(root)), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_safe(_verify(paths, args.config.resolve(), root)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
