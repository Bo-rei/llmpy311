"""Contract-repaired training-participating adaptive multicenter pilot.

This module is deliberately separate from ``joint_adaptive_v1``.  The older
pilot remains immutable; this stage fixes three contracts before making a new
claim:

* the K=1 parent boundary is fitted once and never refit after a candidate
  representation update;
* calibration compactness is measured with the same parent-guarded score used
  by inference;
* candidate prototypes receive explicit load-balance and separation penalties.

Only Known train rows are optimized and Known calibration rows select the
structure.  Test OOS labels are read only for final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file
from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.experiments.joint_adaptive_v1.runner import (
    DATASET,
    KIR,
    OOS_LABEL,
    BoundaryState,
    CenterState,
    JointPrototypeModel,
    _candidate_intents,
    _hash_array,
    _hash_rows,
    _initial_checkpoint,
    _initial_centers,
    _label_map,
    _model_path,
    _normalize,
    _seed_everything,
    _state_from_tensor,
    _fit_diag,
)
from protocol_v2.experiments.racal_v1.representation import RacalMiniLM, choose_device, encode_rows
from protocol_v2.gate.view_loader import GateViews, load_gate_views
from protocol_v2.runtime.paths import ProtocolV2Paths


STAGE = "joint_adaptive_multicenter_contract_repair_v1"
SEEDS = (13, 42, 87)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        atomic_write_text(path, "")
        return
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, mode="w", encoding="utf-8", newline="", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _root(paths: ProtocolV2Paths) -> Path:
    attempt = os.environ.get("JOINT_ADAPTIVE_CONTRACT_ATTEMPT", "repair1").strip() or "repair1"
    return paths.run_root / STAGE / attempt


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Contract repair config must be a mapping: {path}")
    required = {"protocol_version", "dataset", "kir", "seeds", "model_path", "stage"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Contract repair config missing keys: {missing}")
    if payload["stage"] != STAGE or payload["protocol_version"] != "protocol_v2_textoir_v1":
        raise ValueError("Contract repair is restricted to its independent protocol stage")
    if str(payload["dataset"]).lower() != DATASET or abs(float(payload["kir"]) - KIR) > 1e-12:
        raise ValueError("Contract repair is restricted to StackOverflow KIR=0.50")
    seeds = tuple(int(x) for x in payload["seeds"])
    if seeds != SEEDS:
        raise ValueError(f"Contract repair must use seeds {SEEDS}, got {seeds}")
    return payload


def _load_views(paths: ProtocolV2Paths, seed: int) -> GateViews:
    views = load_gate_views(paths, DATASET, seed, KIR)
    if any(int(row["label"]) != 0 for row in views.train + views.calibration):
        raise ValueError("Contract repair train/calibration must be Known-only")
    ids = [str(row["sample_id"]) for row in views.train + views.calibration + views.test]
    if len(ids) != len(set(ids)):
        raise ValueError("Contract repair views overlap sample IDs")
    return views


def _fit_fixed_parent(values: np.ndarray, rows: Sequence[Mapping[str, Any]], state: CenterState, radius_lambda: float, eps: float = 1e-5) -> BoundaryState:
    """Fit the parent boundary once; its parameters are never recomputed."""
    values = _normalize(values)
    parent_centers = _normalize(np.asarray([
        values[[str(row["intent"]) == intent for row in rows]].mean(axis=0)
        for intent in state.parent_intents
    ], dtype=np.float32))
    parent_inv, parent_radii = [], []
    for intent, center in zip(state.parent_intents, parent_centers, strict=True):
        local = values[[str(row["intent"]) == intent for row in rows]]
        inv, radius = _fit_diag(local, center, eps)
        parent_inv.append(inv)
        parent_radii.append(max(float(radius * (1.0 + 0.0 * radius_lambda)), 1e-6))
    return BoundaryState(
        centers=parent_centers.copy(),
        center_intents=state.parent_intents,
        radii=np.asarray(parent_radii, dtype=np.float64),
        inv_diag_cov=np.asarray(parent_inv, dtype=np.float64),
        parent_centers=parent_centers.copy(),
        parent_intents=state.parent_intents,
        parent_radii=np.asarray(parent_radii, dtype=np.float64),
        parent_inv_diag_cov=np.asarray(parent_inv, dtype=np.float64),
    )


def _fit_children_with_fixed_parent(
    train_values: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    state: CenterState,
    parent: BoundaryState,
    radius_lambda: float,
    eps: float = 1e-5,
) -> tuple[BoundaryState, dict[str, Any]]:
    values = _normalize(train_values)
    centers = _normalize(state.centers)
    assignments = np.full(len(values), -1, dtype=np.int64)
    loads: dict[str, list[int]] = {}
    for intent in state.intent_names:
        sample_indices = np.asarray([i for i, row in enumerate(rows) if str(row["intent"]) == intent], dtype=np.int64)
        center_indices = np.asarray([i for i, name in enumerate(state.center_intents) if name == intent], dtype=np.int64)
        distances = 1.0 - values[sample_indices] @ centers[center_indices].T
        chosen = center_indices[distances.argmin(axis=1)]
        assignments[sample_indices] = chosen
        loads[intent] = [int(np.sum(chosen == index)) for index in center_indices]
    inv_cov, radii = [], []
    for index, center in enumerate(centers):
        local = values[assignments == index]
        if len(local) == 0:
            local = values
        inv, radius = _fit_diag(local, center, eps)
        inv_cov.append(inv)
        radii.append(max(float(radius * (1.0 + 0.0 * radius_lambda)), 1e-6))
    child_centers = centers
    boundary = BoundaryState(
        centers=child_centers,
        center_intents=state.center_intents,
        radii=np.asarray(radii, dtype=np.float64),
        inv_diag_cov=np.asarray(inv_cov, dtype=np.float64),
        parent_centers=parent.parent_centers.copy(),
        parent_intents=parent.parent_intents,
        parent_radii=parent.parent_radii.copy(),
        parent_inv_diag_cov=parent.parent_inv_diag_cov.copy(),
    )
    separation = []
    for intent in state.intent_names:
        idx = [i for i, name in enumerate(state.center_intents) if name == intent]
        if len(idx) > 1:
            for left in range(len(idx)):
                for right in range(left + 1, len(idx)):
                    separation.append(float(1.0 - child_centers[idx[left]] @ child_centers[idx[right]]))
    diagnostics = {
        "loads": loads,
        "minimum_child_load": min((load for values_ in loads.values() for load in values_), default=0),
        "minimum_child_separation": min(separation) if separation else math.inf,
        "mean_child_separation": float(np.mean(separation)) if separation else math.inf,
        "radii": [float(x) for x in radii],
    }
    return boundary, diagnostics


def _scores(boundary: BoundaryState, values: np.ndarray, threshold: float) -> dict[str, Any]:
    values = _normalize(values)
    child_q = np.sqrt(np.maximum(np.einsum("nkd,kd->nk", (values[:, None, :] - boundary.centers[None, :, :]) ** 2, boundary.inv_diag_cov), 0.0))
    child_norm = child_q / boundary.radii[None, :]
    parent_q = np.sqrt(np.maximum(np.einsum("nkd,kd->nk", (values[:, None, :] - boundary.parent_centers[None, :, :]) ** 2, boundary.parent_inv_diag_cov), 0.0))
    parent_norm = parent_q / boundary.parent_radii[None, :]
    guarded = child_norm.copy()
    for index, intent in enumerate(boundary.center_intents):
        parent_index = boundary.parent_intents.index(intent)
        guarded[:, index] = np.where(parent_norm[:, parent_index] <= threshold, guarded[:, index], np.inf)
    intent_scores = np.full((len(values), len(boundary.parent_intents)), np.inf, dtype=np.float64)
    unconstrained = np.full_like(intent_scores, np.inf)
    for index, intent in enumerate(boundary.parent_intents):
        child = [i for i, name in enumerate(boundary.center_intents) if name == intent]
        intent_scores[:, index] = np.min(guarded[:, child], axis=1)
        unconstrained[:, index] = np.min(child_norm[:, child], axis=1)
    best = intent_scores.argmin(axis=1)
    raw_score = intent_scores[np.arange(len(values)), best]
    score = np.where(np.isfinite(raw_score), raw_score, 1.0e6)
    accepted = np.isfinite(raw_score) & (raw_score <= threshold)
    best_intent = np.asarray(boundary.parent_intents, dtype=object)[best]
    return {
        "score": score,
        "raw_score": raw_score,
        "unconstrained_score": unconstrained.min(axis=1),
        "accepted": accepted,
        "best_intent": best_intent,
        "parent_norm": parent_norm,
    }


def _evaluate(boundary: BoundaryState, embeddings: np.ndarray, rows: Sequence[Mapping[str, Any]], threshold: float) -> tuple[dict[str, float], list[dict[str, Any]]]:
    output = _scores(boundary, embeddings, threshold)
    gold_binary = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    binary = compute_binary_oos_metrics(gold_binary, output["score"], threshold)
    names = list(boundary.parent_intents)
    truth = [OOS_LABEL if int(row["label"]) else str(row["intent"]) for row in rows]
    predicted = [OOS_LABEL if not output["accepted"][i] else str(output["best_intent"][i]) for i in range(len(rows))]
    labels = [*names, OOS_LABEL]
    metrics = {
        **binary,
        "f1_all": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "f1_u": float(f1_score(truth, predicted, labels=[OOS_LABEL], average="macro", zero_division=0)),
        "f1_k": float(f1_score(truth, predicted, labels=names, average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(truth, dtype=object) == np.asarray(predicted, dtype=object))),
        "known_recall": float(1.0 - binary["false_reject_rate"]) if np.isfinite(binary["false_reject_rate"]) else math.nan,
        "guarded_mean_score": float(np.mean(output["score"])) if len(output["score"]) else math.nan,
        "unconstrained_mean_score": float(np.mean(output["unconstrained_score"])) if len(output["unconstrained_score"]) else math.nan,
    }
    predictions = []
    for index, row in enumerate(rows):
        predictions.append({
            "sample_id": str(row["sample_id"]),
            "gold_intent": str(row["intent"]),
            "gold_is_oos": int(row["label"]),
            "predicted_intent": predicted[index],
            "predicted_is_oos": int(not output["accepted"][index]),
            "guarded_score": float(output["score"][index]),
            "unconstrained_score": float(output["unconstrained_score"][index]),
            "parent_norm": float(np.min(output["parent_norm"][index])),
        })
    return metrics, predictions


def _loss(model: JointPrototypeModel, features: torch.Tensor, targets: torch.Tensor, names: Sequence[str], config: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    features = torch.nn.functional.normalize(features, dim=-1)
    centers = model.normalized_prototypes()
    distances = 1.0 - features @ centers.T
    logits = model.logits(features, names, float(config["temperature"]))
    own, other = [], []
    balance_terms, separation_terms = [], []
    for row_index, name in enumerate([names[int(x)] for x in targets.tolist()]):
        own_indices = [i for i, value in enumerate(model.center_intents) if value == name]
        other_indices = [i for i, value in enumerate(model.center_intents) if value != name]
        own.append(distances[row_index, own_indices].min())
        other.append(distances[row_index, other_indices].min())
    own_tensor = torch.stack(own)
    other_tensor = torch.stack(other)
    for name in names:
        indices = [i for i, value in enumerate(model.center_intents) if value == name]
        if len(indices) > 1:
            probabilities = torch.softmax(-distances[:, indices] / float(config["temperature"]), dim=1)
            mean_probability = probabilities.mean(dim=0)
            target_probability = torch.full_like(mean_probability, 1.0 / len(indices))
            balance_terms.append(torch.mean((mean_probability - target_probability) ** 2))
            for left in range(len(indices)):
                for right in range(left + 1, len(indices)):
                    separation_distance = 1.0 - torch.sum(centers[indices[left]] * centers[indices[right]])
                    separation_terms.append(torch.relu(separation_distance.new_tensor(float(config["minimum_center_separation"])) - separation_distance))
    ce = torch.nn.functional.cross_entropy(logits, targets)
    intra = own_tensor.mean()
    inter = torch.relu(own_tensor - other_tensor + float(config["inter_margin"])).mean()
    balance = torch.stack(balance_terms).mean() if balance_terms else ce.new_zeros(())
    separation = torch.stack(separation_terms).mean() if separation_terms else ce.new_zeros(())
    total = ce + float(config["intra_weight"]) * intra + float(config["inter_weight"]) * inter + float(config["load_balance_weight"]) * balance + float(config["separation_weight"]) * separation
    return total, {"ce": float(ce.detach().cpu()), "intra": float(intra.detach().cpu()), "inter": float(inter.detach().cpu()), "balance": float(balance.detach().cpu()), "separation": float(separation.detach().cpu()), "total": float(total.detach().cpu())}


def _train_candidate(model: JointPrototypeModel, tokenizer: Any, rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], device: torch.device, seed: int, checkpoint_path: Path) -> dict[str, Any]:
    names, label_map = _label_map(rows)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["candidate_learning_rate"]))
    history = []
    for epoch in range(1, int(config["split_epochs"]) + 1):
        model.train()
        order = np.random.default_rng(seed + epoch * 1009).permutation(len(rows))
        parts = []
        for start in range(0, len(order), int(config["batch_size"])):
            batch_rows = [rows[int(i)] for i in order[start : start + int(config["batch_size"])]]
            tokens = tokenizer([str(row["text"]) for row in batch_rows], padding=True, truncation=True, max_length=int(config["max_length"]), return_tensors="pt").to(device)
            targets = torch.as_tensor([label_map[str(row["intent"])] for row in batch_rows], dtype=torch.long, device=device)
            loss, detail = _loss(model, model(tokens), targets, names, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            parts.append(detail)
        history.append({key: float(np.mean([item[key] for item in parts])) for key in parts[0]})
    state = _state_from_tensor(model.prototype, CenterState(model.prototype.detach().cpu().numpy(), model.center_intents, model.prototype.detach().cpu().numpy(), model.center_intents, ()))
    payload = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "center_intents": list(model.center_intents), "history": history}
    _atomic_torch_save(checkpoint_path, payload)
    return {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path), "history": history, "checkpoint_centers": len(model.center_intents)}


def _objective(metrics: Mapping[str, float], complexity: int) -> float:
    return float(metrics["known_recall"] - 0.02 * metrics["guarded_mean_score"] - 0.002 * complexity)


def _candidate_state(state: CenterState, candidate: Mapping[str, Any], values: np.ndarray) -> CenterState:
    intent = str(candidate["intent"])
    left, right = candidate["split"]
    local = _normalize(values)
    left_center = _normalize(local[left].mean(axis=0, keepdims=True))[0]
    right_center = _normalize(local[right].mean(axis=0, keepdims=True))[0]
    centers, intents = [], []
    for center, name in zip(state.centers, state.center_intents, strict=True):
        if name == intent:
            centers.extend([left_center, right_center])
            intents.extend([name, name])
        else:
            centers.append(center)
            intents.append(name)
    return CenterState(np.asarray(centers, dtype=np.float32), tuple(intents), state.parent_centers.copy(), state.parent_intents, state.split_events)


def _run_seed(paths: ProtocolV2Paths, config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    started = time.time()
    _seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    views = _load_views(paths, seed)
    model_path = _model_path(paths, config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    initial = _initial_checkpoint(paths, config, seed)
    if initial is None:
        raise FileNotFoundError("Contract repair requires the validated RACAL trainable K=1 checkpoint")
    base = RacalMiniLM(model_path, "last2_minilm_plus_projection", int(config["projection_hidden_dim"])).to(device)
    base.load_state_dict(torch.load(initial, map_location="cpu", weights_only=False)["model"])
    base.eval()
    train_base = encode_rows(base, tokenizer, views.train, device, int(config["batch_size"]), int(config["max_length"]))
    cal_base = encode_rows(base, tokenizer, views.calibration, device, int(config["batch_size"]), int(config["max_length"]))
    test_base = encode_rows(base, tokenizer, views.test, device, int(config["batch_size"]), int(config["max_length"]))
    state = _initial_centers(train_base, views.train)
    parent_boundary = _fit_fixed_parent(train_base, views.train, state, float(config["radius_lambda"]))
    base_metrics, _ = _evaluate(parent_boundary, cal_base, views.calibration, float(config["threshold"]))
    model = JointPrototypeModel(model_path, state, int(config["projection_hidden_dim"])).to(device)
    model.encoder.load_state_dict(base.state_dict())
    with torch.no_grad():
        model.prototype.copy_(torch.as_tensor(state.centers, dtype=torch.float32, device=device))
    candidates = _candidate_intents(train_base, views.train, int(config["max_candidate_splits"]), int(config["min_child_samples"]))
    events, accepted = [], 0
    current_model, current_boundary, current_state = model, parent_boundary, state
    for candidate in candidates:
        candidate_state = _candidate_state(current_state, candidate, train_base)
        candidate_model = JointPrototypeModel(model_path, candidate_state, int(config["projection_hidden_dim"])).to(device)
        candidate_model.encoder.load_state_dict(current_model.encoder.state_dict())
        with torch.no_grad():
            candidate_model.prototype.copy_(torch.as_tensor(candidate_state.centers, dtype=torch.float32, device=device))
        root = _root(paths) / "runs" / f"seed_{seed}"
        candidate_training = _train_candidate(candidate_model, tokenizer, views.train, config, device, seed + 100, root / f"candidate_{candidate['intent']}.pt")
        candidate_model.eval()
        train_candidate = encode_rows(candidate_model.encoder, tokenizer, views.train, device, int(config["batch_size"]), int(config["max_length"]))
        cal_candidate = encode_rows(candidate_model.encoder, tokenizer, views.calibration, device, int(config["batch_size"]), int(config["max_length"]))
        child_boundary, child_diag = _fit_children_with_fixed_parent(train_candidate, views.train, candidate_state, parent_boundary, float(config["radius_lambda"]))
        candidate_metrics, _ = _evaluate(child_boundary, cal_candidate, views.calibration, float(config["threshold"]))
        score_gain = float(base_metrics["guarded_mean_score"] - candidate_metrics["guarded_mean_score"])
        objective_before = _objective(base_metrics, len(current_state.centers))
        objective_after = _objective(candidate_metrics, len(candidate_state.centers))
        event = {
            "intent": candidate["intent"],
            "accepted": False,
            "before_centers": len(current_state.centers),
            "after_centers": len(candidate_state.centers),
            "calibration_before": base_metrics,
            "calibration_after": candidate_metrics,
            "calibration_score_gain_guarded": score_gain,
            "objective_before": objective_before,
            "objective_after": objective_after,
            "known_recall_delta": float(candidate_metrics["known_recall"] - base_metrics["known_recall"]),
            "candidate_training": candidate_training,
            "child_diagnostics": child_diag,
            "parent_boundary_sha256": _hash_array(np.concatenate([parent_boundary.parent_centers.ravel(), parent_boundary.parent_radii.ravel(), parent_boundary.parent_inv_diag_cov.ravel()])),
            "reject_reason": None,
        }
        if event["known_recall_delta"] < -float(config["max_calibration_recall_drop"]):
            event["reject_reason"] = "calibration_known_recall_drop"
        elif child_diag["minimum_child_load"] < int(config["min_child_samples"]):
            event["reject_reason"] = "child_load_below_minimum"
        elif child_diag["minimum_child_separation"] < float(config["minimum_center_separation"]):
            event["reject_reason"] = "center_separation_below_minimum"
        elif score_gain < float(config["minimum_calibration_score_gain"]):
            event["reject_reason"] = "no_guarded_known_only_compactness_gain"
        elif objective_after < objective_before - float(config["maximum_objective_regression"]):
            event["reject_reason"] = "no_guarded_known_only_objective_gain"
        else:
            event["accepted"] = True
            accepted += 1
            current_model, current_boundary, current_state = candidate_model, child_boundary, candidate_state
            base_metrics = candidate_metrics
        events.append(event)
        if not event["accepted"]:
            del candidate_model
        if accepted >= int(config["max_accepted_splits"]):
            break
    current_model.eval()
    test_values = encode_rows(current_model.encoder, tokenizer, views.test, device, int(config["batch_size"]), int(config["max_length"]))
    test_metrics, predictions = _evaluate(current_boundary, test_values, views.test, float(config["threshold"]))
    test_metrics.update({"stage": STAGE, "dataset": DATASET, "kir": KIR, "seed": seed, "method": "joint_adaptive_contract_repair", "accepted_split_count": accepted, "mean_k_y": float(np.mean(list(current_state.counts().values()))), "max_k_y": max(current_state.counts().values()), "test_used_for_selection": False, "oos_used_for_training": False, "initial_checkpoint": str(initial), "initial_checkpoint_sha256": sha256_file(initial), "train_ids_sha256": _hash_rows(views.train), "calibration_ids_sha256": _hash_rows(views.calibration), "test_ids_sha256": _hash_rows(views.test), "test_embedding_sha256": _hash_array(test_values), "elapsed_seconds": time.time() - started, "trainable_parameters": int(sum(parameter.numel() for parameter in current_model.parameters() if parameter.requires_grad))})
    root = _root(paths) / "runs" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "metrics.json", test_metrics)
    atomic_write_json(root / "split_events.json", events)
    atomic_write_json(root / "center_counts.json", current_state.counts())
    atomic_write_json(root / "run_manifest.json", {"schema_version": "s2c.joint_adaptive_contract_repair_v1.run.v1", "status": "complete", "metrics": test_metrics, "split_events": events, "center_counts": current_state.counts(), "parent_boundary_frozen": True, "parent_boundary_sha256": _hash_array(np.concatenate([parent_boundary.parent_centers.ravel(), parent_boundary.parent_radii.ravel(), parent_boundary.parent_inv_diag_cov.ravel()]))})
    _write_csv(root / "predictions.csv", predictions)
    del current_model, tokenizer, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {"metrics": test_metrics, "events": events, "center_counts": current_state.counts()}


def freeze_provenance(paths: ProtocolV2Paths, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(paths)
    root.mkdir(parents=True, exist_ok=True)
    source_files = [paths.project_root / "src/protocol_v2/experiments/joint_adaptive_contract_repair_v1.py", paths.project_root / "scripts/experiments/run_joint_adaptive_contract_repair_v1.py", config_path]
    source_manifest = {str(path.relative_to(paths.project_root)): sha256_file(path) for path in source_files}
    patch = root / "JOINT_ADAPTIVE_CONTRACT_REPAIR_CODE.patch"
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=paths.project_root.parent, check=True, capture_output=True).stdout.decode("utf-8", errors="replace")
    atomic_write_text(patch, diff + "\n# untracked source hashes\n" + json.dumps(source_manifest, ensure_ascii=False, sort_keys=True, indent=2))
    initial_hashes = {str(seed): {"path": str(_initial_checkpoint(paths, config, seed)), "sha256": sha256_file(_initial_checkpoint(paths, config, seed))} for seed in SEEDS}
    payload = {"schema_version": "s2c.joint_adaptive_contract_repair_v1.provenance.v1", "stage": STAGE, "protocol_version": paths.dataset_version, "dataset": DATASET, "kir": KIR, "seeds": list(SEEDS), "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.project_root.parent, check=True, capture_output=True, text=True).stdout.strip(), "git_dirty": bool(subprocess.run(["git", "status", "--short"], cwd=paths.project_root.parent, check=True, capture_output=True, text=True).stdout.strip()), "code_patch_sha256": sha256_file(patch), "source_files": source_manifest, "config_sha256": sha256_file(config_path), "initial_checkpoint_hashes": initial_hashes, "model_path": str(_model_path(paths, config)), "selection": "Known train + guarded Known calibration only", "parent_boundary_frozen": True, "test_used_for_selection": False, "oos_used_for_training": False, "created_at": datetime.now(UTC).isoformat(), "python": os.sys.version, "torch": torch.__version__, "numpy": np.__version__}
    atomic_write_json(root / "JOINT_ADAPTIVE_CONTRACT_REPAIR_PROVENANCE.json", payload)
    return payload


def _verify(paths: ProtocolV2Paths, config_path: Path) -> dict[str, Any]:
    root = _root(paths)
    provenance = json.loads((root / "JOINT_ADAPTIVE_CONTRACT_REPAIR_PROVENANCE.json").read_text(encoding="utf-8"))
    if provenance["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("Contract repair config changed after provenance freeze")
    if provenance["code_patch_sha256"] != sha256_file(root / "JOINT_ADAPTIVE_CONTRACT_REPAIR_CODE.patch"):
        raise RuntimeError("Contract repair code patch changed after provenance freeze")
    runs = []
    failures = []
    for seed in SEEDS:
        run_manifest = root / "runs" / f"seed_{seed}" / "run_manifest.json"
        if not run_manifest.is_file():
            failures.append(f"missing seed {seed}")
        else:
            runs.append(seed)
    return {"stage": STAGE, "status": "complete" if not failures and len(runs) == len(SEEDS) else "failed", "planned_seeds": len(SEEDS), "completed_seeds": len(runs), "failures": failures, "provenance": True}


def _summarize(paths: ProtocolV2Paths) -> dict[str, Any]:
    root = _root(paths)
    rows = [json.loads((root / "runs" / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8")) for seed in SEEDS]
    summary = {"stage": STAGE, "dataset": DATASET, "kir": KIR, "seed_count": len(rows), "metrics": {}, "mean_k_y": {"values": [row["mean_k_y"] for row in rows], "mean": float(np.mean([row["mean_k_y"] for row in rows]))}}
    for key in ("oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"):
        values = [float(row[key]) for row in rows]
        summary["metrics"][key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)), "values": values}
    atomic_write_json(root / "CONTRACT_REPAIR_SUMMARY.json", summary)
    return summary


def run(paths: ProtocolV2Paths, config_path: Path, resume: bool, selected_seed: int | None = None) -> dict[str, Any]:
    config = _load_config(config_path)
    paths.require_experiment_admission(DATASET)
    root = _root(paths)
    if (root / "JOINT_ADAPTIVE_CONTRACT_REPAIR_PROVENANCE.json").is_file():
        _verify(paths, config_path)
    else:
        freeze_provenance(paths, config_path, config)
    seeds = (int(selected_seed),) if selected_seed is not None else SEEDS
    results = []
    for seed in seeds:
        run_dir = root / "runs" / f"seed_{seed}"
        if resume and (run_dir / "run_manifest.json").is_file():
            results.append(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")))
            continue
        results.append(_run_seed(paths, config, seed))
    atomic_write_json(root / "CONTRACT_REPAIR_RESULTS.json", {"stage": STAGE, "status": "complete", "results": results})
    return {"stage": STAGE, "status": "complete", "completed_seeds": len(results), "root": str(root)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contract-repaired training-participating adaptive multicenter pilot")
    parser.add_argument("command", choices=("run", "summarize", "verify", "freeze"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_contract_repair_v1.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    paths = ProtocolV2Paths.discover(Path(__file__).resolve().parents[3])
    config = _load_config(args.config)
    if args.command == "freeze":
        output = freeze_provenance(paths, args.config, config)
    elif args.command == "run":
        output = run(paths, args.config, args.resume, args.seed)
    elif args.command == "summarize":
        output = _summarize(paths)
    else:
        output = _verify(paths, args.config)
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0 if output.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
