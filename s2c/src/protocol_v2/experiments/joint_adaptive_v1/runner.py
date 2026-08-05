"""Bounded, training-participating adaptive multicenter pilot.

This stage is intentionally independent from E2/E3/RACAL artifacts.  A
MiniLM adapter and trainable intent prototypes are optimized together using
Known train rows.  Candidate splits are proposed from train embeddings and
accepted only with Known calibration evidence; test OOS labels are read once
for the final report and never enter model or structure selection.
"""

from __future__ import annotations

import argparse
import copy
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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file
from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.experiments.racal_v1.representation import (
    RacalMiniLM,
    choose_device,
    encode_rows,
    mean_pool,
    set_seed,
)
from protocol_v2.gate.view_loader import GateViews, load_gate_views
from protocol_v2.runtime.paths import ProtocolV2Paths


STAGE = "joint_adaptive_multicenter_v1"
DATASET = "stackoverflow"
KIR = 0.50
SEEDS = (13, 42, 87)
OOS_LABEL = "__oos__"


@dataclass(frozen=True)
class CenterState:
    centers: np.ndarray
    center_intents: tuple[str, ...]
    parent_centers: np.ndarray
    parent_intents: tuple[str, ...]
    split_events: tuple[dict[str, Any], ...]

    @property
    def intent_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.center_intents)))

    def counts(self) -> dict[str, int]:
        return {intent: self.center_intents.count(intent) for intent in self.intent_names}


@dataclass(frozen=True)
class BoundaryState:
    centers: np.ndarray
    center_intents: tuple[str, ...]
    radii: np.ndarray
    inv_diag_cov: np.ndarray
    parent_centers: np.ndarray
    parent_intents: tuple[str, ...]
    parent_radii: np.ndarray
    parent_inv_diag_cov: np.ndarray


def _hash_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(row["sample_id"]) for row in rows).encode("utf-8")
    ).hexdigest()


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    out = []
    if fields:
        buffer = tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
            mode="w", encoding="utf-8", newline="", delete=False,
        )
        try:
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            buffer.flush()
            os.fsync(buffer.fileno())
            temporary = Path(buffer.name)
        finally:
            buffer.close()
        temporary.replace(path)
    else:
        atomic_write_text(path, "")


def _model_path(paths: ProtocolV2Paths, config: Mapping[str, Any]) -> Path:
    configured = Path(str(config["model_path"]))
    candidates = (
        configured,
        paths.project_root / configured,
        paths.project_root.parent / configured,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Local MiniLM model is unavailable: {configured}")


def _initial_checkpoint(paths: ProtocolV2Paths, config: Mapping[str, Any], seed: int) -> Path | None:
    configured = config.get("initial_checkpoint_root")
    if not configured:
        return None
    root = Path(str(configured))
    candidates = (
        root / f"seed_{seed}" / "checkpoint.pt",
        paths.project_root / root / f"seed_{seed}" / "checkpoint.pt",
        paths.project_root.parent / root / f"seed_{seed}" / "checkpoint.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Configured initial checkpoint is unavailable for seed {seed}: {configured}")


def _root(paths: ProtocolV2Paths) -> Path:
    root = paths.run_root / STAGE
    attempt = os.environ.get("JOINT_ADAPTIVE_ATTEMPT", "").strip()
    return root / attempt if attempt else root


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Adaptive pilot config must be a mapping: {path}")
    required = {"protocol_version", "dataset", "kir", "seeds", "model_path"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Adaptive pilot config is missing keys: {missing}")
    if payload["protocol_version"] != "protocol_v2_textoir_v1":
        raise ValueError("Adaptive pilot is restricted to protocol_v2_textoir_v1")
    if str(payload["dataset"]).lower() != DATASET or abs(float(payload["kir"]) - KIR) > 1e-12:
        raise ValueError("First adaptive pilot is restricted to StackOverflow KIR=0.50")
    seeds = tuple(int(x) for x in payload["seeds"])
    if seeds != SEEDS:
        raise ValueError(f"First adaptive pilot must use seeds {SEEDS}, got {seeds}")
    return payload


def _load_views(paths: ProtocolV2Paths, seed: int) -> GateViews:
    views = load_gate_views(paths, DATASET, seed, KIR)
    if any(int(row["label"]) != 0 for row in views.train + views.calibration):
        raise ValueError("Adaptive training/calibration must be Known-only")
    if len({str(row["sample_id"]) for row in views.train + views.calibration + views.test}) != len(views.train) + len(views.calibration) + len(views.test):
        raise ValueError("Adaptive views contain overlapping sample IDs")
    return views


def _label_map(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, ...], dict[str, int]]:
    names = tuple(sorted({str(row["intent"]) for row in rows}))
    return names, {name: i for i, name in enumerate(names)}


def _initial_centers(values: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> CenterState:
    values = _normalize(values)
    names, _ = _label_map(rows)
    centers = []
    intents = []
    for name in names:
        local = values[[str(row["intent"]) == name for row in rows]]
        center = _normalize(local.mean(axis=0, keepdims=True))[0]
        centers.append(center)
        intents.append(name)
    centers_arr = np.asarray(centers, dtype=np.float32)
    return CenterState(centers_arr, tuple(intents), centers_arr.copy(), tuple(intents), ())


def _split_candidate(values: np.ndarray, rows: Sequence[Mapping[str, Any]], intent: str, min_size: int, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    indices = np.asarray([i for i, row in enumerate(rows) if str(row["intent"]) == intent], dtype=np.int64)
    if len(indices) < 2 * min_size:
        return None
    local = _normalize(values[indices])
    local_center = local.mean(axis=0, keepdims=True)
    centered = local - local_center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projections = centered @ direction
    order = np.argsort(projections, kind="mergesort")
    cut = len(order) // 2
    left, right = order[:cut], order[cut:]
    if len(left) < min_size or len(right) < min_size:
        return None
    del seed  # deterministic PCA split; seed is recorded in the event contract
    return indices[left], indices[right]


def _candidate_intents(values: np.ndarray, rows: Sequence[Mapping[str, Any]], max_candidates: int, min_size: int) -> list[dict[str, Any]]:
    candidates = []
    for intent in sorted({str(row["intent"]) for row in rows}):
        indices = np.asarray([i for i, row in enumerate(rows) if str(row["intent"]) == intent], dtype=np.int64)
        if len(indices) < 2 * min_size:
            continue
        local = _normalize(values[indices])
        center = _normalize(local.mean(axis=0, keepdims=True))[0]
        residual = np.linalg.norm(local - center[None, :], axis=1)
        split = _split_candidate(values, rows, intent, min_size, 0)
        if split is None:
            continue
        candidates.append({
            "intent": intent,
            "indices": indices,
            "split": split,
            "residual_mean": float(residual.mean()),
            "residual_p90": float(np.quantile(residual, 0.90)),
            "sample_count": int(len(indices)),
        })
    candidates.sort(key=lambda item: (-item["residual_p90"], item["intent"]))
    return candidates[:max_candidates]


def _make_center_state_after_split(state: CenterState, candidate: Mapping[str, Any], values: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> CenterState:
    intent = str(candidate["intent"])
    left, right = candidate["split"]
    local_values = _normalize(values)
    left_center = _normalize(local_values[left].mean(axis=0, keepdims=True))[0]
    right_center = _normalize(local_values[right].mean(axis=0, keepdims=True))[0]
    centers = []
    intents = []
    for center, name in zip(state.centers, state.center_intents, strict=True):
        if name == intent:
            centers.extend([left_center, right_center])
            intents.extend([name, name])
        else:
            centers.append(center)
            intents.append(name)
    event = {
        "event": "candidate_split",
        "intent": intent,
        "left_count": int(len(left)),
        "right_count": int(len(right)),
        "residual_p90": float(candidate["residual_p90"]),
        "status": "pending",
    }
    return CenterState(
        np.asarray(centers, dtype=np.float32),
        tuple(intents),
        state.parent_centers.copy(),
        state.parent_intents,
        (*state.split_events, event),
    )


def _state_from_tensor(centers: torch.Tensor, state: CenterState) -> CenterState:
    return CenterState(_normalize(centers.detach().cpu().numpy()), state.center_intents, state.parent_centers, state.parent_intents, state.split_events)


class JointPrototypeModel(torch.nn.Module):
    """Trainable MiniLM adapter plus trainable normalized intent prototypes."""

    def __init__(self, model_path: Path, center_state: CenterState, projection_hidden_dim: int) -> None:
        super().__init__()
        self.encoder = RacalMiniLM(model_path, "last2_minilm_plus_projection", projection_hidden_dim)
        self.prototype = torch.nn.Parameter(torch.as_tensor(center_state.centers, dtype=torch.float32))
        self.center_intents = center_state.center_intents

    def forward(self, tokens: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(tokens)

    def normalized_prototypes(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.prototype, dim=-1)

    def logits(self, features: torch.Tensor, names: Sequence[str], temperature: float) -> torch.Tensor:
        features = torch.nn.functional.normalize(features, dim=-1)
        centers = self.normalized_prototypes()
        distances = 1.0 - features @ centers.T
        per_intent = []
        for name in names:
            indices = [i for i, value in enumerate(self.center_intents) if value == name]
            per_intent.append(-distances[:, indices].min(dim=1).values / max(float(temperature), 1e-6))
        return torch.stack(per_intent, dim=1)


def _joint_loss(model: JointPrototypeModel, features: torch.Tensor, targets: torch.Tensor, names: Sequence[str], temperature: float, intra_weight: float, inter_weight: float, margin: float) -> tuple[torch.Tensor, dict[str, float]]:
    features = torch.nn.functional.normalize(features, dim=-1)
    centers = model.normalized_prototypes()
    distances = 1.0 - features @ centers.T
    logits = model.logits(features, names, temperature)
    own_center_indices = []
    other_distance = []
    own_distance = []
    for target, name in zip(targets.tolist(), [names[int(x)] for x in targets.tolist()], strict=True):
        own = [i for i, value in enumerate(model.center_intents) if value == name]
        own_d = distances[len(own_distance), own]
        own_distance.append(own_d.min())
        other = [i for i, value in enumerate(model.center_intents) if value != name]
        other_distance.append(distances[len(other_distance), other].min())
        own_center_indices.append(own[0])
    own_tensor = torch.stack(own_distance)
    other_tensor = torch.stack(other_distance)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    intra = own_tensor.mean()
    inter = torch.relu(own_tensor - other_tensor + float(margin)).mean()
    total = ce + float(intra_weight) * intra + float(inter_weight) * inter
    return total, {"ce": float(ce.detach().cpu()), "intra": float(intra.detach().cpu()), "inter": float(inter.detach().cpu()), "total": float(total.detach().cpu())}


def _train_model(model: JointPrototypeModel, tokenizer: Any, rows: Sequence[Mapping[str, Any]], calibration: Sequence[Mapping[str, Any]], state: CenterState, config: Mapping[str, Any], device: torch.device, seed: int, epochs: int, checkpoint_path: Path, learning_rate: float | None = None) -> tuple[JointPrototypeModel, CenterState, dict[str, Any]]:
    names, label_map = _label_map(rows)
    train_indices = np.arange(len(rows), dtype=np.int64)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(config["learning_rate"] if learning_rate is None else learning_rate))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch * 1009).permutation(train_indices)
        parts = []
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start : start + int(config["batch_size"])]
            batch = [rows[int(i)] for i in indices]
            tokens = tokenizer([str(row["text"]) for row in batch], padding=True, truncation=True, max_length=int(config["max_length"]), return_tensors="pt").to(device)
            target = torch.as_tensor([label_map[str(row["intent"])] for row in batch], dtype=torch.long, device=device)
            features = model(tokens)
            total, item = _joint_loss(model, features, target, names, float(config["temperature"]), float(config["intra_weight"]), float(config["inter_weight"]), float(config["inter_margin"]))
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            parts.append(item)
        model.eval()
        with torch.no_grad():
            train_emb = encode_rows(model.encoder, tokenizer, rows, device, int(config["batch_size"]), int(config["max_length"]))
            cal = encode_rows(model.encoder, tokenizer, calibration, device, int(config["batch_size"]), int(config["max_length"]))
        current_state = _state_from_tensor(model.prototype, state)
        calibration_boundary = fit_boundary(train_embeddings=train_emb, calibration_embeddings=cal, rows=rows, state=current_state, radius_lambda=float(config["radius_lambda"]))
        cal_metrics = evaluate_boundary(calibration_boundary, cal, calibration, threshold=float(config["threshold"]))[0]
        history.append({"epoch": epoch, "loss": float(np.mean([x["total"] for x in parts])), "ce": float(np.mean([x["ce"] for x in parts])), "intra": float(np.mean([x["intra"] for x in parts])), "inter": float(np.mean([x["inter"] for x in parts])), "calibration_known_recall": cal_metrics["id_recall"]})
    state = _state_from_tensor(model.prototype, state)
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "center_intents": list(state.center_intents), "parent_intents": list(state.parent_intents), "history": history, "center_state": state.centers, "parent_centers": state.parent_centers}
    _atomic_torch_save(checkpoint_path, payload)
    return model, state, {"history": history, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path), "checkpoint_centers": int(len(state.centers))}


def _fit_diag(values: np.ndarray, center: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    diff = values - center[None, :]
    var = np.var(diff, axis=0) + eps
    dist = np.sqrt(np.sum((diff * diff) / var[None, :], axis=1))
    radius = max(float(dist.mean() + dist.std()), 1e-6)
    return 1.0 / var, radius


def fit_boundary(*, train_embeddings: np.ndarray | None, calibration_embeddings: np.ndarray, rows: Sequence[Mapping[str, Any]], state: CenterState, radius_lambda: float, eps: float = 1e-5) -> BoundaryState:
    values = _normalize(calibration_embeddings if train_embeddings is None else train_embeddings)
    if len(values) != len(rows):
        raise ValueError("Boundary rows and embeddings are not aligned")
    centers = _normalize(state.centers)
    # The parent boundary must live in the *current* trained representation,
    # not in the pre-training embedding space.  Recompute one parent centroid
    # per intent from the same Known train rows used for the child radii.
    parent_centers = _normalize(np.asarray([
        values[[str(row["intent"]) == intent for row in rows]].mean(axis=0)
        for intent in state.parent_intents
    ], dtype=np.float32))
    center_intents = state.center_intents
    assignments = np.full(len(values), -1, dtype=np.int64)
    for intent in state.intent_names:
        sample_indices = np.asarray([i for i, row in enumerate(rows) if str(row["intent"]) == intent], dtype=np.int64)
        center_indices = np.asarray([i for i, name in enumerate(center_intents) if name == intent], dtype=np.int64)
        dist = 1.0 - values[sample_indices] @ centers[center_indices].T
        assignments[sample_indices] = center_indices[dist.argmin(axis=1)]
    inv_cov, radii = [], []
    for center_index in range(len(centers)):
        local = values[assignments == center_index]
        if len(local) == 0:
            local = values
        inv, radius = _fit_diag(local, centers[center_index], eps)
        inv_cov.append(inv)
        radii.append(radius)
    parent_inv, parent_radii = [], []
    for intent, center in zip(state.parent_intents, parent_centers, strict=True):
        local = values[[str(row["intent"]) == intent for row in rows]]
        inv, radius = _fit_diag(local, center, eps)
        parent_inv.append(inv)
        parent_radii.append(radius)
    return BoundaryState(centers, center_intents, np.asarray(radii), np.asarray(inv_cov), parent_centers, state.parent_intents, np.asarray(parent_radii), np.asarray(parent_inv))


def _scores(boundary: BoundaryState, values: np.ndarray, threshold: float) -> dict[str, Any]:
    values = _normalize(values)
    d = 1.0 - values @ boundary.centers.T
    q = np.sqrt(np.maximum(d, 0.0) * boundary.inv_diag_cov[None, :]).sum(axis=2) if False else np.sqrt(np.maximum(np.einsum("nkd,kd->nk", (values[:, None, :] - boundary.centers[None, :, :]) ** 2, boundary.inv_diag_cov), 0.0))
    normalized = q / boundary.radii[None, :]
    parent_q = np.sqrt(np.maximum(np.einsum("nkd,kd->nk", (values[:, None, :] - boundary.parent_centers[None, :, :]) ** 2, boundary.parent_inv_diag_cov), 0.0))
    parent_norm = parent_q / boundary.parent_radii[None, :]
    center_scores = normalized.copy()
    for index, intent in enumerate(boundary.center_intents):
        parent_index = boundary.parent_intents.index(intent)
        center_scores[:, index] = np.where(parent_norm[:, parent_index] <= threshold, center_scores[:, index], np.inf)
    intent_scores = np.full((len(values), len(boundary.parent_intents)), np.inf, dtype=np.float64)
    for index, intent in enumerate(boundary.parent_intents):
        child = [i for i, name in enumerate(boundary.center_intents) if name == intent]
        intent_scores[:, index] = np.min(center_scores[:, child], axis=1)
    unconstrained_intent_scores = np.full((len(values), len(boundary.parent_intents)), np.inf, dtype=np.float64)
    for index, intent in enumerate(boundary.parent_intents):
        child = [i for i, name in enumerate(boundary.center_intents) if name == intent]
        unconstrained_intent_scores[:, index] = np.min(normalized[:, child], axis=1)
    best = intent_scores.argmin(axis=1)
    best_score = intent_scores[np.arange(len(values)), best]
    # The evaluator requires finite scores.  A parent-guard rejection is
    # represented by a large finite sentinel rather than ``inf`` so AUROC and
    # threshold metrics remain well-defined.
    best_score = np.where(np.isfinite(best_score), best_score, 1.0e6)
    accepted = np.isfinite(best_score) & (best_score <= threshold)
    unconstrained_best = unconstrained_intent_scores.min(axis=1)
    return {"intent_scores": intent_scores, "score": best_score, "unconstrained_score": unconstrained_best, "accepted": accepted, "best_intent": np.asarray(boundary.parent_intents, dtype=object)[best], "center_scores": center_scores}


def evaluate_boundary(boundary: BoundaryState, embeddings: np.ndarray, rows: Sequence[Mapping[str, Any]], threshold: float = 1.0) -> tuple[dict[str, float], list[dict[str, Any]]]:
    output = _scores(boundary, embeddings, threshold)
    gold_binary = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    binary = compute_binary_oos_metrics(gold_binary, output["score"], threshold)
    known_names = list(boundary.parent_intents)
    truth = [OOS_LABEL if int(row["label"]) == 1 else str(row["intent"]) for row in rows]
    predicted = [OOS_LABEL if not output["accepted"][i] else str(output["best_intent"][i]) for i in range(len(rows))]
    labels = [*known_names, OOS_LABEL]
    metrics = {
        **binary,
        "f1_all": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "f1_u": float(f1_score(truth, predicted, labels=[OOS_LABEL], average="macro", zero_division=0)),
        "f1_k": float(f1_score(truth, predicted, labels=known_names, average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(truth, dtype=object) == np.asarray(predicted, dtype=object))),
        "calibration_mean_score": float(np.mean(output["unconstrained_score"])) if len(output["unconstrained_score"]) else math.nan,
    }
    predictions = []
    for i, row in enumerate(rows):
        predictions.append({"sample_id": str(row["sample_id"]), "gold_intent": str(row["intent"]), "gold_is_oos": int(row["label"]), "predicted_intent": predicted[i], "predicted_is_oos": int(not output["accepted"][i]), "score": float(output["score"][i]), "best_intent": str(output["best_intent"][i])})
    return metrics, predictions


def _calibration_objective(metrics: Mapping[str, float], complexity: int) -> float:
    return float(metrics["id_recall"] - 0.02 * metrics["calibration_mean_score"] - 0.002 * complexity)


def _make_candidate_model(base: JointPrototypeModel, candidate_state: CenterState, values: np.ndarray, rows: Sequence[Mapping[str, Any]], model_path: Path, projection_hidden_dim: int, device: torch.device) -> JointPrototypeModel:
    candidate = JointPrototypeModel(model_path, candidate_state, projection_hidden_dim).to(device)
    candidate.encoder.load_state_dict(base.encoder.state_dict())
    with torch.no_grad():
        candidate.prototype.copy_(torch.as_tensor(candidate_state.centers, dtype=torch.float32, device=device))
    return candidate


def _train_one_seed(paths: ProtocolV2Paths, config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    started = time.time()
    _seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    views = _load_views(paths, seed)
    model_path = _model_path(paths, config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    base = RacalMiniLM(model_path, "last2_minilm_plus_projection", int(config["projection_hidden_dim"])).to(device)
    initial_checkpoint = _initial_checkpoint(paths, config, seed)
    if initial_checkpoint is not None:
        initial_payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        base.load_state_dict(initial_payload["model"])
    base.eval()
    # Initial K=1 prototypes use only Known train embeddings.
    train_initial = encode_rows(base, tokenizer, views.train, device, int(config["batch_size"]), int(config["max_length"]))
    state = _initial_centers(train_initial, views.train)
    model = JointPrototypeModel(model_path, state, int(config["projection_hidden_dim"])).to(device)
    model.encoder.load_state_dict(base.state_dict())
    with torch.no_grad():
        model.prototype.copy_(torch.as_tensor(state.centers, dtype=torch.float32, device=device))
    del base
    root = _root(paths) / "runs" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    model, state, training = _train_model(model, tokenizer, views.train, views.calibration, state, config, device, seed, int(config["base_epochs"]), root / "joint_k1.pt")
    model.eval()
    train_emb = encode_rows(model.encoder, tokenizer, views.train, device, int(config["batch_size"]), int(config["max_length"]))
    cal_emb = encode_rows(model.encoder, tokenizer, views.calibration, device, int(config["batch_size"]), int(config["max_length"]))
    test_emb = encode_rows(model.encoder, tokenizer, views.test, device, int(config["batch_size"]), int(config["max_length"]))
    current_boundary = fit_boundary(train_embeddings=train_emb, calibration_embeddings=cal_emb, rows=views.train, state=state, radius_lambda=float(config["radius_lambda"]))
    base_cal_metrics = evaluate_boundary(current_boundary, cal_emb, views.calibration, float(config["threshold"]))[0]
    events = []
    candidates = _candidate_intents(train_emb, views.train, int(config["max_candidate_splits"]), int(config["min_child_samples"]))
    accepted = 0
    for candidate in candidates:
        if accepted >= int(config["max_accepted_splits"]):
            break
        candidate_state = _make_center_state_after_split(state, candidate, train_emb, views.train)
        candidate_model = _make_candidate_model(model, candidate_state, train_emb, views.train, model_path, int(config["projection_hidden_dim"]), device)
        candidate_model, candidate_state, candidate_training = _train_model(candidate_model, tokenizer, views.train, views.calibration, candidate_state, config, device, seed + accepted + 100, int(config["split_epochs"]), root / f"candidate_{accepted + 1}_{candidate['intent']}.pt", learning_rate=float(config.get("candidate_learning_rate", config["learning_rate"])))
        candidate_model.eval()
        candidate_train = encode_rows(candidate_model.encoder, tokenizer, views.train, device, int(config["batch_size"]), int(config["max_length"]))
        candidate_cal = encode_rows(candidate_model.encoder, tokenizer, views.calibration, device, int(config["batch_size"]), int(config["max_length"]))
        candidate_boundary = fit_boundary(train_embeddings=candidate_train, calibration_embeddings=candidate_cal, rows=views.train, state=candidate_state, radius_lambda=float(config["radius_lambda"]))
        cal_metrics = evaluate_boundary(candidate_boundary, candidate_cal, views.calibration, float(config["threshold"]))[0]
        objective_before = _calibration_objective(base_cal_metrics, len(state.centers))
        objective_after = _calibration_objective(cal_metrics, len(candidate_state.centers))
        event = {"intent": candidate["intent"], "before_centers": len(state.centers), "after_centers": len(candidate_state.centers), "candidate_residual_p90": candidate["residual_p90"], "calibration_before": base_cal_metrics, "calibration_after": cal_metrics, "objective_before": objective_before, "objective_after": objective_after, "known_recall_delta": float(cal_metrics["id_recall"] - base_cal_metrics["id_recall"]), "accepted": False, "reject_reason": None, "candidate_checkpoint": candidate_training["checkpoint_sha256"]}
        score_gain = float(base_cal_metrics["calibration_mean_score"] - cal_metrics["calibration_mean_score"])
        event["calibration_score_gain"] = score_gain
        if event["known_recall_delta"] < -float(config["max_calibration_recall_drop"]):
            event["reject_reason"] = "calibration_known_recall_drop"
        elif score_gain < float(config["minimum_calibration_score_gain"]):
            event["reject_reason"] = "no_known_only_compactness_gain"
        elif objective_after < objective_before - float(config["maximum_objective_regression"]):
            event["reject_reason"] = "no_known_only_objective_gain"
        else:
            event["accepted"] = True
            accepted += 1
            old_model = model
            state = candidate_state
            model = candidate_model
            train_emb, cal_emb = candidate_train, candidate_cal
            current_boundary = candidate_boundary
            base_cal_metrics = cal_metrics
            del old_model
        events.append(event)
        if not event["accepted"]:
            del candidate_model
        else:
            # Keep only the accepted model; the previous model and its CUDA
            # allocations are no longer part of the adaptive state.
            del candidate_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    model.eval()
    final_boundary = fit_boundary(train_embeddings=train_emb, calibration_embeddings=cal_emb, rows=views.train, state=state, radius_lambda=float(config["radius_lambda"]))
    final_test = encode_rows(model.encoder, tokenizer, views.test, device, int(config["batch_size"]), int(config["max_length"]))
    metrics, predictions = evaluate_boundary(final_boundary, final_test, views.test, float(config["threshold"]))
    metrics.update({"stage": STAGE, "dataset": DATASET, "kir": KIR, "seed": seed, "method": "joint_adaptive", "device": str(device), "accepted_split_count": accepted, "mean_k_y": float(np.mean(list(state.counts().values()))), "max_k_y": max(state.counts().values()), "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "elapsed_seconds": time.time() - started, "test_used_for_selection": False, "oos_used_for_training": False, "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None, "initial_checkpoint_sha256": sha256_file(initial_checkpoint) if initial_checkpoint else None, "train_ids_sha256": _hash_rows(views.train), "calibration_ids_sha256": _hash_rows(views.calibration), "test_ids_sha256": _hash_rows(views.test), "test_embedding_sha256": _hash_array(final_test)})
    payload = {"schema_version": "s2c.joint_adaptive_v1.run.v1", "status": "complete", "config": dict(config), "metrics": metrics, "center_counts": state.counts(), "split_events": events, "training": training, "parent_guard": True, "center_intents": list(state.center_intents), "checkpoint_sha256": training["checkpoint_sha256"], "views": {"train": len(views.train), "calibration": len(views.calibration), "test": len(views.test), "test_oos": int(sum(int(row["label"]) for row in views.test))}}
    atomic_write_json(root / "run_manifest.json", payload)
    atomic_write_json(root / "metrics.json", metrics)
    atomic_write_json(root / "split_events.json", events)
    atomic_write_json(root / "center_counts.json", state.counts())
    _write_csv(root / "predictions.csv", predictions)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def freeze_provenance(paths: ProtocolV2Paths, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(paths)
    root.mkdir(parents=True, exist_ok=True)
    patch_path = root / "JOINT_ADAPTIVE_CODE.patch"
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=paths.project_root.parent, check=True, capture_output=True)
    source_files = [
        paths.project_root / "src/protocol_v2/experiments/joint_adaptive_v1/__init__.py",
        paths.project_root / "src/protocol_v2/experiments/joint_adaptive_v1/runner.py",
        paths.project_root / "scripts/experiments/run_joint_adaptive_multicenter_v1.py",
        config_path,
    ]
    source_manifest = {str(path.relative_to(paths.project_root)): sha256_file(path) for path in source_files}
    atomic_write_json(root / "JOINT_ADAPTIVE_SOURCE_MANIFEST.json", source_manifest)
    patch_text = diff.stdout.decode("utf-8", errors="replace")
    patch_text += "\n# Untracked/independent source hashes\n"
    patch_text += json.dumps(source_manifest, ensure_ascii=False, sort_keys=True, indent=2)
    atomic_write_text(patch_path, patch_text)
    model_path = _model_path(paths, config)
    initial_checkpoint_hashes = {}
    for seed in SEEDS:
        checkpoint = _initial_checkpoint(paths, config, seed)
        initial_checkpoint_hashes[str(seed)] = None if checkpoint is None else {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    payload = {"schema_version": "s2c.joint_adaptive_v1.provenance.v1", "stage": STAGE, "protocol_version": paths.dataset_version, "dataset": DATASET, "kir": KIR, "seeds": list(SEEDS), "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.project_root.parent, check=True, capture_output=True, text=True).stdout.strip(), "git_dirty": bool(subprocess.run(["git", "status", "--short"], cwd=paths.project_root.parent, check=True, capture_output=True, text=True).stdout.strip()), "code_patch_sha256": sha256_file(patch_path), "source_files": source_manifest, "config_sha256": sha256_file(config_path), "model_path": str(model_path), "model_config_sha256": sha256_file(model_path / "config.json"), "initial_checkpoint_hashes": initial_checkpoint_hashes, "python": os.sys.version, "torch": torch.__version__, "numpy": np.__version__, "selection": "Known train residual + Known calibration only", "test_used_for_selection": False, "oos_used_for_training": False, "created_at": datetime.now(UTC).isoformat()}
    atomic_write_json(root / "JOINT_ADAPTIVE_PROVENANCE.json", payload)
    return payload


def _verify_provenance(paths: ProtocolV2Paths, config_path: Path) -> dict[str, Any]:
    root = _root(paths)
    payload = json.loads((root / "JOINT_ADAPTIVE_PROVENANCE.json").read_text(encoding="utf-8"))
    if payload["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("Adaptive pilot config changed after provenance freeze")
    if payload["code_patch_sha256"] != sha256_file(root / "JOINT_ADAPTIVE_CODE.patch"):
        raise RuntimeError("Adaptive pilot code patch changed after provenance freeze")
    return payload


def run(paths: ProtocolV2Paths, config_path: Path, resume: bool, selected_seeds: Sequence[int] | None = None) -> dict[str, Any]:
    config = _load_config(config_path)
    paths.require_experiment_admission(DATASET)
    root = _root(paths)
    if (root / "JOINT_ADAPTIVE_PROVENANCE.json").is_file():
        _verify_provenance(paths, config_path)
    else:
        freeze_provenance(paths, config_path, config)
    selected = tuple(int(seed) for seed in (selected_seeds or SEEDS))
    if any(seed not in SEEDS for seed in selected):
        raise ValueError(f"Requested seeds must be drawn from {SEEDS}: {selected}")
    results = []
    for seed in selected:
        run_dir = root / "runs" / f"seed_{seed}"
        if resume and (run_dir / "run_manifest.json").is_file():
            results.append(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")))
            continue
        results.append(_train_one_seed(paths, config, seed))
    rows = []
    for result in results:
        row = dict(result["metrics"])
        row["center_counts"] = json.dumps(result["center_counts"], ensure_ascii=False, sort_keys=True)
        row["accepted_split_count"] = result["metrics"]["accepted_split_count"]
        rows.append(row)
    _write_csv(root / "JOINT_ADAPTIVE_RESULTS.csv", rows)
    return {"stage": STAGE, "status": "complete", "completed_seeds": len(results), "results": rows, "root": str(root)}


def summarize(paths: ProtocolV2Paths) -> dict[str, Any]:
    path = _root(paths) / "JOINT_ADAPTIVE_RESULTS.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    numeric = ("oos_f1", "f1_all", "f1_k", "accuracy", "id_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos")
    summary = {"stage": STAGE, "dataset": DATASET, "kir": KIR, "seed_count": len(rows), "metrics": {}}
    for name in numeric:
        vals = [float(row[name]) for row in rows]
        summary["metrics"][name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "values": vals}
    summary["mean_k_y"] = {"mean": float(np.mean([float(row["mean_k_y"]) for row in rows])), "values": [float(row["mean_k_y"]) for row in rows]}
    atomic_write_json(_root(paths) / "JOINT_ADAPTIVE_SUMMARY.json", summary)
    lines = ["# Joint adaptive multicenter pilot", "", f"- Dataset: `{DATASET}`, KIR=`{KIR}`, seeds=`{SEEDS}`", "- Training: RACAL MiniLM adapter plus trainable intent prototypes.", "- Adaptation: PCA candidate split on train embeddings; acceptance uses Known calibration only.", "- Parent guard: enabled; child centers cannot expand the parent class boundary.", "", "## Aggregate metrics", ""]
    for name, item in summary["metrics"].items():
        lines.append(f"- `{name}`: {item['mean']:.6f} ± {item['std']:.6f}")
    lines.extend(["", f"- mean K_y: {summary['mean_k_y']['mean']:.3f}", "- Test OOS labels were not used for training, split acceptance, or checkpoint selection."])
    atomic_write_text(_root(paths) / "JOINT_ADAPTIVE_REPORT.md", "\n".join(lines) + "\n")
    return summary


def verify(paths: ProtocolV2Paths) -> dict[str, Any]:
    root = _root(paths)
    manifests = sorted((root / "runs").glob("seed_*/run_manifest.json"))
    failures = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            failures.append(f"incomplete:{path}")
        if payload.get("metrics", {}).get("test_used_for_selection"):
            failures.append(f"test_selection:{path}")
        if payload.get("metrics", {}).get("oos_used_for_training"):
            failures.append(f"oos_training:{path}")
    report = {"stage": STAGE, "planned_seeds": len(SEEDS), "completed_seeds": len(manifests), "failures": failures, "provenance": (root / "JOINT_ADAPTIVE_PROVENANCE.json").is_file(), "status": "complete" if len(manifests) == len(SEEDS) and not failures else "incomplete"}
    atomic_write_json(root / "JOINT_ADAPTIVE_VERIFY.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "summarize", "verify", "freeze"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, action="append", dest="selected_seeds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProtocolV2Paths.discover()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if args.command == "freeze":
        result = freeze_provenance(paths, config_path, config)
    elif args.command == "run":
        result = run(paths, config_path, args.resume, args.selected_seeds)
    elif args.command == "summarize":
        result = summarize(paths)
    else:
        result = verify(paths)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
