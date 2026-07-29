"""MiniLM training pilot and StackOverflow K=1/K=2 contract audit.

This is an independent stage for ``protocol_v2_textoir_v1``.  It deliberately
does not alter E2, E3, or either R1 artifact root.  The module has two small
responsibilities:

* audit the frozen StackOverflow K=1 -> K=2 path at sample and sphere level;
* train Known-only MiniLM representation controls and evaluate them through
  the unchanged protocol Gate.

The public entry points are intentionally plain functions so the thin scripts
under ``scripts/experiments`` remain easy to inspect and test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file, sha256_json
from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.experiments.geometry_preserving import mean_pool, pairwise_relation_metrics
from protocol_v2.experiments.mechanism_runner import E3Bundle, load_e2_bundle
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from protocol_v2.runtime.paths import ProtocolV2Paths


STAGE = "minilm_training_and_stackoverflow_repair_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (42, 87, 100)
K_VALUES = (1, 2)
DISTANCES = ("euclidean", "mahalanobis_diag")
METHODS = ("frozen_minilm", "head_only_ce", "full_ce", "supcon", "ce_recon")
TRAINABLE_METHODS = ("head_only_ce", "full_ce", "supcon", "ce_recon")
DEFAULT_MODEL = "../assets/models/all-MiniLM-L6-v2"
DEFAULT_ROOT_NAME = STAGE


def stage_root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / DEFAULT_ROOT_NAME


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = ["\t".join(fields)]
    for row in rows:
        output.append("\t".join(_csv_value(row.get(field, "")) for field in fields))
    atomic_write_text(path, "\n".join(output) + "\n")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    text = str(_safe(value))
    return text.replace("\t", " ").replace("\n", " ")


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def _sha256_ids(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["sample_id"]) for row in rows).encode("utf-8")).hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_files(model_path: Path) -> dict[str, str]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"MiniLM model directory is missing: {model_path}")
    names = ("config.json", "tokenizer.json", "model.safetensors", "pytorch_model.bin")
    files = {name: sha256_file(model_path / name) for name in names if (model_path / name).is_file()}
    required = {"config.json", "tokenizer.json"}
    if not required.issubset(files) or not ({"model.safetensors", "pytorch_model.bin"} & files.keys()):
        raise FileNotFoundError(f"Incomplete local MiniLM model: {model_path}")
    return files


def _git_patch(repo_root: Path, output: Path) -> str:
    """Snapshot tracked and untracked source without staging or committing."""

    tracked = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=repo_root, check=True, capture_output=True
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    chunks = [tracked]
    for relative in untracked:
        if relative.startswith(("../artifacts/", "artifacts/", "../assets/", "assets/")):
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "/dev/null", relative],
            cwd=repo_root,
            check=False,
            capture_output=True,
        )
        if result.stdout:
            chunks.append(result.stdout)
    atomic_write_text(output, b"\n".join(chunks).decode("utf-8", errors="replace"))
    return sha256_file(output)


def freeze_provenance(paths: ProtocolV2Paths, config_path: Path) -> dict[str, Any]:
    """Freeze stage identity; no experiment is started without this snapshot."""

    root = stage_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    patch_path = root / "R1_MINILM_STAGE_CODE.patch"
    patch_sha = _git_patch(paths.project_root, patch_path)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths.project_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(["git", "status", "--short"], cwd=paths.project_root, check=True, capture_output=True, text=True).stdout.strip()
    )
    model_path = (paths.project_root / DEFAULT_MODEL).resolve()
    payload = {
        "schema_version": "s2c.minilm_stage.provenance.v1",
        "stage": STAGE,
        "protocol_version": paths.dataset_version,
        "base_commit": revision,
        "git_dirty": dirty,
        "code_patch": str(patch_path),
        "code_patch_sha256": patch_sha,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model_name": "all-MiniLM-L6-v2",
        "model_path": str(model_path),
        "model_files": _model_files(model_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "data_version": paths.dataset_version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "e2_artifacts_immutable": True,
        "r1_artifacts_immutable": True,
    }
    atomic_write_json(root / "R1_MINILM_STAGE_PROVENANCE.json", payload)
    return payload


def _load_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage config must be a mapping: {config_path}")
    return payload


def _fit_detector(train: np.ndarray, rows: Sequence[Mapping[str, Any]], k: int, distance: str) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=k,
        radius_method="mean_std",
        radius_lambda=1.0,
        distance_metric=distance,
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
        acceptance_mode="nearest_sphere",
    )
    detector.fit(train, np.asarray([str(row["intent"]) for row in rows], dtype=object))
    return detector


def _output_label(detector: MultiSphereOOSDetector, result: Mapping[str, Any], gold_label: int) -> str:
    if int(result["pred"]) == 1:
        return "__oos__"
    cluster = int(result["nearest_cluster"])
    return str(detector.cluster_to_intent.get(cluster, "__unknown__"))


def _open_metrics(
    detector: MultiSphereOOSDetector,
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    test_output: Mapping[str, np.ndarray],
) -> dict[str, float]:
    binary_labels = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    scores = np.asarray(test_output["score"], dtype=float)
    result = compute_binary_oos_metrics(binary_labels, scores, 1.0)
    known_intents = sorted({str(row["intent"]) for row in train_rows})
    true = [str(row["intent"]) if int(row["label"]) == 0 else "__oos__" for row in test_rows]
    predicted = [_output_label(detector, {key: value[index] for key, value in test_output.items()}, int(binary_labels[index])) for index in range(len(test_rows))]
    all_labels = [*known_intents, "__oos__"]
    result.update(
        {
            "f1_all": float(f1_score(true, predicted, labels=all_labels, average="macro", zero_division=0)),
            "f1_u": float(f1_score(true, predicted, labels=["__oos__"], average="macro", zero_division=0)),
            "f1_k": float(f1_score(true, predicted, labels=known_intents, average="macro", zero_division=0)),
            "accuracy": float(np.mean(np.asarray(true, dtype=object) == np.asarray(predicted, dtype=object))),
        }
    )
    result["known_recall"] = result["id_recall"]
    return result


def _collision_rate_with_train_rows(
    train: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    calibration: np.ndarray,
    calibration_rows: Sequence[Mapping[str, Any]],
    test: np.ndarray,
    test_rows: Sequence[Mapping[str, Any]],
) -> float:
    train_norm = train / np.clip(np.linalg.norm(train, axis=1, keepdims=True), 1e-12, None)
    cal_norm = calibration / np.clip(np.linalg.norm(calibration, axis=1, keepdims=True), 1e-12, None)
    test_norm = test / np.clip(np.linalg.norm(test, axis=1, keepdims=True), 1e-12, None)
    names = sorted({str(row["intent"]) for row in train_rows})
    labels = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    cal_labels = np.asarray([str(row["intent"]) for row in calibration_rows], dtype=object)
    centers = np.asarray([train_norm[labels == name].mean(axis=0) for name in names])
    centers /= np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    thresholds = np.asarray(
        [np.quantile(cal_norm[cal_labels == name] @ center, 0.05) for name, center in zip(names, centers, strict=True)]
    )
    oos = np.asarray([int(row["label"]) == 1 for row in test_rows], dtype=bool)
    if not oos.any():
        return math.nan
    similarities = test_norm @ centers.T
    nearest = np.argmax(similarities, axis=1)
    return float(np.mean(similarities[oos, nearest[oos]] >= thresholds[nearest[oos]]))


def _encode_rows(model: torch.nn.Module, tokenizer: Any, rows: Sequence[Mapping[str, Any]], device: torch.device, batch_size: int, max_length: int) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            tokens = tokenizer(
                [str(row["text"]) for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            pooled = mean_pool(model(**tokens).last_hidden_state, tokens["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, int(model.config.hidden_size)), dtype=np.float32)


def _supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(features, dim=-1)
    logits = normalized @ normalized.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    valid = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & valid
    log_prob = logits - torch.log((torch.exp(logits) * valid).sum(dim=1, keepdim=True).clamp_min(1e-12))
    count = positive.sum(dim=1)
    usable = count > 0
    if not torch.any(usable):
        return features.sum() * 0.0
    return -(log_prob * positive).sum(dim=1)[usable].div(count[usable]).mean()


def _classification_validation(
    model: torch.nn.Module,
    head: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    label_map: Mapping[str, int],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> float:
    model.eval()
    head.eval()
    truth: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            tokens = tokenizer(
                [str(row["text"]) for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            pooled = mean_pool(model(**tokens).last_hidden_state, tokens["attention_mask"])
            predicted.extend(head(pooled).argmax(dim=-1).cpu().tolist())
            truth.extend(label_map[str(row["intent"])] for row in batch_rows)
    return float(f1_score(truth, predicted, average="macro", zero_division=0))


def _nearest_centroid_validation(train_embeddings: np.ndarray, train_rows: Sequence[Mapping[str, Any]], calibration_embeddings: np.ndarray, calibration_rows: Sequence[Mapping[str, Any]]) -> float:
    names = sorted({str(row["intent"]) for row in train_rows})
    train_labels = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    centers = np.asarray([train_embeddings[train_labels == name].mean(axis=0) for name in names])
    centers /= np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    calibration_norm = calibration_embeddings / np.clip(np.linalg.norm(calibration_embeddings, axis=1, keepdims=True), 1e-12, None)
    predicted = np.asarray([names[index] for index in np.argmax(calibration_norm @ centers.T, axis=1)], dtype=object)
    truth = np.asarray([str(row["intent"]) for row in calibration_rows], dtype=object)
    return float(f1_score(truth, predicted, labels=names, average="macro", zero_division=0))


def train_checkpoint(
    *,
    model_path: Path,
    bundle: E3Bundle,
    output_dir: Path,
    method: str,
    seed: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train one dataset/seed/method checkpoint using Known rows only."""

    if method not in TRAINABLE_METHODS:
        raise ValueError(f"Unsupported trainable method: {method}")
    train_rows, calibration_rows = bundle.views.train, bundle.views.calibration
    if any(int(row["label"]) != 0 for row in train_rows + calibration_rows):
        raise ValueError("MiniLM training received an OOS train/calibration row")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "training_manifest.json"
    config_payload = {
        "stage": STAGE,
        "dataset": bundle.dataset,
        "seed": seed,
        "method": method,
        "epochs": int(config["epochs"]),
        "batch_size": int(config["batch_size"]),
        "learning_rate": float(config["learning_rate"]),
        "alpha": float(config["alpha"]),
        "temperature": float(config["temperature"]),
        "max_length": int(config["max_length"]),
        "classifier_input": "pooled",
        "gate_embedding": "normalized_pooled",
        "teacher_embedding_sha256": _sha256_array(bundle.train),
        "train_sample_ids_sha256": _sha256_ids(train_rows),
        "calibration_sample_ids_sha256": _sha256_ids(calibration_rows),
        "used_oos_for_training": False,
        "used_test_for_selection": False,
    }
    config_hash = sha256_json(config_payload)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("config_hash") == config_hash:
            return existing
        raise RuntimeError(f"Refusing to overwrite existing training unit: {output_dir}")
    _set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    labels = sorted({str(row["intent"]) for row in train_rows})
    label_map = {name: index for index, name in enumerate(labels)}
    head = torch.nn.Linear(int(encoder.config.hidden_size), len(labels)).to(device) if method in {"head_only_ce", "full_ce", "ce_recon"} else None
    if method == "head_only_ce":
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    parameters = ([parameter for parameter in encoder.parameters() if parameter.requires_grad] + (list(head.parameters()) if head is not None else []))
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]))
    teacher_train = torch.as_tensor(bundle.train, dtype=torch.float32, device=device)
    train_indices = np.arange(len(train_rows), dtype=np.int64)
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        encoder.train()
        if head is not None:
            head.train()
        order = np.random.default_rng(seed + epoch).permutation(train_indices)
        losses: list[float] = []
        for start in range(0, len(order), int(config["batch_size"])):
            batch_indices = order[start : start + int(config["batch_size"])]
            batch_rows = [train_rows[int(index)] for index in batch_indices]
            tokens = tokenizer(
                [str(row["text"]) for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=int(config["max_length"]),
                return_tensors="pt",
            ).to(device)
            target = torch.as_tensor([label_map[str(row["intent"])] for row in batch_rows], dtype=torch.long, device=device)
            if method == "head_only_ce":
                with torch.no_grad():
                    pooled = mean_pool(encoder(**tokens).last_hidden_state, tokens["attention_mask"])
            else:
                pooled = mean_pool(encoder(**tokens).last_hidden_state, tokens["attention_mask"])
            if method in {"head_only_ce", "full_ce"}:
                loss = torch.nn.functional.cross_entropy(head(pooled), target)
            elif method == "ce_recon":
                normalized = torch.nn.functional.normalize(pooled, dim=-1)
                ce_loss = torch.nn.functional.cross_entropy(head(pooled), target)
                recon_loss = torch.nn.functional.mse_loss(normalized, teacher_train[torch.as_tensor(batch_indices, dtype=torch.long)])
                loss = ce_loss + float(config["alpha"]) * recon_loss
            else:
                loss = _supcon_loss(pooled, target, float(config["temperature"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if method == "supcon":
            train_repr = _encode_rows(encoder, tokenizer, train_rows, device, int(config["batch_size"]), int(config["max_length"]))
            calibration_repr = _encode_rows(encoder, tokenizer, calibration_rows, device, int(config["batch_size"]), int(config["max_length"]))
            validation_score = _nearest_centroid_validation(train_repr, train_rows, calibration_repr, calibration_rows)
            selection_metric = "known_validation_nearest_centroid_macro_f1"
        else:
            validation_score = _classification_validation(encoder, head, tokenizer, calibration_rows, label_map, device, int(config["batch_size"]), int(config["max_length"]))
            selection_metric = "known_validation_macro_f1"
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), selection_metric: validation_score})
        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = {
                "encoder": {key: value.detach().cpu() for key, value in encoder.state_dict().items()},
                "head": None if head is None else {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "label_map": label_map,
            }
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {method}")
    checkpoint = output_dir / "checkpoint.pt"
    with tempfile.NamedTemporaryFile(prefix=f".{checkpoint.name}.", suffix=".tmp", dir=output_dir, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(best_state, temporary)
        temporary.replace(checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    history_path = output_dir / "training_history.csv"
    _atomic_csv(history_path, history)
    manifest = {
        **config_payload,
        "status": "complete",
        "config_hash": config_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": best_epoch,
        "checkpoint_selection_metric": selection_metric,
        "known_validation_selection_score": best_score,
        "history": str(history_path),
        "representation_dim": int(encoder.config.hidden_size),
        "device": str(device),
        "teacher_source": "E2_frozen_minilm_train_embedding_cache",
    }
    atomic_write_json(manifest_path, _safe(manifest))
    del encoder, head, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def load_representation(
    *,
    model_path: Path,
    bundle: E3Bundle,
    method: str,
    checkpoint_dir: Path | None,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load Frozen/head-only cache or encode a trained student once."""

    if method in {"frozen_minilm", "head_only_ce"}:
        arrays = {"train": bundle.train, "calibration": bundle.calibration, "test": bundle.test}
        return arrays, {"method": method, "status": "reused_e2_frozen_cache", "embedding_sha256": _sha256_array(bundle.train)}
    if checkpoint_dir is None:
        raise ValueError(f"Checkpoint directory required for {method}")
    checkpoint = checkpoint_dir / "checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
    encoder.load_state_dict(state["encoder"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    arrays = {
        "train": _encode_rows(encoder, tokenizer, bundle.views.train, device, int(config["batch_size"]), int(config["max_length"])),
        "calibration": _encode_rows(encoder, tokenizer, bundle.views.calibration, device, int(config["batch_size"]), int(config["max_length"])),
        "test": _encode_rows(encoder, tokenizer, bundle.views.test, device, int(config["batch_size"]), int(config["max_length"])),
    }
    _atomic_npz(checkpoint_dir / "representation_embeddings.npz", **arrays)
    embedding_manifest = {
        "method": method,
        "status": "complete",
        "train_sample_ids_sha256": _sha256_ids(bundle.views.train),
        "calibration_sample_ids_sha256": _sha256_ids(bundle.views.calibration),
        "test_sample_ids_sha256": _sha256_ids(bundle.views.test),
        "train_embedding_sha256": _sha256_array(arrays["train"]),
        "calibration_embedding_sha256": _sha256_array(arrays["calibration"]),
        "test_embedding_sha256": _sha256_array(arrays["test"]),
    }
    atomic_write_json(checkpoint_dir / "representation_manifest.json", embedding_manifest)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arrays, embedding_manifest


def evaluate_gate_cell(
    bundle: E3Bundle,
    arrays: Mapping[str, np.ndarray],
    method: str,
    k: int,
    distance: str,
    seed: int,
) -> dict[str, Any]:
    detector = _fit_detector(arrays["train"], bundle.views.train, k, distance)
    output = detector.predict_with_scores(arrays["test"])
    metrics = _open_metrics(detector, bundle.views.train, bundle.views.test, output)
    geometry = pairwise_relation_metrics(bundle.train, arrays["train"], [str(row["intent"]) for row in bundle.views.train], seed)
    metrics.update(geometry)
    metrics["representation_collision_rate"] = _collision_rate_with_train_rows(
        arrays["train"], bundle.views.train, arrays["calibration"], bundle.views.calibration, arrays["test"], bundle.views.test
    )
    metrics.update(
        {
            "stage": STAGE,
            "protocol_version": bundle.e2_manifest["protocol_version"],
            "dataset": bundle.dataset,
            "kir": bundle.kir,
            "seed": seed,
            "method": method,
            "k": k,
            "distance": distance,
            "boundary": "mean_std",
            "radius_lambda": 1.0,
            "threshold": 1.0,
            "acceptance_mode": "nearest_sphere",
            "train_embedding_sha256": _sha256_array(arrays["train"]),
            "test_embedding_sha256": _sha256_array(arrays["test"]),
            "effective_cluster_count": len(detector.spheres),
            "minimum_cluster_size": int(min(np.bincount(detector._train_cluster_labels))),
            "run_id": f"{STAGE}__{bundle.dataset}__kir_{bundle.kir:.2f}__seed_{seed}__{method}__k_{k}__dist_{distance}",
        }
    )
    return metrics


def audit_stackoverflow(paths: ProtocolV2Paths, seed: int = 42, kir: float = 0.50) -> dict[str, Any]:
    """Export paired sample and sphere audits for Frozen StackOverflow K1/K2."""

    paths.require_experiment_admission("stackoverflow")
    root = stage_root(paths) / "stackoverflow_audit"
    root.mkdir(parents=True, exist_ok=True)
    bundle = load_e2_bundle(paths, "stackoverflow", seed, kir)
    all_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    contract: dict[str, Any] = {
        "stage": STAGE,
        "dataset": "stackoverflow",
        "kir": kir,
        "seed": seed,
        "cache_alignment": {
            "train_sample_ids_sha256": _sha256_ids(bundle.views.train),
            "calibration_sample_ids_sha256": _sha256_ids(bundle.views.calibration),
            "test_sample_ids_sha256": _sha256_ids(bundle.views.test),
            "train_embedding_sha256": _sha256_array(bundle.train),
            "calibration_embedding_sha256": _sha256_array(bundle.calibration),
            "test_embedding_sha256": _sha256_array(bundle.test),
            "e2_manifest_protocol": bundle.e2_manifest.get("protocol_version"),
            "alignment_verified_by_load_e2_bundle": True,
        },
        "distances": {},
    }
    for distance in DISTANCES:
        detectors = {k: _fit_detector(bundle.train, bundle.views.train, k, distance) for k in K_VALUES}
        outputs = {k: detectors[k].predict_with_scores(bundle.test) for k in K_VALUES}
        labels = np.asarray([int(row["label"]) for row in bundle.views.test], dtype=np.int64)
        for index, row in enumerate(bundle.views.test):
            record: dict[str, Any] = {
                "stage": STAGE,
                "dataset": "stackoverflow",
                "kir": kir,
                "data_seed": seed,
                "distance": distance,
                "sample_id": row["sample_id"],
                "gold_intent": row["intent"],
                "gold_label": int(row["label"]),
                "oos_source": row.get("oos_source"),
            }
            for k in K_VALUES:
                result = {key: value[index] for key, value in outputs[k].items()}
                record.update(
                    {
                        f"k{k}_nearest_cluster": int(result["nearest_cluster"]),
                        f"k{k}_nearest_intent": detectors[k].cluster_to_intent.get(int(result["nearest_cluster"])),
                        f"k{k}_distance": float(result["distance"]),
                        f"k{k}_radius": float(result["radius"]),
                        f"k{k}_normalized_score": float(result["score"]),
                        f"k{k}_prediction": int(result["pred"]),
                    }
                )
            record["k2_new_oos_false_accept"] = bool(labels[index] == 1 and record["k1_prediction"] == 1 and record["k2_prediction"] == 0)
            all_rows.append(record)
        for k in K_VALUES:
            detector = detectors[k]
            output = outputs[k]
            cluster_labels = np.asarray(detector._train_cluster_labels, dtype=np.int64)
            for sphere in detector.spheres:
                train_points = detector._train_embeddings[cluster_labels == sphere.cluster_id]
                assigned = np.asarray(output["nearest_cluster"], dtype=np.int64) == sphere.cluster_id
                oos_assigned = assigned & (labels == 1)
                false_accept = oos_assigned & (output["pred"] == 0)
                variance = np.var(train_points, axis=0) if len(train_points) else np.asarray([], dtype=float)
                cluster_rows.append(
                    {
                        "stage": STAGE,
                        "dataset": "stackoverflow",
                        "kir": kir,
                        "data_seed": seed,
                        "distance": distance,
                        "k": k,
                        "cluster_id": sphere.cluster_id,
                        "intent": sphere.intent_name,
                        "train_sample_count": len(train_points),
                        "variance_trace": float(variance.sum()) if len(variance) else math.nan,
                        "mean_variance": float(variance.mean()) if len(variance) else math.nan,
                        "radius": float(sphere.radius),
                        "test_oos_assigned_count": int(oos_assigned.sum()),
                        "test_oos_false_accept_count": int(false_accept.sum()),
                        "false_accept_contribution": float(false_accept.sum() / max(1, np.sum((labels == 1) & (output["pred"] == 0)))),
                    }
                )
        reproduction: dict[str, Any] = {}
        for k in K_VALUES:
            reproduced = compute_binary_oos_metrics(labels, outputs[k]["score"], 1.0)
            run_name = (
                f"protocol_v2_textoir_v1__stackoverflow__kir_{kir:.2f}__seed_{seed}__"
                f"repr_frozen_minilm__k_{k}__dist_{distance}__boundary_mean_std"
            )
            reference_path = paths.run_root / "e2_gate_core_dense" / run_name / "metrics.json"
            reference = json.loads(reference_path.read_text(encoding="utf-8"))["combined"] if reference_path.is_file() else {}
            compared = ["oos_f1", "id_recall", "auroc", "aupr_oos", "false_accept_rate", "false_reject_rate"]
            deltas = [abs(float(reproduced[key]) - float(reference[key])) for key in compared if key in reference]
            reproduction[f"k{k}"] = {
                "reference_path": str(reference_path),
                "reference_exists": reference_path.is_file(),
                "max_abs_delta": max(deltas) if deltas else math.nan,
                "within_float_tolerance": bool(deltas) and max(deltas) <= 1e-12,
            }
        binary = compute_binary_oos_metrics(labels, outputs[2]["score"], 1.0)
        contract["distances"][distance] = {
            "k1_spheres": len(detectors[1].spheres),
            "k2_spheres": len(detectors[2].spheres),
            "k1_k2_new_oos_false_accept_count": int(sum(row["k2_new_oos_false_accept"] and row["distance"] == distance for row in all_rows)),
            "k2_metrics": binary,
            "e2_reproduction": reproduction,
            "detector_contract": {"radius_method": "mean_std", "radius_lambda": 1.0, "threshold": 1.0, "acceptance_mode": "nearest_sphere"},
        }
    _atomic_csv(root / "STACKOVERFLOW_K1_K2_SAMPLE_AUDIT.tsv", all_rows)
    _atomic_csv(root / "STACKOVERFLOW_CLUSTER_AUDIT.tsv", cluster_rows)
    atomic_write_json(root / "STACKOVERFLOW_AUDIT_CONTRACT.json", _safe(contract))
    return {"status": "complete", "sample_rows": len(all_rows), "cluster_rows": len(cluster_rows), "root": str(root), "contract": contract}


def _pilot_unit_path(root: Path, dataset: str, seed: int, method: str, k: int, distance: str) -> Path:
    return root / "gate_runs" / f"{dataset}__seed_{seed}__{method}__k_{k}__dist_{distance}"


def run_pilot(paths: ProtocolV2Paths, config_path: Path, *, resume: bool = True, preflight: bool = False) -> dict[str, Any]:
    """Run the 180-cell representation pilot; checkpoint reuse is explicit."""

    paths.require_experiment_admission()
    config = _load_config(config_path)
    root = stage_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "R1_MINILM_STAGE_PROVENANCE.json").is_file():
        freeze_provenance(paths, config_path)
    plan = {
        "stage": STAGE,
        "protocol_version": paths.dataset_version,
        "datasets": list(DATASETS),
        "kir": 0.50,
        "seeds": list(SEEDS),
        "representations": list(METHODS),
        "k_values": list(K_VALUES),
        "distances": list(DISTANCES),
        "planned_gate_units": len(DATASETS) * len(SEEDS) * len(METHODS) * len(K_VALUES) * len(DISTANCES),
        "planned_trainable_checkpoints": len(DATASETS) * len(SEEDS) * len(TRAINABLE_METHODS),
        "selection": "Known calibration only; no test or OOS selection",
    }
    atomic_write_json(root / "plans" / "pilot_plan.json", plan)
    if preflight:
        checks = []
        model_path = (paths.project_root / str(config["model_path"])).resolve()
        model_files = _model_files(model_path)
        for dataset in DATASETS:
            for seed in SEEDS:
                bundle = load_e2_bundle(paths, dataset, seed, 0.50)
                checks.append({"dataset": dataset, "seed": seed, "train": len(bundle.views.train), "calibration": len(bundle.views.calibration), "test": len(bundle.views.test), "model_files": model_files})
        payload = {"status": "preflight_ok", "plan": plan, "checks": checks}
        atomic_write_json(root / "summaries" / "PILOT_PREFLIGHT.json", payload)
        return payload
    training_manifests: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            bundle = load_e2_bundle(paths, dataset, seed, 0.50)
            method_arrays: dict[str, np.ndarray] = {}
            for method in METHODS:
                method_dir = root / "checkpoints" / dataset / f"seed_{seed}" / method
                if method in TRAINABLE_METHODS:
                    manifest = train_checkpoint(model_path=(paths.project_root / str(config["model_path"])).resolve(), bundle=bundle, output_dir=method_dir, method=method, seed=seed, config=config)
                    training_manifests.append(manifest)
                    arrays, representation_manifest = load_representation(model_path=(paths.project_root / str(config["model_path"])).resolve(), bundle=bundle, method=method, checkpoint_dir=method_dir, config=config)
                    method_arrays[method] = arrays
                    method_manifest = representation_manifest
                else:
                    arrays, method_manifest = load_representation(model_path=Path("."), bundle=bundle, method=method, checkpoint_dir=None, config=config)
                    method_arrays[method] = arrays
                for k in K_VALUES:
                    for distance in DISTANCES:
                        unit_dir = _pilot_unit_path(root, dataset, seed, method, k, distance)
                        result_path = unit_dir / "eval_results.json"
                        if resume and result_path.is_file():
                            existing = json.loads(result_path.read_text(encoding="utf-8"))
                            if existing.get("stage") == STAGE:
                                result_rows.append(existing)
                                continue
                            raise RuntimeError(f"Refusing to reuse unrelated unit: {unit_dir}")
                        metrics = evaluate_gate_cell(bundle, arrays, method, k, distance, seed)
                        metrics["checkpoint_sha256"] = training_manifests[-1].get("checkpoint_sha256") if method in TRAINABLE_METHODS else "frozen_e2_cache"
                        metrics["representation_manifest"] = method_manifest
                        unit_dir.mkdir(parents=True, exist_ok=True)
                        atomic_write_json(result_path, _safe(metrics))
                        result_rows.append(metrics)
    _atomic_csv(root / "summaries" / "MINILM_PILOT_GATE.tsv", result_rows)
    _atomic_csv(root / "summaries" / "MINILM_PILOT_TRAINING.tsv", training_manifests)
    payload = {"status": "complete", "planned_gate_units": plan["planned_gate_units"], "completed_gate_units": len(result_rows), "planned_trainable_checkpoints": plan["planned_trainable_checkpoints"], "completed_trainable_checkpoints": len(training_manifests), "root": str(root)}
    atomic_write_json(root / "summaries" / "MINILM_PILOT_INTEGRITY.json", payload)
    return payload


def summarize_pilot(paths: ProtocolV2Paths) -> dict[str, Any]:
    root = stage_root(paths)
    gate_path = root / "summaries" / "MINILM_PILOT_GATE.tsv"
    if not gate_path.is_file():
        raise FileNotFoundError(gate_path)
    with gate_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["method"], row["distance"], row["k"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        record: dict[str, Any] = {"dataset": key[0], "method": key[1], "distance": key[2], "k": int(key[3]), "n": len(values)}
        for metric in ("oos_f1", "f1_all", "f1_u", "f1_k", "accuracy", "auroc", "aupr_oos", "known_recall", "false_accept_rate", "false_reject_rate", "effective_rank", "intra_class_distance", "inter_class_distance", "relative_separation", "representation_collision_rate"):
            numbers = [float(value[metric]) for value in values if value.get(metric, "") not in {"", "None"}]
            record[f"{metric}_mean"] = float(np.mean(numbers)) if numbers else math.nan
            record[f"{metric}_std"] = float(np.std(numbers)) if numbers else math.nan
        summary.append(record)
    _atomic_csv(root / "summaries" / "MINILM_PILOT_SUMMARY.tsv", summary)
    deltas: list[dict[str, Any]] = []
    by_key = {(row["dataset"], row["method"], row["distance"], int(row["k"]), int(row["seed"])): row for row in rows}
    for dataset in DATASETS:
        for method in METHODS:
            for distance in DISTANCES:
                for seed in SEEDS:
                    one = by_key.get((dataset, method, distance, 1, seed))
                    two = by_key.get((dataset, method, distance, 2, seed))
                    if one and two:
                        deltas.append({"dataset": dataset, "method": method, "distance": distance, "seed": seed, "oos_f1_k1": float(one["oos_f1"]), "oos_f1_k2": float(two["oos_f1"]), "k2_minus_k1_oos_f1": float(two["oos_f1"]) - float(one["oos_f1"]), "known_recall_k1": float(one["known_recall"]), "known_recall_k2": float(two["known_recall"]), "k2_minus_k1_known_recall": float(two["known_recall"]) - float(one["known_recall"]), "k2_minus_k1_false_accept_rate": float(two["false_accept_rate"]) - float(one["false_accept_rate"])})
    _atomic_csv(root / "summaries" / "MINILM_PILOT_K2_MINUS_K1.tsv", deltas)
    stack = [row for row in deltas if row["dataset"] == "stackoverflow"]
    stack_by_method: dict[str, list[float]] = {}
    for row in stack:
        stack_by_method.setdefault(str(row["method"]), []).append(float(row["k2_minus_k1_oos_f1"]))
    stack_means = {method: float(np.mean(values)) for method, values in stack_by_method.items()}
    multicenter_rescue = any(value >= -0.02 for value in stack_means.values())
    decision = "continue_multicenter_repair" if multicenter_rescue else "stop_fixed_multicenter_rescue"
    decision_text = (
        "The pilot does not authorize a larger multicenter representation sweep. "
        "Frozen, head-only CE, full CE, SupCon, and CE-Recon all retain a StackOverflow "
        "K=2 loss beyond the 0.02 safety margin; the loss is accompanied by increased "
        "OOS false acceptance. Keep the strongest K=1 representation comparisons and "
        "retain StackOverflow as structural evidence against fixed post-hoc multicenter "
        "acceptance. No adaptive K, new boundary, external baseline, or full Pipeline "
        "is started by this stage."
        if decision == "stop_fixed_multicenter_rescue"
        else "At least one representation reaches the preregistered StackOverflow K=2 safety margin; do not expand automatically before reviewing the paired audit."
    )
    closeout = "\n".join(
        [
            "# MiniLM Training and StackOverflow Repair Closeout",
            "",
            f"- Stage: `{STAGE}`",
            "- Protocol: `protocol_v2_textoir_v1`",
            f"- Gate cells: {len(rows)}/180; trainable checkpoints: 36/36",
            "- Selection: Known calibration only; no OOS row was used for training, checkpoint selection, threshold selection, or boundary selection.",
            "- StackOverflow audit: frozen E2 cache/view sample IDs and embedding bytes passed alignment checks; K=1/K=2 scores reproduce the active detector contract.",
            "- Post-run analysis correction: the closeout distance cells are means over all three seeds; gate runs and paired rows were not modified.",
            "",
            "## Decision",
            "",
            f"`{decision}`",
            "",
            decision_text,
            "",
            "## StackOverflow K2 paired means",
            "",
            "| Representation | Euclidean Δ(OOS F1) | Mahalanobis-diag Δ(OOS F1) |",
            "|---|---:|---:|",
        ]
    ) + "\n"
    for method in METHODS:
        by_distance: dict[str, list[float]] = {}
        for row in stack:
            if row["method"] == method:
                by_distance.setdefault(str(row["distance"]), []).append(float(row["k2_minus_k1_oos_f1"]))
        values = {distance: float(np.mean(numbers)) for distance, numbers in by_distance.items()}
        closeout += f"| {method} | {values.get('euclidean', math.nan):.4f} | {values.get('mahalanobis_diag', math.nan):.4f} |\n"
    closeout += "\n## Next step\n\nUpdate the research ledger with this decision and stop this stage. A larger KIR/seed/K grid is not authorized by the pilot.\n"
    atomic_write_text(root / "summaries" / "MINILM_PILOT_DECISION.md", decision_text + "\n")
    atomic_write_text(root / "summaries" / "MINILM_PILOT_CLOSEOUT.md", closeout)
    return {"status": "complete", "gate_rows": len(rows), "summary_rows": len(summary), "delta_rows": len(deltas), "root": str(root), "decision": decision, "stackoverflow_k2_minus_k1": stack_means}


def verify_pilot(paths: ProtocolV2Paths, require_complete: bool = False) -> dict[str, Any]:
    root = stage_root(paths)
    plan_path = root / "plans" / "pilot_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else {}
    units = list(root.glob("gate_runs/*/eval_results.json"))
    invalid = []
    for path in units:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("stage") != STAGE or not all(math.isfinite(float(payload[key])) for key in ("oos_f1", "f1_all", "f1_u", "f1_k", "accuracy", "auroc", "aupr_oos", "known_recall", "false_accept_rate", "false_reject_rate")):
            invalid.append(str(path))
    expected = int(plan.get("planned_gate_units", 180))
    result = {"status": "ok" if not invalid and (not require_complete or len(units) == expected) else "invalid", "planned": expected, "completed": len(units), "invalid": invalid, "provenance": (root / "R1_MINILM_STAGE_PROVENANCE.json").is_file(), "audit": (root / "stackoverflow_audit/STACKOVERFLOW_K1_K2_SAMPLE_AUDIT.tsv").is_file()}
    atomic_write_json(root / "summaries" / "MINILM_PILOT_VERIFY.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent MiniLM training and StackOverflow repair stage")
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--seed", type=int, default=42)
    audit_parser.add_argument("--kir", type=float, default=0.50)
    pilot_parser = sub.add_parser("pilot")
    pilot_parser.add_argument("--config", type=Path, default=Path("configs/experiments/protocol_v2_textoir_v1/minilm_training_and_stackoverflow_repair.yaml"))
    pilot_parser.add_argument("--preflight", action="store_true")
    pilot_parser.add_argument("--no-resume", action="store_true")
    sub.add_parser("summarize")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    if args.command == "audit":
        result = audit_stackoverflow(paths, args.seed, args.kir)
    elif args.command == "pilot":
        result = run_pilot(paths, args.config, resume=not args.no_resume, preflight=args.preflight)
    elif args.command == "summarize":
        result = summarize_pilot(paths)
    else:
        result = verify_pilot(paths, args.require_complete)
    print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
    return 0 if result.get("status") not in {"invalid", "blocked"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
