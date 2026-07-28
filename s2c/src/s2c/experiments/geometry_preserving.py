"""Geometry-preserving MiniLM adaptation for the R1 pilot.

This module deliberately stays below the Gate API.  It trains a Known-only
MiniLM representation, then sends the resulting embeddings through the same
``MultiSphereOOSDetector`` used by the active protocol.  The frozen teacher is
never updated and no OOS row is read by the training or beta-selection paths.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer

from s2c.data.hashing import atomic_write_json, atomic_write_text, sha256_file
from s2c.evaluation.metrics import compute_binary_oos_metrics
from s2c.experiments.mechanism_runner import E3Bundle, load_e2_bundle
from gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from s2c.runtime.paths import ProtocolV2Paths


DEFAULT_MODEL = "all-MiniLM-L6-v2"
REPRESENTATIONS = ("frozen_minilm", "ce_recon", "ce_recon_geometry")
DISTANCES = ("euclidean", "mahalanobis_diag")
K_VALUES = (1, 2)
BETAS = (0.1, 0.5, 1.0)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_safe(row) for row in rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_module(module: torch.nn.Module) -> None:
    """Freeze a teacher explicitly; this is also covered by a unit test."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)


def pairwise_cosine_relation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Match within-batch cosine relations; diagonal entries are harmless zeros."""

    student_norm = torch.nn.functional.normalize(student, dim=-1)
    teacher_norm = torch.nn.functional.normalize(teacher.detach(), dim=-1)
    if student_norm.shape[0] < 2:
        return student_norm.sum() * 0.0
    student_rel = student_norm @ student_norm.T
    teacher_rel = teacher_norm @ teacher_norm.T
    mask = ~torch.eye(student_rel.shape[0], dtype=torch.bool, device=student_rel.device)
    return torch.nn.functional.mse_loss(student_rel[mask], teacher_rel[mask])


def effective_rank(embeddings: np.ndarray) -> float:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.shape[0] < 2:
        return 1.0
    singular = np.linalg.svd(values - values.mean(axis=0, keepdims=True), compute_uv=False)
    singular = singular[singular > 1e-12]
    if not singular.size:
        return 0.0
    probabilities = singular / singular.sum()
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def pairwise_relation_metrics(
    teacher: np.ndarray,
    student: np.ndarray,
    labels: Sequence[str],
    seed: int,
    max_points: int = 1500,
    neighbors: int = 10,
) -> dict[str, float]:
    """Measure geometry without constructing an unbounded all-pairs table."""

    teacher = np.asarray(teacher, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    if teacher.shape != student.shape:
        raise ValueError("Teacher and student geometry arrays must have identical shapes")
    if teacher.shape[0] > max_points:
        indices = np.sort(np.random.default_rng(seed).choice(teacher.shape[0], max_points, replace=False))
        teacher = teacher[indices]
        student = student[indices]
        labels = [str(labels[int(index)]) for index in indices]
    teacher = teacher / np.clip(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-12, None)
    student = student / np.clip(np.linalg.norm(student, axis=1, keepdims=True), 1e-12, None)
    teacher_rel = teacher @ teacher.T
    student_rel = student @ student.T
    mask = ~np.eye(len(teacher), dtype=bool)
    t_values = teacher_rel[mask]
    s_values = student_rel[mask]
    correlation = float(np.corrcoef(t_values, s_values)[0, 1]) if np.std(t_values) and np.std(s_values) else 1.0
    k = min(int(neighbors), max(1, len(teacher) - 1))
    teacher_near = np.argpartition(-teacher_rel, kth=k, axis=1)[:, 1 : k + 1]
    student_near = np.argpartition(-student_rel, kth=k, axis=1)[:, 1 : k + 1]
    overlap = np.mean([
        len(set(teacher_near[i].tolist()).intersection(student_near[i].tolist())) / k
        for i in range(len(teacher))
    ]) if len(teacher) else math.nan
    label_array = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(seed)
    pair_count = min(20000, len(teacher) * max(1, len(teacher) - 1) // 2)
    if pair_count:
        first = rng.integers(0, len(teacher), pair_count)
        second = rng.integers(0, len(teacher), pair_count)
        valid = first != second
        first, second = first[valid], second[valid]
        # Class-separation metrics must be measured in the representation they
        # claim to describe.  The old implementation accidentally reused the
        # teacher distances for both teacher and student fields, which made a
        # collapsed student appear to preserve teacher class geometry.
        teacher_distances = 1.0 - teacher_rel[first, second]
        student_distances = 1.0 - student_rel[first, second]
        same = label_array[first] == label_array[second]
        teacher_intra = float(np.mean(teacher_distances[same])) if same.any() else math.nan
        teacher_inter = float(np.mean(teacher_distances[~same])) if (~same).any() else math.nan
        intra = float(np.mean(student_distances[same])) if same.any() else math.nan
        inter = float(np.mean(student_distances[~same])) if (~same).any() else math.nan
    else:
        intra = inter = teacher_intra = teacher_inter = math.nan
    return {
        "effective_rank": effective_rank(student),
        "teacher_effective_rank": effective_rank(teacher),
        "pairwise_distance_correlation": correlation,
        "knn_neighborhood_preservation": float(overlap),
        "intra_class_distance": intra,
        "inter_class_distance": inter,
        "relative_separation": (inter - intra) / inter if math.isfinite(intra) and math.isfinite(inter) and inter else math.nan,
        "teacher_intra_class_distance": teacher_intra,
        "teacher_inter_class_distance": teacher_inter,
        "teacher_relative_separation": (
            (teacher_inter - teacher_intra) / teacher_inter
            if math.isfinite(teacher_intra) and math.isfinite(teacher_inter) and teacher_inter
            else math.nan
        ),
        "embedding_norm_mean": float(np.linalg.norm(student, axis=1).mean()),
        "embedding_norm_std": float(np.linalg.norm(student, axis=1).std()),
    }


def _encode(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
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
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, int(model.config.hidden_size)), dtype=np.float32)


def _classification_metrics(
    model: torch.nn.Module,
    head: torch.nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    label_map: Mapping[str, int],
    device: torch.device,
    batch_size: int,
    max_length: int,
    classifier_input: str = "pooled",
) -> dict[str, float]:
    model.eval()
    head.eval()
    predictions: list[int] = []
    labels: list[int] = []
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
            pooled_norm = torch.nn.functional.normalize(pooled, dim=-1)
            classifier_features = pooled if classifier_input == "pooled" else pooled_norm
            predictions.extend(head(classifier_features).argmax(dim=-1).cpu().tolist())
            labels.extend(label_map[str(row["intent"])] for row in batch_rows)
    return {
        "known_validation_macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)) if labels else math.nan,
        "known_validation_accuracy": float(np.mean(np.asarray(labels) == np.asarray(predictions))) if labels else math.nan,
    }


def train_representation(
    *,
    model_path: Path,
    train_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    teacher_train: np.ndarray,
    output_dir: Path,
    method: str,
    seed: int,
    beta: float,
    alpha: float = 1.0,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    classifier_input: str = "pooled",
    geometry_input: str = "normalized_pooled",
    gate_embedding: str = "normalized_pooled",
) -> dict[str, Any]:
    """Train CE-Recon or CE-Recon-Geometry using Known train only."""

    if method not in {"ce_recon", "ce_recon_geometry"}:
        raise ValueError(f"Unsupported trainable representation: {method}")
    valid_inputs = {"pooled", "normalized_pooled"}
    if classifier_input not in valid_inputs or geometry_input not in valid_inputs or gate_embedding not in valid_inputs:
        raise ValueError("Representation inputs must be pooled or normalized_pooled")
    if len(train_rows) != len(teacher_train):
        raise ValueError("Teacher cache and train rows are not aligned")
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "encoder.pt"
    manifest_path = output_dir / "training_manifest.json"
    if checkpoint.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("method") == method
            and float(manifest.get("beta", -1.0)) == float(beta)
            and manifest.get("classifier_input") == classifier_input
            and manifest.get("geometry_input") == geometry_input
            and manifest.get("gate_embedding") == gate_embedding
            and manifest.get("status") == "complete"
        ):
            return manifest
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    student = AutoModel.from_pretrained(model_path, local_files_only=True)
    teacher = AutoModel.from_pretrained(model_path, local_files_only=True)
    freeze_module(teacher)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    teacher.to(device)
    labels = sorted({str(row["intent"]) for row in train_rows})
    label_map = {name: index for index, name in enumerate(labels)}
    head = torch.nn.Linear(int(student.config.hidden_size), len(labels)).to(device)
    optimizer = torch.optim.AdamW([*student.parameters(), *head.parameters()], lr=learning_rate)
    teacher_train_tensor = torch.as_tensor(teacher_train, dtype=torch.float32)
    history: list[dict[str, float]] = []
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    train_indices = np.arange(len(train_rows))
    for epoch in range(1, epochs + 1):
        student.train()
        head.train()
        order = np.random.default_rng(seed + epoch).permutation(train_indices)
        losses: list[float] = []
        ce_losses: list[float] = []
        recon_losses: list[float] = []
        geometry_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_rows = [train_rows[int(index)] for index in batch_indices]
            tokens = tokenizer(
                [str(row["text"]) for row in batch_rows],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            pooled = mean_pool(student(**tokens).last_hidden_state, tokens["attention_mask"])
            pooled_norm = torch.nn.functional.normalize(pooled, dim=-1)
            teacher_pooled = teacher_train_tensor[torch.as_tensor(batch_indices, dtype=torch.long)].to(device)
            teacher_norm = torch.nn.functional.normalize(teacher_pooled, dim=-1)
            classifier_features = pooled if classifier_input == "pooled" else pooled_norm
            geometry_student = pooled if geometry_input == "pooled" else pooled_norm
            geometry_teacher = teacher_pooled if geometry_input == "pooled" else teacher_norm
            target = torch.as_tensor([label_map[str(row["intent"])] for row in batch_rows], dtype=torch.long, device=device)
            ce_loss = torch.nn.functional.cross_entropy(head(classifier_features), target)
            recon_loss = torch.nn.functional.mse_loss(geometry_student, geometry_teacher.detach())
            geometry_loss = pairwise_cosine_relation_loss(geometry_student, geometry_teacher) if method == "ce_recon_geometry" else geometry_student.sum() * 0.0
            total = ce_loss + alpha * recon_loss + beta * geometry_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            losses.append(float(total.detach().cpu()))
            ce_losses.append(float(ce_loss.detach().cpu()))
            recon_losses.append(float(recon_loss.detach().cpu()))
            geometry_losses.append(float(geometry_loss.detach().cpu()))
        validation = _classification_metrics(
            student, head, tokenizer, calibration_rows, label_map, device, batch_size, max_length, classifier_input
        )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "ce_loss": float(np.mean(ce_losses)),
                "reconstruction_loss": float(np.mean(recon_losses)),
                "geometry_loss": float(np.mean(geometry_losses)),
                **validation,
            }
        )
        score = validation["known_validation_macro_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                "encoder": {key: value.detach().cpu() for key, value in student.state_dict().items()},
                "label_map": label_map,
            }
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    torch.save(best_state, checkpoint)
    history_path = output_dir / "training_history.csv"
    write_csv(history_path, history)
    manifest = {
        "schema_version": "s2c.r1_training.v1",
        "status": "complete",
        "method": method,
        "seed": seed,
        "beta": beta,
        "alpha": alpha,
        "train_sample_count": len(train_rows),
        "calibration_sample_count": len(calibration_rows),
        "label_count": len(labels),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "checkpoint_epoch": best_epoch,
        "checkpoint_selection_metric": "known_validation_macro_f1",
        "known_validation_macro_f1": best_score,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "history": str(history_path),
        "used_oos_for_training": False,
        "used_test_for_selection": False,
        "teacher_frozen": True,
        "teacher_source": "frozen_minilm_teacher_cache",
        "classifier_input": classifier_input,
        "geometry_input": geometry_input,
        "gate_embedding": gate_embedding,
        "pooled_semantics": "mean_pool(last_hidden_state, attention_mask)",
        "teacher_pooled_source": "frozen_minilm_train_embedding_cache",
    }
    atomic_write_json(manifest_path, manifest)
    del student, teacher, head, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def load_checkpoint(model_path: Path, checkpoint: Path, device: torch.device) -> tuple[Any, torch.nn.Module, dict[str, int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["encoder"])
    model.to(device)
    model.eval()
    return tokenizer, model, {str(key): int(value) for key, value in state["label_map"].items()}


def class_centers(embeddings: np.ndarray, intents: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray([str(value) for value in intents], dtype=object)
    names = sorted(np.unique(labels).tolist())
    centers = np.asarray([values[labels == name].mean(axis=0) for name in names], dtype=np.float64)
    centers /= np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    return centers, names


def fixed_oos_buckets(
    frozen_train: np.ndarray,
    frozen_test: np.ndarray,
    test_rows: Sequence[Mapping[str, Any]],
    intents: Sequence[str],
    *,
    frozen_validation: np.ndarray | None = None,
    validation_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Frozen-reference OOS cut points learned from validation only.

    The active protocol deliberately uses Known-only calibration.  In that
    case there is no legal validation OOS population from which to estimate
    q20/q80.  We therefore return an explicit exploratory/unavailable status
    instead of silently leaking test OOS distances into the bucket definition.
    """

    if frozen_validation is None or validation_rows is None:
        return np.full(len(test_rows), "all", dtype=object), {
            "q20": math.nan,
            "q80": math.nan,
            "source": "validation_oos_unavailable_known_only_calibration",
            "bucket_status": "exploratory_unavailable_validation_oos",
            "used_test_oos_for_cutpoints": False,
        }
    centers, _ = class_centers(frozen_train, intents)
    validation_norm = frozen_validation / np.clip(np.linalg.norm(frozen_validation, axis=1, keepdims=True), 1e-12, None)
    test_norm = frozen_test / np.clip(np.linalg.norm(frozen_test, axis=1, keepdims=True), 1e-12, None)
    validation_distances = 1.0 - np.max(validation_norm @ centers.T, axis=1)
    test_distances = 1.0 - np.max(test_norm @ centers.T, axis=1)
    validation_labels = np.asarray([int(row["label"]) for row in validation_rows], dtype=np.int64)
    oos_distances = validation_distances[validation_labels == 1]
    if not len(oos_distances):
        return np.full(len(test_rows), "all", dtype=object), {
            "q20": math.nan,
            "q80": math.nan,
            "source": "validation_oos_unavailable_known_only_calibration",
            "bucket_status": "exploratory_unavailable_validation_oos",
            "used_test_oos_for_cutpoints": False,
        }
    q20, q80 = np.quantile(oos_distances, [0.20, 0.80])
    buckets = np.where(test_distances <= q20, "near", np.where(test_distances <= q80, "medium", "far"))
    return buckets.astype(object), {
        "q20": float(q20),
        "q80": float(q80),
        "source": "frozen_k1_validation_oos_quantiles",
        "bucket_status": "formal_validation_oos_cutpoints",
        "used_test_oos_for_cutpoints": False,
    }


def representation_collision(train: np.ndarray, calibration: np.ndarray, test: np.ndarray, train_rows: Sequence[Mapping[str, Any]], calibration_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]]) -> float:
    train_norm = train / np.clip(np.linalg.norm(train, axis=1, keepdims=True), 1e-12, None)
    calibration_norm = calibration / np.clip(np.linalg.norm(calibration, axis=1, keepdims=True), 1e-12, None)
    test_norm = test / np.clip(np.linalg.norm(test, axis=1, keepdims=True), 1e-12, None)
    train_intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    calibration_intents = np.asarray([str(row["intent"]) for row in calibration_rows], dtype=object)
    names = sorted(np.unique(train_intents).tolist())
    centers = np.asarray([train_norm[train_intents == name].mean(axis=0) for name in names])
    centers /= np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    thresholds = []
    for name, center in zip(names, centers, strict=True):
        values = calibration_norm[calibration_intents == name] @ center
        thresholds.append(float(np.quantile(values, 0.05)) if len(values) else math.nan)
    similarities = test_norm @ centers.T
    nearest = np.argmax(similarities, axis=1)
    labels = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    oos = labels == 1
    if not oos.any():
        return math.nan
    return float(np.mean(similarities[np.arange(len(test_norm))[oos], nearest[oos]] >= np.asarray(thresholds)[nearest[oos]]))


def fit_gate(train: np.ndarray, train_rows: Sequence[Mapping[str, Any]], k: int, distance: str) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=k,
        radius_method="mean_std",
        radius_lambda=1.0,
        distance_metric=distance,
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
    )
    detector.fit(train, np.asarray([str(row["intent"]) for row in train_rows], dtype=object))
    return detector


def evaluate_gate(
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    k: int,
    distance: str,
    buckets: np.ndarray,
) -> dict[str, Any]:
    detector = fit_gate(train, train_rows, k, distance)
    test_scores = np.asarray(detector.predict_with_scores(test)["score"], dtype=float)
    labels = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)
    result: dict[str, Any] = compute_binary_oos_metrics(labels, test_scores, 1.0)
    for bucket in ("near", "medium", "far"):
        oos = (labels == 1) & (buckets == bucket)
        known = labels == 0
        selected = known | oos
        bucket_metrics = compute_binary_oos_metrics(labels[selected], test_scores[selected], 1.0) if oos.any() else {}
        result[f"{bucket}_oos_f1"] = bucket_metrics.get("oos_f1", math.nan)
        result[f"{bucket}_oos_recall"] = bucket_metrics.get("oos_recall", math.nan)
        result[f"{bucket}_false_accept_rate"] = bucket_metrics.get("false_accept_rate", math.nan)
    result.update(
        {
            "effective_cluster_count": len(detector.spheres),
            "minimum_cluster_size": int(min(np.bincount(detector._train_cluster_labels))),
            "scoring_seconds": math.nan,
            "samples_per_second": math.nan,
            "score_mean": float(test_scores.mean()),
            "representation_collision_rate": math.nan,
        }
    )
    return result


def geometry_metrics(
    frozen_train: np.ndarray,
    frozen_calibration: np.ndarray,
    frozen_test: np.ndarray,
    student_train: np.ndarray,
    student_calibration: np.ndarray,
    student_test: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    labels = [str(row["intent"]) for row in train_rows]
    teacher = np.asarray(frozen_train, dtype=np.float64)
    student = np.asarray(student_train, dtype=np.float64)
    metrics = pairwise_relation_metrics(teacher, student, labels, seed)
    metrics["representation_collision_rate"] = representation_collision(
        student_train, student_calibration, student_test, train_rows, calibration_rows, test_rows
    )
    metrics["teacher_representation_collision_rate"] = representation_collision(
        frozen_train, frozen_calibration, frozen_test, train_rows, calibration_rows, test_rows
    )
    return metrics


def load_bundle(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float) -> E3Bundle:
    return load_e2_bundle(paths, dataset, seed, kir)


def file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths if path.is_file()}


def git_patch_hash(repo_root: Path) -> tuple[str, Path]:
    import subprocess

    patch_path = repo_root.parent / "artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation/R1_CODE_SNAPSHOT.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=repo_root.parent, capture_output=True, check=True).stdout
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo_root.parent, capture_output=True, text=True, check=True).stdout.splitlines()
    chunks = [tracked]
    for relative in untracked:
        if relative.startswith("artifacts/") or relative.startswith("assets/"):
            continue
        result = subprocess.run(["git", "diff", "--no-index", "--binary", "/dev/null", relative], cwd=repo_root.parent, capture_output=True, check=False)
        if result.stdout:
            chunks.append(result.stdout)
    atomic_write_text(patch_path, b"\n".join(chunks).decode("utf-8", errors="replace"))
    return sha256_file(patch_path), patch_path
