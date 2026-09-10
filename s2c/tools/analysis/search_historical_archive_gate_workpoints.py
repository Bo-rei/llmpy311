#!/usr/bin/env python3
"""Search archive Gate work-points without changing existing checkpoints.

Validation rows, including their OOS labels, are used only to rank candidate
work-points in this explicitly development-tuned diagnostic.  Test rows are
scored for confirmation and are never used for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from protocol_v2.experiments.racal_v1.representation import (  # noqa: E402
    build_racal_model,
    choose_device,
    encode_rows,
)
from tools.legacy.analysis_v19.run_trainable_minilm_historical_v1 import (  # noqa: E402
    load_archive_views,
)
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector  # noqa: E402


DATA_ROOT = ROOT.parent / "archives" / "submissions" / "s2c-submission" / "data"
MODEL_ROOT = ROOT.parent / "assets" / "models"
DEFAULT_OUTPUT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_archive_gate_search_seed42"
DATASETS = ("clinc150", "stackoverflow", "banking77")
METHODS = ("frozen_k1", "partial_k1", "lora_k1")
K_VALUES = (1, 2, 3, 4, 5)
LAMBDA_VALUES = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
THRESHOLD_VALUES = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
ACCEPTANCE_MODES = ("nearest_sphere", "normalized_union")


def _checkpoint_path(dataset: str, method: str) -> Path:
    if method == "frozen_k1":
        return (
            ROOT.parent
            / "artifacts"
            / "s2c"
            / "runs"
            / "historical_protocol_v1"
            / "minilm_k1"
            / dataset
            / "kir50_seed42"
            / "frozen_k1"
        )
    if method == "partial_k1" and dataset != "banking77":
        return (
            ROOT.parent
            / "artifacts"
            / "s2c"
            / "runs"
            / "historical_protocol_v1"
            / "minilm_k1"
            / dataset
            / "kir50_seed42"
            / "trainable_k1"
        )
    return (
        ROOT.parent
        / "artifacts"
        / "s2c"
        / "runs"
        / "historical_archive_minilm_seed42"
        / dataset
        / ("trainable_k1" if method == "partial_k1" else "lora_k1")
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_detector(
    train_values: np.ndarray,
    train_rows: list[dict[str, Any]],
    k: int,
    radius_lambda: float,
    acceptance_mode: str,
) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=int(k),
        radius_method="mean_std",
        radius_lambda=float(radius_lambda),
        distance_metric="mahalanobis_diag",
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
        acceptance_mode=acceptance_mode,
    )
    detector.fit(train_values, np.asarray([str(row["intent"]) for row in train_rows], dtype=object))
    return detector


def _vectorized_output(
    detector: MultiSphereOOSDetector,
    embeddings: np.ndarray,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Match detector score semantics while avoiding Python sample/sphere loops."""
    values = np.asarray(embeddings, dtype=np.float64)
    if detector.l2_normalize:
        values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    centers = np.asarray([sphere.center for sphere in detector.spheres], dtype=np.float64)
    radii = np.asarray([sphere.radius for sphere in detector.spheres], dtype=np.float64)
    inv_cov = None
    if detector.distance_metric == "mahalanobis_diag":
        inv_cov = np.asarray([sphere.inv_diag_cov for sphere in detector.spheres], dtype=np.float64)
    score_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    distance_chunks: list[np.ndarray] = []
    prediction_chunks: list[np.ndarray] = []
    for start in range(0, len(values), chunk_size):
        batch = values[start : start + chunk_size]
        diff = batch[:, None, :] - centers[None, :, :]
        if inv_cov is None:
            distances = np.linalg.norm(diff, axis=2)
        else:
            distances = np.sqrt(np.sum(np.square(diff) * inv_cov[None, :, :], axis=2))
        raw_indices = np.argmin(distances, axis=1)
        if detector.acceptance_mode == "normalized_union":
            ratios = distances / np.clip(radii[None, :], 1e-12, None)
            indices = np.argmin(ratios, axis=1)
            scores = ratios[np.arange(len(batch)), indices]
            selected_distances = distances[np.arange(len(batch)), indices]
            predictions = (~np.any(ratios <= 1.0, axis=1)).astype(np.int64)
        else:
            indices = raw_indices
            selected_distances = distances[np.arange(len(batch)), indices]
            scores = selected_distances / np.clip(radii[indices], 1e-12, None)
            predictions = (selected_distances > radii[indices]).astype(np.int64)
        score_chunks.append(scores)
        index_chunks.append(indices)
        distance_chunks.append(selected_distances)
        prediction_chunks.append(predictions)
    return {
        "score": np.concatenate(score_chunks),
        "nearest_cluster": np.concatenate(index_chunks),
        "distance": np.concatenate(distance_chunks),
        "pred": np.concatenate(prediction_chunks),
    }


def _evaluate_fast(
    detector: MultiSphereOOSDetector,
    embeddings: np.ndarray,
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    output = _vectorized_output(detector, embeddings)
    return _metrics_from_output(detector, output, rows, threshold)


def _metrics_from_output(
    detector: MultiSphereOOSDetector,
    output: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    predicted_oos = (output["score"] > float(threshold)).astype(np.int64)
    known_mask = labels == 0
    oos_mask = labels == 1
    known_intents = sorted({str(row["intent"]) for row in rows if int(row["label"]) == 0})
    truth = [str(row["intent"]) if not is_oos else "__oos__" for row, is_oos in zip(rows, labels, strict=True)]
    predicted = []
    for index, is_oos in enumerate(predicted_oos):
        if is_oos:
            predicted.append("__oos__")
        else:
            predicted.append(str(detector.cluster_to_intent.get(int(output["nearest_cluster"][index]), "__unknown__")))
    return {
        "f1_u": float(f1_score(labels, predicted_oos, pos_label=1, zero_division=0)),
        "f1_k": float(f1_score(truth, predicted, labels=known_intents, average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(truth, dtype=object) == np.asarray(predicted, dtype=object))),
        "known_recall": float(np.mean(predicted_oos[known_mask] == 0)),
        "false_accept_rate": float(np.mean(predicted_oos[oos_mask] == 0)),
        "false_reject_rate": float(np.mean(predicted_oos[known_mask] == 1)),
        "auroc": float(roc_auc_score(labels, output["score"])),
        "aupr_oos": float(average_precision_score(labels, output["score"])),
    }


def _load_embeddings(
    dataset: str,
    method: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    data_root = DATA_ROOT / dataset / "kir50_seed42"
    snapshot, views = load_archive_views(data_root)
    model_root = MODEL_ROOT / "all-MiniLM-L6-v2"
    if method == "frozen_k1":
        encoder = SentenceTransformer(str(model_root), device=str(device))
        values = {
            split: np.asarray(
                encoder.encode(
                    [str(row["text"]) for row in rows],
                    batch_size=128,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                ),
                dtype=np.float32,
            )
            for split, rows in views.items()
        }
        del encoder
    else:
        checkpoint_dir = _checkpoint_path(dataset, method)
        checkpoint = torch.load(checkpoint_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        hidden_dim = int(checkpoint["model"]["projection.fc1.weight"].shape[0])
        model = build_racal_model(model_root, str(checkpoint["mode"]), hidden_dim).to(device)
        model.load_state_dict(checkpoint["model"])
        tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
        values = {
            split: encode_rows(model, tokenizer, rows, device, 128, 256)
            for split, rows in views.items()
        }
        del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return snapshot, views, values


def _metric_row(
    dataset: str,
    method: str,
    split: str,
    k: int,
    radius_lambda: float,
    threshold: float,
    acceptance_mode: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "split": split,
        "k": int(k),
        "radius_lambda": float(radius_lambda),
        "threshold": float(threshold),
        "acceptance_mode": acceptance_mode,
        "oos_f1": float(metrics["f1_u"]),
        "known_f1": float(metrics["f1_k"]),
        "accuracy": float(metrics["accuracy"]),
        "known_recall": float(metrics["known_recall"]),
        "false_accept_rate": float(metrics["false_accept_rate"]),
        "false_reject_rate": float(metrics["false_reject_rate"]),
        "auroc": float(metrics["auroc"]),
        "aupr_oos": float(metrics["aupr_oos"]),
    }


def _select_validation(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if float(row["known_f1"]) >= float(baseline["known_f1"]) - 0.01
        and float(row["accuracy"]) >= float(baseline["accuracy"]) - 0.01
    ]
    if not eligible:
        raise ValueError("no candidate satisfies the one-point Known F1/accuracy guard")
    return max(
        eligible,
        key=lambda row: (
            float(row["oos_f1"]),
            float(row["known_f1"]),
            float(row["accuracy"]),
            -int(row["k"]),
            -abs(float(row["radius_lambda"]) - 1.0),
        ),
    )


def _iter_configs() -> Iterable[tuple[int, float, float, str]]:
    for k in K_VALUES:
        for radius_lambda in LAMBDA_VALUES:
            for threshold in THRESHOLD_VALUES:
                for acceptance_mode in ACCEPTANCE_MODES:
                    yield k, radius_lambda, threshold, acceptance_mode


def build_dataset(dataset: str, method: str, device: torch.device, output_root: Path) -> dict[str, Any]:
    snapshot, views, values = _load_embeddings(dataset, method, device)
    all_rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        for radius_lambda in LAMBDA_VALUES:
            detector = _fit_detector(values["train"], views["train"], k, radius_lambda, "nearest_sphere")
            for acceptance_mode in ACCEPTANCE_MODES:
                detector.acceptance_mode = acceptance_mode
                outputs = {split: _vectorized_output(detector, values[split]) for split in ("val", "test")}
                for threshold in THRESHOLD_VALUES:
                    for split in ("val", "test"):
                        metrics = _metrics_from_output(detector, outputs[split], views[split], threshold=threshold)
                        all_rows.append(_metric_row(dataset, method, split, k, radius_lambda, threshold, acceptance_mode, metrics))
    val_rows = [row for row in all_rows if row["split"] == "val"]
    baseline = next(
        row
        for row in val_rows
        if row["k"] == 1
        and np.isclose(row["radius_lambda"], 1.0)
        and np.isclose(row["threshold"], 1.0)
        and row["acceptance_mode"] == "nearest_sphere"
    )
    selected = _select_validation(val_rows, baseline)
    selected_test = next(
        row
        for row in all_rows
        if row["split"] == "test"
        and row["k"] == selected["k"]
        and np.isclose(row["radius_lambda"], selected["radius_lambda"])
        and np.isclose(row["threshold"], selected["threshold"])
        and row["acceptance_mode"] == selected["acceptance_mode"]
    )
    dataset_root = output_root / dataset / method
    _write_csv(dataset_root / "workpoints.csv", all_rows)
    _write_csv(dataset_root / "selected_validation.csv", [selected])
    _write_csv(dataset_root / "selected_test_confirmation.csv", [selected_test])
    manifest = {
        "schema_version": "s2c.historical_archive_gate_search.v1",
        "dataset": dataset,
        "method": method,
        "seed": 42,
        "kir": 0.50,
        "protocol": "historical_v19_archive_development_tuning",
        "device": str(device),
        "selection_split": "val",
        "selection_uses_validation_oos_labels": True,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "known_f1_accuracy_guard_pp": 1.0,
        "baseline_validation": baseline,
        "selected_validation": selected,
        "selected_test_confirmation": selected_test,
        "candidate_count_per_split": len(list(_iter_configs())),
        "data_snapshot": snapshot,
    }
    (dataset_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    global DATA_ROOT, MODEL_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    DATA_ROOT = args.data_root.resolve()
    MODEL_ROOT = args.model_root.resolve()
    device = choose_device(args.device)
    datasets = tuple(args.dataset or DATASETS)
    methods = tuple(args.method or METHODS)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = [build_dataset(dataset, method, device, output_root) for dataset in datasets for method in methods]
    summary = {
        "schema_version": "s2c.historical_archive_gate_search.summary.v1",
        "protocol": "historical_v19_archive_development_tuning",
        "datasets": list(datasets),
        "methods": list(methods),
        "device": str(device),
        "completed_units": len(manifests),
        "selection_split": "val_with_oos_labels",
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    (output_root / "MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
