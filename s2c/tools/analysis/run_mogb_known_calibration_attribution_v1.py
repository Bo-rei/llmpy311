#!/usr/bin/env python3
"""Run Known-only MOGB boundary calibration and loss-contract attribution.

This experiment reuses the frozen MiniLM cache and the existing MOGB-Fair
partition adapter.  It never encodes text, trains an encoder, or uses test OOS
labels to select a threshold.  Radius multipliers are determined solely by a
pre-registered target coverage on ``calibration_known``.

The second half evaluates the upstream sub-centroid loss formula on actual
MOGB-Fair distance tables.  It is a numerical/gradient diagnostic, not a new
training result and not an official MOGB reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from protocol_v2.evaluation.metrics import compute_binary_oos_metrics  # noqa: E402
from protocol_v2.experiments.mogb import (  # noqa: E402
    AdaptiveGranularBallClusterer,
    MOGBBoundary,
    make_mogb_boundaries,
    score_mogb_boundaries,
)
from protocol_v2.experiments.partitions import normalize_for_detector  # noqa: E402


PROTOCOL = "protocol_v2_textoir_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
TARGET_COVERAGES = (0.80, 0.85, 0.90, 0.95)
LOSS_VARIANTS = ("official_l1", "raw_tau_0.05", "raw_tau_0.10", "raw_tau_0.20", "raw_tau_0.50", "raw_tau_1.00")

ARTIFACTS = ROOT.parent / "artifacts" / "s2c"
CACHE_ROOT = ARTIFACTS / "cache" / "embeddings" / PROTOCOL
VIEW_ROOT = ROOT / "data" / "views" / PROTOCOL
FAIR_ROOT = ARTIFACTS / "runs" / PROTOCOL / "mogb_baseline_v1"
RUN_ROOT = ARTIFACTS / "runs" / PROTOCOL / "mogb_known_calibration_attribution_v1"
OUT = ROOT / "results" / "analysis" / "mogb_known_calibration_attribution_v1"
FIG = ROOT / "figures" / "mogb_known_calibration_attribution_v1"
REPORT = ROOT / "docs" / "analysis" / "MOGB_KNOWN_CALIBRATION_ATTRIBUTION_V1.md"
TRAINABLE_SOURCE = ROOT / "results" / "analysis" / "minilm_trainable_5seed_fair_v1" / "trainable_per_seed.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["sample_id"]) for row in rows).encode()).hexdigest()


def embedding_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes(order="C")).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_rows(dataset: str, seed: int, kir: float, split: str) -> list[dict[str, Any]]:
    path = VIEW_ROOT / dataset / f"seed_{seed}" / f"kir_{kir:.2f}" / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError(f"Duplicate sample_id in {path}")
    return rows


def load_embeddings(dataset: str, seed: int, kir: float, split: str, rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any], Path]:
    root = CACHE_ROOT / dataset / f"seed_{seed}" / f"kir_{kir:.2f}"
    ids_hash = sha256_ids(rows)
    matches: list[tuple[Path, dict[str, Any]]] = []
    for metadata_path in root.glob(f"{split}_*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sample_ids_sha256") == ids_hash and metadata_path.with_suffix(".npz").is_file():
            matches.append((metadata_path, metadata))
    if not matches:
        raise FileNotFoundError(f"No sample-aligned cache: {dataset}/{seed}/{kir:.2f}/{split}")
    hashes = {item[1].get("embedding_sha256") for item in matches}
    if len(hashes) != 1:
        raise ValueError(f"Conflicting aligned embedding caches: {dataset}/{seed}/{kir:.2f}/{split}")
    metadata_path, metadata = sorted(matches, key=lambda item: item[0].name)[0]
    npz_path = metadata_path.with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as archive:
        values = np.ascontiguousarray(archive["embeddings"], dtype=np.float32)
    if values.shape[0] != len(rows):
        raise ValueError(f"Row count mismatch: {npz_path}")
    if embedding_sha256(values) != metadata["embedding_sha256"]:
        raise ValueError(f"Embedding content hash mismatch: {npz_path}")
    return values, metadata, npz_path


def gold_oos(row: dict[str, Any]) -> int:
    return int(str(row.get("evaluation_label", row.get("intent", ""))) == "oos")


def open_metrics(rows: list[dict[str, Any]], scores: np.ndarray, nearest_labels: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray([gold_oos(row) for row in rows], dtype=np.int64)
    binary = compute_binary_oos_metrics(labels, scores, threshold=threshold)
    predicted_oos = scores > threshold
    predicted = np.asarray(nearest_labels, dtype=object).copy()
    predicted[predicted_oos] = "oos"
    gold = np.asarray(["oos" if gold_oos(row) else str(row["intent"]) for row in rows], dtype=object)
    known_labels = sorted({str(row["intent"]) for row in rows if not gold_oos(row)})
    all_labels = known_labels + ["oos"]
    return {
        **binary,
        "accuracy": float(accuracy_score(gold, predicted)),
        "f1_all": float(f1_score(gold, predicted, labels=all_labels, average="macro", zero_division=0)),
        "f1_k": float(f1_score(gold, predicted, labels=known_labels, average="macro", zero_division=0)),
        "f1_u": float(f1_score(gold == "oos", predicted == "oos", average="binary", zero_division=0)),
    }


def nearest_labels(output: dict[str, np.ndarray], boundaries: list[MOGBBoundary]) -> np.ndarray:
    by_id = {boundary.ball_id: boundary.label for boundary in boundaries}
    return np.asarray([by_id[int(ball_id)] for ball_id in output["nearest_ball"]], dtype=object)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def distance_table(sample: np.ndarray, clusterer: AdaptiveGranularBallClusterer, class_labels: list[str]) -> np.ndarray:
    result = np.empty((sample.shape[0], len(class_labels)), dtype=np.float64)
    by_label: dict[str, list[np.ndarray]] = {label: [] for label in class_labels}
    for ball in clusterer.selected_balls:
        by_label[ball.majority_label].append(np.asarray(ball.centroid, dtype=np.float64))
    missing = [label for label, centers in by_label.items() if not centers]
    if missing:
        raise RuntimeError(f"Selected MOGB balls omit train labels: {missing}")
    sample_norm = np.einsum("ij,ij->i", sample, sample)[:, None]
    for class_index, label in enumerate(class_labels):
        centers = np.stack(by_label[label], axis=0)
        squared = sample_norm + np.einsum("ij,ij->i", centers, centers)[None, :] - 2.0 * sample @ centers.T
        np.maximum(squared, 0.0, out=squared)
        result[:, class_index] = np.sqrt(squared.min(axis=1))
    return result


def loss_contract_metrics(distances: np.ndarray, true_index: np.ndarray, variant: str) -> dict[str, float]:
    row = np.arange(distances.shape[0])
    if variant == "official_l1":
        denominator = np.maximum(distances.sum(axis=1, keepdims=True), 1e-12)
        normalized = distances / denominator
        probabilities = stable_softmax(-normalized)
        grad_u = -probabilities
        grad_u[row, true_index] += 1.0
        weighted = np.sum(grad_u * distances, axis=1, keepdims=True)
        gradients = (grad_u * denominator - weighted) / np.square(denominator)
        temperature = math.nan
    else:
        temperature = float(variant.rsplit("_", 1)[1])
        probabilities = stable_softmax(-distances / temperature)
        gradients = -probabilities / temperature
        gradients[row, true_index] += 1.0 / temperature
    true_probability = probabilities[row, true_index]
    masked = probabilities.copy()
    masked[row, true_index] = -np.inf
    next_probability = masked.max(axis=1)
    loss = -np.log(np.maximum(true_probability, 1e-15))
    predicted = np.argmin(distances, axis=1)
    return {
        "temperature": temperature,
        "mean_loss": float(loss.mean()),
        "mean_true_probability": float(true_probability.mean()),
        "median_true_probability": float(np.median(true_probability)),
        "mean_probability_margin": float(np.mean(true_probability - next_probability)),
        "nearest_class_accuracy": float(np.mean(predicted == true_index)),
        "mean_gradient_l2": float(np.mean(np.linalg.norm(gradients, axis=1))),
        "median_gradient_l2": float(np.median(np.linalg.norm(gradients, axis=1))),
        "mean_distance_span": float(np.mean(distances.max(axis=1) - distances.min(axis=1))),
    }


def theoretical_bound(class_count: int) -> dict[str, float]:
    uniform_probability = 1.0 / class_count
    maximum_probability = 1.0 / (1.0 + (class_count - 1) * math.exp(-1.0 / (class_count - 1)))
    return {
        "class_count": class_count,
        "uniform_probability": uniform_probability,
        "maximum_true_probability": maximum_probability,
        "uniform_ce": -math.log(uniform_probability),
        "minimum_possible_ce": -math.log(maximum_probability),
        "probability_headroom": maximum_probability - uniform_probability,
    }


def equivalence_check(dataset: str, seed: int, kir: float, scores: np.ndarray, nearest: np.ndarray, metrics: dict[str, float]) -> dict[str, Any]:
    run_dir = FAIR_ROOT / dataset / f"kir_{kir:.2f}" / f"seed_{seed}" / "mogb_minilm"
    prediction_path = run_dir / "predictions.tsv"
    metric_path = run_dir / "metrics.json"
    if not prediction_path.is_file() or not metric_path.is_file():
        raise FileNotFoundError(f"Missing frozen MOGB-Fair reference: {run_dir}")
    reference_predictions = pd.read_csv(prediction_path, sep="\t")
    if len(reference_predictions) != len(scores):
        raise ValueError(f"Prediction count mismatch: {run_dir}")
    reference_metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    metric_keys = ("oos_f1", "id_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos", "accuracy", "f1_all", "f1_k")
    metric_delta = max(abs(float(metrics[key]) - float(reference_metrics[key])) for key in metric_keys)
    score_delta = float(np.max(np.abs(scores - reference_predictions["normalized_score"].to_numpy(dtype=float))))
    label_mismatch = int(np.sum(nearest.astype(str) != reference_predictions["predicted_label"].astype(str).where(reference_predictions["predicted_is_oos"] == 0, nearest.astype(str))))
    # Nearest labels are not exported separately for rejected samples.  Score
    # equality plus accepted-label equality is the reproducibility gate.
    accepted = reference_predictions["predicted_is_oos"].to_numpy(dtype=int) == 0
    accepted_label_mismatch = int(np.sum(nearest[accepted].astype(str) != reference_predictions.loc[accepted, "predicted_label"].astype(str).to_numpy()))
    passed = metric_delta <= 1e-12 and score_delta <= 1e-12 and accepted_label_mismatch == 0
    return {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "metric_max_abs_delta": metric_delta,
        "score_max_abs_delta": score_delta,
        "accepted_label_mismatch": accepted_label_mismatch,
        "diagnostic_label_mismatch": label_mismatch,
        "passed": passed,
        "reference_metrics_sha256": sha256_file(metric_path),
        "reference_predictions_sha256": sha256_file(prediction_path),
    }


def run_cell(dataset: str, seed: int, kir: float) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    train_rows = load_rows(dataset, seed, kir, "train_known")
    calibration_rows = load_rows(dataset, seed, kir, "calibration_known")
    test_rows = load_rows(dataset, seed, kir, "test_combined")
    train, train_meta, train_path = load_embeddings(dataset, seed, kir, "train_known", train_rows)
    calibration, calibration_meta, calibration_path = load_embeddings(dataset, seed, kir, "calibration_known", calibration_rows)
    test, test_meta, test_path = load_embeddings(dataset, seed, kir, "test_combined", test_rows)
    train_norm = normalize_for_detector(train)
    calibration_norm = normalize_for_detector(calibration)
    test_norm = normalize_for_detector(test)
    labels = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    started = time.perf_counter()
    clusterer = AdaptiveGranularBallClusterer(seed=seed).fit(train_norm, labels)
    boundaries = make_mogb_boundaries(clusterer, boundary="mean", distance="euclidean")
    calibration_output = score_mogb_boundaries(calibration_norm, boundaries, distance="euclidean")
    test_output = score_mogb_boundaries(test_norm, boundaries, distance="euclidean")
    nearest = nearest_labels(test_output, boundaries)
    default_metrics = open_metrics(test_rows, test_output["score"], nearest, 1.0)
    equivalence = equivalence_check(dataset, seed, kir, test_output["score"], nearest, default_metrics)
    if not equivalence["passed"]:
        raise RuntimeError(f"MOGB-Fair equivalence failed: {equivalence}")

    workpoints: list[dict[str, Any]] = []
    configurations = [("default_radius", math.nan, 1.0)]
    for coverage in TARGET_COVERAGES:
        configurations.append((f"calibration_coverage_{coverage:.2f}", coverage, float(np.quantile(calibration_output["score"], coverage))))
    for name, coverage, multiplier in configurations:
        metrics = open_metrics(test_rows, test_output["score"], nearest, multiplier)
        calibration_recall = float(np.mean(calibration_output["score"] <= multiplier))
        workpoints.append(
            {
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "workpoint": name,
                "target_calibration_coverage": coverage,
                "radius_multiplier": multiplier,
                "calibration_known_recall": calibration_recall,
                "selected_using_test_oos": False,
                "selected_using_test_known": False,
                "selected_split": "calibration_known" if name != "default_radius" else "paper_default_radius",
                "selected_balls": len(boundaries),
                "mean_balls_per_intent": clusterer.ball_statistics()["mean_balls_per_intent"],
                "minimum_ball_size": min(ball.sample_count for ball in clusterer.selected_balls),
                "fit_and_score_seconds": time.perf_counter() - started,
                **metrics,
            }
        )

    all_class_labels = sorted(set(labels.tolist()))
    selected_class_labels = sorted({ball.majority_label for ball in clusterer.selected_balls})
    missing_selected_classes = sorted(set(all_class_labels).difference(selected_class_labels))
    supported_mask = np.isin(labels, selected_class_labels)
    supported_indices = np.flatnonzero(supported_mask)
    rng = np.random.default_rng(seed + int(round(kir * 1000)) + len(dataset) * 1009)
    sample_count = min(1024, len(supported_indices))
    sample_indices = np.sort(rng.choice(supported_indices, size=sample_count, replace=False))
    sampled = train_norm[sample_indices]
    sampled_labels = labels[sample_indices]
    distances = distance_table(sampled, clusterer, selected_class_labels)
    class_to_index = {label: index for index, label in enumerate(selected_class_labels)}
    true_index = np.asarray([class_to_index[str(label)] for label in sampled_labels], dtype=np.int64)
    loss_rows = []
    for variant in LOSS_VARIANTS:
        loss_rows.append(
            {
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "loss_variant": variant,
                "class_count": len(selected_class_labels),
                "registered_known_class_count": len(all_class_labels),
                "missing_selected_class_count": len(missing_selected_classes),
                "missing_selected_classes": "|".join(missing_selected_classes),
                "excluded_train_row_count": int(np.sum(~supported_mask)),
                "sample_count": sample_count,
                "selected_balls": len(boundaries),
                **loss_contract_metrics(distances, true_index, variant),
            }
        )
    provenance = {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "train_sample_ids_sha256": sha256_ids(train_rows),
        "calibration_sample_ids_sha256": sha256_ids(calibration_rows),
        "test_sample_ids_sha256": sha256_ids(test_rows),
        "train_embedding_sha256": train_meta["embedding_sha256"],
        "calibration_embedding_sha256": calibration_meta["embedding_sha256"],
        "test_embedding_sha256": test_meta["embedding_sha256"],
        "train_cache": str(train_path.relative_to(ROOT.parent)),
        "calibration_cache": str(calibration_path.relative_to(ROOT.parent)),
        "test_cache": str(test_path.relative_to(ROOT.parent)),
    }
    return workpoints, equivalence, loss_rows, provenance


def summarize(workpoints: pd.DataFrame, losses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "radius_multiplier",
        "calibration_known_recall",
        "oos_f1",
        "f1_all",
        "f1_k",
        "accuracy",
        "id_recall",
        "false_accept_rate",
        "false_reject_rate",
        "auroc",
        "aupr_oos",
    ]
    grouped = workpoints.groupby(["dataset", "kir", "workpoint"], as_index=False)[metric_columns].agg(["mean", "std"])
    grouped.columns = ["_".join(part for part in col if part) for col in grouped.columns]
    loss_metrics = ["mean_loss", "mean_true_probability", "mean_probability_margin", "nearest_class_accuracy", "mean_gradient_l2", "mean_distance_span"]
    loss_summary = losses.groupby(["dataset", "kir", "loss_variant"], as_index=False)[loss_metrics].agg(["mean", "std"])
    loss_summary.columns = ["_".join(part for part in col if part) for col in loss_summary.columns]
    return grouped, loss_summary


def plot_results(
    workpoints: pd.DataFrame,
    summary: pd.DataFrame,
    losses: pd.DataFrame,
    bounds: pd.DataFrame,
    trainable: pd.DataFrame,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    labels = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        subset = workpoints[(workpoints.dataset == dataset) & (workpoints.kir == 0.50)]
        for workpoint, frame in subset.groupby("workpoint"):
            axis.scatter(frame.id_recall.mean(), frame.oos_f1.mean(), s=55, label=workpoint.replace("calibration_coverage_", "cal "))
        trainable_point = trainable[(trainable.dataset == dataset) & (trainable.kir == 0.50)]
        axis.scatter(
            trainable_point.id_recall.mean(),
            trainable_point.oos_f1.mean(),
            marker="*",
            s=150,
            color="black",
            label="S2C Trainable K=1",
            zorder=5,
        )
        axis.set_title(labels[dataset])
        axis.set_xlabel("Test Known Recall")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test OOS F1")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("MOGB-Fair Known-only calibration work points (KIR=0.50)")
    fig.tight_layout()
    fig.savefig(FIG / "known_recall_oos_f1_workpoints.png", dpi=220)
    plt.close(fig)

    calibration = workpoints[workpoints.workpoint != "default_radius"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        subset = calibration[calibration.dataset == dataset]
        for kir, frame in subset.groupby("kir"):
            means = frame.groupby("target_calibration_coverage").radius_multiplier.mean()
            axis.plot(means.index, means.values, marker="o", label=f"KIR={kir:.2f}")
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="MOGB default")
        axis.set_title(labels[dataset])
        axis.set_xlabel("Target calibration Known coverage")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Required radius multiplier")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Default mean radius is narrower than Known-only calibration requires")
    fig.tight_layout()
    fig.savefig(FIG / "radius_multiplier_by_coverage.png", dpi=220)
    plt.close(fig)

    merged = workpoints.pivot_table(index=["dataset", "kir", "seed"], columns="workpoint", values=["oos_f1", "f1_all", "id_recall", "false_accept_rate"])
    delta = {}
    for metric in ("oos_f1", "f1_all", "id_recall", "false_accept_rate"):
        delta[metric] = merged[(metric, "calibration_coverage_0.95")] - merged[(metric, "default_radius")]
    delta_frame = pd.DataFrame(delta).reset_index().groupby("dataset")[["oos_f1", "f1_all", "id_recall", "false_accept_rate"]].mean()
    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    delta_frame.mul(100).plot(kind="bar", ax=axis)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Mean change (percentage points)")
    axis.set_xlabel("")
    axis.set_title("Effect of calibration-95 radius versus MOGB default")
    axis.legend(["OOS F1", "F1-All", "Known Recall", "False acceptance"], fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "cal95_delta_vs_default.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for variant, frame in losses.groupby("loss_variant"):
        means = frame.groupby("class_count").mean_true_probability.mean().sort_index()
        axes[0].plot(means.index, means.values, marker="o", label=variant)
        gradients = frame.groupby("class_count").mean_gradient_l2.mean().sort_index()
        axes[1].plot(gradients.index, gradients.values, marker="o", label=variant)
    axes[0].plot(bounds.class_count, bounds.maximum_true_probability, color="black", linestyle="--", label="official theoretical max")
    axes[0].set_ylabel("Mean true-class probability")
    axes[1].set_ylabel("Mean gradient L2 w.r.t. class distances")
    for axis in axes:
        axis.set_xlabel("Known class count")
        axis.grid(alpha=0.25)
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Official L1-normalized sub-centroid loss compresses probability and gradient signal")
    fig.tight_layout()
    fig.savefig(FIG / "subcentroid_loss_signal.png", dpi=220)
    plt.close(fig)

    comparison = workpoints.merge(
        trainable[["dataset", "kir", "seed", "oos_f1", "f1_all", "id_recall", "false_accept_rate"]],
        on=["dataset", "kir", "seed"],
        suffixes=("_mogb", "_trainable"),
    )
    cal80 = comparison[comparison.workpoint == "calibration_coverage_0.80"].copy()
    cal80["oos_f1_delta"] = cal80.oos_f1_trainable - cal80.oos_f1_mogb
    cal80["f1_all_delta"] = cal80.f1_all_trainable - cal80.f1_all_mogb
    delta_summary = cal80.groupby(["dataset", "kir"])[["oos_f1_delta", "f1_all_delta"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    for dataset, frame in delta_summary.groupby("dataset"):
        axes[0].plot(frame.kir, 100 * frame.oos_f1_delta, marker="o", label=labels[dataset])
        axes[1].plot(frame.kir, 100 * frame.f1_all_delta, marker="o", label=labels[dataset])
    axes[0].set_ylabel("S2C Trainable K=1 − MOGB cal-80 OOS F1 (pp)")
    axes[1].set_ylabel("S2C Trainable K=1 − MOGB cal-80 F1-All (pp)")
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("KIR")
        axis.set_xticks(KIRS)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Same-protocol comparison: current S2C candidate versus calibrated MOGB-Fair")
    fig.tight_layout()
    fig.savefig(FIG / "trainable_k1_vs_mogb_cal80.png", dpi=220)
    plt.close(fig)


def build_report(
    workpoints: pd.DataFrame,
    losses: pd.DataFrame,
    equivalence: pd.DataFrame,
    trainable: pd.DataFrame,
    manifest_sha: str,
) -> str:
    default = workpoints[workpoints.workpoint == "default_radius"]
    cal95 = workpoints[workpoints.workpoint == "calibration_coverage_0.95"]
    paired = default.merge(cal95, on=["dataset", "kir", "seed"], suffixes=("_default", "_cal95"))
    dataset_rows = []
    for dataset, frame in paired.groupby("dataset"):
        dataset_rows.append(
            f"| {dataset} | {frame.radius_multiplier_cal95.mean():.3f} | "
            f"{100 * (frame.oos_f1_cal95 - frame.oos_f1_default).mean():+.2f} | "
            f"{100 * (frame.f1_all_cal95 - frame.f1_all_default).mean():+.2f} | "
            f"{100 * (frame.id_recall_cal95 - frame.id_recall_default).mean():+.2f} | "
            f"{100 * (frame.false_accept_rate_cal95 - frame.false_accept_rate_default).mean():+.2f} |"
        )
    official = losses[losses.loss_variant == "official_l1"]
    raw = losses[losses.loss_variant == "raw_tau_0.10"]
    missing_rows = []
    for dataset, frame in official.groupby("dataset"):
        missing_rows.append(
            f"| {dataset} | {int(np.sum(frame.missing_selected_class_count > 0))}/15 | "
            f"{frame.missing_selected_class_count.mean():.2f} | {int(frame.missing_selected_class_count.max())} | "
            f"{frame.excluded_train_row_count.mean():.1f} | {int(frame.excluded_train_row_count.max())} |"
        )
    comparison = workpoints.merge(
        trainable[["dataset", "kir", "seed", "oos_f1", "f1_all", "id_recall", "false_accept_rate"]],
        on=["dataset", "kir", "seed"],
        suffixes=("_mogb", "_trainable"),
    )
    cal80 = comparison[comparison.workpoint == "calibration_coverage_0.80"].copy()
    comparison_rows = []
    for dataset in DATASETS:
        trainable_cell = trainable[(trainable.dataset == dataset) & (trainable.kir == 0.50)]
        default_cell = workpoints[
            (workpoints.dataset == dataset) & (workpoints.kir == 0.50) & (workpoints.workpoint == "default_radius")
        ]
        cal80_cell = workpoints[
            (workpoints.dataset == dataset)
            & (workpoints.kir == 0.50)
            & (workpoints.workpoint == "calibration_coverage_0.80")
        ]
        for method, frame in (
            ("S2C Trainable K=1", trainable_cell),
            ("MOGB-Fair default", default_cell),
            ("MOGB-Fair cal-80", cal80_cell),
        ):
            comparison_rows.append(
                f"| {dataset} | {method} | {100 * frame.oos_f1.mean():.2f} | "
                f"{100 * frame.f1_all.mean():.2f} | {100 * frame.id_recall.mean():.2f} | "
                f"{100 * frame.false_accept_rate.mean():.2f} |"
            )
    trainable_oos_wins = int(np.sum(cal80.oos_f1_trainable > cal80.oos_f1_mogb))
    trainable_f1_all_wins = int(np.sum(cal80.f1_all_trainable > cal80.f1_all_mogb))
    return f"""# MOGB Known-only 校准与子中心损失归因 V1

## 结论

本实验完成 **45 个 dataset×KIR×seed 粒球单元、225 个边界工作点评价单元和 270 个损失契约评价单元**。
45/45 个重拟合单元与冻结的 `mogb_minilm` 参考结果达到 score/指标数值等价，因此以下差异不是缓存错位或重新实现漂移。

MOGB-Fair 的平均距离半径在 Known calibration 上普遍过窄。把半径扩大到仅由 `calibration_known` 决定的 95% 覆盖工作点，可以显著恢复 Known Recall，但同时增加 OOS false acceptance；这说明本地 MOGB 的弱综合性能包含明确的**工作点失配**，但不能仅靠半径放大自动得到更好的开放集排序，因为 AUROC/AUPR 不随单调阈值变化。

官方代码中的子中心损失先对每个样本的类别距离向量做 L1 归一化，再对负距离做 softmax。实际 MOGB-Fair 距离表上，`official_l1` 的平均真实类概率为 `{official.mean_true_probability.mean():.6f}`，平均距离梯度范数为 `{official.mean_gradient_l2.mean():.6f}`；未经这一步压缩的 `raw distance / tau=0.10` 分别为 `{raw.mean_true_probability.mean():.6f}` 和 `{raw.mean_gradient_l2.mean():.6f}`。这与严格兼容运行中“CE 收敛而 sub-centroid loss 接近均匀分布”的现象一致，是当前最强的代码级根因证据之一。

粒球筛选还会让部分 Known 类没有任何 selected ball。该现象不是脚本假设：45/45 Gate 重放与历史结果等价，但损失诊断必须排除没有中心的类别，不能伪造距离。

| 数据集 | 出现缺类的单元 | 平均缺失类数 | 最大缺失类数 | 平均排除训练行 | 最大排除训练行 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(missing_rows)}

## Calibration-95 相对默认平均变化

| 数据集 | 所需半径倍率 | OOS F1 (pp) | F1-All (pp) | Known Recall (pp) | False acceptance (pp) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(dataset_rows)}

这些变化是 3 个 KIR×5 个 seed 的均值。Calibration-95 是预注册 Known coverage 工作点，不使用 test Known 或 test OOS 选择；它不是声称最优的半径参数。

## 对“为什么没有复现论文 MOGB”的回答

1. **不是简单的训练轮数不足。** 既有 BERT 严格兼容单格中 CE 和 Known dev accuracy 已收敛。
2. **边界工作点确实有系统性失配。** 默认 mean radius 会拒绝大量 Known；Known-only 校准能恢复覆盖，但必然增加开放空间接受风险。
3. **子中心训练信号被公式压缩。** L1-normalized distance softmax 的概率动态范围随 Known 类数增长快速缩小；本实验在真实距离表上验证了这一点。
4. **仍不能把差距完全归因于代码缺陷。** 论文原始样本 ID、Known intent 列表、旧运行环境和完整作者数据契约不可恢复；本实验是官方公式和适配组件的归因，不是对论文结果真实性的否定。

## 当前自有方法与 MOGB-Fair 的同协议比较

这里比较的自有方法是 **S2C Trainable MiniLM K=1 Gate**：Known-only 训练 MiniLM 最后两层和 residual projection，随后使用单中心对角马氏 Gate；不是 `fulltex.tex` 中的完整 Gate→Router→Expert Cascade。MOGB-Fair 使用相同 TEXTOIR split 和冻结 MiniLM，但只适配 MOGB 粒球与边界组件；不是官方 BERT 完整 MOGB。

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False acceptance |
|---|---|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

在全部 45 个同 seed 配对中，S2C Trainable K=1 相对预注册的 MOGB cal-80 工作点取得 OOS F1 胜出 `{trainable_oos_wins}/45`、F1-All 胜出 `{trainable_f1_all_wins}/45`。这表明当前差异不能只用 MOGB 默认半径过窄解释；即使把 MOGB 调到较合理的 Known 覆盖，其开放集排序和综合 Known/OOS 权衡仍通常弱于当前 Trainable K=1。

## 图表

- `figures/archive/analysis/mogb_known_calibration_attribution_v1/known_recall_oos_f1_workpoints.png`
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/radius_multiplier_by_coverage.png`
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/cal95_delta_vs_default.png`
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/subcentroid_loss_signal.png`
- `figures/archive/analysis/mogb_known_calibration_attribution_v1/trainable_k1_vs_mogb_cal80.png`

## 证据与边界

- 等价门：`equivalence.csv`，通过 `{int(equivalence.passed.sum())}/{len(equivalence)}`。
- 正式工作点表：`workpoint_per_seed.csv`，选择 split 始终为 Known calibration。
- 损失数值实验：`loss_contract_per_cell.csv`，只评价训练信号，不宣称测试性能提升。
- Manifest SHA256：`{manifest_sha}`。
- 原始文本、embedding、checkpoint 和逐样本预测没有复制到轻量结果目录。
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Reuse a complete 45-cell artifact table when present")
    parser.add_argument("--max-cells", type=int, help="Debug-only cell cap; omitted means the registered 45 cells")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    workpoint_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    cells = [(dataset, seed, kir) for dataset in DATASETS for kir in KIRS for seed in SEEDS]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    started = time.time()
    existing_manifest = OUT / "MANIFEST.json"
    reuse_complete = args.resume and args.max_cells is None and existing_manifest.is_file()
    if reuse_complete and json.loads(existing_manifest.read_text(encoding="utf-8")).get("status") == "complete":
        workpoints = pd.read_csv(OUT / "workpoint_per_seed.csv")
        equivalence = pd.read_csv(OUT / "equivalence.csv")
        losses = pd.read_csv(OUT / "loss_contract_per_cell.csv")
        provenance = pd.read_csv(RUN_ROOT / "input_provenance.csv")
    else:
        for index, (dataset, seed, kir) in enumerate(cells, start=1):
            print(f"[{index}/{len(cells)}] {dataset} kir={kir:.2f} seed={seed}", flush=True)
            workpoints, equivalence, losses, provenance = run_cell(dataset, seed, kir)
            workpoint_rows.extend(workpoints)
            equivalence_rows.append(equivalence)
            loss_rows.extend(losses)
            provenance_rows.append(provenance)
        workpoints = pd.DataFrame(workpoint_rows)
        equivalence = pd.DataFrame(equivalence_rows)
        losses = pd.DataFrame(loss_rows)
        provenance = pd.DataFrame(provenance_rows)
    if len(cells) == 45:
        if len(workpoints) != 225 or len(losses) != 270 or len(equivalence) != 45:
            raise RuntimeError("Registered matrix coverage mismatch")
        if not equivalence.passed.all():
            raise RuntimeError("At least one MOGB-Fair equivalence gate failed")
    workpoint_summary, loss_summary = summarize(workpoints, losses)
    bounds = pd.DataFrame([theoretical_bound(count) for count in sorted(set(losses.class_count.astype(int)))])
    trainable = pd.read_csv(TRAINABLE_SOURCE)
    trainable = trainable[trainable.method == "trainable_k1"].copy()
    if len(trainable) != 45 or trainable.seed.nunique() != 5:
        raise RuntimeError("Expected the registered 45-cell Trainable K=1 comparison table")
    comparison = workpoints.merge(
        trainable[["dataset", "kir", "seed", "oos_f1", "f1_all", "id_recall", "false_accept_rate"]],
        on=["dataset", "kir", "seed"],
        suffixes=("_mogb", "_trainable"),
    )
    for metric in ("oos_f1", "f1_all", "id_recall", "false_accept_rate"):
        comparison[f"{metric}_trainable_minus_mogb"] = comparison[f"{metric}_trainable"] - comparison[f"{metric}_mogb"]

    atomic_csv(workpoints, OUT / "workpoint_per_seed.csv")
    atomic_csv(workpoint_summary, OUT / "workpoint_summary.csv")
    atomic_csv(equivalence, OUT / "equivalence.csv")
    atomic_csv(losses, OUT / "loss_contract_per_cell.csv")
    atomic_csv(loss_summary, OUT / "loss_contract_summary.csv")
    atomic_csv(
        losses[losses.loss_variant == "official_l1"][
            [
                "dataset",
                "kir",
                "seed",
                "registered_known_class_count",
                "class_count",
                "missing_selected_class_count",
                "missing_selected_classes",
                "excluded_train_row_count",
                "selected_balls",
            ]
        ],
        OUT / "missing_selected_classes.csv",
    )
    atomic_csv(bounds, OUT / "class_count_probability_bound.csv")
    atomic_csv(comparison, OUT / "trainable_vs_calibrated_mogb.csv")
    atomic_csv(provenance, RUN_ROOT / "input_provenance.csv")
    plot_results(workpoints, workpoint_summary, losses, bounds, trainable)

    manifest = {
        "experiment_id": "mogb_known_calibration_attribution_v1",
        "protocol_version": PROTOCOL,
        "status": "complete" if len(cells) == 45 else "debug_partial",
        "completed_cells": len(cells),
        "planned_cells": 45,
        "workpoint_units": len(workpoints),
        "loss_contract_units": len(losses),
        "equivalence_passed": int(equivalence.passed.sum()),
        "test_used_for_selection": False,
        "selection_split": "calibration_known",
        "target_coverages": TARGET_COVERAGES,
        "loss_variants": LOSS_VARIANTS,
        "elapsed_seconds": time.time() - started,
        "script_sha256": sha256_file(Path(__file__)),
        "source_hashes": {
            "mogb_adapter": sha256_file(ROOT / "src/protocol_v2/experiments/mogb.py"),
            "official_loss": sha256_file(ROOT / "third_party/mogb_official/myloss.py"),
            "trainable_comparison": sha256_file(TRAINABLE_SOURCE),
        },
    }
    atomic_json(manifest, RUN_ROOT / "CLOSEOUT.json")
    manifest["output_hashes"] = {
        path.name: sha256_file(path)
        for path in sorted(OUT.glob("*.csv"))
    }
    atomic_json(manifest, OUT / "MANIFEST.json")
    manifest_sha = sha256_file(OUT / "MANIFEST.json")
    atomic_text(build_report(workpoints, losses, equivalence, trainable, manifest_sha), REPORT)
    print(json.dumps({"status": manifest["status"], "cells": len(cells), "workpoints": len(workpoints), "loss_units": len(losses), "manifest_sha256": manifest_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
