#!/usr/bin/env python3
"""Build paper-style H1 v19 visual explanations for Frozen vs Trainable MiniLM.

This is a post-hoc analysis of completed historical controlled Gate runs.  It
does not train a model, change the score=1 decision rule, or mix H1 evidence
with the strict historical H0 Cascade or protocol_v2 results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RUN_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_protocol_v2" / "minilm_k1"
DATA_ROOT = ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19"
MODEL_ROOT = ROOT.parent / "assets" / "models" / "all-MiniLM-L6-v2"
RESULT_ROOT = ROOT / "results" / "analysis" / "historical_oos_visual_explanation"
FIGURE_ROOT = ROOT / "figures" / "historical_oos_visual_explanation"

DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87)
METHODS = ("frozen_k1", "trainable_k1")
DATASET_LABELS = {
    "clinc150": "CLINC150",
    "stackoverflow": "StackOverflow",
    "banking77_oos": "BANKING77-OOS",
}
METHOD_LABELS = {"frozen_k1": "Frozen", "trainable_k1": "Trainable"}

BLUE = "#4C78A8"
RED = "#C44E52"
GRAY = "#8B929A"
LIGHT_GRAY = "#D9DDE1"
DARK = "#252525"
GOLD = "#F2B134"
METHOD_COLORS = {"frozen_k1": BLUE, "trainable_k1": RED}
TRANSITIONS = ("both_correct", "trainable_only", "frozen_only", "both_wrong")
TRANSITION_LABELS = {
    "both_correct": "both reject",
    "trainable_only": "corrected by Trainable",
    "frozen_only": "regressed under Trainable",
    "both_wrong": "both accept",
}
FIGURE_STEMS = (
    "paper_style_local_boundary_geometry",
    "paper_style_local_boundary_geometry_3d",
    "paper_style_score_distribution",
    "paired_oos_score_crossing",
    "corrected_oos_score_decomposition",
    "cross_dataset_oos_density",
    "oos_margin_difficulty_profile",
    "paired_kir_seed_stability",
    "oos_subtype_residual_risk",
    "oos_representation_movement_map",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"empty JSONL: {path}")
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(dataset: str, kir: float, seed: int) -> Path:
    return RUN_ROOT / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}"


def data_dir(dataset: str, kir: float, seed: int) -> Path:
    return DATA_ROOT / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}"


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def score_from_signature(embeddings: np.ndarray, signature: Mapping[str, Any], chunk_size: int = 512) -> dict[str, np.ndarray]:
    """Apply the stored nearest-sphere diagonal-Mahalanobis detector."""
    if signature.get("distance_metric") != "mahalanobis_diag":
        raise ValueError("visual explanation expects mahalanobis_diag signatures")
    if signature.get("acceptance_mode") != "nearest_sphere":
        raise ValueError("visual explanation expects nearest_sphere acceptance")
    spheres = sorted(signature["spheres"], key=lambda item: int(item["cluster_id"]))
    centers = np.asarray([item["center"] for item in spheres], dtype=np.float64)
    inv_cov = np.asarray([item["inv_diag_cov"] for item in spheres], dtype=np.float64)
    radii = np.asarray([item["radius"] for item in spheres], dtype=np.float64)
    cluster_ids = np.asarray([item["cluster_id"] for item in spheres], dtype=np.int64)
    values = l2_normalize(embeddings)
    nearest_rows: list[np.ndarray] = []
    distance_rows: list[np.ndarray] = []
    for start in range(0, len(values), chunk_size):
        diff = values[start : start + chunk_size, None, :] - centers[None, :, :]
        distances = np.sqrt(np.sum(diff * diff * inv_cov[None, :, :], axis=2))
        nearest = np.argmin(distances, axis=1)
        nearest_rows.append(nearest)
        distance_rows.append(distances[np.arange(len(nearest)), nearest])
    nearest_index = np.concatenate(nearest_rows) if nearest_rows else np.empty(0, dtype=np.int64)
    distance = np.concatenate(distance_rows) if distance_rows else np.empty(0, dtype=np.float64)
    radius = radii[nearest_index]
    score = distance / np.clip(radius, 1e-12, None)
    return {
        "nearest_cluster": cluster_ids[nearest_index],
        "distance": distance,
        "radius": radius,
        "score": score,
        "pred": (score > 1.0).astype(np.int64),
    }


def smoothed_histogram(values: Sequence[float], bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a normalized five-tap-smoothed density without dropping tails."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("density input is empty")
    upper = float(bins[-1])
    overflow = int(np.sum(array >= upper))
    clipped = np.clip(array, float(bins[0]), np.nextafter(upper, -np.inf))
    density, _ = np.histogram(clipped, bins=bins, density=True)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    density = np.convolve(density, kernel, mode="same")
    width = float(bins[1] - bins[0])
    density /= max(float(np.sum(density) * width), 1e-12)
    return (bins[:-1] + bins[1:]) / 2.0, density, overflow


def style_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.tick_params(length=2.8, width=0.65, pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E7E9EC", linewidth=0.45)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def load_completed_artifacts() -> tuple[pd.DataFrame, dict[tuple[str, int, str], pd.DataFrame]]:
    metric_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int, str], pd.DataFrame] = {}
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                base = run_dir(dataset, kir, seed)
                for method in METHODS:
                    metrics = read_json(base / method / "metrics.json")
                    metric_rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": DATASET_LABELS[dataset],
                            "kir": kir,
                            "seed": seed,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "oos_f1": float(metrics["oos_f1"]),
                            "false_accept_rate": float(metrics["false_accept_rate"]),
                            "known_recall": float(metrics["known_recall"]),
                            "auroc": float(metrics["auroc"]),
                            "aupr_oos": float(metrics["aupr_oos"]),
                        }
                    )
                    if math.isclose(kir, 0.50):
                        predictions[(dataset, seed, method)] = read_jsonl(base / method / "predictions.jsonl")
    metrics = pd.DataFrame(metric_rows)
    if len(metrics) != len(DATASETS) * len(KIRS) * len(SEEDS) * len(METHODS):
        raise AssertionError("incomplete H1 metric matrix")
    if len(predictions) != len(DATASETS) * len(SEEDS) * len(METHODS):
        raise AssertionError("incomplete H1 KIR=.50 prediction matrix")
    metrics.to_csv(RESULT_ROOT / "h1_performance_run_level.csv", index=False)
    return metrics, predictions


def merge_oos_pair(predictions: Mapping[tuple[str, int, str], pd.DataFrame], dataset: str, seed: int) -> pd.DataFrame:
    columns = [
        "sample_id",
        "gold_intent",
        "gold_is_oos",
        "predicted_is_oos",
        "oos_score",
        "distance",
        "radius",
        "nearest_cluster",
    ]
    frozen = predictions[(dataset, seed, "frozen_k1")][columns].rename(
        columns={
            "gold_intent": "gold_intent_frozen",
            "gold_is_oos": "gold_is_oos_frozen",
            "predicted_is_oos": "frozen_pred",
            "oos_score": "frozen_score",
            "distance": "frozen_distance",
            "radius": "frozen_radius",
            "nearest_cluster": "frozen_cluster",
        }
    )
    trainable = predictions[(dataset, seed, "trainable_k1")][columns].rename(
        columns={
            "gold_intent": "gold_intent_trainable",
            "gold_is_oos": "gold_is_oos_trainable",
            "predicted_is_oos": "trainable_pred",
            "oos_score": "trainable_score",
            "distance": "trainable_distance",
            "radius": "trainable_radius",
            "nearest_cluster": "trainable_cluster",
        }
    )
    merged = frozen.merge(trainable, on="sample_id", validate="one_to_one")
    if not merged["gold_intent_frozen"].equals(merged["gold_intent_trainable"]):
        raise AssertionError(f"gold intent mismatch: {dataset}/seed={seed}")
    if not merged["gold_is_oos_frozen"].equals(merged["gold_is_oos_trainable"]):
        raise AssertionError(f"gold OOS mismatch: {dataset}/seed={seed}")
    merged = merged[merged["gold_is_oos_frozen"].astype(int).eq(1)].copy()
    frozen_pred = merged["frozen_pred"].astype(int)
    trainable_pred = merged["trainable_pred"].astype(int)
    merged["transition"] = np.select(
        [
            frozen_pred.eq(1) & trainable_pred.eq(1),
            frozen_pred.eq(0) & trainable_pred.eq(1),
            frozen_pred.eq(1) & trainable_pred.eq(0),
        ],
        ["both_correct", "trainable_only", "frozen_only"],
        default="both_wrong",
    )
    merged["dataset"] = dataset
    merged["seed"] = seed
    merged["gold_intent"] = merged["gold_intent_frozen"]
    merged["delta_score"] = merged["trainable_score"] - merged["frozen_score"]
    merged["distance_contribution"] = (
        merged["trainable_distance"] / merged["frozen_radius"] - merged["frozen_score"]
    )
    merged["radius_contribution"] = (
        merged["trainable_score"] - merged["trainable_distance"] / merged["frozen_radius"]
    )
    if not np.allclose(
        merged["distance_contribution"] + merged["radius_contribution"],
        merged["delta_score"],
        atol=1e-10,
        rtol=1e-10,
    ):
        raise AssertionError(f"score decomposition mismatch: {dataset}/seed={seed}")
    return merged


def build_pairwise_tables(
    predictions: Mapping[tuple[str, int, str], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairwise = pd.concat(
        [merge_oos_pair(predictions, dataset, seed) for dataset in DATASETS for seed in SEEDS],
        ignore_index=True,
    )
    transition = (
        pairwise.groupby(["dataset", "seed", "transition"], as_index=False)
        .size()
        .rename(columns={"size": "n_oos"})
    )
    totals = transition.groupby(["dataset", "seed"])["n_oos"].transform("sum")
    transition["rate"] = transition["n_oos"] / totals
    transition.to_csv(RESULT_ROOT / "score_crossing_summary.csv", index=False)

    corrected = pairwise[pairwise["transition"].eq("trainable_only")]
    decomposition = (
        corrected.groupby(["dataset", "seed"], as_index=False)
        .agg(
            corrected_count=("sample_id", "size"),
            distance_contribution_mean=("distance_contribution", "mean"),
            radius_contribution_mean=("radius_contribution", "mean"),
            total_delta_score_mean=("delta_score", "mean"),
            total_delta_score_median=("delta_score", "median"),
        )
    )
    decomposition["closure_error"] = (
        decomposition["distance_contribution_mean"]
        + decomposition["radius_contribution_mean"]
        - decomposition["total_delta_score_mean"]
    ).abs()
    decomposition.to_csv(RESULT_ROOT / "corrected_score_decomposition.csv", index=False)

    false_accepts = pairwise[pairwise["frozen_pred"].astype(int).eq(0)].copy()
    false_accepts["margin_to_boundary"] = 1.0 - false_accepts["frozen_score"].astype(float)
    false_accepts["difficulty_band"] = pd.cut(
        false_accepts["margin_to_boundary"],
        bins=[-np.inf, 0.05, 0.10, 0.20, np.inf],
        labels=["≤0.05", "0.05–0.10", "0.10–0.20", ">0.20"],
        ordered=True,
    )
    false_accepts["outcome"] = np.where(
        false_accepts["trainable_pred"].astype(int).eq(1), "corrected", "still accepted"
    )
    difficulty = (
        false_accepts.groupby(["dataset", "seed", "difficulty_band", "outcome"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    difficulty.to_csv(RESULT_ROOT / "oos_margin_difficulty_profile.csv", index=False)

    so = pairwise[pairwise["dataset"].eq("stackoverflow")].copy()
    subtype = (
        so.groupby(["seed", "gold_intent"], as_index=False)
        .agg(
            n_oos=("sample_id", "size"),
            frozen_false_accepts=("frozen_pred", lambda values: int(np.sum(np.asarray(values) == 0))),
            trainable_false_accepts=("trainable_pred", lambda values: int(np.sum(np.asarray(values) == 0))),
        )
    )
    subtype["frozen_false_accept_rate"] = subtype["frozen_false_accepts"] / subtype["n_oos"]
    subtype["trainable_false_accept_rate"] = subtype["trainable_false_accepts"] / subtype["n_oos"]
    subtype.to_csv(RESULT_ROOT / "oos_subtype_residual_risk.csv", index=False)
    return pairwise, transition, decomposition, difficulty


def load_stackoverflow_embeddings(
    predictions: Mapping[tuple[str, int, str], pd.DataFrame], device_name: str
) -> dict[str, Any]:
    """Re-encode one fixed cell for the paper-style geometry and validation curves."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    from protocol_v2.experiments.racal_v1.representation import RacalMiniLM, choose_device, encode_rows

    dataset, seed = "stackoverflow", 42
    source = data_dir(dataset, 0.50, seed) / "gate"
    rows = {split: read_json(source / f"{split}.json") for split in ("train", "val", "test")}
    for split_rows in rows.values():
        for row in split_rows:
            row["text"] = str(row["text"])
            row["intent"] = str(row["intent"])
            row["label"] = int(row["label"])

    for method in METHODS:
        frame = predictions[(dataset, seed, method)]
        if frame["gold_intent"].astype(str).tolist() != [row["intent"] for row in rows["test"]]:
            raise AssertionError(f"test row order mismatch for {method}")
        if frame["gold_is_oos"].astype(int).tolist() != [row["label"] for row in rows["test"]]:
            raise AssertionError(f"test labels mismatch for {method}")

    device = choose_device(device_name)
    frozen_encoder = SentenceTransformer(str(MODEL_ROOT), device=str(device))
    frozen_values = {
        split: np.asarray(
            frozen_encoder.encode(
                [row["text"] for row in split_rows],
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )
        for split, split_rows in rows.items()
    }
    del frozen_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint_path = run_dir(dataset, 0.50, seed) / "trainable_k1" / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hidden_dim = int(checkpoint["model"]["projection.fc1.weight"].shape[0])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True)
    model = RacalMiniLM(MODEL_ROOT, str(checkpoint["mode"]), hidden_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    trainable_values = {
        split: encode_rows(model, tokenizer, split_rows, device, 128, 256)
        for split, split_rows in rows.items()
    }
    del model, tokenizer, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    signatures = {
        method: read_json(run_dir(dataset, 0.50, seed) / method / "detector_signature.json")
        for method in METHODS
    }
    values = {"frozen_k1": frozen_values, "trainable_k1": trainable_values}
    scored: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    verification_rows: list[dict[str, Any]] = []
    for method in METHODS:
        scored[method] = {
            split: score_from_signature(values[method][split], signatures[method])
            for split in ("val", "test")
        }
        stored = predictions[(dataset, seed, method)]
        max_delta = float(np.max(np.abs(scored[method]["test"]["score"] - stored["oos_score"].to_numpy(float))))
        decision_mismatch = int(
            np.sum(scored[method]["test"]["pred"] != stored["predicted_is_oos"].to_numpy(int))
        )
        cluster_mismatch = int(
            np.sum(scored[method]["test"]["nearest_cluster"] != stored["nearest_cluster"].to_numpy(int))
        )
        verification_rows.append(
            {
                "method": method,
                "test_rows": len(stored),
                "score_max_abs_delta": max_delta,
                "decision_mismatch_count": decision_mismatch,
                "nearest_cluster_mismatch_count": cluster_mismatch,
            }
        )
        if decision_mismatch or cluster_mismatch or max_delta > 1e-4:
            raise AssertionError(f"re-encoding does not reproduce stored {method} scores")
    verification = pd.DataFrame(verification_rows)
    verification.to_csv(RESULT_ROOT / "stack_overflow_reencoding_verification.csv", index=False)
    return {
        "rows": rows,
        "values": values,
        "scores": scored,
        "signatures": signatures,
        "verification": verification,
    }


def signature_sphere(signature: Mapping[str, Any], *, cluster: int | None = None, intent: str | None = None) -> Mapping[str, Any]:
    matches = [
        sphere
        for sphere in signature["spheres"]
        if (cluster is None or int(sphere["cluster_id"]) == cluster)
        and (intent is None or str(sphere["intent"]) == intent)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one detector sphere, found {len(matches)}")
    return matches[0]


def boundary_cross_section(pca: Any, sphere: Mapping[str, Any]) -> np.ndarray:
    center = np.asarray(sphere["center"], dtype=np.float64)
    inv_cov = np.asarray(sphere["inv_diag_cov"], dtype=np.float64)
    radius = float(sphere["radius"])
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    directions = np.cos(angles)[:, None] * pca.components_[0] + np.sin(angles)[:, None] * pca.components_[1]
    scales = radius / np.sqrt(np.sum(directions * directions * inv_cov[None, :], axis=1))
    return pca.transform(center[None, :] + scales[:, None] * directions)


def plot_local_boundary_geometry(
    predictions: Mapping[tuple[str, int, str], pd.DataFrame], encoded: Mapping[str, Any]
) -> dict[str, Any]:
    from sklearn.decomposition import PCA

    frozen = predictions[("stackoverflow", 42, "frozen_k1")]
    trainable = predictions[("stackoverflow", 42, "trainable_k1")]
    false_accept = frozen[
        frozen["gold_is_oos"].astype(int).eq(1) & frozen["predicted_is_oos"].astype(int).eq(0)
    ]
    counts = false_accept["nearest_cluster"].astype(int).value_counts()
    maximum = int(counts.max())
    selected_cluster = int(sorted(int(value) for value in counts[counts.eq(maximum)].index)[0])
    frozen_sphere = signature_sphere(encoded["signatures"]["frozen_k1"], cluster=selected_cluster)
    selected_intent = str(frozen_sphere["intent"])
    trainable_sphere = signature_sphere(encoded["signatures"]["trainable_k1"], intent=selected_intent)

    rows = encoded["rows"]
    train_mask = np.asarray([row["intent"] == selected_intent for row in rows["train"]], dtype=bool)
    boundary_low, boundary_high = 0.90, 1.10
    val_unknown = np.asarray([row["label"] == 1 for row in rows["val"]], dtype=bool)
    frozen_val_score = encoded["scores"]["frozen_k1"]["val"]["score"]
    val_mask = (
        val_unknown
        & (encoded["scores"]["frozen_k1"]["val"]["nearest_cluster"].astype(int) == selected_cluster)
        & (frozen_val_score >= boundary_low)
        & (frozen_val_score <= boundary_high)
    )
    test_oos = frozen["gold_is_oos"].to_numpy(int).astype(bool)
    frozen_test_score = frozen["oos_score"].to_numpy(float)
    test_mask = (
        test_oos
        & (frozen["nearest_cluster"].to_numpy(int) == selected_cluster)
        & (frozen_test_score >= boundary_low)
        & (frozen_test_score <= boundary_high)
    )
    corrected_mask = (
        test_mask
        & (frozen["predicted_is_oos"].to_numpy(int) == 0)
        & (trainable["predicted_is_oos"].to_numpy(int) == 1)
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.19, top=0.70, wspace=0.23)
    for column, method in enumerate(METHODS):
        ax = axes[column]
        sphere = frozen_sphere if method == "frozen_k1" else trainable_sphere
        values = encoded["values"][method]
        known_values = l2_normalize(values["train"])[train_mask]
        test_oos_values = l2_normalize(values["test"])[test_mask]
        center = np.asarray(sphere["center"], dtype=np.float64)[None, :]
        pca = PCA(n_components=2, random_state=0)
        pca.fit(np.vstack([known_values, test_oos_values, center]))
        known_xy = pca.transform(known_values)
        oos_xy = pca.transform(test_oos_values)
        center_xy = pca.transform(center)
        boundary_xy = boundary_cross_section(pca, sphere)

        ax.scatter(known_xy[:, 0], known_xy[:, 1], s=3, color=BLUE, alpha=0.20, linewidths=0, rasterized=True)
        ax.scatter(oos_xy[:, 0], oos_xy[:, 1], s=4, color=RED, alpha=0.24, marker="x", linewidths=0.35, rasterized=True)
        ax.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="#7C838A", linestyle="--", linewidth=0.58, alpha=0.75)
        ax.scatter(center_xy[:, 0], center_xy[:, 1], marker="*", s=58, color=GOLD, edgecolor=DARK, linewidth=0.45, zorder=5)
        ax.set_title(f"{chr(ord('a') + column)}   {METHOD_LABELS[method]}", loc="left", fontweight="bold", fontsize=7.8, pad=4)
        ax.set_xlabel("local PC1")
        ax.set_ylabel("local PC2")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(False)
        style_axis(ax)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=BLUE, markersize=4, label="Known"),
        Line2D([], [], marker="x", linestyle="none", color=RED, markersize=4, label="OOS"),
        Line2D([], [], color="#7C838A", linestyle="--", linewidth=0.8, label="score=1 boundary"),
        Line2D([], [], marker="*", linestyle="none", markerfacecolor=GOLD, markeredgecolor=DARK, markersize=8, label="Known center"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.86), ncol=4, handlelength=1.4, columnspacing=0.8, frameon=False)
    fig.text(0.07, 0.965, "Local OOS boundary", fontsize=7.9, fontweight="bold", va="top")
    fig.text(
        0.07,
        0.035,
        f"StackOverflow, KIR=.50, seed=42; {selected_intent}; Frozen score band [{boundary_low:.2f}, {boundary_high:.2f}]. Known/OOS only; PCA is descriptive.",
        fontsize=5.8,
        color="#555B61",
    )
    save_figure(fig, "paper_style_local_boundary_geometry")

    summary = {
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seed": 42,
        "selection_rule": "Frozen OOS false-accept nearest cluster with largest count; smallest cluster id breaks ties; display uses fixed Frozen score band [0.90,1.10]",
        "selected_frozen_cluster": selected_cluster,
        "selected_known_intent": selected_intent,
        "frozen_score_band_low": boundary_low,
        "frozen_score_band_high": boundary_high,
        "frozen_false_accepts_at_selected_cluster": maximum,
        "known_train_points": int(train_mask.sum()),
        "validation_unknown_points_not_plotted": int(val_mask.sum()),
        "test_oos_points": int(test_mask.sum()),
        "corrected_oos_points": int(corrected_mask.sum()),
        "projection_scope": "separate local PCA per method; descriptive only",
    }
    pd.DataFrame([summary]).to_csv(RESULT_ROOT / "local_boundary_geometry_summary.csv", index=False)
    return summary


def plot_local_boundary_geometry_3d(
    predictions: Mapping[tuple[str, int, str], pd.DataFrame], encoded: Mapping[str, Any]
) -> dict[str, Any]:
    """Show the selected local boundary in 3D with true high-dimensional outcomes.

    The point cloud uses a common PCA fitted on both per-method locally
    whitened representations.  The translucent surface is a unit sphere in
    that normalized local coordinate system; accepted OOS are drawn as open
    circles and rejected OOS as faint red crosses so the projection cannot be
    read as the decision itself.
    """
    from sklearn.decomposition import PCA
    from matplotlib import font_manager

    font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

    frozen = predictions[("stackoverflow", 42, "frozen_k1")]
    trainable = predictions[("stackoverflow", 42, "trainable_k1")]
    false_accept = frozen[
        frozen["gold_is_oos"].astype(int).eq(1) & frozen["predicted_is_oos"].astype(int).eq(0)
    ]
    counts = false_accept["nearest_cluster"].astype(int).value_counts()
    maximum = int(counts.max())
    selected_cluster = int(sorted(int(value) for value in counts[counts.eq(maximum)].index)[0])
    frozen_sphere = signature_sphere(encoded["signatures"]["frozen_k1"], cluster=selected_cluster)
    selected_intent = str(frozen_sphere["intent"])
    trainable_sphere = signature_sphere(encoded["signatures"]["trainable_k1"], intent=selected_intent)

    rows = encoded["rows"]
    train_mask = np.asarray([row["intent"] == selected_intent for row in rows["train"]], dtype=bool)
    boundary_low, boundary_high = 0.90, 1.10
    test_oos = frozen["gold_is_oos"].to_numpy(int).astype(bool)
    frozen_test_score = frozen["oos_score"].to_numpy(float)
    test_mask = (
        test_oos
        & (frozen["nearest_cluster"].to_numpy(int) == selected_cluster)
        & (frozen_test_score >= boundary_low)
        & (frozen_test_score <= boundary_high)
    )

    values = {
        method: np.asarray(encoded["values"][method]["test"], dtype=np.float64)
        for method in METHODS
    }
    train_values_raw = {
        method: l2_normalize(encoded["values"][method]["train"])[train_mask]
        for method in METHODS
    }
    test_values_raw = {
        method: l2_normalize(values[method])[test_mask]
        for method in METHODS
    }
    centers_raw = {
        "frozen_k1": np.asarray(frozen_sphere["center"], dtype=np.float64),
        "trainable_k1": np.asarray(trainable_sphere["center"], dtype=np.float64),
    }

    def local_whiten(values: np.ndarray, sphere: Mapping[str, Any]) -> np.ndarray:
        center = np.asarray(sphere["center"], dtype=np.float64)
        inv_diag_cov = np.asarray(sphere["inv_diag_cov"], dtype=np.float64)
        radius = float(sphere["radius"])
        if not np.all(np.isfinite(inv_diag_cov)) or np.any(inv_diag_cov <= 0.0):
            raise ValueError("unit-sphere visualization requires positive finite inverse diagonal covariance")
        return (values - center) * np.sqrt(inv_diag_cov)[None, :] / radius

    train_values = {
        "frozen_k1": local_whiten(train_values_raw["frozen_k1"], frozen_sphere),
        "trainable_k1": local_whiten(train_values_raw["trainable_k1"], trainable_sphere),
    }
    test_values = {
        "frozen_k1": local_whiten(test_values_raw["frozen_k1"], frozen_sphere),
        "trainable_k1": local_whiten(test_values_raw["trainable_k1"], trainable_sphere),
    }
    centers = {method: np.zeros_like(center) for method, center in centers_raw.items()}
    pca = PCA(n_components=3, random_state=0)
    pca.fit(
        np.vstack(
            [
                train_values["frozen_k1"],
                train_values["trainable_k1"],
                test_values["frozen_k1"],
                test_values["trainable_k1"],
                centers["frozen_k1"][None, :],
                centers["trainable_k1"][None, :],
            ]
        )
    )

    def unit_surface() -> np.ndarray:
        theta = np.linspace(0.0, 2.0 * np.pi, 72)
        phi = np.linspace(0.0, np.pi, 36)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        directions = (
            np.sin(phi_grid)[..., None] * np.cos(theta_grid)[..., None] * pca.components_[0]
            + np.sin(phi_grid)[..., None] * np.sin(theta_grid)[..., None] * pca.components_[1]
            + np.cos(phi_grid)[..., None] * pca.components_[2]
        )
        return pca.transform(directions.reshape(-1, directions.shape[-1])).reshape(
            directions.shape[0], directions.shape[1], 3
        )

    projected: dict[str, np.ndarray] = {
        method: pca.transform(test_values[method]) for method in METHODS
    }
    projected_train: dict[str, np.ndarray] = {
        method: pca.transform(train_values[method]) for method in METHODS
    }
    projected_centers = {
        method: pca.transform(centers[method][None, :])[0]
        for method in METHODS
    }
    projected_surfaces = {method: unit_surface() for method in METHODS}
    # Preserve the selected center's full-dimensional score as radial distance.
    local_scores = {}
    local_train_scores = {}
    for method in METHODS:
        origin = projected_centers[method].copy()
        local_scores[method] = np.linalg.norm(test_values[method], axis=1)
        local_train_scores[method] = np.linalg.norm(train_values[method], axis=1)
        for points, full_values in ((projected[method], test_values[method]),
                                    (projected_train[method], train_values[method])):
            directions = points - origin
            lengths = np.linalg.norm(directions, axis=1)
            if np.any(lengths < 1e-12):
                raise ValueError("Undefined projected direction for radial visualization")
            points[:] = directions / lengths[:, None] * np.linalg.norm(full_values, axis=1)[:, None]
        projected_surfaces[method] -= origin
        projected_centers[method][:] = 0
        np.testing.assert_allclose(np.linalg.norm(projected[method], axis=1), local_scores[method], atol=1e-10)
    display_radius = 1.16

    def display_points(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(points, axis=1)
        directions = points / np.maximum(lengths[:, None], 1e-12)
        return directions * np.minimum(radii, display_radius)[:, None]

    displayed = {method: display_points(projected[method], local_scores[method]) for method in METHODS}
    displayed_train = {method: display_points(projected_train[method], local_train_scores[method]) for method in METHODS}
    all_points = np.vstack(
        [
            displayed_train["frozen_k1"],
            displayed_train["trainable_k1"],
            displayed["frozen_k1"],
            displayed["trainable_k1"],
            *[value.reshape(-1, 3) for value in projected_surfaces.values()],
        ]
    )
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    margin = np.maximum((upper - lower) * 0.06, 0.05)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.6, 5.50),
        subplot_kw={"projection": "3d", "computed_zorder": False},
    )
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.28, top=0.99, wspace=0.01)
    decisions: dict[str, np.ndarray] = {
        "frozen_k1": encoded["scores"]["frozen_k1"]["test"]["pred"][test_mask].astype(bool),
        "trainable_k1": encoded["scores"]["trainable_k1"]["test"]["pred"][test_mask].astype(bool),
    }
    for column, method in enumerate(METHODS):
        ax = axes[column]
        train_xy = displayed_train[method]
        oos_xy = displayed[method]
        # Fig.1 explains the selected local sphere only.  Do not overlay the
        # full Gate union decision here: other centers can accept an OOS sample
        # even when it lies outside this displayed sphere.
        local_inside = local_scores[method] <= 1.0
        local_outside = ~local_inside
        ax.scatter(train_xy[:, 0], train_xy[:, 1], train_xy[:, 2], s=7, color="#4477AA", alpha=0.30, linewidths=0, depthshade=False, zorder=3, rasterized=True)
        ax.scatter(oos_xy[local_outside, 0], oos_xy[local_outside, 1], oos_xy[local_outside, 2], s=5, color="#CC6677", alpha=0.10, marker="o", linewidths=0, depthshade=False, zorder=2, rasterized=True)
        ax.scatter(oos_xy[local_inside, 0], oos_xy[local_inside, 1], oos_xy[local_inside, 2], s=18, facecolors="none", edgecolors="#CC6677", alpha=0.85, marker="o", linewidths=0.8, depthshade=False, zorder=4)
        boundary = projected_surfaces[method]
        ax.plot_surface(boundary[:, :, 0], boundary[:, :, 1], boundary[:, :, 2], color="#9ABAD0", alpha=0.055, linewidth=0, shade=False, zorder=1)
        ax.plot_wireframe(boundary[:, :, 0], boundary[:, :, 1], boundary[:, :, 2], rstride=9, cstride=12, color="#9AB0C0", alpha=0.35, linewidth=0.45, zorder=1)
        center = projected_centers[method]
        ax.scatter([center[0]], [center[1]], [center[2]], marker="*", s=100, color="#334155", edgecolor="white", linewidth=0.9, depthshade=False, zorder=10)
        ax.set_xlim(-display_radius, display_radius)
        ax.set_ylim(-display_radius, display_radius)
        ax.set_zlim(-display_radius, display_radius)
        ax.set_box_aspect((1, 1, 1), zoom=1.55)
        ax.set_proj_type("ortho")
        # View the point cloud obliquely from its side, rather than through the sphere.
        mean_direction = np.mean(np.vstack(list(displayed.values())), axis=0)
        azimuth = np.degrees(np.arctan2(mean_direction[1], mean_direction[0])) + 55
        ax.view_init(elev=20, azim=azimuth)
        ax.set_xlabel(r"$u_1$", labelpad=-1, fontsize=22)
        ax.set_ylabel(r"$u_2$", labelpad=-1, fontsize=22)
        ax.set_zlabel("")
        ax.text2D(0.93, 0.56, r"$u_3$", transform=ax.transAxes, fontsize=22, color="#64748B")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticks([-1, 0, 1])
            axis.line.set_color("#9BA7B4")
        for axis_name in ("x", "y", "z"):
            ax.tick_params(axis=axis_name, which="major", pad=0, labelsize=20, colors="#64748B")
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    handles = [
        Line2D([], [], marker="o", linestyle="none", color="#4477AA", markersize=5, label="Known"),
        Line2D([], [], marker="o", linestyle="none", color="#CC6677", markersize=5, label="OOS outside"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="#CC6677", markersize=6, label="OOS inside"),
        Line2D([], [], marker="*", linestyle="none", color="#334155", markersize=8, label="Center"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        fontsize=20,
        handlelength=1.2,
        columnspacing=1.6,
        frameon=False,
    )
    save_figure(fig, "paper_style_local_boundary_geometry_3d")

    summary = {
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seed": 42,
        "selected_frozen_cluster": selected_cluster,
        "selected_known_intent": selected_intent,
        "oos_cohort_count": int(test_mask.sum()),
        "frozen_true_384d_accept_count": int((~decisions["frozen_k1"]).sum()),
        "frozen_true_384d_reject_count": int(decisions["frozen_k1"].sum()),
        "trainable_true_384d_accept_count": int((~decisions["trainable_k1"]).sum()),
        "trainable_true_384d_reject_count": int(decisions["trainable_k1"].sum()),
        "trainable_repairs_from_frozen_false_accept": int(
            ((~decisions["frozen_k1"]) & decisions["trainable_k1"]).sum()
        ),
        "projection_scope": "radial schematic: PCA direction and exact 384-D local normalized distance; sphere inclusion equals selected-center inclusion, not necessarily full Gate decision",
        "frozen_local_inside": int((local_scores['frozen_k1'] <= 1).sum()),
        "trainable_local_inside": int((local_scores['trainable_k1'] <= 1).sum()),
        "coordinate_transform": "u=(z-center)*sqrt(inv_diag_cov)/radius; each method has its own local coordinate system",
        "display_transform": "radial display clips normalized distances above 1.28 for readability while preserving inside/outside status and direction",
    }
    pd.DataFrame([summary]).to_csv(RESULT_ROOT / "local_boundary_geometry_3d_summary.csv", index=False)
    return summary


def plot_oos_representation_movement(
    predictions: Mapping[tuple[str, int, str], pd.DataFrame], encoded: Mapping[str, Any]
) -> None:
    """Show the representation movement of OOS samples whose decision changed."""
    from matplotlib.collections import LineCollection
    from sklearn.decomposition import PCA

    frozen = predictions[("stackoverflow", 42, "frozen_k1")]
    trainable = predictions[("stackoverflow", 42, "trainable_k1")]
    oos = frozen["gold_is_oos"].to_numpy(int) == 1
    changed = oos & (
        frozen["predicted_is_oos"].to_numpy(int) != trainable["predicted_is_oos"].to_numpy(int)
    )
    frozen_values = l2_normalize(encoded["values"]["frozen_k1"]["test"])[changed]
    trainable_values = l2_normalize(encoded["values"]["trainable_k1"]["test"])[changed]
    if len(frozen_values) == 0:
        raise ValueError("no decision-changing OOS samples available for movement map")
    pca = PCA(n_components=2, random_state=0)
    pca.fit(np.vstack([frozen_values, trainable_values]))
    frozen_xy = pca.transform(frozen_values)
    trainable_xy = pca.transform(trainable_values)
    corrected = (
        frozen["predicted_is_oos"].to_numpy(int)[changed] == 0
    ) & (trainable["predicted_is_oos"].to_numpy(int)[changed] == 1)
    regressed = (
        frozen["predicted_is_oos"].to_numpy(int)[changed] == 1
    ) & (trainable["predicted_is_oos"].to_numpy(int)[changed] == 0)

    segments = np.stack([frozen_xy, trainable_xy], axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.72)
    corrected_lines = LineCollection(segments[corrected], colors=RED, linewidths=0.55, alpha=0.22, zorder=1)
    regressed_lines = LineCollection(segments[regressed], colors=BLUE, linewidths=0.65, alpha=0.35, zorder=2)
    corrected_lines.set_rasterized(True)
    regressed_lines.set_rasterized(True)
    ax.add_collection(corrected_lines)
    ax.add_collection(regressed_lines)
    ax.scatter(frozen_xy[:, 0], frozen_xy[:, 1], s=3, color=BLUE, alpha=0.20, marker="x", linewidths=0.35, rasterized=True)
    ax.scatter(trainable_xy[:, 0], trainable_xy[:, 1], s=3, color=RED, alpha=0.20, marker="o", linewidths=0, rasterized=True)
    ax.scatter(trainable_xy[corrected, 0], trainable_xy[corrected, 1], s=7, color=RED, edgecolors=DARK, linewidths=0.20, alpha=0.52, rasterized=True)
    ax.scatter(trainable_xy[regressed, 0], trainable_xy[regressed, 1], s=7, facecolors="none", edgecolors=BLUE, linewidths=0.35, alpha=0.70, rasterized=True)
    all_xy = np.vstack([frozen_xy, trainable_xy])
    x_margin = max((float(all_xy[:, 0].max()) - float(all_xy[:, 0].min())) * 0.06, 0.08)
    y_margin = max((float(all_xy[:, 1].max()) - float(all_xy[:, 1].min())) * 0.06, 0.08)
    ax.set_xlim(float(all_xy[:, 0].min()) - x_margin, float(all_xy[:, 0].max()) + x_margin)
    ax.set_ylim(float(all_xy[:, 1].min()) - y_margin, float(all_xy[:, 1].max()) + y_margin)
    ax.set_xlabel("common PC1")
    ax.set_ylabel("common PC2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    style_axis(ax)
    handles = [
        Line2D([], [], marker="x", linestyle="none", color=BLUE, markersize=4, label="Frozen position"),
        Line2D([], [], marker="o", linestyle="none", color=RED, markersize=4, label="Trainable position"),
        Line2D([], [], marker="o", markerfacecolor=RED, markeredgecolor=DARK, linestyle="none", color=RED, markersize=4, label="OOS repaired"),
        Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=BLUE, linestyle="none", color=BLUE, markersize=4, label="OOS regressed"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.58, 0.87), ncol=4, handlelength=1.1, columnspacing=0.8, frameon=False)
    fig.text(0.08, 0.965, "OOS representation movement", fontsize=7.9, fontweight="bold", va="top")
    fig.text(0.08, 0.04, "StackOverflow, KIR=.50, seed=42; only decision-changing OOS. Common PCA is descriptive.", fontsize=5.9, color="#555B61")
    save_figure(fig, "oos_representation_movement_map")
    pd.DataFrame(
        [
            {
                "dataset": "stackoverflow",
                "kir": 0.50,
                "seed": 42,
                "oos_decision_changed": int(changed.sum()),
                "corrected_oos": int(corrected.sum()),
                "regressed_oos": int(regressed.sum()),
                "projection_scope": "common PCA fitted on Frozen and Trainable endpoints; descriptive only",
            }
        ]
    ).to_csv(RESULT_ROOT / "oos_representation_movement_summary.csv", index=False)


def plot_paper_score_distribution(
    predictions: Mapping[tuple[str, int, str], pd.DataFrame], encoded: Mapping[str, Any]
) -> pd.DataFrame:
    populations: dict[tuple[str, str], np.ndarray] = {}
    for method in METHODS:
        test = predictions[("stackoverflow", 42, method)]
        populations[(method, "Known")] = test.loc[test["gold_is_oos"].astype(int).eq(0), "oos_score"].to_numpy(float)
        populations[(method, "OOS")] = test.loc[test["gold_is_oos"].astype(int).eq(1), "oos_score"].to_numpy(float)
    all_values = np.concatenate(list(populations.values()))
    display_max = max(1.45, float(np.quantile(all_values, 0.995) * 1.03))
    bins = np.linspace(0.0, display_max, 72)
    population_colors = {"Known": BLUE, "OOS": RED}
    rows: list[dict[str, Any]] = []

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.40), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.24, top=0.96, wspace=0.16)
    for column, method in enumerate(METHODS):
        ax = axes[column]
        for population in ("Known", "OOS"):
            centers, density, overflow = smoothed_histogram(populations[(method, population)], bins)
            ax.plot(centers, density, color=population_colors[population], linewidth=1.45, label=population)
            if population == "OOS":
                accept = centers <= 1.0
                ax.fill_between(centers[accept], 0, density[accept], color=RED, alpha=0.14, linewidth=0)
            for x, y in zip(centers, density, strict=True):
                rows.append(
                    {
                        "dataset": "stackoverflow",
                        "kir": 0.50,
                        "seed": 42,
                        "method": method,
                        "population": population,
                        "bin_center": float(x),
                        "density": float(y),
                        "n": len(populations[(method, population)]),
                        "display_max": display_max,
                        "overflow_in_last_bin": overflow,
                    }
                )
        ax.axvline(1.0, color=DARK, linestyle="--", linewidth=0.9)
        ax.set_xlabel("Gate score (normalized)", fontsize=19, labelpad=2)
        ax.set_xlim(0.0, display_max)
        style_axis(ax)
        ax.tick_params(axis="both", which="major", labelsize=18, pad=2)
        ax.grid(False)
    axes[0].set_ylabel("density", fontsize=19, labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        fontsize=17,
        handlelength=1.4,
        columnspacing=1.0,
        frameon=False,
    )
    save_figure(fig, "paper_style_score_distribution")
    density_frame = pd.DataFrame(rows)
    density_frame.to_csv(RESULT_ROOT / "score_distribution_summary.csv", index=False)
    return density_frame


def plot_score_crossing(pairwise: pd.DataFrame) -> None:
    global_values = np.concatenate([pairwise["frozen_score"].to_numpy(float), pairwise["trainable_score"].to_numpy(float)])
    high = max(1.45, float(np.quantile(global_values, 0.995) * 1.03))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.68, wspace=0.29)
    for column, dataset in enumerate(DATASETS):
        ax = axes[column]
        part = pairwise[pairwise["dataset"].eq(dataset)]
        x = np.clip(part["frozen_score"].to_numpy(float), 0, np.nextafter(high, -np.inf))
        y = np.clip(part["trainable_score"].to_numpy(float), 0, np.nextafter(high, -np.inf))
        ax.hexbin(x, y, gridsize=38, extent=(0, high, 0, high), mincnt=1, cmap="Greys", linewidths=0, alpha=0.30, rasterized=True)
        corrected = part["transition"].eq("trainable_only").to_numpy()
        regressed = part["transition"].eq("frozen_only").to_numpy()
        ax.scatter(x[corrected], y[corrected], s=4, color=RED, alpha=0.45, linewidths=0, rasterized=True)
        ax.scatter(x[regressed], y[regressed], s=6, facecolors="none", edgecolors=BLUE, alpha=0.65, linewidths=0.45, rasterized=True)
        ax.axvline(1.0, color=DARK, linestyle="--", linewidth=0.75)
        ax.axhline(1.0, color=DARK, linestyle="--", linewidth=0.75)
        ax.plot([0, high], [0, high], color="#AEB3B8", linestyle=":", linewidth=0.7)
        corrected_rate = float(np.mean(corrected)) * 100
        regressed_rate = float(np.mean(regressed)) * 100
        ax.text(0.04, 0.96, f"corrected {corrected_rate:.1f}%\nregressed {regressed_rate:.1f}%", transform=ax.transAxes, va="top", fontsize=6.1)
        ax.set_title(f"{chr(ord('a') + column)}   {DATASET_LABELS[dataset]}", loc="left", fontweight="bold", pad=4)
        ax.set_xlabel("Frozen score")
        if column == 0:
            ax.set_ylabel("Trainable score")
        ax.set_xlim(0, high)
        ax.set_ylim(0, high)
        ax.set_aspect("equal", adjustable="box")
        style_axis(ax)
        ax.grid(False)
    handles = [
        Patch(facecolor="#A0A0A0", alpha=0.45, label="all OOS density"),
        Line2D([], [], marker="o", linestyle="none", color=RED, markersize=4, label="corrected"),
        Line2D([], [], marker="o", markerfacecolor="none", linestyle="none", color=BLUE, markersize=4, label="regressed"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.58, 0.84), ncol=3, handlelength=1.3, columnspacing=0.9, frameon=False)
    fig.text(0.075, 0.965, "Paired OOS score crossings", fontsize=8.2, fontweight="bold", va="top")
    fig.text(0.075, 0.04, "KIR=.50; three seeds pooled as run-observations. Shared axes; the upper 0.5% tail remains at the edge.", fontsize=6.0, color="#555B61")
    save_figure(fig, "paired_oos_score_crossing")


def plot_corrected_decomposition(decomposition: pd.DataFrame) -> None:
    summary = (
        decomposition.groupby("dataset", as_index=False)
        .agg(
            corrected_count=("corrected_count", "sum"),
            distance=("distance_contribution_mean", "mean"),
            radius=("radius_contribution_mean", "mean"),
            total=("total_delta_score_mean", "mean"),
        )
        .set_index("dataset")
        .loc[list(DATASETS)]
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.20, right=0.985, bottom=0.23, top=0.69)
    y = np.arange(len(DATASETS))
    axis_high = 0.0
    for index, dataset in enumerate(DATASETS):
        row = summary.loc[dataset]
        distance = float(row["distance"])
        total = float(row["total"])
        radius = float(row["radius"])
        axis_high = max(axis_high, distance, total)
        ax.annotate(
            "",
            xy=(distance, index),
            xytext=(0.0, index),
            arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 4.5, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 8},
        )
        ax.annotate(
            "",
            xy=(total, index),
            xytext=(distance, index),
            arrowprops={"arrowstyle": "-|>", "color": GOLD, "lw": 4.5, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 8},
        )
        seed_rows = decomposition[decomposition["dataset"].eq(dataset)]
        jitter = np.linspace(0.16, 0.28, len(seed_rows))
        ax.scatter(seed_rows["total_delta_score_mean"], index + jitter, s=15, facecolors="white", edgecolors=DARK, linewidths=0.60, zorder=4)
        ax.scatter([total], [index], marker="D", s=28, color=RED, edgecolor=DARK, linewidth=0.45, zorder=5)
        ax.text(
            0.0,
            index - 0.27,
            f"distance {distance:+.3f}   radius {radius:+.3f}   total {total:+.3f}   n={int(row['corrected_count']):,}",
            ha="left",
            va="bottom",
            fontsize=5.8,
            color="#555B61",
        )
    ax.axvline(0, color=DARK, linewidth=0.75)
    ax.set_yticks(y, [DATASET_LABELS[dataset] for dataset in DATASETS])
    ax.invert_yaxis()
    ax.set_ylim(len(DATASETS) - 0.55, -0.55)
    ax.set_xlim(-0.02, axis_high * 1.12)
    ax.set_xlabel("mean contribution to Trainable − Frozen score")
    style_axis(ax, "x")
    handles = [
        Line2D([], [], color=BLUE, linewidth=3.2, label="distance step"),
        Line2D([], [], color=GOLD, linewidth=3.2, label="radius step"),
        Line2D([], [], marker="D", linestyle="none", color=RED, markeredgecolor=DARK, markersize=5, label="total mean"),
        Line2D([], [], marker="o", markerfacecolor="white", linestyle="none", color=DARK, markersize=4, label="seed mean"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.59, 0.84), ncol=4, handlelength=1.2, columnspacing=0.9, frameon=False)
    fig.text(0.20, 0.965, "Exact score decomposition for corrected OOS", fontsize=8.2, fontweight="bold", va="top")
    fig.text(0.20, 0.04, "Arrows follow signed addition: Δscore = distance step + radius step. Only Frozen-accepted / Trainable-rejected OOS are shown.", fontsize=6.0, color="#555B61")
    save_figure(fig, "corrected_oos_score_decomposition")


def plot_cross_dataset_density(
    predictions: Mapping[tuple[str, int, str], pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.68, wspace=0.25)
    for column, dataset in enumerate(DATASETS):
        ax = axes[column]
        all_values = np.concatenate(
            [
                predictions[(dataset, seed, method)].loc[
                    predictions[(dataset, seed, method)]["gold_is_oos"].astype(int).eq(1), "oos_score"
                ].to_numpy(float)
                for method in METHODS
                for seed in SEEDS
            ]
        )
        high = max(1.45, float(np.quantile(all_values, 0.995) * 1.03))
        bins = np.linspace(0.0, high, 68)
        for method in METHODS:
            seed_curves = []
            for seed in SEEDS:
                values = predictions[(dataset, seed, method)].loc[
                    predictions[(dataset, seed, method)]["gold_is_oos"].astype(int).eq(1), "oos_score"
                ].to_numpy(float)
                centers, density, overflow = smoothed_histogram(values, bins)
                seed_curves.append(density)
                for x, y in zip(centers, density, strict=True):
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "bin_center": float(x),
                            "density": float(y),
                            "n_oos": len(values),
                            "display_max": high,
                            "overflow_in_last_bin": overflow,
                        }
                    )
            mean_curve = np.mean(seed_curves, axis=0)
            ax.plot(centers, mean_curve, color=METHOD_COLORS[method], linewidth=1.45, label=METHOD_LABELS[method])
        ax.axvspan(0, 1.0, color=RED, alpha=0.055, linewidth=0)
        ax.axvline(1.0, color=DARK, linestyle="--", linewidth=0.8)
        ax.set_title(f"{chr(ord('a') + column)}   {DATASET_LABELS[dataset]}", loc="left", fontweight="bold", pad=4)
        ax.set_xlabel("normalized OOS score")
        if column == 0:
            ax.set_ylabel("OOS density")
        ax.set_xlim(0, high)
        style_axis(ax)
        ax.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.58, 0.83), ncol=2, handlelength=1.7, columnspacing=1.1, frameon=False)
    fig.text(0.075, 0.965, "OOS score density across datasets", fontsize=8.2, fontweight="bold", va="top")
    fig.text(0.075, 0.04, "KIR=.50; three-seed mean densities. Shading marks score≤1; display tails remain in the final bin.", fontsize=6.0, color="#555B61")
    save_figure(fig, "cross_dataset_oos_density")
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULT_ROOT / "cross_dataset_oos_density.csv", index=False)
    return frame


def plot_margin_difficulty(difficulty: pd.DataFrame) -> None:
    bands = ["≤0.05", "0.05–0.10", "0.10–0.20", ">0.20"]
    pooled = difficulty.groupby(["dataset", "difficulty_band", "outcome"], observed=False)["count"].sum()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.25, top=0.66, wspace=0.19)
    for column, dataset in enumerate(DATASETS):
        ax = axes[column]
        corrected = np.asarray([float(pooled.get((dataset, band, "corrected"), 0)) for band in bands])
        remaining = np.asarray([float(pooled.get((dataset, band, "still accepted"), 0)) for band in bands])
        total = corrected + remaining
        corrected_rate = np.divide(corrected, total, out=np.zeros_like(corrected), where=total > 0) * 100
        remaining_rate = 100 - corrected_rate
        x = np.arange(len(bands))
        ax.bar(x, corrected_rate, width=0.68, color=RED, label="corrected")
        ax.bar(x, remaining_rate, bottom=corrected_rate, width=0.68, color=LIGHT_GRAY, label="still accepted")
        for i, (rate, n) in enumerate(zip(corrected_rate, total, strict=True)):
            if n > 0:
                ax.text(i, max(min(rate / 2, 47), 3), f"{rate:.0f}%", ha="center", va="center", fontsize=6.0, color="white" if rate > 18 else DARK)
                ax.text(i, 102, f"n={int(n):,}", ha="center", va="bottom", fontsize=5.6, color="#555B61")
        ax.set_title(f"{chr(ord('a') + column)}   {DATASET_LABELS[dataset]}", loc="left", fontweight="bold", pad=4)
        ax.set_xticks(x, bands, rotation=25, ha="right")
        ax.set_xlabel("Frozen margin below 1")
        ax.set_ylim(0, 113)
        if column == 0:
            ax.set_ylabel("share of Frozen false accepts (%)")
        style_axis(ax, "y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.60, 0.81), ncol=2, handlelength=1.2, frameon=False)
    fig.text(0.075, 0.965, "Which Frozen false accepts are repaired?", fontsize=8.2, fontweight="bold", va="top")
    fig.text(0.075, 0.035, "KIR=.50; three seeds pooled as run-observations. Fixed bands include every Frozen false accept.", fontsize=6.0, color="#555B61")
    save_figure(fig, "oos_margin_difficulty_profile")


def build_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index=["dataset", "kir", "seed"], columns="method", values="oos_f1").reset_index()
    pivot["delta_oos_f1_pp"] = (pivot["trainable_k1"] - pivot["frozen_k1"]) * 100.0
    pivot["positive"] = pivot["delta_oos_f1_pp"] > 0
    pivot.to_csv(RESULT_ROOT / "paired_kir_seed_stability.csv", index=False)
    return pivot


def plot_stability(stability: pd.DataFrame) -> None:
    order = [(dataset, kir) for dataset in DATASETS for kir in KIRS]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.subplots_adjust(left=0.23, right=0.985, bottom=0.16, top=0.82)
    for position, (dataset, kir) in enumerate(order):
        part = stability[stability["dataset"].eq(dataset) & stability["kir"].eq(kir)].sort_values("seed")
        jitter = np.linspace(-0.12, 0.12, len(part))
        colors = np.where(part["delta_oos_f1_pp"].to_numpy() > 0, GRAY, BLUE)
        ax.scatter(part["delta_oos_f1_pp"], position + jitter, s=22, c=colors, edgecolor="white", linewidth=0.35, zorder=3)
        mean = float(part["delta_oos_f1_pp"].mean())
        ax.scatter([mean], [position], marker="D", s=32, color=RED, edgecolor=DARK, linewidth=0.45, zorder=4)
        ax.plot([float(part["delta_oos_f1_pp"].min()), float(part["delta_oos_f1_pp"].max())], [position, position], color="#C9CDD1", linewidth=0.8, zorder=1)
    ax.axvline(0, color=DARK, linewidth=0.85)
    labels = [f"{DATASET_LABELS[dataset]}  KIR={kir:.2f}" for dataset, kir in order]
    ax.set_yticks(np.arange(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Trainable − Frozen OOS F1 (percentage points)")
    style_axis(ax, "x")
    for y in (2.5, 5.5):
        ax.axhline(y, color="#D7DADF", linewidth=0.7)
    positive_runs = int(stability["positive"].sum())
    group_means = stability.groupby(["dataset", "kir"])["delta_oos_f1_pp"].mean()
    positive_groups = int((group_means > 0).sum())
    ax.text(0.99, 1.04, f"{positive_runs}/{len(stability)} seed runs positive; {positive_groups}/{len(group_means)} group means positive", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=GRAY, markersize=4, label="seed run"),
        Line2D([], [], marker="o", linestyle="none", color=BLUE, markersize=4, label="negative exception"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor=RED, markeredgecolor=DARK, color=RED, markersize=5, label="three-seed mean"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.60, 0.94), ncol=3, handlelength=1.1, columnspacing=0.9)
    fig.text(0.23, 0.975, "Paired H1 stability across KIR and seeds", fontsize=8.2, fontweight="bold", va="top")
    save_figure(fig, "paired_kir_seed_stability")


def plot_subtype_risk(pairwise: pd.DataFrame) -> pd.DataFrame:
    part = pairwise[pairwise["dataset"].eq("stackoverflow")]
    summary = (
        part.groupby("gold_intent", as_index=False)
        .agg(
            n_oos=("sample_id", "size"),
            n_seeds=("seed", "nunique"),
            frozen_false_accepts=("frozen_pred", lambda values: int(np.sum(np.asarray(values) == 0))),
            trainable_false_accepts=("trainable_pred", lambda values: int(np.sum(np.asarray(values) == 0))),
        )
    )
    summary["frozen_false_accept_rate"] = summary["frozen_false_accepts"] / summary["n_oos"] * 100
    summary["trainable_false_accept_rate"] = summary["trainable_false_accepts"] / summary["n_oos"] * 100
    summary["reduction_pp"] = summary["frozen_false_accept_rate"] - summary["trainable_false_accept_rate"]
    summary = summary.sort_values(["trainable_false_accept_rate", "frozen_false_accept_rate"], ascending=False).reset_index(drop=True)
    summary.to_csv(RESULT_ROOT / "oos_subtype_residual_risk_summary.csv", index=False)

    height = max(3.5, 0.27 * len(summary) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, height))
    fig.subplots_adjust(left=0.29, right=0.985, bottom=0.13, top=0.86)
    y = np.arange(len(summary))
    frozen = summary["frozen_false_accept_rate"].to_numpy(float)
    trainable = summary["trainable_false_accept_rate"].to_numpy(float)
    for index in range(len(summary)):
        ax.plot([frozen[index], trainable[index]], [index, index], color="#BFC4C9", linewidth=1.0, zorder=1)
    ax.scatter(frozen, y, color=BLUE, s=25, label="Frozen", zorder=3)
    ax.scatter(trainable, y, color=RED, s=25, label="Trainable", zorder=3)
    labels = [str(value).replace("_", " ") for value in summary["gold_intent"]]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("OOS false-accept rate within subtype (%)")
    style_axis(ax, "x")
    ax.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.9)
    ax.text(0.0, 1.035, "red leftward = repair; red rightward = regression; sorted by residual risk", transform=ax.transAxes, fontsize=6.2, color="#555B61")
    fig.text(0.29, 0.97, "Residual OOS subtype risk (StackOverflow)", fontsize=8.2, fontweight="bold", va="top")
    fig.text(0.29, 0.035, "KIR=.50; pooled run-observations from seeds 13/42/87 when each intent is OOS. Seed coverage and counts are retained in the source table.", fontsize=6.0, color="#555B61")
    save_figure(fig, "oos_subtype_residual_risk")
    return summary


def write_summary_and_manifest(
    metrics: pd.DataFrame,
    transition: pd.DataFrame,
    decomposition: pd.DataFrame,
    stability: pd.DataFrame,
    geometry: Mapping[str, Any],
    encoded: Mapping[str, Any],
) -> None:
    kir50 = metrics[metrics["kir"].eq(0.50)]
    performance = (
        kir50.groupby(["dataset", "method"], as_index=False)
        .agg(oos_f1_mean=("oos_f1", "mean"), false_accept_rate_mean=("false_accept_rate", "mean"), known_recall_mean=("known_recall", "mean"))
    )
    performance["oos_f1_mean"] *= 100
    performance["false_accept_rate_mean"] *= 100
    performance["known_recall_mean"] *= 100
    performance.to_csv(RESULT_ROOT / "kir50_performance_summary.csv", index=False)

    group_means = stability.groupby(["dataset", "kir"])["delta_oos_f1_pp"].mean()
    decomp_max_error = float(decomposition["closure_error"].max())
    summary = {
        "schema_version": "s2c.h1_historical_oos_visual_explanation.v1",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_only",
        "not_protocols": ["strict_H0_full_Cascade", "protocol_v2_textoir_v1"],
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "kir50_positive_dataset_means": int(
            (
                kir50.pivot(index=["dataset", "seed"], columns="method", values="oos_f1")["trainable_k1"]
                - kir50.pivot(index=["dataset", "seed"], columns="method", values="oos_f1")["frozen_k1"]
            )
            .groupby("dataset")
            .mean()
            .gt(0)
            .sum()
        ),
        "all_kir_positive_group_means": int((group_means > 0).sum()),
        "all_kir_group_count": int(len(group_means)),
        "positive_seed_runs": int(stability["positive"].sum()),
        "seed_run_count": int(len(stability)),
        "transition_counts": {
            key: int(value)
            for key, value in transition.groupby("transition")["n_oos"].sum().to_dict().items()
        },
        "decomposition_max_closure_error": decomp_max_error,
        "geometry": dict(geometry),
        "reencoding_verification": encoded["verification"].to_dict(orient="records"),
        "selection_and_display_notes": {
            "local_geometry": geometry["selection_rule"],
            "density_smoothing": "fixed histogram bins followed by [1,2,3,2,1]/9 convolution",
            "display_tail": "values at or above the per-figure 99.5th-percentile cap are retained in the final bin/edge",
            "sample_export": "aggregate only; no raw embeddings, text, sample ids, or per-sample scores written",
        },
    }
    (RESULT_ROOT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_files = sorted(path for path in RESULT_ROOT.iterdir() if path.is_file() and path.name != "MANIFEST.json")
    manifest = {
        "schema_version": "s2c.analysis_bundle.v1",
        "bundle": "historical_oos_visual_explanation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_layer": "historical_controlled_gate_H1",
        "question": "How does Trainable MiniLM reduce OOS false acceptance under the identical H1 v19 K=1 Gate?",
        "source_experiment": "historical_protocol_reconciliation_20260826",
        "source_artifact_root": str(RUN_ROOT),
        "data_root": str(DATA_ROOT),
        "builder": str(Path(__file__).relative_to(ROOT)),
        "builder_sha256": sha256_file(Path(__file__)),
        "report": "docs/analysis/RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md",
        "result_files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for path in result_files
        ],
        "figure_files": [
            str((FIGURE_ROOT / f"{stem}.{suffix}").relative_to(ROOT))
            for stem in FIGURE_STEMS
            for suffix in ("png", "tiff", "pdf", "svg")
        ],
        "guardrails": {
            "new_training": False,
            "threshold_changed": False,
            "test_oos_used_for_selection": False,
            "fulltex_modified": False,
            "per_sample_output_written": False,
        },
    }
    (RESULT_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    metrics, predictions = load_completed_artifacts()
    pairwise, transition, decomposition, difficulty = build_pairwise_tables(predictions)
    stability = build_stability(metrics)
    encoded = load_stackoverflow_embeddings(predictions, args.device)

    geometry = plot_local_boundary_geometry(predictions, encoded)
    geometry["three_d"] = plot_local_boundary_geometry_3d(predictions, encoded)
    plot_oos_representation_movement(predictions, encoded)
    plot_paper_score_distribution(predictions, encoded)
    plot_score_crossing(pairwise)
    plot_corrected_decomposition(decomposition)
    plot_cross_dataset_density(predictions)
    plot_margin_difficulty(difficulty)
    plot_stability(stability)
    plot_subtype_risk(pairwise)
    write_summary_and_manifest(metrics, transition, decomposition, stability, geometry, encoded)

    print(
        json.dumps(
            {
                "bundle": "historical_oos_visual_explanation",
                "figures": len(FIGURE_STEMS),
                "result_root": str(RESULT_ROOT),
                "figure_root": str(FIGURE_ROOT),
                "new_training": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
