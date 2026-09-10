#!/usr/bin/env python3
"""Build the authoritative Chinese analysis bundle for the current fair matrix.

This is analysis-only: it consumes the audited five-seed summary and never
selects a model, threshold, K, or external baseline from test predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
OUT = ROOT / "results/analysis/archive/analysis/experiment_analysis_master_v1"
FIG = ROOT / "figures/archive/analysis/experiment_analysis_master_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
METHODS = (
    "trainable_k1",
    "single_centroid",
    "fixed_k2",
    "random_partition",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
    "mogb_minilm",
)
LABELS = {
    "trainable_k1": "Trainable K=1",
    "single_centroid": "Frozen K=1",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_partition_ours_boundary": "MOGB partition + S2C boundary",
    "ours_partition_mogb_boundary": "S2C partition + MOGB boundary",
    "mogb_minilm": "MOGB-MiniLM",
}
COLORS = {
    "trainable_k1": "#0072B2",
    "single_centroid": "#777777",
    "fixed_k2": "#D55E00",
    "random_partition": "#009E73",
    "mogb_partition_ours_boundary": "#CC79A7",
    "ours_partition_mogb_boundary": "#E69F00",
    "mogb_minilm": "#56B4E9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_source() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE)
    required = {"dataset", "kir", "method", "oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate", "n_seeds"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing source columns: {missing}")
    frame = frame[frame.dataset.isin(DATASETS) & frame.method.isin(METHODS) & frame.kir.isin(KIRS)].copy()
    expected = len(DATASETS) * len(KIRS) * len(METHODS)
    if len(frame) != expected:
        raise ValueError(f"expected {expected} summary rows, got {len(frame)}")
    if frame.duplicated(["dataset", "kir", "method"]).any():
        raise ValueError("duplicate dataset/kir/method rows")
    if set(frame.n_seeds.astype(int)) != {5}:
        raise ValueError("fair matrix must contain exactly five seeds per cell")
    numeric = ["oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate", "accuracy", "auroc", "aupr_oos"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("non-finite metrics in source")
    frame["method_label"] = frame.method.map(LABELS)
    frame["dataset_label"] = frame.dataset.map({"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"})
    frame["kir_label"] = frame.kir.map(lambda x: f"{x:.2f}")
    return frame


def rank_cells(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, kir), group in frame.groupby(["dataset", "kir"], sort=False):
        group = group.copy()
        # Higher is better for performance and lower is better for error rates.
        group["rank_oos_f1"] = group.oos_f1.rank(method="min", ascending=False).astype(int)
        group["rank_f1_all"] = group.f1_all.rank(method="min", ascending=False).astype(int)
        group["rank_known_recall"] = group.known_recall.rank(method="min", ascending=False).astype(int)
        group["rank_false_accept_rate"] = group.false_accept_rate.rank(method="min", ascending=True).astype(int)
        group["rank_false_reject_rate"] = group.false_reject_rate.rank(method="min", ascending=True).astype(int)
        for _, row in group.iterrows():
            value = row.to_dict()
            value["dataset"] = dataset
            value["kir"] = kir
            rows.append(value)
    result = pd.DataFrame(rows)
    return result.sort_values(["dataset", "kir", "rank_oos_f1", "method"]).reset_index(drop=True)


def global_summary(frame: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in frame.groupby("method", sort=False):
        ranked = ranks[ranks.method == method]
        row: dict[str, Any] = {"method": method, "method_label": LABELS[method], "n_cells": len(group)}
        for metric in ("oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate", "accuracy", "auroc", "aupr_oos"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std_across_cells"] = float(group[metric].std(ddof=1))
        for metric in ("oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate"):
            row[f"rank1_{metric}_cells"] = int((ranked[f"rank_{metric}"] == 1).sum())
            row[f"mean_rank_{metric}"] = float(ranked[f"rank_{metric}"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("oos_f1_mean", ascending=False).reset_index(drop=True)


def trainable_effects(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame.method == "trainable_k1"].set_index(["dataset", "kir"])
    rows: list[dict[str, Any]] = []
    for _, row in frame[frame.method != "trainable_k1"].iterrows():
        ref = baseline.loc[(row.dataset, row.kir)]
        item = {"dataset": row.dataset, "kir": row.kir, "comparison_method": row.method, "comparison_label": LABELS[row.method]}
        for metric in ("oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate"):
            item[f"trainable_minus_{metric}"] = float(ref[metric] - row[metric])
            item[f"trainable_minus_{metric}_pp"] = float((ref[metric] - row[metric]) * 100)
        rows.append(item)
    return pd.DataFrame(rows)


def plot_heatmap(frame: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    pivot = frame.pivot_table(index="method_label", columns=["dataset_label", "kir_label"], values=metric)
    order = [LABELS[m] for m in METHODS]
    columns = [(d, f"{kir:.2f}") for d in ("CLINC150", "Banking77", "StackOverflow") for kir in KIRS]
    pivot = pivot.reindex(order).reindex(columns=pd.MultiIndex.from_tuples(columns))
    fig, ax = plt.subplots(figsize=(15, 5.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy() * 100, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(columns)), [f"{d}\nKIR={k}" for d, k in columns], fontsize=8)
    ax.set_yticks(np.arange(len(order)), order, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j] * 100:.1f}", ha="center", va="center", color="white" if pivot.iloc[i, j] > 0.55 else "black", fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Percent")
    fig.savefig(FIG / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        part = frame[(frame.dataset == dataset) & (frame.kir == 0.50)]
        for _, row in part.iterrows():
            ax.scatter(row.f1_all * 100, row.oos_f1 * 100, s=75, color=COLORS[row.method], label=LABELS[row.method])
            ax.annotate(LABELS[row.method], (row.f1_all * 100, row.oos_f1 * 100), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("F1-All (%)")
        ax.set_xlim(35, 90)
        ax.set_ylim(35, 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.suptitle("KIR=0.50: OOS F1 vs F1-All coverage/rejection trade-off", y=1.16)
    fig.savefig(FIG / "pareto_oos_f1_f1_all_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rank_counts(ranks: pd.DataFrame) -> None:
    summary = ranks.groupby("method").agg(
        oos_rank1=("rank_oos_f1", lambda x: int((x == 1).sum())),
        f1all_rank1=("rank_f1_all", lambda x: int((x == 1).sum())),
        mean_oos_rank=("rank_oos_f1", "mean"),
    ).reindex(METHODS)
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    x = np.arange(len(METHODS))
    width = 0.35
    ax.bar(x - width / 2, summary.oos_rank1, width, label="OOS F1 rank 1", color="#0072B2")
    ax.bar(x + width / 2, summary.f1all_rank1, width, label="F1-All rank 1", color="#D55E00")
    ax.set_xticks(x, [LABELS[m].replace(" + ", "\n+") for m in METHODS], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Rank-1 cells among 9 dataset x KIR cells")
    ax.set_title("Cross-dataset/KIR rank-1 counts (correct metric direction)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG / "method_rank1_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_kir_curves(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = frame[frame.dataset == dataset]
        for method in METHODS:
            values = part[part.method == method].sort_values("kir")
            ax.plot(values.kir, values.oos_f1 * 100, marker="o", linewidth=1.7, color=COLORS[method], label=LABELS[method])
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("KIR")
        ax.set_xticks(KIRS)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("OOS F1 across KIR (five-seed means)", y=1.02)
    fig.savefig(FIG / "oos_f1_kir_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame = load_source()
    ranks = rank_cells(frame)
    summary = global_summary(frame, ranks)
    effects = trainable_effects(frame)
    atomic_csv(frame.sort_values(["dataset", "kir", "method"]), OUT / "audited_summary.csv")
    atomic_csv(ranks, OUT / "cell_ranking.csv")
    atomic_csv(summary, OUT / "method_global_summary.csv")
    atomic_csv(effects, OUT / "trainable_paired_effects.csv")
    plot_heatmap(frame, "oos_f1", "oos_f1_heatmap.png", "OOS F1 (five-seed means)")
    plot_heatmap(frame, "f1_all", "f1_all_heatmap.png", "F1-All (five-seed means)")
    plot_pareto(frame)
    plot_rank_counts(ranks)
    plot_kir_curves(frame)
    manifest = {
        "analysis_id": "experiment_analysis_master_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "rows": int(len(frame)),
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "methods": list(METHODS),
        "seeds_per_cell": 5,
        "analysis_only": True,
        "test_selection": False,
        "figures": sorted(path.name for path in FIG.glob("*.png")),
    }
    atomic_json(manifest, OUT / "MANIFEST.json")
    print(json.dumps({"status": "ok", "rows": len(frame), "output": str(OUT), "figures": len(manifest["figures"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
