#!/usr/bin/env python3
"""Render a multi-metric effect landscape from completed fair comparisons."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/analysis/archive/analysis/statistical_stability_v1/paired_effects.csv"
OUT = ROOT / "results/analysis/archive/analysis/fair_effect_landscape_v1"
FIG = ROOT / "figures/archive/analysis/fair_effect_landscape_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
COMPARISONS = (
    "single_centroid",
    "fixed_k2",
    "random_partition",
    "mogb_minilm",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
)
METRICS = (
    "oos_f1",
    "f1_all",
    "f1_k",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
    "auroc",
    "aupr_oos",
)
METRIC_LABELS = {
    "oos_f1": "OOS F1",
    "f1_all": "F1-All",
    "f1_k": "F1-K",
    "known_recall": "Known Recall",
    "false_accept_rate": "False acceptance",
    "false_reject_rate": "False rejection",
    "auroc": "AUROC",
    "aupr_oos": "AUPR-OOS",
}
METHOD_LABELS = {
    "single_centroid": "Frozen K=1",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_minilm": "MOGB MiniLM",
    "mogb_partition_ours_boundary": "MOGB partition + ours boundary",
    "ours_partition_mogb_boundary": "Ours partition + MOGB boundary",
}
BOOTSTRAP_SOURCE_VERSION = "statistical_stability_v1"


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


def load() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE)
    frame = frame[
        frame.dataset.isin(DATASETS)
        & frame.kir.isin(KIRS)
        & frame.comparison_method.isin(COMPARISONS)
        & frame.metric.isin(METRICS)
    ].copy()
    expected = len(DATASETS) * len(KIRS) * len(COMPARISONS) * len(METRICS)
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, got {len(frame)}")
    if frame.duplicated(["dataset", "kir", "comparison_method", "metric"]).any():
        raise ValueError("duplicate effect rows")
    frame["effect_pp"] = frame["mean_better_delta"] * 100.0
    frame["ci95_low_pp"] = frame["ci95_low"] * 100.0
    frame["ci95_high_pp"] = frame["ci95_high"] * 100.0
    frame["significant_95"] = (frame.ci95_low_pp > 0) | (frame.ci95_high_pp < 0)
    frame["comparison_label"] = frame.comparison_method.map(METHOD_LABELS)
    frame["metric_label"] = frame.metric.map(METRIC_LABELS)
    return frame


def plot_oos_heatmap(frame: pd.DataFrame) -> None:
    sub = frame[frame.metric == "oos_f1"].copy()
    sub["cell"] = sub.dataset.str.upper() + "\nKIR=" + sub.kir.map(lambda value: f"{value:.2f}")
    order = [f"{dataset.upper()}\nKIR={kir:.2f}" for dataset in DATASETS for kir in KIRS]
    table = sub.pivot(index="comparison_label", columns="cell", values="effect_pp").reindex(columns=order)
    significant = sub.assign(cell=sub.cell).pivot(index="comparison_label", columns="cell", values="significant_95").reindex(columns=order)
    annotations = table.copy().astype(object)
    for row in annotations.index:
        for col in annotations.columns:
            value = table.loc[row, col]
            star = "*" if bool(significant.loc[row, col]) else ""
            annotations.loc[row, col] = f"{value:+.1f}{star}"
    fig, ax = plt.subplots(figsize=(15, 5.8))
    sns.heatmap(table, annot=annotations, fmt="", cmap="RdBu_r", center=0, linewidths=0.6, linecolor="white", cbar_kws={"label": "Trainable advantage (pp)"}, ax=ax)
    ax.set_xlabel("Dataset × KIR")
    ax.set_ylabel("Comparison")
    ax.set_title("Trainable K=1 OOS F1 advantage across the fair matrix\n* 95% paired bootstrap CI excludes zero")
    fig.tight_layout()
    fig.savefig(FIG / "trainable_oos_f1_effect_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(frame: pd.DataFrame) -> None:
    wide = frame[frame.metric.isin(["oos_f1", "false_accept_rate"])]
    wide = wide.pivot(index=["dataset", "kir", "comparison_method", "comparison_label"], columns="metric", values="effect_pp").reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = wide[wide.dataset == dataset]
        for method in COMPARISONS:
            row = sub[sub.comparison_method == method]
            ax.plot(row.false_accept_rate, row.oos_f1, marker="o", linewidth=1.4, label=METHOD_LABELS[method])
            for _, point in row.iterrows():
                ax.annotate(f"{point.kir:.2f}", (point.false_accept_rate, point.oos_f1), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(dataset)
        ax.set_xlabel("False-acceptance advantage (pp; higher is safer)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("OOS F1 advantage (pp)")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Trainable advantage: OOS F1 versus false acceptance across KIR", y=1.03)
    fig.savefig(FIG / "trainable_oos_fa_effect_landscape.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(frame: pd.DataFrame) -> None:
    sub = frame.groupby(["comparison_label", "metric_label"], as_index=False).agg(mean_effect_pp=("effect_pp", "mean"), significant_cells=("significant_95", "sum"))
    table = sub.pivot(index="comparison_label", columns="metric_label", values="mean_effect_pp")
    table = table.reindex(columns=[METRIC_LABELS[m] for m in METRICS])
    fig, ax = plt.subplots(figsize=(14, 5.8))
    sns.heatmap(table, annot=True, fmt="+.1f", cmap="RdBu_r", center=0, linewidths=0.6, linecolor="white", cbar_kws={"label": "Mean Trainable advantage (pp)"}, ax=ax)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Comparison")
    ax.set_title("Mean multi-metric effect across 9 dataset×KIR cells")
    fig.tight_layout()
    fig.savefig(FIG / "trainable_multimetric_effect_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame = load()
    atomic_csv(frame, OUT / "effect_landscape.csv")
    summary = frame.groupby(["comparison_method", "comparison_label", "metric", "metric_label"], as_index=False).agg(
        mean_effect_pp=("effect_pp", "mean"),
        std_effect_pp=("effect_pp", "std"),
        significant_cells=("significant_95", "sum"),
        cells=("significant_95", "size"),
    )
    atomic_csv(summary, OUT / "effect_landscape_summary.csv")
    plot_oos_heatmap(frame)
    plot_tradeoff(frame)
    plot_metric_summary(frame)
    atomic_json(
        {
            "analysis_id": "fair_effect_landscape_v1",
            "protocol_version": "protocol_v2_textoir_v1",
            "source": str(SOURCE),
            "source_sha256": sha256(SOURCE),
            "source_version": BOOTSTRAP_SOURCE_VERSION,
            "datasets": list(DATASETS),
            "kirs": list(KIRS),
            "comparisons": list(COMPARISONS),
            "metrics": list(METRICS),
            "rows": int(len(frame)),
            "significance": "95% paired bootstrap CI inherited from statistical_stability_v1; no new model run",
            "outputs": [
                "effect_landscape.csv",
                "effect_landscape_summary.csv",
                "trainable_oos_f1_effect_heatmap.png",
                "trainable_oos_fa_effect_landscape.png",
                "trainable_multimetric_effect_summary.png",
            ],
        },
        OUT / "MANIFEST.json",
    )


if __name__ == "__main__":
    main()
