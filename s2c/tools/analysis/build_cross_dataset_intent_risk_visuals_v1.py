#!/usr/bin/env python3
"""Visualize intent-level OOS acceptance and Known rejection risk.

This is an analysis-only consumer of the already audited
``cross_dataset_error_attribution_v1`` intent summary.  It does not read raw
text, tune a threshold, or create a new prediction run.  The purpose is to
show which intents account for the coverage/open-space trade-off across
datasets and KIR values.
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
SOURCE = ROOT / "results/analysis/archive/analysis/cross_dataset_error_attribution_v1/intent_error_attribution_summary.csv"
OUT = ROOT / "results/analysis/archive/analysis/cross_dataset_intent_risk_visuals_v1"
FIG = ROOT / "figures/archive/analysis/cross_dataset_intent_risk_visuals_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
METHODS = (
    "trainable_k1",
    "fixed_k2",
    "random_partition",
    "mogb_minilm",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
)
LABELS = {
    "trainable_k1": "Trainable K=1",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_minilm": "MOGB-MiniLM",
    "mogb_partition_ours_boundary": "MOGB partition + ours",
    "ours_partition_mogb_boundary": "Ours partition + MOGB",
}
COLORS = {
    "trainable_k1": "#2b6cb0",
    "fixed_k2": "#c53030",
    "random_partition": "#dd6b20",
    "mogb_minilm": "#805ad5",
    "mogb_partition_ours_boundary": "#319795",
    "ours_partition_mogb_boundary": "#718096",
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
    required = {
        "dataset",
        "kir",
        "method",
        "method_label",
        "intent",
        "known_count_mean",
        "known_false_reject_rate_mean",
        "oos_false_accept_rate_mean",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = frame[frame.dataset.isin(DATASETS) & frame.kir.isin(KIRS) & frame.method.isin(METHODS)].copy()
    if frame.empty:
        raise ValueError("empty intent summary")
    if frame[["known_false_reject_rate_mean", "oos_false_accept_rate_mean"]].isna().any().any():
        raise ValueError("NaN in intent risk summary")
    if not np.isfinite(frame[["known_false_reject_rate_mean", "oos_false_accept_rate_mean"]].to_numpy()).all():
        raise ValueError("non-finite intent risk summary")
    return frame


def top_intents(frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for keys, group in frame.groupby(["dataset", "kir", "method"], sort=False):
        ordered = group.sort_values(
            ["oos_false_accept_rate_mean", "known_false_reject_rate_mean", "intent"],
            ascending=[False, False, True],
        ).head(limit).copy()
        total = float(group["oos_false_accept_rate_mean"].sum())
        ordered["rank"] = np.arange(1, len(ordered) + 1)
        ordered["method_oos_fa_total"] = total
        ordered["oos_fa_share_of_method_total"] = ordered["oos_false_accept_rate_mean"] / max(total, 1e-12)
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True).sort_values(["dataset", "kir", "method", "rank"])


def concentration(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["dataset", "kir", "method"], sort=False):
        ordered = group.sort_values("oos_false_accept_rate_mean", ascending=False)
        total = float(ordered["oos_false_accept_rate_mean"].sum())
        row = {"dataset": keys[0], "kir": keys[1], "method": keys[2], "method_label": LABELS[keys[2]], "oos_fa_total": total}
        for n in (1, 3, 5, 10):
            row[f"top{n}_share"] = float(ordered.head(n)["oos_false_accept_rate_mean"].sum() / max(total, 1e-12))
        row["top_intent"] = str(ordered.iloc[0]["intent"])
        row["top_intent_fa_rate"] = float(ordered.iloc[0]["oos_false_accept_rate_mean"])
        rows.append(row)
    return pd.DataFrame(rows)


def deltas(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[frame.method == "trainable_k1"].rename(
        columns={
            "known_false_reject_rate_mean": "trainable_known_fr",
            "oos_false_accept_rate_mean": "trainable_oos_fa",
        }
    )[["dataset", "kir", "intent", "trainable_known_fr", "trainable_oos_fa"]]
    comparisons = frame[frame.method.isin(METHODS[1:])].merge(base, on=["dataset", "kir", "intent"], how="left", validate="many_to_one")
    comparisons["delta_known_fr_vs_trainable"] = comparisons.known_false_reject_rate_mean - comparisons.trainable_known_fr
    comparisons["delta_oos_fa_vs_trainable"] = comparisons.oos_false_accept_rate_mean - comparisons.trainable_oos_fa
    comparisons["method_label"] = comparisons.method.map(LABELS)
    return comparisons


def plot_risk_scatter(frame: pd.DataFrame) -> None:
    selected = frame[(frame.kir == 0.50) & frame.method.isin(("trainable_k1", "fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary"))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        group = selected[selected.dataset == dataset]
        for method in ("trainable_k1", "fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary"):
            sub = group[group.method == method]
            ax.scatter(
                sub.known_false_reject_rate_mean * 100,
                sub.oos_false_accept_rate_mean * 100,
                s=np.maximum(20, np.sqrt(sub.known_count_mean) * 3),
                alpha=0.72,
                label=LABELS[method],
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_title(dataset)
        ax.grid(alpha=0.22)
        ax.axvline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Intent-level OOS false acceptance contribution (%)")
    for ax in axes:
        ax.set_xlabel("Intent-level Known false rejection (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("KIR=0.50: intent-level coverage/open-space risk", y=1.15)
    fig.savefig(FIG / "intent_risk_scatter_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_concentration(summary: pd.DataFrame) -> None:
    selected = summary[summary.kir == 0.50].copy()
    methods = ("trainable_k1", "fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for ax, column, title in zip(axes, ("top1_share", "top5_share"), ("Top-1 intent share of OOS false acceptance", "Top-5 intent share of OOS false acceptance")):
        x = np.arange(len(DATASETS))
        width = 0.18
        for index, method in enumerate(methods):
            values = [float(selected[(selected.dataset == dataset) & (selected.method == method)][column].iloc[0]) * 100 for dataset in DATASETS]
            ax.bar(x + (index - 1.5) * width, values, width, label=LABELS[method], color=COLORS[method])
        ax.set_xticks(x, DATASETS)
        ax.set_ylabel("Share (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("KIR=0.50: OOS false acceptance concentration by intent", y=1.04)
    fig.savefig(FIG / "oos_acceptance_concentration_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_scatter(delta: pd.DataFrame) -> None:
    selected = delta[(delta.kir == 0.50) & delta.method.isin(("fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary"))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        group = selected[selected.dataset == dataset]
        for method in ("fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary"):
            sub = group[group.method == method]
            ax.scatter(sub.delta_known_fr_vs_trainable * 100, sub.delta_oos_fa_vs_trainable * 100, s=24, alpha=0.68, label=LABELS[method], color=COLORS[method])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(dataset)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Δ OOS false acceptance vs Trainable (pp)")
    for ax in axes:
        ax.set_xlabel("Δ Known false rejection vs Trainable (pp)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("KIR=0.50: per-intent error shifts relative to Trainable K=1", y=1.15)
    fig.savefig(FIG / "intent_error_delta_scatter_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rank_curve(frame: pd.DataFrame) -> None:
    selected = frame[frame.method.isin(("trainable_k1", "fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary"))]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        for kir in KIRS:
            sub = selected[(selected.dataset == dataset) & (selected.kir == kir) & (selected.method == "fixed_k2")].sort_values("oos_false_accept_rate_mean", ascending=False)
            total = sub.oos_false_accept_rate_mean.sum()
            cumulative = sub.oos_false_accept_rate_mean.cumsum() / max(total, 1e-12)
            ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=2.5, label=f"KIR={kir:.2f}")
        ax.set_title(dataset)
        ax.set_xlabel("Intent rank by Fixed K=2 OOS acceptance")
        ax.grid(alpha=0.22)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Cumulative share of Fixed K=2 OOS false acceptance")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Which intents absorb OOS under Fixed K=2?", y=1.02)
    fig.savefig(FIG / "fixed_k2_oos_acceptor_rank_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source = load_source()
    top = top_intents(source)
    conc = concentration(source)
    delta = deltas(source)
    atomic_csv(top, OUT / "top_oos_acceptor_intents.csv")
    atomic_csv(conc, OUT / "oos_acceptance_concentration.csv")
    atomic_csv(delta, OUT / "intent_error_deltas_vs_trainable.csv")
    plot_risk_scatter(source)
    plot_concentration(conc)
    plot_delta_scatter(delta)
    plot_rank_curve(source)
    manifest = {
        "analysis_id": "cross_dataset_intent_risk_visuals_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "methods": list(METHODS),
        "source_rows": int(len(source)),
        "top_intent_limit": 10,
        "threshold_selection": "not applicable; consumes frozen summary at formal score<=1",
        "test_usage": "post-hoc attribution only; no selection or tuning",
        "outputs": [
            "top_oos_acceptor_intents.csv",
            "oos_acceptance_concentration.csv",
            "intent_error_deltas_vs_trainable.csv",
            "intent_risk_scatter_kir050.png",
            "oos_acceptance_concentration_kir050.png",
            "intent_error_delta_scatter_kir050.png",
            "fixed_k2_oos_acceptor_rank_curve.png",
        ],
    }
    atomic_json(manifest, OUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
