#!/usr/bin/env python3
"""Decompose OOS F1 into OOS precision/recall and error counts.

The source is the already audited per-seed prediction summary.  OOS precision
and recall are reconstructed from the known/OOS denominators and audited error
rates, then checked against the original Trainable rows where available.
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
SOURCE = ROOT / "results/analysis/archive/analysis/cross_dataset_error_attribution_v1/method_metrics_per_seed.csv"
REFERENCE = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
OUT = ROOT / "results/analysis/archive/analysis/oos_error_budget_v1"
FIG = ROOT / "figures/archive/analysis/oos_error_budget_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
METHODS = (
    "trainable_k1",
    "single_centroid",
    "fixed_k2",
    "random_partition",
    "mogb_partition_ours_boundary",
    "mogb_minilm",
    "ours_partition_mogb_boundary",
)
METHOD_LABELS = {
    "trainable_k1": "Trainable K=1",
    "single_centroid": "Frozen K=1",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_partition_ours_boundary": "MOGB partition + ours",
    "mogb_minilm": "MOGB MiniLM",
    "ours_partition_mogb_boundary": "Ours partition + MOGB",
}
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_REPS = 10_000


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
    required = {
        "dataset",
        "kir",
        "seed",
        "method",
        "n_known",
        "n_oos",
        "oos_f1",
        "known_recall",
        "false_accept_rate",
        "false_reject_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing source columns: {missing}")
    frame = frame[frame.dataset.isin(DATASETS) & frame.kir.isin(KIRS) & frame.method.isin(METHODS)].copy()
    expected = len(DATASETS) * len(KIRS) * 5 * len(METHODS)
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, got {len(frame)}")
    if frame.duplicated(["dataset", "kir", "seed", "method"]).any():
        raise ValueError("duplicate source keys")
    frame["oos_false_accept_count"] = frame.n_oos * frame.false_accept_rate
    frame["known_false_reject_count"] = frame.n_known * frame.false_reject_rate
    frame["oos_true_reject_count"] = frame.n_oos - frame.oos_false_accept_count
    frame["oos_recall"] = frame.oos_true_reject_count / frame.n_oos
    predicted_oos = frame.oos_true_reject_count + frame.known_false_reject_count
    frame["oos_precision"] = frame.oos_true_reject_count / predicted_oos.replace(0, np.nan)
    frame["oos_f1_reconstructed"] = 2 * frame.oos_precision * frame.oos_recall / (frame.oos_precision + frame.oos_recall)
    frame["method_label"] = frame.method.map(METHOD_LABELS)
    if not np.isfinite(frame[["oos_precision", "oos_recall", "oos_f1_reconstructed"]].to_numpy()).all():
        raise ValueError("non-finite reconstructed OOS metrics")
    if (frame.oos_f1_reconstructed - frame.oos_f1).abs().max() > 1e-9:
        raise ValueError("reconstructed OOS F1 does not match audited source")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "oos_f1": "oos_f1",
        "oos_precision": "oos_precision",
        "oos_recall": "oos_recall",
        "known_recall": "known_recall",
        "false_accept_rate": "false_accept_rate",
        "false_reject_rate": "false_reject_rate",
    }
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["dataset", "kir", "method", "method_label"], sort=False):
        row: dict[str, object] = dict(zip(["dataset", "kir", "method", "method_label"], keys))
        row["n_seeds"] = int(group.seed.nunique())
        row["n_known"] = int(group.n_known.iloc[0])
        row["n_oos"] = int(group.n_oos.iloc[0])
        for source, label in metrics.items():
            row[f"{label}_mean"] = float(group[source].mean())
            row[f"{label}_std"] = float(group[source].std())
        for source in ("oos_false_accept_count", "known_false_reject_count"):
            row[f"{source}_mean"] = float(group[source].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ("oos_f1", "oos_precision", "oos_recall", "known_recall", "false_accept_rate", "false_reject_rate")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for (dataset, kir, comparison), group in frame.groupby(["dataset", "kir", "method"], sort=False):
        if comparison == "trainable_k1":
            continue
        train = frame[(frame.dataset == dataset) & (frame.kir == kir) & (frame.method == "trainable_k1")].set_index("seed")
        comp = group.set_index("seed")
        seeds = sorted(set(train.index) & set(comp.index))
        if len(seeds) != 5:
            raise ValueError(f"paired seeds incomplete for {dataset}/{kir}/{comparison}")
        for metric in metrics:
            values = train.loc[seeds, metric].to_numpy() - comp.loc[seeds, metric].to_numpy()
            draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
            means = values[draws].mean(axis=1)
            rows.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "comparison_method": comparison,
                    "comparison_label": METHOD_LABELS[comparison],
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean_trainable_minus_comparison": float(values.mean()),
                    "mean_delta_pp": float(values.mean() * 100),
                    "ci95_low": float(np.quantile(means, 0.025)),
                    "ci95_high": float(np.quantile(means, 0.975)),
                    "wins": int((values > 0).sum()),
                    "ties": int((values == 0).sum()),
                    "losses": int((values < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_precision_recall(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = summary[(summary.dataset == dataset) & (summary.kir == 0.50)]
        for method in METHODS:
            row = sub[sub.method == method].iloc[0]
            ax.scatter(row.oos_recall_mean * 100, row.oos_precision_mean * 100, s=70, label=METHOD_LABELS[method])
            ax.annotate(METHOD_LABELS[method], (row.oos_recall_mean * 100, row.oos_precision_mean * 100), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_title(dataset)
        ax.set_xlabel("OOS Recall (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS Precision (%)")
    fig.suptitle("OOS precision–recall work points at KIR=0.50", y=1.02)
    fig.savefig(FIG / "oos_precision_recall_workpoints.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_budget(summary: pd.DataFrame) -> None:
    sub = summary[summary.kir == 0.50].copy()
    order = list(METHODS)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        part = sub[sub.dataset == dataset].set_index("method").reindex(order)
        x = np.arange(len(order))
        width = 0.36
        ax.bar(x - width / 2, part.false_accept_rate_mean * 100, width, label="OOS → Known (false acceptance)", color="#d95f02")
        ax.bar(x + width / 2, part.false_reject_rate_mean * 100, width, label="Known → OOS (false rejection)", color="#1b9e77")
        ax.set_title(dataset)
        ax.set_xticks(x, [METHOD_LABELS[m].replace(" + ", "\n+") for m in order], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Error rate (%)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("OOS error budget at KIR=0.50", y=1.02)
    fig.savefig(FIG / "oos_error_budget_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paired_effects(effects: pd.DataFrame) -> None:
    sub = effects[effects.kir == 0.50].copy()
    order = ["oos_f1", "oos_precision", "oos_recall", "known_recall", "false_accept_rate", "false_reject_rate"]
    labels = {"oos_f1": "OOS F1", "oos_precision": "OOS Precision", "oos_recall": "OOS Recall", "known_recall": "Known Recall", "false_accept_rate": "FA risk", "false_reject_rate": "FR risk"}
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        part = sub[sub.dataset == dataset]
        for i, metric in enumerate(order):
            rows = part[part.metric == metric].set_index("comparison_method").reindex(METHODS[1:])
            y = np.arange(len(rows)) + i * (len(rows) + 1)
            ax.errorbar(rows.mean_delta_pp, y, xerr=[rows.mean_delta_pp - rows.ci95_low * 100, rows.ci95_high * 100 - rows.mean_delta_pp], fmt="o", capsize=2, markersize=3, label=labels[metric] if dataset == DATASETS[0] else None)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_title(dataset)
        ax.set_xlabel("Trainable − comparison (pp)")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("Metric blocks (comparison rows)")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Paired OOS error-budget effects at KIR=0.50", y=1.02)
    fig.savefig(FIG / "trainable_oos_error_budget_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame = load()
    summary = summarize(frame)
    effects = paired_effects(frame)
    atomic_csv(frame, OUT / "per_seed_reconstructed.csv")
    atomic_csv(summary, OUT / "summary.csv")
    atomic_csv(effects, OUT / "paired_effects.csv")
    plot_precision_recall(summary)
    plot_error_budget(summary)
    plot_paired_effects(effects)
    atomic_json(
        {
            "analysis_id": "oos_error_budget_v1",
            "protocol_version": "protocol_v2_textoir_v1",
            "source": str(SOURCE),
            "source_sha256": sha256(SOURCE),
            "reference": str(REFERENCE),
            "datasets": list(DATASETS),
            "kirs": list(KIRS),
            "methods": list(METHODS),
            "rows_per_seed": int(len(frame)),
            "summary_rows": int(len(summary)),
            "paired_effect_rows": int(len(effects)),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "reconstruction": "OOS precision and recall reconstructed from audited n_known/n_oos and error rates; OOS F1 checked against source.",
            "outputs": [
                "per_seed_reconstructed.csv",
                "summary.csv",
                "paired_effects.csv",
                "oos_precision_recall_workpoints.png",
                "oos_error_budget_kir050.png",
                "trainable_oos_error_budget_effects.png",
            ],
        },
        OUT / "MANIFEST.json",
    )


if __name__ == "__main__":
    main()
