#!/usr/bin/env python3
"""Paired seed stability and method ranking for completed fair predictions.

Only the frozen ``cross_protocol_tradeoff_v1/per_seed.csv`` table is read.
Every comparison is paired on the same dataset, KIR, and seed.  The script
does not run a model or select a threshold; it produces statistical evidence
for the already completed fair matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
OUT = ROOT / "results/analysis/archive/analysis/statistical_stability_v1"
FIG = ROOT / "figures/archive/analysis/statistical_stability_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
BASELINE = "trainable_k1"
METHODS = (
    "trainable_k1",
    "single_centroid",
    "fixed_k2",
    "random_partition",
    "mogb_minilm",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
)
LABELS = {
    "trainable_k1": "Trainable K=1",
    "single_centroid": "Frozen K=1",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_minilm": "MOGB-MiniLM",
    "mogb_partition_ours_boundary": "MOGB partition + ours",
    "ours_partition_mogb_boundary": "Ours partition + MOGB",
}
COLORS = {
    "trainable_k1": "#2b6cb0",
    "single_centroid": "#718096",
    "fixed_k2": "#c53030",
    "random_partition": "#dd6b20",
    "mogb_minilm": "#805ad5",
    "mogb_partition_ours_boundary": "#319795",
    "ours_partition_mogb_boundary": "#4a5568",
}
HIGHER_BETTER = {"oos_f1", "f1_all", "f1_k", "f1_u", "known_recall", "auroc", "aupr_oos"}
METRICS = tuple(HIGHER_BETTER | {"false_accept_rate", "false_reject_rate"})
BOOTSTRAP_SEED = 20260725
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


def load_source() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE)
    frame = frame[frame.dataset.isin(DATASETS) & frame.kir.isin(KIRS) & frame.seed.isin(SEEDS) & frame.method.isin(METHODS)].copy()
    expected = len(DATASETS) * len(KIRS) * len(SEEDS) * len(METHODS)
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, got {len(frame)}")
    if frame.duplicated(["dataset", "kir", "seed", "method"]).any():
        raise ValueError("duplicate dataset/kir/seed/method rows")
    missing = [metric for metric in METRICS if metric not in frame.columns]
    if missing:
        raise ValueError(f"missing metrics: {missing}")
    values = frame[list(METRICS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite source metrics")
    return frame


def sign_test_pvalue(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > 1e-12]
    n = int(nonzero.size)
    if n == 0:
        return 1.0
    positive = int(np.sum(nonzero > 0))
    lower = sum(math.comb(n, k) for k in range(0, positive + 1)) / (2**n)
    upper = sum(math.comb(n, k) for k in range(positive, n + 1)) / (2**n)
    return float(min(1.0, 2.0 * min(lower, upper)))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    sample = rng.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True)
    means = sample.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            base = frame[(frame.dataset == dataset) & (frame.kir == kir) & (frame.method == BASELINE)].set_index("seed")
            for method in METHODS[1:]:
                comparison = frame[(frame.dataset == dataset) & (frame.kir == kir) & (frame.method == method)].set_index("seed")
                if set(base.index) != set(comparison.index):
                    raise ValueError(f"seed mismatch {dataset} {kir} {method}")
                for metric in METRICS:
                    raw = base.loc[list(SEEDS), metric].to_numpy() - comparison.loc[list(SEEDS), metric].to_numpy()
                    if metric in {"false_accept_rate", "false_reject_rate"}:
                        better = -raw
                        interpretation = "comparison_minus_trainable"  # positive means Trainable has lower risk
                    else:
                        better = raw
                        interpretation = "trainable_minus_comparison"  # positive means Trainable is better
                    ci_low, ci_high = bootstrap_ci(better, rng)
                    wins = int(np.sum(better > 1e-12))
                    ties = int(np.sum(np.abs(better) <= 1e-12))
                    losses = int(len(better) - wins - ties)
                    std = float(np.std(better, ddof=1)) if len(better) > 1 else 0.0
                    rows.append(
                        {
                            "dataset": dataset,
                            "kir": kir,
                            "comparison_method": method,
                            "comparison_label": LABELS[method],
                            "metric": metric,
                            "n_seeds": len(better),
                            "mean_better_delta": float(np.mean(better)),
                            "median_better_delta": float(np.median(better)),
                            "std_better_delta": std,
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "wins": wins,
                            "ties": ties,
                            "losses": losses,
                            "sign_test_pvalue": sign_test_pvalue(better),
                            "effect_size": float(np.mean(better) / std) if std > 0 else 0.0,
                            "interpretation": interpretation,
                        }
                    )
    return pd.DataFrame(rows)


def rankings(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    for metric in ("oos_f1", "f1_all", "known_recall", "false_accept_rate"):
        sub = frame[["dataset", "kir", "seed", "method", metric]].copy()
        ascending = metric == "false_accept_rate"
        sub["rank"] = sub.groupby(["dataset", "kir", "seed"])[metric].rank(method="average", ascending=ascending)
        sub["is_best"] = sub["rank"].eq(1.0)
        sub["metric"] = metric
        records.append(sub)
    per_seed = pd.concat(records, ignore_index=True)
    summary = per_seed.groupby(["dataset", "kir", "method", "metric"], as_index=False).agg(mean_rank=("rank", "mean"), std_rank=("rank", "std"), wins=("is_best", "sum"), n_seeds=("seed", "nunique"))
    return per_seed, summary


def plot_forest(effects: pd.DataFrame) -> None:
    subset = effects[effects.metric == "oos_f1"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 11), sharex=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        group = subset[subset.dataset == dataset].copy()
        group["label"] = group.kir.map(lambda value: f"KIR={value:.2f}") + " · " + group.comparison_label
        group = group.sort_values(["kir", "comparison_method"], ascending=[True, True])
        y = np.arange(len(group))
        ax.errorbar(group.mean_better_delta * 100, y, xerr=[(group.mean_better_delta - group.ci95_low) * 100, (group.ci95_high - group.mean_better_delta) * 100], fmt="o", color="#2b6cb0", ecolor="#718096", capsize=2, markersize=4)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_yticks(y, group.label, fontsize=8)
        ax.set_title(dataset)
        ax.set_xlabel("Trainable OOS F1 advantage (percentage points)")
        ax.grid(axis="x", alpha=0.22)
    fig.suptitle("Paired five-seed OOS F1 effects: Trainable K=1 vs completed fair methods", y=1.01)
    fig.savefig(FIG / "paired_oos_f1_forest.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rank_heatmap(summary: pd.DataFrame) -> None:
    sub = summary[summary.metric == "oos_f1"].copy()
    sub["row"] = sub.dataset + " / KIR=" + sub.kir.map(lambda value: f"{value:.2f}")
    matrix = sub.pivot(index="row", columns="method", values="mean_rank").reindex(columns=METHODS)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(), cmap="viridis_r", aspect="auto", vmin=1, vmax=len(METHODS))
    ax.set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS], rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if matrix.iloc[i, j] > 4 else "black")
    ax.set_title("Mean OOS F1 rank across five seeds (1 is best)")
    fig.colorbar(im, ax=ax, label="Mean rank")
    fig.savefig(FIG / "method_rank_heatmap_oos_f1.png", dpi=180)
    plt.close(fig)


def plot_seed_curves(frame: pd.DataFrame) -> None:
    selected = ("trainable_k1", "single_centroid", "fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = frame[(frame.dataset == dataset) & frame.method.isin(selected)]
        for method in selected:
            stats = sub[sub.method == method].groupby("kir").oos_f1.agg(["mean", "std"]).reindex(KIRS)
            ax.plot(np.array(KIRS) * 100, stats["mean"] * 100, marker="o", label=LABELS[method], color=COLORS[method])
            ax.fill_between(np.array(KIRS) * 100, (stats["mean"] - stats["std"].fillna(0)) * 100, (stats["mean"] + stats["std"].fillna(0)) * 100, color=COLORS[method], alpha=0.08)
        ax.set_title(dataset)
        ax.set_xlabel("KIR (%)")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Five-seed OOS F1 stability across KIR", y=1.02)
    fig.savefig(FIG / "five_seed_oos_f1_kir_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame = load_source()
    effects = paired_effects(frame)
    rank_per_seed, rank_summary = rankings(frame)
    atomic_csv(effects, OUT / "paired_effects.csv")
    atomic_csv(rank_per_seed, OUT / "rank_per_seed.csv")
    atomic_csv(rank_summary, OUT / "rank_summary.csv")
    plot_forest(effects)
    plot_rank_heatmap(rank_summary)
    plot_seed_curves(frame)
    manifest = {
        "analysis_id": "statistical_stability_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "metrics": list(METRICS),
        "baseline": BASELINE,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "source_rows": int(len(frame)),
        "test_usage": "post-hoc statistical analysis only; no selection or tuning",
        "outputs": ["paired_effects.csv", "rank_per_seed.csv", "rank_summary.csv", "paired_oos_f1_forest.png", "method_rank_heatmap_oos_f1.png", "five_seed_oos_f1_kir_curves.png"],
    }
    atomic_json(manifest, OUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
