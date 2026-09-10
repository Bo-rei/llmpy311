#!/usr/bin/env python3
"""Paired Gate-to-Cascade bridge analysis for completed three-seed rows.

The Trainable row is Gate-only while the comparison rows are Cascade rows.
This script therefore reports a *bridge*, not a unified end-to-end ranking.
It only reads frozen metrics and never changes thresholds or reruns a model.
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
SOURCE = ROOT / "results/analysis/archive/analysis/gate_cascade_bridge_v1/per_seed.csv"
OUT = ROOT / "results/analysis/archive/analysis/gate_cascade_paired_bridge_v2"
FIG = ROOT / "figures/archive/analysis/gate_cascade_paired_bridge_v2"
DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87)
GATE = "trainable_k1_gate"
CASCADE_METHODS = ("frozen_k1", "frozen_selected_k", "ce_recon_selected_k", "best_controlled_baseline")
LABELS = {
    "frozen_k1": "Frozen K=1 Cascade",
    "frozen_selected_k": "Frozen selected-K Cascade",
    "ce_recon_selected_k": "CE-Recon selected-K Cascade",
    "best_controlled_baseline": "Best controlled Cascade",
}
COLORS = {
    "frozen_k1": "#718096",
    "frozen_selected_k": "#dd6b20",
    "ce_recon_selected_k": "#805ad5",
    "best_controlled_baseline": "#319795",
}
BRIDGE_METRICS = ("oos_f1", "id_recall", "oos_false_accept_rate", "known_false_reject_rate")
CASCADE_METRICS = ("oos_f1", "id_recall", "oos_false_accept_rate", "known_false_reject_rate", "router_error_rate", "expert_error_rate", "overall_accuracy", "known_macro_f1")
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
    frame = frame[frame.dataset.isin(DATASETS) & frame.seed.isin(SEEDS)].copy()
    expected = len(DATASETS) * len(SEEDS) * (1 + len(CASCADE_METHODS))
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, got {len(frame)}")
    if frame.duplicated(["dataset", "seed", "gate"]).any():
        raise ValueError("duplicate dataset/seed/gate rows")
    return frame


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    sample = rng.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True)
    means = sample.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pvalue_sign(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positive = int(np.sum(nonzero > 0))
    lower = sum(math.comb(n, i) for i in range(positive + 1)) / 2**n
    upper = sum(math.comb(n, i) for i in range(positive, n + 1)) / 2**n
    return float(min(1.0, 2 * min(lower, upper)))


def bridge_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        base = frame[(frame.dataset == dataset) & (frame.gate == GATE)].set_index("seed")
        for method in CASCADE_METHODS:
            comp = frame[(frame.dataset == dataset) & (frame.gate == method)].set_index("seed")
            for metric in BRIDGE_METRICS:
                if base[metric].isna().any() or comp[metric].isna().any():
                    raise ValueError(f"missing bridge metric {dataset} {method} {metric}")
                raw = base.loc[list(SEEDS), metric].to_numpy() - comp.loc[list(SEEDS), metric].to_numpy()
                better = -raw if metric in {"oos_false_accept_rate", "known_false_reject_rate"} else raw
                low, high = bootstrap_ci(better, rng)
                std = float(np.std(better, ddof=1))
                rows.append({
                    "dataset": dataset,
                    "comparison": method,
                    "comparison_label": LABELS[method],
                    "metric": metric,
                    "n_seeds": len(better),
                    "trainable_gate_minus_cascade": float(np.mean(raw)),
                    "trainable_better_delta": float(np.mean(better)),
                    "std_better_delta": std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": int(np.sum(better > 1e-12)),
                    "ties": int(np.sum(np.abs(better) <= 1e-12)),
                    "losses": int(np.sum(better < -1e-12)),
                    "sign_test_pvalue": pvalue_sign(better),
                    "effect_size": float(np.mean(better) / std) if std > 0 else 0.0,
                })
    return pd.DataFrame(rows)


def cascade_pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in ("frozen_selected_k", "ce_recon_selected_k", "best_controlled_baseline"):
            ref = "frozen_k1"
            left = frame[(frame.dataset == dataset) & (frame.gate == method)].set_index("seed")
            right = frame[(frame.dataset == dataset) & (frame.gate == ref)].set_index("seed")
            for metric in CASCADE_METRICS:
                if left[metric].isna().any() or right[metric].isna().any():
                    continue
                delta = left.loc[list(SEEDS), metric].to_numpy() - right.loc[list(SEEDS), metric].to_numpy()
                low, high = bootstrap_ci(delta, rng)
                rows.append({
                    "dataset": dataset,
                    "comparison": method,
                    "comparison_label": LABELS[method],
                    "reference": ref,
                    "metric": metric,
                    "n_seeds": len(delta),
                    "mean_delta": float(np.mean(delta)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": int(np.sum(delta > 1e-12)),
                    "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                    "losses": int(np.sum(delta < -1e-12)),
                })
    return pd.DataFrame(rows)


def plot_bridge(effects: pd.DataFrame) -> None:
    sub = effects[effects.metric == "oos_f1"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharex=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        group = sub[sub.dataset == dataset].copy().sort_values("comparison")
        y = np.arange(len(group))
        ax.errorbar(group.trainable_better_delta * 100, y, xerr=[(group.trainable_better_delta - group.ci95_low) * 100, (group.ci95_high - group.trainable_better_delta) * 100], fmt="o", color="#2b6cb0", ecolor="#718096", capsize=3)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_yticks(y, group.comparison_label)
        ax.set_title(dataset)
        ax.set_xlabel("Trainable Gate OOS F1 advantage (pp)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Gate-only Trainable K=1 versus three-seed Cascade baselines", y=1.01)
    fig.savefig(FIG / "gate_cascade_oos_f1_bridge_forest.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cascade_tradeoff(frame: pd.DataFrame) -> None:
    sub = frame[frame.gate.isin(CASCADE_METHODS)].groupby(["dataset", "gate"], as_index=False)[["oos_f1", "overall_accuracy", "known_macro_f1", "expert_error_rate"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        group = sub[sub.dataset == dataset]
        for _, row in group.iterrows():
            ax.scatter(row.overall_accuracy * 100, row.oos_f1 * 100, color=COLORS[row.gate], s=65)
            ax.annotate(LABELS[row.gate], (row.overall_accuracy * 100, row.oos_f1 * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(dataset)
        ax.set_xlabel("Cascade Overall Accuracy (%)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Cascade OOS F1 (%)")
    fig.suptitle("Cascade-only trade-off: OOS rejection versus overall classification", y=1.02)
    fig.savefig(FIG / "cascade_only_oos_accuracy_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_decomposition(frame: pd.DataFrame) -> None:
    sub = frame[frame.gate.isin(CASCADE_METHODS)].groupby(["dataset", "gate"], as_index=False)[["router_error_rate", "expert_error_rate", "known_false_reject_rate", "oos_false_accept_rate"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        group = sub[sub.dataset == dataset].set_index("gate").reindex(CASCADE_METHODS)
        x = np.arange(len(group))
        ax.bar(x, group.router_error_rate * 100, label="Router error", color="#4a5568")
        ax.bar(x, group.expert_error_rate * 100, bottom=group.router_error_rate * 100, label="Expert error", color="#805ad5")
        ax.set_xticks(x, ["Frozen\nK=1", "Frozen\nselected", "CE-Recon", "Best\ncontrolled"], rotation=20)
        ax.set_ylabel("Error rate (%)")
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Cascade error decomposition (router + expert; Gate errors shown in bridge CSV)", y=1.03)
    fig.savefig(FIG / "cascade_router_expert_error_decomposition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame = load_source()
    effects = bridge_effects(frame)
    cascade = cascade_pairwise(frame)
    atomic_csv(effects, OUT / "gate_vs_cascade_effects.csv")
    atomic_csv(cascade, OUT / "cascade_pairwise_effects.csv")
    plot_bridge(effects)
    plot_cascade_tradeoff(frame)
    plot_error_decomposition(frame)
    atomic_json({
        "analysis_id": "gate_cascade_paired_bridge_v2",
        "protocol_version": "protocol_v2_textoir_v1",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "gate_only_method": GATE,
        "cascade_methods": list(CASCADE_METHODS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "source_rows": int(len(frame)),
        "warning": "Trainable row is Gate-only; bridge effects are not a unified end-to-end ranking.",
        "outputs": ["gate_vs_cascade_effects.csv", "cascade_pairwise_effects.csv", "gate_cascade_oos_f1_bridge_forest.png", "cascade_only_oos_accuracy_tradeoff.png", "cascade_router_expert_error_decomposition.png"],
    }, OUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
