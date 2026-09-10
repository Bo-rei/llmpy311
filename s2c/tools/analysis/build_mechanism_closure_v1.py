#!/usr/bin/env python3
"""Build an analysis-only closure package for the current S2C comparisons.

This script deliberately reads already frozen summaries.  It does not train,
select a threshold, choose K, or rerun an external baseline.  Its purpose is
to answer one practical question: why does the current Trainable-K1 candidate
look better than the fair frozen/MOGB components, and where is that claim not
valid because the contracts differ?
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2c-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FAIR = ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
TRANSITIONS = ROOT / "results/analysis/archive/analysis/trainable_mogb_open_intent_transitions_v1/cell_decomposition_summary.csv"
MOGB_RISK = ROOT / "results/analysis/archive/analysis/mogb_ball_risk_attribution_v1/dataset_kir_summary.csv"
KIR = ROOT / "results/analysis/archive/analysis/kir_sensitivity_decomposition_v1/kir_sensitivity_summary.csv"
BUILDER = Path(__file__).resolve()
OUT = ROOT / "results/analysis/archive/analysis/mechanism_closure_v1"
FIG = ROOT / "figures/archive/analysis/mechanism_closure_v1"
REPORT = ROOT / "docs/archive/analysis/MECHANISM_CLOSURE_V1.md"

_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
if _FONT.is_file():
    font_manager.fontManager.addfont(str(_FONT))
    _FAMILY = font_manager.FontProperties(fname=str(_FONT)).get_name()
else:
    _FAMILY = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = [_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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
    "trainable_k1": "S2C Trainable K=1",
    "single_centroid": "S2C Frozen K=1",
    "fixed_k2": "S2C Frozen K=2",
    "random_partition": "S2C Random K=2",
    "mogb_partition_ours_boundary": "MOGB partition + S2C boundary",
    "ours_partition_mogb_boundary": "S2C partition + MOGB boundary",
    "mogb_minilm": "MOGB-MiniLM",
}
DATASET_LABELS = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}
COLORS = {
    "trainable_k1": "#0072B2",
    "single_centroid": "#777777",
    "fixed_k2": "#D55E00",
    "random_partition": "#E69F00",
    "mogb_partition_ours_boundary": "#009E73",
    "ours_partition_mogb_boundary": "#56B4E9",
    "mogb_minilm": "#CC79A7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(temporary, format=path.suffix.lstrip("."), dpi=180, bbox_inches="tight")
    plt.close(fig)
    os.replace(temporary, path)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fair = pd.read_csv(FAIR)
    required = {"dataset", "kir", "method", "oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate", "n_seeds"}
    missing = sorted(required - set(fair.columns))
    if missing:
        raise ValueError(f"fair summary missing columns: {missing}")
    fair = fair[fair.dataset.isin(DATASETS) & fair.method.isin(METHODS)].copy()
    fair["kir"] = fair["kir"].astype(float)
    expected = len(DATASETS) * len(KIRS) * len(METHODS)
    if len(fair) != expected or set(fair.n_seeds.astype(int)) != {5}:
        raise ValueError(f"fair matrix is not the expected 63 five-seed rows: {len(fair)}")
    if not np.isfinite(fair[["oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate"]].to_numpy()).all():
        raise ValueError("fair matrix contains non-finite values")

    transitions = pd.read_csv(TRANSITIONS)
    transition_required = {"dataset", "kir", "net_known_correct_gain_mean", "net_oos_correct_gain_mean", "net_known_correct_gain_rate", "net_oos_correct_gain_rate", "f1_all_delta_mean", "n_seeds"}
    missing = sorted(transition_required - set(transitions.columns))
    if missing:
        raise ValueError(f"transition summary missing columns: {missing}")
    transitions["kir"] = transitions["kir"].astype(float)
    if len(transitions) != len(DATASETS) * len(KIRS):
        raise ValueError("transition summary does not contain all dataset/KIR cells")

    risk = pd.read_csv(MOGB_RISK)
    risk_required = {"dataset", "kir", "method", "selected_balls_mean", "tiny_support_lt20_ratio_mean", "false_accept_rate_mean", "false_reject_rate_mean", "known_recall_mean", "oos_f1_mean", "f1_all_mean"}
    missing = sorted(risk_required - set(risk.columns))
    if missing:
        raise ValueError(f"MOGB risk summary missing columns: {missing}")
    risk["kir"] = risk["kir"].astype(float)
    risk = risk[risk.method.isin(("mogb_minilm", "mogb_partition_ours_boundary", "ours_partition_mogb_boundary"))].copy()

    kir = pd.read_csv(KIR)
    kir_required = {"dataset", "method", "oos_f1_delta_pp", "oos_f1_slope_pp_per_kir", "f1_all_delta_pp", "known_recall_delta_pp", "false_accept_rate_delta_pp"}
    missing = sorted(kir_required - set(kir.columns))
    if missing:
        raise ValueError(f"KIR sensitivity summary missing columns: {missing}")
    return fair, transitions, risk, kir


def build_outputs(fair: pd.DataFrame, transitions: pd.DataFrame, risk: pd.DataFrame, kir: pd.DataFrame) -> dict[str, pd.DataFrame]:
    performance = fair.copy()
    performance["dataset_label"] = performance["dataset"].map(DATASET_LABELS)
    performance["method_label"] = performance["method"].map(LABELS)
    for metric in ("oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate", "accuracy", "auroc", "aupr_oos"):
        if metric in performance:
            performance[f"{metric}_pct"] = performance[metric] * 100

    transition = transitions.copy()
    transition["dataset_label"] = transition["dataset"].map(DATASET_LABELS)
    transition["f1_all_delta_pp"] = transition["f1_all_delta_mean"] * 100
    transition["known_gain_pct_of_known"] = transition["net_known_correct_gain_rate"] * 100
    transition["oos_gain_pct_of_oos"] = transition["net_oos_correct_gain_rate"] * 100
    transition["net_oos_correct_loss_mean"] = -transition["net_oos_correct_gain_mean"]

    risk_out = risk.copy()
    risk_out["dataset_label"] = risk_out["dataset"].map(DATASET_LABELS)
    risk_out["method_label"] = risk_out["method"].map(LABELS)
    for metric in ("oos_f1_mean", "f1_all_mean", "known_recall_mean", "false_accept_rate_mean", "false_reject_rate_mean", "tiny_support_lt20_ratio_mean"):
        risk_out[f"{metric}_pct"] = risk_out[metric] * 100

    kir_out = kir.copy()
    kir_out["dataset_label"] = kir_out["dataset"].map(DATASET_LABELS)
    return {
        "method_kir_performance": performance,
        "trainable_mogb_transitions": transition,
        "mogb_risk_summary": risk_out,
        "kir_robustness": kir_out,
    }


def plot_performance(performance: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    xlabels = [f"{DATASET_LABELS[d]}\nKIR={k:.2f}" for d in DATASETS for k in KIRS]
    x = np.arange(len(xlabels))
    for ax, metric, title in zip(axes, ("oos_f1_pct", "f1_all_pct", "known_recall_pct"), ("OOS F1", "F1-All", "Known Recall")):
        for method in METHODS:
            vals = []
            for dataset in DATASETS:
                for kir in KIRS:
                    row = performance[(performance.dataset == dataset) & (performance.kir == kir) & (performance.method == method)]
                    vals.append(float(row.iloc[0][metric]))
            ax.plot(x, vals, marker="o", linewidth=1.6, markersize=3.8, color=COLORS[method], label=LABELS[method])
        ax.set_ylabel(f"{title} (%)")
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 100)
    axes[-1].set_xticks(x, xlabels, rotation=0, fontsize=8)
    axes[0].legend(ncol=3, fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Current fair matrix: performance and coverage across datasets/KIR", fontsize=14)
    path = FIG / "performance_operating_curves.png"
    save_figure(fig, path)
    return path


def plot_error_budget(transition: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    markers = {"clinc150": "o", "banking77": "s", "stackoverflow": "^"}
    for _, row in transition.iterrows():
        x = float(row["known_gain_pct_of_known"])
        y = float(row["oos_gain_pct_of_oos"])
        ax.scatter(x, y, s=70, marker=markers[row.dataset], color="#0072B2", edgecolor="white", linewidth=0.7)
        ax.annotate(f"{DATASET_LABELS[row.dataset]}\n{row.kir:.2f}", (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_xlabel("Trainable K=1 net Known-correct gain (% of Known)")
    ax.set_ylabel("Trainable K=1 net OOS-correct gain (% of OOS)")
    ax.set_title("Trainable vs MOGB-Fair: correctness budget (post-hoc diagnostic)")
    ax.text(0.02, 0.03, "右侧/下方：Known 恢复伴随少量 OOS 正确数损失\n不代表测试集选择，仅用于解释 F1-All 差异", transform=ax.transAxes, fontsize=8)
    ax.grid(alpha=0.25)
    path = FIG / "trainable_mogb_error_budget.png"
    save_figure(fig, path)
    return path


def plot_mogb_risk(risk: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for method, group in risk.groupby("method", sort=False):
        axes[0].scatter(group["false_accept_rate_mean_pct"], group["false_reject_rate_mean_pct"], s=55, label=LABELS[method], color=COLORS.get(method, "#333333"))
        axes[1].scatter(group["selected_balls_mean"], group["tiny_support_lt20_ratio_mean_pct"], s=55, label=LABELS[method], color=COLORS.get(method, "#333333"))
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].set_xlabel("False acceptance (%)")
    axes[0].set_ylabel("False rejection (%)")
    axes[0].set_title("MOGB component workpoints")
    axes[1].set_xlabel("Selected balls (mean)")
    axes[1].set_ylabel("Tiny-support balls <20 (%)")
    axes[1].set_title("Ball count and support risk")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("MOGB attribution: conservative rejection versus coverage loss", fontsize=13)
    path = FIG / "mogb_risk_workpoints.png"
    save_figure(fig, path)
    return path


def plot_dashboard(performance: pd.DataFrame, transition: pd.DataFrame, risk: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    # Panel 1: mean OOS F1 by dataset.
    panel = performance.groupby(["dataset", "method"], as_index=False)["oos_f1_pct"].mean()
    for method in METHODS:
        vals = [float(panel[(panel.dataset == d) & (panel.method == method)].iloc[0].oos_f1_pct) for d in DATASETS]
        axes[0, 0].plot(DATASETS, vals, marker="o", label=LABELS[method], color=COLORS[method])
    axes[0, 0].set_title("Mean OOS F1 by dataset")
    axes[0, 0].set_ylabel("OOS F1 (%)")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].tick_params(axis="x", rotation=15)
    # Panel 2: F1-All versus Known Recall.
    for method in METHODS:
        sub = performance[performance.method == method]
        axes[0, 1].scatter(sub.known_recall_pct, sub.f1_all_pct, s=35, color=COLORS[method], label=LABELS[method], alpha=0.82)
    axes[0, 1].set_xlabel("Known Recall (%)")
    axes[0, 1].set_ylabel("F1-All (%)")
    axes[0, 1].set_title("Coverage-quality frontier")
    # Panel 3: transition budget.
    axes[1, 0].scatter(transition.known_gain_pct_of_known, transition.oos_gain_pct_of_oos, s=38, color="#0072B2")
    axes[1, 0].axhline(0, color="#555555", linewidth=0.8)
    axes[1, 0].axvline(0, color="#555555", linewidth=0.8)
    axes[1, 0].set_xlabel("Known-correct gain (% of Known)")
    axes[1, 0].set_ylabel("OOS-correct gain (% of OOS)")
    axes[1, 0].set_title("Trainable-MOGB correctness budget")
    # Panel 4: MOGB workpoints.
    for method, group in risk.groupby("method", sort=False):
        axes[1, 1].scatter(group.false_accept_rate_mean_pct, group.false_reject_rate_mean_pct, s=42, label=LABELS[method], color=COLORS.get(method, "#333333"))
    axes[1, 1].set_xlabel("False acceptance (%)")
    axes[1, 1].set_ylabel("False rejection (%)")
    axes[1, 1].set_title("MOGB boundary/partition operating points")
    for ax in axes.flat:
        ax.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False, loc="lower left")
    fig.suptitle("S2C mechanism closure: what improves, and what does not", fontsize=15)
    path = FIG / "mechanism_closure_dashboard.png"
    save_figure(fig, path)
    return path


def main() -> None:
    fair, transitions, risk, kir = load_inputs()
    outputs = build_outputs(fair, transitions, risk, kir)
    OUT.mkdir(parents=True, exist_ok=True)
    save_csv(outputs["method_kir_performance"], OUT / "method_kir_performance.csv")
    save_csv(outputs["trainable_mogb_transitions"], OUT / "trainable_mogb_transition_summary.csv")
    save_csv(outputs["mogb_risk_summary"], OUT / "mogb_risk_summary.csv")
    save_csv(outputs["kir_robustness"], OUT / "kir_robustness.csv")
    figures = [
        plot_performance(outputs["method_kir_performance"]),
        plot_error_budget(outputs["trainable_mogb_transitions"]),
        plot_mogb_risk(outputs["mogb_risk_summary"]),
        plot_dashboard(outputs["method_kir_performance"], outputs["trainable_mogb_transitions"], outputs["mogb_risk_summary"]),
    ]
    manifest = {
        "schema": 1,
        "analysis_only": True,
        "protocol_version": "protocol_v2_textoir_v1",
        "builder_sha256": sha256(BUILDER),
        "source_files": {str(path.relative_to(ROOT)): sha256(path) for path in (FAIR, TRANSITIONS, MOGB_RISK, KIR)},
        "outputs": [str(path.relative_to(ROOT)) for path in sorted(OUT.glob("*.csv"))],
        "figures": [str(path.relative_to(ROOT)) for path in figures],
        "rows": {name: int(len(frame)) for name, frame in outputs.items()},
        "notes": [
            "All values are post-hoc summaries of frozen runs; no test result was used to choose a model, K, threshold, or radius.",
            "Trainable K=1 versus MOGB-Fair transitions explain F1-All/coverage tradeoffs and are not a causal claim.",
            "External ADB/DCLOOS and historical full Cascade are intentionally not appended to the fair matrix.",
        ],
    }
    save_json(manifest, OUT / "MANIFEST.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
