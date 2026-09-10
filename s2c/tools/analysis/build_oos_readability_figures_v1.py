#!/usr/bin/env python3
"""Build the low-reading-cost OOS-first figure bundle.

The builder consumes frozen analysis tables and completed prediction files only.
It does not train, calibrate, select a threshold, select K, or overwrite an
experiment artifact root.  Test OOS labels are used only for post-hoc grouping
and mechanism annotation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts/s2c/runs/protocol_v2_textoir_v1"
RESULT_DIR = ROOT / "results/analysis/oos_readability_figures_v1"
FIGURE_DIR = ROOT / "figures/oos_readability_figures_v1"

DATASETS = ("clinc150", "banking77", "stackoverflow")
DATASET_LABELS = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}
SEEDS = (13, 42, 87, 100, 123)
MECHANISM_SEEDS = (13, 42, 87)
KIRS = (0.25, 0.50, 0.75)
METHODS = ("frozen", "trainable")
METHOD_LABELS = {"frozen": "Frozen E2 K=1", "trainable": "Trainable K=1"}
METHOD_COLORS = {"frozen": "#4C78A8", "trainable": "#C44E52"}
OOS_COLOR = "#C44E52"
KNOWN_COLOR = "#B8C0CC"
THRESHOLD_COLOR = "#222222"
QUANTILES_PATH = ROOT / "results/analysis/archive/analysis/minilm_boundary_diagnostics_v1/score_quantiles.csv"
SUMMARY_PATH = ROOT / "results/analysis/archive/analysis/minilm_trainable_5seed_fair_v1/trainable_vs_frozen_paired.csv"
TRANSITIONS_PATH = ROOT / "results/analysis/deep_geometry_mechanism_v1/movement_transition_summary.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_scores() -> pd.DataFrame:
    frame = pd.read_csv(QUANTILES_PATH)
    required = {"dataset", "kir", "seed", "representation", "gold_is_oos", "n", "score_p10", "score_p50", "score_p90", "score_p95", "score_p99", "score_leq_1_rate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"score quantile table is missing columns: {sorted(missing)}")
    subset = frame[frame["kir"].eq(0.50)].copy()
    if set(subset["dataset"]) != set(DATASETS) or set(subset["representation"]) != set(METHODS):
        raise ValueError("central-KIR score table does not cover the declared datasets and methods")
    return subset


def load_paired_summary() -> pd.DataFrame:
    frame = pd.read_csv(SUMMARY_PATH)
    expected = {"dataset", "kir", "seed", "oos_f1_trainable", "known_recall_trainable", "false_accept_rate_trainable", "auroc_trainable", "oos_f1_frozen", "known_recall_frozen", "false_accept_rate_frozen", "auroc_frozen"}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"paired table is missing columns: {sorted(missing)}")
    subset = frame[frame["kir"].isin(KIRS)].copy()
    if len(subset) != len(DATASETS) * len(KIRS) * len(SEEDS):
        raise ValueError(f"paired table has {len(subset)} rows, expected 45")
    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        for method in METHODS:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "kir": row["kir"],
                    "seed": row["seed"],
                    "representation": method,
                    "oos_f1": row[f"oos_f1_{method}"],
                    "known_recall": row[f"known_recall_{method}"],
                    "false_accept_rate": row[f"false_accept_rate_{method}"],
                    "auroc": row[f"auroc_{method}"],
                }
            )
    return pd.DataFrame(rows)


def load_transitions() -> pd.DataFrame:
    frame = pd.read_csv(TRANSITIONS_PATH)
    required = {"dataset", "seed", "comparison", "gold_type", "transition", "count", "rate_within_gold_type", "mean_delta_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"transition table is missing columns: {sorted(missing)}")
    subset = frame[(frame["comparison"] == "trainable_vs_frozen") & (frame["gold_type"] == "OOS")].copy()
    required_transitions = {"both_correct", "trainable_only_correct", "baseline_only_correct", "both_wrong"}
    if set(subset["dataset"]) != set(DATASETS) or set(subset["transition"]) != required_transitions:
        raise ValueError("OOS transition table does not cover the declared panels")
    return subset


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": FIGURE_DIR / f"{stem}.png",
        "tiff": FIGURE_DIR / f"{stem}.tiff",
        "svg": FIGURE_DIR / f"{stem}.svg",
        "pdf": FIGURE_DIR / f"{stem}.pdf",
    }
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs["pdf"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {kind: str(path.relative_to(ROOT)) for kind, path in outputs.items()}


def plot_score_distribution(scores: pd.DataFrame) -> dict[str, str]:
    """Show the OOS score mass and a quiet Known reference at KIR=.50."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2047, 2.44), sharey=True)
    x = np.array([0.10, 0.50, 0.90, 0.95, 0.99])
    for ax, dataset in zip(axes, DATASETS):
        panel = scores[scores["dataset"] == dataset]
        for method in METHODS:
            oos = panel[(panel["representation"] == method) & (panel["gold_is_oos"] == 1)]
            known = panel[(panel["representation"] == method) & (panel["gold_is_oos"] == 0)]
            # The five seed quantiles form a compact uncertainty ribbon rather than
            # a dense histogram.  All observations remain in the source table.
            oos_q = oos[["score_p10", "score_p50", "score_p90", "score_p95", "score_p99"]].to_numpy(dtype=float)
            known_q = known[["score_p10", "score_p50", "score_p90", "score_p95", "score_p99"]].to_numpy(dtype=float)
            oos_center = np.nanmedian(oos_q, axis=0)
            known_center = np.nanmedian(known_q, axis=0)
            oos_low = np.nanmin(oos_q, axis=0)
            oos_high = np.nanmax(oos_q, axis=0)
            ax.plot(x, oos_center, color=METHOD_COLORS[method], linewidth=2.0, marker="o", markersize=3.8, label=METHOD_LABELS[method])
            ax.fill_between(x, oos_low, oos_high, color=METHOD_COLORS[method], alpha=0.10, linewidth=0)
            # Known is a low-contrast guard, not a second signal family.
            ax.plot(x, known_center, color=KNOWN_COLOR, linewidth=1.1, linestyle="--", marker="", alpha=0.9)
        ax.axhline(1.0, color=THRESHOLD_COLOR, linestyle=(0, (4, 3)), linewidth=1.0)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks([0.10, 0.50, 0.90, 0.99], ["10", "50", "90", "99"])
        ax.set_xlabel("score percentile")
        ax.set_ylim(0.55, 1.85)
        ax.grid(axis="y", color="#E6E8EB", linewidth=0.7)
        panel_oos = panel[panel["gold_is_oos"] == 1]
        fa = panel_oos.groupby("representation")["score_leq_1_rate"].mean()
        ax.text(0.03, 0.05, f"FA {fa['frozen']:.1%} → {fa['trainable']:.1%}", transform=ax.transAxes, fontsize=7, color="#444444")
    axes[0].set_ylabel("normalized OOS score")
    axes[-1].legend(loc="upper left", frameon=False, handlelength=2.0)
    fig.suptitle("Trainable shifts OOS scores above the rejection boundary", y=1.02, fontsize=10)
    fig.text(0.01, -0.02, "Colored lines: OOS; light dashed line: Known reference; black dashed line: score = 1", fontsize=7, color="#555555")
    fig.tight_layout()
    return save_figure(fig, "oos_score_distribution_kir050")


def plot_operating_point(summary: pd.DataFrame) -> dict[str, str]:
    """Show only the OOS work point; Known Recall is a text-side guard."""
    means = summary.groupby(["dataset", "kir", "representation"], as_index=False)[["oos_f1", "false_accept_rate", "known_recall"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.2047, 2.44), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        panel = means[means["dataset"] == dataset]
        for method in METHODS:
            part = panel[panel["representation"] == method].sort_values("kir")
            ax.plot(part["false_accept_rate"] * 100, part["oos_f1"] * 100, color=METHOD_COLORS[method], marker="o", linewidth=1.8, markersize=4, label=METHOD_LABELS[method])
            for _, row in part.iterrows():
                ax.annotate(f"{row.kir:.2f}", (row.false_accept_rate * 100, row.oos_f1 * 100), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=METHOD_COLORS[method])
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("OOS false acceptance (%)")
        ax.set_xlim(left=0)
        ax.set_ylim(45, 100)
        ax.grid(color="#E6E8EB", linewidth=0.7)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(frameon=False, loc="lower left")
    fig.suptitle("Trainable improves the OOS work point across KIR", y=1.02, fontsize=10)
    fig.text(0.01, -0.02, "Each point is the five-seed mean; KIR labels are printed beside the points. Known Recall remains a coverage guard in the source table.", fontsize=7, color="#555555")
    fig.tight_layout()
    return save_figure(fig, "oos_operating_point_kir")


def plot_mechanism(transitions: pd.DataFrame) -> dict[str, str]:
    """Show the OOS correctness transition and its score movement."""
    order = ["both_correct", "trainable_only_correct", "baseline_only_correct", "both_wrong"]
    labels = ["both\ncorrect", "Trainable\nonly", "Frozen\nonly", "both\nwrong"]
    colors = ["#9EA4AC", OOS_COLOR, "#7B8794", "#D7DADF"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2047, 2.44), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        panel = transitions[transitions["dataset"] == dataset]
        rates = panel.groupby("transition")["rate_within_gold_type"].mean().reindex(order)
        delta = panel.groupby("transition")["mean_delta_score"].mean().reindex(order)
        ax.bar(np.arange(len(order)), rates.to_numpy() * 100, color=colors, width=0.68)
        ax.set_xticks(np.arange(len(order)), labels)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_ylim(0, 105)
        ax.set_xlabel("OOS correctness transition")
        ax.grid(axis="y", color="#E6E8EB", linewidth=0.7)
        trainable_only = rates["trainable_only_correct"]
        mean_delta = delta["trainable_only_correct"]
        ax.text(
            0.03,
            0.96,
            f"Trainable-only {trainable_only:.1%}\nΔscore +{mean_delta:.3f}",
            transform=ax.transAxes,
            fontsize=7,
            color=OOS_COLOR,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )
    axes[0].set_ylabel("share of OOS samples (%)")
    fig.suptitle("Trainable-only OOS corrections coincide with higher scores", y=1.02, fontsize=10)
    fig.text(0.01, -0.02, "Rates and mean Δscore are three-seed means at KIR=.50; Δscore = Trainable − Frozen.", fontsize=7, color="#555555")
    fig.tight_layout()
    return save_figure(fig, "oos_mechanism_transitions_kir050")


def write_summary(scores: pd.DataFrame, summary: pd.DataFrame, transitions: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    score_summary = scores[scores["gold_is_oos"] == 1].groupby(["dataset", "representation"], as_index=False)[["n", "score_p10", "score_p50", "score_p90", "score_p95", "score_p99", "score_leq_1_rate"]].mean()
    score_summary.to_csv(RESULT_DIR / "oos_score_summary_kir050.csv", index=False)
    summary.groupby(["dataset", "kir", "representation"], as_index=False)[["oos_f1", "false_accept_rate", "known_recall", "auroc"]].mean().to_csv(RESULT_DIR / "oos_operating_point_summary.csv", index=False)
    transitions.groupby(["dataset", "transition"], as_index=False)[["rate_within_gold_type", "mean_delta_score"]].mean().to_csv(RESULT_DIR / "oos_transition_summary_kir050.csv", index=False)


def write_manifest(figures: dict[str, dict[str, str]]) -> None:
    source_paths = [QUANTILES_PATH, SUMMARY_PATH, TRANSITIONS_PATH, Path(__file__)]
    output_paths = [RESULT_DIR / name for name in ("oos_score_summary_kir050.csv", "oos_operating_point_summary.csv", "oos_transition_summary_kir050.csv")]
    for group in figures.values():
        output_paths.extend(ROOT / path for path in group.values())
    figure_paths = [path for group in figures.values() for path in group.values()]
    source_paths_relative = [str(path.relative_to(ROOT)) for path in source_paths]
    manifest = {
        "analysis_id": "oos_readability_figures_v1",
        "analysis_stage": "ANALYSIS_OOS_READABILITY_FIGURES_V1",
        "protocol_version": "protocol_v2_textoir_v1",
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "central_kir_for_score_and_mechanism": 0.50,
        "seeds": list(SEEDS),
        "mechanism_seeds": list(MECHANISM_SEEDS),
        "methods": [METHOD_LABELS[method] for method in METHODS],
        "k_values": [1],
        "distances": ["mahalanobis_diag"],
        "selection_used_test_oos": False,
        "test_labels_used_post_hoc": True,
        "oos_used_for_training": False,
        "threshold": 1.0,
        "threshold_semantics": "normalized score > 1 rejects as OOS",
        "figure_count": 3,
        "figure_export_formats": ["png", "tiff", "svg", "pdf"],
        "sources": source_paths_relative,
        "figures": figure_paths,
        "figure_contract": [
            {"figure_id": "oos_score_distribution", "question": "Trainable 是否把 OOS score 推向拒绝侧？", "population": "test OOS plus quiet Known reference at KIR=.50", "primary_metric": "oos_score", "guard_metric": "false_acceptance", "output": figures["score"]},
            {"figure_id": "oos_operating_point", "question": "Trainable 的 OOS F1—false acceptance 工作点是否跨 KIR 稳定改善？", "population": "test metrics, 5-seed means", "primary_metric": "oos_f1", "guard_metric": "known_recall", "output": figures["operating_point"]},
            {"figure_id": "oos_mechanism_transitions", "question": "Trainable-only OOS 正确是否伴随拒绝侧 score 移动？", "population": "test OOS, KIR=.50, seeds=13/42/87, post-hoc state transitions", "primary_metric": "trainable_only_correct_rate", "guard_metric": "both_wrong_rate", "output": figures["mechanism"]},
        ],
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "builder_sha256": sha256(Path(__file__)),
        "output_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
        "outputs": {
            "score_summary": "results/analysis/oos_readability_figures_v1/oos_score_summary_kir050.csv",
            "operating_point_summary": "results/analysis/oos_readability_figures_v1/oos_operating_point_summary.csv",
            "transition_summary": "results/analysis/oos_readability_figures_v1/oos_transition_summary_kir050.csv",
            "figures": [path for group in figures.values() for path in group.values()],
        },
    }
    (RESULT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    scores = load_scores()
    summary = load_paired_summary()
    transitions = load_transitions()
    write_summary(scores, summary, transitions)
    figures = {
        "score": plot_score_distribution(scores),
        "operating_point": plot_operating_point(summary),
        "mechanism": plot_mechanism(transitions),
    }
    write_manifest(figures)
    print(json.dumps({"status": "complete", "figures": figures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
