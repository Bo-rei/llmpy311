#!/usr/bin/env python3
"""Build contract-aware, OOS-first comparison figures.

The figures reuse completed lightweight summaries.  They deliberately keep
same-protocol MiniLM experiments, TextOIR-native compatibility, external
BERT/DA-ADB results, and DCLOOS evidence-status rows in separate visual
layers.  No model is trained and no threshold or checkpoint is selected here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2c-matplotlib")

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
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/analysis/cross_method_oos_mechanism"
FIG = ROOT / "figures/cross_method_oos_mechanism"

FAIR_PATH = ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
TEXT_OIR_PATH = ROOT / "../artifacts/s2c/outputs/experiments/cluster_separability_v19/textoir_protocol/official_runs/textoir_results_by_seed.csv"
ADB_PATH = ROOT / "results/analysis/adb_kir_sensitivity_v2/adb_summary.csv"
DA_ADB_PATH = ROOT / "results/analysis/da_adb_current_protocol_summary_v1/summary_mean_std_ci.csv"
MOGB_BALL_PATH = ROOT / "results/analysis/deep_geometry_mechanism_v1/mogb_ball_risk_summary.csv"
TRANSITION_PATH = ROOT / "results/analysis/deep_geometry_mechanism_v1/multi_method_error_transitions.csv"
BLOCKED_PATH = ROOT / "results/analysis/unified_prediction_contract_v1/blocked_methods.csv"
DCLOOS_METRICS = ROOT / "../artifacts/s2c/external/dcloos_official_oos_kir75_seed888_reduced_v2/recovery_metrics.json"

DATASETS = ("clinc150", "banking77", "stackoverflow")
DATASET_LABELS = {
    "clinc150": "CLINC150",
    "banking77": "Banking77",
    "stackoverflow": "StackOverflow",
}
TEXT_OIR_DATASET_LABELS = {
    "oos": "CLINC150",
    "banking": "Banking source key",
    "stackoverflow": "StackOverflow",
}
KIRS = (0.25, 0.50, 0.75)

# Visual vocabulary follows the user's reference figure: Known blue circles,
# OOS red crosses, yellow stars for the proposed Trainable S2C point, gray
# dashed guides, and a white canvas with black axes.
BLUE = "#4C78A8"
RED = "#C44E52"
YELLOW = "#F1B82D"
GRAY = "#8E969E"
LIGHT_GRAY = "#C5CBD1"
DARK = "#252525"
PURPLE = "#8C6D9C"
BROWN = "#9A7655"
GREEN = "#6A9F7B"

FAIR_LABELS = {
    "trainable_k1": "S2C Trainable K=1",
    "single_centroid": "Frozen single",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_partition_ours_boundary": "MOGB partition + S2C",
    "ours_partition_mogb_boundary": "S2C partition + MOGB",
    "mogb_minilm": "MOGB-MiniLM",
}
FAIR_ORDER = tuple(FAIR_LABELS)
FAIR_STYLE = {
    "trainable_k1": {"marker": "*", "color": YELLOW, "edge": DARK, "size": 150, "zorder": 6},
    "single_centroid": {"marker": "o", "color": BLUE, "edge": "white", "size": 45, "zorder": 3},
    "fixed_k2": {"marker": "o", "color": "white", "edge": GRAY, "size": 50, "zorder": 3},
    "random_partition": {"marker": "D", "color": BROWN, "edge": "white", "size": 42, "zorder": 3},
    "mogb_partition_ours_boundary": {"marker": "^", "color": GREEN, "edge": "white", "size": 50, "zorder": 4},
    "ours_partition_mogb_boundary": {"marker": "v", "color": PURPLE, "edge": "white", "size": 50, "zorder": 4},
    "mogb_minilm": {"marker": "x", "color": RED, "edge": RED, "size": 65, "zorder": 5},
}


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return frame


def _oos_precision(f1: pd.Series, recall: pd.Series) -> pd.Series:
    denominator = 2.0 * recall - f1
    result = f1 * recall / denominator.where(denominator > 1e-12)
    return result.clip(0.0, 1.0).fillna(0.0)


def _style(ax: plt.Axes, grid: str | None = None) -> None:
    ax.tick_params(length=2.8, width=0.65, pad=2)
    if grid:
        ax.grid(axis=grid, color="#E7E9EC", linewidth=0.45)
        ax.set_axisbelow(True)


def _save(fig: plt.Figure, stem: str) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    png_path = FIG / f"{stem}.png"
    tiff_path = FIG / f"{stem}.tiff"
    pdf_path = FIG / f"{stem}.pdf"
    svg_path = FIG / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    outputs = [str(path.relative_to(ROOT)) for path in (png_path, tiff_path, pdf_path, svg_path)]
    plt.close(fig)
    return outputs


def _fair_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _read_csv(
        FAIR_PATH,
        {
            "dataset",
            "kir",
            "method",
            "oos_f1",
            "f1_all",
            "known_recall",
            "false_accept_rate",
            "false_reject_rate",
            "n_seeds",
        },
    ).copy()
    frame["oos_recall"] = 1.0 - frame["false_accept_rate"]
    frame["oos_precision"] = _oos_precision(frame["oos_f1"], frame["oos_recall"])
    frame["contract_layer"] = "same_protocol_fair"
    frame["backbone"] = "MiniLM"
    frame["supervision"] = "Known-only"
    frame["method_label"] = frame["method"].map(FAIR_LABELS)
    frame50 = frame[np.isclose(frame["kir"], 0.50)].copy()
    frame50["dataset_label"] = frame50["dataset"].map(DATASET_LABELS)
    frame50.to_csv(OUT / "fair_kir50_summary.csv", index=False)
    return frame, frame50


def _textoir_data() -> pd.DataFrame:
    frame = _read_csv(
        TEXT_OIR_PATH,
        {"dataset", "method", "known_cls_ratio", "seed", "accuracy", "known_macro_f1", "open_oos_f1", "macro_f1"},
    ).copy()
    frame = frame[frame["method"].isin(("MSP", "DOC", "ADB"))].copy()
    grouped = (
        frame.groupby(["dataset", "method", "known_cls_ratio"], as_index=False)
        .agg(
            oos_f1=("open_oos_f1", "mean"),
            oos_f1_std=("open_oos_f1", "std"),
            accuracy=("accuracy", "mean"),
            known_f1=("known_macro_f1", "mean"),
            f1_all=("macro_f1", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    grouped["dataset_label"] = grouped["dataset"].map(TEXT_OIR_DATASET_LABELS)
    grouped["contract_layer"] = "textoir_native_compatibility"
    grouped["backbone"] = "BERT/TextOIR"
    grouped["supervision"] = "Known-only"
    grouped.to_csv(OUT / "textoir_native_summary.csv", index=False)
    return grouped


def _adb_data(fair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    adb = _read_csv(
        ADB_PATH,
        {
            "dataset",
            "kir",
            "method",
            "n_seeds",
            "oos_f1_mean",
            "oos_f1_std",
            "known_recall_mean",
            "false_accept_rate_mean",
            "false_reject_rate_mean",
        },
    ).copy()
    adb = adb[adb["method"].eq("ADB")].copy()
    adb = adb.rename(
        columns={
            "oos_f1_mean": "oos_f1",
            "oos_f1_std": "oos_f1_std",
            "known_recall_mean": "known_recall",
            "false_accept_rate_mean": "false_accept_rate",
            "false_reject_rate_mean": "false_reject_rate",
        }
    )
    adb["oos_recall"] = 1.0 - adb["false_accept_rate"]
    adb["oos_precision"] = _oos_precision(adb["oos_f1"], adb["oos_recall"])
    adb["method_label"] = "ADB"
    adb["contract_layer"] = "external_backbone_reference"
    adb["backbone"] = "BERT/TextOIR"
    adb["supervision"] = "Known-only"
    adb.to_csv(OUT / "adb_current_summary.csv", index=False)

    s2c = fair[fair["method"].eq("trainable_k1")][
        ["dataset", "kir", "oos_f1", "oos_f1_std", "known_recall", "false_accept_rate", "false_reject_rate", "oos_recall", "oos_precision", "n_seeds"]
    ].copy()
    s2c["method"] = "S2C-Trainable-K1"
    s2c["method_label"] = "S2C Trainable K=1"
    s2c["contract_layer"] = "same_protocol_fair"
    s2c["backbone"] = "MiniLM"
    s2c["supervision"] = "Known-only"
    pair = pd.concat([s2c, adb], ignore_index=True, sort=False)
    pair.to_csv(OUT / "s2c_vs_adb_kir_summary.csv", index=False)

    rows: list[dict[str, Any]] = []
    for (dataset, kir), group in pair.groupby(["dataset", "kir"], sort=True):
        s2c_row = group[group["method"].eq("S2C-Trainable-K1")].iloc[0]
        adb_row = group[group["method"].eq("ADB")].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "kir": kir,
                "delta_oos_f1": float(s2c_row["oos_f1"] - adb_row["oos_f1"]),
                "delta_known_recall": float(s2c_row["known_recall"] - adb_row["known_recall"]),
                "delta_false_accept_rate": float(s2c_row["false_accept_rate"] - adb_row["false_accept_rate"]),
                "delta_false_reject_rate": float(s2c_row["false_reject_rate"] - adb_row["false_reject_rate"]),
                "n_s2c_seeds": int(s2c_row["n_seeds"]),
                "n_adb_seeds": int(adb_row["n_seeds"]),
            }
        )
    deltas = pd.DataFrame(rows)
    deltas.to_csv(OUT / "s2c_vs_adb_kir_deltas.csv", index=False)
    return pair, deltas


def _external_data(adb_pair: pd.DataFrame) -> pd.DataFrame:
    da = _read_csv(
        DA_ADB_PATH,
        {
            "method",
            "oos_f1_mean",
            "oos_f1_std",
            "f1_all_mean",
            "accuracy_mean",
            "known_recall_mean",
            "false_acceptance_mean",
            "false_rejection_mean",
            "n_seeds",
        },
    ).copy()
    da_rows: list[dict[str, Any]] = []
    for _, row in da.iterrows():
        da_rows.append(
            {
                "method": str(row["method"]),
                "dataset": "stackoverflow",
                "kir": 0.50,
                "oos_f1": float(row["oos_f1_mean"]),
                "oos_f1_std": float(row["oos_f1_std"]),
                "f1_all": float(row["f1_all_mean"]),
                "accuracy": float(row["accuracy_mean"]),
                "known_recall": float(row["known_recall_mean"]),
                "false_accept_rate": float(row["false_acceptance_mean"]),
                "false_reject_rate": float(row["false_rejection_mean"]),
                "n_seeds": int(row["n_seeds"]),
                "backbone": "MiniLM" if str(row["method"]).startswith("S2C") else "BERT/TextOIR",
                "contract_layer": "same_data_external_compatibility",
            }
        )
    adb50 = adb_pair[(adb_pair["method"].eq("ADB")) & np.isclose(adb_pair["kir"], 0.50)].copy()
    adb50_row = adb50.iloc[0]
    da_rows.append(
        {
            "method": "ADB",
            "dataset": "stackoverflow",
            "kir": 0.50,
            "oos_f1": float(adb50_row["oos_f1"]),
            "oos_f1_std": float(adb50_row["oos_f1_std"]),
            "f1_all": np.nan,
            "accuracy": np.nan,
            "known_recall": float(adb50_row["known_recall"]),
            "false_accept_rate": float(adb50_row["false_accept_rate"]),
            "false_reject_rate": float(adb50_row["false_reject_rate"]),
            "n_seeds": int(adb50_row["n_seeds"]),
            "backbone": "BERT/TextOIR",
            "contract_layer": "external_backbone_reference",
        }
    )
    external = pd.DataFrame(da_rows)
    external.to_csv(OUT / "stackoverflow_external_summary.csv", index=False)
    return external


def _dcloos_status() -> pd.DataFrame:
    blocked = _read_csv(
        BLOCKED_PATH,
        {"method", "status", "contract_layer", "dataset", "kir", "seed", "reason", "final_metrics_available", "include_in_unified_rows"},
    ).copy()
    rows = blocked[blocked["method"].str.contains("DCLOOS|KNNCL", case=False, regex=True)].copy()
    if DCLOOS_METRICS.is_file():
        metrics = json.loads(DCLOOS_METRICS.read_text(encoding="utf-8"))
        recovered = pd.DataFrame(
            [
                {
                    "method": "DCLOOS-reduced",
                    "status": "reference_only",
                    "contract_layer": "different_supervision",
                    "dataset": "oos+squad",
                    "kir": 0.75,
                    "seed": "888",
                    "reason": "BERT + pseudo-OOS + external SQuAD OOS; recovered intermediate prediction, not Known-only fair",
                    "final_metrics_available": True,
                    "include_in_unified_rows": False,
                    "oos_f1": float(metrics["oos_f1"]) / 100.0,
                    "f1_all": float(metrics["f1_all"]) / 100.0,
                    "known_recall": float(metrics["known_recall"]) / 100.0,
                    "accuracy": float(metrics["accuracy"]) / 100.0,
                }
            ]
        )
        rows = pd.concat([rows, recovered], ignore_index=True, sort=False)
    rows.to_csv(OUT / "external_status.csv", index=False)
    return rows


def plot_fair_tradeoff(fair50: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.24, top=0.70, wspace=0.18)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS, strict=True)):
        part = fair50[fair50["dataset"].eq(dataset)]
        for method in FAIR_ORDER:
            row = part[part["method"].eq(method)].iloc[0]
            style = FAIR_STYLE[method]
            ax.scatter(
                float(row["known_recall"]) * 100,
                float(row["oos_f1"]) * 100,
                marker=style["marker"],
                s=style["size"],
                color=style["color"],
                edgecolors=style["edge"] if style["marker"] != "x" else "none",
                linewidths=0.85 if method == "trainable_k1" else 0.55,
                zorder=style["zorder"],
            )
        # A gray guide marks a coverage guard rather than a selected threshold.
        ax.axvline(80, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.8)
        ax.text(80.5, 71.2, "80%\ncoverage", color=GRAY, fontsize=5.3, va="bottom")
        s2c = part[part["method"].eq("trainable_k1")].iloc[0]
        ax.annotate(
            "S2C",
            (float(s2c["known_recall"]) * 100, float(s2c["oos_f1"]) * 100),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=6.0,
            fontweight="bold",
            color=DARK,
        )
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlim(25, 92)
        ax.set_ylim(42, 94)
        ax.set_xlabel("Known recall (%)")
        if index == 0:
            ax.set_ylabel("OOS F1 (%)")
        _style(ax)
    handles = [
        Line2D([], [], marker=FAIR_STYLE["trainable_k1"]["marker"], color=FAIR_STYLE["trainable_k1"]["color"], markeredgecolor=DARK, linestyle="None", markersize=9, label="S2C Trainable"),
        Line2D([], [], marker=FAIR_STYLE["single_centroid"]["marker"], color=BLUE, linestyle="None", markersize=5, label="Frozen single"),
        Line2D([], [], marker=FAIR_STYLE["fixed_k2"]["marker"], color="white", markeredgecolor=GRAY, linestyle="None", markersize=5, label="Frozen K=2"),
        Line2D([], [], marker=FAIR_STYLE["mogb_partition_ours_boundary"]["marker"], color=GREEN, linestyle="None", markersize=5, label="MOGB partition + S2C"),
        Line2D([], [], marker=FAIR_STYLE["ours_partition_mogb_boundary"]["marker"], color=PURPLE, linestyle="None", markersize=5, label="S2C partition + MOGB"),
        Line2D([], [], marker=FAIR_STYLE["mogb_minilm"]["marker"], color=RED, linestyle="None", markersize=6, label="MOGB-MiniLM"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.88), ncol=3, frameon=False, columnspacing=0.9, handletextpad=0.35)
    fig.text(0.08, 0.965, "Fair OOS boundary: S2C occupies the high-F1 / high-coverage region", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "KIR=.50; five seeds; same protocol_v2 MiniLM Gate. Star = proposed method; x = MOGB-MiniLM.", fontsize=5.9, color="#555B61")
    return _save(fig, "fair_oos_boundary_tradeoff")


def plot_precision_recall(fair50: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.24, top=0.70, wspace=0.18)
    x_grid = np.linspace(55, 100, 250)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS, strict=True)):
        part = fair50[fair50["dataset"].eq(dataset)]
        for f1_level in (70, 80, 90):
            precision = f1_level * x_grid / (2 * x_grid - f1_level)
            valid = precision <= 100
            ax.plot(x_grid[valid], precision[valid], color=LIGHT_GRAY, linestyle="--", linewidth=0.55)
            label_x = 96 if f1_level == 90 else 98
            label_y = float(f1_level * label_x / (2 * label_x - f1_level))
            ax.text(label_x - 3, label_y + 0.4, f"F1={f1_level}", color=GRAY, fontsize=5.2)
        for method in FAIR_ORDER:
            row = part[part["method"].eq(method)].iloc[0]
            style = FAIR_STYLE[method]
            ax.scatter(
                float(row["oos_recall"]) * 100,
                float(row["oos_precision"]) * 100,
                marker=style["marker"],
                s=style["size"],
                color=style["color"],
                edgecolors=style["edge"] if style["marker"] != "x" else "none",
                linewidths=0.85 if method == "trainable_k1" else 0.55,
                zorder=style["zorder"],
            )
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlim(55, 101)
        ax.set_ylim(35, 101)
        ax.set_xlabel("OOS recall (%)")
        if index == 0:
            ax.set_ylabel("OOS precision (%)")
        _style(ax)
    handles = [
        Line2D([], [], marker="*", color=YELLOW, markeredgecolor=DARK, linestyle="None", markersize=9, label="S2C Trainable"),
        Line2D([], [], marker="o", color=BLUE, linestyle="None", markersize=5, label="Frozen single"),
        Line2D([], [], marker="x", color=RED, linestyle="None", markersize=6, label="MOGB-MiniLM"),
        Line2D([], [], linestyle="--", color=LIGHT_GRAY, linewidth=0.8, label="iso-F1 guides"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.88), ncol=4, frameon=False, columnspacing=0.9, handletextpad=0.35)
    fig.text(0.08, 0.965, "OOS F1 is a balance: S2C avoids both MOGB under-coverage and fixed-K rejection loss", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "Precision is reconstructed from OOS F1 and OOS recall = 1 − false acceptance; points are five-seed means.", fontsize=5.9, color="#555B61")
    return _save(fig, "fair_oos_precision_recall")


def plot_textoir_native(textoir: pd.DataFrame) -> list[str]:
    method_style = {
        "MSP": {"marker": "x", "color": RED},
        "DOC": {"marker": "o", "color": BLUE},
        "ADB": {"marker": "*", "color": YELLOW},
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.24, top=0.70, wspace=0.18)
    for index, (ax, dataset) in enumerate(zip(axes, ("oos", "banking", "stackoverflow"), strict=True)):
        part = textoir[textoir["dataset"].eq(dataset)]
        for method in ("MSP", "DOC", "ADB"):
            line = part[part["method"].eq(method)].sort_values("known_cls_ratio")
            style = method_style[method]
            ax.plot(
                line["known_cls_ratio"],
                line["oos_f1"] * 100,
                marker=style["marker"],
                color=style["color"],
                linewidth=1.25,
                markersize=6 if method != "ADB" else 8,
                markeredgecolor=DARK if method == "ADB" else style["color"],
                markeredgewidth=0.6,
                label=method,
            )
        ax.set_title(TEXT_OIR_DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("KIR")
        ax.set_xticks(KIRS, [".25", ".50", ".75"])
        ax.set_ylim(20, 100)
        if index == 0:
            ax.set_ylabel("OOS F1 (%)")
        _style(ax, "y")
    handles = [
        Line2D([], [], marker="x", color=RED, linestyle="-", linewidth=1.1, markersize=6, label="MSP"),
        Line2D([], [], marker="o", color=BLUE, linestyle="-", linewidth=1.1, markersize=5, label="DOC"),
        Line2D([], [], marker="*", color=YELLOW, markeredgecolor=DARK, linestyle="-", linewidth=1.1, markersize=8, label="ADB"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.88), ncol=3, frameon=False, columnspacing=0.9, handletextpad=0.35)
    fig.text(0.08, 0.965, "TextOIR-native compatibility: the detector family changes sharply with KIR", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "BERT/TextOIR native runs; three seeds per point. This panel is a compatibility reference, not a MiniLM fair ranking.", fontsize=5.9, color="#555B61")
    return _save(fig, "textoir_native_oos_kir_stability")


def plot_s2c_adb_frontier(pair: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.24, top=0.70, wspace=0.18)
    styles = {
        "S2C-Trainable-K1": {"marker": "*", "color": YELLOW, "edge": DARK, "size": 130},
        "ADB": {"marker": "x", "color": RED, "edge": RED, "size": 60},
    }
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS, strict=True)):
        part = pair[pair["dataset"].eq(dataset)]
        for method in ("S2C-Trainable-K1", "ADB"):
            line = part[part["method"].eq(method)].sort_values("kir")
            style = styles[method]
            ax.plot(line["known_recall"] * 100, line["oos_f1"] * 100, color=style["color"], linewidth=1.15, alpha=0.9)
            ax.scatter(
                line["known_recall"] * 100,
                line["oos_f1"] * 100,
                marker=style["marker"],
                s=style["size"],
                color=style["color"],
                edgecolors=style["edge"] if style["marker"] != "x" else "none",
                linewidths=0.8,
                zorder=5,
            )
            for _, row in line.iterrows():
                ax.annotate(f"{float(row['kir']):.2f}", (float(row["known_recall"]) * 100, float(row["oos_f1"]) * 100), xytext=(3, 3), textcoords="offset points", fontsize=5.3, color=style["color"])
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("Known recall (%)")
        ax.set_xlim(25, 95)
        ax.set_ylim(40, 100)
        if index == 0:
            ax.set_ylabel("OOS F1 (%)")
        _style(ax)
    handles = [
        Line2D([], [], marker="*", color=YELLOW, markeredgecolor=DARK, linestyle="-", linewidth=1.1, markersize=9, label="S2C Trainable (MiniLM)"),
        Line2D([], [], marker="x", color=RED, linestyle="-", linewidth=1.1, markersize=6, label="ADB (BERT/TextOIR)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.88), ncol=2, frameon=False, columnspacing=1.0, handletextpad=0.35)
    fig.text(0.08, 0.965, "S2C versus ADB: the advantage is dataset- and KIR-dependent, not a universal dominance claim", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "Current protocol KIR sweep; S2C is MiniLM fair and ADB is BERT/TextOIR external. Labels mark KIR.", fontsize=5.9, color="#555B61")
    return _save(fig, "s2c_vs_adb_oos_frontier")


def plot_mogb_bridge(fair50: pd.DataFrame) -> list[str]:
    bridge = ("mogb_minilm", "mogb_partition_ours_boundary", "trainable_k1")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.24, top=0.70, wspace=0.18)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS, strict=True)):
        part = fair50[fair50["dataset"].eq(dataset)].set_index("method")
        coords = [(float(part.loc[m, "known_recall"]) * 100, float(part.loc[m, "oos_f1"]) * 100) for m in bridge]
        for left, right in zip(coords[:-1], coords[1:], strict=True):
            ax.annotate("", xy=right, xytext=left, arrowprops={"arrowstyle": "->", "color": GRAY, "linewidth": 1.1, "shrinkA": 8, "shrinkB": 8})
        for method, (x, y) in zip(bridge, coords, strict=True):
            style = FAIR_STYLE[method]
            ax.scatter(x, y, marker=style["marker"], s=style["size"], color=style["color"], edgecolors=style["edge"] if style["marker"] != "x" else "none", linewidths=0.85, zorder=5)
        first = part.loc["mogb_minilm"]
        middle = part.loc["mogb_partition_ours_boundary"]
        last = part.loc["trainable_k1"]
        ax.text(float(first["known_recall"]) * 100 - 2, float(first["oos_f1"]) * 100 - 4, "MOGB", ha="right", fontsize=5.6, color=RED)
        ax.text(float(middle["known_recall"]) * 100 + 2, float(middle["oos_f1"]) * 100 - 4, "boundary", fontsize=5.6, color=GREEN)
        ax.text(float(last["known_recall"]) * 100 + 2, float(last["oos_f1"]) * 100 + 1, "S2C", fontsize=6.0, fontweight="bold", color=DARK)
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("Known recall (%)")
        ax.set_xlim(25, 92)
        ax.set_ylim(42, 94)
        if index == 0:
            ax.set_ylabel("OOS F1 (%)")
        _style(ax)
    handles = [
        Line2D([], [], marker="x", color=RED, linestyle="None", markersize=6, label="MOGB-MiniLM"),
        Line2D([], [], marker="^", color=GREEN, linestyle="None", markersize=5, label="MOGB partition + S2C boundary"),
        Line2D([], [], marker="*", color=YELLOW, markeredgecolor=DARK, linestyle="None", markersize=9, label="S2C Trainable"),
        Line2D([], [], color=GRAY, linewidth=1.0, label="component bridge"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.88), ncol=4, frameon=False, columnspacing=0.75, handletextpad=0.3)
    fig.text(0.08, 0.965, "MOGB component bridge: partition and boundary changes help, but representation adaptation closes the gap", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "KIR=.50; five seeds; same MiniLM fair contract. Arrows are descriptive component swaps, not a causal intervention.", fontsize=5.9, color="#555B61")
    return _save(fig, "mogb_component_bridge")


def plot_error_budget(fair50: pd.DataFrame) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.24, top=0.70, wspace=0.18)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS, strict=True)):
        part = fair50[fair50["dataset"].eq(dataset)]
        for method in FAIR_ORDER:
            row = part[part["method"].eq(method)].iloc[0]
            style = FAIR_STYLE[method]
            ax.scatter(float(row["false_accept_rate"]) * 100, float(row["false_reject_rate"]) * 100, marker=style["marker"], s=style["size"] * 0.72, color=style["color"], edgecolors=style["edge"] if style["marker"] != "x" else "none", linewidths=0.8, zorder=5)
        s2c = part[part["method"].eq("trainable_k1")].iloc[0]
        mogb = part[part["method"].eq("mogb_minilm")].iloc[0]
        ax.annotate("", xy=(float(s2c["false_accept_rate"]) * 100, float(s2c["false_reject_rate"]) * 100), xytext=(float(mogb["false_accept_rate"]) * 100, float(mogb["false_reject_rate"]) * 100), arrowprops={"arrowstyle": "->", "color": GRAY, "linewidth": 1.1, "shrinkA": 8, "shrinkB": 8})
        ax.text(float(s2c["false_accept_rate"]) * 100 + 1.3, float(s2c["false_reject_rate"]) * 100 - 1.2, "S2C", fontsize=6.0, fontweight="bold")
        ax.text(float(mogb["false_accept_rate"]) * 100 + 1.3, float(mogb["false_reject_rate"]) * 100 + 1.0, "MOGB", fontsize=5.6, color=RED)
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("OOS false acceptance (%)")
        ax.set_xlim(-1, 74)
        ax.set_ylim(-1, 74)
        if index == 0:
            ax.set_ylabel("Known false rejection (%)")
        _style(ax)
    handles = [
        Line2D([], [], marker="*", color=YELLOW, markeredgecolor=DARK, linestyle="None", markersize=9, label="S2C Trainable"),
        Line2D([], [], marker="x", color=RED, linestyle="None", markersize=6, label="MOGB-MiniLM"),
        Line2D([], [], marker="o", color=BLUE, linestyle="None", markersize=5, label="Frozen single"),
        Line2D([], [], color=GRAY, linewidth=1.0, label="MOGB → S2C trade-off"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.88), ncol=4, frameon=False, columnspacing=0.8, handletextpad=0.3)
    fig.text(0.08, 0.965, "Error budget: MOGB is conservative; S2C trades a small open-space cost for much higher coverage", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "KIR=.50; five-seed means. Lower-left is better; arrows show the MOGB-MiniLM to S2C shift.", fontsize=5.9, color="#555B61")
    return _save(fig, "oos_error_budget_frontier")


def plot_ball_risk() -> list[str]:
    frame = _read_csv(MOGB_BALL_PATH, {"dataset", "seed", "ball_id", "support", "radius", "known_reject_rate", "oos_false_accept_rate"})
    frame = frame[frame["seed"].eq(42)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.24, top=0.70, wspace=0.18)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS, strict=True)):
        part = frame[frame["dataset"].eq(dataset)].copy()
        part["oos_false_accept_rate"] = part["oos_false_accept_rate"].fillna(0.0)
        for _, row in part.iterrows():
            polluted = float(row["oos_false_accept_rate"]) > 0
            ax.scatter(
                float(row["support"]),
                float(row["oos_false_accept_rate"]) * 100,
                marker="x" if polluted else "o",
                color=RED if polluted else BLUE,
                s=18 + 55 * float(row["radius"]),
                alpha=0.58,
                linewidths=0.7,
            )
        risky = part.sort_values("oos_false_accept_rate", ascending=False).head(1)
        if not risky.empty and float(risky.iloc[0]["oos_false_accept_rate"]) > 0:
            row = risky.iloc[0]
            ax.annotate(f"ball {int(row['ball_id'])}", (float(row["support"]), float(row["oos_false_accept_rate"]) * 100), xytext=(4, 4), textcoords="offset points", fontsize=5.6, color=RED)
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("ball support")
        ax.set_xlim(0, 105)
        ax.set_ylim(-2, 100)
        if index == 0:
            ax.set_ylabel("OOS false acceptance (%)")
        _style(ax)
    handles = [
        Line2D([], [], marker="o", color=BLUE, linestyle="None", markersize=5, label="no observed OOS contamination"),
        Line2D([], [], marker="x", color=RED, linestyle="None", markersize=6, label="OOS-contaminated ball"),
        Line2D([], [], marker="o", color=DARK, alpha=0.5, linestyle="None", markersize=8, label="larger marker = larger radius"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.88), ncol=3, frameon=False, columnspacing=0.8, handletextpad=0.3)
    fig.text(0.08, 0.965, "MOGB risk is local: a small set of balls absorbs OOS while neighboring balls reject Known", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "KIR=.50; seed=42; each mark is a MOGB ball. Marker size encodes radius; this is a descriptive risk map.", fontsize=5.9, color="#555B61")
    return _save(fig, "mogb_ball_risk_surface")


def plot_state_transitions() -> list[str]:
    frame = _read_csv(TRANSITION_PATH, {"dataset", "seed", "gold_type", "baseline", "s2c_state", "baseline_state", "count", "rate_within_gold_type"})
    frame = frame[(frame["seed"].eq(42)) & (frame["baseline"].eq("mogb"))].copy()
    transitions = [
        ("Known", "Known rejected", "Known correct", "Known rescue", BLUE),
        ("Known", "Known correct", "Known rejected", "Known coverage cost", GRAY),
        ("OOS", "OOS accepted", "OOS rejected", "OOS fixed", RED),
        ("OOS", "OOS rejected", "OOS accepted", "OOS regression", PURPLE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.26, top=0.70, wspace=0.18)
    for index, (ax, dataset) in enumerate(zip(axes, DATASETS, strict=True)):
        values: list[float] = []
        for gold_type, baseline_state, s2c_state, label, color in transitions:
            row = frame[(frame["dataset"].eq(dataset)) & (frame["gold_type"].eq(gold_type)) & (frame["baseline_state"].eq(baseline_state)) & (frame["s2c_state"].eq(s2c_state))]
            rate = float(row["rate_within_gold_type"].iloc[0]) * 100 if not row.empty else 0.0
            values.append(rate)
        y = np.arange(len(transitions))
        for yi, (info, value) in enumerate(zip(transitions, values, strict=True)):
            _, _, _, label, color = info
            ax.plot([0, value], [yi, yi], color=LIGHT_GRAY, linewidth=1.0)
            ax.scatter(value, yi, marker="o" if color in (BLUE, GRAY) else "x", color=color, s=52 if color != RED else 60, linewidths=1.0, zorder=4)
            ax.text(min(value + 1.0, 57.0), yi, f"{value:.1f}%", va="center", fontsize=5.6, color=color)
        ax.set_yticks(y, [item[3] for item in transitions])
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("share of the gold group (%)")
        ax.set_xlim(0, 60)
        if index == 0:
            ax.set_ylabel("same-sample transition")
        _style(ax)
    handles = [
        Line2D([], [], marker="o", color=BLUE, linestyle="None", markersize=5, label="Known rescue"),
        Line2D([], [], marker="o", color=GRAY, linestyle="None", markersize=5, label="Known cost"),
        Line2D([], [], marker="x", color=RED, linestyle="None", markersize=6, label="OOS fixed"),
        Line2D([], [], marker="x", color=PURPLE, linestyle="None", markersize=6, label="OOS regression"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.88), ncol=4, frameon=False, columnspacing=0.8, handletextpad=0.3)
    fig.text(0.08, 0.965, "Same-sample transitions: S2C recovers Known and corrects MOGB false acceptance, with dataset-specific costs", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "KIR=.50; seed=42; MOGB-Fair component versus S2C Trainable. Rates are within Known or OOS gold groups.", fontsize=5.9, color="#555B61")
    return _save(fig, "same_sample_mogb_s2c_transitions")


def plot_evidence_status(status: pd.DataFrame) -> list[str]:
    rows = [
        ("S2C Trainable", "same protocol", True, True),
        ("TextOIR MSP", "native compatibility", True, True),
        ("TextOIR DOC", "native compatibility", True, True),
        ("TextOIR ADB", "native compatibility", True, True),
        ("MOGB-Fair", "same MiniLM component", True, True),
        ("DA-ADB", "external BERT", True, False),
        ("KNNCL", "current final metrics", False, False),
        ("DCLOOS", "official default budget", False, False),
        ("DCLOOS reduced", "reference only", True, False),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    fig.subplots_adjust(left=0.19, right=0.995, bottom=0.20, top=0.70)
    columns = ("final metrics", "same MiniLM/data contract", "direct fair ranking")
    x = np.arange(len(columns))
    for y, (label, _, final, fair) in enumerate(rows):
        ax.text(-0.05, y, label, ha="right", va="center", fontsize=6.4, color=DARK)
        values = (final, fair, fair)
        for xi, value in zip(x, values, strict=True):
            if xi == 0 and label == "S2C Trainable":
                ax.scatter(xi, y, marker="*", s=145, color=YELLOW, edgecolors=DARK, linewidths=0.8, zorder=4)
            elif value:
                ax.scatter(xi, y, marker="o", s=42, color=BLUE, edgecolors="white", linewidths=0.5, zorder=3)
            else:
                ax.scatter(xi, y, marker="x", s=55, color=RED, linewidths=1.0, zorder=3)
    ax.set_xticks(x, columns)
    ax.set_yticks([])
    ax.set_xlim(-0.45, len(columns) - 0.55)
    ax.set_ylim(-1, len(rows))
    for yi in range(len(rows)):
        ax.axhline(yi - 0.5, color="#EEF0F2", linewidth=0.45, zorder=0)
    _style(ax)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE, linestyle="None", markersize=5, label="available"),
            Line2D([], [], marker="x", color=RED, linestyle="None", markersize=6, label="not closed"),
            Line2D([], [], marker="*", color=YELLOW, markeredgecolor=DARK, linestyle="None", markersize=8, label="proposed method"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.56, 0.88),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.text(0.08, 0.965, "Evidence map: more baseline numbers do not mean more fair evidence", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "Blue circle = an auditable result; red x = blocked or contract-incompatible; reduced DCLOOS is reference-only, not a ranking row.", fontsize=5.9, color="#555B61")
    return _save(fig, "cross_method_evidence_status")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fair, fair50 = _fair_data()
    textoir = _textoir_data()
    adb_pair, _ = _adb_data(fair)
    _external_data(adb_pair)
    status = _dcloos_status()
    figures = {
        "fair_oos_boundary_tradeoff": plot_fair_tradeoff(fair50),
        "fair_oos_precision_recall": plot_precision_recall(fair50),
        "textoir_native_oos_kir_stability": plot_textoir_native(textoir),
        "s2c_vs_adb_oos_frontier": plot_s2c_adb_frontier(adb_pair),
        "mogb_component_bridge": plot_mogb_bridge(fair50),
        "oos_error_budget_frontier": plot_error_budget(fair50),
        "mogb_ball_risk_surface": plot_ball_risk(),
        "same_sample_mogb_s2c_transitions": plot_state_transitions(),
        "cross_method_evidence_status": plot_evidence_status(status),
    }
    manifest = {
        "schema_version": "s2c.cross_method_oos_mechanism.v1",
        "bundle": "cross_method_oos_mechanism",
        "contract_layers": [
            "same_protocol_fair",
            "textoir_native_compatibility",
            "external_backbone_reference",
            "different_supervision_or_incomplete",
        ],
        "question": "Why does S2C occupy a different OOS/coverage/error regime from TextOIR methods, MOGB components, ADB/DA-ADB, and DCLOOS?",
        "source_data": [
            str(FAIR_PATH.relative_to(ROOT)),
            str(TEXT_OIR_PATH.relative_to(ROOT)),
            str(ADB_PATH.relative_to(ROOT)),
            str(DA_ADB_PATH.relative_to(ROOT)),
            str(MOGB_BALL_PATH.relative_to(ROOT)),
            str(TRANSITION_PATH.relative_to(ROOT)),
            str(BLOCKED_PATH.relative_to(ROOT)),
            str(DCLOOS_METRICS.relative_to(ROOT)),
        ],
        "figures": [path for figure_paths in figures.values() for path in figure_paths],
        "figure_index": figures,
        "style_reference": {
            "level": "style_only_inheritance",
            "vocabulary": "Known blue circles, OOS red crosses, yellow stars, gray dashed guides, white canvas, black axes",
            "reference": "RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md Fig. 1",
        },
        "selection_used_test_oos": False,
        "new_training": False,
        "historical_banking_note": "The paper-facing historical H1 mechanism figures use banking77_oos; archive/current fair figures use standard banking77. No cross-line banking row is pooled.",
        "dc_lo_os_note": "DCLOOS reduced metrics are retained as reference-only evidence and excluded from fair ranking.",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
