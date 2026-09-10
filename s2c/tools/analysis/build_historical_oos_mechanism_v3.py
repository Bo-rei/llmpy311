#!/usr/bin/env python3
"""Build paper-oriented, OOS-only mechanism evidence for H1 v19."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.4,
        "axes.labelsize": 7.4,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "legend.fontsize": 6.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_protocol_v2" / "minilm_k1"
DATA_ROOT = ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19"
RESULT_ROOT = ROOT / "results" / "analysis" / "historical_protocol_v3"
FIGURE_ROOT = ROOT / "figures" / "historical_protocol_oos_v3"

DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
DATASET_LABELS = {"clinc150": "CLINC150", "stackoverflow": "StackOverflow", "banking77_oos": "BANKING77-OOS"}
METHODS = ("frozen_k1", "trainable_k1")
METHOD_LABELS = {"frozen_k1": "Frozen", "trainable_k1": "Trainable"}
METHOD_COLORS = {"frozen_k1": "#4C78A8", "trainable_k1": "#C44E52"}
TRANSITIONS = ("both_correct", "trainable_only", "frozen_only", "both_wrong")
TRANSITION_LABELS = {"both_correct": "both correct", "trainable_only": "Trainable only", "frozen_only": "Frozen only", "both_wrong": "both wrong"}
TRANSITION_COLORS = {"both_correct": "#A5ABB2", "trainable_only": "#C44E52", "frozen_only": "#4C78A8", "both_wrong": "#D9DDE1"}
OOS_COLOR = "#C44E52"
THRESHOLD_COLOR = "#252525"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"empty prediction file: {path}")
    return frame


def run_dir(dataset: str, kir: float, seed: int) -> Path:
    return RUN_ROOT / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}"


def data_dir(dataset: str, kir: float, seed: int) -> Path:
    return DATA_ROOT / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}"


def load_results() -> tuple[pd.DataFrame, dict[tuple[str, float, int, str], pd.DataFrame], dict[tuple[str, float, int], list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, float, int, str], pd.DataFrame] = {}
    test_rows: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        for kir in (0.25, 0.50, 0.75):
            for seed in (13, 42, 87):
                root = run_dir(dataset, kir, seed)
                data_root = data_dir(dataset, kir, seed)
                test = read_json(data_root / "gate" / "test.json")
                test_rows[(dataset, kir, seed)] = test
                for method in METHODS:
                    method_root = root / method
                    metrics_path = method_root / "metrics.json"
                    prediction_path = method_root / "predictions.jsonl"
                    metrics = read_json(metrics_path)
                    frame = read_jsonl(prediction_path)
                    if len(frame) != len(test):
                        raise ValueError(f"test/prediction length mismatch: {method_root}")
                    frame["text"] = [str(row["text"]) for row in test]
                    frame["oos_source"] = [str(row.get("split", "unknown")) for row in test]
                    predictions[(dataset, kir, seed, method)] = frame
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": DATASET_LABELS[dataset],
                            "kir": kir,
                            "seed": seed,
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "oos_f1": float(metrics["oos_f1"]),
                            "known_recall": float(metrics["known_recall"]),
                            "false_accept_rate": float(metrics["false_accept_rate"]),
                            "auroc": float(metrics["auroc"]),
                            "aupr_oos": float(metrics["aupr_oos"]),
                            "metrics_path": str(metrics_path),
                            "predictions_path": str(prediction_path),
                        }
                    )
    return pd.DataFrame(rows), predictions, test_rows


def merge_oos_pair(predictions: dict[tuple[str, float, int, str], pd.DataFrame], dataset: str, seed: int) -> pd.DataFrame:
    frozen = predictions[(dataset, 0.50, seed, "frozen_k1")]
    trainable = predictions[(dataset, 0.50, seed, "trainable_k1")]
    required = {"sample_id", "gold_is_oos", "predicted_is_oos", "oos_score", "distance", "radius", "text", "oos_source"}
    if not required.issubset(frozen.columns) or not required.issubset(trainable.columns):
        raise ValueError(f"mechanism fields missing: {dataset}/{seed}")
    left = frozen[list(required)].rename(
        columns={
            "gold_is_oos": "gold_is_oos_frozen",
            "predicted_is_oos": "frozen_pred",
            "oos_score": "frozen_score",
            "distance": "frozen_distance",
            "radius": "frozen_radius",
            "oos_source": "frozen_source",
        }
    )
    right = trainable[list(required)].rename(
        columns={
            "gold_is_oos": "gold_is_oos_trainable",
            "predicted_is_oos": "trainable_pred",
            "oos_score": "trainable_score",
            "distance": "trainable_distance",
            "radius": "trainable_radius",
            "text": "trainable_text",
            "oos_source": "trainable_source",
        }
    )
    merged = left.merge(right, on="sample_id", validate="one_to_one")
    merged = merged[merged["gold_is_oos_frozen"].astype(int).eq(1)].copy()
    frozen_pred = merged["frozen_pred"].astype(int)
    trainable_pred = merged["trainable_pred"].astype(int)
    merged["transition"] = np.select(
        [frozen_pred.eq(1) & trainable_pred.eq(1), frozen_pred.eq(0) & trainable_pred.eq(1), frozen_pred.eq(1) & trainable_pred.eq(0)],
        ["both_correct", "trainable_only", "frozen_only"],
        default="both_wrong",
    )
    merged["text"] = merged["trainable_text"]
    merged["oos_source"] = merged["trainable_source"]
    merged["delta_distance"] = merged["trainable_distance"].astype(float) - merged["frozen_distance"].astype(float)
    merged["delta_radius"] = merged["trainable_radius"].astype(float) - merged["frozen_radius"].astype(float)
    merged["delta_score"] = merged["trainable_score"].astype(float) - merged["frozen_score"].astype(float)
    merged["score_distance_contribution"] = merged["trainable_distance"].astype(float) / merged["frozen_radius"].astype(float) - merged["frozen_score"].astype(float)
    merged["score_radius_contribution"] = merged["trainable_score"].astype(float) - merged["trainable_distance"].astype(float) / merged["frozen_radius"].astype(float)
    if not np.allclose(merged["score_distance_contribution"] + merged["score_radius_contribution"], merged["delta_score"], atol=1e-6, rtol=1e-6):
        raise ValueError(f"score decomposition mismatch: {dataset}/{seed}")
    return merged


def build_mechanism_tables(predictions: dict[tuple[str, float, int, str], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_frames = []
    for dataset in DATASETS:
        for seed in (13, 42, 87):
            frame = merge_oos_pair(predictions, dataset, seed)
            frame.insert(0, "seed", seed)
            frame.insert(0, "dataset", dataset)
            sample_frames.append(frame)
    sample = pd.concat(sample_frames, ignore_index=True)
    sample_columns = [
        "dataset", "seed", "sample_id", "oos_source", "transition",
        "frozen_distance", "trainable_distance", "delta_distance",
        "frozen_radius", "trainable_radius", "delta_radius",
        "frozen_score", "trainable_score", "delta_score", "score_distance_contribution", "score_radius_contribution",
    ]
    sample[sample_columns].to_csv(RESULT_ROOT / "oos_pairwise_mechanism.csv", index=False)
    summary = (
        sample.groupby(["dataset", "transition"], as_index=False)
        .agg(
            n_oos=("sample_id", "size"),
            delta_distance_mean=("delta_distance", "mean"),
            delta_distance_median=("delta_distance", "median"),
            delta_radius_mean=("delta_radius", "mean"),
            delta_radius_median=("delta_radius", "median"),
            delta_score_mean=("delta_score", "mean"),
            delta_score_median=("delta_score", "median"),
            score_distance_contribution_mean=("score_distance_contribution", "mean"),
            score_distance_contribution_median=("score_distance_contribution", "median"),
            score_radius_contribution_mean=("score_radius_contribution", "mean"),
            score_radius_contribution_median=("score_radius_contribution", "median"),
            n_seeds=("seed", "nunique"),
        )
    )
    totals = sample.groupby("dataset")["sample_id"].size().rename("total_oos")
    summary = summary.merge(totals, on="dataset")
    summary["rate"] = summary["n_oos"] / summary["total_oos"]
    summary.to_csv(RESULT_ROOT / "oos_mechanism_summary.csv", index=False)
    return sample, summary


def build_training_table() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for seed in (13, 42, 87):
            history_path = run_dir(dataset, 0.50, seed) / "trainable_k1" / "training_history.json"
            for item in read_json(history_path):
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "seed": seed,
                        "epoch": int(item["epoch"]),
                        "phase": str(item["phase"]),
                        "loss": float(item["loss"]),
                        "selection_score": float(item["selection_score"]),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULT_ROOT / "training_dynamics.csv", index=False)
    return frame


def build_performance_table(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kir50 = results[results["kir"].eq(0.50)].copy()
    summary = (
        kir50.groupby(["dataset", "dataset_label", "method", "method_label"], as_index=False)
        .agg(
            oos_f1=("oos_f1", "mean"),
            oos_f1_std=("oos_f1", "std"),
            false_accept_rate=("false_accept_rate", "mean"),
            false_accept_rate_std=("false_accept_rate", "std"),
            auroc=("auroc", "mean"),
            aupr_oos=("aupr_oos", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(RESULT_ROOT / "oos_performance_summary.csv", index=False)
    kir_summary = (
        results.groupby(["dataset", "dataset_label", "kir", "method", "method_label"], as_index=False)
        .agg(oos_f1=("oos_f1", "mean"), oos_f1_std=("oos_f1", "std"), n_seeds=("seed", "nunique"))
    )
    kir_summary.to_csv(RESULT_ROOT / "oos_kir_summary.csv", index=False)
    return summary, kir_summary


def build_case_table(sample: pd.DataFrame) -> None:
    selected = []
    for dataset in DATASETS:
        part = sample[(sample["dataset"] == dataset) & (sample["seed"] == 42)]
        part = part[part["text"].astype(str).str.strip().ne("")]
        for transition, ascending in (("trainable_only", False), ("frozen_only", True)):
            chosen = part[part["transition"] == transition].sort_values("delta_score", ascending=ascending).head(3)
            selected.append(chosen)
        wrong = part[part["transition"] == "both_wrong"].copy()
        wrong["threshold_distance"] = np.minimum((wrong["frozen_score"] - 1.0).abs(), (wrong["trainable_score"] - 1.0).abs())
        selected.append(wrong.sort_values("threshold_distance").head(3))
    cases = pd.concat(selected, ignore_index=True)
    cases[["dataset", "seed", "sample_id", "oos_source", "transition", "text", "frozen_distance", "trainable_distance", "delta_distance", "frozen_radius", "trainable_radius", "delta_radius", "frozen_score", "trainable_score", "delta_score", "score_distance_contribution", "score_radius_contribution"]].to_csv(RESULT_ROOT / "oos_case_examples_seed42.csv", index=False)


def style(ax: plt.Axes) -> None:
    ax.tick_params(length=3, width=0.7, pad=2)
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.45)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_main_performance(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.82, wspace=0.27)
    x = np.arange(len(DATASETS))
    for method in METHODS:
        part = summary[summary["method"] == method].set_index("dataset").loc[list(DATASETS)]
        axes[0].errorbar(x + (-0.11 if method == "frozen_k1" else 0.11), part["oos_f1"] * 100, yerr=part["oos_f1_std"].fillna(0) * 100, fmt="o-", color=METHOD_COLORS[method], linewidth=1.25, markersize=4.2, capsize=2, label=METHOD_LABELS[method])
        axes[1].errorbar(x + (-0.11 if method == "frozen_k1" else 0.11), part["false_accept_rate"] * 100, yerr=part["false_accept_rate_std"].fillna(0) * 100, fmt="o-", color=METHOD_COLORS[method], linewidth=1.25, markersize=4.2, capsize=2, label=METHOD_LABELS[method])
    for index, dataset in enumerate(DATASETS):
        f = summary[(summary.dataset == dataset) & (summary.method == "frozen_k1")].iloc[0]
        t = summary[(summary.dataset == dataset) & (summary.method == "trainable_k1")].iloc[0]
        axes[0].text(index, max(f.oos_f1, t.oos_f1) * 100 + 2.0, f"Δ {(t.oos_f1 - f.oos_f1) * 100:+.1f} pp", ha="center", fontsize=6.2, color="#4D5156")
        axes[1].text(index, min(f.false_accept_rate, t.false_accept_rate) * 100 - 3.5, f"Δ {(t.false_accept_rate - f.false_accept_rate) * 100:+.1f} pp", ha="center", fontsize=6.2, color="#4D5156")
    axes[0].set_title("OOS F1", loc="left", fontweight="bold")
    axes[1].set_title("OOS false acceptance", loc="left", fontweight="bold")
    axes[0].set_ylabel("OOS F1 (%)")
    axes[1].set_ylabel("false acceptance (%)")
    axes[0].set_xticks(x, [DATASET_LABELS[d] for d in DATASETS])
    axes[1].set_xticks(x, [DATASET_LABELS[d] for d in DATASETS])
    axes[0].set_ylim(72, 96)
    axes[1].set_ylim(0, 32)
    for ax in axes:
        style(ax)
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="upper right", handlelength=1.6)
    fig.text(0.01, 0.965, "a", fontsize=9, fontweight="bold", va="top")
    fig.text(0.51, 0.965, "b", fontsize=9, fontweight="bold", va="top")
    save_figure(fig, "main_oos_performance")


def mean_density(frames: list[pd.DataFrame], column: str, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for frame in frames:
        values = frame.loc[frame["gold_is_oos"].astype(int).eq(1), column].astype(float).to_numpy()
        density, _ = np.histogram(values, bins=bins, density=True)
        curves.append(density)
    return (bins[:-1] + bins[1:]) / 2, np.mean(curves, axis=0)


def plot_main_mechanism(predictions: dict[tuple[str, float, int, str], pd.DataFrame], sample: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.55), gridspec_kw={"height_ratios": (1.1, 0.85)})
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.13, top=0.82, hspace=0.55, wspace=0.25)
    for col, dataset in enumerate(DATASETS):
        all_scores = np.concatenate([predictions[(dataset, 0.50, seed, method)].loc[predictions[(dataset, 0.50, seed, method)]["gold_is_oos"].astype(int).eq(1), "oos_score"].astype(float).to_numpy() for seed in (13, 42, 87) for method in METHODS])
        high = max(1.35, float(np.quantile(all_scores, 0.995) * 1.06))
        bins = np.linspace(0, high, 64)
        ax = axes[0, col]
        for method in METHODS:
            centers, curve = mean_density([predictions[(dataset, 0.50, seed, method)] for seed in (13, 42, 87)], "oos_score", bins)
            ax.plot(centers, curve, color=METHOD_COLORS[method], linewidth=1.45, label=METHOD_LABELS[method])
        ax.axvline(1.0, color=THRESHOLD_COLOR, linestyle="--", linewidth=0.8)
        f = summary[(summary.dataset == dataset) & (summary.method == "frozen_k1")].iloc[0]
        t = summary[(summary.dataset == dataset) & (summary.method == "trainable_k1")].iloc[0]
        mechanism = sample[sample.dataset == dataset]
        correlation = float(mechanism[["delta_distance", "delta_score"]].corr().iloc[0, 1])
        ax.text(0.03, 0.94, f"F1 {f.oos_f1 * 100:.1f} → {t.oos_f1 * 100:.1f}\nFA {f.false_accept_rate * 100:.1f}% → {t.false_accept_rate * 100:.1f}%\nr(Δd, Δs)={correlation:.2f}", transform=ax.transAxes, va="top", fontsize=6.1, color="#4D5156")
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("normalized OOS score")
        if col == 0:
            ax.set_ylabel("OOS density")
        ax.set_xlim(0, high)
        style(ax)
        ax.grid(axis="x", visible=False)

        ax = axes[1, col]
        part = sample[sample.dataset == dataset]
        rates = part.groupby("transition")["sample_id"].size().reindex(TRANSITIONS, fill_value=0) / len(part) * 100
        bottom = 0.0
        for transition in TRANSITIONS:
            value = float(rates[transition])
            ax.bar(0, value, bottom=bottom, width=0.62, color=TRANSITION_COLORS[transition], edgecolor="white", linewidth=0.35, label=TRANSITION_LABELS[transition])
            if value >= 7:
                ax.text(0, bottom + value / 2, f"{value:.0f}%", ha="center", va="center", fontsize=6.0, color="white" if transition in {"trainable_only", "frozen_only"} else "#30353A")
            bottom += value
        delta = float(t.oos_f1 - f.oos_f1) * 100
        ax.text(0, 103, f"ΔF1 {delta:+.1f} pp", ha="center", va="bottom", fontsize=6.3, color=OOS_COLOR if delta >= 0 else METHOD_COLORS["frozen_k1"])
        ax.set_title("OOS decision transition", fontsize=7.8, loc="left")
        ax.set_ylim(0, 116)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel("share of OOS (%)")
        style(ax)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.905), ncol=2, handlelength=1.5, columnspacing=0.8)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, handlelength=1.1, columnspacing=0.8)
    axes[0, 0].text(-0.16, 1.11, "a", transform=axes[0, 0].transAxes, fontsize=9, fontweight="bold", va="top")
    axes[1, 0].text(-0.16, 1.11, "b", transform=axes[1, 0].transAxes, fontsize=9, fontweight="bold", va="top")
    fig.text(0.075, 0.94, "OOS score formation and decision correction", fontsize=8.8, va="center")
    save_figure(fig, "main_oos_mechanism")


def plot_training_dynamics(training: pd.DataFrame) -> None:
    grouped = training.groupby(["dataset", "epoch"], as_index=False).agg(loss=("loss", "mean"), loss_std=("loss", "std"), selection_score=("selection_score", "mean"), selection_std=("selection_score", "std"))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.85), sharex="col")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.82, hspace=0.45, wspace=0.25)
    for col, dataset in enumerate(DATASETS):
        part = grouped[grouped.dataset == dataset].sort_values("epoch")
        x = part.epoch.to_numpy()
        axes[0, col].plot(x, part.loss, color="#4C78A8", marker="o", markersize=3.2, linewidth=1.25)
        axes[0, col].fill_between(x, part.loss - part.loss_std.fillna(0), part.loss + part.loss_std.fillna(0), color="#4C78A8", alpha=0.15, linewidth=0)
        axes[1, col].plot(x, part.selection_score, color="#C44E52", marker="o", markersize=3.2, linewidth=1.25)
        axes[1, col].fill_between(x, part.selection_score - part.selection_std.fillna(0), part.selection_score + part.selection_std.fillna(0), color="#C44E52", alpha=0.15, linewidth=0)
        axes[0, col].set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        axes[1, col].set_xlabel("epoch")
        axes[0, col].set_xticks(part.epoch)
        axes[1, col].set_xticks(part.epoch)
        for ax in (axes[0, col], axes[1, col]):
            style(ax)
            ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("training loss")
    axes[1, 0].set_ylabel("ID-only selection score")
    axes[0, 0].text(-0.16, 1.11, "a", transform=axes[0, 0].transAxes, fontsize=9, fontweight="bold", va="top")
    axes[1, 0].text(-0.16, 1.11, "b", transform=axes[1, 0].transAxes, fontsize=9, fontweight="bold", va="top")
    fig.text(0.08, 0.94, "Trainable MiniLM adapts under the fixed ID-only selection rule", fontsize=8.8, va="center")
    fig.text(0.08, 0.02, "Curves are three-seed means at KIR=.50; the test OOS set is not used for selection.", fontsize=6.1, color="#5D646B")
    save_figure(fig, "supp_training_dynamics")


def plot_component_decomposition(sample: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.80, wspace=0.32)
    components = (("score_distance_contribution", "Δscore from distance", "#4C78A8"), ("score_radius_contribution", "Δscore from radius", "#8C9BAA"), ("delta_score", "total Δscore", "#C44E52"))
    x = np.arange(len(DATASETS))
    for ax, (column, label, color) in zip(axes, components):
        means = sample.groupby("dataset")[column].mean().reindex(DATASETS)
        medians = sample.groupby("dataset")[column].median().reindex(DATASETS)
        ax.axhline(0, color=THRESHOLD_COLOR, linewidth=0.7)
        ax.bar(x, means.to_numpy(), color=color, alpha=0.75, width=0.55)
        ax.plot(x, medians.to_numpy(), "o", color="#252525", markersize=3.4, label="median")
        for i, value in enumerate(means.to_numpy()):
            ax.text(i, value + (0.01 if value >= 0 else -0.01), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=5.9)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xticks(x, [DATASET_LABELS[d] for d in DATASETS], rotation=25, ha="right")
        ax.set_ylabel("score contribution")
        style(ax)
        ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.965, "a", fontsize=9, fontweight="bold", va="top")
    fig.text(0.34, 0.965, "b", fontsize=9, fontweight="bold", va="top")
    fig.text(0.67, 0.965, "c", fontsize=9, fontweight="bold", va="top")
    fig.text(0.08, 0.02, "Distance contribution: trainable distance / frozen radius − frozen score; radius contribution closes the exact Δscore decomposition.", fontsize=6.1, color="#5D646B")
    save_figure(fig, "supp_distance_radius_score")


def plot_oos_geometry(sample: pd.DataFrame) -> None:
    from matplotlib.patches import Patch
    from sklearn.decomposition import PCA
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from protocol_v2.experiments.racal_v1.representation import RacalMiniLM, encode_rows

    dataset = "clinc150"
    seed = 42
    data_root = data_dir(dataset, 0.50, seed)
    known = {str(value) for value in read_json(data_root / "KNOWN_INTENTS.json")["known_intents"]}
    train_rows = read_json(data_root / "gate" / "train.json")
    test_rows = read_json(data_root / "gate" / "test.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SentenceTransformer(str(ROOT.parent / "assets" / "models" / "all-MiniLM-L6-v2"), device=str(device))
    frozen_train = np.asarray(encoder.encode([str(row["text"]) for row in train_rows], batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False), dtype=np.float32)
    frozen_test = np.asarray(encoder.encode([str(row["text"]) for row in test_rows], batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False), dtype=np.float32)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint = torch.load(run_dir(dataset, 0.50, seed) / "trainable_k1" / "checkpoint.pt", map_location="cpu", weights_only=True)
    hidden = int(checkpoint["model"]["projection.fc1.weight"].shape[0])
    tokenizer = AutoTokenizer.from_pretrained(ROOT.parent / "assets" / "models" / "all-MiniLM-L6-v2", local_files_only=True)
    model = RacalMiniLM(ROOT.parent / "assets" / "models" / "all-MiniLM-L6-v2", str(checkpoint["mode"]), hidden).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    trainable_train = encode_rows(model, tokenizer, train_rows, device, 64, 256)
    trainable_test = encode_rows(model, tokenizer, test_rows, device, 64, 256)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.80, wspace=0.26)
    for ax, method, train_values, test_values in ((axes[0], "frozen_k1", frozen_train, frozen_test), (axes[1], "trainable_k1", trainable_train, trainable_test)):
        xy = PCA(n_components=2).fit(train_values).transform(test_values)
        is_oos = np.asarray([str(row["intent"]) not in known for row in test_rows])
        values = xy[is_oos]
        histogram, x_edges, y_edges = np.histogram2d(values[:, 0], values[:, 1], bins=42)
        positive = histogram[histogram > 0]
        levels = np.unique(np.quantile(positive, [0.70, 0.85, 0.95]))
        if levels.size >= 2:
            levels = np.r_[levels, positive.max() + 1e-6]
            xx, yy = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2, (y_edges[:-1] + y_edges[1:]) / 2)
            ax.contourf(xx, yy, histogram.T, levels=levels, colors=[OOS_COLOR] * (len(levels) - 1), alpha=0.23)
            ax.contour(xx, yy, histogram.T, levels=levels[:-1], colors=[OOS_COLOR], linewidths=0.55, alpha=0.8)
        ax.set_title(METHOD_LABELS[method], loc="left", fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(False)
        ax.set_aspect("equal", adjustable="box")
    fig.legend(handles=[Patch(facecolor=OOS_COLOR, edgecolor=OOS_COLOR, alpha=0.45, label="OOS density")], loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=1, handlelength=1.2)
    axes[0].text(-0.16, 1.11, "a", transform=axes[0].transAxes, fontsize=9, fontweight="bold", va="top")
    axes[1].text(-0.16, 1.11, "b", transform=axes[1].transAxes, fontsize=9, fontweight="bold", va="top")
    fig.text(0.08, 0.94, "OOS density changes after representation adaptation", fontsize=8.8, va="center")
    fig.text(0.08, 0.02, "CLINC150/KIR=.50/seed=42; PCA is descriptive and is not used for the Gate decision.", fontsize=6.1, color="#5D646B")
    save_figure(fig, "supp_oos_geometry")


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    results, predictions, _ = load_results()
    performance, _ = build_performance_table(results)
    sample, _ = build_mechanism_tables(predictions)
    training = build_training_table()
    build_case_table(sample)
    plot_main_performance(performance)
    plot_main_mechanism(predictions, sample, performance)
    plot_training_dynamics(training)
    plot_component_decomposition(sample)
    plot_oos_geometry(sample)
    print(json.dumps({"results": str(RESULT_ROOT), "figures": str(FIGURE_ROOT), "result_rows": len(results), "mechanism_rows": len(sample), "figures_written": 5}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
