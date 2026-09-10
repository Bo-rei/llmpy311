#!/usr/bin/env python3
"""Summarize the H1 protocol experiment and build OOS-first paper figures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
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
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs"
H1_ROOT = ARTIFACT_ROOT / "historical_protocol_v2" / "minilm_k1"
CURRENT_ROOT = ARTIFACT_ROOT / "protocol_v2_textoir_v1" / "e2_gate_core_dense"
OLD_DATA_ROOT = ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19"
MODEL_ROOT = ROOT.parent / "assets" / "models" / "all-MiniLM-L6-v2"
RESULT_ROOT = ROOT / "results" / "analysis" / "historical_protocol_v2"
FIGURE_ROOT = ROOT / "figures" / "historical_protocol_oos_v2"

DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
DATASET_LABELS = {
    "clinc150": "CLINC150",
    "stackoverflow": "StackOverflow",
    "banking77_oos": "BANKING77-OOS",
    "banking77": "Banking77",
}
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87)
METHODS = ("frozen_k1", "trainable_k1")
METHOD_LABELS = {"frozen_k1": "Frozen K=1", "trainable_k1": "Trainable K=1"}
METHOD_COLORS = {"frozen_k1": "#4C78A8", "trainable_k1": "#C44E52"}
OOS_COLOR = "#C44E52"
KNOWN_COLOR = "#B7BEC7"
THRESHOLD_COLOR = "#252525"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"empty prediction file: {path}")
    return frame


def old_run_dir(dataset: str, kir: float, seed: int) -> Path:
    return H1_ROOT / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}"


def current_run_dir(dataset: str, seed: int) -> Path:
    return CURRENT_ROOT / (
        f"protocol_v2_textoir_v1__{dataset}__kir_0.50__seed_{seed}__"
        "repr_frozen_minilm__k_1__dist_mahalanobis_diag__boundary_mean_std"
    )


def source_summary(frame: pd.DataFrame) -> str:
    oos = frame[frame["gold_is_oos"].astype(int) == 1]
    counts = Counter(str(value) for value in oos.get("oos_source", pd.Series(dtype=str)))
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def load_old_results() -> tuple[pd.DataFrame, dict[tuple[str, float, int, str], pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, float, int, str], pd.DataFrame] = {}
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                run_root = old_run_dir(dataset, kir, seed)
                for method in METHODS:
                    method_root = run_root / method
                    metrics_path = method_root / "metrics.json"
                    prediction_path = method_root / "predictions.jsonl"
                    if not metrics_path.is_file() or not prediction_path.is_file():
                        raise FileNotFoundError(f"incomplete H1 result: {method_root}")
                    metrics = read_json(metrics_path)
                    frame = read_jsonl(prediction_path)
                    predictions[(dataset, kir, seed, method)] = frame
                    rows.append(
                        {
                            "protocol": "h1_v19",
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
                            "known_count": int((frame["gold_is_oos"].astype(int) == 0).sum()),
                            "oos_count": int((frame["gold_is_oos"].astype(int) == 1).sum()),
                            "oos_sources": source_summary(frame),
                            "metrics_path": str(metrics_path),
                            "predictions_path": str(prediction_path),
                        }
                    )
    return pd.DataFrame(rows), predictions


def load_current_results() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in ("clinc150", "stackoverflow", "banking77"):
        for seed in SEEDS:
            run_root = current_run_dir(dataset, seed)
            metrics_path = run_root / "metrics.json"
            prediction_path = run_root / "predictions" / "test.jsonl"
            if not metrics_path.is_file() or not prediction_path.is_file():
                raise FileNotFoundError(f"incomplete current Frozen result: {run_root}")
            metrics = read_json(metrics_path)["combined"]
            frame = read_jsonl(prediction_path)
            rows.append(
                {
                    "protocol": "protocol_v2",
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "kir": 0.50,
                    "seed": seed,
                    "method": "frozen_k1",
                    "method_label": "Frozen K=1",
                    "oos_f1": float(metrics["oos_f1"]),
                    "known_recall": float(metrics["id_recall"]),
                    "false_accept_rate": float(metrics["false_accept_rate"]),
                    "auroc": float(metrics["auroc"]),
                    "aupr_oos": float(metrics["aupr_oos"]),
                    "known_count": int((frame["gold_is_oos"].astype(int) == 0).sum()),
                    "oos_count": int((frame["gold_is_oos"].astype(int) == 1).sum()),
                    "oos_sources": source_summary(frame),
                    "metrics_path": str(metrics_path),
                    "predictions_path": str(prediction_path),
                }
            )
    return pd.DataFrame(rows)


def write_tables(old: pd.DataFrame, current: pd.DataFrame, predictions: dict[tuple[str, float, int, str], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = pd.concat([old, current], ignore_index=True, sort=False)
    all_results.to_csv(RESULT_ROOT / "historical_protocol_supplement_results.csv", index=False)

    protocol = current.copy()
    old_kir = old[old["kir"].eq(0.50)].copy()
    protocol = pd.concat([old_kir, protocol], ignore_index=True, sort=False)
    protocol.to_csv(RESULT_ROOT / "historical_protocol_attribution.csv", index=False)

    summary = (
        old.groupby(["dataset", "dataset_label", "kir", "method", "method_label"], as_index=False)
        .agg(
            oos_f1=("oos_f1", "mean"),
            oos_f1_std=("oos_f1", "std"),
            known_recall=("known_recall", "mean"),
            false_accept_rate=("false_accept_rate", "mean"),
            auroc=("auroc", "mean"),
            aupr_oos=("aupr_oos", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(RESULT_ROOT / "historical_protocol_kir_summary.csv", index=False)

    transitions = build_transitions(predictions)
    transitions.to_csv(RESULT_ROOT / "historical_oos_transitions.csv", index=False)
    return summary, transitions


def build_transitions(predictions: dict[tuple[str, float, int, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            frozen = predictions[(dataset, 0.50, seed, "frozen_k1")]
            trainable = predictions[(dataset, 0.50, seed, "trainable_k1")]
            required = {"sample_id", "gold_is_oos", "predicted_is_oos", "oos_score"}
            if not required.issubset(frozen.columns) or not required.issubset(trainable.columns):
                raise ValueError(f"prediction schema missing for {dataset}/{seed}")
            left = frozen[list(required)].rename(columns={"predicted_is_oos": "frozen_pred", "oos_score": "frozen_score"})
            right = trainable[list(required)].rename(columns={"gold_is_oos": "gold_is_oos_trainable", "predicted_is_oos": "trainable_pred", "oos_score": "trainable_score"})
            merged = left.merge(right, on="sample_id", validate="one_to_one")
            merged = merged[merged["gold_is_oos"].astype(int).eq(1)].copy()
            frozen_pred = merged["frozen_pred"].astype(int)
            trainable_pred = merged["trainable_pred"].astype(int)
            merged["transition"] = np.select(
                [frozen_pred.eq(1) & trainable_pred.eq(1), frozen_pred.eq(0) & trainable_pred.eq(1), frozen_pred.eq(1) & trainable_pred.eq(0)],
                ["both_correct", "trainable_only", "frozen_only"],
                default="both_wrong",
            )
            merged["delta_score"] = merged["trainable_score"].astype(float) - merged["frozen_score"].astype(float)
            for transition, group in merged.groupby("transition", sort=False):
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "kir": 0.50,
                        "seed": seed,
                        "transition": transition,
                        "rate": len(group) / len(merged),
                        "mean_delta_score": float(group["delta_score"].mean()),
                        "n_oos": len(merged),
                    }
                )
    raw = pd.DataFrame(rows)
    return (
        raw.groupby(["dataset", "dataset_label", "kir", "transition"], as_index=False)
        .agg(rate=("rate", "mean"), rate_std=("rate", "std"), mean_delta_score=("mean_delta_score", "mean"), n_seeds=("seed", "nunique"))
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.tick_params(length=3, width=0.7, pad=2)
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_protocol_effect(protocol: pd.DataFrame) -> None:
    protocol = protocol[protocol["method"] == "frozen_k1"].copy()
    grouped = (
        protocol.groupby(["protocol", "dataset"], as_index=False)
        .agg(oos_f1=("oos_f1", "mean"), oos_f1_std=("oos_f1", "std"), known_count=("known_count", "mean"), oos_count=("oos_count", "mean"))
    )
    fig, (ax_comp, ax_f1) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": (0.92, 1.08)})
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.84, wspace=0.38)

    composition = [
        ("CLINC150", "h1_v19", "clinc150"),
        ("CLINC150", "protocol_v2", "clinc150"),
        ("StackOverflow", "h1_v19", "stackoverflow"),
        ("StackOverflow", "protocol_v2", "stackoverflow"),
        ("BANKING77-OOS", "h1_v19", "banking77_oos"),
        ("Banking77", "protocol_v2", "banking77"),
    ]
    y = np.arange(len(composition))
    for i, (_, protocol_name, dataset) in enumerate(composition):
        row = grouped[(grouped["protocol"] == protocol_name) & (grouped["dataset"] == dataset)]
        if row.empty:
            continue
        known = float(row.iloc[0]["known_count"])
        oos = float(row.iloc[0]["oos_count"])
        ax_comp.barh(i, known, color=KNOWN_COLOR, height=0.56)
        ax_comp.barh(i, oos, left=known, color=OOS_COLOR, height=0.56)
        if i == 0:
            ax_comp.text(known / 2, i, "Known", ha="center", va="center", fontsize=6.2, color="#4D5156")
        ax_comp.text(known + oos + max(20, (known + oos) * 0.02), i, f"OOS {int(oos)}", va="center", fontsize=6.2, color=OOS_COLOR)
    ax_comp.set_yticks(y, [f"{name}\n{('old v19' if p == 'h1_v19' else 'protocol v2')}" for name, p, _ in composition])
    ax_comp.invert_yaxis()
    ax_comp.set_xlabel("Gate-test samples")
    ax_comp.set_title("Test composition", loc="left", fontweight="bold")
    ax_comp.set_xlim(0, max(grouped["known_count"] + grouped["oos_count"]) * 1.23)
    _style_axes(ax_comp)
    ax_comp.grid(axis="x", color="#E5E8EB", linewidth=0.45)
    ax_comp.grid(axis="y", visible=False)

    x = np.arange(3)
    old_map = {"clinc150": "CLINC150", "stackoverflow": "StackOverflow", "banking77_oos": "BANKING77-OOS"}
    current_map = {"clinc150": "CLINC150", "stackoverflow": "StackOverflow", "banking77": "Banking77"}
    for index, (old_ds, current_ds) in enumerate(zip(old_map, current_map)):
        old_row = grouped[(grouped.protocol == "h1_v19") & (grouped.dataset == old_ds)].iloc[0]
        current_row = grouped[(grouped.protocol == "protocol_v2") & (grouped.dataset == current_ds)].iloc[0]
        if index < 2:
            ax_f1.plot([index - 0.13, index + 0.13], [old_row.oos_f1 * 100, current_row.oos_f1 * 100], color="#A5ABB2", linewidth=0.8, zorder=1)
        ax_f1.errorbar(index - 0.13, old_row.oos_f1 * 100, yerr=0 if pd.isna(old_row.oos_f1_std) else old_row.oos_f1_std * 100, fmt="o", color="#4C78A8", markersize=4.5, capsize=2, zorder=3)
        ax_f1.errorbar(index + 0.13, current_row.oos_f1 * 100, yerr=0 if pd.isna(current_row.oos_f1_std) else current_row.oos_f1_std * 100, fmt="o", color="#C44E52", markersize=4.5, capsize=2, zorder=3)
        label = f"Δ {(current_row.oos_f1 - old_row.oos_f1) * 100:+.1f} pp" if index < 2 else "not paired"
        ax_f1.text(index, max(old_row.oos_f1, current_row.oos_f1) * 100 + 2.0, label, ha="center", fontsize=6.2, color="#4D5156")
    ax_f1.set_xticks(x, ["CLINC150", "StackOverflow", "Banking77\nvs BANKING77-OOS"])
    ax_f1.set_ylabel("Frozen K=1 OOS F1 (%)")
    ax_f1.set_title("Same detector, different input contract", loc="left", fontweight="bold")
    ax_f1.set_ylim(max(45, float(grouped.oos_f1.min() * 100 - 8)), min(100, float(grouped.oos_f1.max() * 100 + 8)))
    ax_f1.axhline(0, color="#252525", linewidth=0.5)
    _style_axes(ax_f1)
    ax_f1.grid(axis="x", visible=False)
    ax_f1.text(0.02, -0.22, "Blue: old v19   Red: protocol v2; Banking77 and BANKING77-OOS are not the same dataset key.", transform=ax_f1.transAxes, fontsize=6.1, color="#5D646B")
    fig.text(0.01, 0.965, "a", fontsize=9, fontweight="bold", va="top")
    fig.text(0.52, 0.965, "b", fontsize=9, fontweight="bold", va="top")
    save_figure(fig, "protocol_effect_oos")


def _mean_histogram(frames: list[pd.DataFrame], mask_column: str, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for frame in frames:
        values = frame.loc[frame[mask_column].astype(bool), "oos_score"].astype(float).to_numpy()
        density, _ = np.histogram(values, bins=bins, density=True)
        curves.append(density)
    return (bins[:-1] + bins[1:]) / 2, np.mean(curves, axis=0)


def plot_score_and_transitions(old: pd.DataFrame, predictions: dict[tuple[str, float, int, str], pd.DataFrame], transitions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), gridspec_kw={"height_ratios": (1.16, 0.84)})
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.13, top=0.80, hspace=0.58, wspace=0.27)
    state_order = ("both_correct", "trainable_only", "frozen_only", "both_wrong")
    state_labels = ("both correct", "Trainable only", "Frozen only", "both wrong")
    state_colors = ("#A5ABB2", "#C44E52", "#4C78A8", "#D9DDE1")
    for col, dataset in enumerate(DATASETS):
        panel = old[(old.dataset == dataset) & old.kir.eq(0.50)]
        all_frames = [predictions[(dataset, 0.50, seed, method)] for seed in SEEDS for method in METHODS]
        values = np.concatenate([frame["oos_score"].astype(float).to_numpy() for frame in all_frames])
        high = max(1.35, float(np.quantile(values, 0.995) * 1.06))
        bins = np.linspace(0.0, high, 64)
        ax = axes[0, col]
        for method, color, label in (("frozen_k1", METHOD_COLORS["frozen_k1"], "Frozen"), ("trainable_k1", METHOD_COLORS["trainable_k1"], "Trainable")):
            frames = [predictions[(dataset, 0.50, seed, method)] for seed in SEEDS]
            centers, oos_curve = _mean_histogram(frames, "gold_is_oos", bins)
            ax.plot(centers, oos_curve, color=color, linewidth=1.45, label=label)
        known_frames = []
        for seed in SEEDS:
            for method in METHODS:
                frame = predictions[(dataset, 0.50, seed, method)].copy()
                frame["known_mask"] = ~frame["gold_is_oos"].astype(bool)
                known_frames.append(frame)
        centers, known_curve = _mean_histogram(known_frames, "known_mask", bins)
        axes[0, col].plot(centers, known_curve, color=KNOWN_COLOR, linewidth=1.0, linestyle="--", label="Known reference")
        ax.axvline(1.0, color=THRESHOLD_COLOR, linestyle="--", linewidth=0.8)
        metric = panel.groupby("method", as_index=False).mean(numeric_only=True).set_index("method")
        ax.text(0.03, 0.94, f"F1 {metric.loc['frozen_k1', 'oos_f1'] * 100:.1f} → {metric.loc['trainable_k1', 'oos_f1'] * 100:.1f}\nFA {metric.loc['frozen_k1', 'false_accept_rate'] * 100:.1f}% → {metric.loc['trainable_k1', 'false_accept_rate'] * 100:.1f}%", transform=ax.transAxes, va="top", fontsize=6.2, color="#4D5156")
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("normalized OOS score")
        if col == 0:
            ax.set_ylabel("mean density")
        ax.set_xlim(0, high)
        _style_axes(ax)
        ax.grid(axis="y", color="#E5E8EB", linewidth=0.4)

        ax = axes[1, col]
        t = transitions[transitions.dataset == dataset].set_index("transition").reindex(state_order)
        bottom = 0.0
        for state, label, color in zip(state_order, state_labels, state_colors):
            value = float(t.loc[state, "rate"] * 100)
            ax.bar(0, value, bottom=bottom, color=color, width=0.62, edgecolor="white", linewidth=0.35, label=label)
            if value >= 7:
                ax.text(0, bottom + value / 2, f"{value:.0f}%", ha="center", va="center", fontsize=6.1, color="white" if state in {"trainable_only", "frozen_only"} else "#30353A")
            bottom += value
        delta = float(metric.loc["trainable_k1", "oos_f1"] - metric.loc["frozen_k1", "oos_f1"]) * 100
        ax.text(0, 103, f"ΔF1 {delta:+.1f} pp", ha="center", va="bottom", fontsize=6.4, color="#C44E52" if delta >= 0 else "#4C78A8")
        ax.set_title("OOS decision transitions", fontsize=7.8, loc="left")
        ax.set_ylim(0, 116)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel("share of OOS (%)")
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
    score_handles, score_labels = axes[0, 2].get_legend_handles_labels()
    transition_handles, transition_labels = axes[1, 0].get_legend_handles_labels()
    fig.text(0.075, 0.94, "OOS score distribution and decision transitions", fontsize=8.8, va="center")
    fig.legend(score_handles, score_labels, loc="upper right", bbox_to_anchor=(0.99, 0.95), ncol=3, handlelength=1.6, columnspacing=0.9)
    fig.legend(transition_handles, transition_labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, handlelength=1.3, columnspacing=1.0)
    axes[0, 0].text(-0.16, 1.11, "a", transform=axes[0, 0].transAxes, fontsize=9, fontweight="bold", va="top")
    axes[1, 0].text(-0.16, 1.11, "b", transform=axes[1, 0].transAxes, fontsize=9, fontweight="bold", va="top")
    save_figure(fig, "oos_score_and_transition")


def plot_kir_robustness(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.80, wspace=0.23)
    for index, dataset in enumerate(DATASETS):
        ax = axes[index]
        panel = summary[summary.dataset == dataset]
        for method in METHODS:
            line = panel[panel.method == method].sort_values("kir")
            x = line.kir.to_numpy() * 100
            y = line.oos_f1.to_numpy() * 100
            spread = line.oos_f1_std.fillna(0).to_numpy() * 100
            ax.plot(x, y, color=METHOD_COLORS[method], linewidth=1.45, marker="o", markersize=3.5, label=METHOD_LABELS[method])
            ax.fill_between(x, y - spread, y + spread, color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("KIR (%)")
        ax.set_xticks([25, 50, 75])
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
        if index == 0:
            ax.set_ylabel("OOS F1 (%)")
        if index == 2:
            ax.legend(loc="lower left", handlelength=1.8)
    lo = float(summary.oos_f1.min() * 100 - 5)
    hi = float(summary.oos_f1.max() * 100 + 5)
    axes[0].set_ylim(max(0, lo), min(100, hi))
    fig.suptitle("Trainable representation improves the OOS work point across KIR", fontsize=9, y=0.98)
    fig.text(0.01, 0.01, "Lines: three-seed means; ribbons: ±1 s.d.; detector and threshold are fixed across methods.", fontsize=6.2, color="#5D646B")
    save_figure(fig, "oos_kir_robustness")


def plot_oos_sources(protocol: pd.DataFrame) -> None:
    mapping = {
        "test": "held-out intent",
        "unknown": "held-out intent",
        "heldout_oos_test": "held-out intent",
        "heldout_intent": "held-out intent",
        "oos_test": "native OOS",
        "native": "native OOS",
        "id_oos": "ID OOS",
        "id-oos_test": "ID OOS",
        "ood_oos": "OOD OOS",
        "ood-oos_test": "OOD OOS",
        "other_oos": "other OOS",
    }
    rows = []
    for _, row in protocol.drop_duplicates(["protocol", "dataset", "seed"]).iterrows():
        counts = {}
        for item in str(row["oos_sources"]).split(";"):
            if not item:
                continue
            key, value = item.rsplit(":", 1)
            counts[mapping.get(key, key)] = counts.get(mapping.get(key, key), 0) + int(value)
        for source, count in counts.items():
            rows.append({"protocol": row["protocol"], "dataset": row["dataset"], "dataset_label": row["dataset_label"], "source": source, "count": count})
    source_df = pd.DataFrame(rows).groupby(["protocol", "dataset", "dataset_label", "source"], as_index=False).mean(numeric_only=True)
    source_order = ("held-out intent", "native OOS", "ID OOS", "OOD OOS", "other OOS")
    colors = {"held-out intent": "#C44E52", "native OOS": "#E68A8A", "ID OOS": "#D76B6B", "OOD OOS": "#8C9BAA", "other OOS": "#AEB6BE"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.29, top=0.80, wspace=0.24)
    groups = (
        ("clinc150", "clinc150"),
        ("stackoverflow", "stackoverflow"),
        ("banking77", "banking77_oos"),
    )
    for index, (group, old_group) in enumerate(groups):
        ax = axes[index]
        subset = source_df[source_df.dataset.isin({group, old_group})]
        protocols = ["h1_v19", "protocol_v2"]
        x = np.arange(2)
        bottom = np.zeros(2)
        for source in source_order:
            values = []
            for protocol_name in protocols:
                match = subset[(subset.protocol == protocol_name) & (subset.source == source)]
                values.append(float(match.iloc[0]["count"]) if not match.empty else 0.0)
            ax.bar(x, values, bottom=bottom, color=colors[source], width=0.58, edgecolor="white", linewidth=0.3, label=source)
            bottom += np.asarray(values)
        ax.set_xticks(x, ["old v19", "protocol v2"])
        ax.set_title("Banking77\nvs BANKING77-OOS" if group == "banking77" else DATASET_LABELS[group], loc="left", fontweight="bold")
        ax.set_xlabel("input contract")
        if index == 0:
            ax.set_ylabel("OOS test samples")
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, handlelength=1.4, columnspacing=1.0)
    fig.suptitle("The OOS test distribution changes with the data protocol", fontsize=9, y=0.98)
    save_figure(fig, "oos_source_composition")


def load_geometry_arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    import sys

    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "src"))
    from protocol_v2.experiments.racal_v1.representation import RacalMiniLM, encode_rows

    data_root = OLD_DATA_ROOT / "clinc150" / "kir50_seed42"
    known_manifest = read_json(data_root / "KNOWN_INTENTS.json")
    known = {str(value) for value in known_manifest["known_intents"]}
    train_rows = json.loads((data_root / "gate" / "train.json").read_text(encoding="utf-8"))
    test_rows = json.loads((data_root / "gate" / "test.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen_encoder = SentenceTransformer(str(MODEL_ROOT), device=str(device))
    frozen_train = np.asarray(frozen_encoder.encode([str(row["text"]) for row in train_rows], batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False), dtype=np.float32)
    frozen_test = np.asarray(frozen_encoder.encode([str(row["text"]) for row in test_rows], batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False), dtype=np.float32)
    del frozen_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    trainable_checkpoint = H1_ROOT / "clinc150" / "kir50_seed42" / "trainable_k1" / "checkpoint.pt"
    checkpoint = torch.load(trainable_checkpoint, map_location="cpu", weights_only=True)
    hidden = int(checkpoint["model"]["projection.fc1.weight"].shape[0])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True)
    trainable_encoder = RacalMiniLM(MODEL_ROOT, str(checkpoint["mode"]), hidden).to(device)
    trainable_encoder.load_state_dict(checkpoint["model"])
    trainable_encoder.eval()
    trainable_train = encode_rows(trainable_encoder, tokenizer, train_rows, device, 64, 256)
    trainable_test = encode_rows(trainable_encoder, tokenizer, test_rows, device, 64, 256)
    arrays: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for method, train_values, test_values in (("frozen_k1", frozen_train, frozen_test), ("trainable_k1", trainable_train, trainable_test)):
        projection = PCA(n_components=2).fit(train_values)
        arrays[method] = projection.transform(test_values)
        labels[method] = np.asarray([int(str(row["intent"]) not in known) for row in test_rows], dtype=np.int8)
    return arrays, labels


def plot_oos_geometry(predictions: dict[tuple[str, float, int, str], pd.DataFrame]) -> None:
    from matplotlib.patches import Patch

    arrays, labels = load_geometry_arrays()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.82, wspace=0.28)

    def fill_density(ax: plt.Axes, values: np.ndarray, color: str, alpha: float) -> None:
        histogram, x_edges, y_edges = np.histogram2d(values[:, 0], values[:, 1], bins=42)
        positive = histogram[histogram > 0]
        if positive.size == 0:
            return
        levels = np.unique(np.quantile(positive, [0.70, 0.85, 0.95]))
        if levels.size < 2:
            return
        levels = np.r_[levels, positive.max() + 1e-6]
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        xx, yy = np.meshgrid(x_centers, y_centers)
        ax.contourf(xx, yy, histogram.T, levels=levels, colors=[color] * (len(levels) - 1), alpha=alpha, antialiased=True)
        ax.contour(xx, yy, histogram.T, levels=levels[:-1], colors=[color], linewidths=0.55, alpha=min(0.9, alpha + 0.25))

    for ax, method in zip(axes, METHODS):
        xy = arrays[method]
        is_oos = labels[method].astype(bool)
        fill_density(ax, xy[~is_oos], KNOWN_COLOR, 0.28)
        fill_density(ax, xy[is_oos], OOS_COLOR, 0.22)
        frame = predictions[("clinc150", 0.50, 42, method)]
        fa = float(frame.loc[frame["gold_is_oos"].astype(int).eq(1), "predicted_is_oos"].astype(int).eq(0).mean())
        ax.text(0.03, 0.96, f"OOS false acceptance {fa * 100:.1f}%", transform=ax.transAxes, va="top", fontsize=6.5, color="#4D5156")
        ax.set_title(METHOD_LABELS[method], loc="left", fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(False)
        ax.set_aspect("equal", adjustable="box")
    axes[1].text(0.01, -0.24, "Grey: Known test samples   Red: OOS test samples   PCA is for visualization only; decisions use the 384-D representation.", transform=axes[1].transAxes, fontsize=6.1, color="#5D646B")
    fig.suptitle("OOS occupies a different region after representation adaptation", fontsize=8.8, y=0.98)
    fig.legend(
        handles=[Patch(facecolor=KNOWN_COLOR, edgecolor=KNOWN_COLOR, alpha=0.45, label="Known density"), Patch(facecolor=OOS_COLOR, edgecolor=OOS_COLOR, alpha=0.45, label="OOS density")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=2,
        handlelength=1.2,
        columnspacing=1.0,
    )
    axes[0].text(-0.16, 1.11, "a", transform=axes[0].transAxes, fontsize=9, fontweight="bold", va="top")
    axes[1].text(-0.16, 1.11, "b", transform=axes[1].transAxes, fontsize=9, fontweight="bold", va="top")
    save_figure(fig, "oos_geometry_pca")


def main() -> None:
    old, predictions = load_old_results()
    current = load_current_results()
    summary, transitions = write_tables(old, current, predictions)
    protocol = pd.read_csv(RESULT_ROOT / "historical_protocol_attribution.csv")
    plot_protocol_effect(protocol)
    plot_score_and_transitions(old, predictions, transitions)
    plot_kir_robustness(summary)
    plot_oos_sources(protocol)
    plot_oos_geometry(predictions)
    print(json.dumps({"results": str(RESULT_ROOT), "figures": str(FIGURE_ROOT), "old_rows": len(old), "current_rows": len(current), "figures_written": 5}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
