#!/usr/bin/env python3
"""Build sample- and structure-level explanations for the current baselines.

This is a post-hoc analysis stage.  It consumes the audited prediction
contract plus existing representation and MOGB ball summaries.  It does not
train, tune, choose a threshold, choose K, or write into an experiment run.
The outputs are intentionally aggregate; sample IDs are used only while
joining same-sample predictions and are not exported.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2c-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
CONTRACT = WORKSPACE / "artifacts/s2c/analysis/unified_prediction_contract_v1/predictions.jsonl.gz"
OUT = ROOT / "results/analysis/mechanism_explanation_v1"
FIG = ROOT / "figures/mechanism_explanation_v1"
BALLS = ROOT / "results/analysis/archive/analysis/mogb_ball_risk_attribution_v1/per_ball.csv"
GEOMETRY = ROOT / "results/representation/representation_geometry_summary.csv"
REP_RESULTS = ROOT / "results/representation/representation_fixed_results.csv"
COLLISIONS = ROOT / "results/representation/collision_summary.csv"

DATASETS = ("clinc150", "banking77", "stackoverflow")
DATASET_LABELS = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}

METHOD_LABELS = {
    "S2C-Trainable-K1": "S2C Trainable K=1",
    "S2C-Frozen-K1": "Frozen K=1",
    "S2C-Frozen-K2": "Frozen K=2",
    "S2C-random-K2": "Random K=2",
    "MOGB-MiniLM": "MOGB-MiniLM",
    "MOGB-partition-S2C-boundary": "MOGB partition + S2C boundary",
    "S2C-partition-MOGB-boundary": "S2C partition + MOGB boundary",
    "Native-MSP-Trainable-MiniLM": "MSP",
    "Native-Energy-Trainable-MiniLM": "Energy",
    "Native-kNN-Trainable-MiniLM": "kNN",
    "Native-LOF-Trainable-MiniLM": "LOF",
    "ADB-external-BERT": "ADB (external BERT)",
}
COLORS = {
    "S2C-Trainable-K1": "#0072B2",
    "S2C-Frozen-K1": "#666666",
    "S2C-Frozen-K2": "#D55E00",
    "S2C-random-K2": "#009E73",
    "MOGB-MiniLM": "#CC79A7",
    "MOGB-partition-S2C-boundary": "#E69F00",
    "S2C-partition-MOGB-boundary": "#56B4E9",
    "Native-MSP-Trainable-MiniLM": "#332288",
    "Native-Energy-Trainable-MiniLM": "#117733",
    "Native-kNN-Trainable-MiniLM": "#AA4499",
    "Native-LOF-Trainable-MiniLM": "#CC6677",
}
FAIR_METHODS = (
    "S2C-Trainable-K1",
    "S2C-Frozen-K1",
    "S2C-Frozen-K2",
    "S2C-random-K2",
    "MOGB-MiniLM",
    "MOGB-partition-S2C-boundary",
    "S2C-partition-MOGB-boundary",
)
NATIVE_METHODS = (
    "S2C-Trainable-K1",
    "Native-MSP-Trainable-MiniLM",
    "Native-Energy-Trainable-MiniLM",
    "Native-kNN-Trainable-MiniLM",
    "Native-LOF-Trainable-MiniLM",
)
CONTRACT_METHODS = set(FAIR_METHODS) | set(NATIVE_METHODS) | {"ADB-external-BERT"}
CORRECT_OUTCOMES = {"known_correct", "oos_correct"}
OUTCOME_LABELS = {
    "known_correct": "Known correct",
    "known_wrong_intent": "Known wrong intent",
    "known_rejected": "Known rejected",
    "oos_correct": "OOS correct",
    "oos_false_accept": "OOS false accept",
}
TRANSITION_COLORS = {
    "both_correct": "#999999",
    "s2c_only_correct": "#0072B2",
    "baseline_only_correct": "#D55E00",
    "both_wrong": "#C44E52",
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


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, stem: str) -> None:
    """Write an editable vector pair plus a high-resolution preview."""
    figure.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight")
    figure.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")


def classify(row: pd.Series) -> str:
    if bool(row["is_true_oos"]):
        return "oos_correct" if bool(row["predicted_oos"]) else "oos_false_accept"
    if bool(row["predicted_oos"]):
        return "known_rejected"
    if str(row["predicted_label"]) == str(row["true_label"]):
        return "known_correct"
    return "known_wrong_intent"


def load_contract() -> pd.DataFrame:
    if not CONTRACT.is_file():
        raise FileNotFoundError(CONTRACT)
    rows: list[dict[str, Any]] = []
    missing_score_rows = 0
    missing_score_methods: set[str] = set()
    with gzip.open(CONTRACT, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("split") != "test":
                continue
            if abs(float(item.get("kir", -1.0)) - 0.50) > 1e-9:
                continue
            method = str(item.get("method", ""))
            if method not in CONTRACT_METHODS:
                continue
            raw_score = item.get("oos_score")
            score = float(raw_score) if raw_score is not None else np.nan
            if raw_score is None:
                missing_score_rows += 1
                missing_score_methods.add(method)
            elif not np.isfinite(score):
                raise ValueError(f"non-finite score for {method}")
            rows.append(
                {
                    "dataset": str(item["dataset"]),
                    "seed": int(item["seed"]),
                    "sample_id": str(item["sample_id"]),
                    "method": method,
                    "true_label": str(item.get("true_label", "")),
                    "is_true_oos": int(item["is_true_oos"]),
                    "predicted_label": str(item.get("predicted_label", "")),
                    "predicted_oos": int(item["predicted_oos"]),
                    "oos_score": score,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("contract filter returned no KIR=0.50 test rows")
    frame["outcome"] = frame.apply(classify, axis=1)
    frame["method_label"] = frame["method"].map(METHOD_LABELS)
    frame["dataset_label"] = frame["dataset"].map(DATASET_LABELS)
    expected = set(DATASETS)
    if set(frame["dataset"]) != expected:
        raise ValueError(f"unexpected datasets: {sorted(set(frame['dataset']))}")
    frame.attrs["missing_score_rows"] = missing_score_rows
    frame.attrs["missing_score_methods"] = sorted(missing_score_methods)
    return frame


def paired_scores(frame: pd.DataFrame, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = frame[frame.method.eq("S2C-Trainable-K1")].copy()
    right = frame[frame.method.eq(baseline)].copy()
    keys = ["dataset", "seed", "sample_id"]
    merged = left.merge(right, on=keys, suffixes=("_s2c", "_baseline"), validate="one_to_one")
    if merged.empty:
        raise ValueError(f"no paired rows for {baseline}")
    s2c_correct = merged.outcome_s2c.isin(CORRECT_OUTCOMES)
    baseline_correct = merged.outcome_baseline.isin(CORRECT_OUTCOMES)
    merged["transition"] = np.select(
        [s2c_correct & baseline_correct, s2c_correct & ~baseline_correct, ~s2c_correct & baseline_correct],
        ["both_correct", "s2c_only_correct", "baseline_only_correct"],
        default="both_wrong",
    )
    merged["gold_type"] = np.where(merged.is_true_oos_s2c.astype(bool), "OOS", "Known")
    merged["score_delta_s2c_minus_baseline"] = merged.oos_score_s2c - merged.oos_score_baseline
    summary = (
        merged.groupby(["dataset", "gold_type", "transition"], as_index=False)
        .agg(
            count=("sample_id", "size"),
            score_delta_mean=("score_delta_s2c_minus_baseline", "mean"),
            score_delta_median=("score_delta_s2c_minus_baseline", "median"),
        )
    )
    summary["rate_within_dataset_gold_type"] = summary["count"] / summary.groupby(["dataset", "gold_type"])["count"].transform("sum")
    summary.insert(1, "baseline_method", baseline)
    return merged, summary


def plot_paired_scores(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = ("S2C-Frozen-K1", "MOGB-MiniLM")
    pair_frames: list[pd.DataFrame] = []
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True, sharex=False, sharey=False)
    for row_index, baseline in enumerate(pairs):
        merged, summary = paired_scores(frame, baseline)
        pair_frames.append(summary)
        for col_index, dataset in enumerate(DATASETS):
            ax = axes[row_index, col_index]
            sub = merged[merged.dataset.eq(dataset)]
            ax.scatter(
                sub.oos_score_baseline,
                sub.oos_score_s2c,
                s=4,
                alpha=0.10,
                color="#777777",
                rasterized=True,
            )
            for transition, color in TRANSITION_COLORS.items():
                points = sub[sub.transition.eq(transition)]
                if len(points) > 3500:
                    points = points.sort_values(["seed", "sample_id"]).head(3500)
                ax.scatter(
                    points.oos_score_baseline,
                    points.oos_score_s2c,
                    s=5,
                    alpha=0.32,
                    color=color,
                    label=transition.replace("_", " "),
                    rasterized=True,
                )
            lo = float(np.nanquantile(np.r_[sub.oos_score_baseline, sub.oos_score_s2c], 0.002))
            hi = float(np.nanquantile(np.r_[sub.oos_score_baseline, sub.oos_score_s2c], 0.998))
            lo = min(lo, 0.85)
            hi = max(hi, 1.15)
            ax.plot([lo, hi], [lo, hi], color="#aaaaaa", linewidth=0.8, linestyle="--")
            ax.axvline(1.0, color="#333333", linewidth=0.8)
            ax.axhline(1.0, color="#333333", linewidth=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{DATASET_LABELS[dataset]} / {METHOD_LABELS[baseline]}")
            ax.set_xlabel("Baseline oos_score")
            ax.set_ylabel("S2C Trainable K=1 oos_score")
            ax.grid(alpha=0.18)
            counts = sub.transition.value_counts()
            ax.text(
                0.03,
                0.97,
                f"S2C-only={counts.get('s2c_only_correct', 0)}\nBaseline-only={counts.get('baseline_only_correct', 0)}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Same-sample score reordering at KIR=0.50", y=1.06, fontsize=12)
    save_figure(fig, "paired_score_reordering")
    plt.close(fig)
    return pd.concat(pair_frames, ignore_index=True)


def build_margin_rates(frame: pd.DataFrame) -> pd.DataFrame:
    methods = ("S2C-Trainable-K1", "S2C-Frozen-K1", "S2C-Frozen-K2", "MOGB-MiniLM", "MOGB-partition-S2C-boundary", "S2C-partition-MOGB-boundary")
    bins = [-np.inf, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, np.inf]
    labels = ("<0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0", "1.0–1.2", "1.2–1.4", "1.4–1.6", "1.6–2.0", ">=2.0")
    data = frame[frame.method.isin(methods)].copy()
    data["score_bin"] = pd.cut(data.oos_score, bins=bins, labels=labels, include_lowest=True)
    data["failure"] = np.where(
        data.is_true_oos.astype(bool),
        data.outcome.eq("oos_false_accept"),
        data.outcome.eq("known_rejected"),
    )
    data["gold_type"] = np.where(data.is_true_oos.astype(bool), "OOS", "Known")
    result = (
        data.groupby(["dataset", "method", "gold_type", "score_bin"], observed=True, as_index=False)
        .agg(sample_count=("failure", "size"), failure_rate=("failure", "mean"))
    )
    result["class_mass_rate"] = result["sample_count"] / result.groupby(["dataset", "method", "gold_type"])["sample_count"].transform("sum")
    result["method_label"] = result.method.map(METHOD_LABELS)
    result["score_bin"] = result.score_bin.astype(str)
    return result


def plot_margin_rates(margin: pd.DataFrame) -> None:
    methods = ("S2C-Trainable-K1", "S2C-Frozen-K1", "S2C-Frozen-K2", "MOGB-MiniLM", "MOGB-partition-S2C-boundary", "S2C-partition-MOGB-boundary")
    bins = ("<0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0", "1.0–1.2", "1.2–1.4", "1.4–1.6", "1.6–2.0", ">=2.0")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True, sharey="row")
    for col, dataset in enumerate(DATASETS):
        for row, gold_type in enumerate(("Known", "OOS")):
            ax = axes[row, col]
            sub = margin[(margin.dataset.eq(dataset)) & (margin.gold_type.eq(gold_type))]
            for method in methods:
                line = sub[sub.method.eq(method)].set_index("score_bin").reindex(bins)
                ax.plot(
                    np.arange(len(bins)),
                    line.class_mass_rate.to_numpy(float) * 100,
                    marker="o",
                    markersize=3,
                    linewidth=1.3,
                    color=COLORS[method],
                    label=METHOD_LABELS[method],
                )
            ax.axvline(3.5, color="#333333", linewidth=0.8, linestyle="--")
            ax.set_xticks(np.arange(len(bins)), bins, rotation=35, ha="right")
            ax.set_title(f"{DATASET_LABELS[dataset]} — {gold_type}")
            ax.set_xlabel("Normalized score bin (boundary=1.0)")
            ax.set_ylabel("Known score mass (%)" if gold_type == "Known" else "OOS score mass (%)")
            ax.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=8)
    fig.suptitle("Score mass around the boundary: overlap across methods", y=1.09, fontsize=12)
    save_figure(fig, "boundary_margin_fingerprint")
    plt.close(fig)


def build_intent_localization(frame: pd.DataFrame) -> pd.DataFrame:
    methods = ("S2C-Trainable-K1", "S2C-Frozen-K1", "MOGB-MiniLM")
    known = frame[(frame.method.isin(methods)) & (frame.is_true_oos.eq(0))].copy()
    grouped = (
        known.groupby(["dataset", "true_label", "method", "outcome"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = known.groupby(["dataset", "true_label", "method"], as_index=False).size().rename(columns={"size": "total"})
    grouped = grouped.merge(totals, on=["dataset", "true_label", "method"], how="left")
    grouped["rate"] = grouped["count"] / grouped["total"]
    pivot = grouped.pivot_table(index=["dataset", "true_label"], columns=["method", "outcome"], values="rate", fill_value=0.0)
    pivot.columns = [f"{method}__{outcome}" for method, outcome in pivot.columns]
    result = pivot.reset_index()
    for method in methods:
        for outcome in ("known_correct", "known_wrong_intent", "known_rejected"):
            column = f"{method}__{outcome}"
            if column not in result:
                result[column] = 0.0
    result["n_samples"] = known.groupby(["dataset", "true_label"]).size().reindex(pd.MultiIndex.from_frame(result[["dataset", "true_label"]])).to_numpy()
    result["mogb_minus_s2c_reject_gap_pp"] = (result["MOGB-MiniLM__known_rejected"] - result["S2C-Trainable-K1__known_rejected"]) * 100
    result["mogb_minus_s2c_correct_gap_pp"] = (result["S2C-Trainable-K1__known_correct"] - result["MOGB-MiniLM__known_correct"]) * 100
    return result


def plot_intent_localization(intent: pd.DataFrame) -> None:
    columns = (
        "MOGB-MiniLM__known_rejected",
        "S2C-Trainable-K1__known_rejected",
        "MOGB-MiniLM__known_wrong_intent",
        "S2C-Trainable-K1__known_wrong_intent",
    )
    labels = ("MOGB rejected", "S2C rejected", "MOGB wrong", "S2C wrong")
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        sub = intent[intent.dataset.eq(dataset)].nlargest(20, "mogb_minus_s2c_reject_gap_pp").sort_values("mogb_minus_s2c_reject_gap_pp")
        values = sub[list(columns)].to_numpy(float) * 100
        vmax = max(100.0, float(values.max()) if values.size else 0.0)
        image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=0, vmax=vmax)
        ax.set_yticks(np.arange(len(sub)), sub.true_label, fontsize=6)
        ax.set_xticks(np.arange(len(columns)), labels, rotation=35, ha="right")
        ax.set_title(f"{DATASET_LABELS[dataset]}\nTop MOGB excess-rejection intents")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=axes, label="Share of Known intent (%)", shrink=0.8)
    fig.suptitle("Error localization: which Known intents explain the baseline gap?", y=1.03, fontsize=12)
    save_figure(fig, "intent_error_localization")
    plt.close(fig)


def load_ball_risk() -> pd.DataFrame:
    if not BALLS.is_file():
        raise FileNotFoundError(BALLS)
    data = pd.read_csv(BALLS)
    data = data[(data.method.eq("mogb_minilm")) & (np.isclose(data.kir, 0.50))].copy()
    required = {"dataset", "seed", "ball_id", "majority_label", "radius", "sample_count", "test_oos_false_accept_count", "test_known_false_reject_count", "false_accept_share_cell", "known_false_reject_share_cell", "test_assignment_count"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"ball risk source missing {missing}")
    data["dataset"] = data.dataset.replace({"banking77": "banking77", "banking77_oos": "banking77"})
    return data


def plot_ball_risk(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True, sharex=False, sharey=False)
    panels = (
        ("false_accept_share_cell", "OOS false-accept share", "magma"),
        ("known_false_reject_share_cell", "Known false-reject share", "viridis"),
    )
    for row, (color_column, color_label, cmap) in enumerate(panels):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            # Keep only strictly positive support before applying the log axis.
            sub = data[(data.dataset.eq(dataset)) & (data.sample_count > 0)].copy()
            if sub.empty:
                ax.axis("off")
                continue
            size = 12 + 70 * np.sqrt(np.maximum(sub.test_assignment_count.to_numpy(float), 1)) / np.sqrt(max(sub.test_assignment_count.max(), 1))
            points = ax.scatter(
                sub.sample_count,
                sub.radius,
                c=sub[color_column],
                s=size,
                cmap=cmap,
                alpha=0.70,
                edgecolors="white",
                linewidths=0.25,
                rasterized=True,
            )
            top = sub.nlargest(3, color_column)
            for _, item in top.iterrows():
                ax.annotate(str(item.majority_label), (item.sample_count, item.radius), fontsize=5, xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_title(f"{DATASET_LABELS[dataset]} — {color_label}")
            ax.set_xlabel("Train support per ball (log scale)")
            ax.set_ylabel("Ball radius")
            ax.grid(alpha=0.18)
            fig.colorbar(points, ax=ax, label="Share of cell error", shrink=0.86)
    fig.suptitle("MOGB boundary risk surface: support and radius concentrate errors", y=1.02, fontsize=12)
    save_figure(fig, "mogb_ball_risk_surface")
    plt.close(fig)


def build_geometry_coupling() -> pd.DataFrame:
    geometry = pd.read_csv(GEOMETRY)
    geometry["dataset"] = geometry.dataset.replace({"banking77_oos": "banking77"})
    geometry = geometry.groupby(["dataset", "representation"], as_index=False).agg(
        relative_separation=("relative_separation", "mean"),
        effective_rank=("effective_rank", "mean"),
        purity_at_10=("purity_at_10", "mean"),
        same_intent_alignment=("same_intent_alignment", "mean"),
    )
    results = pd.read_csv(REP_RESULTS)
    results["dataset"] = results.dataset.replace({"banking77_oos": "banking77"})
    pivot = results.pivot_table(index=["dataset", "representation", "data_seed"], columns="k_variant", values=["near_oos_f1", "oos_recall"])
    pivot.columns = ["_".join(column) for column in pivot.columns]
    pivot = pivot.reset_index()
    pivot["k2_minus_k1_near_oos_f1_pp"] = (pivot["near_oos_f1_k2"] - pivot["near_oos_f1_k1"]) * 100
    # The source reports OOS recall, so false acceptance is its complement.
    pivot["k2_minus_k1_false_accept_pp"] = (pivot["oos_recall_k1"] - pivot["oos_recall_k2"]) * 100
    effects = pivot.groupby(["dataset", "representation"], as_index=False)[["k2_minus_k1_near_oos_f1_pp", "k2_minus_k1_false_accept_pp"]].mean()
    collisions = pd.read_csv(COLLISIONS)
    collisions["dataset"] = collisions.dataset.replace({"banking77_oos": "banking77"})
    collisions = collisions[(collisions.group.eq("near")) & (collisions.k_name.astype(str).eq("1"))].groupby(["dataset", "representation"], as_index=False).agg(
        near_collision_rate=("mean_representation_collision_rate", "mean"),
        near_false_accept_rate=("mean_false_accept_rate", "mean"),
        near_boundary_overcoverage_rate=("mean_boundary_overcoverage_rate", "mean"),
    )
    result = geometry.merge(effects, on=["dataset", "representation"], how="inner").merge(collisions, on=["dataset", "representation"], how="left")
    result["dataset_label"] = result.dataset.map(DATASET_LABELS)
    result["representation_label"] = result.representation.map({"frozen": "Frozen", "ce": "CE", "supcon": "SupCon"})
    return result


def plot_geometry_coupling(coupling: pd.DataFrame) -> None:
    colors = {"frozen": "#666666", "ce": "#D55E00", "supcon": "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    plots = (
        ("relative_separation", "k2_minus_k1_near_oos_f1_pp", "Relative separation", "K=2 − K=1 Near-OOS F1 (pp)"),
        ("effective_rank", "k2_minus_k1_false_accept_pp", "Effective rank", "K=2 − K=1 false acceptance (pp)"),
        ("near_collision_rate", "near_false_accept_rate", "Near-OOS representation collision rate", "Near-OOS false acceptance rate"),
        ("near_collision_rate", "k2_minus_k1_near_oos_f1_pp", "Near-OOS representation collision rate", "K=2 − K=1 Near-OOS F1 (pp)"),
    )
    for ax, (x_column, y_column, xlabel, ylabel) in zip(axes.ravel(), plots, strict=True):
        for _, row in coupling.iterrows():
            color = colors.get(row.representation, "#0072B2")
            ax.scatter(row[x_column], row[y_column], s=70, color=color, edgecolor="white", linewidth=0.5)
            ax.annotate(f"{row.dataset_label}/{row.representation_label}", (row[x_column], row[y_column]), fontsize=6, xytext=(4, 3), textcoords="offset points")
        if "k2_minus" in y_column:
            ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=label, markersize=6) for label, color in (("Frozen", colors["frozen"]), ("CE", colors["ce"]), ("SupCon", colors["supcon"]))]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Representation geometry does not determine boundary safety by itself", y=1.07, fontsize=12)
    save_figure(fig, "geometry_boundary_coupling")
    plt.close(fig)


def build_detector_composition(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[frame.method.isin(NATIVE_METHODS)].copy()
    result = (
        data.groupby(["dataset", "method", "outcome"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = data.groupby(["dataset", "method", "is_true_oos"], as_index=False).size().rename(columns={"size": "total"})
    result["is_true_oos"] = result.outcome.isin(("oos_correct", "oos_false_accept")).astype(int)
    result = result.merge(totals, on=["dataset", "method", "is_true_oos"], how="left")
    result["rate"] = result["count"] / result["total"]
    result["method_label"] = result.method.map(METHOD_LABELS)
    result["gold_type"] = np.where(result.is_true_oos.eq(1), "OOS", "Known")
    return result


def plot_detector_composition(composition: pd.DataFrame) -> None:
    known_outcomes = ("known_correct", "known_wrong_intent", "known_rejected")
    oos_outcomes = ("oos_correct", "oos_false_accept")
    outcome_colors = {"known_correct": "#4C78A8", "known_wrong_intent": "#F2CF5B", "known_rejected": "#E45756", "oos_correct": "#59A14F", "oos_false_accept": "#E15759"}
    methods = NATIVE_METHODS
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True, sharey="row")
    for col, dataset in enumerate(DATASETS):
        for row, outcomes in enumerate((known_outcomes, oos_outcomes)):
            ax = axes[row, col]
            x = np.arange(len(methods))
            bottom = np.zeros(len(methods))
            for outcome in outcomes:
                values = []
                for method in methods:
                    value = composition[(composition.dataset.eq(dataset)) & (composition.method.eq(method)) & (composition.outcome.eq(outcome))]["rate"]
                    values.append(float(value.iloc[0]) if not value.empty else 0.0)
                values = np.asarray(values)
                ax.bar(x, values * 100, bottom=bottom * 100, color=outcome_colors[outcome], label=OUTCOME_LABELS[outcome], width=0.72)
                bottom += values
            ax.set_xticks(x, [METHOD_LABELS[m] for m in methods], rotation=35, ha="right")
            ax.set_ylim(0, 100)
            ax.set_title(f"{DATASET_LABELS[dataset]} — {'Known' if row == 0 else 'OOS'}")
            ax.set_ylabel("Share (%)")
            ax.grid(axis="y", alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Same Trainable MiniLM representation, different detector failure modes", y=1.07, fontsize=12)
    save_figure(fig, "native_detector_error_fingerprint")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    configure_plot()
    frame = load_contract()
    pair_summary = plot_paired_scores(frame)
    margin = build_margin_rates(frame)
    plot_margin_rates(margin)
    intent = build_intent_localization(frame)
    plot_intent_localization(intent)
    balls = load_ball_risk()
    plot_ball_risk(balls)
    coupling = build_geometry_coupling()
    plot_geometry_coupling(coupling)
    composition = build_detector_composition(frame)
    plot_detector_composition(composition)
    atomic_csv(pair_summary, OUT / "paired_score_reordering_summary.csv")
    atomic_csv(margin, OUT / "boundary_margin_rates.csv")
    atomic_csv(intent, OUT / "intent_error_localization.csv")
    atomic_csv(balls, OUT / "mogb_ball_risk_surface.csv")
    atomic_csv(coupling, OUT / "geometry_boundary_coupling.csv")
    atomic_csv(composition, OUT / "native_detector_error_composition.csv")
    source_paths = [CONTRACT, BALLS, GEOMETRY, REP_RESULTS, COLLISIONS]
    manifest = {
        "analysis": "mechanism_explanation_pack_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "analysis_only": True,
        "kir": 0.50,
        "datasets": list(DATASETS),
        "test_labels_used_post_hoc": True,
        "selection_used_test_oos": False,
        "sample_ids_exported": False,
        "figure_export_formats": ["png_preview_600dpi", "svg", "pdf"],
        "figure_count": 6,
        "source_hashes": {str(path.relative_to(ROOT.parent)): sha256(path) for path in source_paths},
        "input_rows_after_filter": int(len(frame)),
        "methods_after_filter": sorted(frame.method.unique()),
        "missing_oos_score_rows": int(frame.attrs.get("missing_score_rows", 0)),
        "methods_with_missing_oos_score": frame.attrs.get("missing_score_methods", []),
        "score_figure_methods": sorted(set(FAIR_METHODS) | set(NATIVE_METHODS)),
        "outputs": {
            "paired_score_reordering_summary": str((OUT / "paired_score_reordering_summary.csv").relative_to(ROOT)),
            "boundary_margin_rates": str((OUT / "boundary_margin_rates.csv").relative_to(ROOT)),
            "intent_error_localization": str((OUT / "intent_error_localization.csv").relative_to(ROOT)),
            "mogb_ball_risk_surface": str((OUT / "mogb_ball_risk_surface.csv").relative_to(ROOT)),
            "geometry_boundary_coupling": str((OUT / "geometry_boundary_coupling.csv").relative_to(ROOT)),
            "native_detector_error_composition": str((OUT / "native_detector_error_composition.csv").relative_to(ROOT)),
        },
        "figures": [str(path.relative_to(ROOT)) for path in sorted(FIG.glob("*.png"))],
    }
    atomic_json(manifest, OUT / "MANIFEST.json")
    print(json.dumps({"status": "ok", "rows": len(frame), "figures": len(manifest["figures"]), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
