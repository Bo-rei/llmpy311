#!/usr/bin/env python3
"""Quantify KIR sensitivity from the completed five-seed fair matrix.

This is an analysis-only stage.  It does not run a detector, select a KIR,
or alter any existing run.  The input is the already aggregated current
protocol fair summary, so every point is tied to the existing five seeds.
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
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
OUT = ROOT / "results/analysis/archive/analysis/kir_sensitivity_decomposition_v1"
FIG = ROOT / "figures/archive/analysis/kir_sensitivity_decomposition_v1"
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
METRICS = ("oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate")
METRIC_LABELS = {
    "oos_f1": "OOS F1",
    "f1_all": "F1-All",
    "f1_k": "F1-K",
    "known_recall": "Known Recall",
    "false_accept_rate": "False acceptance",
    "false_reject_rate": "False rejection",
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


def load() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE)
    required = {"dataset", "kir", "method", *METRICS, "n_seeds"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    frame = frame[frame.dataset.isin(DATASETS) & frame.kir.isin(KIRS) & frame.method.isin(METHODS)].copy()
    expected = len(DATASETS) * len(KIRS) * len(METHODS)
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, got {len(frame)}")
    if frame.duplicated(["dataset", "kir", "method"]).any():
        raise ValueError("duplicate dataset/kir/method rows")
    if frame.n_seeds.nunique() != 1 or int(frame.n_seeds.iloc[0]) != 5:
        raise ValueError("source summary is not a five-seed fair summary")
    return frame


def build_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (dataset, method), group in frame.groupby(["dataset", "method"], sort=False):
        group = group.sort_values("kir")
        x = group.kir.to_numpy(dtype=float)
        if tuple(x) != KIRS:
            raise ValueError(f"incomplete KIR values for {dataset}/{method}: {tuple(x)}")
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            slope = float(pd.Series(values).cov(pd.Series(x)) / pd.Series(x).var())
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n_seeds": int(group.n_seeds.iloc[0]),
                    "kir_025": float(values[0]),
                    "kir_050": float(values[1]),
                    "kir_075": float(values[2]),
                    "endpoint_delta": float(values[2] - values[0]),
                    "endpoint_delta_pp": float((values[2] - values[0]) * 100.0),
                    "slope_per_kir": slope,
                    "slope_pp_per_kir": slope * 100.0,
                    "worst_value": float(values.min()),
                    "best_value": float(values.max()),
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (dataset, method), group in detail.groupby(["dataset", "method"], sort=False):
        item: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_seeds": int(group.n_seeds.iloc[0]),
        }
        for metric in METRICS:
            row = group[group.metric == metric].iloc[0]
            item[f"{metric}_delta_pp"] = float(row.endpoint_delta_pp)
            item[f"{metric}_slope_pp_per_kir"] = float(row.slope_pp_per_kir)
        summary_rows.append(item)
    return detail, pd.DataFrame(summary_rows)


def plot_oos_lines(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = frame[frame.dataset == dataset]
        for method in METHODS:
            row = sub[sub.method == method].sort_values("kir")
            ax.plot(row.kir, row.oos_f1 * 100, marker="o", linewidth=1.8, label=METHOD_LABELS[method])
        ax.set_title(dataset)
        ax.set_xlabel("Known Intent Ratio")
        ax.set_xticks(KIRS)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("KIR sensitivity of OOS F1 in the current five-seed fair matrix", y=1.02)
    fig.savefig(FIG / "oos_f1_kir_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_slope_heatmap(summary: pd.DataFrame) -> None:
    rows = []
    for _, row in summary.iterrows():
        for metric in METRICS:
            rows.append(
                {
                    "dataset_method": f"{row.dataset.upper()} / {row.method_label}",
                    "metric": METRIC_LABELS[metric],
                    "delta_pp": row[f"{metric}_delta_pp"],
                }
            )
    table = pd.DataFrame(rows).pivot(index="dataset_method", columns="metric", values="delta_pp")
    table = table.reindex(columns=[METRIC_LABELS[m] for m in METRICS])
    fig, ax = plt.subplots(figsize=(13.5, 12), constrained_layout=True)
    sns.heatmap(table, annot=True, fmt="+.1f", center=0, cmap="RdBu_r", linewidths=0.5, linecolor="white", cbar_kws={"label": "KIR=.75 minus KIR=.25 (pp)"}, ax=ax)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Dataset / method")
    ax.set_title("Endpoint sensitivity to increasing KIR")
    fig.savefig(FIG / "kir_endpoint_delta_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_trajectory(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = frame[frame.dataset == dataset]
        for method in METHODS:
            row = sub[sub.method == method].sort_values("kir")
            ax.plot(row.known_recall * 100, row.oos_f1 * 100, marker="o", linewidth=1.5, label=METHOD_LABELS[method])
            for _, point in row.iterrows():
                ax.annotate(f"{point.kir:.2f}", (point.known_recall * 100, point.oos_f1 * 100), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.set_title(dataset)
        ax.set_xlabel("Known Recall (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("KIR trajectory in the coverage–open-space plane", y=1.02)
    fig.savefig(FIG / "coverage_oos_trajectory_by_kir.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame = load()
    detail, summary = build_tables(frame)
    atomic_csv(detail, OUT / "kir_sensitivity_detail.csv")
    atomic_csv(summary, OUT / "kir_sensitivity_summary.csv")
    plot_oos_lines(frame)
    plot_slope_heatmap(summary)
    plot_coverage_trajectory(frame)
    atomic_json(
        {
            "analysis_id": "kir_sensitivity_decomposition_v1",
            "protocol_version": "protocol_v2_textoir_v1",
            "source": str(SOURCE),
            "source_sha256": sha256(SOURCE),
            "source_contract": "experimental_mechanism_pack_v3 five-seed summary; no new model run",
            "datasets": list(DATASETS),
            "kirs": list(KIRS),
            "methods": list(METHODS),
            "metrics": list(METRICS),
            "detail_rows": int(len(detail)),
            "summary_rows": int(len(summary)),
            "outputs": [
                "kir_sensitivity_detail.csv",
                "kir_sensitivity_summary.csv",
                "oos_f1_kir_sensitivity.png",
                "kir_endpoint_delta_heatmap.png",
                "coverage_oos_trajectory_by_kir.png",
            ],
        },
        OUT / "MANIFEST.json",
    )


if __name__ == "__main__":
    main()
