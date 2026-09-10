#!/usr/bin/env python3
"""Build readable full-pipeline figures from completed H1 Cascade artifacts.

The primary comparison is the current H1 Trainable K=1 Gate against the cached
H1 Frozen K=1 Gate with the same v19 Router/Expert components.  Existing
selected-K and CE-Recon Cascade rows remain labeled reference variants.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
        "axes.linewidth": 0.75,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DARK = "#252525"
CASCADE_ROOT = ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cascade_full" / "gpu_kir50"
CURRENT_ROOT = ROOT / "results" / "analysis" / "historical_trainable_full_pipeline"
OUT = ROOT / "results" / "analysis" / "full_pipeline_visual_explanation"
FIG = ROOT / "figures" / "full_pipeline_visual_explanation"

DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
DATASET_LABELS = {"clinc150": "CLINC150", "stackoverflow": "StackOverflow", "banking77_oos": "BANKING77-OOS"}
SEEDS = (13, 42, 87)
METHODS = ("frozen_k1", "trainable_k1_current_h1", "frozen_selected_k", "ce_recon_selected_k", "best_controlled_baseline")
METHOD_LABELS = {
    "frozen_k1": "Frozen K=1",
    "trainable_k1_current_h1": "Trainable H1 K=1",
    "frozen_selected_k": "Frozen selected-K",
    "ce_recon_selected_k": "Trainable CE-Recon selected-K",
    "best_controlled_baseline": "Best controlled",
}
METHOD_COLORS = {
    "frozen_k1": "#4C78A8",
    "trainable_k1_current_h1": "#C44E52",
    "frozen_selected_k": "#8B929A",
    "ce_recon_selected_k": "#E39C35",
    "best_controlled_baseline": "#6B6F73",
}
METHOD_MARKERS = {
    "frozen_k1": "o",
    "frozen_selected_k": "D",
    "ce_recon_selected_k": "^",
    "best_controlled_baseline": "s",
}
STAGES = (
    "correct_oos_rejection",
    "oos_accepted_by_gate",
    "known_rejected_by_gate",
    "known_wrong_domain",
    "known_wrong_expert",
    "correct_known_prediction",
)
STAGE_LABELS = {
    "correct_oos_rejection": "OOS correct",
    "oos_accepted_by_gate": "OOS accept",
    "known_rejected_by_gate": "Known reject",
    "known_wrong_domain": "Router",
    "known_wrong_expert": "Expert",
    "correct_known_prediction": "Known correct",
}
STAGE_COLORS = {
    "correct_oos_rejection": "#C44E52",
    "oos_accepted_by_gate": "#F1B6B8",
    "known_rejected_by_gate": "#F2C14E",
    "known_wrong_domain": "#B7A5D3",
    "known_wrong_expert": "#8B929A",
    "correct_known_prediction": "#4C78A8",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty CSV: {path}")
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.tick_params(length=2.8, width=0.65, pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E7E9EC", linewidth=0.45)
        ax.set_axisbelow(True)


def _counts_from_eval(metrics: Mapping[str, Any]) -> tuple[int, int]:
    counts = metrics.get("counts", {})
    return int(counts.get("known_id", 0)), int(counts.get("oos", 0))


def _gate_f1(known_count: int, oos_count: int, id_recall: float, oos_rejection: float) -> float:
    true_positive = float(oos_count) * float(oos_rejection)
    false_positive = float(known_count) * (1.0 - float(id_recall))
    precision = true_positive / max(true_positive + false_positive, 1e-12)
    return 2.0 * precision * float(oos_rejection) / max(precision + float(oos_rejection), 1e-12)


def load_full_pipeline_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cached = read_csv(CASCADE_ROOT / "cascade_summary.csv")
    cached_rows: list[dict[str, Any]] = []
    for item in cached.to_dict(orient="records"):
        eval_path = Path(str(item["result_path"]))
        payload = read_json(eval_path)
        metrics = payload["metrics"]
        known_count, oos_count = _counts_from_eval(metrics)
        gate = metrics["fast_gate_metrics"]
        cached_rows.append(
            {
                "dataset": str(item["dataset"]),
                "seed": int(str(item["kir_seed"]).split("seed")[-1]),
                "method": str(item["gate"]),
                "method_label": METHOD_LABELS[str(item["gate"])],
                "representation": "frozen MiniLM" if str(item["gate"]).startswith("frozen") else ("CE-Recon MiniLM" if str(item["gate"]) == "ce_recon_selected_k" else "linear baseline"),
                "oos_f1": float(item["oos_f1"]),
                "full_macro_f1": float(metrics["macro_f1"]),
                "known_macro_f1": float(item["known_macro_f1"]),
                "overall_accuracy": float(item["overall_accuracy"]),
                "known_recall": float(item["id_recall"]),
                "false_accept_rate": float(item["oos_false_accept_rate"]),
                "false_reject_rate": float(item["known_false_reject_rate"]),
                "router_error_rate": float(item["router_error_rate"]),
                "expert_error_rate": float(item["expert_error_rate"]),
                "gate_oos_f1": _gate_f1(known_count, oos_count, float(gate["gate_id_recall"]), float(gate["gate_oos_rejection"])),
                "source": "cached_cascade_full",
                "source_path": str(eval_path),
            }
        )
    current = read_csv(CURRENT_ROOT / "per_seed.csv")
    current_rows: list[dict[str, Any]] = []
    for item in current.to_dict(orient="records"):
        current_rows.append(
            {
                "dataset": str(item["dataset"]),
                "seed": int(item["seed"]),
                "method": "trainable_k1_current_h1",
                "method_label": METHOD_LABELS["trainable_k1_current_h1"],
                "representation": "last2 MiniLM + projection",
                "oos_f1": float(item["oos_f1"]),
                "full_macro_f1": float(item["f1_all"]),
                "known_macro_f1": float(item["known_macro_f1"]),
                "overall_accuracy": float(item["overall_accuracy"]),
                "known_recall": float(item["known_recall"]),
                "false_accept_rate": float(item["false_accept_rate"]),
                "false_reject_rate": float(item["false_reject_rate"]),
                "router_error_rate": float(item["router_error_rate"]),
                "expert_error_rate": float(item["expert_error_rate"]),
                "gate_oos_f1": _gate_f1(int(item["known_count"]), int(item["oos_count"]), float(item["gate_id_recall"]), float(item["gate_oos_rejection"])),
                "source": "current_h1_trainable_full_pipeline",
                "source_path": str(CURRENT_ROOT / "per_seed.csv"),
            }
        )
    performance = pd.DataFrame([*cached_rows, *current_rows])
    performance.to_csv(OUT / "full_pipeline_performance.csv", index=False)

    primary = performance[performance["method"].isin(("frozen_k1", "trainable_k1_current_h1"))]
    pivot = primary.pivot(index=["dataset", "seed"], columns="method")
    pair_rows: list[dict[str, Any]] = []
    for (dataset, seed), group in primary.groupby(["dataset", "seed"]):
        frozen = group[group["method"].eq("frozen_k1")].iloc[0]
        trainable = group[group["method"].eq("trainable_k1_current_h1")].iloc[0]
        row: dict[str, Any] = {"dataset": dataset, "seed": int(seed)}
        for metric in ("oos_f1", "full_macro_f1", "overall_accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "router_error_rate", "expert_error_rate"):
            row[f"delta_{metric}"] = float(trainable[metric] - frozen[metric])
        pair_rows.append(row)
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(OUT / "full_pipeline_paired_effects.csv", index=False)

    cached_error = read_csv(CASCADE_ROOT / "cascade_error_decomposition.csv")
    cached_error = cached_error[cached_error["gate"].eq("frozen_k1")].copy()
    cached_error["method"] = "frozen_k1"
    cached_error["method_label"] = METHOD_LABELS["frozen_k1"]
    current_error = read_csv(CURRENT_ROOT / "error_paths.csv")
    current_error["method"] = "trainable_k1_current_h1"
    current_error["method_label"] = METHOD_LABELS["trainable_k1_current_h1"]
    errors = pd.concat(
        [
            cached_error[["dataset", "stage", "count", "rate", "method", "method_label"]].assign(seed=cached_error["kir_seed"].str.extract(r"(\d+)$", expand=False).astype(int)),
            current_error[["dataset", "seed", "stage", "count", "rate", "method", "method_label"]],
        ],
        ignore_index=True,
    )
    errors.to_csv(OUT / "full_pipeline_error_paths.csv", index=False)
    return performance, pairwise, errors


def plot_workpoint(performance: pd.DataFrame) -> None:
    primary = performance[performance["method"].isin(("frozen_k1", "trainable_k1_current_h1"))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.70, wspace=0.22)
    x = np.arange(len(DATASETS))
    for ax, metric, ylabel in zip(axes, ("oos_f1", "full_macro_f1"), ("OOS F1", "full-pipeline macro F1"), strict=True):
        for method in ("frozen_k1", "trainable_k1_current_h1"):
            mean = primary[primary["method"].eq(method)].groupby("dataset")[metric].mean().reindex(DATASETS)
            std = primary[primary["method"].eq(method)].groupby("dataset")[metric].std().reindex(DATASETS).fillna(0)
            offset = -0.10 if method == "frozen_k1" else 0.10
            ax.errorbar(x + offset, mean.to_numpy() * 100, yerr=std.to_numpy() * 100, fmt="o-", color=METHOD_COLORS[method], linewidth=1.2, markersize=4, capsize=2, label=METHOD_LABELS[method])
        ax.set_xticks(x, [DATASET_LABELS[d] for d in DATASETS])
        ax.set_ylabel(f"{ylabel} (%)")
        ax.set_ylim(55 if metric == "full_macro_f1" else 72, 96 if metric == "oos_f1" else 90)
        ax.set_title(ylabel, loc="left", fontweight="bold")
        style_axis(ax, "y")
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="upper right", frameon=False, handlelength=1.4)
    fig.text(0.08, 0.965, "Current H1 Trainable K=1 in the full pipeline", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.04, "Three seeds; same v19 Router/Expert components. OOS F1 is the Gate decision; macro F1 includes downstream intent prediction.", fontsize=5.9, color="#555B61")
    save_figure(fig, "full_pipeline_workpoint")


def plot_error_paths(errors: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.95), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.23, top=0.68, wspace=0.18)
    error_stages = (
        "oos_accepted_by_gate",
        "known_rejected_by_gate",
        "known_wrong_domain",
        "known_wrong_expert",
    )
    for column, dataset in enumerate(DATASETS):
        ax = axes[column]
        part = errors[errors["dataset"].eq(dataset)]
        y = np.arange(2)
        left = np.zeros(2)
        for stage in error_stages:
            values = []
            for method in ("frozen_k1", "trainable_k1_current_h1"):
                values.append(float(part[part["method"].eq(method) & part["stage"].eq(stage)]["rate"].mean()) * 100)
            values_array = np.asarray(values)
            ax.barh(y, values_array, left=left, height=0.58, color=STAGE_COLORS[stage], label=STAGE_LABELS[stage])
            for position, value in enumerate(values_array):
                if value >= 5.0:
                    ax.text(left[position] + value / 2, position, STAGE_LABELS[stage], ha="center", va="center", fontsize=5.2, color="white" if stage in {"oos_accepted_by_gate", "known_wrong_expert"} else DARK)
            left += values_array
        for position, value in enumerate(left):
            ax.text(value + 0.45, position, f"correct {100.0 - value:.0f}%", ha="left", va="center", fontsize=5.7, color="#555B61")
        ax.set_yticks(y, ["Frozen K=1", "Trainable H1 K=1"])
        ax.set_xlim(0, 30)
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        if column == 0:
            ax.set_ylabel("pipeline path")
        ax.set_xlabel("share of all test samples (%)")
        style_axis(ax, "x")
        ax.grid(axis="y", visible=False)
    fig.text(0.08, 0.965, "Where does the full pipeline lose samples?", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "Direct labels show Gate losses first (OOS accepted, Known rejected), then Router and Expert losses; the endpoint is the correct share.", fontsize=5.9, color="#555B61")
    save_figure(fig, "full_pipeline_error_paths")


def plot_gate_to_pipeline(performance: pd.DataFrame) -> None:
    primary_methods = ("frozen_k1", "trainable_k1_current_h1")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85))
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.21, top=0.70, wspace=0.22)
    for column, dataset in enumerate(DATASETS):
        ax = axes[column]
        part = performance[performance["dataset"].eq(dataset) & performance["method"].isin(primary_methods)]
        for method in primary_methods:
            rows = part[part["method"].eq(method)]
            ax.scatter(rows["gate_oos_f1"] * 100, rows["full_macro_f1"] * 100, s=34, color=METHOD_COLORS[method], edgecolor="white", linewidth=0.45, label=METHOD_LABELS[method], zorder=3)
        limits = [float(part["gate_oos_f1"].min()), float(part["gate_oos_f1"].max()), float(part["full_macro_f1"].min()), float(part["full_macro_f1"].max())]
        ax.set_xlim(max(0, limits[0] * 100 - 2), min(100, limits[1] * 100 + 2))
        ax.set_ylim(max(45, limits[2] * 100 - 5), min(100, limits[3] * 100 + 5))
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("Gate OOS F1 (%)")
        if column == 0:
            ax.set_ylabel("full-pipeline macro F1 (%)")
        style_axis(ax, "both")
    handles = [Line2D([], [], marker="o", linestyle="none", color=METHOD_COLORS[m], markersize=5, label=METHOD_LABELS[m]) for m in primary_methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.58, 0.86), ncol=2, frameon=False, handlelength=1.0, columnspacing=0.8)
    fig.text(0.08, 0.965, "Gate quality versus the final pipeline", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.04, "Only Frozen K=1 and current H1 Trainable K=1 are shown; x is Gate OOS F1 and y includes downstream intent prediction.", fontsize=5.9, color="#555B61")
    save_figure(fig, "full_pipeline_gate_to_pipeline")


def plot_gate_variants(performance: pd.DataFrame) -> None:
    variants = ("frozen_k1", "frozen_selected_k", "ce_recon_selected_k", "best_controlled_baseline")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85))
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.21, top=0.70, wspace=0.22)
    for column, dataset in enumerate(DATASETS):
        ax = axes[column]
        part = performance[performance["dataset"].eq(dataset) & performance["method"].isin(variants)]
        for method in variants:
            rows = part[part["method"].eq(method)]
            ax.scatter(rows["false_accept_rate"] * 100, rows["full_macro_f1"] * 100, s=34, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], edgecolor="white", linewidth=0.45, label=METHOD_LABELS[method], zorder=3)
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("OOS false acceptance (%)")
        if column == 0:
            ax.set_ylabel("full-pipeline macro F1 (%)")
        ax.set_xlim(left=0)
        ax.set_ylim(45, 90)
        style_axis(ax, "both")
    handles = [Line2D([], [], marker=METHOD_MARKERS[m], linestyle="none", color=METHOD_COLORS[m], markersize=5, label=METHOD_LABELS[m]) for m in variants]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.86), ncol=2, frameon=False, handlelength=1.0, columnspacing=0.8)
    fig.text(0.08, 0.965, "Full-pipeline Gate variants", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.04, "Marker shape separates the four cached Gate variants; lower x is better OOS rejection, higher y is better final macro F1.", fontsize=5.9, color="#555B61")
    save_figure(fig, "full_pipeline_gate_variants")


def write_manifest(performance: pd.DataFrame) -> None:
    summary = {
        "schema_version": "s2c.full_pipeline_visual_explanation.v1",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert",
        "primary_comparison": "cached Frozen K=1 vs current H1 Trainable K=1 with the same cascade_full downstream components",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "cached_full_pipeline_units": 36,
        "current_trainable_full_pipeline_units": int((performance["method"] == "trainable_k1_current_h1").sum()),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "raw_predictions_written": False,
        "representation_boundary": "current Trainable uses last2_minilm_plus_projection; cached CE-Recon is a separate reference variant",
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "MANIFEST.json")
    manifest = {
        "schema_version": "s2c.analysis_bundle.v1",
        "bundle": "full_pipeline_visual_explanation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_layer": "historical_controlled_gate_to_router_to_expert_H1",
        "question": "How does the Trainable H1 Gate change the final Gate-to-Router-to-Expert outcome?",
        "source_experiment": "historical_protocol_reconciliation_20260826",
        "source_cached_cascade": str(CASCADE_ROOT),
        "source_current_trainable": str(CURRENT_ROOT),
        "builder": str(Path(__file__).relative_to(ROOT)),
        "builder_sha256": sha256_file(Path(__file__)),
        "report": "docs/analysis/RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md",
        "result_files": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for path in files],
        "figure_files": [str((FIG / f"{stem}.{suffix}").relative_to(ROOT)) for stem in ("full_pipeline_workpoint", "full_pipeline_error_paths", "full_pipeline_gate_to_pipeline", "full_pipeline_gate_variants") for suffix in ("png", "tiff", "pdf", "svg")],
        "guardrails": {"new_training": False, "threshold_changed": False, "test_used_for_selection": False, "oos_used_for_training": False, "raw_predictions_written": False},
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    performance, pairwise, errors = load_full_pipeline_rows()
    plot_workpoint(performance)
    plot_error_paths(errors)
    plot_gate_to_pipeline(performance)
    plot_gate_variants(performance)
    write_manifest(performance)
    print(json.dumps({"bundle": "full_pipeline_visual_explanation", "performance_rows": len(performance), "pair_rows": len(pairwise), "error_rows": len(errors), "figures": 4}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
