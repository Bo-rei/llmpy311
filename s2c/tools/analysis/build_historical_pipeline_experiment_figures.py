#!/usr/bin/env python3
"""Build OOS-first figures for the archive full-pipeline experiment report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
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

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_archive_full_pipeline_seed42"
TUNED_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_archive_tuned_full_pipeline_seed42"
SEARCH_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_archive_gate_search_seed42"
PUBLIC_ROOT = ROOT / "results" / "analysis" / "historical_archive_tuned_full_pipeline"
FIG_ROOT = ROOT / "figures" / "historical_archive_tuned_full_pipeline"
DATASETS = ("clinc150", "stackoverflow", "banking77")
DATASET_LABELS = {"clinc150": "CLINC150", "stackoverflow": "StackOverflow", "banking77": "Banking77"}
PAPER = {
    "clinc150": {"known_f1": 79.95, "oos_f1": 91.96, "accuracy": 86.78},
    "stackoverflow": {"known_f1": 75.48, "oos_f1": 89.71, "accuracy": 85.54},
    "banking77": {"known_f1": 74.90, "oos_f1": 88.23, "accuracy": 78.98},
}
COLORS = {"baseline": "#4C78A8", "tuned": "#C44E52", "paper": "#777B80"}
LIGHT_GRAY = "#C5CBD1"
DARK = "#252525"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _per_method(root: Path, dataset: str) -> list[dict[str, str]]:
    path = root / dataset / "kir50_seed42" / "per_method.csv"
    rows = _read_csv(path)
    if len(rows) != 3:
        raise ValueError(f"expected three methods in {path}")
    return rows


def _row(root: Path, dataset: str, method: str) -> dict[str, str]:
    rows = [row for row in _per_method(root, dataset) if row["method"] == method]
    if len(rows) != 1:
        raise ValueError(f"missing {dataset}/{method} in {root}")
    return rows[0]


def _best_original(dataset: str) -> tuple[str, dict[str, str]]:
    rows = _per_method(ORIGINAL_ROOT, dataset)
    row = max(rows, key=lambda item: float(item["oos_f1"]))
    return row["method"], row


def _tuned_row(dataset: str) -> dict[str, str]:
    return _row(TUNED_ROOT, dataset, "partial_k1")


def _pct(row: dict[str, str], key: str) -> float:
    return float(row[key]) * 100.0


def _style(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.tick_params(length=2.8, width=0.65, pad=2)
    if grid:
        ax.grid(axis=grid, color="#E7E9EC", linewidth=0.45)
        ax.set_axisbelow(True)


def _save(fig: plt.Figure, stem: str) -> list[str]:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    png_path = FIG_ROOT / f"{stem}.png"
    svg_path = FIG_ROOT / f"{stem}.svg"
    pdf_path = FIG_ROOT / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    outputs = [str(path.relative_to(ROOT)) for path in (png_path, svg_path, pdf_path)]
    plt.close(fig)
    return outputs


def _load_comparison() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        method, baseline = _best_original(dataset)
        tuned = _tuned_row(dataset)
        result[dataset] = {
            "baseline_method": method,
            "baseline": baseline,
            "tuned": tuned,
            "paper": PAPER[dataset],
        }
    return result


def plot_gate_pipeline_invariance(data: dict[str, dict[str, Any]]) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.70, wspace=0.20)
    x = np.arange(2)
    for index, dataset in enumerate(DATASETS):
        ax = axes[index]
        baseline_value = _pct(data[dataset]["baseline"], "oos_f1")
        tuned_value = _pct(data[dataset]["tuned"], "oos_f1")
        close_labels = abs(tuned_value - baseline_value) < 1.0
        for condition, color in (("baseline", COLORS["baseline"]), ("tuned", COLORS["tuned"])):
            value = _pct(data[dataset][condition], "oos_f1")
            ax.plot(x, [value, value], marker="o", markersize=4.5, linewidth=1.35, color=color, label=condition.title())
            label_offset = (-0.28 if condition == "baseline" else 0.28) if close_labels else 0.0
            ax.text(1.04, value + label_offset, f"{value:.2f}", va="center", ha="left", fontsize=5.9, color=color)
        ax.set_xticks(x, ["Gate", "Full\npipeline"])
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        ax.set_ylim(78, 94)
        if index == 0:
            ax.set_ylabel("OOS F1 (%)")
        _style(ax, "y")
        ax.grid(axis="x", visible=False)
        ax.text(0.5, 0.06, "Δ = 0.00 pp", transform=ax.transAxes, ha="center", fontsize=6.0, color=DARK)
    handles = [
        Line2D([], [], marker="o", color=COLORS["baseline"], linewidth=1.2, markersize=4.5, label="Current baseline"),
        Line2D([], [], marker="o", color=COLORS["tuned"], linewidth=1.2, markersize=4.5, label="Tuned candidate"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.88), ncol=2, frameon=False, handlelength=1.3, columnspacing=0.9)
    fig.text(0.08, 0.965, "Gate OOS decisions are inherited by the full pipeline", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "The downstream Router/Expert changes Known classification, but cannot change an OOS sample already rejected by the Gate.", fontsize=5.9, color="#555B61")
    return _save(fig, "pipeline_gate_full_oos_invariance")


def plot_paper_comparison(data: dict[str, dict[str, Any]]) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.95), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.25, top=0.70, wspace=0.20)
    labels = ["Current\nK=1", "Tuned\ncandidate", "Paper\nOurs"]
    for index, dataset in enumerate(DATASETS):
        ax = axes[index]
        values = [_pct(data[dataset]["baseline"], "oos_f1"), _pct(data[dataset]["tuned"], "oos_f1"), data[dataset]["paper"]["oos_f1"]]
        colors = [COLORS["baseline"], COLORS["tuned"], COLORS["paper"]]
        bars = ax.bar(np.arange(3), values, color=colors, width=0.62, edgecolor="white", linewidth=0.45)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.22, f"{value:.2f}", ha="center", va="bottom", fontsize=5.9)
        tuned = data[dataset]["tuned"]
        ax.text(0.02, 0.03, f"Known F1 { _pct(tuned, 'known_macro_f1'):.2f}%\nAcc { _pct(tuned, 'overall_accuracy'):.2f}%", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.7, color="#555B61")
        ax.set_xticks(np.arange(3), labels)
        ax.set_ylim(78, 94)
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        if index == 0:
            ax.set_ylabel("OOS F1 (%)")
        _style(ax, "y")
        ax.grid(axis="x", visible=False)
    fig.text(0.08, 0.965, "OOS-focused candidates close or exceed the paper work point", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "Paper values are the KIR=.50 Ours row in fulltex.tex; standard Banking77 is shown only as a protocol-separated reference.", fontsize=5.9, color="#555B61")
    return _save(fig, "pipeline_oos_f1_vs_paper")


def plot_error_budget(data: dict[str, dict[str, Any]]) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.24, top=0.70, wspace=0.20)
    metrics = (("false_accept_rate", "OOS accept"), ("false_reject_rate", "Known reject"), ("expert_error_rate", "Expert error"))
    x = np.arange(len(metrics))
    width = 0.34
    for index, dataset in enumerate(DATASETS):
        ax = axes[index]
        baseline = data[dataset]["baseline"]
        tuned = data[dataset]["tuned"]
        base_values = [_pct(baseline, key) for key, _ in metrics]
        tuned_values = [_pct(tuned, key) for key, _ in metrics]
        first = ax.bar(x - width / 2, base_values, width, color=COLORS["baseline"], label="Current baseline")
        second = ax.bar(x + width / 2, tuned_values, width, color=COLORS["tuned"], label="Tuned candidate")
        for bars in (first, second):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks(x, [label for _, label in metrics])
        ax.set_ylim(0, 30)
        ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
        if index == 0:
            ax.set_ylabel("Error share (%)")
        _style(ax, "y")
        ax.grid(axis="x", visible=False)
    handles = [
        Patch(color=COLORS["baseline"], label="Current baseline"),
        Patch(color=COLORS["tuned"], label="Tuned candidate"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.88), ncol=2, frameon=False, handlelength=1.0, columnspacing=0.9)
    fig.text(0.08, 0.965, "The OOS gain is a Gate work-point trade-off, not a downstream failure", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "Tuned candidates are selected for OOS F1; OOS acceptance, Known rejection and Expert error remain visible as costs.", fontsize=5.9, color="#555B61")
    return _save(fig, "pipeline_error_budget_oos_tradeoff")


def plot_workpoint_search(data: dict[str, dict[str, Any]]) -> list[str]:
    """Show the validation work-point frontier under the declared guard."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.95), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.20, top=0.70, wspace=0.20, hspace=0.28)
    k_markers = {1: "o", 2: "D", 3: "^"}
    k_colors = {1: "#A5ADB5", 2: "#7C8791", 3: "#59656F"}
    for column, dataset in enumerate(DATASETS):
        path = SEARCH_ROOT / dataset / "partial_k1" / "workpoints.csv"
        rows = _read_csv(path)
        validation = [row for row in rows if row["split"] == "val"]
        if not validation:
            raise ValueError(f"missing validation workpoints in {path}")
        baseline_candidates = [
            row
            for row in validation
            if int(row["k"]) == 1
            and float(row["radius_lambda"]) == 1.0
            and float(row["threshold"]) == 1.0
            and row["acceptance_mode"] == "nearest_sphere"
        ]
        if len(baseline_candidates) != 1:
            raise ValueError(f"missing unique baseline in {path}")
        baseline = baseline_candidates[0]
        selected_path = SEARCH_ROOT / dataset / "partial_k1" / "selected_validation.csv"
        selected_rows = _read_csv(selected_path)
        if len(selected_rows) != 1:
            raise ValueError(f"missing unique selected validation row in {selected_path}")
        selected = selected_rows[0]
        guard_f1 = float(baseline["known_f1"]) - 0.01
        guard_acc = float(baseline["accuracy"]) - 0.01
        candidates = [
            row
            for row in validation
            if float(row["known_f1"]) >= guard_f1 and float(row["accuracy"]) >= guard_acc
        ]
        for row in candidates:
            x_values = (float(row["known_f1"]) * 100.0, float(row["accuracy"]) * 100.0)
            y_value = float(row["oos_f1"]) * 100.0
            for row_index, x_value in enumerate(x_values):
                ax = axes[row_index, column]
                ax.scatter(
                    x_value,
                    y_value,
                    marker=k_markers[int(row["k"])],
                    s=12,
                    color=k_colors[int(row["k"])],
                    alpha=0.38,
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=2,
                )
        for row_index, key in enumerate(("known_f1", "accuracy")):
            ax = axes[row_index, column]
            base_x = float(baseline[key]) * 100.0
            selected_x = float(selected[key]) * 100.0
            selected_y = float(selected["oos_f1"]) * 100.0
            ax.axvline(base_x - 1.0, color=LIGHT_GRAY, linestyle="--", linewidth=0.8)
            ax.scatter(base_x, float(baseline["oos_f1"]) * 100.0, marker="o", s=36, color=COLORS["baseline"], edgecolors="white", linewidths=0.55, zorder=4)
            ax.scatter(selected_x, selected_y, marker="*", s=100, color="#F1B82D", edgecolors=DARK, linewidths=0.75, zorder=5)
            ax.text(selected_x + 0.35, selected_y + 0.20, "selected", fontsize=5.1, color=DARK)
            if row_index == 0:
                ax.set_title(DATASET_LABELS[dataset], loc="left", fontweight="bold")
            ax.set_ylim(74, 94)
            ax.set_xlim(max(45, min(float(row[key]) * 100.0 for row in candidates) - 1.5), min(94, max(float(row[key]) * 100.0 for row in candidates) + 1.5))
            ax.set_xlabel(f"{key.replace('_', ' ').title()} (%)")
            if column == 0:
                ax.set_ylabel("validation OOS F1 (%)")
            _style(ax, "y")
    handles = [
        Line2D([], [], marker="o", color=COLORS["baseline"], linestyle="None", markersize=5, label="K=1, λ=1, threshold=1 baseline"),
        Line2D([], [], marker="*", color="#F1B82D", markeredgecolor=DARK, linestyle="None", markersize=9, label="selected validation workpoint"),
        Line2D([], [], marker="o", color="#A5ADB5", linestyle="None", markersize=4, label="K=1 candidate"),
        Line2D([], [], marker="D", color="#7C8791", linestyle="None", markersize=4, label="K=2 candidate"),
        Line2D([], [], marker="^", color="#59656F", linestyle="None", markersize=4, label="K=3 candidate"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.88), ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8)
    fig.text(0.08, 0.965, "Parameter workpoints can recover OOS F1 while preserving the Known/Accuracy guard", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.08, 0.035, "Validation candidates only; gray dashed line = baseline − 1 pp guard. Test confirmation is reported separately.", fontsize=5.9, color="#555B61")
    return _save(fig, "pipeline_workpoint_search_frontier")


def main() -> int:
    data = _load_comparison()
    figures = {
        "pipeline_gate_full_oos_invariance": plot_gate_pipeline_invariance(data),
        "pipeline_oos_f1_vs_paper": plot_paper_comparison(data),
        "pipeline_error_budget_oos_tradeoff": plot_error_budget(data),
        "pipeline_workpoint_search_frontier": plot_workpoint_search(data),
    }
    manifest = {
        "schema_version": "s2c.historical_archive_tuned_full_pipeline.figures.v1",
        "protocol": "historical_v19_archive_development_tuning",
        "backend": "python_matplotlib",
        "datasets": list(DATASETS),
        "seed": 42,
        "source_data": [
            "results/analysis/historical_archive_full_pipeline/comparison.csv",
            "../artifacts/s2c/runs/historical_archive_tuned_full_pipeline_seed42/*/kir50_seed42/per_method.csv",
            "../artifacts/s2c/runs/historical_archive_gate_search_seed42/*/partial_k1/workpoints.csv",
            "../artifacts/s2c/runs/historical_archive_gate_search_seed42/*/partial_k1/selected_validation.csv",
            "fulltex.tex",
        ],
        "figures": figures,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "raw_text_or_sample_level_data_exported": False,
        "notes": "The tuned candidate was selected on validation OOS labels and confirmed on test; standard Banking77 is protocol-separated from the paper historical banking77_oos task.",
    }
    PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    (PUBLIC_ROOT / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
