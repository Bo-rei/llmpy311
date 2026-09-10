#!/usr/bin/env python3
"""Build the low-reading-cost OOS figure for the historical protocol audit.

Claim: under the paper's controlled K=1 Gate contract, Trainable K=1 improves
OOS F1 over Frozen K=1 on the H1 KIR=.50/seed=42 snapshots; the reported
full-Cascade values are shown separately because they are not Gate replay evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_protocol_v1"
FIGURE_ROOT = PROJECT_ROOT / "figures" / "historical_protocol_oos_v1"
RESULT_ROOT = PROJECT_ROOT / "results" / "analysis" / "historical_protocol_v1"


DATASETS = ("CLINC150", "StackOverflow", "BANKING77-OOS")
DATASET_KEYS = {"CLINC150": "clinc150", "StackOverflow": "stackoverflow", "BANKING77-OOS": "banking77_oos"}
PAPER_OOS = {"CLINC150": 91.96, "StackOverflow": 89.71, "BANKING77-OOS": 88.23}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        key = DATASET_KEYS[dataset]
        minilm_root = ARTIFACT_ROOT / "minilm_k1" / key / "kir50_seed42"
        if key == "banking77_oos":
            minilm_root = minilm_root / "lambda_1.0"
        frozen = _read_json(minilm_root / "frozen_k1" / "metrics.json")
        trainable = _read_json(minilm_root / "trainable_k1" / "metrics.json")
        rows.extend(
            [
                {"dataset": dataset, "method": "Frozen K=1\n(controlled Gate)", "oos_f1": 100.0 * float(frozen["oos_f1"]), "false_accept_rate": 100.0 * float(frozen["false_accept_rate"]), "source": str(minilm_root / "frozen_k1" / "metrics.json")},
                {"dataset": dataset, "method": "Trainable K=1\n(controlled Gate)", "oos_f1": 100.0 * float(trainable["oos_f1"]), "false_accept_rate": 100.0 * float(trainable["false_accept_rate"]), "source": str(minilm_root / "trainable_k1" / "metrics.json")},
                {"dataset": dataset, "method": "Paper Ours\n(full Cascade reference)", "oos_f1": PAPER_OOS[dataset], "false_accept_rate": None, "source": "fulltex.tex:main_results_all/KIR=.50"},
            ]
        )
    return rows


def write_source_data(rows: list[dict[str, object]]) -> Path:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    path = RESULT_ROOT / "historical_protocol_oos_comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "method", "oos_f1_percent", "false_accept_rate_percent", "source"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"dataset": row["dataset"], "method": row["method"], "oos_f1_percent": f"{float(row['oos_f1']):.6f}", "false_accept_rate_percent": "" if row["false_accept_rate"] is None else f"{float(row['false_accept_rate']):.6f}", "source": row["source"]})
    return path


def build_figure(rows: list[dict[str, object]]) -> plt.Figure:
    colors = {
        "Frozen K=1\n(controlled Gate)": "#4C78A8",
        "Trainable K=1\n(controlled Gate)": "#E45756",
        "Paper Ours\n(full Cascade reference)": "#B7B7B7",
    }
    methods = tuple(colors)
    fig, (ax_bar, ax_delta) = plt.subplots(1, 2, figsize=(7.204724, 3.622047), gridspec_kw={"width_ratios": (1.25, 0.95)})
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.27, top=0.84, wspace=0.34)

    x = np.arange(len(DATASETS))
    width = 0.22
    for index, method in enumerate(methods):
        values = [float(next(row["oos_f1"] for row in rows if row["dataset"] == dataset and row["method"] == method)) for dataset in DATASETS]
        bars = ax_bar.bar(x + (index - 1.5) * width, values, width=width, color=colors[method], edgecolor="#263238", linewidth=0.35, label=method, zorder=3)
        if method.startswith("Paper"):
            for bar in bars:
                bar.set_hatch("//")
                bar.set_edgecolor("#5F6368")
        for bar, value in zip(bars, values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2.0, value + 1.1, f"{value:.1f}", ha="center", va="bottom", fontsize=6.1, color="#263238")
    ax_bar.set_ylim(75, 97)
    ax_bar.set_ylabel("OOS F1 (%)")
    ax_bar.set_xticks(x, DATASETS)
    ax_bar.grid(axis="y", color="#D9DEE3", linewidth=0.5, zorder=0)
    ax_bar.set_title("H1 controlled Gate · KIR=.50 · seed=42", loc="left", fontweight="bold", pad=8)
    handles, labels = ax_bar.get_legend_handles_labels()

    frozen = np.array([float(next(row["oos_f1"] for row in rows if row["dataset"] == dataset and row["method"].startswith("Frozen"))) for dataset in DATASETS])
    trainable = np.array([float(next(row["oos_f1"] for row in rows if row["dataset"] == dataset and row["method"].startswith("Trainable"))) for dataset in DATASETS])
    frozen_fa = np.array([float(next(row["false_accept_rate"] for row in rows if row["dataset"] == dataset and row["method"].startswith("Frozen"))) for dataset in DATASETS])
    trainable_fa = np.array([float(next(row["false_accept_rate"] for row in rows if row["dataset"] == dataset and row["method"].startswith("Trainable"))) for dataset in DATASETS])
    delta_oos = trainable - frozen
    ax_delta.axhline(0, color="#263238", linewidth=0.7)
    bars = ax_delta.bar(np.arange(len(DATASETS)), delta_oos, color="#E45756", edgecolor="#263238", linewidth=0.35, zorder=3)
    fa_reduction = frozen_fa - trainable_fa
    for bar, delta, reduction in zip(bars, delta_oos, fa_reduction):
        ax_delta.text(bar.get_x() + bar.get_width() / 2.0, delta + 0.18, f"+{delta:.1f}", ha="center", va="bottom", fontsize=6.6)
        ax_delta.text(bar.get_x() + bar.get_width() / 2.0, max(0.25, delta * 0.45), f"FA −{reduction:.1f}pp", ha="center", va="center", fontsize=5.8, color="white", fontweight="bold")
    ax_delta.set_xlabel("Dataset")
    ax_delta.set_ylabel("Δ OOS F1 (pp)\nTrainable − Frozen")
    ax_delta.set_title("Trainable gain and FA reduction", loc="left", fontweight="bold", pad=8)
    ax_delta.set_xticks(np.arange(len(DATASETS)), DATASETS)
    ax_delta.set_ylim(0, max(8.5, float(delta_oos.max()) + 1.8))
    ax_delta.grid(color="#E5E9EC", linewidth=0.5, zorder=0)
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.065), handlelength=1.4, columnspacing=1.0, handletextpad=0.45)
    fig.text(0.01, 0.965, "a", fontsize=9, fontweight="bold", va="top")
    fig.text(0.56, 0.965, "b", fontsize=9, fontweight="bold", va="top")
    fig.text(0.08, 0.015, "Hatched bars are full-Cascade paper references; solid bars are same-contract Gate replay. OOS is the positive class.", fontsize=6.2, color="#455A64")
    return fig


def main() -> None:
    rows = load_rows()
    source_path = write_source_data(rows)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig = build_figure(rows)
    # Final physical canvas: 183 mm wide by 92 mm high (double-column layout).
    fig.savefig(FIGURE_ROOT / "historical_protocol_oos_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / "historical_protocol_oos_comparison.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / "historical_protocol_oos_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / "historical_protocol_oos_comparison.svg", bbox_inches="tight")
    manifest = {
        "claim": "On H1 controlled KIR=.50/seed=42 snapshots, Trainable K=1 improves OOS F1 over Frozen K=1; paper full-Cascade values are separate references.",
        "archetype": "low_reading_cost_bar_plus_delta_bar",
        "backend": "python_matplotlib",
        "source_data": str(source_path),
        "outputs": [str(path) for path in sorted(FIGURE_ROOT.glob("historical_protocol_oos_comparison.*"))],
        "n": "one fixed H1 snapshot per dataset; seed=42; no seed aggregation",
        "metric": "standard OOS-positive F1 at score threshold=1.0",
        "panel_b_secondary": "False-acceptance reduction is annotated as a secondary OOS error signal; Known Recall is not plotted.",
        "field_mapping": {
            "panel_a": "metrics.oos_f1 x dataset x method",
            "panel_b": "Trainable oos_f1 - Frozen oos_f1; Frozen false_accept_rate - Trainable false_accept_rate",
        },
    }
    (FIGURE_ROOT / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
