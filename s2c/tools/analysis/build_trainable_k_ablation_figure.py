#!/usr/bin/env python3
"""Build the paper figure for the Trainable MiniLM K ablation.

The source table is the completed protocol_v2_textoir_v1 Gate sweep.  The
figure uses one shared y-axis so that the cross-dataset effect of K is visible
without adding a second panel or a redundant legend.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep the visual vocabulary of the existing paper-style OOS figures.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 12.0,
        "axes.labelsize": 14.0,
        "axes.titlesize": 14.0,
        "xtick.labelsize": 13.0,
        "ytick.labelsize": 13.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT.parent
    / "artifacts"
    / "s2c"
    / "runs"
    / "protocol_v2_textoir_v1"
    / "minilm_trainable_k_sweep_v1"
    / "summary.csv"
)
FIGURE_ROOT = ROOT / "figures"
RESULT_ROOT = ROOT / "results" / "analysis" / "trainable_k_ablation_figure"

DATASET_ORDER = ("clinc150", "banking77", "stackoverflow")
DATASET_LABELS = {
    "clinc150": "CLINC150",
    "banking77": "Banking77",
    "stackoverflow": "StackOverflow",
}
COLORS = {
    "clinc150": "#4C78A8",
    "banking77": "#C44E52",
    "stackoverflow": "#F2B134",
}
MARKERS = {"clinc150": "o", "banking77": "s", "stackoverflow": "^"}
DARK = "#252525"
GRID = "#E7E9EC"
GRAY = "#6F767D"


def load_source() -> pd.DataFrame:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing K-ablation summary: {SOURCE}")
    frame = pd.read_csv(SOURCE)
    required = {
        "dataset",
        "kir",
        "k",
        "n_seeds",
        "oos_f1_mean",
        "oos_f1_std",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"K-ablation summary is missing columns: {missing}")
    frame = frame.loc[np.isclose(frame["kir"].astype(float), 0.50)].copy()
    if len(frame) != 15:
        raise ValueError(f"Expected 15 K-ablation rows at KIR=0.50, found {len(frame)}")
    if set(frame["dataset"]) != set(DATASET_ORDER):
        raise ValueError(f"Unexpected datasets: {sorted(set(frame['dataset']))}")
    if set(frame["k"].astype(int)) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected K values 1 through 5")
    if not np.all(frame["n_seeds"].astype(int).to_numpy() == 5):
        raise ValueError("The figure requires five-seed summaries")
    if not np.isfinite(frame[["oos_f1_mean", "oos_f1_std"]].to_numpy()).all():
        raise ValueError("OOS F1 summaries must be finite")
    return frame.sort_values(["dataset", "k"]).reset_index(drop=True)


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.204724, 2.95))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.235, top=0.98)

    k_values = np.arange(1, 6, dtype=float)
    for dataset in DATASET_ORDER:
        part = frame.loc[frame["dataset"] == dataset].sort_values("k")
        mean = 100.0 * part["oos_f1_mean"].to_numpy(dtype=float)
        std = 100.0 * part["oos_f1_std"].to_numpy(dtype=float)
        ax.errorbar(
            k_values,
            mean,
            yerr=std,
            color=COLORS[dataset],
            marker=MARKERS[dataset],
            markersize=6.0,
            markerfacecolor=COLORS[dataset],
            markeredgecolor=DARK,
            markeredgewidth=0.45,
            linewidth=2.0,
            elinewidth=0.95,
            capsize=3.0,
            capthick=0.95,
            label=DATASET_LABELS[dataset],
            zorder=3,
        )

    ax.set_xlabel("Number of centroids ($K$)", labelpad=5)
    ax.set_ylabel("OOS F1 (%)", labelpad=7)
    ax.set_xticks(k_values, ["1", "2", "3", "4", "5"])
    ax.set_xlim(0.75, 5.35)
    ax.set_ylim(52.0, 94.0)
    ax.set_yticks(np.arange(55.0, 95.0, 5.0))
    ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=4.0, width=0.8, pad=3, colors=DARK)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(DARK)

    # Direct labels avoid a separate legend and preserve the style of the
    # existing paper figures while keeping the plot area compact.  Anchor each
    # label at the final marker and move it into the right-side gutter so that
    # neither the line nor its marker/error bar runs through the text.
    label_offsets = {
        "clinc150": (20.0, 4.0),
        "banking77": (20.0, 4.0),
        "stackoverflow": (20.0, 4.0),
    }
    for dataset in DATASET_ORDER:
        part = frame.loc[frame["dataset"] == dataset].sort_values("k")
        last_mean = 100.0 * float(part.iloc[-1]["oos_f1_mean"])
        ax.annotate(
            DATASET_LABELS[dataset],
            xy=(k_values[-1], last_mean),
            xytext=label_offsets[dataset],
            textcoords="offset points",
            color=COLORS[dataset],
            fontsize=12.5,
            fontweight="bold",
            ha="left",
            va="center",
            clip_on=False,
        )
    return fig


def write_manifest(frame: pd.DataFrame, outputs: list[Path]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = frame[["dataset", "kir", "k", "n_seeds", "oos_f1_mean", "oos_f1_std"]].copy()
    selected.to_csv(RESULT_ROOT / "figure_source_data.csv", index=False)
    manifest = {
        "claim": "The effect of the centroid number K is dataset-dependent under the Trainable MiniLM Gate.",
        "archetype": "quantitative trend figure",
        "backend": "python_matplotlib",
        "source_data": str(SOURCE),
        "source_data_copy": str(RESULT_ROOT / "figure_source_data.csv"),
        "metric": "OOS F1 (%)",
        "kir": 0.50,
        "distance": "mahalanobis_diag",
        "representation": "last2_minilm_plus_projection",
        "n_seeds": 5,
        "spread": "standard deviation across seeds",
        "exclusions": "none; all 15 dataset-by-K rows from summary.csv are plotted",
        "field_mapping": {
            "x": "k",
            "y": "100 * oos_f1_mean",
            "error": "100 * oos_f1_std",
            "group": "dataset",
        },
        "outputs": [str(path) for path in outputs],
        "review_risk": "K is evaluated on the test split for descriptive ablation only; it must not be selected from this figure.",
    }
    (RESULT_ROOT / "FIGURE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    frame = load_source()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig = build_figure(frame)
    stem = FIGURE_ROOT / "k_ablation_compact"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    outputs = sorted(FIGURE_ROOT.glob("k_ablation_compact.*"))
    write_manifest(frame, outputs)
    print(json.dumps({"source": str(SOURCE), "outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
