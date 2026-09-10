#!/usr/bin/env python3
"""Build a current, analysis-only mechanism evidence bundle.

The bundle uses already audited five-seed results.  It does not train a model,
select a threshold, select K, or merge incompatible external-baseline rows.
The purpose is to make the error-budget explanation visual: higher OOS F1 is
only useful when it is not purchased with excessive Known false rejection or
OOS false acceptance.
"""

from __future__ import annotations

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
FAIR = ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
K2 = ROOT / "results/diagnostics/minilm_trainable_k2_control_v1/per_seed.csv"
GEOMETRY = ROOT / "results/analysis/archive/analysis/representation_geometry_visuals_v1/geometry_summary.csv"
GEOMETRY_K = ROOT / "results/analysis/archive/analysis/representation_geometry_visuals_v1/k_effects.csv"
SCORE_GAP = ROOT / "results/analysis/archive/analysis/representation_geometry_visuals_v1/score_gap_false_accept.csv"
OUT = ROOT / "results/analysis/archive/analysis/mechanism_evidence_v2"
FIG = ROOT / "figures/archive/analysis/mechanism_evidence_v2"

DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
METHODS = (
    "trainable_k1",
    "single_centroid",
    "fixed_k2",
    "random_partition",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
    "mogb_minilm",
)
LABELS = {
    "trainable_k1": "Trainable K=1",
    "single_centroid": "Frozen K=1",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_partition_ours_boundary": "MOGB partition + S2C boundary",
    "ours_partition_mogb_boundary": "S2C partition + MOGB boundary",
    "mogb_minilm": "MOGB-MiniLM",
}
COLORS = {
    "trainable_k1": "#0072B2",
    "single_centroid": "#666666",
    "fixed_k2": "#D55E00",
    "random_partition": "#009E73",
    "mogb_partition_ours_boundary": "#CC79A7",
    "ours_partition_mogb_boundary": "#E69F00",
    "mogb_minilm": "#56B4E9",
}
DATASET_LABELS = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}


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


def load_fair() -> pd.DataFrame:
    frame = pd.read_csv(FAIR)
    required = {
        "dataset", "kir", "method", "oos_f1", "f1_all", "f1_k",
        "known_recall", "false_accept_rate", "false_reject_rate", "n_seeds",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fair summary missing columns: {missing}")
    frame = frame[frame.dataset.isin(DATASETS) & frame.method.isin(METHODS)].copy()
    frame["kir"] = frame["kir"].astype(float)
    expected = len(DATASETS) * len(KIRS) * len(METHODS)
    if len(frame) != expected:
        raise ValueError(f"expected {expected} fair rows, got {len(frame)}")
    if set(frame.n_seeds.astype(int)) != {5}:
        raise ValueError("mechanism bundle requires five seeds per fair cell")
    numeric = ["oos_f1", "f1_all", "f1_k", "known_recall", "false_accept_rate", "false_reject_rate"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("non-finite fair metrics")
    frame["method_label"] = frame["method"].map(LABELS)
    frame["dataset_label"] = frame["dataset"].map(DATASET_LABELS)
    return frame


def build_budget(fair: pd.DataFrame) -> pd.DataFrame:
    """Compute the K=2 acceptance budget against the same Frozen K=1 cell."""
    rows: list[dict[str, Any]] = []
    for (dataset, kir), group in fair.groupby(["dataset", "kir"], sort=True):
        base = group[group.method.eq("single_centroid")]
        if len(base) != 1:
            raise ValueError(f"missing Frozen K=1 baseline for {dataset}/{kir}")
        base_row = base.iloc[0]
        for method in ("fixed_k2", "random_partition", "trainable_k1", "mogb_partition_ours_boundary", "ours_partition_mogb_boundary", "mogb_minilm"):
            part = group[group.method.eq(method)]
            if len(part) != 1:
                raise ValueError(f"missing {method} for {dataset}/{kir}")
            row = part.iloc[0]
            rows.append({
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "kir": float(kir),
                "method": method,
                "method_label": LABELS[method],
                "oos_f1_delta_vs_frozen_k1_pp": float((row.oos_f1 - base_row.oos_f1) * 100),
                "f1_all_delta_vs_frozen_k1_pp": float((row.f1_all - base_row.f1_all) * 100),
                "known_recall_delta_vs_frozen_k1_pp": float((row.known_recall - base_row.known_recall) * 100),
                "false_accept_delta_vs_frozen_k1_pp": float((row.false_accept_rate - base_row.false_accept_rate) * 100),
                "false_reject_delta_vs_frozen_k1_pp": float((row.false_reject_rate - base_row.false_reject_rate) * 100),
                "absolute_oos_f1": float(row.oos_f1),
                "absolute_false_accept_rate": float(row.false_accept_rate),
            })
    return pd.DataFrame(rows)


def build_trainable_k2() -> pd.DataFrame:
    if not K2.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(K2)
    required = {"dataset", "seed", "k2_minus_k1_oos_f1", "k2_minus_k1_known_recall", "k2_minus_k1_false_accept_rate", "k2_minus_k1_f1_all"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"trainable K2 control missing columns: {missing}")
    frame = frame.copy()
    frame["kir"] = 0.50
    frame["method"] = "trainable_fixed_k2"
    frame["method_label"] = "Trainable K=2"
    frame["oos_f1_delta_pp"] = frame["k2_minus_k1_oos_f1"] * 100
    frame["f1_all_delta_pp"] = frame["k2_minus_k1_f1_all"] * 100
    frame["known_recall_delta_pp"] = frame["k2_minus_k1_known_recall"] * 100
    frame["false_accept_delta_pp"] = frame["k2_minus_k1_false_accept_rate"] * 100
    return frame[["dataset", "kir", "seed", "method", "method_label", "oos_f1_delta_pp", "f1_all_delta_pp", "known_recall_delta_pp", "false_accept_delta_pp"]]


def build_geometry_risk() -> pd.DataFrame:
    geometry = pd.read_csv(GEOMETRY)
    effects = pd.read_csv(GEOMETRY_K)
    # The legacy geometry summary and the R1 contract-repair summary expose
    # different optional columns.  Keep the common representation diagnostics
    # and leave collision_rate explicitly missing rather than silently mixing
    # two incompatible representation contracts.
    keep = geometry[["dataset", "representation", "effective_rank", "relative_separation"]].copy()
    keep["collision_rate"] = np.nan
    effect_columns = {
        "k2_minus_k1_oos_f1_mean": "k2_minus_k1_oos_f1",
        "k2_minus_k1_id_recall_mean": "k2_minus_k1_id_recall",
        "k2_minus_k1_false_accept_mean": "k2_minus_k1_false_accept_rate",
        "k2_minus_k1_auroc_mean": "k2_minus_k1_auroc",
    }
    missing = sorted(set(effect_columns) - set(effects.columns))
    if missing:
        raise ValueError(f"geometry K-effect summary missing columns: {missing}")
    risk = effects[["dataset", "representation", *effect_columns]].rename(columns=effect_columns)
    result = keep.merge(risk, on=["dataset", "representation"], how="inner")
    result["dataset_label"] = result["dataset"].map(DATASET_LABELS)
    result["oos_delta_pp"] = result["k2_minus_k1_oos_f1"] * 100
    result["known_recall_delta_pp"] = result["k2_minus_k1_id_recall"] * 100
    result["false_accept_delta_pp"] = result["k2_minus_k1_false_accept_rate"] * 100
    return result


def build_score_gap() -> pd.DataFrame:
    frame = pd.read_csv(SCORE_GAP)
    frame["score_gap_pp"] = frame["score_gap_median_oos_minus_known"]
    frame["false_accept_pp"] = frame["false_accept_rate"] * 100
    frame["dataset_label"] = frame["dataset"].map(DATASET_LABELS)
    return frame


def plot_frontier(fair: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        part = fair[fair.dataset.eq(dataset)]
        for method in METHODS:
            sub = part[part.method.eq(method)]
            ax.scatter(sub.false_accept_rate * 100, sub.oos_f1 * 100, s=55, color=COLORS[method], alpha=0.85, label=LABELS[method])
            for _, row in sub.iterrows():
                ax.annotate(f"{row.kir:.2f}", (row.false_accept_rate * 100, row.oos_f1 * 100), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("False acceptance (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[: len(METHODS)], labels[: len(METHODS)], frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Fair matrix: OOS performance versus open-space false acceptance", y=1.14)
    fig.savefig(FIG / "oos_f1_vs_false_acceptance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_budget(budget: pd.DataFrame) -> None:
    methods = ["fixed_k2", "random_partition", "trainable_k1", "mogb_partition_ours_boundary", "ours_partition_mogb_boundary", "mogb_minilm"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True, sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = budget[budget.dataset.eq(dataset)]
        for method in methods:
            sub = part[part.method.eq(method)].sort_values("kir")
            ax.plot(sub.known_recall_delta_vs_frozen_k1_pp, sub.false_accept_delta_vs_frozen_k1_pp, marker="o", color=COLORS[method], label=LABELS[method])
            for _, row in sub.iterrows():
                ax.annotate(f"{row.kir:.2f}", (row.known_recall_delta_vs_frozen_k1_pp, row.false_accept_delta_vs_frozen_k1_pp), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.axvline(0, color="#999999", linewidth=0.8)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Known Recall delta vs Frozen K=1 (pp)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("False acceptance delta vs Frozen K=1 (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[: len(methods)], labels[: len(methods)], frameon=False, ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.suptitle("Error budget of adding/altering local acceptance regions", y=1.17)
    fig.savefig(FIG / "known_recovery_vs_oos_acceptance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_geometry(risk: pd.DataFrame) -> None:
    labels = {"frozen": "Frozen", "ce": "CE", "supcon": "SupCon"}
    colors = {"frozen": "#777777", "ce": "#D55E00", "supcon": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    for _, row in risk.iterrows():
        rep = str(row.representation)
        axes[0].scatter(row.relative_separation, row.oos_delta_pp, color=colors.get(rep, "#0072B2"), s=70)
        axes[0].annotate(f"{row.dataset_label}\n{labels.get(rep, rep)}", (row.relative_separation, row.oos_delta_pp), fontsize=7, xytext=(4, 3), textcoords="offset points")
        axes[1].scatter(row.effective_rank, row.false_accept_delta_pp, color=colors.get(rep, "#0072B2"), s=70)
        axes[1].annotate(f"{row.dataset_label}\n{labels.get(rep, rep)}", (row.effective_rank, row.false_accept_delta_pp), fontsize=7, xytext=(4, 3), textcoords="offset points")
    axes[0].axhline(0, color="#999999", linewidth=0.8)
    axes[1].axhline(0, color="#999999", linewidth=0.8)
    axes[0].set_xlabel("Relative separation (representation diagnostic)")
    axes[0].set_ylabel("K=2 - K=1 OOS F1 (pp)")
    axes[1].set_xlabel("Effective rank")
    axes[1].set_ylabel("K=2 - K=1 false acceptance (pp)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Representation geometry does not by itself guarantee safe multicentroid boundaries")
    fig.savefig(FIG / "representation_geometry_vs_k2_risk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_score_gap(score_gap: pd.DataFrame) -> None:
    order = ["frozen", "trainable"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for dataset in DATASETS:
        part = score_gap[score_gap.dataset.eq(dataset)].set_index("representation").reindex(order)
        axes[0].plot(order, part.score_gap_pp, marker="o", label=DATASET_LABELS[dataset])
        axes[1].plot(order, part.false_accept_pp, marker="o", label=DATASET_LABELS[dataset])
    axes[0].set_title("Median OOS-known score gap")
    axes[0].set_ylabel("Score gap (raw score units)")
    axes[1].set_title("False acceptance at fixed threshold")
    axes[1].set_ylabel("False acceptance (%)")
    for ax in axes:
        ax.set_xlabel("Representation")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Trainable representation changes the K=1 operating margin")
    fig.savefig(FIG / "score_gap_and_false_acceptance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fair = load_fair()
    budget = build_budget(fair)
    trainable_k2 = build_trainable_k2()
    geometry = build_geometry_risk()
    score_gap = build_score_gap()
    atomic_csv(fair.sort_values(["dataset", "kir", "method"]), OUT / "fair_matrix_audited.csv")
    atomic_csv(budget, OUT / "acceptance_budget.csv")
    atomic_csv(trainable_k2, OUT / "trainable_k1_k2_budget.csv")
    atomic_csv(geometry, OUT / "representation_geometry_k2_risk.csv")
    atomic_csv(score_gap, OUT / "score_gap_false_acceptance.csv")
    plot_frontier(fair)
    plot_budget(budget)
    plot_geometry(geometry)
    plot_score_gap(score_gap)
    manifest = {
        "analysis_id": "mechanism_evidence_v2",
        "protocol_version": "protocol_v2_textoir_v1",
        "analysis_only": True,
        "test_selection": False,
        "sources": {name: str(path.relative_to(ROOT)) for name, path in {"fair": FAIR, "trainable_k2": K2, "geometry": GEOMETRY, "geometry_k": GEOMETRY_K, "score_gap": SCORE_GAP}.items()},
        "source_sha256": {name: sha256(path) for name, path in {"fair": FAIR, "trainable_k2": K2, "geometry": GEOMETRY, "geometry_k": GEOMETRY_K, "score_gap": SCORE_GAP}.items()},
        "fair_rows": int(len(fair)),
        "budget_rows": int(len(budget)),
        "trainable_k2_rows": int(len(trainable_k2)),
        "geometry_rows": int(len(geometry)),
        "figures": sorted(path.name for path in FIG.glob("*.png")),
        "interpretation": "descriptive mechanism analysis; no test-derived selection rule",
    }
    atomic_json(manifest, OUT / "MANIFEST.json")
    print(json.dumps({"status": "ok", "fair_rows": len(fair), "budget_rows": len(budget), "figures": len(manifest["figures"]), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
