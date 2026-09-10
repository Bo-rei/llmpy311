#!/usr/bin/env python3
"""Summarize the completed native-detector controls on MiniLM representations.

The experiment is analysis-only: it reuses the three-seed KIR=.50 outputs for
the same Trainable/Frozen MiniLM checkpoints and compares MSP, Energy, kNN,
LOF with the Trainable K=1 Gate.  It is intended to isolate representation
gain from detector/Gate gain without launching another run.
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
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NATIVE_TRAIN = ROOT / "results/analysis/archive/analysis/native_baselines_trainable_v1/trainable_native_per_seed.csv"
NATIVE_FROZEN = ROOT / "results/analysis/archive/analysis/native_baselines_trainable_v1/frozen_native_per_seed.csv"
GATE_TRAIN = ROOT / "results/analysis/archive/analysis/native_baselines_trainable_v1/trainable_gate_per_seed.csv"
PAIRED_REPR = ROOT / "results/analysis/archive/analysis/native_baselines_trainable_v1/trainable_vs_frozen_native_paired.csv"
PAIRED_GATE = ROOT / "results/analysis/archive/analysis/native_baselines_trainable_v1/trainable_native_vs_gate_paired.csv"
OUT = ROOT / "results/analysis/archive/analysis/detector_mechanism_v1"
FIG = ROOT / "figures/archive/analysis/detector_mechanism_v1"

DATASETS = ("clinc150", "banking77", "stackoverflow")
DETECTORS = ("trainable_k1", "msp", "energy", "knn", "lof")
NATIVE = ("msp", "energy", "knn", "lof")
METRICS = ("oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos")
LABELS = {
    "trainable_k1": "Trainable Gate K=1",
    "msp": "MSP",
    "energy": "Energy",
    "knn": "kNN",
    "lof": "LOF",
}
COLORS = {
    "trainable_k1": "#0072B2",
    "msp": "#D55E00",
    "energy": "#009E73",
    "knn": "#E69F00",
    "lof": "#CC79A7",
}
DATASET_LABELS = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}

_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
if _FONT.is_file():
    font_manager.fontManager.addfont(str(_FONT))
    _FAMILY = font_manager.FontProperties(fname=str(_FONT)).get_name()
else:
    _FAMILY = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = [_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(temporary, format=path.suffix.lstrip("."), dpi=180, bbox_inches="tight")
    plt.close(fig)
    os.replace(temporary, path)


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(NATIVE_TRAIN)
    frozen = pd.read_csv(NATIVE_FROZEN)
    gate = pd.read_csv(GATE_TRAIN)
    for frame, name in ((train, "trainable_native"), (frozen, "frozen_native"), (gate, "trainable_gate")):
        required = {"dataset", "kir", "seed", "method", "oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc", "aupr_oos"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if "false_reject_rate" not in frame.columns:
            frame["false_reject_rate"] = 1.0 - frame["known_recall"]
        if set(frame.dataset) != set(DATASETS) or set(frame.kir.astype(float)) != {0.5}:
            raise ValueError(f"{name} is not the registered KIR=.50 three-dataset scope")
        if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contains non-finite metrics")
    if len(train) != 36 or len(frozen) != 36 or len(gate) != 9:
        raise ValueError(f"unexpected row counts: train={len(train)}, frozen={len(frozen)}, gate={len(gate)}")
    train["representation"] = "trainable_minilm"
    frozen["representation"] = "frozen_minilm"
    gate["representation"] = "trainable_minilm"
    return train, frozen, gate


def load_paired(path: Path) -> pd.DataFrame:
    """Load an already-computed paired-bootstrap table without recomputing it."""
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "kir",
        "comparison_method",
        "metric",
        "mean_left_minus_right",
        "ci95_low",
        "ci95_high",
        "wins",
        "ties",
        "losses",
        "bootstrap_seed",
        "bootstrap_samples",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    if set(frame.dataset) != set(DATASETS) or set(frame.kir.astype(float)) != {0.5}:
        raise ValueError(f"{path.name} is not the registered KIR=.50 three-dataset scope")
    numeric = ["mean_left_minus_right", "ci95_low", "ci95_high"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name} contains non-finite confidence intervals")
    if set(frame.bootstrap_seed.astype(int)) != {20260725} or set(frame.bootstrap_samples.astype(int)) != {10000}:
        raise ValueError(f"{path.name} does not use the registered bootstrap contract")
    return frame


def summarize(train: pd.DataFrame, frozen: pd.DataFrame, gate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    frozen = frozen.copy()
    gate = gate.copy()
    train["method_group"] = train["method"]
    frozen["method_group"] = frozen["method"]
    gate["method_group"] = "trainable_k1"
    gate["method"] = "trainable_k1"
    frames = []
    for source, scope in ((gate, "trainable_gate"), (train, "trainable_native"), (frozen, "frozen_native")):
        frame = source.copy()
        frame["scope"] = scope
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    summary = all_rows.groupby(["dataset", "method_group", "representation", "scope"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
        **{f"{metric}_std": (metric, "std") for metric in METRICS},
    )
    summary["dataset_label"] = summary["dataset"].map(DATASET_LABELS)
    summary["method_label"] = summary["method_group"].map(LABELS)
    for metric in METRICS:
        summary[f"{metric}_mean_pct"] = summary[f"{metric}_mean"] * 100
        summary[f"{metric}_std_pct"] = summary[f"{metric}_std"] * 100

    # Representation gain is paired by dataset/method/seed, not by separate means.
    native_train = train.set_index(["dataset", "method", "seed"])
    native_frozen = frozen.set_index(["dataset", "method", "seed"])
    keys = native_train.index.intersection(native_frozen.index)
    rows = []
    for dataset, method, seed in keys:
        left = native_train.loc[(dataset, method, seed)]
        right = native_frozen.loc[(dataset, method, seed)]
        row = {"dataset": dataset, "method": method, "seed": seed, "dataset_label": DATASET_LABELS[dataset], "method_label": LABELS[method]}
        for metric in METRICS:
            row[f"{metric}_delta"] = float(left[metric] - right[metric])
            row[f"{metric}_delta_pp"] = float((left[metric] - right[metric]) * 100)
        rows.append(row)
    repr_delta = pd.DataFrame(rows)
    repr_summary = repr_delta.groupby(["dataset", "method", "method_label"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        **{f"{metric}_delta_pp_mean": (f"{metric}_delta_pp", "mean") for metric in METRICS},
        **{f"{metric}_delta_pp_std": (f"{metric}_delta_pp", "std") for metric in METRICS},
    )

    # Gate versus each native detector, paired by the same trainable seed.
    gate_idx = gate.set_index(["dataset", "seed"])
    rows = []
    for _, native_row in train.iterrows():
        gate_row = gate_idx.loc[(native_row["dataset"], native_row["seed"])]
        row = {"dataset": native_row["dataset"], "method": native_row["method"], "seed": native_row["seed"], "dataset_label": DATASET_LABELS[native_row["dataset"]], "method_label": LABELS[native_row["method"]]}
        for metric in METRICS:
            row[f"gate_minus_native_{metric}"] = float(gate_row[metric] - native_row[metric])
            row[f"gate_minus_native_{metric}_pp"] = float((gate_row[metric] - native_row[metric]) * 100)
        rows.append(row)
    gate_delta = pd.DataFrame(rows)
    return summary, repr_summary, gate_delta


def paired_ci_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize source paired tables to the direction used in the report.

    The stored Gate table is native - Gate, while the report needs Gate - native.
    Confidence limits and wins/losses are inverted together; no bootstrap is
    recomputed here.
    """
    gate_source = load_paired(PAIRED_GATE)
    repr_source = load_paired(PAIRED_REPR)
    keep_metrics = {"oos_f1", "f1_all", "known_recall", "false_accept_rate"}

    gate = gate_source[gate_source.metric.isin(keep_metrics)].copy()
    gate["comparison"] = "Trainable Gate K=1 - native detector"
    gate["reference_method"] = gate["comparison_method"]
    gate["mean_delta"] = -gate["mean_left_minus_right"]
    gate["ci95_low_delta"] = -gate["ci95_high"]
    gate["ci95_high_delta"] = -gate["ci95_low"]
    gate["wins_delta"] = gate["losses"]
    gate["ties_delta"] = gate["ties"]
    gate["losses_delta"] = gate["wins"]
    gate = gate[[
        "dataset", "kir", "reference_method", "metric", "comparison",
        "mean_delta", "ci95_low_delta", "ci95_high_delta", "wins_delta",
        "ties_delta", "losses_delta", "n_seeds", "bootstrap_seed",
        "bootstrap_samples",
    ]].sort_values(["dataset", "metric", "reference_method"]).reset_index(drop=True)

    repr_frame = repr_source[repr_source.metric.isin(keep_metrics)].copy()
    repr_frame["comparison"] = "Trainable native - Frozen native"
    repr_frame["reference_method"] = repr_frame["comparison_method"]
    repr_frame["mean_delta"] = repr_frame["mean_left_minus_right"]
    repr_frame["ci95_low_delta"] = repr_frame["ci95_low"]
    repr_frame["ci95_high_delta"] = repr_frame["ci95_high"]
    repr_frame["wins_delta"] = repr_frame["wins"]
    repr_frame["ties_delta"] = repr_frame["ties"]
    repr_frame["losses_delta"] = repr_frame["losses"]
    repr_frame = repr_frame[[
        "dataset", "kir", "reference_method", "metric", "comparison",
        "mean_delta", "ci95_low_delta", "ci95_high_delta", "wins_delta",
        "ties_delta", "losses_delta", "n_seeds", "bootstrap_seed",
        "bootstrap_samples",
    ]].sort_values(["dataset", "metric", "reference_method"]).reset_index(drop=True)
    for frame in (gate, repr_frame):
        frame["mean_delta_pp"] = frame["mean_delta"] * 100
        frame["ci95_low_delta_pp"] = frame["ci95_low_delta"] * 100
        frame["ci95_high_delta_pp"] = frame["ci95_high_delta"] * 100
    return gate, repr_frame


def plot_comparison(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    x = np.arange(len(DETECTORS))
    width = 0.17
    for idx, (metric, title) in enumerate((("oos_f1", "OOS F1"), ("f1_all", "F1-All"), ("false_accept_rate", "False acceptance"))):
        ax = axes[idx]
        for dataset_index, dataset in enumerate(DATASETS):
            sub = summary[(summary.dataset == dataset) & summary.method_group.isin(DETECTORS) & summary.scope.isin(("trainable_gate", "trainable_native"))].set_index("method_group")
            vals = [float(sub.loc[method, f"{metric}_mean_pct"]) for method in DETECTORS]
            offset = (dataset_index - 1) * width
            ax.bar(x + offset, vals, width=width, label=DATASET_LABELS[dataset] if idx == 0 else None, alpha=0.84)
        ax.set_xticks(x, [LABELS[m].replace("Trainable Gate K=1", "Gate K=1") for m in DETECTORS], rotation=25, ha="right")
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if metric != "false_accept_rate":
            ax.set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Detector comparison on the same Trainable MiniLM checkpoints (KIR=.50)", fontsize=13)
    path = FIG / "native_detector_comparison.png"
    save_figure(fig, path)
    return path


def plot_frontier(summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = summary[(summary.dataset == dataset) & summary.scope.isin(("trainable_gate", "trainable_native"))]
        for method in DETECTORS:
            row = sub[sub.method_group == method].iloc[0]
            ax.scatter(row.false_accept_rate_mean_pct, row.oos_f1_mean_pct, s=90, color=COLORS[method], label=LABELS[method])
            ax.annotate(LABELS[method].replace("Trainable Gate K=1", "Gate K=1"), (row.false_accept_rate_mean_pct, row.oos_f1_mean_pct), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("False acceptance (%)")
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("OOS F1 (%)")
    fig.suptitle("OOS F1–false-acceptance frontier: Gate versus native detectors", fontsize=13)
    path = FIG / "native_detector_frontier.png"
    save_figure(fig, path)
    return path


def plot_representation_gain(repr_summary: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(NATIVE))
    width = 0.24
    for di, dataset in enumerate(DATASETS):
        sub = repr_summary[repr_summary.dataset == dataset].set_index("method")
        vals = [float(sub.loc[m, "oos_f1_delta_pp_mean"]) for m in NATIVE]
        axes[0].bar(x + (di - 1) * width, vals, width=width, label=DATASET_LABELS[dataset])
        vals_fa = [float(sub.loc[m, "false_accept_rate_delta_pp_mean"]) for m in NATIVE]
        axes[1].bar(x + (di - 1) * width, vals_fa, width=width, label=DATASET_LABELS[dataset])
    for ax, title, ylabel in ((axes[0], "Trainable representation gain", "Trainable - Frozen OOS F1 (pp)"), (axes[1], "Representation effect on false acceptance", "Trainable - Frozen false acceptance (pp)")):
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(x, [LABELS[m] for m in NATIVE], rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Separating representation training from detector choice", fontsize=13)
    path = FIG / "representation_vs_detector_gain.png"
    save_figure(fig, path)
    return path


def plot_paired_forest(frame: pd.DataFrame, filename: str, title: str, x_label: str) -> Path:
    """Plot paired-bootstrap intervals for each dataset and reference method."""
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 5.0), constrained_layout=True, sharex=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = frame[frame.dataset == dataset].copy()
        methods = list(dict.fromkeys(sub.reference_method.tolist()))
        y = np.arange(len(methods))
        # Use OOS F1 as the forest metric; caller filters to that metric.
        sub = sub.set_index("reference_method").reindex(methods)
        means = sub.mean_delta_pp.to_numpy(dtype=float)
        low = means - sub.ci95_low_delta_pp.to_numpy(dtype=float)
        high = sub.ci95_high_delta_pp.to_numpy(dtype=float) - means
        ax.errorbar(means, y, xerr=[low, high], fmt="o", color="#0072B2", ecolor="#0072B2", capsize=3, linewidth=1.4)
        ax.axvline(0, color="#555555", linewidth=0.9)
        ax.set_yticks(y, [LABELS.get(m, m) for m in methods])
        ax.set_title(DATASET_LABELS[dataset])
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    axes[0].set_ylabel("Reference detector")
    for ax in axes:
        ax.set_xlabel(x_label)
    fig.suptitle(title, fontsize=13)
    path = FIG / filename
    save_figure(fig, path)
    return path


def main() -> None:
    train, frozen, gate = load()
    summary, repr_summary, gate_delta = summarize(train, frozen, gate)
    paired_gate, paired_repr = paired_ci_tables()
    OUT.mkdir(parents=True, exist_ok=True)
    save_csv(summary, OUT / "detector_summary.csv")
    save_csv(repr_summary, OUT / "representation_gain_summary.csv")
    save_csv(gate_delta, OUT / "gate_vs_native_paired.csv")
    save_csv(paired_gate, OUT / "detector_paired_ci.csv")
    save_csv(paired_repr, OUT / "representation_paired_ci.csv")
    gate_oos = paired_gate[paired_gate.metric == "oos_f1"]
    repr_oos = paired_repr[paired_repr.metric == "oos_f1"]
    figures = [
        plot_comparison(summary),
        plot_frontier(summary),
        plot_representation_gain(repr_summary),
        plot_paired_forest(
            gate_oos,
            "gate_vs_native_oos_f1_ci.png",
            "配对 bootstrap：Trainable Gate K=1 相对原生 detector",
            "Gate - native OOS F1 (百分点)",
        ),
        plot_paired_forest(
            repr_oos,
            "trainable_vs_frozen_native_oos_f1_ci.png",
            "配对 bootstrap：Trainable 相对 Frozen 的原生 detector 增益",
            "Trainable - Frozen OOS F1 (百分点)",
        ),
    ]
    manifest = {
        "schema": 2,
        "analysis_only": True,
        "protocol_version": "protocol_v2_textoir_v1",
        "scope": {"datasets": list(DATASETS), "kir": 0.50, "seeds": [13, 42, 87]},
        "builder_sha256": sha256(Path(__file__).resolve()),
        "source_files": {str(path.relative_to(ROOT)): sha256(path) for path in (NATIVE_TRAIN, NATIVE_FROZEN, GATE_TRAIN, PAIRED_REPR, PAIRED_GATE)},
        "outputs": [str(path.relative_to(ROOT)) for path in sorted(OUT.glob("*.csv"))],
        "figures": [str(path.relative_to(ROOT)) for path in figures],
        "rows": {
            "detector_summary": len(summary),
            "representation_gain_summary": len(repr_summary),
            "gate_vs_native_paired": len(gate_delta),
            "detector_paired_ci": len(paired_gate),
            "representation_paired_ci": len(paired_repr),
        },
        "notes": [
            "Trainable and Frozen native detector rows use the same three-seed KIR=.50 scope.",
            "Thresholds were fixed by the registered Known-calibration conformal contract in the source experiment.",
            "No test result is used for selection here; the outputs are post-hoc analysis only.",
            "Paired confidence intervals are copied from the registered 10,000-resample bootstrap tables; no new resampling is performed.",
            "Gate-vs-native source direction is inverted so positive values mean Gate is better; wins/losses are inverted accordingly.",
        ],
    }
    save_json(manifest, OUT / "MANIFEST.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
