#!/usr/bin/env python3
"""Compare Trainable K=1 Gate with native detectors on the same representation.

This is an analysis-only layer. It reads completed per-seed rows, verifies the
Known-only/provenance contract, and never changes a checkpoint or threshold.
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
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GATE_SOURCE = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
NATIVE_SOURCE = ROOT / "results/analysis/archive/analysis/native_baselines_trainable_v1/trainable_native_per_seed.csv"
OUT = ROOT / "results/analysis/archive/analysis/trainable_detector_mechanism_v1"
FIG = ROOT / "figures/archive/analysis/trainable_detector_mechanism_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87)
KIR = 0.50
DETECTORS = ("msp", "energy", "knn", "lof")
METRICS = (
    "oos_f1",
    "f1_all",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
    "auroc",
    "aupr_oos",
    "accuracy",
)
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_REPS = 10_000
LABELS = {
    "trainable_k1": "Trainable Gate K=1",
    "msp": "MSP",
    "energy": "Energy",
    "knn": "kNN",
    "lof": "LOF",
}
COLORS = {
    "trainable_k1": "#c53030",
    "msp": "#2b6cb0",
    "energy": "#2f855a",
    "knn": "#805ad5",
    "lof": "#dd6b20",
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


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    gate = pd.read_csv(GATE_SOURCE)
    gate = gate[
        gate.dataset.isin(DATASETS)
        & gate.seed.isin(SEEDS)
        & np.isclose(gate.kir, KIR)
        & (gate.method == "trainable_k1")
    ].copy()
    native = pd.read_csv(NATIVE_SOURCE)
    native = native[
        native.dataset.isin(DATASETS)
        & native.seed.isin(SEEDS)
        & np.isclose(native.kir, KIR)
        & native.method.isin(DETECTORS)
    ].copy()
    if len(gate) != len(DATASETS) * len(SEEDS):
        raise ValueError(f"expected 9 Gate rows, got {len(gate)}")
    if len(native) != len(DATASETS) * len(SEEDS) * len(DETECTORS):
        raise ValueError(f"expected 36 native rows, got {len(native)}")
    if gate.duplicated(["dataset", "seed"]).any() or native.duplicated(["dataset", "seed", "method"]).any():
        raise ValueError("duplicate dataset/seed rows")
    if gate.test_used_for_selection.astype(bool).any() or gate.oos_used_for_training.astype(bool).any():
        raise ValueError("Gate source violates test/OOS selection contract")
    if native.test_used_for_selection.astype(bool).any() or native.uses_oos_for_training.astype(bool).any():
        raise ValueError("native source violates test/OOS selection contract")
    return gate, native


def paired_effects(gate: pd.DataFrame, native: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        g = gate[gate.dataset == dataset].set_index("seed").loc[list(SEEDS)]
        for detector in DETECTORS:
            n = native[(native.dataset == dataset) & (native.method == detector)].set_index("seed").loc[list(SEEDS)]
            for metric in METRICS:
                delta = g[metric].to_numpy(dtype=float) - n[metric].to_numpy(dtype=float)
                low, high = bootstrap_ci(delta, rng)
                if metric in {"false_accept_rate", "false_reject_rate"}:
                    better = -delta
                else:
                    better = delta
                rows.extend(
                    {
                        "dataset": dataset,
                        "kir": KIR,
                        "seed": int(seed),
                        "comparison": "trainable_gate_minus_native",
                        "native_method": detector,
                        "metric": metric,
                        "raw_delta": float(d),
                        "better_delta": float(b),
                    }
                    for seed, d, b in zip(SEEDS, delta, better)
                )
                std = float(np.std(better, ddof=1))
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "kir": KIR,
                        "native_method": detector,
                        "metric": metric,
                        "n_seeds": len(better),
                        "mean_delta": float(np.mean(delta)),
                        "mean_better_delta": float(np.mean(better)),
                        "std_better_delta": std,
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": int(np.sum(better > 1e-12)),
                        "ties": int(np.sum(np.abs(better) <= 1e-12)),
                        "losses": int(np.sum(better < -1e-12)),
                        "effect_size": float(np.mean(better) / std) if std > 0 else 0.0,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def performance_table(gate: pd.DataFrame, native: pd.DataFrame) -> pd.DataFrame:
    gate = gate.assign(method="trainable_k1")
    gate = gate[["dataset", "kir", "seed", "method", *METRICS]]
    native = native[["dataset", "kir", "seed", "method", *METRICS]]
    frame = pd.concat([gate, native], ignore_index=True)
    table = frame.groupby(["dataset", "kir", "method"], as_index=False)[list(METRICS)].agg(["mean", "std"]).reset_index()
    table.columns = [
        "_".join(str(part) for part in column if str(part) not in {"", "None"}).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in table.columns
    ]
    if "index" in table.columns:
        table = table.drop(columns=["index"])
    return table


def plot_pareto(gate: pd.DataFrame, native: pd.DataFrame) -> None:
    frame = pd.concat(
        [gate.assign(method="trainable_k1"), native], ignore_index=True
    )
    means = frame.groupby(["dataset", "method"], as_index=False)[["oos_f1", "known_recall", "false_accept_rate"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    for ax, dataset in zip(axes, DATASETS):
        sub = means[means.dataset == dataset]
        for _, row in sub.iterrows():
            method = row.method
            ax.scatter(row.known_recall * 100, row.oos_f1 * 100, s=80 + row.false_accept_rate * 180, color=COLORS[method], alpha=0.9)
            ax.annotate(LABELS[method], (row.known_recall * 100, row.oos_f1 * 100), fontsize=8, xytext=(4, 3), textcoords="offset points")
        ax.set_title(dataset)
        ax.set_xlabel("Known Recall (%)")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("OOS F1 (%)")
    fig.suptitle("Same Trainable MiniLM representation: Gate vs native detectors\nMarker size = false acceptance", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG / "trainable_detector_pareto.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_forest(summary: pd.DataFrame) -> None:
    metrics = [("oos_f1", "OOS F1"), ("f1_all", "F1-All"), ("known_recall", "Known Recall")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharex=False, constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        sub = summary[summary.metric == metric].copy()
        labels = []
        for dataset in DATASETS:
            for detector in DETECTORS:
                labels.append(f"{dataset} / {LABELS[detector]}")
        sub["label"] = [f"{row.dataset} / {LABELS[row.native_method]}" for _, row in sub.iterrows()]
        sub = sub.set_index("label").loc[labels].reset_index()
        y = np.arange(len(sub))
        err_low = (sub.mean_delta - sub.ci95_low) * 100
        err_high = (sub.ci95_high - sub.mean_delta) * 100
        ax.errorbar(sub.mean_delta * 100, y, xerr=[err_low, err_high], fmt="o", color="#2b6cb0", ecolor="#718096", capsize=3)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_yticks(y, sub.label, fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Trainable Gate − native detector (pp)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Trainable Gate minus native detector (3 seeds; 95% bootstrap CI)", y=1.01)
    fig.savefig(FIG / "trainable_detector_paired_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_errors(gate: pd.DataFrame, native: pd.DataFrame) -> None:
    frame = pd.concat([gate.assign(method="trainable_k1"), native], ignore_index=True)
    means = frame.groupby(["dataset", "method"], as_index=False)[["false_accept_rate", "false_reject_rate"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = means[means.dataset == dataset]
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width / 2, sub.false_accept_rate * 100, width, label="False acceptance", color="#c53030")
        ax.bar(x + width / 2, sub.false_reject_rate * 100, width, label="False rejection", color="#2b6cb0")
        ax.set_title(dataset)
        ax.set_xticks(x, [LABELS[m] for m in sub.method], rotation=25)
        ax.set_xlabel("Detector")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Error rate (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Error composition on the same Trainable MiniLM representation", y=1.02)
    fig.savefig(FIG / "trainable_detector_error_balance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    gate, native = load_inputs()
    per_seed, summary = paired_effects(gate, native)
    table = performance_table(gate, native)
    atomic_csv(per_seed, OUT / "paired_effects_per_seed.csv")
    atomic_csv(summary, OUT / "paired_effects_summary.csv")
    atomic_csv(table, OUT / "performance_summary.csv")
    plot_pareto(gate, native)
    plot_forest(summary)
    plot_errors(gate, native)
    atomic_json(
        {
            "analysis_id": "trainable_detector_mechanism_v1",
            "protocol_version": "protocol_v2_textoir_v1",
            "kir": KIR,
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "gate_source": str(GATE_SOURCE),
            "gate_source_sha256": sha256(GATE_SOURCE),
            "native_source": str(NATIVE_SOURCE),
            "native_source_sha256": sha256(NATIVE_SOURCE),
            "native_detectors": list(DETECTORS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPS,
            "gate_rows": int(len(gate)),
            "native_rows": int(len(native)),
            "contract": {
                "trainable_gate": "Known-only Trainable MiniLM K=1 Gate",
                "native_detectors": "same Trainable MiniLM representation; Known-only conformal calibration",
                "test_used_for_selection": False,
                "oos_used_for_training": False,
            },
            "outputs": [
                "paired_effects_per_seed.csv",
                "paired_effects_summary.csv",
                "performance_summary.csv",
                "trainable_detector_pareto.png",
                "trainable_detector_paired_effects.png",
                "trainable_detector_error_balance.png",
            ],
        },
        OUT / "MANIFEST.json",
    )


if __name__ == "__main__":
    main()
