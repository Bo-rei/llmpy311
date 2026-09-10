#!/usr/bin/env python3
"""Decompose OOS F1 into precision/recall using aligned error states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE = Path("results/analysis/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/per_seed_error_budget.csv")
OUT = Path("results/analysis/archive/analysis/trainable_vs_adb_oos_decomposition_v1")
FIG = Path("figures/archive/analysis/trainable_vs_adb_oos_decomposition_v1")
REPORT = Path("docs/archive/analysis/TRAINABLE_VS_ADB_OOS_DECOMPOSITION_V1.md")
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
METHODS = ("Trainable-K1", "ADB")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def decompose_counts(
    n_known: int,
    n_oos: int,
    oos_correct: int,
    oos_false_accept: int,
    known_rejected: int,
    known_correct: int,
    known_wrong: int,
) -> dict[str, float | int]:
    """Return binary OOS metrics from the five mutually exclusive states."""
    precision = oos_correct / max(1, oos_correct + known_rejected)
    recall = oos_correct / max(1, n_oos)
    known_acceptance = (known_correct + known_wrong) / max(1, n_known)
    return {
        "oos_precision": precision,
        "oos_recall": recall,
        "oos_f1": f1(precision, recall),
        "known_acceptance": known_acceptance,
        "oos_correct": oos_correct,
        "oos_false_accept": oos_false_accept,
        "known_rejected": known_rejected,
    }


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE)
    expected_states = {"known_correct", "known_wrong", "known_rejected", "oos_correct_rejected", "oos_false_accept"}
    if set(source.state.unique()) != expected_states:
        raise ValueError("unexpected error states")
    expected = {(d, k, s, m) for d in DATASETS for k in KIRS for s in (13, 42, 87, 100, 123) for m in METHODS}
    got = set(zip(source.dataset, source.kir, source.seed, source.method))
    if got != expected:
        raise ValueError(f"coverage mismatch: missing={len(expected - got)} extra={len(got - expected)}")
    rows = []
    for (dataset, kir, seed, method), group in source.groupby(["dataset", "kir", "seed", "method"], sort=False):
        values = {str(row.state): row for _, row in group.iterrows()}
        n_known = int(values["known_correct"].n_known)
        n_oos = int(values["oos_correct_rejected"].n_oos)
        oos_correct = int(values["oos_correct_rejected"]["count"])
        oos_false = int(values["oos_false_accept"]["count"])
        known_rejected = int(values["known_rejected"]["count"])
        metrics = decompose_counts(
            n_known=n_known,
            n_oos=n_oos,
            oos_correct=oos_correct,
            oos_false_accept=oos_false,
            known_rejected=known_rejected,
            known_correct=int(values["known_correct"]["count"]),
            known_wrong=int(values["known_wrong"]["count"]),
        )
        rows.append({"dataset": dataset, "kir": kir, "seed": seed, "method": method, "n_known": n_known, "n_oos": n_oos, **metrics})
    per_seed = pd.DataFrame(rows).sort_values(["dataset", "kir", "seed", "method"]).reset_index(drop=True)
    paired = per_seed.pivot(index=["dataset", "kir", "seed"], columns="method", values=["oos_precision", "oos_recall", "oos_f1", "known_acceptance"]).reset_index()
    paired.columns = ["dataset" if c[0] == "dataset" else "kir" if c[0] == "kir" else "seed" if c[0] == "seed" else f"{c[1]}_{c[0]}" for c in paired.columns]
    for metric in ("oos_precision", "oos_recall", "oos_f1", "known_acceptance"):
        paired[f"delta_{metric}"] = paired[f"Trainable-K1_{metric}"] - paired[f"ADB_{metric}"]
    return per_seed, paired


def write_outputs(per_seed: pd.DataFrame, paired: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(OUT / "per_seed_oos_decomposition.csv", index=False, float_format="%.10f")
    paired.to_csv(OUT / "paired_per_seed_decomposition.csv", index=False, float_format="%.10f")
    summary = paired.groupby(["dataset", "kir"], as_index=False).agg(
        delta_oos_precision_pp=("delta_oos_precision", lambda x: 100 * x.mean()),
        delta_oos_recall_pp=("delta_oos_recall", lambda x: 100 * x.mean()),
        delta_oos_f1_pp=("delta_oos_f1", lambda x: 100 * x.mean()),
        delta_known_acceptance_pp=("delta_known_acceptance", lambda x: 100 * x.mean()),
        n_seeds=("seed", "nunique"),
    )
    summary.to_csv(OUT / "cell_summary.csv", index=False, float_format="%.10f")
    dataset = summary.groupby("dataset", as_index=False).mean(numeric_only=True)
    dataset.to_csv(OUT / "dataset_summary.csv", index=False, float_format="%.10f")
    manifest = {"schema_version": "trainable_vs_adb_oos_decomposition_v1", "protocol_version": "protocol_v2_textoir_v1", "cells": 45, "paired_rows": len(paired), "source_sha256": sha256(SOURCE), "raw_text_exported": False, "sample_level_predictions_exported": False, "selection_used_test_oos": False}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_figures(summary, per_seed)
    write_report(summary, dataset, manifest["source_sha256"])


def write_figures(summary: pd.DataFrame, per_seed: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "figure.dpi": 140})
    specs = [("delta_oos_precision_pp", "Δ OOS precision (pp)"), ("delta_oos_recall_pp", "Δ OOS recall (pp)"), ("delta_oos_f1_pp", "Δ OOS F1 (pp)"), ("delta_known_acceptance_pp", "Δ Known acceptance (pp)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, (col, title) in zip(axes.flat, specs):
        pivot = summary.pivot(index="dataset", columns="kir", values=col).reindex(index=DATASETS, columns=KIRS)
        values = pivot.to_numpy(dtype=float)
        vmax = max(1.0, float(np.nanmax(np.abs(values))))
        im = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(KIRS)), [f"{k:.2f}" for k in KIRS])
        ax.set_yticks(range(len(DATASETS)), DATASETS)
        for i in range(len(DATASETS)):
            for j in range(len(KIRS)):
                ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(FIG / "oos_precision_recall_decomposition_heatmaps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = per_seed[per_seed.dataset.eq(dataset)]
        for method, color in (("Trainable-K1", "#1f77b4"), ("ADB", "#d62728")):
            means = subset[subset.method.eq(method)].groupby("kir", as_index=False)[["oos_precision", "oos_recall"]].mean()
            ax.plot(means.oos_recall * 100, means.oos_precision * 100, marker="o", label=method, color=color)
            for _, row in means.iterrows():
                ax.annotate(f"{row.kir:.2f}", (row.oos_recall * 100, row.oos_precision * 100), xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.set_title(dataset)
        ax.set_xlabel("OOS recall (%)")
        ax.set_ylabel("OOS precision (%)")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.savefig(FIG / "oos_precision_recall_workpoints.png", dpi=180)
    plt.close(fig)


def write_report(summary: pd.DataFrame, dataset: pd.DataFrame, source_hash: str) -> None:
    lines = ["# Trainable-K1 与 ADB 的 OOS Precision/Recall 分解 V1", "", "更新时间：2026-08-10", "", "本报告从已经完成的逐样本对齐状态计数重算 OOS precision、OOS recall、OOS F1 和 Known acceptance，按同一 dataset×KIR×seed 比较 Trainable-K1 与 ADB。它不训练、不调阈值、不选择 checkpoint。ADB 仍是 BERT/TextOIR 外部合同。", "", "## 工作点分解", "", "| 数据集 | KIR | Δ OOS precision | Δ OOS recall | Δ OOS F1 | Δ Known acceptance |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in summary.iterrows():
        lines.append(f"| {row.dataset} | {row.kir:.2f} | {row.delta_oos_precision_pp:+.2f}pp | {row.delta_oos_recall_pp:+.2f}pp | {row.delta_oos_f1_pp:+.2f}pp | {row.delta_known_acceptance_pp:+.2f}pp |")
    lines.extend(["", "## 数据集平均", "", "| 数据集 | Δ OOS precision | Δ OOS recall | Δ OOS F1 | Δ Known acceptance |", "|---|---:|---:|---:|---:|"])
    for _, row in dataset.iterrows():
        lines.append(f"| {row.dataset} | {row.delta_oos_precision_pp:+.2f}pp | {row.delta_oos_recall_pp:+.2f}pp | {row.delta_oos_f1_pp:+.2f}pp | {row.delta_known_acceptance_pp:+.2f}pp |")
    lines.extend(["", "## 解释", "", "- CLINC150 和 Banking77 的 OOS F1 改善主要来自 OOS recall 增加，也就是 OOS false acceptance 减少；OOS precision 反而下降，原因是更多 Known 被判成 OOS，体现保守拒识代价。", "- StackOverflow 的 Trainable 在 KIR=.25 同时提高 OOS precision 和 recall；KIR=.50/.75 的 recall 略降，但 precision 提升足以使 OOS F1 仍为正。其主要收益是减少 Known rejection，并没有重现固定 K>1 的 acceptance-union 爆炸。", "- 这组分解支持“Trainable 的优势是数据集相关的分数/边界工作点”，不支持跨 BERT/MiniLM 合同的无条件 SOTA 结论。", "", "## 产物", "", "- `results/analysis/archive/analysis/trainable_vs_adb_oos_decomposition_v1/`", "- `figures/archive/analysis/trainable_vs_adb_oos_decomposition_v1/oos_precision_recall_decomposition_heatmaps.png`", "- `figures/archive/analysis/trainable_vs_adb_oos_decomposition_v1/oos_precision_recall_workpoints.png`", "", f"源文件 SHA256：`{source_hash}`。"])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    per_seed, paired = load()
    write_outputs(per_seed, paired)
    print(json.dumps({"status": "ok", "per_seed_rows": len(per_seed), "paired_rows": len(paired)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
