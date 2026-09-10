#!/usr/bin/env python3
"""Join Trainable-vs-ADB metric deltas with aligned error-budget deltas.

This is a read-only analysis of completed artifacts.  It does not tune a
threshold, select a checkpoint, or export sample-level predictions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(".")
METRICS = ROOT / "results/analysis/archive/analysis/trainable_vs_adb_kir_v1/paired_effects_pp.csv"
ERRORS = ROOT / "results/analysis/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/state_deltas.csv"
OUT = ROOT / "results/analysis/archive/analysis/trainable_vs_adb_mechanism_summary_v1"
FIG = ROOT / "figures/archive/analysis/trainable_vs_adb_mechanism_summary_v1"
REPORT = ROOT / "docs/archive/analysis/TRAINABLE_VS_ADB_MECHANISM_SUMMARY_V1.md"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load() -> pd.DataFrame:
    metrics = pd.read_csv(METRICS)
    metrics = metrics[metrics.metric.isin(["oos_f1", "f1_all", "known_recall", "false_acceptance", "false_rejection"])].copy()
    if metrics.duplicated(["dataset", "kir", "metric"]).any():
        raise ValueError("duplicate metric cells")
    expected = {(d, k) for d in DATASETS for k in KIRS}
    got = set(zip(metrics.dataset, metrics.kir))
    if not expected.issubset(got):
        raise ValueError(f"missing metric cells: {expected - got}")
    wide = metrics.pivot(index=["dataset", "kir"], columns="metric", values="mean_delta_pp").reset_index()
    errors = pd.read_csv(ERRORS)
    errors = errors[errors.state.isin(["known_rejected", "oos_false_accept"])].copy()
    if errors.duplicated(["dataset", "kir", "state"]).any():
        raise ValueError("duplicate error cells")
    # The source stores the relevant rate in separate columns by state.
    selected = errors.pivot(index=["dataset", "kir"], columns="state", values=["delta_rate_known", "delta_rate_oos"]).reset_index()
    selected.columns = [
        "dataset" if column[0] == "dataset" else "kir" if column[0] == "kir" else f"{column[1]}_{column[0]}"
        for column in selected.columns
    ]
    selected = selected.rename(
        columns={
            "known_rejected_delta_rate_known": "delta_known_rejected_rate",
            "oos_false_accept_delta_rate_oos": "delta_oos_false_accept_rate",
        }
    )
    joined = wide.merge(selected[["dataset", "kir", "delta_known_rejected_rate", "delta_oos_false_accept_rate"]], on=["dataset", "kir"], validate="one_to_one")
    joined["mechanism"] = joined.apply(classify, axis=1)
    joined = joined.sort_values(["dataset", "kir"]).reset_index(drop=True)
    if len(joined) != 9 or joined[["oos_f1", "f1_all", "known_recall", "delta_known_rejected_rate", "delta_oos_false_accept_rate"]].isna().any().any():
        raise ValueError("joined mechanism table is incomplete")
    return joined


def classify(row: pd.Series) -> str:
    known = float(row["delta_known_rejected_rate"])
    oos = float(row["delta_oos_false_accept_rate"])
    if known > 0 and oos < 0:
        return "conservative_rejection"
    if known < 0 and oos > 0:
        return "coverage_false_accept_tradeoff"
    if known < 0 and oos <= 0:
        return "coverage_recovery"
    if known >= 0 and oos >= 0:
        return "joint_error_increase"
    return "mixed"


def write_outputs(table: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "dataset_kir_mechanism.csv", index=False, float_format="%.10f")
    dataset = table.groupby("dataset", as_index=False).agg(
        mean_oos_f1_delta_pp=("oos_f1", "mean"),
        mean_f1_all_delta_pp=("f1_all", "mean"),
        mean_known_rejected_delta_pp=("delta_known_rejected_rate", lambda x: 100 * x.mean()),
        mean_oos_false_accept_delta_pp=("delta_oos_false_accept_rate", lambda x: 100 * x.mean()),
        n_kir=("kir", "nunique"),
    )
    mechanism_map = table.groupby("dataset")["mechanism"].apply(classify_dataset_group).to_dict()
    dataset["dominant_mechanism"] = dataset["dataset"].map(mechanism_map)
    dataset.to_csv(OUT / "dataset_mechanism_summary.csv", index=False, float_format="%.10f")
    mechanism = table.groupby(["dataset", "mechanism"], as_index=False).size().rename(columns={"size": "n_kir"})
    mechanism.to_csv(OUT / "mechanism_counts.csv", index=False)
    source_hashes = {"metrics": sha256(METRICS), "errors": sha256(ERRORS)}
    manifest = {
        "schema_version": "trainable_vs_adb_mechanism_summary_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "scope": "read_only_metric_and_sample_aligned_error_budget_join",
        "dataset_count": len(DATASETS),
        "kir_count": len(KIRS),
        "cells": len(table),
        "source_hashes": source_hashes,
        "raw_text_exported": False,
        "sample_level_predictions_exported": False,
        "selection_used_test_oos": False,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_figures(table)
    write_report(table, dataset, source_hashes)


def classify_dataset_group(values: pd.Series) -> str:
    mechanisms = set(values.tolist())
    if "conservative_rejection" in mechanisms and len(mechanisms) == 1:
        return "保守拒识：Known 误拒增加，但 OOS 误接收下降"
    if "coverage_false_accept_tradeoff" in mechanisms and "coverage_recovery" in mechanisms:
        return "覆盖恢复，但中高 KIR 出现误接收权衡"
    if "coverage_false_accept_tradeoff" in mechanisms:
        return "覆盖—误接收权衡：Known 覆盖恢复，但 OOS 误接收上升"
    if "coverage_recovery" in mechanisms:
        return "覆盖恢复：Known 误拒和 OOS 误接收同时下降"
    return "混合工作点"


def write_figures(table: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "figure.dpi": 140})
    specs = [
        ("oos_f1", "Trainable - ADB OOS F1 (pp)"),
        ("f1_all", "Trainable - ADB F1-All (pp)"),
        ("delta_known_rejected_rate", "Trainable - ADB Known rejection (pp)"),
        ("delta_oos_false_accept_rate", "Trainable - ADB OOS false accept (pp)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, (column, title) in zip(axes.flat, specs):
        pivot = table.pivot(index="dataset", columns="kir", values=column).reindex(index=DATASETS, columns=KIRS)
        values = pivot.to_numpy(dtype=float)
        if column.startswith("delta_"):
            values = values * 100
        vmax = max(1.0, float(np.nanmax(np.abs(values))))
        im = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(KIRS)), [f"{k:.2f}" for k in KIRS])
        ax.set_yticks(range(len(DATASETS)), DATASETS)
        for i in range(len(DATASETS)):
            for j in range(len(KIRS)):
                ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(FIG / "dataset_kir_mechanism_heatmaps.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
    colors = {"clinc150": "#1f77b4", "banking77": "#ff7f0e", "stackoverflow": "#2ca02c"}
    for dataset in DATASETS:
        subset = table[table.dataset.eq(dataset)]
        ax.scatter(subset.delta_known_rejected_rate * 100, subset.delta_oos_false_accept_rate * 100, s=70, color=colors[dataset], label=dataset)
        for _, row in subset.iterrows():
            ax.annotate(f"{row.kir:.2f}", (row.delta_known_rejected_rate * 100, row.delta_oos_false_accept_rate * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Trainable − ADB: Known rejection delta (pp)")
    ax.set_ylabel("Trainable − ADB: OOS false-accept delta (pp)")
    ax.set_title("Dataset/KIR error-budget mechanisms")
    ax.legend(title="dataset")
    fig.savefig(FIG / "dataset_kir_error_budget_quadrants.png", dpi=180)
    plt.close(fig)


def write_report(table: pd.DataFrame, dataset: pd.DataFrame, hashes: dict[str, str]) -> None:
    lines = [
        "# Trainable-K1 与 ADB 数据集/KIR 机制汇总 V1",
        "",
        "更新时间：2026-08-10",
        "",
        "本报告把已经完成的 Trainable-K1/ADB 指标差值与逐样本错误预算差值按同一 `dataset×KIR` 对齐。它只做后验分析，不训练、不调阈值、不选择 checkpoint，也不导出原始文本或逐样本公共结果。ADB 仍是 BERT/TextOIR 外部合同，不能与 MiniLM fair 行合并成 SOTA 排名。",
        "",
        "## 工作点表",
        "",
        "| 数据集 | KIR | Δ OOS F1 | Δ F1-All | Δ Known Recall | Δ Known 误拒 | Δ OOS 误接收 | 机制 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in table.iterrows():
        lines.append(f"| {row.dataset} | {row.kir:.2f} | {row.oos_f1:+.2f}pp | {row.f1_all:+.2f}pp | {row.known_recall:+.2f}pp | {row.delta_known_rejected_rate*100:+.2f}pp | {row.delta_oos_false_accept_rate*100:+.2f}pp | {row.mechanism} |")
    lines.extend(["", "## 数据集结论", "", "| 数据集 | 平均 Δ OOS F1 | 平均 Δ F1-All | 平均 Δ Known 误拒 | 平均 Δ OOS 误接收 | 主要机制 |", "|---|---:|---:|---:|---:|---|"])
    for _, row in dataset.iterrows():
        lines.append(f"| {row.dataset} | {row.mean_oos_f1_delta_pp:+.2f}pp | {row.mean_f1_all_delta_pp:+.2f}pp | {row.mean_known_rejected_delta_pp:+.2f}pp | {row.mean_oos_false_accept_delta_pp:+.2f}pp | {row.dominant_mechanism} |")
    lines.extend([
        "",
        "## 解释",
        "",
        "- CLINC150 和 Banking77 的 Trainable 工作点更保守：Known 误拒上升，但 OOS 误接收明显下降，因此 OOS 指标改善不能只解释为 Known 覆盖恢复。",
        "- StackOverflow 的方向不同：Trainable 在三个 KIR 都减少 Known 误拒；KIR=.25 同时减少 OOS 误接收，而 KIR=.50/.75 出现轻微 OOS 误接收增加，体现覆盖—开放空间风险权衡。",
        "- 因此不存在一个跨数据集的“Trainable 必然减少所有错误”机制。后续解释必须同时报告 OOS F1、F1-All、Known Recall 和两类条件错误率。",
        "- 这些差异同时包含 MiniLM 与 BERT、训练目标和边界实现差异；本报告用于机制归因，不用于宣称超过完整 ADB/MOGB/DCLOOS 或 SOTA。",
        "",
        "## 产物",
        "",
        "- `results/analysis/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_kir_mechanism.csv`",
        "- `results/analysis/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_mechanism_summary.csv`",
        "- `figures/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_kir_mechanism_heatmaps.png`",
        "- `figures/archive/analysis/trainable_vs_adb_mechanism_summary_v1/dataset_kir_error_budget_quadrants.png`",
        "",
        f"源文件 SHA256：metrics=`{hashes['metrics']}`；error_budget=`{hashes['errors']}`。",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    table = load()
    write_outputs(table)
    print(json.dumps({"status": "ok", "cells": len(table), "datasets": len(DATASETS), "kirs": len(KIRS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
