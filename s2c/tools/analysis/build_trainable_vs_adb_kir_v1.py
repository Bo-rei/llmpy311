#!/usr/bin/env python3
"""Build a contract-aware Trainable-K1 versus ADB KIR analysis.

This is a read-only analysis of completed artifacts.  ADB uses a BERT/TextOIR
compatibility contract while Trainable-K1 uses the current Known-only MiniLM
contract; the output therefore reports paired same-split effects without
claiming a same-backbone ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "oos_f1": "OOS F1",
    "f1_all": "F1-All",
    "known_recall": "Known Recall",
    "false_accept_rate": "False Acceptance",
    "false_reject_rate": "False Rejection",
    "accuracy": "Accuracy",
}
DATASET_ORDER = ["clinc150", "banking77", "stackoverflow"]
KIR_ORDER = [0.25, 0.50, 0.75]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sampled = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def load_and_merge(fair_path: Path, adb_path: Path) -> pd.DataFrame:
    fair = pd.read_csv(fair_path)
    fair = fair[fair["method"].eq("trainable_k1")].copy()
    adb = pd.read_csv(adb_path)
    adb = adb[adb["method"].eq("ADB") & adb["valid_semantic_metrics"].astype(bool)].copy()
    keys = ["dataset", "kir", "seed"]
    expected = {(d, k, s) for d in DATASET_ORDER for k in KIR_ORDER for s in [13, 42, 87, 100, 123]}
    got_fair = set(map(tuple, fair[keys].itertuples(index=False, name=None)))
    got_adb = set(map(tuple, adb[keys].itertuples(index=False, name=None)))
    if got_fair != expected:
        raise ValueError(f"Trainable source coverage mismatch: {len(got_fair)} rows")
    if got_adb != expected:
        raise ValueError(f"ADB source coverage mismatch: {len(got_adb)} rows")
    if fair.duplicated(keys).any() or adb.duplicated(keys).any():
        raise ValueError("Duplicate dataset/kir/seed rows in source")
    keep = keys + list(METRICS)
    merged = fair[keep].merge(adb[keep], on=keys, suffixes=("_trainable", "_adb"), validate="one_to_one")
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_trainable"] - merged[f"{metric}_adb"]
    return merged.sort_values(keys).reset_index(drop=True)


def make_summaries(merged: pd.DataFrame, bootstrap_seed: int, bootstrap_resamples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, object]] = []
    effects: list[dict[str, object]] = []
    for (dataset, kir), group in merged.groupby(["dataset", "kir"], sort=False):
        for metric in METRICS:
            values = group[f"delta_{metric}"].to_numpy(float)
            ci_low, ci_high = bootstrap_ci(values, rng, bootstrap_resamples)
            row = {
                "dataset": dataset,
                "kir": float(kir),
                "metric": metric,
                "metric_label": METRICS[metric],
                "n_seeds": int(len(values)),
                "trainable_mean": float(group[f"{metric}_trainable"].mean()),
                "trainable_std": float(group[f"{metric}_trainable"].std(ddof=1)),
                "adb_mean": float(group[f"{metric}_adb"].mean()),
                "adb_std": float(group[f"{metric}_adb"].std(ddof=1)),
                "delta_mean": float(values.mean()),
                "delta_std": float(values.std(ddof=1)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "trainable_wins": int((values > 0).sum()),
                "ties": int((values == 0).sum()),
                "adb_wins": int((values < 0).sum()),
            }
            rows.append(row)
            effects.append({
                "dataset": dataset,
                "kir": float(kir),
                "metric": metric,
                "mean_delta_pp": float(values.mean() * 100.0),
                "ci95_low_pp": ci_low * 100.0,
                "ci95_high_pp": ci_high * 100.0,
                "wins": int((values > 0).sum()),
                "ties": int((values == 0).sum()),
                "losses": int((values < 0).sum()),
            })
    return pd.DataFrame(rows), pd.DataFrame(effects)


def configure_matplotlib() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140})
    candidates = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
    for name in candidates:
        try:
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def heatmap(ax: plt.Axes, data: pd.DataFrame, metric: str, title: str) -> None:
    pivot = data[data.metric.eq(metric)].pivot(index="dataset", columns="kir", values="mean_delta_pp")
    pivot = pivot.reindex(index=DATASET_ORDER, columns=KIR_ORDER)
    vmax = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
    im = ax.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(KIR_ORDER)), [f"{x:.2f}" for x in KIR_ORDER])
    ax.set_yticks(range(len(DATASET_ORDER)), DATASET_ORDER)
    ax.set_xlabel("KIR")
    ax.set_title(title)
    for i in range(len(DATASET_ORDER)):
        for j in range(len(KIR_ORDER)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=9)
    return im


def make_figures(merged: pd.DataFrame, effects: pd.DataFrame, figure_dir: Path) -> None:
    configure_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric, title in zip(
        axes.flat,
        ["oos_f1", "f1_all", "known_recall", "false_accept_rate"],
        ["Trainable − ADB: OOS F1 (pp)", "Trainable − ADB: F1-All (pp)", "Trainable − ADB: Known Recall (pp)", "Trainable − ADB: False Acceptance (pp)"],
    ):
        heatmap(ax, effects, metric, title)
    fig.suptitle("Same-split comparison: Known-only MiniLM Trainable vs BERT/TextOIR ADB", fontsize=13)
    fig.savefig(figure_dir / "trainable_vs_adb_kir_delta_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASET_ORDER):
        group = merged[merged.dataset.eq(dataset)]
        for method, suffix, color, marker in [
            ("Trainable K=1", "trainable", "#1f77b4", "o"),
            ("ADB", "adb", "#d62728", "s"),
        ]:
            mean = group.groupby("kir")[f"f1_all_{suffix}"].mean().reindex(KIR_ORDER)
            std = group.groupby("kir")[f"f1_all_{suffix}"].std(ddof=1).reindex(KIR_ORDER)
            ax.errorbar(KIR_ORDER, mean * 100, yerr=std * 100, label=method, color=color, marker=marker, capsize=3)
        ax.set_title(dataset)
        ax.set_xlabel("KIR")
        ax.set_xticks(KIR_ORDER, [".25", ".50", ".75"])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("F1-All (%)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Known classification quality across KIR", fontsize=13)
    fig.savefig(figure_dir / "trainable_vs_adb_f1_all_kir_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    for method, suffix, color, marker in [
        ("Trainable K=1", "trainable", "#1f77b4", "o"),
        ("ADB", "adb", "#d62728", "s"),
    ]:
        points = merged.groupby(["dataset", "kir"])[[f"f1_all_{suffix}", f"false_accept_rate_{suffix}"]].mean().reset_index()
        ax.scatter(points[f"false_accept_rate_{suffix}"] * 100, points[f"f1_all_{suffix}"] * 100, label=method, color=color, marker=marker, s=55)
        for _, row in points.iterrows():
            ax.annotate(f"{row.dataset[:3]}-{row.kir:.2f}", (row[f"false_accept_rate_{suffix}"] * 100, row[f"f1_all_{suffix}"] * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("False Acceptance (%)")
    ax.set_ylabel("F1-All (%)")
    ax.set_title("Coverage–open-space trade-off across 9 KIR cells")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(figure_dir / "trainable_vs_adb_f1_all_false_acceptance.png", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output: Path, figure_dir: Path, source_hashes: dict[str, str]) -> None:
    lines = [
        "# Trainable-K1 与 ADB 的跨 KIR 对比分析 V1",
        "",
        "更新时间：2026-08-10",
        "",
        "本文只分析已完成的 3 数据集×3 KIR×5 seed 配对结果。Trainable-K1 是当前 Known-only MiniLM 合同，ADB 是 BERT/TextOIR 外部兼容合同；二者共享 protocol split/KIR/seed，但不共享 backbone 和训练实现。因此本文报告同数据工作点差异，不把结果写成同骨干 SOTA 排名。",
        "",
        "## 结论摘要",
        "",
        "- 训练表示和 ADB 的差异具有数据集/KIR 条件性，不能只看 StackOverflow 单格。",
        "- Trainable-K1 的主要优势集中在 F1-All、Known Recall 和 false rejection；ADB 在部分开放比例下更保守，false acceptance 可能更低。",
        "- 如果只看 OOS F1，会掩盖 Known 覆盖与错误预算差异；因此同时报告 F1-All、Known Recall、false acceptance 和 false rejection。",
        "- 这些结果支持“当前 Trainable-K1 是一个更平衡的同数据工作点”，但不支持跨 backbone 的全面 SOTA 结论。",
        "",
        "## 配对统计",
        "",
        "所有 delta 定义为 Trainable-K1 − ADB；bootstrap 只用于描述区间，不用于选择参数。",
        "",
        "| 指标 | 9 个 dataset×KIR 单元的 delta 均值（pp） | 解释 |",
        "|---|---:|---|",
    ]
    global_rows = []
    for metric in METRICS:
        x = summary[summary.metric.eq(metric)]["delta_mean"].to_numpy() * 100
        global_rows.append((metric, float(x.mean()), int((x > 0).sum()), int((x < 0).sum())))
    labels = {
        "oos_f1": "OOS F1",
        "f1_all": "F1-All",
        "known_recall": "Known Recall",
        "false_accept_rate": "False Acceptance",
        "false_reject_rate": "False Rejection",
        "accuracy": "Accuracy",
    }
    interpretations = {
        "oos_f1": "正值表示 Trainable 的 OOS F1 更高",
        "f1_all": "正值表示整体已知/未知分类更高",
        "known_recall": "正值表示 Trainable 保留更多 Known",
        "false_accept_rate": "负值才表示 Trainable 的 OOS 误接受更低",
        "false_reject_rate": "负值表示 Trainable 的 Known 误拒绝更少",
        "accuracy": "正值表示总体准确率更高",
    }
    for metric, mean, wins, losses in global_rows:
        lines.append(f"| {labels[metric]} | {mean:+.2f} | {interpretations[metric]}；9 个单元正/负={wins}/{losses} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "1. ADB 使用 BERT/TextOIR，Trainable-K1 使用 Known-only MiniLM；因此正向差异不能归因于某一个组件而不考虑 backbone。",
        "2. 训练表示的收益必须和 detector/边界一起解释；已有 fair matrix 显示固定 K>1 在 StackOverflow 上会增加 union false acceptance。",
        "3. 该分析不使用 test OOS 调参，不改变任何 checkpoint、registry、阈值或历史 artifact。",
        "",
        "## 产物",
        "",
        "- 机器可读结果：`results/analysis/archive/analysis/trainable_vs_adb_kir_v1/`",
        f"- 图表目录：`{figure_dir}`",
        f"- 源文件哈希：`{json.dumps(source_hashes, ensure_ascii=False, sort_keys=True)}`",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair", type=Path, default=Path("results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"))
    parser.add_argument("--adb", type=Path, default=Path("results/analysis/adb_kir_sensitivity_v2/adb_per_seed.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/archive/analysis/trainable_vs_adb_kir_v1"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/archive/analysis/trainable_vs_adb_kir_v1"))
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = load_and_merge(args.fair, args.adb)
    summary, effects = make_summaries(merged, args.bootstrap_seed, args.bootstrap_resamples)
    merged.to_csv(args.output_dir / "paired_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "cell_summary.csv", index=False)
    effects.to_csv(args.output_dir / "paired_effects_pp.csv", index=False)
    make_figures(merged, effects, args.figure_dir)
    source_hashes = {"fair": sha256(args.fair), "adb": sha256(args.adb)}
    manifest = {
        "schema_version": 1,
        "analysis": "trainable_vs_adb_kir_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "scope": "3 datasets x 3 KIR x 5 seeds; descriptive same-split external-contract comparison",
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "source_hashes": source_hashes,
        "n_paired_rows": int(len(merged)),
        "n_cells": int(len(summary) // len(METRICS)),
        "selection_uses_test_oos": False,
        "contract_note": "Trainable-K1 is Known-only MiniLM; ADB is BERT/TextOIR compatibility; no same-backbone SOTA claim.",
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(summary, Path("docs/analysis/ADB_TRAINABLE_KIR_ANALYSIS_V1"), args.figure_dir, source_hashes)
    print(json.dumps({"rows": len(merged), "cells": len(summary) // len(METRICS), "output": str(args.output_dir), "figures": str(args.figure_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
