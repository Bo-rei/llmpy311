#!/usr/bin/env python3
"""导出 s2c 多簇/OOS 实验的论文级聚合图。

脚本只读取 ``cluster_separability_v19`` 已冻结的 CSV，不重新编码文本、拟合
KMeans 或选择阈值。这样绘图失败不会污染实验结果，也能保证图和表来自同一批
产物。图中默认使用 KIR=50，避免把三个 KIR 混成一个不易解释的平均数。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import WorkspacePaths

PATHS = WorkspacePaths.discover(PROJECT_ROOT)
DEFAULT_ROOT = PATHS.artifact_root / "outputs" / "experiments" / "cluster_separability_v19"
DATASET_ORDER = ["clinc150", "banking77_oos", "stackoverflow"]
DATASET_LABELS = {
    "clinc150": "CLINC150",
    "banking77_oos": "BANKING77-OOS",
    "stackoverflow": "StackOverflow",
}
DISTANCE_LABELS = {"euclidean": "Euclidean", "mahalanobis_diag": "Diag. Mahalanobis"}


def _read(root: Path, relative: str) -> pd.DataFrame:
    """读取并检查绘图所需的列，尽早暴露不完整的产物。"""

    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing required aggregate: {path}")
    return pd.read_csv(path)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    """同时保存 PNG/PDF，并返回相对路径和 hash 供 manifest 审计。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    digest = hashlib.sha256(png.read_bytes()).hexdigest()
    return {"png": str(png), "pdf": str(pdf), "png_sha256": digest}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_k_sweep(root: Path, output_dir: Path) -> dict[str, str]:
    """画 K=1..5 的 fixed-boundary OOS F1，突出数据集交互。"""

    rows = _read(root, "kir_k_fixed_mean_std.csv")
    rows = rows[(rows["kir"] == 50) & rows["dataset"].isin(DATASET_ORDER)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = rows[rows["dataset"] == dataset]
        for distance, group in subset.groupby("distance", sort=False):
            group = group.sort_values("k_gate")
            ax.errorbar(
                group["k_gate"], group["test_oos_f1_mean"],
                yerr=group["test_oos_f1_std"], marker="o", capsize=3,
                label=DISTANCE_LABELS.get(distance, distance),
            )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Number of local centers K")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("OOS F1 (fixed boundary)")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Known-intent multi-cluster ablation (KIR=50)", y=1.02)
    return _save(fig, output_dir, "k_sweep_oos_f1")


def plot_baselines(root: Path, output_dir: Path) -> dict[str, str]:
    """比较统一 MiniLM 表征上的 Gate-only Baseline。"""

    rows = _read(root, "gate_baseline_summary.csv")
    rows = rows[(rows["kir"] == 50) & rows["dataset"].isin(DATASET_ORDER)]
    methods = list(dict.fromkeys(rows["method"].astype(str)))
    x = np.arange(len(DATASET_ORDER))
    width = 0.82 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for index, method in enumerate(methods):
        values = []
        errors = []
        for dataset in DATASET_ORDER:
            match = rows[(rows["dataset"] == dataset) & (rows["method"] == method)]
            values.append(float(match["test_oos_f1_mean"].iloc[0]) if len(match) else np.nan)
            errors.append(float(match["test_oos_f1_std"].iloc[0]) if len(match) else 0.0)
        ax.bar(x + (index - (len(methods) - 1) / 2) * width, values, width, yerr=errors,
               capsize=2, label=method)
    ax.set_xticks(x, [DATASET_LABELS[name] for name in DATASET_ORDER])
    ax.set_ylabel("OOS F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Controlled Gate-only baselines (KIR=50)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    return _save(fig, output_dir, "gate_baseline_oos_f1")


def plot_near_far(root: Path, output_dir: Path) -> dict[str, str]:
    """展示 near-OOS 是主要失败区，而不是只报告总体平均值。"""

    rows = _read(root, "analysis/near_far_oos_summary.csv")
    rows = rows[
        (rows["phase"] == "tuned") & (rows["kir"] == 50) &
        (rows["data_seed"] == 42) & (rows["oos_source"] == "combined")
    ]
    buckets = ["near", "medium", "far"]
    x = np.arange(len(DATASET_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for index, bucket in enumerate(buckets):
        values = []
        for dataset in DATASET_ORDER:
            match = rows[(rows["dataset"] == dataset) & (rows["bucket"] == bucket)]
            values.append(float(match["oos_f1"].iloc[0]) if len(match) else np.nan)
        ax.bar(x + (index - 1) * width, values, width, label=bucket)
    ax.set_xticks(x, [DATASET_LABELS[name] for name in DATASET_ORDER])
    ax.set_ylabel("OOS F1 (bucket + all Known test)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Known/OOS separability by OOS proximity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return _save(fig, output_dir, "near_medium_far_oos_f1")


def plot_fragmentation(root: Path, output_dir: Path) -> dict[str, str]:
    """把聚类碎片化与 OOS F1 放在同一张图，避免只看 K 的黑盒曲线。"""

    quality = _read(root, "cluster_quality_summary.csv")
    scores = _read(root, "kir_k_fixed_mean_std.csv")
    quality = quality[(quality["kir"] == 50) & (quality["k_gate"] > 1)]
    scores = scores[(scores["kir"] == 50) & (scores["k_gate"] > 1)]
    merged = quality.merge(scores, on=["phase", "dataset", "kir", "distance", "k_gate"], how="inner")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for dataset in DATASET_ORDER:
        group = merged[merged["dataset"] == dataset]
        ax.scatter(group["fragmented_intent_rate"], group["test_oos_f1_mean"],
                   s=45, alpha=0.85, label=DATASET_LABELS[dataset])
        for _, row in group.iterrows():
            ax.annotate(f"K{int(row['k_gate'])}", (row["fragmented_intent_rate"], row["test_oos_f1_mean"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Fragmented-intent rate")
    ax.set_ylabel("OOS F1 (fixed boundary)")
    ax.set_title("Cluster fragmentation and OOS separability (KIR=50)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _save(fig, output_dir, "fragmentation_vs_oos_f1")


def export_figures(root: Path, output_dir: Path) -> dict[str, object]:
    """导出四张图并写入一个可审计 manifest。"""

    _style()
    figures = {
        "k_sweep_oos_f1": plot_k_sweep(root, output_dir),
        "gate_baseline_oos_f1": plot_baselines(root, output_dir),
        "near_medium_far_oos_f1": plot_near_far(root, output_dir),
        "fragmentation_vs_oos_f1": plot_fragmentation(root, output_dir),
    }
    manifest = {"source_root": str(root), "kir": 50, "figures": figures}
    (output_dir / "cluster_separability_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export s2c cluster/OOS aggregate figures")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "figures" / "paper_v19"
    print(json.dumps(export_figures(root, output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
