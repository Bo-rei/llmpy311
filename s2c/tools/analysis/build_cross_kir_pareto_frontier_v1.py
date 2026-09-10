#!/usr/bin/env python3
"""Build a multi-metric Pareto view of the frozen fair Gate matrix.

The source contains completed five-seed runs only.  This script performs no
training or parameter selection; it reports descriptive mean/std frontiers for
OOS F1, F1-All, Known Recall and false acceptance.
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
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "analysis" / "cross_protocol_tradeoff_v1" / "per_seed.csv"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "cross_kir_pareto_frontier_v1"
DEFAULT_FIGURES = ROOT / "figures" / "cross_kir_pareto_frontier_v1"
DEFAULT_REPORT = ROOT / "docs" / "analysis" / "CROSS_KIR_PARETO_FRONTIER_V1.md"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
METHODS = (
    "trainable_k1",
    "single_centroid",
    "random_partition",
    "fixed_k2",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
    "mogb_minilm",
)
METHOD_LABELS = {
    "trainable_k1": "Trainable K=1",
    "single_centroid": "Frozen K=1",
    "random_partition": "Random K=2",
    "fixed_k2": "Frozen K=2",
    "mogb_partition_ours_boundary": "MOGB part. + S2C bound.",
    "ours_partition_mogb_boundary": "S2C part. + MOGB bound.",
    "mogb_minilm": "MOGB-MiniLM",
}
COLORS = {
    "trainable_k1": "#0072B2",
    "single_centroid": "#56B4E9",
    "random_partition": "#E69F00",
    "fixed_k2": "#D55E00",
    "mogb_partition_ours_boundary": "#009E73",
    "ours_partition_mogb_boundary": "#CC79A7",
    "mogb_minilm": "#6A3D9A",
}

_CJK_FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
if Path(_CJK_FONT).exists():
    font_manager.fontManager.addfont(_CJK_FONT)
    plt.rcParams["font.family"] = [font_manager.FontProperties(fname=_CJK_FONT).get_name()]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


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
    temporary.replace(path)


def validate(frame: pd.DataFrame) -> None:
    required = {"dataset", "kir", "seed", "method", "oos_f1", "f1_all", "known_recall", "false_accept_rate"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    expected = len(DATASETS) * len(KIRS) * len(SEEDS) * len(METHODS)
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, found {len(frame)}")
    key = ["dataset", "kir", "seed", "method"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate fair rows")
    if set(frame.dataset) != set(DATASETS) or set(frame.kir) != set(KIRS) or set(frame.seed) != set(SEEDS):
        raise ValueError("unexpected dataset/KIR/seed")
    if set(frame.method) != set(METHODS):
        raise ValueError("unexpected method set")
    metrics = ["oos_f1", "f1_all", "known_recall", "false_accept_rate"]
    if not np.isfinite(frame[metrics].to_numpy()).all():
        raise ValueError("non-finite fair metric")


def dominates(left: pd.Series, right: pd.Series, tolerance: float = 1e-12) -> bool:
    higher = ["oos_f1_mean", "f1_all_mean", "known_recall_mean"]
    lower = ["false_accept_rate_mean"]
    no_worse = all(left[column] >= right[column] - tolerance for column in higher) and all(
        left[column] <= right[column] + tolerance for column in lower
    )
    strictly = any(left[column] > right[column] + tolerance for column in higher) or any(
        left[column] < right[column] - tolerance for column in lower
    )
    return bool(no_worse and strictly)


def build(input_path: Path, output: Path, figures: Path, report: Path) -> None:
    source = pd.read_csv(input_path)
    validate(source)
    grouped = (
        source.groupby(["dataset", "kir", "method"], as_index=False)
        .agg(
            oos_f1_mean=("oos_f1", "mean"),
            oos_f1_std=("oos_f1", "std"),
            f1_all_mean=("f1_all", "mean"),
            f1_all_std=("f1_all", "std"),
            known_recall_mean=("known_recall", "mean"),
            known_recall_std=("known_recall", "std"),
            false_accept_rate_mean=("false_accept_rate", "mean"),
            false_accept_rate_std=("false_accept_rate", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["dataset", "kir", "method"])
        .reset_index(drop=True)
    )
    grouped["method_label"] = grouped.method.map(METHOD_LABELS)
    grouped["pareto_frontier"] = False
    grouped["dominated_by"] = ""
    for (dataset, kir), index in grouped.groupby(["dataset", "kir"]).groups.items():
        rows = grouped.loc[index]
        for row_index, row in rows.iterrows():
            dominators = [other.method for _, other in rows.iterrows() if other.method != row.method and dominates(other, row)]
            grouped.loc[row_index, "pareto_frontier"] = not dominators
            grouped.loc[row_index, "dominated_by"] = "|".join(dominators)

    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    atomic_csv(grouped, output / "pareto_points.csv")
    pareto_summary = (
        grouped[grouped.pareto_frontier]
        .groupby(["dataset", "kir"], as_index=False)
        .agg(pareto_methods=("method_label", lambda values: "|".join(values)), pareto_count=("method", "size"))
    )
    atomic_csv(pareto_summary, output / "pareto_summary.csv")
    trainable = grouped[grouped.method == "trainable_k1"].copy()
    trainable = trainable.merge(pareto_summary, on=["dataset", "kir"], how="left")
    trainable["trainable_on_pareto"] = trainable.pareto_frontier
    atomic_csv(trainable, output / "trainable_pareto_status.csv")

    manifest = {
        "experiment": "cross_kir_pareto_frontier_v1",
        "analysis_only": True,
        "source": str(input_path.resolve()),
        "source_sha256": sha256(input_path),
        "datasets": DATASETS,
        "kirs": KIRS,
        "seeds": SEEDS,
        "methods": METHODS,
        "objectives": {
            "maximize": ["oos_f1", "f1_all", "known_recall"],
            "minimize": ["false_accept_rate"],
        },
        "selection_used_test_labels": False,
        "interpretation": "descriptive mean/std Pareto diagnostic; not parameter selection",
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(len(DATASETS), len(KIRS), figsize=(16, 12), constrained_layout=True, sharex=True, sharey=True)
    for row_index, dataset in enumerate(DATASETS):
        for col_index, kir in enumerate(KIRS):
            ax = axes[row_index, col_index]
            cell = grouped[(grouped.dataset == dataset) & (grouped.kir == kir)]
            for method in METHODS:
                point = cell[cell.method == method].iloc[0]
                ax.errorbar(
                    point.f1_all_mean * 100,
                    point.oos_f1_mean * 100,
                    xerr=point.f1_all_std * 100,
                    yerr=point.oos_f1_std * 100,
                    fmt="o" if not point.pareto_frontier else "*",
                    markersize=9 if not point.pareto_frontier else 13,
                    capsize=2,
                    color=COLORS[method],
                    alpha=0.9,
                    label=METHOD_LABELS[method],
                )
            ax.set_title(f"{dataset} / KIR={kir:.2f}")
            ax.grid(alpha=0.25)
            if row_index == len(DATASETS) - 1:
                ax.set_xlabel("F1-All (%)")
            if col_index == 0:
                ax.set_ylabel("OOS F1 (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, fontsize=8, loc="best")
    fig.suptitle("当前同协议方法的多指标 Pareto 前沿（误差线为五 seed 标准差）", fontsize=16)
    fig.savefig(figures / "cross_kir_pareto_frontier.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # A compact count plot makes the stability claim auditable without reading all panels.
    counts = grouped.groupby("method", as_index=False).agg(
        pareto_cells=("pareto_frontier", "sum"),
        total_cells=("pareto_frontier", "size"),
    )
    counts["pareto_rate"] = counts.pareto_cells / counts.total_cells
    counts["method_label"] = counts.method.map(METHOD_LABELS)
    atomic_csv(counts.sort_values("pareto_cells", ascending=False), output / "pareto_cell_counts.csv")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    order = counts.sort_values("pareto_cells", ascending=True)
    ax.barh(order.method_label, order.pareto_cells, color=[COLORS[m] for m in order.method])
    for position, value in enumerate(order.pareto_cells):
        ax.text(value + 0.05, position, f"{int(value)}/9", va="center", fontsize=9)
    ax.set_xlim(0, 9.8)
    ax.set_xlabel("Pareto cells (out of 9)")
    ax.set_title("多指标 Pareto 前沿覆盖率")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(figures / "pareto_cell_counts.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# 跨 KIR 多指标 Pareto 前沿 V1",
        "",
        "本分析读取当前 `protocol_v2_textoir_v1` 已完成的 315 个 fair Gate 行（7 个方法×3 数据集×3 KIR×5 seed），只做五 seed 均值/标准差和后验多指标归因，不重训、不调参、不使用测试标签选择方法。",
        "",
        f"输入：`{input_path}`；SHA256：`{sha256(input_path)}`。Pareto 目标为最大化 OOS F1、F1-All、Known Recall，同时最小化 false acceptance。",
        "",
        "## 解读",
        "",
        "- 星形点表示在同一 dataset×KIR 单元的四指标均值上未被其他 fair 方法同时支配；它不是统计显著性结论。",
        "- 误差线表示五个正式 seed 的标准差，用于观察稳定性，不用于选择测试最优点。",
        "- Pareto 前沿可以区分“单项 OOS F1 高”与“Known 分类、OOS 拒识和误接收同时平衡”。",
        "",
        "## Trainable-K1 状态",
        "",
    ]
    trainable_status = trainable.sort_values(["dataset", "kir"])
    lines.append("| 数据集 | KIR | Trainable 是否在 Pareto 前沿 | Pareto 方法 |")
    lines.append("|---|---:|---|---|")
    for row in trainable_status.itertuples(index=False):
        methods = str(row.pareto_methods).replace("|", ", ") if pd.notna(row.pareto_methods) else ""
        lines.append(f"| {row.dataset} | {row.kir:.2f} | {'是' if row.trainable_on_pareto else '否'} | {methods} |")
    lines.extend(
        [
            "",
            "## 输出",
            "",
            "- `results/analysis/archive/analysis/cross_kir_pareto_frontier_v1/pareto_points.csv`：每个方法的均值、标准差和 Pareto 标记。",
            "- `results/analysis/archive/analysis/cross_kir_pareto_frontier_v1/pareto_summary.csv`：每个 dataset×KIR 的前沿方法。",
            "- `figures/archive/analysis/cross_kir_pareto_frontier_v1/cross_kir_pareto_frontier.png`：9 个工作点的多指标前沿图。",
            "- `figures/archive/analysis/cross_kir_pareto_frontier_v1/pareto_cell_counts.png`：方法进入 Pareto 前沿的工作点数量。",
            "",
            "## 边界",
            "",
            "该分析只说明当前统一 MiniLM Known-only Gate 合同下的多指标工作点关系；MOGB 官方 BERT、ADB、DA-ADB 和 DCLOOS 的外部监督/骨干合同不在此图中，不据此宣称跨方法 SOTA。",
        ]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    build(args.input.resolve(), args.output.resolve(), args.figures.resolve(), args.report.resolve())
    print(args.report.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
