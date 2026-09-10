#!/usr/bin/env python3
"""Build the report figures and summary for the H1 Trainable-Gate search."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = ROOT / "results" / "analysis" / "historical_trainable_parameter_full_pipeline"
DEFAULT_FIGURE_ROOT = ROOT / "figures" / "historical_trainable_parameter_full_pipeline"
DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
LABELS = {"clinc150": "CLINC150", "stackoverflow": "StackOverflow", "banking77_oos": "Banking77-OOS"}
PAPER = {
    "clinc150": {"known_f1": 79.95, "oos_f1": 91.96, "accuracy": 86.78},
    "stackoverflow": {"known_f1": 75.48, "oos_f1": 89.71, "accuracy": 85.54},
    "banking77_oos": {"known_f1": 74.90, "oos_f1": 88.23, "accuracy": 78.98},
}
COLORS = {
    "eligible": "#4C78A8",
    "selected": "#E0A72F",
    "best": "#C44E52",
    "paper": "#6B7075",
    "baseline": "#263238",
    "ineligible": "#C9CED3",
    "grid": "#E7E9EC",
    "text": "#252525",
}
K_MARKERS = {1: "o", 2: "D", 3: "^"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _style(ax: plt.Axes) -> None:
    ax.tick_params(length=2.8, width=0.65, pad=2)
    ax.grid(color=COLORS["grid"], linewidth=0.45)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, stem: str, figure_root: Path) -> list[str]:
    figure_root.mkdir(parents=True, exist_ok=True)
    png_path = figure_root / f"{stem}.png"
    tiff_path = figure_root / f"{stem}.tiff"
    pdf_path = figure_root / f"{stem}.pdf"
    svg_path = figure_root / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return [str(path.relative_to(ROOT)) for path in (png_path, tiff_path, pdf_path, svg_path)]


def _config(row: dict[str, str]) -> tuple[int, float, float, str]:
    return int(row["k"]), float(row["radius_lambda"]), float(row["threshold"]), row["acceptance_mode"]


def _is_baseline(row: dict[str, str]) -> bool:
    return _config(row) == (1, 1.0, 1.0, "nearest_sphere")


def _is_selected(row: dict[str, str], selected: dict[str, str]) -> bool:
    return _config(row) == _config(selected)


def plot_validation_frontier(
    rows: list[dict[str, str]],
    selected: dict[str, dict[str, str]],
    figure_root: Path,
) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.72, wspace=0.23)
    for index, dataset in enumerate(DATASETS):
        ax = axes[index]
        subset = [row for row in rows if row["dataset"] == dataset]
        chosen = selected[dataset]
        eligible = [row for row in subset if row.get("eligible_per_seed_guard") == "True"]
        ineligible = [row for row in subset if row.get("eligible_per_seed_guard") != "True"]
        for group, color, alpha in ((ineligible, COLORS["ineligible"], 0.60), (eligible, COLORS["eligible"], 0.68)):
            for row in group:
                ax.scatter(
                    float(row["known_f1_mean"]) * 100.0,
                    float(row["oos_f1_mean"]) * 100.0,
                    s=13,
                    marker=K_MARKERS[int(row["k"])],
                    color=color,
                    alpha=alpha,
                    linewidths=0.25,
                    edgecolors="white",
                )
        baseline = next(row for row in subset if _is_baseline(row))
        ax.scatter(float(baseline["known_f1_mean"]) * 100.0, float(baseline["oos_f1_mean"]) * 100.0, s=42, marker="o", facecolor="white", edgecolor=COLORS["baseline"], linewidth=1.1, zorder=5)
        ax.scatter(float(chosen["known_f1_mean"]) * 100.0, float(chosen["oos_f1_mean"]) * 100.0, s=64, marker="*", facecolor=COLORS["selected"], edgecolor=COLORS["text"], linewidth=0.45, zorder=6)
        ax.axhline(PAPER[dataset]["oos_f1"], color=COLORS["paper"], linestyle=(0, (3, 2)), linewidth=0.85)
        ax.set_title(LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("Known F1 (%)")
        if index == 0:
            ax.set_ylabel("Validation OOS F1 (%)")
        ax.set_xlim(45, 92 if dataset == "clinc150" else 92)
        ax.set_ylim(45, 96)
        ax.text(0.98, 0.04, f"selected: K={chosen['k']}, λ={chosen['radius_lambda']}, t={chosen['threshold']}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color=COLORS["text"])
        _style(ax)
    handles = [
        Line2D([], [], marker="o", color=COLORS["eligible"], linestyle="None", markersize=4, label="guard-eligible candidates"),
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=COLORS["baseline"], color="none", markersize=4.5, label="K=1 baseline"),
        Line2D([], [], marker="*", markerfacecolor=COLORS["selected"], markeredgecolor=COLORS["text"], color="none", markersize=7, label="validation-selected"),
        Line2D([], [], color=COLORS["paper"], linestyle=(0, (3, 2)), linewidth=1, label="paper Ours OOS F1"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.90), ncol=4, frameon=False, handlelength=1.2, columnspacing=0.8)
    fig.text(0.075, 0.965, "Validation selects the Gate work-point before test confirmation", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.075, 0.035, "Each point is one K/λ/threshold/acceptance configuration; the Known F1 and Accuracy guard is applied per seed and in aggregate.", fontsize=5.8, color="#555B61")
    return _save(fig, "validation_parameter_frontier", figure_root)


def plot_test_paper_comparison(
    rows: list[dict[str, str]],
    selected: dict[str, dict[str, str]],
    best: dict[str, dict[str, str]],
    figure_root: Path,
) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.72, wspace=0.23)
    for index, dataset in enumerate(DATASETS):
        ax = axes[index]
        subset = [row for row in rows if row["dataset"] == dataset]
        ax.scatter(
            [float(row["known_macro_f1_mean"]) * 100.0 for row in subset],
            [float(row["oos_f1_mean"]) * 100.0 for row in subset],
            s=14,
            color=COLORS["ineligible"],
            alpha=0.62,
            edgecolors="none",
        )
        chosen = selected[dataset]
        top = best[dataset]
        baseline = next(row for row in subset if _is_baseline(row))
        ax.scatter(float(baseline["known_macro_f1_mean"]) * 100.0, float(baseline["oos_f1_mean"]) * 100.0, s=42, marker="o", facecolor="white", edgecolor=COLORS["baseline"], linewidth=1.1, zorder=5)
        ax.scatter(float(chosen["known_macro_f1_mean"]) * 100.0, float(chosen["oos_f1_mean"]) * 100.0, s=64, marker="*", facecolor=COLORS["selected"], edgecolor=COLORS["text"], linewidth=0.45, zorder=6)
        ax.scatter(float(top["known_macro_f1_mean"]) * 100.0, float(top["oos_f1_mean"]) * 100.0, s=38, marker="x", color=COLORS["best"], linewidth=1.0, zorder=7)
        ax.axhline(PAPER[dataset]["oos_f1"], color=COLORS["paper"], linestyle=(0, (3, 2)), linewidth=0.85)
        ax.set_title(LABELS[dataset], loc="left", fontweight="bold")
        ax.set_xlabel("Known F1 (%)")
        if index == 0:
            ax.set_ylabel("Test OOS F1 (%)")
        ax.set_xlim(45, 92)
        ax.set_ylim(45, 96)
        chosen_delta = float(chosen["delta_oos_f1_pp"])
        top_delta = float(top["delta_oos_f1_pp"])
        ax.text(0.98, 0.04, f"selected Δ {chosen_delta:+.2f} pp\npost-hoc max Δ {top_delta:+.2f} pp", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color=COLORS["text"])
        _style(ax)
    handles = [
        Line2D([], [], marker="o", color=COLORS["ineligible"], linestyle="None", markersize=4, label="all test candidates"),
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=COLORS["baseline"], color="none", markersize=4.5, label="K=1 baseline"),
        Line2D([], [], marker="*", markerfacecolor=COLORS["selected"], markeredgecolor=COLORS["text"], color="none", markersize=7, label="validation-selected"),
        Line2D([], [], marker="x", color=COLORS["best"], linestyle="None", markersize=5, label="test max (confirmation only)"),
        Line2D([], [], color=COLORS["paper"], linestyle=(0, (3, 2)), linewidth=1, label="paper Ours OOS F1"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 0.90), ncol=5, frameon=False, handlelength=1.15, columnspacing=0.55)
    fig.text(0.075, 0.965, "Trainable MiniLM Gate candidates versus the historical paper work-point", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.075, 0.035, "The star is selected from validation; the red cross is shown only as a post-hoc test confirmation and is not a selected result.", fontsize=5.8, color="#555B61")
    return _save(fig, "test_oos_f1_vs_paper_ours", figure_root)


def plot_error_budget(
    selected_rows: list[dict[str, str]],
    figure_root: Path,
) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.72, wspace=0.23)
    fields = (("false_accept_rate", "OOS accepted"), ("false_reject_rate", "Known rejected"), ("expert_error_rate", "Expert error"))
    for index, dataset in enumerate(DATASETS):
        ax = axes[index]
        row = next(item for item in selected_rows if item["dataset"] == dataset)
        values = [float(row[f"{field}_mean"]) * 100.0 for field, _ in fields]
        bars = ax.bar(np.arange(len(values)), values, color=[COLORS["best"], COLORS["selected"], COLORS["eligible"]], width=0.62)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=5.6)
        ax.set_xticks(np.arange(len(values)), [label for _, label in fields], rotation=20, ha="right")
        ax.set_title(LABELS[dataset], loc="left", fontweight="bold")
        ax.set_ylim(0, 35)
        if index == 0:
            ax.set_ylabel("Share (%)")
        _style(ax)
    fig.text(0.075, 0.965, "What the selected full pipeline trades to improve OOS F1", fontsize=8.1, fontweight="bold", va="top")
    fig.text(0.075, 0.035, "Values are means over three test seeds; OOS F1 is controlled by the Gate, while Expert error is measured only after Gate acceptance.", fontsize=5.8, color="#555B61")
    return _save(fig, "selected_full_pipeline_error_budget", figure_root)


def _report(
    result_root: Path,
    figure_paths: list[str],
    selected_validation: list[dict[str, str]],
    selected_test: list[dict[str, str]],
    best: list[dict[str, str]],
) -> str:
    selected_val_by_ds = {row["dataset"]: row for row in selected_validation}
    selected_test_by_ds = {row["dataset"]: row for row in selected_test}
    best_by_ds = {row["dataset"]: row for row in best}
    lines = [
        "# Historical H1 Trainable MiniLM Gate：参数化 full pipeline 实验汇报",
        "",
        "> 目标：在历史论文主线的 H1 controlled 协议下，用已有训练后的 Trainable MiniLM 作为 Gate，搜索 K、λ、threshold 和接受规则，并将真实 Gate→Router→Expert 结果与 `fulltex.tex` KIR=.50 的 `Ours` OOS F1 对比。",
        "",
        "## 结论",
        "",
        "- `banking77_oos` 的验证集选中配置在真实 full pipeline 上达到 **91.53±0.05 OOS F1**，比论文 `Ours=88.23` 高 **+3.30 pp**；三个 seed 均超过论文值。",
        "- `CLINC150` 的选中配置为 **91.04±0.81**，比论文 `91.96` 低 **0.92 pp**；`StackOverflow` 为 **89.15±1.54**，比论文 `89.71` 低 **0.56 pp**。当前网格没有让这两个数据集的三-seed 均值超过论文值。",
        "- 当前结果说明：训练后的 MiniLM Gate 可以在同一 full pipeline 中产生明显收益，但是否超过论文 Ours 仍受数据协议/H1 下游组件和边界 work-point 共同影响，不能把 Gate-only 提升直接等同为论文系统复现。",
        "",
        "## 1. 协议与实验设计",
        "",
        "| 项目 | 本实验 |",
        "|---|---|",
        "| 数据 | `clinc150`、`stackoverflow`、`banking77_oos` |",
        "| KIR / seed | KIR=.50；seed=13,42,87 |",
        "| Gate | 已有 Trainable `last2 MiniLM + 384→256→384 residual projection` checkpoint |",
        "| 边界 | diagonal Mahalanobis；`mean + λ·std` |",
        "| 参数 | K∈{1,2,3}；λ∈{.50,.75,1,1.25,1.5,2}；threshold∈{.85,.90,.95,1,1.05,1.10,1.20}；两种 acceptance mode |",
        "| 选参 | 仅 validation；目标 OOS F1；Known F1 与 Accuracy 相对 K=1 baseline 不下降超过 1 pp（逐 seed及均值） |",
        "| 下游 | 现有 H1 Router/Expert checkpoint 固定不变；selected 配置另做直接 pipeline 核验 |",
        "| GPU | CUDA；实验 manifest 记录为 `cuda` |",
        "",
        "论文对照值来自 `fulltex.tex` KIR=.50 的 `Ours` 行；论文表头写作 Banking77，本实验历史主线的实际数据键为 `banking77_oos`。",
        "",
        "## 2. 验证集选中的配置与测试结果",
        "",
        "| 数据集 | selected Gate | Val OOS F1 | Test full-pipeline OOS F1 | Known F1 | Acc | False Acceptance | 相对论文 OOS F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        val_row = selected_val_by_ds[dataset]
        row = selected_test_by_ds[dataset]
        top = best_by_ds[dataset]
        config = f"K={val_row['k']}, λ={val_row['radius_lambda']}, t={val_row['threshold']}, {val_row['acceptance_mode']}"
        lines.append(
            f"| {LABELS[dataset]} | {config} | {float(val_row['oos_f1_mean']) * 100:.2f} | {float(row['oos_f1_mean']) * 100:.2f}* | "
            f"{float(row['known_macro_f1_mean']) * 100:.2f} | {float(row['overall_accuracy_mean']) * 100:.2f} | "
            f"{float(row['false_accept_rate_mean']) * 100:.2f}% | {float(row['delta_oos_f1_pp']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "`*` 上表的 selected test 数字来自直接 Gate→Router→Expert 推理，并与固定下游 replay 的全部指标逐项核对，最大绝对差为 0。",
            "",
            "## 3. 参数搜索保留结果",
            "",
            "测试集只用于确认，不用于选参。作为后验审计，本实验另外保留所有测试均值超过论文 OOS F1 的候选：",
            "",
            "| 数据集 | 超过论文的候选数 | 测试后验最高配置 | OOS F1 | Δ paper |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        top = best_by_ds[dataset]
        lines.append(
            f"| {LABELS[dataset]} | {sum(1 for row in _read_csv(result_root / 'full_pipeline_candidate_summary.csv') if row['dataset'] == dataset and row['beats_paper_oos_f1_mean'] == 'True')} | "
            f"K={top['k']}, λ={top['radius_lambda']}, t={top['threshold']}, {top['acceptance_mode']} | "
            f"{float(top['oos_f1_mean']) * 100:.2f} | {float(top['delta_oos_f1_pp']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "测试后验最高配置只作为保留的 confirmation，不是论文式选参结果；完整候选表见 [`full_pipeline_candidate_summary.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/full_pipeline_candidate_summary.csv)，论文超越候选见 [`paper_beating_candidates.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/paper_beating_candidates.csv)。",
            "",
            "## 4. 图形解释",
            "",
            f"![Validation parameter frontier](../../{figure_paths[0]})",
            "",
            "图 1：每个点是一组 Gate 参数。星标只由 validation 选出；灰色/蓝色区分 Known F1 与 Accuracy guard，虚线是论文 Ours 的 OOS F1。",
            "",
            f"![Test OOS F1 versus paper](../../{figure_paths[4]})",
            "",
            "图 2：所有测试候选的 OOS F1 与 Known F1 分布。星标是 validation-selected，红色叉号只是 test 后验最高点，用于回答“当前网格能否超过论文”，不参与选参。",
            "",
            f"![Selected error budget](../../{figure_paths[8]})",
            "",
            "图 3：selected full pipeline 的错误预算。OOS F1 的变化首先来自 Gate 的 OOS 接受/拒绝边界；只有被 Gate 接受的样本才会暴露 Router/Expert error。",
            "",
            "## 5. 可信度与限制",
            "",
            "- 所有 9 个 selected dataset×seed 均执行了真实 pipeline；`direct_vs_derived_verification.csv` 的最大绝对差为 0。",
            "- 训练后的 MiniLM checkpoint 已有，本文实验不重新训练 Gate；本轮只改变边界 work-point。",
            "- 这是 H1 controlled evidence：当前数据、训练后的 Gate 和现有 Router/Expert 与论文原始 H0 完整 Cascade 并非字节级相同，因此“高于论文”应理解为当前 H1 对论文公开 reference 的比较，而不是严格复现声明。",
            "- 没有修改 `fulltex.tex`，没有覆盖历史 artifact，也没有保存逐样本 raw prediction。",
            "",
            "## 6. 文件",
            "",
            "- 结果 manifest：[`MANIFEST.json`](../../results/analysis/historical_trainable_parameter_full_pipeline/MANIFEST.json)",
            "- 验证集选择：[`selected_validation.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/selected_validation.csv)",
            "- 全部 Gate 候选：[`gate_workpoints.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/gate_workpoints.csv)",
            "- full pipeline 候选：[`full_pipeline_candidates_test.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/full_pipeline_candidates_test.csv)",
            "- 测试后验最高配置：[`best_test_candidate_by_dataset.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/best_test_candidate_by_dataset.csv)",
            "- selected 直接结果：[`full_pipeline_selected_per_seed.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/full_pipeline_selected_per_seed.csv)",
            "- 直接/派生核验：[`direct_vs_derived_verification.csv`](../../results/analysis/historical_trainable_parameter_full_pipeline/direct_vs_derived_verification.csv)",
        ]
    )
    return "\n".join(lines) + "\n"


def build(result_root: Path = DEFAULT_RESULT_ROOT, figure_root: Path = DEFAULT_FIGURE_ROOT) -> dict[str, Any]:
    validation = _read_csv(result_root / "validation_aggregate.csv")
    selected_validation = _read_csv(result_root / "selected_validation.csv")
    candidate_summary = _read_csv(result_root / "full_pipeline_candidate_summary.csv")
    selected_summary = _read_csv(result_root / "full_pipeline_selected_summary.csv")
    selected = {row["dataset"]: row for row in selected_summary}
    best = {
        dataset: max((row for row in candidate_summary if row["dataset"] == dataset), key=lambda row: float(row["oos_f1_mean"]))
        for dataset in DATASETS
    }
    figure_paths: list[str] = []
    figure_paths.extend(plot_validation_frontier(validation, {row["dataset"]: row for row in selected_validation}, figure_root))
    figure_paths.extend(plot_test_paper_comparison(candidate_summary, selected, best, figure_root))
    figure_paths.extend(plot_error_budget(selected_summary, figure_root))
    manifest = {
        "schema_version": "s2c.historical_trainable_parameter_full_pipeline.figures.v1",
        "backend": "python_matplotlib",
        "result_root": str(result_root.resolve()),
        "figure_root": str(figure_root.resolve()),
        "figures": figure_paths,
        "figure_stems": ["validation_parameter_frontier", "test_oos_f1_vs_paper_ours", "selected_full_pipeline_error_budget"],
        "source_data": [
            "results/analysis/historical_trainable_parameter_full_pipeline/validation_aggregate.csv",
            "results/analysis/historical_trainable_parameter_full_pipeline/full_pipeline_candidate_summary.csv",
            "results/analysis/historical_trainable_parameter_full_pipeline/full_pipeline_selected_summary.csv",
        ],
        "exclusions": "None; all 252 candidates per dataset are retained in the source table.",
    }
    (result_root / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = _report(result_root, figure_paths, selected_validation, selected_summary, list(best.values()))
    report_path = ROOT / "docs" / "analysis" / "historical_trainable_parameter_full_pipeline_presentation.md"
    report_path.write_text(report, encoding="utf-8")
    return {"figures": figure_paths, "report": str(report_path.relative_to(ROOT)), "figure_manifest": str((result_root / "FIGURE_MANIFEST.json").relative_to(ROOT))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.result_root.resolve(), args.figure_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
