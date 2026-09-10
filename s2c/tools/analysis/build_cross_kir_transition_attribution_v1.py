#!/usr/bin/env python3
"""Summarise Trainable-K1 versus MOGB-Fair error-budget transitions across KIR.

This is an analysis-only companion to the frozen 45-cell transition audit.  It
does not retrain models, choose a threshold, or use test labels to select a
method.  It converts the already audited per-cell decomposition into a compact
cross-KIR view of coverage recovery and OOS trade-offs.
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

_CJK_FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
if Path(_CJK_FONT).exists():
    font_manager.fontManager.addfont(_CJK_FONT)
    plt.rcParams["font.family"] = [font_manager.FontProperties(fname=_CJK_FONT).get_name()]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "analysis" / "trainable_mogb_open_intent_transitions_v1" / "cell_decomposition_summary.csv"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "cross_kir_transition_attribution_v1"
DEFAULT_FIGURES = ROOT / "figures" / "cross_kir_transition_attribution_v1"
DEFAULT_REPORT = ROOT / "docs" / "analysis" / "CROSS_KIR_TRANSITION_ATTRIBUTION_V1.md"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)


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
    required = {
        "dataset",
        "kir",
        "n_seeds",
        "f1_all_delta_mean",
        "net_known_correct_gain_rate",
        "net_oos_correct_gain_rate",
        "known_trainable_recovers_mogb_reject_mean",
        "known_trainable_fixes_mogb_wrong_mean",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if len(frame) != len(DATASETS) * len(KIRS):
        raise ValueError(f"expected {len(DATASETS) * len(KIRS)} rows, found {len(frame)}")
    if frame.duplicated(["dataset", "kir"]).any():
        raise ValueError("duplicate dataset/KIR rows")
    if set(frame.dataset) != set(DATASETS) or set(frame.kir) != set(KIRS):
        raise ValueError("unexpected dataset or KIR")
    if not (frame.n_seeds == 5).all():
        raise ValueError("transition summary is not five-seed")
    numeric = [
        "f1_all_delta_mean",
        "net_known_correct_gain_rate",
        "net_oos_correct_gain_rate",
        "known_trainable_recovers_mogb_reject_mean",
        "known_trainable_fixes_mogb_wrong_mean",
    ]
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("non-finite transition metric")


def build(input_path: Path, output: Path, figures: Path, report: Path) -> None:
    source = pd.read_csv(input_path)
    validate(source)
    source = source.sort_values(["dataset", "kir"]).reset_index(drop=True)
    data = source.copy()
    data["kir_label"] = data["kir"].map(lambda value: f"{value:.2f}")
    data["f1_all_delta_pp"] = data["f1_all_delta_mean"] * 100.0
    data["net_known_gain_pp"] = data["net_known_correct_gain_rate"] * 100.0
    data["net_oos_gain_pp"] = data["net_oos_correct_gain_rate"] * 100.0
    data["known_recovery_rate_pp"] = (
        data["known_trainable_recovers_mogb_reject_mean"] / data["n_known_mean"] * 100.0
    )
    data["known_wrong_fix_rate_pp"] = (
        data["known_trainable_fixes_mogb_wrong_mean"] / data["n_known_mean"] * 100.0
    )

    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    atomic_csv(data, output / "cross_kir_transition_summary.csv")

    manifest = {
        "experiment": "cross_kir_transition_attribution_v1",
        "analysis_only": True,
        "source": str(input_path.resolve()),
        "source_sha256": sha256(input_path),
        "datasets": DATASETS,
        "kirs": KIRS,
        "n_cells": int(len(data)),
        "n_seeds_per_cell": 5,
        "test_labels_used_for_posthoc_attribution_only": True,
        "selection_used_test_labels": False,
        "interpretation": "coverage/error-budget attribution, not method selection",
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = [
        ("f1_all_delta_pp", "F1-All 差值（pp）", "RdYlGn"),
        ("net_known_gain_pp", "Known 正确净增（pp）", "RdYlGn"),
        ("net_oos_gain_pp", "OOS 正确净增（pp）", "RdYlGn"),
        ("known_recovery_rate_pp", "恢复 MOGB 拒绝 Known（%）", "Blues"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    for ax, (column, title, cmap) in zip(axes.flat, metrics):
        matrix = data.pivot(index="dataset", columns="kir", values=column).reindex(DATASETS, columns=KIRS)
        values = matrix.to_numpy(dtype=float)
        limit = max(1.0, float(np.nanmax(np.abs(values))))
        if column == "known_recovery_rate_pp":
            image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=limit, aspect="auto")
        else:
            image = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(KIRS)), [f"{kir:.2f}" for kir in KIRS])
        ax.set_yticks(range(len(DATASETS)), DATASETS)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix.iloc[row, col]:.1f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.suptitle("Trainable-K1 相对 MOGB-Fair 的跨 KIR 错误预算归因", fontsize=15)
    fig.savefig(figures / "cross_kir_transition_attribution.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    line_frame = data.melt(
        id_vars=["dataset", "kir"],
        value_vars=["net_known_gain_pp", "net_oos_gain_pp"],
        var_name="budget_component",
        value_name="delta_pp",
    )
    labels = {"net_known_gain_pp": "Known 正确净增", "net_oos_gain_pp": "OOS 正确净增"}
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 4.6), constrained_layout=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        cell = line_frame[line_frame.dataset == dataset]
        for component in ("net_known_gain_pp", "net_oos_gain_pp"):
            line = cell[cell.budget_component == component].sort_values("kir")
            ax.plot(line.kir, line.delta_pp, marker="o", linewidth=2, label=labels[component])
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.set_title(dataset)
        ax.set_xticks(KIRS, [f"{kir:.2f}" for kir in KIRS])
        ax.set_xlabel("KIR")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Trainable - MOGB correct-count delta (pp)")
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle("性能差距的来源：Known 覆盖恢复 vs OOS 正确拒识", fontsize=14)
    fig.savefig(figures / "known_oos_budget_cross_kir.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    mean = data.groupby("dataset", as_index=False)[
        ["f1_all_delta_pp", "net_known_gain_pp", "net_oos_gain_pp", "known_recovery_rate_pp"]
    ].mean()
    lines = [
        "# 跨 KIR 错误预算归因 V1",
        "",
        "本分析只读取已经完成的 Trainable-K1 与 MOGB-Fair 五 seed 逐样本转移审计。它不重训模型、不选择阈值、不改变 KIR，也不把测试标签用于方法选择；测试标签仅用于事后统计错误来源。",
        "",
        f"输入：`{input_path}`；SHA256：`{sha256(input_path)}`。覆盖 3 个数据集、3 个 KIR、9 个 dataset×KIR 单元，每单元 5 个 seed。",
        "",
        "## 结论",
        "",
        "- Trainable-K1 的 F1-All 优势主要来自恢复 MOGB 判为 Known rejected 的正确 Known，而不是增加 OOS 正确拒识。",
        "- 随 KIR 增大，Known 覆盖恢复规模扩大；StackOverflow 的恢复量最大，说明 MOGB-Fair 的平均半径在更高 Known 比例下更保守。",
        "- Trainable 的 OOS 正确净增通常为负，说明它不是通过更激进地拒绝 OOS 获得优势；优势来自更好的 Known 覆盖—边界排序平衡。",
        "- 该结论只适用于当前 MiniLM Known-only 的 MOGB-Fair 组件合同，不能写成超过完整 BERT MOGB 或论文 SOTA。",
        "",
        "## 数据集均值（跨三个 KIR）",
        "",
        "| 数据集 | F1-All 差值（pp） | Known 正确净增（pp） | OOS 正确净增（pp） | 恢复 MOGB 拒绝 Known（%） |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in mean.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.f1_all_delta_pp:.2f} | {row.net_known_gain_pp:.2f} | "
            f"{row.net_oos_gain_pp:.2f} | {row.known_recovery_rate_pp:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            "- `cross_kir_transition_summary.csv`：9 个 dataset×KIR 单元的错误预算指标。",
            "- `figures/archive/analysis/cross_kir_transition_attribution_v1/cross_kir_transition_attribution.png`：四项跨 KIR 热图。",
            "- `figures/archive/analysis/cross_kir_transition_attribution_v1/known_oos_budget_cross_kir.png`：Known/OOS 正确预算折线图。",
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
