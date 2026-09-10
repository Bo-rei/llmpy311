#!/usr/bin/env python3
"""Build a cross-KIR, matched-known-coverage S2C/MOGB diagnostic.

This is analysis-only.  It consumes the frozen operating-curve table created
by ``build_s2c_mogb_operating_curve_attribution_v1.py``.  The threshold at a
target Known Recall is derived from Known scores and the resulting OOS metrics
are evaluated as a post-hoc diagnostic.  No model, threshold, or K is selected
from these rows.
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "analysis" / "s2c_mogb_operating_curve_attribution_v1" / "matched_known_recall.csv"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "cross_kir_matched_frontier_v1"
DEFAULT_FIGURES = ROOT / "figures" / "cross_kir_matched_frontier_v1"
DEFAULT_REPORT = ROOT / "docs" / "analysis" / "CROSS_KIR_MATCHED_FRONTIER_V1.md"

DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
TARGETS = (0.80, 0.90, 0.95)
METHODS = ("s2c_trainable_k1", "mogb_minilm_fair")
LABELS = {
    "s2c_trainable_k1": "S2C Trainable K=1",
    "mogb_minilm_fair": "MOGB-MiniLM-Fair",
}
COLORS = {"s2c_trainable_k1": "#0072B2", "mogb_minilm_fair": "#009E73"}


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
        "method",
        "dataset",
        "kir",
        "seed",
        "target_known_recall",
        "oos_f1",
        "known_recall",
        "false_accept_rate",
        "threshold",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    subset = ["method", "dataset", "kir", "seed", "target_known_recall"]
    if frame.duplicated(subset).any():
        raise ValueError("duplicate operating-point rows")
    expected = len(DATASETS) * len(KIRS) * 5 * len(TARGETS) * len(METHODS)
    actual = int(frame[frame.target_known_recall.isin(TARGETS)].shape[0])
    if actual != expected:
        raise ValueError(f"expected {expected} target rows, found {actual}")
    if set(frame.method.unique()) != set(METHODS):
        raise ValueError("unexpected methods")
    if not np.isfinite(frame[["oos_f1", "known_recall", "false_accept_rate", "threshold"]].to_numpy()).all():
        raise ValueError("non-finite operating-point value")


def build(input_path: Path, output: Path, figures: Path, report: Path) -> None:
    source = pd.read_csv(input_path)
    validate(source)
    source = source[source.target_known_recall.isin(TARGETS)].copy()
    source["method_label"] = source["method"].map(LABELS)
    source["kir_label"] = source["kir"].map(lambda value: f"{value:.2f}")

    key = ["dataset", "kir", "seed", "target_known_recall"]
    pivot = source.pivot(index=key, columns="method", values=["oos_f1", "known_recall", "false_accept_rate", "threshold"]).reset_index()
    pivot.columns = ["_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col) for col in pivot.columns]
    pivot["delta_oos_f1_s2c_minus_mogb"] = pivot["oos_f1_s2c_trainable_k1"] - pivot["oos_f1_mogb_minilm_fair"]
    pivot["delta_known_recall_s2c_minus_mogb"] = pivot["known_recall_s2c_trainable_k1"] - pivot["known_recall_mogb_minilm_fair"]
    pivot["delta_fa_s2c_minus_mogb"] = pivot["false_accept_rate_s2c_trainable_k1"] - pivot["false_accept_rate_mogb_minilm_fair"]
    pivot["delta_threshold_s2c_minus_mogb"] = pivot["threshold_s2c_trainable_k1"] - pivot["threshold_mogb_minilm_fair"]
    pivot = pivot.sort_values(key).reset_index(drop=True)

    summary = (
        pivot.groupby(["dataset", "kir", "target_known_recall"], as_index=False)
        .agg(
            mean_delta_oos_f1=("delta_oos_f1_s2c_minus_mogb", "mean"),
            std_delta_oos_f1=("delta_oos_f1_s2c_minus_mogb", "std"),
            mean_delta_known_recall=("delta_known_recall_s2c_minus_mogb", "mean"),
            mean_delta_false_acceptance=("delta_fa_s2c_minus_mogb", "mean"),
            wins=("delta_oos_f1_s2c_minus_mogb", lambda values: int((values > 0).sum())),
            ties=("delta_oos_f1_s2c_minus_mogb", lambda values: int(np.isclose(values, 0.0, atol=1e-12).sum())),
            losses=("delta_oos_f1_s2c_minus_mogb", lambda values: int((values < 0).sum())),
            n_seeds=("delta_oos_f1_s2c_minus_mogb", "size"),
        )
    )

    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    atomic_csv(source.sort_values(key + ["method"]), output / "operating_points.csv")
    atomic_csv(pivot, output / "paired_frontier.csv")
    atomic_csv(summary, output / "summary.csv")
    manifest = {
        "experiment": "cross_kir_matched_frontier_v1",
        "analysis_only": True,
        "source": str(input_path.resolve()),
        "source_sha256": sha256(input_path),
        "datasets": DATASETS,
        "kirs": KIRS,
        "target_known_recall": TARGETS,
        "methods": METHODS,
        "selection_used_test_labels": False,
        "interpretation": "post_hoc matched-coverage diagnostic; not a parameter-selection rule",
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(15, 4.8), constrained_layout=True, sharey=True)
    for ax, target in zip(axes, TARGETS):
        frame = summary[summary.target_known_recall == target]
        matrix = frame.pivot(index="dataset", columns="kir", values="mean_delta_oos_f1").reindex(DATASETS, columns=KIRS)
        image = ax.imshow(matrix.to_numpy() * 100.0, cmap="RdBu_r", vmin=-20, vmax=20, aspect="auto")
        ax.set_title(f"Known Recall≈{target:.0%}")
        ax.set_xticks(range(len(KIRS)), [f"{kir:.2f}" for kir in KIRS])
        ax.set_yticks(range(len(DATASETS)), DATASETS if ax is axes[0] else [])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j] * 100:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes, label="S2C − MOGB OOS F1 (pp)", shrink=0.82)
    fig.suptitle("Matched Known-coverage frontier across KIR", fontsize=14)
    fig.savefig(figures / "matched_frontier_delta_heatmaps.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    mean_curve = source.groupby(["dataset", "kir", "target_known_recall", "method"], as_index=False)[["oos_f1", "false_accept_rate"]].mean()
    fig, axes = plt.subplots(len(DATASETS), len(KIRS), figsize=(15, 11), constrained_layout=True, sharex=True, sharey=True)
    for i, dataset in enumerate(DATASETS):
        for j, kir in enumerate(KIRS):
            ax = axes[i, j]
            cell = mean_curve[(mean_curve.dataset == dataset) & (mean_curve.kir == kir)]
            for method in METHODS:
                line = cell[cell.method == method].sort_values("target_known_recall")
                ax.plot(line.target_known_recall * 100, line.oos_f1 * 100, marker="o", label=LABELS[method], color=COLORS[method])
            ax.set_title(f"{dataset} / KIR={kir:.2f}")
            ax.grid(alpha=0.25)
            if i == len(DATASETS) - 1:
                ax.set_xlabel("matched Known Recall (%)")
            if j == 0:
                ax.set_ylabel("OOS F1 (%)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("S2C vs MOGB-Fair at matched Known coverage", fontsize=14)
    fig.savefig(figures / "matched_frontier_oos_f1_curves.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    mean_delta = summary.groupby(["dataset", "kir"], as_index=False)["mean_delta_oos_f1"].mean()
    lines = [
        "# 跨 KIR 匹配 Known 覆盖率前沿分析 V1",
        "",
        "该阶段只读取已完成 operating-curve 结果，不重新训练、不选择新的阈值或方法。每个工作点先用 Known 分数得到目标覆盖率，再报告对应 OOS F1；因此它是事后机制诊断，不是测试集调参。",
        "",
        f"输入：`{input_path}`；SHA256：`{sha256(input_path)}`。覆盖三个数据集、KIR=0.25/0.50/0.75、五个 seed、Known Recall 80%/90%/95%。",
        "",
        "## 结论",
        "",
        "- 如果在相同 Known Recall 下 S2C 仍保持更高 OOS F1，优势更接近分数排序/表示分离，而不是默认阈值的偶然选择。",
        "- 若 MOGB 在低覆盖率工作点更高而 S2C 在高覆盖率工作点更高，则说明二者主要是保守拒识与 Known 覆盖的工作点差异。",
        "- 任何 heatmap 数值都不能写成新的 SOTA 排名，因为 MOGB-Fair 与 S2C 仍属于当前 MiniLM Known-only 组件合同，且目标覆盖率是测试后诊断。",
        "",
        "## 输出",
        "",
        "- `results/analysis/archive/analysis/cross_kir_matched_frontier_v1/paired_frontier.csv`：同 dataset×KIR×seed×目标覆盖率的配对差值。",
        "- `results/analysis/archive/analysis/cross_kir_matched_frontier_v1/summary.csv`：五 seed 均值、标准差、win/tie/loss。",
        "- `figures/archive/analysis/cross_kir_matched_frontier_v1/matched_frontier_delta_heatmaps.png`：跨 KIR 的 OOS F1 差值热图。",
        "- `figures/archive/analysis/cross_kir_matched_frontier_v1/matched_frontier_oos_f1_curves.png`：匹配 Known 覆盖率曲线。",
        "",
        "## 按 dataset/KIR 的平均差值（跨三个目标覆盖率）",
        "",
        "| 数据集 | KIR | S2C−MOGB OOS F1 pp |",
        "|---|---:|---:|",
    ]
    for row in mean_delta.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.kir:.2f} | {row.mean_delta_oos_f1 * 100:.2f} |")
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
