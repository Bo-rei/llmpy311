#!/usr/bin/env python3
"""Build a compact, protocol-aware analysis pack from existing light CSVs.

This tool deliberately does not load checkpoints, embeddings, raw text, or
test-time artifacts.  It joins the already exported five-seed rows for the
Trainable/Frozen/MOGB-component comparison and produces paired effects,
work-point plots, and an evidence report.  It is therefore safe to run after
large local artifacts have been archived or removed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "analysis" / "minilm_trainable_5seed_fair_v1" / "all_methods_per_seed.csv"
NATIVE_INPUT = ROOT / "results" / "analysis" / "native_baselines_trainable_v1" / "trainable_native_per_seed.csv"
OUT = ROOT / "results" / "analysis" / "experimental_mechanism_pack_v3"
FIGURES = ROOT / "figures" / "experimental_mechanism_pack_v3"
REPORT = ROOT / "docs" / "analysis" / "EXPERIMENTAL_MECHANISM_PACK_V3.md"

METHOD_ORDER = (
    "trainable_k1",
    "single_centroid",
    "fixed_k2",
    "random_partition",
    "mogb_minilm",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
)
SHORT_LABELS = {
    "trainable_k1": "Trainable K=1",
    "single_centroid": "Frozen single",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_minilm": "MOGB MiniLM",
    "mogb_partition_ours_boundary": "MOGB split + ours",
    "ours_partition_mogb_boundary": "Ours split + MOGB",
}
METRICS = ("oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc", "aupr_oos")
HIGHER = {"oos_f1", "f1_all", "known_recall", "auroc", "aupr_oos"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if key == "known_recall" and value in (None, ""):
        # MOGB/fair component exports use the historical ``id_recall`` name.
        value = row.get("id_recall", "")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value


def group_key(row: dict[str, str]) -> tuple[str, float, int]:
    return (row["dataset"], float(row["kir"]), int(row["seed"]))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def paired_bootstrap(values: list[float], seed: int = 20260806, samples: int = 10000) -> tuple[float, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if clean.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, clean.size, size=(samples, clean.size))
    means = clean[indices].mean(axis=1)
    return percentile(means, 0.025), percentile(means, 0.975)


def effect_size(values: list[float]) -> float:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if clean.size < 2:
        return math.nan
    std = float(clean.std(ddof=1))
    return float(clean.mean() / std) if std > 1e-12 else math.inf if clean.mean() > 0 else -math.inf


def pair_effects(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    index = {(group_key(row), row["method"]): row for row in rows}
    baselines = [method for method in METHOD_ORDER if method != "trainable_k1"]
    output: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for kir in sorted({float(row["kir"]) for row in rows if row["dataset"] == dataset}):
            for baseline in baselines:
                for metric in METRICS:
                    deltas: list[float] = []
                    for seed in sorted({int(row["seed"]) for row in rows}):
                        left = index.get(((dataset, kir, seed), "trainable_k1"))
                        right = index.get(((dataset, kir, seed), baseline))
                        if left is None or right is None:
                            continue
                        value = number(left, metric) - number(right, metric)
                        if np.isfinite(value):
                            deltas.append(value)
                    if not deltas:
                        continue
                    low, high = paired_bootstrap(deltas, seed=20260806)
                    output.append(
                        {
                            "dataset": dataset,
                            "kir": kir,
                            "baseline": baseline,
                            "metric": metric,
                            "n_seeds": len(deltas),
                            "mean_delta": float(np.mean(deltas)),
                            "median_delta": float(np.median(deltas)),
                            "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                            "ci95_low": low,
                            "ci95_high": high,
                            "wins": int(sum(value > 1e-12 for value in deltas)),
                            "ties": int(sum(abs(value) <= 1e-12 for value in deltas)),
                            "losses": int(sum(value < -1e-12 for value in deltas)),
                            "effect_size_d": effect_size(deltas),
                            "bootstrap_seed": 20260806,
                            "bootstrap_samples": 10000,
                        }
                    )
    return output


def grouped_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], float(row["kir"]), row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, kir, method), values in sorted(grouped.items()):
        item: dict[str, Any] = {"dataset": dataset, "kir": kir, "method": method, "n_seeds": len(values)}
        for metric in METRICS:
            numbers = np.asarray([number(row, metric) for row in values], dtype=float)
            numbers = numbers[np.isfinite(numbers)]
            item[f"{metric}_mean"] = float(numbers.mean()) if numbers.size else math.nan
            item[f"{metric}_std"] = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0 if numbers.size else math.nan
        output.append(item)
    return output


def pareto_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return method-level Pareto flags for OOS F1, F1-All, recall and FA."""
    output: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in summary}):
        for kir in sorted({float(row["kir"]) for row in summary if row["dataset"] == dataset}):
            subset = [row for row in summary if row["dataset"] == dataset and float(row["kir"]) == kir]
            for row in subset:
                dominated = False
                for other in subset:
                    if other is row:
                        continue
                    comparisons = (
                        other["oos_f1_mean"] >= row["oos_f1_mean"],
                        other["f1_all_mean"] >= row["f1_all_mean"],
                        other["known_recall_mean"] >= row["known_recall_mean"],
                        other["false_accept_rate_mean"] <= row["false_accept_rate_mean"],
                    )
                    strict = (
                        other["oos_f1_mean"] > row["oos_f1_mean"]
                        or other["f1_all_mean"] > row["f1_all_mean"]
                        or other["known_recall_mean"] > row["known_recall_mean"]
                        or other["false_accept_rate_mean"] < row["false_accept_rate_mean"]
                    )
                    if all(comparisons) and strict:
                        dominated = True
                        break
                output.append({"dataset": dataset, "kir": kir, "method": row["method"], "pareto_efficient": not dominated})
    return output


def load_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_heatmap(summary: list[dict[str, Any]], path: Path, metric: str = "oos_f1_mean") -> None:
    plt = load_plotting()
    datasets = sorted({row["dataset"] for row in summary})
    kirs = sorted({float(row["kir"]) for row in summary})
    methods = ["single_centroid", "fixed_k2", "mogb_partition_ours_boundary", "trainable_k1"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(13, 4.2), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        matrix = []
        labels = []
        for method in methods:
            row_values = []
            for kir in kirs:
                match = next((row for row in summary if row["dataset"] == dataset and float(row["kir"]) == kir and row["method"] == method), None)
                row_values.append(float(match[metric]) * 100 if match and np.isfinite(match[metric]) else np.nan)
            matrix.append(row_values)
            labels.append(SHORT_LABELS.get(method, method))
        image = axis.imshow(np.asarray(matrix), cmap="RdYlGn", aspect="auto", vmin=35, vmax=100)
        axis.set_title(dataset)
        axis.set_xticks(range(len(kirs)), [f"{value:.2f}" for value in kirs])
        axis.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(kirs)):
                value = matrix[i][j]
                if np.isfinite(value):
                    axis.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
        axis.set_xlabel("KIR")
    fig.colorbar(image, ax=axes[0].tolist(), label="OOS F1 (%)")
    fig.suptitle("OOS F1 by method and KIR (five-seed mean)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary: list[dict[str, Any]], path: Path) -> None:
    plt = load_plotting()
    datasets = sorted({row["dataset"] for row in summary})
    colors = {method: color for method, color in zip(METHOD_ORDER, ("#d62728", "#1f77b4", "#9467bd", "#2ca02c", "#8c564b", "#ff7f0e", "#17becf"))}
    labels = {method: method.replace("_", " ") for method in METHOD_ORDER}
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 4.5), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        rows = [row for row in summary if row["dataset"] == dataset and abs(float(row["kir"]) - 0.50) < 1e-9]
        for row in rows:
            x = float(row["known_recall_mean"]) * 100
            y = float(row["oos_f1_mean"]) * 100
            size = 40 + 500 * float(row["false_accept_rate_mean"])
            method = row["method"]
            axis.scatter(x, y, s=size, color=colors.get(method, "#333333"), alpha=0.8, edgecolor="white", linewidth=0.5)
            axis.annotate(labels.get(method, method), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
        axis.set_title(f"{dataset} · KIR=0.50")
        axis.set_xlabel("Known Recall (%)")
        axis.set_ylabel("OOS F1 (%)")
        axis.grid(alpha=0.25)
    fig.suptitle("Coverage--rejection trade-off (point size = false acceptance)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_kir_curves(summary: list[dict[str, Any]], path: Path) -> None:
    plt = load_plotting()
    datasets = sorted({row["dataset"] for row in summary})
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 4.5), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        for method in METHOD_ORDER:
            rows = sorted((row for row in summary if row["dataset"] == dataset and row["method"] == method), key=lambda row: float(row["kir"]))
            if not rows:
                continue
            x = [float(row["kir"]) for row in rows]
            y = [float(row["oos_f1_mean"]) * 100 for row in rows]
            e = [float(row["oos_f1_std"]) * 100 for row in rows]
            axis.errorbar(x, y, yerr=e, marker="o", linewidth=1.4, capsize=2, label=SHORT_LABELS.get(method, method))
        axis.set_title(dataset)
        axis.set_xlabel("KIR")
        axis.set_ylabel("OOS F1 (%)")
        axis.set_xticks([0.25, 0.50, 0.75])
        axis.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.suptitle("OOS F1 across KIR (mean +/- standard deviation)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_effect_grid(effects: list[dict[str, Any]], path: Path) -> None:
    plt = load_plotting()
    baselines = ["single_centroid", "fixed_k2", "random_partition", "mogb_partition_ours_boundary"]
    datasets = sorted({row["dataset"] for row in effects})
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 4.5), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        matrix = []
        labels = []
        for baseline in baselines:
            values = []
            for kir in (0.25, 0.50, 0.75):
                row = next((item for item in effects if item["dataset"] == dataset and float(item["kir"]) == kir and item["baseline"] == baseline and item["metric"] == "oos_f1"), None)
                values.append(float(row["mean_delta"]) * 100 if row else np.nan)
            matrix.append(values)
            labels.append(SHORT_LABELS.get(baseline, baseline))
        image = axis.imshow(np.asarray(matrix), cmap="RdBu_r", aspect="auto", vmin=-25, vmax=25)
        axis.set_title(dataset)
        axis.set_xticks(range(3), [".25", ".50", ".75"])
        axis.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(3):
                value = matrix[i][j]
                if np.isfinite(value):
                    axis.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8)
        axis.set_xlabel("KIR")
    fig.colorbar(image, ax=axes[0].tolist(), label="Trainable - baseline OOS F1 (pp)")
    fig.suptitle("Paired Trainable K=1 minus frozen-component OOS F1 (pp)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(summary: list[dict[str, Any]], effects: list[dict[str, Any]], pareto: list[dict[str, Any]], input_hash: str, native_rows: int) -> str:
    lines = [
        "# 实验机制分析包 V3（现有结果复核）",
        "",
        "> 本报告只读取已经提交到 `s2c/results/analysis/` 的轻量 CSV；不读取 checkpoint、embedding 或原始文本，也不重新训练。当前工作区的 `../artifacts/s2c/runs/` 已不在磁盘，因此报告同时标记重产物缺失风险。",
        "",
        "## 1. 数据与证据边界",
        "",
        f"- 输入：`results/analysis/minilm_trainable_5seed_fair_v1/all_methods_per_seed.csv`，SHA256=`{input_hash}`。",
        "- 主表：315 行（3 数据集 × 3 KIR × 5 seed × 7 方法）；Trainable K=1、Frozen 单/双中心、随机分簇和三种 MOGB 组件。",
        f"- Trainable native 归因轻量表：{native_rows} 行（仅 KIR=.50、3 seed、四种 detector）；不与主表混合。",
        "- 所有主表差值按 dataset×KIR×seed 配对；Bootstrap RNG=20260806、10,000 次。",
        "- `fulltex.tex`、ADB/DA-ADB、DCLOOS 只作为外部/历史层参照，不参与同协议配对排名。",
        "",
        "## 2. 当前最稳定的实验事实",
        "",
        "| 数据集 | KIR | Trainable K=1 OOS F1 | Frozen 单中心 OOS F1 | 差值 | Trainable false acceptance | Trainable Known Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted({row["dataset"] for row in summary}):
        for kir in (0.25, 0.50, 0.75):
            t = next(row for row in summary if row["dataset"] == dataset and float(row["kir"]) == kir and row["method"] == "trainable_k1")
            f = next(row for row in summary if row["dataset"] == dataset and float(row["kir"]) == kir and row["method"] == "single_centroid")
            lines.append(f"| {dataset} | {kir:.2f} | {t['oos_f1_mean']*100:.2f}±{t['oos_f1_std']*100:.2f} | {f['oos_f1_mean']*100:.2f}±{f['oos_f1_std']*100:.2f} | {(t['oos_f1_mean']-f['oos_f1_mean'])*100:+.2f} pp | {t['false_accept_rate_mean']*100:.2f}% | {t['known_recall_mean']*100:.2f}% |")
    lines.extend([
        "",
        "## 3. 机制解释",
        "",
        "### 3.1 Trainable 的主要收益在 K=1 分数排序",
        "",
        "Trainable K=1 在三个数据集和多数 KIR 上提高 OOS F1，尤其是 StackOverflow 的 KIR=.50/.75；同时 false acceptance 通常下降。这个现象在同一 seed 的配对差值中保持，说明收益不是单个 seed 的偶然峰值。它支持“最后两层 MiniLM+投影改变 Known/OOS 分数排序”的解释，而不是“增加中心数量带来收益”。",
        "",
        "### 3.2 多中心不是表示训练的自动副产物",
        "",
        "StackOverflow 的 fixed K=2 仍然明显差于 Trainable K=1；MOGB 分区组件通常进一步牺牲 Known Recall 换取低 false acceptance。E2/E3/RACAL 的逐样本诊断已经显示稳定聚类仍会新增大量 OOS 误接受，因此当前瓶颈是接受区域的组合语义，而非 KMeans 是否收敛。",
        "",
        "### 3.3 数据集差异是真实的",
        "",
        "Banking77 在某些 KIR 下固定多中心或 MOGB 分区能提高 OOS F1，但往往伴随 Known Recall/F1-All 下降；CLINC150 的收益小且不稳定；StackOverflow 在高 KIR 下多中心风险最大。下一步不能用一个跨数据集的“最优 K”解释这些现象。",
        "",
        "## 4. Pareto 工作点",
        "",
        "Pareto 标记同时考虑 OOS F1、F1-All、Known Recall（越高越好）和 false acceptance（越低越好）。它不是 SOTA 排名，而是展示为什么需要同时报告覆盖和拒识。每个 dataset/KIR 的候选方法见 `pareto_flags.csv`。",
        "",
        "## 5. 与外部方法的边界",
        "",
        "- ADB/DA-ADB：已有单 seed、BERT/兼容环境数字；不是当前 protocol_v2 五 seed 同合同结果。",
        "- MOGB：作者 BERT 单格严格复现未达到论文数字；Frozen MiniLM MOGB 行是公平组件对照，不是官方 MOGB。",
        "- DCLOOS：使用伪 OOS/外部 OOS 监督；reduced-budget 结果不能与 Known-only Trainable 直接排名。",
        "- 因此当前最稳妥的“自有方法胜出”表述是：在相同当前协议的 Gate-only、Known-only 条件下，Trainable K=1 通常比 Frozen/MOGB 组件有更好的覆盖—拒识折中；尚不能声称超过完整 MOGB 或端到端 DCLOOS。",
        "",
        "## 6. 当前实验瓶颈与下一步",
        "",
        "1. 现有轻量结果已经足以支持 KIR/K/表示/组件的机制分析；不应继续重复相同矩阵。",
        "2. 当前工作区缺少原始 run/checkpoint，因此要重新运行新实验，必须先恢复并核对 artifacts provenance，或重新登记一套最小可复现实验。",
        "3. 恢复产物后，最高价值的下一步是同一 Known-only 工作点下复算 Trainable/Frozen Gate、native detectors 和 MOGB 组件；不要再用 test-oracle threshold 做正式选择。",
        "4. 之后再决定是否把最稳定的 Gate 接入 Cascade；外部基线必须单独标注监督条件。",
        "",
        "## 7. 输出与复核",
        "",
        "- `paired_effects.csv`：Trainable 相对每个冻结组件的 paired bootstrap 差值。",
        "- `method_summary.csv`：dataset×KIR×method 的均值、标准差。",
        "- `pareto_flags.csv`：覆盖—拒识多目标 Pareto 标记。",
        "- `figures/experimental_mechanism_pack_v3/`：热图、KIR 曲线、工作点散点和差值热图。",
        "",
        "## 8. 结论",
        "",
        "当前最可信的结果不是“固定多中心达到了 SOTA”，而是：Trainable MiniLM K=1 在当前统一协议下是最稳定的自有 Gate 候选；它的收益来自表示适配后的分数分离和更好的覆盖—拒识平衡。固定 K>1、MOGB 组件和训练参与式自适应 split 尚未在 StackOverflow 上形成安全正收益。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    rows = read_csv(args.input)
    if len(rows) != 315:
        raise RuntimeError(f"Expected the completed 315-row five-seed matrix, got {len(rows)}: {args.input}")
    methods = {row["method"] for row in rows}
    missing = set(METHOD_ORDER) - methods
    if missing:
        raise RuntimeError(f"Five-seed matrix is missing methods: {sorted(missing)}")
    summary = grouped_summary(rows)
    effects = pair_effects(rows)
    pareto = pareto_rows(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "method_summary.csv", summary)
    write_csv(args.output / "paired_effects.csv", effects)
    write_csv(args.output / "pareto_flags.csv", pareto)
    native_rows = len(read_csv(NATIVE_INPUT)) if NATIVE_INPUT.is_file() else 0
    manifest = {
        "stage": "experimental_mechanism_pack_v3",
        "protocol_version": "protocol_v2_textoir_v1",
        "input": str(args.input.relative_to(ROOT)),
        "input_sha256": sha256_file(args.input),
        "input_rows": len(rows),
        "methods": sorted(methods),
        "native_trainable_rows": native_rows,
        "bootstrap_seed": 20260806,
        "bootstrap_samples": 10000,
        "test_used_for_selection": False,
        "uses_raw_text": False,
        "uses_checkpoints": False,
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_heatmap(summary, args.figures / "oos_f1_method_kir_heatmap.png")
    plot_tradeoff(summary, args.figures / "kir050_coverage_oos_tradeoff.png")
    plot_kir_curves(summary, args.figures / "method_kir_curves.png")
    plot_effect_grid(effects, args.figures / "trainable_paired_effects_heatmap.png")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(summary, effects, pareto, manifest["input_sha256"], native_rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary), "effect_rows": len(effects), "pareto_rows": len(pareto), "report": str(args.report), "figures": str(args.figures)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
