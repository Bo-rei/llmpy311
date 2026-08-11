#!/usr/bin/env python3
"""Build the authoritative S2C-vs-MOGB-Fair mechanism dashboard.

This is analysis-only.  It pairs the completed five-seed Trainable-K1 runs
with frozen-MiniLM MOGB component runs on identical dataset/KIR/seed cells.
No checkpoint, prediction, threshold, or historical experiment is modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/analysis/archive/analysis/minilm_trainable_5seed_fair_v1"
TRAINABLE = SOURCE / "trainable_per_seed.csv"
MOGB = SOURCE / "mogb_fair_per_seed.csv"
OUT = ROOT / "results/analysis/s2c_vs_mogb_mechanism_dashboard_v1"
FIG = ROOT / "figures/s2c_vs_mogb_mechanism_dashboard_v1"
REPORT = ROOT / "docs/analysis/S2C_VS_MOGB_MECHANISM_DASHBOARD_V1.md"
DATASET_ORDER = ("clinc150", "banking77", "stackoverflow")
KIR_ORDER = (0.25, 0.50, 0.75)
SEED_ORDER = (13, 42, 87, 100, 123)
COMPONENT_ORDER = ("mogb_minilm", "mogb_partition_ours_boundary", "trainable_k1")
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_SAMPLES = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def oos_precision_from_f1_recall(f1: np.ndarray, recall: np.ndarray) -> np.ndarray:
    f1 = np.asarray(f1, dtype=float)
    recall = np.asarray(recall, dtype=float)
    denominator = 2.0 * recall - f1
    result = np.zeros_like(f1)
    valid = denominator > 1e-15
    result[valid] = f1[valid] * recall[valid] / denominator[valid]
    return np.clip(result, 0.0, 1.0)


def paired_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("paired bootstrap requires values")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    trainable = pd.read_csv(TRAINABLE)
    mogb = pd.read_csv(MOGB)
    trainable = trainable[[
        "dataset", "kir", "seed", "method", "oos_f1", "f1_all", "f1_k", "accuracy",
        "id_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos",
    ]].copy()
    mogb = mogb[mogb["method"].isin(("mogb_minilm", "mogb_partition_ours_boundary"))].copy()
    required = {(dataset, kir, seed) for dataset in DATASET_ORDER for kir in KIR_ORDER for seed in SEED_ORDER}
    observed_trainable = set(map(tuple, trainable[["dataset", "kir", "seed"]].itertuples(index=False, name=None)))
    if observed_trainable != required:
        raise RuntimeError(f"Trainable coverage mismatch: missing={sorted(required - observed_trainable)} extra={sorted(observed_trainable - required)}")
    for method in ("mogb_minilm", "mogb_partition_ours_boundary"):
        observed = set(map(tuple, mogb[mogb["method"] == method][["dataset", "kir", "seed"]].itertuples(index=False, name=None)))
        if observed != required:
            raise RuntimeError(f"{method} coverage mismatch: missing={sorted(required - observed)} extra={sorted(observed - required)}")
    if trainable.duplicated(["dataset", "kir", "seed"]).any() or mogb.duplicated(["dataset", "kir", "seed", "method"]).any():
        raise RuntimeError("duplicate paired cell")
    return trainable, mogb


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["oos_recall"] = 1.0 - result["false_accept_rate"]
    result["oos_precision"] = oos_precision_from_f1_recall(
        result["oos_f1"].to_numpy(), result["oos_recall"].to_numpy()
    )
    result["known_coverage"] = result["id_recall"]
    return result


def paired_tables(trainable: pd.DataFrame, mogb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainable = enrich(trainable)
    mogb = enrich(mogb)
    all_rows = pd.concat([trainable, mogb], ignore_index=True)
    metrics = (
        "oos_f1", "f1_all", "f1_k", "accuracy", "id_recall", "false_accept_rate",
        "false_reject_rate", "oos_precision", "oos_recall", "auroc", "aupr_oos",
    )
    paired_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    train_keyed = trainable.set_index(["dataset", "kir", "seed"])
    for comparator in ("mogb_minilm", "mogb_partition_ours_boundary"):
        comp_keyed = mogb[mogb["method"] == comparator].set_index(["dataset", "kir", "seed"])
        for key in train_keyed.index:
            left = train_keyed.loc[key]
            right = comp_keyed.loc[key]
            row: dict[str, Any] = {"dataset": key[0], "kir": key[1], "seed": key[2], "comparator": comparator}
            for metric in metrics:
                row[f"trainable_{metric}"] = float(left[metric])
                row[f"comparator_{metric}"] = float(right[metric])
                row[f"delta_{metric}"] = float(left[metric] - right[metric])
            paired_rows.append(row)
        pair_frame = pd.DataFrame([row for row in paired_rows if row["comparator"] == comparator])
        for (dataset, kir), group in pair_frame.groupby(["dataset", "kir"], sort=False):
            for metric in metrics:
                values = group[f"delta_{metric}"].to_numpy(dtype=float)
                low, high = paired_ci(values)
                effect_rows.append({
                    "dataset": dataset,
                    "kir": kir,
                    "comparator": comparator,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean_delta": float(values.mean()),
                    "std_delta": float(values.std(ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": int(np.sum(values > 1e-12)),
                    "ties": int(np.sum(np.abs(values) <= 1e-12)),
                    "losses": int(np.sum(values < -1e-12)),
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_samples": BOOTSTRAP_SAMPLES,
                })
    paired = pd.DataFrame(paired_rows)
    effects = pd.DataFrame(effect_rows)
    summary = (
        all_rows.groupby(["dataset", "kir", "method"], as_index=False)
        .agg({metric: ["mean", "std"] for metric in metrics})
    )
    summary.columns = ["_".join(part for part in column if part) for column in summary.columns.to_flat_index()]
    return paired, effects, summary


def label_method(value: str) -> str:
    return {
        "trainable_k1": "S2C Trainable K=1",
        "mogb_minilm": "MOGB-Fair",
        "mogb_partition_ours_boundary": "MOGB partition + S2C boundary",
    }[value]


def plot_delta_heatmaps(effects: pd.DataFrame) -> None:
    metrics = [
        ("oos_f1", "OOS F1"), ("f1_all", "F1-All"), ("id_recall", "Known Recall"),
        ("oos_precision", "OOS Precision"), ("oos_recall", "OOS Recall"),
        ("false_accept_rate", "False Acceptance"),
    ]
    data = effects[effects["comparator"] == "mogb_minilm"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        subset = data[data["metric"] == metric]
        matrix = np.asarray([
            [float(subset[(subset.dataset == dataset) & (np.isclose(subset.kir, kir))].mean_delta.iloc[0]) * 100.0 for kir in KIR_ORDER]
            for dataset in DATASET_ORDER
        ])
        bound = max(float(np.abs(matrix).max()), 1.0)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix[row, col]:+.1f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), ["25%", "50%", "75%"])
        ax.set_yticks(range(3), ["CLINC150", "Banking77", "StackOverflow"])
        ax.set_title(f"Δ {title} (pp)")
        fig.colorbar(image, ax=ax, shrink=0.72)
    fig.suptitle("S2C Trainable K=1 minus MOGB-Fair (five-seed paired means)", fontsize=14)
    fig.savefig(FIG / "paired_delta_heatmaps.png", dpi=220)
    plt.close(fig)


def plot_error_budget(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharex=True, sharey=True, constrained_layout=True)
    colors = {0.25: "#4c78a8", 0.50: "#f58518", 0.75: "#54a24b"}
    for ax, dataset in zip(axes, DATASET_ORDER):
        for kir in KIR_ORDER:
            subset = summary[(summary.dataset == dataset) & np.isclose(summary.kir, kir)]
            left = subset[subset.method == "mogb_minilm"].iloc[0]
            right = subset[subset.method == "trainable_k1"].iloc[0]
            ax.annotate(
                "",
                xy=(right.false_accept_rate_mean * 100, right.false_reject_rate_mean * 100),
                xytext=(left.false_accept_rate_mean * 100, left.false_reject_rate_mean * 100),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": colors[kir]},
            )
            ax.scatter(left.false_accept_rate_mean * 100, left.false_reject_rate_mean * 100, marker="s", s=55, color=colors[kir])
            ax.scatter(right.false_accept_rate_mean * 100, right.false_reject_rate_mean * 100, marker="o", s=70, color=colors[kir], label=f"KIR {kir:.2f}")
        ax.set_title(dataset)
        ax.set_xlabel("OOS false acceptance (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Known false rejection (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3)
    fig.suptitle("MOGB-Fair (square) → S2C Trainable K=1 (circle): error-budget transfer", y=1.02)
    fig.savefig(FIG / "error_budget_arrows.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharex=True, sharey=True, constrained_layout=True)
    method_style = {
        "mogb_minilm": ("s", "#e45756"),
        "trainable_k1": ("o", "#4c78a8"),
    }
    for ax, dataset in zip(axes, DATASET_ORDER):
        for method, (marker, color) in method_style.items():
            subset = summary[(summary.dataset == dataset) & (summary.method == method)].sort_values("kir")
            ax.plot(subset.oos_recall_mean * 100, subset.oos_precision_mean * 100, marker=marker, color=color, linewidth=2, label=label_method(method))
            for row in subset.itertuples(index=False):
                ax.annotate(f"{row.kir:.2f}", (row.oos_recall_mean * 100, row.oos_precision_mean * 100), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_title(dataset)
        ax.set_xlabel("OOS Recall (%)")
    axes[0].set_ylabel("OOS Precision (%)")
    axes[-1].legend(loc="best")
    fig.suptitle("Why OOS F1 differs: conservative rejection vs balanced coverage")
    fig.savefig(FIG / "oos_precision_recall_decomposition.png", dpi=220)
    plt.close(fig)


def plot_kir_curves(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6), sharex=True, constrained_layout=True)
    colors = {"mogb_minilm": "#e45756", "trainable_k1": "#4c78a8"}
    for column, dataset in enumerate(DATASET_ORDER):
        for method in colors:
            subset = summary[(summary.dataset == dataset) & (summary.method == method)].sort_values("kir")
            axes[0, column].errorbar(subset.kir, subset.oos_f1_mean * 100, yerr=subset.oos_f1_std * 100, marker="o", capsize=3, color=colors[method], label=label_method(method))
            axes[1, column].errorbar(subset.kir, subset.f1_all_mean * 100, yerr=subset.f1_all_std * 100, marker="o", capsize=3, color=colors[method])
        axes[0, column].set_title(dataset)
        axes[1, column].set_xlabel("KIR")
    axes[0, 0].set_ylabel("OOS F1 (%)")
    axes[1, 0].set_ylabel("F1-All (%)")
    axes[0, -1].legend(loc="best")
    fig.suptitle("Five-seed KIR trends under the same protocol")
    fig.savefig(FIG / "kir_performance_curves.png", dpi=220)
    plt.close(fig)


def plot_forest(effects: pd.DataFrame) -> None:
    data = effects[(effects.comparator == "mogb_minilm") & effects.metric.isin(("oos_f1", "f1_all", "id_recall"))].copy()
    data["label"] = data.apply(lambda row: f"{row.dataset} / {row.kir:.2f} / {row.metric}", axis=1)
    data = data.sort_values(["dataset", "kir", "metric"]).reset_index(drop=True)
    y = np.arange(len(data))
    mean = data.mean_delta.to_numpy() * 100
    low = data.ci95_low.to_numpy() * 100
    high = data.ci95_high.to_numpy() * 100
    fig, ax = plt.subplots(figsize=(9.4, 10.5))
    ax.errorbar(mean, y, xerr=np.vstack((mean - low, high - mean)), fmt="o", capsize=3, color="#4c78a8")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, data.label)
    ax.set_xlabel("S2C - MOGB-Fair paired mean delta (pp), 95% bootstrap CI")
    ax.set_title("Seed-paired effect stability")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "paired_effect_forest.png", dpi=220)
    plt.close(fig)


def plot_component_bridge(summary: pd.DataFrame) -> None:
    labels = []
    values = {method: [] for method in COMPONENT_ORDER}
    for dataset in DATASET_ORDER:
        for kir in KIR_ORDER:
            labels.append(f"{dataset}\n{kir:.2f}")
            for method in COMPONENT_ORDER:
                row = summary[(summary.dataset == dataset) & np.isclose(summary.kir, kir) & (summary.method == method)].iloc[0]
                values[method].append(row.f1_all_mean * 100)
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    for offset, method, color in zip((-width, 0.0, width), COMPONENT_ORDER, ("#e45756", "#f2cf5b", "#4c78a8")):
        ax.bar(x + offset, values[method], width, label=label_method(method), color=color)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("F1-All (%)")
    ax.set_title("Component bridge: boundary replacement helps, representation adaptation closes more of the gap")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "component_bridge_f1_all.png", dpi=220)
    plt.close(fig)


def build_report(effects: pd.DataFrame, summary: pd.DataFrame, manifest_sha: str) -> str:
    core = effects[(effects.comparator == "mogb_minilm") & effects.metric.isin(("oos_f1", "f1_all", "id_recall", "false_accept_rate", "false_reject_rate", "oos_precision", "oos_recall"))]
    pivot = core.pivot_table(index=["dataset", "kir"], columns="metric", values="mean_delta").reset_index()
    display = pivot.copy()
    for column in display.columns[2:]:
        display[column] = display[column] * 100.0
    all_oos = effects[(effects.comparator == "mogb_minilm") & (effects.metric == "oos_f1")]
    oos_wins = int(all_oos.wins.sum())
    oos_losses = int(all_oos.losses.sum())
    all_f1 = effects[(effects.comparator == "mogb_minilm") & (effects.metric == "f1_all")]
    f1_wins = int(all_f1.wins.sum())
    f1_losses = int(all_f1.losses.sum())
    mean_effect = core.groupby("metric").mean_delta.mean() * 100.0
    component = summary.groupby("method", as_index=False)[["oos_f1_mean", "f1_all_mean", "id_recall_mean", "false_accept_rate_mean"]].mean()
    component["method"] = component.method.map(label_method)
    return f"""# S2C 与 MOGB-Fair 机制对比仪表盘 V1

> 比较对象固定为当前 `S2C-Trainable-K1` 与 `MOGB-MiniLM-Fair`。两者共享 `protocol_v2_textoir_v1` 的 dataset/KIR/seed，但表示和边界合同不同。本报告不把历史 Cascade 或论文 BERT MOGB 混入公平排名。

## 实验覆盖

- 三个数据集 × 三个 KIR × 五个相同 seed，共 45 个严格配对单元。
- 主比较：Known-only Trainable MiniLM K=1 vs 冻结 MiniLM MOGB adaptive balls + mean radius。
- 组件桥：增加 `MOGB partition + S2C mean_std boundary`，用于分离边界工作点与表示适配的贡献。
- 统计：固定 seed={BOOTSTRAP_SEED}、{BOOTSTRAP_SAMPLES:,} 次配对 bootstrap。

## 核心结论

1. S2C 的优势不是“更保守地拒绝更多样本”。45 个配对中，OOS F1 win/loss={oos_wins}/{oos_losses}，F1-All win/loss={f1_wins}/{f1_losses}。
2. 相对 MOGB-Fair，S2C 平均 Known Recall 提高 {mean_effect['id_recall']:+.2f}pp、OOS Precision 提高 {mean_effect['oos_precision']:+.2f}pp；代价是 OOS Recall {mean_effect['oos_recall']:+.2f}pp、false acceptance {mean_effect['false_accept_rate']:+.2f}pp。其 OOS F1 提升来自 precision–recall 工作点更平衡，而不是任一方向全面占优。
3. MOGB-Fair 的主要问题是大量 Known→OOS：mean-radius 粒球非常保守。换成 S2C mean_std 边界通常能恢复部分 F1-All，但完整 Trainable-K1 仍普遍更好，说明优势同时来自表示适配和边界工作点。
4. StackOverflow 固定 K=2 的失败与 MOGB-Fair 的失败方向相反：固定 K=2 是 OOS false acceptance 过多；MOGB-Fair 是 Known false rejection 过多。不能用同一个“多中心无效”概括二者。

## 45 单元平均

{component.to_markdown(index=False, floatfmt='.4f')}

## 每个数据集与 KIR 的 S2C−MOGB 差值（百分点）

{display.to_markdown(index=False, floatfmt='+.2f')}

## 图表

1. `paired_delta_heatmaps.png`：六个指标的 dataset×KIR 配对差值。
2. `error_budget_arrows.png`：从 MOGB 保守拒识工作点移动到 S2C 平衡工作点。
3. `oos_precision_recall_decomposition.png`：OOS F1 的 precision/recall 来源。
4. `kir_performance_curves.png`：五 seed 的 KIR 趋势与标准差。
5. `paired_effect_forest.png`：OOS F1、F1-All、Known Recall 的配对置信区间。
6. `component_bridge_f1_all.png`：MOGB boundary、S2C boundary、Trainable representation 的组件桥。

## 解释边界

- 这能够解释当前 S2C Gate 为什么优于 MOGB 的冻结 MiniLM 公平组件。
- 它不能证明超过论文完整 BERT MOGB；本地官方逻辑 MOGB 的复现差距另见 `MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`。
- 它也不能替代完整 Gate→Router→Expert 与 ADB/DA-ADB/DCLOOS 的同协议主表。
- Manifest SHA256：`{manifest_sha}`。
"""


def main() -> int:
    trainable, mogb = load_inputs()
    paired, effects, summary = paired_tables(trainable, mogb)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / "paired_cells.csv", paired)
    atomic_csv(OUT / "paired_effects.csv", effects)
    atomic_csv(OUT / "method_summary.csv", summary)
    plot_delta_heatmaps(effects)
    plot_error_budget(summary)
    plot_precision_recall(summary)
    plot_kir_curves(summary)
    plot_forest(effects)
    plot_component_bridge(summary)
    manifest = {
        "analysis_id": "s2c_vs_mogb_mechanism_dashboard_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "paired_cells": 45,
        "methods": list(COMPONENT_ORDER),
        "datasets": list(DATASET_ORDER),
        "kirs": list(KIR_ORDER),
        "seeds": list(SEED_ORDER),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "trainable_source_sha256": sha256_file(TRAINABLE),
        "mogb_source_sha256": sha256_file(MOGB),
        "script_sha256": sha256_file(Path(__file__)),
        "selection": "analysis-only; no model, threshold, K, radius, or checkpoint selection",
    }
    atomic_text(OUT / "MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    manifest_sha = sha256_file(OUT / "MANIFEST.json")
    atomic_text(REPORT, build_report(effects, summary, manifest_sha))
    print(json.dumps({"status": "complete", "paired_cells": len(paired) // 2, "effect_rows": len(effects), "figures": 6, "manifest_sha256": manifest_sha}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
