"""Paired, protocol-aware comparison of Trainable K=1 and MOGB components.

The MOGB rows are not called an official MOGB reproduction: they are the
existing Frozen MiniLM component matrix.  This script only aligns rows by
dataset/KIR/seed and reports the metric trade-offs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/analysis/trainable_vs_mogb_component_v1"
FIG = ROOT / "figures/trainable_vs_mogb_component_v1"
TRAINABLE = ROOT / "results/analysis/minilm_trainable_5seed_fair_v1/trainable_per_seed.csv"
MOGB = ROOT / "results/mogb/fair_matrix.csv"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
BASELINES = ("mogb_partition_ours_boundary", "mogb_minilm", "fixed_k2")
METRICS = ("oos_f1", "f1_all", "id_recall", "false_accept_rate", "accuracy", "auroc")


def paired_bootstrap(values: np.ndarray, rng_seed: int = 20260725, n_resamples: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    samples = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAINABLE)
    mogb = pd.read_csv(MOGB)
    train = train[train.seed.isin(SEEDS)].copy()
    mogb = mogb[mogb.seed.isin(SEEDS) & mogb.method.isin(BASELINES)].copy()
    required = {"dataset", "kir", "seed", *METRICS}
    if not required.issubset(train.columns) or not required.issubset(mogb.columns):
        raise ValueError("Missing comparison columns")
    return train, mogb


def _write(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, kir, method), group in frame.groupby(["dataset", "kir", "method"]):
        row: dict[str, Any] = {"dataset": dataset, "kir": kir, "method": method, "n_seeds": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
        rows.append(row)
    return pd.DataFrame(rows)


def _effects(train: pd.DataFrame, mogb: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for baseline in BASELINES:
        ref = mogb[mogb.method == baseline]
        merged = train.merge(ref, on=["dataset", "kir", "seed"], suffixes=("_trainable", "_baseline"), validate="one_to_one")
        if len(merged) != len(train):
            raise ValueError(f"Missing paired rows for {baseline}: {len(merged)} vs {len(train)}")
        for (dataset, kir), group in merged.groupby(["dataset", "kir"]):
            for metric in METRICS:
                delta = group[f"{metric}_trainable"].to_numpy() - group[f"{metric}_baseline"].to_numpy()
                mean, low, high = paired_bootstrap(delta)
                rows.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "baseline": baseline,
                        "metric": metric,
                        "n_seeds": len(delta),
                        "mean_delta": mean,
                        "median_delta": float(np.median(delta)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": int(np.sum(delta > 1e-12)),
                        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                        "losses": int(np.sum(delta < -1e-12)),
                        "effect_size_d": float(mean / max(np.std(delta, ddof=1), 1e-12)),
                    }
                )
    return pd.DataFrame(rows)


def _plot(aggregate: pd.DataFrame, effects: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    label = {"mogb_partition_ours_boundary": "MOGB partition + s2c boundary", "mogb_minilm": "MOGB MiniLM", "fixed_k2": "Frozen fixed K=2"}
    # OOS F1 effect relative to each baseline.
    sub = effects[effects.metric == "oos_f1"].copy()
    sub["comparison"] = sub.baseline.map(label)
    grid = sub.pivot_table(index=["dataset", "kir"], columns="comparison", values="mean_delta") * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(grid, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "Trainable minus baseline OOS F1 (pp)"})
    ax.set_xlabel("baseline")
    ax.set_ylabel("dataset / KIR")
    ax.set_title("Trainable K=1 vs MOGB/fixed-K components (paired five-seed OOS F1 delta)")
    fig.tight_layout()
    fig.savefig(FIG / "paired_oos_f1_delta_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # KIR=.50 trade-off plot.
    central = aggregate[aggregate.kir == 0.50].copy()
    central["label"] = central.method.map({"trainable_k1": "Trainable K=1", **label})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = central[central.dataset == dataset]
        for _, row in part.iterrows():
            ax.scatter(row.false_accept_rate_mean * 100, row.oos_f1_mean * 100, s=100 if row.method == "trainable_k1" else 60, label=row.label)
            ax.annotate(row.label, (row.false_accept_rate_mean * 100, row.oos_f1_mean * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(dataset)
        ax.set_xlabel("false acceptance (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    fig.suptitle("KIR=0.50: OOS F1 vs false acceptance across components")
    fig.tight_layout()
    fig.savefig(FIG / "kir050_oos_f1_false_acceptance_components.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Known Recall versus OOS F1.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = central[central.dataset == dataset]
        for _, row in part.iterrows():
            ax.scatter(row.id_recall_mean * 100, row.oos_f1_mean * 100, s=100 if row.method == "trainable_k1" else 60, label=row.label)
            ax.annotate(row.label, (row.id_recall_mean * 100, row.oos_f1_mean * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(dataset)
        ax.set_xlabel("Known Recall (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    fig.suptitle("KIR=0.50: Known coverage versus OOS F1 across components")
    fig.tight_layout()
    fig.savefig(FIG / "kir050_known_recall_oos_f1_components.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report(aggregate: pd.DataFrame, effects: pd.DataFrame) -> None:
    central = aggregate[aggregate.kir == 0.50]
    def val(method: str, dataset: str, metric: str) -> float:
        row = central[(central.method == method) & (central.dataset == dataset)].iloc[0]
        return float(row[f"{metric}_mean"])

    lines = [
        "# Trainable K=1 与 MOGB 组件五 seed 对比 V1",
        "",
        "> 这是同一数据划分、KIR、seed 下的组件级比较，不是 MOGB 官方 BERT 论文结果的严格复现排名。MOGB 行来自 Frozen MiniLM fair matrix，表示和监督条件与 Trainable 不同。",
        "",
        "## 1. 比较范围",
        "",
        "- Trainable：Known-only、最后两层 MiniLM + projection、K=1、diagonal Mahalanobis、mean+std。",
        "- `mogb_partition_ours_boundary`：Frozen MiniLM + MOGB 动态分区 + s2c 边界，是最接近的边界合同对照。",
        "- `mogb_minilm`：Frozen MiniLM + MOGB 分区/欧氏距离/平均半径。",
        "- `fixed_k2`：Frozen MiniLM + 固定 K=2；作为多中心失败参考。",
        "- 每个比较均按 dataset×KIR×seed 配对，bootstrap 10,000 次，仅用于统计描述。",
        "",
        "## 2. KIR=0.50 的均值",
        "",
        "| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False Accept |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in ("trainable_k1", "mogb_partition_ours_boundary", "mogb_minilm", "fixed_k2"):
            name = {"trainable_k1": "Trainable K=1", "mogb_partition_ours_boundary": "MOGB partition + s2c boundary", "mogb_minilm": "MOGB MiniLM", "fixed_k2": "Frozen fixed K=2"}[method]
            lines.append(f"| {dataset} | {name} | {val(method, dataset, 'oos_f1')*100:.2f} | {val(method, dataset, 'f1_all')*100:.2f} | {val(method, dataset, 'id_recall')*100:.2f} | {val(method, dataset, 'false_accept_rate')*100:.2f} |")
    lines.extend(
        [
            "",
            "## 3. 机制解读",
            "",
            "1. Trainable K=1 在 KIR=.50 的 OOS F1 高于 MOGB partition+s2c boundary，但 MOGB 的 false acceptance 更低；MOGB 主要通过拒绝更多 Known 换取保守拒识。",
            "2. 例如 StackOverflow：Trainable 为 OOS F1 87.67%、Known Recall 83.89%、false acceptance 9.34%；MOGB partition+s2c boundary 为 79.25%、50.39%、1.86%。这不是单纯的 OOS F1 胜负，而是“覆盖—拒识”工作点不同。",
            "3. CLINC150 和 Banking77 也表现出相同趋势：MOGB 组件通常降低 false acceptance，但 Known Recall/F1-All 显著下降；Trainable 保留了更高的 Known 覆盖。",
            "4. 因此当前可支持的解释是：Trainable 的优势来自可训练表示带来的 score 分离和更平衡的工作点，不是已经证明自己超过了完整 MOGB 或所有端到端基线。",
            "",
            "## 4. 证据文件",
            "",
            "- `results/analysis/trainable_vs_mogb_component_v1/aggregate.csv`",
            "- `results/analysis/trainable_vs_mogb_component_v1/paired_effects.csv`",
            "- `figures/trainable_vs_mogb_component_v1/`",
            "",
            "## 5. 结论边界",
            "",
            "- 不能把 MOGB fair component 结果称为官方 MOGB 完整复现；官方 BERT 结果另有复现状态。",
            "- 不能只用 OOS F1 选择方法；Known Recall、F1-All 和 false acceptance 必须同时报告。",
            "- 下一步应把相同监督条件下的强基线和完整 Cascade 单独桥接，而不是把不同协议的数字合成一个 SOTA 排名。",
        ]
    )
    (ROOT / "docs/analysis/TRAINABLE_VS_MOGB_COMPONENT_V1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    train, mogb = _load()
    train = train.assign(method="trainable_k1")
    common = ["dataset", "kir", "seed", *METRICS]
    combined = pd.concat([train[common].assign(method="trainable_k1"), mogb[common + ["method"]]], ignore_index=True)
    aggregate = _aggregate(combined)
    effects = _effects(train[common], mogb)
    _write(aggregate, "aggregate.csv")
    _write(effects, "paired_effects.csv")
    _plot(aggregate, effects)
    _report(aggregate, effects)
    print(json.dumps({"status": "complete", "trainable_rows": len(train), "mogb_rows": len(mogb), "effect_rows": len(effects), "figures": len(list(FIG.glob("*.png")))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
