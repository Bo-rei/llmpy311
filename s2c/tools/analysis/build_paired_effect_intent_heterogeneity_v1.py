"""Build paired-effect and intent-heterogeneity evidence from completed rows.

This stage is analysis-only.  It reuses the protocol_v2_textoir_v1 light
summaries already produced by the fair component runs and the intent-level
oracle diagnostic.  It does not train, select a model, tune a boundary, or
rewrite any historical artifact.

The first part keeps the split and data seed paired and reports Trainable K=1
minus each frozen/component comparator with deterministic bootstrap intervals.
The second part describes how often an intent-level test oracle prefers a
multi-centre configuration.  The latter is explicitly marked diagnostic and
must never be used as a deployment selector.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2c-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "paired_effect_intent_heterogeneity_v1"
FIG = ROOT / "figures" / "paired_effect_intent_heterogeneity_v1"
REPORT = ROOT / "docs" / "analysis" / "PAIRED_EFFECT_INTENT_HETEROGENEITY_V1.md"

FAIR_INPUT = ROOT / "results" / "analysis" / "cross_protocol_tradeoff_v1" / "per_seed.csv"
INTENT_INPUT = ROOT / "results" / "analysis" / "intent_kir_stability_pack_v1" / "intent_kir_rows.csv"

DATASET_ORDER = ["clinc150", "banking77", "stackoverflow"]
DATASET_LABELS = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}
KIR_ORDER = [0.25, 0.50, 0.75]
COMPARATOR_ORDER = [
    "single_centroid",
    "random_partition",
    "fixed_k2",
    "mogb_minilm",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
]
COMPARATOR_LABELS = {
    "single_centroid": "Frozen K=1",
    "random_partition": "Random K=2",
    "fixed_k2": "Frozen K=2",
    "mogb_minilm": "MOGB-MiniLM",
    "mogb_partition_ours_boundary": "MOGB partition + s2c boundary",
    "ours_partition_mogb_boundary": "s2c partition + MOGB boundary",
}
METRICS = ["oos_f1", "f1_all", "known_recall", "false_accept_rate"]
METRIC_LABELS = {
    "oos_f1": "OOS F1",
    "f1_all": "F1-All",
    "known_recall": "Known Recall",
    "false_accept_rate": "False acceptance",
}
METRIC_COLORS = {"oos_f1": "#0072B2", "f1_all": "#009E73", "known_recall": "#D55E00", "false_accept_rate": "#CC79A7"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, format=path.suffix.lstrip("."), dpi=180, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(path)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def load_fair() -> pd.DataFrame:
    frame = pd.read_csv(FAIR_INPUT)
    needed = {"dataset", "kir", "seed", "method", *METRICS}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"fair input missing columns: {sorted(missing)}")
    frame = frame.loc[frame["method"].isin(["trainable_k1", *COMPARATOR_ORDER])].copy()
    frame["kir"] = frame["kir"].astype(float)
    frame["seed"] = frame["seed"].astype(int)
    return frame


def paired_effects(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = frame.loc[frame["method"].eq("trainable_k1"), ["dataset", "kir", "seed", *METRICS]].copy()
    rows: list[dict[str, object]] = []
    for comparator in COMPARATOR_ORDER:
        comp = frame.loc[frame["method"].eq(comparator), ["dataset", "kir", "seed", *METRICS]].copy()
        merged = base.merge(comp, on=["dataset", "kir", "seed"], suffixes=("_trainable", "_comparator"))
        if merged.empty:
            raise ValueError(f"missing paired rows for {comparator}")
        for _, item in merged.iterrows():
            row = {
                "dataset": item["dataset"],
                "kir": float(item["kir"]),
                "seed": int(item["seed"]),
                "comparator": comparator,
                "comparator_label": COMPARATOR_LABELS[comparator],
            }
            for metric in METRICS:
                row[f"trainable_{metric}"] = float(item[f"{metric}_trainable"])
                row[f"comparator_{metric}"] = float(item[f"{metric}_comparator"])
                row[f"delta_{metric}"] = float(item[f"{metric}_trainable"] - item[f"{metric}_comparator"])
            rows.append(row)
    paired = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (dataset, kir, comparator), group in paired.groupby(["dataset", "kir", "comparator"], sort=False):
        for metric_index, metric in enumerate(METRICS):
            values = group[f"delta_{metric}"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(values, seed=20260808 + metric_index + int(round(float(kir) * 100)) + len(summary_rows))
            summary_rows.append(
                {
                    "dataset": dataset,
                    "kir": float(kir),
                    "comparator": comparator,
                    "comparator_label": COMPARATOR_LABELS[comparator],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n_seeds": int(values.size),
                    "mean_delta": float(values.mean()),
                    "std_delta": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "median_delta": float(np.median(values)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wins": int((values > 1e-12).sum()),
                    "ties": int((np.abs(values) <= 1e-12).sum()),
                    "losses": int((values < -1e-12).sum()),
                }
            )
    return paired, pd.DataFrame(summary_rows)


def load_intent() -> pd.DataFrame:
    frame = pd.read_csv(INTENT_INPUT)
    required = {"dataset", "kir", "distance", "intent", "seed", "best_k", "delta_oos_f1_vs_k1", "delta_known_recall_vs_k1", "safe_gain_oracle", "multi_center_oracle", "best_k_false_accept_rate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"intent input missing columns: {sorted(missing)}")
    frame = frame.loc[frame["kir"].astype(float).isin(KIR_ORDER)].copy()
    frame["kir"] = frame["kir"].astype(float)
    frame["best_k"] = frame["best_k"].astype(int)
    frame["safe_gain_oracle"] = frame["safe_gain_oracle"].astype(bool)
    frame["multi_center_oracle"] = frame["multi_center_oracle"].astype(bool)
    return frame


def intent_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (dataset, kir, distance), group in frame.groupby(["dataset", "kir", "distance"], sort=False):
        rows.append(
            {
                "dataset": dataset,
                "kir": float(kir),
                "distance": distance,
                "intent_seed_cells": int(len(group)),
                "safe_gain_rate": float(group["safe_gain_oracle"].mean()),
                "multi_center_oracle_rate": float(group["multi_center_oracle"].mean()),
                "mean_best_k": float(group["best_k"].mean()),
                "median_best_k": float(group["best_k"].median()),
                "mean_delta_oos_f1": float(group["delta_oos_f1_vs_k1"].mean()),
                "mean_delta_known_recall": float(group["delta_known_recall_vs_k1"].mean()),
                "mean_best_k_false_accept_rate": float(group["best_k_false_accept_rate"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    distribution = (
        frame.groupby(["dataset", "kir", "distance", "best_k"], as_index=False).size().rename(columns={"size": "count"})
    )
    distribution["proportion"] = distribution["count"] / distribution.groupby(["dataset", "kir", "distance"])["count"].transform("sum")
    return summary, distribution


def plot_forest(summary: pd.DataFrame, path: Path) -> None:
    frame = summary.loc[summary["metric"].eq("oos_f1")].copy()
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(18, 8), sharex=True)
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = frame.loc[frame["dataset"].eq(dataset)].copy()
        sub["order"] = [KIR_ORDER.index(float(k)) * len(COMPARATOR_ORDER) + COMPARATOR_ORDER.index(c) for k, c in zip(sub["kir"], sub["comparator"])]
        sub = sub.sort_values("order")
        y = np.arange(len(sub))
        x = sub["mean_delta"].to_numpy() * 100
        low = x - sub["ci95_low"].to_numpy() * 100
        high = sub["ci95_high"].to_numpy() * 100 - x
        colors = ["#0072B2" if value >= 0 else "#D55E00" for value in x]
        ax.errorbar(x, y, xerr=[low, high], fmt="none", ecolor="#555555", elinewidth=1.1, capsize=2, zorder=1)
        ax.scatter(x, y, c=colors, s=38, zorder=2)
        ax.axvline(0, color="#333333", linewidth=0.9)
        labels = [f"{kir:.2f} | {COMPARATOR_LABELS[c]}" for kir, c in zip(sub["kir"], sub["comparator"])]
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_title(DATASET_LABELS[dataset])
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_ylabel("KIR | comparator")
    for ax in axes:
        ax.set_xlabel("Trainable K=1 - comparator OOS F1 (pp)")
    fig.suptitle("配对 OOS F1 效应：同一 dataset×KIR×seed，95% bootstrap CI")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    atomic_figure(fig, path)


def plot_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    keep = summary.loc[summary["kir"].eq(0.50)].copy()
    keep["row"] = keep.apply(lambda row: f"{DATASET_LABELS[row['dataset']]} / {row['comparator_label']}", axis=1)
    rows = [f"{DATASET_LABELS[d]} / {COMPARATOR_LABELS[c]}" for d in DATASET_ORDER for c in COMPARATOR_ORDER]
    columns = ["oos_f1", "f1_all", "known_recall", "false_accept_rate"]
    matrix = keep.pivot(index="row", columns="metric", values="mean_delta").reindex(index=rows, columns=columns)
    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(matrix.to_numpy(dtype=float) * 100, aspect="auto", cmap="RdBu_r", vmin=-35, vmax=35)
    ax.set_yticks(np.arange(len(matrix)), matrix.index, fontsize=8)
    ax.set_xticks(np.arange(len(columns)), [METRIC_LABELS[c] + "\nTrainable-comp." for c in columns])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value * 100:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="差值（百分点）")
    ax.set_title("KIR=0.50 的配对效应：OOS 改善是否伴随 Known/FA 代价")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_safe_gain(summary: pd.DataFrame, path: Path) -> None:
    frame = summary.copy()
    frame["row"] = frame.apply(lambda row: f"{DATASET_LABELS[row['dataset']]}\n{row['distance']}", axis=1)
    rows = [f"{DATASET_LABELS[d]}\n{distance}" for d in DATASET_ORDER for distance in ["euclidean", "mahalanobis_diag"]]
    columns = [f"KIR={kir:.2f}" for kir in KIR_ORDER]
    frame["column"] = frame["kir"].map(lambda value: f"KIR={value:.2f}")
    matrix = frame.pivot(index="row", columns="column", values="safe_gain_rate").reindex(index=rows, columns=columns)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix.to_numpy(dtype=float) * 100, aspect="auto", cmap="YlGn", vmin=0, vmax=100)
    ax.set_yticks(np.arange(len(matrix)), matrix.index, fontsize=8)
    ax.set_xticks(np.arange(len(columns)), columns)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value * 100:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="intent×seed 单元中 safe_gain_oracle 比例（%）")
    ax.set_title("逐意图多中心安全收益的异质性\n（测试 oracle 诊断，不是选择规则）")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_best_k_distribution(distribution: pd.DataFrame, path: Path) -> None:
    frame = distribution.loc[distribution["distance"].eq("euclidean")].copy()
    frame["group"] = frame.apply(lambda row: f"{DATASET_LABELS[row['dataset']]}\n{row['kir']:.2f}", axis=1)
    groups = [f"{DATASET_LABELS[d]}\n{kir:.2f}" for d in DATASET_ORDER for kir in KIR_ORDER]
    fig, ax = plt.subplots(figsize=(12, 5))
    bottoms = np.zeros(len(groups))
    colors = {1: "#999999", 2: "#56B4E9", 3: "#009E73", 4: "#E69F00", 5: "#D55E00"}
    for k in range(1, 6):
        values = []
        for group in groups:
            sub = frame.loc[(frame["group"].eq(group)) & (frame["best_k"].eq(k)), "proportion"]
            values.append(float(sub.iloc[0]) if not sub.empty else 0.0)
        ax.bar(np.arange(len(groups)), values, bottom=bottoms, color=colors[k], label=f"oracle best K={k}")
        bottoms += np.asarray(values)
    ax.set_xticks(np.arange(len(groups)), groups, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("intent×seed 比例")
    ax.set_title("逐意图 oracle-best-K 分布（Euclidean）")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    atomic_figure(fig, path)


def write_report(paired_summary: pd.DataFrame, intent_summary_frame: pd.DataFrame, distribution: pd.DataFrame, manifest: dict[str, object]) -> None:
    trainable = paired_summary.loc[paired_summary["comparator"].eq("single_centroid") & paired_summary["metric"].eq("oos_f1")]
    safe = intent_summary_frame.loc[intent_summary_frame["distance"].eq("euclidean")]
    trainable_lines = []
    for _, row in trainable.iterrows():
        trainable_lines.append(f"- {DATASET_LABELS[row['dataset']]} KIR={row['kir']:.2f}：相对 Frozen K=1 的 OOS F1 差值 {row['mean_delta'] * 100:+.2f}pp，95% CI [{row['ci95_low'] * 100:+.2f}, {row['ci95_high'] * 100:+.2f}]。")
    safe_lines = []
    for _, row in safe.iterrows():
        safe_lines.append(f"- {DATASET_LABELS[row['dataset']]} KIR={row['kir']:.2f}：safe_gain_oracle 比例 {row['safe_gain_rate'] * 100:.1f}%，平均 best K={row['mean_best_k']:.2f}，平均 ΔOOS F1={row['mean_delta_oos_f1'] * 100:+.2f}pp，平均 ΔKnown Recall={row['mean_delta_known_recall'] * 100:+.2f}pp。")
    text = "\n".join(
        [
            "# 配对效应与 intent-level 多中心异质性 V1",
            "",
            "更新时间：2026-08-08",
            "活动协议：`protocol_v2_textoir_v1`",
            "证据类型：analysis-only；不训练、不调参、不覆盖历史 artifact。",
            "",
            "## 做了什么",
            "",
            "1. 对同一 `dataset × KIR × seed` 配对 Trainable K=1 与六个 Frozen/MOGB 组件，计算 OOS F1、F1-All、Known Recall 和 false acceptance 的差值。",
            "2. 使用固定 RNG seed、10,000 次 bootstrap 生成配对均值的 95% CI，并输出 win/tie/loss。",
            "3. 从已有 intent-level test-oracle 诊断中统计不同数据集/KIR/距离下的多中心安全收益比例和 oracle-best-K 分布。",
            "",
            "## Trainable K=1 对 Frozen K=1 的配对结果",
            "",
            *trainable_lines,
            "",
            "这一步只说明表示适配在同一 split/seed 下的后验差异，不把结果解释成外部 SOTA。",
            "",
            "## intent-level 异质性（Euclidean）",
            "",
            *safe_lines,
            "",
            "`safe_gain_oracle` 和 `oracle-best-K` 依赖测试标签，只能用于解释研究空间，不能用于正式选择中心数、半径或阈值。它们的价值是说明：多中心潜在收益集中在部分 intent，而不是整个数据集统一受益。",
            "",
            "## 图表与数据",
            "",
            "- `figures/archive/analysis/paired_effect_intent_heterogeneity_v1/paired_oos_f1_forest.png`：配对 OOS F1 及 95% bootstrap CI。",
            "- `figures/archive/analysis/paired_effect_intent_heterogeneity_v1/kir50_tradeoff_heatmap.png`：KIR=0.50 下四项核心指标的配对差值。",
            "- `figures/archive/analysis/paired_effect_intent_heterogeneity_v1/safe_gain_heatmap.png`：逐意图安全收益比例。",
            "- `figures/archive/analysis/paired_effect_intent_heterogeneity_v1/oracle_best_k_distribution.png`：Euclidean 下 oracle-best-K 分布。",
            "- `results/analysis/archive/analysis/paired_effect_intent_heterogeneity_v1/paired_effects.csv`、`paired_summary.csv`、`intent_summary.csv`、`best_k_distribution.csv`。",
            "",
            "## 解释边界",
            "",
            "- 该阶段没有新增训练或测试选择；所有输入来自已完成协议结果。",
            "- 外部 ADB、DA-ADB、DCLOOS 不在本阶段被强行并入同协议统计。",
            "- 配对 CI 反映五个正式 seed 的不确定性，不替代同监督外部基线复现。",
            "",
            f"输入与输出 SHA256 见 `results/analysis/archive/analysis/paired_effect_intent_heterogeneity_v1/MANIFEST.json`。生成摘要：{json.dumps(manifest, ensure_ascii=False, sort_keys=True)}",
        ]
    )
    atomic_text(text, REPORT)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fair = load_fair()
    intent = load_intent()
    paired, paired_summary = paired_effects(fair)
    intent_summary_frame, distribution = intent_summary(intent)
    atomic_csv(paired, OUT / "paired_effects.csv")
    atomic_csv(paired_summary, OUT / "paired_summary.csv")
    atomic_csv(intent_summary_frame, OUT / "intent_summary.csv")
    atomic_csv(distribution, OUT / "best_k_distribution.csv")
    plot_forest(paired_summary, FIG / "paired_oos_f1_forest.png")
    plot_tradeoff(paired_summary, FIG / "kir50_tradeoff_heatmap.png")
    plot_safe_gain(intent_summary_frame, FIG / "safe_gain_heatmap.png")
    plot_best_k_distribution(distribution, FIG / "oracle_best_k_distribution.png")
    manifest: dict[str, object] = {
        "schema_version": "paired_effect_intent_heterogeneity_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "analysis_only": True,
        "bootstrap_replicates": 10000,
        "bootstrap_seed_base": 20260808,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in [FAIR_INPUT, INTENT_INPUT]},
        "outputs": {},
        "paired_rows": int(len(paired)),
        "paired_summary_rows": int(len(paired_summary)),
        "intent_rows": int(len(intent)),
        "intent_summary_rows": int(len(intent_summary_frame)),
        "best_k_distribution_rows": int(len(distribution)),
    }
    for path in [OUT / "paired_effects.csv", OUT / "paired_summary.csv", OUT / "intent_summary.csv", OUT / "best_k_distribution.csv"]:
        manifest["outputs"][str(path.relative_to(ROOT))] = sha256(path)
    atomic_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", OUT / "MANIFEST.json")
    write_report(paired_summary, intent_summary_frame, distribution, manifest)
    print(json.dumps({"status": "ok", "paired_rows": len(paired), "paired_summary_rows": len(paired_summary), "intent_summary_rows": len(intent_summary_frame), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
