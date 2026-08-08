"""Summarize existing StackOverflow K=1/K=2 intent diagnostics.

This is an analysis-only artifact builder.  It never reads test text or
re-runs a detector; it consumes the immutable RACAL stage-2 diagnostic CSV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "diagnostics" / "racal_v1" / "stage2_fixed_k2" / "RACAL_V1_STAGE2_INTENT_DIAGNOSTICS.csv"
OUT = ROOT / "results" / "analysis" / "stackoverflow_intent_diagnostic_v1"
FIG = ROOT / "figures" / "stackoverflow_intent_diagnostic_v1"
REPORT = ROOT / "docs" / "analysis" / "STACKOVERFLOW_INTENT_MULTI_CENTER_DIAGNOSTIC_V1.md"


def load() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    required = {
        "dataset",
        "kir",
        "seed",
        "intent",
        "bootstrap_ari_mean",
        "silhouette",
        "known_reject_count_k1",
        "known_reject_count_k2",
        "known_recall_delta",
        "newly_accepted_oos_count",
        "recovered_known_count",
        "net_benefit_recovered_known_minus_new_oos",
        "k2_cluster_1_sample_count",
        "k2_cluster_2_sample_count",
        "k2_cluster_1_radius",
        "k2_cluster_2_radius",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing diagnostic columns: {missing}")
    if set(df["dataset"].astype(str)) != {"stackoverflow"}:
        raise ValueError("This report is restricted to StackOverflow diagnostics")
    df["cluster_size_imbalance"] = (
        df[["k2_cluster_1_sample_count", "k2_cluster_2_sample_count"]].max(axis=1)
        / df[["k2_cluster_1_sample_count", "k2_cluster_2_sample_count"]].min(axis=1)
    )
    df["radius_ratio"] = (
        df[["k2_cluster_1_radius", "k2_cluster_2_radius"]].max(axis=1)
        / df[["k2_cluster_1_radius", "k2_cluster_2_radius"]].min(axis=1)
    )
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("intent", as_index=False)
        .agg(
            diagnostic_rows=("seed", "count"),
            seed_count=("seed", "nunique"),
            mean_newly_accepted_oos=("newly_accepted_oos_count", "mean"),
            mean_recovered_known=("recovered_known_count", "mean"),
            mean_net_benefit=("net_benefit_recovered_known_minus_new_oos", "mean"),
            mean_ari=("bootstrap_ari_mean", "mean"),
            mean_silhouette=("silhouette", "mean"),
            mean_known_recall_delta=("known_recall_delta", "mean"),
            mean_cluster_imbalance=("cluster_size_imbalance", "mean"),
            mean_radius_ratio=("radius_ratio", "mean"),
            mean_k1_reject=("known_reject_count_k1", "mean"),
            mean_k2_reject=("known_reject_count_k2", "mean"),
        )
        .sort_values("mean_net_benefit")
    )


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bootstrap_ari_mean",
        "silhouette",
        "cluster_size_imbalance",
        "radius_ratio",
        "newly_accepted_oos_count",
        "recovered_known_count",
        "net_benefit_recovered_known_minus_new_oos",
        "known_recall_delta",
    ]
    rows = []
    for left in cols:
        for right in cols:
            rows.append({"x": left, "y": right, "spearman": df[left].corr(df[right], method="spearman")})
    return pd.DataFrame(rows)


def save_figures(summary: pd.DataFrame, df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ranked = summary.sort_values("mean_net_benefit")

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#d62728" if x < 0 else "#2ca02c" for x in ranked["mean_net_benefit"]]
    ax.barh(ranked["intent"], ranked["mean_net_benefit"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Recovered Known - newly accepted OOS (mean count)")
    ax.set_title("StackOverflow: per-intent cost of fixed K=2")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "intent_net_benefit.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(df["bootstrap_ari_mean"], df["newly_accepted_oos_count"], c=df["known_recall_delta"], cmap="coolwarm", s=55, alpha=0.85)
    ax.set_xlabel("Bootstrap ARI")
    ax.set_ylabel("Newly accepted OOS count")
    ax.set_title("Stability does not imply OOS safety")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Known Recall delta")
    fig.tight_layout()
    fig.savefig(FIG / "ari_vs_new_oos.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ranked2 = summary.sort_values("mean_newly_accepted_oos", ascending=False)
    x = range(len(ranked2))
    ax.bar(x, ranked2["mean_newly_accepted_oos"], label="new OOS accepted", color="#d62728")
    ax.bar(x, ranked2["mean_recovered_known"], label="Known recovered", color="#2ca02c", alpha=0.8)
    ax.set_xticks(list(x), ranked2["intent"], rotation=45, ha="right")
    ax.set_ylabel("Mean sample count")
    ax.set_title("StackOverflow: recovered Known versus newly accepted OOS")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "intent_recovered_vs_oos.png", dpi=180)
    plt.close(fig)

    pivot = ranked.pivot_table(index="intent", values=["mean_ari", "mean_silhouette", "mean_cluster_imbalance", "mean_net_benefit"])
    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.35)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)), ["ARI", "silhouette", "size imbalance", "net benefit"], rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("StackOverflow intent diagnostics")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "intent_diagnostic_heatmap.png", dpi=180)
    plt.close(fig)


def write_report(df: pd.DataFrame, summary: pd.DataFrame, corr: pd.DataFrame) -> None:
    overall = df[["newly_accepted_oos_count", "recovered_known_count", "net_benefit_recovered_known_minus_new_oos", "bootstrap_ari_mean", "known_recall_delta"]].mean()
    worst = summary.head(5)[["intent", "mean_newly_accepted_oos", "mean_recovered_known", "mean_net_benefit", "mean_ari"]]
    best = summary.tail(5).sort_values("mean_net_benefit", ascending=False)[["intent", "mean_newly_accepted_oos", "mean_recovered_known", "mean_net_benefit", "mean_ari"]]
    report = f"""# StackOverflow 固定 K=2 按意图机制诊断

更新时间：2026-08-06  
协议：`protocol_v2_textoir_v1`  
来源：`RACAL_V1_STAGE2_INTENT_DIAGNOSTICS.csv`；本报告只做已有诊断汇总，不重新运行实验。

## 结论

在当前诊断样本中，K=2 平均每个意图恢复约 `{overall['recovered_known_count']:.2f}` 个 Known 样本，却新增接受约 `{overall['newly_accepted_oos_count']:.2f}` 个 OOS 样本，净收益为 `{overall['net_benefit_recovered_known_minus_new_oos']:.2f}`。因此，StackOverflow 的固定多中心问题不是聚类不稳定：平均 bootstrap ARI 为 `{overall['bootstrap_ari_mean']:.2f}`，而是稳定的子簇划分仍然把更多 OOS 带入接受区域。

这与“增加中心可以覆盖更多 Known”同时发生：平均 Known Recall 变化为 `{overall['known_recall_delta']:.3f}`，但新增 OOS 的代价更大。稳定性指标不能作为接受 K>1 的充分条件。

## 最差的意图

| intent | 新增 OOS | 恢复 Known | 净收益 | ARI |
|---|---:|---:|---:|---:|
"""
    for _, row in worst.iterrows():
        report += f"| {row['intent']} | {row['mean_newly_accepted_oos']:.1f} | {row['mean_recovered_known']:.1f} | {row['mean_net_benefit']:.1f} | {row['mean_ari']:.2f} |\n"
    report += "\n## 相对安全的意图（仅限本诊断样本）\n\n| intent | 新增 OOS | 恢复 Known | 净收益 | ARI |\n|---|---:|---:|---:|---:|\n"
    for _, row in best.iterrows():
        report += f"| {row['intent']} | {row['mean_newly_accepted_oos']:.1f} | {row['mean_recovered_known']:.1f} | {row['mean_net_benefit']:.1f} | {row['mean_ari']:.2f} |\n"
    report += """
## 解释边界

- 诊断文件只有当前 RACAL Stage-2 选取的 30 个 intent/seed 记录，不代表 StackOverflow 全部 20 个 intent；不能把该表写成全数据集的逐意图定律。
- `newly_accepted_oos_count` 是 K=2 相对于 K=1 的新增 OOS 接受量，直接用于解释 false-accept 代价；它不是 OOS F1 本身。
- ARI、silhouette、簇规模和半径只描述 Known 训练结构，没有使用 test OOS 选择参数。
- 这些结果支持“固定 K=2 在 StackOverflow 存在接受并集风险”，不支持“所有多中心方法都必然失败”。

## 原始证据与图

- [`RACAL_V1_STAGE2_INTENT_DIAGNOSTICS.csv`](../../results/diagnostics/racal_v1/stage2_fixed_k2/RACAL_V1_STAGE2_INTENT_DIAGNOSTICS.csv)
- [`intent_diagnostic_summary.csv`](../../results/analysis/stackoverflow_intent_diagnostic_v1/intent_diagnostic_summary.csv)
- [`intent_net_benefit.png`](../../figures/stackoverflow_intent_diagnostic_v1/intent_net_benefit.png)
- [`ari_vs_new_oos.png`](../../figures/stackoverflow_intent_diagnostic_v1/ari_vs_new_oos.png)
- [`intent_recovered_vs_oos.png`](../../figures/stackoverflow_intent_diagnostic_v1/intent_recovered_vs_oos.png)
- [`intent_diagnostic_heatmap.png`](../../figures/stackoverflow_intent_diagnostic_v1/intent_diagnostic_heatmap.png)

## Spearman 相关

相关矩阵已保存为 [`intent_diagnostic_correlations.csv`](../../results/analysis/stackoverflow_intent_diagnostic_v1/intent_diagnostic_correlations.csv)，仅作探索性机制分析，未进行多重比较校正。
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    summary = summarize(df)
    corr = correlations(df)
    df.to_csv(OUT / "intent_diagnostic_rows.csv", index=False)
    summary.to_csv(OUT / "intent_diagnostic_summary.csv", index=False)
    corr.to_csv(OUT / "intent_diagnostic_correlations.csv", index=False)
    save_figures(summary, df)
    write_report(df, summary, corr)
    print(f"rows={len(df)} intents={summary['intent'].nunique()} figures={len(list(FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
