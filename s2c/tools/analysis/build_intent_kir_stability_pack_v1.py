"""Aggregate the completed intent-level K/KIR sensitivity audit.

The source is explicitly test-sensitivity/oracle analysis.  This script only
summarizes completed rows and must not be used to choose a production K.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "diagnostics" / "adaptive_k" / "intent_level.csv"
OUT = ROOT / "results" / "analysis" / "intent_kir_stability_pack_v1"
FIG = ROOT / "figures" / "intent_kir_stability_pack_v1"
REPORT = ROOT / "docs" / "analysis" / "INTENT_KIR_STABILITY_PACK_V1.md"


def load() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    required = {
        "dataset",
        "kir",
        "distance",
        "intent",
        "seed",
        "best_k",
        "delta_oos_f1_vs_k1",
        "delta_known_recall_vs_k1",
        "best_k_false_accept_rate",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["kir_pct"] = df["kir"] * 100
    df["multi_center_oracle"] = df["best_k"] > 1
    df["safe_gain_oracle"] = (df["delta_oos_f1_vs_k1"] > 0) & (df["delta_known_recall_vs_k1"] >= -0.01)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "kir", "distance"], as_index=False)
        .agg(
            rows=("intent", "size"),
            intents=("intent", "nunique"),
            oracle_multi_center_rate=("multi_center_oracle", "mean"),
            oracle_safe_gain_rate=("safe_gain_oracle", "mean"),
            mean_best_k=("best_k", "mean"),
            median_best_k=("best_k", "median"),
            mean_delta_oos_f1=("delta_oos_f1_vs_k1", "mean"),
            mean_delta_known_recall=("delta_known_recall_vs_k1", "mean"),
            mean_false_accept_rate=("best_k_false_accept_rate", "mean"),
        )
    )


def mode_stability(df: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for keys, part in df.groupby(["dataset", "kir", "distance", "intent"]):
        counts = part["best_k"].value_counts()
        mode_k = int(counts.index[0])
        grouped.append(
            {
                "dataset": keys[0],
                "kir": keys[1],
                "distance": keys[2],
                "intent": keys[3],
                "seed_count": int(part["seed"].nunique()),
                "mode_best_k": mode_k,
                "mode_agreement": float(counts.iloc[0] / len(part)),
                "all_seeds_multi_center": bool((part["best_k"] > 1).all()),
                "all_seeds_k1": bool((part["best_k"] == 1).all()),
                "mean_delta_oos_f1": float(part["delta_oos_f1_vs_k1"].mean()),
                "mean_delta_known_recall": float(part["delta_known_recall_vs_k1"].mean()),
            }
        )
    return pd.DataFrame(grouped)


def save_figures(df: pd.DataFrame, summ: pd.DataFrame, stable: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ordered = ["clinc150", "banking77", "stackoverflow"]
    pivot = summ.pivot_table(index=["dataset", "distance"], columns="kir", values="oracle_multi_center_rate")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)), [f"{d} / {dist}" for d, dist in pivot.index])
    ax.set_xticks(range(len(pivot.columns)), [f"KIR={int(k*100)}%" for k in pivot.columns], rotation=35, ha="right")
    ax.set_title("Oracle intent-level rate with best K>1 (test-sensitivity audit)")
    fig.colorbar(im, ax=ax, label="rate")
    fig.tight_layout()
    fig.savefig(FIG / "oracle_multicenter_rate_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, dataset in zip(axes, ordered):
        part = df[df["dataset"].eq(dataset)]
        for distance, line in part.groupby("distance"):
            grouped = line.groupby("kir")["delta_oos_f1_vs_k1"].mean().sort_index() * 100
            ax.plot(grouped.index * 100, grouped.values, marker="o", label=distance)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(dataset)
        ax.set_xlabel("KIR (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Oracle best-K OOS F1 delta vs K=1 (pp)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Intent-level oracle multi-center gain changes with KIR")
    fig.tight_layout()
    fig.savefig(FIG / "oracle_gain_by_kir.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for dataset, part in df.groupby("dataset"):
        ax.scatter(part["delta_known_recall_vs_k1"] * 100, part["delta_oos_f1_vs_k1"] * 100, s=12, alpha=0.25, label=dataset)
        means = part.groupby("kir")[["delta_known_recall_vs_k1", "delta_oos_f1_vs_k1"]].mean()
        ax.plot(means["delta_known_recall_vs_k1"] * 100, means["delta_oos_f1_vs_k1"] * 100, marker="o", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Known Recall delta (pp)")
    ax.set_ylabel("OOS F1 delta (pp)")
    ax.set_title("Intent-level K selection trade-off (oracle diagnostic)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "oracle_oos_known_tradeoff.png", dpi=180)
    plt.close(fig)

    stable_rate = stable.groupby(["dataset", "kir"], as_index=False).agg(
        intents=("intent", "size"),
        all_seed_k1=("all_seeds_k1", "mean"),
        all_seed_multi=("all_seeds_multi_center", "mean"),
        mean_mode_agreement=("mode_agreement", "mean"),
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for dataset, part in stable_rate.groupby("dataset"):
        ax.plot(part["kir"] * 100, part["all_seed_multi"] * 100, marker="o", label=dataset)
    ax.set_xlabel("KIR (%)")
    ax.set_ylabel("Intents with K>1 for every seed (%)")
    ax.set_title("Cross-seed stability of oracle multi-center selection")
    ax.set_xticks(sorted(df["kir_pct"].unique()))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "oracle_multicenter_seed_stability.png", dpi=180)
    plt.close(fig)


def write_report(df: pd.DataFrame, summ: pd.DataFrame, stable: pd.DataFrame) -> None:
    overall = df.groupby("dataset").agg(
        rows=("intent", "size"),
        oracle_multi_center_rate=("multi_center_oracle", "mean"),
        oracle_safe_gain_rate=("safe_gain_oracle", "mean"),
        mean_delta_oos=("delta_oos_f1_vs_k1", "mean"),
        mean_delta_recall=("delta_known_recall_vs_k1", "mean"),
    )
    text = """# 意图级 KIR 稳定性与多中心收益分析

更新时间：2026-08-06  
协议：`protocol_v2_textoir_v1`  
性质：analysis-only；来源是已完成的 `adaptive_k/intent_level.csv` oracle/test-sensitivity 审计，不用于正式选择 K。

## 结论

数据集级别的 KIR/K 曲线会掩盖意图异质性：同一数据集内只有部分意图在 oracle 口径下从 K>1 获益，且这种选择通常不能跨 seed 稳定重复。该结果支持“每个意图的几何结构不同”，但不等于已经构造出一个无泄漏的 adaptive-K 规则。

| 数据集 | 诊断行数 | oracle 中 K>1 比例 | 同时 OOS 增益且 Known Recall 下降≤1pp 的比例 | 平均 OOS F1 变化 | 平均 Known Recall 变化 |
|---|---:|---:|---:|---:|---:|
"""
    for dataset, row in overall.iterrows():
        text += f"| {dataset} | {int(row['rows'])} | {row['oracle_multi_center_rate']*100:.1f}% | {row['oracle_safe_gain_rate']*100:.1f}% | {row['mean_delta_oos']*100:.2f}pp | {row['mean_delta_recall']*100:.2f}pp |\n"
    text += """
## 如何解读

- `best_k` 是利用测试敏感性审计得到的 oracle 诊断量；不能用于训练、阈值、结构选择或正式主表。
- `safe_gain_oracle` 只是一个描述性筛选：OOS F1 增加且 Known Recall 降幅不超过 1pp；它不是已经验证的 calibration 规则。
- 真正的 adaptive-K 需要在 Known train/calibration 上预注册规则，并在冻结后只评估一次 test。
- 如果某意图的 `best_k` 在不同 seed 间反复变化，说明固定 K 或简单 oracle 选择不够稳定。

## 证据文件

- [`intent_level.csv`](../../results/diagnostics/adaptive_k/intent_level.csv)
- [`intent_kir_summary.csv`](../../results/analysis/intent_kir_stability_pack_v1/intent_kir_summary.csv)
- [`intent_kir_seed_stability.csv`](../../results/analysis/intent_kir_stability_pack_v1/intent_kir_seed_stability.csv)
- [`oracle_multicenter_rate_heatmap.png`](../../figures/intent_kir_stability_pack_v1/oracle_multicenter_rate_heatmap.png)
- [`oracle_gain_by_kir.png`](../../figures/intent_kir_stability_pack_v1/oracle_gain_by_kir.png)
- [`oracle_oos_known_tradeoff.png`](../../figures/intent_kir_stability_pack_v1/oracle_oos_known_tradeoff.png)
- [`oracle_multicenter_seed_stability.png`](../../figures/intent_kir_stability_pack_v1/oracle_multicenter_seed_stability.png)
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    summ = summary(df)
    stable = mode_stability(df)
    df.to_csv(OUT / "intent_kir_rows.csv", index=False)
    summ.to_csv(OUT / "intent_kir_summary.csv", index=False)
    stable.to_csv(OUT / "intent_kir_seed_stability.csv", index=False)
    save_figures(df, summ, stable)
    write_report(df, summ, stable)
    print(f"rows={len(df)} summary={len(summ)} intent_groups={len(stable)} figures={len(list(FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
