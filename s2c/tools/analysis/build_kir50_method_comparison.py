"""Build a protocol-separated KIR=0.50 method comparison from existing CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "kir50_method_comparison_v1"
FIG = ROOT / "figures" / "active_experiment_dashboard_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")


def _load() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    baseline = pd.read_csv(ROOT / "results/final_baselines/summary.csv")
    fair = baseline[
        baseline["kir"].eq(0.50)
        & baseline["scope"].eq("protocol_v2_fair_mean_over_5_seeds")
        & baseline["dataset"].isin(DATASETS)
    ].copy()
    fair["scope_group"] = "same protocol / frozen MiniLM / 5 seeds"
    fair["ranking_note"] = "same split/evaluator, frozen representation; fair component comparison"
    fair["n_seeds"] = 5
    fair["method_group"] = "frozen_fair_component"
    rows.append(fair)

    trainable = pd.read_csv(ROOT / "results/diagnostics/minilm_trainable_k2_control_v1/per_seed.csv")
    trainable_rows: list[dict[str, object]] = []
    for _, record in trainable.iterrows():
        for k in (1, 2):
            trainable_rows.append(
                {
                    "dataset": record["dataset"],
                    "kir": 0.50,
                    "method": f"Trainable K={k}",
                    "scope": "current_protocol_trainable_3seed",
                    "training_regime": "Known-only last2 MiniLM + projection",
                    "supervision": "Known-only",
                    "oos_f1": record[f"k{k}_oos_f1"],
                    "known_macro_f1": record[f"k{k}_f1_k"],
                    "known_recall": record[f"k{k}_known_recall"],
                    "f1_all": record[f"k{k}_f1_all"],
                    "accuracy": record[f"k{k}_accuracy"],
                    "scope_group": "same protocol / Trainable MiniLM / 3 seeds",
                    "ranking_note": "same split/evaluator; representation differs from frozen fair rows",
                    "n_seeds": 3,
                    "method_group": "current_trainable",
                }
            )
    rows.append(pd.DataFrame(trainable_rows))

    external = baseline[
        baseline["kir"].eq(0.50)
        & baseline["dataset"].isin(DATASETS)
        & baseline["method"].isin(
            [
                "ADB",
                "DA-ADB",
                "DCLOOS-official",
                "DCLOOS-unified",
                "DCLOOS-official (reduced-budget recovered)",
                "MOGB-official (compatibility)",
                "MOGB-official (strict single-cell)",
                "BRAK",
            ]
        )
    ].copy()
    external["scope_group"] = external["scope"].map(
        lambda value: "external compatibility / BERT or pseudo-OOS" if "official" in str(value) or "compatibility" in str(value) or "brak" in str(value) else "external compatibility"
    )
    external["ranking_note"] = "not direct ranking: different representation, supervision, seed count or data contract"
    external["n_seeds"] = external["scope"].map(lambda value: 5 if "mean_over_5" in str(value) else 1)
    external["method_group"] = "external_compatibility"
    rows.append(external)

    frame = pd.concat(rows, ignore_index=True, sort=False)
    keep = [
        "dataset",
        "kir",
        "method",
        "scope",
        "training_regime",
        "supervision",
        "oos_f1",
        "known_macro_f1",
        "known_recall",
        "f1_all",
        "accuracy",
        "scope_group",
        "ranking_note",
        "n_seeds",
        "method_group",
        "status",
        "source",
    ]
    for col in keep:
        if col not in frame:
            frame[col] = np.nan
    return frame[keep].sort_values(["dataset", "method_group", "method"]).reset_index(drop=True)


def _write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# KIR=0.50 方法对比（协议分层）",
        "",
        "本报告只整理已有结果，不重新训练。比较先按数据、表示、监督条件和 seed 数分层；不同合同的结果不合并为单一 SOTA 排名。",
        "",
        "## StackOverflow 当前协议结果",
        "",
        "| 方法 | OOS F1 | Known Macro-F1 | Known Recall | F1-All | Accuracy | 结果层 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    so = summary[summary["dataset"].eq("stackoverflow")]
    order = ["Trainable K=1", "Trainable K=2", "Single centroid", "Fixed K=2", "Random partition", "MOGB-MiniLM", "MOGB partition + s2c boundary", "ADB", "DA-ADB", "BRAK"]
    for method in order:
        hit = so[so["method"].eq(method)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        def fmt(col: str) -> str:
            value = pd.to_numeric(row.get(col), errors="coerce")
            return "—" if pd.isna(value) else f"{float(value) * 100:.2f}%"
        lines.append(f"| {method} | {fmt('oos_f1_mean')} | {fmt('known_macro_f1_mean')} | {fmt('known_recall_mean')} | {fmt('f1_all_mean')} | {fmt('accuracy_mean')} | {row['scope_group']} |")
    lines += [
        "",
        "## 解释",
        "",
        "- Trainable K=1 是当前协议下的 3-seed Known-only 表示训练结果；它相对当前 Frozen K=1 有稳定收益，但与 5-seed frozen fair rows 的 seed 数不同。",
        "- MOGB-MiniLM、固定 K、随机分簇和 MOGB 分区/边界替换属于同一 frozen MiniLM 组件层，可用于分析分区和半径贡献。",
        "- ADB/DA-ADB 是 BERT/兼容单格结果，BRAK 也使用不同实验合同；它们显示性能差距，但不能直接证明 Trainable MiniLM 方法优于或劣于这些方法。",
        "- DCLOOS/MOGB 官方行若缺少可比 OOS F1，则保留为复现/兼容性证据，不伪造排名。",
        "",
        "## 当前可支持的结论",
        "",
        "1. 在当前统一协议内部，Trainable MiniLM K=1 是比 Frozen K=1 更强的表示基线。",
        "2. Trainable K=2 在 StackOverflow 明显退化，不能把 K=1 表示收益归因于多中心。",
        "3. MOGB fair 结果能回答组件机制问题，但不能替代作者 BERT 原始协议复现。",
        "4. 要宣称超过 ADB/DA-ADB/DCLOOS，必须先统一表示、监督条件、known list、seed 和评价器；当前证据不足。",
        "",
        "## 文件",
        "",
        "- `results/analysis/kir50_method_comparison_v1/rows.csv`：逐方法分层表。",
        "- `results/analysis/kir50_method_comparison_v1/mean_std.csv`：按数据集/方法的轻量汇总。",
        "- `figures/active_experiment_dashboard_v1/kir50_method_layers.png`：协议分层柱状图。",
        "- `figures/active_experiment_dashboard_v1/kir50_method_tradeoff.png`：Known/OOS 权衡图。",
    ]
    (ROOT / "docs/analysis/KIR50_METHOD_COMPARISON_V1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame = _load()
    frame.to_csv(OUT / "rows.csv", index=False)
    numeric = ["oos_f1", "known_macro_f1", "known_recall", "f1_all", "accuracy"]
    summary = frame.groupby(["dataset", "method", "scope_group"], as_index=True)[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = ["dataset", "method", "scope_group", *[f"{metric}_{stat}" for metric in numeric for stat in ("mean", "std")]]
    summary.to_csv(OUT / "mean_std.csv", index=False)

    method_order = ["Trainable K=1", "Trainable K=2", "Single centroid", "Fixed K=2", "Random partition", "MOGB-MiniLM", "MOGB partition + s2c boundary", "ADB", "DA-ADB", "BRAK"]
    labels = {"Trainable K=1": "Train K1", "Trainable K=2": "Train K2", "Single centroid": "Single", "Fixed K=2": "Fixed K2", "Random partition": "Random", "MOGB-MiniLM": "MOGB fair", "MOGB partition + s2c boundary": "MOGB part+s2c", "ADB": "ADB", "DA-ADB": "DA-ADB", "BRAK": "BRAK"}
    colors = {"current_trainable": "#1b9e77", "frozen_fair_component": "#7570b3", "external_compatibility": "#d95f02"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = summary[summary["dataset"].eq(dataset)].copy()
        subset["order"] = subset["method"].map({name: index for index, name in enumerate(method_order)}).fillna(999)
        subset = subset.sort_values("order")
        x = np.arange(len(subset))
        ax.bar(x, pd.to_numeric(subset["oos_f1_mean"], errors="coerce") * 100, color=[colors.get(group, "#999999") for group in subset["scope_group"].map(lambda value: "current_trainable" if "Trainable" in str(value) else ("frozen_fair_component" if "frozen MiniLM" in str(value) else "external_compatibility"))])
        ax.set_xticks(x, [labels.get(name, name) for name in subset["method"]], rotation=42, ha="right")
        ax.set_title(dataset)
        ax.set_ylabel("OOS F1 (%)")
        ax.grid(axis="y", alpha=0.25)
        for index, value in enumerate(pd.to_numeric(subset["oos_f1_mean"], errors="coerce")):
            if pd.notna(value):
                ax.text(index, float(value) * 100 + 1, f"{float(value) * 100:.1f}", ha="center", fontsize=7)
    fig.suptitle("KIR=0.50: protocol-separated method comparison (not a single ranking)")
    fig.savefig(FIG / "kir50_method_layers.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for group, color in colors.items():
        subset = summary[summary["scope_group"].map(lambda value: "current_trainable" if "Trainable" in str(value) else ("frozen_fair_component" if "frozen MiniLM" in str(value) else "external_compatibility")).eq(group)].copy()
        subset["oos_f1_mean"] = pd.to_numeric(subset["oos_f1_mean"], errors="coerce")
        subset["known_macro_f1_mean"] = pd.to_numeric(subset["known_macro_f1_mean"], errors="coerce")
        subset = subset.dropna(subset=["oos_f1_mean", "known_macro_f1_mean"])
        ax.scatter(subset["known_macro_f1_mean"] * 100, subset["oos_f1_mean"] * 100, color=color, label=group, s=70, alpha=.82)
        for _, row in subset.iterrows():
            if row["dataset"] == "stackoverflow":
                ax.annotate(str(row["method"]).replace("MOGB partition + s2c boundary", "MOGB part+s2c"), (row["known_macro_f1_mean"] * 100, row["oos_f1_mean"] * 100), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Known Macro-F1 (%)")
    ax.set_ylabel("OOS F1 (%)")
    ax.set_title("KIR=0.50: Known/OOS trade-off by protocol layer")
    ax.grid(alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(FIG / "kir50_method_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    _write_report(summary)
    print({"rows": len(frame), "summary": str(OUT / "mean_std.csv"), "report": str(ROOT / "docs/analysis/KIR50_METHOD_COMPARISON_V1.md")})


if __name__ == "__main__":
    main()
