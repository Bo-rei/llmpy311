#!/usr/bin/env python3
"""Summarize the current-protocol Cascade bridge across all three datasets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1"
SO_ROOT = ARTIFACT_ROOT / "cascade_bridge_v1"
MULTI_ROOT = ARTIFACT_ROOT / "cascade_bridge_multidataset_v1"
OUT = ROOT / "results" / "analysis" / "cascade_bridge_cross_dataset_v1"
FIG = ROOT / "figures" / "cascade_bridge_cross_dataset_v1"
SEEDS = (13, 42, 87)
DATASETS = ("clinc150", "banking77", "stackoverflow")


def _read_metrics(path: Path) -> pd.DataFrame:
    payload = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    return pd.DataFrame(payload["metrics"])


def _cascade_prediction_path(dataset: str, seed: int, variant: str) -> Path:
    if dataset == "stackoverflow":
        return SO_ROOT / f"seed_{seed}" / f"{variant}_predictions.jsonl"
    return MULTI_ROOT / dataset / f"seed_{seed}" / f"{variant}_predictions.jsonl"


def _gate_source_path(metric: pd.Series, dataset: str, seed: int, variant: str) -> Path:
    # The original StackOverflow bridge predates the explicit gate_source
    # column.  Its provenance manifest pins the legacy-compatible source.
    source = metric.get("gate_source")
    if isinstance(source, str) and source and source != "nan":
        return Path(source)
    if dataset == "stackoverflow":
        return ARTIFACT_ROOT / "racal_v1" / "runs" / variant / f"seed_{seed}" / "predictions.jsonl"
    raise ValueError(f"Missing gate source for {dataset} seed={seed} variant={variant}")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _error_budget(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, metric in metrics.iterrows():
        dataset = str(metric["dataset"])
        seed = int(metric["seed"])
        variant = str(metric["gate_variant"])
        cascade = _load_jsonl(_cascade_prediction_path(dataset, seed, variant))
        gate = _load_jsonl(_gate_source_path(metric, dataset, seed, variant))
        gate_by_id = {str(row["sample_id"]): row for row in gate}
        gate_fr = gate_fa = gate_known_correct = gate_known_wrong = 0
        expert_known_correct = expert_known_wrong = 0
        for row in cascade:
            sample_id = str(row["sample_id"])
            g = gate_by_id[sample_id]
            gold_oos = int(row["gold_is_oos"]) == 1
            gate_accept = int(g["predicted_is_oos"]) == 0
            if gold_oos and gate_accept:
                gate_fa += 1
            if not gold_oos and not gate_accept:
                gate_fr += 1
            if not gold_oos and gate_accept:
                gate_intent = g.get("predicted_intent", g.get("nearest_known_intent", ""))
                if str(gate_intent) == str(row["gold_intent"]):
                    gate_known_correct += 1
                else:
                    gate_known_wrong += 1
                if str(row["predicted_intent"]) == str(row["gold_intent"]):
                    expert_known_correct += 1
                else:
                    expert_known_wrong += 1
        oos_count = int(sum(int(r["gold_is_oos"]) for r in cascade))
        known_count = len(cascade) - oos_count
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "gate_variant": variant,
                "gate_false_accept": gate_fa,
                "gate_false_reject": gate_fr,
                "gate_known_correct": gate_known_correct,
                "gate_known_wrong": gate_known_wrong,
                "expert_known_correct": expert_known_correct,
                "expert_known_wrong": expert_known_wrong,
                "oos_count": oos_count,
                "known_count": known_count,
                "false_accept_rate": gate_fa / max(oos_count, 1),
                "false_reject_rate": gate_fr / max(known_count, 1),
            }
        )
    return pd.DataFrame(rows)


def _write_report(summary: pd.DataFrame, paired: pd.DataFrame, budget: pd.DataFrame) -> None:
    def pct(x: float) -> str:
        return f"{x * 100:.2f}%"

    lines = [
        "# 当前协议三数据集 Cascade 桥接分析（V1）",
        "",
        "> 这是当前 `protocol_v2_textoir_v1` 的下游桥接，不是 `fulltex.tex` 历史 Cascade 的逐字节复现；Gate、Expert、数据和选择协议均单独记录。",
        "",
        "## 1. 实验范围与合同",
        "",
        "- 数据集：CLINC150、Banking77、StackOverflow；KIR=0.50；seeds=13/42/87。",
        "- Frozen K=1 使用已完成 E2 的 `test.jsonl`；Trainable K=1 使用当前协议 Known-only MiniLM control 的预测。",
        "- 两个 Gate 变体共享同一数据 views、同一 SmolLM Expert 训练代码和同一 Expert checkpoint（同 dataset×seed）。",
        "- Expert 只读 `train_known`，用 `calibration_known` 选 epoch；测试 OOS 不参与训练或选择。",
        "- StackOverflow 的已有 `cascade_bridge_v1` 不覆盖；CLINC150/Banking77 写入独立 `cascade_bridge_multidataset_v1`。",
        "",
        "## 2. 均值±标准差",
        "",
        "| 数据集 | Gate | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | FA | FR | AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.dataset} | {row.gate_variant} | {pct(row.oos_f1_mean)}±{pct(row.oos_f1_std)} | "
            f"{pct(row.f1_all_mean)}±{pct(row.f1_all_std)} | {pct(row.f1_k_mean)}±{pct(row.f1_k_std)} | "
            f"{pct(row.accuracy_mean)}±{pct(row.accuracy_std)} | {pct(row.known_recall_mean)}±{pct(row.known_recall_std)} | "
            f"{pct(row.false_accept_rate_mean)}±{pct(row.false_accept_rate_std)} | {pct(row.false_reject_rate_mean)}±{pct(row.false_reject_rate_std)} | "
            f"{pct(row.auroc_mean)}±{pct(row.auroc_std)} |"
        )
    lines += [
        "",
        "## 3. Trainable 相对 Frozen 的配对差值",
        "",
        "| 数据集 | OOS F1 | F1-All | Accuracy | Known Recall | FA | FR | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in paired.iterrows():
        lines.append(
            f"| {row.dataset} | {pct(row.oos_f1_delta)} | {pct(row.f1_all_delta)} | {pct(row.accuracy_delta)} | "
            f"{pct(row.known_recall_delta)} | {pct(row.false_accept_rate_delta)} | {pct(row.false_reject_rate_delta)} | {pct(row.auroc_delta)} |"
        )
    lines += [
        "",
        "## 4. Gate→Cascade 错误预算",
        "",
        "这里 FA 的分母是 OOS 数量，FR 的分母是 Known 数量。Trainable 的主要变化仍应先看 Gate 的 OOS false acceptance，不能把 Expert 的闭集分类变化误读成 OOS 检测来源。",
        "",
        "| 数据集 | Gate | FA均值 | FR均值 | Known正确均值 | Known错误均值 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    bsum = budget.groupby(["dataset", "gate_variant"], as_index=False).agg(
        false_accept_rate=("false_accept_rate", "mean"),
        false_reject_rate=("false_reject_rate", "mean"),
        gate_known_correct=("gate_known_correct", "mean"),
        gate_known_wrong=("gate_known_wrong", "mean"),
    )
    for _, row in bsum.iterrows():
        lines.append(
            f"| {row.dataset} | {row.gate_variant} | {pct(row.false_accept_rate)} | {pct(row.false_reject_rate)} | "
            f"{row.gate_known_correct:.1f} | {row.gate_known_wrong:.1f} |"
        )
    lines += [
        "",
        "## 5. 当前可以下的结论",
        "",
        "1. `S2C-Trainable-K1` 的当前 Cascade 证据已从 StackOverflow 扩展到三个数据集，但样本量仍为每个数据集 3 seeds，不替代已有 Gate 5-seed 主矩阵。",
        "2. 若 Trainable 在三个数据集均降低 FA 且 OOS F1 上升，说明当前最可重复的收益来源是 Trainable Gate 的 score separation，而不是 Expert 单独变强。",
        "3. 这仍不是对论文 MOGB/DCLOOS 的公平 SOTA 排名：MOGB 论文使用 BERT、交替粒球训练和作者原始数据合同；DCLOOS 使用 pseudo-OOS 与外部 OOS。它们必须保留为不同监督合同。",
        "4. `fulltex.tex` 的历史 `Ours` 是旧版完整 Cascade；当前 Trainable K=1 是当前协议候选，二者不能直接混排。",
        "",
        "## 6. 证据路径",
        "",
        "- 逐 seed：`results/analysis/cascade_bridge_cross_dataset_v1/per_seed.csv`",
        "- 配对差值：`results/analysis/cascade_bridge_cross_dataset_v1/paired_effects.csv`",
        "- 错误预算：`results/analysis/cascade_bridge_cross_dataset_v1/error_budget.csv`",
        "- 图：`figures/cascade_bridge_cross_dataset_v1/`",
        "- StackOverflow 单独报告：`docs/analysis/CASCADE_BRIDGE_V1.md`",
        "- MOGB 复现差距：`docs/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`",
    ]
    (ROOT / "docs" / "analysis" / "CASCADE_BRIDGE_CROSS_DATASET_V1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([_read_metrics(SO_ROOT), _read_metrics(MULTI_ROOT)], ignore_index=True)
    metrics.to_csv(OUT / "per_seed.csv", index=False)
    numeric = ["oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"]
    grouped = metrics.groupby(["dataset", "gate_variant"], as_index=False)[numeric].agg(["mean", "std"])
    grouped.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(OUT / "summary_mean_std.csv", index=False)

    paired_rows = []
    for dataset, frame in metrics.groupby("dataset"):
        piv = frame.pivot(index="seed", columns="gate_variant", values=numeric)
        row = {"dataset": dataset, "n_seeds": int(len(piv))}
        for col in numeric:
            row[f"{col}_delta"] = float((piv[col]["trainable_k1"] - piv[col]["frozen_k1"]).mean())
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "paired_effects.csv", index=False)
    budget = _error_budget(metrics)
    budget.to_csv(OUT / "error_budget.csv", index=False)

    # Performance view.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ["oos_f1", "f1_all"], ["Cascade OOS F1", "Cascade F1-All"]):
        plot = grouped.pivot(index="dataset", columns="gate_variant", values=f"{metric}_mean")
        plot = plot.reindex(list(DATASETS))
        plot.plot(kind="bar", ax=ax, color={"frozen_k1": "#9aa0a6", "trainable_k1": "#1f77b4"}, rot=0)
        ax.set_title(title)
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Gate", fontsize=8)
    fig.savefig(FIG / "trainable_vs_frozen_cascade_metrics.png", dpi=180)
    plt.close(fig)

    # Paired delta view.
    plot_cols = ["oos_f1_delta", "f1_all_delta", "accuracy_delta", "known_recall_delta", "false_accept_rate_delta"]
    labels = ["OOS F1", "F1-All", "Accuracy", "Known Recall", "FA"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(DATASETS))
    width = 0.15
    for idx, col in enumerate(plot_cols):
        ax.bar(x + (idx - 2) * width, paired[col], width, label=labels[idx])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, list(DATASETS))
    ax.set_ylabel("Trainable - Frozen")
    ax.set_title("Current-protocol Cascade paired effects")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "trainable_minus_frozen_cascade_effect.png", dpi=180)
    plt.close(fig)

    # Error-budget view.
    bsum = budget.groupby(["dataset", "gate_variant"], as_index=False)[["false_accept_rate", "false_reject_rate"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, metric, title in zip(axes, ["false_accept_rate", "false_reject_rate"], ["Gate false acceptance", "Gate false rejection"]):
        plot = bsum.pivot(index="dataset", columns="gate_variant", values=metric).reindex(list(DATASETS))
        plot.plot(kind="bar", ax=ax, color={"frozen_k1": "#d95f02", "trainable_k1": "#1b9e77"}, rot=0)
        ax.set_title(title)
        ax.set_ylabel("rate")
        ax.set_ylim(0, max(0.4, float(plot.max().max()) * 1.2))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(FIG / "cascade_error_budget_cross_dataset.png", dpi=180)
    plt.close(fig)
    _write_report(grouped, paired, budget)
    manifest = {
        "stage": "cascade_bridge_cross_dataset_v1",
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "source_roots": [str(SO_ROOT), str(MULTI_ROOT)],
        "output_files": [str(p.relative_to(ROOT)) for p in OUT.glob("*")],
        "figures": [str(p.relative_to(ROOT)) for p in FIG.glob("*")],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(metrics), "datasets": list(metrics["dataset"].unique()), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
