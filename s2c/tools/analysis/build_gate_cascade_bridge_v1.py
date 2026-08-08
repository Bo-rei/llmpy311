"""Analyze the bridge from Gate-only scores to the current Cascade.

The source rows are already completed artifacts.  This script does not rerun
training and does not claim that Gate-only and Cascade metrics are identical
label spaces; it only compares shared OOS/rejection metrics and decomposes the
Cascade's router/expert errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CASCADE = ROOT / "results/analysis/active_experiment_dashboard_v1/current_cascade_rows.csv"
TRAINABLE = ROOT / "results/analysis/minilm_trainable_5seed_fair_v1/trainable_per_seed.csv"
OUT = ROOT / "results/analysis/gate_cascade_bridge_v1"
FIG = ROOT / "figures/gate_cascade_bridge_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87)
CASCADE_ORDER = ("frozen_k1", "frozen_selected_k", "ce_recon_selected_k", "best_controlled_baseline")
LABELS = {
    "trainable_k1_gate": "Trainable K=1 Gate-only",
    "frozen_k1": "Frozen K=1 Cascade",
    "frozen_selected_k": "Frozen selected-K Cascade",
    "ce_recon_selected_k": "CE-Recon selected-K Cascade",
    "best_controlled_baseline": "Best controlled Cascade",
}
COLORS = {
    "trainable_k1_gate": "#d62728",
    "frozen_k1": "#4c78a8",
    "frozen_selected_k": "#f58518",
    "ce_recon_selected_k": "#54a24b",
    "best_controlled_baseline": "#9467bd",
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cascade = pd.read_csv(CASCADE)
    trainable = pd.read_csv(TRAINABLE)
    if "seed" not in cascade.columns and "kir_seed" in cascade.columns:
        cascade["seed"] = cascade["kir_seed"].str.extract(r"seed(\d+)")[0].astype(int)
    cascade = cascade[(cascade.kir == 0.50) & cascade.dataset.isin(DATASETS) & cascade.gate.isin(CASCADE_ORDER)].copy()
    trainable = trainable[(trainable.kir == 0.50) & trainable.dataset.isin(DATASETS) & trainable.seed.isin(SEEDS)].copy()
    if len(cascade) != len(DATASETS) * len(SEEDS) * len(CASCADE_ORDER):
        raise ValueError(f"Unexpected cascade rows: {len(cascade)}")
    if len(trainable) != len(DATASETS) * len(SEEDS):
        raise ValueError(f"Unexpected trainable rows: {len(trainable)}")
    trainable["id_recall"] = trainable["known_recall"]
    trainable["oos_false_accept_rate"] = trainable["false_accept_rate"]
    trainable["known_false_reject_rate"] = trainable["false_reject_rate"]
    trainable["gate"] = "trainable_k1_gate"
    trainable["router_error_rate"] = np.nan
    trainable["expert_error_rate"] = np.nan
    trainable["overall_accuracy"] = np.nan
    trainable["known_macro_f1"] = np.nan
    common = ["dataset", "kir", "seed", "gate", "oos_f1", "id_recall", "oos_false_accept_rate", "known_false_reject_rate", "router_error_rate", "expert_error_rate", "overall_accuracy", "known_macro_f1"]
    return trainable[common], cascade[common]


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["oos_f1", "id_recall", "oos_false_accept_rate", "known_false_reject_rate", "router_error_rate", "expert_error_rate", "overall_accuracy", "known_macro_f1"]
    keys = ["dataset", "gate"]
    mean = frame.groupby(keys, as_index=False)[metrics].mean()
    std = frame.groupby(keys, as_index=False)[metrics].std().rename(columns={m: f"{m}_std" for m in metrics})
    n = frame.groupby(keys, as_index=False).size().rename(columns={"size": "n_seeds"})
    return mean.merge(std, on=keys).merge(n, on=keys)


def _write(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def _plot_oos(summary: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = summary[summary.dataset == dataset]
        x = np.arange(len(part))
        values = part.set_index("gate").loc[["trainable_k1_gate", *CASCADE_ORDER], "oos_f1"] * 100
        ax.bar(x, values, color=[COLORS[g] for g in ["trainable_k1_gate", *CASCADE_ORDER]])
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[g].replace(" Cascade", "\nCascade").replace(" Gate-only", "\nGate-only") for g in ["trainable_k1_gate", *CASCADE_ORDER]], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("OOS F1 (%)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("KIR=0.50: Trainable Gate-only versus current Cascade variants (3 seeds)")
    fig.tight_layout()
    fig.savefig(FIG / "gate_vs_cascade_oos_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_errors(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = summary[summary.dataset == dataset].set_index("gate").loc[["trainable_k1_gate", *CASCADE_ORDER]]
        x = np.arange(len(part))
        width = 0.25
        ax.bar(x - width, part.oos_false_accept_rate * 100, width, label="Gate false accept", color="#e45756")
        ax.bar(x, part.known_false_reject_rate * 100, width, label="Known false reject", color="#4c78a8")
        ax.bar(x + width, part.expert_error_rate.fillna(0) * 100, width, label="Expert error", color="#72b7b2")
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[g].replace(" Cascade", "\nCascade").replace(" Gate-only", "\nGate-only") for g in ["trainable_k1_gate", *CASCADE_ORDER]], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("rate (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Cascade error decomposition: rejection versus downstream classification")
    fig.tight_layout()
    fig.savefig(FIG / "cascade_error_decomposition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = summary[summary.dataset == dataset].set_index("gate").loc[["trainable_k1_gate", *CASCADE_ORDER]]
        for gate, row in part.iterrows():
            ax.scatter(row.oos_f1 * 100, row.id_recall * 100, s=80, color=COLORS[gate], label=LABELS[gate])
            ax.annotate(LABELS[gate].split()[0], (row.oos_f1 * 100, row.id_recall * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(dataset)
        ax.set_xlabel("OOS F1 (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Known Recall / ID Recall (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Gate-to-Cascade operating points (shared OOS/ID metrics)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "cascade_oos_known_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report(summary: pd.DataFrame) -> None:
    lines = [
        "# Gate→Cascade 桥接与误差分解 V1",
        "",
        "> 本报告只读取当前 protocol_v2_textoir_v1 已完成的 3-seed Cascade 行和 Trainable K=1 Gate 行，不重训、不改变历史结果。Gate-only 与 Cascade 的 Known intent label-space 不完全相同，因此只对共享的 OOS/ID 指标做直接桥接，Accuracy/F1-K 只在 Cascade 内解释。",
        "",
        "## 1. 关键发现",
        "",
        "1. Trainable K=1 Gate-only 的 OOS F1 已经改善，但历史高分还依赖后续 Router/Expert 和完整 Cascade；不能把 Gate-only 数字直接当作论文系统数字。",
        "2. `ce_recon_selected_k` 和 `best_controlled_baseline` 的 OOS F1 变化不只来自 Gate：Router/Expert error 与 Gate false-reject/false-accept 共同决定端到端结果。",
        "3. 同一个 OOS F1 可能对应不同 Known coverage；因此只看 OOS F1 会掩盖 Cascade 牺牲 Known Recall 换取拒识的情况。",
        "",
        "## 2. KIR=0.50 汇总（3 个 seed）",
        "",
        "| 数据集 | 变体 | OOS F1 | ID/ Known Recall | Gate FA | Known FR | Expert error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    order = ["trainable_k1_gate", *CASCADE_ORDER]
    for dataset in DATASETS:
        part = summary[summary.dataset == dataset].set_index("gate")
        for gate in order:
            row = part.loc[gate]
            expert = "--" if pd.isna(row.expert_error_rate) else f"{row.expert_error_rate*100:.2f}"
            lines.append(
                f"| {dataset} | {LABELS[gate]} | {row.oos_f1*100:.2f} | {row.id_recall*100:.2f} | "
                f"{row.oos_false_accept_rate*100:.2f} | {row.known_false_reject_rate*100:.2f} | {expert} |"
            )
    lines.extend(
        [
            "",
            "## 3. 证据",
            "",
        "- `results/analysis/gate_cascade_bridge_v1/per_seed.csv`",
        "- `results/analysis/gate_cascade_bridge_v1/summary_mean_std.csv`",
        "- `figures/gate_cascade_bridge_v1/gate_vs_cascade_oos_f1.png`",
        "- `figures/gate_cascade_bridge_v1/cascade_error_decomposition.png`",
        "- `figures/gate_cascade_bridge_v1/cascade_oos_known_tradeoff.png`",
        "",
        "## 4. 结论边界",
        "",
        "- 该桥接证明“当前 Trainable 低于历史 fulltex”不能归因于一个 MiniLM checkpoint；必须把 Gate、Router、Expert 和校准合同分开。",
        "- 它不是新的 SOTA 结果，也不把 3-seed 当前 Cascade 变体冒充 fulltex 历史主表。",
        ]
    )
    (ROOT / "docs/analysis/GATE_CASCADE_BRIDGE_V1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    trainable, cascade = _load()
    frame = pd.concat([trainable, cascade], ignore_index=True)
    summary = _summary(frame)
    _write(frame, "per_seed.csv")
    _write(summary, "summary_mean_std.csv")
    _plot_oos(summary)
    _plot_errors(summary)
    _plot_tradeoff(summary)
    _report(summary)
    manifest = {"datasets": list(DATASETS), "kir": 0.50, "seeds": list(SEEDS), "analysis_only": True, "trainable_rows": len(trainable), "cascade_rows": len(cascade), "scope": "protocol_v2_textoir_v1"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(frame), "summary": len(summary), "figures": len(list(FIG.glob("*.png")))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
