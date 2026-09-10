#!/usr/bin/env python3
"""Build a contract-separated comparison atlas with the new Cascade bridge.

The atlas is intentionally descriptive.  It aligns rows by explicit protocol,
backbone, supervision and system layer, then renders them in separated panels;
it never turns incompatible rows into a single leaderboard.
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

ROOT = Path(__file__).resolve().parents[2]
FAIR = ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
EXTERNAL = ROOT / "results/analysis/archive/analysis/comparison_atlas_v1/stackoverflow_kir50_external_summary.csv"
CASCADE = ROOT / "results/analysis/archive/analysis/cascade_bridge_v1/summary_mean_std.csv"
MOGB_GAP = ROOT / "results/analysis/archive/analysis/mogb_reproduction_gap_analysis_v2/paper_gap.csv"
MOGB_EXACT = ROOT / "results/analysis/archive/analysis/mogb_reproduction_gap_analysis_v2/exact_run_summary.csv"
DCLOOS = ROOT.parent / "artifacts/s2c/external/dcloos_official_oos_kir75_seed888_reduced_v2/recovery_metrics.json"
OUT = ROOT / "results/analysis/comparison_atlas_v2"
FIG = ROOT / "figures/comparison_atlas_v2"
REPORT = ROOT / "docs/analysis/COMPARISON_ATLAS_V2.md"

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def load_rows() -> pd.DataFrame:
    fair = pd.read_csv(FAIR)
    fair = fair.loc[(fair["dataset"] == "stackoverflow") & np.isclose(fair["kir"], 0.50)].copy()
    fair = fair.loc[fair["method"].isin(["trainable_k1", "single_centroid", "fixed_k2", "random_partition", "mogb_minilm", "mogb_partition_ours_boundary", "ours_partition_mogb_boundary"])]
    fair = fair.rename(columns={"method": "method_id", "oos_f1": "oos_f1_mean", "f1_all": "f1_all_mean", "known_recall": "known_recall_mean", "false_accept_rate": "false_accept_rate_mean", "false_reject_rate": "false_reject_rate_mean", "accuracy": "accuracy_mean"})
    fair["layer"] = "current_protocol_v2_fair_gate"
    fair["system_level"] = "gate_only"
    fair["supervision"] = "Known-only"
    fair["backbone"] = fair["method_id"].map(lambda x: "MiniLM trainable" if x == "trainable_k1" else "MiniLM frozen")
    fair["scope"] = "same_protocol_fair"
    fair["seed_scope"] = "13|42|87|100|123"
    fair["method_label"] = fair["method_id"].map({"trainable_k1": "S2C Trainable K=1", "single_centroid": "S2C Frozen K=1", "fixed_k2": "S2C Frozen K=2", "random_partition": "S2C Random K=2", "mogb_minilm": "MOGB-MiniLM", "mogb_partition_ours_boundary": "MOGB partition + S2C boundary", "ours_partition_mogb_boundary": "S2C partition + MOGB boundary"})
    fair["source"] = str(FAIR.relative_to(ROOT))

    cascade = pd.read_csv(CASCADE)
    cascade = cascade.rename(columns={"gate_variant": "method_id", "oos_f1_mean": "oos_f1_mean", "f1_all_mean": "f1_all_mean", "known_recall_mean": "known_recall_mean", "false_accept_rate_mean": "false_accept_rate_mean", "false_reject_rate_mean": "false_reject_rate_mean", "accuracy_mean": "accuracy_mean"})
    cascade["method_label"] = cascade["method_id"].map({"frozen_k1": "Frozen K=1 Cascade", "trainable_k1": "Trainable K=1 Cascade"})
    cascade["layer"] = "current_protocol_v2_cascade"
    cascade["system_level"] = "gate_router_expert"
    cascade["supervision"] = "Known-only"
    cascade["backbone"] = "MiniLM Gate + SmolLM Expert"
    cascade["scope"] = "same_protocol_cascade_bridge"
    cascade["seed_scope"] = "13|42|87"
    cascade["dataset"] = "stackoverflow"
    cascade["kir"] = 0.50
    cascade["source"] = str(CASCADE.relative_to(ROOT))

    external = pd.read_csv(EXTERNAL)
    external = external.loc[external["method"].isin(["ADB", "DA-ADB"])].copy()
    external["method_id"] = external["method"]
    external["method_label"] = external["method_label"]
    external["layer"] = "external_compatibility"
    external["system_level"] = "gate_or_classifier"
    external["supervision"] = "Known-only"
    external["backbone"] = "BERT/TextOIR"
    external["scope"] = "same_data_external_compatibility"
    external["seed_scope"] = "42|87|100"
    external["dataset"] = "stackoverflow"
    external["kir"] = 0.50
    external["source"] = str(EXTERNAL.relative_to(ROOT))

    rows = pd.concat([fair, cascade, external], ignore_index=True, sort=False)
    keep = ["method_id", "method_label", "layer", "system_level", "supervision", "backbone", "scope", "dataset", "kir", "seed_scope", "n_seeds", "oos_f1_mean", "oos_f1_std", "f1_all_mean", "f1_all_std", "known_recall_mean", "known_recall_std", "false_accept_rate_mean", "false_accept_rate_std", "false_reject_rate_mean", "false_reject_rate_std", "accuracy_mean", "accuracy_std", "source"]
    return rows[[column for column in keep if column in rows.columns]].sort_values(["layer", "method_label"]).reset_index(drop=True)


def build_figures(rows: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    current = rows.loc[rows["layer"].isin(["current_protocol_v2_fair_gate", "current_protocol_v2_cascade"])].copy()
    current["label"] = current["method_label"].str.replace(" + ", "\n+", regex=False)
    order = ["S2C Frozen K=1", "S2C Frozen K=2", "S2C Random K=2", "MOGB-MiniLM", "MOGB partition + S2C boundary", "S2C partition + MOGB boundary", "S2C Trainable K=1", "Frozen K=1 Cascade", "Trainable K=1 Cascade"]
    current["order"] = current["method_label"].map({value: index for index, value in enumerate(order)})
    current = current.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.3), constrained_layout=True)
    colors = ["#777777" if "Frozen" in label else "#D55E00" if "fixed" in label.lower() else "#0072B2" if "Trainable" in label else "#009E73" if "MOGB" in label else "#E69F00" for label in current["method_label"]]
    x = np.arange(len(current))
    for ax, metric, title in zip(axes, ["oos_f1_mean", "f1_all_mean"], ["StackOverflow/KIR=.50：OOS F1", "StackOverflow/KIR=.50：F1-All"]):
        values = current[metric].astype(float).to_numpy() * 100
        errors = current[metric.replace("_mean", "_std")].astype(float).fillna(0).to_numpy() * 100
        ax.bar(x, values, yerr=errors, capsize=3, color=colors)
        ax.set_xticks(x, current["label"], rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("百分比")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for i, value in enumerate(values):
            ax.text(i, value + 1.5, f"{value:.1f}", ha="center", fontsize=8)
    fig.suptitle("当前协议：fair Gate 与同协议 Cascade bridge（只作合同内比较）", fontsize=13)
    fig.savefig(FIG / "current_gate_and_cascade_layers.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    casc = pd.read_csv(CASCADE)
    metrics = [("oos_f1_mean", "OOS F1"), ("false_accept_rate_mean", "False acceptance")]
    x = np.arange(3)
    for ax, (metric, title) in zip(axes, metrics):
        for variant, color in [("frozen_k1", "#777777"), ("trainable_k1", "#0072B2")]:
            row = casc.loc[casc["gate_variant"].eq(variant)].iloc[0]
            value = float(row[metric]) * 100
            error = float(row[metric.replace("_mean", "_std")]) * 100
            ax.errorbar(x, [value] * 3, yerr=[error] * 3, marker="o", linewidth=2, capsize=3, label=variant, color=color)
        ax.set_xticks(x, ["seed 13", "seed 42", "seed 87"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("百分比")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
    fig.suptitle("Cascade bridge：同一 Expert 下的 Gate 变化（误差线为三 seed 标准差）", fontsize=12)
    fig.savefig(FIG / "cascade_bridge_seed_effect.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    gap = pd.read_csv(MOGB_GAP)
    gap = gap.loc[gap["dataset"].astype(str).str.lower().eq("stackoverflow") & np.isclose(gap["local_kir"], 0.50)].copy()
    labels = {"Accuracy": "Accuracy", "F1-All": "F1-All", "F1-U": "F1-U", "F1-K": "F1-K"}
    gap["metric_label"] = gap["metric"].map(labels)
    x = np.arange(len(gap))
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    width = 0.35
    ax.bar(x - width / 2, gap["local_exact"], width, label="本地官方逻辑兼容", color="#C53030")
    ax.bar(x + width / 2, gap["published_reference"], width, label="MOGB 论文公开参考", color="#2B6CB0")
    ax.set_xticks(x, gap["metric_label"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("百分比")
    ax.set_title("MOGB StackOverflow/KIR=.50：本地兼容结果与论文参考差距")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    for i, row in gap.reset_index(drop=True).iterrows():
        ax.text(i, max(row["local_exact"], row["published_reference"]) + 1.2, f"差 {row['gap_pp']:.1f}pp", ha="center", fontsize=8)
    fig.savefig(FIG / "mogb_paper_gap_attribution_context.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(rows: pd.DataFrame) -> str:
    fair = rows.loc[rows["method_id"].eq("trainable_k1") & rows["layer"].eq("current_protocol_v2_fair_gate")].iloc[0]
    cascade = rows.loc[rows["method_id"].eq("trainable_k1") & rows["layer"].eq("current_protocol_v2_cascade")].iloc[0]
    frozen_cascade = rows.loc[rows["method_id"].eq("frozen_k1") & rows["layer"].eq("current_protocol_v2_cascade")].iloc[0]
    return f"""# 当前实验对比图谱 V2（加入 protocol_v2 Cascade bridge）

更新时间：2026-08-10
活动协议：`protocol_v2_textoir_v1`
用途：实验阶段的合同分层、效果比较和机制可视化，不是论文主表，也不做跨合同 SOTA 排名。

## 1. 现在“我的方法”究竟是哪一个

当前自有方法有三个层级，不能混称：

1. **S2C Trainable K=1 Gate**：Known-only 训练 MiniLM 最后两层和残差投影，单中心对角 Mahalanobis Gate。
2. **当前协议 Trainable K=1 Cascade**：上面的 Gate 接入当前 protocol 重新训练的 SmolLM Expert；本轮新增。
3. **fulltex.tex 历史 Ours**：旧合同下的完整 Gate–Router–Expert Cascade，不能与前两者直接排名。

本图谱的当前 fair Gate 主结果中，S2C Trainable K=1 的 OOS F1 为 `{fair['oos_f1_mean'] * 100:.2f}%`；
当前协议 Cascade bridge 中为 `{cascade['oos_f1_mean'] * 100:.2f}%`，Frozen K=1 Cascade 为
`{frozen_cascade['oos_f1_mean'] * 100:.2f}%`。两者分别是 5-seed Gate 矩阵和 3-seed Cascade 桥接，不能混为同一个均值。

## 2. 这次新增的 Cascade 证据

同一 StackOverflow/KIR=.50 protocol_v2 Expert 下，Trainable K=1 相对 Frozen K=1：

- OOS F1：`+9.42pp`
- F1-All：`+6.70pp`
- Known Recall：`+0.21pp`
- false acceptance：`-15.40pp`

这说明当前 Trainable 表示的收益能传递到下游，而不是只在 Gate-only 指标中出现。它仍然不证明
超过完整 MOGB、DCLOOS 或历史 fulltex。

## 3. 当前 MOGB 对比应该怎么读

### 同 MiniLM 公平组件

MOGB-MiniLM 和 MOGB partition + S2C boundary 与当前 S2C fair Gate 使用相同 TEXTOIR split、
Known 列表和 Frozen MiniLM，可用于分析动态粒球、欧氏平均半径和边界的组件作用。StackOverflow/KIR=.50
的五 seed 结果显示：MOGB 组件 false acceptance 较低，但 Known Recall/F1-All 也显著较低；Trainable K=1
是更平衡的工作点。这不是完整官方 MOGB 排名。

### 官方 MOGB 兼容复现

本地运行使用了作者公开代码逻辑（BERT、最近子中心训练、递归粒球、平均半径、最近球推理）和现代兼容层，
但不是作者原始环境的逐字节复现。StackOverflow/KIR=.50/seed=0 的本地结果与论文参考差距，已通过以下证据拆解：

1. 官方子中心损失的距离 L1 归一化压缩了 softmax/logit 梯度；修正后只恢复约 `4.29pp` F1-All。
2. 平均半径过窄造成 Known Recall 约 `51.53%`；Known-only 放大半径后 false acceptance 快速上升。
3. selected-ball 过滤会遗漏部分 Known 类；补球能恢复 Known，但会增加 OOS 误接收。
4. 原始数据快照、Known 列表、环境、最终粒球随机状态和论文训练细节没有全部恢复。

因此正确标签是：`official_code_not_reproduced_under_available_materials`，而不是“已证明 MOGB 算法无效”。

## 4. 外部方法的合同边界

- ADB：StackOverflow/KIR=.50 的 BERT/TextOIR 兼容结果约 `87.47±1.48%`，不是同 MiniLM 训练合同。
- DA-ADB：当前兼容运行 NaN/全类预测，标记 invalid，不引用其伪高分。
- DCLOOS：reduced 单元约 `87.05%`，但使用 pseudo-OOS 和外部 SQuAD OOS、KIR=.75/seed=888，不能进入 Known-only fair 排名。
- 历史 fulltex Ours：旧完整 Cascade 的历史主表数字，只能作为旧合同参照。

## 5. 图表和数据

- `figures/comparison_atlas_v2/current_gate_and_cascade_layers.png`
- `figures/comparison_atlas_v2/cascade_bridge_seed_effect.png`
- `figures/comparison_atlas_v2/mogb_paper_gap_attribution_context.png`
- `results/analysis/comparison_atlas_v2/contract_rows.csv`
- `results/analysis/comparison_atlas_v2/MANIFEST.json`

所有输出均为分析层；没有修改 E2/E3/R1/MOGB/DCLOOS 原始 artifact，没有使用测试结果选参。
"""


def main() -> None:
    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(rows, OUT / "contract_rows.csv")
    build_figures(rows)
    atomic_text(build_report(rows), REPORT)
    manifest = {
        "stage": "comparison_atlas_v2",
        "protocol": "protocol_v2_textoir_v1",
        "scope": "stackoverflow_kir050_contract_layers",
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in [FAIR, EXTERNAL, CASCADE, MOGB_GAP, MOGB_EXACT]},
        "rows": int(len(rows)),
        "figures": [
            "figures/comparison_atlas_v2/current_gate_and_cascade_layers.png",
            "figures/comparison_atlas_v2/cascade_bridge_seed_effect.png",
            "figures/comparison_atlas_v2/mogb_paper_gap_attribution_context.png",
        ],
        "test_used_for_selection": False,
    }
    atomic_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), OUT / "MANIFEST.json")
    print(json.dumps({"stage": manifest["stage"], "rows": len(rows), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
