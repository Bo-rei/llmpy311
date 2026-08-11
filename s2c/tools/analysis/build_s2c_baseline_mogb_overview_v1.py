#!/usr/bin/env python3
"""Build one Chinese overview of historical S2C and the MOGB loss audit.

The report keeps three contracts separate: the historical full Cascade table,
the local official-code compatibility run, and the corrected-loss diagnostic.
It never presents the diagnostic ablation as an official MOGB reproduction.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts" / "s2c"
OUT = ROOT / "results/analysis/s2c_baseline_mogb_overview_v1"
FIG = ROOT / "figures/s2c_baseline_mogb_overview_v1"
REPORT = ROOT / "docs/analysis/S2C_BASELINE_MOGB_COMPARISON_OVERVIEW_V1.md"

ORIGINAL_MANIFEST = (
    ARTIFACTS
    / "external/mogb_exact_reproduction_v1/audit/official_fixed/mode_manifest.json"
)
CORRECTED_MANIFEST = (
    ARTIFACTS
    / "external/mogb_corrected_subcentroid_loss_v1/audit/official_fixed/mode_manifest.json"
)
ORIGINAL_RESULTS = ROOT / "results/mogb_exact_reproduction"
CORRECTED_RESULTS = ROOT / "results/mogb_corrected_subcentroid_loss_v1"
HISTORICAL = ROOT / "results/analysis/historical_sota_comparison_v1"
PUBLISHED = {"Accuracy": 88.67, "F1-All": 87.49, "F1-U": 89.71, "F1-K": 87.27}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    atomic_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", path)


def run_row(name: str, contract: str, manifest: dict[str, Any]) -> dict[str, Any]:
    metrics = manifest["metrics"]
    return {
        "method": name,
        "contract": contract,
        "accuracy": float(metrics["Accuracy"]),
        "f1_all": float(metrics["F1-All"]),
        "f1_u": float(metrics["F1-U"]),
        "f1_k": float(metrics["F1-K"]),
        "known_recall": float(metrics["Known Recall"]),
        "oos_precision": float(metrics["OOS Precision"]),
        "oos_recall": float(metrics["OOS Recall"]),
        "known_to_oos": int(metrics["Known->OOS"]),
        "oos_to_known": int(metrics["OOS->Known"]),
        "best_epoch": int(manifest["best_epoch"]),
        "best_dev_accuracy": float(manifest["best_dev_accuracy"]),
        "ball_count": len(manifest["ball_rows"]),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }


def metric_comparison(frame: pd.DataFrame) -> None:
    metrics = ["accuracy", "f1_all", "f1_u", "f1_k"]
    labels = ["Accuracy", "F1-All", "F1-U", "F1-K"]
    methods = frame.method.tolist()
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for index, (_, row) in enumerate(frame.iterrows()):
        values = [float(row[metric]) for metric in metrics]
        bars = ax.bar(x + (index - 1) * width, values, width, label=methods[index])
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("StackOverflow / KIR=0.50: published MOGB vs local loss contracts")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "mogb_published_local_corrected_metrics.png", dpi=200)
    plt.close(fig)


def error_budget(frame: pd.DataFrame) -> None:
    local = frame[frame.method != "MOGB published reference"].copy()
    x = np.arange(len(local))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, local.known_to_oos, width, label="Known→OOS")
    ax.bar(x + width / 2, local.oos_to_known, width, label="OOS→Known")
    ax.set_xticks(x, local.method, rotation=12)
    ax.set_ylabel("Number of test samples")
    ax.set_title("Error budget: Known rejection versus OOS acceptance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "mogb_error_budget.png", dpi=200)
    plt.close(fig)


def training_curves(original: pd.DataFrame, corrected: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for name, frame, color in (
        ("Official L1 loss", original, "#756bb1"),
        ("Raw-distance diagnostic", corrected, "#31a354"),
    ):
        axes[0].plot(frame.epoch, frame.train_ce_loss, label=name, color=color)
        axes[1].plot(frame.epoch, frame.subcentroid_loss, label=name, color=color)
        axes[2].plot(frame.epoch, frame.dev_accuracy, label=name, color=color)
    axes[0].set_title("Known CE loss")
    axes[1].set_title("Nearest-subcentroid loss")
    axes[2].set_title("Known dev accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Loss")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "mogb_training_contract_curves.png", dpi=200)
    plt.close(fig)


def ball_distributions(original: pd.DataFrame, corrected: pd.DataFrame) -> None:
    rows = []
    for name, frame in (
        ("Official L1 loss", original),
        ("Raw-distance diagnostic", corrected),
    ):
        selected = frame[frame["mode"] == "official_fixed"].copy()
        selected["method"] = name
        rows.append(selected)
    combined = pd.concat(rows, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    groups = [
        combined.loc[combined.method == method, "sample_count"].to_numpy()
        for method in combined.method.unique()
    ]
    axes[0].boxplot(groups, tick_labels=combined.method.unique())
    groups = [
        combined.loc[combined.method == method, "radius"].to_numpy()
        for method in combined.method.unique()
    ]
    axes[1].boxplot(groups, tick_labels=combined.method.unique())
    axes[0].set_title("Selected granular-ball sizes")
    axes[1].set_title("Selected granular-ball radii")
    for axis in axes:
        axis.tick_params(axis="x", rotation=10)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "mogb_ball_distribution_comparison.png", dpi=200)
    plt.close(fig)
    atomic_csv(combined, OUT / "ball_level_comparison.csv")


def historical_summary() -> pd.DataFrame:
    path = HISTORICAL / "ours_minus_best_baseline.csv"
    frame = pd.read_csv(path)
    required = {"dataset", "kir", "ours_oos_f1", "best_baseline_oos_f1", "margin_pp"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"historical margin table missing columns: {sorted(missing)}")
    return frame


def write_report(
    comparison: pd.DataFrame,
    deltas: pd.DataFrame,
    historical: pd.DataFrame,
    current_fair: pd.DataFrame,
    original: dict[str, Any],
    corrected: dict[str, Any],
) -> None:
    local = comparison.set_index("method").loc["MOGB local official logic"]
    adapted = comparison.set_index("method").loc["MOGB raw-distance diagnostic"]
    fair_means = current_fair.groupby("method", as_index=True)[["oos_f1", "f1_all"]].mean()
    fair_trainable = fair_means.loc["trainable_k1"]
    fair_mogb = fair_means.loc["mogb_minilm"]
    stack_fair = current_fair[
        (current_fair.dataset == "stackoverflow") & (current_fair.kir == 0.50)
    ].set_index("method")
    stack_trainable = stack_fair.loc["trainable_k1"]
    stack_mogb = stack_fair.loc["mogb_minilm"]
    margin_lines = [
        f"- {row.dataset} / KIR={row.kir:.2f}：+{row.margin_pp:.2f} pp"
        for row in historical.itertuples()
    ]
    verdict = (
        "修正损失后已明显缩小论文差距"
        if adapted.f1_all - local.f1_all >= 5
        else "单独修正损失不足以解释或关闭论文差距"
    )
    lines = [
        "# S2C 历史基线优势与 MOGB 复现差距总览（V1）",
        "",
        "> 这是当前问题的统一入口。历史论文、当前 Gate 和 MOGB 诊断属于不同实验合同，禁止把数字直接混成一张 SOTA 排名表。",
        "",
        "## 1. `fulltex.tex` 中哪个方法超过了基线",
        "",
        "历史表中的 `Ours` 是完整的 Gate–Router–Expert Cascade：冻结 all-MiniLM-L6-v2 Gate、每意图固定 K=2 KMeans、多局部对角马氏边界，以及 SmolLM-135M LoRA Router/Experts。它不是当前的 Trainable-K1，也不是 RC-AMBL。",
        "",
        "历史表比较 MSP、OpenMax、DOC、DeepUnk、KNNCL、ADB 和 DA-ADB。`Ours` 在九个 dataset×KIR 格子的 OOS F1 都高于表中最强基线；逐格优势为：",
        "",
        *margin_lines,
        "",
        "这只支持‘历史合同下完整 Cascade 的 OOS F1 领先’，不支持所有 Known F1/Accuracy 都为 SOTA，也不能直接证明超过后来发表的 MOGB。",
        "",
        "## 2. 相关对比和可视化是否存在",
        "",
        "已经生成，但此前分散。主要入口：",
        "",
        "- `figures/historical_sota_comparison_v1/`：历史 Ours 与七个基线的热力图和逐格优势；",
        "- `figures/experiment_analysis_master_v1/`：当前 protocol_v2 的方法总览；",
        "- `figures/mogb_reproduction_gap_analysis_v2/`：MOGB 训练、论文差距和粒球分布；",
        "- `figures/mogb_operating_point_visuals_v1/`：MOGB Known/OOS 工作点；",
        "- `figures/s2c_baseline_mogb_overview_v1/`：本报告新增的统一对照图。",
        "",
        "## 3. MOGB 当前到底复现了什么",
        "",
        "本地 exact 单格确实加载了作者公开仓库的 BERT、CE、递归粒球、最近子中心损失、平均半径和最近粒球推理；第三方源码保持 pinned，现代 PyTorch 兼容修复位于外部适配层。它属于‘官方逻辑的现代兼容复现’，不是作者原环境的逐字节复现。",
        "",
        f"StackOverflow/KIR=0.50/seed=0 本地结果为 Acc={local.accuracy:.2f}、F1-All={local.f1_all:.2f}、F1-U={local.f1_u:.2f}、F1-K={local.f1_k:.2f}；论文公开参考为 88.67/87.49/89.71/87.27。",
        "",
        "## 4. 为什么本地结果与论文差很多",
        "",
        "已经排除‘没有训练’和‘没有生成动态粒球’：旧单格 Known dev accuracy 达到 91.60%，最终生成 28 个粒球。主要可验证问题有两层：",
        "",
        "1. 官方 `myloss.py` 将非负类别距离先 L1 归一化，再 `softmax(-distance)`；十个 Known 类时 true-class probability 的理论上限只有约 0.1105，最近子中心监督非常弱。",
        "2. 最终平均距离半径边界过于保守：旧单格 OOS Recall=98.80%，但 Known Recall 只有 51.53%，Known→OOS=1449，主要差距来自过度拒绝 Known。",
        "3. 作者原始 sample ID、Known 列表、旧依赖环境和完整数据生成链未恢复，因此即使算法逻辑相同，也不能证明当前 split 与论文逐样本一致。",
        "",
        "## 5. 只修正子中心损失后的受控结果",
        "",
        "本次只把 L1 归一化距离改成 `raw_distance / temperature=1.0`；BERT、数据、粒球、early stopping、平均半径和推理规则全部保持。它是 diagnostic ablation，不是 MOGB-official。",
        "",
        f"修正后 Acc={adapted.accuracy:.2f}、F1-All={adapted.f1_all:.2f}、F1-U={adapted.f1_u:.2f}、F1-K={adapted.f1_k:.2f}、Known Recall={adapted.known_recall:.2f}。相对旧本地运行，F1-All {adapted.f1_all - local.f1_all:+.2f} pp，F1-K {adapted.f1_k - local.f1_k:+.2f} pp，Known Recall {adapted.known_recall - local.known_recall:+.2f} pp。",
        "",
        f"结论：**{verdict}**。即使修正后更好，也只能说明公开损失合同是复现差距的一项来源；若仍明显低于论文，剩余差距主要需要从边界 calibration、数据合同和旧运行时语义继续定位。",
        "",
        "## 6. 当前能否说 S2C 超过 MOGB",
        "",
        "不能把 fulltex 的历史 Cascade 数字与 MOGB 论文数字直接做公平排名，因为 Known 列表、split、backbone、训练监督和指标合同未完全对齐。当前可以严谨地说：",
        "",
        "- 历史完整 S2C 在其旧主表中超过了当时列出的七个基线；",
        "- 当前 `S2C-Trainable-K1` 在 protocol_v2 的统一 Gate/组件矩阵中优于冻结 K1/K2 与 MOGB-MiniLM-Fair；",
        "- 本地官方逻辑 MOGB 未复现论文公开结果，且差距集中在 Known coverage；",
        "- 还缺同一 split、同一 seeds、统一评估器下的完整 S2C Cascade、MOGB、ADB/DA-ADB 与 DCLOOS 主表，才能作新的 SOTA 结论。",
        "",
        "当前 protocol_v2 的五 seed 组件矩阵中，`S2C-Trainable-K1` 跨九个 dataset×KIR 单元的平均 OOS F1/F1-All 为 "
        f"{fair_trainable.oos_f1 * 100:.2f}/{fair_trainable.f1_all * 100:.2f}，`MOGB-MiniLM-Fair` 为 "
        f"{fair_mogb.oos_f1 * 100:.2f}/{fair_mogb.f1_all * 100:.2f}；StackOverflow/KIR=.50 分别为 "
        f"{stack_trainable.oos_f1 * 100:.2f}/{stack_trainable.f1_all * 100:.2f} 与 "
        f"{stack_mogb.oos_f1 * 100:.2f}/{stack_mogb.f1_all * 100:.2f}。这证明当前 S2C Gate 优于冻结 MiniLM 的 MOGB 组件，"
        "不等于超过论文中的完整 BERT MOGB。",
        "",
        "历史 fulltex 的 StackOverflow/KIR=.50 `Ours` OOS F1 恰为89.71，MOGB论文同格公开 F1-U也为89.71；"
        "数值相同只是描述性巧合，因为样本、Known列表、训练骨干和完整评价合同尚未证明一致。",
        "",
        "## 7. 本次证据文件",
        "",
        "- `results/analysis/s2c_baseline_mogb_overview_v1/method_comparison.csv`",
        "- `loss_contract_deltas.csv`",
        "- `ball_level_comparison.csv`",
        "- `figures/s2c_baseline_mogb_overview_v1/mogb_published_local_corrected_metrics.png`",
        "- `mogb_error_budget.png`",
        "- `mogb_training_contract_curves.png`",
        "- `mogb_ball_distribution_comparison.png`",
        "",
        "## 8. 唯一下一步",
        "",
        "若损失修正仍不能接近论文，下一步只做 Known-calibration 半径归因：保持修正后的表示和粒球不变，比较官方 mean radius 与 Known-only coverage-calibrated radius；不再扩 seed/KIR，直到解释清楚 Known Recall 为什么从约 92% dev 分类准确率降到约一半的边界覆盖。",
        "",
        f"原始 checkpoint：`{original['checkpoint_sha256']}`；修正 checkpoint：`{corrected['checkpoint_sha256']}`。",
    ]
    atomic_text("\n".join(lines) + "\n", REPORT)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    original = load_json(ORIGINAL_MANIFEST)
    corrected = load_json(CORRECTED_MANIFEST)
    if corrected.get("subcentroid_loss_contract", {}).get("mode") != "raw_temperature":
        raise ValueError("corrected manifest does not record raw_temperature loss")

    rows = [
        {
            "method": "MOGB published reference",
            "contract": "paper_reference_not_same_split",
            "accuracy": PUBLISHED["Accuracy"],
            "f1_all": PUBLISHED["F1-All"],
            "f1_u": PUBLISHED["F1-U"],
            "f1_k": PUBLISHED["F1-K"],
            "known_recall": np.nan,
            "oos_precision": np.nan,
            "oos_recall": np.nan,
            "known_to_oos": np.nan,
            "oos_to_known": np.nan,
            "best_epoch": np.nan,
            "best_dev_accuracy": np.nan,
            "ball_count": np.nan,
            "checkpoint_sha256": "not_available",
        },
        run_row(
            "MOGB local official logic",
            "official_logic_modern_compatibility",
            original,
        ),
        run_row(
            "MOGB raw-distance diagnostic",
            "adapted_loss_diagnostic_not_official",
            corrected,
        ),
    ]
    comparison = pd.DataFrame(rows)
    local = comparison.iloc[1]
    adapted = comparison.iloc[2]
    delta_rows = []
    for metric in ("accuracy", "f1_all", "f1_u", "f1_k", "known_recall", "oos_precision", "oos_recall"):
        delta_rows.append(
            {
                "metric": metric,
                "official_local": float(local[metric]),
                "raw_distance_diagnostic": float(adapted[metric]),
                "delta_pp": float(adapted[metric] - local[metric]),
            }
        )
    deltas = pd.DataFrame(delta_rows)
    historical = historical_summary()
    current_fair = pd.read_csv(
        ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
    )

    original_history = pd.read_csv(ORIGINAL_RESULTS / "training_history.csv")
    original_history = original_history[original_history["mode"] == "official_fixed"]
    corrected_history = pd.read_csv(CORRECTED_RESULTS / "training_history.csv")
    corrected_history = corrected_history[corrected_history["mode"] == "official_fixed"]
    original_balls = pd.read_csv(ORIGINAL_RESULTS / "ball_statistics.csv")
    corrected_balls = pd.read_csv(CORRECTED_RESULTS / "ball_statistics.csv")

    atomic_csv(comparison, OUT / "method_comparison.csv")
    atomic_csv(deltas, OUT / "loss_contract_deltas.csv")
    atomic_csv(historical, OUT / "historical_ours_margins.csv")
    atomic_csv(current_fair, OUT / "current_fair_component_summary.csv")
    metric_comparison(comparison)
    error_budget(comparison)
    training_curves(original_history, corrected_history)
    ball_distributions(original_balls, corrected_balls)
    write_report(comparison, deltas, historical, current_fair, original, corrected)

    outputs = [
        OUT / "method_comparison.csv",
        OUT / "loss_contract_deltas.csv",
        OUT / "historical_ours_margins.csv",
        OUT / "current_fair_component_summary.csv",
        OUT / "ball_level_comparison.csv",
        REPORT,
        *sorted(FIG.glob("*.png")),
    ]
    atomic_json(
        {
            "status": "complete",
            "experiment_id": "mogb_corrected_subcentroid_loss_v1",
            "analysis_id": "s2c_baseline_mogb_overview_v1",
            "source_manifests": {
                "official_local": {"path": str(ORIGINAL_MANIFEST), "sha256": sha256(ORIGINAL_MANIFEST)},
                "corrected": {"path": str(CORRECTED_MANIFEST), "sha256": sha256(CORRECTED_MANIFEST)},
            },
            "outputs": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in outputs
            ],
        },
        OUT / "closeout_manifest.json",
    )
    print(json.dumps({"status": "complete", "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
