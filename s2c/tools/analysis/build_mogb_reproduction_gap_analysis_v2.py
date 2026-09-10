#!/usr/bin/env python3
"""Explain the gap between published MOGB results and local reproductions.

This tool is analysis-only.  It reads frozen official-logic artifacts and the
pinned third-party source, then produces contract-separated tables and plots.
It never trains a model, changes a checkpoint, or re-scores test predictions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts" / "s2c"
OUT = ROOT / "results" / "analysis" / "mogb_reproduction_gap_analysis_v2"
FIG = ROOT / "figures" / "mogb_reproduction_gap_analysis_v2"
REPORT = ROOT / "docs" / "analysis" / "MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md"

EXACT_RUNS = {
    "stackoverflow": {
        "kir": 0.50,
        "known_classes": 10,
        "manifest": ARTIFACTS
        / "external/mogb_exact_reproduction_v1/audit/official_fixed/mode_manifest.json",
        "history": ROOT / "results/mogb_exact_reproduction/training_history.csv",
        "balls": ROOT / "results/mogb_exact_reproduction/ball_statistics.csv",
        "paper": {"Accuracy": 88.67, "F1-All": 87.49, "F1-U": 89.71, "F1-K": 87.27},
        "paper_contract_note": "论文 StackOverflow KIR=0.50 公开参考；原始样本 ID 和 Known 列表未恢复。",
    },
    "banking77": {
        "kir": 0.75,
        "known_classes": 58,
        "manifest": ARTIFACTS
        / "external/mogb_exact_reproduction_banking_v1/audit/official_fixed/mode_manifest.json",
        "history": ROOT / "results/mogb_exact_reproduction_banking/training_history.csv",
        "balls": ROOT / "results/mogb_exact_reproduction_banking/ball_statistics.csv",
        "paper": {"Accuracy": 80.58, "F1-All": 81.52, "F1-U": 81.04, "F1-K": 81.53},
        "paper_contract_note": "仅作论文公开参考；本地 Banking 单格为 KIR=0.75，公开数字的精确 split/KIR 合同尚未闭合。",
    },
}

SHORT_ROOT = ARTIFACTS / "external/mogb_official_converged_v1"
OFFICIAL_LOSS = ROOT / "third_party/mogb_official/myloss.py"
OFFICIAL_TRAIN = ROOT / "third_party/mogb_official/pretrain.py"
OFFICIAL_ARGS = ROOT / "third_party/mogb_official/init_parameter.py"
COMPAT_RUNTIME = ROOT / "third_party/mogb_compat/mogb_runtime.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_distance_probability_bound(class_count: int) -> dict[str, float]:
    """Bound imposed by L1-normalizing non-negative class distances.

    The code normalizes the C non-negative nearest-centroid distances so they
    sum to one, then applies softmax to their negatives.  The highest possible
    probability for the true class is achieved by assigning it distance zero
    and spreading the remaining unit mass equally across the other classes.
    """

    uniform_probability = 1.0 / class_count
    maximum_probability = 1.0 / (
        1.0 + (class_count - 1) * math.exp(-1.0 / (class_count - 1))
    )
    uniform_loss = math.log(class_count)
    minimum_loss = -math.log(maximum_probability)
    return {
        "uniform_probability": uniform_probability,
        "maximum_probability": maximum_probability,
        "uniform_loss": uniform_loss,
        "theoretical_minimum_loss": minimum_loss,
        "maximum_loss_reduction": uniform_loss - minimum_loss,
    }


def load_exact_runs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    balls: list[pd.DataFrame] = []
    paper_gaps: list[dict[str, Any]] = []

    for dataset, spec in EXACT_RUNS.items():
        manifest = load_json(spec["manifest"])
        history = pd.read_csv(spec["history"])
        history = history[history["mode"] == "official_fixed"].copy()
        history.insert(0, "dataset", dataset)
        bounds = normalized_distance_probability_bound(int(spec["known_classes"]))
        history["uniform_subcentroid_loss"] = bounds["uniform_loss"]
        history["theoretical_minimum_subcentroid_loss"] = bounds["theoretical_minimum_loss"]
        history["subcentroid_improvement_from_uniform"] = (
            bounds["uniform_loss"] - history["subcentroid_loss"]
        )
        histories.append(history)

        ball_frame = pd.read_csv(spec["balls"])
        ball_frame = ball_frame[ball_frame["mode"] == "official_fixed"].copy()
        ball_frame.insert(0, "dataset", dataset)
        ball_frame["kir"] = float(spec["kir"])
        balls.append(ball_frame)

        metrics = manifest["metrics"]
        first = history.iloc[0]
        last = history.iloc[-1]
        summaries.append(
            {
                "dataset": dataset,
                "kir": float(spec["kir"]),
                "known_classes": int(spec["known_classes"]),
                "epochs_completed": int(manifest["epochs_completed"]),
                "best_epoch": int(manifest["best_epoch"]),
                "best_dev_accuracy": float(manifest["best_dev_accuracy"]),
                "train_ce_loss_first": float(first["train_ce_loss"]),
                "train_ce_loss_last": float(last["train_ce_loss"]),
                "subcentroid_loss_first": float(first["subcentroid_loss"]),
                "subcentroid_loss_last": float(last["subcentroid_loss"]),
                "uniform_subcentroid_loss": bounds["uniform_loss"],
                "theoretical_minimum_subcentroid_loss": bounds["theoretical_minimum_loss"],
                "maximum_loss_reduction": bounds["maximum_loss_reduction"],
                "observed_loss_reduction": bounds["uniform_loss"] - float(last["subcentroid_loss"]),
                "uniform_true_class_probability": bounds["uniform_probability"],
                "maximum_true_class_probability_under_code": bounds["maximum_probability"],
                "ball_count": int(len(ball_frame)),
                "ball_size_min": int(ball_frame["sample_count"].min()),
                "ball_size_mean": float(ball_frame["sample_count"].mean()),
                "ball_size_max": int(ball_frame["sample_count"].max()),
                "radius_min": float(ball_frame["radius"].min()),
                "radius_mean": float(ball_frame["radius"].mean()),
                "radius_max": float(ball_frame["radius"].max()),
                "accuracy": float(metrics["Accuracy"]),
                "f1_all": float(metrics["F1-All"]),
                "f1_u": float(metrics["F1-U"]),
                "f1_k": float(metrics["F1-K"]),
                "known_recall": float(metrics["Known Recall"]),
                "oos_precision": float(metrics["OOS Precision"]),
                "oos_recall": float(metrics["OOS Recall"]),
                "known_to_oos": int(metrics["Known->OOS"]),
                "oos_to_known": int(metrics["OOS->Known"]),
                "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
                "contract": "official_logic_modern_compatibility_on_local_snapshot",
            }
        )
        for metric, published in spec["paper"].items():
            observed = float(metrics[metric])
            paper_gaps.append(
                {
                    "dataset": dataset,
                    "local_kir": float(spec["kir"]),
                    "metric": metric,
                    "local_exact": observed,
                    "published_reference": float(published),
                    "gap_pp": observed - float(published),
                    "contract_note": spec["paper_contract_note"],
                }
            )

    return (
        pd.DataFrame(summaries),
        pd.concat(histories, ignore_index=True),
        pd.concat(balls, ignore_index=True),
        pd.DataFrame(paper_gaps),
    )


def load_short_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(SHORT_ROOT.glob("*/kir_*/seed_*/results/results.csv")):
        run_dir = result_path.parents[1]
        manifest = load_json(run_dir / "manifest.json")
        config = load_json(run_dir / "resolved_config.json")
        result = pd.read_csv(result_path).iloc[0]
        rows.append(
            {
                "dataset": str(result["dataset"]),
                "kir": float(result["known_cls_ratio"]),
                "seed": int(result["seed"]),
                "known_f1": float(result["Known"]),
                "f1_u": float(result["Open"]),
                "f1_all": float(result["F1-score"]),
                "accuracy": float(result["Accuracy"]),
                "epochs": float(manifest["epochs"]),
                "wait_patient": int(config["wait_patient"]),
                "freeze_bert_parameters": bool(config["freeze_bert_parameters"]),
                "strict_official_reproduction": bool(manifest["strict_official_reproduction"]),
                "runtime_repair": bool(manifest["runtime_repair"]),
                "result_path": str(result_path.relative_to(ROOT.parent)),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 10:
        raise ValueError(f"Expected 10 five-epoch compatibility rows, found {len(frame)}")
    metrics = ["known_f1", "f1_u", "f1_all", "accuracy"]
    summary = frame.groupby(["dataset", "kir"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(part for part in col if part) for col in summary.columns]
    return frame, summary


def build_root_cause_evidence(exact: pd.DataFrame) -> pd.DataFrame:
    stack = exact.set_index("dataset").loc["stackoverflow"]
    bank = exact.set_index("dataset").loc["banking77"]
    rows = [
        {
            "hypothesis": "训练没有收敛",
            "status": "not_supported_as_primary_cause",
            "evidence": (
                f"CE loss {stack.train_ce_loss_first:.4f}->{stack.train_ce_loss_last:.4f} / "
                f"{bank.train_ce_loss_first:.4f}->{bank.train_ce_loss_last:.4f}; best dev accuracy "
                f"{stack.best_dev_accuracy:.2f}% / {bank.best_dev_accuracy:.2f}%."
            ),
        },
        {
            "hypothesis": "没有生成自适应粒球",
            "status": "refuted",
            "evidence": f"最终产生 {int(stack.ball_count)} / {int(bank.ball_count)} 个粒球，覆盖 10 / 58 个 Known 类。",
        },
        {
            "hypothesis": "公开子中心损失的动态范围过窄",
            "status": "strongly_supported_by_code_and_trajectory",
            "evidence": (
                "myloss.py 先对非负类别距离做 L1 归一化，再 softmax(-distance)。"
                f"理论上 true-class probability 最多仅从 {stack.uniform_true_class_probability:.4f} 到 "
                f"{stack.maximum_true_class_probability_under_code:.4f}（10类），以及 "
                f"{bank.uniform_true_class_probability:.4f} 到 {bank.maximum_true_class_probability_under_code:.4f}（58类）。"
            ),
        },
        {
            "hypothesis": "最终边界过度拒绝 Known",
            "status": "strongly_supported",
            "evidence": (
                f"OOS Recall 为 {stack.oos_recall:.2f}% / {bank.oos_recall:.2f}%，但 Known Recall 仅 "
                f"{stack.known_recall:.2f}% / {bank.known_recall:.2f}%；Known->OOS 为 "
                f"{int(stack.known_to_oos)} / {int(bank.known_to_oos)}。"
            ),
        },
        {
            "hypothesis": "本地结果可直接否定 MOGB 论文",
            "status": "not_supported",
            "evidence": "作者原始数据、完整旧依赖环境和逐样本 Known/split 合同未恢复；现代兼容层还修复了 stale-graph 训练路径。",
        },
    ]
    return pd.DataFrame(rows)


def plot_losses(history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, (dataset, group) in zip(axes, history.groupby("dataset"), strict=True):
        axis.plot(group["epoch"], group["train_ce_loss"], label="CE loss", color="#2b8cbe", linewidth=2)
        axis.plot(group["epoch"], group["subcentroid_loss"], label="sub-centroid loss", color="#d95f0e", linewidth=2)
        axis.axhline(group["uniform_subcentroid_loss"].iloc[0], color="#756bb1", linestyle="--", label="log(C) uniform")
        axis.axhline(group["theoretical_minimum_subcentroid_loss"].iloc[0], color="#31a354", linestyle=":", label="code-imposed lower bound")
        axis.set_title(dataset)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("MOGB exact cells: CE converges while sub-centroid loss stays in a compressed range")
    fig.savefig(FIG / "ce_vs_subcentroid_loss.png", dpi=190)
    plt.close(fig)


def plot_operating_point(exact: pd.DataFrame) -> None:
    frame = exact.set_index("dataset")
    labels = ["StackOverflow\nKIR=.50", "Banking77\nKIR=.75"]
    x = np.arange(2)
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.bar(x - width, frame.loc[["stackoverflow", "banking77"], "best_dev_accuracy"], width, label="best Known dev accuracy", color="#3182bd")
    ax.bar(x, frame.loc[["stackoverflow", "banking77"], "known_recall"], width, label="test Known recall", color="#e6550d")
    ax.bar(x + width, frame.loc[["stackoverflow", "banking77"], "oos_recall"], width, label="test OOS recall", color="#31a354")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("percent")
    ax.set_title("Closed-set checkpoint looks strong, but the mean-radius boundary rejects Known")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(FIG / "dev_accuracy_vs_known_recall.png", dpi=190)
    plt.close(fig)


def plot_paper_gap(gaps: pd.DataFrame) -> None:
    metrics = ["Accuracy", "F1-All", "F1-U", "F1-K"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True, sharey=True)
    for axis, dataset in zip(axes, ["stackoverflow", "banking77"], strict=True):
        group = gaps[gaps["dataset"] == dataset].set_index("metric").loc[metrics]
        x = np.arange(len(metrics))
        axis.bar(x - 0.18, group["published_reference"], 0.36, label="published reference", color="#756bb1")
        axis.bar(x + 0.18, group["local_exact"], 0.36, label="local exact attempt", color="#31a354")
        axis.set_xticks(x, metrics)
        axis.set_title(dataset)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("score (%)")
    fig.suptitle("Published MOGB reference versus local official-logic compatibility cells")
    fig.savefig(FIG / "paper_gap_metrics.png", dpi=190)
    plt.close(fig)


def plot_balls(balls: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"stackoverflow": "#3182bd", "banking77": "#e6550d"}
    for dataset, group in balls.groupby("dataset"):
        axes[0].scatter(group["sample_count"], group["radius"], alpha=0.72, s=30, label=dataset, color=colors[dataset])
        axes[1].hist(group["sample_count"], bins=16, alpha=0.55, label=dataset, color=colors[dataset])
    axes[0].set_xlabel("ball sample count")
    axes[0].set_ylabel("mean-distance radius")
    axes[0].set_title("Ball size versus radius")
    axes[0].grid(alpha=0.2)
    axes[1].set_xlabel("ball sample count")
    axes[1].set_ylabel("count")
    axes[1].set_title("Final ball-size distribution")
    for axis in axes:
        axis.legend(fontsize=8)
    fig.savefig(FIG / "ball_size_radius_distribution.png", dpi=190)
    plt.close(fig)


def plot_short_runs(short: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, sharey=True)
    for axis, dataset in zip(axes, ["stackoverflow", "banking77"], strict=True):
        group = short[short["dataset"] == dataset].sort_values("seed")
        axis.plot(group["seed"].astype(str), group["f1_all"], marker="o", label="5-epoch F1-All", color="#3182bd")
        paper = EXACT_RUNS[dataset]["paper"]["F1-All"]
        axis.axhline(paper, linestyle="--", color="#756bb1", label="published reference")
        axis.set_title(dataset)
        axis.set_xlabel("seed")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("F1-All (%)")
    fig.suptitle("Five-epoch compatibility runs are stable but intentionally under-budget")
    fig.savefig(FIG / "five_seed_short_run_stability.png", dpi=190)
    plt.close(fig)


def write_report(exact: pd.DataFrame, gaps: pd.DataFrame, evidence: pd.DataFrame) -> None:
    stack = exact.set_index("dataset").loc["stackoverflow"]
    bank = exact.set_index("dataset").loc["banking77"]
    lines = [
        "# MOGB 复现差距与方法合同分析（V2）",
        "",
        "> 结论先行：本地 MOGB 单格不是简单的“没训练好”。闭集 CE 与 Known dev accuracy 已收敛；主要异常是公开代码中的子中心损失被 L1 距离归一化压缩，以及最终平均半径边界严重拒绝 Known。由于作者原始数据和旧环境不完整，本报告只能定位本地差距，不能据此否定论文结果。",
        "",
        "## 1. 现在到底在比较哪些方法",
        "",
        "| 名称 | 实际方法 | 合同 | 可回答的问题 |",
        "|---|---|---|---|",
        "| fulltex `Ours` | 冻结 MiniLM 多中心 Gate + SmolLM Router/Expert 的完整 Cascade | 历史论文合同 | 为什么旧论文 OOS F1 高于其表内基线 |",
        "| `S2C-Trainable-K1` | Known-only 训练 MiniLM 最后两层与 projection + 单中心 Gate | 当前 protocol_v2，五 seed | 当前自有 Gate 候选是否优于冻结/固定多中心组件 |",
        "| `MOGB-MiniLM-Fair` | 同一冻结 MiniLM embedding + MOGB 粒球/平均半径 | 当前 fair 组件合同 | 固定表示下，粒球与边界组件是否有效 |",
        "| MOGB exact local | BERT + CE + 最近子中心损失 + 自适应粒球 + 平均半径 | 官方逻辑的现代兼容单格 | 公开实现能否在现有材料下接近论文数字 |",
        "| MOGB paper | 论文报告的完整方法 | 作者原始合同 | 公开参考，不与当前 Gate 行直接排名 |",
        "",
        "因此，“我的方法超过谁”必须带限定：历史 fulltex `Ours` 在旧表中比较的是 MSP、OpenMax、DOC、DeepUnk、KNNCL、ADB、DA-ADB；当前 `S2C-Trainable-K1` 只在当前统一 Gate/组件矩阵中优于 Frozen K1/K2、random K2 和 MOGB-MiniLM 组件。它尚不能宣称超过完整 MOGB、DCLOOS 或历史完整 Cascade。",
        "",
        "## 2. 本地 exact 单格有没有真正训练",
        "",
        f"- StackOverflow KIR=.50：训练 {int(stack.epochs_completed)} epoch，best epoch={int(stack.best_epoch)}，Known dev accuracy={stack.best_dev_accuracy:.2f}%，CE loss {stack.train_ce_loss_first:.4f}→{stack.train_ce_loss_last:.4f}。",
        f"- Banking77 KIR=.75：训练 {int(bank.epochs_completed)} epoch，best epoch={int(bank.best_epoch)}，Known dev accuracy={bank.best_dev_accuracy:.2f}%，CE loss {bank.train_ce_loss_first:.4f}→{bank.train_ce_loss_last:.4f}。",
        "",
        "所以不能再把主要差距解释成 epoch 太少或 CE 完全未收敛。另有十个五 epoch 兼容运行，它们只是执行稳定性证据，明确 `strict_official_reproduction=false`，不能放进论文主表。",
        "",
        "## 3. 子中心损失为什么可疑",
        "",
        "公开 `myloss.py` 对每个样本先计算到各类别最近粒球中心的距离，然后执行 `F.normalize(distances, p=1)`，再计算 `softmax(-distances)`。距离非负且归一化后总和为 1，这把所有类别 logit 压在 [-1,0] 的极窄区间。",
        "",
        f"对 StackOverflow 的 10 个 Known 类，均匀概率为 {stack.uniform_true_class_probability:.4f}，即使达到该代码约束下的理论最佳，true-class probability 也最多为 {stack.maximum_true_class_probability_under_code:.4f}；子中心交叉熵只能从 log(10)={stack.uniform_subcentroid_loss:.4f} 最多降到 {stack.theoretical_minimum_subcentroid_loss:.4f}。实测末轮为 {stack.subcentroid_loss_last:.4f}。",
        "",
        f"对 Banking77 的 58 个 Known 类，均匀概率为 {bank.uniform_true_class_probability:.5f}，理论最大也仅 {bank.maximum_true_class_probability_under_code:.5f}；loss 的全部理论动态范围只有 {bank.maximum_loss_reduction:.4f}，实测 {bank.subcentroid_loss_first:.4f}→{bank.subcentroid_loss_last:.4f}。类别越多，这个压缩越严重。",
        "",
        "这不证明论文概念无效，但证明当前公开实现的最近子中心目标很难形成高置信的类别梯度，尤其在 Banking77 的多 Known 类设置下。",
        "",
        "## 4. 最终边界的主要失败不是误收 OOS，而是拒绝 Known",
        "",
        f"StackOverflow：OOS Recall={stack.oos_recall:.2f}%，Known Recall={stack.known_recall:.2f}%，Known→OOS={int(stack.known_to_oos)}，OOS→Known={int(stack.oos_to_known)}。",
        f"Banking77：OOS Recall={bank.oos_recall:.2f}%，Known Recall={bank.known_recall:.2f}%，Known→OOS={int(bank.known_to_oos)}，OOS→Known={int(bank.oos_to_known)}。",
        "",
        "这说明 MOGB 本地工作点非常保守：平均距离半径能拒绝几乎全部 OOS，却只覆盖约一半 Known。闭集 checkpoint 是按 Known dev accuracy 选择的，并没有校准最终多粒球边界的 Known coverage，因此 dev accuracy 约 92% 与 test Known Recall 约 44%–52% 可以同时出现。",
        "",
        "## 5. 动态粒球是否实际运行",
        "",
        f"是。最终 StackOverflow 有 {int(stack.ball_count)} 个粒球、10 个 Known 类；Banking77 有 {int(bank.ball_count)} 个粒球、58 个 Known 类。球样本数均值分别为 {stack.ball_size_mean:.1f} 和 {bank.ball_size_mean:.1f}，平均半径为 {stack.radius_mean:.3f} 和 {bank.radius_mean:.3f}。所以差距不是“代码退化成单中心”，而是表示损失、粒球结构、平均半径和 checkpoint 选择之间没有形成论文报告中的工作点。",
        "",
        "## 6. 与论文的差距",
        "",
        "StackOverflow 的本地 exact 单格相对论文公开参考：Accuracy -13.50 pp、F1-All -19.14 pp、F1-U -9.74 pp、F1-K -20.08 pp。Known 侧差距明显大于 Unknown 侧，和上述过度拒绝一致。Banking 本地单格也显著偏低，但其本地 KIR=.75 与公开参考的精确合同仍未闭合，只能描述，不能作严格一一复现判定。",
        "",
        "## 7. 现代兼容层带来的不可消除差异",
        "",
        "官方 `pretrain.py` 先完成 CE optimizer step，再把 CE 更新后的 features 保存在 memory bank，最后对整库 loss2 反向传播；现代 PyTorch 会遇到 stale graph。兼容层改为：no-grad 重新提取 feature bank 生成粒球，再对固定 centroids 做第二次可微前向和 optimizer2 更新。这是合理修复，但不是作者旧环境的逐字节执行，可能改变梯度、聚类更新和早停轨迹。",
        "",
        "另外，作者原始数据、逐样本 ID、Known 类列表、完整旧依赖环境均未恢复；pinned checkout 本身还缺少 `utils` 包。四组合静态诊断在 `results/diagnostics/mogb_diff/`，结论仍是 `public_code_not_reproduced_under_available_materials`。",
        "",
        "## 8. 为什么当前 S2C 行看起来比本地 MOGB 好",
        "",
        "当前 `S2C-Trainable-K1` 的 Known-only 训练直接改善 Gate score separation，且 S2C 的 `μ+λσ` 边界比 MOGB 平均半径保留更多 Known coverage。相反，本地 MOGB exact 的 OOS Recall 已接近 98%，但 Known Recall 只有约一半；MOGB-MiniLM-Fair 更是同样呈现低 false acceptance、高 false rejection。当前优势主要是工作点更平衡，不是已经证明 S2C 全面超过论文完整 MOGB。",
        "",
        "## 9. 新增证据文件",
        "",
        "- `results/analysis/archive/analysis/mogb_reproduction_gap_analysis_v2/exact_run_summary.csv`：两个 exact 单格；",
        "- `loss_trajectory.csv`：CE、子中心 loss 和理论界；",
        "- `paper_gap.csv`：论文公开参考差值；",
        "- `ball_distribution.csv`：127 个最终粒球；",
        "- `five_seed_compatibility_runs.csv`：十个五 epoch 非严格运行；",
        "- `root_cause_evidence.csv`：各归因假设的证据状态；",
        "- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/`：五张图。",
        "",
        "## 10. 当前结论和下一步",
        "",
        "1. 历史 SOTA 方法是完整 Cascade `Ours`，不是当前 Trainable-K1，也不是 RC-AMBL。",
        "2. 现有可视化很多，但之前没有把方法合同、MOGB loss 数学动态范围和边界过拒串成一条证据链；本报告补齐了这部分。",
        "3. 不应继续盲目扩大 MOGB 复现矩阵。若要追近论文，优先做两个受控诊断：去掉距离 L1 归一化/加入温度的 loss 公式校验；在不看 test OOS 的前提下，用 Known calibration 检查 mean radius 的 coverage。二者必须注册为 adapted ablation，不能冒充 official reproduction。",
        "4. 当前论文级公平结论仍需同一数据、Known 列表、split、seed 和评估器下的完整 S2C Cascade、MOGB、ADB/DA-ADB/DCLOOS 主表。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_suffix(REPORT.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, REPORT)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__).resolve(), OFFICIAL_LOSS, OFFICIAL_TRAIN, OFFICIAL_ARGS, COMPAT_RUNTIME]
    for spec in EXACT_RUNS.values():
        sources.extend([spec["manifest"], spec["history"], spec["balls"]])
    if any(not path.is_file() for path in sources):
        missing = [str(path) for path in sources if not path.is_file()]
        raise FileNotFoundError(f"Missing sources: {missing}")

    exact, history, balls, gaps = load_exact_runs()
    short, short_summary = load_short_runs()
    evidence = build_root_cause_evidence(exact)

    atomic_csv(exact, OUT / "exact_run_summary.csv")
    atomic_csv(history, OUT / "loss_trajectory.csv")
    atomic_csv(balls, OUT / "ball_distribution.csv")
    atomic_csv(gaps, OUT / "paper_gap.csv")
    atomic_csv(short, OUT / "five_seed_compatibility_runs.csv")
    atomic_csv(short_summary, OUT / "five_seed_compatibility_summary.csv")
    atomic_csv(evidence, OUT / "root_cause_evidence.csv")

    plot_losses(history)
    plot_operating_point(exact)
    plot_paper_gap(gaps)
    plot_balls(balls)
    plot_short_runs(short)
    write_report(exact, gaps, evidence)

    manifest = {
        "stage": "mogb_reproduction_gap_analysis_v2",
        "analysis_only": True,
        "training_launched": False,
        "exact_cells": int(len(exact)),
        "exact_history_rows": int(len(history)),
        "ball_rows": int(len(balls)),
        "short_compatibility_rows": int(len(short)),
        "source_sha256": {str(path.relative_to(ROOT.parent)): sha256(path) for path in sources},
        "outputs": sorted(
            str(path.relative_to(ROOT))
            for path in OUT.glob("*")
            if path.is_file() and path.name != "MANIFEST.json"
        ),
        "figures": sorted(str(path.relative_to(ROOT)) for path in FIG.glob("*") if path.is_file()),
        "contract_warning": (
            "Published references, exact local compatibility cells, five-epoch compatibility runs, "
            "MOGB-MiniLM fair components, and S2C Gate/Cascade results are separate contracts."
        ),
    }
    atomic_json(manifest, OUT / "MANIFEST.json")
    print(json.dumps({"exact_cells": len(exact), "history_rows": len(history), "balls": len(balls), "short_runs": len(short)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
