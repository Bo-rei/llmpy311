#!/usr/bin/env python3
"""Paired open-intent outcome attribution for Trainable K1 vs MOGB-Fair.

This analysis only reads frozen protocol_v2 predictions.  It extends the
binary Gate attribution with five mutually exclusive open-intent outcomes:
correct Known, wrong Known intent, rejected Known, correctly rejected OOS,
and falsely accepted OOS.  Test labels are used for post-hoc attribution only.
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
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
SOURCE = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
OUT = ROOT / "results/analysis/archive/analysis/trainable_mogb_open_intent_transitions_v1"
FIG = ROOT / "figures/archive/analysis/trainable_mogb_open_intent_transitions_v1"
ARTIFACT = WORKSPACE / "artifacts/s2c/runs/protocol_v2_textoir_v1/trainable_mogb_open_intent_transitions_v1"
REPORT = ROOT / "docs/archive/analysis/TRAINABLE_MOGB_OPEN_INTENT_TRANSITIONS_V1.md"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
METHODS = ("trainable_k1", "mogb_minilm")
OUTCOMES = (
    "known_correct",
    "known_wrong_intent",
    "known_rejected",
    "oos_correct",
    "oos_false_accept",
)
LABELS = {
    "known_correct": "Known correct",
    "known_wrong_intent": "Known wrong intent",
    "known_rejected": "Known rejected",
    "oos_correct": "OOS correct",
    "oos_false_accept": "OOS false accept",
}
MOGB_SCORE_BINS = (-np.inf, 0.75, 1.0, 1.25, 1.50, 2.0, np.inf)
MOGB_SCORE_LABELS = ("<=0.75", "0.75-1.00", "1.00-1.25", "1.25-1.50", "1.50-2.00", ">2.00")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def resolve(path_value: str) -> Path:
    for candidate in (Path(path_value), ROOT / path_value, WORKSPACE / path_value):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(path_value)


def classify_outcomes(
    gold_intent: pd.Series,
    gold_is_oos: pd.Series,
    predicted_label: pd.Series,
    predicted_is_oos: pd.Series,
) -> pd.Series:
    """Return the five-state open-intent outcome for every row."""
    gold_oos = gold_is_oos.astype(bool).to_numpy()
    pred_oos = predicted_is_oos.astype(bool).to_numpy()
    label_correct = predicted_label.astype(str).to_numpy() == gold_intent.astype(str).to_numpy()
    values = np.full(len(gold_intent), "known_wrong_intent", dtype=object)
    values[~gold_oos & pred_oos] = "known_rejected"
    values[~gold_oos & ~pred_oos & label_correct] = "known_correct"
    values[gold_oos & pred_oos] = "oos_correct"
    values[gold_oos & ~pred_oos] = "oos_false_accept"
    return pd.Series(values, index=gold_intent.index, dtype="string")


def load_prediction(row: pd.Series) -> tuple[pd.DataFrame, Path]:
    method = str(row["method"])
    if method == "trainable_k1":
        path = resolve(str(row["metrics_path"])).with_name("predictions.jsonl")
        raw = pd.read_json(path, lines=True)
        score_col = "oos_score"
        label_col = "predicted_intent"
    elif method == "mogb_minilm":
        path = resolve(str(row["run_dir"])) / "predictions.tsv"
        raw = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
        score_col = "normalized_score"
        label_col = "predicted_label"
    else:
        raise ValueError(method)
    required = {"sample_id", "gold_intent", "gold_is_oos", score_col, label_col}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    frame = pd.DataFrame(
        {
            "sample_id": raw["sample_id"].astype(str),
            "gold_intent": raw["gold_intent"].astype(str),
            "gold_is_oos": raw["gold_is_oos"].astype(int),
            "score": pd.to_numeric(raw[score_col], errors="coerce"),
            "predicted_label": raw[label_col].astype(str),
        }
    )
    if frame.sample_id.duplicated().any() or frame.score.isna().any() or not np.isfinite(frame.score).all():
        raise ValueError(f"invalid prediction file: {path}")
    frame["predicted_is_oos"] = frame.score.gt(1.0).astype(int)
    frame["open_prediction"] = np.where(frame.predicted_is_oos.eq(1), "__oos__", frame.predicted_label)
    frame["open_gold"] = np.where(frame.gold_is_oos.eq(1), "__oos__", frame.gold_intent)
    frame["outcome"] = classify_outcomes(
        frame.gold_intent,
        frame.gold_is_oos,
        frame.predicted_label,
        frame.predicted_is_oos,
    )
    return frame.sort_values("sample_id").reset_index(drop=True), path


def transition_table(left: pd.DataFrame, right: pd.DataFrame, dataset: str, kir: float, seed: int) -> pd.DataFrame:
    if not left.sample_id.equals(right.sample_id):
        raise ValueError(f"sample mismatch: {dataset}/{kir}/{seed}")
    for column in ("gold_intent", "gold_is_oos"):
        if not left[column].equals(right[column]):
            raise ValueError(f"gold mismatch {column}: {dataset}/{kir}/{seed}")
    counts = pd.crosstab(left.outcome, right.outcome).reindex(index=OUTCOMES, columns=OUTCOMES, fill_value=0)
    rows: list[dict[str, Any]] = []
    for left_outcome in OUTCOMES:
        for right_outcome in OUTCOMES:
            count = int(counts.loc[left_outcome, right_outcome])
            rows.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "seed": seed,
                    "trainable_outcome": left_outcome,
                    "mogb_outcome": right_outcome,
                    "count": count,
                    "rate_all": count / len(left),
                }
            )
    return pd.DataFrame(rows)


def cell_decomposition(left: pd.DataFrame, right: pd.DataFrame, dataset: str, kir: float, seed: int) -> dict[str, Any]:
    left_correct = left.open_prediction.eq(left.open_gold)
    right_correct = right.open_prediction.eq(right.open_gold)
    known = left.gold_is_oos.eq(0)
    oos = ~known

    def count(mask: pd.Series) -> int:
        return int(mask.sum())

    row: dict[str, Any] = {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "n_samples": len(left),
        "n_known": count(known),
        "n_oos": count(oos),
        "known_trainable_only_correct": count(known & left_correct & ~right_correct),
        "known_mogb_only_correct": count(known & ~left_correct & right_correct),
        "oos_trainable_only_correct": count(oos & left_correct & ~right_correct),
        "oos_mogb_only_correct": count(oos & ~left_correct & right_correct),
        "known_trainable_recovers_mogb_reject": count(known & left.outcome.eq("known_correct") & right.outcome.eq("known_rejected")),
        "known_trainable_fixes_mogb_wrong": count(known & left.outcome.eq("known_correct") & right.outcome.eq("known_wrong_intent")),
        "known_mogb_recovers_trainable_reject": count(known & right.outcome.eq("known_correct") & left.outcome.eq("known_rejected")),
        "known_mogb_fixes_trainable_wrong": count(known & right.outcome.eq("known_correct") & left.outcome.eq("known_wrong_intent")),
        "trainable_f1_all": float(f1_score(left.open_gold, left.open_prediction, average="macro", zero_division=0)),
        "mogb_f1_all": float(f1_score(right.open_gold, right.open_prediction, average="macro", zero_division=0)),
    }
    row["net_known_correct_gain"] = row["known_trainable_only_correct"] - row["known_mogb_only_correct"]
    row["net_oos_correct_gain"] = row["oos_trainable_only_correct"] - row["oos_mogb_only_correct"]
    row["net_total_correct_gain"] = row["net_known_correct_gain"] + row["net_oos_correct_gain"]
    row["f1_all_delta"] = row["trainable_f1_all"] - row["mogb_f1_all"]
    return row


def known_intent_rows(left: pd.DataFrame, right: pd.DataFrame, dataset: str, kir: float, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    known = left.gold_is_oos.eq(0)
    for intent in sorted(left.loc[known, "gold_intent"].unique()):
        mask = known & left.gold_intent.eq(intent)
        n = int(mask.sum())
        rows.append(
            {
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "intent": intent,
                "known_count": n,
                "trainable_correct_rate": float(left.loc[mask, "outcome"].eq("known_correct").mean()),
                "mogb_correct_rate": float(right.loc[mask, "outcome"].eq("known_correct").mean()),
                "trainable_reject_rate": float(left.loc[mask, "outcome"].eq("known_rejected").mean()),
                "mogb_reject_rate": float(right.loc[mask, "outcome"].eq("known_rejected").mean()),
                "trainable_wrong_intent_rate": float(left.loc[mask, "outcome"].eq("known_wrong_intent").mean()),
                "mogb_wrong_intent_rate": float(right.loc[mask, "outcome"].eq("known_wrong_intent").mean()),
            }
        )
    return pd.DataFrame(rows)


def score_bin_rows(left: pd.DataFrame, right: pd.DataFrame, dataset: str, kir: float, seed: int) -> pd.DataFrame:
    known = left.gold_is_oos.eq(0)
    data = pd.DataFrame(
        {
            "mogb_score_bin": pd.cut(right.loc[known, "score"], MOGB_SCORE_BINS, labels=MOGB_SCORE_LABELS),
            "trainable_correct": left.loc[known, "outcome"].eq("known_correct").astype(int),
            "mogb_correct": right.loc[known, "outcome"].eq("known_correct").astype(int),
        }
    )
    grouped = data.groupby("mogb_score_bin", observed=False).agg(
        sample_count=("trainable_correct", "size"),
        trainable_correct_rate=("trainable_correct", "mean"),
        mogb_correct_rate=("mogb_correct", "mean"),
    ).reset_index()
    grouped.insert(0, "seed", seed)
    grouped.insert(0, "kir", kir)
    grouped.insert(0, "dataset", dataset)
    return grouped


def plot_transition_heatmaps(summary: pd.DataFrame) -> None:
    data = summary[summary.kir.eq(0.50)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    vmax = float(data.rate_mean.max())
    for ax, dataset in zip(axes, DATASETS, strict=True):
        sub = data[data.dataset.eq(dataset)].pivot(index="trainable_outcome", columns="mogb_outcome", values="rate_mean").reindex(index=OUTCOMES, columns=OUTCOMES)
        values = sub.to_numpy(float)
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(5), [LABELS[x] for x in OUTCOMES], rotation=38, ha="right")
        ax.set_yticks(range(5), [LABELS[x] for x in OUTCOMES])
        ax.set_title(f"{dataset} / KIR=0.50")
        ax.set_xlabel("MOGB-Fair outcome")
        ax.set_ylabel("S2C-Trainable-K1 outcome")
        for i in range(5):
            for j in range(5):
                if values[i, j] >= 0.005:
                    ax.text(j, i, f"{values[i, j]*100:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes, label="Share of all test samples", shrink=0.82)
    fig.savefig(FIG / "open_intent_transition_heatmaps_kir050.png", dpi=190)
    plt.close(fig)


def plot_gain_decomposition(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values(["dataset", "kir"]).copy()
    labels = [f"{d}\n{k:.2f}" for d, k in zip(ordered.dataset, ordered.kir, strict=True)]
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.bar(x, ordered.net_known_correct_gain_mean / ordered.n_known_mean, label="Net Known correct gain", color="#2b6cb0")
    ax.bar(x, ordered.net_oos_correct_gain_mean / ordered.n_oos_mean, label="Net OOS correct gain", color="#dd6b20", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=0)
    ax.set_ylabel("Net paired correctness gain (rate)")
    ax.set_title("Where S2C-Trainable-K1 gains over MOGB-Fair")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG / "paired_correctness_gain_decomposition.png", dpi=190)
    plt.close(fig)


def plot_score_bin_recovery(summary: pd.DataFrame) -> None:
    data = summary[summary.kir.eq(0.50)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        sub = data[data.dataset.eq(dataset)].set_index("mogb_score_bin").reindex(MOGB_SCORE_LABELS)
        x = np.arange(len(sub))
        ax.plot(x, sub.trainable_correct_rate_mean, marker="o", label="S2C Trainable K1")
        ax.plot(x, sub.mogb_correct_rate_mean, marker="s", label="MOGB-Fair")
        ax.axvline(1.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x, MOGB_SCORE_LABELS, rotation=35, ha="right")
        ax.set_title(dataset)
        ax.set_xlabel("MOGB normalized score bin")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Known intent correct rate")
    axes[0].legend()
    fig.suptitle("Known recovery by MOGB boundary margin (KIR=0.50)")
    fig.savefig(FIG / "known_recovery_by_mogb_score_bin.png", dpi=190)
    plt.close(fig)


def plot_intent_recovery(intent_summary: pd.DataFrame) -> None:
    data = intent_summary[intent_summary.kir.eq(0.50) & intent_summary.n_seeds.ge(3)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        sub = data[data.dataset.eq(dataset)].nlargest(15, "correct_rate_delta").sort_values("correct_rate_delta")
        ax.barh(sub.intent, sub.correct_rate_delta * 100, color="#2f855a")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(dataset)
        ax.set_xlabel("S2C minus MOGB Known correct rate (pp)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Known intents most recovered by S2C-Trainable-K1 (KIR=0.50; >=3 seeds)")
    fig.savefig(FIG / "top_intent_known_recovery_kir050.png", dpi=190)
    plt.close(fig)


def build_report(cell: pd.DataFrame, cell_summary: pd.DataFrame, transition_summary: pd.DataFrame, score_summary: pd.DataFrame) -> str:
    dataset = cell_summary.groupby("dataset", as_index=False).agg(
        f1_all_delta=("f1_all_delta_mean", "mean"),
        known_gain=("net_known_correct_gain_rate", "mean"),
        oos_gain=("net_oos_correct_gain_rate", "mean"),
        recovery_from_reject=("known_trainable_recovers_mogb_reject_mean", "mean"),
        recovery_from_wrong=("known_trainable_fixes_mogb_wrong_mean", "mean"),
    )
    table = dataset.to_markdown(index=False, floatfmt=".4f")
    kir50 = cell_summary[cell_summary.kir.eq(0.50)][[
        "dataset", "f1_all_delta_mean", "net_known_correct_gain_rate", "net_oos_correct_gain_rate",
        "known_trainable_recovers_mogb_reject_mean", "known_trainable_fixes_mogb_wrong_mean",
    ]]
    transition_nonzero = transition_summary[transition_summary.rate_mean.gt(0)].shape[0]
    score_rows = int(score_summary.sample_count_sum.sum())
    prediction_rows = int(cell.n_samples.sum() * 2)
    test_sizes = cell.groupby("dataset").n_samples.first().to_dict()
    return f"""# S2C Trainable K1 与 MOGB-Fair 开放意图结果转移分析 V1

更新时间：2026-08-09  
活动协议：`protocol_v2_textoir_v1`

## 1. 问题与合同

本阶段不再只比较 Known/OOS 二分类 Gate，而是把每条测试样本分成五种互斥结果：Known 正确、Known 错类、Known 被拒、OOS 正确拒绝、OOS 误接收。比较范围为三数据集、KIR=0.25/0.50/0.75、五个正式 seed，共45个逐样本配对单元。

`S2C-Trainable-K1` 是当前 Known-only、最后两层 MiniLM+projection、单中心 Gate；`MOGB-Fair` 是相同 TEXTOIR split 上冻结 MiniLM 的自适应粒球组件。后者不是官方 BERT MOGB。本分析只读取冻结预测，test标签只用于事后归因。

## 2. 完整性

- 配对单元：45/45；Banking77/CLINC150/StackOverflow 每方法测试样本数分别为 `{test_sizes['banking77']}`/`{test_sizes['clinc150']}`/`{test_sizes['stackoverflow']}`；
- 总预测读取量：`{prediction_rows}`行；逐配对 sample_id、gold_intent、Known/OOS标记完全一致；
- 五状态转移组合：25种，非零汇总组合 `{transition_nonzero}`；
- score-bin Known 样本累计计数：`{score_rows}`；
- 每单元重新计算 F1-All，并与冻结汇总核对。

## 3. 跨数据集差距来源

{table}

`known_gain` 和 `oos_gain` 分别是同一配对单元中 S2C-only correct 减去 MOGB-only correct，再除以对应 Known/OOS 样本数。结果直接显示：S2C 相对 MOGB-Fair 的主要优势来自恢复被 MOGB 边界拒绝的 Known 样本，而不是通过额外接受 OOS 换取表面指标。

## 4. KIR=0.50 的配对结果

{kir50.to_markdown(index=False, floatfmt='.4f')}

`known_trainable_recovers_mogb_reject` 是 S2C 正确分类、但 MOGB 判成 OOS 的 Known 样本数；`known_trainable_fixes_mogb_wrong` 是 S2C 修正 MOGB Known 错类的数量。前者明显更大，说明两者差距首先是 MOGB 边界覆盖问题，其次才是粒球之间的 Known 分类竞争。

## 5. 机制结论

1. **F1-All 优势主要来自 Known 覆盖恢复。** MOGB-Fair 的平均 false acceptance 很低，但大量 Known 样本落在所有平均半径粒球之外；S2C 的单中心边界恢复了这些样本。
2. **表示排序仍然重要。** S2C 不只把 MOGB 拒绝的样本强行接收；它需要同时给出正确 Known intent。意图级正确率与 MOGB score-bin 曲线显示，很多 MOGB score>1 的样本仍可被 S2C 正确分类。
3. **MOGB 并非所有错误都来自边界。** 仍存在双方都接受但 MOGB 选错 Known intent、S2C 修正的样本；因此子中心表示/最近球竞争也是次级差距来源。
4. **不能据此比较官方论文 SOTA。** 当前结论只适用于 MOGB-Fair 组件合同；官方 BERT MOGB 的公开代码兼容复现仍需单独解释数据、训练目标和半径合同差异。

## 6. 图表

- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/open_intent_transition_heatmaps_kir050.png`
- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/paired_correctness_gain_decomposition.png`
- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/known_recovery_by_mogb_score_bin.png`
- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/top_intent_known_recovery_kir050.png`

意图图只显示至少在3/5个seed中进入Known集合的意图，避免偶然类别划分产生的单seed极值；完整CSV保留全部意图并记录`n_seeds`。

## 7. 研究边界

- 没有重新训练、调阈值、改split或覆盖历史artifact；
- test标签没有参与模型、边界、K或工作点选择；
- 不输出原始文本或逐样本ID到Git轻量结果；
- 该结果用于解释“为什么当前同协议 S2C 行优于 MOGB-Fair”，不能写成超过完整 MOGB、ADB、DA-ADB或DCLOOS。
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    source = source[source.dataset.isin(DATASETS) & source.kir.isin(KIRS) & source.seed.isin(SEEDS) & source.method.isin(METHODS)].copy()
    expected = len(DATASETS) * len(KIRS) * len(SEEDS) * len(METHODS)
    if len(source) != expected:
        raise ValueError(f"expected {expected} source rows, got {len(source)}")
    hashes = {str(SOURCE.relative_to(ROOT)): sha256(SOURCE)}
    transitions: list[pd.DataFrame] = []
    cells: list[dict[str, Any]] = []
    intents: list[pd.DataFrame] = []
    score_bins: list[pd.DataFrame] = []
    metric_deltas: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                loaded: dict[str, pd.DataFrame] = {}
                source_rows: dict[str, pd.Series] = {}
                for method in METHODS:
                    row = source[(source.dataset.eq(dataset)) & (source.kir.eq(kir)) & (source.seed.eq(seed)) & (source.method.eq(method))].iloc[0]
                    prediction, path = load_prediction(row)
                    hashes[str(path)] = sha256(path)
                    loaded[method] = prediction
                    source_rows[method] = row
                left = loaded["trainable_k1"]
                right = loaded["mogb_minilm"]
                transitions.append(transition_table(left, right, dataset, kir, seed))
                cell = cell_decomposition(left, right, dataset, kir, seed)
                for method, prediction in loaded.items():
                    recomputed = float(f1_score(prediction.open_gold, prediction.open_prediction, average="macro", zero_division=0))
                    stored = float(source_rows[method]["f1_all"])
                    metric_deltas.append({"dataset": dataset, "kir": kir, "seed": seed, "method": method, "recomputed_f1_all": recomputed, "source_f1_all": stored, "abs_delta": abs(recomputed - stored)})
                cells.append(cell)
                intents.append(known_intent_rows(left, right, dataset, kir, seed))
                score_bins.append(score_bin_rows(left, right, dataset, kir, seed))

    transition = pd.concat(transitions, ignore_index=True)
    cell = pd.DataFrame(cells)
    intent = pd.concat(intents, ignore_index=True)
    score_bin = pd.concat(score_bins, ignore_index=True)
    metric_delta = pd.DataFrame(metric_deltas)
    if metric_delta.abs_delta.max() > 1e-10:
        raise ValueError(f"F1-All replay mismatch: {metric_delta.abs_delta.max()}")

    transition_summary = transition.groupby(["dataset", "kir", "trainable_outcome", "mogb_outcome"], as_index=False).agg(rate_mean=("rate_all", "mean"), rate_std=("rate_all", "std"), count_sum=("count", "sum"), n_seeds=("seed", "nunique"))
    cell_summary = cell.groupby(["dataset", "kir"], as_index=False).agg(
        n_known_mean=("n_known", "mean"), n_oos_mean=("n_oos", "mean"),
        net_known_correct_gain_mean=("net_known_correct_gain", "mean"), net_known_correct_gain_std=("net_known_correct_gain", "std"),
        net_oos_correct_gain_mean=("net_oos_correct_gain", "mean"), net_oos_correct_gain_std=("net_oos_correct_gain", "std"),
        known_trainable_recovers_mogb_reject_mean=("known_trainable_recovers_mogb_reject", "mean"),
        known_trainable_fixes_mogb_wrong_mean=("known_trainable_fixes_mogb_wrong", "mean"),
        known_mogb_recovers_trainable_reject_mean=("known_mogb_recovers_trainable_reject", "mean"),
        known_mogb_fixes_trainable_wrong_mean=("known_mogb_fixes_trainable_wrong", "mean"),
        trainable_f1_all_mean=("trainable_f1_all", "mean"), mogb_f1_all_mean=("mogb_f1_all", "mean"),
        f1_all_delta_mean=("f1_all_delta", "mean"), f1_all_delta_std=("f1_all_delta", "std"), n_seeds=("seed", "nunique"),
    )
    cell_summary["net_known_correct_gain_rate"] = cell_summary.net_known_correct_gain_mean / cell_summary.n_known_mean
    cell_summary["net_oos_correct_gain_rate"] = cell_summary.net_oos_correct_gain_mean / cell_summary.n_oos_mean
    intent_summary = intent.groupby(["dataset", "kir", "intent"], as_index=False).agg(
        known_count_mean=("known_count", "mean"), trainable_correct_rate=("trainable_correct_rate", "mean"), mogb_correct_rate=("mogb_correct_rate", "mean"),
        trainable_reject_rate=("trainable_reject_rate", "mean"), mogb_reject_rate=("mogb_reject_rate", "mean"),
        trainable_wrong_intent_rate=("trainable_wrong_intent_rate", "mean"), mogb_wrong_intent_rate=("mogb_wrong_intent_rate", "mean"), n_seeds=("seed", "nunique"),
    )
    intent_summary["correct_rate_delta"] = intent_summary.trainable_correct_rate - intent_summary.mogb_correct_rate
    score_summary = score_bin.groupby(["dataset", "kir", "mogb_score_bin"], observed=False, as_index=False).agg(
        sample_count_sum=("sample_count", "sum"), trainable_correct_rate_mean=("trainable_correct_rate", "mean"), mogb_correct_rate_mean=("mogb_correct_rate", "mean"), n_seeds=("seed", "nunique")
    )

    atomic_csv(cell, OUT / "cell_decomposition_per_seed.csv")
    atomic_csv(cell_summary, OUT / "cell_decomposition_summary.csv")
    atomic_csv(transition_summary, OUT / "open_intent_transition_summary.csv")
    atomic_csv(intent_summary, OUT / "intent_known_outcome_summary.csv")
    atomic_csv(score_summary, OUT / "mogb_score_bin_known_recovery.csv")
    atomic_csv(metric_delta, OUT / "metric_replay_audit.csv")
    # Raw paired transitions contain no sample IDs/text and remain in the artifact root.
    atomic_csv(transition, ARTIFACT / "open_intent_transitions_per_seed.csv")
    atomic_csv(intent, ARTIFACT / "intent_known_outcomes_per_seed.csv")
    atomic_csv(score_bin, ARTIFACT / "score_bin_known_recovery_per_seed.csv")

    plot_transition_heatmaps(transition_summary)
    plot_gain_decomposition(cell_summary)
    plot_score_bin_recovery(score_summary)
    plot_intent_recovery(intent_summary)
    atomic_text(build_report(cell, cell_summary, transition_summary, score_summary), REPORT)

    outputs = [*OUT.glob("*.csv"), *FIG.glob("*.png"), REPORT]
    manifest = {
        "analysis_id": "trainable_mogb_open_intent_transitions_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "analysis_only": True,
        "paired_cells": len(cell),
        "prediction_rows_read": int(cell.n_samples.sum() * 2),
        "datasets": list(DATASETS), "kirs": list(KIRS), "seeds": list(SEEDS), "methods": list(METHODS),
        "max_f1_all_replay_abs_delta": float(metric_delta.abs_delta.max()),
        "test_labels_used_for_selection": False,
        "source_sha256": hashes,
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(outputs)},
    }
    atomic_json(manifest, OUT / "MANIFEST.json")
    closeout = {"status": "complete", "analysis_id": manifest["analysis_id"], "paired_cells": len(cell), "failed_cells": 0, "manifest_sha256": sha256(OUT / "MANIFEST.json"), "report": str(REPORT)}
    atomic_json(closeout, ARTIFACT / "CLOSEOUT.json")
    print(json.dumps(closeout, ensure_ascii=False))


if __name__ == "__main__":
    main()
