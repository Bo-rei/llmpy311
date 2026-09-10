"""Bridge MOGB ball structure to per-intent S2C recovery and OOS cost.

The inputs are frozen, Git-light analysis tables produced by completed
protocol_v2_textoir_v1 stages.  This stage does not read text, embeddings, or
checkpoints and never changes a detector.  Test outcomes are used only for
post-hoc mechanism attribution, never for selecting balls or thresholds.
"""

from __future__ import annotations

import argparse
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
ARCHIVE_ANALYSIS = ROOT / "results" / "analysis" / "archive" / "analysis"
ERROR_INPUT = ARCHIVE_ANALYSIS / "cross_dataset_error_attribution_v1" / "intent_error_attribution_per_seed.csv"
BALL_INPUT = ARCHIVE_ANALYSIS / "mogb_ball_risk_attribution_v1" / "per_ball.csv"
DEFAULT_OUTPUT = ARCHIVE_ANALYSIS / "s2c_mogb_intent_structure_bridge_v1"
DEFAULT_FIGURES = ROOT / "figures" / "archive" / "analysis" / "s2c_mogb_intent_structure_bridge_v1"
DEFAULT_REPORT = ROOT / "docs" / "archive" / "analysis" / "S2C_MOGB_INTENT_STRUCTURE_BRIDGE_V1.md"

DATASETS = ("banking77", "clinc150", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
KEYS = ["dataset", "kir", "seed", "intent"]
BALL_GROUPS = ("0", "1", "2", "3+")
DATASET_COLORS = {"banking77": "#0072B2", "clinc150": "#CC79A7", "stackoverflow": "#D55E00"}
KIR_MARKERS = {0.25: "o", 0.50: "s", 0.75: "^"}

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, format=path.suffix.lstrip("."), dpi=190, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def ball_group(value: float | int) -> str:
    count = int(value)
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if len(values) == 0 or float(values.sum()) == 0.0:
        return 0.0
    ordered = np.sort(values)
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    return float((2.0 * np.sum(ranks * ordered) / (len(ordered) * ordered.sum())) - (len(ordered) + 1) / len(ordered))


def top_fraction_share(values: np.ndarray, fraction: float) -> float:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive) == 0:
        return 0.0
    count = max(1, int(np.ceil(len(positive) * fraction)))
    return float(np.sort(positive)[::-1][:count].sum() / positive.sum())


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    errors = pd.read_csv(ERROR_INPUT)
    balls = pd.read_csv(BALL_INPUT)
    required_errors = {
        "dataset",
        "kir",
        "seed",
        "method",
        "intent",
        "known_count",
        "known_false_reject",
        "known_false_reject_rate",
        "oos_false_accept",
        "oos_false_accept_rate",
    }
    required_balls = {
        "dataset",
        "kir",
        "seed",
        "method",
        "majority_label",
        "ball_id",
        "radius",
        "sample_count",
        "depth",
        "tiny_support_lt_20",
        "selected",
    }
    if missing := required_errors.difference(errors.columns):
        raise ValueError(f"intent error input missing columns: {sorted(missing)}")
    if missing := required_balls.difference(balls.columns):
        raise ValueError(f"ball input missing columns: {sorted(missing)}")
    errors = errors[errors["method"].isin(["trainable_k1", "mogb_minilm"])].copy()
    balls = balls[balls["method"].eq("mogb_minilm") & balls["selected"].astype(bool)].copy()
    if errors.duplicated(KEYS + ["method"]).any():
        raise ValueError("duplicate intent-method rows")
    if balls.duplicated(["dataset", "kir", "seed", "ball_id"]).any():
        raise ValueError("duplicate MOGB ball rows")
    return errors, balls


def aggregate_ball_structure(balls: pd.DataFrame) -> pd.DataFrame:
    grouped = balls.groupby(["dataset", "kir", "seed", "majority_label"], as_index=False).agg(
        ball_count=("ball_id", "nunique"),
        train_support_total=("sample_count", "sum"),
        mean_ball_support=("sample_count", "mean"),
        min_ball_support=("sample_count", "min"),
        max_ball_support=("sample_count", "max"),
        mean_radius=("radius", "mean"),
        max_radius=("radius", "max"),
        max_depth=("depth", "max"),
        tiny_ball_ratio=("tiny_support_lt_20", "mean"),
    )
    grouped["dominant_ball_ratio"] = grouped["max_ball_support"] / grouped["train_support_total"].clip(lower=1)
    return grouped.rename(columns={"majority_label": "intent"})


def build_intent_bridge(errors: pd.DataFrame, balls: pd.DataFrame) -> pd.DataFrame:
    trainable = errors[errors["method"].eq("trainable_k1")].copy()
    mogb = errors[errors["method"].eq("mogb_minilm")].copy()
    keep = KEYS + [
        "known_count",
        "known_false_reject",
        "known_false_reject_rate",
        "oos_false_accept",
        "oos_false_accept_rate",
    ]
    trainable = trainable[keep].rename(
        columns={
            "known_count": "trainable_known_count",
            "known_false_reject": "trainable_known_false_reject",
            "known_false_reject_rate": "trainable_known_false_reject_rate",
            "oos_false_accept": "trainable_oos_false_accept",
            "oos_false_accept_rate": "trainable_oos_false_accept_rate",
        }
    )
    mogb = mogb[keep].rename(
        columns={
            "known_count": "mogb_known_count",
            "known_false_reject": "mogb_known_false_reject",
            "known_false_reject_rate": "mogb_known_false_reject_rate",
            "oos_false_accept": "mogb_oos_false_accept",
            "oos_false_accept_rate": "mogb_oos_false_accept_rate",
        }
    )
    bridge = trainable.merge(mogb, on=KEYS, validate="one_to_one")
    if not np.array_equal(bridge["trainable_known_count"], bridge["mogb_known_count"]):
        raise ValueError("Known intent sample counts differ between methods")
    bridge["known_count"] = bridge["trainable_known_count"]
    bridge["known_recovery_count"] = (
        bridge["mogb_known_false_reject"] - bridge["trainable_known_false_reject"]
    )
    bridge["known_recovery_rate"] = (
        bridge["mogb_known_false_reject_rate"] - bridge["trainable_known_false_reject_rate"]
    )
    bridge["extra_oos_accept_count"] = (
        bridge["trainable_oos_false_accept"] - bridge["mogb_oos_false_accept"]
    )
    bridge["extra_oos_accept_rate"] = (
        bridge["trainable_oos_false_accept_rate"] - bridge["mogb_oos_false_accept_rate"]
    )
    bridge["coverage_tradeoff_net_count"] = bridge["known_recovery_count"] - bridge["extra_oos_accept_count"]

    structure = aggregate_ball_structure(balls)
    bridge = bridge.merge(structure, on=KEYS, how="left", validate="one_to_one")
    fill_zero = [
        "ball_count",
        "train_support_total",
        "mean_ball_support",
        "min_ball_support",
        "max_ball_support",
        "max_depth",
        "tiny_ball_ratio",
        "dominant_ball_ratio",
    ]
    bridge[fill_zero] = bridge[fill_zero].fillna(0.0)
    bridge["missing_selected_ball"] = bridge["ball_count"].eq(0)
    bridge["ball_count_group"] = bridge["ball_count"].map(ball_group)
    return bridge.sort_values(KEYS).reset_index(drop=True)


def concentration_table(bridge: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, kir, seed), group in bridge.groupby(["dataset", "kir", "seed"], sort=True):
        positive = group["known_recovery_count"].clip(lower=0).to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "known_intents": int(len(group)),
                "positive_recovery_intents": int(np.sum(positive > 0)),
                "positive_recovery_intent_ratio": float(np.mean(positive > 0)),
                "total_positive_known_recovery": float(positive.sum()),
                "top10pct_recovery_share": top_fraction_share(positive, 0.10),
                "top20pct_recovery_share": top_fraction_share(positive, 0.20),
                "top50pct_recovery_share": top_fraction_share(positive, 0.50),
                "recovery_gini": gini(positive),
            }
        )
    return pd.DataFrame(rows)


def build_summaries(bridge: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_summary = bridge.groupby(["dataset", "kir", "ball_count_group"], as_index=False, observed=True).agg(
        intent_seed_rows=("intent", "size"),
        intent_count=("intent", "nunique"),
        known_recovery_rate_mean=("known_recovery_rate", "mean"),
        known_recovery_count_mean=("known_recovery_count", "mean"),
        extra_oos_accept_rate_mean=("extra_oos_accept_rate", "mean"),
        extra_oos_accept_count_mean=("extra_oos_accept_count", "mean"),
        coverage_tradeoff_net_count_mean=("coverage_tradeoff_net_count", "mean"),
        missing_selected_ball_ratio=("missing_selected_ball", "mean"),
    )
    group_summary["ball_count_group"] = pd.Categorical(
        group_summary["ball_count_group"], categories=BALL_GROUPS, ordered=True
    )
    group_summary = group_summary.sort_values(["dataset", "kir", "ball_count_group"])

    cell_summary = bridge.groupby(["dataset", "kir", "seed"], as_index=False).agg(
        known_recovery_count=("known_recovery_count", "sum"),
        extra_oos_accept_count=("extra_oos_accept_count", "sum"),
        coverage_tradeoff_net_count=("coverage_tradeoff_net_count", "sum"),
        positive_recovery_intent_ratio=("known_recovery_rate", lambda series: float(np.mean(series > 0))),
        negative_recovery_intent_ratio=("known_recovery_rate", lambda series: float(np.mean(series < 0))),
        missing_selected_ball_ratio=("missing_selected_ball", "mean"),
        mean_ball_count=("ball_count", "mean"),
        mean_radius=("mean_radius", "mean"),
    )
    dataset_summary = cell_summary.groupby(["dataset", "kir"], as_index=False).agg(
        known_recovery_count_mean=("known_recovery_count", "mean"),
        known_recovery_count_std=("known_recovery_count", "std"),
        extra_oos_accept_count_mean=("extra_oos_accept_count", "mean"),
        extra_oos_accept_count_std=("extra_oos_accept_count", "std"),
        coverage_tradeoff_net_count_mean=("coverage_tradeoff_net_count", "mean"),
        positive_recovery_intent_ratio_mean=("positive_recovery_intent_ratio", "mean"),
        missing_selected_ball_ratio_mean=("missing_selected_ball_ratio", "mean"),
        mean_ball_count=("mean_ball_count", "mean"),
    )

    intent_summary = bridge.groupby(["dataset", "kir", "intent"], as_index=False).agg(
        observed_seeds=("seed", "nunique"),
        known_count_mean=("known_count", "mean"),
        known_recovery_rate_mean=("known_recovery_rate", "mean"),
        known_recovery_rate_std=("known_recovery_rate", "std"),
        known_recovery_count_mean=("known_recovery_count", "mean"),
        extra_oos_accept_count_mean=("extra_oos_accept_count", "mean"),
        extra_oos_accept_rate_mean=("extra_oos_accept_rate", "mean"),
        mean_ball_count=("ball_count", "mean"),
        mean_radius=("mean_radius", "mean"),
        min_ball_support_mean=("min_ball_support", "mean"),
        missing_selected_ball_ratio=("missing_selected_ball", "mean"),
        positive_seed_ratio=("known_recovery_rate", lambda series: float(np.mean(series > 0))),
    )
    return group_summary, cell_summary, dataset_summary, intent_summary


def correlation_table(bridge: pd.DataFrame) -> pd.DataFrame:
    predictors = ("ball_count", "mean_radius", "min_ball_support", "tiny_ball_ratio", "dominant_ball_ratio")
    outcomes = ("known_recovery_rate", "extra_oos_accept_rate")
    rows: list[dict[str, object]] = []
    for (dataset, kir), group in bridge.groupby(["dataset", "kir"], sort=True):
        for predictor in predictors:
            for outcome in outcomes:
                finite = group[[predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
                rho = float(finite[predictor].corr(finite[outcome], method="spearman")) if len(finite) >= 5 else float("nan")
                rows.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "predictor": predictor,
                        "outcome": outcome,
                        "spearman_rho": rho,
                        "n_intent_seed_rows": int(len(finite)),
                    }
                )
    return pd.DataFrame(rows)


def plot_recovery_distributions(bridge: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        values = [
            bridge[bridge["dataset"].eq(dataset) & bridge["kir"].eq(kir)]["known_recovery_rate"].to_numpy() * 100
            for kir in KIRS
        ]
        boxes = axis.boxplot(values, patch_artist=True, tick_labels=[f"{kir:.2f}" for kir in KIRS], showfliers=False)
        for patch, color in zip(boxes["boxes"], ("#B3E2CD", "#FDCDAC", "#CBD5E8"), strict=True):
            patch.set_facecolor(color)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(dataset)
        axis.set_xlabel("KIR")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Known拒绝率差值：MOGB - S2C（pp）")
    figure.suptitle("逐 intent 的 S2C Known覆盖恢复分布", fontsize=14)
    atomic_figure(figure, path)


def plot_ball_group_tradeoff(group_summary: pd.DataFrame, path: Path) -> None:
    selected = group_summary[group_summary["kir"].eq(0.50)].copy()
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.5), sharey=True, constrained_layout=True)
    x = np.arange(len(BALL_GROUPS), dtype=float)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        rows = selected[selected["dataset"].eq(dataset)].set_index("ball_count_group").reindex(BALL_GROUPS)
        recovery = rows["known_recovery_rate_mean"].fillna(0).to_numpy() * 100
        oos_cost = rows["extra_oos_accept_rate_mean"].fillna(0).to_numpy() * 100
        axis.bar(x - 0.18, recovery, 0.36, label="Known恢复", color="#0072B2")
        axis.bar(x + 0.18, oos_cost, 0.36, label="新增OOS误收", color="#D55E00")
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(x, BALL_GROUPS)
        axis.set_title(dataset)
        axis.set_xlabel("MOGB selected balls / intent")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("平均变化（pp）")
    axes[-1].legend(loc="best", fontsize=9)
    figure.suptitle("KIR=0.50：粒球数量与S2C覆盖收益/OOS代价", fontsize=14)
    atomic_figure(figure, path)


def plot_structure_scatter(intent_summary: pd.DataFrame, path: Path) -> None:
    selected = intent_summary[intent_summary["kir"].eq(0.50)].copy()
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.5), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(20260809)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        rows = selected[selected["dataset"].eq(dataset)]
        jitter = rng.normal(0, 0.06, size=len(rows))
        axis.scatter(
            rows["mean_ball_count"] + jitter,
            rows["known_recovery_rate_mean"] * 100,
            s=np.clip(rows["known_count_mean"], 20, 120),
            color=DATASET_COLORS[dataset],
            alpha=0.65,
            edgecolor="white",
            linewidth=0.4,
        )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(dataset)
        axis.set_xlabel("每 intent 平均 selected balls")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Known覆盖恢复（pp）")
    figure.suptitle("KIR=0.50：MOGB结构复杂度与S2C恢复幅度", fontsize=14)
    atomic_figure(figure, path)


def plot_concentration(concentration: pd.DataFrame, path: Path) -> None:
    summary = concentration.groupby(["dataset", "kir"], as_index=False)[
        ["top10pct_recovery_share", "top20pct_recovery_share", "top50pct_recovery_share"]
    ].mean()
    selected = summary[summary["kir"].eq(0.50)]
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.23
    for index, (column, label) in enumerate(
        (
            ("top10pct_recovery_share", "Top 10% intents"),
            ("top20pct_recovery_share", "Top 20% intents"),
            ("top50pct_recovery_share", "Top 50% intents"),
        )
    ):
        values = selected.set_index("dataset").reindex(DATASETS)[column].to_numpy()
        axis.bar(x + (index - 1) * width, values, width, label=label)
    axis.set_xticks(x, DATASETS)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("正向 Known 恢复贡献占比")
    axis.set_title("KIR=0.50：S2C Known恢复是否只来自少数intent")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    atomic_figure(figure, path)


def plot_top_intents(intent_summary: pd.DataFrame, path: Path) -> None:
    selected = intent_summary[intent_summary["kir"].eq(0.50)]
    figure, axes = plt.subplots(1, 3, figsize=(16, 6.2), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        rows = selected[selected["dataset"].eq(dataset)].nlargest(10, "known_recovery_rate_mean").sort_values(
            "known_recovery_rate_mean"
        )
        bars = axis.barh(rows["intent"], rows["known_recovery_rate_mean"] * 100, color=DATASET_COLORS[dataset])
        for bar, balls in zip(bars, rows["mean_ball_count"], strict=True):
            axis.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2, f"K={balls:.1f}", va="center", fontsize=8)
        axis.set_title(dataset)
        axis.set_xlabel("Known覆盖恢复（pp）")
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("KIR=0.50：S2C相对MOGB恢复最大的Known intents", fontsize=14)
    atomic_figure(figure, path)


def plot_recovery_cost(cell_summary: pd.DataFrame, path: Path) -> None:
    summary = cell_summary.groupby(["dataset", "kir"], as_index=False)[
        ["known_recovery_count", "extra_oos_accept_count"]
    ].mean()
    figure, axis = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    for _, row in summary.iterrows():
        dataset = str(row["dataset"])
        kir = float(row["kir"])
        axis.scatter(
            row["extra_oos_accept_count"],
            row["known_recovery_count"],
            s=100,
            color=DATASET_COLORS[dataset],
            marker=KIR_MARKERS[kir],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.annotate(
            f"{dataset} {kir:.2f}",
            (row["extra_oos_accept_count"], row["known_recovery_count"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    bounds = axis.get_xlim() + axis.get_ylim()
    low = min(bounds)
    high = max(bounds)
    axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1, label="样本数收益=代价")
    axis.set_xlabel("S2C 相对 MOGB 新增 OOS 误接收数")
    axis.set_ylabel("S2C 相对 MOGB 恢复的 Known 覆盖数")
    axis.set_title("逐单元 Known恢复与OOS代价（五seed均值）")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    atomic_figure(figure, path)


def render_report(
    bridge: pd.DataFrame,
    group_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    correlations: pd.DataFrame,
    manifest_hash: str,
) -> str:
    overall = {
        "intent_seed_rows": int(len(bridge)),
        "known_recovery_rate_mean": float(bridge["known_recovery_rate"].mean()),
        "extra_oos_accept_rate_mean": float(bridge["extra_oos_accept_rate"].mean()),
        "missing_selected_ball_ratio": float(bridge["missing_selected_ball"].mean()),
    }
    dataset_table = dataset_summary.copy()
    for column in [
        "positive_recovery_intent_ratio_mean",
        "missing_selected_ball_ratio_mean",
    ]:
        dataset_table[column] *= 100
    selected_groups = group_summary[group_summary["kir"].eq(0.50)].copy()
    selected_groups["known_recovery_rate_mean"] *= 100
    selected_groups["extra_oos_accept_rate_mean"] *= 100
    concentration_summary = concentration.groupby(["dataset", "kir"], as_index=False)[
        ["positive_recovery_intent_ratio", "top10pct_recovery_share", "top20pct_recovery_share", "recovery_gini"]
    ].mean()
    concentration_summary[[
        "positive_recovery_intent_ratio",
        "top10pct_recovery_share",
        "top20pct_recovery_share",
    ]] *= 100
    recovery_correlations = correlations[correlations["outcome"].eq("known_recovery_rate")]
    ball_rho = recovery_correlations[recovery_correlations["predictor"].eq("ball_count")]["spearman_rho"]
    return f"""# S2C 与 MOGB-Fair 逐意图结构--收益桥接 V1

> 本报告联合两份已冻结的 analysis-only 证据：逐 intent 错误归因与 MOGB selected-ball 结构。
> test错误只用于事后解释，不能删球、选K、调阈值或生成自适应方法。

## 核心结论

1. 共桥接 `{overall['intent_seed_rows']}` 条 intent×seed 记录。S2C 相对 MOGB 的逐intent平均 Known
   拒绝率降低 `{overall['known_recovery_rate_mean'] * 100:.2f}pp`，同时每个intent平均新增 OOS
   误接收率 `{overall['extra_oos_accept_rate_mean'] * 100:.2f}pp`。
2. `ball_count=0` 的行明确表示 MOGB selected-ball 阶段遗漏该注册 Known intent，其占比为
   `{overall['missing_selected_ball_ratio'] * 100:.2f}%`；它是覆盖缺陷，但不是全部差距。
3. 每个 dataset/KIR 中，正向 Known 恢复 intent 的比例和恢复集中度见表。该分析区分“广泛恢复”与
   “少数高风险 intent 主导”，避免只看全局均值。
4. selected-ball数量与 Known恢复率的九个 dataset×KIR Spearman rho 范围为
   `{ball_rho.min():+.3f}` 到 `{ball_rho.max():+.3f}`。这说明自适应粒球复杂度与S2C优势的关系具有
   数据集/KIR依赖性，不能把更多粒球直接解释为更好的开放集结构。
5. 样本数恢复--代价图只衡量 Gate 覆盖，不等于完整 Known 分类正确率；完整五状态分类证据仍以
   `TRAINABLE_MOGB_OPEN_INTENT_TRANSITIONS_V1.md` 为准。

## dataset×KIR 汇总

{dataset_table.to_markdown(index=False, floatfmt='.2f')}

## KIR=0.50 按 MOGB 每intent粒球数分组

{selected_groups[['dataset', 'ball_count_group', 'intent_seed_rows', 'known_recovery_rate_mean', 'extra_oos_accept_rate_mean', 'coverage_tradeoff_net_count_mean']].to_markdown(index=False, floatfmt='.2f')}

## 恢复集中度

{concentration_summary.to_markdown(index=False, floatfmt='.2f')}

## 六张机制图

1. `intent_known_recovery_distribution.png`：逐intent Known覆盖恢复的全分布。
2. `ball_count_recovery_tradeoff.png`：粒球数分组后的 Known恢复与新增OOS误收。
3. `ball_count_vs_recovery.png`：结构复杂度与恢复幅度散点。
4. `known_recovery_concentration.png`：Top intents承担多少恢复贡献。
5. `top_intent_recovery_kir050.png`：三个数据集恢复最大的具体intent及其平均粒球数。
6. `known_recovery_vs_oos_cost.png`：九个dataset×KIR的样本数收益--代价。

## 解释边界

- MOGB-Fair 使用 Frozen MiniLM；S2C 使用 Known-only Trainable MiniLM K=1，因此结构与表示同时变化。
- 该桥接说明 MOGB 的 adaptive balls 在哪些intent上形成覆盖风险，不证明完整BERT MOGB无效。
- 所有 test-defined intent风险和相关性只用于解释，不得成为新的ball selector。
- Manifest SHA256：`{manifest_hash}`。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors, balls = load_inputs()
    bridge = build_intent_bridge(errors, balls)
    group_summary, cell_summary, dataset_summary, intent_summary = build_summaries(bridge)
    concentration = concentration_table(bridge)
    correlations = correlation_table(bridge)

    outputs = {
        "intent_seed_bridge.csv": bridge,
        "ball_count_group_summary.csv": group_summary,
        "cell_tradeoff_summary.csv": cell_summary,
        "dataset_kir_summary.csv": dataset_summary,
        "intent_summary.csv": intent_summary,
        "recovery_concentration.csv": concentration,
        "structure_correlations.csv": correlations,
    }
    for name, frame in outputs.items():
        atomic_csv(frame, args.output_dir / name)

    figures = {
        "intent_known_recovery_distribution.png": lambda path: plot_recovery_distributions(bridge, path),
        "ball_count_recovery_tradeoff.png": lambda path: plot_ball_group_tradeoff(group_summary, path),
        "ball_count_vs_recovery.png": lambda path: plot_structure_scatter(intent_summary, path),
        "known_recovery_concentration.png": lambda path: plot_concentration(concentration, path),
        "top_intent_recovery_kir050.png": lambda path: plot_top_intents(intent_summary, path),
        "known_recovery_vs_oos_cost.png": lambda path: plot_recovery_cost(cell_summary, path),
    }
    for name, builder in figures.items():
        builder(args.figure_dir / name)

    output_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in [
            *(args.output_dir / name for name in outputs),
            *(args.figure_dir / name for name in figures),
        ]
    }
    manifest = {
        "stage": "ANALYSIS_S2C_MOGB_INTENT_STRUCTURE_BRIDGE_V1",
        "protocol_version": "protocol_v2_textoir_v1",
        "status": "complete",
        "analysis_only": True,
        "test_labels_used_for_posthoc_attribution": True,
        "intent_seed_rows": int(len(bridge)),
        "cell_rows": int(len(cell_summary)),
        "intent_summary_rows": int(len(intent_summary)),
        "correlation_rows": int(len(correlations)),
        "source_sha256": {
            str(ERROR_INPUT.relative_to(ROOT)): sha256(ERROR_INPUT),
            str(BALL_INPUT.relative_to(ROOT)): sha256(BALL_INPUT),
        },
        "output_sha256": output_hashes,
    }
    manifest_path = args.output_dir / "MANIFEST.json"
    atomic_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", manifest_path)
    manifest_hash = sha256(manifest_path)
    atomic_text(
        render_report(bridge, group_summary, dataset_summary, concentration, correlations, manifest_hash),
        args.report,
    )
    atomic_text(
        "# S2C--MOGB intent structure bridge closeout\n\n"
        f"- status: complete\n- intent-seed rows: {len(bridge)}\n- cell rows: {len(cell_summary)}\n"
        f"- failed units: 0\n- manifest SHA256: `{manifest_hash}`\n"
        "- scope: post-hoc mechanism attribution; no model or structure selection\n",
        args.output_dir / "CLOSEOUT.md",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "intent_seed_rows": len(bridge),
                "cell_rows": len(cell_summary),
                "intent_summary_rows": len(intent_summary),
                "correlation_rows": len(correlations),
                "figures": len(figures),
                "manifest_sha256": manifest_hash,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
