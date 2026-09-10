"""Build representation-geometry and near-OOS visual evidence.

This is an analysis-only stage.  It joins already completed light summaries;
it does not train, choose a checkpoint, tune a threshold, or rewrite any
historical run.  The figures are intended to answer two separate questions:

* does a trainable representation improve the K=1 score geometry?
* does that improvement survive when a fixed multi-centre union is used?

The latter is deliberately displayed as a diagnostic delta, never as a
selection rule.  Test labels are used only for post-hoc aggregation.
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

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "representation_geometry_visuals_v1"
FIG = ROOT / "figures" / "representation_geometry_visuals_v1"
REPORT = ROOT / "docs" / "analysis" / "REPRESENTATION_GEOMETRY_VISUALS_V1.md"

DATASET_ORDER = ["clinc150", "banking77", "stackoverflow"]
DATASET_LABELS = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}
REP_ORDER = ["frozen", "ce", "supcon"]
REP_LABELS = {"frozen": "Frozen", "ce": "CE", "supcon": "SupCon", "trainable": "Trainable"}
REP_COLORS = {"frozen": "#666666", "ce": "#0072B2", "supcon": "#D55E00", "trainable": "#009E73"}
REP_MARKERS = {"frozen": "o", "ce": "s", "supcon": "^"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def atomic_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, format=path.suffix.lstrip("."), dpi=180, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(path)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = {
        "geometry": ROOT / "results" / "analysis" / "representation_boundary_pack_v1" / "representation_geometry.csv",
        "k_summary": ROOT / "results" / "analysis" / "representation_boundary_pack_v1" / "representation_k1_k2_summary.csv",
        "boundary": ROOT / "results" / "analysis" / "minilm_boundary_diagnostics_v1" / "run_summary.csv",
        "transitions": ROOT / "results" / "analysis" / "raw_gate_error_visualization_v1" / "pairwise_transitions.csv",
    }
    geometry = read_csv(paths["geometry"])
    k_summary = read_csv(paths["k_summary"])
    boundary = read_csv(paths["boundary"])
    transitions = read_csv(paths["transitions"])
    if set(geometry["dataset"]) != set(DATASET_ORDER):
        raise ValueError("geometry input does not contain all three datasets")
    if not set(k_summary["representation"]).issuperset(REP_ORDER):
        raise ValueError("K summary is missing Frozen/CE/SupCon")
    if not {"trainable", "frozen"}.issubset(set(boundary["representation"])):
        raise ValueError("boundary input is missing Trainable/Frozen rows")
    return geometry, k_summary, boundary, transitions, {key: sha256(value) for key, value in paths.items()}


def aggregate_geometry(geometry: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "linear_probe_val_macro_f1",
        "purity_at_10",
        "relative_separation",
        "effective_rank",
        "uniformity",
        "same_intent_alignment",
    ]
    result = geometry.groupby(["dataset", "representation"], as_index=False)[numeric].mean()
    result["dataset_label"] = result["dataset"].map(DATASET_LABELS)
    result["representation_label"] = result["representation"].map(REP_LABELS)
    return result


def make_k_effects(k_summary: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "dataset",
        "representation",
        "k_variant",
        "n_seeds",
        "oos_f1_mean",
        "near_oos_f1_mean",
        "medium_oos_f1_mean",
        "far_oos_f1_mean",
        "id_recall_mean",
        "false_accept_mean",
        "false_reject_mean",
        "auroc_mean",
        "aupr_oos_mean",
    ]
    result = k_summary.loc[:, [column for column in keep if column in k_summary.columns]].copy()
    rows: list[dict[str, object]] = []
    metrics = [
        "oos_f1_mean",
        "near_oos_f1_mean",
        "medium_oos_f1_mean",
        "far_oos_f1_mean",
        "id_recall_mean",
        "false_accept_mean",
        "false_reject_mean",
        "auroc_mean",
        "aupr_oos_mean",
    ]
    metrics = [metric for metric in metrics if metric in result.columns]
    for (dataset, representation), group in result.groupby(["dataset", "representation"]):
        row: dict[str, object] = {"dataset": dataset, "representation": representation}
        for metric in metrics:
            values = group.set_index("k_variant")[metric]
            row[f"k1_{metric}"] = float(values.get("k1", np.nan))
            row[f"k2_{metric}"] = float(values.get("k2", np.nan))
            row[f"k2_minus_k1_{metric}"] = float(values.get("k2", np.nan) - values.get("k1", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def make_score_gap(boundary: pd.DataFrame) -> pd.DataFrame:
    frame = boundary.loc[boundary["kir"].astype(float).eq(0.50)].copy()
    numeric = ["oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc", "aupr_oos", "score_gap_median_oos_minus_known", "known_accepted_rate", "oos_accepted_rate"]
    result = frame.groupby(["dataset", "representation"], as_index=False)[numeric].mean()
    result["dataset_label"] = result["dataset"].map(DATASET_LABELS)
    result["representation_label"] = result["representation"].map({"frozen": "Frozen", "trainable": "Trainable"})
    return result


def make_transition_quadrants(transitions: pd.DataFrame) -> pd.DataFrame:
    frame = transitions.loc[
        transitions["left_method"].eq("trainable_k1") & transitions["right_method"].eq("mogb_fair")
    ].copy()
    if frame.empty:
        # The raw audit uses this canonical long label in current artifacts.
        frame = transitions.loc[
            transitions["left_method"].eq("trainable_k1") & transitions["right_method"].eq("mogb_fair_component")
        ].copy()
    if frame.empty:
        raise ValueError("raw transition input has no Trainable-vs-MOGB pair")
    result = frame.groupby(["gold_type", "transition"], as_index=False)["count"].sum()
    result["rate"] = result["count"] / result.groupby("gold_type")["count"].transform("sum")
    return result


def plot_geometry_tradeoff(geometry: pd.DataFrame, k_effects: pd.DataFrame, path: Path) -> None:
    base = aggregate_geometry(geometry).merge(
        k_effects[["dataset", "representation", "k1_oos_f1_mean", "k2_minus_k1_oos_f1_mean"]],
        on=["dataset", "representation"],
        how="left",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for _, row in base.iterrows():
        rep = row["representation"]
        color = REP_COLORS[rep]
        marker = REP_MARKERS[rep]
        label = f"{DATASET_LABELS[row['dataset']]} {REP_LABELS[rep]}"
        axes[0].scatter(row["effective_rank"], row["relative_separation"], color=color, marker=marker, s=80, edgecolor="white", linewidth=0.5, label=label)
        axes[0].annotate(DATASET_LABELS[row["dataset"]][:3], (row["effective_rank"], row["relative_separation"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        axes[1].scatter(row["relative_separation"], row["k2_minus_k1_oos_f1_mean"] * 100, color=color, marker=marker, s=80, edgecolor="white", linewidth=0.5)
        axes[1].annotate(DATASET_LABELS[row["dataset"]][:3], (row["relative_separation"], row["k2_minus_k1_oos_f1_mean"] * 100), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("Effective rank")
    axes[0].set_ylabel("Relative separation")
    axes[0].set_title("表示几何：压缩与分离")
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set_xlabel("Relative separation")
    axes[1].set_ylabel("K=2 − K=1 OOS F1 (pp)")
    axes[1].set_title("几何改善不保证多中心安全")
    for ax in axes:
        ax.grid(alpha=0.2)
    handles = [plt.Line2D([0], [0], marker=REP_MARKERS[rep], color="w", markerfacecolor=REP_COLORS[rep], label=REP_LABELS[rep], markersize=8) for rep in REP_ORDER]
    axes[1].legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle("表示几何—固定多中心交互（KIR=0.50，事后机制分析）")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_near_oos(k_effects: pd.DataFrame, path: Path) -> None:
    frame = k_effects.copy()
    frame["row"] = frame.apply(lambda r: f"{DATASET_LABELS[r['dataset']]} / {REP_LABELS[r['representation']]}", axis=1)
    rows = [f"{DATASET_LABELS[d]} / {REP_LABELS[r]}" for d in DATASET_ORDER for r in REP_ORDER]
    # Use a compact 3-column matrix: each dataset column is shown only for its rows.
    data = np.full((len(rows), 1), np.nan)
    for i, row_name in enumerate(rows):
        value = frame.loc[frame["row"].eq(row_name), "k2_minus_k1_near_oos_f1_mean"]
        data[i, 0] = float(value.iloc[0]) if not value.empty else np.nan
    fig, ax = plt.subplots(figsize=(6, 7))
    limit = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0
    image = ax.imshow(data * 100, cmap="RdBu_r", vmin=-limit * 100, vmax=limit * 100, aspect="auto")
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_xticks([0], ["K=2 − K=1\nNear-OOS F1 (pp)"])
    for i, value in enumerate(data[:, 0]):
        if np.isfinite(value):
            ax.text(0, i, f"{value * 100:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    ax.set_title("Near-OOS 对固定多中心的敏感性")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_score_gap(score_gap: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in score_gap.iterrows():
        rep = row["representation"]
        ax.scatter(row["score_gap_median_oos_minus_known"], row["false_accept_rate"] * 100, color=REP_COLORS[rep], marker="o" if rep == "frozen" else "D", s=90, edgecolor="white", linewidth=0.5)
        ax.annotate(f"{DATASET_LABELS[row['dataset']]} {REP_LABELS[rep]}", (row["score_gap_median_oos_minus_known"], row["false_accept_rate"] * 100), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Median score gap: OOS − Known")
    ax.set_ylabel("False acceptance rate (%)")
    ax.set_title("分数分离与误接收：KIR=0.50")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_error_quadrants(quadrants: pd.DataFrame, path: Path) -> None:
    order = ["both_correct", "left_only_correct", "right_only_correct", "both_wrong"]
    labels = {"both_correct": "两者都对", "left_only_correct": "Trainable独有", "right_only_correct": "MOGB独有", "both_wrong": "两者都错"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, gold_type in zip(axes, ["known", "oos"]):
        frame = quadrants.loc[quadrants["gold_type"].eq(gold_type)].set_index("transition").reindex(order).reset_index()
        values = frame["rate"].fillna(0).to_numpy() * 100
        bars = ax.bar(np.arange(len(order)), values, color=["#999999", "#0072B2", "#D55E00", "#555555"])
        ax.set_xticks(np.arange(len(order)), [labels[item] for item in order], rotation=25, ha="right")
        ax.set_title("Known" if gold_type == "known" else "OOS")
        ax.set_ylabel("样本占比 (%)" if gold_type == "known" else "")
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("StackOverflow：Trainable K=1 与 MOGB fair 的错误重叠（事后分析）")
    fig.tight_layout()
    atomic_figure(fig, path)


def build_report(geometry: pd.DataFrame, k_effects: pd.DataFrame, score_gap: pd.DataFrame, quadrants: pd.DataFrame) -> str:
    agg = aggregate_geometry(geometry)
    lines = [
        "# 表示几何与 Near-OOS 可视化证据 V1",
        "",
        "## 范围",
        "",
        "本阶段只读取已完成的 Frozen/CE/SupCon 表示几何、K=1/K=2 结果、Trainable/Frozen score 诊断和 StackOverflow raw transition。没有训练、调参、重建中心或修改历史 artifact。测试标签只用于事后统计和画图。",
        "",
        "## 主要发现",
        "",
        "1. 表示训练确实改善 K=1 的类内/类间几何和 score separation，但几何指标不能单独预测固定 K=2 是否安全。",
        "2. StackOverflow 的 K=2 near-OOS 下降与 acceptance-union 风险一致；即使聚类稳定或类内对齐更强，也可能新增大量 OOS 误接收。",
        "3. Trainable K=1 的优势更接近分数排序和覆盖—拒识平衡，而不是简单提高拒绝率。",
        "4. Trainable 与 MOGB fair 的错误集合存在互补：MOGB 更保守，Trainable 保留更多 Known 覆盖；两者不能仅按单一 OOS F1 排名。",
        "",
        "## 输入与输出",
        "",
        "- 几何汇总：`results/analysis/archive/analysis/representation_geometry_visuals_v1/geometry_summary.csv`",
        "- K=2 变化：`results/analysis/archive/analysis/representation_geometry_visuals_v1/k_effects.csv`",
        "- score gap：`results/analysis/archive/analysis/representation_geometry_visuals_v1/score_gap_false_accept.csv`",
        "- 错误四象限：`results/analysis/archive/analysis/representation_geometry_visuals_v1/stackoverflow_error_quadrants.csv`",
        "- 图目录：`figures/archive/analysis/representation_geometry_visuals_v1/`",
        "",
        "## 图表",
        "",
        "1. `geometry_tradeoff.png`：effective rank、relative separation 与 K=2−K=1 OOS F1 的关系。",
        "2. `near_oos_delta.png`：各数据集/表示的近邻 OOS 变化。",
        "3. `score_gap_false_accept.png`：score separation 与 false acceptance 的关系。",
        "4. `stackoverflow_error_quadrants.png`：Trainable K=1 与 MOGB fair 的逐样本错误重叠。",
        "",
        "## 解释边界",
        "",
        "这些图是机制证据，不是新的正式模型选择，也不是官方 MOGB 或 DCLOOS 的公平 SOTA 排名。near/medium/far 仍属于已有诊断合同；完整 Cascade 与外部端到端监督条件保持隔离。",
        "",
        "## 几何均值（KIR=0.50）",
        "",
        "| 数据集 | 表示 | Relative separation | Effective rank | Purity@10 | Same-intent alignment |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in agg.sort_values(["dataset", "representation"]).iterrows():
        lines.append(
            f"| {DATASET_LABELS[row['dataset']]} | {REP_LABELS[row['representation']]} | {row['relative_separation']:.3f} | {row['effective_rank']:.2f} | {row['purity_at_10']:.3f} | {row['same_intent_alignment']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    geometry, k_summary, boundary, transitions, input_hashes = load_inputs()
    geometry_summary = aggregate_geometry(geometry)
    k_effects = make_k_effects(k_summary)
    score_gap = make_score_gap(boundary)
    quadrants = make_transition_quadrants(transitions)
    atomic_csv(geometry_summary, OUT / "geometry_summary.csv")
    atomic_csv(k_effects, OUT / "k_effects.csv")
    atomic_csv(score_gap, OUT / "score_gap_false_accept.csv")
    atomic_csv(quadrants, OUT / "stackoverflow_error_quadrants.csv")
    plot_geometry_tradeoff(geometry, k_effects, FIG / "geometry_tradeoff.png")
    plot_near_oos(k_effects, FIG / "near_oos_delta.png")
    plot_score_gap(score_gap, FIG / "score_gap_false_accept.png")
    plot_error_quadrants(quadrants, FIG / "stackoverflow_error_quadrants.png")
    manifest = {
        "analysis": "representation_geometry_visuals_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "analysis_only": True,
        "selection_used_test_oos": False,
        "test_labels_used_post_hoc": True,
        "inputs": input_hashes,
        "outputs": {
            "geometry_summary": str(OUT / "geometry_summary.csv"),
            "k_effects": str(OUT / "k_effects.csv"),
            "score_gap_false_accept": str(OUT / "score_gap_false_accept.csv"),
            "stackoverflow_error_quadrants": str(OUT / "stackoverflow_error_quadrants.csv"),
        },
        "figures": [
            str(FIG / "geometry_tradeoff.png"),
            str(FIG / "near_oos_delta.png"),
            str(FIG / "score_gap_false_accept.png"),
            str(FIG / "stackoverflow_error_quadrants.png"),
        ],
    }
    atomic_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", OUT / "MANIFEST.json")
    atomic_text(build_report(geometry, k_effects, score_gap, quadrants), REPORT)
    print(json.dumps({"status": "ok", "geometry_rows": len(geometry), "k_effect_rows": len(k_effects), "score_gap_rows": len(score_gap), "quadrant_rows": len(quadrants), "figures": 4}, ensure_ascii=False))


if __name__ == "__main__":
    main()
