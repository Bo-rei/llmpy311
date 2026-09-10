"""Build a cross-dataset visual evidence chain from completed light results.

This is an analysis-only companion to the raw StackOverflow error audit.  It
does not train, select a model, or rewrite any historical artifact.  The
inputs are already completed five-seed fair-component summaries, the fixed-K
light sweep, and the StackOverflow intent diagnostic.  The outputs make the
three layers explicit:

* performance: where each method wins or loses;
* representation/score trade-off: OOS F1 versus Known coverage and false
  acceptance;
* decision mechanism: how adding centres trades recovered Known samples for
  newly admitted OOS samples.

External ADB/DA-ADB/MOGB compatibility rows are copied into a separate table,
never pooled with the same-protocol fair matrix.
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
OUT = ROOT / "results" / "analysis" / "visual_evidence_chain_v1"
FIG = ROOT / "figures" / "visual_evidence_chain_v1"
REPORT = ROOT / "docs" / "analysis" / "VISUAL_EVIDENCE_CHAIN_V1.md"
METHOD_ORDER = [
    "trainable_k1",
    "single_centroid",
    "random_partition",
    "fixed_k2",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
    "mogb_minilm",
]
METHOD_LABELS = {
    "trainable_k1": "Trainable K=1",
    "single_centroid": "Single centroid",
    "random_partition": "Random partition",
    "fixed_k2": "Fixed K=2",
    "mogb_partition_ours_boundary": "MOGB partition + s2c boundary",
    "ours_partition_mogb_boundary": "s2c partition + MOGB boundary",
    "mogb_minilm": "MOGB MiniLM",
}
METHOD_SHORT = {
    "trainable_k1": "T",
    "single_centroid": "C1",
    "random_partition": "Rand",
    "fixed_k2": "K2",
    "mogb_partition_ours_boundary": "M-P",
    "ours_partition_mogb_boundary": "S-P",
    "mogb_minilm": "M",
}
METHOD_COLORS = {
    "trainable_k1": "#0072B2",
    "single_centroid": "#666666",
    "random_partition": "#56B4E9",
    "fixed_k2": "#D55E00",
    "mogb_partition_ours_boundary": "#009E73",
    "ours_partition_mogb_boundary": "#CC79A7",
    "mogb_minilm": "#E69F00",
}
DATASET_ORDER = ["clinc150", "banking77", "stackoverflow"]
DATASET_LABELS = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}
KIR_ORDER = [0.25, 0.50, 0.75]


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


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, format=path.suffix.lstrip("."), dpi=180, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(path)


def load_fair_summary() -> tuple[pd.DataFrame, Path]:
    path = ROOT / "results" / "analysis" / "experimental_mechanism_pack_v3" / "method_summary.csv"
    frame = read_csv(path)
    frame = frame.loc[frame["method"].isin(METHOD_ORDER)].copy()
    frame["dataset"] = frame["dataset"].astype(str).replace({"banking77_oos": "banking77"})
    frame["kir"] = frame["kir"].astype(float)
    frame["method_label"] = frame["method"].map(METHOD_LABELS)
    expected = set(DATASET_ORDER) | set(frame["dataset"])
    if not set(DATASET_ORDER).issubset(expected):
        raise ValueError("fair summary is missing one of the three datasets")
    return frame, path


def load_fixed_k_sweep() -> tuple[pd.DataFrame, Path]:
    path = ROOT / "results" / "gate_only" / "kir_k_fixed_mean_std.csv"
    frame = read_csv(path)
    frame = frame.loc[frame["phase"].eq("fixed")].copy()
    frame["dataset"] = frame["dataset"].astype(str).replace({"banking77_oos": "banking77"})
    frame["kir"] = frame["kir"].astype(float) / 100.0
    frame["k"] = frame["k_gate"].astype(int)
    return frame, path


def load_intent_diagnostic() -> tuple[pd.DataFrame, Path]:
    path = ROOT / "results" / "analysis" / "stackoverflow_intent_diagnostic_v1" / "intent_diagnostic_summary.csv"
    return read_csv(path), path


def load_external_rows() -> tuple[pd.DataFrame, Path]:
    path = ROOT / "results" / "final_baselines" / "summary.csv"
    frame = read_csv(path)
    wanted = frame[frame["method"].astype(str).str.contains("ADB|MOGB|DCLOOS", case=False, regex=True)].copy()
    wanted["comparison_scope"] = np.where(wanted["scope"].astype(str).str.contains("protocol_v2_fair"), "same_protocol_component", "external_or_compatibility")
    return wanted, path


def performance_tables(fair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    value_columns = {
        "oos_f1": "oos_f1_mean",
        "f1_all": "f1_all_mean",
        "known_recall": "known_recall_mean",
        "false_accept_rate": "false_accept_rate_mean",
        "auroc": "auroc_mean",
    }
    rows: list[dict[str, object]] = []
    for _, item in fair.iterrows():
        row = {"dataset": item["dataset"], "kir": float(item["kir"]), "method": item["method"], "method_label": item["method_label"], "n_seeds": int(item["n_seeds"])}
        for name, source in value_columns.items():
            row[name] = float(item[source])
        rows.append(row)
    long = pd.DataFrame(rows)
    ranking = long.copy()
    ranking["oos_f1_rank"] = ranking.groupby(["dataset", "kir"])["oos_f1"].rank(method="min", ascending=False).astype(int)
    ranking["f1_all_rank"] = ranking.groupby(["dataset", "kir"])["f1_all"].rank(method="min", ascending=False).astype(int)
    base = long[long["method"].eq("trainable_k1")][["dataset", "kir", "oos_f1", "f1_all", "known_recall", "false_accept_rate"]].rename(columns={"oos_f1": "trainable_oos_f1", "f1_all": "trainable_f1_all", "known_recall": "trainable_known_recall", "false_accept_rate": "trainable_false_accept_rate"})
    gaps = long[~long["method"].eq("trainable_k1")].merge(base, on=["dataset", "kir"], how="left")
    for metric in ("oos_f1", "f1_all", "known_recall", "false_accept_rate"):
        gaps[f"trainable_minus_{metric}"] = gaps[f"trainable_{metric}"] - gaps[metric]
    return long, ranking, gaps


def plot_heatmaps(long: pd.DataFrame, path: Path) -> None:
    data = long.copy()
    data["column"] = data.apply(lambda row: f"{DATASET_LABELS[row['dataset']]}\nKIR={row['kir']:.2f}", axis=1)
    columns = [f"{DATASET_LABELS[d]}\nKIR={kir:.2f}" for d in DATASET_ORDER for kir in KIR_ORDER]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for ax, metric, title in zip(axes, ("oos_f1", "f1_all"), ("OOS F1", "F1-All")):
        matrix = data.pivot(index="method_label", columns="column", values=metric).reindex(index=[METHOD_LABELS[m] for m in METHOD_ORDER], columns=columns)
        image = ax.imshow(matrix.to_numpy(dtype=float) * 100.0, aspect="auto", cmap="viridis", vmin=0, vmax=100)
        ax.set_yticks(np.arange(len(matrix)), matrix.index)
        ax.set_title(title + "（五 seed 均值，百分数）")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value * 100:.1f}", ha="center", va="center", fontsize=7, color="white" if value < 0.62 else "black")
        fig.colorbar(image, ax=ax, fraction=0.015, pad=0.01)
    axes[-1].set_xticks(np.arange(len(columns)), columns, rotation=30, ha="right")
    fig.suptitle("同一 protocol_v2_textoir_v1：方法×数据集×KIR 性能总览")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_pareto(long: pd.DataFrame, path: Path) -> None:
    averaged = long.groupby(["dataset", "method", "method_label"], as_index=False)[["oos_f1", "f1_all", "known_recall", "false_accept_rate"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, dataset in zip(axes, DATASET_ORDER):
        frame = averaged[averaged["dataset"].eq(dataset)]
        for _, row in frame.iterrows():
            method = row["method"]
            ax.scatter(row["f1_all"] * 100, row["oos_f1"] * 100, s=85, color=METHOD_COLORS[method], edgecolor="white", linewidth=0.5)
            ax.annotate(METHOD_SHORT[method], (row["f1_all"] * 100, row["oos_f1"] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8, fontweight="bold")
        ax.set_title(DATASET_LABELS[dataset] + "（跨 KIR 平均）")
        ax.set_xlabel("F1-All (%)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("OOS F1 (%)")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS[m], label=f"{METHOD_SHORT[m]}={METHOD_LABELS[m]}", markersize=7) for m in METHOD_ORDER]
    axes[-1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle("OOS F1—Known 分类能力 Pareto 视图：不把拒绝 Known 当作纯收益")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_kir_curves(long: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, dataset in zip(axes, DATASET_ORDER):
        frame = long[long["dataset"].eq(dataset)]
        for method in METHOD_ORDER:
            rows = frame[frame["method"].eq(method)].sort_values("kir")
            ax.plot(rows["kir"], rows["oos_f1"] * 100, marker="o", linewidth=2 if method == "trainable_k1" else 1.2, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(KIR_ORDER, ["0.25", "0.50", "0.75"])
        ax.set_xlabel("KIR")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("KIR 曲线：Trainable K=1 的收益是否跨开放程度稳定")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_rank_heatmap(ranking: pd.DataFrame, path: Path) -> None:
    data = ranking.copy()
    data["column"] = data.apply(lambda row: f"{DATASET_LABELS[row['dataset']]}\n{row['kir']:.2f}", axis=1)
    columns = [f"{DATASET_LABELS[d]}\n{kir:.2f}" for d in DATASET_ORDER for kir in KIR_ORDER]
    matrix = data.pivot(index="method_label", columns="column", values="oos_f1_rank").reindex(index=[METHOD_LABELS[m] for m in METHOD_ORDER], columns=columns)
    fig, ax = plt.subplots(figsize=(15, 5))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=len(METHOD_ORDER))
    ax.set_yticks(np.arange(len(matrix)), matrix.index)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=30, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, str(int(value)), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="OOS F1 rank（1=best）")
    ax.set_title("方法排名稳定性（仅同协议 fair matrix）")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_relative_heatmap(gaps: pd.DataFrame, path: Path) -> None:
    data = gaps.copy()
    data["column"] = data.apply(lambda row: f"{DATASET_LABELS[row['dataset']]}\n{row['kir']:.2f}", axis=1)
    columns = [f"{DATASET_LABELS[d]}\n{kir:.2f}" for d in DATASET_ORDER for kir in KIR_ORDER]
    method_labels = [METHOD_LABELS[m] for m in METHOD_ORDER if m != "trainable_k1"]
    matrix = data.assign(comparator=data["method_label"])[["comparator", "dataset", "kir", "trainable_minus_oos_f1"]]
    matrix["column"] = matrix.apply(lambda row: f"{DATASET_LABELS[row['dataset']]}\n{row['kir']:.2f}", axis=1)
    matrix = matrix.pivot(index="comparator", columns="column", values="trainable_minus_oos_f1").reindex(index=method_labels, columns=columns)
    fig, ax = plt.subplots(figsize=(15, 5))
    limit = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))
    image = ax.imshow(matrix.to_numpy(dtype=float) * 100, aspect="auto", cmap="RdBu_r", vmin=-limit * 100, vmax=limit * 100)
    ax.set_yticks(np.arange(len(matrix)), matrix.index)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=30, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value * 100:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Trainable − comparator OOS F1 (pp)")
    ax.set_title("Trainable K=1 相对同协议组件的 OOS F1 差值")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_k_sweep(k_sweep: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True, sharey=True)
    for row_idx, dataset in enumerate(DATASET_ORDER):
        for col_idx, distance in enumerate(["euclidean", "mahalanobis_diag"]):
            ax = axes[row_idx, col_idx]
            frame = k_sweep[(k_sweep["dataset"].eq(dataset)) & k_sweep["distance"].eq(distance)]
            for kir, color in zip(KIR_ORDER, ["#0072B2", "#D55E00", "#009E73"]):
                rows = frame[frame["kir"].eq(kir)].sort_values("k")
                if rows.empty:
                    continue
                ax.plot(rows["k"], rows["test_oos_f1_mean"] * 100, marker="o", color=color, linewidth=2, label=f"OOS KIR={kir:.2f}")
                ax.plot(rows["k"], rows["test_id_recall_mean"] * 100, marker="x", linestyle="--", color=color, alpha=0.65, label=f"Known KIR={kir:.2f}")
            ax.set_title(f"{DATASET_LABELS[dataset]} · {distance}")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(alpha=0.2)
            if row_idx == 2:
                ax.set_xlabel("固定中心数 K")
            if col_idx == 0:
                ax.set_ylabel("指标 (%)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("固定 K 消融：OOS F1 与 Known Recall 的覆盖—拒识折中（3 seed light sweep）")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_intent_risk(intent: pd.DataFrame, path: Path) -> None:
    frame = intent.copy().sort_values("mean_newly_accepted_oos", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    size = np.clip(frame["mean_cluster_imbalance"].to_numpy(dtype=float), 1, 12) * 12
    scatter = ax.scatter(frame["mean_recovered_known"], frame["mean_newly_accepted_oos"], c=frame["mean_ari"], s=size, cmap="viridis", alpha=0.85, edgecolor="white", linewidth=0.4)
    limit = max(float(frame["mean_recovered_known"].max()), float(frame["mean_newly_accepted_oos"].max())) * 1.05
    ax.plot([0, limit], [0, limit], "k--", linewidth=1, label="新增误接收=恢复 Known")
    top = frame.head(8)
    for _, row in top.iterrows():
        ax.annotate(str(row["intent"]), (row["mean_recovered_known"], row["mean_newly_accepted_oos"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("K=2 相对 K=1 恢复的 Known 样本数")
    ax.set_ylabel("K=2 新增误接收的 OOS 样本数")
    ax.set_title("StackOverflow intent-level 机制：稳定聚类不等于开放边界有效")
    fig.colorbar(scatter, ax=ax, label="bootstrap ARI")
    ax.legend(fontsize=8)
    fig.tight_layout()
    atomic_figure(fig, path)


def build_report(long: pd.DataFrame, ranking: pd.DataFrame, gaps: pd.DataFrame, k_sweep: pd.DataFrame, intent: pd.DataFrame, external: pd.DataFrame, sources: dict[str, str]) -> None:
    trainable = long[long["method"].eq("trainable_k1")]
    mean_trainable = trainable.groupby("dataset")[["oos_f1", "f1_all", "known_recall", "false_accept_rate"]].mean()
    lines = [
        "# 跨数据集可视化证据链 V1",
        "",
        "## 证据范围",
        "",
        "本报告是 analysis-only：读取已完成的 `experimental_mechanism_pack_v3` 五 seed fair-component 汇总、固定 K=1…5 的 3-seed light sweep 和 StackOverflow intent 诊断；不训练、不重新选择超参数、不覆盖历史 artifact。相同协议 fair matrix 与外部/兼容 baseline 分开。",
        "",
        "## 当前同协议结论",
        "",
        "- Trainable K=1 是当前三个数据集、三个 KIR 下最稳定的自有 Gate 候选；它的主要优势是 OOS 分数排序和覆盖—拒识平衡，而不是单纯把 Known 拒成 OOS。",
        "- MOGB MiniLM / MOGB partition 组件通常降低 false acceptance，但同时牺牲大量 Known Recall/F1-All；因此不能仅按 OOS F1 或 FA 单指标排名。",
        "- 固定 K=2 在 Banking77 某些 KIR 可能提高 OOS F1，但 Known Recall 与 F1-All 的代价必须同时报告；在 StackOverflow 上固定多中心的风险最明显。",
        "- StackOverflow intent 诊断显示，K=2 的聚类 ARI 可以很高，但新增 OOS 误接收仍远多于恢复 Known，说明“聚类稳定”不是开放边界有效性的充分条件。",
        "",
        "## Trainable 的跨数据集均值（跨 KIR，五 seed）",
        "",
        "| 数据集 | OOS F1 | F1-All | Known Recall | False Acceptance |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASET_ORDER:
        row = mean_trainable.loc[dataset]
        lines.append(f"| {DATASET_LABELS[dataset]} | {row.oos_f1 * 100:.2f} | {row.f1_all * 100:.2f} | {row.known_recall * 100:.2f} | {row.false_accept_rate * 100:.2f} |")
    lines += [
        "",
        "## 图表解释",
        "",
        "1. `performance_heatmap.png` 显示方法优势集中在哪个数据集/KIR，而不是把所有结果压成一个平均数。",
        "2. `pareto_tradeoff.png` 用 F1-All/OOS F1 展示 Known 分类与 OOS 拒识的联合工作点；`relative_trainable_heatmap.png` 直接显示 Trainable 相对同协议组件的差值。",
        "3. `kir_curves.png` 用 KIR 曲线检查收益是否随开放程度变化；曲线不能被解释成超出当前三 KIR 的外推。",
        "4. `k_sweep_tradeoff.png` 显示增加固定中心数后 OOS F1 与 Known Recall 的不同方向，使用 3-seed light sweep，只作机制消融，不构成 adaptive-K 选择规则。",
        "5. `intent_risk_scatter.png` 把 StackOverflow 的 intent-level 恢复 Known 与新增误接收 OOS 放在同一坐标；点在对角线之上表示边界扩张的风险大于覆盖收益。",
        "",
        "## 与外部 baseline 的边界",
        "",
        "ADB、DA-ADB、官方 BERT MOGB 和 DCLOOS（如有记录）保存在 `external_baselines_isolated.csv`，不与同协议 MiniLM fair matrix 合并。它们的表示、训练监督、数据/划分或评价合同不同，只能作为复现/兼容性参照，不能从本报告推导无条件 SOTA 排名。",
        "",
        "## 当前最重要的机制判断",
        "",
        "Trainable 的正结果来自表示适配后更好的 K=1 score separation；固定多中心的负结果来自 acceptance union 扩张。下一步若继续做实验，应优先补同一监督条件下的强 baseline 工作点与表示几何图，而不是盲目增加 K 或重新搜索 test-oracle 最优点。",
        "",
        "## 可复现与限制",
        "",
        "- 输入 SHA256 记录在 `MANIFEST.json`；没有输出原始文本、embedding、checkpoint 或逐样本预测。",
        "- 所有 test OOS 使用仅限事后可视化/误差分析；没有用于选择 K、阈值、半径或 checkpoint。",
        "- 本报告仍是 Gate-only/组件级分析，不是完整 Cascade 论文主表。",
    ]
    atomic_text("\n".join(lines) + "\n", REPORT)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fair, fair_path = load_fair_summary()
    k_sweep, k_path = load_fixed_k_sweep()
    intent, intent_path = load_intent_diagnostic()
    external, external_path = load_external_rows()
    long, ranking, gaps = performance_tables(fair)
    atomic_csv(long, OUT / "performance_long.csv")
    atomic_csv(ranking, OUT / "method_rankings.csv")
    atomic_csv(gaps, OUT / "relative_trainable_effects.csv")
    atomic_csv(k_sweep, OUT / "fixed_k_sweep.csv")
    atomic_csv(intent, OUT / "stackoverflow_intent_mechanism.csv")
    atomic_csv(external, OUT / "external_baselines_isolated.csv")
    plot_heatmaps(long, FIG / "performance_heatmap.png")
    plot_pareto(long, FIG / "pareto_tradeoff.png")
    plot_kir_curves(long, FIG / "kir_curves.png")
    plot_rank_heatmap(ranking, FIG / "method_rank_heatmap.png")
    plot_relative_heatmap(gaps, FIG / "relative_trainable_heatmap.png")
    plot_k_sweep(k_sweep, FIG / "k_sweep_tradeoff.png")
    plot_intent_risk(intent, FIG / "intent_risk_scatter.png")
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in (fair_path, k_path, intent_path, external_path)}
    manifest = {
        "analysis_stage": "visual_evidence_chain_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "same_protocol_scope": "five_seed_light_fair_component_summary",
        "fixed_k_scope": "three_seed_light_sweep",
        "test_oos_used_for_selection": False,
        "analysis_uses_test_labels_post_hoc": True,
        "raw_text_exported": False,
        "sources": sources,
        "outputs": sorted(str(path.relative_to(OUT)) for path in OUT.iterdir() if path.is_file()),
        "figures": sorted(str(path.relative_to(FIG)) for path in FIG.iterdir() if path.is_file()),
    }
    atomic_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", OUT / "MANIFEST.json")
    build_report(long, ranking, gaps, k_sweep, intent, external, sources)
    print(json.dumps({"status": "ok", "fair_rows": int(len(long)), "k_rows": int(len(k_sweep)), "intent_rows": int(len(intent)), "figures": int(len(list(FIG.glob("*.png")))), "report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
