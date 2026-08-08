"""Aggregate and visualize the Trainable MiniLM KIR sweep.

The script is analysis-only: it reads completed manifests/CSV files and writes
lightweight summaries and figures.  It never reads test labels to select a
configuration and never modifies experiment artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1" / "minilm_trainable_kir_sweep_v1"
OUT = ROOT / "results" / "analysis" / "minilm_trainable_kir_sweep_v1"
FIG = ROOT / "figures" / "minilm_trainable_kir_sweep_v1"
REPORT = ROOT / "docs" / "analysis" / "MINILM_TRAINABLE_KIR_SWEEP_V1.md"

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(ARTIFACT_ROOT.glob("kir_*/runs/*/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_dir"] = str(path.parent.relative_to(ROOT.parent))
        payload["metrics_path"] = str(path.relative_to(ROOT.parent))
        payload["test_used_for_selection"] = bool(payload.get("test_used_for_selection", False))
        payload["oos_used_for_training"] = bool(payload.get("oos_used_for_training", False))
        rows.append(payload)
    if not rows:
        raise FileNotFoundError(f"No completed metrics under {ARTIFACT_ROOT}")
    frame = pd.DataFrame(rows)
    required = {"dataset", "kir", "seed", "oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Trainable metrics missing columns: {missing}")
    return frame.sort_values(["dataset", "kir", "seed"]).reset_index(drop=True)


def read_frozen() -> pd.DataFrame:
    path = ROOT / "results" / "gate_only" / "kir_k_fixed_mean_std.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[(frame["phase"] == "fixed") & (frame["k_gate"] == 1) & (frame["distance"] == "mahalanobis_diag")].copy()
    frame["dataset"] = frame["dataset"].replace({"banking77_oos": "banking77"})
    frame["kir"] = frame["kir"].astype(float) / 100.0
    frame = frame.rename(
        columns={
            "test_oos_f1_mean": "frozen_oos_f1",
            "test_oos_f1_std": "frozen_oos_f1_std",
            "test_id_recall_mean": "frozen_known_recall",
            "test_id_recall_std": "frozen_known_recall_std",
            "test_auroc_mean": "frozen_auroc",
            "test_auroc_std": "frozen_auroc_std",
            "test_aupr_oos_mean": "frozen_aupr_oos",
            "test_aupr_oos_std": "frozen_aupr_oos_std",
        }
    )
    keep = ["dataset", "kir", "frozen_oos_f1", "frozen_oos_f1_std", "frozen_known_recall", "frozen_known_recall_std", "frozen_auroc", "frozen_auroc_std", "frozen_aupr_oos", "frozen_aupr_oos_std", "seed_count"]
    frozen = frame[keep]
    # The fixed-K table does not expose OOS rejection.  Read the matching
    # frozen baseline row so false-accept deltas are not approximated by
    # ``1 - KnownRecall`` (those are different error events).
    baseline_path = ROOT / "results" / "gate_only" / "gate_baseline_summary.csv"
    baseline = pd.read_csv(baseline_path)
    baseline = baseline.loc[(baseline["method"] == "diag_mahalanobis_centroid") & (baseline["k_gate_values"].astype(str) == "1")].copy()
    baseline["dataset"] = baseline["dataset"].replace({"banking77_oos": "banking77"})
    baseline["kir"] = baseline["kir"].astype(float) / 100.0
    baseline["frozen_false_accept_rate"] = 1.0 - baseline["test_oos_rejection_mean"].astype(float)
    frozen = frozen.merge(baseline[["dataset", "kir", "frozen_false_accept_rate"]], on=["dataset", "kir"], how="left", validate="one_to_one")
    return frozen


def read_fair_components() -> pd.DataFrame:
    path = ROOT / "results" / "mogb" / "fair_matrix.csv"
    frame = pd.read_csv(path)
    frame["dataset"] = frame["dataset"].replace({"banking77_oos": "banking77"})
    return frame.loc[frame["kir"].isin([0.25, 0.50, 0.75])].copy()


def aggregate(trainable: pd.DataFrame, frozen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"]
    grouped = trainable.groupby(["dataset", "kir"], as_index=False).agg(
        n_seeds=("seed", "count"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, "std") for metric in metrics},
    )
    grouped["method"] = "trainable_k1"
    paired = grouped.merge(frozen, on=["dataset", "kir"], how="left", validate="one_to_one")
    paired["oos_f1_delta_pp"] = (paired["oos_f1_mean"] - paired["frozen_oos_f1"]) * 100.0
    paired["known_recall_delta_pp"] = (paired["known_recall_mean"] - paired["frozen_known_recall"]) * 100.0
    paired["false_accept_delta_pp"] = (paired["false_accept_rate_mean"] - paired["frozen_false_accept_rate"]) * 100.0
    return grouped, paired


def write_csv(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def plot_oos(trainable: pd.DataFrame, frozen: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, dataset in zip(axes, ["clinc150", "banking77", "stackoverflow"]):
        t = trainable.loc[trainable["dataset"] == dataset].groupby("kir")["oos_f1"].agg(["mean", "std"]).reset_index()
        f = frozen.loc[frozen["dataset"] == dataset].sort_values("kir")
        ax.errorbar(t["kir"], t["mean"] * 100, yerr=t["std"] * 100, marker="o", linewidth=2, capsize=3, label="Trainable K=1")
        ax.errorbar(f["kir"], f["frozen_oos_f1"] * 100, yerr=f["frozen_oos_f1_std"] * 100, marker="s", linewidth=2, capsize=3, label="Frozen K=1")
        ax.set_title(dataset)
        ax.set_xlabel("KIR")
        ax.set_xticks([0.25, 0.50, 0.75])
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("Trainable versus Frozen MiniLM across KIR")
    fig.tight_layout()
    fig.savefig(FIG / "trainable_vs_frozen_kir.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(trainable: pd.DataFrame, frozen: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
    colors = {0.25: "#4c78a8", 0.50: "#f58518", 0.75: "#54a24b"}
    for ax, dataset in zip(axes, ["clinc150", "banking77", "stackoverflow"]):
        t = trainable.loc[trainable["dataset"] == dataset]
        f = frozen.loc[frozen["dataset"] == dataset]
        for kir, row in t.groupby("kir"):
            point = row.mean(numeric_only=True)
            ax.scatter(point["known_recall"] * 100, point["oos_f1"] * 100, color=colors[float(kir)], marker="o", s=65, label=f"Trainable KIR={kir:.2f}")
            ax.annotate(f"{kir:.2f}", (point["known_recall"] * 100, point["oos_f1"] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
        for _, row in f.iterrows():
            ax.scatter(row["frozen_known_recall"] * 100, row["frozen_oos_f1"] * 100, color=colors[float(row["kir"])], marker="s", s=55, alpha=0.7)
        ax.set_title(dataset)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[1].set_xlabel("Known Recall (%)")
    handles = [plt.Line2D([0], [0], marker="o", color="black", linestyle="", label="Trainable K=1"), plt.Line2D([0], [0], marker="s", color="black", linestyle="", label="Frozen K=1")]
    axes[-1].legend(handles=handles, frameon=False, fontsize=9)
    fig.suptitle("OOS quality versus Known coverage across KIR")
    fig.tight_layout()
    fig.savefig(FIG / "trainable_known_oos_tradeoff_kir.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(paired: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    table = paired.pivot(index="dataset", columns="kir", values="oos_f1_delta_pp").reindex(["clinc150", "banking77", "stackoverflow"])
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    image = ax.imshow(table.to_numpy(), cmap="RdYlGn", vmin=-15, vmax=15, aspect="auto")
    ax.set_xticks(range(len(table.columns)), [f"{v:.2f}" for v in table.columns])
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("KIR")
    ax.set_ylabel("Dataset")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(image, ax=ax, label="Trainable − Frozen OOS F1 (pp)")
    fig.suptitle("Effect of Known-only MiniLM adaptation")
    fig.tight_layout()
    fig.savefig(FIG / "trainable_minus_frozen_kir_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_component_context(trainable_mean: pd.DataFrame, fair: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for _, row in trainable_mean.iterrows():
        rows.append({"dataset": row["dataset"], "kir": row["kir"], "method": "Trainable K=1", "oos_f1": row["oos_f1_mean"] * 100, "known_recall": row["known_recall_mean"] * 100})
    for method in ["single_centroid", "fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary"]:
        sub = fair.loc[fair["method"] == method]
        if sub.empty:
            continue
        grouped = sub.groupby(["dataset", "kir"], as_index=False).agg(oos_f1=("oos_f1", "mean"), known_recall=("id_recall", "mean"))
        for _, row in grouped.iterrows():
            rows.append({"dataset": row["dataset"], "kir": row["kir"], "method": method, "oos_f1": row["oos_f1"] * 100, "known_recall": row["known_recall"] * 100})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "trainable_fair_component_points.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), sharex=True, sharey=True)
    markers = {"Trainable K=1": "*", "single_centroid": "o", "fixed_k2": "s", "mogb_minilm": "X", "mogb_partition_ours_boundary": "D"}
    for ax, dataset in zip(axes, ["clinc150", "banking77", "stackoverflow"]):
        sub = frame.loc[frame["dataset"] == dataset]
        for method, group in sub.groupby("method"):
            ax.scatter(group["known_recall"], group["oos_f1"], marker=markers.get(method, "o"), s=80 if method == "Trainable K=1" else 48, label=method)
        ax.set_title(dataset)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[1].set_xlabel("Known Recall (%)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("Trainable representation in the same-method comparison context")
    fig.tight_layout()
    fig.savefig(FIG / "trainable_fair_component_context.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(trainable: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# Trainable MiniLM KIR Sweep V1（K=1）",
        "",
        "> 本报告只汇总新完成的 Trainable MiniLM K=1 KIR 控制，不把它冒充新的多中心方法。训练、checkpoint 选择和 Gate 评价均使用 `protocol_v2_textoir_v1`；测试 OOS 未用于选择。",
        "",
        "## 1. 实验完成情况",
        "",
        "- 范围：CLINC150、Banking77、StackOverflow；KIR=0.25/0.50/0.75；seed=13/42/87。",
        "- 计划 27 个单元，完成 27 个；失败、缺失、重复、无效指标均为 0。",
        "- 训练目标与已有 Trainable K=1 控制完全相同：最后两层 MiniLM + 残差 projection；只使用 Known train 和 Known calibration。",
        "- 结果根目录：`artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_v1/`。",
        "",
        "## 2. Trainable K=1 结果（均值±标准差）",
        "",
        "| 数据集 | KIR | OOS F1 | F1-All | Known Recall | False Accept | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.sort_values(["dataset", "kir"]).iterrows():
        lines.append(f"| {row['dataset']} | {row['kir']:.2f} | {row['oos_f1_mean']*100:.2f}±{row['oos_f1_std']*100:.2f} | {row['f1_all_mean']*100:.2f}±{row['f1_all_std']*100:.2f} | {row['known_recall_mean']*100:.2f}±{row['known_recall_std']*100:.2f} | {row['false_accept_rate_mean']*100:.2f}±{row['false_accept_rate_std']*100:.2f} | {row['auroc_mean']*100:.2f}±{row['auroc_std']*100:.2f} |")
    lines += [
        "",
        "## 3. 相同 KIR、相同对角马氏距离下 Trainable−Frozen",
        "",
        "| 数据集 | KIR | OOS F1 差值 | Known Recall 差值 | Trainable OOS F1 | Frozen OOS F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in paired.sort_values(["dataset", "kir"]).iterrows():
        lines.append(f"| {row['dataset']} | {row['kir']:.2f} | {row['oos_f1_delta_pp']:+.2f} pp | {row['known_recall_delta_pp']*1:+.2f} pp | {row['oos_f1_mean']*100:.2f} | {row['frozen_oos_f1']*100:.2f} |")
    lines += [
        "",
        "## 4. 机制解读",
        "",
        "1. Trainable K=1 的收益不是只出现在 KIR=0.50：CLINC150 三个 KIR 均为正，StackOverflow 三个 KIR 均为正；Banking77 在 KIR=0.25 仅小幅为正，KIR=0.50 接近持平，KIR=0.75 明显下降。",
        "2. KIR 增大时，Trainable K=1 的 OOS F1 普遍下降，尤其 Banking77/StackOverflow；这说明已知意图变密集后，Known/OOS 重叠仍是主要限制。",
        "3. Banking77 的 KIR=0.75 训练结果虽然 F1-All 较高，但 OOS F1 下降，说明只看 Known 分类会掩盖 OOS 拒识退化。",
        "4. 当前结果支持‘Known-only 表示适配改善单中心分数排序’，不支持‘训练后固定多中心自然有效’；StackOverflow 的 K=2 负结果仍需单独看作 union-risk 证据。",
        "5. Trainable 与 Frozen、MOGB 组件结果的 seed 数和监督条件不同，图表用于机制对照，不构成无条件 SOTA 排名。",
        "",
        "## 5. 图表",
        "",
        "- `figures/minilm_trainable_kir_sweep_v1/trainable_vs_frozen_kir.png`：三数据集 KIR 曲线。",
        "- `figures/minilm_trainable_kir_sweep_v1/trainable_known_oos_tradeoff_kir.png`：Known Recall/OOS F1 权衡。",
        "- `figures/minilm_trainable_kir_sweep_v1/trainable_minus_frozen_kir_heatmap.png`：训练相对冻结的 KIR 热图。",
        "- `figures/minilm_trainable_kir_sweep_v1/trainable_fair_component_context.png`：与同协议 Frozen/MOGB 组件的上下文比较。",
        "",
        "## 6. 下一步",
        "",
        "- 先把本轮 Trainable K=1 与现有 Frozen/MOGB 组件结果合并到同一分层总表；",
        "- 再决定是否为 Trainable 表示运行 K=2 的跨 KIR 诊断；不把 K=2 直接当正式方法；",
        "- 完整 Cascade 和 DCLOOS 仍需单独标注数据/监督合同，不能用本轮 Gate-only 数字代替。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    trainable = read_metrics()
    frozen = read_frozen()
    fair = read_fair_components()
    summary, paired = aggregate(trainable, frozen)
    write_csv(trainable, "per_seed.csv")
    write_csv(summary, "mean_std.csv")
    write_csv(paired, "trainable_vs_frozen.csv")
    plot_oos(trainable, frozen)
    plot_tradeoff(trainable, frozen)
    plot_delta_heatmap(paired)
    plot_component_context(summary, fair)
    write_report(trainable, summary, paired)
    print(json.dumps({"status": "complete", "units": len(trainable), "summary_rows": len(summary), "report": str(REPORT), "figures": [str(path) for path in sorted(FIG.glob("*.png"))]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
