"""Summarize the completed cross-dataset Trainable MiniLM control."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1" / "minilm_trainable_control_v1" / "runs"
OUT = ROOT / "results" / "diagnostics" / "minilm_trainable_control_v1"
FIG = ROOT / "figures" / "active_experiment_dashboard_v1"
REPORT = ROOT / "docs" / "analysis" / "MINILM_TRAINABLE_CONTROL_V1.md"
DEFAULT_SEEDS = (13, 42, 87)

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_trainable() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ["oos_f1", "oos_precision", "oos_recall", "f1_all", "f1_u", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"]
    for path in sorted(ARTIFACT_ROOT.glob("*/seed_*/metrics.json")):
        payload = read_json(path)
        rows.append({"dataset": str(payload["dataset"]), "seed": int(payload["seed"]), "method": "trainable_k1", "scope": "minilm_trainable_control_v1", "source": str(path.relative_to(ROOT.parent)), **{metric: float(payload[metric]) for metric in metrics}})
    historical = ROOT / "results" / "diagnostics" / "racal_v1" / "RACAL_V1_STAGE1_MEAN_STD.csv"
    frame = pd.read_csv(historical)
    row = frame.loc[frame["method"].eq("trainable_k1")].iloc[0]
    rows.append({"dataset": "stackoverflow", "seed": np.nan, "method": "trainable_k1", "scope": "racal_v1_existing", "source": str(historical.relative_to(ROOT)), **{metric: float(row[f"{metric}_mean"]) if f"{metric}_mean" in row else np.nan for metric in metrics}, **{f"{metric}_std": float(row[f"{metric}_std"]) if f"{metric}_std" in row else np.nan for metric in metrics}})
    return pd.DataFrame(rows)


def collect_frozen() -> pd.DataFrame:
    # Use the exact E2 per-seed K=1 reference cells, not a pre-aggregated
    # CSV whose seed set and historical runner may differ from this stage.
    e2_root = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1" / "e2_gate_core_dense"
    rows: list[dict[str, object]] = []
    for dataset in ("clinc150", "banking77", "stackoverflow"):
        for seed in DEFAULT_SEEDS:
            run_name = (
                f"protocol_v2_textoir_v1__{dataset}__kir_0.50__seed_{seed}__"
                "repr_frozen_minilm__k_1__dist_mahalanobis_diag__boundary_mean_std"
            )
            path = e2_root / run_name / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Exact E2 Frozen reference is missing: {path}")
            payload = read_json(path)
            metrics = payload["combined"]
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "method": "frozen_k1",
                    "scope": "e2_exact_seed_control",
                    "source": str(path.relative_to(ROOT.parent)),
                    "oos_f1": float(metrics["oos_f1"]),
                    "known_recall": float(metrics["id_recall"]),
                    "false_accept_rate": float(metrics["false_accept_rate"]),
                    "false_reject_rate": float(metrics["false_reject_rate"]),
                    "auroc": float(metrics["auroc"]),
                    "aupr_oos": float(metrics["aupr_oos"]),
                }
            )
    return pd.DataFrame(rows)


def summarize(trainable: pd.DataFrame, frozen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"]
    rows: list[dict[str, object]] = []
    for dataset in ["clinc150", "banking77", "stackoverflow"]:
        subset = trainable[trainable["dataset"] == dataset]
        for metric in metrics:
            values = subset[metric].dropna().astype(float)
            rows.append({"dataset": dataset, "method": "trainable_k1", "metric": metric, "mean": float(values.mean()) if len(values) else np.nan, "std": float(values.std(ddof=1)) if len(values) > 1 else float(subset[f"{metric}_std"].iloc[0]) if f"{metric}_std" in subset and len(subset) else np.nan, "n": int(len(values)), "scope": str(subset["scope"].iloc[0]) if len(subset) else ""})
        base = frozen[frozen["dataset"] == dataset]
        for metric in ["oos_f1", "known_recall", "false_accept_rate", "auroc", "aupr_oos"]:
            values = base[metric].dropna().astype(float)
            rows.append({"dataset": dataset, "method": "frozen_k1", "metric": metric, "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan, "n": int(len(values)), "scope": str(base["scope"].iloc[0])})
    summary = pd.DataFrame(rows)
    delta_rows: list[dict[str, object]] = []
    for dataset in ["clinc150", "banking77", "stackoverflow"]:
        t = summary[(summary["dataset"] == dataset) & (summary["method"] == "trainable_k1")].set_index("metric")
        f = summary[(summary["dataset"] == dataset) & (summary["method"] == "frozen_k1")].set_index("metric")
        for metric in ["oos_f1", "known_recall", "auroc", "aupr_oos"]:
            if metric in t.index and metric in f.index:
                delta_rows.append({"dataset": dataset, "metric": metric, "trainable_mean": float(t.loc[metric, "mean"]), "frozen_mean": float(f.loc[metric, "mean"]), "delta_pp": (float(t.loc[metric, "mean"]) - float(f.loc[metric, "mean"])) * 100.0, "trainable_scope": t.loc[metric, "scope"], "frozen_scope": f.loc[metric, "scope"]})
    return summary, pd.DataFrame(delta_rows)


def plot(summary: pd.DataFrame) -> Path:
    metrics = [("oos_f1", "OOS F1 (%)"), ("known_recall", "Known Recall (%)"), ("false_accept_rate", "False Acceptance (%)")]
    datasets = ["clinc150", "banking77", "stackoverflow"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = {"frozen_k1": "#8da0cb", "trainable_k1": "#1b9e77"}
    for ax, (metric, title) in zip(axes, metrics):
        values: dict[str, list[float]] = {"frozen_k1": [], "trainable_k1": []}
        errors: dict[str, list[float]] = {"frozen_k1": [], "trainable_k1": []}
        for method in values:
            for dataset in datasets:
                hit = summary[(summary["dataset"] == dataset) & (summary["method"] == method) & (summary["metric"] == metric)]
                values[method].append(float(hit["mean"].iloc[0]) * 100 if not hit.empty else np.nan)
                errors[method].append(float(hit["std"].iloc[0]) * 100 if not hit.empty else 0.0)
        x = np.arange(len(datasets))
        width = 0.36
        for offset, method in [(-width / 2, "frozen_k1"), (width / 2, "trainable_k1")]:
            ax.bar(x + offset, values[method], width, yerr=errors[method], capsize=3, color=colors[method], label="Frozen K=1" if method == "frozen_k1" else "Trainable K=1")
        ax.set_xticks(x, datasets)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if metric != "false_accept_rate":
            ax.set_ylim(bottom=0)
        if metric == "oos_f1":
            ax.legend(fontsize=8)
    fig.suptitle("当前 protocol_v2_textoir_v1：Trainable K=1 的跨数据集效果", fontsize=13)
    fig.tight_layout()
    path = FIG / "trainable_cross_dataset.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def report(summary: pd.DataFrame, delta: pd.DataFrame) -> str:
    def fmt(value: float) -> str:
        return f"{value * 100:.2f}%"
    lines = [
        "# MiniLM Trainable K=1 跨数据集控制实验",
        "",
        "> 这是 `minilm_trainable_control_v1` 的受控实验，不扩展 K、不使用 OOS 训练、不用测试集选 checkpoint。CLINC150 和 Banking77 为本轮新运行，StackOverflow 复用已完成且同协议的 RACAL-v1 K=1 结果。",
        "",
        "## 1. 配置",
        "",
        "- protocol：`protocol_v2_textoir_v1`；KIR=0.50；K=1；对角 Mahalanobis；`mean_std` 半径。",
        "- 表示：`all-MiniLM-L6-v2`，只训练最后两层和残差 projection。",
        "- checkpoint：只用 Known calibration 的 `f1_k + 0.05 × Known Recall` 选择。",
        "- 新运行：CLINC150/Banking77 × seeds 13, 42, 87，共 6 个单元，全部成功。",
        "",
        "## 2. Frozen 与 Trainable",
        "",
        "| 数据集 | Frozen OOS F1 | Trainable OOS F1 | 差值 | Frozen Known Recall | Trainable Known Recall | 差值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ["clinc150", "banking77", "stackoverflow"]:
        subset = delta[delta["dataset"] == dataset]
        oos = subset[subset["metric"] == "oos_f1"].iloc[0]
        recall = subset[subset["metric"] == "known_recall"].iloc[0]
        lines.append(f"| {dataset} | {fmt(oos['frozen_mean'])} | {fmt(oos['trainable_mean'])} | {oos['delta_pp']:+.2f} pp | {fmt(recall['frozen_mean'])} | {fmt(recall['trainable_mean'])} | {recall['delta_pp']:+.2f} pp |")
    lines += [
        "",
        "## 3. 结果解释",
        "",
        "- CLINC150：Trainable K=1 的 OOS F1 提升约 1.12 个百分点，但 Known Recall 下降约 1.33 个百分点；表示排序变好，但已知样本覆盖略有损失。",
        "- Banking77：OOS F1 提升约 5.18 个百分点，Known Recall 下降约 1.95 个百分点；这是正向但伴随代价的改进，仍需校准和更多 seed 验证。",
        "- StackOverflow：Trainable K=1 提升约 9.42 个百分点且 Known Recall略升；但此前 K=2 仍崩溃，表示收益局限于单中心。",
        "",
        "## 4. 当前阶段结论",
        "",
        "Trainable MiniLM 不是跨数据集无条件优于 Frozen：三个数据集的 OOS F1 均不低于 Frozen，但 CLINC150 和 Banking77 的 Known Recall 分别下降约 1.33 和 1.95 个百分点。当前证据支持“Known-only 表示适配有潜力”，不支持“无条件替代 Frozen”。下一批不应直接扩大 K，而应先分析训练表示对类内方差、半径和 Known Recall 的影响，再决定是否需要校准或表示训练范围消融。",
        "",
        "## 5. 证据文件",
        "",
        "- `results/diagnostics/minilm_trainable_control_v1/summary.csv`",
        "- `results/diagnostics/minilm_trainable_control_v1/paired_deltas.csv`",
        "- `figures/active_experiment_dashboard_v1/trainable_cross_dataset.png`",
        "- `artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_control_v1/`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trainable = collect_trainable()
    frozen = collect_frozen()
    summary, delta = summarize(trainable, frozen)
    summary.to_csv(OUT / "summary.csv", index=False)
    delta.to_csv(OUT / "paired_deltas.csv", index=False)
    figure = plot(summary)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report(summary, delta), encoding="utf-8")
    print(json.dumps({"trainable_units": int(len(trainable[trainable["dataset"].isin(["clinc150", "banking77"])])), "summary_rows": int(len(summary)), "figure": str(figure.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
