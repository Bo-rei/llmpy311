"""Aggregate the Trainable MiniLM K=1/K=2 control across three datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = (13, 42, 87)
METRICS = (
    "oos_f1",
    "oos_precision",
    "oos_recall",
    "f1_all",
    "f1_k",
    "accuracy",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
    "auroc",
    "aupr_oos",
    "fpr95",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_root(project_root: Path) -> Path:
    return project_root.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1" / "minilm_trainable_k2_control_v1"


def _load_new_rows(project_root: Path) -> list[dict[str, Any]]:
    root = _artifact_root(project_root)
    rows: list[dict[str, Any]] = []
    for dataset in ("clinc150", "banking77"):
        for seed in SEEDS:
            metrics_path = root / "runs" / dataset / f"seed_{seed}" / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(metrics_path)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            row: dict[str, Any] = {
                "dataset": dataset,
                "kir": 0.50,
                "seed": seed,
                "representation": "trainable_minilm_last2_plus_projection",
                "source": "minilm_trainable_k2_control_v1",
                "source_path": str(metrics_path),
            }
            for k in (1, 2):
                for metric in METRICS:
                    row[f"k{k}_{metric}"] = float(payload[f"k{k}"][metric])
            for metric in METRICS:
                row[f"k2_minus_k1_{metric}"] = float(payload["k2_minus_k1"][metric])
            row["k1_replay_max_abs_delta"] = payload.get("k1_replay_max_abs_delta")
            rows.append(row)
    return rows


def _load_stackoverflow_rows(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "results" / "diagnostics" / "racal_v1" / "stage2_fixed_k2" / "RACAL_V1_STAGE2_PER_SEED.csv"
    frame = pd.read_csv(path)
    frame = frame[(frame["dataset"] == "stackoverflow") & frame["seed"].isin(SEEDS) & (frame["kir"] == 0.5)].copy()
    if len(frame) != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} StackOverflow rows, got {len(frame)}")
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        row: dict[str, Any] = {
            "dataset": "stackoverflow",
            "kir": float(record["kir"]),
            "seed": int(record["seed"]),
            "representation": "trainable_minilm_last2_plus_projection",
            "source": "racal_v1_stage2_fixed_k2_existing",
            "source_path": str(path),
        }
        for k in (1, 2):
            for metric in METRICS:
                row[f"k{k}_{metric}"] = float(record[f"k{k}_{metric}"])
        for metric in METRICS:
            row[f"k2_minus_k1_{metric}"] = float(record[f"k2_minus_k1_{metric}"])
        row["k1_replay_max_abs_delta"] = np.nan
        rows.append(row)
    return rows


def _write_markdown(frame: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Trainable MiniLM K=1/K=2 跨数据集控制",
        "",
        "本阶段只评价已经训练完成的 Trainable MiniLM checkpoint，不重新训练、不重新划分数据，也不使用测试集选择 checkpoint、K、半径或阈值。CLINC150/Banking77 使用 `minilm_trainable_control_v1` checkpoint；StackOverflow 读取已经完成的同协议 RACAL-v1 K=1/K=2 配对结果。",
        "",
        "## 协议与完整性",
        "",
        f"- 数据集：CLINC150、Banking77、StackOverflow；KIR=0.50；seed={list(SEEDS)}。",
        "- 表示：Trainable MiniLM last-2 layers + projection；K=1 与 K=2 共用同一 seed 对应 checkpoint。",
        "- 距离/边界：对角 Mahalanobis、`mean + 1.0*std`、threshold=1.0、固定 partition seed=42。",
        f"- 配对行数：{len(frame)}；每个数据集 3 个 seed；CLINC/Banking 新增 6 个评价单元，StackOverflow 为既有只读结果。",
        "- 所有新增 run 的 `k1_replay_max_abs_delta` 均为 0（浮点容差内）；历史 artifacts 未覆盖。",
        "",
        "## 逐数据集均值（0-1 指标）",
        "",
        "| 数据集 | K | OOS F1 | F1-All | F1-K | Known Recall | False Acceptance | AUROC | AUPR-OOS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['k'])} | {row['oos_f1_mean']:.4f} ± {row['oos_f1_std']:.4f} | {row['f1_all_mean']:.4f} ± {row['f1_all_std']:.4f} | {row['f1_k_mean']:.4f} ± {row['f1_k_std']:.4f} | {row['known_recall_mean']:.4f} ± {row['known_recall_std']:.4f} | {row['false_accept_rate_mean']:.4f} ± {row['false_accept_rate_std']:.4f} | {row['auroc_mean']:.4f} ± {row['auroc_std']:.4f} | {row['aupr_oos_mean']:.4f} ± {row['aupr_oos_std']:.4f} |"
        )
    lines += [
        "",
        "## K=2 相对 K=1 的配对差值",
        "",
        "| 数据集 | OOS F1 Δ | F1-All Δ | F1-K Δ | Known Recall Δ | False Acceptance Δ | AUROC Δ | 方向（OOS F1） |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in deltas.iterrows():
        direction = "K=2较好" if row["oos_f1_delta_mean"] > 0 else "K=1较好"
        lines.append(
            f"| {row['dataset']} | {row['oos_f1_delta_mean'] * 100:+.2f} pp | {row['f1_all_delta_mean'] * 100:+.2f} pp | {row['f1_k_delta_mean'] * 100:+.2f} pp | {row['known_recall_delta_mean'] * 100:+.2f} pp | {row['false_accept_rate_delta_mean'] * 100:+.2f} pp | {row['auroc_delta_mean'] * 100:+.2f} pp | {direction} |"
        )
    lines += [
        "",
        "## 当前解释边界",
        "",
        "- 该阶段只回答“同一 Trainable MiniLM 表示下，固定 K=2 是否仍优于 K=1”。它不是自适应 K，也不是完整 Cascade 结果。",
        "- K=2 若提高 Known Recall 但降低 OOS F1，说明新增局部球扩大了接受区域；不能只看 Known Recall 判断多中心有效。",
        "- StackOverflow 的 K=2 结果来自既有 RACAL-v1 Stage 2，不与本阶段 CLINC/Banking 的新 run 混写；来源列保留了这一差异。",
        "- 需要把这些配对结果与 Frozen、CE-Recon、MOGB、DCLOOS 等方法放到同一 split/指标协议后，才可作 baseline 排名。",
        "",
        "## 机器可读文件",
        "",
        "- `per_seed.csv`：逐 seed 的 K=1/K=2 配对结果。",
        "- `mean_std.csv`：各数据集、各 K 的均值和标准差。",
        "- `delta_summary.csv`：K=2−K=1 的配对汇总。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=_project_root())
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = project_root / "results" / "diagnostics" / "minilm_trainable_k2_control_v1"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(_load_new_rows(project_root) + _load_stackoverflow_rows(project_root))
    frame = frame.sort_values(["dataset", "seed"]).reset_index(drop=True)
    frame.to_csv(output / "per_seed.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for dataset, dataset_group in frame.groupby("dataset"):
        for k in (1, 2):
            row: dict[str, Any] = {"dataset": dataset, "k": int(k)}
            for metric in METRICS:
                values = dataset_group[f"k{k}_{metric}"].astype(float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
            row["n_seeds"] = int(len(dataset_group))
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "k"])
    summary.to_csv(output / "mean_std.csv", index=False)

    delta_rows: list[dict[str, Any]] = []
    for dataset, group in frame.groupby("dataset"):
        row: dict[str, Any] = {"dataset": dataset, "n_seeds": int(len(group))}
        for metric in METRICS:
            values = group[f"k2_minus_k1_{metric}"].astype(float)
            row[f"{metric}_delta_mean"] = float(values.mean())
            row[f"{metric}_delta_std"] = float(values.std(ddof=1))
            row[f"{metric}_wins_k2"] = int((values > 0).sum())
            row[f"{metric}_ties"] = int(np.isclose(values, 0.0, atol=1e-12).sum())
            row[f"{metric}_wins_k1"] = int((values < 0).sum())
        delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows).sort_values("dataset")
    deltas.to_csv(output / "delta_summary.csv", index=False)

    figures = project_root / "figures" / "active_experiment_dashboard_v1"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    plot_metrics = (("oos_f1", "OOS F1"), ("known_recall", "Known Recall"), ("false_accept_rate", "False Acceptance"))
    datasets = ["clinc150", "banking77", "stackoverflow"]
    x = np.arange(len(datasets))
    width = 0.34
    for axis, (metric, title) in zip(axes, plot_metrics, strict=True):
        k1 = [summary.loc[(summary.dataset == dataset) & (summary.k == 1), f"{metric}_mean"].iloc[0] for dataset in datasets]
        k1_std = [summary.loc[(summary.dataset == dataset) & (summary.k == 1), f"{metric}_std"].iloc[0] for dataset in datasets]
        k2 = [summary.loc[(summary.dataset == dataset) & (summary.k == 2), f"{metric}_mean"].iloc[0] for dataset in datasets]
        k2_std = [summary.loc[(summary.dataset == dataset) & (summary.k == 2), f"{metric}_std"].iloc[0] for dataset in datasets]
        axis.bar(x - width / 2, k1, width, yerr=k1_std, capsize=3, label="K=1", color="#4C78A8")
        axis.bar(x + width / 2, k2, width, yerr=k2_std, capsize=3, label="K=2", color="#F58518")
        axis.set_title(title)
        axis.set_xticks(x, ["CLINC150", "Banking77", "StackOverflow"], rotation=18)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("mean ± std")
    axes[-1].legend(frameon=False, loc="best")
    fig.savefig(figures / "trainable_k_interaction_cross_dataset.png", dpi=220)
    plt.close(fig)

    report_path = project_root / "docs" / "analysis" / "MINILM_TRAINABLE_K2_CONTROL_V1.md"
    _write_markdown(frame, summary, deltas, report_path)
    print(json.dumps({"rows": len(frame), "summary": str(output / "mean_std.csv"), "deltas": str(output / "delta_summary.csv"), "report": str(report_path), "figure": str(figures / "trainable_k_interaction_cross_dataset.png")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
