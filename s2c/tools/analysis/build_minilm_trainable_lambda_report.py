"""Aggregate the Known-only lambda calibration for Trainable MiniLM."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87)
K_VALUES = (1, 2)
LAMBDA_VALUES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
METRICS = (
    "test_oos_f1",
    "test_f1_all",
    "test_f1_k",
    "test_accuracy",
    "test_known_recall",
    "test_false_accept_rate",
    "test_false_reject_rate",
    "test_auroc",
    "test_aupr_oos",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_root(project_root: Path) -> Path:
    return project_root.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1" / "minilm_trainable_lambda_control_v1"


def _load_rows(project_root: Path) -> pd.DataFrame:
    root = _artifact_root(project_root)
    frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = root / "runs" / dataset / f"seed_{seed}" / "lambda_metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            expected = len(K_VALUES) * len(LAMBDA_VALUES)
            if len(frame) != expected:
                raise ValueError(f"{path}: expected {expected} rows, got {len(frame)}")
            frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["dataset", "seed", "k", "radius_lambda"]).any():
        raise ValueError("Duplicate dataset/seed/K/lambda rows")
    if not np.isfinite(output[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite test metric in lambda results")
    return output.sort_values(["dataset", "seed", "k", "radius_lambda"]).reset_index(drop=True)


def _summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = frame.groupby(["dataset", "k", "radius_lambda"], as_index=True)
    mean_std = grouped[list(METRICS)].agg(["mean", "std"]).reset_index()
    mean_std.columns = [
        "dataset",
        "k",
        "radius_lambda",
        *[f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")],
    ]
    selected = frame[frame["radius_lambda"].eq(frame["known_only_selected_lambda"])].copy()
    selected_mean = selected.groupby(["dataset", "k"], as_index=True)[list(METRICS)].agg(["mean", "std"]).reset_index()
    selected_mean.columns = [
        "dataset",
        "k",
        *[f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")],
    ]
    selected_lambdas = selected.groupby(["dataset", "k"], as_index=False)["radius_lambda"].agg(
        selected_lambda_mean="mean", selected_lambda_values=lambda values: ",".join(f"{float(value):.2f}" for value in values)
    )
    selected_summary = selected_mean.merge(selected_lambdas, on=["dataset", "k"], how="left")

    delta_rows: list[dict[str, float | int | str]] = []
    for dataset in DATASETS:
        subset = frame[frame["dataset"].eq(dataset)]
        for lambda_value, label in ((1.0, "fixed_lambda_1"), (None, "known_only_selected")):
            if lambda_value is None:
                selected_rows = subset[subset["radius_lambda"].eq(subset["known_only_selected_lambda"])]
            else:
                selected_rows = subset[subset["radius_lambda"].eq(lambda_value)]
            pivot = selected_rows.pivot(index="seed", columns="k", values=list(METRICS))
            row: dict[str, float | int | str] = {"dataset": dataset, "selection": label, "n_seeds": int(len(selected_rows["seed"].unique()))}
            for metric in METRICS:
                delta = pivot[metric][2] - pivot[metric][1]
                row[f"k2_minus_k1_{metric}_mean"] = float(delta.mean())
                row[f"k2_minus_k1_{metric}_std"] = float(delta.std(ddof=1))
                row[f"k2_wins_{metric}"] = int((delta > 0).sum())
                row[f"k1_wins_{metric}"] = int((delta < 0).sum())
            delta_rows.append(row)
    return mean_std, selected_summary, pd.DataFrame(delta_rows)


def _write_report(frame: pd.DataFrame, selected: pd.DataFrame, deltas: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Trainable MiniLM Known-only λ/K 受控实验",
        "",
        "本阶段复用已经完成的 Trainable MiniLM checkpoint，只改变半径系数 λ，并在 Known calibration 上选择每个 dataset/seed/K 的最小可行 λ（calibration false-reject rate ≤ 0.05）。测试 OOS 仅用于冻结选择规则后的评价，不参与训练、选 λ 或选 K。",
        "",
        "## 完整性",
        "",
        f"- 数据集：{', '.join(DATASETS)}；KIR=0.50；seed={list(SEEDS)}；K={list(K_VALUES)}；λ={list(LAMBDA_VALUES)}。",
        f"- 计划与实际结果：{len(frame)}/108 个 λ 评价单元；每个 dataset×seed 12 行。",
        f"- test_used_for_selection={bool(frame['test_used_for_selection'].any())}；oos_used_for_training={bool(frame['oos_used_for_training'].any())}。",
        "- 该阶段不重新训练 encoder，因此不会把半径修正误报为表示学习收益。",
        "",
        "## Known-only 选择出的 λ",
        "",
        "| 数据集 | K | 选择的 λ（seed 逐项） | 约束在三个 seed 是否都满足 | OOS F1 | Known Recall | False Acceptance | F1-All |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for _, row in selected.sort_values(["dataset", "k"]).iterrows():
        dataset = row["dataset"]
        k = int(row["k"])
        raw = frame[(frame["dataset"] == dataset) & (frame["k"] == k)]
        feasible = bool(raw["known_only_selection_constraint_met"].all())
        lines.append(
            f"| {dataset} | {k} | {row['selected_lambda_values']} | {feasible} | {row['test_oos_f1_mean']:.4f} ± {row['test_oos_f1_std']:.4f} | {row['test_known_recall_mean']:.4f} ± {row['test_known_recall_std']:.4f} | {row['test_false_accept_rate_mean']:.4f} ± {row['test_false_accept_rate_std']:.4f} | {row['test_f1_all_mean']:.4f} ± {row['test_f1_all_std']:.4f} |"
        )
    lines += [
        "",
        "## K=2−K=1 配对差值",
        "",
        "| 数据集 | 选择方式 | OOS F1 Δ | F1-All Δ | Known Recall Δ | False Acceptance Δ | AUROC Δ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in deltas.sort_values(["dataset", "selection"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['selection']} | {row['k2_minus_k1_test_oos_f1_mean'] * 100:+.2f} pp | {row['k2_minus_k1_test_f1_all_mean'] * 100:+.2f} pp | {row['k2_minus_k1_test_known_recall_mean'] * 100:+.2f} pp | {row['k2_minus_k1_test_false_accept_rate_mean'] * 100:+.2f} pp | {row['k2_minus_k1_test_auroc_mean'] * 100:+.2f} pp |"
        )
    lines += [
        "",
        "## 结论边界",
        "",
        "- 若 λ=2 仍无法满足 Known calibration false-reject ≤5%，说明当前训练表示与半径统计之间存在契约失配，单纯扩大半径不能同时保留 Known 覆盖和 OOS 拒识。",
        "- StackOverflow 即使使用 Known-only 选择的 K=2 λ（约 1.25–1.50），仍保持显著的 OOS F1 损失和较高 false acceptance；因此 K=2 退化不是仅由 λ=1 的偶然设置造成。",
        "- 本阶段只能支持“Trainable MiniLM 改善 K=1，但没有普遍修复固定 K=2”的结论；不能据此宣称自适应多中心已解决。",
        "",
        "## 文件",
        "",
        "- `per_seed.csv`：全部 108 个逐 seed×K×λ 结果。",
        "- `mean_std.csv`：λ 曲线均值和标准差。",
        "- `selected.csv`：Known-only 选择结果。",
        "- `k_delta_by_lambda.csv`：固定 λ=1 与 Known-only 选择下的 K=2−K=1 配对差值。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=_project_root())
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = project_root / "results" / "diagnostics" / "minilm_trainable_lambda_control_v1"
    output.mkdir(parents=True, exist_ok=True)
    frame = _load_rows(project_root)
    mean_std, selected, deltas = _summaries(frame)
    frame.to_csv(output / "per_seed.csv", index=False)
    mean_std.to_csv(output / "mean_std.csv", index=False)
    selected.to_csv(output / "selected.csv", index=False)
    deltas.to_csv(output / "k_delta_by_lambda.csv", index=False)

    figure_path = project_root / "figures" / "active_experiment_dashboard_v1" / "trainable_lambda_k_interaction.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), constrained_layout=True)
    for column, dataset in enumerate(DATASETS):
        subset = frame[frame["dataset"] == dataset]
        for row_index, metric in enumerate(("test_oos_f1", "test_known_recall")):
            axis = axes[row_index, column]
            for k, color in ((1, "#4C78A8"), (2, "#F58518")):
                curve = subset[subset["k"] == k].groupby("radius_lambda")[metric].agg(["mean", "std"]).reset_index()
                axis.plot(curve["radius_lambda"], curve["mean"], marker="o", label=f"K={k}", color=color)
                axis.fill_between(curve["radius_lambda"], curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=color, alpha=0.12)
            axis.set_title(f"{dataset}: {metric.replace('test_', '')}")
            axis.set_xlabel("lambda")
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel("mean ± std")
            if row_index == 0 and column == 2:
                axis.legend(frameon=False)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    report_path = project_root / "docs" / "analysis" / "MINILM_TRAINABLE_LAMBDA_CONTROL_V1.md"
    _write_report(frame, selected, deltas, report_path)
    print({"rows": len(frame), "report": str(report_path), "figure": str(figure_path), "selected": str(output / "selected.csv")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
