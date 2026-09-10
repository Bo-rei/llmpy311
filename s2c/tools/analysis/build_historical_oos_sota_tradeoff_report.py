#!/usr/bin/env python3
"""Build the external-baseline OOS-SOTA / Known-coverage tradeoff report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19"
RESULT_ROOT = ROOT / "results" / "analysis" / "historical_trainable_sota_tradeoff"
REPORT = ROOT / "docs" / "analysis" / "historical_oos_sota_tradeoff.md"
CURRENT = ROOT / "results" / "analysis" / "historical_paper_ablation" / "current_h1_full_pipeline_summary.csv"

PAPER = {
    ("clinc150", 0.25): (71.75, 95.01, 90.45),
    ("clinc150", 0.50): (79.95, 91.96, 86.78),
    ("clinc150", 0.75): (66.32, 87.10, 79.83),
    ("stackoverflow", 0.25): (73.61, 94.47, 91.04),
    ("stackoverflow", 0.50): (75.48, 89.71, 85.54),
    ("stackoverflow", 0.75): (80.18, 75.57, 81.32),
    ("banking77_oos", 0.25): (75.83, 93.99, 89.07),
    ("banking77_oos", 0.50): (74.90, 88.23, 78.98),
    ("banking77_oos", 0.75): (70.28, 86.49, 77.84),
}
EXTERNAL = {
    ("clinc150", 0.25): (79.57, 93.56, 89.48),
    ("clinc150", 0.50): (85.58, 90.10, 87.93),
    ("clinc150", 0.75): (88.97, 86.00, 87.39),
    ("stackoverflow", 0.25): (80.87, 92.65, 89.07),
    ("stackoverflow", 0.50): (86.71, 88.86, 87.78),
    ("stackoverflow", 0.75): (87.66, 74.55, 83.56),
    ("banking77_oos", 0.25): (73.05, 86.57, 81.19),
    ("banking77_oos", 0.50): (82.54, 79.93, 81.51),
    ("banking77_oos", 0.75): (86.44, 69.37, 81.39),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def known_count(dataset: str, kir: float) -> int:
    path = DATA_ROOT / dataset / f"kir{int(kir * 100):02d}_seed13" / "KNOWN_INTENTS.json"
    return len(json.loads(path.read_text(encoding="utf-8"))["known_intents"])


def corrected_known(f1_all: float, oos_f1: float, count: int) -> float:
    return ((count + 1.0) * f1_all - oos_f1) / count


def current_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in read_csv(CURRENT):
        dataset = source["dataset"]
        kir = float(source["kir"])
        count = known_count(dataset, kir)
        f1_all = float(source["f1_all_mean"])
        oos = float(source["oos_f1_mean"])
        acc = float(source["accuracy_mean"])
        rows.append(
            {
                "dataset": dataset,
                "kir": kir,
                "configuration": source["configuration"],
                "source": "historical_paper_ablation/current_h1_full_pipeline_summary.csv",
                "known_f1": corrected_known(f1_all, oos, count),
                "oos_f1": oos,
                "accuracy": acc,
                "oos_f1_std": float(source["oos_f1_std"]),
                "known_recall": float(source["known_recall_mean"]),
                "false_acceptance": float(source["false_accept_rate_mean"]),
                "direct_pipeline_verified": True,
            }
        )
    return rows


def banking_tradeoff_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kir in (0.25, 0.50, 0.75):
        root = RESULT_ROOT / f"kir{int(kir * 100):02d}" / "full_pipeline_selected_summary.csv"
        source = read_csv(root)[0]
        count = known_count("banking77_oos", kir)
        f1_all = float(source["f1_all_mean"]) * 100.0
        oos = float(source["oos_f1_mean"]) * 100.0
        rows.append(
            {
                "dataset": "banking77_oos",
                "kir": kir,
                "configuration": f"K={source['k']},lambda={source['radius_lambda']},threshold={source['threshold']},{source['acceptance_mode']}",
                "source": f"historical_trainable_sota_tradeoff/kir{int(kir * 100):02d}",
                "known_f1": corrected_known(f1_all, oos, count),
                "oos_f1": oos,
                "accuracy": float(source["overall_accuracy_mean"]) * 100.0,
                "oos_f1_std": float(source["oos_f1_std"]) * 100.0,
                "known_recall": float(source["known_recall_mean"]) * 100.0,
                "false_acceptance": float(source["false_accept_rate_mean"]) * 100.0,
                "direct_pipeline_verified": True,
            }
        )
    return rows


def build() -> dict[str, Any]:
    rows = [row for row in current_rows() if row["dataset"] != "banking77_oos"] + banking_tradeoff_rows()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = (row["dataset"], float(row["kir"]))
        paper = PAPER[key]
        external = EXTERNAL[key]
        enriched = dict(row)
        enriched.update(
            {
                "external_best_known_f1": external[0],
                "external_best_oos_f1": external[1],
                "external_best_accuracy": external[2],
                "delta_known_vs_external_pp": row["known_f1"] - external[0],
                "delta_oos_vs_external_pp": row["oos_f1"] - external[1],
                "delta_accuracy_vs_external_pp": row["accuracy"] - external[2],
                "paper_ours_known_f1": paper[0],
                "paper_ours_oos_f1": paper[1],
                "paper_ours_accuracy": paper[2],
                "delta_oos_vs_paper_ours_pp": row["oos_f1"] - paper[1],
                "oos_external_sota": row["oos_f1"] >= external[1],
            }
        )
        output.append(enriched)

    output.sort(key=lambda row: (row["dataset"], float(row["kir"])))
    write_csv(RESULT_ROOT / "summary.csv", output)
    sota = [row for row in output if row["oos_external_sota"]]
    sota.sort(
        key=lambda row: min(row["delta_known_vs_external_pp"], row["delta_accuracy_vs_external_pp"]),
        reverse=True,
    )

    lines = [
        "# OOS SOTA 与 Known/Accuracy 代价权衡",
        "",
        "目标：OOS F1 至少达到论文其他 baseline 的最好值，再尽量保留 Known F1 和 Accuracy。这里的 Known F1 使用全测试集 Known 类 macro F1，包含 OOS 被误接受带来的 precision 损失。",
        "",
        "| 数据集 | KIR | 当前配置 | Known F1 | OOS F1 | Acc | ΔKnown / ΔOOS / ΔAcc vs 其他 baseline 最好值 | OOS达标 |",
        "|---|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for row in output:
        lines.append(
            f"| {row['dataset']} | {float(row['kir']):.2f} | {row['configuration']} | "
            f"{row['known_f1']:.2f} | {row['oos_f1']:.2f}±{row['oos_f1_std']:.2f} | {row['accuracy']:.2f} | "
            f"{row['delta_known_vs_external_pp']:+.2f} / {row['delta_oos_vs_external_pp']:+.2f} / {row['delta_accuracy_vs_external_pp']:+.2f} pp | "
            f"{'是' if row['oos_external_sota'] else '否'} |"
        )
    lines += [
        "",
        "## 最值得保留的工作点",
        "",
        "以下按 OOS 达标后，Known F1/Accuracy 的最差相对差值排序；它们是当前已有直接 pipeline 结果，不是测试集重新选参。",
        "",
        "| 排名 | 数据集/KIR | OOS F1 | Known F1 | Acc | 主要代价 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for index, row in enumerate(sota, 1):
        weakest = min(row["delta_known_vs_external_pp"], row["delta_accuracy_vs_external_pp"])
        metric = "Known F1" if row["delta_known_vs_external_pp"] <= row["delta_accuracy_vs_external_pp"] else "Accuracy"
        lines.append(
            f"| {index} | {row['dataset']} / {float(row['kir']):.2f} | {row['oos_f1']:.2f} | "
            f"{row['known_f1']:.2f} | {row['accuracy']:.2f} | {metric} {weakest:+.2f} pp |"
        )
    near_ours_keys = [("clinc150", 0.50), ("stackoverflow", 0.50), ("banking77_oos", 0.75)]
    near_ours = [next(row for row in output if (row["dataset"], float(row["kir"])) == key) for key in near_ours_keys]
    lines += [
        "",
        "## OOS 接近论文 Ours 时的 Known 优先工作点",
        "",
        "下表选取每个数据集当前最适合的工作点：OOS F1 至少达到论文 Ours 附近，同时在已有结果中尽量保留 Known F1。",
        "",
        "| 数据集/KIR | 配置 | Known F1 | OOS F1 | Acc | Known Recall | False Acceptance | Δ vs 论文 Ours (K/O/Acc) | Δ vs 其他 baseline 最好 (K/O/Acc) |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in near_ours:
        paper = PAPER[(row["dataset"], float(row["kir"]))]
        lines.append(
            f"| {row['dataset']} / {float(row['kir']):.2f} | {row['configuration']} | {row['known_f1']:.2f} | "
            f"{row['oos_f1']:.2f}±{row['oos_f1_std']:.2f} | {row['accuracy']:.2f} | {row['known_recall']:.2f} | {row['false_acceptance']:.2f} | "
            f"{row['known_f1']-paper[0]:+.2f} / {row['oos_f1']-paper[1]:+.2f} / {row['accuracy']-paper[2]:+.2f} | "
            f"{row['delta_known_vs_external_pp']:+.2f} / {row['delta_oos_vs_external_pp']:+.2f} / {row['delta_accuracy_vs_external_pp']:+.2f} |"
        )
    lines += [
        "",
        "结论：StackOverflow KIR=.25 是当前最平衡的设置，OOS F1 高于其他 baseline 最好值约 2.73 pp，Known F1 仅低约 0.80 pp，Accuracy 高约 2.51 pp。CLINC KIR=.50 也达到外部 baseline OOS SOTA，Known F1 约低 3.13 pp，Accuracy 基本持平。BANKING77-OOS 三个 KIR 都能达到外部 baseline OOS SOTA，但 Known F1 代价明显更大，KIR=.25 的代价最小。",
        "",
        "所有比较属于 historical_v19 H1 controlled evidence；论文 Ours 另列为历史参照。选择使用验证集 OOS target、Accuracy 和 Known Recall，三组 Banking 结果均用 CUDA 直接 full pipeline 确认；没有训练新的 MiniLM、Router 或 Expert。",
        "",
        "结果入口：",
        "",
        "- [机器可读汇总](../../results/analysis/historical_trainable_sota_tradeoff/summary.csv)",
        "- [Known 口径修正](historical_known_oos_tradeoff.md)",
        "- [论文主表](../../fulltex.tex:372)",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": "s2c.historical_trainable_sota_tradeoff.v1",
        "stage": "historical_trainable_sota_tradeoff",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert",
        "objective": "validation_oos_f1_at_external_baseline_max_accuracy_then_known_recall",
        "datasets": ["clinc150", "stackoverflow", "banking77_oos"],
        "kirs": [0.25, 0.50, 0.75],
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "direct_banking_cuda_cells": 9,
        "summary": "results/analysis/historical_trainable_sota_tradeoff/summary.csv",
        "report": "docs/analysis/historical_oos_sota_tradeoff.md",
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
