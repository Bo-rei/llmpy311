#!/usr/bin/env python3
"""Build the Banking77-OOS OOS-first Trainable Gate report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOT = ROOT / "results" / "analysis" / "historical_trainable_oos_priority_search"
REPORT_PATH = ROOT / "docs" / "analysis" / "historical_trainable_oos_priority_presentation.md"
BALANCED_ROOT = ROOT / "results" / "analysis" / "historical_trainable_parameter_full_pipeline"
BALANCED_SUMMARY_PATH = BALANCED_ROOT / "full_pipeline_selected_summary.csv"
BALANCED_COMPARISON_PATH = SEARCH_ROOT / "balanced_vs_oos_priority_kir50.csv"
DATASET = "banking77_oos"
KIRS = (0.25, 0.50, 0.75)
PAPER = {
    0.25: {"known_f1": 75.83, "oos_f1": 93.99, "accuracy": 89.07},
    0.50: {"known_f1": 74.90, "oos_f1": 88.23, "accuracy": 78.98},
    0.75: {"known_f1": 70.28, "oos_f1": 86.49, "accuracy": 77.84},
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


def f(value: Any) -> float:
    return float(value)


def fmt(value: Any) -> str:
    return f"{float(value):.2f}"


def default_rows() -> dict[float, dict[str, str]]:
    path = ROOT / "results" / "analysis" / "historical_paper_ablation" / "current_h1_full_pipeline_summary.csv"
    return {float(row["kir"]): row for row in read_csv(path) if row["dataset"] == DATASET}


def load_search(kir: float) -> tuple[dict[str, str], list[dict[str, str]], dict[str, Any]]:
    root = SEARCH_ROOT / f"kir{int(kir * 100):02d}"
    summary = read_csv(root / "full_pipeline_selected_summary.csv")[0]
    seeds = read_csv(root / "full_pipeline_selected_per_seed.csv")
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    return summary, seeds, manifest


def load_balanced_kir50() -> dict[str, str]:
    rows = [
        row
        for row in read_csv(BALANCED_SUMMARY_PATH)
        if row["dataset"] == DATASET
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one balanced {DATASET} row, found {len(rows)}")
    return rows[0]


def build_balanced_comparison(
    oos_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    balanced = load_balanced_kir50()
    oos50 = next(row for row in oos_summary_rows if float(row["kir"]) == 0.50)

    def current_row(
        label: str,
        objective: str,
        configuration: str,
        known_f1: float,
        oos_f1: float,
        oos_f1_std: float | str,
        accuracy: float,
        known_recall: float | str,
        false_acceptance: float | str,
        source: str,
    ) -> dict[str, Any]:
        return {
            "dataset": DATASET,
            "kir": 0.50,
            "selection": label,
            "selection_objective": objective,
            "configuration": configuration,
            "known_f1": known_f1,
            "oos_f1": oos_f1,
            "oos_f1_std": oos_f1_std,
            "accuracy": accuracy,
            "known_recall": known_recall,
            "false_acceptance": false_acceptance,
            "delta_oos_vs_paper_pp": oos_f1 - PAPER[0.50]["oos_f1"],
            "delta_oos_vs_balanced_pp": "",
            "source": source,
        }

    rows = [
        current_row(
            "paper_ours",
            "historical reference",
            "Historical Ours",
            PAPER[0.50]["known_f1"],
            PAPER[0.50]["oos_f1"],
            "",
            PAPER[0.50]["accuracy"],
            "",
            "",
            "fulltex.tex / historical paper reference",
        ),
        current_row(
            "balanced_parameter_search",
            "validation OOS F1 with Known F1 and Accuracy <=1 pp guard",
            f"K={balanced['k']},lambda={balanced['radius_lambda']},threshold={balanced['threshold']},{balanced['acceptance_mode']}",
            f(balanced["known_macro_f1_mean"]) * 100.0,
            f(balanced["oos_f1_mean"]) * 100.0,
            f(balanced["oos_f1_std"]) * 100.0,
            f(balanced["overall_accuracy_mean"]) * 100.0,
            f(balanced["known_recall_mean"]) * 100.0,
            f(balanced["false_accept_rate_mean"]) * 100.0,
            "historical_trainable_parameter_full_pipeline/full_pipeline_selected_summary.csv",
        ),
        current_row(
            "current_adaptive_default",
            "current H1 adaptive-centers selected workpoint",
            "adaptive_centers_overall",
            f(oos50["default_known_f1_mean"]),
            f(oos50["default_oos_f1_mean"]),
            f(oos50["default_oos_f1_std"]),
            f(oos50["default_accuracy_mean"]),
            f(default_rows()[0.50]["known_recall_mean"]),
            f(default_rows()[0.50]["false_accept_rate_mean"]),
            "historical_paper_ablation/current_h1_full_pipeline_summary.csv",
        ),
        current_row(
            "oos_priority",
            "validation OOS F1 only; no Known guard",
            oos50["configuration"],
            f(oos50["oos_priority_known_f1_mean"]),
            f(oos50["oos_priority_oos_f1_mean"]),
            f(oos50["oos_priority_oos_f1_std"]),
            f(oos50["oos_priority_accuracy_mean"]),
            f(oos50["oos_priority_known_recall_mean"]),
            f(oos50["oos_priority_false_acceptance_mean"]),
            "historical_trainable_oos_priority_search/kir50/full_pipeline_selected_summary.csv",
        ),
    ]
    balanced_oos = next(row for row in rows if row["selection"] == "balanced_parameter_search")["oos_f1"]
    for row in rows:
        row["delta_oos_vs_balanced_pp"] = f(row["oos_f1"]) - f(balanced_oos)
    return rows


def build() -> dict[str, Any]:
    defaults = default_rows()
    summary_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for kir in KIRS:
        selected, seeds, manifest = load_search(kir)
        paper = PAPER[kir]
        summary_rows.append(
            {
                "dataset": DATASET,
                "kir": kir,
                "configuration": f"K={selected['k']},lambda={selected['radius_lambda']},threshold={selected['threshold']},{selected['acceptance_mode']}",
                "paper_known_f1": paper["known_f1"],
                "paper_oos_f1": paper["oos_f1"],
                "paper_accuracy": paper["accuracy"],
                "default_oos_f1_mean": f(defaults[kir]["oos_f1_mean"]),
                "default_oos_f1_std": f(defaults[kir]["oos_f1_std"]),
                "default_known_f1_mean": f(defaults[kir]["known_macro_f1_mean"]),
                "default_accuracy_mean": f(defaults[kir]["accuracy_mean"]),
                "oos_priority_known_f1_mean": f(selected["known_macro_f1_mean"]) * 100.0,
                "oos_priority_oos_f1_mean": f(selected["oos_f1_mean"]) * 100.0,
                "oos_priority_oos_f1_std": f(selected["oos_f1_std"]) * 100.0,
                "oos_priority_accuracy_mean": f(selected["overall_accuracy_mean"]) * 100.0,
                "oos_priority_known_recall_mean": f(selected["known_recall_mean"]) * 100.0,
                "oos_priority_false_acceptance_mean": f(selected["false_accept_rate_mean"]) * 100.0,
                "delta_oos_priority_vs_default_pp": f(selected["oos_f1_mean"]) * 100.0 - f(defaults[kir]["oos_f1_mean"]),
                "delta_oos_priority_vs_paper_pp": f(selected["delta_oos_f1_pp"]),
                "delta_known_f1_vs_default_pp": f(selected["known_macro_f1_mean"]) * 100.0 - f(defaults[kir]["known_macro_f1_mean"]),
                "delta_accuracy_vs_default_pp": f(selected["overall_accuracy_mean"]) * 100.0 - f(defaults[kir]["accuracy_mean"]),
                "candidate_count_per_seed": manifest["candidate_count_per_seed_dataset"],
                "selection_guard": manifest["selection_guard"],
                "device": manifest["device"],
            }
        )
        for row in seeds:
            seed_rows.append(
                {
                    "dataset": DATASET,
                    "kir": kir,
                    "seed": row["seed"],
                    "k": row["k"],
                    "radius_lambda": row["radius_lambda"],
                    "threshold": row["threshold"],
                    "acceptance_mode": row["acceptance_mode"],
                    "known_f1": f(row["known_macro_f1"]) * 100.0,
                    "oos_f1": f(row["oos_f1"]) * 100.0,
                    "accuracy": f(row["overall_accuracy"]) * 100.0,
                    "known_recall": f(row["known_recall"]) * 100.0,
                    "false_acceptance": f(row["false_accept_rate"]) * 100.0,
                    "paper_known_f1": paper["known_f1"],
                    "paper_oos_f1": paper["oos_f1"],
                    "paper_accuracy": paper["accuracy"],
                    "direct_pipeline_verified": row["direct_pipeline_verified"],
                }
            )
    write_csv(SEARCH_ROOT / "summary.csv", summary_rows)
    write_csv(SEARCH_ROOT / "per_seed.csv", seed_rows)
    balanced_comparison = build_balanced_comparison(summary_rows)
    write_csv(BALANCED_COMPARISON_PATH, balanced_comparison)
    lines = [
        "# Banking77-OOS OOS-first Trainable Gate 搜索",
        "",
        "更正：本页旧 Known F1 是真实 Known 子集指标，未包含 OOS false positives；论文对比请使用[全测试集口径修正及折中配置](historical_known_oos_tradeoff.md)。OOS F1 和 Accuracy 不变。",
        "",
        "本轮只用 validation OOS F1 选择边界，Known F1、Accuracy、Known Recall 和 False Acceptance 记录为代价。",
        "",
        "| KIR | OOS-first 配置 | 论文 Known F1 / OOS F1 / Acc | 当前默认 OOS F1 | OOS-first Known F1 / OOS F1 / Acc | 相对默认 OOS F1 | 相对论文 OOS F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {float(row['kir']):.2f} | {row['configuration']} | "
            f"{fmt(row['paper_known_f1'])} / {fmt(row['paper_oos_f1'])} / {fmt(row['paper_accuracy'])} | "
            f"{fmt(row['default_oos_f1_mean'])}±{fmt(row['default_oos_f1_std'])} | "
            f"{fmt(row['oos_priority_known_f1_mean'])} / {fmt(row['oos_priority_oos_f1_mean'])} / {fmt(row['oos_priority_accuracy_mean'])} | "
            f"{float(row['delta_oos_priority_vs_default_pp']):+.2f} pp | "
            f"{float(row['delta_oos_priority_vs_paper_pp']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 之前的‘均衡结果’与当前 OOS-first 结果",
            "",
            "之前的均衡搜索只在 KIR=.50 进行：validation 仍优化 OOS F1，但要求 Known F1 和 Accuracy 相对 K=1 基线最多下降 1 pp。它不是三种 KIR 的统一最优，也不是简单平均三个指标。",
            "",
            "| 结果线 | 选择目标 | 配置 | Known F1 | OOS F1 | Acc | Known Recall | False Acceptance | 相对均衡 OOS F1 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in balanced_comparison:
        std = f"±{fmt(row['oos_f1_std'])}" if row["oos_f1_std"] != "" else ""
        known_recall = fmt(row["known_recall"]) if row["known_recall"] != "" else "—"
        false_acceptance = fmt(row["false_acceptance"]) if row["false_acceptance"] != "" else "—"
        lines.append(
            f"| {row['selection']} | {row['selection_objective']} | {row['configuration']} | "
            f"{fmt(row['known_f1'])} | {fmt(row['oos_f1'])}{std} | {fmt(row['accuracy'])} | "
            f"{known_recall} | {false_acceptance} | {float(row['delta_oos_vs_balanced_pp']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "关键结论：在 KIR=.50，OOS-first 相比均衡结果只增加约 +0.20 pp OOS F1，但 Known F1 下降约 4.54 pp、Known Recall 下降约 7.67 pp；False Acceptance 下降约 2.48 pp，Accuracy 仅增加约 0.25 pp。因此这次搜索确实沿着‘牺牲 Known coverage 换 OOS 拒识’方向移动，但 OOS F1 的增益很小，并没有出现同等幅度的跃升。当前 H1 adaptive default 的 OOS F1 为 91.89，仍略高于 KIR=.50 OOS-first 的 91.73。",
            "",
            "KIR=.25 和 KIR=.75 的 OOS-first 结果分别是 95.95±0.48 和 88.56±0.48；它们相对各自论文 Ours 高约 +1.96 和 +2.07 pp，但不能与 KIR=.50 的均衡结果直接当作同一个 operating point。",
            "",
            "KIR=.25 达到 95.95±0.48 OOS F1，相对历史 Ours 93.99 提升 +1.96 pp；Known F1 约 71.86，Known Recall 约 61.53%。",
            "KIR=.50 的本轮 OOS-first 配置为 91.73±0.08，低于此前 adaptive-centers 的 91.89；KIR=.75 达到 88.56±0.48，相对历史 Ours 提升约 2.07 pp。",
            "",
            "搜索范围：K={1,2,3,5}，lambda={.25,.5,.75,1,1.25,1.5,2,2.5}，threshold=.20–1.20，两种 acceptance mode。测试集只在 validation 选择锁定后确认。",
            "",
            "结果入口：",
            "",
            "- [三 KIR 汇总](../../results/analysis/historical_trainable_oos_priority_search/summary.csv)",
            "- [KIR=.50 均衡 vs OOS-first 汇总](../../results/analysis/historical_trainable_oos_priority_search/balanced_vs_oos_priority_kir50.csv)",
            "- [原始均衡参数搜索汇报](historical_trainable_parameter_full_pipeline_presentation.md)",
            "- [逐 seed 结果](../../results/analysis/historical_trainable_oos_priority_search/per_seed.csv)",
            "- [KIR=.25 manifest](../../results/analysis/historical_trainable_oos_priority_search/kir25/MANIFEST.json)",
            "- [KIR=.50 manifest](../../results/analysis/historical_trainable_oos_priority_search/kir50/MANIFEST.json)",
            "- [KIR=.75 manifest](../../results/analysis/historical_trainable_oos_priority_search/kir75/MANIFEST.json)",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "schema_version": "s2c.historical_trainable_oos_priority_search.v1",
        "stage": "historical_trainable_oos_priority_search",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert",
        "dataset": DATASET,
        "kirs": list(KIRS),
        "seeds": [13, 42, 87],
        "selection_objective": "validation_oos_f1_only",
        "known_metrics_are_diagnostics": True,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "completed_selected_cells": len(summary_rows),
        "completed_direct_seed_cells": len(seed_rows),
        "summary": "results/analysis/historical_trainable_oos_priority_search/summary.csv",
        "balanced_comparison_kir50": "results/analysis/historical_trainable_oos_priority_search/balanced_vs_oos_priority_kir50.csv",
        "per_seed": "results/analysis/historical_trainable_oos_priority_search/per_seed.csv",
        "report": "docs/analysis/historical_trainable_oos_priority_presentation.md",
        "raw_predictions_exported": False,
    }
    (SEARCH_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
