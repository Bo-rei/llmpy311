#!/usr/bin/env python3
"""Build the paper-style Trainable-Gate extension report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results" / "analysis" / "historical_paper_style_trainable_gate_ablation"
PAPER_ABLATION = ROOT / "results" / "analysis" / "historical_paper_ablation" / "paper_ablation_summary.csv"
REPORT = ROOT / "docs" / "analysis" / "historical_paper_style_trainable_gate_ablation_report.md"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build() -> None:
    trainable = read(RESULT_ROOT / "per_cell.csv")
    paper = read(PAPER_ABLATION)
    lines = [
        "# 论文设置下的 Trainable Gate 扩展消融",
        "",
        "论文原始消融包含 Ours、Without Gate、Cascade-MiniLM 和 Cascade-SmolLM。本报告在同样的 KIR、K=2、lambda、threshold=1 几何设定下，新增一个 Trainable-Gate 变体。",
        "",
        "由于论文原始 Router/Expert 的部分历史 checkpoint 路径无法在当前工作区完整恢复，本变体使用当前 H1 对应 KIR 的固定 Router/Expert；因此它是 paper-style H1 extension，不是论文 H0 的严格复现。",
        "",
        "| 数据集 | KIR | 变体 | Known F1 | OOS F1 | Accuracy |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in sorted(trainable, key=lambda item: (item["dataset"], float(item["kir"]))):
        lines.append(
            f"| {row['dataset']} | {float(row['kir']):.2f} | Trainable Gate + paper geometry | "
            f"{float(row['known_f1']):.2f} | {float(row['oos_f1']):.2f} | {float(row['accuracy']):.2f} |"
        )
    lines += [
        "",
        "## 论文历史四变体的对应表",
        "",
        "以下数值直接来自历史 paper_results artifact，表中保留论文原始消融的 Acc/OOS F1 口径。",
        "",
        "| 数据集 | KIR | Ours | Without Gate | Cascade-MiniLM | Cascade-SmolLM |",
        "|---|---:|---|---|---|---|",
    ]
    grouped: dict[tuple[str, float], dict[str, dict[str, str]]] = {}
    for row in paper:
        grouped.setdefault((row["dataset"], float(row["kir"])), {})[row["variant_label"]] = row
    for key in sorted(grouped, key=lambda item: (item[0], item[1])):
        values = grouped[key]
        cells = []
        for label in ("Ours", "Without Gate", "Cascade-MiniLM", "Cascade-SmolLM"):
            row = values[label]
            cells.append(f"{float(row['overall_accuracy']) * 100:.2f} / {float(row['oos_f1']) * 100:.2f}")
        lines.append(f"| {key[0]} | {key[1]:.2f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 关键限制",
        "",
        "- 论文四变体是历史 seed42 artifact；Trainable-Gate extension 也是 seed42，但下游来源是当前 H1 fixed Router/Expert。",
        "- Trainable-Gate extension 使用 Trainable embedding、每 intent K=2、对角 Mahalanobis、CLINC lambda=.5、其余 lambda=1、normalized boundary threshold=1。",
        "- Known F1 使用全测试集 Known 类 macro F1；论文历史表主要展示 Acc/OOS F1，因此两类 Known F1 不能直接拿旧 artifact 的 legacy 字段替代。",
        "- 没有修改 fulltex.tex，也没有使用测试集选择配置。",
        "",
        "结果入口：",
        "",
        "- [Trainable Gate 9 个单元](../../results/analysis/historical_paper_style_trainable_gate_ablation/per_cell.csv)",
        "- [实验 manifest](../../results/analysis/historical_paper_style_trainable_gate_ablation/MANIFEST.json)",
        "- [论文四变体历史报告](historical_paper_ablation_report.md)",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
