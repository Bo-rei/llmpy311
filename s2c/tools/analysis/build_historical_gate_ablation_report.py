#!/usr/bin/env python3
"""Build the matched Frozen-Gate versus Trainable-Gate ablation report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results" / "analysis" / "historical_gate_ablation"
REPORT = ROOT / "docs" / "analysis" / "historical_gate_ablation_report.md"
TRAINABLE_ROOTS = {
    0.25: ROOT / "results" / "analysis" / "historical_trainable_kir25_full_pipeline",
    0.50: ROOT / "results" / "analysis" / "historical_trainable_full_pipeline",
    0.75: ROOT / "results" / "analysis" / "historical_trainable_kir75_full_pipeline",
}
FROZEN_ROOTS = {
    0.25: RESULT_ROOT / "frozen_kir25",
    0.50: RESULT_ROOT / "frozen_kir50",
    0.75: RESULT_ROOT / "frozen_kir75",
}
KIRS = (0.25, 0.50, 0.75)
DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
METRICS = ("known_f1", "oos_f1", "accuracy", "known_recall", "false_acceptance")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def corrected_known(row: dict[str, str]) -> float:
    count = int(row["known_count"])
    f1_all = float(row["f1_all"])
    oos = float(row["oos_f1"])
    return 100.0 * ((count + 1.0) * f1_all - oos) / count


def load_rows(root: Path, gate: str, kir: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in read_csv(root / "per_seed.csv"):
        rows.append(
            {
                "dataset": source["dataset"],
                "kir": float(source["kir"]),
                "seed": int(source["seed"]),
                "gate": gate,
                "known_f1": corrected_known(source),
                "oos_f1": 100.0 * float(source["oos_f1"]),
                "f1_all": 100.0 * float(source["f1_all"]),
                "accuracy": 100.0 * float(source["overall_accuracy"]),
                "known_recall": 100.0 * float(source["known_recall"]),
                "false_acceptance": 100.0 * float(source["false_accept_rate"]),
                "router_error": 100.0 * float(source["router_error_rate"]),
                "expert_error": 100.0 * float(source["expert_error_rate"]),
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["kir"], row["gate"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (dataset, kir, gate), group in sorted(grouped.items()):
        item: dict[str, Any] = {"dataset": dataset, "kir": kir, "gate": gate, "seed_count": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = pstdev(values)
        item["router_error_mean"] = mean(float(row["router_error"]) for row in group)
        item["expert_error_mean"] = mean(float(row["expert_error"]) for row in group)
        output.append(item)
    return output


def build() -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for kir in KIRS:
        per_seed.extend(load_rows(FROZEN_ROOTS[kir], "frozen_k1", kir))
        per_seed.extend(load_rows(TRAINABLE_ROOTS[kir], "trainable_k1", kir))
    summaries = aggregate(per_seed)
    by_key = {(row["dataset"], row["kir"], row["gate"]): row for row in summaries}
    paired: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            frozen = by_key[(dataset, kir, "frozen_k1")]
            trainable = by_key[(dataset, kir, "trainable_k1")]
            item = {"dataset": dataset, "kir": kir}
            for metric in METRICS:
                item[f"frozen_{metric}"] = frozen[f"{metric}_mean"]
                item[f"trainable_{metric}"] = trainable[f"{metric}_mean"]
                item[f"delta_{metric}"] = trainable[f"{metric}_mean"] - frozen[f"{metric}_mean"]
            paired.append(item)

    write_csv(RESULT_ROOT / "per_seed.csv", per_seed)
    write_csv(RESULT_ROOT / "summary.csv", summaries)
    write_csv(RESULT_ROOT / "paired_delta.csv", paired)

    lines = [
        "# Frozen MiniLM Gate vs Trainable MiniLM Gate 消融",
        "",
        "这组消融只改变 Gate 表示：Frozen all-MiniLM-L6-v2 与当前 Trainable MiniLM；KIR、K=1、对角 Mahalanobis、mean+1×std、threshold=1、Router、Expert、数据和 seed 均保持一致。",
        "",
        "论文中的 Without Gate / Cascade-MiniLM / Cascade-SmolLM 36 个 artifact 仍作为历史论文设置参照；本报告新增的是更直接回答当前方法改动的 Frozen-vs-Trainable Gate 对照。",
        "",
        "Known F1 使用全测试集 Known 类 macro F1，包含 OOS 被接受为 Known 造成的 precision 损失；旧报告中的 Known 子集指标不用于本表。",
        "",
        "| 数据集 | KIR | Gate | Known F1 | OOS F1 | Accuracy | Known Recall | False Acceptance |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {float(row['kir']):.2f} | {row['gate']} | "
            f"{row['known_f1_mean']:.2f}±{row['known_f1_std']:.2f} | "
            f"{row['oos_f1_mean']:.2f}±{row['oos_f1_std']:.2f} | "
            f"{row['accuracy_mean']:.2f}±{row['accuracy_std']:.2f} | "
            f"{row['known_recall_mean']:.2f} | {row['false_acceptance_mean']:.2f} |"
        )
    lines += [
        "",
        "## Trainable − Frozen 配对差值",
        "",
        "| 数据集 | KIR | Δ Known F1 | Δ OOS F1 | Δ Accuracy | Δ Known Recall | Δ False Acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['dataset']} | {float(row['kir']):.2f} | {row['delta_known_f1']:+.2f} | "
            f"{row['delta_oos_f1']:+.2f} | {row['delta_accuracy']:+.2f} | "
            f"{row['delta_known_recall']:+.2f} | {row['delta_false_acceptance']:+.2f} |"
        )
    lines += [
        "",
        "## 读法",
        "",
        "如果 Trainable 的 OOS F1 上升且 False Acceptance 下降，同时 Known F1/Accuracy 没有同步大幅下降，可以把收益归因到 Gate 表示适配在当前固定下游链路上的作用。若 OOS 改善伴随 Known Recall 下降，则应报告为 coverage–rejection trade-off，而不是三个指标全面改善。",
        "",
        "## 证据范围",
        "",
        "- 3 个数据集 × 3 个 KIR × 3 个 seed × 2 个 Gate = 54 个 CUDA full-pipeline 单元。",
        "- Router/Expert 使用对应 KIR 的同一固定组件；没有重新训练下游模型。",
        "- Trainable 和 Frozen 使用同样的 K=1、lambda=1、nearest-sphere detector contract。",
        "- fulltex.tex 未修改；论文四变体历史 artifact 不与本表混合排名。",
        "",
        "结果入口：",
        "",
        "- [逐 seed 结果](../../results/analysis/historical_gate_ablation/per_seed.csv)",
        "- [汇总](../../results/analysis/historical_gate_ablation/summary.csv)",
        "- [Trainable−Frozen 差值](../../results/analysis/historical_gate_ablation/paired_delta.csv)",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "s2c.historical_gate_ablation.v1",
        "stage": "historical_gate_ablation_full_pipeline_v1",
        "protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert",
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "seeds": [13, 42, 87],
        "gate_variants": ["frozen_k1", "trainable_k1"],
        "completed_units": len(per_seed),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "summary": "results/analysis/historical_gate_ablation/summary.csv",
        "paired_delta": "results/analysis/historical_gate_ablation/paired_delta.csv",
        "report": "docs/analysis/historical_gate_ablation_report.md",
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
