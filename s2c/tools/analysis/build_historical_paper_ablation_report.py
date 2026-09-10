#!/usr/bin/env python3
"""Materialize the paper-setting ablation table and current H1 result bridge.

The paper ablation artifact lives outside the repository because it contains the
historical evaluation roots.  This builder copies only aggregate metrics into a
public, auditable result bundle and adds the current H1 Trainable full-pipeline
summary when the corresponding result bundles are available.  It deliberately
does not expose paths to raw predictions, text, embeddings, or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT.parent / "artifacts" / "s2c" / "outputs" / "paper_results"
OUTPUT_ROOT = ROOT / "results" / "analysis" / "historical_paper_ablation"
REPORT_PATH = ROOT / "docs" / "analysis" / "historical_paper_ablation_report.md"

DATASET_ORDER = ("clinc150", "stackoverflow", "banking77_oos")
DATASET_LABELS = {
    "clinc150": "CLINC150",
    "stackoverflow": "StackOverflow",
    "banking77_oos": "Banking77-OOS",
}
PAPER_REFERENCE_DATASET = {
    "clinc150": "clinc150",
    "stackoverflow": "stackoverflow",
    "banking77_oos": "banking77_oos",
}
PAPER_REFERENCE_COMPARABLE = {
    "clinc150": True,
    "stackoverflow": True,
    "banking77_oos": True,
}
KIR_ORDER = (0.25, 0.50, 0.75)
PAPER = {
    ("clinc150", 0.25): {"oos_f1": 95.01, "accuracy": 90.45},
    ("clinc150", 0.50): {"oos_f1": 91.96, "accuracy": 86.78},
    ("clinc150", 0.75): {"oos_f1": 87.10, "accuracy": 79.83},
    ("stackoverflow", 0.25): {"oos_f1": 94.47, "accuracy": 91.04},
    ("stackoverflow", 0.50): {"oos_f1": 89.71, "accuracy": 85.54},
    ("stackoverflow", 0.75): {"oos_f1": 75.57, "accuracy": 81.32},
    ("banking77_oos", 0.25): {"oos_f1": 93.99, "accuracy": 89.07},
    ("banking77_oos", 0.50): {"oos_f1": 88.23, "accuracy": 78.98},
    ("banking77_oos", 0.75): {"oos_f1": 86.49, "accuracy": 77.84},
}
VARIANT_ORDER = ("ours", "without_gate", "cascade_minilm", "cascade_smollm")
VARIANT_LABELS = {
    "ours": "Ours",
    "without_gate": "Without Gate",
    "cascade_minilm": "Cascade-MiniLM",
    "cascade_smollm": "Cascade-SmolLM",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _kir_from_tag(tag: str) -> float:
    match = re.search(r"kir(25|50|75)", str(tag).lower())
    if not match:
        raise ValueError(f"cannot parse KIR from {tag!r}")
    return int(match.group(1)) / 100.0


def canonical_variant(name: str) -> str:
    value = str(name).strip().lower()
    if value == "full_anchor":
        return "ours"
    if value == "cascade_minilm":
        return "cascade_minilm"
    if value == "cascade_smollm":
        return "cascade_smollm"
    if value == "wo_gate_confidence" or value.startswith("banking_wo_"):
        return "without_gate"
    raise ValueError(f"unknown paper ablation variant: {name!r}")


def normalize_paper_ablation(
    source_rows: list[dict[str, str]],
    *,
    use_actual_eval: bool = True,
) -> list[dict[str, Any]]:
    """Return the complete 36-cell table from the actual paper-results eval JSONs.

    ``ablation_summary.csv`` is retained as audit metadata, but its nine Ours
    rows intentionally override metrics with paper-table values and its paths
    point to the historical source tree.  The public ablation table must use
    the materialized ``paper_results/<dataset>/<kir>/<variant>/eval_results.json``
    files instead.
    """

    expected = {
        (dataset, kir, variant)
        for dataset in DATASET_ORDER
        for kir in KIR_ORDER
        for variant in VARIANT_ORDER
    }
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str]] = set()
    for source in source_rows:
        dataset = str(source["slug"])
        kir = _kir_from_tag(source["kir_tag"])
        variant = canonical_variant(source["variant"])
        key = (dataset, kir, variant)
        if key in seen:
            raise ValueError(f"duplicate paper ablation cell: {key}")
        seen.add(key)
        paper = PAPER[(dataset, kir)]
        actual_path = SOURCE_ROOT / dataset / str(source["kir_tag"]) / str(source["variant"]) / "eval_results.json"
        if use_actual_eval:
            if not actual_path.is_file():
                raise FileNotFoundError(f"missing actual paper ablation eval: {actual_path}")
            payload = json.loads(actual_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            oos_f1 = _float(metrics.get("oos_f1", metrics.get("f1_u")))
            accuracy = _float(metrics.get("overall_accuracy", metrics.get("accuracy")))
            macro_f1 = _float(metrics.get("macro_f1", metrics.get("f1_all")))
            known_macro_f1 = _float(metrics.get("known_macro_f1", metrics.get("f1_k")))
            evidence = "historical_paper_eval_artifact"
        else:
            oos_f1 = _float(source.get("oos_f1"))
            accuracy = _float(source.get("overall_accuracy"))
            macro_f1 = _float(source.get("macro_f1"))
            known_macro_f1 = _float(source.get("known_macro_f1"))
            evidence = "historical_paper_summary_metadata"
        normalized.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "kir": kir,
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "status": "existing_eval_artifact" if use_actual_eval else str(source.get("status", "unknown")),
                "overall_accuracy": accuracy,
                "macro_f1": macro_f1,
                "known_macro_f1": known_macro_f1,
                "oos_f1": oos_f1,
                "paper_ours_accuracy": paper["accuracy"],
                "paper_ours_oos_f1": paper["oos_f1"],
                "paper_reference_dataset": PAPER_REFERENCE_DATASET[dataset],
                "paper_reference_comparable": PAPER_REFERENCE_COMPARABLE[dataset],
                "actual_eval_artifact": str(actual_path.relative_to(SOURCE_ROOT)),
                "summary_derived_from_anchor": str(source.get("derived_from_anchor", "")),
                "summary_metric_override_source": str(source.get("metric_override_source", "")),
                "delta_vs_paper_ours_accuracy_pp": None if accuracy is None else accuracy * 100.0 - paper["accuracy"],
                "delta_vs_paper_ours_oos_f1_pp": None if oos_f1 is None else oos_f1 * 100.0 - paper["oos_f1"],
                "delta_vs_same_anchor_accuracy_pp": None,
                "delta_vs_same_anchor_oos_f1_pp": None,
                "evidence": evidence,
            }
        )
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"paper ablation coverage mismatch; missing={missing}, extra={extra}")
    return sorted(normalized, key=lambda row: (DATASET_ORDER.index(row["dataset"]), KIR_ORDER.index(row["kir"]), VARIANT_ORDER.index(row["variant"])))


def _current_row(dataset: str, kir: float, seed: int, source: dict[str, str], prefix: str = "") -> dict[str, Any]:
    def get(name: str) -> Any:
        value = source.get(f"{prefix}{name}")
        return _float(value)

    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "kir": kir,
        "seed": int(seed),
        "configuration": str(source.get("configuration", source.get("selection_scope", "trainable_k1"))),
        "oos_f1": get("oos_f1"),
        "f1_all": get("f1_all"),
        "known_macro_f1": get("known_macro_f1"),
        "accuracy": get("overall_accuracy") if source.get(f"{prefix}overall_accuracy") is not None else get("accuracy"),
        "known_recall": get("known_recall"),
        "false_accept_rate": get("false_accept_rate"),
        "router_error_rate": get("router_error_rate"),
        "expert_error_rate": get("expert_error_rate"),
        "device": str(source.get("device", "cuda")),
        "test_used_for_selection": str(source.get("test_used_for_selection", "False")),
        "oos_used_for_training": str(source.get("oos_used_for_training", "False")),
    }


def _load_current_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kir25_path = RESULTS_ROOT / "historical_trainable_kir25_full_pipeline" / "per_seed.csv"
    if kir25_path.is_file():
        for source in _read_csv(kir25_path):
            rows.append(_current_row(str(source["dataset"]), 0.25, int(source["seed"]), source))

    recipe_path = RESULTS_ROOT / "historical_trainable_gate_recipe_search" / "selected_test.csv"
    if recipe_path.is_file():
        for source in _read_csv(recipe_path):
            if str(source["dataset"]) != "clinc150":
                continue
            row = _current_row("clinc150", 0.50, int(source["seed"]), source, prefix="pipeline_")
            row["configuration"] = f"recipe:{source['name']}"
            rows.append(row)

    adaptive_path = RESULTS_ROOT / "historical_trainable_adaptive_centers" / "selected_test_per_seed.csv"
    if adaptive_path.is_file():
        for source in _read_csv(adaptive_path):
            if str(source.get("selection_scope")) != "overall":
                continue
            dataset = str(source["dataset"])
            if dataset not in {"stackoverflow", "banking77_oos"}:
                continue
            row = _current_row(dataset, 0.50, int(source["seed"]), source, prefix="full_pipeline_")
            row["configuration"] = "adaptive_centers_overall"
            rows.append(row)

    kir75_path = RESULTS_ROOT / "historical_trainable_kir75_full_pipeline" / "per_seed.csv"
    if kir75_path.is_file():
        for source in _read_csv(kir75_path):
            row = _current_row(str(source["dataset"]), 0.75, int(source["seed"]), source)
            row["configuration"] = "trainable_k1_direct"
            rows.append(row)
    return sorted(rows, key=lambda row: (DATASET_ORDER.index(row["dataset"]), KIR_ORDER.index(row["kir"]), int(row["seed"])))


RESULTS_ROOT = ROOT / "results" / "analysis"


def _aggregate_current(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), float(row["kir"]))].append(row)
    output: list[dict[str, Any]] = []
    metrics = ("oos_f1", "f1_all", "known_macro_f1", "accuracy", "known_recall", "false_accept_rate", "router_error_rate", "expert_error_rate")
    for dataset in DATASET_ORDER:
        for kir in KIR_ORDER:
            group = groups.get((dataset, kir), [])
            if not group:
                continue
            row: dict[str, Any] = {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "kir": kir,
                "configuration": ";".join(sorted({str(item["configuration"]) for item in group})),
                "seed_count": len(group),
                "paper_ours_oos_f1": PAPER[(dataset, kir)]["oos_f1"],
                "paper_reference_dataset": PAPER_REFERENCE_DATASET[dataset],
                "paper_reference_comparable": PAPER_REFERENCE_COMPARABLE[dataset],
            }
            for metric in metrics:
                values = [float(item[metric]) for item in group if item.get(metric) is not None]
                if not values:
                    row[f"{metric}_mean"] = None
                    row[f"{metric}_std"] = None
                    continue
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                row[f"{metric}_mean"] = mean * 100.0
                row[f"{metric}_std"] = math.sqrt(variance) * 100.0
            row["delta_oos_f1_pp"] = row["oos_f1_mean"] - PAPER[(dataset, kir)]["oos_f1"]
            row["beats_paper_mean"] = row["delta_oos_f1_pp"] > 0.0
            row["all_seeds_beat_paper"] = all(float(item["oos_f1"]) * 100.0 > PAPER[(dataset, kir)]["oos_f1"] for item in group)
            output.append(row)
    return output


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _pct_pm(mean: Any, std: Any) -> str:
    if mean is None:
        return "—"
    return f"{float(mean):.2f}±{float(std or 0.0):.2f}"


def _write_report(paper_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]], current_summary: list[dict[str, Any]]) -> None:
    by_cell = {(row["dataset"], row["kir"], row["variant"]): row for row in paper_rows}
    anchor_rows = [row for row in paper_rows if row["variant"] == "ours"]
    anchor_mismatches = [
        row
        for row in anchor_rows
        if abs(float(row["delta_vs_paper_ours_oos_f1_pp"] or 0.0)) > 0.01
        or abs(float(row["delta_vs_paper_ours_accuracy_pp"] or 0.0)) > 0.01
    ]
    lines = [
        "# 完整主实验结果与论文设置消融实验",
        "",
        "## 结论先行",
        "",
        "本报告补齐两类结果，但不混淆证据层级：",
        "",
        "1. **当前 H1 Trainable full pipeline**：使用当前工作区已经完成的 CUDA 组件/推理结果，按数据集和 KIR 汇总，并与 `fulltex.tex` 的 Ours 逐格比较。",
        "2. **论文设置消融**：读取已有 `paper_results` 下实际 materialized `eval_results.json`，完整覆盖 `3 数据集 × 3 KIR × 4 变体 = 36/36` 个单元。它是历史 CPU/旧环境 eval artifact，不是本轮新训练的 CUDA H1 消融，不能与第一部分直接合并排名。",
        "",
        "因此，“消融实验完成”在本报告中表示：36 个实际 eval JSON 均存在并已重新读取指标；但它仍不是本轮重新训练的 CUDA 消融。当前 H1 的 Trainable 结果则单独报告其真实 CUDA provenance。",
        "",
        "## 1. 当前 H1 Trainable full pipeline",
        "",
        "当前结果保持 Gate→Router→Expert 结构，OOS 训练和测试选择均为 false；不同 KIR 的最优配置不强行视为同一个模型：CLINC KIR=.50 使用 Known-only recipe search，StackOverflow/Banking77-OOS KIR=.50 使用 adaptive-centers overall 结果，KIR=.25 与 KIR=.75 使用直接 Trainable K=1 结果。",
        "",
        "| 数据集 | KIR | 当前配置 | OOS F1 | full macro F1 | Accuracy | Known Recall | False Acceptance | 相对论文 Ours |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in current_summary:
        comparison_note = "" if bool(summary.get("paper_reference_comparable")) else " †"
        lines.append(
            f"| {summary['dataset_label']} | {float(summary['kir']):.2f} | {summary['configuration']} | "
            f"{_pct_pm(summary.get('oos_f1_mean'), summary.get('oos_f1_std'))} | "
            f"{_pct_pm(summary.get('f1_all_mean'), summary.get('f1_all_std'))} | "
            f"{_pct_pm(summary.get('accuracy_mean'), summary.get('accuracy_std'))} | "
            f"{_pct_pm(summary.get('known_recall_mean'), summary.get('known_recall_std'))} | "
            f"{_pct_pm(summary.get('false_accept_rate_mean'), summary.get('false_accept_rate_std'))} | "
            f"{float(summary['delta_oos_f1_pp']):+.2f} pp{comparison_note} |"
        )
    if not current_summary:
        lines.append("| — | — | 当前 H1 结果尚未汇总 | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "说明：均值和标准差使用 3 个 seed，std 为总体标准差（ddof=0）。当前 H1 表中的 OOS F1 与 Gate 的 OOS 二分类一致；Router/Expert 主要影响 Known 分类、full macro F1 和 Accuracy。",
            "论文表头把该列显示为 `Banking77`，但当前历史 artifact 的实际数据键是 `banking77_oos`；`banking77` archive/protocol_v2 线是另一条数据线，不能替代它。",
            "",
            "### 93.99 的来源与 Banking 协议拆分",
            "",
            "`fulltex.tex` 主表的列名是 **Banking77**，但 artifacts 中的实际 Ours eval 位于 `paper_results/banking77_oos/kir25_seed42/full_anchor/eval_results.json`，其 OOS F1 为 `93.9852%`、Accuracy 为 `89.0686%`，即论文表中的 `93.99/89.07`。该文件的 `data_root` 也是 `data/multidataset/v19/banking77_oos/kir25_seed42`。因此，`93.99` 确实存在于 artifacts，不是当前 `banking77_oos` 中不存在的隐藏数字。",
            "",
            "| 线 | Known | OOS | Test | 结果/含义 |",
            "|---|---:|---:|---:|---|",
            "| 历史 Ours artifact `banking77_oos/kir25_seed42` | 12 Known intents | 66 OOS intents | 4080 | 实际保存了 `93.9852/89.0686`，与论文 `93.99/89.07` 对应；这是历史 Ours，不是当前 Trainable K=1 结果 |",
            "| archive/protocol_v2 `banking77/kir25_seed42` | 19 Known intents | 58 OOS intents | 3080 | 独立的标准 Banking77 数据线，不是上述 `93.99` artifact 的数据根 |",
            "",
            "因此，当前 H1 Trainable 的 Banking77-OOS 结果可以与历史 Ours artifact 做同数据键下的历史系统参照，但仍必须标记方法、Gate、语义校准和下游模型合同不同；标准 `banking77` 结果不能混入这条比较。",
            "",
            "## 2. 论文消融设置与覆盖情况",
            "",
            "论文消融的四个变体为：",
            "",
            "- **Ours**：论文原始 Gate–Router–Expert 配置；",
            "- **Without Gate**：移除几何 Gate，使用论文规定的下游置信度拒识替代；Banking77-OOS 的历史 artifact 使用其对应的 expert-confidence 变体名称；",
            "- **Cascade-MiniLM**：级联各阶段使用 MiniLM；",
            "- **Cascade-SmolLM**：级联各阶段使用 SmolLM。",
            "",
            "每个数据集均覆盖 KIR=.25/.50/.75，历史 anchor key 为 seed42；36 个实际 eval JSON 均存在。`banking77_oos` 与 archive `banking77` 仍是两条不同数据线。",
            "",
        ]
    )
    for dataset in DATASET_ORDER:
        lines.extend(
            [
                f"### {DATASET_LABELS[dataset]}",
                "",
                "| KIR | Historical full anchor OOS F1 / Acc | Without Gate OOS F1 / Acc | Cascade-MiniLM OOS F1 / Acc | Cascade-SmolLM OOS F1 / Acc |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for kir in KIR_ORDER:
            cells = []
            for variant in VARIANT_ORDER:
                row = by_cell[(dataset, kir, variant)]
                cells.append(f"{_pct(float(row['oos_f1']) * 100.0)} / {_pct(float(row['overall_accuracy']) * 100.0)}")
            lines.append(f"| {kir:.2f} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "### Historical anchor 与 `fulltex.tex` 的一致性",
            "",
            f"36 个消融组合均有结果，但历史 `full_anchor` 在 OOS F1 与 Accuracy 两个展示指标上只有 **{9 - len(anchor_mismatches)}/9** 个数据集×KIR 单元同时与 `fulltex.tex` 一致；因此上面的第一列明确标为历史 anchor，而不把它自动改写成论文 Ours。",
            "",
            "| 数据集 | KIR | 历史 full anchor OOS F1 / Acc | `fulltex.tex` Ours OOS F1 / Acc | 差值 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in anchor_rows:
        lines.append(
            f"| {row['dataset_label']} | {float(row['kir']):.2f} | "
            f"{_pct(float(row['oos_f1']) * 100.0)} / {_pct(float(row['overall_accuracy']) * 100.0)} | "
            f"{_pct(row['paper_ours_oos_f1'])} / {_pct(row['paper_ours_accuracy'])} | "
            f"OOS {_pct(row['delta_vs_paper_ours_oos_f1_pp'])} pp; Acc {_pct(row['delta_vs_paper_ours_accuracy_pp'])} pp |"
        )
    if anchor_mismatches:
        lines.extend(["", "需要特别保留的 mismatch：", ""])
        for row in anchor_mismatches:
            lines.append(
                f"- {row['dataset_label']}/KIR={float(row['kir']):.2f}：历史 anchor 与论文差值为 "
                f"OOS F1 {_pct(row['delta_vs_paper_ours_oos_f1_pp'])} pp、Accuracy {_pct(row['delta_vs_paper_ours_accuracy_pp'])} pp。"
            )
    lines.extend(
        [
            "",
            "## 3. 消融结果的直接解释",
            "",
            "- **Gate 是必要的结构组件**：Without Gate 在三个数据集和多数 KIR 下 OOS F1 明显低于 Ours，说明只依赖下游置信度不能替代几何 OOS Gate。",
            "- **MiniLM 级联优于 SmolLM 级联**：Cascade-MiniLM 通常保留更高的 OOS F1 和 Accuracy；SmolLM 在 StackOverflow 和 Banking77-OOS 上的退化尤其明显。",
            "- **消融不是当前 Trainable MiniLM 的新结论**：这些表格回答的是论文原始系统组件是否必要；Trainable MiniLM 的表示适配属于当前 H1 follow-up，应看第一节和既有 Frozen/Trainable 配对结果。",
            "- **不能用单个 OOS F1 排名替代整体判断**：例如更保守的变体可能降低 OOS false acceptance，却同时牺牲 Known coverage，因此同时保留 Accuracy、macro F1 和 Known 指标。",
            "",
            "## 4. 证据边界",
            "",
            "- 历史消融实际源文件：`../artifacts/s2c/outputs/paper_results/<dataset>/<kir>/<variant>/eval_results.json`；`ablation_summary.csv` 只作为覆盖/派生关系 metadata，不能覆盖实际 eval 指标。",
            "- 历史 ledger 记录的 CUDA 不可用，因此 36 个消融单元应标记为 **historical paper-eval evidence**；不能写成“本轮 GPU 重跑完成”。",
            "- 当前 H1 结果属于 `historical_v19_paper_main` 的 controlled evidence，严格 H0 的原始目录和完整旧 Cascade 尚未完全逐字恢复。",
            "- `fulltex.tex` 未修改；论文 Ours 数字仅作为 reported reference。",
            "",
            "## 5. 机器可读入口",
            "",
            "- [当前 H1 full-pipeline seed 表](../../results/analysis/historical_paper_ablation/current_h1_full_pipeline_per_seed.csv)",
            "- [当前 H1 full-pipeline 汇总](../../results/analysis/historical_paper_ablation/current_h1_full_pipeline_summary.csv)",
            "- [36 个论文消融单元](../../results/analysis/historical_paper_ablation/paper_ablation_summary.csv)",
            "- [bundle manifest](../../results/analysis/historical_paper_ablation/MANIFEST.json)",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build() -> dict[str, Any]:
    source_csv = SOURCE_ROOT / "ablation_summary.csv"
    source_ledger = SOURCE_ROOT / "ablation_ledger.json"
    paper_rows = normalize_paper_ablation(_read_csv(source_csv))
    current_rows = _load_current_rows()
    current_summary = _aggregate_current(current_rows)
    paper_fields = list(paper_rows[0])
    current_fields = list(current_rows[0]) if current_rows else [
        "dataset", "dataset_label", "kir", "seed", "configuration", "oos_f1", "f1_all",
        "known_macro_f1", "accuracy", "known_recall", "false_accept_rate", "router_error_rate",
        "expert_error_rate", "device", "test_used_for_selection", "oos_used_for_training",
    ]
    summary_fields = list(current_summary[0]) if current_summary else ["dataset", "dataset_label", "kir", "configuration", "seed_count"]
    _write_csv(OUTPUT_ROOT / "paper_ablation_summary.csv", paper_rows, paper_fields)
    _write_csv(OUTPUT_ROOT / "current_h1_full_pipeline_per_seed.csv", current_rows, current_fields)
    _write_csv(OUTPUT_ROOT / "current_h1_full_pipeline_summary.csv", current_summary, summary_fields)
    _write_report(paper_rows, current_rows, current_summary)
    manifest = {
        "schema_version": "s2c.historical_paper_ablation.v1",
        "stage": "historical_paper_ablation_and_current_h1_bridge",
        "paper_protocol": "historical_v19_paper_main_paper_ablation_anchor",
        "current_protocol": "historical_v19_paper_main__H1_controlled_gate_to_router_to_expert",
        "datasets": list(DATASET_ORDER),
        "kir_values": list(KIR_ORDER),
        "paper_variants": [VARIANT_LABELS[key] for key in VARIANT_ORDER],
        "paper_reference_dataset_by_current_key": dict(PAPER_REFERENCE_DATASET),
        "paper_reference_comparable_by_current_key": dict(PAPER_REFERENCE_COMPARABLE),
        "paper_ablation_expected_units": 36,
        "paper_ablation_completed_units": len(paper_rows),
        "paper_ablation_missing_units": 36 - len(paper_rows),
        "paper_ablation_actual_eval_files": sum(1 for row in paper_rows if row["evidence"] == "historical_paper_eval_artifact"),
        "paper_anchor_key": "seed42 historical artifact",
        "paper_anchor_cells_matching_fulltex_both_metrics": len([
            row for row in paper_rows
            if row["variant"] == "ours"
            and abs(float(row["delta_vs_paper_ours_oos_f1_pp"] or 0.0)) <= 0.01
            and abs(float(row["delta_vs_paper_ours_accuracy_pp"] or 0.0)) <= 0.01
        ]),
        "paper_anchor_mismatches": [
            {
                "dataset": row["dataset"],
                "kir": row["kir"],
                "delta_oos_f1_pp": row["delta_vs_paper_ours_oos_f1_pp"],
                "delta_accuracy_pp": row["delta_vs_paper_ours_accuracy_pp"],
            }
            for row in paper_rows
            if row["variant"] == "ours"
            and (
                abs(float(row["delta_vs_paper_ours_oos_f1_pp"] or 0.0)) > 0.01
                or abs(float(row["delta_vs_paper_ours_accuracy_pp"] or 0.0)) > 0.01
            )
        ],
        "paper_ablation_fresh_cuda_rerun": False,
        "paper_ablation_metadata_source": str(source_csv.relative_to(ROOT.parent)),
        "paper_ablation_eval_root": str(SOURCE_ROOT.relative_to(ROOT.parent)),
        "paper_ablation_source": str(source_csv.relative_to(ROOT.parent)),
        "paper_ablation_ledger": str(source_ledger.relative_to(ROOT.parent)),
        "historical_source_cuda_available": False,
        "current_h1_completed_units": len(current_rows),
        "current_h1_summary_units": len(current_summary),
        "current_h1_source_bundles": [
            "results/analysis/historical_trainable_kir25_full_pipeline",
            "results/analysis/historical_trainable_gate_recipe_search",
            "results/analysis/historical_trainable_adaptive_centers",
            "results/analysis/historical_trainable_kir75_full_pipeline",
        ],
        "current_h1_test_used_for_selection": False,
        "current_h1_oos_used_for_training": False,
        "paper_ablation_summary": "results/analysis/historical_paper_ablation/paper_ablation_summary.csv",
        "current_h1_per_seed": "results/analysis/historical_paper_ablation/current_h1_full_pipeline_per_seed.csv",
        "current_h1_summary": "results/analysis/historical_paper_ablation/current_h1_full_pipeline_summary.csv",
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "raw_predictions_exported": False,
        "fulltex_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.parse_args()
    print(json.dumps(build(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
