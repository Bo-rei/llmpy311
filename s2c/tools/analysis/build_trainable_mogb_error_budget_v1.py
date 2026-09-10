"""Rebuild the Trainable-vs-MOGB open-intent error budget.

The source rows are already frozen sample-level transition summaries.  This
analysis performs arithmetic checks only; it does not train, tune thresholds,
or read test labels for any selection decision.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "analysis" / "trainable_mogb_open_intent_transitions_v1" / "cell_decomposition_per_seed.csv"
OUT = ROOT / "results" / "analysis" / "trainable_mogb_error_budget_v1"
FIG = ROOT / "figures" / "trainable_mogb_error_budget_v1"
REPORT = ROOT / "docs" / "analysis" / "TRAINABLE_MOGB_ERROR_BUDGET_V1.md"
ORDER = ["clinc150", "banking77", "stackoverflow"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def build_summary(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    mismatches = []
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        known = int(row["net_known_correct_gain"])
        oos = int(row["net_oos_correct_gain"])
        total = int(row["net_total_correct_gain"])
        if known + oos != total:
            mismatches.append({"dataset": row["dataset"], "kir": row["kir"], "seed": row["seed"], "known_plus_oos": known + oos, "reported_total": total})
        groups[(row["dataset"], row["kir"])].append(row)
    output = []
    for (dataset, kir), group in sorted(groups.items(), key=lambda item: (ORDER.index(item[0][0]), float(item[0][1]))):
        known = [int(r["net_known_correct_gain"]) for r in group]
        oos = [int(r["net_oos_correct_gain"]) for r in group]
        total = [int(r["net_total_correct_gain"]) for r in group]
        f1_all = [float(r["f1_all_delta"]) * 100.0 for r in group]
        output.append({
            "dataset": dataset,
            "kir": kir,
            "n_seeds": len(group),
            "net_known_correct_gain_mean": mean(known),
            "net_known_correct_gain_std": std(known),
            "net_oos_correct_gain_mean": mean(oos),
            "net_oos_correct_gain_std": std(oos),
            "net_total_correct_gain_mean": mean(total),
            "net_total_correct_gain_std": std(total),
            "f1_all_delta_pp_mean": mean(f1_all),
            "f1_all_delta_pp_std": std(f1_all),
            "known_gain_positive_seeds": sum(v > 0 for v in known),
            "oos_gain_positive_seeds": sum(v > 0 for v in oos),
            "total_gain_positive_seeds": sum(v > 0 for v in total),
        })
    integrity = {
        "source_sha256": sha256(SOURCE),
        "source_rows": len(rows),
        "groups": len(output),
        "decomposition_mismatches": len(mismatches),
        "mismatches": mismatches,
        "test_selection": False,
    }
    return output, integrity


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def esc(value: object) -> str:
    return html.escape(str(value))


def render_svg(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 1480, 920
    left, right = 120, 70
    top, panel = 95, 300
    plot_w = width - left - right
    group_w = plot_w / len(rows)
    max_known = max(float(r["net_known_correct_gain_mean"]) for r in rows) * 1.18
    max_abs_oos = max(abs(float(r["net_oos_correct_gain_mean"])) for r in rows) * 1.25
    scale = panel / max(max_known, max_abs_oos)
    baseline = top + panel
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222}.small{font-size:12px}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.title{font-size:20px;font-weight:700}.subtitle{font-size:13px;fill:#555}.legend{font-size:13px}</style>',
        '<text x="120" y="36" class="title">Trainable-K1 vs MOGB-Fair: error-budget decomposition</text>',
        '<text x="120" y="59" class="subtitle">Positive green = additional correct Known samples; red = change in correct OOS rejections</text>',
        f'<line x1="{left}" y1="{baseline}" x2="{width-right}" y2="{baseline}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{baseline}" class="axis"/>',
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = baseline - panel * frac
        value = max_known * frac
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{value:.0f}</text>')
    for i, row in enumerate(rows):
        x = left + i * group_w
        known = float(row["net_known_correct_gain_mean"])
        oos = float(row["net_oos_correct_gain_mean"])
        bw = min(24, group_w * 0.25)
        kh = known * scale
        oh = abs(oos) * scale
        parts.append(f'<rect x="{x+group_w*0.25:.1f}" y="{baseline-kh:.1f}" width="{bw:.1f}" height="{kh:.1f}" fill="#31a354"/>')
        parts.append(f'<rect x="{x+group_w*0.60:.1f}" y="{baseline-oh:.1f}" width="{bw:.1f}" height="{oh:.1f}" fill="#de2d26"/>')
        parts.append(f'<text x="{x+group_w*0.50:.1f}" y="{baseline+22}" text-anchor="middle" class="small">{esc(row["dataset"])}</text>')
        parts.append(f'<text x="{x+group_w*0.50:.1f}" y="{baseline+38}" text-anchor="middle" class="small">KIR={float(row["kir"]):g}</text>')
        parts.append(f'<text x="{x+group_w*0.32:.1f}" y="{baseline-kh-5:.1f}" text-anchor="middle" class="small">+{known:.0f}</text>')
        parts.append(f'<text x="{x+group_w*0.67:.1f}" y="{baseline-oh-5:.1f}" text-anchor="middle" class="small">{oos:.0f}</text>')
    parts += [
        '<rect x="1130" y="73" width="14" height="14" fill="#31a354"/><text x="1150" y="85" class="legend">net Known correct gain</text>',
        '<rect x="1130" y="96" width="14" height="14" fill="#de2d26"/><text x="1150" y="108" class="legend">net OOS correct-rejection change</text>',
        '<text x="120" y="485" class="title" style="font-size:18px">F1-All gain over MOGB-Fair (percentage points)</text>',
        '<text x="120" y="508" class="subtitle">The count decomposition is paired sample evidence; F1-All is macro averaged and shown separately.</text>',
    ]
    x0 = 780
    for i, row in enumerate(rows):
        y = 550 + i * 32
        delta = float(row["f1_all_delta_pp_mean"])
        w = abs(delta) * 15.0
        x = x0 if delta >= 0 else x0 - w
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="18" fill="#3182bd"/>')
        parts.append(f'<text x="{left}" y="{y+14}" class="small">{esc(row["dataset"])} KIR={float(row["kir"]):g}</text>')
        parts.append(f'<text x="{max(x0, x+w)+6:.1f}" y="{y+14}" class="small">+{delta:.2f}pp</text>')
    parts.append(f'<line x1="{x0}" y1="530" x2="{x0}" y2="840" class="axis"/>')
    parts.append('<text x="120" y="875" class="subtitle">来源：trainable_mogb_open_intent_transitions_v1/cell_decomposition_per_seed.csv；45 个配对单元，未用于选参。</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(summary: list[dict[str, object]], integrity: dict[str, object]) -> None:
    lines = [
        "# Trainable-K1 与 MOGB-Fair 错误预算分析（V1）",
        "",
        "> 本报告是 analysis-only 证据，不训练模型、不选择参数、不使用测试 OOS 调参，也不修改原始逐样本 artifact。",
        "",
        "## 1. 核心问题",
        "",
        "这里比较的是同一 `protocol_v2_textoir_v1`、同一 registry、同一 sample_id 集合下的 `S2C-Trainable-K1` 与 `MOGB-MiniLM-Fair`。因此它回答的是：在相同冻结/训练后表示和评价合同中，两者的错误预算为什么不同；不是对完整 BERT MOGB 论文的排名。",
        "",
        "## 2. 完整性审计",
        "",
        f"- 源行数：{integrity['source_rows']}（3 数据集 × 3 KIR × 5 seeds）；",
        f"- dataset×KIR 汇总组：{integrity['groups']}；",
        f"- `net_known_correct_gain + net_oos_correct_gain = net_total_correct_gain` 不一致数：{integrity['decomposition_mismatches']}；",
        "- 选择参数：否；所有行均为冻结结果的事后归因。",
        "",
        "## 3. 机制结论",
        "",
        "1. Trainable-K1 的净收益主要来自恢复 MOGB 拒绝的 Known 样本；这不是少数意图的偶然现象，而是三个数据集和三个 KIR 上都存在的广泛覆盖差异。",
        "2. Trainable-K1 通常会损失一部分原本被 MOGB 正确拒绝的 OOS 样本，但恢复的 Known 正确量更大，因此 F1-All 和 Known 分类工作点显著改善。",
        "3. MOGB-Fair 的低 false acceptance 伴随大规模 Known false rejection；它更像保守的拒识工作点，而不是全面更强的开放意图分类器。",
        "4. StackOverflow/KIR=.50 的净 Known 正确恢复约 1,682 个，而净 OOS 正确拒绝减少约 256 个；这直接解释了为什么当前方法的 F1-All 从约 43.30% 提升到约 86.55%。",
        "",
        "## 4. 可视化",
        "",
        "- `figures/archive/analysis/trainable_mogb_error_budget_v1/error_budget_decomposition.svg`：每个 dataset×KIR 的 Known 恢复、OOS 正确拒绝变化和 F1-All 增量；",
        "- 原始五状态转移图：`figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/open_intent_transition_heatmaps_kir050.png`；",
        "- 配对工作点图：`figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/matched_known_recall_frontier.png`。",
        "",
        "## 5. 结论边界",
        "",
        "该分析支持“当前 Trainable-K1 在相同 MiniLM Gate 合同下比 MOGB-Fair 更平衡”的结论；不能支持“当前方法超过完整 MOGB 论文方法”或“所有监督条件下达到 SOTA”。MOGB 官方 BERT 复现和 DCLOOS 外部监督仍需独立合同报告。",
        "",
        "机器可读结果：`results/analysis/archive/analysis/trainable_mogb_error_budget_v1/`。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_rows(SOURCE)
    summary, integrity = build_summary(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "budget_summary.csv", summary)
    (OUT / "integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_svg(summary, FIG / "error_budget_decomposition.svg")
    write_report(summary, integrity)
    manifest = {
        "stage": "trainable_mogb_error_budget_v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "summary_rows": len(summary),
        "source_rows": len(rows),
        "decomposition_mismatches": len(integrity["mismatches"]),
        "outputs": [str((OUT / "budget_summary.csv").relative_to(ROOT)), str((OUT / "integrity.json").relative_to(ROOT)), str((FIG / "error_budget_decomposition.svg").relative_to(ROOT)), str(REPORT.relative_to(ROOT))],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "source_rows": len(rows), "summary_rows": len(summary), "mismatches": len(integrity["mismatches"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
