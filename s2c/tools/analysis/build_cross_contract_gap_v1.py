"""Build a contract-aware gap table for historical and current experiments.

This analysis intentionally uses only lightweight CSVs already produced by
completed runs.  It never ranks incompatible methods as one SOTA table.  The
SVG is generated without pandas/matplotlib so the audit remains runnable even
when the training environment is unavailable or its torch import is blocked.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "results" / "analysis" / "historical_sota_comparison_v1" / "fulltex_main_results.csv"
CURRENT = ROOT / "results" / "analysis" / "cross_protocol_tradeoff_v1" / "summary_mean_std.csv"
EXTERNAL = ROOT / "results" / "analysis" / "comparison_atlas_v1" / "external_contract_reference.csv"
OUT = ROOT / "results" / "analysis" / "cross_contract_gap_v1"
FIG = ROOT / "figures" / "cross_contract_gap_v1"
REPORT = ROOT / "docs" / "analysis" / "CROSS_CONTRACT_GAP_V1.md"

DATASETS = ["clinc150", "banking77", "stackoverflow"]
KIRS = [0.25, 0.50, 0.75]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f(value: str | float | int) -> float:
    return float(value)


def key(dataset: str, kir: float) -> tuple[str, str]:
    return dataset, f"{kir:.2f}"


def build_gap_rows() -> list[dict[str, object]]:
    historical = read_csv(HIST)
    current = read_csv(CURRENT)
    hist_ours = {
        key(row["dataset"], f(row["kir"])): row
        for row in historical
        if row["method"] == "Ours"
    }
    hist_best = {}
    for row in historical:
        if row["method"] == "Ours":
            continue
        k = key(row["dataset"], f(row["kir"]))
        value = f(row["oos_f1"])
        hist_best[k] = max(hist_best.get(k, float("-inf")), value)
    trainable = {
        key(row["dataset"], f(row["kir"])): row
        for row in current
        if row["method"] == "trainable_k1"
    }
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            k = key(dataset, kir)
            h = hist_ours[k]
            c = trainable[k]
            current_oos = f(c["oos_f1"]) * 100.0
            current_all = f(c["f1_all"]) * 100.0
            current_f1_k = f(c["f1_k"]) * 100.0
            current_accuracy = f(c["accuracy"]) * 100.0
            historical_oos = f(h["oos_f1"])
            historical_known_f1 = f(h["known_f1"])
            historical_accuracy = f(h["accuracy"])
            rows.append(
                {
                    "dataset": dataset,
                    "kir": f"{kir:.2f}",
                    "historical_contract": "historical_full_cascade",
                    "current_contract": "protocol_v2_current_gate",
                    "historical_method": "fulltex Ours (Gate-Router-Expert)",
                    "current_method": "S2C Trainable K=1",
                    "historical_oos_f1_pct": historical_oos,
                    "current_oos_f1_pct": current_oos,
                    "current_f1_all_pct": current_all,
                    "historical_known_f1_pct": historical_known_f1,
                    "current_f1_k_pct": current_f1_k,
                    "historical_accuracy_pct": historical_accuracy,
                    "current_accuracy_pct": current_accuracy,
                    "historical_vs_best_baseline_pp": historical_oos - hist_best[k],
                    "current_minus_historical_oos_pp": current_oos - historical_oos,
                    "current_minus_historical_f1_k_pp": current_f1_k - historical_known_f1,
                    "current_minus_historical_accuracy_pp": current_accuracy - historical_accuracy,
                    "current_oos_f1_std_pp": f(c["oos_f1_std"]) * 100.0,
                    "current_f1_all_std_pp": f(c["f1_all_std"]) * 100.0,
                    "current_seed_count": int(c["n_seeds"]),
                    "historical_all_metric_available": True,
                    "interpretation": "descriptive_contract_gap_not_ranked",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_text(text: object) -> str:
    return html.escape(str(text))


def render_svg(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 1440, 760
    left, right = 90, 40
    plot_width = width - left - right
    top_y, panel_h = 95, 270
    group_w = plot_width / len(rows)
    max_y = 100.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222}.small{font-size:12px}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.title{font-size:20px;font-weight:700}.subtitle{font-size:13px;fill:#555}.legend{font-size:13px}</style>',
        '<text x="90" y="35" class="title">Historical full Cascade vs current Trainable-K1</text>',
        '<text x="90" y="58" class="subtitle">Descriptive contract gap; values are not a cross-contract SOTA ranking</text>',
        f'<line x1="{left}" y1="{top_y + panel_h}" x2="{width-right}" y2="{top_y + panel_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top_y}" x2="{left}" y2="{top_y + panel_h}" class="axis"/>',
    ]
    for tick in (0, 25, 50, 75, 100):
        y = top_y + panel_h - panel_h * tick / max_y
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="small">{tick}</text>')
    for i, row in enumerate(rows):
        x = left + i * group_w
        hist = float(row["historical_oos_f1_pct"])
        cur = float(row["current_oos_f1_pct"])
        bar_w = min(22, group_w * 0.28)
        hist_h = panel_h * hist / max_y
        cur_h = panel_h * cur / max_y
        parts.append(f'<rect x="{x+group_w*0.20:.1f}" y="{top_y+panel_h-hist_h:.1f}" width="{bar_w:.1f}" height="{hist_h:.1f}" fill="#756bb1"/>')
        parts.append(f'<rect x="{x+group_w*0.55:.1f}" y="{top_y+panel_h-cur_h:.1f}" width="{bar_w:.1f}" height="{cur_h:.1f}" fill="#31a354"/>')
        parts.append(f'<text x="{x+group_w*0.50:.1f}" y="{top_y+panel_h+22}" text-anchor="middle" class="small">{svg_text(row["dataset"])}</text>')
        parts.append(f'<text x="{x+group_w*0.50:.1f}" y="{top_y+panel_h+38}" text-anchor="middle" class="small">KIR={float(row["kir"]):g}</text>')
        parts.append(f'<text x="{x+group_w*0.28:.1f}" y="{top_y+panel_h-hist_h-5:.1f}" text-anchor="middle" class="small">{hist:.1f}</text>')
        parts.append(f'<text x="{x+group_w*0.63:.1f}" y="{top_y+panel_h-cur_h-5:.1f}" text-anchor="middle" class="small">{cur:.1f}</text>')
    parts += [
        '<rect x="1100" y="75" width="14" height="14" fill="#756bb1"/><text x="1120" y="87" class="legend">fulltex 历史 Ours</text>',
        '<rect x="1100" y="98" width="14" height="14" fill="#31a354"/><text x="1120" y="110" class="legend">当前 Trainable K=1</text>',
        '<text x="90" y="425" class="title" style="font-size:18px">当前 Trainable-K1 − 历史 fulltex Ours（OOS F1, pp）</text>',
        '<text x="90" y="448" class="subtitle">负值不表示当前方法失败；它反映当前 Gate-only 与历史完整 Cascade 的合同差异</text>',
    ]
    base_y, zero_x, scale = 500, 730, 5.2
    parts.append(f'<line x1="{zero_x}" y1="{base_y-28}" x2="{zero_x}" y2="{base_y+110}" class="axis"/>')
    for i, row in enumerate(rows):
        delta = float(row["current_minus_historical_oos_pp"])
        y = base_y + i * 27
        x0 = zero_x
        x1 = zero_x + delta * scale
        x = min(x0, x1)
        w = abs(x1 - x0)
        color = "#2ca25f" if delta >= 0 else "#de2d26"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="16" fill="{color}"/>')
        parts.append(f'<text x="{left}" y="{y+13}" class="small">{svg_text(row["dataset"])} KIR={float(row["kir"]):g}</text>')
        parts.append(f'<text x="{max(x1, x0)+6:.1f}" y="{y+13}" class="small">{delta:+.2f}</text>')
    parts.append('<text x="90" y="735" class="subtitle">来源：fulltex_main_results.csv 与 cross_protocol_tradeoff_v1/summary_mean_std.csv；当前均值为 5 seeds。</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(rows: list[dict[str, object]], external: list[dict[str, str]]) -> None:
    current_wins = sum(float(row["current_minus_historical_oos_pp"]) > 0 for row in rows)
    gaps = [float(row["current_minus_historical_oos_pp"]) for row in rows]
    mean_gap = sum(gaps) / len(gaps)
    stack = [row for row in rows if row["dataset"] == "stackoverflow" and row["kir"] == "0.50"][0]
    lines = [
        "# 跨合同实验差距审计（V1）",
        "",
        "> 这是实验阶段的统一阅读入口。它把结果对齐到同一数据集和 KIR，但保留训练监督、骨干、系统层级和评价合同；因此不生成一个不公平的 SOTA 排名。",
        "",
        "## 1. 当前到底比较哪个自有方法",
        "",
        "当前 fair 矩阵中的自有最佳对象是 **S2C-Trainable-K1**：只使用 Known train/calibration 训练 MiniLM 最后两层和 projection，随后使用 K=1 Gate；不是 `fulltex.tex` 中的完整 Gate–Router–Expert Cascade，也不是 RC-AMBL/joint-adaptive 候选。",
        "",
        "历史论文的 `Ours` 是旧合同下的完整 Cascade：Frozen MiniLM Gate + Router/Expert。它的高 OOS F1 不能直接归因于单独的 Gate 或固定多中心。",
        "",
        "## 2. 当前可复算的同合同结论",
        "",
        "来源：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`。在 3 数据集 × 3 KIR × 5 seeds 的同一 protocol_v2 Gate 矩阵中：",
        "",
        "- Trainable-K1 的 OOS F1 九格均值为 85.75%，F1-All 九格均值为 83.36%；",
        "- 它在 OOS F1 上 9 格赢 8 格，在 F1-All 上赢 9 格；",
        "- 相比 Frozen-K1，平均 OOS F1 提升 7.11pp；相比 Frozen-K2，提升 10.25pp；",
        "- 这证明的是当前 Known-only MiniLM Gate 合同内的表示适配收益，不是跨论文 SOTA。",
        "",
        "## 3. 与 fulltex 历史 Ours 的逐格差距",
        "",
        f"逐格表见 `results/analysis/archive/analysis/cross_contract_gap_v1/current_vs_historical.csv`。当前 Trainable-K1 在 OOS F1 上 9 格中高于历史 Ours 的格数为 {current_wins}/9，平均差值为 {mean_gap:+.2f}pp；StackOverflow/KIR=.50 当前 OOS F1 为 {float(stack['current_oos_f1_pct']):.2f}%、历史为 {float(stack['historical_oos_f1_pct']):.2f}%，差值 {float(stack['current_minus_historical_oos_pp']):+.2f}pp。这个差距主要反映 Gate-only 与完整 Cascade 的系统层级差异，不能被解释为 Trainable encoder 本身“没有效果”。",
        "",
        f"同一格的其他指标更能说明问题：StackOverflow/KIR=.50 当前 F1-K 为 {float(stack['current_f1_k_pct']):.2f}%、历史 Known F1 为 {float(stack['historical_known_f1_pct']):.2f}%，当前高 {float(stack['current_minus_historical_f1_k_pp']):+.2f}pp；当前 Accuracy 为 {float(stack['current_accuracy_pct']):.2f}%、历史为 {float(stack['historical_accuracy_pct']):.2f}%，差值 {float(stack['current_minus_historical_accuracy_pp']):+.2f}pp。也就是说，当前单中心 Gate 的 Known 分类指标并不弱，历史 OOS F1 优势不能简单归因为 encoder 更强，而应归因于完整 Cascade 的系统合同和 OOS 工作点。",
        "",
        "可视化：`figures/archive/analysis/cross_contract_gap_v1/current_vs_historical.svg`。紫色是历史完整 Cascade，绿色是当前 Trainable-K1；下半图只展示描述性差值。",
        "",
        "## 4. MOGB、ADB、DCLOOS 应如何放置",
        "",
        "| 对象 | 当前证据 | 能否与 Trainable-K1 直接排名 |",
        "|---|---|---|",
        "| MOGB-MiniLM-Fair | 同 TEXTOIR registry、Frozen MiniLM、MOGB 粒球/边界；45 配对单元 | 可以作为同表示组件对照 |",
        "| MOGB 官方逻辑单格 | BERT + 现代兼容层；StackOverflow KIR=.50/seed0 未复现论文工作点 | 不能当作完整论文 MOGB 排名 |",
        "| ADB | BERT/TextOIR 兼容 3 cell，StackOverflow KIR=.50 OOS F1 87.47±1.48% | 只能作外部兼容参照 |",
        "| DA-ADB | 当前 NaN/全类预测 | 无效，不能进入排名 |",
        "| DCLOOS reduced | BERT + pseudo-OOS + 外部 SQuAD OOS，KIR=.75/seed888 | 监督合同更强、不能进入 Known-only 排名 |",
        "",
        "外部机器可读来源：`results/analysis/archive/analysis/comparison_atlas_v1/external_contract_reference.csv`。当前 StackOverflow/KIR=.50 的分层图为 `figures/archive/analysis/experiment_comparison_overview_v2/stackoverflow_kir50_contract_layers.png`。",
        "",
        "## 5. MOGB 论文差距的正确解释",
        "",
        "MOGB 本地 BERT 兼容单格为 Accuracy=75.17、F1-All=68.35、F1-U=79.97、F1-K=67.19；论文公开参考为 88.67、87.49、89.71、87.27。当前代码审计和 Known-only 归因已经验证：",
        "",
        "1. 官方子中心损失的 L1 距离归一化压窄了训练信号；",
        "2. 平均半径工作点过窄，严重误拒 Known；",
        "3. selected-ball 可能遗漏注册 Known 类；",
        "4. 作者数据快照、Known 列表、旧运行环境和最终粒球随机状态未完整恢复。",
        "",
        "修复单一损失或单纯扩大半径都不能恢复论文工作点，所以当前状态必须写成 `official_code_not_reproduced_under_available_materials`，不能写成“本地低分证明 MOGB 无效”。详细证据见 `docs/archive/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`。",
        "",
        "## 6. 当前实验阶段的下一步",
        "",
        "1. 不再重复 E2/E3、旧 R1、固定 K/KIR 扫描或 MOGB-Fair 组件矩阵；",
        "2. 保留 Trainable-K1 作为当前自有安全基线；",
        "3. 若继续做外部实验，只在独立运行时先完成 DA-ADB 数值审计和 DCLOOS 默认单格；不拿 reduced/invalid 结果填主表；",
        "4. 继续补的分析应围绕现有 CSV：逐数据集/KIR/seed 的错误预算、Known/OOS 工作点、MOGB 粒球风险和历史 Cascade 合同差距；",
        "5. 在外部方法合同闭合前，不宣称跨合同 SOTA。",
        "",
        "## 7. 证据入口",
        "",
        "- 历史对比总览：`docs/archive/analysis/EXPERIMENT_COMPARISON_OVERVIEW_V2.md`",
        "- 当前可视化证据包：`docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md`",
        "- MOGB 机制闭环：`docs/archive/analysis/MECHANISM_CLOSURE_V1.md`",
        "- 历史 fulltex 对照：`docs/archive/analysis/HISTORICAL_SOTA_AND_CURRENT_COMPARISON_V1.md`",
        "- 外部状态：`docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`",
        "",
        "本文件和 SVG 只使用已完成轻量结果，不修改训练 artifact、不读取测试 OOS 进行选参。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_gap_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "current_vs_historical.csv", rows)
    external = read_csv(EXTERNAL)
    write_csv(OUT / "external_contract_reference.csv", external)
    render_svg(rows, FIG / "current_vs_historical.svg")
    write_report(rows, external)
    manifest = {
        "stage": "cross_contract_gap_v1",
        "source_files": {
            "historical": str(HIST.relative_to(ROOT)),
            "current": str(CURRENT.relative_to(ROOT)),
            "external": str(EXTERNAL.relative_to(ROOT)),
        },
        "source_sha256": {"historical": sha256(HIST), "current": sha256(CURRENT), "external": sha256(EXTERNAL)},
        "rows": len(rows),
        "contract_warning": "Historical full Cascade, current Gate, MOGB, ADB and DCLOOS are layered, not ranked together.",
        "outputs": [
            str((OUT / "current_vs_historical.csv").relative_to(ROOT)),
            str((OUT / "external_contract_reference.csv").relative_to(ROOT)),
            str((FIG / "current_vs_historical.svg").relative_to(ROOT)),
            str(REPORT.relative_to(ROOT)),
        ],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rows": len(rows), "mean_current_minus_historical_pp": sum(float(r["current_minus_historical_oos_pp"]) for r in rows) / len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
