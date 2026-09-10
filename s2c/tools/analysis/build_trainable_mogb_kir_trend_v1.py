"""Summarize Trainable-K1 versus MOGB-Fair effects across KIR values."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "analysis" / "cross_protocol_tradeoff_v1" / "summary_mean_std.csv"
OUT = ROOT / "results" / "analysis" / "trainable_mogb_kir_trend_v1"
FIG = ROOT / "figures" / "trainable_mogb_kir_trend_v1"
REPORT = ROOT / "docs" / "analysis" / "TRAINABLE_MOGB_KIR_TREND_V1.md"
DATASETS = ["clinc150", "banking77", "stackoverflow"]
KIRS = ["0.25", "0.5", "0.75"]
METRICS = ["oos_f1", "f1_all", "known_recall", "false_accept_rate"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows(source_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    keyed = {(r["dataset"], r["kir"], r["method"]): r for r in source_rows if r["method"] in {"trainable_k1", "mogb_minilm"}}
    rows = []
    for dataset in DATASETS:
        for kir in KIRS:
            train = keyed[(dataset, kir, "trainable_k1")]
            mogb = keyed[(dataset, kir, "mogb_minilm")]
            row: dict[str, object] = {"dataset": dataset, "kir": kir, "n_seeds_trainable": int(train["n_seeds"]), "n_seeds_mogb": int(mogb["n_seeds"])}
            for metric in METRICS:
                row[f"trainable_{metric}"] = float(train[metric])
                row[f"mogb_{metric}"] = float(mogb[metric])
                row[f"delta_{metric}_pp"] = (float(train[metric]) - float(mogb[metric])) * 100.0
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def esc(value: object) -> str:
    return html.escape(str(value))


def render_svg(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 1480, 980
    left, right = 105, 70
    panel_top = [90, 350, 610]
    panel_h = 185
    plot_w = width - left - right
    colors = {"clinc150": "#3182bd", "banking77": "#e6550d", "stackoverflow": "#31a354"}
    labels = {"oos_f1": "OOS F1 delta (pp)", "f1_all": "F1-All delta (pp)", "known_recall": "Known Recall delta (pp)"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222}.small{font-size:12px}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.title{font-size:20px;font-weight:700}.subtitle{font-size:13px;fill:#555}.legend{font-size:13px}</style>',
        '<text x="105" y="35" class="title">Trainable-K1 minus MOGB-Fair across KIR</text>',
        '<text x="105" y="58" class="subtitle">Same protocol_v2 registry, five seeds; positive values favor Trainable-K1</text>',
    ]
    for panel_idx, metric in enumerate(("oos_f1", "f1_all", "known_recall")):
        top = panel_top[panel_idx]
        vals = [float(r[f"delta_{metric}_pp"]) for r in rows]
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        span = max(hi - lo, 1.0)
        lo -= span * 0.08
        hi += span * 0.08
        def y(value: float) -> float:
            return top + panel_h - (value - lo) / (hi - lo) * panel_h
        zero = y(0.0)
        parts.append(f'<text x="{left}" y="{top-15}" class="title" style="font-size:17px">{labels[metric]}</text>')
        parts.append(f'<line x1="{left}" y1="{zero:.1f}" x2="{width-right}" y2="{zero:.1f}" class="axis"/>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+panel_h}" class="axis"/>')
        for tick in (0.0, 0.5, 1.0):
            value = lo + (hi - lo) * tick
            yy = y(value)
            parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>')
            parts.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="small">{value:.1f}</text>')
        for dataset in DATASETS:
            sub = [r for r in rows if r["dataset"] == dataset]
            points = []
            for i, row in enumerate(sub):
                x = left + i * (plot_w / 2) + plot_w / 4
                yy = y(float(row[f"delta_{metric}_pp"]))
                points.append((x, yy))
                parts.append(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="5" fill="{colors[dataset]}"/>')
                parts.append(f'<text x="{x:.1f}" y="{yy-9:.1f}" text-anchor="middle" class="small">{float(row[f"delta_{metric}_pp"]):+.1f}</text>')
            path_d = " ".join(("M" if j == 0 else "L") + f" {x:.1f} {yy:.1f}" for j, (x, yy) in enumerate(points))
            parts.append(f'<path d="{path_d}" fill="none" stroke="{colors[dataset]}" stroke-width="2.5"/>')
        for i, kir in enumerate(KIRS):
            x = left + i * (plot_w / 2) + plot_w / 4
            parts.append(f'<text x="{x:.1f}" y="{top+panel_h+21}" text-anchor="middle" class="small">KIR={float(kir):g}</text>')
    parts += [
        '<rect x="1130" y="70" width="12" height="12" fill="#3182bd"/><text x="1150" y="81" class="legend">CLINC150</text>',
        '<rect x="1130" y="91" width="12" height="12" fill="#e6550d"/><text x="1150" y="102" class="legend">Banking77</text>',
        '<rect x="1130" y="112" width="12" height="12" fill="#31a354"/><text x="1150" y="123" class="legend">StackOverflow</text>',
        '<text x="105" y="940" class="subtitle">来源：cross_protocol_tradeoff_v1/summary_mean_std.csv；差值是五 seed 均值的 paired descriptive comparison。</text>',
        '</svg>',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Trainable-K1 与 MOGB-Fair 的 KIR 趋势分析（V1）",
        "",
        "> 本报告只使用已完成的 protocol_v2 五 seed汇总，不重新训练、不使用测试集选参，也不将 MOGB 论文 BERT 数字混入同合同结果。",
        "",
        "## 1. 结果",
        "",
        "- Trainable-K1 相对 MOGB-Fair 的 OOS F1、F1-All 和 Known Recall 差值在三个数据集上均为正；",
        "- 随 KIR 从 0.25 增加到 0.75，三项优势整体扩大；",
        "- StackOverflow 的增幅最明显：OOS F1 差值从 +6.04pp 增加到 +30.72pp，F1-All 从 +33.57pp 增加到 +54.19pp；",
        "- Banking77 的 OOS F1 差值从 +1.35pp 增加到 +21.02pp，但 false acceptance 差值也从 +8.83pp 增加到 +18.84pp，说明 Trainable 的覆盖恢复伴随更高的开放空间风险；",
        "- CLINC150 的 OOS F1 差值从 +2.78pp 增加到 +16.92pp，Known Recall 差值从 +36.53pp 增加到 +45.92pp。",
        "",
        "## 2. 机制解释",
        "",
        "MOGB-Fair 的平均半径和粒球边界在 Known 类比例较低时已经偏保守；KIR 增大后，MOGB 需要覆盖更多 Known intent，但仍保留同样的保守边界逻辑，因此 Known false rejection 的差距扩大。Trainable-K1 的 Known-only 表示适配改善了单中心分数排序和覆盖，因而差距随 KIR 放大。",
        "",
        "这不是“Trainable 在所有方面都更好”：false acceptance 在部分数据集会增加，尤其 Banking77；正确结论是 Trainable-K1 在当前合同下取得了更平衡的 Known/OOS 工作点。",
        "",
        "## 3. 可视化与数据",
        "",
        "- `figures/archive/analysis/trainable_mogb_kir_trend_v1/kir_trend.svg`：OOS F1、F1-All、Known Recall 的三面板趋势图；",
        "- `results/analysis/archive/analysis/trainable_mogb_kir_trend_v1/kir_trend.csv`：Trainable、MOGB 绝对值和逐 KIR 差值；",
        "- 同合同五 seed 主表：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`。",
        "",
        "## 4. 边界",
        "",
        "本趋势只证明当前 protocol_v2 中 Trainable-K1 相对冻结 MiniLM 的 MOGB 组件对照更稳定；不能外推为超过完整 BERT MOGB 论文、ADB 或 DCLOOS。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source_rows = read_csv(SOURCE)
    rows = build_rows(source_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "kir_trend.csv", rows)
    render_svg(rows, FIG / "kir_trend.svg")
    write_report(rows)
    manifest = {"stage": "trainable_mogb_kir_trend_v1", "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE), "rows": len(rows), "outputs": [str((OUT / "kir_trend.csv").relative_to(ROOT)), str((FIG / "kir_trend.svg").relative_to(ROOT)), str(REPORT.relative_to(ROOT))]}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rows": len(rows), "source_sha256": manifest["source_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
