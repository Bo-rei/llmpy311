#!/usr/bin/env python3
"""Build a compact, contract-aware dashboard from frozen experiment summaries.

This is analysis-only: it never reads text, embeddings, or test labels and never
selects a model.  The dashboard is deliberately small so that it can be used as
the first status artifact before opening the larger mechanism reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAIR = ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
EXTERNAL = ROOT / "results/analysis/archive/analysis/external_gpu_runtime_comparison_v1/external_same_protocol_cells.csv"
MOGB_EXACT = ROOT / "results/analysis/archive/analysis/mogb_reproduction_gap_analysis_v2/exact_run_summary.csv"
OUT = ROOT / "results/analysis/archive/analysis/experiment_decision_dashboard_v1"
FIG = ROOT / "figures/archive/analysis/experiment_decision_dashboard_v1"
DOC = ROOT / "docs/analysis/EXPERIMENT_DECISION_DASHBOARD_V1.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "nan", "NaN", "None"):
        return None
    return float(value)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fair = read_csv(FAIR)
    external = read_csv(EXTERNAL)
    exact = read_csv(MOGB_EXACT)
    fair_rows: list[dict[str, object]] = []
    for row in fair:
        fair_rows.append(
            {
                "dataset": row["dataset"],
                "kir": float(row["kir"]),
                "method": row["method"],
                "method_label": row["method_label"],
                "layer": "current_fair_gate",
                "contract": "same TEXTOIR registry + Known-only MiniLM Gate",
                "status": "valid_same_protocol",
                "oos_f1": f(row, "oos_f1"),
                "f1_all": f(row, "f1_all"),
                "known_recall": f(row, "known_recall"),
                "false_accept_rate": f(row, "false_accept_rate"),
            }
        )

    # The external file contains five seed rows for fair methods and a small
    # number of external rows.  Keep only methods that are not already in the
    # fair summary and aggregate valid external cells by method.
    external_rows: list[dict[str, object]] = []
    for method in ("ADB", "DCLOOS reduced"):
        selected = [
            row
            for row in external
            if (
                (row.get("method_label", "").startswith("ADB") if method == "ADB" else row.get("method_label") == method)
                and row.get("valid_semantic_metrics") == "True"
            )
        ]
        if not selected:
            continue
        for row in selected:
            external_rows.append(
                {
                    "dataset": row["dataset"],
                    "kir": float(row["kir"]),
                    "method": method,
                    "method_label": "ADB" if method == "ADB" else row["method_label"],
                    "layer": row.get("layer", "external"),
                    "contract": row.get("contract", "external"),
                    "status": "valid_external_reference",
                    "oos_f1": f(row, "oos_f1"),
                    "f1_all": f(row, "f1_all"),
                    "known_recall": f(row, "known_recall"),
                    "false_accept_rate": f(row, "false_accept_rate"),
                }
            )

    # DCLOOS reduced is stored separately because it uses pseudo/external OOS
    # and is intentionally not part of the same-protocol external CSV.
    dcloos_path = ROOT.parent / "artifacts/s2c/external/dcloos_official_oos_kir75_seed888_reduced_v2/recovery_metrics.json"
    if dcloos_path.is_file():
        payload = json.loads(dcloos_path.read_text(encoding="utf-8"))
        external_rows.append(
            {
                "dataset": "stackoverflow",
                "kir": 0.75,
                "method": "DCLOOS reduced",
                "method_label": "DCLOOS reduced",
                "layer": "external_oos_reference",
                "contract": "BERT + pseudo-OOS + external SQuAD OOS; seed=888",
                "status": "valid_external_reference_not_fair",
                "oos_f1": float(payload.get("f1_u", payload.get("oos_f1"))),
                "f1_all": float(payload.get("f1_all")),
                "known_recall": float(payload.get("known_recall")),
                "false_accept_rate": None,
            }
        )

    # Add the official-logic local MOGB cells as a separate paper-gap layer.
    for row in exact:
        external_rows.append(
            {
                "dataset": row["dataset"],
                "kir": float(row["kir"]),
                "method": "MOGB official local",
                "method_label": "MOGB official local",
                "layer": "official_logic_compatibility",
                "contract": row["contract"],
                "status": "not_reproduced_strict",
                "oos_f1": f(row, "f1_u"),
                "f1_all": f(row, "f1_all"),
                "known_recall": f(row, "known_recall"),
                "false_accept_rate": None,
            }
        )
    return fair_rows, external_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "kir", "method", "method_label", "layer", "contract", "status",
        "oos_f1", "f1_all", "known_recall", "false_accept_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def render_svg(fair_rows: list[dict[str, object]], external_rows: list[dict[str, object]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    labels = [
        "Trainable K=1", "Frozen K=1", "Random K=2", "Frozen K=2",
        "MOGB partition + s2c boundary", "s2c partition + MOGB boundary", "MOGB-MiniLM",
    ]
    means: dict[str, float] = {}
    for label in labels:
        values = [
            float(row["oos_f1"])
            for row in fair_rows
            if row["method_label"] == label and row["oos_f1"] is not None
        ]
        means[label] = sum(values) / len(values)

    width, height = 1200, 760
    top, bar_w, bar_h = 86, 490, 38
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#172033} .small{font-size:13px} .title{font-size:22px;font-weight:700} .section{font-size:16px;font-weight:700} .muted{fill:#5a6475}</style>',
        '<text x="42" y="36" class="title">S2C 当前实验决策面板（冻结结果，不混合同）</text>',
        '<text x="42" y="61" class="small muted">当前公平矩阵：3 数据集 × 3 KIR × 5 seeds；外部方法仅作合同参照</text>',
        '<text x="42" y="90" class="section">A. 同 protocol_v2 fair Gate：九格 OOS F1 均值</text>',
        '<line x1="70" y1="610" x2="560" y2="610" stroke="#8793a6"/>',
        '<text x="70" y="635" class="small muted">0%</text><text x="520" y="635" class="small muted">100%</text>',
        '<text x="690" y="90" class="section">B. StackOverflow/KIR=.50：可比较与不可比较层</text>',
    ]
    colors = ["#1769aa", "#7b8a9a", "#9b59b6", "#d9822b", "#20a39e", "#b85c5c", "#8c8c8c"]
    for i, label in enumerate(labels):
        y = top + i * 58
        value = 100.0 * means[label]
        parts.extend([
            f'<text x="70" y="{y+24}" class="small">{label}</text>',
            f'<rect x="250" y="{y}" width="{bar_w}" height="{bar_h}" rx="5" fill="#e7ebf2"/>',
            f'<rect x="250" y="{y}" width="{bar_w * value / 100:.1f}" height="{bar_h}" rx="5" fill="{colors[i]}"/>',
            f'<text x="{250 + bar_w * value / 100 + 8:.1f}" y="{y+24}" class="small">{value:.2f}%</text>',
        ])

    so_rows = [
        row for row in fair_rows + external_rows
        if row["dataset"] == "stackoverflow" and abs(float(row["kir"]) - 0.5) < 1e-9
    ]
    order = ["Trainable K=1", "ADB", "MOGB partition + s2c boundary", "MOGB-MiniLM", "MOGB official local"]
    so_map = {str(row["method_label"]): row for row in so_rows}
    # ADB is averaged below from the three valid rows; fair rows are already mean rows.
    adb = [row for row in external_rows if row["method_label"] == "ADB" and row["dataset"] == "stackoverflow"]
    if adb:
        so_map["ADB"] = {**adb[0], "oos_f1": sum(float(r["oos_f1"]) for r in adb) / len(adb), "f1_all": sum(float(r["f1_all"]) for r in adb) / len(adb)}
    x0, y0 = 690, 120
    for i, label in enumerate(order):
        row = so_map.get(label)
        y = y0 + i * 72
        if row is None:
            continue
        value = 100 * float(row["oos_f1"])
        all_value = 100 * float(row["f1_all"])
        status = str(row["status"])
        fill = "#1769aa" if label == "Trainable K=1" else ("#c98b2e" if "not_fair" in status else "#7b8a9a")
        parts.extend([
            f'<text x="{x0}" y="{y+20}" class="small">{label}</text>',
            f'<rect x="860" y="{y}" width="260" height="22" rx="4" fill="#e7ebf2"/><rect x="860" y="{y}" width="{2.6*value:.1f}" height="22" rx="4" fill="{fill}"/>',
            f'<text x="1130" y="{y+17}" class="small">OOS {value:.2f}%</text>',
            f'<text x="860" y="{y+43}" class="small muted">F1-All {all_value:.2f}% · {status}</text>',
        ])
    parts.extend([
        '<rect x="680" y="520" width="470" height="120" rx="8" fill="#eef3f8" stroke="#b5c3d3"/>',
        '<text x="700" y="548" class="small">读图规则：</text>',
        '<text x="700" y="572" class="small">蓝色：当前 Known-only fair 候选；灰色：组件/外部参照；</text>',
        '<text x="700" y="596" class="small">橙色：监督或 KIR 不同，不得并入同协议排名。</text>',
        '<text x="700" y="620" class="small muted">MOGB official local 仅表示官方逻辑兼容单格，未复现论文工作点。</text>',
        '</svg>',
    ])
    (FIG / "decision_dashboard.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    fair_rows, external_rows = make_rows()
    all_rows = fair_rows + external_rows
    write_csv(OUT / "decision_rows.csv", all_rows)
    render_svg(fair_rows, external_rows)
    manifest = {
        "schema_version": 1,
        "analysis_only": True,
        "protocol_version": "protocol_v2_textoir_v1",
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in (FAIR, EXTERNAL, MOGB_EXACT)},
        "fair_rows": len(fair_rows),
        "external_reference_rows": len(external_rows),
        "outputs": [
            "results/analysis/archive/analysis/experiment_decision_dashboard_v1/decision_rows.csv",
            "figures/archive/analysis/experiment_decision_dashboard_v1/decision_dashboard.svg",
        ],
        "selection_or_training": False,
        "notes": "External rows are contract-labelled references and are never merged into fair ranking.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fair_mean = {}
    for row in fair_rows:
        fair_mean.setdefault(row["method_label"], []).append(float(row["oos_f1"]))
    lines = [
        "# 实验决策面板 V1",
        "",
        "本面板只读取已冻结结果，集中回答当前比较对象、同协议结果和外部参照的合同边界。它不新增训练、不调参、不读取测试样本。",
        "",
        "## 同协议公平矩阵",
        "",
        "| 方法 | 九格 OOS F1 均值 | 状态 |",
        "|---|---:|---|",
    ]
    for label in labels_from_fair(fair_rows):
        values = fair_mean[label]
        lines.append(f"| {label} | {100*sum(values)/len(values):.2f}% | `valid_same_protocol` |")
    lines += [
        "",
        "## 解释",
        "",
        "- 当前最强自有对象是 `S2C-Trainable-K1`，不是历史 `fulltex.tex` Cascade。",
        "- MOGB-Fair 是冻结 MiniLM 组件对照；MOGB 官方 BERT 单格仍是 `not_reproduced_strict`。",
        "- ADB 和 DCLOOS reduced 只作外部合同参照，不能与 fair Gate 行直接排名。",
        "- 需要继续完成的是可审计外部单格，而不是重复 E2/E3 或继续扩大固定 K。",
        "",
        "## 产物",
        "",
        "- `results/analysis/archive/analysis/experiment_decision_dashboard_v1/decision_rows.csv`",
        "- `figures/archive/analysis/experiment_decision_dashboard_v1/decision_dashboard.svg`",
        "- `results/analysis/archive/analysis/experiment_decision_dashboard_v1/MANIFEST.json`",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def labels_from_fair(rows: list[dict[str, object]]) -> list[str]:
    preferred = [
        "Trainable K=1", "Frozen K=1", "Random K=2", "Frozen K=2",
        "MOGB partition + s2c boundary", "s2c partition + MOGB boundary", "MOGB-MiniLM",
    ]
    present = {str(row["method_label"]) for row in rows}
    return [label for label in preferred if label in present]


if __name__ == "__main__":
    main()
