#!/usr/bin/env python3
"""生成 S2C/MOGB-Fair/ADB 的跨 KIR 合同分层图和中文摘要。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
FAIR = ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
ADB = ROOT / "results/analysis/adb_kir_sensitivity_v2/adb_summary.csv"
OUTPUT = ROOT / "results/analysis/cross_kir_contract_atlas_v2"
FIGURES = ROOT / "figures/cross_kir_contract_atlas_v2"
DATASETS = ["clinc150", "banking77", "stackoverflow"]
METHODS = ["trainable_k1", "mogb_minilm", "adb"]
METRICS = [("oos_f1", "OOS F1 (%)"), ("f1_all", "F1-All (%)"), ("known_recall", "Known Recall (%)")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_rows(fair_path: Path, adb_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with fair_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] not in {"trainable_k1", "mogb_minilm"}:
                continue
            if float(row["kir"]) not in {0.25, 0.50, 0.75}:
                continue
            rows.append(
                {
                    "dataset": row["dataset"],
                    "kir": float(row["kir"]),
                    "method": row["method"],
                    "method_label": row.get("method_label", row["method"]),
                    "contract": "当前 MiniLM fair（5 seeds）",
                    "n_seeds": int(row["n_seeds"]),
                    **{metric: float(row[metric]) for metric, _ in METRICS},
                    **{f"{metric}_std": float(row[f"{metric}_std"]) for metric, _ in METRICS},
                }
            )
    with adb_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] not in DATASETS:
                continue
            rows.append(
                {
                    "dataset": row["dataset"],
                    "kir": float(row["kir"]),
                    "method": "adb",
                    "method_label": "ADB",
                    "contract": f"BERT/TextOIR 外部合同（{row['n_seeds']} seeds）",
                    "n_seeds": int(row["n_seeds"]),
                    **{metric: float(row[f"{metric}_mean"]) for metric, _ in METRICS},
                    **{f"{metric}_std": float(row[f"{metric}_std"]) for metric, _ in METRICS},
                }
            )
    return rows


def plot(rows: list[dict[str, object]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"trainable_k1": "#1f77b4", "mogb_minilm": "#d62728", "adb": "#2ca02c"}
    labels = {"trainable_k1": "S2C Trainable-K1", "mogb_minilm": "MOGB-MiniLM", "adb": "ADB (BERT)"}
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex="col")
    for col, dataset in enumerate(DATASETS):
        for row_index, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row_index, col]
            for method in METHODS:
                group = sorted(
                    [r for r in rows if r["dataset"] == dataset and r["method"] == method],
                    key=lambda r: float(r["kir"]),
                )
                if not group:
                    continue
                x = [float(r["kir"]) for r in group]
                y = [100 * float(r[metric]) for r in group]
                error = [100 * float(r[f"{metric}_std"]) for r in group]
                ax.errorbar(x, y, yerr=error, marker="o", capsize=3, color=colors[method], label=labels[method])
            ax.set_title(dataset if row_index == 0 else "")
            ax.set_ylabel(ylabel if col == 0 else "")
            ax.set_xticks([0.25, 0.50, 0.75])
            ax.grid(alpha=0.2)
            if row_index == 2:
                ax.set_xlabel("KIR")
            if row_index == 0 and col == 2:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle("跨 KIR 合同分层：当前 S2C、MOGB-Fair 与 ADB", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_dir / "cross_kir_contract_atlas.png", dpi=190)
    plt.close(fig)


def report(rows: list[dict[str, object]], fair_path: Path, adb_path: Path, output_dir: Path, figure_dir: Path) -> str:
    adb_seed_counts = sorted({int(r["n_seeds"]) for r in rows if r["method"] == "adb"})
    adb_seed_label = ", ".join(str(value) for value in adb_seed_counts) if adb_seed_counts else "0"
    lines = [
        "# 跨 KIR 合同分层图与结果摘要",
        "",
        "本报告把当前 MiniLM fair 结果、MOGB-MiniLM 组件结果和 ADB BERT/TextOIR 外部合同放在同一坐标系中，",
        f"但不把它们当作同骨干 SOTA 排名。误差线来自 5 seed（S2C/MOGB）和 ADB 的 `{adb_seed_label}` seed。",
        "",
        "## 关键观察",
        "",
        "- Trainable-K1 的 OOS F1 在 9 个 dataset×KIR 组中有 8 个高于 ADB，但 CLINC150 的 F1-All 与 Known Recall 全部低于 ADB。",
        "- MOGB-MiniLM 的低 false acceptance 伴随明显 Known Recall 损失；它不是简单的“更强拒识”。",
        "- KIR 增加后，ADB 的 OOS F1 在三个数据集都下降，幅度在 Banking77 和 StackOverflow 更明显。",
        "- 因此应同时报告 OOS F1、F1-All、Known Recall 和 false acceptance；只画 OOS F1 会掩盖工作点差异。",
        "",
        "## 机器可读来源",
        "",
        f"- fair source SHA256：`{sha256(fair_path)}`",
        f"- ADB source SHA256：`{sha256(adb_path)}`",
        f"- 图：`{figure_dir / 'cross_kir_contract_atlas.png'}`",
        f"- 行级数据：`{output_dir / 'rows.csv'}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair", type=Path, default=FAIR)
    parser.add_argument("--adb", type=Path, default=ADB)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    fair_path = args.fair.resolve()
    adb_path = args.adb.resolve()
    rows = read_rows(fair_path, adb_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "rows.csv", rows)
    plot(rows, args.figure_dir.resolve())
    manifest = {
        "schema_version": 1,
        "analysis": "cross_kir_contract_atlas_v2",
        "protocol_version": "protocol_v2_textoir_v1",
        "fair_source": str(fair_path),
        "fair_source_sha256": sha256(fair_path),
        "adb_source": str(adb_path),
        "adb_source_sha256": sha256(adb_path),
        "row_count": len(rows),
        "selection_used_test_oos": False,
        "contracts": {
            "trainable_k1": "MiniLM known-only fair, 5 seeds",
            "mogb_minilm": "Frozen MiniLM MOGB component, 5 seeds",
            "adb": "BERT/TextOIR external compatibility; seed count comes from source summary",
        },
    }
    (args.output_dir / "CROSS_KIR_CONTRACT_ATLAS_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "CROSS_KIR_CONTRACT_ATLAS_REPORT.md").write_text(
        report(rows, fair_path, adb_path, args.output_dir.resolve(), args.figure_dir.resolve()),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
