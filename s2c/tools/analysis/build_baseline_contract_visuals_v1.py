"""Visualise external-baseline contract differences without false ranking.

This analysis joins the completed five-seed fair MiniLM rows with the existing
single-cell ADB/DA-ADB/BRAK/MOGB compatibility rows.  It deliberately labels
the contracts and keeps DCLOOS (different data and external OOS supervision)
out of the same scatter.  No training or test-based selection is performed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2c-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "baseline_contract_visuals_v1"
FIG = ROOT / "figures" / "baseline_contract_visuals_v1"
REPORT = ROOT / "docs" / "analysis" / "BASELINE_CONTRACT_VISUALS_V1.md"
ADB_EXTERNAL = ROOT / "results" / "analysis" / "external_gpu_runtime_comparison_v1" / "stackoverflow_kir50_external_summary.csv"

METHOD_ORDER = [
    "Trainable K=1",
    "Single centroid",
    "Fixed K=2",
    "MOGB partition + s2c boundary",
    "MOGB-MiniLM",
    "BRAK",
    "ADB",
    "DA-ADB",
    "MOGB-official strict single-cell",
    "DCLOOS reduced-budget",
]
METHOD_COLORS = {
    "Trainable K=1": "#0072B2",
    "Single centroid": "#666666",
    "Fixed K=2": "#D55E00",
    "MOGB partition + s2c boundary": "#009E73",
    "MOGB-MiniLM": "#E69F00",
    "BRAK": "#CC79A7",
    "ADB": "#117733",
    "DA-ADB": "#882255",
    "MOGB-official strict single-cell": "#AA4499",
    "DCLOOS reduced-budget": "#999999",
}
SCOPE_COLORS = {"same_protocol_fair": "#0072B2", "compatibility_single_cell": "#D55E00", "official_or_other_contract": "#777777"}
SCOPE_MARKERS = {"same_protocol_fair": "o", "compatibility_single_cell": "^", "official_or_other_contract": "X"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def atomic_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, format=path.suffix.lstrip("."), dpi=180, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(path)


def load_points() -> tuple[pd.DataFrame, dict[str, str]]:
    fair_path = ROOT / "results" / "analysis" / "experimental_mechanism_pack_v3" / "method_summary.csv"
    baseline_path = ROOT / "results" / "final_baselines" / "summary.csv"
    fair = pd.read_csv(fair_path)
    baseline = pd.read_csv(baseline_path)
    rows: list[dict[str, object]] = []
    fair_names = {
        "trainable_k1": "Trainable K=1",
        "single_centroid": "Single centroid",
        "fixed_k2": "Fixed K=2",
        "mogb_partition_ours_boundary": "MOGB partition + s2c boundary",
        "mogb_minilm": "MOGB-MiniLM",
    }
    fair_slice = fair[(fair["dataset"] == "stackoverflow") & fair["kir"].astype(float).eq(0.50)]
    for _, item in fair_slice.iterrows():
        name = fair_names.get(str(item["method"]))
        if name is None:
            continue
        rows.append(
            {
                "method": name,
                "dataset": "stackoverflow",
                "kir": 0.50,
                "scope": "same_protocol_fair",
                "training_regime": "Known-only Trainable/ Frozen MiniLM",
                "supervision": "Known-only",
                "oos_f1": float(item["oos_f1_mean"]),
                "f1_all": float(item["f1_all_mean"]),
                "known_macro_f1": np.nan,
                "known_recall": float(item["known_recall_mean"]),
                "accuracy": np.nan,
                "n_seeds": int(item["n_seeds"]),
                "status": "complete_fair_component",
                "source": str(fair_path.relative_to(ROOT)),
                "comparable_oos": True,
            }
        )
    # Prefer the latest three-seed CUDA ADB audit over the historical
    # single-cell compatibility row.  Keep the contracts explicitly separate.
    adb_external = pd.read_csv(ADB_EXTERNAL)
    adb_external = adb_external[adb_external["method"].eq("ADB")]
    if not adb_external.empty:
        item = adb_external.iloc[0]
        rows.append(
            {
                "method": "ADB",
                "dataset": "stackoverflow",
                "kir": 0.50,
                "scope": "compatibility_single_cell",
                "training_regime": "end-to-end BERT/TextOIR",
                "supervision": "Known-only",
                "oos_f1": float(item["oos_f1_mean"]),
                "f1_all": float(item["f1_all_mean"]),
                "known_macro_f1": np.nan,
                "known_recall": float(item["known_recall_mean"]),
                "accuracy": float(item["accuracy_mean"]),
                "n_seeds": 3,
                "status": "complete_external_three_seed",
                "source": str(ADB_EXTERNAL.relative_to(ROOT)),
                "comparable_oos": False,
            }
        )

    wanted = {"DA-ADB", "BRAK", "MOGB-official (strict single-cell)"}
    external = baseline[baseline["method"].isin(wanted)].copy()
    for _, item in external.iterrows():
        if item["dataset"] != "stackoverflow" or float(item["kir"]) != 0.50:
            continue
        rows.append(
            {
                "method": str(item["method"]),
                "dataset": str(item["dataset"]),
                "kir": float(item["kir"]),
                "scope": "compatibility_single_cell" if item["method"] in {"ADB", "DA-ADB"} else "official_or_other_contract",
                "training_regime": str(item["training_regime"]),
                "supervision": str(item["supervision"]),
                "oos_f1": float(item["oos_f1"]) if pd.notna(item["oos_f1"]) else np.nan,
                "f1_all": float(item["f1_all"]) if pd.notna(item["f1_all"]) else np.nan,
                "known_macro_f1": float(item["known_macro_f1"]) if pd.notna(item["known_macro_f1"]) else np.nan,
                "known_recall": float(item["known_recall"]) if pd.notna(item["known_recall"]) else np.nan,
                "accuracy": float(item["accuracy"]) if pd.notna(item["accuracy"]) else np.nan,
                "n_seeds": 1,
                "status": str(item["status"]),
                "source": str(item["source"]),
                # 兼容性单格可作为带合同标签的参照，但不能进入当前
                # protocol_v2 的同协议主排名或显著性检验。
                "comparable_oos": False,
            }
        )
    # Keep DCLOOS visible as a separate contract row, not in StackOverflow scatter.
    dcloos = baseline[baseline["method"].eq("DCLOOS-official (reduced-budget recovered)")]
    for _, item in dcloos.iterrows():
        rows.append(
            {
                "method": "DCLOOS reduced-budget",
                "dataset": str(item["dataset"]),
                "kir": float(item["kir"]),
                "scope": "official_or_other_contract",
                "training_regime": str(item["training_regime"]),
                "supervision": str(item["supervision"]),
                "oos_f1": float(item["oos_f1"]) if pd.notna(item["oos_f1"]) else np.nan,
                "f1_all": float(item["f1_all"]) if pd.notna(item["f1_all"]) else np.nan,
                "known_macro_f1": float(item["known_macro_f1"]) if pd.notna(item["known_macro_f1"]) else np.nan,
                "known_recall": float(item["known_recall"]) if pd.notna(item["known_recall"]) else np.nan,
                "accuracy": float(item["accuracy"]) if pd.notna(item["accuracy"]) else np.nan,
                "n_seeds": 1,
                "status": str(item["status"]),
                "source": str(item["source"]),
                "comparable_oos": False,
            }
        )
    points = pd.DataFrame(rows)
    if points.empty:
        raise ValueError("no baseline rows were loaded")
    return points, {
        "fair_summary": sha256(fair_path),
        "baseline_summary": sha256(baseline_path),
        "adb_external_summary": sha256(ADB_EXTERNAL),
    }


def plot_pareto(points: pd.DataFrame, path: Path) -> None:
    frame = points[(points["dataset"] == "stackoverflow") & points["oos_f1"].notna() & points["f1_all"].notna()].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in frame.iterrows():
        method = row["method"]
        ax.scatter(row["f1_all"] * 100, row["oos_f1"] * 100, s=100, color=METHOD_COLORS.get(method, "#555555"), marker=SCOPE_MARKERS.get(row["scope"], "o"), edgecolor="white", linewidth=0.6)
        ax.annotate(method, (row["f1_all"] * 100, row["oos_f1"] * 100), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("F1-All (%)")
    ax.set_ylabel("OOS F1 (%)")
    ax.set_title("StackOverflow/KIR=0.50：当前方法与外部基线的工作点（合同分层）")
    ax.grid(alpha=0.2)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", label="同协议 fair component", markersize=8),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#D55E00", label="兼容性单格", markersize=8),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#777777", label="官方/其他合同", markersize=8),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_contract_table(points: pd.DataFrame, path: Path) -> None:
    frame = points.copy()
    frame["display_scope"] = frame["scope"].map({"same_protocol_fair": "same protocol", "compatibility_single_cell": "compatibility single-cell", "official_or_other_contract": "official/other"})
    frame = frame.sort_values(["dataset", "scope", "method"])
    columns = ["method", "dataset", "kir", "display_scope", "training_regime", "supervision", "n_seeds", "comparable_oos"]
    cell_text = []
    for _, row in frame[columns].iterrows():
        cell_text.append([str(row[col]) if col not in {"kir"} else f"{row[col]:.2f}" for col in columns])
    fig, ax = plt.subplots(figsize=(16, max(4, 0.42 * len(cell_text) + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)
    ax.set_title("基线合同矩阵：可比性必须先于数字排名", pad=15)
    fig.tight_layout()
    atomic_figure(fig, path)


def build_report(points: pd.DataFrame) -> str:
    stack = points[(points["dataset"] == "stackoverflow") & points["oos_f1"].notna()].copy()
    lines = [
        "# 外部基线合同与可比性可视化 V1",
        "",
        "## 结论",
        "",
        "本报告把同协议 Frozen/Trainable 组件、ADB/DA-ADB 兼容性单格、MOGB 官方严格单格和 DCLOOS reduced-budget 结果放在一个带合同标签的证据层中。它不把不同监督、表示、数据或 seed 合同混成 SOTA 排名。",
        "",
        "在 StackOverflow/KIR=0.50 的已有数字中，最新三 seed ADB 外部均值高于当前 Trainable K=1；但它是端到端 BERT/TextOIR 合同，不是当前 protocol_v2 的五 seed MiniLM fair matrix，因此这里只能记录外部参照工作点，不能称为公平优胜结论。",
        "",
        "## StackOverflow/KIR=0.50 可见结果",
        "",
        "| 方法 | OOS F1 | F1-All | Known macro-F1 | seed 数 | 合同 | 是否进入当前同协议主排名 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for _, row in stack.sort_values("oos_f1", ascending=False).iterrows():
        scope = {"same_protocol_fair": "same protocol", "compatibility_single_cell": "compatibility single-cell", "official_or_other_contract": "official/other"}[row["scope"]]
        lines.append(f"| {row['method']} | {row['oos_f1'] * 100:.2f} | {row['f1_all'] * 100:.2f} | {row['known_macro_f1'] * 100:.2f} | {int(row['n_seeds'])} | {scope} | {'是' if row['comparable_oos'] else '否'} |" )
    lines.extend(
        [
            "",
            "## 监督与表示差异",
            "",
            "- 当前 Trainable K=1：Known-only MiniLM 适配，当前协议，五 seed fair summary。",
            "- MOGB-MiniLM 与 MOGB partition 组件：冻结 MiniLM，Known-only，五 seed fair component。",
            "- ADB：最新三 seed 的端到端 BERT/TextOIR 兼容性参照；DA-ADB 仍为旧兼容性单格/invalid 记录，不能与当前 MiniLM 五 seed 结果直接做显著性结论。",
            "- MOGB official strict：官方 BERT 逻辑单格，当前 OOS F1 字段与 Gate 表不完全同构，不能强行补值。",
            "- DCLOOS reduced-budget：使用 pseudo-OOS/外部 OOS，且当前记录为不同数据/不同 KIR 合同，不进入 StackOverflow fair scatter。",
            "",
            "## 图表",
            "",
            "- `figures/baseline_contract_visuals_v1/stackoverflow_baseline_pareto_contract.png`：性能工作点与合同形状。",
            "- `figures/baseline_contract_visuals_v1/baseline_contract_table.png`：方法、监督、表示和可比性矩阵。",
            "",
            "## 研究含义",
            "",
            "当前证据说明：Trainable K=1 已经是当前自有 Gate 的稳定工作点，但还没有同合同证据证明它超过端到端 ADB/DA-ADB。下一步应优先补齐同协议的端到端基线或明确将其保留为兼容性参照，而不是继续用外部数字宣称 SOTA。",
            "",
            "所有结果来自已完成的 summary 文件；没有训练、调参或覆盖历史 artifact。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    points, hashes = load_points()
    atomic_csv(points, OUT / "contract_matrix.csv")
    plot_pareto(points, FIG / "stackoverflow_baseline_pareto_contract.png")
    plot_contract_table(points, FIG / "baseline_contract_table.png")
    manifest = {
        "analysis": "baseline_contract_visuals_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "analysis_only": True,
        "selection_used_test_oos": False,
        "external_rows_isolated": True,
        "inputs": hashes,
        "output": str(OUT / "contract_matrix.csv"),
        "figures": [str(FIG / "stackoverflow_baseline_pareto_contract.png"), str(FIG / "baseline_contract_table.png")],
    }
    atomic_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", OUT / "MANIFEST.json")
    atomic_text(build_report(points), REPORT)
    print(json.dumps({"status": "ok", "rows": len(points), "same_dataset_rows": int((points["dataset"] == "stackoverflow").sum()), "figures": 2}, ensure_ascii=False))


if __name__ == "__main__":
    main()
