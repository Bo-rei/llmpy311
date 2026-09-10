#!/usr/bin/env python3
"""Build a contract-aware Chinese comparison atlas for the S2C experiments.

The atlas deliberately keeps four evidence layers separate:

1. historical ``fulltex.tex`` Cascade results;
2. the current protocol_v2 fair Gate matrix;
3. MOGB frozen-MiniLM/component results;
4. legacy or external compatibility references.

It only reads frozen CSV artifacts and never trains, tunes, or rewrites an
existing result.  Every derived table records its source layer and whether a
direct ranking is valid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


_CJK_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
if _CJK_FONT.is_file():
    font_manager.fontManager.addfont(str(_CJK_FONT))
    _CJK_FAMILY = font_manager.FontProperties(fname=str(_CJK_FONT)).get_name()
else:
    _CJK_FAMILY = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = [_CJK_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
FAIR_SUMMARY = ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
FAIR_PER_SEED = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
HISTORICAL = ROOT / "results/analysis/archive/analysis/historical_sota_comparison_v1/fulltex_main_results.csv"
HISTORICAL_MARGIN = ROOT / "results/analysis/archive/analysis/historical_sota_comparison_v1/ours_minus_best_baseline.csv"
CONTRACT_MATRIX = ROOT / "results/analysis/archive/analysis/baseline_contract_visuals_v1/contract_matrix.csv"
MOGB_EXACT = ROOT / "results/analysis/archive/analysis/mogb_reproduction_gap_analysis_v2/exact_run_summary.csv"


METHOD_ORDER = [
    "trainable_k1",
    "single_centroid",
    "fixed_k2",
    "random_partition",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
    "mogb_minilm",
]
METHOD_LABELS = {
    "trainable_k1": "S2C Trainable K=1",
    "single_centroid": "Frozen K=1",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_partition_ours_boundary": "MOGB partition + S2C boundary",
    "ours_partition_mogb_boundary": "S2C partition + MOGB boundary",
    "mogb_minilm": "MOGB-MiniLM",
}
DATASET_LABELS = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}
METRICS = [
    "oos_f1",
    "f1_all",
    "f1_k",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
    "accuracy",
    "auroc",
    "aupr_oos",
]


def _check(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return frame


def _pp(series: pd.Series) -> pd.Series:
    return series.astype(float) * 100.0


def _heatmap(frame: pd.DataFrame, value: str, path: Path, title: str, fmt: str = ".1f") -> None:
    columns = [f"{DATASET_LABELS[d]}\nKIR={k:g}" for d in ("clinc150", "banking77", "stackoverflow") for k in (0.25, 0.5, 0.75)]
    keys = [(d, k) for d in ("clinc150", "banking77", "stackoverflow") for k in (0.25, 0.5, 0.75)]
    methods = [m for m in METHOD_ORDER if m in set(frame["method"])]
    matrix = np.full((len(methods), len(keys)), np.nan)
    for i, method in enumerate(methods):
        for j, (dataset, kir) in enumerate(keys):
            row = frame[(frame.method == method) & (frame.dataset == dataset) & (frame.kir == kir)]
            if not row.empty:
                matrix[i, j] = float(row.iloc[0][value]) * 100.0
    fig, ax = plt.subplots(figsize=(14, max(3.8, 0.55 * len(methods) + 1.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    ax.set_yticks(range(len(methods)), [METHOD_LABELS.get(m, m) for m in methods])
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label="百分比")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def build(output_dir: Path, figure_dir: Path, report_path: Path) -> dict:
    fair = _check(FAIR_SUMMARY, ["dataset", "kir", "method", *METRICS, "n_seeds"])
    fair_seed = _check(FAIR_PER_SEED, ["dataset", "kir", "method", *METRICS, "seed"])
    historical = _check(HISTORICAL, ["protocol", "dataset", "kir", "method", "known_f1", "oos_f1", "accuracy"])
    historical_margin = _check(HISTORICAL_MARGIN, ["dataset", "kir", "ours_oos_f1", "best_baseline_oos_f1", "margin_pp"])
    contract = _check(CONTRACT_MATRIX, ["method", "dataset", "kir", "scope", "training_regime", "supervision", "status", "source", "comparable_oos"])
    mogb_exact = _check(MOGB_EXACT, ["dataset", "kir", "accuracy", "f1_all", "f1_u", "f1_k", "known_recall", "oos_precision", "oos_recall", "contract"])

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fair_summary = (
        fair.groupby("method", as_index=False)[METRICS]
        .mean(numeric_only=True)
        .assign(method_label=lambda x: x.method.map(METHOD_LABELS).fillna(x.method))
    )
    fair_summary["n_cells"] = fair.groupby("method").size().reindex(fair_summary.method).to_numpy()
    fair_summary.to_csv(output_dir / "fair_method_summary.csv", index=False)

    baseline_rows = []
    for baseline in [m for m in METHOD_ORDER if m != "trainable_k1"]:
        left = fair[fair.method == "trainable_k1"].set_index(["dataset", "kir"])
        right = fair[fair.method == baseline].set_index(["dataset", "kir"])
        common = left.index.intersection(right.index)
        for dataset, kir in common:
            row = {"dataset": dataset, "kir": kir, "baseline": baseline, "baseline_label": METHOD_LABELS.get(baseline, baseline)}
            for metric in METRICS:
                row[f"delta_{metric}"] = float(left.loc[(dataset, kir), metric] - right.loc[(dataset, kir), metric]) * 100.0
            baseline_rows.append(row)
    effects = pd.DataFrame(baseline_rows)
    effects.to_csv(output_dir / "trainable_vs_all_fair_effects.csv", index=False)
    effects.groupby("baseline", as_index=False).mean(numeric_only=True).to_csv(output_dir / "trainable_vs_all_fair_effects_mean.csv", index=False)

    # Explicit decomposition of the three MOGB/S2C component rows.
    comp_pairs = [
        ("mogb_minilm", "single_centroid", "MOGB-MiniLM minus Frozen K=1"),
        ("mogb_partition_ours_boundary", "mogb_minilm", "MOGB partition + S2C boundary minus MOGB-MiniLM"),
        ("ours_partition_mogb_boundary", "mogb_partition_ours_boundary", "S2C partition + MOGB boundary minus MOGB partition + S2C boundary"),
    ]
    component_rows = []
    for child, parent, label in comp_pairs:
        c = fair[fair.method == child].set_index(["dataset", "kir"])
        p = fair[fair.method == parent].set_index(["dataset", "kir"])
        for key in c.index.intersection(p.index):
            row = {"dataset": key[0], "kir": key[1], "child": child, "parent": parent, "comparison": label}
            for metric in METRICS:
                row[f"delta_{metric}"] = float(c.loc[key, metric] - p.loc[key, metric]) * 100.0
            component_rows.append(row)
    components = pd.DataFrame(component_rows)
    components.to_csv(output_dir / "component_decomposition.csv", index=False)

    # Long contract-aware matrix.  The status and comparability fields are
    # intentionally explicit so downstream plots cannot silently rank layers.
    fair_long = fair[["dataset", "kir", "method", *METRICS, "n_seeds"]].copy()
    fair_long["layer"] = "current_protocol_v2_fair_gate"
    fair_long["method_label"] = fair_long.method.map(METHOD_LABELS).fillna(fair_long.method)
    fair_long["status"] = "complete_fair_component"
    fair_long["directly_comparable"] = True
    fair_long["source"] = str(FAIR_SUMMARY.relative_to(ROOT))
    fair_long = fair_long.rename(columns={m: f"{m}_mean" for m in METRICS})

    historical_long = historical.copy()
    historical_long["layer"] = "historical_fulltex_cascade"
    historical_long["method_label"] = historical_long.method
    historical_long["status"] = "historical_contract"
    historical_long["directly_comparable"] = False
    historical_long["source"] = str(HISTORICAL.relative_to(ROOT))
    historical_long["n_seeds"] = np.nan
    historical_long = historical_long.rename(columns={"known_f1": "f1_k_mean", "oos_f1": "oos_f1_mean", "accuracy": "accuracy_mean"})
    for metric in METRICS:
        if f"{metric}_mean" not in historical_long:
            historical_long[f"{metric}_mean"] = np.nan

    comparison = pd.concat(
        [
            fair_long[["dataset", "kir", "method_label", "layer", "status", "directly_comparable", "source", "n_seeds", *[f"{m}_mean" for m in METRICS]]],
            historical_long[["dataset", "kir", "method_label", "layer", "status", "directly_comparable", "source", "n_seeds", *[f"{m}_mean" for m in METRICS]]],
        ],
        ignore_index=True,
    )
    comparison.to_csv(output_dir / "contract_aware_comparison_matrix.csv", index=False)

    contract.to_csv(output_dir / "external_contract_reference.csv", index=False)
    mogb_exact.to_csv(output_dir / "mogb_exact_local_reference.csv", index=False)
    historical_margin.to_csv(output_dir / "historical_ours_margin.csv", index=False)

    _heatmap(fair, "oos_f1", figure_dir / "fair_oos_f1_heatmap.png", "当前 protocol_v2 公平 Gate：OOS F1")
    _heatmap(fair, "f1_all", figure_dir / "fair_f1_all_heatmap.png", "当前 protocol_v2 公平 Gate：F1-All")

    # A compact dataset x KIR Trainable-minus-Frozen heatmap.
    t = effects[effects.baseline == "single_centroid"].copy()
    mat = t.pivot(index="dataset", columns="kir", values="delta_oos_f1").reindex(["clinc150", "banking77", "stackoverflow"])
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    im = ax.imshow(mat.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(mat.columns)), [f"KIR={x:g}" for x in mat.columns])
    ax.set_yticks(range(len(mat.index)), [DATASET_LABELS[x] for x in mat.index])
    ax.set_title("Trainable K=1 相对 Frozen K=1 的 OOS F1 提升")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="百分点")
    _savefig(figure_dir / "trainable_minus_frozen_delta_heatmap.png")

    # Pareto points across all current cells: this makes the coverage tradeoff
    # visible instead of hiding it behind one averaged ranking.
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for method in METHOD_ORDER:
        rows = fair[fair.method == method]
        if rows.empty:
            continue
        ax.scatter(_pp(rows.known_recall), _pp(rows.oos_f1), label=METHOD_LABELS[method], s=34, alpha=0.75)
    ax.set_xlabel("Known Recall (%)")
    ax.set_ylabel("OOS F1 (%)")
    ax.set_title("当前公平协议：OOS F1 与 Known Recall 工作点")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    _savefig(figure_dir / "fair_pareto_oos_known_recall.png")

    # K=1 versus K=2 risk in the current fixed partition, by dataset/KIR.
    k1 = fair[fair.method == "single_centroid"].set_index(["dataset", "kir"])
    k2 = fair[fair.method == "fixed_k2"].set_index(["dataset", "kir"])
    keys = k1.index.intersection(k2.index)
    risk = pd.DataFrame(
        [
            {
                "dataset": d,
                "kir": k,
                "delta_oos_f1": (k2.loc[(d, k), "oos_f1"] - k1.loc[(d, k), "oos_f1"]) * 100,
                "delta_false_accept": (k2.loc[(d, k), "false_accept_rate"] - k1.loc[(d, k), "false_accept_rate"]) * 100,
                "delta_known_recall": (k2.loc[(d, k), "known_recall"] - k1.loc[(d, k), "known_recall"]) * 100,
            }
            for d, k in keys
        ]
    )
    risk.to_csv(output_dir / "fixed_k2_risk_attribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(risk))
    width = 0.26
    ax.bar(x - width, risk.delta_oos_f1, width, label="Δ OOS F1")
    ax.bar(x, risk.delta_false_accept, width, label="Δ false acceptance")
    ax.bar(x + width, risk.delta_known_recall, width, label="Δ Known Recall")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [f"{DATASET_LABELS[d]}\n{float(k):g}" for d, k in zip(risk.dataset, risk.kir)], rotation=0)
    ax.set_ylabel("K=2 - K=1（百分点）")
    ax.set_title("固定双中心的收益与风险归因")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    _savefig(figure_dir / "fixed_k2_risk_attribution.png")

    # MOGB component operating points.
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for method, marker in [("single_centroid", "o"), ("mogb_minilm", "s"), ("mogb_partition_ours_boundary", "^"), ("ours_partition_mogb_boundary", "D"), ("trainable_k1", "*")]:
        rows = fair[fair.method == method]
        if rows.empty:
            continue
        ax.scatter(_pp(rows.false_accept_rate), _pp(rows.f1_all), label=METHOD_LABELS[method], marker=marker, s=42 if marker != "*" else 90, alpha=0.8)
    ax.set_xlabel("False acceptance (%)")
    ax.set_ylabel("F1-All (%)")
    ax.set_title("MOGB/S2C 组件边界：拒识保守性与完整分类权衡")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    _savefig(figure_dir / "mogb_component_tradeoff.png")

    # Seed stability, aggregated over cells, with raw points retained.
    seed_agg = fair_seed.groupby(["method", "seed"], as_index=False)[["oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate"]].mean()
    seed_agg.to_csv(output_dir / "seed_stability_aggregate.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    data, labels = [], []
    for method in METHOD_ORDER:
        rows = seed_agg[seed_agg.method == method]
        if rows.empty:
            continue
        data.append(_pp(rows.oos_f1).to_numpy())
        labels.append(METHOD_LABELS[method])
    ax.boxplot(data, tick_labels=labels, orientation="vertical", showmeans=True)
    ax.set_ylabel("跨数据集/KIR平均 OOS F1 (%)")
    ax.set_title("五个 data seed 的 OOS F1 稳定性")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    _savefig(figure_dir / "seed_stability_oos_f1.png")

    fair_mean = fair_summary.set_index("method")
    def val(method: str, metric: str) -> float:
        return float(fair_mean.loc[method, metric]) * 100.0
    report_lines = [
        "# 当前实验对比图谱 V1",
        "",
        "> 本报告只做实验整理、对比和机制分析，不提出新方法，也不把不同数据合同的数字混合排名。",
        "",
        "## 1. 当前到底比较哪些方法",
        "",
        "当前主分析对象是 `protocol_v2_textoir_v1` 下的 3 数据集 × 3 KIR × 7 方法 × 5 seeds。",
        "历史 `fulltex.tex` 的 `Ours` 是完整 Gate–Router–Expert Cascade；当前 `S2C Trainable K=1` 是 Known-only 训练后的 Gate-only 方法，两者不能直接合并成一个排名。",
        "MOGB 公平行是冻结 MiniLM 组件；MOGB BERT 单格属于现代兼容复现，外部 ADB/DA-ADB/DCLOOS 属于兼容或不同监督合同。",
        "",
        "## 2. 当前公平矩阵的总体结果",
        "",
        f"- S2C Trainable K=1：平均 OOS F1 **{val('trainable_k1','oos_f1'):.2f}%**，F1-All **{val('trainable_k1','f1_all'):.2f}%**。",
        f"- Frozen K=1：平均 OOS F1 **{val('single_centroid','oos_f1'):.2f}%**，F1-All **{val('single_centroid','f1_all'):.2f}%**。",
        f"- MOGB-MiniLM：平均 OOS F1 **{val('mogb_minilm','oos_f1'):.2f}%**，F1-All **{val('mogb_minilm','f1_all'):.2f}%**，Known Recall **{val('mogb_minilm','known_recall'):.2f}%**。",
        "- Trainable K=1 在当前公平矩阵的 OOS F1 9 个设置中 8 个第一，F1-All 9 个设置全部第一；这只是当前自有/组件矩阵结论，不是对完整官方基线的 SOTA 宣称。",
        "",
        "## 3. 为什么当前自有方法在公平矩阵中更好",
        "",
        "1. **表示层**：Trainable K=1 改善了单中心的 Known/OOS 分数分离；相对 Frozen K=1，结果表中的 delta 热图直接显示每个数据集和 KIR 的提升。",
        "2. **决策层**：它使用单中心边界，避免固定多个球的接受区域并集快速扩大；因此相比 Frozen K=2，通常能减少 OOS 误接受，同时只付出有限 Known Recall。",
        "3. **工作点层**：MOGB-MiniLM 的 false acceptance 很低，但 false rejection 极高；它更像保守拒识器，而不是完整分类性能更好的替代方案。",
        "",
        "## 4. 固定多中心为什么受限",
        "",
        "固定 K=2 的风险图把 OOS F1、false acceptance 和 Known Recall 的变化放在同一张图中。StackOverflow 的主要问题是新增中心扩大了接受区域：false acceptance 上升明显，而 OOS F1 下降；这不是简单的中心数不足。",
        "",
        "## 5. MOGB 组件归因",
        "",
        "- MOGB-MiniLM → MOGB partition + S2C boundary：隔离粒球划分与边界规则的变化。",
        "- MOGB partition + S2C boundary → S2C partition + MOGB boundary：隔离 MOGB 平均半径/欧氏边界的影响。",
        "- 这些组件行可以说明当前差距来自 Known 覆盖、半径和接受区域工作点，但不能替代 MOGB 官方 BERT 端到端复现。",
        "",
        "## 6. 历史论文结果和外部基线边界",
        "",
        "历史 `fulltex.tex` 的完整 Cascade 在旧协议下 9/9 个 dataset×KIR 单元的 OOS F1 高于表内基线；这证明旧论文主结果强，但不能直接证明当前 Gate-only Trainable K=1 已超过历史 Cascade。",
        "ADB、DA-ADB 和 DCLOOS 的旧兼容数字只作参考，不进入当前公平主排名。StackOverflow/KIR=.50 的 ADB seed=42/87/100 已形成同数据合同的 BERT/TextOIR 外部单格参照；DA-ADB 兼容运行出现 NaN/全类预测并被标记无效。",
        "",
        "## 7. 本轮图和机器可读结果",
        "",
        "- `figures/archive/analysis/comparison_atlas_v1/fair_oos_f1_heatmap.png`：当前公平 OOS F1 热图。",
        "- `figures/archive/analysis/comparison_atlas_v1/fair_f1_all_heatmap.png`：当前公平 F1-All 热图。",
        "- `figures/archive/analysis/comparison_atlas_v1/trainable_minus_frozen_delta_heatmap.png`：Trainable 相对 Frozen 的提升。",
        "- `figures/archive/analysis/comparison_atlas_v1/fixed_k2_risk_attribution.png`：固定双中心风险归因。",
        "- `figures/archive/analysis/comparison_atlas_v1/mogb_component_tradeoff.png`：MOGB/S2C 组件工作点。",
        "- `figures/archive/analysis/comparison_atlas_v1/fair_pareto_oos_known_recall.png`：OOS F1–Known Recall 权衡。",
        "- `figures/archive/analysis/comparison_atlas_v1/seed_stability_oos_f1.png`：五 seed 稳定性。",
        "- `docs/analysis/STACKOVERFLOW_EXTERNAL_COMPARISON_V1.md`：ADB 三 seed 与 DA-ADB 逐样本有效性审计。",
        "- `figures/archive/analysis/comparison_atlas_v1/stackoverflow_external_single_cell_comparison.png`：同协议单格外部参照。",
        "",
        "## 8. 结论边界和下一步",
        "",
        "当前最可靠结论是：Trainable K=1 是当前统一 Known-only Gate 矩阵中最平衡的自有方法；固定 K>1 在 StackOverflow 有明显 boundary-union 风险；MOGB-MiniLM 的低误接受伴随严重 Known 拒绝。",
        "仍缺少同一监督/数据合同下的 DA-ADB、DCLOOS 多 seed 主表，以及作者数据合同闭合后的官方 MOGB 复现。因此下一步应优先修复 DA-ADB 的数值稳定性并审计 DCLOOS 的外部 OOS 合同，而不是继续堆 K 或新增损失。",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "tool": "build_comparison_atlas_v1.py",
        "sources": {
            "fair_summary": str(FAIR_SUMMARY.relative_to(ROOT)),
            "fair_per_seed": str(FAIR_PER_SEED.relative_to(ROOT)),
            "historical": str(HISTORICAL.relative_to(ROOT)),
            "historical_margin": str(HISTORICAL_MARGIN.relative_to(ROOT)),
            "contract_matrix": str(CONTRACT_MATRIX.relative_to(ROOT)),
            "mogb_exact": str(MOGB_EXACT.relative_to(ROOT)),
        },
        "fair_rows": int(len(fair)),
        "fair_per_seed_rows": int(len(fair_seed)),
        "historical_rows": int(len(historical)),
        "contract_rows": int(len(contract)),
        "mogb_exact_rows": int(len(mogb_exact)),
        "output_tables": sorted(p.name for p in output_dir.glob("*.csv")),
        "output_figures": sorted(p.name for p in figure_dir.glob("*.png")),
        "direct_ranking_layers": ["current_protocol_v2_fair_gate"],
        "non_comparable_layers": ["historical_fulltex_cascade", "legacy_external_compatibility", "mogb_official_modern_compatibility"],
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/analysis/archive/analysis/comparison_atlas_v1")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures/archive/analysis/comparison_atlas_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/archive/analysis/COMPARISON_ATLAS_V1.md")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(args.output_dir, args.figure_dir, args.report), ensure_ascii=False, indent=2))
