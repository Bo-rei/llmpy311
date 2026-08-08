"""Build a protocol-aware method trade-off and visualization pack.

This is an analysis-only stage.  It consumes completed five-seed rows from the
current protocol and never merges historical fulltex or externally supervised
baselines into the fair table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/analysis/minilm_trainable_5seed_fair_v1/all_methods_per_seed.csv"
OUT = ROOT / "results/analysis/cross_protocol_tradeoff_v1"
FIG = ROOT / "figures/cross_protocol_tradeoff_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEED = 20260725
BOOTSTRAPS = 10_000
METHOD_ORDER = (
    "trainable_k1",
    "single_centroid",
    "fixed_k2",
    "random_partition",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
    "mogb_minilm",
)
METHOD_LABELS = {
    "trainable_k1": "Trainable K=1",
    "single_centroid": "Frozen K=1",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_partition_ours_boundary": "MOGB partition + s2c boundary",
    "ours_partition_mogb_boundary": "s2c partition + MOGB boundary",
    "mogb_minilm": "MOGB-MiniLM",
}
COLORS = {
    "trainable_k1": "#d62728",
    "single_centroid": "#4c78a8",
    "fixed_k2": "#f58518",
    "random_partition": "#72b7b2",
    "mogb_partition_ours_boundary": "#54a24b",
    "ours_partition_mogb_boundary": "#b279a2",
    "mogb_minilm": "#9d755d",
}
METRICS = (
    "oos_f1",
    "f1_all",
    "f1_k",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
    "accuracy",
    "auroc",
    "aupr_oos",
)


def _bootstrap_delta(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_summary(frame: pd.DataFrame, baseline: str, metrics: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for (dataset, kir), group in frame.groupby(["dataset", "kir"], sort=True):
        ours = group[group.method == "trainable_k1"].set_index("seed")
        ref = group[group.method == baseline].set_index("seed")
        common = sorted(set(ours.index) & set(ref.index))
        if not common:
            continue
        for metric in metrics:
            delta = ours.loc[common, metric].to_numpy(dtype=float) - ref.loc[common, metric].to_numpy(dtype=float)
            mean_delta, ci_low, ci_high = _bootstrap_delta(delta, rng)
            rows.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS[baseline],
                    "metric": metric,
                    "n_seeds": len(delta),
                    "mean_delta": mean_delta,
                    "median_delta": float(np.median(delta)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wins": int((delta > 1e-12).sum()),
                    "ties": int((np.abs(delta) <= 1e-12).sum()),
                    "losses": int((delta < -1e-12).sum()),
                }
            )
    return pd.DataFrame(rows)


def _load() -> pd.DataFrame:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    frame = pd.read_csv(INPUT)
    required = {"dataset", "kir", "seed", "method", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame[frame.method.isin(METHOD_ORDER)].copy()
    if set(frame.method.unique()) != set(METHOD_ORDER):
        raise ValueError(f"Unexpected method coverage: {sorted(frame.method.unique())}")
    expected = len(DATASETS) * len(KIRS) * 5 * len(METHOD_ORDER)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(frame)}")
    # The Trainable runner calls this field ``known_recall`` while the frozen
    # baseline/MOGB adapters call the same quantity ``id_recall``.  Normalize
    # the schema before any paired analysis; otherwise baseline coverage would
    # silently appear as NaN.
    if "id_recall" in frame.columns:
        frame["known_recall"] = frame["known_recall"].fillna(frame["id_recall"])
    frame["method_label"] = frame.method.map(METHOD_LABELS)
    frame["scope"] = "protocol_v2_textoir_v1 / Frozen-or-Trainable MiniLM / Known-only"
    return frame.sort_values(["dataset", "kir", "seed", "method"]).reset_index(drop=True)


def _write(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["dataset", "kir", "method", "method_label"], as_index=False)
    mean = grouped[list(METRICS)].mean()
    std = grouped[list(METRICS)].std().rename(columns={metric: f"{metric}_std" for metric in METRICS})
    n = grouped.size().rename(columns={"size": "n_seeds"})
    return mean.merge(std, on=["dataset", "kir", "method", "method_label"]).merge(n, on=["dataset", "kir", "method", "method_label"])


def _plot_pareto(summary: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = summary[summary.dataset == dataset]
        for method in METHOD_ORDER:
            rows = part[part.method == method]
            ax.scatter(
                rows.oos_f1 * 100,
                rows.known_recall * 100,
                color=COLORS[method],
                label=METHOD_LABELS[method],
                s=70,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            for _, row in rows.iterrows():
                ax.text(row.oos_f1 * 100 + 0.15, row.known_recall * 100 + 0.15, f"{row.kir:.2f}", fontsize=7)
        ax.set_title(dataset)
        ax.grid(alpha=0.25)
        ax.set_xlabel("OOS F1 (%)")
    axes[0].set_ylabel("Known Recall (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Fair five-seed operating points: OOS quality versus Known coverage")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "pareto_oos_f1_known_recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_errors(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = summary[(summary.dataset == dataset) & (summary.kir == 0.50)]
        x = np.arange(len(METHOD_ORDER))
        width = 0.35
        ax.bar(x - width / 2, part.set_index("method").loc[list(METHOD_ORDER), "false_accept_rate"] * 100, width, label="False acceptance", color="#e45756")
        ax.bar(x + width / 2, part.set_index("method").loc[list(METHOD_ORDER), "false_reject_rate"] * 100, width, label="False rejection", color="#4c78a8")
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m].replace(" + ", " +\n") for m in METHOD_ORDER], rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Error rate (%)")
    axes[-1].legend(frameon=False)
    fig.suptitle("KIR=0.50 error decomposition: coverage versus rejection")
    fig.tight_layout()
    fig.savefig(FIG / "error_decomposition_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_kir(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), sharex=True)
    for row, dataset in enumerate(DATASETS):
        part = summary[summary.dataset == dataset]
        for col, metric in enumerate(("oos_f1", "f1_all")):
            ax = axes[row, col]
            for method in METHOD_ORDER:
                line = part[part.method == method].sort_values("kir")
                ax.plot(line.kir, line[metric] * 100, marker="o", color=COLORS[method], label=METHOD_LABELS[method])
            ax.set_title(f"{dataset}: {metric}")
            ax.grid(alpha=0.25)
            ax.set_ylabel("score (%)")
            if row == 2:
                ax.set_xlabel("KIR")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("KIR sensitivity under the same Frozen/Trainable MiniLM fair contract")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "kir_curves_oos_f1_f1_all.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_variance(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = frame[(frame.dataset == dataset) & (frame.kir == 0.50)]
        data = [part[part.method == method].oos_f1.to_numpy() * 100 for method in METHOD_ORDER]
        ax.boxplot(data, tick_labels=[METHOD_LABELS[m].replace(" + ", " +\n") for m in METHOD_ORDER], showmeans=True)
        ax.set_title(dataset)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("OOS F1 (%)")
    fig.suptitle("KIR=0.50 five-seed OOS F1 variability")
    fig.tight_layout()
    fig.savefig(FIG / "seed_variance_oos_f1_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# 同协议方法权衡与可视化分析 V1",
        "",
        "> 本报告只汇总已完成的 `protocol_v2_textoir_v1` 五 seed 结果，不重新训练，不加入历史 `fulltex.tex`、官方 MOGB BERT 或 DCLOOS 外部 OOS 监督结果。",
        "",
        "## 1. 分析范围",
        "",
        "- 三个数据集、KIR={0.25, 0.50, 0.75}、5 个 seed；",
        "- Trainable K=1、Frozen K=1、Frozen K=2、Random K=2、MOGB 组件三种变体；",
        "- 主要观察 OOS F1、F1-All、Known Recall、false acceptance/rejection 和 seed 方差；",
        "- 所有行仍是 Gate-only 或同协议组件，不是完整 Cascade 的 SOTA 排名。",
        "",
        "## 2. KIR=0.50 的 Trainable 与组件对照",
        "",
        "| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | FA | FR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        part = summary[(summary.dataset == dataset) & (summary.kir == 0.50)].set_index("method")
        for method in METHOD_ORDER:
            row = part.loc[method]
            lines.append(
                f"| {dataset} | {METHOD_LABELS[method]} | {row.oos_f1*100:.2f} | {row.f1_all*100:.2f} | "
                f"{row.known_recall*100:.2f} | {row.false_accept_rate*100:.2f} | {row.false_reject_rate*100:.2f} |"
            )
    lines.extend(
        [
            "",
            "## 3. 机制结论",
            "",
            "1. Trainable K=1 通常位于更好的覆盖—拒识折中区域：它保留较高 Known Recall/F1-All，同时比 Frozen K=1 降低 false acceptance。",
            "2. MOGB 风格组件有时提高 OOS F1，但常以显著牺牲 Known Recall 和 F1-All 为代价；这说明它们更像保守拒识工作点，而不是全面替代。",
            "3. StackOverflow 的 Frozen K=2 仍出现明显 OOS 误接受，说明训练表示的 K=1 收益不能直接外推为固定多中心安全性。",
            "4. KIR 增大时，方法之间的差距和误差权衡发生变化；因此不能用单一 KIR 的最好数字宣称跨数据集统一优势。",
            "",
            "## 4. 图和机器结果",
            "",
            "- `results/analysis/cross_protocol_tradeoff_v1/per_seed.csv`",
            "- `results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`",
            "- `results/analysis/cross_protocol_tradeoff_v1/trainable_vs_components_paired.csv`",
            "- `figures/cross_protocol_tradeoff_v1/pareto_oos_f1_known_recall.png`",
            "- `figures/cross_protocol_tradeoff_v1/error_decomposition_kir050.png`",
            "- `figures/cross_protocol_tradeoff_v1/kir_curves_oos_f1_f1_all.png`",
            "- `figures/cross_protocol_tradeoff_v1/seed_variance_oos_f1_kir050.png`",
            "",
            "## 5. 结论边界",
            "",
            "- 历史 `fulltex.tex` 是完整 Gate→Router→Expert Cascade，不能混入本报告的 fair rows。",
            "- MOGB 这里是 Frozen MiniLM 组件适配，不是作者 BERT 完整复现。",
            "- DCLOOS 使用伪 OOS/外部 OOS 监督，不能与 Known-only 行直接排名。",
            "- 后续应在统一监督、split、seed 和系统层级后再做强基线主表；本报告本身不启动新训练。",
        ]
    )
    (ROOT / "docs/analysis/CROSS_PROTOCOL_TRADEOFF_V1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    frame = _load()
    summary = _summary(frame)
    baselines = [method for method in METHOD_ORDER if method != "trainable_k1"]
    paired = pd.concat([_paired_summary(frame, baseline, METRICS) for baseline in baselines], ignore_index=True)
    _write(frame, "per_seed.csv")
    _write(summary, "summary_mean_std.csv")
    _write(paired, "trainable_vs_components_paired.csv")
    _plot_pareto(summary)
    _plot_errors(summary)
    _plot_kir(summary)
    _plot_variance(frame)
    _report(summary, paired)
    manifest = {
        "input": str(INPUT),
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "methods": list(METHOD_ORDER),
        "seed_count": 5,
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "analysis_only": True,
        "historical_fulltex_included": False,
        "external_oos_supervision_included": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "per_seed": len(frame), "summary": len(summary), "paired": len(paired), "figures": len(list(FIG.glob("*.png")))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
