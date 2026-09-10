"""Build a contract-aware Chinese experiment comparison overview.

This is an analysis-only artifact.  It never combines historical Cascade,
current Gate-only, and external compatibility rows into one SOTA ranking.
The figure has four panels so the reader can see (a) the current fair matrix,
(b) the StackOverflow external reference cells, (c) the MOGB paper/local gap,
and (d) the historical fulltex margin separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FAIR = ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv"
EXTERNAL = ROOT / "results/analysis/archive/analysis/comparison_atlas_v1/stackoverflow_kir50_external_summary.csv"
HISTORICAL = ROOT / "results/analysis/archive/analysis/historical_sota_comparison_v1/ours_minus_best_baseline.csv"
MOGB_EXACT = ROOT / "results/analysis/archive/analysis/mogb_reproduction_gap_analysis_v2/exact_run_summary.csv"
MOGB_PAPER_GAP = ROOT / "results/analysis/archive/analysis/mogb_reproduction_gap_analysis_v2/paper_gap.csv"
BASELINE_SUMMARY = ROOT / "results/final_baselines/summary.csv"
DCLOOS_RECOVERY = ROOT.parent / "artifacts/s2c/external/dcloos_official_oos_kir75_seed888_reduced_v2/recovery_metrics.json"

OUT = ROOT / "results/analysis/archive/analysis/experiment_comparison_overview_v2"
FIG = ROOT / "figures/archive/analysis/experiment_comparison_overview_v2"
REPORT = ROOT / "docs/archive/analysis/EXPERIMENT_COMPARISON_OVERVIEW_V2.md"

_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
if _FONT.is_file():
    font_manager.fontManager.addfont(str(_FONT))
    _FAMILY = font_manager.FontProperties(fname=str(_FONT)).get_name()
else:
    _FAMILY = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = [_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FAIR_LABELS = {
    "trainable_k1": "S2C Trainable K=1",
    "single_centroid": "S2C Frozen K=1",
    "fixed_k2": "S2C Frozen K=2",
    "random_partition": "S2C Random K=2",
    "mogb_partition_ours_boundary": "MOGB partition + S2C boundary",
    "ours_partition_mogb_boundary": "S2C partition + MOGB boundary",
    "mogb_minilm": "MOGB-MiniLM",
}
FAIR_ORDER = list(FAIR_LABELS)
FAIR_COLORS = {
    "trainable_k1": "#0072B2",
    "single_centroid": "#777777",
    "fixed_k2": "#D55E00",
    "random_partition": "#E69F00",
    "mogb_partition_ours_boundary": "#009E73",
    "ours_partition_mogb_boundary": "#56B4E9",
    "mogb_minilm": "#CC79A7",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path, required: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return frame


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, format=path.suffix.lstrip("."), dpi=180, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(path)


def build() -> dict:
    fair = _read(FAIR, ["dataset", "kir", "method", "oos_f1", "f1_all", "known_recall", "n_seeds"])
    external = _read(EXTERNAL, ["method", "method_label", "layer", "oos_f1_mean", "f1_all_mean", "known_recall_mean", "accuracy_mean"])
    # The existing external summary predates the explicit n_seeds column.  ADB
    # has three valid same-data cells; the fair component rows have five; all
    # other compatibility rows are single-cell or invalid.  This inferred
    # count is only a display annotation, never a statistical reweighting.
    external["n_seeds"] = external["method"].map({"ADB": 3}).fillna(
        external["method"].isin(FAIR_LABELS).map({True: 5, False: 1})
    ).astype(int)
    historical = _read(HISTORICAL, ["dataset", "kir", "ours_oos_f1", "best_baseline_oos_f1", "margin_pp"])
    mogb_exact = _read(MOGB_EXACT, ["dataset", "kir", "accuracy", "f1_all", "f1_u", "f1_k", "known_recall", "contract"])
    mogb_paper = _read(MOGB_PAPER_GAP, ["dataset", "local_kir", "metric", "published_reference", "local_exact", "gap_pp"])
    mogb_paper = mogb_paper.rename(columns={"local_kir": "kir", "published_reference": "paper_value", "local_exact": "local_value"})
    if not DCLOOS_RECOVERY.is_file():
        raise FileNotFoundError(DCLOOS_RECOVERY)
    dcloos = json.loads(DCLOOS_RECOVERY.read_text(encoding="utf-8"))

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    fair_mean = (
        fair.groupby("method", as_index=False)[["oos_f1", "f1_all", "known_recall"]]
        .mean(numeric_only=True)
        .assign(method_label=lambda x: x["method"].map(FAIR_LABELS))
    )
    fair_mean["cells"] = fair.groupby("method").size().reindex(fair_mean["method"]).to_numpy()
    _save_csv(fair_mean, OUT / "current_fair_mean.csv")

    external_out = external.copy()
    external_out["scope"] = np.where(
        external_out["layer"].eq("current_protocol_v2_fair_gate"),
        "same_protocol_fair",
        np.where(external_out["layer"].eq("same_protocol_external_compatibility"), "same_data_external_compatibility", "invalid_or_other"),
    )
    _save_csv(external_out, OUT / "stackoverflow_kir50_contract_rows.csv")

    historical_summary = historical.groupby("dataset", as_index=False)["margin_pp"].mean()
    _save_csv(historical_summary, OUT / "historical_margin_mean.csv")

    # External supervision reference is deliberately a separate table: the
    # reduced DCLOOS cell uses KIR=.75/seed=888, pseudo-OOS and external SQuAD
    # OOS, so it must never be appended to the KIR=.50 fair ranking.
    external_reference = pd.DataFrame(
        [
            {
                "method": "trainable_k1",
                "method_label": "S2C Trainable K=1",
                "dataset": "stackoverflow",
                "kir": 0.50,
                "seed_scope": "13|42|87|100|123",
                "supervision": "Known-only",
                "backbone": "MiniLM",
                "budget": "formal_five_seed",
                "oos_f1": float(external.loc[external["method"].eq("trainable_k1"), "oos_f1_mean"].iloc[0]) * 100,
                "f1_all": float(external.loc[external["method"].eq("trainable_k1"), "f1_all_mean"].iloc[0]) * 100,
                "known_recall": float(external.loc[external["method"].eq("trainable_k1"), "known_recall_mean"].iloc[0]) * 100,
                "accuracy": float(external.loc[external["method"].eq("trainable_k1"), "accuracy_mean"].iloc[0]) * 100,
            },
            {
                "method": "ADB",
                "method_label": "ADB",
                "dataset": "stackoverflow",
                "kir": 0.50,
                "seed_scope": "42|87|100",
                "supervision": "Known-only",
                "backbone": "BERT/TextOIR",
                "budget": "compatibility_three_cell",
                "oos_f1": float(external.loc[external["method"].eq("ADB"), "oos_f1_mean"].iloc[0]) * 100,
                "f1_all": float(external.loc[external["method"].eq("ADB"), "f1_all_mean"].iloc[0]) * 100,
                "known_recall": float(external.loc[external["method"].eq("ADB"), "known_recall_mean"].iloc[0]) * 100,
                "accuracy": float(external.loc[external["method"].eq("ADB"), "accuracy_mean"].iloc[0]) * 100,
            },
            {
                "method": "DCLOOS_reduced",
                "method_label": "DCLOOS reduced",
                "dataset": "stackoverflow",
                "kir": 0.75,
                "seed_scope": "888",
                "supervision": "pseudo-OOS + external SQuAD OOS",
                "backbone": "BERT",
                "budget": "reduced_compatibility",
                "oos_f1": float(dcloos["oos_f1"]),
                "f1_all": float(dcloos["f1_all"]),
                "known_recall": float(dcloos["known_recall"]),
                "accuracy": float(dcloos["accuracy"]),
            },
        ]
    )
    _save_csv(external_reference, OUT / "external_supervision_reference.csv")

    mogb_exact_out = mogb_exact.copy()
    mogb_exact_out["source"] = "local_official_logic_compatibility"
    paper = mogb_paper.copy()
    paper["source"] = "paper_reference"
    _save_csv(mogb_exact_out, OUT / "mogb_local_exact_cells.csv")
    _save_csv(paper, OUT / "mogb_paper_gap.csv")

    # Panel A: current fair Gate-only matrix, averaged over the nine cells.
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    plot_fair = fair_mean.set_index("method").reindex(FAIR_ORDER).dropna(how="all")
    x = np.arange(len(plot_fair))
    for ax, metric, title in zip(axes, ["oos_f1", "f1_all"], ["当前 fair 矩阵：OOS F1（9 个 dataset×KIR 均值）", "当前 fair 矩阵：F1-All（9 个 dataset×KIR 均值）"]):
        values = plot_fair[metric].to_numpy(dtype=float) * 100
        bars = ax.bar(x, values, color=[FAIR_COLORS[m] for m in plot_fair.index])
        ax.set_xticks(x, [FAIR_LABELS[m].replace(" + ", "\n+") for m in plot_fair.index], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("百分比")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(0, max(values) + 8)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("S2C 当前实验与外部参考：合同分层总览（不得混成一个 SOTA 排名）", fontsize=13)
    _save_fig(fig, FIG / "current_fair_matrix_mean.png")

    # Panel B: StackOverflow/KIR=.50, showing valid rows and an explicit invalid DA-ADB marker.
    ext = external_out[external_out["method"].isin(["trainable_k1", "single_centroid", "fixed_k2", "random_partition", "mogb_minilm", "mogb_partition_ours_boundary", "ADB", "DA-ADB"])].copy()
    labels = {
        "trainable_k1": "S2C Trainable K=1",
        "single_centroid": "S2C Frozen K=1",
        "fixed_k2": "S2C Frozen K=2",
        "random_partition": "S2C Random K=2",
        "mogb_minilm": "MOGB-MiniLM",
        "mogb_partition_ours_boundary": "MOGB partition + S2C boundary",
        "ADB": "ADB（BERT兼容，3 seeds）",
        "DA-ADB": "DA-ADB（无效：NaN/全类）",
    }
    ext = ext.copy()
    ext["display"] = ext["method"].map(labels)
    ext["value"] = ext["oos_f1_mean"] * 100
    ext.loc[ext["method"].eq("DA-ADB"), "value"] = np.nan
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(ext))
    colors = [FAIR_COLORS.get(m, "#2F855A" if m == "ADB" else "#999999") for m in ext["method"]]
    bars = ax.bar(x, ext["value"], color=colors, alpha=0.9)
    for i, row in ext.reset_index(drop=True).iterrows():
        if np.isfinite(row["value"]):
            ax.text(i, row["value"] + 1, f"{row['value']:.1f}\n(n={int(row['n_seeds'])})", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(i, 5, "无效\nNaN/全类", ha="center", va="bottom", fontsize=8, color="#882255")
            ax.scatter([i], [5], marker="x", s=120, color="#882255", zorder=5)
    ax.set_xticks(x, [str(v).replace(" + ", "\n+") for v in ext["display"]], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("OOS F1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("StackOverflow / KIR=0.50：同数据参照与外部兼容结果（合同分层）")
    ax.grid(axis="y", alpha=0.22)
    ax.text(0.01, 0.02, "ADB 使用 BERT/TextOIR；S2C/MOGB-Fair 使用当前 MiniLM Gate 合同；不能直接作同监督 SOTA 排名。", transform=ax.transAxes, fontsize=8)
    _save_fig(fig, FIG / "stackoverflow_kir50_contract_layers.png")

    # Panel C: direct visual of MOGB paper/local gap for the available exact cells.
    paper_stack = paper[paper["dataset"].astype(str).str.lower().eq("stackoverflow") & paper["kir"].astype(float).eq(0.5)]
    exact_stack = mogb_exact_out[mogb_exact_out["dataset"].astype(str).str.lower().eq("stackoverflow") & mogb_exact_out["kir"].astype(float).eq(0.5)]
    metric_map = {"accuracy": "Accuracy", "f1_all": "F1-All", "f1_u": "F1-U", "f1_k": "F1-K"}
    if not paper_stack.empty and not exact_stack.empty:
        metrics = [m for m, label in metric_map.items() if label in set(paper_stack["metric"])]
        # The MOGB gap CSV stores percentages (e.g. 88.67), whereas the
        # current fair matrix stores fractions.  Keep this panel in percent
        # without multiplying the already-percent local exact values again.
        paper_vals = [float(paper_stack.loc[paper_stack["metric"].eq(metric_map[m]), "paper_value"].iloc[0]) for m in metrics]
        exact_vals = [float(exact_stack[m].iloc[0]) for m in metrics]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        xx = np.arange(len(metrics))
        width = 0.36
        ax.bar(xx - width / 2, paper_vals, width, label="MOGB 论文公开参考", color="#009E73")
        ax.bar(xx + width / 2, exact_vals, width, label="本地官方逻辑兼容单格", color="#D55E00")
        ax.set_xticks(xx, [metric_map[m] for m in metrics])
        ax.set_ylabel("百分比")
        ax.set_ylim(0, 100)
        ax.set_title("MOGB：论文公开工作点与本地兼容单格（StackOverflow/KIR=.50）")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.22)
        for pos, val in zip(xx - width / 2, paper_vals):
            ax.text(pos, val + 1, f"{val:.1f}", ha="center", fontsize=8)
        for pos, val in zip(xx + width / 2, exact_vals):
            ax.text(pos, val + 1, f"{val:.1f}", ha="center", fontsize=8)
        _save_fig(fig, FIG / "mogb_paper_vs_local_exact.png")

    # Panel D: historical fulltex margin, deliberately separate from current rows.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    hist_labels = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}
    pivot = historical.pivot(index="dataset", columns="kir", values="margin_pp").reindex(["clinc150", "banking77", "stackoverflow"])
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=0, vmax=max(1, float(np.nanmax(pivot.to_numpy()))))
    ax.set_xticks(range(len(pivot.columns)), [f"KIR={float(c):.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [hist_labels.get(i, i) for i in pivot.index])
    ax.set_title("历史 fulltex Cascade：Ours 相对表内最佳基线的 OOS F1 优势（旧合同）")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"+{value:.2f} pp", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="优势（百分点）")
    _save_fig(fig, FIG / "historical_fulltex_margin.png")

    # Panel E: explicit supervision/budget contrast.  This is not a ranking:
    # KIR, seed count, backbone and OOS supervision are written on the x-axis.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.subplots_adjust(bottom=0.28, top=0.82, wspace=0.16)
    display = ["S2C Trainable\nKIR=.50\nKnown-only", "ADB\nKIR=.50\nKnown-only", "DCLOOS reduced\nKIR=.75\npseudo+external OOS"]
    colors = ["#0072B2", "#2F855A", "#CC79A7"]
    xx = np.arange(len(external_reference))
    for ax, metric, title in zip(axes, ["oos_f1", "f1_all"], ["OOS F1", "F1-All"]):
        vals = external_reference[metric].to_numpy(dtype=float)
        bars = ax.bar(xx, vals, color=colors)
        ax.set_xticks(xx, display, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("百分比")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=9)
    fig.suptitle("外部监督条件对照：数值参考，不是统一 SOTA 排名", fontsize=13)
    fig.text(0.5, 0.035, "DCLOOS reduced 使用 BERT、pseudo-OOS、外部 SQuAD OOS、KIR=.75/seed=888；S2C/ADB 为 Known-only。", ha="center", fontsize=8)
    _save_fig(fig, FIG / "external_supervision_reference.png")

    manifest = {
        "schema_version": 3,
        "sources": {str(p.relative_to(ROOT)): _sha(p) for p in [FAIR, EXTERNAL, HISTORICAL, MOGB_EXACT, MOGB_PAPER_GAP]},
        "external_sources": {str(DCLOOS_RECOVERY.relative_to(ROOT.parent)): _sha(DCLOOS_RECOVERY)},
        "figures": [
            "current_fair_matrix_mean.png",
            "stackoverflow_kir50_contract_layers.png",
            "mogb_paper_vs_local_exact.png",
            "historical_fulltex_margin.png",
            "external_supervision_reference.png",
        ],
        "scope": "analysis_only_contract_aware_comparison",
        "no_test_selection": True,
    }
    _save_json(manifest, OUT / "MANIFEST.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
