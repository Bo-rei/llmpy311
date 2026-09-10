#!/usr/bin/env python3
"""Cross-dataset/KIR error attribution for existing fair predictions.

The script only reads frozen protocol_v2 prediction files.  It aligns each
method by sample_id within dataset/KIR/seed, recomputes the formal score<=1
decision, and aggregates error transitions.  It never tunes a threshold.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
OUT = ROOT / "results/analysis/archive/analysis/cross_dataset_error_attribution_v1"
FIG = ROOT / "figures/archive/analysis/cross_dataset_error_attribution_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
METHODS = (
    "trainable_k1",
    "single_centroid",
    "fixed_k2",
    "random_partition",
    "mogb_minilm",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
)
LABELS = {
    "trainable_k1": "Trainable K=1",
    "single_centroid": "Frozen K=1",
    "fixed_k2": "Frozen K=2",
    "random_partition": "Random K=2",
    "mogb_minilm": "MOGB-MiniLM",
    "mogb_partition_ours_boundary": "MOGB partition + ours",
    "ours_partition_mogb_boundary": "Ours partition + MOGB",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def resolve(path: str) -> Path:
    for candidate in (Path(path), ROOT / path, ROOT.parent / path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def load_prediction(row: pd.Series) -> tuple[pd.DataFrame, Path]:
    method = str(row["method"])
    if method == "trainable_k1":
        path = resolve(str(row["metrics_path"])).with_name("predictions.jsonl")
        raw = pd.read_json(path, lines=True)
        score_col = "oos_score"
        label_col = "predicted_intent"
    else:
        path = resolve(str(row["run_dir"])) / "predictions.tsv"
        raw = pd.read_csv(path, sep="\t")
        score_col = "normalized_score"
        label_col = "predicted_label"
    required = {"sample_id", "gold_intent", "gold_is_oos", score_col, label_col}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = pd.DataFrame(
        {
            "sample_id": raw["sample_id"].astype(str),
            "gold_intent": raw["gold_intent"].astype(str),
            "gold_is_oos": raw["gold_is_oos"].astype(int),
            "score": pd.to_numeric(raw[score_col], errors="coerce"),
            "pred_label": raw[label_col].astype(str).replace({"oos": "__oos__", "unknown": "__oos__", "__unknown__": "__oos__"}),
        }
    )
    if df["sample_id"].duplicated().any() or df["score"].isna().any() or not np.isfinite(df["score"]).all():
        raise ValueError(f"invalid prediction file {path}")
    df["pred_oos"] = (df["score"] > 1.0).astype(int)
    df["gold_type"] = np.where(df["gold_is_oos"].eq(1), "oos", "known")
    return df, path


def method_metric(df: pd.DataFrame, row: pd.Series, dataset: str, kir: float, seed: int, path: Path) -> dict[str, Any]:
    gold = df["gold_is_oos"].to_numpy(dtype=int)
    pred = df["pred_oos"].to_numpy(dtype=int)
    known = gold == 0
    oos = gold == 1
    f1 = float(f1_score(gold, pred, zero_division=0))
    return {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "method": str(row["method"]),
        "method_label": LABELS[str(row["method"])],
        "n_samples": int(len(df)),
        "n_known": int(known.sum()),
        "n_oos": int(oos.sum()),
        "oos_f1": f1,
        "known_recall": float(np.mean(pred[known] == 0)),
        "false_accept_rate": float(np.mean(pred[oos] == 0)),
        "false_reject_rate": float(np.mean(pred[known] == 1)),
        "source_oos_f1": float(row["oos_f1"]),
        "source_known_recall": float(row["known_recall"]),
        "metric_recompute_delta": f1 - float(row["oos_f1"]),
        "source_prediction_path": str(path),
    }


def transitions(base: pd.DataFrame, comparison: pd.DataFrame, dataset: str, kir: float, seed: int, method: str) -> pd.DataFrame:
    if not base["sample_id"].equals(comparison["sample_id"]):
        raise ValueError(f"sample order mismatch {dataset} kir={kir} seed={seed} {method}")
    if not base["gold_intent"].equals(comparison["gold_intent"]):
        raise ValueError(f"gold intent mismatch {dataset} kir={kir} seed={seed} {method}")
    left_correct = base["pred_oos"].eq(base["gold_is_oos"])
    right_correct = comparison["pred_oos"].eq(comparison["gold_is_oos"])
    categories = np.select(
        [left_correct & right_correct, left_correct & ~right_correct, ~left_correct & right_correct],
        ["both_correct", "trainable_only_correct", "comparison_only_correct"],
        default="both_incorrect",
    )
    rows: list[dict[str, Any]] = []
    for gold_type in ("known", "oos"):
        mask = base["gold_type"].eq(gold_type).to_numpy()
        counts = pd.Series(categories[mask]).value_counts()
        denominator = int(mask.sum())
        for category in ("both_correct", "trainable_only_correct", "comparison_only_correct", "both_incorrect"):
            count = int(counts.get(category, 0))
            rows.append({"dataset": dataset, "kir": kir, "seed": seed, "comparison_method": method, "gold_type": gold_type, "transition": category, "count": count, "rate": count / denominator, "denominator": denominator})
    return pd.DataFrame(rows)


def intent_rows(df: pd.DataFrame, dataset: str, kir: float, seed: int, method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    intents = sorted(set(df.loc[df.gold_is_oos.eq(0), "gold_intent"]) | set(df.loc[df.pred_oos.eq(0), "pred_label"]))
    oos_total = int(df.gold_is_oos.eq(1).sum())
    for intent in intents:
        known = df[df.gold_is_oos.eq(0) & df.gold_intent.eq(intent)]
        accepted = df[df.gold_is_oos.eq(1) & df.pred_oos.eq(0) & df.pred_label.eq(intent)]
        rows.append({
            "dataset": dataset,
            "kir": kir,
            "seed": seed,
            "method": method,
            "method_label": LABELS[method],
            "intent": intent,
            "known_count": int(len(known)),
            "known_false_reject": int(known.pred_oos.sum()),
            "known_false_reject_rate": float(known.pred_oos.mean()) if len(known) else 0.0,
            "oos_false_accept": int(len(accepted)),
            "oos_false_accept_rate": float(len(accepted) / max(1, oos_total)),
        })
    return pd.DataFrame(rows)


def heatmap(matrix: pd.DataFrame, path: Path, title: str, label: str, cmap: str, vmin: float | None = None, vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)), [LABELS.get(x, x) for x in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:+.2f}" if (vmin is not None and vmin < 0) else f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_method_effects(summary: pd.DataFrame, metric: str, title: str, filename: str, diverging: bool = False) -> None:
    d = summary[summary.comparison_method.isin(METHODS[1:])].copy()
    d["row"] = d.dataset + " / KIR=" + d.kir.map(lambda x: f"{x:.2f}")
    matrix = d.pivot(index="row", columns="comparison_method", values=metric).reindex(columns=METHODS[1:])
    if diverging:
        max_abs = float(np.nanmax(np.abs(matrix.to_numpy())))
        heatmap(matrix, FIG / filename, title, "Trainable - comparison rate", "RdBu_r", -max_abs, max_abs)
    else:
        heatmap(matrix, FIG / filename, title, "Rate difference", "Reds", 0, float(np.nanmax(matrix.to_numpy())))


def plot_tradeoff(metrics: pd.DataFrame) -> None:
    d = metrics.groupby(["dataset", "method", "method_label"], as_index=False)[["oos_f1", "false_accept_rate"]].mean()
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    colors = {"clinc150": "#2b6cb0", "banking77": "#c53030", "stackoverflow": "#2f855a"}
    for dataset in DATASETS:
        sub = d[d.dataset == dataset]
        ax.scatter(sub.false_accept_rate, sub.oos_f1, s=60, color=colors[dataset], label=dataset)
        for _, r in sub.iterrows():
            ax.annotate(LABELS[r.method], (r.false_accept_rate, r.oos_f1), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("False acceptance rate (lower is safer)")
    ax.set_ylabel("OOS F1")
    ax.set_title("Cross-dataset formal operating-point tradeoff")
    ax.legend(title="dataset")
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "cross_dataset_oos_f1_false_acceptance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    source = source[source.dataset.isin(DATASETS) & source.kir.isin(KIRS) & source.seed.isin(SEEDS) & source.method.isin(METHODS)].copy()
    expected = len(DATASETS) * len(KIRS) * len(SEEDS) * len(METHODS)
    if len(source) != expected:
        raise ValueError(f"expected {expected} source rows, got {len(source)}")
    source_hashes: dict[str, str] = {str(SOURCE): sha256(SOURCE)}
    metrics: list[dict[str, Any]] = []
    trans: list[pd.DataFrame] = []
    intent: list[pd.DataFrame] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                loaded: dict[str, pd.DataFrame] = {}
                base: pd.DataFrame | None = None
                for method in METHODS:
                    row = source[(source.dataset == dataset) & (source.kir == kir) & (source.seed == seed) & (source.method == method)].iloc[0]
                    pred, path = load_prediction(row)
                    source_hashes[str(path)] = sha256(path)
                    if base is None:
                        base = pred[["sample_id", "gold_intent", "gold_is_oos", "gold_type"]].copy()
                    else:
                        if not pred["sample_id"].equals(base["sample_id"]) or not pred["gold_intent"].equals(base["gold_intent"]):
                            raise ValueError(f"alignment mismatch {dataset} kir={kir} seed={seed} method={method}")
                    loaded[method] = pred
                    metrics.append(method_metric(pred, row, dataset, kir, seed, path))
                    intent.append(intent_rows(pred, dataset, kir, seed, method))
                assert base is not None
                for method in METHODS[1:]:
                    trans.append(transitions(loaded["trainable_k1"], loaded[method], dataset, kir, seed, method))
    metric_df = pd.DataFrame(metrics)
    trans_df = pd.concat(trans, ignore_index=True)
    intent_df = pd.concat(intent, ignore_index=True)
    trans_summary = trans_df.groupby(["dataset", "kir", "comparison_method", "gold_type", "transition"], as_index=False).agg(rate_mean=("rate", "mean"), rate_std=("rate", "std"), n_seeds=("seed", "nunique"))
    metric_summary = metric_df.groupby(["dataset", "kir", "method", "method_label"], as_index=False).agg(oos_f1_mean=("oos_f1", "mean"), oos_f1_std=("oos_f1", "std"), known_recall_mean=("known_recall", "mean"), false_accept_rate_mean=("false_accept_rate", "mean"), false_reject_rate_mean=("false_reject_rate", "mean"), n_seeds=("seed", "nunique"))
    intent_summary = intent_df.groupby(["dataset", "kir", "method", "method_label", "intent"], as_index=False).agg(known_count_mean=("known_count", "mean"), known_false_reject_rate_mean=("known_false_reject_rate", "mean"), oos_false_accept_rate_mean=("oos_false_accept_rate", "mean"))
    atomic_csv(metric_df, OUT / "method_metrics_per_seed.csv")
    atomic_csv(metric_summary, OUT / "method_metrics_summary.csv")
    atomic_csv(trans_df, OUT / "pairwise_transitions_per_seed.csv")
    atomic_csv(trans_summary, OUT / "pairwise_transition_summary.csv")
    atomic_csv(intent_df, OUT / "intent_error_attribution_per_seed.csv")
    atomic_csv(intent_summary, OUT / "intent_error_attribution_summary.csv")
    # Positive values mean the comparison method is worse than Trainable K=1.
    oos = trans_summary[trans_summary.gold_type == "oos"].copy()
    pivot_oos = oos[oos.transition.isin(["trainable_only_correct", "comparison_only_correct"])].pivot_table(index=["dataset", "kir", "comparison_method"], columns="transition", values="rate_mean").reset_index()
    pivot_oos["trainable_win_minus_comparison_win"] = pivot_oos["trainable_only_correct"] - pivot_oos["comparison_only_correct"]
    atomic_csv(pivot_oos, OUT / "trainable_oos_transition_effects.csv")
    fa = metric_summary.pivot_table(index=["dataset", "kir"], columns="method", values="false_accept_rate_mean")
    fr = metric_summary.pivot_table(index=["dataset", "kir"], columns="method", values="false_reject_rate_mean")
    fa_delta = fa[fa.columns.difference(["trainable_k1"])].subtract(fa["trainable_k1"], axis=0).reindex(columns=METHODS[1:])
    fr_delta = fr[fr.columns.difference(["trainable_k1"])].subtract(fr["trainable_k1"], axis=0).reindex(columns=METHODS[1:])
    atomic_csv(fa_delta.reset_index().rename(columns={m: f"{m}_fa_minus_trainable" for m in METHODS[1:]}), OUT / "false_acceptance_delta_vs_trainable.csv")
    atomic_csv(fr_delta.reset_index().rename(columns={m: f"{m}_fr_minus_trainable" for m in METHODS[1:]}), OUT / "false_rejection_delta_vs_trainable.csv")
    plot_method_effects(trans_summary[trans_summary.gold_type == "oos"].pivot_table(index=["dataset", "kir", "comparison_method", "gold_type"], columns="transition", values="rate_mean").reset_index().assign(trainable_win_minus_comparison_win=lambda x: x["trainable_only_correct"] - x["comparison_only_correct"]), "trainable_win_minus_comparison_win", "Trainable K=1 OOS correctness advantage by dataset and KIR", "trainable_oos_win_heatmap.png", True)
    plot_tradeoff(metric_df)
    # Compact summary heatmaps for false-acceptance and false-rejection deltas.
    fa_plot = fa_delta.copy()
    fa_plot.index = [f"{d} / KIR={k:.2f}" for d, k in fa_plot.index]
    fr_plot = fr_delta.copy()
    fr_plot.index = [f"{d} / KIR={k:.2f}" for d, k in fr_plot.index]
    heatmap(fa_plot, FIG / "false_acceptance_delta_vs_trainable.png", "Comparison false acceptance minus Trainable K=1", "FA difference", "RdBu_r", float(-np.nanmax(np.abs(fa_plot.to_numpy()))), float(np.nanmax(np.abs(fa_plot.to_numpy()))))
    heatmap(fr_plot, FIG / "false_rejection_delta_vs_trainable.png", "Comparison false rejection minus Trainable K=1", "FR difference", "RdBu_r", float(-np.nanmax(np.abs(fr_plot.to_numpy()))), float(np.nanmax(np.abs(fr_plot.to_numpy()))))
    manifest = {
        "analysis_id": "cross_dataset_error_attribution_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "threshold": 1.0,
        "source_hashes": source_hashes,
        "source_run_count": int(len(source)),
        "aligned_prediction_rows": int(len(DATASETS) * len(KIRS) * len(SEEDS) * len(METHODS) * 6000),
        "sample_order_checked": True,
        "gold_intent_checked": True,
        "warning": "Frozen prediction attribution only; no threshold/model selection and no external-baseline fairness claim.",
    }
    atomic_json(manifest, OUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
