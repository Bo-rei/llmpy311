#!/usr/bin/env python3
"""Align StackOverflow predictions and visualize error attribution.

Analysis-only: uses already-produced protocol_v2 predictions at the formal
score threshold (score <= 1 is Known).  No training or threshold selection is
performed here.
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
OUT = ROOT / "results/analysis/archive/analysis/stackoverflow_error_attribution_v2"
FIG = ROOT / "figures/archive/analysis/stackoverflow_error_attribution_v2"
KIR = 0.50
DATASET = "stackoverflow"
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
    candidates = [Path(path), ROOT / path, ROOT.parent / path]
    for candidate in candidates:
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
        raise ValueError(f"{path}: missing {sorted(missing)}")
    df = pd.DataFrame(
        {
            "sample_id": raw["sample_id"].astype(str),
            "gold_intent": raw["gold_intent"].astype(str),
            "gold_is_oos": raw["gold_is_oos"].astype(int),
            "score": pd.to_numeric(raw[score_col], errors="coerce"),
            "pred_label": raw[label_col].astype(str),
        }
    )
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated sample_id")
    if df["score"].isna().any() or not np.isfinite(df["score"]).all():
        raise ValueError(f"{path}: non-finite score")
    df["pred_label"] = df["pred_label"].replace({"oos": "__oos__", "unknown": "__oos__", "__unknown__": "__oos__"})
    df["pred_oos"] = (df["score"] > 1.0).astype(int)
    df["gold_type"] = np.where(df["gold_is_oos"].eq(1), "oos", "known")
    return df, path


def transition(left: pd.DataFrame, right: pd.DataFrame, method: str, seed: int) -> pd.DataFrame:
    if not left["sample_id"].equals(right["sample_id"]):
        raise ValueError(f"sample order mismatch: {method} seed={seed}")
    out = []
    left_correct = left["pred_oos"].eq(left["gold_is_oos"])
    right_correct = right["pred_oos"].eq(right["gold_is_oos"])
    for gold_type in ("known", "oos"):
        mask = left["gold_type"].eq(gold_type)
        categories = np.select(
            [left_correct & right_correct, left_correct & ~right_correct, ~left_correct & right_correct],
            ["both_correct", "trainable_only_correct", "comparison_only_correct"],
            default="both_incorrect",
        )
        counts = pd.Series(categories[mask.to_numpy()]).value_counts()
        denom = int(mask.sum())
        for category in ["both_correct", "trainable_only_correct", "comparison_only_correct", "both_incorrect"]:
            count = int(counts.get(category, 0))
            out.append({"seed": seed, "comparison_method": method, "gold_type": gold_type, "transition": category, "count": count, "rate": count / denom, "denominator": denom})
    return pd.DataFrame(out)


def method_metrics(df: pd.DataFrame, row: pd.Series, path: Path) -> dict[str, Any]:
    gold = df["gold_is_oos"].to_numpy(dtype=int)
    pred = df["pred_oos"].to_numpy(dtype=int)
    known = gold == 0
    oos = gold == 1
    return {
        "dataset": DATASET,
        "kir": KIR,
        "seed": int(row["seed"]),
        "method": str(row["method"]),
        "method_label": LABELS[str(row["method"])],
        "n_samples": int(len(df)),
        "oos_f1": float(f1_score(gold, pred, zero_division=0)),
        "known_recall": float(np.mean(pred[known] == 0)),
        "false_accept_rate": float(np.mean(pred[oos] == 0)),
        "false_reject_rate": float(np.mean(pred[known] == 1)),
        "threshold": 1.0,
        "source_prediction_path": str(path),
        "source_metric_oos_f1": float(row["oos_f1"]),
        "source_metric_known_recall": float(row["known_recall"]),
        "score_vs_source_oos_f1_delta": float(f1_score(gold, pred, zero_division=0) - row["oos_f1"]),
        "score_vs_source_known_recall_delta": float(np.mean(pred[known] == 0) - row["known_recall"]),
    }


def intent_attribution(df: pd.DataFrame, method: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    intents = sorted(set(df.loc[df.gold_is_oos.eq(0), "gold_intent"]) | set(df.loc[df.pred_oos.eq(0), "pred_label"]))
    for intent in intents:
        known = df[df.gold_intent.eq(intent) & df.gold_is_oos.eq(0)]
        accepted_oos = df[df.gold_is_oos.eq(1) & df.pred_oos.eq(0) & df.pred_label.eq(intent)]
        rows.append(
            {
                "dataset": DATASET,
                "kir": KIR,
                "seed": seed,
                "method": method,
                "method_label": LABELS[method],
                "intent": intent,
                "known_count": int(len(known)),
                "known_false_reject": int(known.pred_oos.sum()),
                "known_false_reject_rate": float(known.pred_oos.mean()) if len(known) else 0.0,
                "oos_false_accept": int(len(accepted_oos)),
                "oos_false_accept_rate_of_all_oos": float(len(accepted_oos) / max(1, int((df.gold_is_oos == 1).sum()))),
                "mean_known_score": float(known.score.mean()) if len(known) else np.nan,
                "mean_oos_accepted_score": float(accepted_oos.score.mean()) if len(accepted_oos) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_transition(summary: pd.DataFrame) -> None:
    sub = summary[summary.gold_type.eq("oos")].copy()
    matrix = sub.pivot(index="comparison_method", columns="transition", values="rate_mean").reindex(METHODS[1:])
    matrix = matrix[["trainable_only_correct", "comparison_only_correct", "both_incorrect"]]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), ["Trainable only correct", "Comparison only correct", "Both incorrect"])
    ax.set_yticks(range(matrix.shape[0]), [LABELS[x] for x in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("StackOverflow KIR=0.50: OOS error transitions vs Trainable K=1")
    fig.colorbar(im, ax=ax, label="Rate among OOS test samples")
    fig.savefig(FIG / "stackoverflow_oos_transition_heatmap.png", dpi=180)
    plt.close(fig)


def plot_waterfall(metrics: pd.DataFrame) -> None:
    d = metrics.groupby(["method", "method_label"], as_index=False).agg(oos_fa=("false_accept_rate", "mean"), known_fr=("false_reject_rate", "mean"))
    d["order"] = d.method.map({m: i for i, m in enumerate(METHODS)})
    d = d.sort_values("order")
    x = np.arange(len(d))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(x - width / 2, d.oos_fa, width, label="OOS false acceptance", color="#c53030")
    ax.bar(x + width / 2, d.known_fr, width, label="Known false rejection", color="#2b6cb0")
    ax.set_xticks(x, d.method_label, rotation=25, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("StackOverflow KIR=0.50: error-source comparison")
    ax.legend()
    fig.savefig(FIG / "stackoverflow_error_source_waterfall.png", dpi=180)
    plt.close(fig)


def plot_intent_heatmap(intent: pd.DataFrame, column: str, filename: str, title: str, cmap: str) -> None:
    d = intent.groupby(["intent", "method"], as_index=False)[column].mean()
    matrix = d.pivot(index="intent", columns="method", values=column).reindex(columns=METHODS)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=0, vmax=float(np.nanmax(matrix.to_numpy()) or 1), aspect="auto")
    ax.set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS], rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Rate")
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    source = source[(source.dataset == DATASET) & (source.kir == KIR) & source.method.isin(METHODS) & source.seed.isin(SEEDS)].copy()
    if len(source) != len(METHODS) * len(SEEDS):
        raise ValueError(f"expected {len(METHODS) * len(SEEDS)} source rows, got {len(source)}")
    loaded: dict[tuple[str, int], pd.DataFrame] = {}
    source_hashes: dict[str, str] = {str(SOURCE): sha256(SOURCE)}
    metric_rows: list[dict[str, Any]] = []
    intent_rows: list[pd.DataFrame] = []
    transition_rows: list[pd.DataFrame] = []
    for seed in SEEDS:
        base: pd.DataFrame | None = None
        for method in METHODS:
            row = source[(source.seed == seed) & (source.method == method)].iloc[0]
            pred, path = load_prediction(row)
            source_hashes[str(path)] = sha256(path)
            if len(pred) != 6000 or pred.gold_is_oos.value_counts().to_dict() != {0: 3000, 1: 3000}:
                raise ValueError(f"unexpected StackOverflow shape/split for {method} seed={seed}")
            if base is None:
                base = pred[["sample_id", "gold_intent", "gold_is_oos", "gold_type"]].copy()
            else:
                if not pred["sample_id"].equals(base["sample_id"]):
                    raise ValueError(f"sample_id order mismatch for {method} seed={seed}")
                if not pred["gold_intent"].equals(base["gold_intent"]):
                    raise ValueError(f"gold intent mismatch for {method} seed={seed}")
            loaded[(method, seed)] = pred
            metric_rows.append(method_metrics(pred, row, path))
            intent_rows.append(intent_attribution(pred, method, seed))
        assert base is not None
        for method in METHODS[1:]:
            transition_rows.append(transition(loaded[("trainable_k1", seed)], loaded[(method, seed)], method, seed))
    metrics = pd.DataFrame(metric_rows)
    intents = pd.concat(intent_rows, ignore_index=True)
    transitions = pd.concat(transition_rows, ignore_index=True)
    transition_summary = transitions.groupby(["comparison_method", "gold_type", "transition"], as_index=False).agg(rate_mean=("rate", "mean"), rate_std=("rate", "std"), n_seeds=("seed", "nunique"))
    intent_summary = intents.groupby(["method", "method_label", "intent"], as_index=False).agg(
        known_count_mean=("known_count", "mean"),
        known_false_reject_mean=("known_false_reject", "mean"),
        known_false_reject_rate_mean=("known_false_reject_rate", "mean"),
        oos_false_accept_mean=("oos_false_accept", "mean"),
        oos_false_accept_rate_mean=("oos_false_accept_rate_of_all_oos", "mean"),
    )
    atomic_csv(metrics, OUT / "method_metrics.csv")
    atomic_csv(transitions, OUT / "pairwise_transitions_per_seed.csv")
    atomic_csv(transition_summary, OUT / "pairwise_transition_summary.csv")
    atomic_csv(intents, OUT / "intent_error_attribution_per_seed.csv")
    atomic_csv(intent_summary, OUT / "intent_error_attribution_summary.csv")
    plot_transition(transition_summary)
    plot_waterfall(metrics)
    plot_intent_heatmap(intent_summary, "oos_false_accept_rate_mean", "stackoverflow_intent_oos_acceptor_heatmap.png", "StackOverflow: OOS false-acceptance attribution by intent", "Reds")
    plot_intent_heatmap(intent_summary, "known_false_reject_rate_mean", "stackoverflow_intent_known_reject_heatmap.png", "StackOverflow: Known false-rejection rate by intent", "Blues")
    manifest = {
        "analysis_id": "stackoverflow_error_attribution_v2",
        "protocol_version": "protocol_v2_textoir_v1",
        "dataset": DATASET,
        "kir": KIR,
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "threshold": 1.0,
        "threshold_semantics": "formal score <= 1 accepts Known; no threshold selection performed",
        "source_hashes": source_hashes,
        "run_count": int(len(metrics)),
        "prediction_rows_aligned": int(len(METHODS) * len(SEEDS) * 6000),
        "sample_order_checked": True,
        "intent_mapping_checked": True,
        "outputs": ["method_metrics.csv", "pairwise_transitions_per_seed.csv", "pairwise_transition_summary.csv", "intent_error_attribution_per_seed.csv", "intent_error_attribution_summary.csv"],
        "warning": "This is error attribution on frozen formal predictions; it does not select models or parameters and does not prove external-baseline fairness.",
    }
    atomic_json(manifest, OUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
