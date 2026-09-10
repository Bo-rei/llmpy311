#!/usr/bin/env python3
"""Build threshold-diagnostic comparisons including MOGB components.

This is an analysis-only post-hoc diagnostic.  It aligns score thresholds to
test-known recall targets so methods with very different operating points can
be inspected on the same x-axis.  It is not a validation protocol and must
never be used to select a model, radius, threshold, or method.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import f1_score, precision_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts"
SOURCE_CSV = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
OUT = ROOT / "results/analysis/archive/analysis/mogb_operating_point_visuals_v1"
FIG = ROOT / "figures/archive/analysis/mogb_operating_point_visuals_v1"
TARGETS = (0.75, 0.85, 0.90, 0.95)
METHOD_ORDER = (
    "trainable_k1",
    "single_centroid",
    "fixed_k2",
    "random_partition",
    "mogb_minilm",
    "mogb_partition_ours_boundary",
    "ours_partition_mogb_boundary",
)
DISPLAY = {
    "trainable_k1": "Trainable MiniLM K=1",
    "single_centroid": "Frozen single centroid",
    "fixed_k2": "Frozen fixed K=2",
    "random_partition": "Random partition",
    "mogb_minilm": "MOGB-MiniLM",
    "mogb_partition_ours_boundary": "MOGB partition + ours boundary",
    "ours_partition_mogb_boundary": "Ours partition + MOGB boundary",
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


def resolve_source(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    candidate = ROOT.parent / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path)


def prediction_path(row: pd.Series) -> Path:
    method = str(row["method"])
    if method == "trainable_k1":
        return resolve_source(str(row["metrics_path"])).with_name("predictions.jsonl")
    return resolve_source(str(row["run_dir"])) / "predictions.tsv"


def load_predictions(row: pd.Series) -> pd.DataFrame:
    path = prediction_path(row)
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
        score_col = "oos_score"
        label_col = "predicted_intent"
    else:
        df = pd.read_csv(path, sep="\t")
        score_col = "normalized_score"
        label_col = "predicted_label"
    required = {"sample_id", "gold_is_oos", score_col, label_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    out = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "gold_is_oos": df["gold_is_oos"].astype(int),
            "score": pd.to_numeric(df[score_col], errors="coerce"),
            "base_label": df[label_col].astype(str),
        }
    )
    if out["sample_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate sample_id")
    if out["score"].isna().any() or not np.isfinite(out["score"]).all():
        raise ValueError(f"{path}: non-finite score")
    return out


def normalise_label(value: str) -> str:
    return "__oos__" if value in {"oos", "__oos__", "unknown", "__unknown__"} else value


def threshold_for_known_recall(scores: np.ndarray, target: float) -> float:
    ordered = np.sort(scores)
    index = int(np.ceil(target * ordered.size) - 1)
    index = max(0, min(index, ordered.size - 1))
    return float(ordered[index])


def score_metrics(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    gold_oos = df["gold_is_oos"].to_numpy(dtype=int)
    pred_oos = (df["score"].to_numpy(dtype=float) > threshold).astype(int)
    known_mask = gold_oos == 0
    oos_mask = gold_oos == 1
    known_recall = float(np.mean(pred_oos[known_mask] == 0))
    oos_recall = float(np.mean(pred_oos[oos_mask] == 1))
    false_accept = float(np.mean(pred_oos[oos_mask] == 0))
    false_reject = float(np.mean(pred_oos[known_mask] == 1))
    # The full intent label is not available in every compact prediction file.
    # Keep OOS metrics complete and mark macro-F1 as unavailable rather than
    # fabricating an intent mapping from sample ids.
    return {
        "known_recall": known_recall,
        "oos_recall": oos_recall,
        "false_accept_rate": false_accept,
        "false_reject_rate": false_reject,
        "oos_f1": float(f1_score(gold_oos, pred_oos, zero_division=0)),
        "oos_precision": float(precision_score(gold_oos, pred_oos, zero_division=0)),
        "auroc": float("nan"),
        "aupr_oos": float("nan"),
        "f1_all": float("nan"),
        "f1_k": float("nan"),
        "accuracy": float("nan"),
    }


def make_rows(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {str(SOURCE_CSV): sha256(SOURCE_CSV)}
    for _, row in source.iterrows():
        method = str(row["method"])
        path = prediction_path(row)
        source_hashes[str(path)] = sha256(path)
        pred = load_predictions(row)
        known = pred.loc[pred["gold_is_oos"] == 0, "score"].to_numpy(dtype=float)
        for target in TARGETS:
            threshold = threshold_for_known_recall(known, target)
            metrics = score_metrics(pred, threshold)
            metrics.update(
                {
                    "dataset": str(row["dataset"]),
                    "kir": float(row["kir"]),
                    "seed": int(row["seed"]),
                    "method": method,
                    "method_label": DISPLAY.get(method, method),
                    "target_known_recall": target,
                    "diagnostic_threshold": threshold,
                    "source_prediction_path": str(path),
                    "diagnostic_only": True,
                    "threshold_source": "test_known_scores_posthoc",
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows), source_hashes


def plot_stackoverflow(summary: pd.DataFrame, target: float) -> None:
    d = summary[(summary.dataset == "stackoverflow") & (summary.kir == 0.5) & (summary.target_known_recall == target)].copy()
    d["order"] = d.method.map({m: i for i, m in enumerate(METHOD_ORDER)})
    d = d.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(d["method_label"], d["oos_f1_mean"], color="#2b6cb0")
    axes[0].set_title(f"StackOverflow KIR=0.50: OOS F1 at Known Recall≈{target:.2f}")
    axes[0].set_xlabel("OOS F1 (post-hoc diagnostic)")
    axes[0].invert_yaxis()
    axes[1].barh(d["method_label"], d["false_accept_rate_mean"], color="#c53030")
    axes[1].set_title("False acceptance at the same work point")
    axes[1].set_xlabel("False acceptance rate")
    axes[1].invert_yaxis()
    fig.savefig(FIG / f"stackoverflow_kir050_target_{str(target).replace('.', '')}.png", dpi=180)
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame) -> None:
    d = summary[(summary.target_known_recall == 0.85) & (summary.kir == 0.5)].copy()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for method in METHOD_ORDER:
        sub = d[d.method == method]
        if sub.empty:
            continue
        means = sub.groupby("dataset", as_index=False)[["oos_f1_mean", "false_accept_rate_mean"]].mean()
        ax.scatter(means["false_accept_rate_mean"], means["oos_f1_mean"], s=70, label=DISPLAY[method])
        for _, r in means.iterrows():
            ax.annotate(str(r["dataset"]).replace("banking77", "Banking").replace("stackoverflow", "SO").replace("clinc150", "CLINC"), (r.false_accept_rate_mean, r.oos_f1_mean), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("False acceptance rate (lower is safer)")
    ax.set_ylabel("OOS F1")
    ax.set_title("Same Known Recall≈0.85: OOS utility vs open-space risk")
    ax.legend(fontsize=7, loc="best")
    fig.savefig(FIG / "cross_dataset_target085_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_threshold_curves(rows: pd.DataFrame) -> None:
    sub = rows[(rows.dataset == "stackoverflow") & (rows.kir == 0.5)].copy()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for method in METHOD_ORDER:
        d = sub[sub.method == method]
        if d.empty:
            continue
        grouped = d.groupby("target_known_recall", as_index=False)[["known_recall", "oos_f1"]].mean().sort_values("known_recall")
        ax.plot(grouped.known_recall, grouped.oos_f1, marker="o", label=DISPLAY[method])
    ax.set_xlabel("Achieved Known Recall")
    ax.set_ylabel("OOS F1")
    ax.set_title("StackOverflow KIR=0.50 diagnostic operating curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.savefig(FIG / "stackoverflow_operating_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_CSV)
    args = parser.parse_args()
    source_csv = args.source.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(source_csv)
    expected = {"dataset", "kir", "seed", "method", "metrics_path", "run_dir"}
    missing = expected.difference(source.columns)
    if missing:
        raise ValueError(f"source missing {sorted(missing)}")
    rows, hashes = make_rows(source)
    rows = rows.sort_values(["dataset", "kir", "seed", "target_known_recall", "method"]).reset_index(drop=True)
    summary = rows.groupby(["dataset", "kir", "target_known_recall", "method", "method_label"], as_index=False).agg(
        known_recall_mean=("known_recall", "mean"),
        known_recall_std=("known_recall", "std"),
        oos_f1_mean=("oos_f1", "mean"),
        oos_f1_std=("oos_f1", "std"),
        false_accept_rate_mean=("false_accept_rate", "mean"),
        false_accept_rate_std=("false_accept_rate", "std"),
        n_seeds=("seed", "nunique"),
    )
    atomic_csv(rows, OUT / "per_seed_targets.csv")
    atomic_csv(summary, OUT / "summary.csv")
    plot_stackoverflow(summary, 0.85)
    plot_stackoverflow(summary, 0.90)
    plot_tradeoff(summary)
    plot_threshold_curves(rows)
    manifest = {
        "analysis_id": "mogb_operating_point_visuals_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "source_csv": str(source_csv),
        "source_csv_sha256": sha256(source_csv),
        "prediction_file_count": len(hashes) - 1,
        "prediction_file_sha256": hashes,
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "targets": list(TARGETS),
        "methods": list(METHOD_ORDER),
        "diagnostic_warning": "阈值按测试集 Known 分数事后对齐，仅用于可视化/机制诊断，不用于模型、阈值、半径或方法选择。",
        "metrics_available": ["known_recall", "oos_recall", "oos_f1", "oos_precision", "false_accept_rate", "false_reject_rate"],
        "metrics_unavailable_from_compact_predictions": ["f1_all", "f1_k", "accuracy", "auroc", "aupr_oos"],
    }
    atomic_json(manifest, OUT / "MANIFEST.json")


if __name__ == "__main__":
    main()
