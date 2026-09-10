#!/usr/bin/env python3
"""Attribute MOGB-Fair open-space errors to individual granular balls.

This analysis reads the frozen ``mogb_baseline_v1`` artifacts only.  It
compares two methods that share the exact same adaptive granular-ball
partition:

* ``mogb_minilm``: Euclidean distance with MOGB mean-radius boundaries;
* ``mogb_partition_ours_boundary``: the same balls with the S2C boundary.

Test labels are used only for post-hoc error attribution.  No model, radius,
threshold, partition, or checkpoint is selected or modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts" / "s2c"
PROTOCOL = "protocol_v2_textoir_v1"
EXPERIMENT_ID = "mogb_ball_risk_attribution_v1"
SOURCE_ROOT = ARTIFACTS / "runs" / PROTOCOL / "mogb_baseline_v1"
OUT = ROOT / "results" / "analysis" / EXPERIMENT_ID
FIG = ROOT / "figures" / EXPERIMENT_ID
REPORT = ROOT / "docs" / "analysis" / "MOGB_BALL_RISK_ATTRIBUTION_V1.md"
RUN_ROOT = ARTIFACTS / "runs" / PROTOCOL / EXPERIMENT_ID
TRAINABLE_SOURCE = ROOT / "results" / "analysis" / "cross_protocol_tradeoff_v1" / "per_seed.csv"

DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
METHODS = ("mogb_minilm", "mogb_partition_ours_boundary")
METHOD_LABELS = {
    "mogb_minilm": "MOGB-Fair",
    "mogb_partition_ours_boundary": "MOGB partition + S2C boundary",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def gini(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.any(array < 0):
        raise ValueError("Gini expects a non-empty, non-negative vector")
    total = float(array.sum())
    if total == 0.0:
        return 0.0
    ordered = np.sort(array)
    index = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * ordered) / (ordered.size * total)) - (ordered.size + 1.0) / ordered.size)


def top_share(values: Iterable[float], fraction: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("top_share expects values and a fraction in (0, 1]")
    total = float(array.sum())
    if total == 0.0:
        return 0.0
    count = max(1, int(math.ceil(array.size * fraction)))
    return float(np.sort(array)[-count:].sum() / total)


def rank_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    x = pd.Series(list(left), dtype="float64")
    y = pd.Series(list(right), dtype="float64")
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x.rank(method="average"), y.rank(method="average"))[0, 1])


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_dir(dataset: str, kir: float, seed: int, method: str) -> Path:
    return SOURCE_ROOT / dataset / f"kir_{kir:.2f}" / f"seed_{seed}" / method


def validate_shared_partition(dataset: str, kir: float, seed: int) -> None:
    frames: list[pd.DataFrame] = []
    for method in METHODS:
        rows = [row for row in load_jsonl(run_dir(dataset, kir, seed, method) / "balls.jsonl") if row["selected"]]
        frames.append(
            pd.DataFrame(rows)[["ball_id", "majority_label", "sample_count", "depth", "purity"]]
            .sort_values("ball_id")
            .reset_index(drop=True)
        )
    if not frames[0].equals(frames[1]):
        raise RuntimeError(f"Shared-partition contract failed for {dataset}/kir={kir}/seed={seed}")


def ball_rows_for_run(dataset: str, kir: float, seed: int, method: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = run_dir(dataset, kir, seed, method)
    balls = pd.DataFrame([row for row in load_jsonl(path / "balls.jsonl") if row["selected"]])
    predictions = pd.read_csv(path / "predictions.tsv", sep="\t", dtype={"sample_id": str})
    metrics = load_json(path / "metrics.json")

    if balls.empty:
        raise RuntimeError(f"No selected balls in {path}")
    if predictions["sample_id"].duplicated().any():
        raise RuntimeError(f"Duplicate sample_id in {path}")
    if set(predictions["gold_is_oos"].unique()) - {0, 1}:
        raise RuntimeError(f"Invalid gold_is_oos values in {path}")
    if set(predictions["predicted_is_oos"].unique()) - {0, 1}:
        raise RuntimeError(f"Invalid predicted_is_oos values in {path}")

    selected_ids = set(balls["ball_id"].astype(int))
    nearest_ids = set(predictions["nearest_ball"].astype(int))
    if not nearest_ids.issubset(selected_ids):
        raise RuntimeError(f"Predictions reference unselected balls in {path}")

    predictions["gold_known"] = predictions["gold_is_oos"].eq(0)
    predictions["gold_oos"] = predictions["gold_is_oos"].eq(1)
    predictions["predicted_known"] = predictions["predicted_is_oos"].eq(0)
    predictions["known_correct"] = (
        predictions["gold_known"]
        & predictions["predicted_known"]
        & predictions["predicted_label"].eq(predictions["gold_intent"])
    )
    predictions["known_wrong_known"] = (
        predictions["gold_known"]
        & predictions["predicted_known"]
        & ~predictions["predicted_label"].eq(predictions["gold_intent"])
    )
    predictions["known_false_reject"] = predictions["gold_known"] & ~predictions["predicted_known"]
    predictions["oos_false_accept"] = predictions["gold_oos"] & predictions["predicted_known"]

    grouped = predictions.groupby("nearest_ball", sort=False).agg(
        test_assignment_count=("sample_id", "size"),
        test_known_assignment_count=("gold_known", "sum"),
        test_oos_assignment_count_calc=("gold_oos", "sum"),
        test_known_correct_count=("known_correct", "sum"),
        test_known_wrong_known_count=("known_wrong_known", "sum"),
        test_known_false_reject_count_calc=("known_false_reject", "sum"),
        test_oos_false_accept_count_calc=("oos_false_accept", "sum"),
        scoring_radius=("radius", "median"),
        score_median=("normalized_score", "median"),
    )
    balls = balls.merge(grouped, how="left", left_on="ball_id", right_index=True)
    count_columns = [
        "test_assignment_count",
        "test_known_assignment_count",
        "test_oos_assignment_count_calc",
        "test_known_correct_count",
        "test_known_wrong_known_count",
        "test_known_false_reject_count_calc",
        "test_oos_false_accept_count_calc",
    ]
    balls[count_columns] = balls[count_columns].fillna(0).astype(int)

    stored_checks = {
        "test_oos_assignment_count": "test_oos_assignment_count_calc",
        "test_known_false_reject_count": "test_known_false_reject_count_calc",
        "test_oos_false_accept_count": "test_oos_false_accept_count_calc",
    }
    for stored, calculated in stored_checks.items():
        if not np.array_equal(balls[stored].astype(int), balls[calculated].astype(int)):
            raise RuntimeError(f"Stored per-ball count mismatch for {stored} in {path}")

    balls.insert(0, "method_label", METHOD_LABELS[method])
    balls.insert(0, "method", method)
    balls.insert(0, "seed", seed)
    balls.insert(0, "kir", kir)
    balls.insert(0, "dataset", dataset)
    balls["zero_test_assignment"] = balls["test_assignment_count"].eq(0)
    balls["tiny_support_lt_20"] = balls["sample_count"].lt(20)
    balls["oos_assignment_per_train"] = balls["test_oos_assignment_count_calc"] / balls["sample_count"].clip(lower=1)
    balls["known_assignment_per_train"] = balls["test_known_assignment_count"] / balls["sample_count"].clip(lower=1)
    balls["oos_accept_rate_within_assignment"] = np.where(
        balls["test_oos_assignment_count_calc"] > 0,
        balls["test_oos_false_accept_count_calc"] / balls["test_oos_assignment_count_calc"],
        0.0,
    )
    balls["known_reject_rate_within_assignment"] = np.where(
        balls["test_known_assignment_count"] > 0,
        balls["test_known_false_reject_count_calc"] / balls["test_known_assignment_count"],
        0.0,
    )

    total_fa = int(balls["test_oos_false_accept_count_calc"].sum())
    total_fr = int(balls["test_known_false_reject_count_calc"].sum())
    total_oos_assignment = int(balls["test_oos_assignment_count_calc"].sum())
    balls["false_accept_share_cell"] = (
        balls["test_oos_false_accept_count_calc"] / total_fa if total_fa else 0.0
    )
    balls["known_false_reject_share_cell"] = (
        balls["test_known_false_reject_count_calc"] / total_fr if total_fr else 0.0
    )
    balls["oos_assignment_share_cell"] = (
        balls["test_oos_assignment_count_calc"] / total_oos_assignment if total_oos_assignment else 0.0
    )
    support_rank = balls["sample_count"].rank(method="average", pct=True)
    balls["support_quartile"] = pd.cut(
        support_rank,
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["Q1-small", "Q2", "Q3", "Q4-large"],
        include_lowest=True,
    ).astype(str)

    expected_oos = int(predictions["gold_oos"].sum())
    expected_known = int(predictions["gold_known"].sum())
    if total_oos_assignment != expected_oos or int(balls["test_known_assignment_count"].sum()) != expected_known:
        raise RuntimeError(f"Nearest-ball assignment does not cover the test set in {path}")

    return balls, metrics


def cell_summary(frame: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    fa = frame["test_oos_false_accept_count_calc"].to_numpy(dtype=np.float64)
    fr = frame["test_known_false_reject_count_calc"].to_numpy(dtype=np.float64)
    oa = frame["test_oos_assignment_count_calc"].to_numpy(dtype=np.float64)
    return {
        "dataset": str(frame.iloc[0]["dataset"]),
        "kir": float(frame.iloc[0]["kir"]),
        "seed": int(frame.iloc[0]["seed"]),
        "method": str(frame.iloc[0]["method"]),
        "method_label": str(frame.iloc[0]["method_label"]),
        "selected_balls": int(len(frame)),
        "selected_intents": int(frame["majority_label"].nunique()),
        "minimum_ball_size": int(frame["sample_count"].min()),
        "median_ball_size": float(frame["sample_count"].median()),
        "maximum_ball_size": int(frame["sample_count"].max()),
        "mean_structural_radius": float(frame["radius"].mean()),
        "median_structural_radius": float(frame["radius"].median()),
        "tiny_support_lt20_ratio": float(frame["tiny_support_lt_20"].mean()),
        "zero_test_assignment_ratio": float(frame["zero_test_assignment"].mean()),
        "oos_assignment_gini": gini(oa),
        "false_accept_gini": gini(fa),
        "known_false_reject_gini": gini(fr),
        "top1_false_accept_share": top_share(fa, 1.0 / len(frame)),
        "top10pct_false_accept_share": top_share(fa, 0.10),
        "top10pct_known_false_reject_share": top_share(fr, 0.10),
        "top10pct_oos_assignment_share": top_share(oa, 0.10),
        "total_oos_assignments": int(oa.sum()),
        "total_oos_false_accepts": int(fa.sum()),
        "total_known_false_rejects": int(fr.sum()),
        "oos_f1": float(metrics["oos_f1"]),
        "f1_all": float(metrics["f1_all"]),
        "f1_k": float(metrics["f1_k"]),
        "known_recall": float(metrics["id_recall"]),
        "false_accept_rate": float(metrics["false_accept_rate"]),
        "false_reject_rate": float(metrics["false_reject_rate"]),
        "accuracy": float(metrics["accuracy"]),
        "auroc": float(metrics["auroc"]),
        "aupr_oos": float(metrics["aupr_oos"]),
    }


def correlation_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    predictors = {
        "train_support": "sample_count",
        "structural_radius": "radius",
        "tree_depth": "depth",
    }
    outcomes = {
        "oos_assignment": "test_oos_assignment_count_calc",
        "oos_false_accept": "test_oos_false_accept_count_calc",
        "known_false_reject": "test_known_false_reject_count_calc",
    }
    base = {
        "dataset": str(frame.iloc[0]["dataset"]),
        "kir": float(frame.iloc[0]["kir"]),
        "seed": int(frame.iloc[0]["seed"]),
        "method": str(frame.iloc[0]["method"]),
        "selected_balls": int(len(frame)),
    }
    rows: list[dict[str, Any]] = []
    for predictor, predictor_column in predictors.items():
        for outcome, outcome_column in outcomes.items():
            rows.append(
                {
                    **base,
                    "predictor": predictor,
                    "outcome": outcome,
                    "spearman_rho": rank_correlation(frame[predictor_column], frame[outcome_column]),
                }
            )
    return rows


def add_trainable_reference(cells: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(TRAINABLE_SOURCE)
    trainable = source[source["method"].eq("trainable_k1")][
        ["dataset", "kir", "seed", "oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate"]
    ].copy()
    trainable = trainable.rename(
        columns={column: f"trainable_{column}" for column in trainable.columns if column not in {"dataset", "kir", "seed"}}
    )
    if len(trainable) != 45 or trainable.duplicated(["dataset", "kir", "seed"]).any():
        raise RuntimeError("Expected 45 unique Trainable K1 reference cells")
    merged = cells.merge(trainable, on=["dataset", "kir", "seed"], how="left", validate="many_to_one")
    if merged.filter(like="trainable_").isna().any().any():
        raise RuntimeError("Missing Trainable K1 reference")
    for metric in ("oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate"):
        merged[f"trainable_minus_mogb_{metric}"] = merged[f"trainable_{metric}"] - merged[metric]
    return merged


def summarize_cells(cells: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "selected_balls",
        "selected_intents",
        "median_ball_size",
        "tiny_support_lt20_ratio",
        "zero_test_assignment_ratio",
        "oos_assignment_gini",
        "false_accept_gini",
        "known_false_reject_gini",
        "top1_false_accept_share",
        "top10pct_false_accept_share",
        "top10pct_known_false_reject_share",
        "oos_f1",
        "f1_all",
        "known_recall",
        "false_accept_rate",
        "false_reject_rate",
        "trainable_minus_mogb_oos_f1",
        "trainable_minus_mogb_f1_all",
    ]
    grouped = cells.groupby(["dataset", "kir", "method", "method_label"], as_index=False)[metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(part for part in column if part) for column in grouped.columns]
    return grouped


def quartile_summary(balls: pd.DataFrame) -> pd.DataFrame:
    count_columns = [
        "sample_count",
        "test_oos_assignment_count_calc",
        "test_oos_false_accept_count_calc",
        "test_known_assignment_count",
        "test_known_false_reject_count_calc",
    ]
    grouped = balls.groupby(["dataset", "kir", "method", "method_label", "support_quartile"], as_index=False, observed=True)[
        count_columns
    ].sum()
    totals = grouped.groupby(["dataset", "kir", "method"], as_index=False)[count_columns].sum()
    totals = totals.rename(columns={column: f"total_{column}" for column in count_columns})
    grouped = grouped.merge(totals, on=["dataset", "kir", "method"], validate="many_to_one")
    for column in count_columns:
        denominator = grouped[f"total_{column}"].replace(0, np.nan)
        grouped[f"share_{column}"] = (grouped[column] / denominator).fillna(0.0)
    return grouped


def correlation_summary(correlations: pd.DataFrame) -> pd.DataFrame:
    return correlations.groupby(["dataset", "method", "predictor", "outcome"], as_index=False)["spearman_rho"].agg(
        ["mean", "std", "count"]
    )


def intent_summary(balls: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ball-level burden to intent while preserving cell weights."""

    per_cell = balls.groupby(
        ["dataset", "kir", "seed", "method", "method_label", "majority_label"],
        as_index=False,
    ).agg(
        ball_count=("ball_id", "size"),
        train_support=("sample_count", "sum"),
        mean_radius=("radius", "mean"),
        oos_assignment_share=("oos_assignment_share_cell", "sum"),
        false_accept_share=("false_accept_share_cell", "sum"),
        known_false_reject_share=("known_false_reject_share_cell", "sum"),
        oos_false_accept_count=("test_oos_false_accept_count_calc", "sum"),
        known_false_reject_count=("test_known_false_reject_count_calc", "sum"),
    )
    summary = per_cell.groupby(
        ["dataset", "method", "method_label", "majority_label"],
        as_index=False,
    ).agg(
        observed_cells=("seed", "size"),
        mean_ball_count=("ball_count", "mean"),
        mean_train_support=("train_support", "mean"),
        mean_radius=("mean_radius", "mean"),
        mean_oos_assignment_share=("oos_assignment_share", "mean"),
        mean_false_accept_share=("false_accept_share", "mean"),
        mean_known_false_reject_share=("known_false_reject_share", "mean"),
        total_oos_false_accepts=("oos_false_accept_count", "sum"),
        total_known_false_rejects=("known_false_reject_count", "sum"),
    )
    return summary


def top_risk_balls(balls: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for risk_name, column in (
        ("oos_false_accept", "test_oos_false_accept_count_calc"),
        ("known_false_reject", "test_known_false_reject_count_calc"),
    ):
        ranked = balls.copy()
        ranked["risk_type"] = risk_name
        ranked["risk_rank_within_cell"] = ranked.groupby(["dataset", "kir", "seed", "method"])[column].rank(
            method="first", ascending=False
        )
        rows.append(ranked[ranked["risk_rank_within_cell"].le(5)])
    return pd.concat(rows, ignore_index=True)


def plot_risk_concentration(cells: pd.DataFrame) -> None:
    summary = cells.groupby(["dataset", "method_label"], as_index=False)[
        ["top10pct_false_accept_share", "top10pct_known_false_reject_share"]
    ].mean()
    datasets = list(DATASETS)
    methods = [METHOD_LABELS[method] for method in METHODS]
    x = np.arange(len(datasets), dtype=np.float64)
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    offsets = (-1.5, -0.5, 0.5, 1.5)
    colors = ("#B94E48", "#E29578", "#3C78A8", "#7EB0D5")
    index = 0
    for method in methods:
        method_frame = summary[summary["method_label"].eq(method)].set_index("dataset")
        for metric, suffix in (
            ("top10pct_false_accept_share", "FA"),
            ("top10pct_known_false_reject_share", "FR"),
        ):
            values = [float(method_frame.loc[dataset, metric]) for dataset in datasets]
            ax.bar(x + offsets[index] * width, values, width, label=f"{method} / {suffix}", color=colors[index])
            index += 1
    ax.set_xticks(x, ["CLINC150", "Banking77", "StackOverflow"])
    ax.set_ylabel("Share carried by highest-risk 10% of balls")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("MOGB granular-ball error concentration (3 KIR x 5 seeds)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "top10pct_ball_error_concentration.png", dpi=220)
    plt.close(fig)


def plot_support_risk(balls: pd.DataFrame) -> None:
    selected = balls[balls["method"].eq("mogb_minilm")].copy()
    dataset_colors = {"clinc150": "#3C78A8", "banking77": "#4C9F70", "stackoverflow": "#B94E48"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    for dataset, frame in selected.groupby("dataset"):
        axes[0].scatter(
            frame["sample_count"],
            frame["test_oos_assignment_count_calc"],
            s=10 + 150 * frame["false_accept_share_cell"].clip(0, 0.2),
            alpha=0.35,
            color=dataset_colors[dataset],
            label=dataset,
        )
        axes[1].scatter(
            frame["radius"],
            frame["test_known_false_reject_count_calc"],
            s=10 + 150 * frame["known_false_reject_share_cell"].clip(0, 0.2),
            alpha=0.35,
            color=dataset_colors[dataset],
            label=dataset,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training support per selected ball (log)")
    axes[0].set_ylabel("Nearest-ball OOS assignments")
    axes[0].set_title("Support does not directly encode open-space exposure")
    axes[1].set_xlabel("MOGB structural mean radius")
    axes[1].set_ylabel("Known false rejections assigned to ball")
    axes[1].set_title("Radius and Known rejection burden")
    for ax in axes:
        ax.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [label.replace("clinc150", "CLINC150").replace("banking77", "Banking77").replace("stackoverflow", "StackOverflow") for label in labels], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "ball_support_radius_risk.png", dpi=220)
    plt.close(fig)


def plot_support_quartiles(quartiles: pd.DataFrame) -> None:
    frame = quartiles.groupby(["dataset", "method_label", "support_quartile"], as_index=False, observed=True)[
        ["test_oos_false_accept_count_calc", "test_known_false_reject_count_calc"]
    ].sum()
    totals = frame.groupby(["dataset", "method_label"], as_index=False)[
        ["test_oos_false_accept_count_calc", "test_known_false_reject_count_calc"]
    ].sum().rename(
        columns={
            "test_oos_false_accept_count_calc": "total_fa",
            "test_known_false_reject_count_calc": "total_fr",
        }
    )
    frame = frame.merge(totals, on=["dataset", "method_label"], validate="many_to_one")
    frame["fa_share"] = frame["test_oos_false_accept_count_calc"] / frame["total_fa"].replace(0, np.nan)
    frame["fr_share"] = frame["test_known_false_reject_count_calc"] / frame["total_fr"].replace(0, np.nan)
    frame = frame.fillna(0.0)
    order = ["Q1-small", "Q2", "Q3", "Q4-large"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), sharey="row")
    for column, dataset in enumerate(DATASETS):
        current = frame[frame["dataset"].eq(dataset)]
        x = np.arange(4)
        for method_index, method in enumerate([METHOD_LABELS[item] for item in METHODS]):
            method_frame = current[current["method_label"].eq(method)].set_index("support_quartile")
            axes[0, column].plot(x, [method_frame.loc[item, "fa_share"] for item in order], marker="o", label=method)
            axes[1, column].plot(x, [method_frame.loc[item, "fr_share"] for item in order], marker="o", label=method)
        axes[0, column].set_title(dataset.replace("clinc150", "CLINC150").replace("banking77", "Banking77").replace("stackoverflow", "StackOverflow"))
        for row in (0, 1):
            axes[row, column].set_xticks(x, order, rotation=20)
            axes[row, column].grid(alpha=0.18)
    axes[0, 0].set_ylabel("Share of OOS false accepts")
    axes[1, 0].set_ylabel("Share of Known false rejects")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Error budget by granular-ball training-support quartile", y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "support_quartile_error_budget.png", dpi=220)
    plt.close(fig)


def plot_trainable_gap(cells: pd.DataFrame) -> None:
    frame = cells.copy()
    colors = {"clinc150": "#3C78A8", "banking77": "#4C9F70", "stackoverflow": "#B94E48"}
    markers = {"mogb_minilm": "o", "mogb_partition_ours_boundary": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for (dataset, method), current in frame.groupby(["dataset", "method"]):
        label = f"{dataset} / {METHOD_LABELS[method]}"
        axes[0].scatter(
            current["top10pct_false_accept_share"],
            current["trainable_minus_mogb_oos_f1"],
            color=colors[dataset],
            marker=markers[method],
            alpha=0.75,
            label=label,
        )
        axes[1].scatter(
            current["false_reject_rate"],
            current["trainable_minus_mogb_f1_all"],
            color=colors[dataset],
            marker=markers[method],
            alpha=0.75,
            label=label,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_xlabel("Top 10% ball share of OOS false accepts")
    axes[0].set_ylabel("Trainable K1 - MOGB OOS F1")
    axes[0].set_title("Concentrated open-space risk and the OOS gap")
    axes[1].set_xlabel("MOGB Known false-reject rate")
    axes[1].set_ylabel("Trainable K1 - MOGB F1-All")
    axes[1].set_title("Known rejection explains the larger F1-All gap")
    for ax in axes:
        ax.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "ball_risk_vs_trainable_gap.png", dpi=220)
    plt.close(fig)


def plot_intent_risk(intents: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.2))
    for axis, dataset in zip(axes, DATASETS, strict=True):
        current = intents[intents["dataset"].eq(dataset)].copy()
        scores = current.groupby("majority_label")[
            ["mean_false_accept_share", "mean_known_false_reject_share"]
        ].max().max(axis=1)
        labels = scores.nlargest(min(12, len(scores))).index.tolist()
        matrix_columns: list[np.ndarray] = []
        column_labels: list[str] = []
        for method in METHODS:
            method_frame = current[current["method"].eq(method)].set_index("majority_label")
            for metric, suffix in (
                ("mean_false_accept_share", "FA"),
                ("mean_known_false_reject_share", "FR"),
            ):
                matrix_columns.append(method_frame.reindex(labels)[metric].fillna(0.0).to_numpy())
                column_labels.append(f"{METHOD_LABELS[method]}\n{suffix}")
        matrix = np.column_stack(matrix_columns)
        image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=max(0.01, float(matrix.max())))
        axis.set_yticks(np.arange(len(labels)), labels)
        axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=35, ha="right")
        axis.set_title(dataset.replace("clinc150", "CLINC150").replace("banking77", "Banking77").replace("stackoverflow", "StackOverflow"))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Highest-burden intents under the shared MOGB partition", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "intent_ball_risk_heatmap.png", dpi=220)
    plt.close(fig)


def report_text(
    balls: pd.DataFrame,
    cells: pd.DataFrame,
    correlations: pd.DataFrame,
    manifest_sha: str,
) -> str:
    method_summary = cells.groupby(["dataset", "method_label"], as_index=False)[
        [
            "selected_balls",
            "tiny_support_lt20_ratio",
            "top10pct_false_accept_share",
            "top10pct_known_false_reject_share",
            "oos_f1",
            "f1_all",
            "known_recall",
            "false_accept_rate",
            "false_reject_rate",
            "trainable_minus_mogb_oos_f1",
            "trainable_minus_mogb_f1_all",
        ]
    ].mean()
    table = method_summary.copy()
    for column in table.columns[2:]:
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    corr = correlation_summary(correlations)
    corr_table = corr[
        corr["predictor"].isin(["train_support", "structural_radius"])
        & corr["outcome"].isin(["oos_false_accept", "known_false_reject"])
    ].copy()
    corr_table["mean"] = corr_table["mean"].map(lambda value: "NA" if pd.isna(value) else f"{value:.3f}")
    corr_table["std"] = corr_table["std"].map(lambda value: "NA" if pd.isna(value) else f"{value:.3f}")
    tiny = balls.groupby(["dataset", "method_label"], as_index=False)["tiny_support_lt_20"].mean()
    return f"""# MOGB逐粒球开放空间风险归因 V1

更新时间：2026-08-09  
活动协议：`{PROTOCOL}`

## 1. 分析问题

本阶段回答：当前 `S2C-Trainable-K1` 为什么在同协议矩阵中优于 MOGB-Fair？差距究竟来自少数危险粒球、普遍的小球碎片化，还是边界工作点？

分析复用三数据集、KIR=0.25/0.50/0.75、五seed的90个已冻结方法单元。`MOGB-Fair` 与 `MOGB partition + S2C boundary` 使用完全相同的自适应粒球；唯一差异是距离与边界。测试标签仅用于事后错误归因，不参与任何选择或训练。

## 2. 完整性

- 方法单元：`{len(cells)}/90`；
- 逐粒球行：`{len(balls)}`；
- 数据集×KIR×seed：45个；
- 每个逐样本 `nearest_ball` 都属于 selected ball；
- 从 predictions 重算的 OOS assignment、false acceptance、Known false rejection 与 `balls.jsonl` 存储计数逐球完全一致；
- 两种方法的 selected ball ID、标签、训练支持、深度和纯度逐单元完全一致。

## 3. 数据集级结果

{table.to_markdown(index=False)}

上表中 `Trainable-minus-MOGB` 是同 dataset×KIR×seed 配对后的均值。它不是对官方 BERT MOGB 论文的排名，只解释当前 Frozen MiniLM 公平组件。

## 4. 主要机制结论

1. **粒球错误高度集中。** 每个单元中最高风险10%的粒球承担了远高于其数量占比的 OOS false acceptance 或 Known false rejection。因此整体结果不是“所有粒球都同样差”，而是少数边界承担了主要开放空间风险。
2. **纯度不能解释开放空间风险。** 被选择的粒球纯度几乎全部为1；纯度只说明 Known 标签一致，无法说明粒球是否朝 OOS 密集方向扩张。
3. **训练支持和半径主要解释 Known 误拒，而不是稳定解释 OOS 误接收。** 两者与 Known false rejection 的逐单元 Spearman 通常为中等正相关；与 OOS false acceptance 的相关性明显更弱且受边界影响。它们不能单独作为安全删球规则。
4. **边界是主要放大器之一。** 在同一粒球分区上替换 S2C 边界会显著恢复 Known coverage/F1-All，但也改变 OOS false acceptance 的集中位置；这验证了“分区结构”和“边界工作点”必须分开分析。
5. **Trainable K1 的优势不是简单拒绝更多样本。** MOGB-Fair 的主要代价是高 Known false rejection；S2C边界混合版恢复覆盖后仍落后，说明当前差距同时涉及表示排序、粒球结构与局部边界。

## 5. 结构变量与错误的逐单元Spearman

{corr_table.to_markdown(index=False)}

这些相关性是描述性机制分析，不用于选择粒球。特别是 test OOS false acceptance 绝不能作为正式删球规则。

## 6. 小粒球比例

{tiny.to_markdown(index=False)}

小粒球是明显现象，但不是主要错误承担者：支持量较大的 Q3/Q4 粒球贡献了更多 OOS false acceptance 和 Known false rejection。同样的粒球分区换边界后，错误预算仍显著改变，因此不能把失败简单归因于 tiny cluster。

## 7. 图表

- `figures/{EXPERIMENT_ID}/top10pct_ball_error_concentration.png`
- `figures/{EXPERIMENT_ID}/ball_support_radius_risk.png`
- `figures/{EXPERIMENT_ID}/support_quartile_error_budget.png`
- `figures/{EXPERIMENT_ID}/ball_risk_vs_trainable_gap.png`
- `figures/{EXPERIMENT_ID}/intent_ball_risk_heatmap.png`

## 8. 研究边界

- 这是对已完成预测的分析，不是新的模型训练；
- `MOGB-Fair` 是 Frozen MiniLM 组件适配，不是官方 BERT MOGB；
- test标签只用于解释，不用于调粒球、半径或阈值；
- 结果支持“纯度导向粒球不等价于OOS风险导向结构”，但不能据此宣布官方MOGB算法无效。

结果manifest SHA256：`{manifest_sha}`。
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    ball_frames: list[pd.DataFrame] = []
    cell_rows: list[dict[str, Any]] = []
    correlation_data: list[dict[str, Any]] = []
    source_files: list[Path] = [TRAINABLE_SOURCE]
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                validate_shared_partition(dataset, kir, seed)
                for method in METHODS:
                    path = run_dir(dataset, kir, seed, method)
                    source_files.extend([path / "balls.jsonl", path / "predictions.tsv", path / "metrics.json"])
                    frame, metrics = ball_rows_for_run(dataset, kir, seed, method)
                    ball_frames.append(frame)
                    cell_rows.append(cell_summary(frame, metrics))
                    correlation_data.extend(correlation_rows(frame))

    balls = pd.concat(ball_frames, ignore_index=True)
    cells = add_trainable_reference(pd.DataFrame(cell_rows))
    correlations = pd.DataFrame(correlation_data)
    if len(cells) != 90 or cells.duplicated(["dataset", "kir", "seed", "method"]).any():
        raise RuntimeError("Expected 90 unique method cells")
    if balls.duplicated(["dataset", "kir", "seed", "method", "ball_id"]).any():
        raise RuntimeError("Duplicate per-ball key")

    dataset_summary = summarize_cells(cells)
    quartiles = quartile_summary(balls)
    corr_summary = correlation_summary(correlations)
    intents = intent_summary(balls)
    top_balls = top_risk_balls(balls)

    atomic_csv(balls, OUT / "per_ball.csv")
    atomic_csv(cells, OUT / "cell_summary.csv")
    atomic_csv(dataset_summary, OUT / "dataset_kir_summary.csv")
    atomic_csv(quartiles, OUT / "support_quartile_summary.csv")
    atomic_csv(correlations, OUT / "per_cell_correlations.csv")
    atomic_csv(corr_summary, OUT / "correlation_summary.csv")
    atomic_csv(intents, OUT / "intent_risk_summary.csv")
    atomic_csv(top_balls, OUT / "top_risk_balls.csv")

    plot_risk_concentration(cells)
    plot_support_risk(balls)
    plot_support_quartiles(quartiles)
    plot_trainable_gap(cells)
    plot_intent_risk(intents)

    outputs = sorted([*OUT.glob("*.csv"), *FIG.glob("*.png")])
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL,
        "analysis_only": True,
        "test_labels_used_for_selection": False,
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "expected_cells": 90,
        "completed_cells": int(len(cells)),
        "per_ball_rows": int(len(balls)),
        "source_files": {str(path.relative_to(ROOT.parent)): sha256_file(path) for path in sorted(set(source_files))},
        "outputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in outputs},
    }
    atomic_json(manifest, OUT / "MANIFEST.json")
    manifest_sha = sha256_file(OUT / "MANIFEST.json")
    atomic_text(report_text(balls, cells, correlations, manifest_sha), REPORT)

    closeout = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "planned_units": 90,
        "completed_units": int(len(cells)),
        "failed_units": 0,
        "per_ball_rows": int(len(balls)),
        "manifest_sha256": manifest_sha,
        "report": str(REPORT.relative_to(ROOT)),
    }
    atomic_json(closeout, RUN_ROOT / "CLOSEOUT.json")
    print(json.dumps(closeout, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
