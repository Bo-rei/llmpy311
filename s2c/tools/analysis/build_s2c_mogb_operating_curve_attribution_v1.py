"""Attribute the S2C--MOGB gap to ranking quality and operating point.

This is an analysis-only stage.  It reads frozen test predictions from the
completed protocol_v2_textoir_v1 Trainable-K1 and MOGB-MiniLM-Fair runs.  It
does not train, tune, overwrite a run, or promote any post-hoc threshold into
the formal protocol.

All stored scores use the same orientation: larger means more OOS-like.  The
formal work point is replayed at score > 1.0.  Oracle and matched-coverage
thresholds use test labels only as explicitly labelled post-hoc diagnostics.
They must never be reused for method selection.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1"
DEFAULT_OUTPUT = ROOT / "results" / "analysis" / "s2c_mogb_operating_curve_attribution_v1"
DEFAULT_FIGURES = ROOT / "figures" / "s2c_mogb_operating_curve_attribution_v1"
DEFAULT_REPORT = ROOT / "docs" / "analysis" / "S2C_MOGB_OPERATING_CURVE_ATTRIBUTION_V1.md"

DATASETS = ("banking77", "clinc150", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
METHODS = ("s2c_trainable_k1", "mogb_minilm_fair")
METHOD_LABELS = {
    "s2c_trainable_k1": "S2C Trainable K=1",
    "mogb_minilm_fair": "MOGB-MiniLM-Fair",
}
COLORS = {
    "s2c_trainable_k1": "#0072B2",
    "mogb_minilm_fair": "#009E73",
}
COVERAGE_TARGETS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
DEFAULT_THRESHOLD = 1.0
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_REPETITIONS = 10_000
CURVE_QUANTILES = np.linspace(0.0, 1.0, 101)

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, format=path.suffix.lstrip("."), dpi=190, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def source_paths() -> dict[tuple[str, str, float, int], Path]:
    paths: dict[tuple[str, str, float, int], Path] = {}
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                trainable_stage = (
                    "minilm_trainable_kir_sweep_v1"
                    if seed in (13, 42, 87)
                    else "minilm_trainable_kir_sweep_extension_v1"
                )
                paths[("s2c_trainable_k1", dataset, kir, seed)] = (
                    ARTIFACT_ROOT
                    / trainable_stage
                    / f"kir_{kir:.2f}"
                    / "runs"
                    / dataset
                    / f"seed_{seed}"
                    / "predictions.jsonl"
                )
                paths[("mogb_minilm_fair", dataset, kir, seed)] = (
                    ARTIFACT_ROOT
                    / "mogb_baseline_v1"
                    / dataset
                    / f"kir_{kir:.2f}"
                    / f"seed_{seed}"
                    / "mogb_minilm"
                    / "predictions.tsv"
                )
    return paths


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            rows.append(
                {
                    "sample_id": str(item["sample_id"]),
                    "gold_intent": str(item["gold_intent"]),
                    "gold_is_oos": int(item["gold_is_oos"]),
                    "predicted_intent": str(item["predicted_intent"]),
                    "predicted_is_oos": int(item["predicted_is_oos"]),
                    "score": float(item["oos_score"]),
                }
            )
    return pd.DataFrame(rows)


def read_tsv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
    required = {
        "sample_id",
        "gold_intent",
        "gold_is_oos",
        "predicted_label",
        "predicted_is_oos",
        "normalized_score",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "sample_id": raw["sample_id"].astype(str),
            "gold_intent": raw["gold_intent"].astype(str),
            "gold_is_oos": raw["gold_is_oos"].astype(int),
            "predicted_intent": raw["predicted_label"].astype(str),
            "predicted_is_oos": raw["predicted_is_oos"].astype(int),
            "score": raw["normalized_score"].astype(float),
        }
    )


def validate_prediction_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"empty prediction file: {path}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample_id: {path}")
    if not bool(frame["gold_is_oos"].isin([0, 1]).all()):
        raise ValueError(f"invalid gold_is_oos: {path}")
    if not bool(frame["predicted_is_oos"].isin([0, 1]).all()):
        raise ValueError(f"invalid predicted_is_oos: {path}")
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite score: {path}")
    replay = frame["score"].gt(DEFAULT_THRESHOLD).astype(int)
    mismatch = int((replay != frame["predicted_is_oos"]).sum())
    if mismatch:
        raise ValueError(f"threshold=1 replay mismatch ({mismatch} rows): {path}")
    return frame.sort_values("sample_id").reset_index(drop=True)


def load_predictions() -> tuple[dict[tuple[str, str, float, int], pd.DataFrame], dict[str, str]]:
    frames: dict[tuple[str, str, float, int], pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for key, path in source_paths().items():
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = read_tsv(path) if path.suffix == ".tsv" else read_jsonl(path)
        frames[key] = validate_prediction_frame(raw, path)
        hashes[str(path)] = sha256(path)

    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                left = frames[("s2c_trainable_k1", dataset, kir, seed)]
                right = frames[("mogb_minilm_fair", dataset, kir, seed)]
                if not left["sample_id"].equals(right["sample_id"]):
                    raise ValueError(f"sample_id mismatch: {dataset}/{kir}/{seed}")
                if not left[["gold_intent", "gold_is_oos"]].equals(right[["gold_intent", "gold_is_oos"]]):
                    raise ValueError(f"gold label mismatch: {dataset}/{kir}/{seed}")
    return frames, hashes


def binary_metrics(y_oos: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted_oos = scores > threshold
    known_mask = y_oos == 0
    oos_mask = y_oos == 1
    known_recall = float(np.mean(~predicted_oos[known_mask]))
    false_accept = float(np.mean(~predicted_oos[oos_mask]))
    false_reject = 1.0 - known_recall
    return {
        "threshold": float(threshold),
        "oos_f1": float(f1_score(y_oos, predicted_oos.astype(int), pos_label=1, zero_division=0)),
        "known_recall": known_recall,
        "false_accept_rate": false_accept,
        "false_reject_rate": false_reject,
        "oos_precision": float(
            np.sum(predicted_oos & oos_mask) / max(1, int(np.sum(predicted_oos)))
        ),
        "oos_recall": float(np.mean(predicted_oos[oos_mask])),
    }


def oracle_f1(y_oos: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_oos, scores)
    f1_values = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_index = int(np.argmax(f1_values))
    threshold = float("inf") if best_index >= len(thresholds) else float(thresholds[best_index])
    return float(f1_values[best_index]), threshold


def standardized_separation(known_scores: np.ndarray, oos_scores: np.ndarray) -> float:
    numerator = float(np.mean(oos_scores) - np.mean(known_scores))
    denominator = float(np.sqrt((np.var(known_scores) + np.var(oos_scores)) / 2.0))
    return numerator / denominator if denominator > 0 else float("nan")


def coverage_threshold(known_scores: np.ndarray, target: float) -> float:
    if not 0.0 < target <= 1.0:
        raise ValueError(f"invalid coverage target: {target}")
    return float(np.quantile(known_scores, target, method="higher"))


def dense_operating_curve(y_oos: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    thresholds = np.unique(np.quantile(scores, CURVE_QUANTILES))
    return [binary_metrics(y_oos, scores, float(threshold)) for threshold in thresholds]


def analyse_cell(frame: pd.DataFrame) -> tuple[dict[str, float], list[dict[str, float]], list[dict[str, float]], list[dict[str, float]]]:
    y_oos = frame["gold_is_oos"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    known_scores = scores[y_oos == 0]
    oos_scores = scores[y_oos == 1]
    default = binary_metrics(y_oos, scores, DEFAULT_THRESHOLD)
    oracle_score, oracle_threshold = oracle_f1(y_oos, scores)
    cell = {
        **default,
        "auroc": float(roc_auc_score(y_oos, scores)),
        "aupr_oos": float(average_precision_score(y_oos, scores)),
        "oracle_oos_f1": oracle_score,
        "oracle_threshold": oracle_threshold,
        "default_to_oracle_gap": oracle_score - default["oos_f1"],
        "score_separation_d": standardized_separation(known_scores, oos_scores),
        "known_score_median": float(np.median(known_scores)),
        "oos_score_median": float(np.median(oos_scores)),
        "median_score_gap": float(np.median(oos_scores) - np.median(known_scores)),
        "n_known": int(len(known_scores)),
        "n_oos": int(len(oos_scores)),
    }

    matched: list[dict[str, float]] = []
    for target in COVERAGE_TARGETS:
        threshold = coverage_threshold(known_scores, target)
        metrics = binary_metrics(y_oos, scores, threshold)
        matched.append({"target_known_recall": target, **metrics})

    quantiles: list[dict[str, float]] = []
    for label, values in (("known", known_scores), ("oos", oos_scores)):
        quantiles.append(
            {
                "sample_type": label,
                "q10": float(np.quantile(values, 0.10)),
                "q25": float(np.quantile(values, 0.25)),
                "q50": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q90": float(np.quantile(values, 0.90)),
            }
        )
    return cell, matched, dense_operating_curve(y_oos, scores), quantiles


def paired_ci(values: np.ndarray, rng_seed: int) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPETITIONS, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_tables(frames: dict[tuple[str, str, float, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    for method in METHODS:
        for dataset in DATASETS:
            for kir in KIRS:
                for seed in SEEDS:
                    prefix = {"method": method, "dataset": dataset, "kir": kir, "seed": seed}
                    cell, matched, curve, quantiles = analyse_cell(frames[(method, dataset, kir, seed)])
                    cells.append({**prefix, **cell})
                    matched_rows.extend({**prefix, **row} for row in matched)
                    curve_rows.extend({**prefix, **row} for row in curve)
                    quantile_rows.extend({**prefix, **row} for row in quantiles)

    cell_frame = pd.DataFrame(cells).sort_values(["dataset", "kir", "seed", "method"])
    matched_frame = pd.DataFrame(matched_rows).sort_values(
        ["dataset", "kir", "seed", "method", "target_known_recall"]
    )
    curve_frame = pd.DataFrame(curve_rows).sort_values(["dataset", "kir", "seed", "method", "threshold"])
    quantile_frame = pd.DataFrame(quantile_rows).sort_values(
        ["dataset", "kir", "seed", "method", "sample_type"]
    )

    metrics = (
        "oos_f1",
        "auroc",
        "aupr_oos",
        "known_recall",
        "false_accept_rate",
        "oracle_oos_f1",
        "default_to_oracle_gap",
        "score_separation_d",
    )
    effects: list[dict[str, object]] = []
    pair_index = 0
    for dataset in DATASETS:
        for kir in KIRS:
            left = cell_frame[
                cell_frame["method"].eq("s2c_trainable_k1")
                & cell_frame["dataset"].eq(dataset)
                & cell_frame["kir"].eq(kir)
            ].set_index("seed")
            right = cell_frame[
                cell_frame["method"].eq("mogb_minilm_fair")
                & cell_frame["dataset"].eq(dataset)
                & cell_frame["kir"].eq(kir)
            ].set_index("seed")
            if not left.index.equals(right.index):
                raise ValueError(f"seed mismatch: {dataset}/{kir}")
            for metric_index, metric in enumerate(metrics):
                delta = left[metric].to_numpy(dtype=float) - right[metric].to_numpy(dtype=float)
                low, high = paired_ci(delta, BOOTSTRAP_SEED + pair_index * 100 + metric_index)
                effects.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "metric": metric,
                        "mean_delta": float(np.mean(delta)),
                        "std_delta": float(np.std(delta, ddof=1)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": int(np.sum(delta > 1e-12)),
                        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                        "losses": int(np.sum(delta < -1e-12)),
                    }
                )
            pair_index += 1
    return cell_frame, pd.DataFrame(effects), matched_frame, curve_frame, quantile_frame


def heatmap_delta(effects: pd.DataFrame, figure_path: Path) -> None:
    metrics = ("auroc", "aupr_oos", "score_separation_d")
    titles = ("AUROC差值", "AUPR-OOS差值", "标准化分数分离差值")
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    rows = [f"{dataset}\nKIR={kir:.2f}" for dataset in DATASETS for kir in KIRS]
    for axis, metric, title in zip(axes, metrics, titles, strict=True):
        selected = effects[effects["metric"].eq(metric)].copy()
        selected["row"] = selected["dataset"].astype(str) + "\nKIR=" + selected["kir"].map(lambda x: f"{x:.2f}")
        values = selected.set_index("row").reindex(rows)["mean_delta"].to_numpy()[:, None]
        limit = max(float(np.nanmax(np.abs(values))), 1e-6)
        image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks([0], ["S2C - MOGB"])
        axis.set_yticks(range(len(rows)), rows if axis is axes[0] else [])
        axis.set_title(title)
        for row_index, value in enumerate(values[:, 0]):
            axis.text(0, row_index, f"{value:+.3f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.72)
    figure.suptitle("阈值无关排序能力：S2C Trainable K=1 相对 MOGB-Fair", fontsize=14)
    atomic_figure(figure, figure_path)


def default_oracle_bars(cells: pd.DataFrame, figure_path: Path) -> None:
    summary = cells.groupby(["dataset", "kir", "method"], as_index=False)[["oos_f1", "oracle_oos_f1"]].mean()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        subset = summary[summary["dataset"].eq(dataset)]
        positions = np.arange(len(KIRS), dtype=float)
        width = 0.18
        for method_index, method in enumerate(METHODS):
            rows = subset[subset["method"].eq(method)].set_index("kir").reindex(KIRS)
            offset = (-1.5 + method_index * 2.0) * width
            axis.bar(
                positions + offset,
                rows["oos_f1"],
                width,
                color=COLORS[method],
                alpha=0.95,
                label=f"{METHOD_LABELS[method]} 默认" if dataset == DATASETS[0] else None,
            )
            axis.bar(
                positions + offset + width,
                rows["oracle_oos_f1"],
                width,
                color=COLORS[method],
                alpha=0.35,
                hatch="//",
                label=f"{METHOD_LABELS[method]} 事后上限" if dataset == DATASETS[0] else None,
            )
        axis.set_xticks(positions, [f"{kir:.2f}" for kir in KIRS])
        axis.set_title(dataset)
        axis.set_xlabel("KIR")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("OOS F1")
    figure.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False, fontsize=9)
    figure.suptitle("默认工作点与事后最优阈值上限（仅诊断，不用于选参）", y=1.17, fontsize=14)
    atomic_figure(figure, figure_path)


def matched_coverage_frontier(matched: pd.DataFrame, figure_path: Path) -> None:
    selected = matched[matched["kir"].eq(0.50)]
    summary = selected.groupby(["dataset", "method", "target_known_recall"], as_index=False)[
        ["oos_f1", "false_accept_rate", "known_recall"]
    ].mean()
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.3), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for method in METHODS:
            rows = summary[
                summary["dataset"].eq(dataset) & summary["method"].eq(method)
            ].sort_values("target_known_recall")
            axis.plot(
                rows["known_recall"],
                rows["oos_f1"],
                marker="o",
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(dataset)
        axis.set_xlabel("事后匹配 Known Recall")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1")
    axes[-1].legend(loc="best", fontsize=9)
    figure.suptitle("KIR=0.50：相同 Known 覆盖下的检测前沿", fontsize=14)
    atomic_figure(figure, figure_path)


def score_quantile_plot(quantiles: pd.DataFrame, figure_path: Path) -> None:
    selected = quantiles[quantiles["kir"].eq(0.50)]
    summary = selected.groupby(["dataset", "method", "sample_type"], as_index=False)[
        ["q10", "q25", "q50", "q75", "q90"]
    ].mean()
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 4.4), sharey=False, constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        rows = summary[summary["dataset"].eq(dataset)]
        positions = {"s2c_trainable_k1": 0.0, "mogb_minilm_fair": 1.0}
        for method in METHODS:
            for type_index, sample_type in enumerate(("known", "oos")):
                row = rows[rows["method"].eq(method) & rows["sample_type"].eq(sample_type)].iloc[0]
                x = positions[method] + (-0.12 if type_index == 0 else 0.12)
                color = "#999999" if sample_type == "known" else "#D55E00"
                axis.vlines(x, row["q10"], row["q90"], color=color, linewidth=2)
                axis.vlines(x, row["q25"], row["q75"], color=color, linewidth=7, alpha=0.7)
                axis.scatter([x], [row["q50"]], color="black", s=18, zorder=3)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        axis.set_xticks([0, 1], ["S2C", "MOGB"])
        axis.set_title(dataset)
        axis.set_ylabel("归一化 OOS 分数")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("KIR=0.50：Known/OOS 分数分布（灰=Known，橙=OOS）", fontsize=14)
    atomic_figure(figure, figure_path)


def calibration_gap_heatmap(cells: pd.DataFrame, figure_path: Path) -> None:
    summary = cells.groupby(["dataset", "kir", "method"], as_index=False)["default_to_oracle_gap"].mean()
    matrix = np.zeros((len(DATASETS) * len(KIRS), len(METHODS)))
    labels: list[str] = []
    for row_index, (dataset, kir) in enumerate((d, k) for d in DATASETS for k in KIRS):
        labels.append(f"{dataset}\nKIR={kir:.2f}")
        for method_index, method in enumerate(METHODS):
            matrix[row_index, method_index] = float(
                summary[
                    summary["dataset"].eq(dataset)
                    & summary["kir"].eq(kir)
                    & summary["method"].eq(method)
                ]["default_to_oracle_gap"].iloc[0]
            )
    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = axis.imshow(matrix, cmap="YlOrRd", vmin=0, aspect="auto")
    axis.set_xticks(range(len(METHODS)), ["S2C Trainable K=1", "MOGB-Fair"])
    axis.set_yticks(range(len(labels)), labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column] * 100:.1f}pp", ha="center", va="center")
    axis.set_title("默认阈值到事后最优 OOS F1 的校准缺口")
    figure.colorbar(image, ax=axis, label="OOS F1 gap")
    atomic_figure(figure, figure_path)


def ranking_workpoint_scatter(effects: pd.DataFrame, figure_path: Path) -> None:
    auroc = effects[effects["metric"].eq("auroc")][["dataset", "kir", "mean_delta"]].rename(
        columns={"mean_delta": "delta_auroc"}
    )
    f1 = effects[effects["metric"].eq("oos_f1")][["dataset", "kir", "mean_delta"]].rename(
        columns={"mean_delta": "delta_oos_f1"}
    )
    frame = auroc.merge(f1, on=["dataset", "kir"], validate="one_to_one")
    markers = {0.25: "o", 0.50: "s", 0.75: "^"}
    dataset_colors = {"banking77": "#0072B2", "clinc150": "#CC79A7", "stackoverflow": "#D55E00"}
    figure, axis = plt.subplots(figsize=(7.5, 5.7), constrained_layout=True)
    for _, row in frame.iterrows():
        axis.scatter(
            row["delta_auroc"],
            row["delta_oos_f1"],
            s=90,
            marker=markers[float(row["kir"])],
            color=dataset_colors[str(row["dataset"])],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.annotate(
            f"{row['dataset']} {row['kir']:.2f}",
            (row["delta_auroc"], row["delta_oos_f1"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("AUROC差值（S2C - MOGB）")
    axis.set_ylabel("默认 OOS F1差值（S2C - MOGB）")
    axis.set_title("排序能力与默认工作点优势是否一致")
    axis.grid(alpha=0.2)
    atomic_figure(figure, figure_path)


def render_report(cells: pd.DataFrame, effects: pd.DataFrame, matched: pd.DataFrame, manifest_hash: str) -> str:
    piv = cells.pivot_table(index=["dataset", "kir", "seed"], columns="method", values=[
        "oos_f1", "auroc", "aupr_oos", "oracle_oos_f1", "default_to_oracle_gap", "known_recall"
    ])
    delta = piv.xs("s2c_trainable_k1", axis=1, level=1) - piv.xs("mogb_minilm_fair", axis=1, level=1)
    joint_win = int(((delta["oos_f1"] > 0) & (delta["auroc"] > 0)).sum())
    default_win = int((delta["oos_f1"] > 0).sum())
    ranking_win = int((delta["auroc"] > 0).sum())
    method_means = cells.groupby("method")[[
        "oos_f1", "auroc", "aupr_oos", "oracle_oos_f1", "default_to_oracle_gap", "known_recall"
    ]].mean()
    matched80 = matched[matched["target_known_recall"].eq(0.80)].groupby("method")[[
        "oos_f1", "false_accept_rate", "known_recall"
    ]].mean()
    table = method_means.rename(index=METHOD_LABELS).to_markdown(floatfmt=".4f")
    matched_table = matched80.rename(index=METHOD_LABELS).to_markdown(floatfmt=".4f")
    per_unit = effects[effects["metric"].isin(["oos_f1", "auroc", "aupr_oos", "oracle_oos_f1"])].copy()
    per_unit["mean_delta_pp"] = per_unit["mean_delta"] * 100.0
    effect_table = per_unit.pivot(index=["dataset", "kir"], columns="metric", values="mean_delta_pp").reset_index()
    effect_markdown = effect_table.to_markdown(index=False, floatfmt="+.2f")
    return f"""# S2C 与 MOGB-Fair 工作点和排序能力归因 V1

> 本阶段只重放已冻结的 45 个同 split 配对单元。默认阈值结果是正式已完成结果；oracle threshold 和
> matched-known-recall 使用 test 标签，只作为事后机制诊断，严禁用于选择模型、阈值或论文主结果。

## 核心结论

1. 默认 OOS F1 上 S2C 胜出 `{default_win}/45` 个单元，阈值无关 AUROC 上胜出 `{ranking_win}/45` 个单元；
   两者同时胜出的单元为 `{joint_win}/45`。因此当前优势不只是阈值偏移，还包含分数排序质量差异。
2. 45 单元平均 AUROC 差值为 `{delta['auroc'].mean() * 100:+.2f}pp`，AUPR-OOS 差值为
   `{delta['aupr_oos'].mean() * 100:+.2f}pp`；默认 OOS F1 差值为 `{delta['oos_f1'].mean() * 100:+.2f}pp`。
3. 事后最优阈值下，S2C 相对 MOGB 的 OOS F1 上限仍相差
   `{delta['oracle_oos_f1'].mean() * 100:+.2f}pp`。这说明仅移动 MOGB 默认 mean-radius 工作点不能关闭全部差距。
4. MOGB 的默认工作点更保守，但它的 default-to-oracle gap 平均为
   `{method_means.loc['mogb_minilm_fair', 'default_to_oracle_gap'] * 100:.2f}pp`；S2C 为
   `{method_means.loc['s2c_trainable_k1', 'default_to_oracle_gap'] * 100:.2f}pp`。该差距量化了两者的校准损失。
5. 在事后匹配约 80% Known coverage 时，S2C 与 MOGB 的 OOS F1/false acceptance 见下表；这只用于
   判断同覆盖前沿，不构成新的正式指标。

## 45 单元总体均值

{table}

## 事后匹配 80% Known coverage

{matched_table}

## 每数据集与 KIR 的配对差值（百分点）

{effect_markdown}

## 六张机制图

1. `threshold_free_delta_heatmap.png`：AUROC、AUPR-OOS 和分数分离度的配对差值。
2. `default_vs_oracle_oos_f1.png`：默认阈值与事后 OOS F1 上限。
3. `matched_known_recall_frontier.png`：KIR=.50 相同 Known coverage 下的 OOS F1 前沿。
4. `score_quantiles_kir050.png`：Known/OOS 分数的10%--90%分位区间。
5. `calibration_gap_heatmap.png`：默认工作点离事后最优阈值的距离。
6. `ranking_vs_workpoint.png`：AUROC优势与默认OOS F1优势是否同步。

## 解释边界

- `S2C-Trainable-K1` 与 `MOGB-MiniLM-Fair` 共享 dataset/KIR/seed 和 evaluator，但表示与边界合同不同。
- 该结果解释当前公平组件差距；不能替代完整 BERT MOGB、历史 Cascade、ADB/DA-ADB 或 DCLOOS 主表。
- oracle 和 matched-coverage 结果是测试敏感性分析，不能进入任何参数选择或正式方法声明。
- Manifest SHA256：`{manifest_hash}`。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, source_hashes = load_predictions()
    cells, effects, matched, curves, quantiles = build_tables(frames)

    output_files = {
        "cell_metrics.csv": cells,
        "paired_effects.csv": effects,
        "matched_known_recall.csv": matched,
        "operating_curves.csv": curves,
        "score_quantiles.csv": quantiles,
    }
    for name, frame in output_files.items():
        atomic_csv(frame, args.output_dir / name)

    figure_paths = {
        "threshold_free_delta_heatmap.png": lambda path: heatmap_delta(effects, path),
        "default_vs_oracle_oos_f1.png": lambda path: default_oracle_bars(cells, path),
        "matched_known_recall_frontier.png": lambda path: matched_coverage_frontier(matched, path),
        "score_quantiles_kir050.png": lambda path: score_quantile_plot(quantiles, path),
        "calibration_gap_heatmap.png": lambda path: calibration_gap_heatmap(cells, path),
        "ranking_vs_workpoint.png": lambda path: ranking_workpoint_scatter(effects, path),
    }
    for name, builder in figure_paths.items():
        builder(args.figure_dir / name)

    output_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in [
            *(args.output_dir / name for name in output_files),
            *(args.figure_dir / name for name in figure_paths),
        ]
    }
    manifest = {
        "stage": "ANALYSIS_S2C_MOGB_OPERATING_CURVE_ATTRIBUTION_V1",
        "protocol_version": "protocol_v2_textoir_v1",
        "status": "complete",
        "analysis_only": True,
        "test_labels_used_for_posthoc_diagnostics": True,
        "formal_threshold": DEFAULT_THRESHOLD,
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "paired_cells": 45,
        "method_cells": int(len(cells)),
        "matched_coverage_rows": int(len(matched)),
        "operating_curve_rows": int(len(curves)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
    }
    manifest_path = args.output_dir / "MANIFEST.json"
    atomic_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", manifest_path)
    manifest_hash = sha256(manifest_path)
    atomic_text(render_report(cells, effects, matched, manifest_hash), args.report)
    atomic_text(
        "# S2C--MOGB operating-curve attribution closeout\n\n"
        f"- status: complete\n- paired cells: 45\n- method cells: {len(cells)}\n"
        f"- operating curve rows: {len(curves)}\n- failed units: 0\n"
        f"- manifest SHA256: `{manifest_hash}`\n"
        "- scope: analysis-only; no training or formal threshold selection\n",
        args.output_dir / "CLOSEOUT.md",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "paired_cells": 45,
                "method_cells": len(cells),
                "paired_effect_rows": len(effects),
                "matched_coverage_rows": len(matched),
                "operating_curve_rows": len(curves),
                "figures": len(figure_paths),
                "manifest_sha256": manifest_hash,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
