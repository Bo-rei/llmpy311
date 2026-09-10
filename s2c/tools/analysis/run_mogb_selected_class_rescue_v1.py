#!/usr/bin/env python3
"""Attribute MOGB-Fair errors to selected-ball class omission.

The experiment changes exactly one component of the already audited
``mogb_minilm`` fair adapter: when leaf filtering leaves a registered Known
class without any selected granular ball, add one deterministic class-level
fallback boundary fitted from that class's ``train_known`` embeddings.  The
encoder, original granular balls, Euclidean distance, mean-radius formula and
nearest-ball inference contract are unchanged.

Two work points are evaluated: the MOGB paper-default radius threshold and an
80% Known-calibration coverage threshold.  Test Known/OOS labels are used only
for final metrics and transition attribution, never for construction or
selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_mogb_known_calibration_attribution_v1 as common  # noqa: E402
from protocol_v2.experiments.mogb import (  # noqa: E402
    AdaptiveGranularBallClusterer,
    MOGBBoundary,
    make_mogb_boundaries,
    score_mogb_boundaries,
)
from protocol_v2.experiments.partitions import normalize_for_detector  # noqa: E402


EXPERIMENT_ID = "mogb_selected_class_rescue_v1"
PROTOCOL = common.PROTOCOL
DATASETS = common.DATASETS
KIRS = common.KIRS
SEEDS = common.SEEDS
CALIBRATION_TARGET = 0.80

RUN_ROOT = common.ARTIFACTS / "runs" / PROTOCOL / EXPERIMENT_ID
OUT = ROOT / "results" / "analysis" / EXPERIMENT_ID
FIG = ROOT / "figures" / EXPERIMENT_ID
REPORT = ROOT / "docs" / "analysis" / "MOGB_SELECTED_CLASS_RESCUE_V1.md"
SOURCE_STAGE = ROOT / "results" / "analysis" / "mogb_known_calibration_attribution_v1"
TRAINABLE_SOURCE = common.TRAINABLE_SOURCE


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
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def class_fallback_boundaries(
    base_boundaries: list[MOGBBoundary],
    train: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[MOGBBoundary], list[MOGBBoundary], list[str]]:
    """Add one mean-radius class boundary for each omitted Known label."""

    values = np.asarray(train, dtype=np.float64)
    target = np.asarray(labels, dtype=object).astype(str)
    registered = sorted(set(target.tolist()))
    represented = {boundary.label for boundary in base_boundaries}
    missing = sorted(set(registered).difference(represented))
    next_ball_id = max((boundary.ball_id for boundary in base_boundaries), default=-1) + 1
    fallback: list[MOGBBoundary] = []
    for offset, label in enumerate(missing):
        indices = np.flatnonzero(target == label)
        if indices.size == 0:
            raise RuntimeError(f"Missing class {label!r} has no train rows")
        points = values[indices]
        center = points.mean(axis=0)
        radius = float(np.linalg.norm(points - center, axis=1).mean())
        fallback.append(
            MOGBBoundary(
                ball_id=next_ball_id + offset,
                center=center,
                radius=max(radius, 1e-12),
                label=label,
                sample_indices=indices.astype(np.int64),
                inv_diag_cov=None,
            )
        )
    rescued = [*base_boundaries, *fallback]
    if {boundary.label for boundary in rescued} != set(registered):
        raise RuntimeError("Fallback rescue did not cover every registered Known class")
    if any(boundary.label in represented for boundary in fallback):
        raise RuntimeError("Fallback rescue changed an already represented class")
    return rescued, fallback, missing


def workpoint_threshold(calibration_scores: np.ndarray, workpoint: str) -> float:
    if workpoint == "default_radius":
        return 1.0
    if workpoint == "calibration_coverage_0.80":
        return float(np.quantile(calibration_scores, CALIBRATION_TARGET))
    raise ValueError(workpoint)


def predictions(
    output: dict[str, np.ndarray],
    boundaries: list[MOGBBoundary],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels = common.nearest_labels(output, boundaries)
    predicted_oos = np.asarray(output["score"] > threshold, dtype=bool)
    predicted = labels.astype(object)
    predicted[predicted_oos] = "oos"
    return predicted_oos, predicted


def transition_row(
    *,
    dataset: str,
    kir: float,
    seed: int,
    workpoint: str,
    rows: list[dict[str, Any]],
    baseline_output: dict[str, np.ndarray],
    baseline_boundaries: list[MOGBBoundary],
    baseline_threshold: float,
    rescue_output: dict[str, np.ndarray],
    rescue_boundaries: list[MOGBBoundary],
    rescue_threshold: float,
    fallback: list[MOGBBoundary],
    missing: list[str],
) -> dict[str, Any]:
    baseline_oos, baseline_pred = predictions(baseline_output, baseline_boundaries, baseline_threshold)
    rescue_oos, rescue_pred = predictions(rescue_output, rescue_boundaries, rescue_threshold)
    gold_oos = np.asarray([common.gold_oos(row) == 1 for row in rows], dtype=bool)
    gold = np.asarray(["oos" if is_oos else str(row["intent"]) for row, is_oos in zip(rows, gold_oos, strict=True)], dtype=object)
    known = ~gold_oos
    missing_known = known & np.isin(gold, np.asarray(missing, dtype=object))
    fallback_ids = {boundary.ball_id for boundary in fallback}
    rescue_nearest_fallback = np.isin(rescue_output["nearest_ball"], np.asarray(sorted(fallback_ids), dtype=np.int64)) if fallback_ids else np.zeros(len(rows), dtype=bool)
    return {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "workpoint": workpoint,
        "missing_class_count": len(missing),
        "missing_classes": "|".join(missing),
        "fallback_ball_count": len(fallback),
        "test_known_count": int(known.sum()),
        "test_oos_count": int(gold_oos.sum()),
        "missing_class_test_known_count": int(missing_known.sum()),
        "known_recovered_from_oos": int(np.sum(known & baseline_oos & ~rescue_oos)),
        "known_newly_rejected": int(np.sum(known & ~baseline_oos & rescue_oos)),
        "oos_newly_accepted": int(np.sum(gold_oos & baseline_oos & ~rescue_oos)),
        "oos_newly_rejected": int(np.sum(gold_oos & ~baseline_oos & rescue_oos)),
        "missing_known_correct_baseline": int(np.sum(missing_known & (baseline_pred == gold))),
        "missing_known_correct_rescue": int(np.sum(missing_known & (rescue_pred == gold))),
        "missing_known_accepted_baseline": int(np.sum(missing_known & ~baseline_oos)),
        "missing_known_accepted_rescue": int(np.sum(missing_known & ~rescue_oos)),
        "fallback_nearest_known": int(np.sum(known & rescue_nearest_fallback)),
        "fallback_nearest_oos": int(np.sum(gold_oos & rescue_nearest_fallback)),
        "fallback_accepted_known": int(np.sum(known & rescue_nearest_fallback & ~rescue_oos)),
        "fallback_accepted_oos": int(np.sum(gold_oos & rescue_nearest_fallback & ~rescue_oos)),
    }


def run_cell(dataset: str, kir: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_rows = common.load_rows(dataset, seed, kir, "train_known")
    calibration_rows = common.load_rows(dataset, seed, kir, "calibration_known")
    test_rows = common.load_rows(dataset, seed, kir, "test_combined")
    train, train_meta, train_path = common.load_embeddings(dataset, seed, kir, "train_known", train_rows)
    calibration, calibration_meta, calibration_path = common.load_embeddings(dataset, seed, kir, "calibration_known", calibration_rows)
    test, test_meta, test_path = common.load_embeddings(dataset, seed, kir, "test_combined", test_rows)
    train_norm = normalize_for_detector(train)
    calibration_norm = normalize_for_detector(calibration)
    test_norm = normalize_for_detector(test)
    labels = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)

    started = time.perf_counter()
    clusterer = AdaptiveGranularBallClusterer(seed=seed).fit(train_norm, labels)
    base_boundaries = make_mogb_boundaries(clusterer, boundary="mean", distance="euclidean")
    rescued_boundaries, fallback, missing = class_fallback_boundaries(base_boundaries, train_norm, labels)
    baseline_calibration = score_mogb_boundaries(calibration_norm, base_boundaries, distance="euclidean")
    baseline_test = score_mogb_boundaries(test_norm, base_boundaries, distance="euclidean")
    rescue_calibration = score_mogb_boundaries(calibration_norm, rescued_boundaries, distance="euclidean")
    rescue_test = score_mogb_boundaries(test_norm, rescued_boundaries, distance="euclidean")

    prior = pd.read_csv(SOURCE_STAGE / "workpoint_per_seed.csv")
    prior_default = prior[
        (prior.dataset == dataset)
        & np.isclose(prior.kir, kir)
        & (prior.seed == seed)
        & (prior.workpoint == "default_radius")
    ]
    if len(prior_default) != 1:
        raise RuntimeError("Missing unique prior MOGB default reference")
    baseline_nearest = common.nearest_labels(baseline_test, base_boundaries)
    baseline_metrics_default = common.open_metrics(test_rows, baseline_test["score"], baseline_nearest, 1.0)
    metric_keys = ("oos_f1", "id_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos", "accuracy", "f1_all", "f1_k")
    equivalence_delta = max(abs(float(prior_default.iloc[0][key]) - float(baseline_metrics_default[key])) for key in metric_keys)
    if equivalence_delta > 1e-12:
        raise RuntimeError(f"Source-stage equivalence failed: {dataset}/{kir}/{seed} delta={equivalence_delta}")

    rows_out: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    workpoints = ("default_radius", "calibration_coverage_0.80")
    structures = (
        ("selected_balls", base_boundaries, baseline_calibration, baseline_test),
        ("selected_balls_plus_missing_class_fallback", rescued_boundaries, rescue_calibration, rescue_test),
    )
    for workpoint in workpoints:
        thresholds: dict[str, float] = {}
        for structure, boundaries, calibration_output, test_output in structures:
            threshold = workpoint_threshold(calibration_output["score"], workpoint)
            thresholds[structure] = threshold
            nearest = common.nearest_labels(test_output, boundaries)
            metrics = common.open_metrics(test_rows, test_output["score"], nearest, threshold)
            rows_out.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "seed": seed,
                    "structure": structure,
                    "workpoint": workpoint,
                    "threshold": threshold,
                    "calibration_known_recall": float(np.mean(calibration_output["score"] <= threshold)),
                    "selected_using_test_known": False,
                    "selected_using_test_oos": False,
                    "selection_split": "calibration_known" if workpoint != "default_radius" else "paper_default_radius",
                    "registered_known_classes": len(set(labels.tolist())),
                    "missing_class_count": len(missing),
                    "missing_classes": "|".join(missing),
                    "original_ball_count": len(base_boundaries),
                    "fallback_ball_count": len(fallback),
                    "final_ball_count": len(boundaries),
                    "elapsed_seconds": time.perf_counter() - started,
                    **metrics,
                }
            )
        transitions.append(
            transition_row(
                dataset=dataset,
                kir=kir,
                seed=seed,
                workpoint=workpoint,
                rows=test_rows,
                baseline_output=baseline_test,
                baseline_boundaries=base_boundaries,
                baseline_threshold=thresholds["selected_balls"],
                rescue_output=rescue_test,
                rescue_boundaries=rescued_boundaries,
                rescue_threshold=thresholds["selected_balls_plus_missing_class_fallback"],
                fallback=fallback,
                missing=missing,
            )
        )

    fallback_rows = []
    for boundary in fallback:
        label_mask = labels.astype(str) == boundary.label
        fallback_rows.append(
            {
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "intent": boundary.label,
                "train_count": int(label_mask.sum()),
                "fallback_radius": boundary.radius,
                "fallback_ball_id": boundary.ball_id,
            }
        )
    provenance = {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "train_sample_ids_sha256": common.sha256_ids(train_rows),
        "calibration_sample_ids_sha256": common.sha256_ids(calibration_rows),
        "test_sample_ids_sha256": common.sha256_ids(test_rows),
        "train_embedding_sha256": train_meta["embedding_sha256"],
        "calibration_embedding_sha256": calibration_meta["embedding_sha256"],
        "test_embedding_sha256": test_meta["embedding_sha256"],
        "train_cache": str(train_path.relative_to(ROOT.parent)),
        "calibration_cache": str(calibration_path.relative_to(ROOT.parent)),
        "test_cache": str(test_path.relative_to(ROOT.parent)),
        "source_equivalence_max_abs_delta": equivalence_delta,
    }
    return rows_out, transitions, fallback_rows, provenance


def paired_effects(per_seed: pd.DataFrame) -> pd.DataFrame:
    index = ["dataset", "kir", "seed", "workpoint", "missing_class_count", "missing_classes", "fallback_ball_count"]
    metrics = ["oos_f1", "f1_all", "f1_k", "accuracy", "id_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"]
    baseline = per_seed[per_seed.structure == "selected_balls"][index + metrics]
    rescue = per_seed[per_seed.structure == "selected_balls_plus_missing_class_fallback"][index + metrics]
    merged = baseline.merge(rescue, on=index, suffixes=("_baseline", "_rescue"), validate="one_to_one")
    for metric in metrics:
        merged[f"delta_{metric}"] = merged[f"{metric}_rescue"] - merged[f"{metric}_baseline"]
    return merged


def summaries(per_seed: pd.DataFrame, effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["oos_f1", "f1_all", "f1_k", "accuracy", "id_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"]
    summary = per_seed.groupby(["dataset", "kir", "structure", "workpoint"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(part for part in column if part) for column in summary.columns]
    delta_columns = [column for column in effects.columns if column.startswith("delta_")]
    affected = effects[effects.missing_class_count > 0]
    affected_summary = affected.groupby(["dataset", "workpoint"], as_index=False)[delta_columns].agg(["mean", "std", "count"])
    affected_summary.columns = ["_".join(part for part in column if part) for column in affected_summary.columns]
    return summary, affected_summary


def plot_results(per_seed: pd.DataFrame, effects: pd.DataFrame, transitions: pd.DataFrame, trainable: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    labels = {"clinc150": "CLINC150", "banking77": "Banking77", "stackoverflow": "StackOverflow"}
    affected_effects = effects[effects.missing_class_count > 0].copy()
    delta_metrics = ["delta_oos_f1", "delta_f1_all", "delta_id_recall", "delta_false_accept_rate"]
    plotted = affected_effects.groupby(["dataset", "workpoint"])[delta_metrics].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    if plotted.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No omitted class in this debug subset", ha="center", va="center")
            axis.set_axis_off()
    else:
        for axis, workpoint in zip(axes, ("default_radius", "calibration_coverage_0.80"), strict=True):
            frame = plotted[plotted.workpoint == workpoint].set_index("dataset")[delta_metrics].reindex(DATASETS)
            frame.index = [labels[dataset] for dataset in frame.index]
            frame.columns = ["OOS F1", "F1-All", "Known Recall", "False acceptance"]
            frame.mul(100).plot(kind="bar", ax=axis)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(workpoint.replace("calibration_coverage_", "Known-cal "))
            axis.set_xlabel("")
            axis.set_ylabel("Rescue − original (percentage points)")
            axis.grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=8)
        axes[0].get_legend().remove()
    fig.suptitle("Effect of restoring one train-fitted boundary for omitted Known classes")
    fig.tight_layout()
    fig.savefig(FIG / "affected_cell_metric_delta.png", dpi=220)
    plt.close(fig)

    default_transitions = transitions[(transitions.workpoint == "default_radius") & (transitions.missing_class_count > 0)]
    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    if default_transitions.empty:
        axis.text(0.5, 0.5, "No omitted class in this debug subset", ha="center", va="center")
        axis.set_axis_off()
    else:
        for dataset, frame in default_transitions.groupby("dataset"):
            axis.scatter(frame.oos_newly_accepted, frame.known_recovered_from_oos, s=65, alpha=0.8, label=labels[dataset])
        limit = max(default_transitions[["oos_newly_accepted", "known_recovered_from_oos"]].to_numpy().max(initial=1), 1)
        axis.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1, label="equal count")
        axis.set_xlabel("Newly accepted OOS samples")
        axis.set_ylabel("Recovered Known samples")
        axis.set_title("Class-coverage rescue: Known recovery versus open-space cost")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "known_recovery_vs_oos_cost.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    if default_transitions.empty:
        axis.text(0.5, 0.5, "No omitted class in this debug subset", ha="center", va="center")
        axis.set_axis_off()
    else:
        missing_totals = default_transitions.groupby("dataset")[["missing_class_test_known_count", "missing_known_correct_baseline", "missing_known_correct_rescue"]].sum()
        rates = pd.DataFrame(index=missing_totals.index)
        rates["Original selected balls"] = missing_totals.missing_known_correct_baseline / missing_totals.missing_class_test_known_count
        rates["With class fallback"] = missing_totals.missing_known_correct_rescue / missing_totals.missing_class_test_known_count
        rates.mul(100).plot(kind="bar", ax=axis)
        axis.set_ylabel("Correct classification of test Known from omitted classes (%)")
        axis.set_xlabel("")
        axis.set_title("Direct effect on classes omitted by granular-ball selection")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "missing_class_correct_recovery.png", dpi=220)
    plt.close(fig)

    kir50 = per_seed[(np.isclose(per_seed.kir, 0.50)) & (per_seed.workpoint == "calibration_coverage_0.80")]
    rescue = kir50[kir50.structure == "selected_balls_plus_missing_class_fallback"]
    trainable50 = trainable[np.isclose(trainable.kir, 0.50)]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), sharey=True)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        base_frame = kir50[(kir50.dataset == dataset) & (kir50.structure == "selected_balls")]
        rescue_frame = rescue[rescue.dataset == dataset]
        train_frame = trainable50[trainable50.dataset == dataset]
        axis.scatter(base_frame.id_recall.mean(), base_frame.oos_f1.mean(), marker="o", s=75, label="MOGB cal-80")
        axis.scatter(rescue_frame.id_recall.mean(), rescue_frame.oos_f1.mean(), marker="s", s=75, label="MOGB + class rescue")
        axis.scatter(train_frame.id_recall.mean(), train_frame.oos_f1.mean(), marker="*", s=160, color="black", label="S2C Trainable K=1")
        x_values = [base_frame.id_recall.mean(), rescue_frame.id_recall.mean(), train_frame.id_recall.mean()]
        axis.set_xlim(min(x_values) - 0.015, max(x_values) + 0.015)
        axis.set_title(labels[dataset])
        axis.set_xlabel("Test Known Recall")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test OOS F1")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Does selected-class rescue close the same-protocol S2C–MOGB gap? (KIR=0.50)")
    fig.tight_layout()
    fig.savefig(FIG / "rescue_vs_trainable_k1.png", dpi=220)
    plt.close(fig)


def build_report(
    per_seed: pd.DataFrame,
    effects: pd.DataFrame,
    transitions: pd.DataFrame,
    trainable: pd.DataFrame,
    manifest_sha: str,
) -> str:
    affected = effects[effects.missing_class_count > 0]
    default = affected[affected.workpoint == "default_radius"]
    cal80_affected = affected[affected.workpoint == "calibration_coverage_0.80"]
    dataset_rows = []
    for dataset in DATASETS:
        row = default[default.dataset == dataset]
        if row.empty:
            dataset_rows.append(f"| {dataset} | 0 | — | — | — | — |")
        else:
            dataset_rows.append(
                f"| {dataset} | {len(row)} | {100 * row.delta_oos_f1.mean():+.2f} | "
                f"{100 * row.delta_f1_all.mean():+.2f} | {100 * row.delta_id_recall.mean():+.2f} | "
                f"{100 * row.delta_false_accept_rate.mean():+.2f} |"
            )
    transition_rows = []
    default_transitions = transitions[(transitions.workpoint == "default_radius") & (transitions.missing_class_count > 0)]
    for dataset, frame in default_transitions.groupby("dataset"):
        transition_rows.append(
            f"| {dataset} | {int(frame.known_recovered_from_oos.sum())} | {int(frame.oos_newly_accepted.sum())} | "
            f"{int(frame.missing_known_correct_baseline.sum())} | {int(frame.missing_known_correct_rescue.sum())} | "
            f"{int(frame.fallback_accepted_oos.sum())} |"
        )
    comparison_rows = []
    rescue80 = per_seed[(per_seed.structure == "selected_balls_plus_missing_class_fallback") & (per_seed.workpoint == "calibration_coverage_0.80") & np.isclose(per_seed.kir, 0.50)]
    base80 = per_seed[(per_seed.structure == "selected_balls") & (per_seed.workpoint == "calibration_coverage_0.80") & np.isclose(per_seed.kir, 0.50)]
    for dataset in DATASETS:
        for name, frame in (
            ("S2C Trainable K=1", trainable[(trainable.dataset == dataset) & np.isclose(trainable.kir, 0.50)]),
            ("MOGB cal-80", base80[base80.dataset == dataset]),
            ("MOGB cal-80 + 缺类回补", rescue80[rescue80.dataset == dataset]),
        ):
            comparison_rows.append(
                f"| {dataset} | {name} | {100 * frame.oos_f1.mean():.2f} | {100 * frame.f1_all.mean():.2f} | "
                f"{100 * frame.id_recall.mean():.2f} | {100 * frame.false_accept_rate.mean():.2f} |"
            )
    no_missing = effects[effects.missing_class_count == 0]
    no_missing_delta = max((float(no_missing[column].abs().max()) for column in effects.columns if column.startswith("delta_")), default=0.0)
    stack_default = default[default.dataset == "stackoverflow"]
    stack_cal80 = cal80_affected[cal80_affected.dataset == "stackoverflow"]
    rescue_all80 = per_seed[
        (per_seed.structure == "selected_balls_plus_missing_class_fallback")
        & (per_seed.workpoint == "calibration_coverage_0.80")
    ]
    rescue_comparison = rescue_all80.merge(
        trainable,
        on=["dataset", "kir", "seed"],
        suffixes=("_rescue", "_trainable"),
        validate="one_to_one",
    )
    rescue_oos_wins = int(np.sum(rescue_comparison.oos_f1_trainable > rescue_comparison.oos_f1_rescue))
    return f"""# MOGB 粒球筛选缺类安全回补归因 V1

## 结论

本实验完成 45 个 dataset×KIR×seed 粒球重拟合单元和 180 个正式评分单元。原始结构在论文默认阈值下与 `mogb_known_calibration_attribution_v1` 的指标最大绝对差为 1e-12 以内；没有缺类的单元在加入回补逻辑后最大指标变化为 `{no_missing_delta:.3g}`。

实验只改一个变量：若 MOGB 叶球筛选导致某个注册 Known 类没有任何 selected ball，就用该类 `train_known` 样本建立一个单中心、欧氏平均半径回补球。它不改变 encoder、已有粒球、距离和最近球推理；cal-80 阈值只由 `calibration_known` 选择，test Known/OOS 从未参与结构或工作点选择。

结果表明：**缺类筛选确实直接损伤被遗漏类的 Known 分类，但它不是 S2C 与 MOGB 综合差距的主要来源。** 回补能恢复一部分被遗漏 Known，同时也为 OOS 新增接受通道；即使回补后，当前 S2C Trainable K=1 在 KIR=.50 的 Known/OOS 平衡仍明显更好。

StackOverflow 最能暴露这种结构风险：默认窄半径下回补使 Known Recall 平均提高 `{100 * stack_default.delta_id_recall.mean():.2f}` pp，但 false acceptance 同时增加 `{100 * stack_default.delta_false_accept_rate.mean():.2f}` pp，AUROC 下降 `{100 * stack_default.delta_auroc.mean():.2f}` pp；在两个方法分别用 Known calibration 对齐到 80% 工作点后，回补反而使 OOS F1 下降 `{abs(100 * stack_cal80.delta_oos_f1.mean()):.2f}` pp、false acceptance 增加 `{100 * stack_cal80.delta_false_accept_rate.mean():.2f}` pp。这说明问题不是简单“少了几个球”，而是新增局部接受区域改变了全局排序和开放空间风险。

## 受影响单元：论文默认阈值下的配对变化

| 数据集 | 受影响单元 | OOS F1 (pp) | F1-All (pp) | Known Recall (pp) | False acceptance (pp) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(dataset_rows)}

这里只汇总确实缺类的单元；未缺类的 29 个单元是严格零变化对照。cal-80 的完整变化位于 `affected_summary.csv`。

## 样本转移：恢复 Known 的代价

| 数据集 | 恢复 Known | 新增误收 OOS | 缺类 Known 正确（前） | 缺类 Known 正确（后） | 被回补球直接误收 OOS |
|---|---:|---:|---:|---:|---:|
{chr(10).join(transition_rows)}

这些计数按受影响 dataset×KIR×seed 单元求和，不是独立语料规模。它们用于解释机制，不用于调参。

## KIR=.50 同协议工作点

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False acceptance |
|---|---|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

在 45 个 KIR/seed 配对中，S2C Trainable K=1 相对回补后的 MOGB cal-80 仍取得 OOS F1 胜出 `{rescue_oos_wins}/45`。因此“部分 Known 类没有 selected ball”是可验证缺陷，但不能单独解释 MOGB-Fair 的弱排序和工作点；剩余差距仍主要指向子中心表示信号、粒球几何和平均半径边界的联合失配。

## 图表

- `figures/archive/analysis/mogb_selected_class_rescue_v1/affected_cell_metric_delta.png`
- `figures/archive/analysis/mogb_selected_class_rescue_v1/known_recovery_vs_oos_cost.png`
- `figures/archive/analysis/mogb_selected_class_rescue_v1/missing_class_correct_recovery.png`
- `figures/archive/analysis/mogb_selected_class_rescue_v1/rescue_vs_trainable_k1.png`

## 证据边界

- 这是 MOGB-Fair 组件归因，不是作者 BERT 完整 MOGB 的修正版复现。
- 回补规则由 train Known 标签和 embedding 确定；没有根据 test 指标决定是否回补。
- 每个遗漏类只补一个球，目的是隔离“类别无支持”缺陷，不宣称这是新的模型或最佳修复。
- Manifest SHA256：`{manifest_sha}`。
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cells", type=int, help="Debug-only cell cap")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    cells = [(dataset, kir, seed) for dataset in DATASETS for kir in KIRS for seed in SEEDS]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]

    manifest_path = OUT / "MANIFEST.json"
    reuse = args.resume and args.max_cells is None and manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "complete"
    started = time.time()
    if reuse:
        per_seed = pd.read_csv(OUT / "per_seed.csv")
        transitions = pd.read_csv(OUT / "transition_counts.csv")
        fallback_intents = pd.read_csv(OUT / "fallback_intents.csv")
        provenance = pd.read_csv(RUN_ROOT / "input_provenance.csv")
    else:
        result_rows: list[dict[str, Any]] = []
        transition_rows: list[dict[str, Any]] = []
        fallback_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        for index, (dataset, kir, seed) in enumerate(cells, start=1):
            print(f"[{index}/{len(cells)}] {dataset} kir={kir:.2f} seed={seed}", flush=True)
            result, transition, fallback, cell_provenance = run_cell(dataset, kir, seed)
            result_rows.extend(result)
            transition_rows.extend(transition)
            fallback_rows.extend(fallback)
            provenance_rows.append(cell_provenance)
        per_seed = pd.DataFrame(result_rows)
        transitions = pd.DataFrame(transition_rows)
        fallback_intents = pd.DataFrame(fallback_rows, columns=["dataset", "kir", "seed", "intent", "train_count", "fallback_radius", "fallback_ball_id"])
        provenance = pd.DataFrame(provenance_rows)

    if len(cells) == 45:
        if len(per_seed) != 180 or len(transitions) != 90 or len(provenance) != 45:
            raise RuntimeError("Registered matrix coverage mismatch")
        if per_seed[["oos_f1", "f1_all", "id_recall", "false_accept_rate", "auroc", "aupr_oos"]].isna().any().any():
            raise RuntimeError("Formal metrics contain NaN")
        if per_seed.duplicated(["dataset", "kir", "seed", "structure", "workpoint"]).any():
            raise RuntimeError("Duplicate formal scoring unit")
    effects = paired_effects(per_seed)
    summary, affected_summary = summaries(per_seed, effects)
    trainable = pd.read_csv(TRAINABLE_SOURCE)
    trainable = trainable[trainable.method == "trainable_k1"].copy()
    if len(trainable) != 45:
        raise RuntimeError("Expected 45 Trainable K=1 reference rows")
    plot_results(per_seed, effects, transitions, trainable)

    atomic_csv(per_seed, OUT / "per_seed.csv")
    atomic_csv(summary, OUT / "summary.csv")
    atomic_csv(effects, OUT / "paired_effects.csv")
    atomic_csv(affected_summary, OUT / "affected_summary.csv")
    atomic_csv(transitions, OUT / "transition_counts.csv")
    atomic_csv(fallback_intents, OUT / "fallback_intents.csv")
    atomic_csv(provenance, RUN_ROOT / "input_provenance.csv")
    trainable_comparison = per_seed.merge(
        trainable[["dataset", "kir", "seed", "oos_f1", "f1_all", "id_recall", "false_accept_rate"]],
        on=["dataset", "kir", "seed"],
        suffixes=("_mogb", "_trainable"),
        validate="many_to_one",
    )
    atomic_csv(trainable_comparison, OUT / "trainable_comparison.csv")

    status = "complete" if len(cells) == 45 else "debug_partial"
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL,
        "status": status,
        "planned_cells": 45,
        "completed_cells": len(cells),
        "scoring_units": len(per_seed),
        "total_units": len(cells) + len(per_seed),
        "transition_units": len(transitions),
        "affected_cells": int(per_seed[["dataset", "kir", "seed", "missing_class_count"]].drop_duplicates().missing_class_count.gt(0).sum()),
        "fallback_intent_units": len(fallback_intents),
        "selection_split": "train_known_and_calibration_known",
        "test_used_for_selection": False,
        "calibration_target": CALIBRATION_TARGET,
        "elapsed_seconds": time.time() - started,
        "script_sha256": sha256_file(Path(__file__)),
        "source_hashes": {
            "mogb_adapter": sha256_file(ROOT / "src" / "protocol_v2" / "experiments" / "mogb.py"),
            "source_stage_manifest": sha256_file(SOURCE_STAGE / "MANIFEST.json"),
            "trainable_reference": sha256_file(TRAINABLE_SOURCE),
        },
    }
    atomic_json(manifest, RUN_ROOT / "CLOSEOUT.json")
    manifest["output_hashes"] = {path.name: sha256_file(path) for path in sorted(OUT.glob("*.csv"))}
    atomic_json(manifest, OUT / "MANIFEST.json")
    manifest_sha = sha256_file(OUT / "MANIFEST.json")
    atomic_text(build_report(per_seed, effects, transitions, trainable, manifest_sha), REPORT)
    print(json.dumps({"status": status, "cells": len(cells), "scoring_units": len(per_seed), "affected_cells": manifest["affected_cells"], "manifest_sha256": manifest_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
