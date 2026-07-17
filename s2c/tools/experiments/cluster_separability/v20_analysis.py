#!/usr/bin/env python3
"""v20 第一批“零重跑”分析。

该模块只读取 v19 已冻结的 CSV、Parquet、JSON 和 embedding cache，输出到独立的
``cluster_separability_v20``。它不重新训练 encoder、Gate、Router 或 Expert，
也不修改 v19 任何文件。所有阈值仍来自 validation 产物，test 只用于最终统计。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.cluster_separability.analysis import (
    _centroids,
    _json as _read_json,
    _load_cache,
    semantic_distance,
    validation_bucket_thresholds,
)
from tools.experiments.cluster_separability.protocol import compute_binary_oos_metrics

V19_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v19"
V20_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v20"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
KIRS = (25, 50, 75)
SEEDS = (13, 42, 87)
DISTANCES = ("euclidean", "mahalanobis_diag")
BASELINE_METHODS = ("msp", "energy", "entropy", "knn", "lof")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """按首次出现顺序写列，允许后续诊断增加字段而不破坏旧行。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _read_csv(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _unit(root: Path, phase: str, dataset: str, kir: int, seed: int, distance: str, k: int) -> Path:
    return root / phase / dataset / f"kir{kir}_seed{seed}" / distance / f"k{k}"


def _selected_k(root: Path, dataset: str, kir: int, seed: int, distance: str) -> int:
    rows = _read_csv(root, "selected_k_summary.csv")
    match = rows[
        (rows["dataset"] == dataset)
        & (rows["kir"] == kir)
        & (rows["data_seed"] == seed)
        & (rows["distance"] == distance)
    ]
    if len(match) != 1:
        raise ValueError(f"expected one selected K, got {len(match)} for {dataset}/{kir}/{seed}/{distance}")
    return int(match.iloc[0]["selected_k"])


def selection_reliability(v19_root: Path, output_dir: Path) -> dict[str, int]:
    """计算 validation 选 K 对 test 的排序相关、regret 和安全性。"""

    rows = _read_csv(v19_root, "kir_k_tuned_boundary.csv")
    detail: list[dict[str, Any]] = []
    for (dataset, kir, seed, distance), group in rows.groupby(["dataset", "kir", "data_seed", "distance"]):
        group = group.sort_values("k_gate")
        eligible = group[group["guard_violation"].fillna(False) != True]
        pool = eligible if len(eligible) else group
        selected = pool.sort_values(
            ["validation_oos_f1", "validation_fpr95", "validation_id_recall", "k_gate"],
            ascending=[False, True, False, True],
        ).iloc[0]
        oracle = group.sort_values(["test_oos_f1", "test_fpr95", "test_id_recall", "k_gate"],
                                   ascending=[False, True, False, True]).iloc[0]
        rho = spearmanr(group["validation_oos_f1"], group["test_oos_f1"]).statistic
        k1 = group[group["k_gate"] == 1].iloc[0]
        selected_test = group[group["k_gate"] == int(selected["k_gate"])].iloc[0]
        oracle_test = float(oracle["test_oos_f1"])
        selected_test_f1 = float(selected_test["test_oos_f1"])
        detail.append(
            {
                "dataset": dataset,
                "kir": int(kir),
                "data_seed": int(seed),
                "distance": distance,
                "selected_k": int(selected["k_gate"]),
                "oracle_k": int(oracle["k_gate"]),
                "validation_test_spearman": float(rho) if math.isfinite(float(rho)) else math.nan,
                "test_regret": oracle_test - selected_test_f1,
                "delta_oos_f1_vs_k1": selected_test_f1 - float(k1["test_oos_f1"]),
                "delta_id_recall_vs_k1": float(selected_test["test_id_recall"]) - float(k1["test_id_recall"]),
                "selection_success": int(int(selected["k_gate"]) == int(oracle["k_gate"])),
                "selected_test_oos_f1": selected_test_f1,
                "oracle_test_oos_f1": oracle_test,
            }
        )
    _write_csv(output_dir / "k_selection_reliability.csv", detail)
    frame = pd.DataFrame(detail)
    summary: list[dict[str, Any]] = []
    for dataset, group in frame.groupby("dataset"):
        summary.append(
            {
                "dataset": dataset,
                "cell_count": len(group),
                "spearman_mean": group["validation_test_spearman"].mean(),
                "spearman_std": group["validation_test_spearman"].std(ddof=1),
                "selection_accuracy": group["selection_success"].mean(),
                "regret_mean": group["test_regret"].mean(),
                "regret_std": group["test_regret"].std(ddof=1),
                "safe_vs_k1_rate": (group["delta_oos_f1_vs_k1"] >= 0).mean(),
                "id_recall_delta_mean": group["delta_id_recall_vs_k1"].mean(),
            }
        )
    _write_csv(output_dir / "k_selection_reliability_summary.csv", summary)
    return {"detail_rows": len(detail), "summary_rows": len(summary)}


def _load_unit_inputs(unit: Path, v19_root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    manifest = _read_json(unit / "run_manifest.json")
    data_root = Path(manifest["data_root"])
    rows = {split: _read_json(data_root / "gate" / f"{split}.json") for split in ("train", "val", "test")}
    embeddings = {
        split: _load_cache(v19_root, manifest["config"]["dataset"], int(manifest["config"]["kir"]),
                           int(manifest["config"]["data_seed"]), split, manifest)
        for split in rows
    }
    return manifest, rows, embeddings


def _bucket_masks(v19_root: Path, reference_unit: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """按 K=1 单中心语义距离冻结 near/medium/far 分桶。"""

    manifest, rows, embeddings = _load_unit_inputs(reference_unit, v19_root)
    centers = _centroids(rows["train"], embeddings["train"])
    val_dist = semantic_distance(embeddings["val"], centers)
    q20, q80 = validation_bucket_thresholds(
        [int(row["label"]) for row in rows["val"]], val_dist
    )
    test_dist = semantic_distance(embeddings["test"], centers)
    bucket = np.where(test_dist <= q20, "near", np.where(test_dist <= q80, "medium", "far"))
    labels = np.asarray([int(row["label"]) for row in rows["test"]], dtype=np.int64)
    return bucket, labels, np.asarray([str(row.get("source_id", "")) for row in rows["test"]]), {"q20": q20, "q80": q80}


def _method_score_frame(unit: Path) -> tuple[pd.DataFrame, float]:
    scores = pd.read_parquet(unit / "scores.parquet")
    selection = _read_json(unit / "threshold_selection.json")
    if "selected_operating_point" in selection:
        threshold = float(selection["selected_operating_point"]["threshold"])
    else:
        threshold = 1.0
    return scores, threshold


def near_oos_comparison(v19_root: Path, output_dir: Path) -> dict[str, int]:
    """在同一 OOS 分桶内比较 K=1/K=2/selected K 与五个 Baseline。"""

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        kir, seed, distance = 50, 42, "mahalanobis_diag"
        ref = _unit(v19_root, "fixed", dataset, kir, seed, distance, 1)
        bucket, labels, _, quantiles = _bucket_masks(v19_root, ref)
        configs: list[tuple[str, Path]] = [
            ("k1", ref),
            ("k2", _unit(v19_root, "fixed", dataset, kir, seed, distance, 2)),
            ("selected_k", _unit(v19_root, "tuned", dataset, kir, seed, distance,
                                  _selected_k(v19_root, dataset, kir, seed, distance))),
        ]
        configs.extend((method, v19_root / "baselines" / dataset / f"kir{kir}_seed{seed}" / method)
                       for method in BASELINE_METHODS)
        for method, unit in configs:
            scores, threshold = _method_score_frame(unit)
            # 所有 v19 score 表按 test sample_id 对齐；按 sample_id 合并比依赖文件顺序更安全。
            if "sample_id" not in scores:
                raise ValueError(f"scores missing sample_id: {unit}")
            score_map = scores.set_index("sample_id")
            # Gate unit 的 sample_id 与 gate/test.json 中 source_id 一一对应；直接用其
            # score 表顺序可保留无 source_id 数据集的稳定索引。
            if len(scores) != len(labels):
                raise ValueError(f"test row mismatch for {unit}: {len(scores)} != {len(labels)}")
            method_labels = scores["true_binary_label"].to_numpy(dtype=np.int64)
            method_scores = scores["score"].to_numpy(dtype=float)
            if not np.array_equal(method_labels, labels):
                raise ValueError(f"label order mismatch for {unit}; refuse near-OOS comparison")
            for name in ("near", "medium", "far"):
                oos_mask = (labels == 1) & (bucket == name)
                eval_mask = (labels == 0) | oos_mask
                metrics = compute_binary_oos_metrics(method_labels[eval_mask], method_scores[eval_mask], threshold)
                rows.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "data_seed": seed,
                        "distance": distance,
                        "method": method,
                        "bucket": name,
                        "validation_q20": quantiles["q20"],
                        "validation_q80": quantiles["q80"],
                        "known_test_count": int(np.sum(labels == 0)),
                        "bucket_oos_count": int(np.sum(oos_mask)),
                        "threshold": threshold,
                        **metrics,
                    }
                )
    _write_csv(output_dir / "near_oos_method_comparison.csv", rows)
    return {"rows": len(rows), "datasets": len(DATASETS), "methods": len(set(row["method"] for row in rows))}


def efficiency_summary(v19_root: Path, output_dir: Path) -> dict[str, int]:
    """聚合现有 manifest/timing，明确缺失字段，不伪造 Gate scoring 时间。"""

    geometry = _read_csv(v19_root, "kir_k_fixed_boundary.csv")
    geometry = geometry[(geometry["kir"] == 50) & geometry["k_gate"].isin([1, 2, 5])].copy()
    geometry["method"] = geometry.apply(
        lambda row: f"multisphere_k{int(row['k_gate'])}_{row['distance']}", axis=1
    )
    # Baseline 的逐 seed 表没有 ``run_elapsed_seconds``，运行时间在 summary 表中
    # 聚合保存；先读 summary，避免把“缺字段”误报成真实的零开销。
    baseline = _read_csv(v19_root, "gate_baseline_summary.csv")
    baseline = baseline[(baseline["kir"] == 50) & baseline["method"].isin(BASELINE_METHODS)].copy()
    baseline = baseline.rename(
        columns={
            "run_elapsed_seconds_mean": "run_elapsed_seconds",
            "test_scoring_seconds_mean": "test_scoring_seconds",
            "test_samples_per_second_mean": "test_samples_per_second",
            "process_peak_rss_mb_mean": "process_peak_rss_mb",
        }
    )
    baseline["phase_source"] = "baselines"
    geometry["phase_source"] = "fixed"
    combined = pd.concat([geometry, baseline], ignore_index=True, sort=False)
    rows: list[dict[str, Any]] = []
    for (dataset, method), group in combined.groupby(["dataset", "method"]):
        def mean(name: str) -> float:
            values = pd.to_numeric(group.get(name), errors="coerce")
            return float(values.mean()) if values.notna().any() else math.nan
        def std(name: str) -> float:
            values = pd.to_numeric(group.get(name), errors="coerce")
            return float(values.std(ddof=1)) if values.notna().sum() > 1 else math.nan
        rows.append(
            {
                "dataset": dataset,
                "kir": 50,
                "method": method,
                "cell_count": len(group),
                "total_seconds_mean": mean("run_elapsed_seconds"),
                "total_seconds_std": std("run_elapsed_seconds"),
                "scoring_seconds_mean": mean("test_scoring_seconds"),
                "samples_per_second_mean": mean("test_samples_per_second"),
                "peak_rss_mb_mean": mean("process_peak_rss_mb"),
                "missing_scoring_time_count": int(pd.to_numeric(group.get("test_scoring_seconds"), errors="coerce").isna().sum()),
                "complexity": "O(M*K*h)" if method.startswith("multisphere") else "method-dependent",
            }
        )
    _write_csv(output_dir / "efficiency_summary.csv", rows)
    return {"rows": len(rows)}


def run_all(v19_root: Path = V19_ROOT, v20_root: Path = V20_ROOT) -> dict[str, Any]:
    """运行第一批无需模型重算的 v20 分析，并留下协议 manifest。"""

    analysis_dir = v20_root / "analysis"
    result = {
        "protocol": "cluster_separability_v20",
        "source_root": str(v19_root),
        "output_root": str(v20_root),
        "v19_frozen": True,
        "selection_reliability": selection_reliability(v19_root, analysis_dir),
        "near_oos_comparison": near_oos_comparison(v19_root, analysis_dir),
        "efficiency_summary": efficiency_summary(v19_root, analysis_dir),
    }
    v20_root.mkdir(parents=True, exist_ok=True)
    (v20_root / "v20_analysis_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v20 zero-rerun s2c analyses")
    parser.add_argument("--v19-root", default=str(V19_ROOT))
    parser.add_argument("--v20-root", default=str(V20_ROOT))
    args = parser.parse_args()
    print(json.dumps(run_all(Path(args.v19_root), Path(args.v20_root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
