#!/usr/bin/env python3
"""同表征自适应局部边界 Baseline。

这是对“固定 KMeans 多中心”最直接的受控比较，不是 s2c 新方法：每个 intent 的
K 只依据 Known train 内部 silhouette 在 1..5 中选择，边界、MiniLM、协方差、
阈值和评价协议全部沿用 v19 fixed 设置。它用于替代外部 MOGB 在旧依赖下无法
稳定运行时的第一版可审计对照。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from tools.experiments.cluster_separability.runner import _all_sphere_distances, _build_detector
from tools.experiments.cluster_separability.v20_random_partition import (
    _load_inputs,
    _score_split_fast,
)

V19_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v19"
V20_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v20"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
SEEDS = (13, 42, 87)
DISTANCES = ("euclidean", "mahalanobis_diag")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _choose_overrides(train_embeddings: np.ndarray, train_intents: np.ndarray) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """只在 Known train 内按 silhouette 选择每个 intent 的 K。"""

    normalized = _build_detector(1, "euclidean")._normalize_embeddings(train_embeddings)
    overrides: dict[str, int] = {}
    diagnostics: list[dict[str, Any]] = []
    for intent in sorted(np.unique(train_intents).tolist()):
        points = normalized[train_intents == intent]
        candidates: list[tuple[float, int]] = [(-1.0, 1)]
        for k in range(2, min(5, len(points) - 1) + 1):
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(points)
            score = float(silhouette_score(points, labels))
            candidates.append((score, k))
        # 并列时选择更小 K，避免把训练噪声转成无必要的边界数量。
        score, selected = max(candidates, key=lambda item: (item[0], -item[1]))
        overrides[str(intent)] = int(selected)
        diagnostics.append({
            "intent": str(intent),
            "support": int(len(points)),
            "selected_k": int(selected),
            "selected_silhouette": float(score),
            "candidate_scores": "|".join(f"k{k}:{value:.6f}" for value, k in candidates),
        })
    return overrides, diagnostics


def _run_one(v19_root: Path, v20_root: Path, dataset: str, seed: int, distance: str) -> dict[str, Any]:
    known_manifest, rows, embeddings = _load_inputs(v19_root, dataset, seed)
    train_intents = np.asarray([str(row["intent"]) for row in rows["train"]], dtype=object)
    overrides, diagnostics = _choose_overrides(embeddings["train"], train_intents)
    detector = MultiSphereOOSDetector(
        n_clusters=None,
        radius_quantile=0.90,
        radius_method="mean_std",
        radius_lambda=1.0,
        center_mode="class_centroid_mixture",
        distance_metric=distance,
        covariance_eps=1e-6,
        l2_normalize=True,
        subcenters_per_intent=1,
        subcenters_overrides=overrides,
        random_state=42,
    )
    detector.fit(embeddings["train"], train_intents)
    val_metrics, _ = _score_split_fast(detector, rows["val"], embeddings["val"], known_manifest, "val")
    test_metrics, test_scores = _score_split_fast(detector, rows["test"], embeddings["test"], known_manifest, "test")
    unit = v20_root / "adaptive_boundary" / dataset / f"kir50_seed{seed}" / distance
    unit.mkdir(parents=True, exist_ok=True)
    test_scores.to_parquet(unit / "scores.parquet", index=False)
    _write_csv(unit / "intent_k_selection.csv", diagnostics)
    _write_json(unit / "eval_results.json", {
        "protocol": "cluster_separability_v20_adaptive_boundary_baseline",
        "selection_split": "known_train_only",
        "validation": val_metrics,
        "test": test_metrics,
        "selected_k_distribution": pd.Series(list(overrides.values())).value_counts().sort_index().to_dict(),
    })
    _write_json(unit / "run_manifest.json", {
        "status": "complete",
        "baseline_name": "adaptive_train_silhouette",
        "dataset": dataset,
        "kir": 50,
        "data_seed": seed,
        "distance": distance,
        "candidate_k": [1, 2, 3, 4, 5],
        "selection_split": "known_train_only",
        "oos_or_test_used_for_k": False,
        "radius_lambda": 1.0,
        "decision_threshold": 1.0,
        "overrides": overrides,
        "v19_frozen": True,
    })
    return {
        "dataset": dataset,
        "kir": 50,
        "data_seed": seed,
        "distance": distance,
        "method": "adaptive_train_silhouette",
        "test_oos_f1": test_metrics["oos_f1"],
        "test_id_recall": test_metrics["id_recall"],
        "test_auroc": test_metrics["auroc"],
        "test_fpr95": test_metrics["fpr95"],
        "mean_selected_k": float(np.mean(list(overrides.values()))),
        "fraction_intents_k_gt1": float(np.mean(np.asarray(list(overrides.values())) > 1)),
    }


def run(v19_root: Path = V19_ROOT, v20_root: Path = V20_ROOT) -> dict[str, Any]:
    rows = [_run_one(v19_root, v20_root, dataset, seed, distance)
            for dataset in DATASETS for seed in SEEDS for distance in DISTANCES]
    _write_csv(v20_root / "adaptive_boundary_by_seed.csv", rows)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "distance"], as_index=False).agg(
        seed_count=("data_seed", "count"),
        test_oos_f1_mean=("test_oos_f1", "mean"),
        test_oos_f1_std=("test_oos_f1", "std"),
        test_id_recall_mean=("test_id_recall", "mean"),
        test_auroc_mean=("test_auroc", "mean"),
        test_fpr95_mean=("test_fpr95", "mean"),
        mean_selected_k=("mean_selected_k", "mean"),
        fraction_intents_k_gt1=("fraction_intents_k_gt1", "mean"),
    )
    _write_csv(v20_root / "adaptive_boundary_summary.csv", summary.to_dict("records"))
    payload = {
        "protocol": "cluster_separability_v20_adaptive_boundary_baseline",
        "completed_units": len(rows),
        "selection_split": "known_train_only",
        "v19_frozen": True,
        "external_method_status": "not_claimed; this is a controlled same-representation fallback",
    }
    _write_json(v20_root / "adaptive_boundary_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled adaptive local-boundary baseline")
    parser.add_argument("--v19-root", default=str(V19_ROOT))
    parser.add_argument("--v20-root", default=str(V20_ROOT))
    args = parser.parse_args()
    print(json.dumps(run(Path(args.v19_root), Path(args.v20_root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
