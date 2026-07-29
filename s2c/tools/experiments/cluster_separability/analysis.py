#!/usr/bin/env python3
"""从已完成的多簇实验产物中分析 MiniLM 几何结构与 OOS 可分性。

near/medium/far 切分点只在 validation OOS 上估计，然后冻结并应用到 test。
每个 OOS bucket 都与完整 Known test 组成一个新的二分评价集；不会在
只含 OOS 正样本的 bucket 上错误计算 AUROC 或 FPR95。可视化一律使用
ground truth 分组，预测结果只用于错误分析。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from legacy.runtime import WorkspacePaths
from .protocol import compute_binary_oos_metrics

PATHS = WorkspacePaths.discover(PROJECT_ROOT)
DEFAULT_ROOT = PATHS.artifact_root / "outputs" / "experiments" / "cluster_separability_v19"
FIGURE_SEED = 42
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
DATA_SEEDS = (13, 42, 87)
DISTANCES = ("euclidean", "mahalanobis_diag")
STABILITY_K_VALUES = (2, 3, 4, 5)
MIN_MULTIMODAL_SUPPORT = 20
MIN_MULTIMODAL_CLUSTER_RATIO = 0.10
os.environ.setdefault("MPLBACKEND", "Agg")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _centroids(train_rows: Sequence[Mapping[str, Any]], train_embeddings: np.ndarray) -> np.ndarray:
    labels = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    x = _l2(train_embeddings)
    centers = np.stack([np.mean(x[labels == label], axis=0) for label in sorted(set(labels))])
    return _l2(centers)


def semantic_distance(embeddings: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """返回样本与最近 Known-intent 单中心的 cosine distance。"""

    return 1.0 - np.max(_l2(embeddings) @ _l2(centers).T, axis=1)


def validation_bucket_thresholds(
    val_labels: Sequence[int] | np.ndarray, val_semantic_distances: Sequence[float] | np.ndarray
) -> tuple[float, float]:
    """在 validation OOS 距离上估计 q20/q80，Known validation 不参与分位点。"""

    labels = np.asarray(val_labels, dtype=np.int64)
    distances = np.asarray(val_semantic_distances, dtype=np.float64)
    oos = distances[labels == 1]
    if not oos.size:
        raise ValueError("validation split contains no OOS samples for q20/q80")
    q20, q80 = np.quantile(oos, [0.20, 0.80])
    return float(q20), float(q80)


def _bucket(distance: np.ndarray, q20: float, q80: float) -> np.ndarray:
    return np.where(distance <= q20, "near", np.where(distance <= q80, "medium", "far"))


def compute_near_far_rows(
    *,
    validation_labels: Sequence[int],
    validation_embeddings: np.ndarray,
    test_scores: pd.DataFrame,
    test_embeddings: np.ndarray,
    centers: np.ndarray,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """返回 near/medium/far 分层指标，test 数值不参与任何阈值选择。"""

    if len(test_scores) != len(test_embeddings):
        raise ValueError("test score rows and embeddings do not align")
    val_distance = semantic_distance(validation_embeddings, centers)
    q20, q80 = validation_bucket_thresholds(validation_labels, val_distance)
    test_distance = semantic_distance(test_embeddings, centers)
    buckets = _bucket(test_distance, q20, q80)
    labels = test_scores["true_binary_label"].to_numpy(dtype=int)
    gate_scores = test_scores["score"].to_numpy(dtype=float)
    known = labels == 0
    sources = sorted(set(test_scores.loc[labels == 1, "oos_source"].astype(str)))
    source_groups: list[tuple[str, np.ndarray]] = [("combined", labels == 1)]
    source_groups.extend((source, (labels == 1) & (test_scores["oos_source"].astype(str).to_numpy() == source)) for source in sources)

    rows = []
    for source, source_mask in source_groups:
        for name in ("near", "medium", "far"):
            oos_mask = source_mask & (buckets == name)
            # 单个 bucket 本身只包含 OOS。必须加回全部 Known test，
            # 才能定义 precision、AUROC、AUPR 和 FPR95。
            evaluation_mask = known | oos_mask
            if not np.any(oos_mask):
                metrics = {key: math.nan for key in ("oos_precision", "oos_recall", "oos_f1", "id_recall", "oos_rejection", "auroc", "aupr_oos", "fpr95")}
            else:
                metrics = compute_binary_oos_metrics(labels[evaluation_mask], gate_scores[evaluation_mask], 1.0)
            rows.append(
                {
                    **metadata,
                    "oos_source": source,
                    "bucket": name,
                    "validation_q20": q20,
                    "validation_q80": q80,
                    "known_test_count": int(np.sum(known)),
                    "bucket_oos_count": int(np.sum(oos_mask)),
                    **metrics,
                }
            )
    return rows


def compute_overlap_row(scores: pd.DataFrame, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """计算样本级 empirical multi-coverage，不估计高维球/椭球体积。

    384 维空间中的几何体积重叠对协方差与尺度高度敏感，且难以稳定解释。
    这里只回答可验证的问题：真实样本是否同时落入多个 intent 接受区域。
    """

    labels = scores["true_binary_label"].to_numpy(dtype=int)
    prediction = scores["prediction"].to_numpy(dtype=int)
    coverage = scores["coverage_count"].to_numpy(dtype=int)
    known, oos = labels == 0, labels == 1
    false_accept = oos & (prediction == 0)
    return {
        **metadata,
        "known_count": int(np.sum(known)),
        "oos_count": int(np.sum(oos)),
        "known_multi_coverage_rate": float(np.mean(coverage[known] >= 2)) if np.any(known) else math.nan,
        "oos_multi_coverage_rate": float(np.mean(coverage[oos] >= 2)) if np.any(oos) else math.nan,
        "false_accepted_oos_count": int(np.sum(false_accept)),
        "false_accepted_oos_multi_coverage_rate": float(np.mean(coverage[false_accept] >= 2)) if np.any(false_accept) else math.nan,
    }


def compute_intent_pair_overlap_rows(
    detector_state: Mapping[str, Any],
    test_embeddings: np.ndarray,
    test_scores: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """统计 test 样本被每一对不同 intent 接受区域共同覆盖的次数。

    同一 intent 的多个 sphere 先做逻辑 OR，再构造 intent pair；因此结果衡量
    跨意图边界重叠，而不是一个意图内部的子球相交。这里沿用 coverage_count
    的定义，只使用 ``distance <= radius``，margin 仍属于最终 Gate 判决而非
    几何接受区域。所有 pair（包括计数为 0）都会保留，便于审计“无重叠”。
    """

    embeddings = np.asarray(test_embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(test_scores) != len(embeddings):
        raise ValueError("test scores and 2-D embeddings must align")
    if bool(detector_state.get("l2_normalize", False)):
        embeddings = _l2(embeddings)
    metric = str(detector_state.get("distance_metric", "euclidean"))
    if metric not in {"euclidean", "mahalanobis_diag"}:
        raise ValueError(f"unsupported detector distance metric: {metric}")

    cluster_to_intent = detector_state.get("cluster_to_intent", {})
    spheres = list(detector_state.get("spheres", []))
    centers = np.asarray([sphere["center"] for sphere in spheres], dtype=np.float64)
    if metric == "euclidean" and len(spheres):
        # 避免构造 [sample, sphere, dim] 的三维 diff；在 CLINC150 上该张量
        # 可轻易超过 1 GB。矩阵恒等式给出完全相同的平方欧氏距离。
        distance_squared = (
            np.sum(embeddings**2, axis=1)[:, None]
            + np.sum(centers**2, axis=1)[None, :]
            - 2.0 * embeddings @ centers.T
        )
        accepted_matrix = np.sqrt(np.maximum(distance_squared, 0.0)) <= np.asarray(
            [float(sphere["radius"]) for sphere in spheres]
        )[None, :]
    else:
        accepted_matrix = np.zeros((len(embeddings), len(spheres)), dtype=bool)
    accepted_by_intent: dict[str, np.ndarray] = {}
    for sphere_index, sphere in enumerate(spheres):
        cluster_id = sphere.get("cluster_id")
        intent = sphere.get("intent_name")
        if intent is None:
            intent = cluster_to_intent.get(str(cluster_id), cluster_to_intent.get(cluster_id))
        if intent is None:
            raise ValueError(f"sphere {cluster_id!r} has no intent mapping")
        if metric == "mahalanobis_diag":
            inv_diag_cov = sphere.get("inv_diag_cov")
            if inv_diag_cov is None:
                raise ValueError("mahalanobis sphere is missing inv_diag_cov")
            diff = embeddings - centers[sphere_index]
            distance = np.sqrt(np.sum(diff**2 * np.asarray(inv_diag_cov, dtype=np.float64), axis=1))
            accepted = distance <= float(sphere["radius"])
        else:
            accepted = accepted_matrix[:, sphere_index]
        key = str(intent)
        accepted_by_intent[key] = accepted | accepted_by_intent.get(
            key, np.zeros(len(embeddings), dtype=bool)
        )

    labels = test_scores["true_binary_label"].to_numpy(dtype=int)
    predictions = test_scores["prediction"].to_numpy(dtype=int)
    known = labels == 0
    oos = labels == 1
    false_accepted_oos = oos & (predictions == 0)
    false_rejected_known = known & (predictions == 1)
    rows = []
    for intent_a, intent_b in combinations(sorted(accepted_by_intent), 2):
        joint = accepted_by_intent[intent_a] & accepted_by_intent[intent_b]
        rows.append(
            {
                **metadata,
                "intent_a": intent_a,
                "intent_b": intent_b,
                "joint_test_count": int(np.sum(joint)),
                "joint_known_count": int(np.sum(joint & known)),
                "joint_oos_count": int(np.sum(joint & oos)),
                "joint_false_accepted_oos_count": int(np.sum(joint & false_accepted_oos)),
                "joint_false_rejected_known_count": int(np.sum(joint & false_rejected_known)),
            }
        )
    return rows


def compute_intent_stability_rows(
    train_rows: Sequence[Mapping[str, Any]],
    train_embeddings: np.ndarray,
    *,
    requested_k: int,
    repeats: int = 20,
    subsample_fraction: float = 0.8,
    random_seed: int = FIGURE_SEED,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """用重复子采样衡量每个 Known intent 的局部聚类稳定性。

    每次只用 80% 样本拟合 KMeans，但都在该 intent 的**完整原始样本**上
    重新预测簇标签，再计算不同重复之间的 pairwise ARI。这样 ARI 两侧的
    样本集合严格一致，不会比较两个不同 bootstrap 样本上的 label 数组。
    """

    if len(train_rows) != len(train_embeddings):
        raise ValueError("train rows and embeddings do not align")
    if requested_k < 2:
        raise ValueError("stability analysis requires requested_k >= 2")
    if repeats < 2:
        raise ValueError("stability analysis requires at least two repeats")
    if not 0.0 < subsample_fraction <= 1.0:
        raise ValueError("subsample_fraction must be in (0, 1]")

    intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    normalized = _l2(train_embeddings)
    rng = np.random.default_rng(random_seed)
    output: list[dict[str, Any]] = []
    for intent in sorted(set(intents.tolist())):
        points = normalized[intents == intent]
        support = int(len(points))
        effective_k = min(int(requested_k), support)
        subsample_size = min(support, max(effective_k, int(math.floor(support * subsample_fraction))))
        predictions: list[np.ndarray] = []
        for repeat in range(repeats):
            indices = rng.choice(support, size=subsample_size, replace=False)
            model = KMeans(
                n_clusters=effective_k,
                random_state=int(random_seed + repeat),
                n_init=10,
            )
            model.fit(points[indices])
            predictions.append(model.predict(points))

        pairwise = np.asarray(
            [adjusted_rand_score(predictions[left], predictions[right]) for left, right in combinations(range(repeats), 2)],
            dtype=np.float64,
        )
        q25, median, q75 = np.quantile(pairwise, [0.25, 0.50, 0.75])
        output.append(
            {
                **(metadata or {}),
                "intent": intent,
                "support": support,
                "requested_k": int(requested_k),
                "effective_k": effective_k,
                "repeats": int(repeats),
                "subsample_fraction": float(subsample_fraction),
                "subsample_size": subsample_size,
                "pair_count": int(len(pairwise)),
                "ari_q25": float(q25),
                "ari_median": float(median),
                "ari_q75": float(q75),
                "ari_iqr": float(q75 - q25),
            }
        )
    return output


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    if not unique.size:
        raise ValueError("cannot select a threshold from empty validation scores")
    return np.concatenate(([np.nextafter(unique[0], -np.inf)], unique))


def _select_validation_threshold(labels: np.ndarray, scores: np.ndarray, id_recall_guard: float) -> dict[str, Any]:
    """按统一 ID-recall guard 在 validation 上冻结阈值。"""

    candidates = [
        {
            "threshold": float(threshold),
            "metrics": compute_binary_oos_metrics(labels, scores, float(threshold)),
        }
        for threshold in _threshold_candidates(scores)
    ]
    eligible = [row for row in candidates if row["metrics"]["id_recall"] >= id_recall_guard]
    pool = eligible or candidates

    def rank(row: Mapping[str, Any]) -> tuple[float, ...]:
        metrics = row["metrics"]
        inverse_fpr95 = -float(metrics["fpr95"]) if math.isfinite(float(metrics["fpr95"])) else -math.inf
        if eligible:
            return (float(metrics["oos_f1"]), inverse_fpr95, float(metrics["id_recall"]), float(row["threshold"]))
        return (float(metrics["id_recall"]), float(metrics["oos_f1"]), inverse_fpr95, float(row["threshold"]))

    selected = dict(max(pool, key=rank))
    selected["guard_violation"] = not bool(eligible)
    return selected


def _k1_centroid_scores(
    train_rows: Sequence[Mapping[str, Any]], train_embeddings: np.ndarray, query_embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回归一化 K=1 intent centroid 的 Euclidean/cosine 距离矩阵。"""

    intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    train = _l2(train_embeddings)
    query = _l2(query_embeddings)
    centers = np.stack([np.mean(train[intents == intent], axis=0) for intent in sorted(set(intents.tolist()))])
    centers = _l2(centers)
    cosine = 1.0 - query @ centers.T
    # query/center 均已 L2，直接使用平方距离恒等式，避免三维广播张量。
    euclidean_squared = np.maximum(2.0 - 2.0 * (query @ centers.T), 0.0)
    return np.sqrt(np.maximum(euclidean_squared, 0.0)), cosine, centers


def _control_metric_row(
    *,
    control: str,
    score_direction: str,
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    id_recall_guard: float,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _select_validation_threshold(validation_labels, validation_scores, id_recall_guard)
    test_metrics = compute_binary_oos_metrics(test_labels, test_scores, float(selected["threshold"]))
    return {
        **metadata,
        "control": control,
        "score_direction": score_direction,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "id_recall_guard": float(id_recall_guard),
        "guard_violation": bool(selected["guard_violation"]),
        "threshold": float(selected["threshold"]),
        **{f"validation_{key}": value for key, value in selected["metrics"].items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def compute_representation_control_rows(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    id_recall_guard: float,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """计算代表配置上的 MiniLM 表征控制，所有运行点只由 validation 决定。"""

    aligned = (
        len(train_rows) == len(train_embeddings)
        and len(validation_rows) == len(validation_embeddings)
        and len(test_rows) == len(test_embeddings)
    )
    if not aligned:
        raise ValueError("gold rows and embedding arrays do not align")
    validation_labels = np.asarray([int(row["label"]) for row in validation_rows], dtype=np.int64)
    test_labels = np.asarray([int(row["label"]) for row in test_rows], dtype=np.int64)

    val_euclidean, val_cosine, _ = _k1_centroid_scores(train_rows, train_embeddings, validation_embeddings)
    test_euclidean, test_cosine, _ = _k1_centroid_scores(train_rows, train_embeddings, test_embeddings)
    val_euclidean_score = np.min(val_euclidean, axis=1)
    test_euclidean_score = np.min(test_euclidean, axis=1)
    rows = [
        _control_metric_row(
            control="l2_k1_centroid",
            score_direction="higher_euclidean_is_oos",
            validation_labels=validation_labels,
            validation_scores=val_euclidean_score,
            test_labels=test_labels,
            test_scores=test_euclidean_score,
            id_recall_guard=id_recall_guard,
            metadata=metadata,
        )
    ]

    # PCA 仅在 Known train 上拟合；去除 PC1 后才做 L2 和 K=1 centroid scoring。
    pca = PCA(n_components=1, random_state=FIGURE_SEED).fit(train_embeddings)
    component = pca.components_[0]

    def remove_pc1(values: np.ndarray) -> np.ndarray:
        centered = np.asarray(values, dtype=np.float64) - pca.mean_
        return centered - np.outer(centered @ component, component)

    train_residual = remove_pc1(train_embeddings)
    val_residual = remove_pc1(validation_embeddings)
    test_residual = remove_pc1(test_embeddings)
    val_pca_distance, _, _ = _k1_centroid_scores(train_rows, train_residual, val_residual)
    test_pca_distance, _, _ = _k1_centroid_scores(train_rows, train_residual, test_residual)
    pca_row = _control_metric_row(
        control="pca_remove_pc1_k1_centroid",
        score_direction="higher_euclidean_is_oos",
        validation_labels=validation_labels,
        validation_scores=np.min(val_pca_distance, axis=1),
        test_labels=test_labels,
        test_scores=np.min(test_pca_distance, axis=1),
        id_recall_guard=id_recall_guard,
        metadata=metadata,
    )
    pca_row["pc1_explained_variance_ratio"] = float(pca.explained_variance_ratio_[0])
    rows.append(pca_row)

    # 向量范数与 OOS 的相关方向并非预先已知，因此将正负两个方向都在
    # validation 上选择，test 只应用冻结后的方向和阈值。
    val_norm = np.linalg.norm(validation_embeddings, axis=1)
    test_norm = np.linalg.norm(test_embeddings, axis=1)
    norm_candidates = []
    for direction, sign in (("higher_norm_is_oos", 1.0), ("lower_norm_is_oos", -1.0)):
        row = _control_metric_row(
            control="raw_norm_only",
            score_direction=direction,
            validation_labels=validation_labels,
            validation_scores=sign * val_norm,
            test_labels=test_labels,
            test_scores=sign * test_norm,
            id_recall_guard=id_recall_guard,
            metadata=metadata,
        )
        norm_candidates.append(row)
    eligible_norm = [row for row in norm_candidates if not row["guard_violation"]]
    norm_pool = eligible_norm or norm_candidates
    norm_row = max(
        norm_pool,
        key=lambda row: (
            float(row["validation_oos_f1"]),
            -float(row["validation_fpr95"]),
            float(row["validation_id_recall"]),
            row["score_direction"] == "higher_norm_is_oos",
        ),
    )
    norm_row["train_raw_norm_mean"] = float(np.mean(np.linalg.norm(train_embeddings, axis=1)))
    norm_row["train_raw_norm_std"] = float(np.std(np.linalg.norm(train_embeddings, axis=1)))
    rows.append(norm_row)

    # 对 L2-normalized embedding，||x-y||^2 = 2(1-cos(x,y))。
    # 此行是数值 sanity check，不伪装成另一个独立检测方法。
    equivalence = {
        **metadata,
        "control": "l2_euclidean_cosine_equivalence",
        "score_direction": "diagnostic_only",
        "selection_split": None,
        "test_used_for_selection": False,
        "id_recall_guard": math.nan,
        "guard_violation": False,
        "threshold": math.nan,
        "nearest_intent_agreement": float(
            np.mean(np.argmin(test_euclidean, axis=1) == np.argmin(test_cosine, axis=1))
        ),
        "max_distance_identity_error": float(
            np.max(np.abs(test_euclidean**2 - 2.0 * test_cosine))
        ),
    }
    rows.append(equivalence)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 分析行可能携带 control-specific 诊断字段；按首次出现顺序取并集，
    # 避免 DictWriter 因后续行多出合法字段而失败。
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _load_cache(root: Path, dataset: str, kir: int, seed: int, split: str, manifest: Mapping[str, Any]) -> np.ndarray:
    cache = manifest["embedding_cache"][split]
    prefix = str(cache["cache_key"])[:16]
    path = root / "embedding_cache" / dataset / f"kir{kir}_seed{seed}" / f"{split}_{prefix}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.shape[0] != int(cache["sample_count"]):
        raise ValueError(f"cache sample count mismatch: {path}")
    return embeddings


def _sample_indices(rows: Sequence[Mapping[str, Any]], max_samples: int) -> np.ndarray:
    """以固定随机种子做分组均衡下采样，防止大 intent/OOS source 支配图像。"""

    if len(rows) <= max_samples:
        return np.arange(len(rows))
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        key = "known:" + str(row["intent"]) if int(row["label"]) == 0 else "oos:" + str(row.get("source_split", "unknown"))
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(FIGURE_SEED)
    quota = max(1, max_samples // len(groups))
    selected = []
    for key in sorted(groups):
        values = np.asarray(groups[key])
        selected.extend(rng.choice(values, size=min(quota, len(values)), replace=False).tolist())
    return np.asarray(sorted(selected[:max_samples]), dtype=int)


def _plot_projection(coords: np.ndarray, kinds: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {"known": "#377eb8", "heldout_unknown": "#e41a1c", "native_or_provided_oos": "#ff7f00"}
    fig, axis = plt.subplots(figsize=(7.2, 5.6))
    for kind in sorted(set(kinds)):
        mask = kinds == kind
        axis.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.55, label=kind, c=colors.get(kind, "#777777"))
    axis.set_title(title)
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    axis.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_geometry_outputs(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    test_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    max_samples: int = 5000,
    nonlinear: bool = True,
) -> dict[str, Any]:
    """生成 train-fit 的 PCA 诊断与可选非线性投影。

    PCA 只在 Known train embedding 上 fit，test 只 transform。t-SNE/UMAP 仅用于
    定性展示，不用它们选 K、阈值或边界，也不将二维视觉分离当作高维
    可分性的定量证据。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=min(50, train_embeddings.shape[0], train_embeddings.shape[1]), random_state=FIGURE_SEED)
    pca.fit(train_embeddings)
    indices = _sample_indices(test_rows, max_samples)
    sample = test_embeddings[indices]
    kinds = np.asarray([
        "known" if int(test_rows[i]["label"]) == 0 else ("native_or_provided_oos" if str(test_rows[i].get("intent", "")).lower() == "oos" or str(test_rows[i].get("source_split", "")).lower().startswith(("oos", "clinc_oos", "id-oos", "ood-oos")) else "heldout_unknown")
        for i in indices
    ])
    pca_coords = pca.transform(sample)[:, :2]
    _plot_projection(pca_coords, kinds, output_dir / "pca.png", "MiniLM PCA (gold groups)")
    diagnostics = {
        "train_sample_count": int(len(train_embeddings)),
        "figure_sample_count": int(len(indices)),
        "embedding_dimension": int(train_embeddings.shape[1]),
        "raw_norm_mean": float(np.mean(np.linalg.norm(train_embeddings, axis=1))),
        "raw_norm_std": float(np.std(np.linalg.norm(train_embeddings, axis=1))),
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "top10_explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_[:10])),
        "pca_figure": "pca.png",
        "tsne_figure": None,
        "umap_figure": None,
    }
    if nonlinear and len(sample) >= 31:
        tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=FIGURE_SEED)
        _plot_projection(tsne.fit_transform(sample), kinds, output_dir / "tsne.png", "MiniLM t-SNE (gold groups)")
        diagnostics["tsne_figure"] = "tsne.png"
        try:
            import umap
        except ImportError:
            diagnostics["umap_status"] = "skipped: umap-learn not installed"
        else:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=FIGURE_SEED)
            _plot_projection(reducer.fit_transform(sample), kinds, output_dir / "umap.png", "MiniLM UMAP (gold groups)")
            diagnostics["umap_figure"] = "umap.png"
            diagnostics["umap_status"] = "complete"
    (output_dir / "pca_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    return diagnostics


def _known_intent_gate_f1(frame: pd.DataFrame, intent: str) -> float:
    """计算 Gate 最近原型在 Known test 上对单个 intent 的诊断 F1。

    该值不是 Router/Expert 的端到端 Known-intent F1：被 Gate 拒绝的样本视为
    未预测到任何 intent，通过 Gate 的样本使用 ``nearest_intent``。它只用于
    判断 K=1 -> K=2 是否同时改善局部几何匹配，不进入论文主 Known 指标表。
    """

    known = frame["true_binary_label"].eq(0)
    truth = known & frame["true_intent"].astype(str).eq(str(intent))
    predicted = (
        known
        & frame["prediction"].eq(0)
        & frame["nearest_intent"].astype(str).eq(str(intent))
    )
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else math.nan


def _hard_intent_rows(root: Path, phase: str, dataset: str, kir: int, seed: int, distance: str) -> list[dict[str, Any]]:
    """在同一批 sample_id 上对齐 K=1/K=2，导出 intent-level 边界得失。

    OOS false accept 按“K=1 时最近 intent”归因，这是为了固定对比基准，
    避免 K=2 改变 nearest intent 后再改写分析人群。稳定性仅在 KIR50 网格
    中定义；其他 KIR 留空，空值不代表稳定性为 0。
    """

    base = root / phase / dataset / f"kir{kir}_seed{seed}" / distance
    paths = {k: base / f"k{k}" for k in (1, 2)}
    if not all((path / "scores.parquet").is_file() and (path / "cluster_metrics.csv").is_file() for path in paths.values()):
        return []
    frames = {k: pd.read_parquet(path / "scores.parquet").set_index("sample_id") for k, path in paths.items()}
    common = frames[1].index.intersection(frames[2].index)
    frames = {k: frame.loc[common] for k, frame in frames.items()}
    cluster = {k: pd.read_csv(path / "cluster_metrics.csv").set_index("intent") for k, path in paths.items()}
    stability_path = root / "analysis" / "cluster_stability.csv"
    stability: dict[str, float] = {}
    if kir == 50 and stability_path.is_file():
        stability_frame = pd.read_csv(stability_path)
        selected = stability_frame[
            stability_frame["dataset"].eq(dataset)
            & stability_frame["data_seed"].eq(seed)
            & stability_frame["requested_k"].eq(2)
        ]
        stability = {
            str(row["intent"]): float(row["ari_median"])
            for _, row in selected.iterrows()
        }
    rows = []
    for intent in sorted(cluster[1].index.intersection(cluster[2].index)):
        known = frames[1]["true_intent"].astype(str).eq(str(intent)) & frames[1]["true_binary_label"].eq(0)
        attributed = frames[1]["nearest_intent"].astype(str).eq(str(intent)) & frames[1]["true_binary_label"].eq(1)
        wcss1, wcss2 = float(cluster[1].loc[intent, "wcss"]), float(cluster[2].loc[intent, "wcss"])
        wcss_gain = 1.0 - wcss2 / max(wcss1, 1e-12)
        minimum_cluster_ratio = float(cluster[2].loc[intent, "minimum_cluster_ratio"])
        stability_k2 = stability.get(str(intent), math.nan)
        multimodality_eligible = (
            int(np.sum(known)) >= MIN_MULTIMODAL_SUPPORT
            and minimum_cluster_ratio >= MIN_MULTIMODAL_CLUSTER_RATIO
            and math.isfinite(stability_k2)
        )
        false_reject_k1 = float(np.mean(frames[1].loc[known, "prediction"] == 1)) if np.any(known) else math.nan
        false_reject_k2 = float(np.mean(frames[2].loc[known, "prediction"] == 1)) if np.any(known) else math.nan
        known_overlap_k1 = float(np.mean(frames[1].loc[known, "coverage_count"] >= 2)) if np.any(known) else math.nan
        known_overlap_k2 = float(np.mean(frames[2].loc[known, "coverage_count"] >= 2)) if np.any(known) else math.nan
        intent_f1_k1 = _known_intent_gate_f1(frames[1], str(intent))
        intent_f1_k2 = _known_intent_gate_f1(frames[2], str(intent))
        rows.append(
            {
                "phase": phase, "dataset": dataset, "kir": kir, "data_seed": seed, "distance": distance, "intent": intent,
                "support": int(np.sum(known)),
                "wcss_k1": wcss1, "wcss_k2": wcss2,
                "wcss_gain_k2": wcss_gain,
                "minimum_cluster_ratio_k2": minimum_cluster_ratio,
                "stability_k2": stability_k2 if math.isfinite(stability_k2) else "",
                "multimodality_eligible": multimodality_eligible,
                "multimodality_score": wcss_gain * stability_k2 if multimodality_eligible else "",
                "false_reject_k1": false_reject_k1,
                "false_reject_k2": false_reject_k2,
                "false_reject_delta_k2_minus_k1": false_reject_k2 - false_reject_k1,
                "false_reject_improvement_k1_minus_k2": false_reject_k1 - false_reject_k2,
                "known_multi_coverage_k1": known_overlap_k1,
                "known_multi_coverage_k2": known_overlap_k2,
                "known_multi_coverage_delta_k2_minus_k1": known_overlap_k2 - known_overlap_k1,
                "intent_f1_k1": intent_f1_k1,
                "intent_f1_k2": intent_f1_k2,
                "intent_f1_delta_k2_minus_k1": intent_f1_k2 - intent_f1_k1,
                "attributed_oos_count": int(np.sum(attributed)),
                "oos_false_accept_k1": float(np.mean(frames[1].loc[attributed, "prediction"] == 0)) if np.any(attributed) else math.nan,
                "oos_false_accept_k2": float(np.mean(frames[2].loc[attributed, "prediction"] == 0)) if np.any(attributed) else math.nan,
                "oos_false_accept_delta_k2_minus_k1": float(np.mean(frames[2].loc[attributed, "prediction"] == 0) - np.mean(frames[1].loc[attributed, "prediction"] == 0)) if np.any(attributed) else math.nan,
            }
        )
    return rows


def compute_hard_intent_correlations(
    hard_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_repeats: int = 1000,
    random_seed: int = FIGURE_SEED,
) -> list[dict[str, Any]]:
    """关联多模态程度与 K=1 -> K=2 的 intent-level 几何收益。

    只使用 KIR50 且通过 support、最小簇比例、稳定性门槛的 intent。每个
    dataset/distance 单独计算，避免把不同距离尺度或同一 intent 的重复观测
    混成一个伪大样本。95% CI 通过对 intent 行有放回重采样获得。
    """

    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be >= 1")
    frame = pd.DataFrame.from_records(hard_rows)
    if frame.empty:
        return []
    targets = (
        "false_reject_improvement_k1_minus_k2",
        "oos_false_accept_delta_k2_minus_k1",
        "known_multi_coverage_delta_k2_minus_k1",
        "intent_f1_delta_k2_minus_k1",
    )
    eligible = frame[
        frame["kir"].eq(50)
        & frame["multimodality_eligible"].astype(str).str.lower().isin({"true", "1"})
    ].copy()
    output: list[dict[str, Any]] = []
    for group_index, ((dataset, distance), group) in enumerate(
        eligible.groupby(["dataset", "distance"], sort=True)
    ):
        for target_index, target in enumerate(targets):
            values = group[["multimodality_score", target]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(values) < 3:
                continue
            x = values["multimodality_score"].to_numpy(dtype=np.float64)
            y = values[target].to_numpy(dtype=np.float64)
            rho = float(spearmanr(x, y).statistic)
            rng = np.random.default_rng(random_seed + group_index * 100 + target_index)
            bootstrapped: list[float] = []
            for _ in range(bootstrap_repeats):
                indices = rng.integers(0, len(values), size=len(values))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    candidate = float(spearmanr(x[indices], y[indices]).statistic)
                # 某次重采样可能只抽到一个唯一秩，此时 Spearman 未定义。
                if math.isfinite(candidate):
                    bootstrapped.append(candidate)
            if bootstrapped:
                ci_low, ci_high = np.quantile(bootstrapped, [0.025, 0.975])
            else:
                ci_low = ci_high = math.nan
            output.append(
                {
                    "dataset": dataset,
                    "kir": 50,
                    "distance": distance,
                    "target_metric": target,
                    "sample_count": int(len(values)),
                    "spearman_rho": rho,
                    "bootstrap_repeats": int(bootstrap_repeats),
                    "bootstrap_valid_repeats": int(len(bootstrapped)),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                }
            )
    return output


def run_stability_grid(
    root: Path,
    *,
    repeats: int = 20,
    subsample_fraction: float = 0.8,
    resume: bool = True,
) -> dict[str, Any]:
    """运行 KIR50、K=2..5、三数据集、三 data seed 的稳定性网格。"""

    analysis_root = root / "analysis"
    all_rows: list[dict[str, Any]] = []
    completed_units = 0
    for dataset in DATASETS:
        for seed in DATA_SEEDS:
            for requested_k in STABILITY_K_VALUES:
                source = root / "fixed" / dataset / f"kir50_seed{seed}" / "euclidean" / f"k{requested_k}"
                manifest_path = source / "run_manifest.json"
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"missing fixed-grid stability source: {manifest_path}")
                manifest = _json(manifest_path)
                unit_output = analysis_root / "stability" / dataset / f"kir50_seed{seed}" / f"k{requested_k}"
                rows_path = unit_output / "cluster_stability.csv"
                stability_manifest_path = unit_output / "stability_manifest.json"
                expected = {
                    "dataset": dataset,
                    "kir": 50,
                    "data_seed": seed,
                    "requested_k": requested_k,
                    "repeats": repeats,
                    "subsample_fraction": subsample_fraction,
                    "source_embedding_cache_key": manifest["embedding_cache"]["train"]["cache_key"],
                }
                if resume and rows_path.is_file() and stability_manifest_path.is_file():
                    saved_manifest = _json(stability_manifest_path)
                    if saved_manifest.get("config") == expected and saved_manifest.get("status") == "complete":
                        saved = pd.read_csv(rows_path).to_dict("records")
                        if saved:
                            all_rows.extend(saved)
                            completed_units += 1
                            continue

                train_rows = _json(Path(manifest["data_root"]) / "gate" / "train.json")
                train_embeddings = _load_cache(root, dataset, 50, seed, "train", manifest)
                rows = compute_intent_stability_rows(
                    train_rows,
                    train_embeddings,
                    requested_k=requested_k,
                    repeats=repeats,
                    subsample_fraction=subsample_fraction,
                    random_seed=FIGURE_SEED + seed * 100 + requested_k,
                    metadata={"dataset": dataset, "kir": 50, "data_seed": seed},
                )
                _write_csv(rows_path, rows)
                unit_output.mkdir(parents=True, exist_ok=True)
                stability_manifest_path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "config": expected,
                            "source_unit": str(source),
                            "intent_count": len(rows),
                            "test_used_for_selection": False,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                all_rows.extend(rows)
                completed_units += 1

    aggregate_path = analysis_root / "cluster_stability.csv"
    _write_csv(aggregate_path, all_rows)
    summary = {
        "status": "complete",
        "unit_count": completed_units,
        "expected_unit_count": len(DATASETS) * len(DATA_SEEDS) * len(STABILITY_K_VALUES),
        "intent_rows": len(all_rows),
        "repeats": repeats,
        "subsample_fraction": subsample_fraction,
        "output": str(aggregate_path),
        "test_used_for_selection": False,
    }
    (analysis_root / "cluster_stability_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _selected_k(root: Path, dataset: str, kir: int, seed: int, distance: str) -> int:
    """读取 exporter 用 validation 指标选出的 dataset-level K。"""

    path = root / "selected_k_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"selected-K summary is required: {path}")
    frame = pd.read_csv(path)
    selected = frame[
        frame["dataset"].eq(dataset)
        & frame["kir"].eq(kir)
        & frame["data_seed"].eq(seed)
        & frame["distance"].eq(distance)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one selected K for {dataset}/kir{kir}/seed{seed}/{distance}, got {len(selected)}"
        )
    return int(selected.iloc[0]["selected_k"])


def analyze(
    root: Path,
    *,
    dataset: str,
    kir: int,
    seed: int,
    phase: str,
    distance: str,
    k_gate: int,
    output_dir: Path | None = None,
    nonlinear: bool = True,
    max_samples: int = 5000,
    include_geometry: bool = True,
    include_representation_controls: bool = True,
) -> dict[str, Any]:
    """对一个已完成实验单元执行全部几何与分层分析。"""

    unit = root / phase / dataset / f"kir{kir}_seed{seed}" / distance / f"k{k_gate}"
    required = [
        unit / "run_manifest.json",
        unit / "scores.parquet",
        unit / "threshold_selection.json",
        unit / "detector.json",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError("missing analysis inputs: " + ", ".join(missing))
    manifest = _json(unit / "run_manifest.json")
    data_root = Path(manifest["data_root"])
    split_rows = {name: _json(data_root / "gate" / f"{name}.json") for name in ("train", "val", "test")}
    embeddings = {name: _load_cache(root, dataset, kir, seed, name, manifest) for name in ("train", "val", "test")}
    if any(len(split_rows[name]) != len(embeddings[name]) for name in embeddings):
        raise ValueError("gold split and embedding cache row counts do not align")
    scores = pd.read_parquet(unit / "scores.parquet")
    metadata = {"phase": phase, "dataset": dataset, "kir": kir, "data_seed": seed, "distance": distance, "k_gate": k_gate}
    centers = _centroids(split_rows["train"], embeddings["train"])
    near_far = compute_near_far_rows(
        validation_labels=[int(row["label"]) for row in split_rows["val"]],
        validation_embeddings=embeddings["val"], test_scores=scores,
        test_embeddings=embeddings["test"], centers=centers, metadata=metadata,
    )
    output = output_dir or (root / "analysis" / dataset / f"kir{kir}_seed{seed}" / distance / f"k{k_gate}")
    _write_csv(output / "near_far_oos_summary.csv", near_far)
    _write_csv(output / "empirical_overlap_summary.csv", [compute_overlap_row(scores, metadata)])
    intent_pair_rows = compute_intent_pair_overlap_rows(
        _json(unit / "detector.json"), embeddings["test"], scores, metadata
    )
    _write_csv(output / "intent_pair_overlap.csv", intent_pair_rows)
    hard_rows = _hard_intent_rows(root, phase, dataset, kir, seed, distance)
    # hard-intent delta 要求同 phase 的 K=1/K=2 成对实验。如果 tuned 网格尚未
    # 完成，只能回退到真实存在的 fixed/smoke 成对产物，且在每行保留
    # 实际 phase；不得把 fixed 差值标成 tuned 结果。
    if not hard_rows:
        for paired_phase in ("fixed", "smoke"):
            if paired_phase != phase:
                hard_rows = _hard_intent_rows(root, paired_phase, dataset, kir, seed, distance)
                if hard_rows:
                    break
    _write_csv(output / "hard_intent_analysis.csv", hard_rows)
    threshold_selection = _json(unit / "threshold_selection.json")
    representation_rows: list[dict[str, Any]] = []
    if include_representation_controls:
        representation_rows = compute_representation_control_rows(
            train_rows=split_rows["train"],
            validation_rows=split_rows["val"],
            test_rows=split_rows["test"],
            train_embeddings=embeddings["train"],
            validation_embeddings=embeddings["val"],
            test_embeddings=embeddings["test"],
            id_recall_guard=float(threshold_selection.get("id_recall_guard", 0.8)),
            metadata=metadata,
        )
        _write_csv(output / "representation_controls.csv", representation_rows)
    diagnostics = None
    if include_geometry:
        diagnostics = create_geometry_outputs(
            embeddings["train"], embeddings["test"], split_rows["test"],
            output / "figures", max_samples=max_samples, nonlinear=nonlinear,
        )
    analysis_manifest = {
        **metadata,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "source_unit": str(unit),
        "near_far_rows": len(near_far),
        "intent_pair_overlap_rows": len(intent_pair_rows),
        "hard_intent_rows": len(hard_rows),
        "representation_control_rows": len(representation_rows),
        "pca_diagnostics": diagnostics,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return analysis_manifest


def run_selected_grid(
    root: Path,
    *,
    phase: str = "tuned",
    seed: int = 42,
    nonlinear: bool = False,
    max_samples: int = 5000,
) -> dict[str, Any]:
    """分析 seed42 的 3 dataset × 3 KIR × 2 distance，共 18 个 selected-K 点。"""

    aggregate: dict[str, list[dict[str, Any]]] = {
        "near_far_oos_summary.csv": [],
        "empirical_overlap_summary.csv": [],
        "intent_pair_overlap.csv": [],
        "hard_intent_analysis.csv": [],
        "representation_controls.csv": [],
    }
    selected_units = []
    for dataset in DATASETS:
        for kir in (25, 50, 75):
            for distance in DISTANCES:
                k_gate = _selected_k(root, dataset, kir, seed, distance)
                # 只在每个数据集的 KIR50/Mahalanobis 代表点运行表征控制和图，
                # 其余 15 点只计算 near/far、overlap 与 hard-intent 数表。
                representative = kir == 50 and distance == "mahalanobis_diag"
                output = root / "analysis" / dataset / f"kir{kir}_seed{seed}" / distance / f"k{k_gate}"
                result = analyze(
                    root,
                    dataset=dataset,
                    kir=kir,
                    seed=seed,
                    phase=phase,
                    distance=distance,
                    k_gate=k_gate,
                    output_dir=output,
                    nonlinear=nonlinear and representative,
                    max_samples=max_samples,
                    include_geometry=representative,
                    include_representation_controls=representative,
                )
                selected_units.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "data_seed": seed,
                        "distance": distance,
                        "selected_k": k_gate,
                        "source_unit": result["source_unit"],
                    }
                )
                for filename in aggregate:
                    if filename == "representation_controls.csv" and not representative:
                        continue
                    path = output / filename
                    if path.is_file():
                        aggregate[filename].extend(pd.read_csv(path).to_dict("records"))

    analysis_root = root / "analysis"
    for filename, rows in aggregate.items():
        _write_csv(analysis_root / filename, rows)
    correlations = compute_hard_intent_correlations(
        aggregate["hard_intent_analysis.csv"], bootstrap_repeats=1000
    )
    _write_csv(analysis_root / "hard_intent_correlations.csv", correlations)
    manifest = {
        "status": "complete",
        "phase": phase,
        "selection_split": "validation",
        "selected_k_source": str(root / "selected_k_summary.csv"),
        "test_used_for_selection": False,
        "expected_unit_count": len(DATASETS) * 3 * len(DISTANCES),
        "completed_unit_count": len(selected_units),
        "selected_units": selected_units,
        "aggregate_row_counts": {name: len(rows) for name, rows in aggregate.items()},
        "hard_intent_correlation_rows": len(correlations),
    }
    (analysis_root / "selected_grid_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--study", choices=("unit", "stability", "selected-grid", "all"), default="unit")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--kir", type=int, choices=(25, 50, 75))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=("fixed", "tuned", "smoke"), default="tuned")
    parser.add_argument("--distance", choices=DISTANCES, default="mahalanobis_diag")
    parser.add_argument("--k-gate", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-nonlinear", action="store_true")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--stability-repeats", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if args.study == "unit":
        if args.dataset is None or args.kir is None:
            parser.error("--study unit requires --dataset and --kir")
        result = analyze(
            args.root, dataset=args.dataset, kir=args.kir, seed=args.seed,
            phase=args.phase, distance=args.distance, k_gate=args.k_gate,
            output_dir=args.output_dir, nonlinear=not args.skip_nonlinear,
            max_samples=args.max_samples,
        )
    elif args.study == "stability":
        result = run_stability_grid(
            args.root, repeats=args.stability_repeats, resume=not args.no_resume
        )
    elif args.study == "selected-grid":
        result = run_selected_grid(
            args.root, phase=args.phase, seed=args.seed,
            nonlinear=not args.skip_nonlinear, max_samples=args.max_samples,
        )
    else:
        stability = run_stability_grid(
            args.root, repeats=args.stability_repeats, resume=not args.no_resume
        )
        selected = run_selected_grid(
            args.root, phase=args.phase, seed=args.seed,
            nonlinear=not args.skip_nonlinear, max_samples=args.max_samples,
        )
        result = {"stability": stability, "selected_grid": selected}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
