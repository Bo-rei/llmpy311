"""MiniLM 多簇可分性实验的共享评价协议。

本模块只操作数组和普通字典，目的是让 Runner、分析脚本和测试共用同一套指标语义，
避免各脚本分别实现后出现分数方向或正负类不一致。全模块固定约定：``OOS=1``，
分数越大越倾向 OOS。历史 ``f1_like`` 不属于这套标准指标。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)


def _as_binary_vector(values: Sequence[int] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.int64).reshape(-1)
    if not np.isin(vector, (0, 1)).all():
        raise ValueError("y_true_binary must contain only 0 (known) and 1 (OOS)")
    return vector


def compute_binary_oos_metrics(
    y_true_binary: Sequence[int] | np.ndarray,
    oos_scores: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """计算新实验统一使用的 Gate 二分类指标。

    为了与历史 Gate 决策保持一致，只有 ``score > threshold`` 才预测为 OOS；
    等于阈值时仍视为 Known。如果输入只包含一个真实类别，AUROC、AUPR 和
    FPR95 在数学上没有定义，因此返回 NaN，而不是误导性的 0。
    """

    y_true = _as_binary_vector(y_true_binary)
    scores = np.asarray(oos_scores, dtype=np.float64).reshape(-1)
    if y_true.size != scores.size:
        raise ValueError("y_true_binary and oos_scores must have equal length")
    if not np.isfinite(scores).all():
        raise ValueError("oos_scores must be finite")

    prediction = (scores > float(threshold)).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        prediction,
        labels=[1],
        average=None,
        zero_division=0,
    )
    known = y_true == 0
    oos = y_true == 1
    id_recall = float(np.mean(prediction[known] == 0)) if known.any() else math.nan
    oos_rejection = float(np.mean(prediction[oos] == 1)) if oos.any() else math.nan

    rank_defined = np.unique(y_true).size == 2
    if rank_defined:
        auroc = float(roc_auc_score(y_true, scores))
        aupr = float(average_precision_score(y_true, scores))
        fpr, tpr, _ = roc_curve(y_true, scores, pos_label=1)
        # FPR95 表示 OOS 召回率至少 95% 时，Known 被误拒绝的最小比例。
        # 这里不做插值，直接在 ROC 的实际候选点中取最小值，便于复现。
        eligible = fpr[tpr >= 0.95]
        fpr95 = float(np.min(eligible)) if eligible.size else math.nan
    else:
        auroc = aupr = fpr95 = math.nan

    return {
        "oos_precision": float(precision[0]),
        "oos_recall": float(recall[0]),
        "oos_f1": float(f1[0]),
        "id_recall": id_recall,
        "oos_rejection": oos_rejection,
        "auroc": auroc,
        "aupr_oos": aupr,
        "fpr95": fpr95,
    }


def gold_sample_kind(record: Mapping[str, Any], manifest: Mapping[str, Any] | None = None) -> str:
    """仅根据真实标签和数据 manifest 返回样本分组。

    此函数绝不读取 ``is_oos``、``gate_pred`` 等预测字段。否则 false accept
    和 false reject 会被分到错误类别，进而污染可视化与分层统计。
    """

    label = record.get("true_gate_label", record.get("true_binary_label", record.get("label")))
    if label is None:
        raise ValueError("record has no ground-truth binary label")
    if int(label) == 0:
        return "known"
    if int(label) != 1:
        raise ValueError(f"unsupported ground-truth binary label: {label!r}")

    source = str(record.get("oos_source", record.get("source_split", ""))).lower()
    if source in {
        "native_oos",
        "provided_oos",
        "native_or_provided_oos",
        "clinc_oos",
        "id-oos",
        "ood-oos",
    }:
        return "native_or_provided_oos"
    if source in {"heldout_unknown", "heldout_oos", "unknown_intent"}:
        return "heldout_unknown"

    intent = str(record.get("true_intent", record.get("intent", "")))
    known_intents = {str(item) for item in (manifest or {}).get("known_intents", [])}
    unknown_intents = {str(item) for item in (manifest or {}).get("unknown_intents", [])}
    if intent in unknown_intents or (known_intents and intent not in known_intents and intent.lower() != "oos"):
        return "heldout_unknown"
    if intent.lower() in {"oos", "out_of_scope", "out-of-scope"}:
        return "native_or_provided_oos"
    return "heldout_unknown"


def compute_cluster_quality_metrics(
    embeddings: Sequence[Sequence[float]] | np.ndarray,
    cluster_labels: Sequence[int] | np.ndarray,
) -> dict[str, float | int]:
    """计算单个 intent 内的聚类质量诊断。

    K=1 时 Silhouette、Davies-Bouldin 和 Calinski-Harabasz 无定义，
    因此显式返回 NaN；不能填 0，否则汇总时会把“不可计算”误解为差聚类。
    """

    x = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(cluster_labels).reshape(-1)
    if x.ndim != 2 or x.shape[0] != labels.size:
        raise ValueError("embeddings must be 2-D and align with cluster_labels")
    unique = np.unique(labels)
    wcss = 0.0
    sizes: list[int] = []
    for label in unique:
        points = x[labels == label]
        center = np.mean(points, axis=0)
        wcss += float(np.sum((points - center) ** 2))
        sizes.append(int(points.shape[0]))

    valid_rank_metrics = 1 < unique.size < x.shape[0]
    return {
        "cluster_count": int(unique.size),
        "wcss": float(wcss),
        "minimum_cluster_size": int(min(sizes)) if sizes else 0,
        "minimum_cluster_ratio": float(min(sizes) / x.shape[0]) if sizes and x.shape[0] else math.nan,
        "silhouette": float(silhouette_score(x, labels)) if valid_rank_metrics else math.nan,
        "davies_bouldin": float(davies_bouldin_score(x, labels)) if valid_rank_metrics else math.nan,
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)) if valid_rank_metrics else math.nan,
    }


def compute_coverage_counts(
    distances: Sequence[Sequence[float]] | np.ndarray,
    radii: Sequence[float] | np.ndarray,
    sphere_intents: Sequence[str],
) -> np.ndarray:
    """统计每个样本同时被多少个不同 intent 的接受区域覆盖。

    同一 intent 的多个子球只计一次。因此 ``coverage_count>=2`` 表示跨意图
    边界重叠，而不是同一意图内部的子簇重叠。
    """

    matrix = np.asarray(distances, dtype=np.float64)
    radius = np.asarray(radii, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != radius.size or radius.size != len(sphere_intents):
        raise ValueError("distance columns, radii, and sphere_intents must align")
    accepted = matrix <= radius[None, :]
    intents = np.asarray([str(item) for item in sphere_intents], dtype=object)
    counts = np.zeros(matrix.shape[0], dtype=np.int64)
    for intent in np.unique(intents):
        counts += np.any(accepted[:, intents == intent], axis=1).astype(np.int64)
    return counts


def _simple_margin_key(value: Any) -> tuple[int, float]:
    return (1, 0.0) if value is None else (0, float(value))


def select_boundary(candidates: Sequence[Mapping[str, Any]], id_recall_guard: float) -> dict[str, Any]:
    """仅使用 validation 指标选择边界，不接触 test 结果。

    先过滤满足 ID-recall guard 的候选，再依次比较 OOS F1、FPR95 和
    ID recall。并列时优先无 margin、且 ``radius_lambda`` 更接近 1 的简单边界。
    若所有候选均违反 guard，则改为优先保全 ID recall，并显式记录
    ``guard_violation=True``，不隐藏协议失败。
    """

    if not candidates:
        raise ValueError("at least one boundary candidate is required")
    eligible = [item for item in candidates if float(item["metrics"]["id_recall"]) >= id_recall_guard]
    guard_violation = not eligible
    pool = list(candidates) if guard_violation else eligible

    if guard_violation:
        key = lambda item: (
            float(item["metrics"]["id_recall"]),
            float(item["metrics"]["oos_f1"]),
            -float(item["metrics"]["fpr95"]),
            _simple_margin_key(item.get("margin_gamma")),
            -abs(float(item.get("radius_lambda", 1.0)) - 1.0),
        )
    else:
        key = lambda item: (
            float(item["metrics"]["oos_f1"]),
            -float(item["metrics"]["fpr95"]),
            float(item["metrics"]["id_recall"]),
            _simple_margin_key(item.get("margin_gamma")),
            -abs(float(item.get("radius_lambda", 1.0)) - 1.0),
        )
    selected = dict(max(pool, key=key))
    selected["guard_violation"] = guard_violation
    return selected


def select_dataset_k(validation_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """在 dataset-level 上用 validation 指标选 K，并在并列时优先较小 K。

    这不是 per-intent K：整个数据集共享一个 K，以避免在 validation OOS
    上搜索指数级的 intent-wise 组合并导致过拟合。
    """

    if not validation_results:
        raise ValueError("at least one K result is required")
    selected = max(
        validation_results,
        key=lambda item: (
            float(item["metrics"]["oos_f1"]),
            -float(item["metrics"]["fpr95"]),
            float(item["metrics"]["id_recall"]),
            -int(item["k_gate"]),
        ),
    )
    return dict(selected)
