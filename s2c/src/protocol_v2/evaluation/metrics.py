"""Standard Gate metrics: OOS is positive and larger score means more OOS-like."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve


def compute_binary_oos_metrics(
    y_true_binary: Sequence[int] | np.ndarray,
    oos_scores: Sequence[float] | np.ndarray,
    threshold: float = 1.0,
) -> dict[str, float]:
    labels = np.asarray(y_true_binary, dtype=np.int64).reshape(-1)
    scores = np.asarray(oos_scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or not np.isin(labels, (0, 1)).all():
        raise ValueError("OOS labels must be aligned binary values: Known=0, OOS=1")
    if not np.isfinite(scores).all():
        raise ValueError("OOS scores must be finite")
    predicted = (scores > threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=[1], average=None, zero_division=0)
    known, oos = labels == 0, labels == 1
    if np.unique(labels).size == 2:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        valid_fpr = fpr[tpr >= 0.95]
        auroc = float(roc_auc_score(labels, scores))
        aupr = float(average_precision_score(labels, scores))
        fpr95 = float(valid_fpr.min()) if valid_fpr.size else math.nan
    else:
        auroc = aupr = fpr95 = math.nan
    false_accept = float(np.mean(predicted[oos] == 0)) if oos.any() else math.nan
    false_reject = float(np.mean(predicted[known] == 1)) if known.any() else math.nan
    return {
        "oos_precision": float(precision[0]),
        "oos_recall": float(recall[0]),
        "oos_f1": float(f1[0]),
        "id_recall": 1.0 - false_reject if known.any() else math.nan,
        "oos_rejection": float(recall[0]),
        "auroc": auroc,
        "aupr_oos": aupr,
        "fpr95": fpr95,
        "false_accept_rate": false_accept,
        "false_reject_rate": false_reject,
    }

