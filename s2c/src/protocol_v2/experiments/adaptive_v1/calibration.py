"""Leakage-safe calibration split and finite-sample threshold fitting."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from .contracts import CalibrationThresholds
from .evidence import EvidenceModel


def ids_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["sample_id"]) for row in rows).encode("utf-8")).hexdigest()


def split_calibration_rows(rows: list[dict[str, Any]], seed: int, select_fraction: float = 0.5) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Deterministically split Known calibration rows per intent.

    A SeedSequence child is generated from the experiment seed and the stable
    intent order. No text or test row is consulted.
    """

    if not 0.0 < select_fraction < 1.0:
        raise ValueError("select_fraction must be between zero and one")
    by_intent: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row.get("label", 0)) != 0:
            raise ValueError("calibration split must be Known-only")
        by_intent.setdefault(str(row["intent"]), []).append(row)
    root = np.random.SeedSequence(int(seed))
    children = root.spawn(len(sorted(by_intent)))
    select: list[dict[str, Any]] = []
    threshold: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"seed": int(seed), "select_fraction": float(select_fraction), "per_intent": {}}
    for child, intent in zip(children, sorted(by_intent)):
        values = list(by_intent[intent])
        rng = np.random.default_rng(child)
        order = rng.permutation(len(values))
        n_select = min(max(1, int(round(len(values) * select_fraction))), len(values) - 1) if len(values) > 1 else 1
        selected_indices = set(int(i) for i in order[:n_select])
        selected = [row for i, row in enumerate(values) if i in selected_indices]
        held = [row for i, row in enumerate(values) if i not in selected_indices]
        select.extend(selected)
        threshold.extend(held)
        audit["per_intent"][intent] = {"select_ids_sha256": ids_hash(selected), "threshold_ids_sha256": ids_hash(held), "select_count": len(selected), "threshold_count": len(held)}
    audit["select_ids_sha256"] = ids_hash(select)
    audit["threshold_ids_sha256"] = ids_hash(threshold)
    return select, threshold, audit


def finite_order(values: np.ndarray, quantile: float, *, upper: bool) -> tuple[float, int]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    if finite.size == 0 or not np.isfinite(finite).all():
        raise ValueError("threshold values must be finite and non-empty")
    n = finite.size
    if upper:
        rank = min(int(math.ceil((n + 1) * float(quantile))), n)
    else:
        rank = max(int(math.floor((n + 1) * float(quantile))), 1)
    return float(np.partition(finite, rank - 1)[rank - 1]), int(rank)


def fit_thresholds(model: EvidenceModel, values: np.ndarray, target_false_rejection: float = 0.05, margin_quantile: float = 0.05) -> CalibrationThresholds:
    energy, parent, gap, _, _ = model.raw(values)
    tau, upper_rank = finite_order(energy, 1.0 - float(target_false_rejection), upper=True)
    tau_parent, _ = finite_order(parent, 1.0 - float(target_false_rejection), upper=True)
    delta, lower_rank = finite_order(gap, float(margin_quantile), upper=False)
    return CalibrationThresholds(tau=tau, tau_parent=tau_parent, delta=delta, threshold_source="calibration_threshold_known_only", n_threshold=int(len(values)), upper_rank=upper_rank, lower_rank=lower_rank)

