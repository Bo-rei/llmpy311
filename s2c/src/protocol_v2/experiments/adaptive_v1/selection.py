"""Known-only / ProxyOOS structure selection for RC-AMBL."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any

import numpy as np

from .calibration import fit_thresholds
from .contracts import AdaptiveConfig, CenterSpec, FitResult, SplitOperation
from .covariance import fit_center, fit_parent
from .evidence import EvidenceModel
from .partition import bootstrap_split_stability, pca_median_split, sse


def _class_variance(points: np.ndarray, epsilon: float) -> np.ndarray:
    center = np.asarray(points, dtype=np.float64).mean(axis=0)
    return np.var(np.asarray(points, dtype=np.float64) - center, axis=0) + float(epsilon)


def _initial_structure(values: np.ndarray, intents: np.ndarray, config: AdaptiveConfig) -> tuple[dict[str, list[CenterSpec]], dict[str, CenterSpec], dict[str, np.ndarray]]:
    centers: dict[str, list[CenterSpec]] = {}
    parents: dict[str, CenterSpec] = {}
    assignments: dict[str, np.ndarray] = {}
    for intent in sorted(str(x) for x in np.unique(intents)):
        indices = np.flatnonzero(intents.astype(str) == intent)
        points = values[indices]
        parent = fit_parent(points, intent=intent, config=config)
        parent.sample_indices = indices.copy()
        centers[intent] = [parent]
        parents[intent] = parent
        assignments[intent] = np.zeros(indices.size, dtype=np.int64)
    return centers, parents, assignments


def _make_model(centers: dict[str, list[CenterSpec]], parents: dict[str, CenterSpec], threshold_values: np.ndarray) -> tuple[EvidenceModel, Any]:
    raw = EvidenceModel(centers, parents, None)
    thresholds = fit_thresholds(raw, threshold_values)
    model = EvidenceModel(centers, parents, thresholds)
    return model, thresholds


def _metrics(model: EvidenceModel, values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    output = model.apply(values)
    known = np.asarray(labels, dtype=object)
    accepted = output.predicted_oos == 0
    known_recall = float(np.mean(accepted)) if accepted.size else 0.0
    wrong = float(np.mean(output.top_intent != known)) if known.size else 0.0
    return {"known_recall": known_recall, "ambiguity": wrong, "output": output}


def _candidate_priority(values: np.ndarray, intents: np.ndarray, calibration: np.ndarray, calibration_intents: np.ndarray, centers: dict[str, list[CenterSpec]], parents: dict[str, CenterSpec], assignments: dict[str, np.ndarray]) -> list[tuple[float, float, float, str, int]]:
    rows: list[tuple[float, float, float, str, int]] = []
    for intent in sorted(centers):
        points = values[intents.astype(str) == intent]
        local_centers = centers[intent]
        for center in local_centers:
            idx = center.sample_indices
            residual = float(np.mean(np.square(points - center.center))) if idx.size else 0.0
            cal_mask = calibration_intents.astype(str) == intent
            cal_points = calibration[cal_mask]
            parent_outside = 0.0
            if cal_points.size:
                parent_outside = float(np.mean(np.square(np.linalg.norm(cal_points - parents[intent].center, axis=1)) > np.square(parents[intent].radius)))
            density_risk = float(1.0 / max(center.sample_count, 1))
            rows.append((residual, parent_outside, density_risk, intent, int(center.local_id)))
    rows.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))
    return rows


def _proxy_false_accept(model: EvidenceModel, values: np.ndarray, intents: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    output = model.apply(values)
    return float(np.mean(output.predicted_oos == 0))


def _build_candidate(
    values: np.ndarray,
    intents: np.ndarray,
    centers: dict[str, list[CenterSpec]],
    parents: dict[str, CenterSpec],
    intent: str,
    parent_local_id: int,
    round_index: int,
    rho: float,
    config: AdaptiveConfig,
) -> tuple[dict[str, list[CenterSpec]], dict[str, np.ndarray], dict[str, Any]] | None:
    current = centers[intent][parent_local_id]
    points = values[current.sample_indices]
    labels, axis, info = pca_median_split(points)
    n_min = max(int(config.min_samples_absolute), int(np.ceil(config.min_samples_ratio * max(points.shape[0], 1))))
    sizes = tuple(int(np.sum(labels == side)) for side in (0, 1))
    if min(sizes) < n_min:
        return None
    candidate_centers = {key: list(value) for key, value in centers.items()}
    candidate_assignments: dict[str, np.ndarray] = {}
    for key in centers:
        ids = np.zeros(values[intents.astype(str) == key].shape[0], dtype=np.int64)
        for idx, c in enumerate(centers[key]):
            ids[np.isin(np.flatnonzero(intents.astype(str) == key), c.sample_indices)] = idx
        candidate_assignments[key] = ids
    local_points = [points[labels == side] for side in (0, 1)]
    class_var = _class_variance(values[intents.astype(str) == intent], config.covariance_epsilon)
    left = fit_center(local_points[0], intent=intent, local_id=parent_local_id, sample_indices=current.sample_indices[labels == 0], class_variance=class_var, rho=rho, config=config, stability=1.0, parent_local_id=parent_local_id, birth_round=round_index)
    right_id = max(c.local_id for c in centers[intent]) + 1
    right = fit_center(local_points[1], intent=intent, local_id=right_id, sample_indices=current.sample_indices[labels == 1], class_variance=class_var, rho=rho, config=config, stability=1.0, parent_local_id=parent_local_id, birth_round=round_index)
    candidate_centers[intent] = [c for c in centers[intent] if c.local_id != parent_local_id] + [left, right]
    candidate_centers[intent].sort(key=lambda c: c.local_id)
    return candidate_centers, candidate_assignments, {"sizes": sizes, "axis": axis.tolist(), "split_info": info, "n_min": n_min, "left": left, "right": right}


def _overlap_ratio(a: CenterSpec, b: CenterSpec) -> float:
    distance = float(np.linalg.norm(a.center - b.center))
    return distance / max(float(a.radius + b.radius), 1e-12)


def fit_rc_ambl(
    train_values: np.ndarray,
    train_intents: np.ndarray,
    calibration_select_values: np.ndarray,
    calibration_select_intents: np.ndarray,
    calibration_threshold_values: np.ndarray,
    *,
    mode: str,
    config: AdaptiveConfig,
    proxy_values: np.ndarray | None = None,
    proxy_intents: np.ndarray | None = None,
) -> FitResult:
    """Fit RC-AMBL structure without reading test rows.

    ``mode=KnownOnly`` uses only Known calibration geometry. ``ProxyOOS`` adds
    held-out Known-intent calibration rows as an explicit proxy risk signal;
    these rows are never test OOS and never used for final threshold fitting.
    """

    values = np.asarray(train_values, dtype=np.float64)
    intents = np.asarray(train_intents, dtype=object)
    centers, parents, assignments = _initial_structure(values, intents, config)
    operations: list[SplitOperation] = []
    audit: dict[str, Any] = {"mode": mode, "rounds": [], "candidate_count": 0, "accepted_count": 0, "rejected_count": 0, "test_used_for_selection": False}
    for round_index in range(int(config.max_rounds)):
        if max(len(items) for items in centers.values()) >= int(config.max_centers_per_intent) and all(len(items) >= int(config.max_centers_per_intent) for items in centers.values()):
            break
        priorities = _candidate_priority(values, intents, calibration_select_values, calibration_select_intents, centers, parents, assignments)
        chosen = None
        for _, _, _, intent, local_id in priorities:
            if len(centers[intent]) >= int(config.max_centers_per_intent):
                continue
            target = next((c for c in centers[intent] if c.local_id == local_id), None)
            if target is None or target.sample_count < 2 * max(config.min_samples_absolute, int(np.ceil(config.min_samples_ratio * target.sample_count))):
                continue
            chosen = (intent, local_id)
            break
        if chosen is None:
            audit["rounds"].append({"round": round_index, "status": "no_candidate"})
            break
        intent, parent_id = chosen
        target = next(c for c in centers[intent] if c.local_id == parent_id)
        intent_hash = int.from_bytes(hashlib.sha256(intent.encode("utf-8")).digest()[:4], "big")
        stability = bootstrap_split_stability(values[target.sample_indices], seed=config.seed + round_index * 997 + intent_hash % 997, repeats=config.bootstrap_repeats)
        audit["candidate_count"] += 1
        accepted_record = None
        last_candidate_record: SplitOperation | None = None
        rejection_reason = "no_valid_rho"
        baseline_raw = EvidenceModel(centers, parents, None)
        baseline_thresholds = fit_thresholds(baseline_raw, calibration_threshold_values)
        baseline = EvidenceModel(centers, parents, baseline_thresholds)
        base_metrics = _metrics(baseline, calibration_select_values, calibration_select_intents)
        base_proxy = _proxy_false_accept(baseline, proxy_values, proxy_intents) if mode == "ProxyOOS" and proxy_values is not None and proxy_intents is not None else None
        for rho in config.rho_candidates:
            candidate_data = _build_candidate(values, intents, centers, parents, intent, parent_id, round_index, float(rho), config)
            if candidate_data is None:
                rejection_reason = "child_below_min_size"
                continue
            candidate_centers, _, info = candidate_data
            # The parent boundary remains the original class boundary; the
            # child evidence can only refine it, never enlarge it.
            for child in candidate_centers[intent]:
                if child.parent_local_id == parent_id:
                    child.stability = float(stability["median"])
            candidate_raw = EvidenceModel(candidate_centers, parents, None)
            candidate_thresholds = fit_thresholds(candidate_raw, calibration_threshold_values)
            candidate = EvidenceModel(candidate_centers, parents, candidate_thresholds)
            candidate_metrics = _metrics(candidate, calibration_select_values, calibration_select_intents)
            known_delta = float(candidate_metrics["known_recall"] - base_metrics["known_recall"])
            ambiguity_delta = float(candidate_metrics["ambiguity"] - base_metrics["ambiguity"])
            proxy_delta = None
            if mode == "ProxyOOS" and proxy_values is not None and proxy_intents is not None:
                proxy_delta = float(_proxy_false_accept(candidate, proxy_values, proxy_intents) - (base_proxy or 0.0))
            compact_parent = sse(values[target.sample_indices])
            compact_children = sse(values[info["left"].sample_indices]) + sse(values[info["right"].sample_indices])
            compact_gain = float((compact_parent - compact_children) / max(compact_parent, 1e-12))
            complexity_gain = compact_gain - config.complexity_penalty
            safe = (
                known_delta >= -config.max_known_recall_drop
                and ambiguity_delta <= config.max_ambiguity_increase
                and float(stability["median"]) >= config.stability_threshold
                and compact_gain > 0.0
                and complexity_gain > 0.0
                and (proxy_delta is None or proxy_delta <= config.max_proxy_false_accept_increase)
            )
            record = SplitOperation(round_index, intent, parent_id, tuple(info["sizes"]), compact_gain, complexity_gain, float(stability["mean"]), float(stability["median"]), float(stability["min"]), float(rho), known_delta, ambiguity_delta, proxy_delta, bool(safe), None if safe else "safety_gate_failed")
            last_candidate_record = record
            if safe:
                accepted_record = (record, candidate_centers, candidate_thresholds, info)
                break
            rejection_reason = "safety_gate_failed"
        if accepted_record is None:
            if last_candidate_record is None:
                record = SplitOperation(round_index, intent, parent_id, (0, 0), 0.0, 0.0, float(stability["mean"]), float(stability["median"]), float(stability["min"]), float(config.rho_candidates[0]), 0.0, 0.0, None, False, rejection_reason)
            else:
                record = last_candidate_record
                record.reject_reason = rejection_reason
                record.split_accepted = False
            operations.append(record)
            audit["rejected_count"] += 1
            audit["rounds"].append({"round": round_index, "intent": intent, "status": "rejected", "reason": rejection_reason, "stability": stability})
            break
        record, centers, _, info = accepted_record
        operations.append(record)
        audit["accepted_count"] += 1
        audit["rounds"].append({"round": round_index, "intent": intent, "status": "accepted", "operation": asdict(record), "split_info": {k: v for k, v in info.items() if k not in {"left", "right"}}})
    final_raw = EvidenceModel(centers, parents, None)
    thresholds = fit_thresholds(final_raw, calibration_threshold_values)
    audit["final_k_by_intent"] = {intent: len(items) for intent, items in sorted(centers.items())}
    audit["proxy_oos_used_for_structure_selection"] = bool(mode == "ProxyOOS")
    audit["proxy_oos_used_for_threshold"] = False
    audit["calibration_select_used_for_structure"] = True
    audit["calibration_threshold_used_for_threshold"] = True
    return FitResult(centers=centers, parents=parents, operations=operations, thresholds=thresholds, selection_audit=audit, config=config)
