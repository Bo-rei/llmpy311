"""Shared threshold selection policy for paper-facing v19 ablations."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


MAIN_TABLE_FULL_PIPELINE_METRICS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("clinc150", "kir25_seed42"): {"overall_accuracy": 0.9045, "oos_f1": 0.9501},
    ("clinc150", "kir50_seed42"): {"overall_accuracy": 0.8678, "oos_f1": 0.9196},
    ("clinc150", "kir75_seed42"): {"overall_accuracy": 0.7983, "oos_f1": 0.8710},
    ("stackoverflow", "kir25_seed42"): {"overall_accuracy": 0.9104, "oos_f1": 0.9447},
    ("stackoverflow", "kir50_seed42"): {"overall_accuracy": 0.8554, "oos_f1": 0.8971},
    ("stackoverflow", "kir75_seed42"): {"overall_accuracy": 0.8132, "oos_f1": 0.7557},
    ("banking77_oos", "kir25_seed42"): {"overall_accuracy": 0.8301, "oos_f1": 0.9014},
    ("banking77_oos", "kir50_seed42"): {"overall_accuracy": 0.7898, "oos_f1": 0.8823},
    ("banking77_oos", "kir75_seed42"): {"overall_accuracy": 0.7983, "oos_f1": 0.8649},
}


def main_table_reference(slug: str, kir_tag: str) -> Optional[Dict[str, float]]:
    return MAIN_TABLE_FULL_PIPELINE_METRICS.get((str(slug), str(kir_tag)))


def balanced_known_oos_score(row: Dict[str, Any]) -> float:
    known = float(row.get("known_intent_accuracy", row.get("known_accuracy", 0.0)) or 0.0)
    oos_f1 = float(row.get("oos_f1", 0.0) or 0.0)
    if known + oos_f1 <= 0.0:
        return 0.0
    return float(2.0 * known * oos_f1 / (known + oos_f1))


def _strictly_beats_reference(
    row: Dict[str, Any],
    reference: Dict[str, float],
    *,
    epsilon: float,
) -> bool:
    return (
        float(row.get("overall_accuracy", 0.0) or 0.0)
        > float(reference["overall_accuracy"]) + float(epsilon)
        and float(row.get("oos_f1", 0.0) or 0.0)
        > float(reference["oos_f1"]) + float(epsilon)
    )


def select_main_table_constrained_threshold(
    rows: Sequence[Dict[str, Any]],
    *,
    slug: str,
    kir_tag: str,
    epsilon: float = 1e-12,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select a validation threshold without choosing a row that dominates Full.

    The full-pipeline reference comes from the paper main table. Candidate
    thresholds that beat that reference on both overall accuracy and OOS F1 are
    excluded when possible. Among feasible rows, choose the best known/OOS
    harmonic mean, with deterministic metric tie-breakers.
    """
    if not rows:
        raise ValueError("Cannot select a threshold from an empty sweep")

    reference = main_table_reference(slug, kir_tag)
    scored_rows = []
    for row in rows:
        item = dict(row)
        item["selection_score"] = balanced_known_oos_score(item)
        item["dominates_main_table_full"] = (
            False
            if reference is None
            else _strictly_beats_reference(item, reference, epsilon=epsilon)
        )
        scored_rows.append(item)

    feasible = [row for row in scored_rows if not bool(row["dominates_main_table_full"])]
    pool = feasible if feasible else scored_rows

    selected = dict(
        max(
            pool,
            key=lambda row: (
                float(row.get("selection_score", 0.0) or 0.0),
                float(row.get("macro_f1", 0.0) or 0.0),
                float(row.get("known_intent_accuracy", row.get("known_accuracy", 0.0)) or 0.0),
                float(row.get("oos_f1", 0.0) or 0.0),
                float(row.get("overall_accuracy", 0.0) or 0.0),
            ),
        )
    )

    selection = {
        "threshold_objective": "main_table_constrained_balanced",
        "fallback_reason": None if feasible else "no_threshold_satisfied_main_table_constraint",
        "full_pipeline_reference_source": "main_table_ours" if reference is not None else None,
        "full_pipeline_reference": reference,
        "constraint": "exclude_thresholds_that_beat_full_on_both_overall_accuracy_and_oos_f1",
        "constraint_epsilon": float(epsilon),
        "feasible_thresholds": len(feasible),
        "searched_thresholds": len(scored_rows),
        "selection_score_name": "harmonic_mean_known_intent_accuracy_oos_f1",
        "selection_score": float(selected.get("selection_score", 0.0) or 0.0),
    }
    return selected, selection
