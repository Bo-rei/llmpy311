from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.cluster_separability import protocol


def test_binary_oos_metrics_use_oos_as_positive_and_high_scores_as_oos():
    y_true = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    scores = np.asarray([0.1, 0.2, 0.3, 0.8, 0.4, 0.5, 0.6, 0.7])

    metrics = protocol.compute_binary_oos_metrics(y_true, scores, threshold=0.45)

    assert metrics["oos_precision"] == pytest.approx(0.75)
    assert metrics["oos_recall"] == pytest.approx(0.75)
    assert metrics["oos_f1"] == pytest.approx(0.75)
    assert metrics["id_recall"] == pytest.approx(0.75)
    assert metrics["oos_rejection"] == pytest.approx(0.75)
    assert metrics["auroc"] == pytest.approx(0.75)
    assert 0.0 <= metrics["aupr_oos"] <= 1.0


def test_fpr95_is_minimum_known_false_reject_rate_at_oos_tpr_95():
    y_true = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    scores = np.asarray([0.1, 0.2, 0.3, 0.8, 0.4, 0.5, 0.6, 0.7])

    metrics = protocol.compute_binary_oos_metrics(y_true, scores, threshold=0.45)

    # With four OOS examples, TPR >= .95 requires accepting all four as OOS.
    # The highest feasible threshold is .4, at which only the known score .8
    # is falsely rejected.
    assert metrics["fpr95"] == pytest.approx(0.25)


@pytest.mark.parametrize("label", [0, 1])
def test_rank_metrics_are_na_for_a_single_class(label: int):
    y_true = np.full(3, label, dtype=int)
    metrics = protocol.compute_binary_oos_metrics(
        y_true,
        np.asarray([0.1, 0.2, 0.3]),
        threshold=0.2,
    )

    assert math.isnan(metrics["auroc"])
    assert math.isnan(metrics["aupr_oos"])
    assert math.isnan(metrics["fpr95"])


def test_gold_sample_kind_ignores_prediction_fields():
    manifest = {"known_intents": ["weather"]}

    assert protocol.gold_sample_kind(
        {
            "label": 0,
            "true_intent": "weather",
            "is_oos": True,
            "gate_pred": "oos",
        },
        manifest,
    ) == "known"
    assert protocol.gold_sample_kind(
        {
            "label": 1,
            "true_intent": "calendar",
            "oos_source": "heldout_unknown",
            "is_oos": False,
            "gate_pred": "known",
        },
        manifest,
    ) == "heldout_unknown"
    assert protocol.gold_sample_kind(
        {
            "label": 1,
            "true_intent": "oos",
            "oos_source": "native_oos",
            "is_oos": False,
        },
        manifest,
    ) == "native_or_provided_oos"
    assert protocol.gold_sample_kind(
        {
            "label": 1,
            "true_intent": "cash_withdrawal",
            "oos_source": "provided_oos",
        },
        manifest,
    ) == "native_or_provided_oos"


def test_k1_cluster_quality_reports_undefined_indices_as_na():
    embeddings = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    labels = np.zeros(3, dtype=int)

    metrics = protocol.compute_cluster_quality_metrics(embeddings, labels)

    assert metrics["cluster_count"] == 1
    assert metrics["wcss"] == pytest.approx(0.5)
    assert math.isnan(metrics["silhouette"])
    assert math.isnan(metrics["davies_bouldin"])
    assert math.isnan(metrics["calinski_harabasz"])


def test_coverage_count_counts_distinct_intents_not_spheres():
    distances = np.asarray(
        [
            [0.5, 0.7, 2.0],
            [0.5, 0.4, 0.3],
            [1.5, 1.1, 0.2],
        ]
    )
    radii = np.asarray([1.0, 1.0, 0.5])
    sphere_intents = ["intent_a", "intent_a", "intent_b"]

    counts = protocol.compute_coverage_counts(distances, radii, sphere_intents)

    assert counts.tolist() == [1, 2, 1]


def test_boundary_selection_uses_guard_then_validation_oos_f1():
    candidates = [
        {
            "radius_lambda": 0.75,
            "margin_gamma": 0.98,
            "metrics": {"id_recall": 0.89, "oos_f1": 0.99, "fpr95": 0.01},
            "test_metrics": {"oos_f1": 1.0},
        },
        {
            "radius_lambda": 1.0,
            "margin_gamma": None,
            "metrics": {"id_recall": 0.91, "oos_f1": 0.88, "fpr95": 0.10},
            "test_metrics": {"oos_f1": 0.0},
        },
        {
            "radius_lambda": 1.25,
            "margin_gamma": None,
            "metrics": {"id_recall": 0.92, "oos_f1": 0.86, "fpr95": 0.05},
            "test_metrics": {"oos_f1": 1.0},
        },
    ]

    selected = protocol.select_boundary(candidates, id_recall_guard=0.90)

    assert selected["radius_lambda"] == 1.0
    assert selected["guard_violation"] is False


def test_boundary_selection_has_deterministic_guard_violation_fallback():
    candidates = [
        {
            "radius_lambda": 0.75,
            "margin_gamma": 0.98,
            "metrics": {"id_recall": 0.84, "oos_f1": 0.90, "fpr95": 0.10},
        },
        {
            "radius_lambda": 1.0,
            "margin_gamma": None,
            "metrics": {"id_recall": 0.87, "oos_f1": 0.80, "fpr95": 0.20},
        },
        {
            "radius_lambda": 1.25,
            "margin_gamma": None,
            "metrics": {"id_recall": 0.87, "oos_f1": 0.82, "fpr95": 0.30},
        },
    ]

    selected = protocol.select_boundary(candidates, id_recall_guard=0.90)

    assert selected["radius_lambda"] == 1.25
    assert selected["guard_violation"] is True


def test_dataset_k_selection_breaks_metric_ties_toward_smaller_k():
    validation_results = [
        {"k_gate": 3, "metrics": {"oos_f1": 0.90, "fpr95": 0.10, "id_recall": 0.91}},
        {"k_gate": 2, "metrics": {"oos_f1": 0.90, "fpr95": 0.10, "id_recall": 0.91}},
        {"k_gate": 1, "metrics": {"oos_f1": 0.89, "fpr95": 0.05, "id_recall": 0.95}},
    ]

    selected = protocol.select_dataset_k(validation_results)

    assert selected["k_gate"] == 2
