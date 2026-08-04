from __future__ import annotations

import numpy as np

from protocol_v2.experiments.adaptive_split_merge import (
    AdaptiveSplitMergeConfig,
    SplitMetrics,
    compactness_gain,
    evaluate_split,
    fit_split_merge,
    merge_small_clusters,
)


def test_compactness_gain_is_positive_for_two_separated_children() -> None:
    parent = np.asarray([[-2.0, 0.0], [-1.8, 0.0], [1.8, 0.0], [2.0, 0.0]])
    children = [parent[:2], parent[2:]]
    assert compactness_gain(parent, children) > 0.9


def test_merge_small_clusters_obeys_minimum_size_when_target_exists() -> None:
    features = np.asarray([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2], [20.0]])
    labels = np.asarray([0, 0, 0, 1, 1, 1, 2])
    merged = merge_small_clusters(features, labels, n_min=3)
    counts = np.bincount(merged)
    assert counts.min() >= 3


def test_missing_cross_intent_signal_rejects_split() -> None:
    config = AdaptiveSplitMergeConfig(tau_compact=0.1, n_min=2, tau_stability=0.5, epsilon=0.02)
    decision = evaluate_split(
        SplitMetrics(0.9, 4, 0.9, None, 0.8),
        config,
    )
    assert not decision.accepted
    assert "cross_intent_acceptance_not_measured" in decision.reasons


def test_each_gate_can_reject_candidate() -> None:
    config = AdaptiveSplitMergeConfig(tau_compact=0.5, n_min=5, tau_stability=0.8, epsilon=0.02)
    decision = evaluate_split(SplitMetrics(0.1, 2, 0.2, 0.5, -0.1), config)
    assert not decision.accepted
    assert {
        "compactness_gain_below_threshold",
        "child_below_min_size",
        "stability_below_threshold",
        "cross_intent_acceptance_increase_above_epsilon",
        "complexity_adjusted_gain_non_positive",
    } <= set(decision.reasons)


def test_split_merge_is_reproducible_and_requires_cross_intent_signal() -> None:
    rng = np.random.default_rng(3)
    features = np.concatenate([rng.normal(-2, 0.1, size=(10, 2)), rng.normal(2, 0.1, size=(10, 2))])
    config = AdaptiveSplitMergeConfig(max_k=2, n_min=4, tau_stability=0.5, bootstrap_repeats=3)
    rejected = fit_split_merge(features, seed=42, config=config)
    assert len(np.unique(rejected.labels)) == 1
    assert rejected.decisions[0].reasons == ("cross_intent_acceptance_not_measured",)
    left = fit_split_merge(features, seed=42, config=config, cross_intent_acceptance_increase=0.0)
    right = fit_split_merge(features, seed=42, config=config, cross_intent_acceptance_increase=0.0)
    assert len(np.unique(left.labels)) == 2
    assert np.array_equal(left.labels, right.labels)
    assert np.allclose(left.centers, right.centers)
