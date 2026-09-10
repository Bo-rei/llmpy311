from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "analysis" / "build_historical_oos_visual_explanation.py"
SPEC = importlib.util.spec_from_file_location("historical_oos_visual_explanation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_score_from_signature_uses_raw_nearest_center_then_radius() -> None:
    signature = {
        "distance_metric": "mahalanobis_diag",
        "acceptance_mode": "nearest_sphere",
        "spheres": [
            {
                "cluster_id": 3,
                "intent": "left",
                "center": [1.0, 0.0],
                "radius": 0.20,
                "inv_diag_cov": [1.0, 1.0],
            },
            {
                "cluster_id": 9,
                "intent": "right",
                "center": [0.0, 1.0],
                "radius": 2.0,
                "inv_diag_cov": [1.0, 1.0],
            },
        ],
    }
    output = MODULE.score_from_signature(np.asarray([[0.8, 0.2], [0.1, 0.9]]), signature)
    assert output["nearest_cluster"].tolist() == [3, 9]
    assert output["pred"].tolist() == [1, 0]
    np.testing.assert_allclose(output["score"], output["distance"] / output["radius"])


def test_smoothed_histogram_keeps_overflow_in_final_bin() -> None:
    bins = np.linspace(0.0, 1.0, 6)
    centers, density, overflow = MODULE.smoothed_histogram([0.1, 0.3, 0.9, 1.2], bins)
    assert overflow == 1
    assert len(centers) == len(density) == 5
    np.testing.assert_allclose(np.sum(density) * (bins[1] - bins[0]), 1.0)


def test_merge_oos_pair_closes_exact_score_decomposition() -> None:
    frozen = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "gold_intent": "oos_a",
                "gold_is_oos": 1,
                "predicted_is_oos": 0,
                "oos_score": 0.8,
                "distance": 0.8,
                "radius": 1.0,
                "nearest_cluster": 0,
            },
            {
                "sample_id": "b",
                "gold_intent": "known",
                "gold_is_oos": 0,
                "predicted_is_oos": 0,
                "oos_score": 0.5,
                "distance": 0.5,
                "radius": 1.0,
                "nearest_cluster": 0,
            },
        ]
    )
    trainable = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "gold_intent": "oos_a",
                "gold_is_oos": 1,
                "predicted_is_oos": 1,
                "oos_score": 1.2,
                "distance": 1.0,
                "radius": 5.0 / 6.0,
                "nearest_cluster": 0,
            },
            {
                "sample_id": "b",
                "gold_intent": "known",
                "gold_is_oos": 0,
                "predicted_is_oos": 0,
                "oos_score": 0.4,
                "distance": 0.4,
                "radius": 1.0,
                "nearest_cluster": 0,
            },
        ]
    )
    predictions = {
        ("stackoverflow", 42, "frozen_k1"): frozen,
        ("stackoverflow", 42, "trainable_k1"): trainable,
    }
    merged = MODULE.merge_oos_pair(predictions, "stackoverflow", 42)
    assert merged["transition"].tolist() == ["trainable_only"]
    assert np.isclose(merged["delta_score"].iloc[0], 0.4)
    np.testing.assert_allclose(
        merged["distance_contribution"] + merged["radius_contribution"],
        merged["delta_score"],
    )
