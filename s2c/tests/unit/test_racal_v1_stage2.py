from __future__ import annotations

from pathlib import Path

import numpy as np

from protocol_v2.experiments.racal_v1.stage2 import _fit_fixed_detector, _sample_audit, load_stage2_config


def test_stage2_config_is_pure_fixed_k1_k2_control() -> None:
    path = Path(__file__).resolve().parents[2] / "configs/experiments/protocol_v2_textoir_v1/racal_v1_stage2.yaml"
    config = load_stage2_config(path)
    assert config["k_values"] == [1, 2]
    assert config["distance"] == "mahalanobis_diag"
    assert config["radius_method"] == "mean_std"
    assert config["threshold"] == 1.0
    assert config["test_used_for_selection"] is False


def test_fixed_k2_has_two_spheres_per_intent() -> None:
    rng = np.random.default_rng(7)
    values = np.vstack([rng.normal(-2, 0.1, size=(12, 4)), rng.normal(2, 0.1, size=(12, 4))]).astype(np.float32)
    rows = [{"intent": "a"} for _ in range(12)] + [{"intent": "b"} for _ in range(12)]
    detector = _fit_fixed_detector(values, rows, 2)
    assert all(len(clusters) == 2 for clusters in detector.intent_to_clusters.values())
    assert all(sum(detector._train_cluster_labels == cluster_id) > 0 for cluster_id in range(4))


def test_sample_audit_marks_k2_new_oos_acceptance() -> None:
    rows = [{"sample_id": "x", "intent": "unknown", "label": 1}]
    k1 = [{"predicted_is_oos": 1, "nearest_cluster": 0, "selected_intent": "known", "selected_center": 0, "predicted_intent": "__oos__", "distance": 3.0, "radius": 1.0, "normalized_score": 3.0}]
    k2 = [{"predicted_is_oos": 0, "nearest_cluster": 1, "selected_intent": "known", "selected_center": 1, "predicted_intent": "known", "distance": 0.5, "radius": 1.0, "normalized_score": 0.5}]
    result = _sample_audit(rows, k1, k2)
    assert "k1_correct_reject_k2_wrong_accept_oos" in result[0]["categories"]
    assert result[0]["k2_selected_intent"] == "known"
