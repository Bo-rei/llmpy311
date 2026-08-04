from __future__ import annotations

import numpy as np

from protocol_v2.experiments.ccsg import apply_calibration, calibrate, open_metrics, support_margin_features
from protocol_v2.experiments.urcsg import fit_detector


def _toy():
    train = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0], [0.0, 5.0], [0.0, 5.1]],
        dtype=np.float64,
    )
    intents = np.asarray(["a", "a", "b", "b", "c", "c"], dtype=object)
    return train, intents


def test_support_margin_is_finite_and_has_deterministic_top_two():
    train, intents = _toy()
    detector = fit_detector(train, intents, distance="euclidean", seed=42)
    features = support_margin_features(detector, train)
    assert features.support.shape == (6,)
    assert features.margin.shape == (6,)
    assert features.raw_intent.shape == (6,)
    assert np.isfinite(features.support).all()
    assert np.isfinite(features.margin).all()
    assert all(str(value) in {"a", "b", "c"} for value in features.top_intent)


def test_mixture_support_weights_sum_per_intent_and_calibration_is_known_only():
    train, intents = _toy()
    detector = fit_detector(train, intents, distance="euclidean", overrides={"a": 2}, seed=42)
    features = support_margin_features(detector, train)
    calibration = calibrate("ccsg", features, target_false_rejection=0.05)
    output = apply_calibration(features, calibration)
    assert np.isfinite(calibration.threshold)
    assert np.isfinite(output["score"]).all()
    assert calibration.calibration_false_rejection <= 0.05


def test_open_metrics_uses_supplied_predictions_not_a_hidden_threshold():
    rows = [
        {"intent": "a", "oos_source": "known"},
        {"intent": "b", "oos_source": "heldout_intent"},
    ]
    output = {
        "score": np.asarray([0.1, 0.2]),
        "pred": np.asarray([0, 1]),
        "prediction_intent": np.asarray(["a", "__oos__"], dtype=object),
    }
    metrics = open_metrics(rows, output)
    assert metrics["oos_f1"] == 1.0
    assert metrics["known_recall"] == 1.0
    assert metrics["f1_all"] == 1.0
