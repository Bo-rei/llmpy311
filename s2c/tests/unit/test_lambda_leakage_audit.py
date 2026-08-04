from __future__ import annotations

import numpy as np

from tools.analysis.run_lambda_leakage_audit import _calibration_selection, _open_metrics


class _Sphere:
    def __init__(self, intent_name: str):
        self.intent_name = intent_name


class _FakeDetector:
    def __init__(self):
        self.radius_lambda = 0.0
        self.spheres = [_Sphere("known_a"), _Sphere("known_b")]

    def _compute_radii(self):
        return None

    def predict_with_scores(self, values):
        del values
        # The known-only guard becomes valid at lambda >= 1.0.
        pred = np.ones(4, dtype=np.int64) if self.radius_lambda < 1.0 else np.zeros(4, dtype=np.int64)
        return {"pred": pred, "score": pred.astype(float), "nearest_cluster": np.zeros(4, dtype=np.int64)}


def test_open_metrics_keeps_gate_and_known_class_metrics_separate():
    rows = [
        {"intent": "known_a", "oos_source": "known"},
        {"intent": "known_b", "oos_source": "known"},
        {"intent": "unknown", "oos_source": "heldout_intent"},
    ]
    output = {
        "score": np.asarray([0.1, 1.2, 1.4]),
        "pred": np.asarray([0, 1, 1]),
        "nearest_cluster": np.asarray([0, 1, 0]),
    }
    metrics = _open_metrics(rows, output, _FakeDetector())
    assert metrics["known_recall"] == 0.5
    assert metrics["f1_u"] == metrics["oos_f1"]
    assert 0.0 <= metrics["f1_all"] <= 1.0
    assert 0.0 <= metrics["known_macro_f1"] <= 1.0


def test_known_only_lambda_selection_does_not_consult_oos_values():
    selected, constraint_met, calibration_rates = _calibration_selection(_FakeDetector(), np.zeros(4))
    assert selected == 1.0
    assert constraint_met is True
    assert calibration_rates[0.75] == 1.0
    assert calibration_rates[1.0] == 0.0
