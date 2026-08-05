from __future__ import annotations

from pathlib import Path

from protocol_v2.experiments.mechanism_runner import load_e2_bundle
from protocol_v2.experiments.racal_v1.boundary import compare_reference, evaluate_open, fit_k1_detector
from protocol_v2.experiments.racal_v1.contracts import validate_bundle
from protocol_v2.runtime.paths import ProtocolV2Paths


def test_racal_frozen_k1_replays_e2_seed42() -> None:
    paths = ProtocolV2Paths.discover(Path(__file__).resolve().parents[2])
    bundle = load_e2_bundle(paths, "stackoverflow", 42, 0.50)
    contract = validate_bundle(bundle)
    assert contract["overlap_counts"] == {"train_calibration": 0, "train_test": 0, "calibration_test": 0}
    detector = fit_k1_detector(bundle.train, bundle.views.train, "mahalanobis_diag")
    metrics, predictions = evaluate_open(detector, bundle.test, bundle.views.test, 1.0)
    reference = paths.run_root / "e2_gate_core_dense" / (
        "protocol_v2_textoir_v1__stackoverflow__kir_0.50__seed_42__"
        "repr_frozen_minilm__k_1__dist_mahalanobis_diag__boundary_mean_std"
    )
    result = compare_reference(reference, metrics, predictions)
    assert result["within_tolerance"]
