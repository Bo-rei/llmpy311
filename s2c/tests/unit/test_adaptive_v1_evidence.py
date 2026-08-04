import numpy as np

from protocol_v2.experiments.adaptive_v1.calibration import fit_thresholds
from protocol_v2.experiments.adaptive_v1.contracts import AdaptiveConfig
from protocol_v2.experiments.adaptive_v1.covariance import fit_parent
from protocol_v2.experiments.adaptive_v1.evidence import EvidenceModel


def test_evidence_is_finite_and_thresholded():
    x = np.vstack([np.zeros((20, 3)), np.ones((20, 3))])
    labels = np.asarray(["a"] * 20 + ["b"] * 20, dtype=object)
    config = AdaptiveConfig()
    parents = {intent: fit_parent(x[labels == intent], intent=intent, config=config) for intent in ("a", "b")}
    centers = {intent: [parents[intent]] for intent in parents}
    raw = EvidenceModel(centers, parents)
    thresholds = fit_thresholds(raw, x)
    output = EvidenceModel(centers, parents, thresholds).apply(x)
    assert np.isfinite(output.oos_score).all()
    assert np.isfinite(output.energy).all()
