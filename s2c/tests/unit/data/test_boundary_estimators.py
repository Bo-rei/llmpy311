"""Protocol_v2 boundary estimators use only fitted Known support samples."""

from types import SimpleNamespace

import numpy as np

from protocol_v2.experiments.boundaries import apply_radius_estimator, known_conformal_threshold


class _Detector:
    def __init__(self) -> None:
        self._train_embeddings = np.asarray([[0.0], [1.0], [2.0], [10.0], [12.0]], dtype=np.float32)
        self._train_cluster_labels = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
        self.spheres = [
            SimpleNamespace(cluster_id=0, center=np.asarray([0.0]), radius=99.0),
            SimpleNamespace(cluster_id=1, center=np.asarray([10.0]), radius=99.0),
        ]

    @staticmethod
    def _distance(point: np.ndarray, sphere: SimpleNamespace) -> float:
        return float(np.linalg.norm(point - sphere.center))


def test_quantile_and_mad_replace_only_cluster_radii() -> None:
    detector = _Detector()
    quantile = apply_radius_estimator(detector, "quantile_90")
    assert quantile.threshold == 1.0
    assert detector.spheres[0].radius == np.quantile([0.0, 1.0, 2.0], 0.90)
    assert detector.spheres[1].radius == np.quantile([0.0, 2.0], 0.90)

    detector = _Detector()
    mad = apply_radius_estimator(detector, "median_mad")
    assert mad.details["radii"] == [detector.spheres[0].radius, detector.spheres[1].radius]
    assert all(radius > 0.0 for radius in mad.details["radii"])


def test_known_conformal_threshold_is_an_order_statistic_without_oos() -> None:
    result = known_conformal_threshold(np.asarray([0.1, 0.2, 0.4, 0.8]), alpha=0.25)
    assert result.threshold == 0.8
    assert result.details == {"alpha": 0.25, "known_calibration_count": 4, "order_statistic_rank": 4}
