from tools.legacy.analysis_v19.run_historical_trainable_cascade_v1 import serialize_detector_signature


def test_serialize_detector_signature_preserves_sphere_contract():
    payload = serialize_detector_signature(
        {
            "distance_metric": "mahalanobis_diag",
            "radius_method": "mean_std",
            "radius_lambda": 1.0,
            "acceptance_mode": "nearest_sphere",
            "spheres": [
                {
                    "cluster_id": 1,
                    "intent": "b",
                    "center": [1.0, 2.0],
                    "radius": 3.0,
                    "inv_diag_cov": [4.0, 5.0],
                },
                {
                    "cluster_id": 0,
                    "intent": "a",
                    "center": [0.0, 1.0],
                    "radius": 2.0,
                    "inv_diag_cov": [6.0, 7.0],
                },
            ],
        }
    )

    assert payload["n_clusters"] == 2
    assert [item["cluster_id"] for item in payload["spheres"]] == [0, 1]
    assert payload["intent_to_cluster"] == {"a": 0, "b": 1}
    assert payload["distance_metric"] == "mahalanobis_diag"
    assert payload["l2_normalize"] is True
