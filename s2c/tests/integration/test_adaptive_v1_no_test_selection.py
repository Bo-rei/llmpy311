from protocol_v2.experiments.adaptive_v1.calibration import split_calibration_rows


def test_calibration_select_and_threshold_are_disjoint():
    rows = [{"sample_id": str(i), "intent": "a" if i < 10 else "b", "label": 0} for i in range(20)]
    select, threshold, audit = split_calibration_rows(rows, 42)
    assert {row["sample_id"] for row in select}.isdisjoint({row["sample_id"] for row in threshold})
    assert audit["select_ids_sha256"] != audit["threshold_ids_sha256"]
