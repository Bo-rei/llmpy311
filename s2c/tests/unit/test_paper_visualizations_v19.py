from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis import export_paper_visualizations_v19 as viz


def test_sample_kind_splits_known_heldout_and_native_oos():
    assert viz.sample_kind({"label": 0, "is_oos": True, "true_intent": "weather"}) == "known"
    assert viz.sample_kind({"label": 1, "is_oos": False, "true_intent": "oos"}) == "native_oos"
    assert viz.sample_kind({"label": 1, "is_oos": False, "true_intent": "calendar"}) == "heldout_unknown"


def test_gate_score_rows_use_real_prediction_fields(tmp_path: Path):
    predictions = [
        {"label": 0, "is_oos": False, "true_intent": "weather", "gate_score": 0.8},
        {"label": 1, "is_oos": True, "true_intent": "oos", "gate_distance": 9.0, "gate_radius": 3.0},
    ]
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(predictions), encoding="utf-8")

    rows = viz.load_gate_score_rows(path)

    assert rows == [
        {"kind": "known", "score": 0.8},
        {"kind": "native_oos", "score": 3.0},
    ]


def test_error_breakdown_reads_pipeline_metrics(tmp_path: Path):
    payload = {
        "metrics": {
            "cascade_error_breakdown": {
                "known": {
                    "total": 10,
                    "gate_false_reject": 2,
                    "router_error_given_gate_pass": 3,
                    "expert_error_given_router_correct": 4,
                },
                "oos": {"total": 20, "gate_false_accept": 5},
            }
        }
    }
    path = tmp_path / "eval_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    row = viz.error_breakdown_row("CLINC150", path)

    assert row == {
        "dataset": "CLINC150",
        "gate_false_reject": 2,
        "router_wrong_dispatch": 3,
        "expert_wrong_classification": 4,
        "oos_false_accept": 5,
    }


def test_detector_centers_are_loaded_from_spheres(tmp_path: Path):
    detector = {
        "spheres": [
            {"center": [1.0, 2.0], "radius": 0.5, "intent_name": "a"},
            {"center": [3.0, 4.0], "radius": 0.7, "intent_name": "b"},
        ]
    }
    path = tmp_path / "detector.json"
    path.write_text(json.dumps(detector), encoding="utf-8")

    centers = viz.load_detector_centers(path)

    assert centers.centers.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert centers.radii.tolist() == [0.5, 0.7]
    assert centers.labels == ["a", "b"]


def test_select_known_intents_is_seeded_and_supported():
    predictions = [
        {"label": 0, "is_oos": False, "true_intent": "a"},
        {"label": 0, "is_oos": False, "true_intent": "a"},
        {"label": 0, "is_oos": False, "true_intent": "b"},
        {"label": 0, "is_oos": False, "true_intent": "b"},
        {"label": 0, "is_oos": False, "true_intent": "c"},
        {"label": 0, "is_oos": False, "true_intent": "d"},
    ]

    selected = viz.select_known_intents(predictions, count=2, seed=7, min_support=2)

    assert len(selected) == 2
    assert set(selected) <= {"a", "b"}


def test_clean_clinc_filter_uses_gold_kind_instead_of_gate_prediction():
    predictions = [
        {"label": 0, "is_oos": True, "true_intent": "weather"},
        {"label": 1, "is_oos": False, "true_intent": "weather"},
        {"label": 1, "is_oos": False, "true_intent": "oos"},
    ]

    selected = viz._filter_clean_clinc_predictions(
        predictions,
        selected_intents=["weather"],
        max_oos_per_kind=10,
        seed=42,
    )

    assert selected[0] is predictions[0]
    assert predictions[1] in selected
    assert predictions[2] in selected
    assert sum(viz.sample_kind(row) == "known" for row in selected) == 1


def test_space_metrics_separate_known_heldout_and_native_oos():
    centers = viz.DetectorCenters(
        centers=viz.np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=float),
        radii=viz.np.asarray([1.0, 2.0], dtype=float),
        labels=["a", "b"],
    )
    embeddings = viz.np.asarray([[0.5, 0.0], [2.0, 0.0], [20.0, 20.0]], dtype=float)
    kinds = ["known", "heldout_unknown", "native_oos"]

    rows = viz.compute_space_metrics(embeddings, kinds, centers)

    by_kind = {row["kind"]: row for row in rows}
    assert by_kind["known"]["count"] == 1
    assert by_kind["known"]["mean_nearest_center_distance"] == 0.5
    assert by_kind["heldout_unknown"]["mean_distance_radius_ratio"] == 2.0
    assert by_kind["native_oos"]["mean_nearest_center_distance"] > 10.0
