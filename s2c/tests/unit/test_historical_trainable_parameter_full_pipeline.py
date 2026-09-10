from __future__ import annotations

from tools.eval.run_historical_trainable_parameter_full_pipeline import (
    FULL_METRICS,
    _candidate_configs,
    _detector_state,
    _summary_rows,
)
import tools.eval.run_historical_trainable_parameter_full_pipeline as parameter_runner


def test_candidate_grid_has_declared_size() -> None:
    assert len(_candidate_configs()) == 252


def test_oos_only_candidate_grid_and_selection_drop_known_guard(monkeypatch) -> None:
    monkeypatch.setattr(parameter_runner, "OOS_ONLY_SELECTION", True)
    assert len(parameter_runner._candidate_configs()) == 960
    rows = []
    for seed in (13, 42, 87):
        for oos_f1, known_f1, accuracy in ((0.90, 0.90, 0.90), (0.95, 0.50, 0.50)):
            rows.append(
                {
                    "dataset": "banking77_oos",
                    "split": "val",
                    "seed": seed,
                    "k": 1,
                    "radius_lambda": 1.0,
                    "threshold": 1.0 if oos_f1 == 0.90 else 0.8,
                    "acceptance_mode": "nearest_sphere",
                    "oos_f1": oos_f1,
                    "known_f1": known_f1,
                    "accuracy": accuracy,
                    "known_recall": known_f1,
                    "false_accept_rate": 1 - oos_f1,
                    "false_reject_rate": 1 - known_f1,
                    "auroc": 0.9,
                    "aupr_oos": 0.9,
                }
            )
    selected, guard = parameter_runner._select(rows, "banking77_oos")
    assert guard == "none_oos_f1_only"
    assert selected["threshold"] == 0.8
    assert selected["selection_guard"] == "none_oos_f1_only"


def test_external_oos_target_prefers_accuracy_after_oos_target(monkeypatch) -> None:
    monkeypatch.setattr(parameter_runner, "OOS_ONLY_SELECTION", True)
    monkeypatch.setattr(parameter_runner, "EXTERNAL_OOS_TARGET_SELECTION", True)
    rows = []
    for seed in (13, 42, 87):
        for threshold, oos_f1, accuracy in ((0.8, 0.84, 0.80), (0.9, 0.82, 0.90)):
            rows.append(
                {
                    "dataset": "banking77_oos",
                    "split": "val",
                    "seed": seed,
                    "k": 1,
                    "radius_lambda": 1.0,
                    "threshold": threshold,
                    "acceptance_mode": "nearest_sphere",
                    "oos_f1": oos_f1,
                    "known_f1": 0.5,
                    "accuracy": accuracy,
                    "known_recall": accuracy,
                    "false_accept_rate": 1 - oos_f1,
                    "false_reject_rate": 1 - accuracy,
                    "auroc": 0.9,
                    "aupr_oos": 0.9,
                }
            )
    selected, guard = parameter_runner._select(rows, "banking77_oos")
    assert guard == "validation_oos_f1_at_external_baseline_max_accuracy_then_known_recall"
    assert selected["threshold"] == 0.9


def test_detector_state_keeps_all_subcenters_for_one_intent() -> None:
    signature = {
        "radius_method": "mean_std",
        "radius_lambda": 1.0,
        "distance_metric": "mahalanobis_diag",
        "acceptance_mode": "normalized_union",
        "spheres": [
            {"cluster_id": 0, "intent": "known", "center": [0.0], "radius": 1.0, "inv_diag_cov": [1.0]},
            {"cluster_id": 1, "intent": "known", "center": [1.0], "radius": 2.0, "inv_diag_cov": [1.0]},
        ],
    }
    state = _detector_state(signature)
    assert state["intent_to_cluster"] == {"known": 0}
    assert state["intent_to_clusters"] == {"known": [0, 1]}
    assert state["cluster_to_intent"] == {"0": "known", "1": "known"}
    assert state["subcenters_per_intent"] == 2


def test_full_pipeline_summary_does_not_merge_same_config_across_datasets() -> None:
    rows = []
    for dataset, oos_f1 in (("clinc150", 0.90), ("stackoverflow", 0.80)):
        row = {
            "dataset": dataset,
            "seed": "42",
            "k": "1",
            "radius_lambda": "1.0",
            "threshold": "1.0",
            "acceptance_mode": "nearest_sphere",
        }
        row.update({metric: str(oos_f1 if metric == "oos_f1" else 0.5) for metric in FULL_METRICS})
        rows.append(row)
    summary = _summary_rows(rows, "all_test_candidates")
    assert {(row["dataset"], row["oos_f1_mean"]) for row in summary} == {
        ("clinc150", 0.90),
        ("stackoverflow", 0.80),
    }
