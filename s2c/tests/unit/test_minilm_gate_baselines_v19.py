from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.cluster_separability import baselines


def test_energy_score_is_raw_and_not_split_minmax_normalized() -> None:
    classifier = LogisticRegression(max_iter=2000, random_state=42).fit(
        np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]]),
        np.asarray([0, 0, 1, 1, 2, 2]),
    )
    samples = np.asarray([[-0.5], [0.5], [2.5]])

    scores, _ = baselines._linear_oos_scores(classifier, samples, "energy")
    logits = classifier.decision_function(samples)

    assert scores == pytest.approx(-logsumexp(logits, axis=1))
    assert not (float(np.min(scores)) == 0.0 and float(np.max(scores)) == 1.0)


def test_msp_and_entropy_are_high_for_ambiguous_predictions() -> None:
    classifier = LogisticRegression(max_iter=2000, random_state=42).fit(
        np.asarray([[-3.0], [-2.0], [2.0], [3.0]]), np.asarray([0, 0, 1, 1])
    )
    samples = np.asarray([[-2.5], [0.0]])

    msp, _ = baselines._linear_oos_scores(classifier, samples, "msp")
    entropy, _ = baselines._linear_oos_scores(classifier, samples, "entropy")

    assert msp[1] > msp[0]
    assert entropy[1] > entropy[0]


def test_operating_point_enforces_id_guard_before_oos_f1() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.3, 0.8, 0.4, 0.5, 0.6, 0.7])

    selected, candidates = baselines.select_operating_point(labels, scores, 0.75)

    assert candidates
    assert selected["guard_violation"] is False
    assert selected["metrics"]["id_recall"] >= 0.75
    eligible_f1 = [
        row["metrics"]["oos_f1"]
        for row in candidates
        if row["metrics"]["id_recall"] >= 0.75
    ]
    assert selected["metrics"]["oos_f1"] == pytest.approx(max(eligible_f1))


def test_operating_point_reports_guard_violation_when_unreachable() -> None:
    labels = np.asarray([0, 1])
    scores = np.asarray([0.1, 0.9])

    selected, _ = baselines.select_operating_point(labels, scores, 1.01)

    assert selected["guard_violation"] is True
    assert selected["metrics"]["id_recall"] == pytest.approx(1.0)


@pytest.mark.parametrize("method,neighbors", [("knn", 5), ("lof", 10)])
def test_neighbor_scores_have_higher_values_for_far_samples(method: str, neighbors: int) -> None:
    train = np.asarray([[1.0, 0.0], [0.99, 0.1], [0.98, -0.1], [0.97, 0.2], [0.96, -0.2]])
    validation = np.asarray([[1.0, 0.01], [-1.0, 0.0]])

    val_scores, val_nearest = baselines._score_neighbor_method(
        method, neighbors, train, validation
    )
    test_scores, test_nearest = baselines._score_neighbor_method(
        method, neighbors, train, validation
    )

    assert val_scores[1] > val_scores[0]
    assert test_scores == pytest.approx(val_scores)
    assert test_nearest.tolist() == val_nearest.tolist()


def _grid_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset="clinc150",
        kir=50,
        data_seed=42,
        method=None,
        encoder_path=tmp_path / "encoder",
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        batch_size=8,
        dry_run=False,
        resume=True,
        grid=True,
    )


def test_baseline_grid_contains_135_unique_units(tmp_path: Path) -> None:
    units = baselines._grid_arguments(_grid_args(tmp_path))

    keys = {(unit.dataset, unit.kir, unit.data_seed, unit.method) for unit in units}
    assert len(units) == 135
    assert len(keys) == 135
    assert {unit.method for unit in units} == set(baselines.METHODS)
    assert all(unit.grid is False for unit in units)


def test_resume_requires_complete_manifest_matching_inputs_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _grid_args(tmp_path)
    args.method = "msp"
    data_root = baselines._dataset_root(args.data_root, args.dataset, args.kir, args.data_seed)
    gate_root = data_root / "gate"
    gate_root.mkdir(parents=True)
    inputs = {
        data_root / "KNOWN_INTENTS.json": {"known_intents": ["intent_a"]},
        data_root / "MANIFEST.json": {"dataset": "clinc150", "kir": 50, "seed": 42},
        gate_root / "train.json": [{"text": "train", "intent": "intent_a", "label": 0}],
        gate_root / "val.json": [{"text": "val", "intent": "intent_a", "label": 0}],
        gate_root / "test.json": [{"text": "test", "intent": "intent_a", "label": 0}],
    }
    for path, payload in inputs.items():
        path.write_text(json.dumps(payload), encoding="utf-8")

    unit_dir = baselines._unit_dir(
        args.output_root, args.dataset, args.kir, args.data_seed, args.method
    )
    unit_dir.mkdir(parents=True)
    scores = baselines._score_frame(
        inputs[gate_root / "test.json"],
        np.asarray([0.1]),
        threshold=0.5,
        nearest_intents=["intent_a"],
        known_manifest=inputs[data_root / "KNOWN_INTENTS.json"],
    )
    scores.to_parquet(unit_dir / "scores.parquet", index=False)
    (unit_dir / "eval_results.json").write_text(
        json.dumps(
            {
                "protocol": "minilm_gate_baselines_v19",
                "method": "msp",
                "validation": {name: 0.0 for name in baselines.REQUIRED_METRICS},
                "test": {name: 0.0 for name in baselines.REQUIRED_METRICS},
            }
        ),
        encoding="utf-8",
    )
    (unit_dir / "threshold_selection.json").write_text(
        json.dumps(
            {
                "selection_split": "gate/val.json",
                "test_used_for_selection": False,
                "selected_operating_point": {"threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (unit_dir / "timing.json").write_text(json.dumps({"total_seconds": 1.0}), encoding="utf-8")
    model = {"path": "test-encoder"}
    monkeypatch.setattr(baselines, "_model_fingerprint", lambda _: model)
    required = baselines._required_inputs(data_root)
    config = baselines._unit_config(args)
    manifest = {
        "status": "complete",
        "config": config,
        "config_hash": baselines._sha256_payload(config),
        "input_hashes": {
            str(path.relative_to(data_root)): baselines._sha256_file(path) for path in required
        },
        "model": model,
    }
    (unit_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert baselines._completion_error(unit_dir, data_root, required, args) is None

    manifest.pop("status")
    (unit_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert baselines._completion_error(unit_dir, data_root, required, args) == "manifest_not_complete"


def test_missing_audit_reports_incomplete_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _grid_args(tmp_path)
    units = baselines._grid_arguments(args)[:2]
    monkeypatch.setattr(
        baselines,
        "_completion_error",
        lambda unit_dir, data_root, required, unit: None if unit.method == "msp" else "missing_outputs",
    )

    audit_path, audit = baselines._write_missing_audit(units, failures=[])

    assert audit_path == args.output_root / "baseline_missing_cells.json"
    assert audit["expected_unit_count"] == 2
    assert audit["complete_unit_count"] == 1
    assert audit["missing_unit_count"] == 1
    assert audit["missing_units"][0]["reason"] == "missing_outputs"
