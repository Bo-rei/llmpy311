import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.experiments.run_historical_trainable_adaptive_centers import (
    assignments, mixed_detector, select, workpoints, summarize, METRICS,
)
from tools.eval.run_historical_trainable_parameter_full_pipeline import (
    _fit_detector, _metrics_from_output, _vectorized_output,
)


@pytest.fixture
def bank():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(60, 5))
    x[:20, 0] += 3
    x[20:40, 1] += 3
    x[40:, 2] += 3
    rows = [{"intent": str(i // 20), "label": 0} for i in range(60)]
    return {k: _fit_detector(x, rows, k, 1., "nearest_sphere") for k in (1, 2, 3)}, x, rows


@pytest.mark.parametrize("mode", ["nearest_sphere", "normalized_union"])
def test_mixed_spheres_equal_native_per_intent_override(bank, mode):
    banks, x, rows = bank
    assignment = {"0": 1, "1": 2, "2": 3}
    mixed, _ = mixed_detector(banks, assignment)
    mixed.acceptance_mode = mode
    native = _fit_detector(x, rows, 1, 1., mode)
    native.subcenters_overrides = assignment
    native.fit(x, np.array([r["intent"] for r in rows]))
    query = np.random.default_rng(12).normal(size=(21, 5))
    for key, values in _vectorized_output(native, query).items():
        np.testing.assert_allclose(_vectorized_output(mixed, query)[key], values, atol=1e-12)


def test_fast_metrics_match_multiclass_reference(bank):
    banks, x, rows = bank
    rows = [dict(r) for r in rows]
    for row in rows[::4]:
        row.update(intent="unseen", label=1)
    output = _vectorized_output(banks[2], x)
    for actual in workpoints(banks[2], output, rows, (.7, 1., 1.3)):
        expected = _metrics_from_output(banks[2], output, rows, actual["threshold"])
        for field, old in (("oos_f1", "f1_u"), ("known_f1", "f1_k"), ("accuracy", "accuracy"),
                           ("known_recall", "known_recall"), ("false_accept_rate", "false_accept_rate")):
            assert actual[field] == pytest.approx(expected[old], abs=1e-12)


def test_selection_ignores_test_and_auxiliary_guards():
    base = dict(split="val", strategy="fixed_k1", center_count=3, radius_lambda=1.,
                threshold=1., acceptance_mode="nearest_sphere", oos_f1=.8, known_f1=.9, accuracy=.9)
    winner = dict(base, strategy="adaptive", oos_f1=.9, known_f1=.1, accuracy=.1)
    assert select([base, winner, dict(base, split="test", oos_f1=1.)]) == winner


def test_train_rank_assignment_counts_and_all_split_aliases(bank):
    banks, _, _ = bank
    strategies, stats, aliases = assignments(banks)
    assert len(strategies) == 27
    assert len(aliases) == 6
    assert len(stats) == 3
    for rank in ("mean_distance", "variance", "sse_reduction"):
        assert sum(k == 2 for k in strategies[f"{rank}_top10_k2"].values()) == 1
        assert aliases[f"{rank}_top100_k3"] == "fixed_k3"


@pytest.mark.parametrize("verified, expected", [(True, True), (False, False), ("True", True), ("False", False)])
def test_summary_preserves_csv_boolean_meaning(verified, expected):
    rows = [dict(dataset="clinc150", selection_scope=scope, seed=seed,
                 delta_oos_f1_pp="-1", direct_pipeline_verified=verified if seed == 42 else True,
                 **{metric: "0.9" for metric in METRICS})
            for scope in ("overall", "fixed", "adaptive") for seed in (13, 42, 87)]
    for summary in summarize(rows):
        assert summary["direct_pipeline_verified"] is expected
        assert summary["oos_f1_mean"] == pytest.approx(.9)


def test_sample_reconciliation_separates_oos_and_downstream_loss(tmp_path, monkeypatch):
    from tools.analysis import build_historical_trainable_adaptive_centers_report as report
    from tools.eval import run_historical_trainable_full_pipeline as evaluator
    from tools.legacy.analysis_v19.run_trainable_minilm_historical_v1 import _row_id

    rows = [dict(text=str(i), intent=intent, domain="d", label=int(intent == "oos"))
            for i, intent in enumerate(("a", "a", "b", "b", "oos", "oos"))]
    samples = [dict(gold_intent=r["intent"], gold_is_oos=r["label"], sample_id=_row_id(r, "test", i),
                    predicted_intent=r["intent"], predicted_is_oos=r["label"]) for i, r in enumerate(rows)]
    predictions = [dict(is_oos=bool(r["label"]), intent=r["intent"], domain="d", gate_score=r["label"])
                   for r in rows]
    predictions[1].update(intent="b", domain="wrong_domain")
    predictions[2].update(intent="a")
    measured, stages = evaluator.compute_metrics(rows, predictions)
    # A current replay need not exactly reproduce an older downstream aggregate.
    saved = dict(measured, f1_all=measured["f1_all"] + .001)
    table = tmp_path / "results/analysis/historical_trainable_full_pipeline/per_seed.csv"
    table.parent.mkdir(parents=True)
    report.write(table, [dict(dataset="clinc150", seed=42, **saved,
                             **{f"stage_count_{k}": v for k, v in stages.items()})])
    source = tmp_path / "clinc150/kir50_seed42/trainable_k1"
    source.mkdir(parents=True)
    (source / "metrics.json").write_text(json.dumps(dict(oos_f1=1., f1_all=1., accuracy=1.)))
    (source / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in samples))
    (source / "detector_signature.json").write_text("{}")
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "DEFAULT_H1_ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "_load_rows", lambda *args: rows)
    monkeypatch.setattr(evaluator, "_detector_state", lambda signature: {})
    monkeypatch.setattr(evaluator, "_make_pipeline", lambda *args: SimpleNamespace(
        predict_batch=lambda texts, batch_size: predictions))
    actual, = report.historical_reconciliation(replay_pipeline=True)
    assert actual["oos_f1_delta"] == 0
    assert actual["oos_decision_mismatch_count"] == 0
    assert actual["macro_f1_delta"] == pytest.approx(-1 / 3)
    assert actual["accuracy_delta"] == pytest.approx(-2 / 6)
    assert actual["downstream_repaired_count"] == 0
    assert actual["downstream_regressed_count"] == 2
    assert actual["router_error_rate"] == actual["expert_error_rate"] == .25
    assert actual["replay_minus_saved_pipeline_macro_f1"] == pytest.approx(-.001)
    assert actual["pipeline_metrics_evidence"] == "fresh_sample_level_cuda_replay"
    rows[0]["text"] = "same-label different sample"
    with pytest.raises(AssertionError):
        report.historical_reconciliation(replay_pipeline=True)
