from __future__ import annotations

import numpy as np
import pytest

from protocol_v2.gate.view_loader import GateViews
from protocol_v2.gate.clmsg import (
    LocalSupportModel,
    known_order_statistic,
    split_conformal_p_values,
)
from scripts.experiments.run_clmsg import METHOD_KNN, _method_payloads, _validate_split_contract


def _support() -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.20],
            [0.95, -0.20],
            [0.0, 1.0],
            [0.20, 0.98],
            [-0.20, 0.95],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
    return values, labels


def test_local_scale_excludes_self_and_stays_positive() -> None:
    values, labels = _support()
    model = LocalSupportModel(k_neighbors=2, k_scale=1).fit(values, labels)
    assert model.local_scales is not None
    assert np.all(model.local_scales > model.eps)
    # A self-neighbour bug would make every local scale exactly epsilon.
    assert np.all(model.local_scales > 1e-4)


@pytest.mark.parametrize("metric", ["cosine", "normalized_euclidean"])
def test_identical_support_is_near_zero_and_far_query_is_larger(metric: str) -> None:
    values, labels = _support()
    model = LocalSupportModel(metric=metric, k_neighbors=2, k_scale=1).fit(values, labels)
    identical = model.score(values[[0]])
    far = model.score(np.asarray([[-1.0, 0.0]]))
    assert identical.local_scale_score[0] == pytest.approx(0.0, abs=1e-10)
    assert far.local_scale_score[0] > identical.local_scale_score[0]
    assert far.knn_score[0] > identical.knn_score[0]


def test_calibration_is_not_part_of_its_support_model() -> None:
    train, labels = _support()
    calibration = np.asarray([[-1.0, 0.0]], dtype=np.float64)
    model = LocalSupportModel(k_neighbors=2, k_scale=1).fit(train, labels)
    before = model.support.copy()
    score = model.score(calibration)
    assert model.support.shape[0] == train.shape[0]
    assert np.array_equal(model.support, before)
    assert score.local_scale_support_index[0] < train.shape[0]


def test_class_conditional_mode_uses_nearest_support_intent_without_gold_label() -> None:
    train, labels = _support()
    model = LocalSupportModel(k_neighbors=2, k_scale=1).fit(train, labels)
    score = model.score(np.asarray([[0.9, 0.1]], dtype=np.float64))
    nearest_label = model.label_for_indices(score.knn_support_index)[0]
    conditional_label = model.label_for_indices(score.class_local_scale_support_index)[0]
    assert conditional_label == nearest_label == "a"
    hybrid, hybrid_index = score.score_for_mode("hybrid_knn", gamma=0.5)
    assert hybrid[0] == pytest.approx(
        0.5 * score.class_local_scale_score[0] + 0.5 * score.local_scale_score[0]
    )
    assert model.label_for_indices(hybrid_index)[0] == "a"


def test_split_conformal_ties_use_greater_or_equal_and_stay_in_range() -> None:
    calibration = np.asarray([0.1, 0.2, 0.2, 0.4])
    values = split_conformal_p_values(calibration, np.asarray([0.2, 0.5]))
    assert values[0] == pytest.approx(4.0 / 5.0)
    assert values[1] == pytest.approx(1.0 / 5.0)
    assert np.all((values > 0.0) & (values <= 1.0))


def test_rejections_are_monotone_as_alpha_increases() -> None:
    p_values = np.asarray([0.005, 0.02, 0.04, 0.08, 0.2])
    counts = [int(np.sum(p_values < alpha)) for alpha in (0.01, 0.025, 0.05, 0.10)]
    assert counts == sorted(counts)


def test_known_order_statistic_is_deterministic_and_known_only() -> None:
    calibration = np.asarray([0.1, 0.2, 0.4, 0.8])
    first = known_order_statistic(calibration, 0.25)
    second = known_order_statistic(calibration.copy(), 0.25)
    assert first == second == (0.8, 4)


def test_same_seed_free_local_support_is_reproducible() -> None:
    values, labels = _support()
    first = LocalSupportModel(k_neighbors=2, k_scale=1, chunk_size=2).fit(values, labels)
    second = LocalSupportModel(k_neighbors=2, k_scale=1, chunk_size=3).fit(values, labels)
    probe = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)
    a = first.score(probe)
    b = second.score(probe)
    assert np.allclose(a.knn_score, b.knn_score)
    assert np.allclose(a.local_scale_score, b.local_scale_score)
    assert np.array_equal(a.local_scale_support_index, b.local_scale_support_index)


def _row(sample_id: str, label: int = 0, intent: str = "a") -> dict[str, object]:
    return {"sample_id": sample_id, "label": label, "intent": intent}


def test_split_contract_rejects_oos_calibration_and_sample_overlap(tmp_path) -> None:
    with pytest.raises(ValueError, match="Known-only"):
        _validate_split_contract(
            GateViews([_row("train")], [_row("cal", label=1)], [_row("test", label=1)], tmp_path)
        )
    with pytest.raises(ValueError, match="disjoint"):
        _validate_split_contract(
            GateViews([_row("same")], [_row("same")], [_row("test", label=1)], tmp_path)
        )


def test_method_payloads_honours_knn_only_configuration() -> None:
    train, labels = _support()
    model = LocalSupportModel(k_neighbors=2, k_scale=1).fit(train, labels)
    calibration = model.score(np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64))
    test = model.score(
        np.asarray([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    )
    rows = [_row("a", intent="a"), _row("b", intent="b"), _row("o1", 1), _row("o2", 1)]
    metrics, predictions = _method_payloads(
        model,
        calibration,
        test,
        rows,
        [0.05],
        0.05,
        ["global_knn", "class_conditional_knn"],
        [0.5],
        {METHOD_KNN},
    )
    assert set(metrics) == {METHOD_KNN}
    assert len(predictions) == len(rows)
    assert {row["method"] for row in predictions} == {METHOD_KNN}
