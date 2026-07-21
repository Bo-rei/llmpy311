"""代表 Cascade 编排新增的样本级错误归因回归测试。"""

from tools.eval.eval_system_pipeline_v19 import (
    _cascade_error_decomposition,
    _cascade_error_stage,
)
from src.pipeline.system_pipeline import HiLSAMoEV19Pipeline


def test_cascade_error_stage_distinguishes_gate_router_and_expert() -> None:
    known = {"label": 0, "domain": "banking", "intent": "balance"}
    oos = {"label": 1, "domain": "oos", "intent": "oos"}

    assert _cascade_error_stage(known, {"is_oos": True}) == "known_rejected_by_gate"
    assert _cascade_error_stage(oos, {"is_oos": False}) == "oos_accepted_by_gate"
    assert _cascade_error_stage(
        known,
        {"is_oos": False, "domain": "wrong", "intent": "balance"},
    ) == "known_wrong_domain"
    assert _cascade_error_stage(
        known,
        {"is_oos": False, "domain": "banking", "intent": "wrong"},
    ) == "known_wrong_expert"
    assert _cascade_error_stage(
        known,
        {"is_oos": False, "domain": "banking", "intent": "balance"},
    ) == "correct_known_prediction"
    assert _cascade_error_stage(oos, {"is_oos": True}) == "correct_oos_rejection"


def test_cascade_error_decomposition_returns_counts_and_rates() -> None:
    rows = [
        {"error_stage": "correct_oos_rejection"},
        {"error_stage": "oos_accepted_by_gate"},
        {"error_stage": "known_rejected_by_gate"},
        {"error_stage": "known_wrong_domain"},
        {"error_stage": "known_wrong_expert"},
        {"error_stage": "correct_known_prediction"},
    ]

    result = _cascade_error_decomposition(rows)

    assert all(value == 1 for value in result["counts"].values())
    assert all(value == 1 / 6 for value in result["rates"].values())


def test_single_domain_router_returns_constant_predictions_without_model() -> None:
    pipeline = HiLSAMoEV19Pipeline.__new__(HiLSAMoEV19Pipeline)
    pipeline.router_model = None
    pipeline.router_num_classes = 1

    result = pipeline._router_predict(["a", "b", "c"])

    assert result == {"domain_ids": [0, 0, 0], "domain_probs": [1.0, 1.0, 1.0]}
