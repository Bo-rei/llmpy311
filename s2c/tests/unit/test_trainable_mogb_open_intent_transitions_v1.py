from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "tools/analysis/build_trainable_mogb_open_intent_transitions_v1.py"
SPEC = importlib.util.spec_from_file_location("open_intent_transitions", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_five_open_intent_outcomes_are_mutually_exclusive() -> None:
    result = MODULE.classify_outcomes(
        pd.Series(["a", "a", "a", "x", "x"]),
        pd.Series([0, 0, 0, 1, 1]),
        pd.Series(["a", "b", "oos", "a", "oos"]),
        pd.Series([0, 0, 1, 0, 1]),
    )
    assert result.tolist() == [
        "known_correct",
        "known_wrong_intent",
        "known_rejected",
        "oos_false_accept",
        "oos_correct",
    ]


def test_transition_table_contains_all_25_cells() -> None:
    base = pd.DataFrame(
        {
            "sample_id": ["1", "2"],
            "gold_intent": ["a", "x"],
            "gold_is_oos": [0, 1],
            "outcome": ["known_correct", "oos_correct"],
        }
    )
    other = base.copy()
    other["outcome"] = ["known_rejected", "oos_false_accept"]
    result = MODULE.transition_table(base, other, "fixture", 0.5, 42)
    assert len(result) == 25
    assert result["count"].sum() == 2
    assert result["rate_all"].sum() == 1.0


def test_score_threshold_contract_is_strictly_greater_than_one_for_oos() -> None:
    scores = pd.Series([0.9, 1.0, 1.0000001])
    assert scores.gt(1.0).astype(int).tolist() == [0, 0, 1]
