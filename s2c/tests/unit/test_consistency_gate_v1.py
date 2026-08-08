from __future__ import annotations

import numpy as np

from protocol_v2.experiments.consistency_gate_v1 import _conflict_count, _normalise_surface, _select_known_only


def _stats(intents, scores, margins, preds=None):
    intents = np.asarray(intents, dtype=object)
    return {
        "intent": intents,
        "score": np.asarray(scores, dtype=float),
        "margin": np.asarray(margins, dtype=float),
        "pred": np.asarray(preds if preds is not None else [0] * len(intents), dtype=int),
    }


def test_surface_normalization_preserves_content_and_collapses_whitespace():
    assert _normalise_surface("  C#\u00a0  async  ") == "C# async"


def test_conflicts_count_intent_and_oos_disagreement():
    base = _stats(["a", "b"], [0.2, 0.2], [0.4, 0.4])
    view = _stats(["a", "x"], [0.2, 0.2], [0.4, 0.4], [0, 1])
    assert _conflict_count(base, [view]).tolist() == [0, 1]


def test_known_only_selection_respects_recall_drop_budget():
    base = _stats(["a", "a", "b", "b"], [0.2, 0.2, 0.2, 0.2], [0.1, 0.2, 0.3, 0.4])
    view = _stats(["a", "a", "b", "x"], [0.2, 0.2, 0.2, 0.2], [0.1, 0.2, 0.3, 0.4])
    selection = _select_known_only(base, [view], 0.01)
    assert selection["selected_known_recall"] >= selection["target_recall"]
    assert selection["allowed_conflicts"] in {0, 1}
