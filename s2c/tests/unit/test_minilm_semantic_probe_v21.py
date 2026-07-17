"""v21 MiniLM 语义探针的协议回归测试。"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.experiments.cluster_separability.v21_semantic_probe import (
    _distance_geometry,
    _margin_rows,
    _purity_rows,
)


def _rows(intents, labels=None):
    labels = labels or [0] * len(intents)
    return [{"intent": intent, "label": label, "text": intent, "source_split": "test"} for intent, label in zip(intents, labels)]


def test_train_purity_excludes_self_match():
    rows = _rows(["a", "a", "b", "b"])
    x = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    by_intent, summary, confusion = _purity_rows(rows, x, rows, x, split="train", dataset="toy", seed=42)
    purity10 = next(item for item in summary if item["k"] == 10)
    assert purity10["macro_purity"] == 0.5
    assert not any(item["true_intent"] == item["neighbor_intent"] for item in confusion)
    assert len(by_intent) == 6


def test_geometry_reports_positive_inter_intent_separation():
    rows = _rows(["a"] * 4 + ["b"] * 4)
    x = np.asarray([[1.0, 0.0], [0.99, 0.01], [1.0, -0.01], [0.98, 0.02],
                    [0.0, 1.0], [0.01, 0.99], [-0.01, 1.0], [0.02, 0.98]])
    summary, per_intent = _distance_geometry(rows, x, dataset="toy", seed=1)
    assert summary["inter_distance"] > summary["intra_distance"]
    assert summary["relative_separation"] > 0
    assert len(per_intent) == 2


def test_true_intent_margin_is_positive_for_separated_known_samples():
    train_rows = _rows(["a"] * 2 + ["b"] * 2)
    train_x = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    query_rows = _rows(["a", "b"])
    query_x = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    summary, detail = _margin_rows(query_rows, query_x, train_rows, train_x, split="test", dataset="toy", seed=1)
    assert summary["nearest_center_accuracy"] == 1.0
    assert all(item["true_intent_margin"] > 0 for item in detail)
