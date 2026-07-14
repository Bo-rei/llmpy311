from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.train.train_semantic_verifier_v19 import _binary_yes_no_metrics


def test_binary_yes_no_metrics_use_configured_token_ids():
    preds = torch.tensor([7, 9, 7, 9], dtype=torch.long)
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    metrics = _binary_yes_no_metrics(
        preds=preds,
        labels=labels,
        yes_token_id=7,
        no_token_id=9,
    )

    assert metrics["correct"] == 4
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
