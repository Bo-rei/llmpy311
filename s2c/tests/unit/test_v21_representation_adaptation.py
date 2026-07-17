"""v21 表征适配实验的轻量协议测试；不下载模型、不启动训练。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.experiments.cluster_separability.v21_representation_adaptation import (
    _preflight_one,
    _supervised_contrastive_loss,
    build_preflight,
)


def test_preflight_reports_missing_inputs_without_training(tmp_path: Path) -> None:
    result = _preflight_one("clinc150", 50, 42, tmp_path / "not-a-model")
    assert result["ready"] is False
    assert any("missing_model_file" in reason for reason in result["reasons"])


def test_preflight_manifest_is_explicit_and_v19_is_frozen(tmp_path: Path) -> None:
    payload = build_preflight(tmp_path, datasets=("clinc150",), kir=50, seed=42,
                              model_path=tmp_path / "missing")
    saved = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert payload["v19_frozen"] is True
    assert saved["protocol"]["selection_split"] == "validation"
    assert saved["checks"][0]["ready"] is False


def test_supcon_loss_is_finite_and_handles_singletons() -> None:
    import torch

    features = torch.tensor(np.eye(4), dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    loss = _supervised_contrastive_loss(features, labels)
    assert torch.isfinite(loss)
    singleton_loss = _supervised_contrastive_loss(features, torch.arange(4))
    assert float(singleton_loss) == 0.0
