from __future__ import annotations

from pathlib import Path

import pytest
import torch

from protocol_v2.experiments.racal_v1.representation import RacalMiniLM, ResidualProjection


MODEL = Path(__file__).resolve().parents[3] / "assets" / "models" / "all-MiniLM-L6-v2"


def test_projection_residual_preserves_dimension() -> None:
    module = ResidualProjection(384, 64)
    output = module(torch.randn(3, 384))
    assert output.shape == (3, 384)


@pytest.mark.skipif(not MODEL.is_dir(), reason="local MiniLM model is unavailable")
def test_last2_freeze_contract() -> None:
    model = RacalMiniLM(MODEL, "last2_minilm_plus_projection", 64)
    report = model.freeze_report()
    assert report["hidden_size"] == 384
    assert report["num_hidden_layers"] == 6
    assert all(name.startswith("projection.") or ".encoder.layer.4." in name or ".encoder.layer.5." in name for name in report["trainable_parameter_names"])
    assert not any(parameter.requires_grad for parameter in model.encoder.embeddings.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.encoder.layer[0].parameters())
    assert all(parameter.requires_grad for parameter in model.encoder.encoder.layer[-1].parameters())


@pytest.mark.skipif(not MODEL.is_dir(), reason="local MiniLM model is unavailable")
def test_projection_only_keeps_backbone_frozen() -> None:
    model = RacalMiniLM(MODEL, "trainable_projection_only", 64)
    assert model.trainable_parameter_count() > 0
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
