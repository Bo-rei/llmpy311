from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from protocol_v2.experiments.racal_v1.representation import LoraRacalMiniLM


MODEL = Path(__file__).resolve().parents[3] / "assets" / "models" / "all-MiniLM-L6-v2"


@pytest.mark.skipif(not MODEL.is_dir(), reason="local MiniLM model is unavailable")
def test_lora_freeze_contract_and_parameter_count() -> None:
    model = LoraRacalMiniLM(MODEL, 256)
    report = model.freeze_report()

    assert report["mode"] == "lora_minilm_plus_projection"
    assert report["lora_config"] == {
        "target_modules": "all-linear",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION",
    }
    assert report["base_parameter_count"] == 22_713_216
    assert report["base_trainable_parameter_count"] == 0
    assert report["lora_parameter_count"] == 337_920
    assert report["projection_parameter_count"] == 198_016
    assert report["trainable_parameter_count"] == 535_936
    assert all("lora_" in name or name.startswith("projection.") for name in report["trainable_parameter_names"])
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name and not name.startswith("projection."))


@pytest.mark.skipif(not MODEL.is_dir(), reason="local MiniLM model is unavailable")
def test_lora_checkpoint_state_roundtrip() -> None:
    model = LoraRacalMiniLM(MODEL, 64)
    clone = LoraRacalMiniLM(MODEL, 64)
    clone.load_state_dict(model.state_dict())
    assert clone.freeze_report()["trainable_parameter_count"] == model.freeze_report()["trainable_parameter_count"]


@pytest.mark.skipif(not MODEL.is_dir(), reason="local MiniLM model is unavailable")
def test_lora_forward_preserves_gate_shape() -> None:
    model = LoraRacalMiniLM(MODEL, 64)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokens = tokenizer(["check the account balance"], return_tensors="pt")
    with torch.no_grad():
        output = model(tokens)
    assert output.shape == (1, 384)
    assert torch.allclose(output.norm(dim=-1), torch.ones(1), atol=1e-5)
