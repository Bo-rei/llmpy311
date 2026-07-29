from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2)
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, **_):
        hidden = torch.stack((input_ids.float(), input_ids.float() * 2), dim=-1)
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=(hidden,))


def test_shared_encoder_preserves_head_and_base_state_dict_names(monkeypatch):
    import legacy.models.transformer as transformer
    from legacy.models.expert import SmolLMExpert
    from legacy.router import SmolLMRouter

    monkeypatch.setattr(
        transformer.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeBase(),
    )
    monkeypatch.setattr(transformer, "get_peft_model", lambda model, _cfg: model)

    router = SmolLMRouter("unused", num_classes=3)
    expert = SmolLMExpert("unused", projection_dim=2)
    input_ids = torch.tensor([[1, 3, 9]])
    attention_mask = torch.tensor([[1, 1, 0]])

    assert router(input_ids, attention_mask).shape == (1, 3)
    assert expert(input_ids, attention_mask).shape == (1, 2)
    assert "base.weight" in router.state_dict()
    assert "classifier.weight" in router.state_dict()
    assert "base.weight" in expert.state_dict()
    assert "projection.net.0.weight" in expert.state_dict()
