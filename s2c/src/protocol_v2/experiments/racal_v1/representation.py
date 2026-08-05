"""Trainable MiniLM representation for RACAL-v1's K=1 control."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModel

from protocol_v2.experiments.geometry_preserving import mean_pool


class ResidualProjection(torch.nn.Module):
    """Two-layer 384D residual adapter with no change to the Gate interface."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fc1 = torch.nn.Linear(dim, hidden_dim)
        self.activation = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_dim, dim)
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return pooled + self.fc2(self.activation(self.fc1(self.norm(pooled))))


class RacalMiniLM(torch.nn.Module):
    """MiniLM wrapper exposing explicit freeze/unfreeze contracts."""

    def __init__(self, model_path: Path, mode: str, projection_hidden_dim: int) -> None:
        super().__init__()
        if mode not in {"trainable_projection_only", "last2_minilm_plus_projection"}:
            raise ValueError(f"Unsupported trainable RACAL mode: {mode}")
        self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        dim = int(self.encoder.config.hidden_size)
        if dim != 384:
            raise ValueError(f"RACAL requires a 384D MiniLM output, got {dim}")
        self.projection = ResidualProjection(dim, int(projection_hidden_dim))
        self.mode = mode
        self._apply_freeze_contract()

    def _apply_freeze_contract(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.projection.parameters():
            parameter.requires_grad_(True)
        if self.mode == "last2_minilm_plus_projection":
            layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
            if layers is None or len(layers) < 2:
                raise RuntimeError("The local MiniLM model does not expose encoder.layer[-2:]")
            for block in layers[-2:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)

    def forward(self, tokens: Mapping[str, torch.Tensor]) -> torch.Tensor:
        pooled = mean_pool(self.encoder(**tokens).last_hidden_state, tokens["attention_mask"])
        return torch.nn.functional.normalize(self.projection(pooled), dim=-1)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))

    def freeze_report(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "hidden_size": int(self.encoder.config.hidden_size),
            "num_hidden_layers": int(self.encoder.config.num_hidden_layers),
            "trainable_parameter_count": self.trainable_parameter_count(),
            "trainable_parameter_names": self.trainable_parameter_names(),
            "requires_grad": {name: bool(parameter.requires_grad) for name, parameter in self.named_parameters()},
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise RuntimeError("RACAL requested CUDA but CUDA is unavailable")
    return torch.device("cpu")


def encode_rows(
    model: RacalMiniLM,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            tokens = tokenizer(
                [str(row["text"]) for row in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            chunks.append(model(tokens).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 384), dtype=np.float32)
