"""Contrastive intent projector for v19 gate representation learning.

Provides a lightweight projection head trained with hard-negative contrastive
loss to improve known-intent separability in embedding space.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveIntentProjector(nn.Module):
    """Projection head for contrastive intent embedding.

    Wraps a frozen backbone and trains only the linear projection layers.
    Embeddings are L2-normalized after projection for cosine similarity use.

    Args:
        backbone: Pre-trained model with hidden_states output.
        input_dim: Hidden size of the backbone output.
        proj_dim: Projected embedding dimension.
        hidden_dim: Intermediate projection dimension. Defaults to proj_dim * 2.
        max_length: Maximum token sequence length.
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_dim: int,
        proj_dim: int = 128,
        hidden_dim: Optional[int] = None,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.max_length = max_length
        _hidden_dim = hidden_dim if hidden_dim is not None else proj_dim * 2

        self.projector = nn.Sequential(
            nn.Linear(input_dim, _hidden_dim, bias=True),
            nn.GELU(),
            nn.LayerNorm(_hidden_dim),
            nn.Linear(_hidden_dim, proj_dim, bias=False),
        )

    @torch.no_grad()
    def _encode_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract mean-pooled backbone hidden states (no grad).

        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).

        Returns:
            Pooled representation tensor of shape (B, input_dim).
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs.hidden_states[-1]

        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return pooled

    def forward_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L2-normalized projected embeddings.

        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).

        Returns:
            L2-normalized projected embeddings of shape (B, proj_dim).
        """
        pooled = self._encode_backbone(input_ids, attention_mask)
        projected = self.projector(pooled.float())
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for forward_embeddings for nn.Module interface compatibility.

        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).

        Returns:
            L2-normalized projected embeddings of shape (B, proj_dim).
        """
        return self.forward_embeddings(input_ids, attention_mask)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss (SupCon).

    For each anchor, all same-label samples are positives, all different-label
    samples are negatives.

    Args:
        embeddings: L2-normalized embeddings of shape (B, D).
        labels: Integer class labels of shape (B,).
        temperature: Softmax temperature. Lower values sharpen distributions.

    Returns:
        Scalar contrastive loss.
    """
    batch_size = embeddings.size(0)
    if batch_size < 2:
        return embeddings.new_zeros(())

    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    labels_row = labels.unsqueeze(1)
    labels_col = labels.unsqueeze(0)
    positive_mask = (labels_row == labels_col).float()
    self_mask = torch.eye(batch_size, device=embeddings.device)
    positive_mask = positive_mask - self_mask

    log_prob = similarity_matrix - torch.logsumexp(
        similarity_matrix - 1e9 * self_mask, dim=1, keepdim=True
    )

    positive_count = positive_mask.sum(dim=1).clamp(min=1.0)
    loss = -(positive_mask * log_prob).sum(dim=1) / positive_count

    return loss.mean()


def hard_negative_contrastive_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Triplet-style contrastive loss with hard negatives.

    Args:
        anchor_embeddings: Anchor L2-normalized embeddings (B, D).
        positive_embeddings: Positive L2-normalized embeddings (B, D).
        negative_embeddings: Hard-negative L2-normalized embeddings (B, D).
        temperature: Softmax temperature.

    Returns:
        Scalar contrastive loss.
    """
    pos_sim = (anchor_embeddings * positive_embeddings).sum(dim=1) / temperature
    neg_sim = (anchor_embeddings * negative_embeddings).sum(dim=1) / temperature
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)
