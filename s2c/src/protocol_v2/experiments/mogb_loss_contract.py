"""Explicit nearest-subcentroid loss contracts for MOGB diagnostics.

The pinned upstream implementation first L1-normalizes each sample's vector of
nearest per-class distances and then applies ``softmax(-distance)``.  This
module preserves that formula as ``official_l1`` and exposes one isolated
diagnostic alternative, ``raw_temperature``.  The alternative is not presented
as official MOGB; it exists only to test whether the upstream normalization
suppresses the representation-learning signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


SubcentroidLossMode = Literal["official_l1", "raw_temperature"]


@dataclass(frozen=True)
class SubcentroidSignal:
    """Train-only diagnostics for one nearest-subcentroid loss evaluation."""

    loss: float
    mean_true_probability: float
    mean_logit_span: float
    distance_gradient_norm: float


def nearest_class_distances(
    features: torch.Tensor,
    centroids: torch.Tensor,
    centroid_labels: torch.Tensor,
    *,
    num_labels: int,
) -> torch.Tensor:
    """Return each sample's distance to the nearest centroid of every class.

    MOGB training expects at least one generated centroid for every registered
    Known class.  Missing classes are rejected explicitly instead of allowing
    an ``inf`` value to turn the downstream normalization into NaN.
    """

    if features.ndim != 2 or centroids.ndim != 2:
        raise ValueError("features and centroids must be two-dimensional")
    if features.shape[1] != centroids.shape[1]:
        raise ValueError("feature and centroid dimensions do not match")
    if centroid_labels.ndim != 1 or centroid_labels.shape[0] != centroids.shape[0]:
        raise ValueError("centroid_labels must align with centroids")
    if num_labels <= 1:
        raise ValueError("nearest-subcentroid training requires at least two classes")

    centroid_labels = centroid_labels.to(device=features.device, dtype=torch.long)
    missing = [label for label in range(num_labels) if not bool((centroid_labels == label).any())]
    if missing:
        raise ValueError(f"missing nearest-subcentroid classes: {missing}")

    point_to_centroid = torch.cdist(features, centroids.to(features.device), p=2)
    per_class = []
    for label in range(num_labels):
        per_class.append(point_to_centroid[:, centroid_labels == label].min(dim=1).values)
    return torch.stack(per_class, dim=1)


def loss_from_class_distances(
    distances: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: SubcentroidLossMode,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute an explicit MOGB subcentroid classification loss."""

    if distances.ndim != 2 or labels.ndim != 1 or distances.shape[0] != labels.shape[0]:
        raise ValueError("distance table and labels must be aligned")
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("nearest-subcentroid distance table contains NaN or infinity")
    labels = labels.to(device=distances.device, dtype=torch.long)
    if mode == "official_l1":
        logits = -F.normalize(distances, p=1, dim=1)
    elif mode == "raw_temperature":
        if temperature <= 0:
            raise ValueError("subcentroid temperature must be positive")
        logits = -distances / float(temperature)
    else:
        raise ValueError(f"unknown subcentroid loss mode: {mode}")
    return F.cross_entropy(logits, labels)


def nearest_subcentroid_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    centroid_labels: torch.Tensor,
    *,
    num_labels: int,
    mode: SubcentroidLossMode,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the selected loss while preserving gradients to features."""

    distances = nearest_class_distances(
        features,
        centroids,
        centroid_labels,
        num_labels=num_labels,
    )
    return loss_from_class_distances(
        distances,
        labels,
        mode=mode,
        temperature=temperature,
    )


def subcentroid_signal(
    features: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    centroid_labels: torch.Tensor,
    *,
    num_labels: int,
    mode: SubcentroidLossMode,
    temperature: float = 1.0,
) -> SubcentroidSignal:
    """Measure probability and distance-gradient signal without model updates."""

    with torch.no_grad():
        observed = nearest_class_distances(
            features.detach(),
            centroids.detach(),
            centroid_labels.detach(),
            num_labels=num_labels,
        )
    distances = observed.detach().clone().requires_grad_(True)
    labels = labels.to(device=distances.device, dtype=torch.long)
    if mode == "official_l1":
        logits = -F.normalize(distances, p=1, dim=1)
    elif mode == "raw_temperature":
        if temperature <= 0:
            raise ValueError("subcentroid temperature must be positive")
        logits = -distances / float(temperature)
    else:
        raise ValueError(f"unknown subcentroid loss mode: {mode}")
    loss = F.cross_entropy(logits, labels)
    gradient = torch.autograd.grad(loss, distances)[0]
    probabilities = torch.softmax(logits.detach(), dim=1)
    row = torch.arange(labels.shape[0], device=labels.device)
    return SubcentroidSignal(
        loss=float(loss.detach().item()),
        mean_true_probability=float(probabilities[row, labels].mean().item()),
        mean_logit_span=float((logits.detach().max(dim=1).values - logits.detach().min(dim=1).values).mean().item()),
        distance_gradient_norm=float(gradient.norm().item()),
    )
