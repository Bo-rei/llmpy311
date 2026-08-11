from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from protocol_v2.experiments.mogb_loss_contract import (
    loss_from_class_distances,
    nearest_class_distances,
    nearest_subcentroid_loss,
    subcentroid_signal,
)
from scripts.experiments.run_mogb_exact_reproduction import configure_subcentroid_loss


class _DummyClusterLoss:
    num_labels = 2


class _DummyManager:
    clusterLoss = _DummyClusterLoss()


def test_official_l1_matches_upstream_formula() -> None:
    distances = torch.tensor([[1.0, 3.0], [4.0, 2.0]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    expected = -torch.log(
        torch.softmax(-F.normalize(distances, p=1, dim=1), dim=1)[torch.arange(2), labels]
    ).mean()
    actual = loss_from_class_distances(distances, labels, mode="official_l1")
    assert torch.allclose(actual, expected)


def test_raw_temperature_preserves_stronger_distance_signal() -> None:
    distances = torch.tensor([[1.0, 4.0], [5.0, 2.0]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    official = loss_from_class_distances(distances, labels, mode="official_l1")
    corrected = loss_from_class_distances(
        distances,
        labels,
        mode="raw_temperature",
        temperature=1.0,
    )
    assert corrected < official


def test_nearest_subcentroid_loss_updates_features() -> None:
    features = torch.tensor([[0.0, 0.0], [4.0, 4.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    centroids = torch.tensor([[0.5, 0.0], [4.5, 4.0]])
    centroid_labels = torch.tensor([0, 1])
    loss = nearest_subcentroid_loss(
        features,
        labels,
        centroids,
        centroid_labels,
        num_labels=2,
        mode="raw_temperature",
        temperature=1.0,
    )
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert float(features.grad.norm()) > 0.0


def test_missing_centroid_class_fails_instead_of_producing_nan() -> None:
    with pytest.raises(ValueError, match="missing nearest-subcentroid classes"):
        nearest_class_distances(
            torch.zeros((2, 3)),
            torch.zeros((1, 3)),
            torch.tensor([0]),
            num_labels=2,
        )


def test_signal_reports_larger_raw_probability_and_logit_span() -> None:
    features = torch.tensor([[0.0, 0.0], [4.0, 4.0]])
    labels = torch.tensor([0, 1])
    centroids = torch.tensor([[0.5, 0.0], [4.5, 4.0]])
    centroid_labels = torch.tensor([0, 1])
    official = subcentroid_signal(
        features,
        labels,
        centroids,
        centroid_labels,
        num_labels=2,
        mode="official_l1",
    )
    corrected = subcentroid_signal(
        features,
        labels,
        centroids,
        centroid_labels,
        num_labels=2,
        mode="raw_temperature",
        temperature=1.0,
    )
    assert corrected.mean_true_probability > official.mean_true_probability
    assert corrected.mean_logit_span > official.mean_logit_span
    assert corrected.distance_gradient_norm > 0.0


def test_raw_temperature_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        loss_from_class_distances(
            torch.ones((2, 2)),
            torch.tensor([0, 1]),
            mode="raw_temperature",
            temperature=0.0,
        )


def test_runner_defaults_to_the_official_loss_contract() -> None:
    manager = _DummyManager()
    contract = configure_subcentroid_loss(manager, {})
    labels = torch.tensor([0, 1])
    centroids = torch.tensor([[0.0, 0.0], [4.0, 4.0]])
    features = torch.tensor([[1.0, 0.0], [4.0, 2.0]])
    actual = manager.clusterLoss.compute_classification_loss(
        features,
        labels,
        centroids,
        torch.tensor([0, 1]),
    )
    distances = nearest_class_distances(
        features,
        centroids,
        torch.tensor([0, 1]),
        num_labels=2,
    )
    expected = loss_from_class_distances(distances, labels, mode="official_l1")
    assert contract["mode"] == "official_l1"
    assert contract["official_formula_preserved"] is True
    assert torch.allclose(actual, expected)


def test_runner_injects_raw_distance_without_editing_upstream_class() -> None:
    manager = _DummyManager()
    contract = configure_subcentroid_loss(
        manager,
        {"subcentroid_loss_mode": "raw_temperature", "subcentroid_temperature": 2.0},
    )
    labels = torch.tensor([0, 1])
    centroids = torch.tensor([[0.0, 0.0], [4.0, 4.0]])
    features = torch.tensor([[1.0, 0.0], [4.0, 2.0]])
    actual = manager.clusterLoss.compute_classification_loss(
        features,
        labels,
        centroids,
        torch.tensor([0, 1]),
    )
    distances = nearest_class_distances(
        features,
        centroids,
        torch.tensor([0, 1]),
        num_labels=2,
    )
    expected = loss_from_class_distances(
        distances,
        labels,
        mode="raw_temperature",
        temperature=2.0,
    )
    assert contract["diagnostic_only"] is True
    assert torch.allclose(actual, expected)
