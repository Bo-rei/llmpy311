"""Unit contracts for the bounded R1 representation method."""

from __future__ import annotations

import numpy as np
import torch

from s2c.experiments.geometry_preserving import (
    effective_rank,
    freeze_module,
    pairwise_cosine_relation_loss,
    pairwise_relation_metrics,
)


def test_teacher_parameters_are_frozen() -> None:
    module = torch.nn.Linear(4, 3)
    freeze_module(module)
    assert not module.training
    assert all(not parameter.requires_grad for parameter in module.parameters())


def test_geometry_loss_is_zero_for_identical_relations() -> None:
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert float(pairwise_cosine_relation_loss(values, values)) < 1e-12


def test_geometry_metrics_report_preservation_and_rank() -> None:
    teacher = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    student = teacher.copy()
    rows = [{"intent": "a"}, {"intent": "b"}, {"intent": "a"}]
    metrics = pairwise_relation_metrics(teacher, student, [row["intent"] for row in rows], seed=42)
    assert metrics["pairwise_distance_correlation"] > 0.99
    assert metrics["knn_neighborhood_preservation"] > 0.99
    assert effective_rank(student) > 1.0
