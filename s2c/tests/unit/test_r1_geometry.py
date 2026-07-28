"""Unit contracts for the bounded R1 representation method."""

from __future__ import annotations

import numpy as np
import torch
import math

from s2c.experiments.geometry_preserving import (
    effective_rank,
    freeze_module,
    pairwise_cosine_relation_loss,
    pairwise_relation_metrics,
    fixed_oos_buckets,
)
from s2c.experiments.r1_contract_repair import _representation_spec


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


def test_geometry_class_distances_are_measured_from_student() -> None:
    teacher = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    # The student deliberately collapses all samples onto one direction.  Its
    # class distances must therefore differ from the teacher distances.
    student = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    labels = ["a", "b", "a", "b"]
    metrics = pairwise_relation_metrics(teacher, student, labels, seed=42)
    assert metrics["intra_class_distance"] < 1e-9
    assert metrics["inter_class_distance"] < 1e-9
    assert metrics["teacher_inter_class_distance"] > metrics["teacher_intra_class_distance"]
    assert math.isnan(metrics["relative_separation"])


def test_geometry_metrics_report_teacher_and_student_separately() -> None:
    teacher = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    student = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = ["a", "b", "a", "b"]
    metrics = pairwise_relation_metrics(teacher, student, labels, seed=42)
    assert "teacher_relative_separation" in metrics
    assert metrics["teacher_relative_separation"] != metrics["relative_separation"]


def test_oos_buckets_do_not_use_test_quantiles_without_validation_oos() -> None:
    train = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    test = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rows = [{"label": 0, "intent": "a"}, {"label": 1, "intent": "unknown"}]
    buckets, info = fixed_oos_buckets(train, test, rows, ["a"])
    assert set(buckets.tolist()) == {"all"}
    assert info["bucket_status"] == "exploratory_unavailable_validation_oos"
    assert info["used_test_oos_for_cutpoints"] is False


def test_contract_representation_specs_keep_classifier_and_geometry_inputs_explicit() -> None:
    pooled = _representation_spec("geometry_ce_recon_pooled_head")
    normalized = _representation_spec("geometry_ce_recon_normalized_head")
    assert pooled["classifier_input"] == "pooled"
    assert normalized["classifier_input"] == "normalized_pooled"
    assert pooled["geometry_enabled"] is True
    assert normalized["geometry_enabled"] is True


def test_oos_buckets_use_validation_quantiles_when_validation_oos_exists() -> None:
    train = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    validation = np.asarray([[0.0, 1.0], [1.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    test = np.asarray([[0.0, 1.0], [1.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    validation_rows = [{"label": 1}, {"label": 1}, {"label": 0}]
    test_rows = [{"label": 1}, {"label": 1}, {"label": 0}]
    buckets, info = fixed_oos_buckets(
        train,
        test,
        test_rows,
        ["a", "a"],
        frozen_validation=validation,
        validation_rows=validation_rows,
    )
    assert info["bucket_status"] == "formal_validation_oos_cutpoints"
    assert info["used_test_oos_for_cutpoints"] is False
    assert info["q20"] < info["q80"]
    assert set(buckets.tolist()) <= {"near", "medium", "far"}
