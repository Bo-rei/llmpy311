"""Small contract tests for the independent E3 experiment layer."""

from __future__ import annotations

from pathlib import Path

from s2c.experiments.mechanism_runner import (
    DIAGNOSTIC_PARTITION_SEEDS,
    diagnostic_groups,
    diagnostic_partition_seeds,
    partition_control_specs,
)
from s2c.runtime.paths import ProtocolV2Paths


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITION_CONFIG = PROJECT_ROOT / "configs/experiments/protocol_v2_textoir_v1/e3_partition_control.yaml"
DIAGNOSTIC_CONFIG = PROJECT_ROOT / "configs/experiments/protocol_v2_textoir_v1/e3_cluster_diagnostics.yaml"


def test_e3_partition_matrix_has_declared_720_cells() -> None:
    specs = partition_control_specs(PARTITION_CONFIG)
    assert len(specs) == 720
    assert len({spec.run_id for spec in specs}) == 720
    assert {spec.partition for spec in specs} == {"kmeans", "random_balanced"}


def test_e3_diagnostic_groups_and_partition_seeds_are_declared() -> None:
    assert len(diagnostic_groups(DIAGNOSTIC_CONFIG)) == 180
    assert diagnostic_partition_seeds(DIAGNOSTIC_CONFIG) == DIAGNOSTIC_PARTITION_SEEDS


def test_e3_root_is_disjoint_from_frozen_e2_root() -> None:
    paths = ProtocolV2Paths.discover()
    e2_root = paths.run_root / "e2_gate_core_dense"
    e3_root = paths.run_root / "e3_mechanisms"
    assert e2_root != e3_root
    assert e2_root.name == "e2_gate_core_dense"
    assert e3_root.name == "e3_mechanisms"

