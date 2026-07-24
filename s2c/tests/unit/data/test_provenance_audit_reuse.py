"""Regression tests for bounded three-way provenance refreshes."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.audit.audit_dataset_provenance import require_reusable_historical_audit


def test_reuse_requires_both_historical_evidence_files(tmp_path: Path) -> None:
    """A source-only refresh must not silently discard legacy-input provenance."""

    with pytest.raises(FileNotFoundError, match="historical_input_inventory"):
        require_reusable_historical_audit(tmp_path)

    (tmp_path / "historical_input_inventory.csv").write_text("dataset\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="historical_comparison_metrics"):
        require_reusable_historical_audit(tmp_path)

    (tmp_path / "historical_comparison_metrics.csv").write_text("dataset\n", encoding="utf-8")
    require_reusable_historical_audit(tmp_path)
