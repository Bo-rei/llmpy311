from pathlib import Path

from tools.analysis.build_external_single_cell_comparison_v1 import build


def test_external_comparison_audits_predictions_and_contracts(tmp_path: Path) -> None:
    payload = build(tmp_path / "tables", tmp_path / "figures")
    assert payload["fair_rows"] == 35
    assert payload["external_rows"] >= 8
    assert payload["valid_rows"] >= 30
    assert payload["invalid_rows"] >= 1
    assert (tmp_path / "tables" / "stackoverflow_kir50_external_and_fair_cells.csv").is_file()
    assert (tmp_path / "tables" / "external_invalid_semantic_runs.csv").is_file()
    assert (tmp_path / "figures" / "stackoverflow_external_single_cell_comparison.png").is_file()
