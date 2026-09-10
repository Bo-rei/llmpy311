from pathlib import Path

from tools.analysis.build_comparison_atlas_v1 import build


def test_comparison_atlas_preserves_contract_layers(tmp_path: Path) -> None:
    manifest = build(
        tmp_path / "tables",
        tmp_path / "figures",
        tmp_path / "COMPARISON_ATLAS.md",
    )
    assert manifest["fair_rows"] == 63
    assert manifest["fair_per_seed_rows"] == 315
    assert "current_protocol_v2_fair_gate" in manifest["direct_ranking_layers"]
    assert "historical_fulltex_cascade" in manifest["non_comparable_layers"]
    assert (tmp_path / "tables" / "contract_aware_comparison_matrix.csv").is_file()
    assert (tmp_path / "figures" / "fixed_k2_risk_attribution.png").is_file()
    assert (tmp_path / "COMPARISON_ATLAS.md").read_text(encoding="utf-8").startswith("# 当前实验对比图谱")
