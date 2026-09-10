import pytest

from tools.analysis.build_historical_paper_ablation_report import (
    canonical_variant,
    normalize_paper_ablation,
    PAPER_REFERENCE_COMPARABLE,
    PAPER_REFERENCE_DATASET,
)


def _source_rows():
    variants = {
        "clinc150": ("full_anchor", "wo_gate_confidence", "cascade_minilm", "cascade_smollm"),
        "stackoverflow": ("full_anchor", "wo_gate_confidence", "cascade_minilm", "cascade_smollm"),
        "banking77_oos": (
            "full_anchor",
            "banking_wo_geometric_gate_expert_confidence",
            "cascade_minilm",
            "cascade_smollm",
        ),
    }
    rows = []
    for dataset, names in variants.items():
        for kir in ("25", "50", "75"):
            for variant in names:
                rows.append(
                    {
                        "slug": dataset,
                        "kir_tag": f"kir{kir}_seed42",
                        "variant": variant,
                        "status": "existing",
                        "overall_accuracy": "0.8",
                        "macro_f1": "0.7",
                        "known_macro_f1": "0.75",
                        "oos_f1": "0.9",
                        "delta_vs_anchor_overall_accuracy": "0.0",
                        "delta_vs_anchor_oos_f1": "0.0",
                    }
                )
    return rows


def test_variant_aliases_are_explicit():
    assert canonical_variant("full_anchor") == "ours"
    assert canonical_variant("banking_wo_geometric_gate_expert_confidence") == "without_gate"
    with pytest.raises(ValueError):
        canonical_variant("unknown_variant")


def test_paper_ablation_normalization_covers_36_cells_without_private_paths():
    rows = normalize_paper_ablation(_source_rows(), use_actual_eval=False)
    assert len(rows) == 36
    assert len({(row["dataset"], row["kir"], row["variant"]) for row in rows}) == 36
    assert all("path" not in key for row in rows for key in row)
    assert all(row["evidence"] == "historical_paper_summary_metadata" for row in rows)


def test_paper_ablation_normalization_rejects_missing_cell():
    rows = _source_rows()[:-1]
    with pytest.raises(ValueError, match="coverage mismatch"):
        normalize_paper_ablation(rows, use_actual_eval=False)


def test_banking_reference_uses_historical_oos_key_and_not_archive_key():
    assert PAPER_REFERENCE_DATASET["banking77_oos"] == "banking77_oos"
    assert PAPER_REFERENCE_COMPARABLE["banking77_oos"] is True
