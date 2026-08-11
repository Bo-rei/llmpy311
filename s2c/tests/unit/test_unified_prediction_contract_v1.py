from __future__ import annotations

from tools.analysis.unified_prediction_contract_v1 import (
    ALL_FIELDS,
    BASE_FIELDS,
    GroupAudit,
    build_alignment,
    normalise_dataset,
    normalise_oos_label,
    validate_row,
    validate_rows,
)


def make_row(sample_id: str, *, predicted_oos: int = 0, score: float | None = 0.2) -> dict:
    row = {field: None for field in ALL_FIELDS}
    row.update(
        {
            "protocol_version": "protocol_v2_textoir_v1",
            "dataset": "stackoverflow",
            "kir": 0.5,
            "seed": 42,
            "method": "S2C-Trainable-K1",
            "backbone": "MiniLM-trainable",
            "supervision_type": "Known-only",
            "split": "test",
            "sample_id": sample_id,
            "true_label": "svn",
            "is_true_oos": 0,
            "predicted_label": "__oos__" if predicted_oos else "svn",
            "predicted_oos": predicted_oos,
            "accepted_known": int(not predicted_oos),
            "oos_score": score,
            "run_id": "unit-run",
            "registry_sha256": "a" * 64,
            "canonical_manifest_sha256": "b" * 64,
        }
    )
    return row


def make_group(method: str, rows: list[dict], *, contract_layer: str = "same_protocol_fair") -> dict:
    audit = GroupAudit(
        dataset="stackoverflow",
        kir=0.5,
        seed=42,
        method=method,
        run_id=f"{method}-run",
        backbone="MiniLM-trainable",
        supervision_type="Known-only",
        source_path="unit.jsonl",
        contract_layer=contract_layer,
        score_available=True,
        selection_audit="unit-test",
    )
    for row in rows:
        audit.add(row)
    return audit.as_dict()


def test_schema_and_aliases_are_stable() -> None:
    assert BASE_FIELDS[:4] == ["protocol_version", "dataset", "kir", "seed"]
    assert "sample_id" in BASE_FIELDS
    assert "nearest_distance" in ALL_FIELDS
    assert "ball_purity" in ALL_FIELDS
    assert normalise_dataset("banking") == "banking77"
    assert normalise_dataset("oos") == "clinc150"
    assert normalise_oos_label("oos") == "__oos__"


def test_row_validation_checks_acceptance_relation_and_allows_external_missing_score() -> None:
    assert validate_row(make_row("a")) == []
    assert validate_row(make_row("b", predicted_oos=1, score=None)) == []
    invalid = make_row("c")
    invalid["accepted_known"] = 0
    assert "accepted_known_inconsistent" in validate_row(invalid)


def test_small_group_validation_rejects_duplicate_ids() -> None:
    rows = [make_row("a"), make_row("a")]
    errors = validate_rows(rows)
    assert any("duplicate_sample_id" in error for error in errors)


def test_alignment_uses_order_and_label_digests() -> None:
    reference = [make_row("a"), make_row("b")]
    aligned = build_alignment(
        [
            make_group("S2C-Trainable-K1", reference),
            make_group("MOGB-MiniLM", [make_row("a"), make_row("b")]),
        ]
    )
    assert {row["status"] for row in aligned} == {"aligned"}

    mismatch = build_alignment(
        [
            make_group("S2C-Trainable-K1", reference),
            make_group("ADB-external-BERT", [make_row("b"), make_row("a")], contract_layer="external_backbone"),
        ]
    )
    adb = next(row for row in mismatch if row["method"] == "ADB-external-BERT")
    assert adb["status"] == "external_contract_unaligned"
    assert adb["sample_id_order_match"] is False
