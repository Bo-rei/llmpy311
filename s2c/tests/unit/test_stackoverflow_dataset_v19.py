from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.active.rebuild_multi_dataset_v19 import (  # noqa: E402
    STACKOVERFLOW_INTENTS,
    build_stackoverflow_bundle,
    sync_stackoverflow_source,
    _select_stackoverflow_known_intents,
)
from tools.eval.eval_system_pipeline_v19 import _evaluate  # noqa: E402
from src.runtime import load_profile  # noqa: E402


def _write_stackoverflow_split(root: Path, split_name: str, rows: list[dict[str, str]]) -> None:
    path = root / f"{split_name}.csv"
    fieldnames = ["Title", "Body", "Tag"]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_stackoverflow_row(intent: str, split: str, idx: int) -> dict[str, str]:
    return {
        "Title": f"{intent} question {split} {idx}",
        "Body": f"{intent} body {split} {idx}",
        "Tag": intent,
    }


def _build_stackoverflow_source_fixture(root: Path) -> None:
    train_rows = [_make_stackoverflow_row(intent, "train", 0) for intent in STACKOVERFLOW_INTENTS]
    valid_rows = [_make_stackoverflow_row(intent, "valid", 0) for intent in STACKOVERFLOW_INTENTS]
    test_rows = [_make_stackoverflow_row(intent, "test", 0) for intent in STACKOVERFLOW_INTENTS]
    _write_stackoverflow_split(root, "train", train_rows)
    _write_stackoverflow_split(root, "valid", valid_rows)
    _write_stackoverflow_split(root, "test", test_rows)


def test_stackoverflow_config_declares_single_domain_repo_variant():
    profile = load_profile(PROJECT_ROOT / "configs" / "profiles.yaml", "stackoverflow")

    assert profile.policy["num_domains"] == 1
    assert profile.policy["multi_domain"] is False
    assert profile.policy["single_domain"] is True
    assert profile.policy["num_intents"] == 20
    assert profile.policy["oos_strategy"] == "held_out_intents_only"
    assert profile.stackoverflow_known_selection_strategy == "seeded_random"


def test_stackoverflow_orchestrator_reports_single_domain_summary():
    profile = load_profile(PROJECT_ROOT / "configs" / "profiles.yaml", "stackoverflow")

    assert profile.policy["num_domains"] == 1
    assert profile.policy["num_intents"] == 20
    assert profile.policy["single_domain"] is True
    assert profile.policy["oos_strategy"] == "held_out_intents_only"


def test_stackoverflow_kir_known_intents_support_legacy_nested_protocol():
    known_25 = _select_stackoverflow_known_intents(
        STACKOVERFLOW_INTENTS,
        kir=0.25,
        seed=42,
        strategy="nested_prefix",
    )
    known_50 = _select_stackoverflow_known_intents(
        STACKOVERFLOW_INTENTS,
        kir=0.5,
        seed=42,
        strategy="nested_prefix",
    )
    known_75 = _select_stackoverflow_known_intents(
        STACKOVERFLOW_INTENTS,
        kir=0.75,
        seed=42,
        strategy="nested_prefix",
    )

    assert len(known_25) == 5
    assert len(known_50) == 10
    assert len(known_75) == 15
    assert set(known_25).issubset(set(known_50))
    assert set(known_50).issubset(set(known_75))


def test_stackoverflow_kir_known_intents_support_seeded_random_protocol():
    known_75 = _select_stackoverflow_known_intents(
        STACKOVERFLOW_INTENTS,
        kir=0.75,
        seed=42,
        strategy="seeded_random",
    )
    legacy_75 = _select_stackoverflow_known_intents(
        STACKOVERFLOW_INTENTS,
        kir=0.75,
        seed=42,
        strategy="nested_prefix",
    )
    repeat_75 = _select_stackoverflow_known_intents(
        STACKOVERFLOW_INTENTS,
        kir=0.75,
        seed=42,
        strategy="seeded_random",
    )

    assert len(known_75) == 15
    assert known_75 == repeat_75
    assert known_75 != legacy_75
    assert set(known_75).issubset(set(STACKOVERFLOW_INTENTS))


def test_sync_stackoverflow_source_builds_single_domain_manifest_from_official_splits(tmp_path: Path):
    source_root = tmp_path / "stackoverflow_origin"
    source_root.mkdir()
    _build_stackoverflow_source_fixture(source_root)

    manifest = sync_stackoverflow_source(source_root, force=True)

    assert manifest["source_splits"] == {"train": 20, "valid": 20, "test": 20}
    assert manifest["intent_universe"] == STACKOVERFLOW_INTENTS
    assert manifest["text_policy"] == "title_only"
    assert manifest["intent_policy"] == "single_label_tag"
    assert manifest["domain_policy"] == "single_domain"

    records_path = Path(manifest["records_path"])
    rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 60
    assert {row["domain"] for row in rows} == {"stackoverflow"}
    assert {row["source_split"] for row in rows} == {"train", "valid", "test"}


def test_sync_stackoverflow_source_rebuilds_stale_legacy_cache(tmp_path: Path):
    source_root = tmp_path / "stackoverflow_origin"
    source_root.mkdir()
    _build_stackoverflow_source_fixture(source_root)
    (source_root / "records.jsonl").write_text('{"legacy": true}\n', encoding="utf-8")
    (source_root / "SOURCE_MANIFEST.json").write_text(
        json.dumps({"dataset_repo": "imoore/60k-stack-overflow-questions-with-quality-rate"}),
        encoding="utf-8",
    )

    manifest = sync_stackoverflow_source(source_root, force=False)

    assert manifest["domain_policy"] == "single_domain"
    rows = [json.loads(line) for line in (source_root / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0].get("legacy") is None


def test_sync_stackoverflow_source_deduplicates_titles_across_splits(tmp_path: Path):
    source_root = tmp_path / "stackoverflow_origin"
    source_root.mkdir()
    _write_stackoverflow_split(
        source_root,
        "train",
        [
            {"Title": "same title", "Body": "train body", "Tag": STACKOVERFLOW_INTENTS[0]},
            {"Title": "train unique", "Body": "train body 2", "Tag": STACKOVERFLOW_INTENTS[1]},
        ],
    )
    _write_stackoverflow_split(
        source_root,
        "valid",
        [{"Title": "valid unique", "Body": "valid body", "Tag": STACKOVERFLOW_INTENTS[2]}],
    )
    _write_stackoverflow_split(
        source_root,
        "test",
        [{"Title": "same title", "Body": "test body", "Tag": STACKOVERFLOW_INTENTS[3]}],
    )

    manifest = sync_stackoverflow_source(source_root, force=True)

    assert manifest["filters"]["deduplicated_titles"] == 1
    rows = [json.loads(line) for line in (source_root / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    same_title_rows = [row for row in rows if row["title"] == "same title"]
    assert len(same_title_rows) == 1
    assert same_title_rows[0]["source_split"] == "train"


def test_build_stackoverflow_bundle_preserves_official_split_and_marks_heldout_oos(tmp_path: Path):
    source_root = tmp_path / "stackoverflow_origin"
    source_root.mkdir()
    _build_stackoverflow_source_fixture(source_root)

    bundle = build_stackoverflow_bundle(
        stackoverflow_root=source_root,
        kir=0.5,
        seed=42,
        output_root=tmp_path / "rebuilt",
        known_selection_strategy="seeded_random",
    )

    assert bundle.manifest["domains"] == ["stackoverflow"]
    assert bundle.manifest["domain_map"] == {"stackoverflow": 0}
    assert bundle.manifest["known_intents_manifest"]["selection_protocol"]["method"] == (
        "official_split_single_domain_seeded_random_kir"
    )
    assert len(bundle.known_intents) == 10
    assert len(bundle.unknown_intents) == 10
    assert len(bundle.gate["train"]) == 10
    assert len(bundle.gate["val"]) == 20
    assert len(bundle.gate["test"]) == 20
    assert {row["domain"] for row in bundle.router["train"]} == {"stackoverflow"}
    assert {row["label"] for row in bundle.router["train"]} == {0}
    assert set(bundle.experts.keys()) == {"stackoverflow"}
    assert {row["intent"] for row in bundle.gate["train"]}.issubset(set(bundle.known_intents))
    heldout_val = [row for row in bundle.gate["val"] if row["label"] == 1]
    heldout_test = [row for row in bundle.gate["test"] if row["label"] == 1]
    assert {row["source_split"] for row in heldout_val} == {"heldout_oos_valid"}
    assert {row["source_split"] for row in heldout_test} == {"heldout_oos_test"}


def test_evaluate_reports_stackoverflow_heldout_oos_source_bucket():
    records = [
        {
            "text": "how to group by in sql",
            "intent": "sql",
            "domain": "stackoverflow",
            "label": 0,
            "source_split": "train",
        },
        {
            "text": "why is react state stale",
            "intent": "reactjs",
            "domain": "unknown",
            "label": 1,
            "source_split": "heldout_oos_test",
        },
    ]
    preds = [
        {
            "gate_pred": 0,
            "is_oos": False,
            "intent": "sql",
            "domain": "stackoverflow",
            "domain_prob": 0.99,
            "intent_prob": 0.98,
            "gate_score": 0.1,
            "gate_distance": 0.1,
            "gate_radius": 1.0,
        },
        {
            "gate_pred": 1,
            "is_oos": True,
            "intent": "__oos__",
            "domain": "unknown",
            "domain_prob": 0.0,
            "intent_prob": 0.0,
            "gate_score": 0.9,
            "gate_distance": 0.9,
            "gate_radius": 1.0,
        },
    ]

    metrics = _evaluate(records, preds)

    assert "heldout_oos" in metrics["oos_by_source"]
    assert metrics["oos_by_source"]["heldout_oos"]["count"] == 1
    assert metrics["oos_by_source"]["heldout_oos"]["gate_oos_rejection"] == 1.0
