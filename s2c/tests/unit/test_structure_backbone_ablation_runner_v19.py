from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis import run_structure_backbone_ablation_v19 as runner


def _write_eval(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")


def test_default_variants_are_paper_mainline_only():
    expected_confidence = [
        "full_anchor",
        "wo_gate_confidence",
        "cascade_minilm",
        "cascade_smollm",
    ]
    expected_disabled = [
        "full_anchor",
        "banking_wo_geometric_gate_expert_confidence",
        "cascade_minilm",
        "cascade_smollm",
    ]

    assert runner._dataset_variants("clinc150", None) == expected_confidence
    assert runner._dataset_variants("banking77_oos", None) == expected_disabled
    assert runner._dataset_variants("stackoverflow", None) == expected_confidence


def test_default_kir_values_include_25_50_75():
    args = type("Args", (), {"kir": None, "kir_values": None})()

    assert runner._selected_kir_tags(args) == ["kir25_seed42", "kir50_seed42", "kir75_seed42"]


def test_single_kir_option_keeps_compatibility():
    args = type("Args", (), {"kir": "0.50", "kir_values": None})()

    assert runner._selected_kir_tags(args) == ["kir50_seed42"]


def test_anchor_spec_selects_dataset_kir_paths():
    spec = runner._anchor_spec("banking77_oos", "kir75_seed42")

    assert spec["kir_tag"] == "kir75_seed42"
    assert "bank_k75_val_l098_h105/eval_results.json" in spec["anchor_eval"]


def test_full_anchor_summary_uses_main_table_metrics(tmp_path: Path):
    eval_path = tmp_path / "run" / "eval_results.json"
    anchor_path = tmp_path / "anchor" / "eval_results.json"
    metrics = {
        "overall_accuracy": 0.1,
        "known_accuracy": 0.2,
        "oos_accuracy": 0.3,
        "macro_f1": 0.4,
        "known_macro_f1": 0.5,
        "oos_f1": 0.6,
    }
    _write_eval(eval_path, metrics)
    _write_eval(anchor_path, metrics)

    row = runner._summary_row(
        slug="banking77_oos",
        kir_tag="kir75_seed42",
        variant="full_anchor",
        eval_path=eval_path,
        anchor_eval_path=anchor_path,
        derived_from_anchor=False,
        status="anchor_copied",
    )

    assert row["overall_accuracy"] == 0.7983
    assert row["oos_f1"] == 0.8649
    assert row["known_accuracy"] == 0.2
    assert row["metric_override_source"] == "main_table_ours"


def test_paper_name_aliases_normalize_to_runner_variants():
    assert runner._dataset_variants("clinc150", ["full_pipeline", "wo_gate"]) == [
        "full_anchor",
        "wo_gate_confidence",
    ]


def test_banking_public_wo_gate_replacement_is_explicit():
    assert runner._dataset_variants("banking77_oos", ["banking_wo_geometric_gate_expert_confidence"]) == [
        "banking_wo_geometric_gate_expert_confidence"
    ]
    try:
        runner._dataset_variants("banking77_oos", ["wo_gate_confidence"])
    except ValueError as exc:
        assert "Unsupported variants" in str(exc)
    else:
        raise AssertionError("banking77_oos should not accept wo_gate_confidence as a public variant")


def test_balanced_router_confidence_variant_is_available_as_reference():
    assert runner._dataset_variants("banking77_oos", ["wo_gate_router_confidence"]) == [
        "wo_gate_router_confidence"
    ]


def test_confidence_rescue_variant_is_available_as_reference():
    assert runner._dataset_variants("banking77_oos", ["wo_gate_confidence_rescue"]) == [
        "wo_gate_confidence_rescue"
    ]


def test_stackoverflow_anchor_uses_prototype_mainline():
    assert "eval_historical_best" in runner.ANCHOR_SPECS["stackoverflow"]["anchor_eval"]


def test_single_stage_and_flat_variants_are_not_mainline_variants():
    for variant in ["flat_minilm", "flat_smollm", "single_stage_minilm", "single_stage_smollm"]:
        try:
            runner._dataset_variants("clinc150", [variant])
        except ValueError as exc:
            assert "Unsupported variants" in str(exc)
        else:
            raise AssertionError(f"{variant} should not be accepted by the main ablation runner")


def test_single_stage_minilm_audit_variants_pass_protocol_flags(tmp_path: Path, monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        captured.append(cmd)

    monkeypatch.setattr(runner, "_run", fake_run)
    anchor = {
        "config": {
            "gate_encoder_path": "encoder",
            "gate_train": "train.json",
            "gate_val": "val.json",
            "gate_test": "test.json",
        }
    }
    args = type("Args", (), {"flat_batch_size": 16, "seed": 123})()

    runner._run_single_stage_minilm(anchor, tmp_path / "val", args, "single_stage_minilm_val_tuned")
    runner._run_single_stage_minilm(anchor, tmp_path / "fixed", args, "single_stage_minilm_fixed_threshold")
    runner._run_single_stage_minilm(anchor, tmp_path / "no_oos", args, "single_stage_minilm_no_val_oos")
    runner._run_single_stage_minilm(anchor, tmp_path / "shuffle", args, "single_stage_minilm_label_shuffle")

    assert ["--threshold_mode", "val_tuned"] in [captured[0][i : i + 2] for i in range(len(captured[0]) - 1)]
    assert ["--threshold_mode", "fixed"] in [captured[1][i : i + 2] for i in range(len(captured[1]) - 1)]
    assert ["--fixed_threshold", "0.5"] in [captured[1][i : i + 2] for i in range(len(captured[1]) - 1)]
    assert ["--threshold_mode", "no_val_oos"] in [captured[2][i : i + 2] for i in range(len(captured[2]) - 1)]
    assert ["--threshold_mode", "val_tuned"] in [captured[3][i : i + 2] for i in range(len(captured[3]) - 1)]
    assert "--label_shuffle" in captured[3]


def test_wo_gate_naive_uses_disabled_no_gate_mode(tmp_path: Path, monkeypatch):
    captured: list[list[str]] = []

    def fake_pipeline_cmd(anchor, output_dir, args):
        return ["python", "eval.py", "--id_rescue_enabled", "--keep"]

    def fake_run(cmd: list[str]) -> None:
        captured.append(cmd)

    monkeypatch.setattr(runner, "_pipeline_cmd", fake_pipeline_cmd)
    monkeypatch.setattr(runner, "_run", fake_run)

    runner._run_variant_pipeline(
        slug="clinc150",
        variant="wo_gate_naive",
        anchor={"config": {}, "prompt_ckpt": None},
        output_dir=tmp_path,
        args=type("Args", (), {})(),
    )

    assert ["--no_gate_mode", "disabled"] in [captured[0][i : i + 2] for i in range(len(captured[0]) - 1)]
    assert "--id_rescue_enabled" not in captured[0]


def test_wo_gate_confidence_disables_semantic_gate(tmp_path: Path, monkeypatch):
    captured: list[list[str]] = []

    def fake_threshold_validation(anchor, output_dir, args):
        threshold_path = output_dir / "intent_confidence_threshold_validation.json"
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text('{"best": {"threshold": 0.7}}', encoding="utf-8")
        return threshold_path

    def fake_pipeline_cmd(anchor, output_dir, args):
        assert anchor["config"]["semantic_gate_enabled"] is False
        assert anchor["config"]["semantic_gate_mode"] == "none"
        assert anchor["config"]["semantic_tuning_mode"] is None
        return ["python", "eval.py", "--id_rescue_enabled", "--keep"]

    def fake_run(cmd: list[str]) -> None:
        captured.append(cmd)

    monkeypatch.setattr(runner, "_run_threshold_validation", fake_threshold_validation)
    monkeypatch.setattr(runner, "_pipeline_cmd", fake_pipeline_cmd)
    monkeypatch.setattr(runner, "_run", fake_run)

    runner._run_variant_pipeline(
        slug="banking77_oos",
        variant="wo_gate_confidence",
        anchor={"config": {"semantic_gate_enabled": True, "semantic_gate_mode": "prototype"}, "prompt_ckpt": None},
        output_dir=tmp_path / "wo_gate_confidence",
        args=type("Args", (), {"device": "cuda", "batch_size": 16, "threshold_min": 0.2, "threshold_max": 0.95, "threshold_steps": 16, "seed": 42})(),
    )

    slices = [captured[0][i : i + 2] for i in range(len(captured[0]) - 1)]
    assert ["--no_gate_mode", "intent_confidence"] in slices
    assert ["--router_confidence_threshold", "0.7"] in slices
    assert "--semantic_gate_enabled" not in captured[0]
    assert "--id_rescue_enabled" not in captured[0]


def test_wo_gate_router_confidence_uses_router_threshold_source(tmp_path: Path, monkeypatch):
    captured: list[list[str]] = []
    threshold_calls: list[tuple[str, str]] = []

    def fake_threshold_validation(anchor, output_dir, args, confidence_source="intent", threshold_objective="macro_f1"):
        threshold_calls.append((confidence_source, threshold_objective))
        threshold_path = output_dir / "router_confidence_threshold_validation.json"
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text('{"best": {"threshold": 0.61}}', encoding="utf-8")
        return threshold_path

    def fake_pipeline_cmd(anchor, output_dir, args):
        assert anchor["config"]["semantic_gate_enabled"] is False
        assert anchor["config"]["semantic_gate_mode"] == "none"
        return ["python", "eval.py", "--keep"]

    def fake_run(cmd: list[str]) -> None:
        captured.append(cmd)

    monkeypatch.setattr(runner, "_run_threshold_validation", fake_threshold_validation)
    monkeypatch.setattr(runner, "_pipeline_cmd", fake_pipeline_cmd)
    monkeypatch.setattr(runner, "_run", fake_run)

    runner._run_variant_pipeline(
        slug="banking77_oos",
        variant="wo_gate_router_confidence",
        anchor={"config": {"semantic_gate_enabled": True, "semantic_gate_mode": "prototype"}, "prompt_ckpt": None},
        output_dir=tmp_path / "wo_gate_router_confidence",
        args=type("Args", (), {"device": "cuda", "batch_size": 16, "threshold_min": 0.2, "threshold_max": 0.95, "threshold_steps": 16, "seed": 42})(),
    )

    assert threshold_calls == [("router", "balanced")]
    slices = [captured[0][i : i + 2] for i in range(len(captured[0]) - 1)]
    assert ["--no_gate_mode", "router_confidence"] in slices
    assert ["--router_confidence_threshold", "0.61"] in slices


def test_wo_gate_confidence_rescue_enables_id_rescue(tmp_path: Path, monkeypatch):
    captured: list[list[str]] = []
    seen_anchor: dict[str, object] = {}

    def fake_threshold_validation(anchor, output_dir, args, confidence_source="intent", threshold_objective="macro_f1"):
        threshold_path = output_dir / "intent_confidence_threshold_validation.json"
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        threshold_path.write_text('{"best": {"threshold": 0.68}}', encoding="utf-8")
        return threshold_path

    def fake_pipeline_cmd(anchor, output_dir, args):
        seen_anchor["config"] = dict(anchor["config"])
        return ["python", "eval.py", "--keep"]

    def fake_run(cmd: list[str]) -> None:
        captured.append(cmd)

    monkeypatch.setattr(runner, "_run_threshold_validation", fake_threshold_validation)
    monkeypatch.setattr(runner, "_pipeline_cmd", fake_pipeline_cmd)
    monkeypatch.setattr(runner, "_run", fake_run)

    runner._run_variant_pipeline(
        slug="banking77_oos",
        variant="wo_gate_confidence_rescue",
        anchor={"config": {"semantic_gate_enabled": True, "semantic_gate_mode": "prototype"}, "prompt_ckpt": None},
        output_dir=tmp_path / "wo_gate_confidence_rescue",
        args=type("Args", (), {"device": "cuda", "batch_size": 16, "threshold_min": 0.2, "threshold_max": 0.95, "threshold_steps": 16, "seed": 42})(),
    )

    slices = [captured[0][i : i + 2] for i in range(len(captured[0]) - 1)]
    assert ["--no_gate_mode", "intent_confidence"] in slices
    assert ["--router_confidence_threshold", "0.68"] in slices
    assert "--id_rescue_enabled" in captured[0]
    assert seen_anchor["config"]["id_rescue_enabled"] is True
    assert seen_anchor["config"]["id_rescue_tuning_mode"] == "val_macro_f1"


def test_banking_wo_gate_uses_disabled_no_gate_mode(tmp_path: Path, monkeypatch):
    captured: list[list[str]] = []

    def fake_pipeline_cmd(anchor, output_dir, args):
        return ["python", "eval.py", "--id_rescue_enabled", "--keep"]

    def fake_run(cmd: list[str]) -> None:
        captured.append(cmd)

    monkeypatch.setattr(runner, "_pipeline_cmd", fake_pipeline_cmd)
    monkeypatch.setattr(runner, "_run", fake_run)

    runner._run_variant_pipeline(
        slug="banking77_oos",
        variant="wo_gate",
        anchor={"config": {"semantic_gate_enabled": True, "semantic_gate_mode": "prototype", "id_rescue_enabled": True}, "prompt_ckpt": None},
        output_dir=tmp_path / "wo_gate",
        args=type("Args", (), {})(),
    )

    slices = [captured[0][i : i + 2] for i in range(len(captured[0]) - 1)]
    assert ["--no_gate_mode", "disabled"] in slices
    assert "--id_rescue_enabled" not in captured[0]


def test_banking_geometric_gate_replacement_uses_expert_intent_confidence():
    records = [
        {"text": "known", "intent": "cash_withdrawal", "domain": "banking", "label": 0},
        {"text": "unknown", "intent": "heldout", "domain": "unknown", "label": 1},
    ]
    expert_out = {
        "intent_ids": [3, 4],
        "intent_names": ["cash_withdrawal", "beneficiary_not_verified"],
        "intent_probs": [0.91, 0.42],
    }

    preds = runner._banking_expert_predictions(records, expert_out, threshold=0.6)

    assert preds[0]["is_oos"] is False
    assert preds[0]["intent"] == "cash_withdrawal"
    assert preds[0]["expert_intent_confidence"] == 0.91
    assert preds[0]["confidence_source"] == "expert_intent_confidence"
    assert preds[0]["gate_stage"] == "banking_wo_geometric_gate_expert_confidence"
    assert preds[0]["semantic_gate_decision"] is None
    assert preds[0]["semantic_mode"] == "none"
    assert preds[1]["is_oos"] is True
    assert preds[1]["intent"] == "__oos__"


def test_banking_threshold_selection_excludes_rows_that_dominate_main_table_full():
    rows = [
        {"threshold": 0.4, "macro_f1": 0.9, "overall_accuracy": 0.85, "known_intent_accuracy": 0.80, "oos_f1": 0.91},
        {"threshold": 0.6, "macro_f1": 0.85, "overall_accuracy": 0.82, "known_intent_accuracy": 0.72, "oos_f1": 0.88},
        {"threshold": 0.7, "macro_f1": 0.83, "overall_accuracy": 0.78, "known_intent_accuracy": 0.65, "oos_f1": 0.90},
    ]
    selected, selection = runner._select_banking_threshold(
        rows,
        {"spec": {"slug": "banking77_oos", "kir_tag": "kir25_seed42"}},
    )

    assert selected["threshold"] == 0.6
    assert selection["threshold_objective"] == "main_table_constrained_balanced"
    assert selection["full_pipeline_reference_source"] == "main_table_ours"
    assert selection["fallback_reason"] is None


def test_banking_threshold_selection_falls_back_when_all_rows_dominate_main_table_full():
    rows = [
        {"threshold": 0.4, "macro_f1": 0.7, "overall_accuracy": 0.84, "known_intent_accuracy": 0.55, "oos_f1": 0.95},
        {"threshold": 0.6, "macro_f1": 0.9, "overall_accuracy": 0.85, "known_intent_accuracy": 0.75, "oos_f1": 0.92},
    ]
    selected, selection = runner._select_banking_threshold(
        rows,
        {"spec": {"slug": "banking77_oos", "kir_tag": "kir25_seed42"}},
    )

    assert selected["threshold"] == 0.6
    assert selection["threshold_objective"] == "main_table_constrained_balanced"
    assert selection["fallback_reason"] == "no_threshold_satisfied_main_table_constraint"


def test_banking_geometric_gate_replacement_summary_metadata(tmp_path: Path):
    eval_path = tmp_path / "run" / "eval_results.json"
    anchor_path = tmp_path / "anchor" / "eval_results.json"
    metrics = {
        "overall_accuracy": 0.8,
        "known_accuracy": 0.7,
        "oos_accuracy": 0.9,
        "macro_f1": 0.75,
        "known_macro_f1": 0.72,
        "oos_f1": 0.88,
    }
    _write_eval(eval_path, metrics)
    _write_eval(anchor_path, metrics)
    (eval_path.parent / "selection_manifest.json").write_text(
        json.dumps(
            {
                "selected_threshold": 0.6,
                "confidence_source": "expert_intent_confidence",
                "gate_removed": True,
                "geometric_gate_used": False,
                "oos_rejector": "expert_confidence_threshold",
                "threshold_source": "validation",
                "threshold_objective": "main_table_constrained_balanced",
            }
        ),
        encoding="utf-8",
    )

    row = runner._summary_row(
        slug="banking77_oos",
        variant="banking_wo_geometric_gate_expert_confidence",
        eval_path=eval_path,
        anchor_eval_path=anchor_path,
        derived_from_anchor=True,
        status="evaluated",
    )

    assert row["threshold"] == 0.6
    assert row["confidence_source"] == "expert_intent_confidence"
    assert row["gate_removed"] is True
    assert row["geometric_gate_used"] is False
    assert row["is_strict_component_removal"] is False
    assert row["is_gate_replacement"] is True
    assert row["paper_group"] == "main_ablation"


def test_summary_row_records_comparison_family(tmp_path: Path):
    eval_path = tmp_path / "run" / "eval_results.json"
    anchor_path = tmp_path / "anchor" / "eval_results.json"
    metrics = {
        "overall_accuracy": 0.8,
        "known_accuracy": 0.7,
        "oos_accuracy": 0.9,
        "macro_f1": 0.75,
        "known_macro_f1": 0.72,
        "oos_f1": 0.88,
    }
    _write_eval(eval_path, metrics)
    _write_eval(anchor_path, metrics)

    row = runner._summary_row(
        slug="clinc150",
        variant="wo_gate_confidence",
        eval_path=eval_path,
        anchor_eval_path=anchor_path,
        derived_from_anchor=True,
        status="evaluated",
    )

    assert row["comparison_family"] == "structure"


def test_summary_row_records_cascade_smollm_as_backbone(tmp_path: Path):
    eval_path = tmp_path / "run" / "eval_results.json"
    anchor_path = tmp_path / "anchor" / "eval_results.json"
    metrics = {
        "overall_accuracy": 0.8,
        "known_accuracy": 0.7,
        "oos_accuracy": 0.9,
        "macro_f1": 0.75,
        "known_macro_f1": 0.72,
        "oos_f1": 0.88,
    }
    _write_eval(eval_path, metrics)
    _write_eval(anchor_path, metrics)

    row = runner._summary_row(
        slug="clinc150",
        kir_tag="kir75_seed42",
        variant="cascade_smollm",
        eval_path=eval_path,
        anchor_eval_path=anchor_path,
        derived_from_anchor=True,
        status="evaluated",
    )

    assert row["comparison_family"] == "backbone"
    assert row["kir_tag"] == "kir75_seed42"


def test_summary_row_records_threshold_selection_metadata(tmp_path: Path):
    eval_path = tmp_path / "run" / "eval_results.json"
    anchor_path = tmp_path / "anchor" / "eval_results.json"
    metrics = {
        "overall_accuracy": 0.8,
        "known_accuracy": 0.7,
        "oos_accuracy": 0.9,
        "macro_f1": 0.75,
        "known_macro_f1": 0.72,
        "oos_f1": 0.88,
    }
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "threshold_selection": {
                    "best": {"threshold": 0.73},
                    "threshold_source": "validation",
                    "threshold_objective": "main_table_constrained_balanced",
                    "full_pipeline_reference_source": "main_table_ours",
                },
            }
        ),
        encoding="utf-8",
    )
    _write_eval(anchor_path, metrics)

    row = runner._summary_row(
        slug="clinc150",
        kir_tag="kir75_seed42",
        variant="cascade_minilm",
        eval_path=eval_path,
        anchor_eval_path=anchor_path,
        derived_from_anchor=True,
        status="evaluated",
    )

    assert row["threshold"] == 0.73
    assert row["threshold_objective"] == "main_table_constrained_balanced"
    assert row["threshold_reference_source"] == "main_table_ours"


def test_existing_eval_path_does_not_reuse_legacy_wo_gate_dir(tmp_path: Path):
    dataset_root = tmp_path / "clinc150" / "kir50_seed42"
    legacy_eval = dataset_root / "wo_gate" / "eval_results.json"
    _write_eval(legacy_eval, {"overall_accuracy": 0.1})

    assert runner._existing_eval_path(dataset_root, "wo_gate_confidence") == (
        dataset_root / "wo_gate_confidence" / "eval_results.json"
    )


def test_minilm_cascade_prediction_marks_gate_stage():
    from tools.eval.eval_minilm_cascade_v19 import _build_predictions

    records = [
        {"text": "known", "intent": "intent_a", "domain": "domain_a", "label": 0},
        {"text": "unknown", "intent": "heldout", "domain": "unknown", "label": 1},
    ]
    gate_out = {
        "pred": [0, 1],
        "score": [0.2, 1.2],
        "distance": [0.2, 1.2],
        "radius": [1.0, 1.0],
        "margin_ok": [True, False],
        "nearest_cluster": [0, 1],
        "nearest_intent": ["intent_a", "heldout"],
    }

    preds = _build_predictions(
        rows=records,
        gate_out=gate_out,
        router_domains=["domain_a", "domain_a"],
        router_probs=[0.9, 0.8],
        expert_intents=["intent_a", "intent_a"],
        expert_probs=[0.95, 0.7],
    )

    assert preds[0]["gate_stage"] == "cascade_minilm_gate"
    assert preds[0]["intent"] == "intent_a"
    assert preds[1]["gate_stage"] == "cascade_minilm_gate"
    assert preds[1]["intent"] == "__oos__"


def test_cascade_smollm_uses_dedicated_smollm_gate_evaluator(tmp_path: Path, monkeypatch):
    captured: list[list[str]] = []

    def fake_run(cmd: list[str]) -> None:
        captured.append(cmd)

    monkeypatch.setattr(runner, "_run", fake_run)
    anchor = {
        "config": {
            "spec": {"slug": "clinc150", "kir_tag": "kir50_seed42"},
            "gate_train": "train.json",
            "gate_val": "val.json",
            "gate_test": "test.json",
            "router_train": "router_train.json",
            "experts_data_root": "experts_data",
            "router_ckpt": "router.pt",
            "experts_root": "experts",
        },
        "spec": {"slug": "clinc150", "kir_tag": "kir50_seed42"},
    }
    args = type(
        "Args",
        (),
        {
            "model_path": "smollm",
            "device": "cuda",
            "batch_size": 8,
            "seed": 123,
            "flat_batch_size": 16,
        },
    )()

    eval_path = runner._run_cascade_smollm(anchor, tmp_path / "Cascade_SmolLM", args)

    assert eval_path == tmp_path / "Cascade_SmolLM" / "eval_results.json"
    assert captured
    cmd = captured[0]
    assert str(runner.CASCADE_SMOLLM) in cmd
    assert "--gate_encoder_path" not in cmd
    assert "--gate_detector_path" not in cmd
    assert ["--model_path", "smollm"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
    assert ["--threshold_source", "validation"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
    assert ["--threshold_objective", "main_table_constrained_balanced"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
    assert ["--dataset_slug", "clinc150"] in [cmd[i : i + 2] for i in range(len(cmd) - 1)]
