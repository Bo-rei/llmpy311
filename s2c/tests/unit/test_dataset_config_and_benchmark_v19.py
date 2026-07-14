from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.active.rebuild_multi_dataset_v19 import STACKOVERFLOW_INTENTS
from tools.analysis.historical_best_pipeline_v19 import HISTORICAL_BEST_PIPELINE
from tools.analysis.run_multi_dataset_benchmark_v19 import (
    _apply_benchmark_profile_defaults,
    _resolve_effective_eval_config,
)
from src.runtime import load_profile


def test_apply_benchmark_profile_defaults_keeps_explicit_semantic_overrides():
    args = argparse.Namespace(
        gate_mode=HISTORICAL_BEST_PIPELINE.gate_mode,
        gate_radius_scale=0.975,
        semantic_gate_mode="prototype",
        semantic_prompt_version="ranking_v1",
        semantic_gate_threshold=0.91,
        semantic_uncertain_low=0.97,
        semantic_uncertain_high=1.03,
        semantic_top_k=5,
        semantic_tuning_mode="val_macro_f1",
        prototype_centers_default=3,
        multi_proto_id_threshold=0.73,
        multi_proto_threshold_mode="val_macro_f1",
        semantic_verifier_lora_r=8,
        semantic_verifier_lora_alpha=16,
    )

    resolved = _apply_benchmark_profile_defaults(args)

    assert resolved.semantic_gate_threshold == 0.91
    assert resolved.semantic_uncertain_low == 0.97
    assert resolved.semantic_uncertain_high == 1.03
    assert resolved.semantic_tuning_mode == "val_macro_f1"
    assert resolved.gate_radius_scale == 0.975
    assert resolved.multi_proto_threshold_mode == "val_macro_f1"
    assert resolved.prototype_centers_default == 3


def test_resolve_effective_eval_config_auto_uses_historical_profile_for_stackoverflow():
    args = argparse.Namespace(
        eval_profile="auto",
        semantic_gate_mode=HISTORICAL_BEST_PIPELINE.semantic_gate_mode,
        semantic_prompt_version=HISTORICAL_BEST_PIPELINE.semantic_prompt_version,
        semantic_tuning_mode=HISTORICAL_BEST_PIPELINE.semantic_tuning_mode,
        semantic_gate_threshold=HISTORICAL_BEST_PIPELINE.semantic_gate_threshold,
        semantic_uncertain_low=HISTORICAL_BEST_PIPELINE.semantic_uncertain_low,
        semantic_uncertain_high=HISTORICAL_BEST_PIPELINE.semantic_uncertain_high,
        semantic_top_k=HISTORICAL_BEST_PIPELINE.semantic_top_k,
        prototype_centers_default=HISTORICAL_BEST_PIPELINE.prototype_centers_default,
        semantic_fusion_alpha=0.7,
        semantic_fusion_beta=0.3,
        semantic_decision_policy="threshold",
        semantic_low_conf_threshold=0.8,
        semantic_high_conf_threshold=0.9,
        semantic_verifier_threshold=0.5,
        semantic_verifier_lora_r=HISTORICAL_BEST_PIPELINE.semantic_verifier_lora_r,
        semantic_verifier_lora_alpha=HISTORICAL_BEST_PIPELINE.semantic_verifier_lora_alpha,
        gate_radius_scale=1.0,
    )

    effective = _resolve_effective_eval_config("STACKOVERFLOW", args)

    assert effective["eval_profile"] == "historical_best"
    assert effective["semantic_gate_mode"] == HISTORICAL_BEST_PIPELINE.semantic_gate_mode
    assert effective["semantic_tuning_mode"] == HISTORICAL_BEST_PIPELINE.semantic_tuning_mode


def test_resolve_effective_eval_config_verifier_retained_is_dataset_agnostic():
    args = argparse.Namespace(
        eval_profile="verifier_retained",
        semantic_gate_mode=HISTORICAL_BEST_PIPELINE.semantic_gate_mode,
        semantic_prompt_version=HISTORICAL_BEST_PIPELINE.semantic_prompt_version,
        semantic_tuning_mode=HISTORICAL_BEST_PIPELINE.semantic_tuning_mode,
        semantic_gate_threshold=HISTORICAL_BEST_PIPELINE.semantic_gate_threshold,
        semantic_uncertain_low=HISTORICAL_BEST_PIPELINE.semantic_uncertain_low,
        semantic_uncertain_high=HISTORICAL_BEST_PIPELINE.semantic_uncertain_high,
        semantic_top_k=HISTORICAL_BEST_PIPELINE.semantic_top_k,
        prototype_centers_default=HISTORICAL_BEST_PIPELINE.prototype_centers_default,
        semantic_fusion_alpha=0.7,
        semantic_fusion_beta=0.3,
        semantic_decision_policy="threshold",
        semantic_low_conf_threshold=0.8,
        semantic_high_conf_threshold=0.9,
        semantic_verifier_threshold=0.5,
        semantic_verifier_lora_r=HISTORICAL_BEST_PIPELINE.semantic_verifier_lora_r,
        semantic_verifier_lora_alpha=HISTORICAL_BEST_PIPELINE.semantic_verifier_lora_alpha,
        gate_radius_scale=1.0,
    )

    stackoverflow = _resolve_effective_eval_config("STACKOVERFLOW", args)
    banking = _resolve_effective_eval_config("BANKING77-OOS", args)

    assert stackoverflow["eval_profile"] == "verifier_retained"
    assert banking["eval_profile"] == "verifier_retained"
    assert stackoverflow["semantic_gate_mode"] == banking["semantic_gate_mode"] == "llm_verifier"
    assert stackoverflow["semantic_tuning_mode"] == banking["semantic_tuning_mode"] == "val_macro_f1"


def test_resolve_effective_eval_config_keeps_legacy_stackoverflow20k_alias():
    args = argparse.Namespace(
        eval_profile="stackoverflow20k",
        semantic_gate_mode=HISTORICAL_BEST_PIPELINE.semantic_gate_mode,
        semantic_prompt_version=HISTORICAL_BEST_PIPELINE.semantic_prompt_version,
        semantic_tuning_mode=HISTORICAL_BEST_PIPELINE.semantic_tuning_mode,
        semantic_gate_threshold=0.91,
        semantic_uncertain_low=0.97,
        semantic_uncertain_high=1.03,
        semantic_top_k=5,
        prototype_centers_default=3,
        semantic_fusion_alpha=0.6,
        semantic_fusion_beta=0.4,
        semantic_decision_policy="threshold",
        semantic_low_conf_threshold=0.8,
        semantic_high_conf_threshold=0.9,
        semantic_verifier_threshold=0.5,
        semantic_verifier_lora_r=8,
        semantic_verifier_lora_alpha=16,
        gate_radius_scale=0.975,
    )

    effective = _resolve_effective_eval_config("STACKOVERFLOW", args)

    assert effective["eval_profile"] == "stackoverflow20k"
    assert effective["semantic_gate_mode"] == "llm_verifier"
    assert effective["semantic_tuning_mode"] == "val_macro_f1"
    assert effective["semantic_gate_threshold"] == 0.91
    assert effective["gate_radius_scale"] == 0.975
    assert effective["prototype_centers_default"] == 3


def test_banking_config_declares_single_domain_repo_variant():
    profile = load_profile(PROJECT_ROOT / "configs" / "profiles.yaml", "banking77_oos")

    assert profile.policy == {
        "num_intents": 50,
        "num_domains": 1,
        "multi_domain": False,
        "single_domain": True,
        "oos_strategy": "native_id_and_ood",
    }


def test_banking_orchestrator_reports_single_domain_summary():
    profile = load_profile(PROJECT_ROOT / "configs" / "profiles.yaml", "banking77_oos")

    assert profile.policy["num_domains"] == 1
    assert profile.policy["single_domain"] is True


def test_snips_orchestrator_reports_constructed_oos_strategy():
    profile = load_profile(PROJECT_ROOT / "configs" / "profiles.yaml", "snips")

    assert profile.policy["single_domain"] is True
    assert profile.policy["oos_strategy"] == "held_out_intents_plus_clinc_oos"


def test_benchmark_cli_records_requested_eval_config_in_summary(tmp_path: Path):
    data_root_base = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"

    cmd = [
        sys.executable,
        "-m",
        "tools.analysis.run_multi_dataset_benchmark_v19",
        "--datasets",
        "BANKING77-OOS",
        "--kir_values",
        "0.5",
        "--seed",
        "42",
        "--data_root_base",
        str(data_root_base),
        "--artifact_root",
        str(artifact_root),
        "--skip_eval",
        "--semantic_tuning_mode",
        "val_macro_f1",
        "--semantic_gate_threshold",
        "0.91",
        "--semantic_uncertain_low",
        "0.97",
        "--semantic_uncertain_high",
        "1.03",
        "--prototype_centers_default",
        "3",
        "--gate_radius_scale",
        "0.975",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)

    summary = json.loads((artifact_root / "benchmark_summary.json").read_text(encoding="utf-8"))
    run = summary["runs"][0]
    requested = run["requested_eval_config"]

    assert run["status"] == "rebuild_only"
    assert requested["semantic_tuning_mode"] == "val_macro_f1"
    assert requested["semantic_gate_threshold"] == 0.91
    assert requested["semantic_uncertain_low"] == 0.97
    assert requested["semantic_uncertain_high"] == 1.03
    assert requested["prototype_centers_default"] == 3
    assert requested["gate_radius_scale"] == 0.975


def test_benchmark_cli_records_effective_eval_config_for_stackoverflow(tmp_path: Path):
    data_root_base = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"

    cmd = [
        sys.executable,
        "-m",
        "tools.analysis.run_multi_dataset_benchmark_v19",
        "--datasets",
        "STACKOVERFLOW",
        "--kir_values",
        "0.5",
        "--seed",
        "42",
        "--data_root_base",
        str(data_root_base),
        "--artifact_root",
        str(artifact_root),
        "--stackoverflow_root",
        str(tmp_path / "stackoverflow_origin"),
        "--stackoverflow_known_selection_strategy",
        "seeded_random",
        "--skip_eval",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    stackoverflow_root = tmp_path / "stackoverflow_origin"
    stackoverflow_root.mkdir()
    train_rows = ["Title,Body,Tag"] + [f"{intent} train,,{intent}" for intent in STACKOVERFLOW_INTENTS]
    valid_rows = ["Title,Body,Tag"] + [f"{intent} valid,,{intent}" for intent in STACKOVERFLOW_INTENTS]
    test_rows = ["Title,Body,Tag"] + [f"{intent} test,,{intent}" for intent in STACKOVERFLOW_INTENTS]
    (stackoverflow_root / "train.csv").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (stackoverflow_root / "valid.csv").write_text("\n".join(valid_rows) + "\n", encoding="utf-8")
    (stackoverflow_root / "test.csv").write_text("\n".join(test_rows) + "\n", encoding="utf-8")

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)

    summary = json.loads((artifact_root / "benchmark_summary.json").read_text(encoding="utf-8"))
    run = summary["runs"][0]
    effective = run["effective_eval_config"]

    assert run["status"] == "rebuild_only"
    assert run["known_selection_strategy"] == "seeded_random"
    assert effective["eval_profile"] == "historical_best"
    assert effective["semantic_gate_mode"] == HISTORICAL_BEST_PIPELINE.semantic_gate_mode
    assert effective["semantic_tuning_mode"] == HISTORICAL_BEST_PIPELINE.semantic_tuning_mode
    assert effective["gate_radius_scale"] == 1.0
