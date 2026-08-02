from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.compat.textoir import audit_textoir_protocol as audit
from tools.compat.textoir import import_textoir_results as importer
from tools.compat.textoir import run_external_textoir as runner
from tools.compat.textoir import run_textoir_matrix as matrix_runner
from tools.compat.textoir._common import default_textoir_root


def test_migrated_textoir_clone_has_expected_provenance_and_protocol() -> None:
    textoir_root = default_textoir_root()
    if not textoir_root.is_dir():
        pytest.skip("Migrated TEXTOIR clone is not available")

    result = audit.build_audit(textoir_root, ("banking",))

    assert result["repository"]["origin"] == "https://github.com/thuiar/TEXTOIR.git"
    assert result["repository"]["branch"] == "main"
    assert result["repository"]["clean"] is True
    assert len(result["repository"]["head"]) == 40
    assert result["methods_complete"] is True
    assert all(method["registered"] for method in result["methods"].values())
    assert all(method["contract_matches_example"] for method in result["methods"].values())
    assert result["methods"]["KNNCL"]["runner_contract"]["loss"] == "KNNCLoss"
    assert result["methods"]["DA-ADB"]["runner_contract"]["backbone"] == "bert_disaware"
    assert "save_model" not in result["methods"]["DA-ADB"]["runner_contract"]
    assert result["datasets"]["banking"]["splits"]["test"]["samples"] > 0


def test_known_label_selection_is_deterministic_and_split_hashes_are_bytes_exact() -> None:
    textoir_root = default_textoir_root()
    if not textoir_root.is_dir():
        pytest.skip("Migrated TEXTOIR clone is not available")

    first = runner.select_known_labels(textoir_root, "banking", 0.5, seed=0)
    repeated = runner.select_known_labels(textoir_root, "banking", 0.5, seed=0)
    another_seed = runner.select_known_labels(textoir_root, "banking", 0.5, seed=1)
    split = textoir_root / "data" / "banking" / "test.tsv"
    result = audit.build_audit(textoir_root, ("banking",))

    assert first == repeated
    assert first != another_seed
    assert len(first) == round(77 * 0.5)
    assert result["datasets"]["banking"]["splits"]["test"]["sha256"] == hashlib.sha256(
        split.read_bytes()
    ).hexdigest()


def test_dirty_worktree_guard_blocks_execution_but_allows_dry_run() -> None:
    with pytest.raises(ValueError, match="dirty TEXTOIR worktree"):
        runner.require_clean_worktree("?? local-output.txt", dry_run=False)

    runner.require_clean_worktree("?? local-output.txt", dry_run=True)
    runner.require_clean_worktree("", dry_run=False)


def test_real_run_requires_existing_local_bert_model(tmp_path: Path) -> None:
    assert runner.resolve_bert_model(None, dry_run=True) is None
    with pytest.raises(ValueError, match="--bert-model is required"):
        runner.resolve_bert_model(None, dry_run=False)
    with pytest.raises(FileNotFoundError, match="Local BERT model directory"):
        runner.resolve_bert_model(tmp_path / "missing", dry_run=False)

    model = tmp_path / "bert"
    model.mkdir()
    assert runner.resolve_bert_model(model, dry_run=False) == model.resolve()


def test_runtime_overlay_patches_only_selected_config_and_preserves_clean_clone(
    tmp_path: Path,
) -> None:
    textoir_root = default_textoir_root()
    if not textoir_root.is_dir():
        pytest.skip("Migrated TEXTOIR clone is not available")
    model = tmp_path / "local-bert"
    model.mkdir()
    source_config = textoir_root / "open_intent_detection" / "configs" / "MSP.py"
    source_hash_before = hashlib.sha256(source_config.read_bytes()).hexdigest()
    status_before = subprocess.run(
        ["git", "-C", str(textoir_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    overlay, provenance = runner.prepare_runtime_overlay(
        textoir_root, tmp_path / "run", "MSP", model.resolve()
    )

    patched = (overlay / "configs" / "MSP.py").read_text(encoding="utf-8")
    status_after = subprocess.run(
        ["git", "-C", str(textoir_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert str(model.resolve()) in patched
    assert "/home/sharing/disk1/pretrained_embedding/bert" not in patched
    assert hashlib.sha256(source_config.read_bytes()).hexdigest() == source_hash_before
    assert status_before == status_after == ""
    assert provenance["changed_files"] == [
        "backbones/__init__.py",
        "backbones/base.py",
        "configs/MSP.py",
        "dataloaders/__init__.py",
        "methods/__init__.py",
    ]
    assert provenance["config_changed_files"] == ["configs/MSP.py"]
    assert provenance["compatibility_changed_files"] == [
        "backbones/__init__.py",
        "backbones/base.py",
        "dataloaders/__init__.py",
        "methods/__init__.py",
    ]
    assert len(provenance["compatibility_patches"]) == 4
    assert provenance["source_config_sha256"] == source_hash_before
    assert provenance["overlay_config_sha256"] != source_hash_before
    assert provenance["source_tree_sha256"] != provenance["overlay_tree_sha256"]
    assert provenance["bert_model"] == str(model.resolve())


def test_bert_overlay_applies_only_documented_compatibility_routes(tmp_path: Path) -> None:
    textoir_root = default_textoir_root()
    if not textoir_root.is_dir():
        pytest.skip("Migrated TEXTOIR clone is not available")
    model = tmp_path / "local-bert"
    model.mkdir()

    overlay, provenance = runner.prepare_runtime_overlay(
        textoir_root, tmp_path / "run", "MSP", model.resolve()
    )

    overlay_backbones = (overlay / "backbones" / "__init__.py").read_text(encoding="utf-8")
    overlay_methods = (overlay / "methods" / "__init__.py").read_text(encoding="utf-8")
    source_backbones = (
        textoir_root / "open_intent_detection" / "backbones" / "__init__.py"
    ).read_text(encoding="utf-8")
    source_methods = (
        textoir_root / "open_intent_detection" / "methods" / "__init__.py"
    ).read_text(encoding="utf-8")
    overlay_dataloaders = (overlay / "dataloaders" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "LLAMA_lora_Disaware" not in overlay_backbones
    assert "llama_disaware" not in overlay_backbones
    assert "ADBManager_llama" not in overlay_methods
    assert "DA-ADB_llama" not in overlay_methods
    assert "LLAMA_lora_Disaware" in source_backbones
    assert "ADBManager_llama" in source_methods
    assert "'bert': BERT_Loader" in overlay_dataloaders
    assert "'bert_doc': BERT_Loader" in overlay_dataloaders
    assert "'bert_disaware': BERT_Loader" in overlay_dataloaders
    assert all(
        patch["source_sha256"] != patch["overlay_sha256"]
        for patch in provenance["compatibility_patches"]
    )


def test_adb_overlay_moves_only_diagnostic_delta_history_to_cpu(tmp_path: Path) -> None:
    textoir_root = default_textoir_root()
    if not textoir_root.is_dir():
        pytest.skip("Migrated TEXTOIR clone is not available")
    model = tmp_path / "local-bert"
    model.mkdir()
    source = textoir_root / "open_intent_detection" / "methods" / "ADB" / "manager.py"
    source_before = source.read_text(encoding="utf-8")

    overlay, provenance = runner.prepare_runtime_overlay(
        textoir_root, tmp_path / "run", "ADB", model.resolve()
    )

    patched = (overlay / "methods" / "ADB" / "manager.py").read_text(encoding="utf-8")
    assert "point.detach().cpu().numpy()" in patched
    assert "point.detach().cpu().numpy()" not in source_before
    assert source.read_text(encoding="utf-8") == source_before
    assert "methods/ADB/manager.py" in provenance["compatibility_changed_files"]
    adb_patch = next(
        patch
        for patch in provenance["compatibility_patches"]
        if patch["file"] == "methods/ADB/manager.py"
    )
    assert adb_patch["reason"] == "move ADB diagnostic delta history to CPU before np.save"


def test_external_artifact_audit_requires_predictions_and_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt"
    prediction_dir = (
        run_dir
        / "textoir_outputs"
        / "open_intent_detection"
        / "MSP_banking_0.5_1.0_bert_0"
    )
    prediction_dir.mkdir(parents=True)
    np.save(prediction_dir / "y_true.npy", np.asarray([0, 1]))
    np.save(prediction_dir / "y_pred.npy", np.asarray([0, 1]))

    incomplete = runner.audit_run_artifacts(run_dir)
    assert incomplete["complete"] is False
    assert incomplete["prediction_directory_count"] == 1

    results = run_dir / "results" / "results.csv"
    results.parent.mkdir(parents=True)
    results.write_text("dataset,method\nbanking,MSP\n", encoding="utf-8")
    complete = runner.audit_run_artifacts(run_dir)
    assert complete["complete"] is True
    assert len(complete["y_true_sha256"]) == 64
    assert len(complete["y_pred_sha256"]) == 64


def _write_complete_matrix_attempt(attempt_dir: Path, unit: dict) -> None:
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "imported").mkdir()
    (attempt_dir / "imported" / "predictions.jsonl").write_text(
        '{"sample_id":"x"}\n', encoding="utf-8"
    )
    imported = {
        **unit,
        "metrics": {
            "accuracy": 0.5,
            "known_macro_f1": 0.6,
            "open_oos_f1": 0.7,
            "macro_f1": 0.8,
        },
    }
    (attempt_dir / "imported" / "import_summary.json").write_text(
        json.dumps(imported) + "\n", encoding="utf-8"
    )
    (attempt_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                **unit,
                "status": "complete",
                "return_code": 0,
                "upstream_clean_after_run": True,
                "runtime_overlay": {"overlay_unchanged_after_run": True},
                "artifact_audit": {"complete": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_textoir_first_batch_matrix_has_108_units_and_resumes_by_attempt(
    tmp_path: Path,
) -> None:
    units = matrix_runner.build_matrix(
        ("banking", "oos", "stackoverflow"),
        matrix_runner.FIRST_BATCH_METHODS,
        matrix_runner.KIRS,
        matrix_runner.SEEDS,
    )
    assert len(units) == 108
    first_dir = matrix_runner.unit_directory(tmp_path, units[0])
    first_attempt = first_dir / "attempts" / "attempt_0001"
    _write_complete_matrix_attempt(first_attempt, units[0])

    assert matrix_runner.completed_attempt(first_dir) == first_attempt
    assert matrix_runner.next_attempt_directory(first_dir).name == "attempt_0002"
    status = matrix_runner.matrix_status(tmp_path, units)
    assert status["expected_units"] == 108
    assert status["complete_units"] == 1
    assert status["missing_units"] == 107
    matrix_runner.export_metric_summaries(tmp_path)
    raw = (tmp_path / "textoir_results_by_seed.csv").read_text(encoding="utf-8")
    summary = (tmp_path / "textoir_baseline_summary.csv").read_text(encoding="utf-8")
    assert "banking,MSP,0.25,0,0.5,0.6,0.7,0.8" in raw
    assert "banking,MSP,0.25,1,0.5,0.0,0.6,0.0,0.7,0.0,0.8,0.0" in summary


def test_metric_export_keeps_completed_runs_outside_current_resume_subset(
    tmp_path: Path,
) -> None:
    """单方法续跑后重建共享 CSV 时，不能丢失其他方法的成功结果。"""

    msp = {
        "dataset": "banking",
        "method": "MSP",
        "known_cls_ratio": 0.5,
        "seed": 0,
    }
    doc = {**msp, "method": "DOC"}
    for unit in (msp, doc):
        attempt = matrix_runner.unit_directory(tmp_path, unit) / "attempts/attempt_0001"
        _write_complete_matrix_attempt(attempt, unit)

    # 模拟 ``--methods ADB`` 续跑后的收尾：汇总器不再依赖本次命令的子矩阵。
    matrix_runner.export_metric_summaries(tmp_path)

    raw = (tmp_path / "textoir_results_by_seed.csv").read_text(encoding="utf-8")
    summary = (tmp_path / "textoir_baseline_summary.csv").read_text(encoding="utf-8")
    assert "banking,MSP,0.5,0" in raw
    assert "banking,DOC,0.5,0" in raw
    assert "banking,MSP,0.5,1" in summary
    assert "banking,DOC,0.5,1" in summary


def test_incomplete_textoir_attempt_is_preserved_for_retry(tmp_path: Path) -> None:
    unit = {
        "dataset": "banking",
        "method": "MSP",
        "known_cls_ratio": 0.5,
        "seed": 0,
    }
    unit_dir = matrix_runner.unit_directory(tmp_path, unit)
    failed_attempt = unit_dir / "attempts" / "attempt_0001"
    failed_attempt.mkdir(parents=True)
    (failed_attempt / "run_manifest.json").write_text(
        '{"status":"failed","return_code":1}\n', encoding="utf-8"
    )

    audit_result = matrix_runner.audit_attempt(failed_attempt)
    assert audit_result["complete"] is False
    assert matrix_runner.completed_attempt(unit_dir) is None
    assert matrix_runner.next_attempt_directory(unit_dir).name == "attempt_0002"
    assert failed_attempt.is_dir()


def test_matrix_command_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    python_link = tmp_path / "venv" / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    python_link.symlink_to(Path(sys.executable))
    args = SimpleNamespace(
        textoir_root=tmp_path / "textoir",
        bert_model=tmp_path / "bert",
        python_executable=python_link,
        gpu_id="0",
        timeout=None,
    )
    unit = {
        "dataset": "banking",
        "method": "MSP",
        "known_cls_ratio": 0.5,
        "seed": 0,
    }

    command = matrix_runner.build_external_command(args, unit, tmp_path / "attempt")
    selected = command[command.index("--python-executable") + 1]

    assert selected == str(python_link.absolute())
    assert selected != str(python_link.resolve())


def test_importer_metrics_use_complete_known_and_unknown_label_space() -> None:
    y_true = np.asarray([0, 0, 1, 1, 2, 2])
    y_pred = np.asarray([0, 2, 1, 2, 2, 0])

    result = importer.metrics(y_true, y_pred, unknown_id=2)

    assert result["samples"] == 6
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["known_macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert result["open_oos_f1"] == pytest.approx(0.4)
    assert result["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 0.4) / 3)


@pytest.mark.parametrize(
    "module",
    [
        "audit_textoir_protocol",
        "normalize_textoir_splits",
        "run_external_textoir",
        "run_textoir_matrix",
        "import_textoir_results",
    ],
)
def test_compatibility_scripts_support_package_execution(module: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", f"tools.compat.textoir.{module}", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
