"""公开结果导出的白名单、hash 和禁止文件协议测试。"""

from pathlib import Path

import yaml

from tools.maintenance.export_public_results import collect_records, execute_export, verify_export


def _fixture(tmp_path: Path, *, source: str = "../artifacts/s2c/demo.csv") -> tuple[Path, Path, Path]:
    project_root = tmp_path / "s2c"
    source_path = tmp_path / "artifacts" / "s2c" / "demo.csv"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("metric,value\nscore,1\n", encoding="utf-8")
    config_path = project_root / "configs" / "public_results.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "limits": {"max_file_bytes": 1024, "max_total_bytes": 2048},
                "files": [
                    {
                        "experiment_id": "demo",
                        "category": "pipeline",
                        "source": source,
                        "public": "pipeline/demo.csv",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return project_root, config_path, project_root / "results"


def test_execute_and_verify_public_result_snapshot(tmp_path: Path) -> None:
    project_root, config_path, results_root = _fixture(tmp_path)

    limits, records, errors = collect_records(config_path, project_root, results_root)
    assert limits["max_file_bytes"] == 1024
    assert len(records) == 1
    assert errors == []

    assert execute_export(config_path, project_root, results_root) == 0
    assert verify_export(config_path, project_root, results_root) == 0
    assert (results_root / "MANIFEST.csv").is_file()
    assert (results_root / "pipeline" / "demo.csv").read_text(encoding="utf-8") == "metric,value\nscore,1\n"
    manifest = (results_root / "MANIFEST.csv").read_text(encoding="utf-8")
    assert "/home/" not in manifest
    assert "../artifacts/s2c/demo.csv" in manifest


def test_missing_source_is_reported_without_fabrication(tmp_path: Path) -> None:
    project_root, config_path, results_root = _fixture(tmp_path, source="../artifacts/s2c/missing.csv")

    _, records, errors = collect_records(config_path, project_root, results_root)
    assert records == []
    assert any("missing source" in error for error in errors)
    assert not results_root.exists()


def test_model_and_checkpoint_paths_are_rejected(tmp_path: Path) -> None:
    project_root, config_path, results_root = _fixture(
        tmp_path,
        source="../artifacts/s2c/models/demo.csv",
    )

    _, records, errors = collect_records(config_path, project_root, results_root)
    assert records == []
    assert any("prohibited directory" in error for error in errors)
