"""实验登记表审计的最小协议测试。"""

import json
from pathlib import Path

import yaml

from tools.analysis.audit_experiment_registry import audit


def _write_registry(root: Path, expected: int) -> Path:
    (root / "artifacts" / "demo").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "tools" / "entry.py").write_text("# fixture\n", encoding="utf-8")
    (root / "artifacts" / "demo" / "manifest.json").write_text(
        json.dumps({"expected_unit_count": 2}), encoding="utf-8"
    )
    (root / "artifacts" / "demo" / "summary.csv").write_text("id\n1\n2\n", encoding="utf-8")
    registry = {
        "artifact_root": "artifacts",
        "experiments": {
            "pipeline": {
                "demo": {
                    "id": "pipeline.demo",
                    "status": "complete",
                    "entrypoints": ["tools/entry.py"],
                    "artifact_root": "demo",
                    "manifest": ["manifest.json"],
                    "summary": ["summary.csv"],
                    "expected_unit_count": expected,
                    "count_source": "csv_rows",
                    "count_summary": "summary.csv",
                }
            }
        },
    }
    path = root / "registry.yaml"
    path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    return path


def test_registry_audit_reports_csv_count_and_missing_errors(tmp_path: Path, monkeypatch) -> None:
    import tools.analysis.audit_experiment_registry as module

    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    registry = _write_registry(tmp_path, expected=2)

    report = audit(
        registry,
        tmp_path / "active.json",
        tmp_path / "unreferenced.json",
        tmp_path / "freeze.json",
        write_freeze=False,
    )

    assert report["status"] == "pass"
    assert report["entries"][0]["observed_unit_count"] == 2

    bad = _write_registry(tmp_path / "bad", expected=3)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path / "bad")
    bad_report = audit(
        bad,
        tmp_path / "bad-active.json",
        tmp_path / "bad-unreferenced.json",
        tmp_path / "bad-freeze.json",
        write_freeze=False,
    )

    assert bad_report["status"] == "fail"
    assert any("unit count mismatch" in error for error in bad_report["errors"])
