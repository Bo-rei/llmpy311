"""分析资产目录审计的最小协议测试。"""

import json
from pathlib import Path

import yaml

from tools.maintenance.audit_asset_catalog import audit_catalog


def _write_catalog(root: Path, *, bundle_id: str = "demo_bundle", manifest_name: str = "MANIFEST.json") -> Path:
    result_dir = root / "results" / "analysis" / "demo_bundle"
    figure_dir = root / "figures" / "demo_bundle"
    report = root / "docs" / "analysis" / "DEMO.md"
    result_dir.mkdir(parents=True)
    figure_dir.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    (result_dir / "summary.csv").write_text("method\nS2C\n", encoding="utf-8")
    (result_dir / "source.csv").write_text("value\n1\n", encoding="utf-8")
    (figure_dir / "summary.png").write_bytes(b"not-a-real-png")
    report.write_text("# Demo\n", encoding="utf-8")
    (result_dir / manifest_name).write_text(
        json.dumps(
            {
                "figures": ["figures/demo_bundle/summary.png"],
                "sources": {"results/analysis/demo_bundle/source.csv": "test"},
            }
        ),
        encoding="utf-8",
    )
    registry = {
        "analysis_bundles": {
            bundle_id: {
                "status": "closed",
                "contract_layer": "same_protocol_fair",
                "question": "Does the catalog remain auditable?",
                "source_experiments": ["demo.experiment"],
                "builder": "tools/build_demo.py",
                "result_dir": "results/analysis/demo_bundle",
                "figure_dir": "figures/demo_bundle",
                "report": "docs/analysis/DEMO.md",
                "manifest": f"results/analysis/demo_bundle/{manifest_name}",
                "selected_for_main_report": False,
                "commit_policy": "closed_evidence",
            }
        }
    }
    (root / "tools").mkdir()
    (root / "tools" / "build_demo.py").write_text("# fixture\n", encoding="utf-8")
    registry_path = root / "configs" / "experiment_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    return registry_path


def test_asset_catalog_checks_manifest_sources_and_paths(tmp_path: Path) -> None:
    registry = _write_catalog(tmp_path)

    report = audit_catalog(project_root=tmp_path, registry_path=registry)

    assert report["status"] == "pass"
    assert report["bundle_count"] == 1
    assert report["bundles"][0]["manifest"]["figures_references"][0]["exists"] is True


def test_asset_catalog_rejects_bad_id_and_missing_manifest_source(tmp_path: Path) -> None:
    registry = _write_catalog(tmp_path, bundle_id="DemoBundle")
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    manifest = tmp_path / "results" / "analysis" / "demo_bundle" / "MANIFEST.json"
    manifest.write_text(
        json.dumps({"figures": ["figures/demo_bundle/missing.png"]}),
        encoding="utf-8",
    )
    registry.write_text(yaml.safe_dump(payload), encoding="utf-8")

    report = audit_catalog(project_root=tmp_path, registry_path=registry)

    assert report["status"] == "fail"
    assert any("lower_snake_case" in error for error in report["errors"])
    assert any("manifest figures reference is missing" in error for error in report["errors"])


def test_asset_catalog_does_not_report_explicit_archive_roots_as_orphans(tmp_path: Path) -> None:
    registry = _write_catalog(tmp_path)
    (tmp_path / "results" / "analysis" / "archive" / "retained_v1").mkdir(parents=True)
    (tmp_path / "figures" / "archive" / "retained_v1").mkdir(parents=True)

    report = audit_catalog(project_root=tmp_path, registry_path=registry)

    assert "results/analysis/archive" not in report["orphan_result_dirs"]
    assert "figures/archive" not in report["orphan_figure_dirs"]
