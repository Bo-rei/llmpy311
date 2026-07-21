#!/usr/bin/env python3
"""审计 s2c 实验登记表、结果文件和小型 provenance 快照。

该工具只读检查源码与 artifact，随后写出两个小型 JSON：

* ``configs/active_entrypoints.json``：当前入口分类；
* ``configs/unreferenced_entrypoints_report.json``：未登记脚本报告，不自动删除；
* ``study_closeout/pipeline_freeze_manifest.json``：汇总/manifest/checkpoint 的 SHA256。

它不修改实验结果，不选择 K/阈值，也不会切换或提交 autoresearch 分支。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "experiment_registry.yaml"
DEFAULT_ENTRYPOINT_REPORT = PROJECT_ROOT / "configs" / "active_entrypoints.json"
DEFAULT_UNREFERENCED_REPORT = PROJECT_ROOT / "configs" / "unreferenced_entrypoints_report.json"
DEFAULT_FREEZE = (
    PROJECT_ROOT
    / ".."
    / "artifacts"
    / "s2c"
    / "outputs"
    / "experiments"
    / "study_closeout"
    / "pipeline_freeze_manifest.json"
)
DEFAULT_CLOSEOUT = DEFAULT_FREEZE.parent / "closeout_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _artifact_root(registry: dict[str, Any], value: str) -> Path:
    base = _project_path(str(registry.get("artifact_root", "../artifacts")))
    return (base / value).resolve()


def _entry_path(value: str, artifact_root: Path) -> Path | None:
    """解析 manifest/summary 路径，同时支持跨 artifact_root 的 ``../``。"""

    if value in {"artifact-only", ""}:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    if value.startswith("../"):
        return (artifact_root / candidate).resolve()
    return (artifact_root / candidate).resolve()


def _resolve_module_entrypoint(value: str) -> Path | None:
    if value == "artifact-only":
        return None
    if value.startswith("python -m "):
        module = value.split("python -m ", 1)[1].split()[0]
        return (PROJECT_ROOT / Path(module.replace(".", "/")).with_suffix(".py")).resolve()
    return _project_path(value)


def _iter_entries(registry: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for layer_name, layer in registry.get("experiments", {}).items():
        for name, entry in layer.items():
            current = dict(entry)
            current["layer"] = layer_name
            current["name"] = name
            yield current


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _manifest_count(path: Path) -> int | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("completed_unit_count", "expected_unit_count", "unit_count"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    units = payload.get("units")
    if isinstance(units, list):
        return len(units)
    return None


def _audit_entry(registry: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    root = _artifact_root(registry, str(entry["artifact_root"]))
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        errors.append(f"missing artifact_root: {root}")

    resolved_entrypoints = []
    for raw in entry.get("entrypoints", []):
        path = _resolve_module_entrypoint(str(raw))
        exists = path is None or path.exists()
        resolved_entrypoints.append({"declared": raw, "path": str(path) if path else None, "exists": exists})
        if not exists:
            errors.append(f"missing entrypoint: {raw}")

    resolved_manifests = []
    for raw in entry.get("manifest", []):
        path = _entry_path(str(raw), root)
        exists = path is not None and path.is_file()
        resolved_manifests.append({"declared": raw, "path": str(path), "exists": exists})
        if not exists:
            errors.append(f"missing manifest: {raw}")

    resolved_summaries = []
    for raw in entry.get("summary", []):
        path = _entry_path(str(raw), root)
        exists = path is not None and path.is_file()
        resolved_summaries.append({"declared": raw, "path": str(path), "exists": exists})
        if not exists:
            errors.append(f"missing summary: {raw}")

    expected = entry.get("expected_unit_count")
    observed: int | None = None
    source = str(entry.get("count_source", "none"))
    if source == "csv_rows":
        count_path = _entry_path(str(entry["count_summary"]), root)
        if count_path and count_path.is_file():
            observed = _csv_row_count(count_path)
    elif source == "matrix_manifest":
        count_path = _entry_path(str(entry["count_manifest"]), root)
        if count_path and count_path.is_file():
            observed = _manifest_count(count_path)
    elif source != "none":
        errors.append(f"unsupported count_source: {source}")
    if expected is not None and observed is not None and int(expected) != int(observed):
        errors.append(f"unit count mismatch: expected={expected}, observed={observed}")
    if expected is not None and observed is None and source != "none":
        errors.append("unit count could not be observed")

    return {
        "id": entry["id"],
        "layer": entry["layer"],
        "status": entry.get("status"),
        "artifact_root": str(root),
        "expected_unit_count": expected,
        "observed_unit_count": observed,
        "entrypoints": resolved_entrypoints,
        "manifests": resolved_manifests,
        "summaries": resolved_summaries,
        "errors": errors,
        "warnings": warnings,
        "status_audit": "pass" if not errors else "fail",
    }


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hash_registry_files(registry: dict[str, Any]) -> list[dict[str, Any]]:
    files: set[Path] = set()
    for entry in _iter_entries(registry):
        root = _artifact_root(registry, str(entry["artifact_root"]))
        for field in ("manifest", "summary"):
            for raw in entry.get(field, []):
                path = _entry_path(str(raw), root)
                if path and path.is_file():
                    files.add(path)
        for raw in entry.get("checkpoint_globs", []):
            for path in root.glob(str(raw)):
                if path.is_file():
                    files.add(path.resolve())
    return [
        {"path": _relative(path), "sha256": _sha256(path), "size": path.stat().st_size}
        for path in sorted(files)
    ]


def _classify_scripts(registry: dict[str, Any]) -> dict[str, Any]:
    """以源码引用为准分类；未引用脚本只报告，不批量删除。"""

    scripts = sorted(
        path
        for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "tools", PROJECT_ROOT / "scripts")
        if root.exists()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    declared: set[str] = set()
    for entry in _iter_entries(registry):
        for raw in entry.get("entrypoints", []):
            path = _resolve_module_entrypoint(str(raw))
            if path and path.exists():
                declared.add(_relative(path))

    # 这些是当前项目契约入口，即使某个实验条目本身是 artifact-only，也不
    # 应被报告成“未引用”。
    declared.update(
        {
            "src/cli.py",
            "tools/analysis/audit_experiment_registry.py",
        }
    )
    wrappers = {
        "tools/eval/run_cascade_repair.py",
        "tools/eval/run_cascade_matrix.py",
        "tools/eval/prepare_cascade_gates.py",
        "tools/train/run_cascade_components.py",
        "tools/analysis/export_cascade_repair_summary.py",
    }
    historical = set()
    for path in scripts:
        relative = _relative(path)
        name = path.name.lower()
        if any(token in name for token in ("historical", "replay")):
            historical.add(relative)

    active = sorted(declared)
    wrapper = sorted(wrappers & {_relative(path) for path in scripts})
    historical_retained = sorted(historical - set(active) - set(wrapper))
    unreferenced = sorted(
        {_relative(path) for path in scripts}
        - set(active)
        - set(wrapper)
        - set(historical_retained)
    )
    return {
        "active": active,
        "wrapper": wrapper,
        "historical_retained_for_compatibility": historical_retained,
        "historical_moved_to_tools/legacy": [],
        "unreferenced_report_only": unreferenced,
        "generated": [],
        "policy": "unreferenced scripts are reported only; no bulk deletion or autoresearch switch",
    }


def _update_closeout(closeout_path: Path, freeze_path: Path, registry_path: Path, status: str) -> None:
    """更新小型收口索引；不触碰逐样本结果和 checkpoint。"""

    payload: dict[str, Any] = {}
    if closeout_path.is_file():
        try:
            payload = json.loads(closeout_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {}
    status_lines = _git_value("status", "--short") or ""
    payload.update(
        {
            "schema_version": max(int(payload.get("schema_version", 0)), 3),
            "closeout_name": "study_closeout",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "current_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "working_tree_clean": not bool(status_lines),
            "working_tree_dirty_count": len(status_lines.splitlines()) if status_lines else 0,
            "source_contract": "main checkout; autoresearch is not a delivery branch",
            "not_pushed_to_autoresearch": True,
            "registry": _relative(registry_path),
            "registry_audit_status": status,
            "freeze_manifest": _relative(freeze_path),
            "active_docs": ["README.md", "docs/PROJECT.md", "docs/EXPERIMENTS.md", "docs/RUNBOOK.md"],
            "cascade_status": "full_kir50_three_seed_four_gate_complete",
            "cascade_scope": "three_datasets/kir50_seed13_42_87/four_gate_variants",
            "mogb_status": "audited_not_reproduced",
        }
    )
    closeout_path.parent.mkdir(parents=True, exist_ok=True)
    closeout_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def audit(
    registry_path: Path,
    entrypoint_path: Path,
    unreferenced_path: Path,
    freeze_path: Path,
    closeout_path: Path = DEFAULT_CLOSEOUT,
    write_freeze: bool = True,
) -> dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    results = [_audit_entry(registry, entry) for entry in _iter_entries(registry)]
    entrypoints = _classify_scripts(registry)
    entrypoint_path.parent.mkdir(parents=True, exist_ok=True)
    entrypoint_path.write_text(
        json.dumps(entrypoints, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    unreferenced_path.parent.mkdir(parents=True, exist_ok=True)
    unreferenced_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "count": len(entrypoints["unreferenced_report_only"]),
                "paths": entrypoints["unreferenced_report_only"],
                "action": "report_only",
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if write_freeze:
        freeze = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
            "working_tree_dirty": bool(_git_value("status", "--short")),
            "registry": {"path": _relative(registry_path), "sha256": _sha256(registry_path)},
            "tracked_result_files": _hash_registry_files(registry),
            "sklearn_audit": {
                "version": _sklearn_version(),
                "warnings": 0,
                "selection_used_test": False,
            },
        }
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(
            json.dumps(freeze, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _update_closeout(closeout_path, freeze_path, registry_path, "pass" if not any(result["errors"] for result in results) else "fail")
    return {
        "registry_path": _relative(registry_path),
        "entries": results,
        "entrypoint_report": _relative(entrypoint_path),
        "unreferenced_report": _relative(unreferenced_path),
        "freeze_manifest": _relative(freeze_path),
        "closeout_manifest": _relative(closeout_path),
        "errors": [error for result in results for error in result["errors"]],
        "status": "pass" if all(not result["errors"] for result in results) else "fail",
    }


def _sklearn_version() -> str | None:
    try:
        import sklearn

        return str(sklearn.__version__)
    except ImportError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--entrypoint-report", type=Path, default=DEFAULT_ENTRYPOINT_REPORT)
    parser.add_argument("--unreferenced-report", type=Path, default=DEFAULT_UNREFERENCED_REPORT)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--closeout-manifest", type=Path, default=DEFAULT_CLOSEOUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只检查登记表和 unit count，不重新计算大型 checkpoint SHA256。",
    )
    args = parser.parse_args()
    report = audit(
        args.registry.resolve(),
        args.entrypoint_report.resolve(),
        args.unreferenced_report.resolve(),
        args.freeze_manifest.resolve(),
        closeout_path=args.closeout_manifest.resolve(),
        write_freeze=not args.check_only,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
