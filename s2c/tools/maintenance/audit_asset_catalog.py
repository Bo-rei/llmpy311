#!/usr/bin/env python3
"""审计分析资产目录中的结果、图、报告和 provenance 关系。

该工具只读检查 ``configs/experiment_registry.yaml`` 的
``analysis_bundles``。它不移动、删除或重写任何研究结果；未登记的历史目录
只作为 warning 输出，便于后续逐项归档。

退出码：

* ``0``：登记关系通过；可能存在待清理的 orphan warning；
* ``1``：登记表或已声明资产存在错误。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "experiment_registry.yaml"
DEFAULT_RESULT_ROOT = "results/analysis"
DEFAULT_FIGURE_ROOT = "figures"

ALLOWED_STATUS = {
    "active",
    "closed",
    "reference",
    "blocked",
    "superseded",
    "archive_candidate",
    "local_only",
}
ALLOWED_COMMIT_POLICY = {
    "closed_evidence",
    "reference_only",
    "do_not_commit_as_main_evidence",
    "local_only",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".npz",
    ".npy",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".safetensors",
}
FORBIDDEN_NAME_PARTS = (
    "checkpoint",
    "embedding",
    "per_sample",
    "sample_level",
)
SLUG_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_repo_path(value: str | Path, root: Path) -> tuple[Path, bool]:
    """Resolve a repository-relative path and flag attempts to escape root."""

    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return path, False
    return path, True


def _iter_files(path: Path) -> Iterable[Path]:
    if not path.is_dir():
        return ()
    return (item for item in path.rglob("*") if item.is_file())


def _manifest_paths(payload: Any, key: str) -> list[str]:
    """Extract explicit path references from a manifest without guessing hashes."""

    value = payload.get(key) if isinstance(payload, dict) else None
    if isinstance(value, dict):
        return [str(item) for item in value]
    if not isinstance(value, list):
        return []
    paths: list[str] = []
    for item in value:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            for field in ("path", "file", "figure", "source"):
                candidate = item.get(field)
                if isinstance(candidate, str):
                    paths.append(candidate)
                    break
    return paths


def _audit_manifest(
    manifest_path: Path,
    root: Path,
    bundle_errors: list[str],
    bundle_warnings: list[str],
) -> dict[str, Any]:
    details: dict[str, Any] = {"path": _relative(manifest_path, root), "format": manifest_path.suffix.lower()}
    if manifest_path.suffix.lower() != ".json":
        details["json_checked"] = False
        return details
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        bundle_errors.append(f"manifest is not readable JSON: {_relative(manifest_path, root)} ({exc})")
        return details

    details["json_checked"] = True
    for key in ("figures", "sources"):
        references = _manifest_paths(payload, key)
        checked: list[dict[str, Any]] = []
        for raw in references:
            candidate, inside_root = _resolve_repo_path(raw, root)
            exists = candidate.is_file()
            checked.append({"declared": raw, "path": _relative(candidate, root), "exists": exists})
            if not inside_root:
                bundle_warnings.append(f"manifest reference is outside repository: {raw}")
            elif not exists:
                bundle_errors.append(
                    f"manifest {key} reference is missing: {_relative(candidate, root)}"
                )
        details[f"{key}_references"] = checked
    return details


def _audit_bundle(bundle_id: str, bundle: dict[str, Any], root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    status = str(bundle.get("status", ""))
    if not SLUG_RE.fullmatch(bundle_id):
        errors.append(f"bundle id is not lower_snake_case: {bundle_id}")
    if status not in ALLOWED_STATUS:
        errors.append(f"unsupported status: {status or '<missing>'}")
    if not str(bundle.get("contract_layer", "")).strip():
        errors.append("missing contract_layer")
    if not str(bundle.get("question", "")).strip():
        errors.append("missing question")
    if not _as_list(bundle.get("source_experiments")):
        errors.append("source_experiments must not be empty")
    commit_policy = str(bundle.get("commit_policy", ""))
    if commit_policy not in ALLOWED_COMMIT_POLICY:
        errors.append(f"unsupported commit_policy: {commit_policy or '<missing>'}")
    selected = bundle.get("selected_for_main_report")
    if not isinstance(selected, bool):
        errors.append("selected_for_main_report must be boolean")
    if status in {"blocked", "superseded", "reference", "archive_candidate"} and selected:
        errors.append(f"{status} bundle cannot be selected_for_main_report")
    if status == "closed" and commit_policy != "closed_evidence":
        warnings.append("closed bundle does not use closed_evidence commit policy")

    paths: dict[str, Any] = {}
    manifest_details: dict[str, Any] | None = None
    required_fields = ("result_dir", "report", "manifest")
    if status in {"active", "closed"}:
        for field in required_fields:
            if not str(bundle.get(field, "")).strip():
                errors.append(f"{field} is required for {status} bundle")

    result_dir = bundle.get("result_dir")
    if result_dir:
        resolved, inside_root = _resolve_repo_path(str(result_dir), root)
        paths["result_dir"] = {"declared": str(result_dir), "path": _relative(resolved, root), "exists": resolved.is_dir()}
        if not inside_root:
            errors.append("result_dir escapes repository root")
        elif not resolved.is_dir():
            errors.append(f"missing result_dir: {_relative(resolved, root)}")
        else:
            source_files = [
                item
                for item in _iter_files(resolved)
                if item.suffix.lower() in {".csv", ".json", ".md", ".yaml", ".yml"}
            ]
            paths["result_file_count"] = len(source_files)
            if not source_files:
                errors.append(f"result_dir has no small result files: {_relative(resolved, root)}")
    elif status not in {"archive_candidate", "local_only"}:
        errors.append("missing result_dir")

    figure_dir = bundle.get("figure_dir")
    if figure_dir:
        resolved, inside_root = _resolve_repo_path(str(figure_dir), root)
        paths["figure_dir"] = {"declared": str(figure_dir), "path": _relative(resolved, root), "exists": resolved.is_dir()}
        if not inside_root:
            errors.append("figure_dir escapes repository root")
        elif not resolved.is_dir():
            errors.append(f"missing figure_dir: {_relative(resolved, root)}")
        else:
            image_files = [
                item
                for item in _iter_files(resolved)
                if item.suffix.lower() in {".png", ".svg", ".pdf", ".jpg", ".jpeg"}
            ]
            paths["figure_file_count"] = len(image_files)
            if not image_files:
                warnings.append(f"figure_dir has no image files: {_relative(resolved, root)}")

    report = bundle.get("report")
    if report:
        resolved, inside_root = _resolve_repo_path(str(report), root)
        paths["report"] = {"declared": str(report), "path": _relative(resolved, root), "exists": resolved.is_file()}
        if not inside_root:
            errors.append("report escapes repository root")
        elif not resolved.is_file():
            errors.append(f"missing report: {_relative(resolved, root)}")

    manifest = bundle.get("manifest")
    if manifest:
        resolved, inside_root = _resolve_repo_path(str(manifest), root)
        paths["manifest"] = {"declared": str(manifest), "path": _relative(resolved, root), "exists": resolved.is_file()}
        if not inside_root:
            errors.append("manifest escapes repository root")
        elif not resolved.is_file():
            errors.append(f"missing manifest: {_relative(resolved, root)}")
        else:
            manifest_details = _audit_manifest(resolved, root, errors, warnings)
            result_dir_path = paths.get("result_dir", {}).get("path")
            if result_dir_path and not str(manifest_details["path"]).startswith(f"{result_dir_path}/"):
                warnings.append("manifest is outside result_dir; keep this explicit in the bundle review")

    builder = bundle.get("builder")
    if builder:
        resolved, inside_root = _resolve_repo_path(str(builder), root)
        paths["builder"] = {"declared": str(builder), "path": _relative(resolved, root), "exists": resolved.is_file()}
        if not inside_root:
            errors.append("builder escapes repository root")
        elif not resolved.is_file():
            errors.append(f"missing builder: {_relative(resolved, root)}")
    elif status in {"active", "closed"}:
        warnings.append("builder is not declared")

    forbidden: list[str] = []
    for field in ("result_dir", "figure_dir"):
        value = bundle.get(field)
        if not value:
            continue
        resolved, inside_root = _resolve_repo_path(str(value), root)
        if not inside_root or not resolved.is_dir():
            continue
        for item in _iter_files(resolved):
            lowered = item.name.lower()
            if item.suffix.lower() in FORBIDDEN_SUFFIXES or any(part in lowered for part in FORBIDDEN_NAME_PARTS):
                forbidden.append(_relative(item, root))
    if forbidden:
        warnings.append(f"forbidden or heavy artifact-like files found ({len(forbidden)}): {forbidden[:5]}")

    return {
        "id": bundle_id,
        "status": status,
        "contract_layer": bundle.get("contract_layer"),
        "selected_for_main_report": selected,
        "paths": paths,
        "manifest": manifest_details,
        "forbidden_files": forbidden,
        "errors": errors,
        "warnings": warnings,
        "status_audit": "pass" if not errors else "fail",
    }


def _orphan_directories(root: Path, registered: set[str], directory: str) -> list[str]:
    base = root / directory
    if not base.is_dir():
        return []
    orphans: list[str] = []
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        relative = _relative(item, root)
        if relative not in registered:
            orphans.append(relative)
    return orphans


def audit_catalog(
    project_root: Path = PROJECT_ROOT,
    registry_path: Path = DEFAULT_REGISTRY,
    *,
    strict_orphans: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serialisable audit report for the analysis asset catalog."""

    root = project_root.resolve()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    bundles = registry.get("analysis_bundles", {})
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(bundles, dict):
        errors.append("analysis_bundles must be a mapping")
        bundles = {}

    bundle_reports: list[dict[str, Any]] = []
    registered_result_dirs: set[str] = set()
    registered_figure_dirs: set[str] = set()
    for bundle_id, bundle in bundles.items():
        if not isinstance(bundle, dict):
            errors.append(f"bundle is not a mapping: {bundle_id}")
            continue
        report = _audit_bundle(str(bundle_id), bundle, root)
        bundle_reports.append(report)
        errors.extend(f"{bundle_id}: {item}" for item in report["errors"])
        warnings.extend(f"{bundle_id}: {item}" for item in report["warnings"])
        for field, target in (("result_dir", registered_result_dirs), ("figure_dir", registered_figure_dirs)):
            value = bundle.get(field)
            if value:
                resolved, inside_root = _resolve_repo_path(str(value), root)
                if inside_root:
                    target.add(_relative(resolved, root))

    archive_reports: list[dict[str, Any]] = []
    archives = registry.get("analysis_archives", {})
    if archives is None:
        archives = {}
    if not isinstance(archives, dict):
        errors.append("analysis_archives must be a mapping")
        archives = {}
    for archive_id, archive in archives.items():
        if not isinstance(archive, dict):
            errors.append(f"archive is not a mapping: {archive_id}")
            continue
        original = str(archive.get("original_path", ""))
        archived = str(archive.get("archived_path", ""))
        if not original or not archived:
            errors.append(f"archive {archive_id} needs original_path and archived_path")
            continue
        original_path, original_inside = _resolve_repo_path(original, root)
        archived_path, archived_inside = _resolve_repo_path(archived, root)
        original_exists = original_path.exists()
        archived_exists = archived_path.is_file()
        archive_reports.append(
            {
                "id": str(archive_id),
                "original_path": _relative(original_path, root),
                "archived_path": _relative(archived_path, root),
                "original_exists": original_exists,
                "archived_exists": archived_exists,
                "reversible": archive.get("reversible") is True,
            }
        )
        if not original_inside or not archived_inside:
            errors.append(f"archive {archive_id} escapes repository root")
        if original_exists:
            errors.append(f"archive {archive_id} still has original path: {original}")
        if not archived_exists:
            errors.append(f"archive {archive_id} is missing archived path: {archived}")
        if archive.get("reversible") is not True:
            errors.append(f"archive {archive_id} must declare reversible: true")

    orphan_results = _orphan_directories(root, registered_result_dirs, DEFAULT_RESULT_ROOT)
    orphan_figures = _orphan_directories(root, registered_figure_dirs, DEFAULT_FIGURE_ROOT)
    if orphan_results:
        warnings.append(f"unregistered result directories: {len(orphan_results)}")
    if orphan_figures:
        warnings.append(f"unregistered figure directories: {len(orphan_figures)}")
    if strict_orphans:
        errors.extend(f"unregistered result directory: {item}" for item in orphan_results)
        errors.extend(f"unregistered figure directory: {item}" for item in orphan_figures)

    return {
        "schema_version": "analysis_asset_catalog_audit_v1",
        "registry": _relative(registry_path.resolve(), root),
        "bundle_count": len(bundle_reports),
        "bundles": bundle_reports,
        "archive_count": len(archive_reports),
        "archives": archive_reports,
        "orphan_result_dirs": orphan_results,
        "orphan_figure_dirs": orphan_figures,
        "errors": errors,
        "warnings": warnings,
        "status": "pass" if not errors else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, help="可选：写出 JSON 审计报告；默认只打印")
    parser.add_argument(
        "--strict-orphans",
        action="store_true",
        help="将尚未登记的历史结果/图目录视为错误；归档前不要使用此选项",
    )
    args = parser.parse_args()
    registry = args.registry.resolve()
    report = audit_catalog(
        project_root=PROJECT_ROOT,
        registry_path=registry,
        strict_orphans=args.strict_orphans,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
