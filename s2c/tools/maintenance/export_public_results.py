"""按白名单导出可提交 GitHub 的轻量 s2c 结果。

脚本只读取 ``../artifacts``，不会训练、改写或移动原始实验产物。公开目录中的
每个文件都必须来自 ``configs/public_results.yaml``，并在 ``MANIFEST.csv`` 中留下
相对路径、大小和 SHA256 provenance。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "public_results.yaml"
DEFAULT_RESULTS = PROJECT_ROOT / "results"
MANIFEST_NAME = "MANIFEST.csv"
ALLOWED_CATEGORIES = {"pipeline", "gate_only", "representation", "robustness", "external"}
BANNED_SUFFIXES = {
    ".bin",
    ".cpp",
    ".log",
    ".npy",
    ".npz",
    ".o",
    ".parquet",
    ".pkl",
    ".pt",
    ".pyc",
    ".safetensors",
    ".so",
}
BANNED_PATH_PARTS = {"checkpoint", "checkpoints", "model", "models", "scores"}
MANIFEST_FIELDS = (
    "experiment_id",
    "category",
    "source_relative_path",
    "public_relative_path",
    "size_bytes",
    "sha256",
)


@dataclass(frozen=True)
class PublicEntry:
    experiment_id: str
    category: str
    source: str
    public: str


@dataclass(frozen=True)
class ExportRecord:
    entry: PublicEntry
    source_path: Path
    public_path: Path
    size_bytes: int
    sha256: str

    def as_manifest_row(self) -> dict[str, str | int]:
        return {
            "experiment_id": self.entry.experiment_id,
            "category": self.entry.category,
            "source_relative_path": Path(self.entry.source).as_posix(),
            "public_relative_path": Path(self.entry.public).as_posix(),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def load_config(config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, int], list[PublicEntry]]:
    """读取并校验公开结果白名单，不访问或修改 artifact。"""

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    limits = raw.get("limits", {})
    max_file = int(limits.get("max_file_bytes", 10 * 1024 * 1024))
    max_total = int(limits.get("max_total_bytes", 100 * 1024 * 1024))
    if max_file <= 0 or max_total <= 0:
        raise ValueError("limits must be positive")

    entries: list[PublicEntry] = []
    for index, item in enumerate(raw.get("files", [])):
        if not isinstance(item, dict):
            raise ValueError(f"files[{index}] must be a mapping")
        entry = PublicEntry(
            experiment_id=str(item["experiment_id"]),
            category=str(item["category"]),
            source=str(item["source"]),
            public=str(item["public"]),
        )
        if entry.category not in ALLOWED_CATEGORIES:
            raise ValueError(f"unsupported category: {entry.category}")
        if Path(entry.source).is_absolute() or Path(entry.public).is_absolute():
            raise ValueError("public result paths must be relative")
        if Path(entry.public).suffix.lower() not in {".csv", ".json"}:
            raise ValueError(f"public file must be CSV or JSON: {entry.public}")
        entries.append(entry)

    if not entries:
        raise ValueError("public result whitelist is empty")
    public_paths = [entry.public for entry in entries]
    if len(public_paths) != len(set(public_paths)):
        raise ValueError("duplicate public paths in whitelist")
    return {"max_file_bytes": max_file, "max_total_bytes": max_total}, entries


def _validate_entry(entry: PublicEntry, project_root: Path, results_root: Path) -> tuple[Path, Path, list[str]]:
    errors: list[str] = []
    workspace_root = project_root.parent
    source_path = (project_root / entry.source).resolve()
    public_path = (results_root / entry.public).resolve()
    if not _within(source_path, workspace_root):
        errors.append(f"source escapes workspace: {entry.source}")
    if not _within(public_path, results_root.resolve()):
        errors.append(f"public path escapes results: {entry.public}")
    source_parts = {part.lower() for part in Path(entry.source).parts}
    if source_parts & BANNED_PATH_PARTS:
        errors.append(f"source is under a prohibited directory: {entry.source}")
    if source_path.suffix.lower() in BANNED_SUFFIXES:
        errors.append(f"prohibited source suffix: {entry.source}")
    return source_path, public_path, errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_records(
    config_path: Path = DEFAULT_CONFIG,
    project_root: Path = PROJECT_ROOT,
    results_root: Path = DEFAULT_RESULTS,
) -> tuple[dict[str, int], list[ExportRecord], list[str]]:
    """解析白名单并计算源文件 provenance；缺失文件只进入 errors。"""

    limits, entries = load_config(config_path)
    records: list[ExportRecord] = []
    errors: list[str] = []
    for entry in entries:
        source_path, public_path, entry_errors = _validate_entry(entry, project_root, results_root)
        errors.extend(entry_errors)
        if entry_errors:
            continue
        if not source_path.is_file():
            errors.append(f"missing source: {entry.source}")
            continue
        size_bytes = source_path.stat().st_size
        if size_bytes > limits["max_file_bytes"]:
            errors.append(f"file exceeds limit ({size_bytes} bytes): {entry.source}")
            continue
        records.append(
            ExportRecord(
                entry=entry,
                source_path=source_path,
                public_path=public_path,
                size_bytes=size_bytes,
                sha256=_sha256(source_path),
            )
        )
    total_size = sum(record.size_bytes for record in records)
    if total_size > limits["max_total_bytes"]:
        errors.append(f"public result total exceeds limit ({total_size} bytes)")
    return limits, sorted(records, key=lambda record: record.entry.public), errors


def _manifest_rows(records: Iterable[ExportRecord]) -> list[dict[str, str | int]]:
    return [record.as_manifest_row() for record in records]


def _write_manifest(path: Path, records: list[ExportRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(_manifest_rows(records))


def _print_report(mode: str, records: list[ExportRecord], errors: list[str]) -> None:
    payload = {
        "mode": mode,
        "record_count": len(records),
        "total_bytes": sum(record.size_bytes for record in records),
        "missing_or_invalid": errors,
        "files": [record.as_manifest_row() for record in records],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def execute_export(
    config_path: Path = DEFAULT_CONFIG,
    project_root: Path = PROJECT_ROOT,
    results_root: Path = DEFAULT_RESULTS,
) -> int:
    _, records, errors = collect_records(config_path, project_root, results_root)
    _print_report("execute", records, errors)
    if errors:
        return 2
    for record in records:
        record.public_path.parent.mkdir(parents=True, exist_ok=True)
        if record.public_path.is_file() and _sha256(record.public_path) == record.sha256:
            continue
        shutil.copyfile(record.source_path, record.public_path)
    _write_manifest(results_root / MANIFEST_NAME, records)
    return 0


def verify_export(
    config_path: Path = DEFAULT_CONFIG,
    project_root: Path = PROJECT_ROOT,
    results_root: Path = DEFAULT_RESULTS,
) -> int:
    _, records, errors = collect_records(config_path, project_root, results_root)
    manifest_path = results_root / MANIFEST_NAME
    if not manifest_path.is_file():
        errors.append(f"missing manifest: {manifest_path}")
        rows: list[dict[str, str]] = []
    else:
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(MANIFEST_FIELDS):
                errors.append("manifest header mismatch")
            rows = list(reader)

    expected_rows = _manifest_rows(records)
    if rows != [{key: str(row[key]) for key in MANIFEST_FIELDS} for row in expected_rows]:
        errors.append("manifest contents do not match the current whitelist/source hashes")

    expected_public = {record.public_path for record in records}
    actual_public = {
        path
        for path in results_root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME and path.name != "README.md"
    }
    if actual_public != expected_public:
        errors.append("results contains files outside the public whitelist")
    for record in records:
        if not record.public_path.is_file():
            errors.append(f"missing exported file: {record.entry.public}")
        elif record.public_path.stat().st_size != record.size_bytes or _sha256(record.public_path) != record.sha256:
            errors.append(f"hash/size mismatch: {record.entry.public}")
    _print_report("verify", records, errors)
    return 0 if not errors else 1


def dry_run(
    config_path: Path = DEFAULT_CONFIG,
    project_root: Path = PROJECT_ROOT,
    results_root: Path = DEFAULT_RESULTS,
) -> int:
    _, records, errors = collect_records(config_path, project_root, results_root)
    _print_report("dry-run", records, errors)
    return 0 if not errors else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export small, auditable s2c result snapshots")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.dry_run:
            return dry_run(args.config, PROJECT_ROOT, args.results_root)
        if args.execute:
            return execute_export(args.config, PROJECT_ROOT, args.results_root)
        return verify_export(args.config, PROJECT_ROOT, args.results_root)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"public result export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
