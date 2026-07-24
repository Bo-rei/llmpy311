"""验证实质性 s2c 修改是否同步追加 DEVELOPMENT_LOG。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


LOG_PATH = "docs/DEVELOPMENT_LOG.md"
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".log"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def changed_paths(root: Path) -> set[str]:
    """读取 Git porcelain 输出；同时覆盖已跟踪与未跟踪文件。"""
    git_root = Path(
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    project_prefix = root.relative_to(git_root).as_posix() + "/"
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "-z"],
        check=True,
        capture_output=True,
        text=False,
    )
    records = [item for item in result.stdout.decode("utf-8").split("\0") if item]
    paths: set[str] = set()
    for record in records:
        path = record[3:]
        # rename/copy porcelain 记录紧跟旧路径；两条路径均应被检查。
        candidates = path.split(" -> ", 1) if " -> " in path else [path]
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate.startswith(project_prefix):
                paths.add(candidate.removeprefix(project_prefix))
    return paths


def is_substantive(path: str) -> bool:
    candidate = Path(path)
    return (
        path != LOG_PATH
        and not any(part in IGNORED_PARTS for part in candidate.parts)
        and candidate.suffix not in IGNORED_SUFFIXES
    )


def main() -> int:
    root = project_root()
    paths = changed_paths(root)
    substantive = sorted(path for path in paths if is_substantive(path))
    if not substantive:
        print("development log check: no substantive s2c changes")
        return 0
    if LOG_PATH not in paths:
        print("development log check failed: substantive changes require docs/DEVELOPMENT_LOG.md", file=sys.stderr)
        for path in substantive:
            print(f"  - {path}", file=sys.stderr)
        return 1
    print(f"development log check: covered {len(substantive)} substantive path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
