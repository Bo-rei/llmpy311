"""Atomic run-directory creation and lightweight environment provenance."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def environment_snapshot(project_root: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
    }


@contextmanager
def atomic_run_directory(final_path: Path) -> Iterator[Path]:
    """Yield a sibling temp directory and atomically publish it on success."""
    if final_path.exists():
        raise FileExistsError(f"Run directory already exists: {final_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final_path.name}.", dir=final_path.parent))
    try:
        yield temporary
        os.replace(temporary, final_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

