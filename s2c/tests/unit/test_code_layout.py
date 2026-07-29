from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def test_source_root_has_only_active_and_legacy_namespaces() -> None:
    assert not (SRC_ROOT / "s2c").exists()
    assert (SRC_ROOT / "protocol_v2").is_dir()
    assert (SRC_ROOT / "legacy").is_dir()
    assert not (SRC_ROOT / "__init__.py").exists()


def test_python_imports_use_declared_namespaces() -> None:
    forbidden = re.compile(
        r"^(?:from|import)\s+(?:s2c|src|gate|gate_minimal|models|router|pipeline|runtime|inference|utils)(?:\.|\s)"
    )
    offenders: list[str] = []
    for root in (SRC_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "tools"):
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if forbidden.match(line.strip()):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}: {line.strip()}")
    assert offenders == []
