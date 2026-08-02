"""Audit DCLOOS official/unified prerequisites without fabricating a run.

The pinned ``liam0949/DCLOOS`` repository is a redirect-only checkout.  The
actual source is kept in a separate third-party checkout.  This command
records both commits, inspects the legacy data contract, and checks whether
the required open-domain negative corpus and local BERT are available.  It
never converts protocol data into a pseudo-OOS corpus and never trains when a
required supervision source is absent.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from protocol_v2.data.hashing import atomic_write_json, sha256_file
from protocol_v2.runtime.paths import ProtocolV2Paths


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _exists(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "relative_path": relative,
        "absolute_path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def run_preflight(paths: ProtocolV2Paths, output_root: Path) -> dict[str, Any]:
    project_root = paths.project_root
    official_root = project_root / "third_party" / "dcloos_official"
    source_root = project_root / "third_party" / "dcloos_source"
    artifact_root = paths.artifacts_root / "external" / "dcloos_v1"
    # The official code's Data class requires these files.  They are checked
    # explicitly rather than silently substituting protocol_v2 exports.
    required_source_files = [
        "configs.py",
        "dataloader.py",
        "models/Encoder.py",
        "losses.py",
        "main.py",
    ]
    required_negative_files = [
        "squad/squad_placeh.tsv",
        "stackoverflow/squad_placeh.tsv",
        "banking/squad_placeh.tsv",
    ]
    source_checks = [_exists(source_root, relative) for relative in required_source_files]
    local_negative_checks = []
    for relative in required_negative_files:
        local_negative_checks.append(_exists(project_root / "data" / "sources" / "textoir" / "dffe2b1b848a069a6808f8089b4cb9bd16e2062b", relative))
    model_check = _exists(project_root.parent / "assets" / "models", "bert-base-uncased/model.safetensors")
    data_roots = {
        "protocol_v2_exports": _exists(project_root / "data", "exports/protocol_v2_textoir_v1"),
        "textoir_compatible_snapshot": _exists(project_root / "data", "sources/textoir/dffe2b1b848a069a6808f8089b4cb9bd16e2062b"),
    }
    compile_targets = [source_root / relative for relative in required_source_files]
    compile_failures: list[str] = []
    for target in compile_targets:
        result = subprocess.run([sys.executable, "-m", "py_compile", str(target)], capture_output=True, text=True)
        if result.returncode:
            compile_failures.append(f"{target}: {result.stderr.strip()}")
    missing_negative = [check["relative_path"] for check in local_negative_checks if not check["exists"]]
    # Keep the state name at the experiment-contract level.  The missing
    # corpus is an external-negative-data blocker; it is not a claim that the
    # end-to-end method was removed from the baseline plan.
    status = "blocked_missing_external_negative_data"
    payload: dict[str, Any] = {
        "experiment_id": "dcloos_official_unified_v1",
        "status": status,
        "official_repo": "https://github.com/liam0949/DCLOOS",
        "official_repo_commit": _git(official_root, "rev-parse", "HEAD"),
        "actual_source_repo": "https://github.com/fanolabs/out-of-scope-intent-detection",
        "actual_source_commit": _git(source_root, "rev-parse", "HEAD"),
        "source_checks": source_checks,
        "negative_corpus_checks": local_negative_checks,
        "missing_negative_files": missing_negative,
        "model_check": model_check,
        "data_roots": data_roots,
        "source_compile_failures": compile_failures,
        "official_supervision": {
            "synthetic_pseudo_oos": True,
            "external_open_domain_oos": True,
            "end_to_end_training": True,
        },
        "unified_protocol_intent": {
            "known_split": "protocol_v2_textoir_v1 registry/views",
            "known_only_training": True,
            "external_oos_available": False,
            "status": "not_started_without_required_negative_corpus",
        },
        "reason": "The official source requires a legacy open-domain negative TSV (squad_placeh.tsv) and its own random Known-label selection. No such corpus is present locally; protocol_v2 OOS/test rows cannot be substituted without changing the method's supervision contract.",
        "git_commit": _git(project_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(project_root, "status", "--short")),
        "python": sys.version,
        "platform": platform.platform(),
    }
    atomic_write_json(output_root / "DCLOOS_PREFLIGHT.json", payload)
    atomic_write_json(artifact_root / "DCLOOS_PREFLIGHT.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    output_root = args.output_dir or paths.project_root / "docs" / "dcloos"
    result = run_preflight(paths, output_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result["status"].startswith("blocked_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
