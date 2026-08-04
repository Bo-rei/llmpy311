#!/usr/bin/env python3
"""Freeze the RC-AMBL pilot input and historical-artifact inventory."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file
from protocol_v2.runtime.paths import ProtocolV2Paths


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    paths = ProtocolV2Paths.discover()
    root = paths.run_root / "adaptive_v1"
    plans = root / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    tracked = {}
    for pattern in (
        "data/manifests/protocol_v2_textoir_v1/stackoverflow/**",
        "data/registries/protocol_v2_textoir_v1/stackoverflow/**",
        "data/views/protocol_v2_textoir_v1/stackoverflow/**",
        "data/exports/protocol_v2_textoir_v1/s2c/stackoverflow/**",
        "results/diagnostics/urcsg/**",
        "results/diagnostics/ccsg/**",
        "results/diagnostics/mogb_diff/**",
        "results/diagnostics/lambda_sensitivity/**",
    ):
        for path in paths.project_root.glob(pattern):
            if path.is_file():
                tracked[str(path.relative_to(paths.project_root))] = sha256_file(path)
    payload = {
        "experiment_id": "adaptive_v1",
        "protocol_version": paths.dataset_version,
        "scope": {"dataset": "stackoverflow", "kir": 0.50, "seeds": [13, 42, 87], "representation": "frozen_minilm", "distance": "mahalanobis_diag", "radius_method": "mean_std", "radius_lambda": 1.0},
        "git": {"commit": git(paths.project_root, "rev-parse", "HEAD"), "branch": git(paths.project_root, "branch", "--show-current"), "status": git(paths.project_root, "status", "--short"), "third_party_mogb_status": git(paths.project_root / "third_party" / "mogb_official", "status", "--short")},
        "python": platform.python_version(),
        "inputs": tracked,
        "history": {"reused": ["E2 K=1/K=2", "E3 random-balanced", "BRAK if exact row exists", "URCSG primary", "CCSG K=1/K=2", "MOGB MiniLM-compatible fair row"], "not_repeated": ["E0", "E1", "E2", "E3", "URCSG", "CCSG", "MOGB fair matrix"]},
        "selection_contract": {"calibration_select": True, "calibration_threshold": True, "test_used_for_selection": False, "proxy_oos_used_for_threshold": False, "runtime_textoir_dependency": False},
    }
    atomic_write_json(plans / "preflight.json", payload)
    lines = [
        "# RC-AMBL adaptive_v1 预检",
        "",
        f"- protocol: `{paths.dataset_version}`",
        "- dataset/KIR: StackOverflow / 0.50",
        "- seeds: 13, 42, 87",
        f"- git commit: `{payload['git']['commit']}`",
        f"- git status: `{payload['git']['status'] or 'clean'}`",
        f"- third_party/mogb_official: `{payload['git']['third_party_mogb_status'] or 'clean'}`（只读，不修改）",
        "- cache policy: 只复用既有 Frozen MiniLM cache，cache miss 失败",
        "- selection: calibration_select；threshold: calibration_threshold；test_used_for_selection=false",
        "- 历史 E2/E3/URCSG/CCSG/BRAK/MOGB 结果只验证 hash 后复用，不覆盖、不重跑",
        "",
        "## 输入文件哈希",
        "",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in sorted(tracked.items()))
    atomic_write_text(plans / "PREFLIGHT.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "path": str(plans / "preflight.json"), "tracked_files": len(tracked)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
