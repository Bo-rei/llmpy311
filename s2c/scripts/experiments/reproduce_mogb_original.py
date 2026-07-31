"""Preflight the pinned upstream MOGB checkout without fabricating results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = ("MOGB.py", "pretrain.py", "cluster.py", "cluster2.py", "cluster3.py", "gb_test.py", "model.py", "loss.py", "myloss.py", "dataloader.py", "init_parameter.py", "util.py", "run.sh", "requirements.txt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("third_party/mogb_official"))
    parser.add_argument("--output", type=Path, default=Path("../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_baseline_v1/official_preflight.json"))
    args = parser.parse_args(argv)
    source = args.source.resolve()
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    imports_utils = "from utils" in (source / "MOGB.py").read_text(encoding="utf-8") or any("from utils" in (source / name).read_text(encoding="utf-8") for name in REQUIRED if (source / name).is_file())
    payload = {
        "status": "blocked_legacy_contract",
        "source": str(source),
        "required_files_missing": missing,
        "missing_utils_package": not (source / "utils").is_dir() and imports_utils,
        "old_dependency_contract": "pytorch-pretrained-bert/PyTorch-1.7 pinned in requirements.txt",
        "hardcoded_cuda": True,
        "data_contract": "legacy TextOIR-style files; not protocol_v2 registries",
        "test_used_for_selection": False,
        "reason": "Do not emit an official reproduction number until the legacy source contract is made runnable in an isolated environment.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
