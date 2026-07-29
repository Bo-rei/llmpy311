"""Print or write a deterministic protocol_v2 Gate sweep plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from protocol_v2.runtime.paths import ProtocolV2Paths

from .matrix import load_gate_matrix
from .registry import write_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/protocol_v2/smoke_gate.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    specs = load_gate_matrix(args.config)
    output = args.output or (ProtocolV2Paths.discover().run_root / "plans" / f"{args.config.stem}.json")
    write_plan(output, specs)
    print(f"planned {len(specs)} runs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

