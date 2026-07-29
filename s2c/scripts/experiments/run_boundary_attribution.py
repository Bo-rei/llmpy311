#!/usr/bin/env python3
"""CLI for the pre-registered StackOverflow multi-center boundary attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol_v2.experiments.boundary_attribution import (
    closeout,
    freeze_provenance,
    load_config,
    run_experiment,
    verify,
    write_plan,
)
from protocol_v2.runtime.paths import ProtocolV2Paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "freeze", "run", "closeout", "verify"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    paths = ProtocolV2Paths.discover()
    config = load_config(args.config)
    if args.command == "plan":
        result: object = {"plan": str(write_plan(paths, args.config))}
    elif args.command == "freeze":
        result = freeze_provenance(paths, args.config)
    elif args.command == "run":
        result = run_experiment(paths, config)
    elif args.command == "closeout":
        result = closeout(paths, config)
    else:
        result = verify(paths)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
