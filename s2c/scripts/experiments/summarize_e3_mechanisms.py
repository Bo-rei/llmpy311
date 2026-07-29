"""Summarize E3 partition, stability, coverage and reliability outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol_v2.experiments.mechanism_summary import summarize_e3
from protocol_v2.runtime.paths import ProtocolV2Paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition-config",
        type=Path,
        default=Path("configs/experiments/protocol_v2_textoir_v1/e3_partition_control.yaml"),
    )
    parser.add_argument(
        "--diagnostic-config",
        type=Path,
        default=Path("configs/experiments/protocol_v2_textoir_v1/e3_cluster_diagnostics.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(summarize_e3(ProtocolV2Paths.discover(), args.partition_config, args.diagnostic_config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

