"""Print the declared E3-A and E3-B/C plan sizes without running experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocol_v2.experiments.mechanism_runner import diagnostic_groups, partition_control_specs


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
    partition_specs = partition_control_specs(args.partition_config)
    diagnostic_specs = diagnostic_groups(args.diagnostic_config)
    print(
        json.dumps(
            {
                "stage": "E3",
                "partition_control_units": len(partition_specs),
                "diagnostic_groups": len(diagnostic_specs),
                "diagnostic_runs_per_group": 2 * 10 * 2,
                "e2_k1_reference": True,
                "runs_started": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

