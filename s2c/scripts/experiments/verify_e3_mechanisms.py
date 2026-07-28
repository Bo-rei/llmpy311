"""Verify E3 plans, provenance and immutable E2 references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from s2c.experiments.mechanism_verify import verify_e3
from s2c.runtime.paths import ProtocolV2Paths


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
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--check-equivalence", action="store_true")
    args = parser.parse_args()
    result = verify_e3(
        ProtocolV2Paths.discover(),
        args.partition_config,
        args.diagnostic_config,
        require_complete=args.require_complete,
    )
    if args.check_equivalence:
        from s2c.experiments.mechanism_verify import verify_kmeans_e2_equivalence

        result["kmeans_e2_equivalence"] = verify_kmeans_e2_equivalence(ProtocolV2Paths.discover())
    print(
        json.dumps(
            result,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
