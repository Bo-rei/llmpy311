#!/usr/bin/env python3
"""Dry-run entry point for the split--merge adaptive-K prototype."""

from __future__ import annotations

import argparse
import json

import numpy as np

from protocol_v2.experiments.adaptive_split_merge import AdaptiveSplitMergeConfig, fit_split_merge


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Use a synthetic Known-only fixture (the only supported mode)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("only --dry-run is enabled; no full adaptive-K experiment is registered")
    rng = np.random.default_rng(args.seed)
    fixture = np.concatenate(
        [rng.normal(loc=(-2.0, 0.0), scale=0.15, size=(12, 2)), rng.normal(loc=(2.0, 0.0), scale=0.15, size=(12, 2))]
    )
    result = fit_split_merge(
        fixture,
        seed=args.seed,
        config=AdaptiveSplitMergeConfig(
            tau_compact=0.10,
            n_min=5,
            tau_stability=0.60,
            epsilon=0.02,
            complexity_penalty=0.01,
            max_k=2,
            bootstrap_repeats=3,
        ),
        # A real Gate caller must measure this on Known calibration data.  The
        # fixture supplies a zero increase only to exercise the dry-run path.
        cross_intent_acceptance_increase=0.0,
    )
    print(
        json.dumps(
            {
                "stage": "adaptive_split_merge_prototype",
                "data": "synthetic_known_only_fixture",
                "seed": args.seed,
                "cluster_count": int(len(np.unique(result.labels))),
                "decisions": [
                    {
                        "accepted": decision.accepted,
                        "reasons": list(decision.reasons),
                        "metrics": decision.metrics.__dict__,
                    }
                    for decision in result.decisions
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
