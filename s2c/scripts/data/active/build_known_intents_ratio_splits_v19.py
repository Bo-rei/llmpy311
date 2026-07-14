#!/usr/bin/env python3
"""Generate deterministic known/unknown intent manifests for ratio sweeps.

This script creates KNOWN_INTENTS-style JSON files for a given known-intent
ratio while keeping comparability strict:
1) same CLINC intent universe,
2) deterministic seed,
3) stable domain-balanced sampling where possible.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.runtime import WorkspacePaths

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATHS = WorkspacePaths.discover(PROJECT_ROOT)


def _load_domains(domains_path: Path) -> Dict[str, List[str]]:
    with open(domains_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("domains.json must be a dict of domain -> intents")
    return {str(k): [str(x) for x in v] for k, v in payload.items()}


def _balanced_known_selection(
    domain_to_intents: Dict[str, List[str]],
    known_count: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    domains = sorted(domain_to_intents.keys())
    all_intents = [intent for domain in domains for intent in domain_to_intents[domain]]
    total = len(all_intents)
    if known_count <= 0 or known_count >= total:
        raise ValueError(f"known_count must be in (0, {total}), got {known_count}")

    ratio = known_count / total
    quotas: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []

    for domain in domains:
        size = len(domain_to_intents[domain])
        raw = size * ratio
        base = int(raw)
        quotas[domain] = min(base, size)
        remainders.append((raw - base, domain))

    assigned = sum(quotas.values())
    remaining = known_count - assigned

    if remaining > 0:
        for _, domain in sorted(remainders, key=lambda x: (-x[0], x[1])):
            if remaining <= 0:
                break
            capacity = len(domain_to_intents[domain]) - quotas[domain]
            if capacity <= 0:
                continue
            quotas[domain] += 1
            remaining -= 1

    if remaining > 0:
        for domain in domains:
            if remaining <= 0:
                break
            capacity = len(domain_to_intents[domain]) - quotas[domain]
            if capacity <= 0:
                continue
            step = min(capacity, remaining)
            quotas[domain] += step
            remaining -= step

    if sum(quotas.values()) != known_count:
        raise RuntimeError("Failed to allocate exact known intent quota")

    known: List[str] = []
    for domain in domains:
        candidates = sorted(domain_to_intents[domain])
        k = quotas[domain]
        picked = rng.sample(candidates, k) if k > 0 else []
        known.extend(picked)

    known = sorted(known)
    known_set = set(known)
    unknown = sorted([intent for intent in all_intents if intent not in known_set])
    return known, unknown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KNOWN_INTENTS manifests for strict ratio experiments"
    )
    parser.add_argument(
        "--domains_path",
        default=str(PATHS.source_data_root / "clinc150/data/domains.json"),
        help="Path to domains.json",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        help="Known intent ratio in (0, 1), e.g. 0.25 / 0.5 / 0.75",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output manifest path, e.g. data/ratios/v19/known_intents_ratio_25_seed42.json",
    )
    args = parser.parse_args()

    if not (0.0 < float(args.ratio) < 1.0):
        raise ValueError("--ratio must be in (0, 1)")

    domains = _load_domains(Path(args.domains_path))
    total_intents = sum(len(v) for v in domains.values())
    known_count = int(round(total_intents * float(args.ratio)))
    known_count = max(1, min(total_intents - 1, known_count))

    known_intents, unknown_intents = _balanced_known_selection(
        domain_to_intents=domains,
        known_count=known_count,
        seed=int(args.seed),
    )

    payload = {
        "ratio": float(args.ratio),
        "seed": int(args.seed),
        "known_intents": known_intents,
        "unknown_intents": unknown_intents,
        "known_count": len(known_intents),
        "unknown_count": len(unknown_intents),
        "intent_universe": total_intents,
        "selection_protocol": {
            "method": "domain_balanced_largest_remainder",
            "deterministic": True,
            "domains_path": str(args.domains_path),
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "ratio": float(args.ratio),
                "seed": int(args.seed),
                "known_count": len(known_intents),
                "unknown_count": len(unknown_intents),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
