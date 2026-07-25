"""Summarize completed protocol_v2 Gate runs without touching raw artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from s2c.data.hashing import sha256_file, sha256_json
from s2c.data.manifests import read_json
from s2c.runtime.paths import ProtocolV2Paths

from .matrix import GateRunSpec, filter_gate_specs, load_gate_matrix
from .registry import write_plan
from .resume import completed_run


SUMMARY_FIELDS = (
    "run_id",
    "protocol_version",
    "dataset",
    "kir",
    "seed",
    "representation",
    "k_gate",
    "distance",
    "boundary",
    "status",
    "oos_precision",
    "oos_recall",
    "oos_f1",
    "id_recall",
    "auroc",
    "aupr_oos",
    "fpr95",
    "false_accept_rate",
    "false_reject_rate",
    "effective_cluster_count",
    "minimum_cluster_size",
    "scoring_seconds",
    "samples_per_second",
    "peak_rss_kib",
    "run_relative_path",
    "manifest_sha256",
)


def _run_dir(paths: ProtocolV2Paths, spec: GateRunSpec) -> Path:
    return paths.run_root / spec.experiment_name / spec.run_id


def _logical_run_path(paths: ProtocolV2Paths, run_dir: Path) -> str:
    """Return a portable artifact reference even when roots are overridden."""
    return str(Path("artifacts") / "s2c" / run_dir.relative_to(paths.artifacts_root))


def _row(paths: ProtocolV2Paths, spec: GateRunSpec) -> dict[str, Any]:
    run_dir = _run_dir(paths, spec)
    base: dict[str, Any] = {
        "run_id": spec.run_id,
        "protocol_version": spec.protocol_version,
        "dataset": spec.dataset,
        "kir": f"{spec.kir:.2f}",
        "seed": spec.seed,
        "representation": spec.representation,
        "k_gate": spec.k_gate,
        "distance": spec.distance,
        "boundary": spec.boundary,
        "status": "missing",
        "run_relative_path": _logical_run_path(paths, run_dir),
        "manifest_sha256": "",
    }
    config = {
        "protocol_version": spec.protocol_version,
        "dataset": spec.dataset,
        "kir": spec.kir,
        "seed": spec.seed,
        "representation": spec.representation,
        "k_gate": spec.k_gate,
        "distance": spec.distance,
        "boundary": spec.boundary,
        "radius_lambda": spec.radius_lambda,
        "encoder_name": spec.encoder_name,
        "encoder_device": spec.encoder_device,
        "oos_positive": True,
        "score_direction": "higher_is_more_oos",
        "selection": "fixed_boundary_known_only_calibration",
    }
    manifest_path = run_dir / "manifest.json"
    if not completed_run(run_dir, sha256_json(config)):
        return base
    metrics = read_json(run_dir / "metrics.json").get("combined", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Malformed metrics in {run_dir}")
    # Keep declared configuration fields from ``base``.  Only metric keys that
    # actually exist in metrics.json may enrich the row; otherwise an absent
    # metric must not erase dataset/KIR/run provenance with an empty value.
    for field in SUMMARY_FIELDS:
        if field in metrics:
            base[field] = metrics[field]
    base["status"] = "complete"
    base["manifest_sha256"] = sha256_file(manifest_path)
    return base


def summarize(paths: ProtocolV2Paths, specs: list[GateRunSpec], output: Path) -> dict[str, int]:
    rows = [_row(paths, spec) for spec in specs]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return {"planned": len(rows), "complete": sum(row["status"] == "complete" for row in rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--kir", type=float, action="append")
    parser.add_argument("--shard-name")
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    specs = filter_gate_specs(
        load_gate_matrix(args.config), datasets=args.dataset, seeds=args.seed, kirs=args.kir
    )
    suffix = f".{args.shard_name}" if args.shard_name else ""
    plan_path = paths.run_root / "plans" / f"{args.config.stem}{suffix}.json"
    write_plan(plan_path, specs)
    output = args.output or (paths.run_root / "summaries" / f"{args.config.stem}.csv")
    summary = summarize(paths, specs, output)
    print(f"summarized {summary['complete']}/{summary['planned']} runs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
