"""Export lightweight, provenance-linked protocol_v2 Gate summaries for Git."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from s2c.data.hashing import atomic_write_text, sha256_file
from s2c.runtime.paths import ProtocolV2Paths


METRICS = (
    "oos_precision",
    "oos_recall",
    "oos_f1",
    "id_recall",
    "auroc",
    "aupr_oos",
    "fpr95",
    "false_accept_rate",
    "false_reject_rate",
    "scoring_seconds",
    "samples_per_second",
    "peak_rss_kib",
    "effective_cluster_count",
    "minimum_cluster_size",
)


def _csv(rows: Iterable[dict[str, Any]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    atomic_write_text(path, _csv(rows, fields))


def _float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} in {row.get('run_id', '<unknown>')}")
    return value


def _summary_path(paths: ProtocolV2Paths, name: str) -> Path:
    return paths.run_root / "summaries" / f"{name}.csv"


def _rows(paths: ProtocolV2Paths, name: str) -> list[dict[str, str]]:
    path = _summary_path(paths, name)
    if not path.is_file():
        raise FileNotFoundError(f"Protocol summary is missing: {path}")
    return list(csv.DictReader(path.open(encoding="utf-8", newline="")))


def _aggregate(rows: list[dict[str, str]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    result: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        item: dict[str, Any] = dict(zip(fields, key, strict=True))
        item["seed_count"] = len(group)
        for metric in METRICS:
            values = [_float(row, metric) for row in group]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def _failed_rows(paths: ProtocolV2Paths) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in sorted((paths.run_root / "plans").glob("*.state.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        for failure in state.get("failed", []):
            result.append(
                {
                    "experiment": path.stem.removesuffix(".state"),
                    "run_id": str(failure.get("run_id", "")),
                    "error_type": str(failure.get("error_type", "")),
                    "error": str(failure.get("error", "")),
                }
            )
    return result


def export(paths: ProtocolV2Paths, *, source_name: str, output: Path) -> None:
    rows = _rows(paths, source_name)
    if any(row.get("status") != "complete" for row in rows):
        raise ValueError("Refusing to export a summary containing incomplete runs")
    output.mkdir(parents=True, exist_ok=True)
    per_seed_fields = [
        "run_id",
        "dataset",
        "kir",
        "seed",
        "representation",
        "k_gate",
        "distance",
        "boundary",
        *METRICS,
        "run_relative_path",
        "manifest_sha256",
    ]
    _write_csv(output / "per_seed_results.csv", rows, per_seed_fields)
    method_fields = ("dataset", "kir", "representation", "k_gate", "distance", "boundary")
    aggregated = _aggregate(rows, method_fields)
    aggregate_fields = [*method_fields, "seed_count", *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std"))]
    _write_csv(output / "mean_std_by_dataset_kir_method.csv", aggregated, aggregate_fields)
    _write_csv(output / "k_sensitivity.csv", aggregated, aggregate_fields)
    kir_fields = ("dataset", "kir", "representation", "distance", "k_gate", "boundary")
    kir_rows = _aggregate(rows, kir_fields)
    _write_csv(output / "kir_curves.csv", kir_rows, [*kir_fields, "seed_count", *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std"))])
    efficiency_fields = ("dataset", "representation", "k_gate", "distance", "boundary")
    efficiency_rows = _aggregate(rows, efficiency_fields)
    _write_csv(output / "efficiency_summary.csv", efficiency_rows, [*efficiency_fields, "seed_count", *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std"))])
    _write_csv(output / "boundary_comparison.csv", [], [*method_fields, "seed_count", *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std"))])
    failures = _failed_rows(paths)
    _write_csv(output / "failed_runs.csv", failures, ["experiment", "run_id", "error_type", "error"])
    source = _summary_path(paths, source_name)
    coverage = [{"source_summary": f"artifacts/s2c/runs/protocol_v2/summaries/{source.name}", "source_sha256": sha256_file(source), "complete_rows": len(rows), "failed_rows": len(failures)}]
    _write_csv(output / "experiment_coverage.csv", coverage, list(coverage[0]))
    atomic_write_text(
        output / "README.md",
        "# protocol_v2 Gate results\n\n"
        "This directory contains only lightweight aggregates derived from immutable Gate run manifests and "
        "metrics. It does not contain text, embeddings, checkpoints or predictions. `MANIFEST.csv` records the "
        "source summary SHA256. Gate-only results must not be interpreted as full Cascade results.\n",
    )
    manifest_rows = []
    for path in sorted(item for item in output.iterdir() if item.is_file() and item.name != "MANIFEST.csv"):
        manifest_rows.append(
            {
                "public_relative_path": path.name,
                "source_summary": f"artifacts/s2c/runs/protocol_v2/summaries/{source.name}",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _write_csv(output / "MANIFEST.csv", manifest_rows, list(manifest_rows[0]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="gate_core_dense", help="Experiment summary basename without .csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    output = args.output or paths.results_root / "protocol_v2" / "gate_core"
    export(paths, source_name=args.source, output=output)
    print(f"wrote lightweight protocol_v2 results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
