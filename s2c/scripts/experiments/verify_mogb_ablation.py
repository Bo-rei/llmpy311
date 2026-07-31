"""Verify the frozen-MiniLM MOGB ablation summary contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path

DEFAULT_DATASETS = ("clinc150", "banking77", "stackoverflow")
DEFAULT_KIRS = (0.25, 0.50, 0.75)
DEFAULT_SEEDS = (13, 42, 87, 100, 123)
EXPECTED_VARIANTS = (
    "get_085",
    "get_090",
    "get_095",
    "select_085",
    "select_095",
    "select_100",
    "min_get_10",
    "min_get_20",
    "min_select_5",
    "min_select_20",
    "default_mean_std",
    "default_mahalanobis_mean",
)
EXPECTED_UNITS = len(DEFAULT_DATASETS) * len(DEFAULT_KIRS) * len(DEFAULT_SEEDS) * len(EXPECTED_VARIANTS)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(summary_root: Path) -> dict[str, object]:
    manifest_path = summary_root / "MOGB_ABLATION_SUMMARY_MANIFEST.json"
    required = (
        "all_runs.csv",
        "baseline_reference.csv",
        "boundary_component_runs.csv",
        "boundary_component_summary.csv",
        "dataset_kir_summary.csv",
        "overall_summary.csv",
        "paired_vs_reference.csv",
        "paired_vs_single.csv",
        "significance_tests.csv",
        "known_recall_false_accept_tradeoff.csv",
        "ball_diagnostics.csv",
        "failed_or_invalid_runs.csv",
        "MOGB_ABLATION_SUMMARY_MANIFEST.json",
    )
    missing_files = [name for name in required if not (summary_root / name).is_file()]
    if missing_files:
        return {
            "status": "missing_files",
            "missing_files": missing_files,
            "summary_root": str(summary_root),
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_runs = _read_csv(summary_root / "all_runs.csv")
    baselines = _read_csv(summary_root / "baseline_reference.csv")
    boundary_runs = _read_csv(summary_root / "boundary_component_runs.csv")
    boundary_summary = _read_csv(summary_root / "boundary_component_summary.csv")
    paired_reference = _read_csv(summary_root / "paired_vs_reference.csv")
    paired_single = _read_csv(summary_root / "paired_vs_single.csv")
    significance = _read_csv(summary_root / "significance_tests.csv")
    failed_or_invalid = _read_csv(summary_root / "failed_or_invalid_runs.csv")

    seen = {
        (
            row["dataset"],
            float(row["kir"]),
            int(row["seed"]),
            row["variant"],
        )
        for row in all_runs
    }
    duplicate_runs = len(seen) != len(all_runs)
    missing_units: list[tuple[str, float, int, str]] = []
    for dataset in DEFAULT_DATASETS:
        for kir in DEFAULT_KIRS:
            for seed in DEFAULT_SEEDS:
                for variant in EXPECTED_VARIANTS:
                    key = (dataset, float(kir), int(seed), variant)
                    if key not in seen:
                        missing_units.append(key)
    invalid_rows = [
        row for row in all_runs if row["variant"] not in EXPECTED_VARIANTS or row["protocol_version"] != "protocol_v2_textoir_v1"
    ]
    baseline_methods = {row["method"] for row in baselines}
    expected_baseline_rows = len(DEFAULT_DATASETS) * len(DEFAULT_KIRS) * len(DEFAULT_SEEDS) * 3
    paired_expected = len(DEFAULT_DATASETS) * len(DEFAULT_KIRS) * len(EXPECTED_VARIANTS)
    # 6 metrics in summarize_mogb_ablation COMPARISON_METRICS
    paired_expected_rows = paired_expected * 6
    complete_contract = (
        not missing_units
        and not invalid_rows
        and not duplicate_runs
        and not failed_or_invalid
        and len(all_runs) == EXPECTED_UNITS
        and len(baselines) == expected_baseline_rows
        and baseline_methods == {"mogb_minilm", "single_centroid", "mogb_partition_ours_boundary"}
        and len(boundary_runs) == 180
        and len(boundary_summary) == 36
        and len(paired_reference) == paired_expected_rows
        and len(paired_single) == paired_expected_rows
        and len(significance) == paired_expected_rows * 2
        and int(manifest.get("completed_units", -1)) == EXPECTED_UNITS
        and int(manifest.get("missing_units", -1)) == 0
        and int(manifest.get("invalid_units", -1)) == 0
    )
    return {
        "status": "complete" if complete_contract else "failed",
        "summary_root": str(summary_root),
        "expected_units": EXPECTED_UNITS,
        "completed_units": len(all_runs),
        "missing_units": len(missing_units),
        "duplicate_runs": duplicate_runs,
        "invalid_rows": len(invalid_rows),
        "failed_or_invalid_rows": len(failed_or_invalid),
        "manifest_completed_units": int(manifest.get("completed_units", -1)),
        "manifest_missing_units": int(manifest.get("missing_units", -1)),
        "manifest_invalid_units": int(manifest.get("invalid_units", -1)),
        "baseline_rows": len(baselines),
        "expected_baseline_rows": expected_baseline_rows,
        "baseline_methods": sorted(baseline_methods),
        "boundary_component_rows": len(boundary_runs),
        "expected_boundary_component_rows": 180,
        "boundary_component_summary_rows": len(boundary_summary),
        "expected_boundary_component_summary_rows": 36,
        "paired_reference_rows": len(paired_reference),
        "paired_single_rows": len(paired_single),
        "expected_paired_rows": paired_expected_rows,
        "significance_rows": len(significance),
        "expected_significance_rows": paired_expected_rows * 2,
        "sample_missing_units": [list(item) for item in missing_units[:20]],
        "sample_invalid_rows": invalid_rows[:20],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_ablation_v1/summary"),
    )
    args = parser.parse_args(argv)
    result = verify(args.summary_root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
