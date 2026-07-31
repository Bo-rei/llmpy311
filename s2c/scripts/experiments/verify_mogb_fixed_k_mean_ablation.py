"""Verify the fixed-K MOGB mean-radius ablation closeout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from protocol_v2.data.hashing import sha256_file, sha256_json
from protocol_v2.runtime.paths import ProtocolV2Paths

DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
NEW_K_VALUES = (1, 3, 4)
REFERENCE_K = 2
FIXED_METHOD = "ours_partition_mogb_boundary"
ADAPTIVE_METHOD = "mogb_minilm"
EXPECTED_PROTOCOL = "protocol_v2_textoir_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_metrics(metrics: dict[str, Any]) -> list[str]:
    invalid: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            invalid.append(key)
    return invalid


def _run_dir(root: Path, dataset: str, kir: float, seed: int, k: int) -> Path:
    return root / f"fixed_k{k}" / dataset / f"kir_{kir:.2f}" / f"seed_{seed}"


def _baseline_dir(root: Path, dataset: str, kir: float, seed: int, method: str) -> Path:
    return root / dataset / f"kir_{kir:.2f}" / f"seed_{seed}" / method


def _validate_run(
    run_dir: Path,
    *,
    dataset: str,
    kir: float,
    seed: int,
    expected_method: str,
    expected_protocol: str = EXPECTED_PROTOCOL,
    expected_k: int | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    required = ("manifest.json", "config.json", "inputs.json", "metrics.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        return None, [f"{run_dir}|missing_files={','.join(missing)}"]

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    inputs = json.loads((run_dir / "inputs.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    problems: list[str] = []
    if manifest.get("status") != "complete":
        problems.append(f"{run_dir}|manifest_status={manifest.get('status')}")
    if manifest.get("test_used_for_selection") is not False:
        problems.append(f"{run_dir}|manifest_test_used_for_selection={manifest.get('test_used_for_selection')}")
    if config.get("protocol_version") != expected_protocol:
        problems.append(f"{run_dir}|protocol_version={config.get('protocol_version')}")
    if config.get("dataset") != dataset or _safe_float(config.get("kir")) != float(kir) or int(config.get("seed", -1)) != int(seed):
        problems.append(f"{run_dir}|config_identity_mismatch")
    if "method" in config and config.get("method") != expected_method:
        problems.append(f"{run_dir}|config_method={config.get('method')}")
    if config.get("test_used_for_selection") is not False:
        problems.append(f"{run_dir}|config_test_used_for_selection={config.get('test_used_for_selection')}")
    config_k = config.get("k")
    if expected_k is not None and config_k is not None and int(config_k) != int(expected_k):
        problems.append(f"{run_dir}|config_k={config.get('k')}")
    if manifest.get("config_hash") != sha256_json(config):
        problems.append(f"{run_dir}|config_hash_mismatch")
    if manifest.get("input_hashes") != inputs:
        problems.append(f"{run_dir}|input_hashes_mismatch")
    invalid_metrics = _finite_metrics(metrics)
    if invalid_metrics:
        problems.append(f"{run_dir}|non_finite_metrics={','.join(sorted(invalid_metrics))}")

    if problems:
        return None, problems

    row = {
        "dataset": dataset,
        "kir": float(kir),
        "seed": int(seed),
        "method": str(config.get("method", expected_method)),
        "run_dir": str(run_dir),
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "config_hash": str(manifest["config_hash"]),
        "inputs_sha256": sha256_json(inputs),
    }
    if expected_k is not None:
        row["k"] = int(expected_k)
    return row, []


def _find_summary_manifest(summary_root: Path) -> Path | None:
    if not summary_root.is_dir():
        return None
    candidates = sorted(
        [
            *summary_root.glob("*SUMMARY_MANIFEST.json"),
            *summary_root.glob("*summary_manifest.json"),
            *summary_root.glob("*manifest*.json"),
        ]
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and "manifest_hashes" in payload:
            return path
    return None


def _validate_summary(summary_root: Path, expected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_manifest_path = _find_summary_manifest(summary_root)
    all_runs_path = summary_root / "all_runs.csv"
    result: dict[str, Any] = {
        "summary_present": True,
        "summary_root": str(summary_root),
        "summary_rows": 0,
        "summary_unique_cells": 0,
        "summary_manifest_path": str(summary_manifest_path) if summary_manifest_path else None,
        "summary_manifest_hashes_ok": False,
        "summary_manifest_rows": 0,
    }
    if not all_runs_path.is_file():
        result["summary_error"] = "missing all_runs.csv"
        return result

    rows = _read_csv(all_runs_path)
    cells = {
        (
            row.get("dataset"),
            row.get("kir"),
            row.get("seed"),
            row.get("k"),
            row.get("k_neighbors"),
            row.get("method"),
            row.get("variant"),
        )
        for row in rows
    }
    result["summary_rows"] = len(rows)
    result["summary_unique_cells"] = len(cells)

    expected_count = len(expected_rows)
    if len(rows) != expected_count or len(cells) != expected_count:
        result["summary_error"] = f"row_count_mismatch:{len(rows)}:{len(cells)}:{expected_count}"
        return result

    if summary_manifest_path is None:
        result["summary_error"] = "missing summary manifest"
        return result

    summary_manifest = json.loads(summary_manifest_path.read_text(encoding="utf-8"))
    manifest_hashes = summary_manifest.get("manifest_hashes")
    if not isinstance(manifest_hashes, list):
        result["summary_error"] = "summary manifest missing manifest_hashes"
        return result

    expected_hashes = {entry["manifest_sha256"] for entry in expected_rows}
    actual_hashes = set()
    for entry in manifest_hashes:
        if not isinstance(entry, dict):
            result["summary_error"] = "summary manifest hash entry malformed"
            return result
        actual_hashes.add(str(entry.get("manifest_sha256")))
    result["summary_manifest_rows"] = len(manifest_hashes)
    result["summary_manifest_hashes_ok"] = actual_hashes == expected_hashes and len(manifest_hashes) == expected_count
    if not result["summary_manifest_hashes_ok"]:
        result["summary_error"] = "summary manifest hashes mismatch"
    return result


def verify(run_root: Path, *, summary_root: Path | None = None) -> dict[str, Any]:
    fixed_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []

    baseline_root = run_root.parent / "mogb_baseline_v1"

    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                for k in NEW_K_VALUES:
                    row, problems = _validate_run(
                        _run_dir(run_root, dataset, kir, seed, k),
                        dataset=dataset,
                        kir=kir,
                        seed=seed,
                        expected_method=FIXED_METHOD,
                        expected_k=k,
                    )
                    if row is None:
                        missing.extend(problems)
                        continue
                    fixed_rows.append(row)

                row, problems = _validate_run(
                    _baseline_dir(baseline_root, dataset, kir, seed, FIXED_METHOD),
                    dataset=dataset,
                    kir=kir,
                    seed=seed,
                    expected_method=FIXED_METHOD,
                    expected_k=REFERENCE_K,
                )
                if row is None:
                    missing.extend(problems)
                else:
                    fixed_rows.append(row)

                row, problems = _validate_run(
                    _baseline_dir(baseline_root, dataset, kir, seed, ADAPTIVE_METHOD),
                    dataset=dataset,
                    kir=kir,
                    seed=seed,
                    expected_method=ADAPTIVE_METHOD,
                )
                if row is None:
                    missing.extend(problems)
                else:
                    adaptive_rows.append(row)

    fixed_keys = {(row["dataset"], row["kir"], row["seed"], row.get("k")) for row in fixed_rows}
    adaptive_keys = {(row["dataset"], row["kir"], row["seed"]) for row in adaptive_rows}
    fixed_k_counts = Counter(int(row["k"]) for row in fixed_rows if "k" in row)
    adaptive_methods = sorted({row["method"] for row in adaptive_rows})
    input_groups: dict[tuple[str, float, int], set[str]] = {}
    for row in [*fixed_rows, *adaptive_rows]:
        key = (str(row["dataset"]), float(row["kir"]), int(row["seed"]))
        input_groups.setdefault(key, set()).add(str(row["inputs_sha256"]))
    input_mismatch_groups = sorted(
        ["|".join([dataset, f"{kir:.2f}", str(seed)]) for (dataset, kir, seed), hashes in input_groups.items() if len(hashes) > 1]
    )
    fixed_expected = len(DATASETS) * len(KIRS) * len(SEEDS) * len(NEW_K_VALUES)
    fixed_reference_expected = len(DATASETS) * len(KIRS) * len(SEEDS)
    adaptive_expected = len(DATASETS) * len(KIRS) * len(SEEDS)
    fixed_contract_expected = fixed_expected + fixed_reference_expected

    fixed_complete = len(fixed_rows) == fixed_contract_expected and len(fixed_keys) == fixed_contract_expected
    adaptive_complete = len(adaptive_rows) == adaptive_expected and len(adaptive_keys) == adaptive_expected
    inputs_complete = len(input_mismatch_groups) == 0

    result: dict[str, Any] = {
        "status": "complete" if fixed_complete and adaptive_complete and inputs_complete and not missing and not invalid else "failed",
        "run_root": str(run_root),
        "fixed_new_units_expected": fixed_expected,
        "fixed_new_units_actual": sum(1 for row in fixed_rows if int(row.get("k", REFERENCE_K)) in NEW_K_VALUES),
        "fixed_reference_units_expected": fixed_reference_expected,
        "fixed_reference_units_actual": sum(1 for row in fixed_rows if int(row.get("k", REFERENCE_K)) == REFERENCE_K),
        "fixed_contract_units_expected": fixed_contract_expected,
        "fixed_contract_units_actual": len(fixed_rows),
        "fixed_unique_cells": len(fixed_keys),
        "fixed_k_counts": {str(k): fixed_k_counts.get(k, 0) for k in (1, 2, 3, 4)},
        "adaptive_reference_units_expected": adaptive_expected,
        "adaptive_reference_units_actual": len(adaptive_rows),
        "adaptive_unique_cells": len(adaptive_keys),
        "adaptive_methods": adaptive_methods,
        "input_mismatch_groups": input_mismatch_groups,
        "missing_units": len(missing),
        "invalid_units": len(invalid),
        "sample_missing_units": missing[:20],
        "sample_invalid_units": invalid[:20],
        "test_used_for_selection": False,
        "summary_checked": False,
    }

    summary_candidate = summary_root or (run_root / "summary")
    summary_result: dict[str, Any] | None = None
    if summary_candidate.is_dir():
        summary_result = _validate_summary(summary_candidate, fixed_rows)
        result["summary_checked"] = True
        result.update(summary_result)
    else:
        result["summary_present"] = False

    if summary_result and summary_result.get("summary_error"):
        invalid.append(str(summary_result["summary_error"]))
        result["invalid_units"] = len(invalid)
        result["sample_invalid_units"] = invalid[:20]
        result["status"] = "failed"

    if input_mismatch_groups:
        invalid.extend(f"inputs_mismatch|{group}" for group in input_mismatch_groups)
        result["invalid_units"] = len(invalid)
        result["sample_invalid_units"] = invalid[:20]
        result["status"] = "failed"

    if missing:
        result["status"] = "failed"
    if invalid:
        result["status"] = "failed"
    if result["status"] != "complete":
        result["summary_manifest_hashes_ok"] = bool(summary_result and summary_result.get("summary_manifest_hashes_ok", False))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Override the fixed-K ablation run root.",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=None,
        help="Override the summary root if a closeout summary exists.",
    )
    args = parser.parse_args(argv)

    paths = ProtocolV2Paths.discover()
    run_root = (args.run_root or (paths.run_root / "mogb_fixed_k_mean_ablation_v1")).resolve()
    summary_root = args.summary_root.resolve() if args.summary_root is not None else None

    try:
        result = verify(run_root, summary_root=summary_root)
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        result = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "run_root": str(run_root),
            "summary_checked": False,
            "test_used_for_selection": False,
        }
        print(json.dumps(result, sort_keys=True))
        return 2

    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
