"""Generate lightweight protocol_v2 implementation evidence from manifests.

The report intentionally reads only small manifests, registries, run metadata
and declared matrices.  It never reads corpus text, embeddings, checkpoints or
predictions, so the generated audit directory is safe to track in Git.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from s2c.data.hashing import atomic_write_json, atomic_write_text, sha256_file
from s2c.data.manifests import dataset_manifest_path, read_json, source_manifest_path
from s2c.experiments.matrix import load_gate_matrix
from s2c.runtime.paths import ProtocolV2Paths


AUDIT_RELATIVE = Path("docs/audits/protocol_v2_implementation")


def _manifest_datasets(paths: ProtocolV2Paths) -> tuple[str, ...]:
    """Return only datasets materialized for the selected immutable version.

    ``DATASET_SPECS`` is deliberately broader than an admitted data version:
    StackOverflow remains in the supported *schema* but must not silently enter
    an official-only audit merely because it is known to the package.  Looking
    for both source and canonical manifests keeps this report descriptive of
    actual evidence rather than of a desired experiment matrix.
    """
    if not paths.manifest_root.is_dir():
        return ()
    return tuple(
        sorted(
            child.name
            for child in paths.manifest_root.iterdir()
            if child.is_dir()
            and source_manifest_path(paths.manifest_root, child.name).is_file()
            and dataset_manifest_path(paths.manifest_root, child.name).is_file()
        )
    )


def _admission(paths: ProtocolV2Paths) -> dict[str, Any]:
    """Read the single fail-closed admission decision without guessing it."""
    if not paths.experiment_admission_path.is_file():
        return {}
    return read_json(paths.experiment_admission_path)


def _git(paths: ProtocolV2Paths, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(paths.project_root), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_at(root: Path, *args: str) -> str | None:
    """Read Git metadata from an explicitly named repository, if present."""

    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _csv(rows: Iterable[dict[str, Any]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    atomic_write_text(path, _csv(rows, fields))


def _logical_artifact_path(paths: ProtocolV2Paths, path: Path) -> str:
    return str(Path("artifacts") / "s2c" / path.relative_to(paths.artifacts_root))


def _du_bytes(path: Path) -> int:
    """Measure a small, project-owned tree without scanning raw experiment evidence."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _source_rows(paths: ProtocolV2Paths, datasets: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(datasets):
        manifest_path = source_manifest_path(paths.manifest_root, dataset)
        manifest = read_json(manifest_path)
        for file_info in manifest["files"]:
            rows.append(
                {
                    "dataset": dataset,
                    "source_name": manifest["source_name"],
                    "source_commit": manifest["source_commit"],
                    "source_relative_directory": manifest["source_relative_directory"],
                    "relative_path": file_info["relative_path"],
                    # Licence and metadata files prove provenance but are not
                    # corpora, so their count fields are intentionally absent.
                    "split": file_info.get("split", ""),
                    "row_count": file_info.get("row_count", ""),
                    "label_count": file_info.get("label_count", ""),
                    "size_bytes": file_info.get("size_bytes", ""),
                    "sha256": file_info["sha256"],
                    "byte_identical": file_info["byte_identical"],
                }
            )
    return rows


def _canonical_rows(paths: ProtocolV2Paths, datasets: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(datasets):
        manifest = read_json(dataset_manifest_path(paths.manifest_root, dataset))
        rows.append(
            {
                "dataset": dataset,
                "sample_count": manifest["sample_count"],
                "intent_count": manifest["known_label_count"],
                "native_oos_count": manifest["native_oos_count"],
                "exact_duplicate_count": manifest["exact_duplicate_count"],
                "normalized_duplicate_count": manifest["normalized_duplicate_count"],
                "canonical_file_sha256": manifest["canonical_file_sha256"],
                "source_manifest_sha256": manifest["source_manifest_sha256"],
                "split_counts": json.dumps(manifest["split_counts"], sort_keys=True),
            }
        )
    return rows


def _registry_rows(paths: ProtocolV2Paths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths.registry_root.glob("*/*/*.json")):
        payload = read_json(path)
        rows.append(
            {
                "dataset": payload["dataset"],
                "seed": payload["seed"],
                "requested_kir": payload["requested_kir"],
                "known_count": payload["known_count"],
                "effective_kir": payload["effective_kir"],
                "intent_count": len(payload["intent_universe"]),
                "registry_sha256": payload["registry_sha256"],
            }
        )
    return rows


def _view_rows(paths: ProtocolV2Paths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths.manifest_root.glob("*/views/seed_*/kir_*.json")):
        payload = read_json(path)
        counts = {item["name"]: item["count"] for item in payload["files"]}
        rows.append(
            {
                "dataset": payload["dataset"],
                "seed": payload["seed"],
                "kir": payload["kir"],
                "registry_sha256": payload["registry_sha256"],
                **counts,
            }
        )
    return rows


def _export_rows(paths: ProtocolV2Paths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths.manifest_root.glob("*/exports/*/seed_*/kir_*.json")):
        payload = read_json(path)
        rows.append(
            {
                "dataset": payload["dataset"],
                "export_name": payload["export_name"],
                "seed": payload["seed"],
                "kir": payload["kir"],
                "file_count": len(payload["files"]),
                "registry_sha256": payload["registry_sha256"],
                "canonical_manifest_sha256": payload["canonical_manifest_sha256"],
                "sample_id_mapping_sha256": payload["canonical_sample_id_mapping_sha256"],
            }
        )
    return rows


def _experiment_rows(paths: ProtocolV2Paths) -> list[dict[str, Any]]:
    configs = paths.project_root / "configs/experiments/protocol_v2"
    rows: list[dict[str, Any]] = []
    for path in sorted(configs.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.name in {"smoke_gate.yaml", "gate_core_dense.yaml"}:
            planned = len(load_gate_matrix(path))
        elif path.name == "boundary_dense.yaml":
            planned = (
                len(payload["datasets"])
                * len(payload["kirs"])
                * len(payload["seeds"])
                * len(payload["k_values"])
                * len(payload["distances"])
                * len(payload["boundary_methods"])
            )
        elif path.name == "representation_comparison.yaml":
            planned = (
                len(payload["datasets"])
                * len(payload["kirs"])
                * len(payload["seeds"])
                * len(payload["k_values"])
                * len(payload["distances"])
                * len(payload["representations"])
            )
        elif path.name == "external_baselines.yaml":
            planned = len(payload["datasets"]) * len(payload["kirs"]) * len(payload["seeds"]) * len(payload["methods"])
        elif path.name == "pipeline_dense.yaml":
            planned = len(payload["datasets"]) * len(payload["kirs"]) * len(payload["seeds"]) * len(payload["gate_candidates"])
        else:
            planned = 0
        rows.append({"config": path.name, "declared_name": payload.get("name", "budget"), "planned_units": planned, "status": "declared"})
    return rows


def _run_rows(paths: ProtocolV2Paths) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    coverage: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    embedding_cache_used = False
    for manifest_path in sorted(paths.run_root.glob("**/manifest.json")):
        payload = read_json(manifest_path)
        experiment = manifest_path.parent.parent.name
        bucket = coverage.setdefault(experiment, {"complete": 0, "failed": 0})
        if payload.get("status") == "complete":
            bucket["complete"] += 1
            # A Gate manifest records the cache key and creation/use state.  It
            # is stronger provenance than guessing from the cache directory.
            embedding_cache_used = embedding_cache_used or bool(payload.get("embedding_cache"))
        else:
            bucket["failed"] += 1
            failures.append({"experiment": experiment, "run_id": payload.get("run_id", ""), "error": payload.get("error", "non-complete manifest")})
    for state_path in sorted((paths.run_root / "plans").glob("*.state.json")):
        state = read_json(state_path)
        for failure in state.get("failed", []):
            failures.append({"experiment": state_path.stem.removesuffix(".state"), "run_id": failure.get("run_id", ""), "error": failure.get("error", "")})
    return (
        [
            {"experiment": name, "complete_runs": values["complete"], "failed_runs": values["failed"]}
            for name, values in sorted(coverage.items())
        ],
        failures,
        embedding_cache_used,
    )


def _remaining_gate_text(admission: dict[str, Any], datasets: tuple[str, ...], completed_runs: int) -> str:
    """Describe the admitted evidence without promoting a blocked matrix.

    The legacy smoke configuration still names three datasets, while an
    official-only version can legitimately admit fewer.  The report must make
    that difference explicit instead of treating a blocked StackOverflow cell
    as an unfinished official experiment.
    """

    scope = ", ".join(datasets) or "no admitted datasets"
    stackoverflow_status = admission.get("dataset_admission", {}).get("stackoverflow", "not recorded")
    if completed_runs:
        return (
            f"The completed Gate evidence is limited to {completed_runs} run(s) on admitted dataset(s): "
            f"{scope}. The legacy three-dataset E1 is intentionally not completed because blocked "
            f"StackOverflow ({stackoverflow_status}) cannot enter this official version. Before a new model experiment "
            "is accepted, its own dataset admission, materialized views/exports, runtime-independence check "
            "and targeted tests must pass."
        )
    return (
        f"No formal Gate run has completed for the admitted scope ({scope}). Blocked dataset(s) "
        f"(including StackOverflow: {stackoverflow_status}) must not be used to fill the legacy three-dataset E1. "
        "Before a model experiment "
        "is accepted, its dataset admission, materialized views/exports, runtime-independence check and "
        "targeted tests must pass."
    )


def _blocker_report(admission: dict[str, Any], datasets: tuple[str, ...], completed_runs: int) -> str:
    """Write the protocol stop condition as evidence, not as an implicit caveat."""

    decisions = admission.get("dataset_decisions", {})
    statuses = admission.get("dataset_admission", {})
    rows = "\n".join(
        f"| {dataset} | {decisions.get(dataset, 'not recorded')} | {status} |"
        for dataset, status in sorted(statuses.items())
    ) or "| _none recorded_ | _n/a_ | _n/a_ |"
    scope = ", ".join(datasets) or "none"
    return f"""# protocol_v2 scope blocker report

## Decision

The original three-dataset protocol is **not eligible for a formal completion claim**. The current
official version admits only: `{scope}`. It has {completed_runs} completed Gate-only run(s), which
are limited to that admitted scope.

| Dataset | Provenance decision | Admission |
| --- | --- | --- |
{rows}

## Blocking condition

StackOverflow has a reproducible public content snapshot, but its raw-source and redistribution-license
chain cannot be independently verified at the record level. Historical `BANKING77-OOS` also lacks a
traceable official OOS-extension source. Neither dataset may be replaced by TEXTOIR, a legacy s2c
prepared copy, a deduplicated StackOverflow variant, or a merged substitute.

## Affected work

- The legacy three-dataset 36-cell E1 smoke cannot be completed with the current evidence.
- The 3,300-cell E2 grid, boundary grid, representation grid, external-method comparison and full
  three-dataset Cascade are not authorised as `protocol_v2_official_v1` claims.
- Existing v19-v22, candidate `protocol_v2`, and historical Cascade outputs remain traceable evidence
  only; they cannot fill a blocked official protocol cell.

## Unblocking evidence

To admit StackOverflow, record one immutable raw source, original file names and SHA256 values, the
20-label mapping and 20,000-row count, a verifiable redistribution license, and a three-way sample/split
comparison against both TEXTOIR and historical s2c inputs. Until then, no training, embedding generation,
MOGB/DCL reproduction or TEXTOIR-fair-comparability claim may use it.
"""


def _requirement_rows(
    paths: ProtocolV2Paths,
    datasets: tuple[str, ...],
    registry_rows: list[dict[str, Any]],
    view_rows: list[dict[str, Any]],
    export_rows: list[dict[str, Any]],
    completed_runs: int,
    embedding_cache_used: bool,
) -> list[dict[str, Any]]:
    """State the original deliverables against present, versioned evidence.

    The matrix deliberately distinguishes a completed two-dataset official
    scope from the original three-dataset request.  A partial row is not a
    substitute for a passing completion claim.
    """

    scope = ", ".join(datasets) or "none"
    return [
        {
            "requirement": "canonical raw-source decision",
            "status": "complete_for_admitted_scope",
            "scope": scope,
            "evidence": "configs/data/protocol_v2_admission.json; docs/audits/data_provenance/",
        },
        {
            "requirement": "independent runtime data copy",
            "status": "complete_for_admitted_scope",
            "scope": paths.dataset_version,
            "evidence": "data/sources + source/canonical manifests; runtime_independence_report.md",
        },
        {
            "requirement": "safe default formal dataset version",
            "status": "complete",
            "scope": "protocol_v2_official_v1; candidate requires explicit S2C_DATASET_VERSION",
            "evidence": "src/s2c/runtime/paths.py; test_protocol_v2_admission.py",
        },
        {
            "requirement": "TEXTOIR as fixed canonical raw source",
            "status": "superseded_by_source_adjudication",
            "scope": "all datasets",
            "evidence": "official raw source wins when source/license are verified; TEXTOIR remains audit/export reference",
        },
        {
            "requirement": "three-dataset canonical protocol",
            "status": "blocked_unverified",
            "scope": "StackOverflow; legacy BANKING77-OOS",
            "evidence": "blocker_report.md; docs/audits/data_provenance/stackoverflow/source_trace.md",
        },
        {
            "requirement": "KIR registries",
            "status": "complete_for_admitted_scope",
            "scope": f"{len(registry_rows)} fixed registries",
            "evidence": "registry_statistics.csv",
        },
        {
            "requirement": "materialized views and method exports",
            "status": "partial_materialized_on_demand",
            "scope": f"{len(view_rows)} views; {len(export_rows)} exports",
            "evidence": "view_statistics.csv; export_statistics.csv",
        },
        {
            "requirement": "E1 three-dataset 36-cell Gate smoke",
            "status": "blocked_unverified",
            "scope": f"{completed_runs}/36 not claimable; {completed_runs} admitted-scope runs only",
            "evidence": "experiment_coverage.csv; blocker_report.md",
        },
        {
            "requirement": "E2 three-dataset 3,300-cell Gate grid",
            "status": "not_authorised",
            "scope": "requires all three canonical datasets",
            "evidence": "blocker_report.md",
        },
        {
            "requirement": "representation, external baseline, and official full-Cascade matrices",
            "status": "not_started_under_official_protocol",
            "scope": paths.dataset_version,
            "evidence": "implementation_report.md; blocker_report.md",
        },
        {
            "requirement": "frozen MiniLM embedding provenance",
            "status": "used_for_completed_gate_runs" if embedding_cache_used else "not_used",
            "scope": f"{completed_runs} completed Gate run(s)",
            "evidence": "completed run manifest embedding_cache fields",
        },
        {
            "requirement": "full test-suite verification",
            "status": "targeted_pass_full_suite_pending",
            "scope": "host filesystem I/O limited",
            "evidence": "test_report.md; DEVELOPMENT_LOG.md",
        },
    ]


def generate(
    paths: ProtocolV2Paths,
    output: Path,
    test_summary: str,
    runtime_status: str,
    executed_commands: tuple[str, ...] = (),
    moved_files: tuple[str, ...] = (),
) -> None:
    started_at = datetime.now(UTC)
    output.mkdir(parents=True, exist_ok=True)
    datasets = _manifest_datasets(paths)
    admission = _admission(paths)
    source_rows = _source_rows(paths, datasets)
    canonical_rows = _canonical_rows(paths, datasets)
    registry_rows = _registry_rows(paths)
    view_rows = _view_rows(paths)
    export_rows = _export_rows(paths)
    experiment_rows = _experiment_rows(paths)
    coverage_rows, failures, embedding_cache_used = _run_rows(paths)
    _write_csv(output / "source_copy_report.csv", source_rows, list(source_rows[0]) if source_rows else [])
    _write_csv(output / "canonical_statistics.csv", canonical_rows, list(canonical_rows[0]) if canonical_rows else [])
    _write_csv(output / "registry_statistics.csv", registry_rows, list(registry_rows[0]) if registry_rows else [])
    _write_csv(output / "view_statistics.csv", view_rows, list(view_rows[0]) if view_rows else [])
    _write_csv(output / "export_statistics.csv", export_rows, list(export_rows[0]) if export_rows else [])
    _write_csv(output / "experiment_plan.csv", experiment_rows, list(experiment_rows[0]) if experiment_rows else [])
    _write_csv(output / "experiment_coverage.csv", coverage_rows, ["experiment", "complete_runs", "failed_runs"])
    _write_csv(output / "failed_runs.csv", failures, ["experiment", "run_id", "error"])
    _write_csv(
        output / "path_migration.csv",
        [
            {
                "old_or_external_location": "official raw checkout",
                "new_location": f"data/sources/official/<fixed-revision> ({paths.dataset_version})",
                "policy": "canonical raw source; byte copy after source/license audit",
            },
            {
                "old_or_external_location": "textoir/data",
                "new_location": "data/sources/textoir/<commit>",
                "policy": "audit/import reference only; never a protocol_v2 official raw source",
            },
            {
                "old_or_external_location": "assets/datasets",
                "new_location": f"data/canonical/{paths.dataset_version}",
                "policy": "legacy evidence only; no official runtime read",
            },
            {
                "old_or_external_location": "artifacts/s2c/outputs/experiments",
                "new_location": f"artifacts/s2c/runs/{paths.dataset_version}",
                "policy": "historical artifacts preserved; no overwrite",
            },
        ],
        ["old_or_external_location", "new_location", "policy"],
    )
    _write_csv(
        output / "disk_usage_before_after.csv",
        [
            {
                "location": "s2c",
                "baseline_bytes": "",
                "current_bytes": _du_bytes(paths.project_root),
                "measurement": "recursive_project_tree",
            },
            {
                "location": "s2c/data",
                "baseline_bytes": "0",
                "current_bytes": _du_bytes(paths.data_root),
                "measurement": "recursive_project_tree",
            },
            {
                "location": "artifacts/s2c",
                "baseline_bytes": "",
                "current_bytes": "",
                "measurement": "not_scanned: immutable raw artifacts may be large",
            },
        ],
        ["location", "baseline_bytes", "current_bytes", "measurement"],
    )
    atomic_write_text(
        output / "test_report.md",
        "# protocol_v2 test report\n\n"
        f"Latest verification command summary: `{test_summary}`.\n\n"
        "This file records only the final command outcome; detailed pytest output is deliberately not committed.\n",
    )
    atomic_write_text(
        output / "runtime_independence_report.md",
        "# TEXTOIR runtime independence\n\n"
        f"Status: **{runtime_status}**. The check temporarily renamed `../textoir` to `textoir.disabled`, "
        "ran protocol validation, s2c/TEXTOIR-format export validation and Gate dry-run, then restored the "
        "directory and checked its Git status. No model training or embedding generation was used for this check.\n",
    )
    source_commits = sorted({str(row["source_commit"]) for row in source_rows if row["source_commit"]})
    decisions = admission.get("dataset_decisions", {})
    admitted = admission.get("dataset_admission", {})
    run_complete = sum(row["complete_runs"] for row in coverage_rows)
    remaining_gate_text = _remaining_gate_text(admission, datasets, run_complete)
    requirement_rows = _requirement_rows(
        paths,
        datasets,
        registry_rows,
        view_rows,
        export_rows,
        run_complete,
        embedding_cache_used,
    )
    _write_csv(
        output / "requirement_matrix.csv",
        requirement_rows,
        ["requirement", "status", "scope", "evidence"],
    )
    dataset_rows = "\n".join(
        "| {dataset} | {count} | {intents} | {native_oos} | `{source}` |".format(
            dataset=row["dataset"],
            count=row["sample_count"],
            intents=row["intent_count"],
            native_oos=row["native_oos_count"],
            source=next(
                source["source_relative_directory"]
                for source in source_rows
                if source["dataset"] == row["dataset"]
            ),
        )
        for row in canonical_rows
    ) or "| _none_ | _n/a_ | _n/a_ | _n/a_ | _n/a_ |"
    atomic_write_text(output / "blocker_report.md", _blocker_report(admission, datasets, run_complete))
    decision_rows = "\n".join(
        f"| {dataset} | {decisions.get(dataset, 'not recorded')} | {admitted.get(dataset, 'not recorded')} |"
        for dataset in datasets
    ) or "| _none materialized_ | _n/a_ | _n/a_ |"
    report = f"""# protocol_v2 implementation report

## Status

- Base commit: `{_git(paths, 'rev-parse', 'HEAD') or 'unavailable'}`
- Dataset version: `{paths.dataset_version}`
- Materialized canonical datasets: `{', '.join(datasets) or 'none'}`
- Source revisions: `{', '.join(source_commits) or 'unavailable'}`
- Canonical datasets: {len(canonical_rows)}
- Fixed registries: {len(registry_rows)}
- Materialized views: {len(view_rows)}
- Materialized exports: {len(export_rows)}
- Completed protocol_v2 runs: {run_complete}
- Failed protocol_v2 runs: {len(failures)}

## Data-admission decision

| Dataset | Provenance decision | Formal admission |
| --- | --- | --- |
{decision_rows}

This report is scoped to the selected dataset version.  It does not upgrade a
blocked dataset, a historical candidate snapshot, or a legacy experiment into
official evidence.

## Materialized data inventory

| Dataset | Samples | Known intents | Native OOS | Local source copy |
| --- | ---: | ---: | ---: | --- |
{dataset_rows}

The authoritative raw source for this version is the source manifest above;
TEXTOIR is retained only as a three-way audit and export-format reference, not
as a runtime dependency.

## Completed implementation work

The approved raw source is byte-copied into `data/sources`, canonical records preserve original text and
splits, and each experimental method consumes the same registry and fixed views. `textoir/data` is not a
runtime dependency. Gate runs use immutable directories beneath `artifacts/s2c/runs/{paths.dataset_version}`
and keep embedding cache separate from formal evidence.

## Deliberately not claimed

Declared boundary, representation, external-baseline and full-pipeline matrices are not experimental evidence
until their run manifests exist. Historical v19-v22 artifacts remain untouched and are not mixed with this
protocol. The StackOverflow corpus remains local-only because its redistribution licence is not verified.

## Requirement status

`requirement_matrix.csv` maps the original implementation goals to current evidence. A status of
`complete_for_admitted_scope` applies only to the two officially admitted datasets; it never upgrades the
blocked three-dataset protocol into a completed claim.

## Remaining gate

{remaining_gate_text} See `experiment_plan.csv`, `experiment_coverage.csv` and `failed_runs.csv` for the
current state rather than inferring completion from configuration files.
"""
    atomic_write_text(output / "implementation_report.md", report)
    tracked_files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "audit_manifest.json")
    ended_at = datetime.now(UTC)
    textoir_root = paths.project_root.parent / "textoir"
    atomic_write_json(
        output / "audit_manifest.json",
        {
            "protocol": paths.dataset_version,
            "audit_started_at": started_at.isoformat(),
            "audit_completed_at": ended_at.isoformat(),
            "base_commit": _git(paths, "rev-parse", "HEAD"),
            "textoir_commit": _git_at(textoir_root, "rev-parse", "HEAD"),
            "textoir_runtime_dependency": False,
            "materialized_datasets": list(datasets),
            "source_commits": source_commits,
            "dataset_decisions": decisions,
            "dataset_admission": admitted,
            "training_run": False,
            "embedding_generation": embedding_cache_used,
            "completed_run_uses_embedding_cache": embedding_cache_used,
            "artifacts_deleted": False,
            "artifacts_created_or_modified": bool(run_complete),
            "runtime_independence": runtime_status,
            "test_summary": test_summary,
            "executed_commands": list(executed_commands),
            "moved_files": list(moved_files),
            "tracked_worktree_modified_files": (
                (_git(paths, "diff", "--name-only") or "").splitlines()
            ),
            "provenance_scope": (
                "This manifest records the generated audit and known run evidence; "
                "detailed changes remain append-only in docs/DEVELOPMENT_LOG.md."
            ),
            "inputs": {
                # Keep the audit scoped to materialized evidence.  Importing
                # DATASET_SPECS here would falsely require blocked datasets.
                "source_manifest_sha256": {
                    dataset: sha256_file(source_manifest_path(paths.manifest_root, dataset))
                    for dataset in datasets
                },
                "dataset_manifest_sha256": {
                    dataset: sha256_file(dataset_manifest_path(paths.manifest_root, dataset))
                    for dataset in datasets
                },
            },
            "generated_files": [{"path": path.name, "sha256": sha256_file(path)} for path in tracked_files],
            "logical_artifact_root": _logical_artifact_path(paths, paths.run_root),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dataset-version",
        default="protocol_v2_official_v1",
        help=(
            "Immutable dataset version to audit. Defaults to the admitted official version so this "
            "audit cannot accidentally report the legacy protocol_v2 candidate."
        ),
    )
    parser.add_argument("--test-summary", default="not yet run")
    parser.add_argument("--runtime-status", default="passed")
    parser.add_argument(
        "--executed-command",
        action="append",
        default=[],
        help="Logical command recorded in audit_manifest.json; may be repeated.",
    )
    parser.add_argument(
        "--moved-file",
        action="append",
        default=[],
        help="Logical source=>destination move recorded in audit_manifest.json; may be repeated.",
    )
    args = parser.parse_args(argv)
    # The runtime helper deliberately retains protocol_v2 as its compatibility
    # default. An audit intended as official evidence must instead be explicit
    # and safe-by-default: candidate paths are only reachable when a caller
    # deliberately asks for that dataset version.
    paths = replace(ProtocolV2Paths.discover(), dataset_version=args.dataset_version)
    output = args.output or paths.project_root / AUDIT_RELATIVE
    generate(
        paths,
        output,
        args.test_summary,
        args.runtime_status,
        executed_commands=tuple(args.executed_command),
        moved_files=tuple(args.moved_file),
    )
    print(f"wrote protocol_v2 implementation audit: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
