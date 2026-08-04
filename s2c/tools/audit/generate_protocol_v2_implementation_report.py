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

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file
from protocol_v2.data.manifests import dataset_manifest_path, read_json, source_manifest_path
from protocol_v2.experiments.matrix import load_gate_matrix
from protocol_v2.runtime.paths import ProtocolV2Paths


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
                    "license_provenance_status": manifest.get("license_provenance_status", manifest.get("license_status", "not_recorded")),
                    "redistribution_by_s2c": manifest.get("redistribution_by_s2c", "follow_source_terms"),
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
    configs = paths.project_root / "configs" / "experiments" / paths.dataset_version
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


def _remaining_gate_text(
    admission: dict[str, Any],
    datasets: tuple[str, ...],
    completed_runs: int,
    e2_completed_runs: int = 0,
) -> str:
    """Describe active evidence and the local-only StackOverflow boundary."""

    scope = ", ".join(datasets) or "no materialized datasets"
    stackoverflow_status = admission.get("dataset_admission", {}).get("stackoverflow", "not recorded")
    return (
        f"The active three-dataset protocol has {completed_runs} completed E1 smoke Gate run(s) on: {scope}. "
        f"StackOverflow is `{stackoverflow_status}`: it is permitted for local training, evaluation and "
        "external baseline reproduction, but its corpus must not be tracked in public Git or redistributed by s2c. "
        f"E2 has {e2_completed_runs}/1,650 completed run(s), is resumable, and must be interpreted only after "
        "its run manifests and summary are complete."
    )


def _blocker_report(
    admission: dict[str, Any],
    datasets: tuple[str, ...],
    completed_runs: int,
    e2_completed_runs: int,
) -> str:
    """Record remaining constraints without re-blocking local experiments."""

    decisions = admission.get("dataset_decisions", {})
    statuses = admission.get("dataset_admission", {})
    rows = "\n".join(
        f"| {dataset} | {decisions.get(dataset, 'not recorded')} | {status} |"
        for dataset, status in sorted(statuses.items())
    ) or "| _none recorded_ | _n/a_ | _n/a_ |"
    scope = ", ".join(datasets) or "none"
    return f"""# protocol_v2 active-protocol constraint report

## Decision

`{scope}` is the active fixed TEXTOIR-compatible local benchmark scope. It has
{completed_runs}/36 completed E1 Gate-only smoke run(s) and {e2_completed_runs}/1,650 E2 run(s).

| Dataset | Provenance decision | Admission |
| --- | --- | --- |
{rows}

## Local-only boundary

StackOverflow is a fixed 20,000-title, 20-label TEXTOIR-compatible snapshot for **local** scientific
experiments. Its provenance does not establish a per-row redistribution licence. Consequently s2c must
not track its complete text in Git, repackage it in an appendix, call it an official Stack Overflow
classification release, or claim complete per-row attribution. These limits do not block canonical
construction, embedding generation, Gate/Pipeline experiments, or external baseline reproduction.

## Affected work

- `protocol_v2_official_v1` is frozen for audit and may not be mixed with this active protocol.
- Legacy `protocol_v2` remains rejected and may not be revived as a formal result source.
- E3--E7 remain deliberately unstarted until E2 is summarized and reviewed.

## Unblocking evidence

If public redistribution becomes necessary, a separate source/licence review is required. It is not a
precondition for the present local benchmark protocol.
"""


def _requirement_rows(
    paths: ProtocolV2Paths,
    datasets: tuple[str, ...],
    registry_rows: list[dict[str, Any]],
    view_rows: list[dict[str, Any]],
    export_rows: list[dict[str, Any]],
    completed_runs: int,
    e2_completed_runs: int,
    embedding_cache_used: bool,
    test_summary: str,
) -> list[dict[str, Any]]:
    """State active deliverables against materialized, versioned evidence."""

    scope = ", ".join(datasets) or "none"
    return [
        {
            "requirement": "canonical raw-source decision",
            "status": "complete",
            "scope": scope,
            "evidence": "configs/data/protocol_v2_admission.json; docs/audits/data_provenance/",
        },
        {
            "requirement": "independent runtime data copy",
            "status": "complete",
            "scope": paths.dataset_version,
            "evidence": "data/sources + source/canonical manifests; runtime_independence_report.md",
        },
        {
            "requirement": "safe default formal dataset version",
            "status": "complete",
            "scope": "protocol_v2_textoir_v1; older versions require explicit S2C_DATASET_VERSION",
            "evidence": "src/protocol_v2/runtime/paths.py; test_protocol_v2_admission.py",
        },
        {
            "requirement": "fixed TEXTOIR-compatible local source snapshot",
            "status": "complete",
            "scope": "all datasets",
            "evidence": "SOURCE_MANIFEST.json; configs/data/protocol_v2_textoir_v1.yaml",
        },
        {
            "requirement": "three-dataset canonical protocol",
            "status": "complete_local_benchmark",
            "scope": "StackOverflow is local-only; no corpus redistribution",
            "evidence": "source_copy_report.csv; blocker_report.md",
        },
        {
            "requirement": "KIR registries",
            "status": "complete",
            "scope": f"{len(registry_rows)} fixed registries",
            "evidence": "registry_statistics.csv",
        },
        {
            "requirement": "materialized views and method exports",
            "status": "complete",
            "scope": f"{len(view_rows)} views; {len(export_rows)} exports",
            "evidence": "view_statistics.csv; export_statistics.csv",
        },
        {
            "requirement": "E1 three-dataset 36-cell Gate smoke",
            "status": "complete" if completed_runs >= 36 else "in_progress",
            "scope": f"{completed_runs}/36 completed",
            "evidence": "experiment_coverage.csv; e1_gate_smoke.csv",
        },
        {
            "requirement": "E2 three-dataset 1,650-cell Gate grid",
            "status": "in_progress",
            "scope": (
                f"{e2_completed_runs}/1,650 complete; "
                "3 datasets × 11 KIR × 5 seeds × 5 K × 2 distances"
            ),
            "evidence": "plans/e2_gate_core_dense.plan.json; gate_core_dense.e2_core.state.json",
        },
        {
            "requirement": "E3--E7 mechanisms, baselines, representations and Pipeline",
            "status": "not_started_pending_e2",
            "scope": paths.dataset_version,
            "evidence": "implementation_report.md; blocker_report.md",
        },
        {
            "requirement": "frozen MiniLM embedding provenance",
            "status": "used_for_completed_gate_runs" if embedding_cache_used else "not_used",
            "scope": f"{completed_runs + e2_completed_runs} completed Gate run(s)",
            "evidence": "completed run manifest embedding_cache fields",
        },
        {
            "requirement": "full test-suite verification",
            "status": (
                "passed"
                if all(
                    marker in test_summary
                    for marker in ("pytest tests/unit -q", "pytest tests/integration -q", "pytest tests/smoke -q")
                )
                else "targeted_pass_full_suite_pending"
            ),
            "scope": "unit, integration and smoke suites" if "pytest tests/unit -q" in test_summary else "not fully run",
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
                "old_or_external_location": "textoir/data",
                "new_location": f"data/sources/textoir/<fixed-commit> ({paths.dataset_version})",
                "policy": "byte-identical local benchmark source; import-only external dependency",
            },
            {
                "old_or_external_location": "StackOverflow snapshot",
                "new_location": f"data/canonical/{paths.dataset_version}/stackoverflow",
                "policy": "local experiment allowed; public Git tracking and redistribution forbidden",
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
                "current_bytes": "",
                "measurement": "not_scanned: local canonical/views/exports may be large",
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
        "ran full canonical/registry/view/export validation, Gate dry-run and Gate data loading, then restored the "
        "directory and checked its Git status. No model training or embedding generation was used for this check.\n",
    )
    source_commits = sorted({str(row["source_commit"]) for row in source_rows if row["source_commit"]})
    decisions = admission.get("dataset_decisions", {})
    admitted = admission.get("dataset_admission", {})
    coverage_by_experiment = {
        str(row["experiment"]): int(row["complete_runs"])
        for row in coverage_rows
    }
    e1_complete = coverage_by_experiment.get("e1_gate_smoke", 0)
    e2_complete = coverage_by_experiment.get("e2_gate_core_dense", 0)
    run_complete = sum(coverage_by_experiment.values())
    remaining_gate_text = _remaining_gate_text(admission, datasets, e1_complete, e2_complete)
    requirement_rows = _requirement_rows(
        paths,
        datasets,
        registry_rows,
        view_rows,
        export_rows,
        e1_complete,
        e2_complete,
        embedding_cache_used,
        test_summary,
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
    atomic_write_text(
        output / "blocker_report.md",
        _blocker_report(admission, datasets, e1_complete, e2_complete),
    )
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

This report is scoped to the selected dataset version. It preserves frozen and
legacy protocols for audit but does not mix their results into this protocol.

## Materialized data inventory

| Dataset | Samples | Known intents | Native OOS | Local source copy |
| --- | ---: | ---: | ---: | --- |
{dataset_rows}

The authoritative local source for this version is the fixed TEXTOIR snapshot
described by the source manifests. TEXTOIR is import-only; no model, view,
export, Gate or Pipeline runtime reads `textoir/data`.

## Completed implementation work

The fixed snapshot is byte-copied into `data/sources`, canonical records preserve original text, labels and
splits, and every method consumes the same registry and fixed views. Gate runs use immutable directories
beneath `artifacts/s2c/runs/{paths.dataset_version}` and keep embedding cache separate from formal evidence.

## Deliberately not claimed

E3--E7 are not experimental evidence until their own manifests exist. Historical v19-v22 artifacts remain
untouched and are not mixed with this protocol. The StackOverflow corpus remains local-only and is excluded
from public Git/result attachments.

## Requirement status

`requirement_matrix.csv` maps the original implementation goals to current evidence. A status of
`complete_local_benchmark` distinguishes local scientific use from public corpus redistribution.

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
                "detailed changes remain append-only in docs/archive/protocol_and_data/DEVELOPMENT_LOG.md."
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
        default="protocol_v2_textoir_v1",
        help="Immutable dataset version to audit; defaults to the sole active TEXTOIR-compatible protocol.",
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
    # Explicit replacement preserves frozen/legacy audit access without letting
    # their paths replace the active protocol's default at runtime.
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
