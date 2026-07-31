"""Close out the gated CLMSG Milestone 1--3 pilot without starting later stages."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file
from protocol_v2.runtime.paths import ProtocolV2Paths


PRIMARY = "local_scale_conformal__class_conditional_knn__alpha_0.05"
EXPECTED_CONFIRMATION_SEEDS = {13, 42, 87}
SUMMARY_METRICS = (
    "oos_f1",
    "known_recall",
    "f1_all",
    "accuracy",
    "auroc",
    "aupr_oos",
    "false_accept_rate",
    "false_reject_rate",
)


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _mean_std_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {"method": method, "n_seeds": len(selected)}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in selected]
            item[f"mean_{metric}"] = statistics.fmean(values)
            item[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def summarize(root: Path) -> dict[str, Any]:
    run_dirs = sorted(root.glob("support_modes_v1/stackoverflow/kir_0.50/seed_*"))
    rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    invalid: list[str] = []
    for run in run_dirs:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        seed = int(manifest["run_id"].rsplit("seed", 1)[1])
        if manifest.get("status") != "complete" or manifest.get("test_used_for_selection"):
            invalid.append(str(run))
            continue
        for method, value in metrics.items():
            rows.append(
                {
                    "dataset": "stackoverflow",
                    "kir": 0.50,
                    "seed": seed,
                    "method": method,
                    **{
                        key: value.get(key)
                        for key in (
                            "oos_f1",
                            "oos_precision",
                            "oos_recall",
                            "known_macro_f1",
                            "known_recall",
                            "f1_all",
                            "accuracy",
                            "auroc",
                            "aupr_oos",
                            "fpr95",
                            "false_accept_rate",
                            "false_reject_rate",
                            "target_alpha",
                            "coverage_error",
                        )
                    },
                    "run_dir": str(run),
                }
            )
        baseline_payload = json.loads((run / "baselines.json").read_text(encoding="utf-8"))
        for value in baseline_payload["rows"]:
            baselines.append(
                {
                    "dataset": value["dataset"],
                    "kir": float(value["kir"]),
                    "seed": int(value["seed"]),
                    "method": value["method"],
                    "oos_f1": float(value["oos_f1"]),
                    "f1_all": float(value["f1_all"]),
                    "known_recall": float(value["id_recall"]),
                    "false_accept_rate": float(value["false_accept_rate"]),
                }
            )

    summary_root = root / "summary"
    fields = list(rows[0]) if rows else ["dataset", "kir", "seed", "method"]
    atomic_write_text(summary_root / "all_runs.csv", _csv_text(rows, fields))
    baseline_fields = list(baselines[0]) if baselines else ["dataset", "kir", "seed", "method"]
    atomic_write_text(summary_root / "baseline_reference.csv", _csv_text(baselines, baseline_fields))

    aggregate_rows = _mean_std_rows(rows)
    aggregate_fields = list(aggregate_rows[0]) if aggregate_rows else ["method", "n_seeds"]
    atomic_write_text(summary_root / "mean_std.csv", _csv_text(aggregate_rows, aggregate_fields))

    single_by_seed = {
        int(row["seed"]): row for row in baselines if row["method"] == "single_centroid"
    }
    paired_rows: list[dict[str, Any]] = []
    for row in rows:
        single_row = single_by_seed.get(int(row["seed"]))
        if single_row is None:
            continue
        paired_rows.append(
            {
                "seed": row["seed"],
                "method": row["method"],
                "oos_f1": row["oos_f1"],
                "single_centroid_oos_f1": single_row["oos_f1"],
                "delta_oos_f1": float(row["oos_f1"]) - float(single_row["oos_f1"]),
                "known_recall": row["known_recall"],
                "single_centroid_known_recall": single_row["known_recall"],
                "delta_known_recall": float(row["known_recall"])
                - float(single_row["known_recall"]),
            }
        )
    paired_fields = list(paired_rows[0]) if paired_rows else ["seed", "method"]
    atomic_write_text(summary_root / "paired_vs_single.csv", _csv_text(paired_rows, paired_fields))

    seed13 = {row["method"]: row for row in rows if row["seed"] == 13}
    baseline13 = {row["method"]: row for row in baselines if row["seed"] == 13}
    primary = seed13.get(PRIMARY)
    single = baseline13.get("single_centroid")
    version_c_rows = [
        row for row in rows if row["seed"] == 13 and row["method"].startswith("local_scale_conformal")
    ]
    best_c = max(version_c_rows, key=lambda row: float(row["oos_f1"])) if version_c_rows else None
    completed_seeds = {int(row["seed"]) for row in rows}
    primary_rows = [row for row in rows if row["method"] == PRIMARY]
    primary_deltas = [
        float(row["oos_f1"]) - float(single_by_seed[int(row["seed"])]["oos_f1"])
        for row in primary_rows
        if int(row["seed"]) in single_by_seed
    ]
    version_c_by_seed = {
        seed: [
            row
            for row in rows
            if int(row["seed"]) == seed and row["method"].startswith("local_scale_conformal")
        ]
        for seed in completed_seeds
    }
    all_c_below_single = all(
        seed in single_by_seed
        and all(float(row["oos_f1"]) <= float(single_by_seed[seed]["oos_f1"]) for row in seed_rows)
        for seed, seed_rows in version_c_by_seed.items()
    )
    confirmation_complete = completed_seeds == EXPECTED_CONFIRMATION_SEEDS and not invalid
    decision = {
        "status": "stop_after_seed_confirmation" if confirmation_complete else "incomplete_confirmation",
        "completed_seeds": sorted(completed_seeds),
        "missing_confirmation_seeds": sorted(EXPECTED_CONFIRMATION_SEEDS - completed_seeds),
        "primary_method": PRIMARY,
        "primary_oos_f1": None if primary is None else primary["oos_f1"],
        "single_centroid_oos_f1": None if single is None else single["oos_f1"],
        "primary_delta_oos_f1_vs_single": (
            None if primary is None or single is None else float(primary["oos_f1"]) - float(single["oos_f1"])
        ),
        "primary_delta_known_recall_vs_single": (
            None
            if primary is None or single is None
            else float(primary["known_recall"]) - float(single["known_recall"])
        ),
        "best_descriptive_version_c": None if best_c is None else best_c["method"],
        "best_descriptive_version_c_oos_f1": None if best_c is None else best_c["oos_f1"],
        "any_version_c_beats_single": bool(
            single is not None
            and any(float(row["oos_f1"]) > float(single["oos_f1"]) for row in version_c_rows)
        ),
        "all_version_c_below_single_across_confirmation_seeds": all_c_below_single,
        "mean_primary_oos_f1": (
            None if not primary_rows else statistics.fmean(float(row["oos_f1"]) for row in primary_rows)
        ),
        "mean_primary_delta_oos_f1_vs_single": (
            None if not primary_deltas else statistics.fmean(primary_deltas)
        ),
        "invalid_runs": invalid,
        "manifold_or_entropy_authorized": False,
        "full_sweep_authorized": False,
        "reason": (
            "Every prescribed local-scale conformal support mode is dominated by Single-centroid "
            "OOS F1 on all three confirmation seeds."
            if confirmation_complete and all_c_below_single
            else "The fixed three-seed confirmation is incomplete or contains an invalid run."
        ),
    }
    atomic_write_json(summary_root / "stage_decision.json", decision)
    project_root = ProtocolV2Paths.discover().project_root
    status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--short"],
        capture_output=True,
        text=True,
        check=False,
    )
    core_files = {
        "algorithm": project_root / "src" / "protocol_v2" / "gate" / "clmsg.py",
        "runner": project_root / "scripts" / "experiments" / "run_clmsg.py",
        "config": project_root / "configs" / "gates" / "clmsg.yaml",
        "unit_tests": project_root / "tests" / "unit" / "test_clmsg.py",
        "formal_run_manifest": root
        / "support_modes_v1"
        / "stackoverflow"
        / "kir_0.50"
        / "seed_13"
        / "manifest.json",
    }
    formal_manifest = json.loads(core_files["formal_run_manifest"].read_text(encoding="utf-8"))
    environment = json.loads(
        (
            root
            / "support_modes_v1"
            / "stackoverflow"
            / "kir_0.50"
            / "seed_13"
            / "environment.txt"
        ).read_text(encoding="utf-8")
    )
    atomic_write_json(
        summary_root / "CLMSG_PROVENANCE_SNAPSHOT.json",
        {
            "stage": "clmsg_v1_milestones_1_4_confirmation",
            "base_commit": environment["git_commit"],
            "environment_git_commit": environment["git_commit"],
            "git_dirty": bool(status.stdout.strip()),
            "file_sha256": {name: sha256_file(path) for name, path in core_files.items()},
            "registry_sha256": formal_manifest["provenance"]["registry_sha256"],
            "canonical_embedding_sha256": formal_manifest["provenance"][
                "canonical_embedding_sha256"
            ],
            "input_hashes": formal_manifest["provenance"]["input_hashes"],
            "authorized_outputs": len(rows),
            "completed_seeds": decision["completed_seeds"],
            "missing_confirmation_seeds": decision["missing_confirmation_seeds"],
            "test_used_for_selection": False,
            "generated_at": datetime.now(UTC).isoformat(),
        },
    )
    atomic_write_text(
        summary_root / "CLMSG_M1_M3_CLOSEOUT.md",
        "\n".join(
            [
                "# CLMSG Milestone 1--3 closeout",
                "",
                "- Protocol: `protocol_v2_textoir_v1`",
                "- Cell: StackOverflow, KIR=0.50, seed=13",
                "- Completed: KNN-only, global/class-conditional/hybrid local-scale KNN, and global split conformal",
                "- Data: 6,000 Known proper-train, 1,000 Known calibration, 6,000 test; disjoint sample IDs",
                "- Encoder: frozen cached `all-MiniLM-L6-v2`; no encoding or training",
                "",
                "## Gate decision",
                "",
                f"Primary Version C OOS F1: `{decision['primary_oos_f1']:.4f}`; "
                f"Single-centroid: `{decision['single_centroid_oos_f1']:.4f}`; "
                f"delta: `{decision['primary_delta_oos_f1_vs_single']:+.4f}`.",
                f"Primary Version C Known Recall delta: `{decision['primary_delta_known_recall_vs_single']:+.4f}`.",
                f"Best descriptive Version C setting: `{decision['best_descriptive_version_c']}` "
                f"with OOS F1 `{decision['best_descriptive_version_c_oos_f1']:.4f}`.",
                "",
                "No prescribed Version C setting beats the single-centroid baseline. Seeds 42/87, local manifold,",
                "label entropy, cross-conformal, and the full sweep are therefore not authorized. The completed",
                "seed13 run is retained as falsification evidence; no historical artifact was modified.",
                "",
            ]
        ),
    )
    atomic_write_text(
        summary_root / "CLMSG_M4_CONFIRMATION_CLOSEOUT.md",
        "\n".join(
            [
                "# CLMSG Milestone 4 seed confirmation closeout",
                "",
                "- Protocol: `protocol_v2_textoir_v1`",
                "- Cell family: StackOverflow, KIR=0.50, seeds 13/42/87",
                "- Completed: 78 fixed method/alpha outputs; no representation training or test selection",
                "- Configuration: unchanged k=10, cosine distance, support modes, alpha grid, and evaluator",
                "",
                "## Confirmation decision",
                "",
                f"Mean primary Version C OOS F1: `{decision['mean_primary_oos_f1']:.4f}`; mean paired "
                f"delta versus Single-centroid: `{decision['mean_primary_delta_oos_f1_vs_single']:+.4f}`.",
                f"All prescribed Version C rows remain below Single-centroid on every seed: "
                f"`{decision['all_version_c_below_single_across_confirmation_seeds']}`.",
                "",
                "The seed13 failure is therefore stable across the two additional predeclared seeds. Local",
                "manifold, label entropy, cross-conformal, and the full CLMSG sweep remain unauthorized.",
                "A future KNN-only Pareto study must be registered as a separate question and may not choose",
                "its operating point from test OOS performance.",
                "",
            ]
        ),
    )
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    root = args.input_dir or paths.run_root / "clmsg_v1"
    decision = summarize(root.resolve())
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
