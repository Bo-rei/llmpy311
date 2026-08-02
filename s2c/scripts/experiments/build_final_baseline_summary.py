"""Build the auditable lightweight baseline table from completed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from protocol_v2.data.hashing import atomic_write_text
from protocol_v2.runtime.paths import ProtocolV2Paths


FIELDS = [
    "dataset",
    "kir",
    "method",
    "scope",
    "training_regime",
    "supervision",
    "oos_f1",
    "oos_precision",
    "oos_recall",
    "known_macro_f1",
    "known_recall",
    "f1_all",
    "accuracy",
    "status",
    "source",
]


def _mean(rows: list[dict[str, str]], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key, "") not in {"", "NA", "nan"}]
    return f"{statistics.mean(values):.6f}" if values else "NA"


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [",".join(FIELDS)]
    for row in rows:
        lines.append(",".join(str(row.get(field, "NA")).replace(",", ";") for field in FIELDS))
    atomic_write_text(path, "\n".join(lines) + "\n")


def build(paths: ProtocolV2Paths, output: Path) -> list[dict[str, str]]:
    fair_path = paths.results_root / "mogb" / "fair_matrix.csv"
    fair_rows = list(csv.DictReader(fair_path.open(encoding="utf-8")))
    fair_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in fair_rows:
        if abs(float(row["kir"]) - 0.50) < 1e-9:
            fair_by_key[(row["dataset"], row["method"])].append(row)
    names = {
        "single_centroid": "Single centroid",
        "random_partition": "Random partition",
        "fixed_k2": "Fixed K=2",
        "mogb_minilm": "MOGB-MiniLM",
        "mogb_partition_ours_boundary": "MOGB partition + s2c boundary",
        "ours_partition_mogb_boundary": "s2c partition + MOGB boundary",
    }
    rows: list[dict[str, str]] = []
    for (dataset, source_method), grouped in sorted(fair_by_key.items()):
        rows.append(
            {
                "dataset": dataset,
                "kir": "0.50",
                "method": names[source_method],
                "scope": "protocol_v2_fair_mean_over_5_seeds",
                "training_regime": "frozen-embedding",
                "supervision": "Known-only",
                "oos_f1": _mean(grouped, "oos_f1"),
                "oos_precision": "NA",
                "oos_recall": "NA",
                "known_macro_f1": _mean(grouped, "f1_k"),
                "known_recall": _mean(grouped, "id_recall"),
                "f1_all": _mean(grouped, "f1_all"),
                "accuracy": _mean(grouped, "accuracy"),
                "status": "complete",
                "source": "results/mogb/fair_matrix.csv",
            }
        )

    brak_path = paths.artifacts_root / "runs" / paths.dataset_version / "brak_v1" / "summaries" / "BRAK_PILOT_SUMMARY.tsv"
    brak_rows = list(csv.DictReader(brak_path.open(encoding="utf-8"), delimiter="\t"))
    for method in ("fixed_k1", "fixed_k2", "brak"):
        grouped = [row for row in brak_rows if row["method"] == method]
        rows.append(
            {
                "dataset": "stackoverflow",
                "kir": "0.50",
                "method": {"fixed_k1": "Single centroid (BRAK control)", "fixed_k2": "Fixed K=2 (BRAK control)", "brak": "BRAK"}[method],
                "scope": "brak_pilot_mean_over_3_seeds",
                "training_regime": "frozen-embedding",
                "supervision": "Known-only",
                "oos_f1": _mean(grouped, "oos_f1"),
                "oos_precision": _mean(grouped, "oos_precision"),
                "oos_recall": _mean(grouped, "oos_recall"),
                "known_macro_f1": _mean(grouped, "f1_k"),
                "known_recall": _mean(grouped, "id_recall"),
                "f1_all": _mean(grouped, "f1_all"),
                "accuracy": _mean(grouped, "accuracy"),
                "status": "complete",
                "source": "artifacts/.../brak_v1/summaries/BRAK_PILOT_SUMMARY.tsv",
            }
        )

    official_root = paths.artifacts_root / "external" / "mogb_official_converged_v1"
    for dataset in ("stackoverflow", "banking77"):
        raw_rows = []
        for path in sorted((official_root / dataset / "kir_0.50").glob("seed_*/results/results.csv")):
            raw_rows.extend(csv.DictReader(path.open(encoding="utf-8")))
        rows.append(
            {
                "dataset": dataset,
                "kir": "0.50",
                "method": "MOGB-official (compatibility)",
                "scope": "official_logic_compatibility_mean_over_5_seeds",
                "training_regime": "end-to-end-BERT",
                "supervision": "Known-only plus official protocol",
                "oos_f1": "NA",
                "oos_precision": "NA",
                "oos_recall": "NA",
                "known_macro_f1": f"{statistics.mean(float(row['Known']) for row in raw_rows) / 100:.6f}",
                "known_recall": "NA",
                "f1_all": f"{statistics.mean(float(row['F1-score']) for row in raw_rows) / 100:.6f}",
                "accuracy": f"{statistics.mean(float(row['Accuracy']) for row in raw_rows) / 100:.6f}",
                "status": "complete_non_strict_reproduction",
                "source": "artifacts/external/mogb_official_converged_v1/*/results/results.csv",
            }
        )

    # Strict single-cell MOGB reproduction is intentionally kept separate from
    # the five-seed modernized compatibility aggregate above.  The official
    # format reports F1-U rather than the protocol_v2 binary OOS evaluator, so
    # leave oos_f1 empty instead of silently mixing metric contracts.
    strict_metrics_path = paths.results_root / "mogb_exact_reproduction" / "final_metrics.json"
    if strict_metrics_path.is_file():
        strict_metrics_payload = json.loads(strict_metrics_path.read_text(encoding="utf-8"))
        strict_metrics = strict_metrics_payload["official_fixed"]["metrics"]
        rows.append(
            {
                "dataset": "stackoverflow",
                "kir": "0.50",
                "method": "MOGB-official (strict single-cell)",
                "scope": "official_stackoverflow_kir50_seed0",
                "training_regime": "end-to-end-BERT",
                "supervision": "Known-only plus official protocol",
                "oos_f1": "NA",
                "oos_precision": "NA",
                "oos_recall": "NA",
                "known_macro_f1": f"{float(strict_metrics['F1-K']) / 100:.6f}",
                "known_recall": f"{float(strict_metrics['Known Recall']) / 100:.6f}",
                "f1_all": f"{float(strict_metrics['F1-All']) / 100:.6f}",
                "accuracy": f"{float(strict_metrics['Accuracy']) / 100:.6f}",
                "status": "not_reproduced_strict",
                "source": "results/mogb_exact_reproduction/final_metrics.json",
            }
        )

    strict_banking_path = paths.results_root / "mogb_exact_reproduction_banking" / "final_metrics.json"
    if strict_banking_path.is_file():
        strict_banking_payload = json.loads(strict_banking_path.read_text(encoding="utf-8"))
        strict_banking = strict_banking_payload["official_fixed"]["metrics"]
        rows.append(
            {
                "dataset": "banking77",
                "kir": "0.75",
                "method": "MOGB-official (strict single-cell)",
                "scope": "official_banking_kir75_seed0",
                "training_regime": "end-to-end-BERT",
                "supervision": "Known-only plus official protocol",
                "oos_f1": "NA",
                "oos_precision": "NA",
                "oos_recall": "NA",
                "known_macro_f1": f"{float(strict_banking['F1-K']) / 100:.6f}",
                "known_recall": f"{float(strict_banking['Known Recall']) / 100:.6f}",
                "f1_all": f"{float(strict_banking['F1-All']) / 100:.6f}",
                "accuracy": f"{float(strict_banking['Accuracy']) / 100:.6f}",
                "status": "not_reproduced_strict",
                "source": "results/mogb_exact_reproduction_banking/final_metrics.json",
            }
        )

    # BRAK on the two MOGB BERT representation snapshots is a separate
    # representation-transfer diagnostic.  It must not be collapsed into the
    # frozen-MiniLM BRAK pilot row above or presented as an official MOGB run.
    brak_mogb_path = (
        paths.results_root
        / "mogb_exact_reproduction"
        / "brak_mogb_representation"
        / "brak_summary.csv"
    )
    if brak_mogb_path.is_file():
        brak_mogb_rows = list(csv.DictReader(brak_mogb_path.open(encoding="utf-8")))
        representation_names = {
            "mogb_initial_bert": "BRAK (MOGB initial BERT)",
            "mogb_trained_hierarchical_bert": "BRAK (MOGB trained BERT)",
        }
        for representation, display_name in representation_names.items():
            grouped = [
                row
                for row in brak_mogb_rows
                if row["representation"] == representation and row["method"] == "brak"
            ]
            if not grouped:
                continue
            row = grouped[0]
            rows.append(
                {
                    "dataset": "stackoverflow",
                    "kir": "0.50",
                    "method": display_name,
                    "scope": "mogb_representation_stackoverflow_kir50_seed0",
                    "training_regime": "Known-only BRAK on BERT representation",
                    "supervision": "Known-only",
                    "oos_f1": row.get("oos_f1", "NA"),
                    "oos_precision": row.get("oos_precision", "NA"),
                    "oos_recall": row.get("oos_recall", "NA"),
                    "known_macro_f1": row.get("f1_k", "NA"),
                    "known_recall": row.get("id_recall", "NA"),
                    "f1_all": row.get("f1_all", "NA"),
                    "accuracy": row.get("accuracy", "NA"),
                    "status": "complete_negative_control",
                    "source": "results/mogb_exact_reproduction/brak_mogb_representation/brak_summary.csv",
                }
            )

    # A historical ADB KIR=.50/seed0 run already exists under the frozen v19
    # official-run root.  It is not silently promoted to protocol_v2: retain
    # the exact source path and label it as a compatibility artifact.  The
    # fallback blocked row is emitted only when no complete result is found.
    historical_external = paths.artifacts_root / "outputs" / "experiments" / "cluster_separability_v19" / "textoir_protocol" / "official_runs"
    completed_external: dict[str, tuple[Path, str, str, str]] = {}
    for method in ("ADB", "DA-ADB"):
        candidates = sorted(
            historical_external.glob(f"stackoverflow/{method}/kir50/seed0/attempts/*/results/results.csv")
        )
        if candidates:
            completed_external[method] = (candidates[-1], "stackoverflow", "0.50", "historical_textoir_compatibility_single_cell")
    current_external = {
        "ADB": paths.artifacts_root / "external" / "adb_compat_single_cell_v2" / "stackoverflow" / "ADB" / "kir50" / "seed0" / "results" / "results.csv",
        "DA-ADB": paths.artifacts_root / "external" / "da_adb_compat_single_cell_v3" / "stackoverflow" / "DA-ADB" / "kir50" / "seed0" / "results" / "results.csv",
    }
    for method, candidate in current_external.items():
        if candidate.is_file():
            completed_external[method] = (candidate, "stackoverflow", "0.50", "modernized_textoir_compatibility_single_cell")
    for method, status, supervision, source in (
        (
            "ADB",
            "blocked_runtime_dependency",
            "Known-only",
            "docs/mogb_integration/ADB_DAADB_AUDIT.md",
        ),
        (
            "DA-ADB",
            "blocked_runtime_dependency",
            "Known-only",
            "docs/mogb_integration/ADB_DAADB_AUDIT.md",
        ),
        ("DCLOOS-official", "blocked_missing_external_negative_data", "pseudo-OOS plus external open-domain OOS", "docs/dcloos/DCLOOS_REPRODUCTION_REPORT.md"),
        ("DCLOOS-unified", "blocked_missing_external_negative_data", "pseudo-OOS plus external open-domain OOS", "docs/dcloos/DCLOOS_REPRODUCTION_REPORT.md"),
    ):
        if method in completed_external:
            result_path, dataset, kir, scope = completed_external[method]
            external_row = next(csv.DictReader(result_path.open(encoding="utf-8")))
            rows.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "method": method,
                    "scope": scope,
                    "training_regime": "end-to-end-BERT",
                    "supervision": "Known-only",
                    "oos_f1": f"{float(external_row['F1-open']) / 100:.6f}",
                    "oos_precision": "NA",
                    "oos_recall": "NA",
                    "known_macro_f1": f"{float(external_row['F1-known']) / 100:.6f}",
                    "known_recall": "NA",
                    "f1_all": f"{float(external_row['F1']) / 100:.6f}",
                    "accuracy": f"{float(external_row['Acc']) / 100:.6f}",
                    "status": "complete_compatibility_artifact",
                    "source": str(result_path.relative_to(paths.project_root.parent)),
                }
            )
        elif method.startswith("DCLOOS"):
            dcloos_manifest = paths.artifacts_root / "external" / "dcloos_official_single_cell_v1" / "run_manifest.json"
            rows.append(
                {
                    "dataset": "oos",
                    "kir": "0.75",
                    "method": method,
                    "scope": "official_external_negative_single_cell",
                    "training_regime": "end-to-end-BERT",
                    "supervision": supervision,
                    "oos_f1": "NA",
                    "oos_precision": "NA",
                    "oos_recall": "NA",
                    "known_macro_f1": "NA",
                    "known_recall": "NA",
                    "f1_all": "NA",
                    "accuracy": "NA",
                    "status": "timeout_incomplete",
                    "source": str(dcloos_manifest.relative_to(paths.project_root.parent)) if dcloos_manifest.is_file() else source,
                }
            )
        else:
            rows.append(
                {
                    "dataset": "all_requested",
                    "kir": "0.50",
                    "method": method,
                    "scope": "requested_final_baseline_slot",
                    "training_regime": "end-to-end",
                    "supervision": supervision,
                    "oos_f1": "NA",
                    "oos_precision": "NA",
                    "oos_recall": "NA",
                    "known_macro_f1": "NA",
                    "known_recall": "NA",
                    "f1_all": "NA",
                    "accuracy": "NA",
                    "status": status,
                    "source": source,
                }
            )

    # A reduced-budget DCLOOS run reached a valid upstream test evaluation but
    # failed only while serializing its final JSON metrics.  Keep its recovered
    # prediction-derived result explicit and separate from the strict/default
    # timeout row above; it is not a paper-table reproduction.
    dcloos_recovery_path = (
        paths.artifacts_root
        / "external"
        / "dcloos_official_oos_kir75_seed888_reduced_v2"
        / "recovery_metrics.json"
    )
    if dcloos_recovery_path.is_file():
        recovered = json.loads(dcloos_recovery_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": "oos",
                "kir": "0.75",
                "method": "DCLOOS-official (reduced-budget recovered)",
                "scope": "official_external_negative_single_cell_recovered",
                "training_regime": "end-to-end-BERT",
                "supervision": "Known-only plus synthetic and external OOS",
                "oos_f1": f"{float(recovered['oos_f1']) / 100:.6f}",
                "oos_precision": f"{float(recovered['oos_precision']) / 100:.6f}",
                "oos_recall": f"{float(recovered['oos_recall']) / 100:.6f}",
                "known_macro_f1": f"{float(recovered['f1_k']) / 100:.6f}",
                "known_recall": f"{float(recovered['known_recall']) / 100:.6f}",
                "f1_all": f"{float(recovered['f1_all']) / 100:.6f}",
                "accuracy": f"{float(recovered['accuracy']) / 100:.6f}",
                "status": "complete_recovered_intermediate_prediction",
                "source": "artifacts/s2c/external/dcloos_official_oos_kir75_seed888_reduced_v2/recovery_metrics.json",
            }
        )
    _write(output, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = ProtocolV2Paths.discover()
    output = args.output or paths.results_root / "final_baselines" / "summary.csv"
    rows = build(paths, output)
    print(f"wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
