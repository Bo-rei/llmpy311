"""Build the auditable lightweight baseline table from completed artifacts."""

from __future__ import annotations

import argparse
import csv
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

    for method, status, supervision, source in (
        ("ADB", "not_run", "Known-only", "not_registered_in_current_protocol"),
        ("DA-ADB", "not_run", "Known-only", "not_registered_in_current_protocol"),
        ("DCLOOS-official", "blocked", "pseudo-OOS plus external open-domain OOS", "docs/dcloos/DCLOOS_REPRODUCTION_REPORT.md"),
        ("DCLOOS-unified", "blocked", "pseudo-OOS plus external open-domain OOS", "docs/dcloos/DCLOOS_REPRODUCTION_REPORT.md"),
    ):
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
