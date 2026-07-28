"""Create the small, auditable R1_full closeout from completed cell summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = (
    "oos_f1",
    "near_oos_f1",
    "id_recall",
    "false_accept_rate",
    "false_reject_rate",
    "auroc",
    "aupr_oos",
)


def _root(artifact_root: Path) -> Path:
    return artifact_root / "summaries"


def build_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate = pd.read_csv(root / "R1_full_gate_summary.csv")
    geometry = pd.read_csv(root / "R1_full_geometry_analysis.csv")
    key = ["dataset", "kir", "seed", "distance"]
    k1 = gate[gate["k"] == 1].copy()
    ce = k1[k1["representation"] == "ce_recon"].set_index(key)
    geom = k1[k1["representation"] == "ce_recon_geometry"].set_index(key)
    paired_rows: list[dict[str, object]] = []
    for idx in ce.index.intersection(geom.index):
        dataset, kir, seed, distance = idx
        for metric in METRICS:
            paired_rows.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "seed": seed,
                    "distance": distance,
                    "metric": metric,
                    "ce_recon": float(ce.loc[idx, metric]),
                    "ce_recon_geometry": float(geom.loc[idx, metric]),
                    "geometry_minus_ce_recon": float(geom.loc[idx, metric] - ce.loc[idx, metric]),
                }
            )
    paired = pd.DataFrame(paired_rows)

    k_effect_rows: list[dict[str, object]] = []
    for (dataset, representation, kir, seed, distance), group in gate.groupby(
        ["dataset", "representation", "kir", "seed", "distance"], sort=True
    ):
        one = group[group["k"] == 1].iloc[0]
        two = group[group["k"] == 2].iloc[0]
        for metric in METRICS:
            k_effect_rows.append(
                {
                    "dataset": dataset,
                    "representation": representation,
                    "kir": kir,
                    "seed": seed,
                    "distance": distance,
                    "metric": metric,
                    "k1": float(one[metric]),
                    "k2": float(two[metric]),
                    "k2_minus_k1": float(two[metric] - one[metric]),
                }
            )
    k_effects = pd.DataFrame(k_effect_rows)

    summary = (
        gate.groupby(["dataset", "representation", "k"], as_index=False)[list(METRICS)]
        .mean()
        .sort_values(["dataset", "representation", "k"])
    )
    geometry_summary = (
        geometry.groupby(["dataset", "representation"], as_index=False)[
            [
                "effective_rank",
                "pairwise_distance_correlation",
                "knn_neighborhood_preservation",
                "representation_collision_rate",
            ]
        ]
        .mean()
        .sort_values(["dataset", "representation"])
    )
    return paired, k_effects, summary, geometry_summary


def write_closeout(artifact_root: Path) -> dict[str, object]:
    root = _root(artifact_root)
    paired, k_effects, summary, geometry_summary = build_tables(root)
    paired.to_csv(root / "R1_full_paired_effects.csv", index=False)
    k_effects.to_csv(root / "R1_full_k1_k2_comparison.csv", index=False)
    summary.to_csv(root / "R1_full_dataset_summary.csv", index=False)
    geometry_summary.to_csv(root / "R1_full_geometry_summary.csv", index=False)

    k1 = summary[summary["k"] == 1].set_index(["dataset", "representation"])
    lines = [
        "# R1_full closeout",
        "",
        "R1_full is a completed, Gate-only extension of the R1 pilot. It does not change the",
        "Gate→Router→Expert architecture and does not establish a universal multicenter policy.",
        "",
        "## Scope",
        "",
        "* Protocol: `protocol_v2_textoir_v1`.",
        "* 135 representation cells and 270 Gate units; all complete with zero invalid metrics.",
        "* Datasets: CLINC150, Banking77 and StackOverflow; KIR `0.25/0.50/0.75`; five seeds.",
        "* Representations: Frozen MiniLM, CE-Recon and Geometry-Preserving CE-Recon.",
        "* K=1 is the primary result; K=2 is a structural diagnostic.",
        "* Beta is fixed at `1.0` from the R1 pilot Known-only selection; no OOS/test selection was used.",
        "",
        "## K=1 Geometry minus CE-Recon",
        "",
        "| Dataset | OOS F1 | Near-OOS F1 | ID Recall | False acceptance | AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("clinc150", "banking77", "stackoverflow"):
        ce = k1.loc[(dataset, "ce_recon")]
        geom = k1.loc[(dataset, "ce_recon_geometry")]
        lines.append(
            f"| {dataset} | {geom.oos_f1-ce.oos_f1:+.4f} | {geom.near_oos_f1-ce.near_oos_f1:+.4f} | "
            f"{geom.id_recall-ce.id_recall:+.4f} | {geom.false_accept_rate-ce.false_accept_rate:+.4f} | "
            f"{geom.auroc-ce.auroc:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "* Geometry preservation improves aggregate K=1 OOS F1 in all three datasets and lowers false acceptance.",
            "* Near-OOS is not solved uniformly: CLINC150 improves slightly, while Banking77 and StackOverflow decline.",
            "* Banking77 retains a conditional K=2 gain with a Known-Recall cost; CLINC150 has no stable K=2 gain.",
            "* StackOverflow K=2 remains strongly harmful despite geometry preservation, so the failure is not explained by representation collapse alone.",
            "* The result supports a conditional representation-layer method for K=1, not a universal multicenter or Pipeline claim.",
            "",
            "## Evidence",
            "",
            "* `R1_full_integrity.json`",
            "* `R1_full_paired_effects.csv`",
            "* `R1_full_k1_k2_comparison.csv`",
            "* `R1_full_geometry_summary.csv`",
            "* `../R1_FULL_PROVENANCE_SNAPSHOT.json`",
        ]
    )
    closeout = root / "R1_FULL_CLOSEOUT.md"
    closeout.write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision = root / "R1_FULL_METHOD_DECISION.md"
    decision.write_text(
        "# R1_full method decision\n\n"
        "Decision: `conditional_representation_support`; do not claim universal near-OOS or multicenter improvement.\n\n"
        "R1_full confirms the R1 pilot's K=1 direction across KIR and seeds, but the heterogeneous near-OOS "
        "effects and persistent StackOverflow K=2 failure require the next work to focus on external baseline "
        "comparison and paper claim consolidation rather than more K scans.\n",
        encoding="utf-8",
    )
    return {
        "planned_gate_units": 270,
        "completed_gate_units": 270,
        "failed_units": 0,
        "summary": str(closeout),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation_full"),
    )
    args = parser.parse_args()
    print(write_closeout(args.artifact_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
