#!/usr/bin/env python3
"""Create compact aggregate views of the lambda leakage-audit CSV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87)
METRICS = (
    "oos_f1",
    "oos_precision",
    "oos_recall",
    "f1_all",
    "f1_k",
    "f1_u",
    "accuracy",
    "known_macro_f1",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
    "auroc",
    "aupr_oos",
)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        records.append(row)
    return pd.DataFrame.from_records(records)


def _selected(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[np.isclose(frame["radius_lambda"], frame["known_only_selected_lambda"])]
    return selected.copy()


def _decision(frame: pd.DataFrame) -> dict[str, object]:
    selected = _selected(frame[frame["k_label"].isin(["k_1", "k_2"])])
    decisions: dict[str, object] = {}
    for dataset in DATASETS:
        current = selected[selected["dataset"] == dataset]
        one = current[current["k_label"] == "k_1"].set_index("seed")
        two = current[current["k_label"] == "k_2"].set_index("seed")
        common = sorted(set(one.index) & set(two.index))
        deltas = []
        for seed in common:
            deltas.append(
                {
                    "seed": int(seed),
                    "oos_f1_delta_k2_minus_k1": float(two.loc[seed, "oos_f1"] - one.loc[seed, "oos_f1"]),
                    "f1_all_delta_k2_minus_k1": float(two.loc[seed, "f1_all"] - one.loc[seed, "f1_all"]),
                    "known_recall_delta_k2_minus_k1": float(two.loc[seed, "known_recall"] - one.loc[seed, "known_recall"]),
                    "false_accept_delta_k2_minus_k1": float(two.loc[seed, "false_accept_rate"] - one.loc[seed, "false_accept_rate"]),
                    "lambda_k1": float(one.loc[seed, "known_only_selected_lambda"]),
                    "lambda_k2": float(two.loc[seed, "known_only_selected_lambda"]),
                }
            )
        oos = [item["oos_f1_delta_k2_minus_k1"] for item in deltas]
        f1_all = [item["f1_all_delta_k2_minus_k1"] for item in deltas]
        known = [item["known_recall_delta_k2_minus_k1"] for item in deltas]
        selected_lambdas = [item["lambda_k1"] for item in deltas] + [item["lambda_k2"] for item in deltas]
        mean_oos = float(np.mean(oos)) if oos else None
        mean_f1_all = float(np.mean(f1_all)) if f1_all else None
        mean_known = float(np.mean(known)) if known else None
        direction_count = int(sum(value >= 0 for value in oos))
        base = {
            "aggregate_known_only_selected_lambda_deltas": deltas,
            "mean_oos_f1_delta_k2_minus_k1": mean_oos,
            "mean_f1_all_delta_k2_minus_k1": mean_f1_all,
            "mean_known_recall_delta_k2_minus_k1": mean_known,
            "direction_consistent_oos_f1_seeds": direction_count,
            "known_only_selected_lambdas": sorted(set(selected_lambdas)),
            "lambda_stable_within_0.50": bool(selected_lambdas and max(selected_lambdas) - min(selected_lambdas) <= 0.50),
        }
        if dataset == "banking77":
            checks = {
                "intent_positive_fraction_ge_0.20": None,
                "f1_all_gain_ge_0.01": mean_f1_all is not None and mean_f1_all >= 0.01,
                "oos_f1_gain_ge_0.02": mean_oos is not None and mean_oos >= 0.02,
                "known_recall_loss_le_0.02": mean_known is not None and mean_known >= -0.02,
                "direction_consistent_in_at_least_2_of_3": direction_count >= 2,
                "lambda_stable": base["lambda_stable_within_0.50"],
            }
            base.update(
                {
                    "status": "not_authorized",
                    "gate_checks": checks,
                    "overall_pass": False,
                    "reason": "Per-intent positive-fraction evidence is unavailable under the Known-only lambda contract; existing intent rows are test-oracle diagnostics only.",
                }
            )
        elif dataset == "stackoverflow":
            checks = {
                "negative_control_f1_all_drop_le_0.01": mean_f1_all is not None and mean_f1_all >= -0.01,
                "negative_control_oos_f1_drop_le_0.02": mean_oos is not None and mean_oos >= -0.02,
                "fallback_to_k1_executed": False,
            }
            base.update(
                {
                    "status": "not_authorized",
                    "gate_checks": checks,
                    "overall_pass": False,
                    "reason": "No split-merge rule was executed; fixed K=2 remains a negative control and the selected-lambda aggregate still shows boundary-union degradation.",
                }
            )
        else:
            base.update(
                {
                    "status": "neutral_control",
                    "gate_checks": {},
                    "overall_pass": None,
                    "reason": "CLINC150 is retained as a neutral dataset; no adaptive-K authorization is inferred from aggregate test sensitivity.",
                }
            )
        decisions[dataset] = base
    return {
        "stage": "lambda_leakage_audit_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "split_merge_pilot_authorized": False,
        "adaptive_k_gate": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/diagnostics/lambda_sensitivity/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnostics/lambda_sensitivity"))
    args = parser.parse_args()
    frame = pd.read_csv(args.summary)
    required = {"dataset", "seed", "k_label", "radius_lambda", "known_only_selected_lambda", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing sensitivity columns: {missing}")
    numeric = frame[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Sensitivity summary contains non-finite metrics")
    core = frame[frame["k_label"].isin(["k_1", "k_2"])].copy()
    mean_std = _aggregate(core, ["dataset", "k_label", "radius_lambda"])
    _atomic_csv(args.output_dir / "mean_std.csv", mean_std)

    one = core[core["k_label"] == "k_1"].set_index(["dataset", "seed", "radius_lambda"])
    two = core[core["k_label"] == "k_2"].set_index(["dataset", "seed", "radius_lambda"])
    common = one.index.intersection(two.index)
    interaction = []
    for key in common:
        left, right = one.loc[key], two.loc[key]
        dataset, seed, value = key
        interaction.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "radius_lambda": float(value),
                **{f"delta_{metric}_k2_minus_k1": float(right[metric] - left[metric]) for metric in METRICS},
            }
        )
    interaction_frame = pd.DataFrame(interaction)
    _atomic_csv(args.output_dir / "lambda_k_interaction.csv", interaction_frame)
    if not interaction_frame.empty:
        interaction_mean = _aggregate(
            interaction_frame.rename(columns={f"delta_{metric}_k2_minus_k1": metric for metric in METRICS}),
            ["dataset", "radius_lambda"],
        )
        _atomic_csv(args.output_dir / "lambda_k_interaction_mean_std.csv", interaction_mean)

    selected = _selected(core)
    _atomic_csv(args.output_dir / "known_only_selected.csv", selected)
    _atomic_json(args.output_dir / "adaptive_k_decision.json", _decision(frame))
    print(json.dumps({"rows": int(len(frame)), "mean_std_rows": int(len(mean_std)), "interaction_rows": int(len(interaction_frame))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
