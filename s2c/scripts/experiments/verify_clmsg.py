"""Verify the fixed three-seed CLMSG confirmation and leakage/provenance contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from protocol_v2.runtime.paths import ProtocolV2Paths


def verify(root: Path) -> dict[str, object]:
    runs = [
        root / "support_modes_v1" / "stackoverflow" / "kir_0.50" / f"seed_{seed}"
        for seed in (13, 42, 87)
    ]
    required = {
        "manifest.json",
        "metrics.json",
        "predictions.csv",
        "calibration_manifest.json",
        "calibration_scores.npy",
        "support_statistics.json",
        "per_intent_metrics.csv",
        "runtime.json",
    }
    total_prediction_rows = 0
    total_calibration_scores = 0
    total_outputs = 0
    verified_seeds: list[int] = []
    numeric = (
        "oos_f1",
        "oos_precision",
        "oos_recall",
        "known_recall",
        "f1_all",
        "accuracy",
        "auroc",
        "aupr_oos",
        "false_accept_rate",
        "false_reject_rate",
    )
    for run in runs:
        missing = sorted(name for name in required if not (run / name).is_file())
        if missing:
            raise FileNotFoundError(f"Missing CLMSG evidence in {run}: {missing}")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        calibration = json.loads((run / "calibration_manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or manifest.get("test_used_for_selection") is not False:
            raise ValueError("CLMSG manifest is not a completed no-test-selection run")
        if not calibration.get("sample_id_sets_disjoint") or not calibration.get(
            "train_and_calibration_known_only"
        ):
            raise ValueError("CLMSG calibration leakage contract failed")
        if calibration.get("uses_oos") or calibration.get("test_used_for_selection"):
            raise ValueError("CLMSG calibration used a forbidden OOS/test source")
        if len(metrics) != 26:
            raise ValueError(f"Expected 26 authorized method/alpha outputs, found {len(metrics)}")
        for method, values in metrics.items():
            if not all(math.isfinite(float(values[key])) for key in numeric):
                raise ValueError(f"Non-finite CLMSG metric: {method}")
        by_method: dict[str, list[str]] = defaultdict(list)
        with (run / "predictions.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                by_method[row["method"]].append(row["sample_id"])
        if set(by_method) != set(metrics):
            raise ValueError("Prediction and metric method sets disagree")
        reference: list[str] | None = None
        for method, sample_ids in by_method.items():
            if len(sample_ids) != 6000 or len(set(sample_ids)) != 6000:
                raise ValueError(f"Prediction coverage mismatch: {method}")
            if reference is None:
                reference = sample_ids
            elif sample_ids != reference:
                raise ValueError(f"Prediction sample order mismatch: {method}")
        groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for method, values in metrics.items():
            if method.startswith("local_scale_conformal"):
                prefix = method.rsplit("__alpha_", 1)[0]
                groups[prefix].append(
                    (float(values["target_alpha"]), float(values["false_reject_rate"]))
                )
        for prefix, values in groups.items():
            ordered = sorted(values)
            rejections = [item[1] for item in ordered]
            if rejections != sorted(rejections):
                raise ValueError(f"Conformal rejection is not alpha-monotone: {prefix}")
        calibration_scores = np.load(run / "calibration_scores.npy", allow_pickle=False)
        if calibration_scores.shape != (1000,) or not np.isfinite(calibration_scores).all():
            raise ValueError("CLMSG calibration score artifact is invalid")
        total_outputs += len(metrics)
        total_prediction_rows += sum(len(value) for value in by_method.values())
        total_calibration_scores += int(calibration_scores.size)
        verified_seeds.append(int(run.name.removeprefix("seed_")))
    decision = json.loads((root / "summary" / "stage_decision.json").read_text(encoding="utf-8"))
    if decision.get("manifold_or_entropy_authorized") or decision.get("full_sweep_authorized"):
        raise ValueError("CLMSG stop decision unexpectedly authorized a later stage")
    if decision.get("completed_seeds") != [13, 42, 87]:
        raise ValueError("CLMSG confirmation seed coverage is incomplete")
    with (root / "summary" / "all_runs.csv").open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != 78:
        raise ValueError(f"Expected 78 summarized outputs, found {len(summary_rows)}")
    return {
        "status": "ok",
        "authorized_outputs": total_outputs,
        "prediction_rows": total_prediction_rows,
        "test_samples_per_method": 6000,
        "calibration_scores": total_calibration_scores,
        "conformal_mode_groups_per_seed": 5,
        "verified_seeds": verified_seeds,
        "sample_id_sets_disjoint": True,
        "test_used_for_selection": False,
        "later_stages_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    result = verify((args.input_dir or paths.run_root / "clmsg_v1").resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
