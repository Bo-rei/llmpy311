"""Lock a pre-declared Known-only Banking77 geometry candidate.

The default candidate reproduces the K=3 diagonal-Mahalanobis operating-point
hypothesis used by the earlier Banking77 holdout campaign, but on the
canonical TextOIR-aligned standard Banking77 views.  Geometry and coverage can
be changed only as explicitly pre-declared command-line candidates.  The
script reads only Known train/dev rows; test evaluation is a separate step.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.experiments import finalize_historical_known_coverage as coverage  # noqa: E402
from scripts.experiments.select_banking_textoir_aligned_geometry import (  # noqa: E402
    ART_ROOT,
    DATA_ROOT,
    KIRS,
    MODEL_ROOT,
    RECIPE_ROOT,
    SEEDS,
    load_known,
    read,
    tag,
)
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder  # noqa: E402


OUT = ART_ROOT / "fixed_k3_mahalanobis_coverage80_known_only"
COVERAGE_TARGET = 0.80
GEOMETRY = {
    "k": 3,
    "distance": "mahalanobis_diag",
    "boundary": "mean_std_1.0",
    "rule": "nearest_sphere",
    "fusion": "none",
    "fusion_weight": 0.0,
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def recipe_record(kir: float, seed: int) -> tuple[str, dict]:
    selection = read(RECIPE_ROOT / f"selection_kir{round(kir * 100):02d}.json")
    recipe = str(selection["winner"])
    for expanded in selection["expanded"]:
        if str(expanded["recipe"]) == recipe:
            for record in expanded["records"]:
                if int(record["seed"]) == seed:
                    return recipe, record
    raise FileNotFoundError(f"missing selected recipe record: KIR={kir}, seed={seed}")


def main() -> None:
    global OUT, COVERAGE_TARGET, GEOMETRY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--coverage", type=float, default=COVERAGE_TARGET)
    parser.add_argument("--k", type=int, default=GEOMETRY["k"])
    parser.add_argument("--distance", choices=("euclidean", "mahalanobis_diag"), default=GEOMETRY["distance"])
    parser.add_argument("--boundary", default=GEOMETRY["boundary"])
    parser.add_argument("--rule", choices=("nearest_sphere", "normalized_union"), default=GEOMETRY["rule"])
    parser.add_argument("--fusion", choices=("none", "ratio", "inverse_margin", "max_ratio"), default=GEOMETRY["fusion"])
    parser.add_argument("--fusion-weight", type=float, default=GEOMETRY["fusion_weight"])
    parser.add_argument("--fixed-threshold", type=float)
    args = parser.parse_args()
    if not 0.0 < args.coverage <= 1.0 or args.k < 1:
        raise ValueError("coverage must be in (0, 1] and k must be positive")
    OUT = args.output.resolve()
    COVERAGE_TARGET = float(args.coverage)
    GEOMETRY = {
        "k": int(args.k),
        "distance": str(args.distance),
        "boundary": str(args.boundary),
        "rule": str(args.rule),
        "fusion": str(args.fusion),
        "fusion_weight": float(args.fusion_weight),
    }
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.zeros(1, device=device)
    torch.set_num_threads(4)
    manifest = {
        "status": "locking",
        "experiment": "banking77_textoir_aligned_geometry_candidate_known_only",
        "dataset": "banking77",
        "dataset_variant": "standard_77_intent",
        "protocol": "protocol_v2_textoir_v1",
        "data_root": str(DATA_ROOT),
        "recipe_root": str(RECIPE_ROOT),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "coverage_target": COVERAGE_TARGET,
        "fixed_threshold": args.fixed_threshold,
        "geometry": GEOMETRY,
        "recipe_selection": "existing Known-dev-only winner for each KIR",
        "threshold_selection": (
            "fixed pre-declared normalized score threshold"
            if args.fixed_threshold is not None
            else "per-seed higher quantile on Known validation scores"
        ),
        "hypothesis_source": "pre-declared Known-only geometry/coverage candidate",
        "real_oos_used_for_training": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_read": False,
        "test_used_for_selection": False,
        "device": str(device),
    }
    dump(OUT / "MANIFEST.json", manifest)
    rows = []
    for kir in KIRS:
        for seed in SEEDS:
            recipe, record = recipe_record(kir, seed)
            train, validation = load_known(kir, seed)
            encoder = _RacalGateEncoder(
                MODEL_ROOT / "all-MiniLM-L6-v2",
                Path(record["checkpoint"]),
                device,
            )
            train_values = encoder.encode([row["text"] for row in train], batch_size=256)
            validation_values = encoder.encode(
                [row["text"] for row in validation], batch_size=256
            )
            detector, output = coverage._fit_and_score(
                train_values, train, validation_values, GEOMETRY
            )
            threshold = (
                float(args.fixed_threshold)
                if args.fixed_threshold is not None
                else float(np.quantile(output["score"], COVERAGE_TARGET, method="higher"))
            )
            accepted = output["score"] <= threshold
            known = np.asarray([str(row["intent"]) for row in validation])
            owners = np.asarray(
                [str(detector.cluster_to_intent[int(item)]) for item in output["nearest_cluster"]]
            )
            lock = {
                "dataset": "banking77",
                "dataset_variant": "standard_77_intent",
                "protocol": "protocol_v2_textoir_v1",
                "kir": kir,
                "seed": seed,
                "recipe_name": recipe,
                "recipe": record["recipe"],
                "epoch": int(record["epoch"]),
                "checkpoint": record["checkpoint"],
                "geometry": {
                    **GEOMETRY,
                    "coverage": COVERAGE_TARGET,
                    "threshold": threshold,
                },
                "scoring_family": "coverage",
                "selection_evidence": {
                    "known_validation_f1": float(record["known_validation_f1"]),
                    "known_dev_coverage": float(np.mean(accepted)),
                    "known_dev_wrong_accept_rate": float(
                        np.mean(accepted & (owners != known))
                    ),
                    "selection_rule": (
                        "fixed geometry and pre-declared normalized score threshold"
                        if args.fixed_threshold is not None
                        else "fixed geometry and 80% Known-dev coverage"
                    ),
                    "test_read": False,
                },
                "real_oos_used_for_training": False,
                "real_oos_used_for_selection": False,
                "pseudo_oos_used": False,
                "test_read": False,
                "test_used_for_selection": False,
            }
            dump(OUT / "locks" / f"banking77_{tag(kir, seed)}.json", lock)
            rows.append(
                {
                    "dataset": "banking77",
                    "kir": kir,
                    "seed": seed,
                    "recipe": recipe,
                    **GEOMETRY,
                    "coverage_target": COVERAGE_TARGET,
                    "threshold": threshold,
                    "known_validation_f1": record["known_validation_f1"],
                    "known_dev_coverage": float(np.mean(accepted)),
                    "known_dev_wrong_accept_rate": float(
                        np.mean(accepted & (owners != known))
                    ),
                }
            )
            del encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                f"LOCK {tag(kir, seed)} threshold={threshold:.6f} "
                f"coverage={np.mean(accepted):.6f}",
                flush=True,
            )
    with (OUT / "known_selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest.update(status="locked", units=len(rows))
    dump(OUT / "MANIFEST.json", manifest)
    dump(
        OUT / "selection_complete.json",
        {"status": "locked", "units": len(rows), "test_read": False, "test_used_for_selection": False},
    )
    print("BANKING_TEXTOIR_ALIGNED_K3_MAHALANOBIS_COVERAGE80_LOCKED", flush=True)


if __name__ == "__main__":
    main()
