"""Select an aligned Banking77 Gate geometry from Known validation only."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.experiments import finalize_historical_known_coverage as coverage  # noqa: E402
from tools.eval.run_historical_trainable_full_pipeline import (  # noqa: E402
    MODEL_ROOT,
    _RacalGateEncoder,
)


ART_ROOT = ROOT.parent / "artifacts" / "s2c" / "analysis" / "banking_textoir_aligned"
DATA_ROOT = ART_ROOT / "data"
RECIPE_ROOT = ART_ROOT / "recipe_search_corrected"
OUT = ART_ROOT / "final_selection"
DATASET = "banking77"
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87)
COVERAGE_TARGET = 0.90


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def tag(kir: float, seed: int) -> str:
    return f"kir{round(kir * 100):02d}_seed{seed}"


def load_known(kir: float, seed: int) -> tuple[list[dict], list[dict]]:
    folder = DATA_ROOT / DATASET / tag(kir, seed) / "gate"
    train = list(read(folder / "train.json"))
    validation = list(read(folder / "val.json"))
    known = {str(row["intent"]) for row in train}
    if not train or not validation or any(int(row["label"]) != 0 for row in train + validation):
        raise ValueError(f"invalid Known-only view: {folder}")
    if any(str(row["intent"]) not in known for row in validation):
        raise ValueError(f"validation contains unknown intent: {folder}")
    return train, validation


def record_path(kir: float, seed: int, recipe: str) -> Path:
    return RECIPE_ROOT / f"{DATASET}_{tag(kir, seed)}_{recipe}.json"


def choose_preference(key: tuple) -> tuple:
    # Utility is the primary Known-only criterion.  This tie-break keeps a
    # duplicate fusion-none candidate and equally useful geometries simple.
    k, distance, boundary, rule, fusion, weight = key
    return (
        int(k == 1),
        int(distance == "mahalanobis_diag"),
        int(boundary == "mean_std_1.0"),
        int(rule == "normalized_union"),
        int(fusion == "none"),
        int(float(weight) == 0.0),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=float, default=COVERAGE_TARGET)
    args = parser.parse_args()
    if not 0.0 < args.coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    if OUT.exists():
        manifest_path = OUT / "MANIFEST.json"
        if not manifest_path.is_file() or read(manifest_path).get("status") != "locked":
            raise FileExistsError(f"refusing to overwrite incomplete selection: {OUT}")
        print("BANKING_TEXT0IR_ALIGNED_GEOMETRY_ALREADY_LOCKED", flush=True)
        return

    coverage.COVERAGE_TARGET = float(args.coverage)
    coverage.FIXED_THRESHOLD = None
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "selecting",
        "experiment": "banking77_textoir_aligned_known_only_geometry_selection",
        "dataset": DATASET,
        "dataset_variant": "standard_77_intent",
        "protocol": "protocol_v2_textoir_v1",
        "data_root": str(DATA_ROOT),
        "recipe_root": str(RECIPE_ROOT),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "coverage_target": float(args.coverage),
        "search_grid": {
            "k": list(coverage.K_VALUES),
            "distance": list(coverage.DISTANCES),
            "lambda": list(coverage.LAMBDA_VALUES),
            "boundary_quantile": list(coverage.BOUNDARY_QUANTILES),
            "rule": ["nearest_sphere", "normalized_union"],
            "fusion": list(coverage.FUSIONS),
            "fusion_weight": list(coverage.FUSION_WEIGHTS),
        },
        "recipe_selection": "mean Known-dev macro F1 across seeds from recipe_search_corrected",
        "geometry_selection": "mean Known-dev utility at fixed Known acceptance coverage across seeds",
        "real_oos_used_for_training": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_read": False,
        "test_used_for_selection": False,
        "device": "cuda",
    }
    dump(OUT / "MANIFEST.json", manifest)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    locks = []
    summary_rows = []
    for kir in KIRS:
        selection = read(RECIPE_ROOT / f"selection_kir{round(kir * 100):02d}.json")
        recipe = str(selection["winner"])
        candidate_groups = []
        records = []
        for seed in SEEDS:
            record = read(record_path(kir, seed, recipe))
            if not (
                record.get("protocol") == "protocol_v2_textoir_v1"
                and record.get("test_read") is False
                and record.get("test_used_for_selection") is False
                and record.get("real_oos_used") is False
                and record.get("pseudo_oos_used") is False
            ):
                raise ValueError(f"non-pure Known-only recipe record: {record_path(kir, seed, recipe)}")
            train, validation = load_known(kir, seed)
            encoder = _RacalGateEncoder(
                MODEL_ROOT / "all-MiniLM-L6-v2",
                Path(record["checkpoint"]),
                device,
            )
            train_values = encoder.encode([row["text"] for row in train], batch_size=256)
            validation_values = encoder.encode([row["text"] for row in validation], batch_size=256)
            candidates = coverage.search_known(train_values, validation_values, train, validation)
            dump(OUT / "validation" / f"{DATASET}_{tag(kir, seed)}_{recipe}.json", candidates)
            candidate_groups.append(candidates)
            records.append(record)
            del encoder
            torch.cuda.empty_cache()
            print(f"KNOWN_GEOMETRY {DATASET}/{tag(kir, seed)} recipe={recipe} candidates={len(candidates)}", flush=True)

        keys = [coverage.geometry_key(row) for row in candidate_groups[0]]
        if not all([coverage.geometry_key(row) for row in group] == keys for group in candidate_groups):
            raise ValueError(f"geometry grids differ for KIR={kir}")
        aggregates = []
        for index, key in enumerate(keys):
            rows = [group[index] for group in candidate_groups]
            aggregates.append(
                {
                    "key": key,
                    "utility_mean": float(np.mean([row["utility"] for row in rows])),
                    "wrong_accept_mean": float(np.mean([row["known_wrong_accept_rate"] for row in rows])),
                    "coverage_mean": float(np.mean([row["known_coverage"] for row in rows])),
                    "rows": rows,
                }
            )
        winner = max(
            aggregates,
            key=lambda row: (
                row["utility_mean"],
                -row["wrong_accept_mean"],
                choose_preference(row["key"]),
            ),
        )
        geometry = dict(zip(coverage.GEOMETRY_FIELDS, winner["key"], strict=True))
        choice = {
            "dataset": DATASET,
            "kir": kir,
            "recipe": recipe,
            "selection": "Known-dev only",
            "candidate_count": len(aggregates),
            "winner": geometry,
            "utility_mean": winner["utility_mean"],
            "wrong_accept_mean": winner["wrong_accept_mean"],
            "coverage_mean": winner["coverage_mean"],
        }
        dump(OUT / "validation" / f"{DATASET}_kir{round(kir * 100):02d}_geometry_choice.json", choice)
        for record, group in zip(records, winner["rows"], strict=True):
            lock = {
                "dataset": DATASET,
                "dataset_variant": "standard_77_intent",
                "protocol": "protocol_v2_textoir_v1",
                "kir": kir,
                "seed": int(record["seed"]),
                "recipe_name": recipe,
                "recipe": record["recipe"],
                "checkpoint": record["checkpoint"],
                "geometry": {
                    **geometry,
                    "coverage": float(args.coverage),
                    "threshold": float(group["threshold"]),
                },
                "scoring_family": "coverage",
                "selection_evidence": {
                    "known_validation_f1": float(record["known_validation_f1"]),
                    "known_dev_utility_mean": winner["utility_mean"],
                    "known_dev_coverage": float(group["known_coverage"]),
                    "known_dev_wrong_accept_rate": float(group["known_wrong_accept_rate"]),
                    "recipe_selection": str(RECIPE_ROOT / f"selection_kir{round(kir * 100):02d}.json"),
                    "geometry_selection": str(OUT / "validation" / f"{DATASET}_kir{round(kir * 100):02d}_geometry_choice.json"),
                },
                "real_oos_used_for_training": False,
                "real_oos_used_for_selection": False,
                "pseudo_oos_used": False,
                "test_read": False,
                "test_used_for_selection": False,
            }
            lock_path = OUT / "locks" / f"{DATASET}_{tag(kir, int(record['seed']))}.json"
            dump(lock_path, lock)
            locks.append(lock)
            summary_rows.append({
                "dataset": DATASET,
                "kir": kir,
                "seed": int(record["seed"]),
                "recipe": recipe,
                **geometry,
                "threshold": float(group["threshold"]),
                "known_validation_f1": float(record["known_validation_f1"]),
                "known_dev_coverage": float(group["known_coverage"]),
                "known_dev_wrong_accept_rate": float(group["known_wrong_accept_rate"]),
            })
        print(
            f"GEOMETRY_SELECTION banking77 KIR={kir:.2f} recipe={recipe} "
            f"K={geometry['k']} distance={geometry['distance']} boundary={geometry['boundary']} "
            f"utility={winner['utility_mean']:.6f}",
            flush=True,
        )
    with (OUT / "known_selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest.update(status="locked", lock_count=len(locks), test_read=False, test_used_for_selection=False)
    dump(OUT / "MANIFEST.json", manifest)
    dump(OUT / "selection_complete.json", {"status": "locked", "units": len(locks), "test_read": False, "test_used_for_selection": False})
    print("BANKING_TEXTOIR_ALIGNED_GEOMETRY_LOCKED", flush=True)


if __name__ == "__main__":
    main()
