"""Known-only high-purity operating-point search for aligned Banking77.

This campaign is intentionally separate from the existing coverage search.  It
uses only Known train/dev rows and a fixed, pre-registered penalty for accepting
a Known validation example with the wrong intent.  The coverage target remains
an explicit candidate in the Known-only selection; no test or OOS row is read
by this script.
"""

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
from scripts.experiments.select_banking_textoir_aligned_geometry import (  # noqa: E402
    ART_ROOT,
    DATASET,
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


OUT = ART_ROOT / "known_precision_search_known_only"
WRONG_ACCEPT_PENALTY = 20.0
COVERAGE_TARGETS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def recipe_records(kir: float) -> dict[str, dict[int, dict]]:
    selection = read(RECIPE_ROOT / f"selection_kir{round(kir * 100):02d}.json")
    records = {}
    for expanded in selection["expanded"]:
        name = str(expanded["recipe"])
        records[name] = {int(record["seed"]): record for record in expanded["records"]}
    if not records or any(set(value) != set(SEEDS) for value in records.values()):
        raise ValueError(f"incomplete Known-only recipe records for KIR={kir}")
    return records


def geometry_key(row: dict) -> tuple:
    return tuple(row[field] for field in coverage.GEOMETRY_FIELDS)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    coverage.FIXED_THRESHOLD = None
    coverage.KNOWN_WRONG_ACCEPT_PENALTY = WRONG_ACCEPT_PENALTY
    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.zeros(1, device=device)

    manifest = {
        "status": "selecting",
        "experiment": "banking77_textoir_aligned_known_only_precision_search",
        "dataset": DATASET,
        "dataset_variant": "standard_77_intent",
        "protocol": "protocol_v2_textoir_v1",
        "data_root": str(DATA_ROOT),
        "recipe_root": str(RECIPE_ROOT),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "coverage_targets": list(COVERAGE_TARGETS),
        "wrong_accept_penalty": WRONG_ACCEPT_PENALTY,
        "selection": (
            "joint recipe, geometry and coverage selection by mean Known-dev "
            "utility across seeds; utility rewards correct acceptance and penalizes "
            "wrong Known acceptance"
        ),
        "utility": "mean(accepted * (1 if Known validation assignment is correct else -20))",
        "candidate_recipe_source": "the two recipes already expanded under Known-dev selection",
        "real_oos_used_for_training": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_read": False,
        "test_used_for_selection": False,
        "device": "cuda",
    }
    dump(OUT / "MANIFEST.json", manifest)
    all_choices = []
    lock_rows = []
    csv_rows = []

    for kir in KIRS:
        records_by_recipe = recipe_records(kir)
        candidates_by_recipe: dict[str, dict[int, dict[float, list[dict]]]] = {
            recipe: {seed: {} for seed in SEEDS} for recipe in records_by_recipe
        }
        for recipe, records in records_by_recipe.items():
            for seed in SEEDS:
                record = records[seed]
                required = {
                    "protocol": "protocol_v2_textoir_v1",
                    "dataset_variant": "standard_77_intent",
                    "test_read": False,
                    "test_used_for_selection": False,
                    "real_oos_used": False,
                    "pseudo_oos_used": False,
                }
                if any(record.get(key) != value for key, value in required.items()):
                    raise ValueError(f"non-Known-only recipe record: {record['path']}")
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
                for target in COVERAGE_TARGETS:
                    coverage.COVERAGE_TARGET = float(target)
                    candidates_by_recipe[recipe][seed][target] = coverage.search_known(
                        train_values, validation_values, train, validation
                    )
                    print(
                        f"KNOWN_PRECISION {DATASET}/{tag(kir, seed)} "
                        f"recipe={recipe} target={target:.2f} "
                        f"candidates={len(candidates_by_recipe[recipe][seed][target])}",
                        flush=True,
                    )
                del encoder
                torch.cuda.empty_cache()

        choices = []
        for recipe, records in records_by_recipe.items():
            recipe_candidates = []
            for target in COVERAGE_TARGETS:
                groups = [candidates_by_recipe[recipe][seed][target] for seed in SEEDS]
                keys = [geometry_key(row) for row in groups[0]]
                if not all([geometry_key(row) for row in group] == keys for group in groups):
                    raise ValueError(f"geometry grid differs for {recipe}, KIR={kir}, target={target}")
                aggregate = []
                for index, key in enumerate(keys):
                    rows = [group[index] for group in groups]
                    aggregate.append(
                        {
                            "key": key,
                            "coverage_target": target,
                            "utility_mean": float(np.mean([row["utility"] for row in rows])),
                            "wrong_accept_mean": float(
                                np.mean([row["known_wrong_accept_rate"] for row in rows])
                            ),
                            "coverage_mean": float(
                                np.mean([row["known_coverage"] for row in rows])
                            ),
                            "rows": rows,
                        }
                    )
                winner = max(
                    aggregate,
                    key=lambda row: (row["utility_mean"], -row["wrong_accept_mean"]),
                )
                recipe_candidates.append({"recipe": recipe, **winner})
            choices.extend(recipe_candidates)

        winner = max(
            choices,
            key=lambda row: (
                row["utility_mean"],
                -row["wrong_accept_mean"],
                -abs(row["coverage_target"] - 0.75),
                -list(records_by_recipe).index(row["recipe"]),
            ),
        )
        geometry = dict(zip(coverage.GEOMETRY_FIELDS, winner["key"], strict=True))
        choice = {
            "dataset": DATASET,
            "kir": kir,
            "selection": "Known-dev only",
            "wrong_accept_penalty": WRONG_ACCEPT_PENALTY,
            "coverage_targets": list(COVERAGE_TARGETS),
            "recipes_considered": list(records_by_recipe),
            "winner": {
                "recipe": winner["recipe"],
                "coverage_target": winner["coverage_target"],
                **geometry,
            },
            "utility_mean": winner["utility_mean"],
            "wrong_accept_mean": winner["wrong_accept_mean"],
            "coverage_mean": winner["coverage_mean"],
            "candidate_count": len(choices),
        }
        dump(OUT / "validation" / f"{DATASET}_kir{round(kir * 100):02d}_choices.json", choices)
        dump(OUT / "validation" / f"{DATASET}_kir{round(kir * 100):02d}_choice.json", choice)
        all_choices.append(choice)

        selected_records = records_by_recipe[winner["recipe"]]
        for record, selected in zip(
            [selected_records[seed] for seed in SEEDS], winner["rows"], strict=True
        ):
            lock = {
                "dataset": DATASET,
                "dataset_variant": "standard_77_intent",
                "protocol": "protocol_v2_textoir_v1",
                "kir": kir,
                "seed": int(record["seed"]),
                "recipe_name": winner["recipe"],
                "recipe": record["recipe"],
                "epoch": int(record["epoch"]),
                "checkpoint": record["checkpoint"],
                "geometry": {
                    **geometry,
                    "coverage": float(winner["coverage_target"]),
                    "threshold": float(selected["threshold"]),
                },
                "selection_contract": str(OUT / "MANIFEST.json"),
                "selection_evidence": {
                    "known_validation_f1": float(record["known_validation_f1"]),
                    "known_dev_utility_mean": winner["utility_mean"],
                    "known_dev_coverage": float(selected["known_coverage"]),
                    "known_dev_wrong_accept_rate": float(
                        selected["known_wrong_accept_rate"]
                    ),
                    "wrong_accept_penalty": WRONG_ACCEPT_PENALTY,
                    "coverage_targets": list(COVERAGE_TARGETS),
                    "recipe_pool": list(records_by_recipe),
                },
                "real_oos_used_for_training": False,
                "real_oos_used_for_selection": False,
                "pseudo_oos_used": False,
                "test_read": False,
                "test_used_for_selection": False,
            }
            dump(OUT / "locks" / f"{DATASET}_{tag(kir, int(record['seed']))}.json", lock)
            lock_rows.append(lock)
            csv_rows.append(
                {
                    "dataset": DATASET,
                    "kir": kir,
                    "seed": int(record["seed"]),
                    "recipe": winner["recipe"],
                    "wrong_accept_penalty": WRONG_ACCEPT_PENALTY,
                    "coverage_target": winner["coverage_target"],
                    **geometry,
                    "threshold": selected["threshold"],
                    "known_validation_f1": record["known_validation_f1"],
                    "known_dev_coverage": selected["known_coverage"],
                    "known_dev_wrong_accept_rate": selected["known_wrong_accept_rate"],
                }
            )
        print(
            f"KNOWN_PRECISION_LOCK {DATASET} KIR={kir:.2f} "
            f"recipe={winner['recipe']} target={winner['coverage_target']:.2f} "
            f"K={geometry['k']} distance={geometry['distance']} "
            f"utility={winner['utility_mean']:.6f}",
            flush=True,
        )

    with (OUT / "known_selection.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(csv_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    manifest.update(status="locked", choices=all_choices, lock_count=len(lock_rows))
    dump(OUT / "MANIFEST.json", manifest)
    dump(
        OUT / "selection_complete.json",
        {
            "status": "locked",
            "units": len(lock_rows),
            "test_read": False,
            "test_used_for_selection": False,
        },
    )
    print("BANKING_TEXTOIR_ALIGNED_KNOWN_PRECISION_LOCKED", flush=True)


if __name__ == "__main__":
    main()
