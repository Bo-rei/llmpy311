"""Known-only search over the existing Banking77 Gate recipes and coverage.

This is a second, pre-declared selection campaign.  It does not read test
rows, real OOS rows, or pseudo-OOS rows.  The candidate operating point is
selected by the same Known-only selective utility used by the locked geometry
search, while expanding the fixed Known acceptance-coverage target.
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
    DATA_ROOT,
    DATASET,
    KIRS,
    MODEL_ROOT,
    RECIPE_ROOT,
    SEEDS,
    load_known,
    read,
    tag,
)
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder  # noqa: E402


OUT = ART_ROOT / "coverage_recipe_search_known_only"
COVERAGE_TARGETS = (0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def preference(key: tuple) -> tuple:
    k, distance, boundary, rule, fusion, weight = key
    return (
        int(k == 1),
        int(distance == "mahalanobis_diag"),
        int(boundary == "mean_std_1.0"),
        int(rule == "normalized_union"),
        int(fusion == "none"),
        int(float(weight) == 0.0),
    )


def recipe_records(kir: float) -> dict[str, dict[int, dict]]:
    selection = read(RECIPE_ROOT / f"selection_kir{round(kir * 100):02d}.json")
    records = {}
    for expanded in selection["expanded"]:
        name = str(expanded["recipe"])
        records[name] = {int(record["seed"]): record for record in expanded["records"]}
    if len(records) < 1 or any(set(values) != set(SEEDS) for values in records.values()):
        raise ValueError(f"expanded recipe records are incomplete for KIR={kir}")
    return records


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    coverage.FIXED_THRESHOLD = None
    manifest = {
        "status": "selecting",
        "experiment": "banking77_textoir_aligned_known_only_coverage_recipe_search",
        "dataset": DATASET,
        "dataset_variant": "standard_77_intent",
        "protocol": "protocol_v2_textoir_v1",
        "data_root": str(DATA_ROOT),
        "recipe_root": str(RECIPE_ROOT),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "coverage_targets": list(COVERAGE_TARGETS),
        "candidate_recipe_source": "top expanded recipes already selected on Known-dev only",
        "selection": (
            "joint recipe, coverage and geometry selection by mean Known-dev selective "
            "utility across three seeds; coverage target is a pre-declared candidate"
        ),
        "search_grid": {
            "k": list(coverage.K_VALUES),
            "distance": list(coverage.DISTANCES),
            "lambda": list(coverage.LAMBDA_VALUES),
            "boundary_quantile": list(coverage.BOUNDARY_QUANTILES),
            "rule": ["nearest_sphere", "normalized_union"],
            "fusion": list(coverage.FUSIONS),
            "fusion_weight": list(coverage.FUSION_WEIGHTS),
        },
        "utility": "mean(accepted * (1 if Known validation assignment is correct else -4))",
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
    all_choices = []
    lock_rows = []
    summary_rows = []

    for kir in KIRS:
        records_by_recipe = recipe_records(kir)
        recipe_choices = []
        for recipe, records in records_by_recipe.items():
            per_seed = {seed: {} for seed in SEEDS}
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
                del encoder
                torch.cuda.empty_cache()
                for target in COVERAGE_TARGETS:
                    coverage.COVERAGE_TARGET = float(target)
                    candidates = coverage.search_known(
                        train_values, validation_values, train, validation
                    )
                    per_seed[seed][target] = candidates
                    print(
                        f"KNOWN_COVERAGE {DATASET}/{tag(kir, seed)} "
                        f"recipe={recipe} target={target:.2f} candidates={len(candidates)}",
                        flush=True,
                    )

            for target in COVERAGE_TARGETS:
                keys = [coverage.geometry_key(row) for row in per_seed[SEEDS[0]][target]]
                if not all(
                    [
                        [coverage.geometry_key(row) for row in per_seed[seed][target]] == keys
                        for seed in SEEDS
                    ]
                ):
                    raise ValueError(f"geometry grids differ for {recipe}, KIR={kir}, target={target}")
                aggregates = []
                for index, key in enumerate(keys):
                    rows = [per_seed[seed][target][index] for seed in SEEDS]
                    aggregates.append(
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
                    aggregates,
                    key=lambda row: (
                        row["utility_mean"],
                        -row["wrong_accept_mean"],
                        preference(row["key"]),
                    ),
                )
                recipe_choices.append({"recipe": recipe, **winner})

        winner = max(
            recipe_choices,
            key=lambda row: (
                row["utility_mean"],
                -row["wrong_accept_mean"],
                preference(row["key"]),
                -list(records_by_recipe).index(row["recipe"]),
            ),
        )
        geometry = dict(zip(coverage.GEOMETRY_FIELDS, winner["key"], strict=True))
        choice = {
            "dataset": DATASET,
            "kir": kir,
            "recipes_considered": list(records_by_recipe),
            "coverage_targets": list(COVERAGE_TARGETS),
            "selection": "Known-dev only",
            "winner": {"recipe": winner["recipe"], "coverage_target": winner["coverage_target"], **geometry},
            "utility_mean": winner["utility_mean"],
            "wrong_accept_mean": winner["wrong_accept_mean"],
            "coverage_mean": winner["coverage_mean"],
            "candidate_count": len(recipe_choices),
        }
        dump(OUT / "validation" / f"{DATASET}_kir{round(kir * 100):02d}_choice.json", choice)
        all_choices.append(choice)
        selected_recipe = winner["recipe"]
        for record, selected in zip(
            [records_by_recipe[selected_recipe][seed] for seed in SEEDS],
            winner["rows"],
            strict=True,
        ):
            geometry_with_threshold = {
                **geometry,
                "coverage": float(winner["coverage_target"]),
                "threshold": float(selected["threshold"]),
            }
            lock = {
                "dataset": DATASET,
                "dataset_variant": "standard_77_intent",
                "protocol": "protocol_v2_textoir_v1",
                "kir": kir,
                "seed": int(record["seed"]),
                "recipe_name": selected_recipe,
                "recipe": record["recipe"],
                "checkpoint": record["checkpoint"],
                "geometry": geometry_with_threshold,
                "scoring_family": "coverage",
                "selection_evidence": {
                    "known_validation_f1": float(record["known_validation_f1"]),
                    "known_dev_utility_mean": winner["utility_mean"],
                    "known_dev_coverage": float(selected["known_coverage"]),
                    "known_dev_wrong_accept_rate": float(selected["known_wrong_accept_rate"]),
                    "recipe_and_coverage_selection": str(
                        OUT / "validation" / f"{DATASET}_kir{round(kir * 100):02d}_choice.json"
                    ),
                },
                "real_oos_used_for_training": False,
                "real_oos_used_for_selection": False,
                "pseudo_oos_used": False,
                "test_read": False,
                "test_used_for_selection": False,
            }
            lock_path = OUT / "locks" / f"{DATASET}_{tag(kir, int(record['seed']))}.json"
            dump(lock_path, lock)
            lock_rows.append(lock)
            summary_rows.append(
                {
                    "dataset": DATASET,
                    "kir": kir,
                    "seed": int(record["seed"]),
                    "recipe": selected_recipe,
                    "coverage_target": winner["coverage_target"],
                    **geometry,
                    "threshold": selected["threshold"],
                    "known_validation_f1": record["known_validation_f1"],
                    "known_dev_coverage": selected["known_coverage"],
                    "known_dev_wrong_accept_rate": selected["known_wrong_accept_rate"],
                }
            )
        print(
            f"COVERAGE_RECIPE_SELECTION banking77 KIR={kir:.2f} "
            f"recipe={selected_recipe} target={winner['coverage_target']:.2f} "
            f"K={geometry['k']} distance={geometry['distance']} "
            f"boundary={geometry['boundary']} utility={winner['utility_mean']:.6f}",
            flush=True,
        )

    with (OUT / "known_selection.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(summary_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest.update(status="locked", lock_count=len(lock_rows), choices=all_choices)
    dump(OUT / "MANIFEST.json", manifest)
    dump(
        OUT / "selection_complete.json",
        {"status": "locked", "units": len(lock_rows), "test_read": False, "test_used_for_selection": False},
    )
    print("BANKING_TEXTOIR_ALIGNED_COVERAGE_RECIPE_LOCKED", flush=True)


if __name__ == "__main__":
    main()
