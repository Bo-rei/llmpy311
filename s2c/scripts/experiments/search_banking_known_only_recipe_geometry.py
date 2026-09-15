"""Pre-registered Known-only recipe and geometry search on standard Banking77.

The search stage reads only the materialized Known train/dev views.  It locks
the recipe, geometry and per-seed Known-dev coverage threshold before the
separate evaluation stage is allowed to read test rows.  The coverage targets
are explicit operating-point hypotheses, not values selected from test OOS.

This experiment is intentionally separate from ``banking_fixed_known`` and
``coverage_repair``.  It reuses checkpoints whose training records declare
``test_read=false`` and expands the boundary operating-point analysis without
mutating either previous result bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.experiments import finalize_historical_known_coverage as coverage  # noqa: E402
from scripts.experiments.search_historical_known_training import RECIPES  # noqa: E402
from tools.eval.run_historical_trainable_full_pipeline import (  # noqa: E402
    MODEL_ROOT,
    _RacalGateEncoder,
)


ART = ROOT.parent / "artifacts" / "s2c" / "analysis" / "kir_sensitivity_known_only"
DATA_ROOT = ART / "data"
OUT = ART / "banking_known_recipe_geometry_search"
DATASET = "banking77"
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87)

# These targets are registered before any test evaluation.  0.75 is the
# aggressive OOS-priority hypothesis requested for this follow-up; the other
# targets are sensitivity controls, not a test-selected ensemble.
COVERAGE_TARGETS = (0.70, 0.75, 0.80, 0.85)
RECIPE_ORDER = (
    "last2",
    "last1",
    "last4",
    "lr_low",
    "projection128",
    "temperature10",
)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def tag(kir: float, seed: int) -> str:
    return f"kir{round(kir * 100):02d}_seed{seed}"


def record_path(kir: float, seed: int, recipe: str) -> Path:
    name = f"banking77_{tag(kir, seed)}_{recipe}.json"
    candidates = [
        ART / "training" / name,
        ART.parent / "banking_known_holdout" / "training_records" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no Known-only training record for {name}")


def load_views(kir: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folder = DATA_ROOT / DATASET / tag(kir, seed) / "gate"
    train = read(folder / "train.json")
    validation = read(folder / "val.json")
    known = {str(row["intent"]) for row in train}
    validation = [row for row in validation if str(row["intent"]) in known]
    if not train or not validation:
        raise ValueError(f"empty Known view: {folder}")
    if any(int(row["label"]) != 0 for row in train + validation):
        raise ValueError(f"non-Known row entered selection view: {folder}")
    return train, validation


def complete_recipes(kir: float) -> list[str]:
    names = []
    for name in RECIPE_ORDER:
        try:
            paths = [record_path(kir, seed, name) for seed in SEEDS]
        except FileNotFoundError:
            continue
        records = [read(path) for path in paths]
        if all(
            record.get("dataset") == DATASET
            and float(record.get("kir")) == float(kir)
            and int(record.get("seed")) == seed
            and record.get("test_read") is False
            and record.get("real_oos_used") is False
            and record.get("pseudo_oos_used") is False
            and Path(record["checkpoint"]).is_file()
            for record, seed in zip(records, SEEDS, strict=True)
        ):
            names.append(name)
    if not names:
        raise RuntimeError(f"no complete Known-only recipe for KIR={kir}")
    return names


def select_one_target(target: float, manifest: dict[str, Any]) -> None:
    target_root = OUT / f"coverage{round(target * 100):02d}"
    if target_root.exists():
        existing = {path.name for path in target_root.iterdir()}
        if existing - {"MANIFEST.json"}:
            raise FileExistsError(f"refusing to overwrite non-empty {target_root}")
    else:
        target_root.mkdir(parents=True)
    target_manifest = {
        "status": "selecting",
        "dataset": DATASET,
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "coverage_target": target,
        "protocol": "shared_known_labels_textoir_source",
        "data_source": "textoir/data/banking",
        "selection": (
            "joint recipe/geometry selection by mean Known-dev utility at "
            "fixed pre-registered Known coverage; per-seed quantile threshold"
        ),
        "recipe_pool": {},
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_used_for_selection": False,
        "test_read": False,
        "prior_test_results_exist": True,
        "parent_manifest": str(OUT / "MANIFEST.json"),
    }
    dump(target_root / "MANIFEST.json", target_manifest)

    for kir in KIRS:
        recipes = complete_recipes(kir)
        target_manifest["recipe_pool"][str(kir)] = recipes
        recipe_choices: list[dict[str, Any]] = []
        encoded: dict[tuple[str, int], tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]] = {}

        # Recipe records are already selected from Known validation.  Geometry
        # is then selected from the same Known train/dev rows.  No test path is
        # constructed in this function.
        for recipe in recipes:
            records = []
            utilities_by_seed = []
            geometry_by_seed: list[list[dict[str, Any]]] = []
            for seed in SEEDS:
                record_file = record_path(kir, seed, recipe)
                record = read(record_file)
                records.append({"path": str(record_file), **record})
                train, validation = load_views(kir, seed)
                encoder = _RacalGateEncoder(
                    MODEL_ROOT / "all-MiniLM-L6-v2",
                    Path(record["checkpoint"]),
                    device=__import__("torch").device("cuda"),
                )
                train_values = encoder.encode(
                    [row["text"] for row in train], batch_size=256
                )
                validation_values = encoder.encode(
                    [row["text"] for row in validation], batch_size=256
                )
                encoded[(recipe, seed)] = (train, validation, train_values, validation_values)
                coverage.COVERAGE_TARGET = float(target)
                coverage.FIXED_THRESHOLD = None
                candidates = coverage.search_known(
                    train_values, validation_values, train, validation
                )
                dump(
                    target_root
                    / "validation"
                    / f"{DATASET}_{tag(kir, seed)}_{recipe}.json",
                    candidates,
                )
                geometry_by_seed.append(candidates)
                utilities_by_seed.append(float(record["known_validation_f1"]))
                del encoder
                import torch

                torch.cuda.empty_cache()

            keys = [coverage.geometry_key(row) for row in geometry_by_seed[0]]
            if not all(
                [coverage.geometry_key(row) for row in rows] == keys
                for rows in geometry_by_seed
            ):
                raise AssertionError("geometry candidate order differs by seed")
            aggregate = []
            for index, key in enumerate(keys):
                rows = [group[index] for group in geometry_by_seed]
                aggregate.append(
                    {
                        "key": key,
                        "utility": float(np.mean([row["utility"] for row in rows])),
                        "wrong_accept": float(
                            np.mean([row["known_wrong_accept_rate"] for row in rows])
                        ),
                        "coverage": float(
                            np.mean([row["known_coverage"] for row in rows])
                        ),
                        "rows": rows,
                    }
                )
            geometry_winner = max(
                aggregate,
                key=lambda row: (row["utility"], -row["wrong_accept"]),
            )
            recipe_choices.append(
                {
                    "recipe": recipe,
                    "known_validation_f1_mean": float(np.mean(utilities_by_seed)),
                    "geometry": dict(
                        zip(coverage.GEOMETRY_FIELDS, geometry_winner["key"], strict=True)
                    ),
                    "geometry_utility_mean": geometry_winner["utility"],
                    "geometry_wrong_accept_mean": geometry_winner["wrong_accept"],
                    "geometry_coverage_mean": geometry_winner["coverage"],
                    "records": records,
                    "selected_rows": geometry_winner["rows"],
                }
            )

        # The primary key is a Known-dev quantity.  The validation F1 tie
        # breaker favors the representation with better known classification;
        # fixed recipe order makes exact ties reproducible.
        winner = max(
            recipe_choices,
            key=lambda row: (
                row["geometry_utility_mean"],
                -row["geometry_wrong_accept_mean"],
                row["known_validation_f1_mean"],
            ),
        )
        dump(target_root / "validation" / f"{DATASET}_kir{round(kir * 100):02d}_choices.json", recipe_choices)
        locks = []
        for record, selected in zip(winner["records"], winner["selected_rows"], strict=True):
            geometry = {
                **winner["geometry"],
                "coverage": float(target),
                "threshold": float(selected["threshold"]),
            }
            lock = {
                "dataset": DATASET,
                "kir": kir,
                "seed": int(record["seed"]),
                "recipe_name": winner["recipe"],
                "recipe": record["recipe"],
                "epoch": int(record["epoch"]),
                "checkpoint": record["checkpoint"],
                "training_record": record["path"],
                "geometry": geometry,
                "selection_contract": str(target_root / "MANIFEST.json"),
                "selection_evidence": {
                    "known_validation_f1": float(record["known_validation_f1"]),
                    "known_dev_coverage": float(selected["known_coverage"]),
                    "known_correct_accept_rate": float(selected["known_correct_accept_rate"]),
                    "known_wrong_accept_rate": float(selected["known_wrong_accept_rate"]),
                    "geometry_utility_mean": winner["geometry_utility_mean"],
                    "recipe_pool": recipes,
                },
                "real_oos_used": False,
                "pseudo_oos_used": False,
                "test_used_for_selection": False,
                "test_read": False,
                "prior_test_results_exist": True,
            }
            locks.append(lock)
            dump(target_root / "locks" / f"{DATASET}_{tag(kir, int(record['seed']))}.json", lock)
        target_manifest.setdefault("locks", []).extend(locks)
        dump(target_root / "selection.json", target_manifest["locks"])
        print(
            f"LOCK coverage={target:.2f} KIR={kir:.2f} recipe={winner['recipe']} "
            f"K={winner['geometry']['k']} distance={winner['geometry']['distance']} "
            f"utility={winner['geometry_utility_mean']:.6f}",
            flush=True,
        )

    target_manifest.update(
        status="all_choices_locked",
        lock_count=len(target_manifest["locks"]),
        test_read=False,
        test_used_for_selection=False,
    )
    dump(target_root / "MANIFEST.json", target_manifest)
    dump(target_root / "selection_complete.json", {
        "status": "locked",
        "units": len(target_manifest["locks"]),
        "coverage_target": target,
        "test_read": False,
        "test_used_for_selection": False,
    })


def write_summary(target_root: Path, target: float) -> None:
    rows = []
    for lock_path in sorted((target_root / "locks").glob("*.json")):
        lock = read(lock_path)
        result = read(target_root / "final" / DATASET / tag(float(lock["kir"]), int(lock["seed"])) / "Ours.json")
        metrics = result["metrics"]
        rows.append({
            "dataset": DATASET,
            "kir": float(lock["kir"]),
            "seed": int(lock["seed"]),
            "coverage_target": target,
            "recipe": lock["recipe_name"],
            "k": lock["geometry"]["k"],
            "distance": lock["geometry"]["distance"],
            "boundary": lock["geometry"]["boundary"],
            "rule": lock["geometry"]["rule"],
            "fusion": lock["geometry"]["fusion"],
            "fusion_weight": lock["geometry"]["fusion_weight"],
            "threshold": lock["geometry"]["threshold"],
            "oos_f1": 100.0 * metrics["oos_f1"],
            "known_f1": 100.0 * metrics["known_f1"],
            "accuracy": 100.0 * metrics["accuracy"],
            "known_recall": 100.0 * metrics["known_recall"],
            "false_accept_rate": 100.0 * metrics["false_accept_rate"],
            "oos_recall": 100.0 * metrics["oos_recall"],
        })
    fields = list(rows[0]) if rows else []
    with (target_root / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for kir in KIRS:
        group = [row for row in rows if row["kir"] == kir]
        if len(group) != len(SEEDS):
            raise ValueError(f"missing evaluated rows for KIR={kir}")
        row = {
            "dataset": DATASET,
            "kir": kir,
            "coverage_target": target,
            "recipe": group[0]["recipe"],
            "k": group[0]["k"],
            "distance": group[0]["distance"],
            "boundary": group[0]["boundary"],
            "rule": group[0]["rule"],
            "fusion": group[0]["fusion"],
            "fusion_weight": group[0]["fusion_weight"],
        }
        for metric in ("oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate", "oos_recall"):
            values = np.asarray([item[metric] for item in group], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        summary.append(row)
    fields = list(summary[0]) if summary else []
    with (target_root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def evaluate() -> None:
    script = ROOT / "scripts" / "experiments" / "evaluate_kir_ours.py"
    for target in COVERAGE_TARGETS:
        target_root = OUT / f"coverage{round(target * 100):02d}"
        manifest = read(target_root / "MANIFEST.json")
        if manifest.get("status") != "all_choices_locked":
            raise RuntimeError(f"selection is not locked: {target_root}")
        if manifest.get("test_read") is not False:
            raise RuntimeError(f"test was read before evaluation stage: {target_root}")
    for target in COVERAGE_TARGETS:
        target_root = OUT / f"coverage{round(target * 100):02d}"
        command = [
            sys.executable,
            str(script),
            "--datasets",
            DATASET,
            "--kirs",
            *[str(kir) for kir in KIRS],
            "--seeds",
            *[str(seed) for seed in SEEDS],
            "--selection-root",
            str(target_root),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        write_summary(target_root, target)
        manifest = read(target_root / "MANIFEST.json")
        manifest.update(
            status="complete",
            test_read=True,
            test_used_for_selection=False,
            completed_test_units=len(SEEDS) * len(KIRS),
            evaluator="scripts/experiments/evaluate_kir_ours.py",
            evaluator_command=command,
        )
        dump(target_root / "MANIFEST.json", manifest)

    root_manifest = read(OUT / "MANIFEST.json")
    root_manifest.update(
        status="complete",
        selection_complete=True,
        evaluation_complete=True,
        test_read=True,
        test_used_for_selection=False,
        target_status={
            str(target): read(OUT / f"coverage{round(target * 100):02d}" / "MANIFEST.json")["status"]
            for target in COVERAGE_TARGETS
        },
    )
    dump(OUT / "MANIFEST.json", root_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "evaluate"), required=True)
    args = parser.parse_args()
    if args.stage == "select":
        if OUT.exists():
            manifest = read(OUT / "MANIFEST.json")
            if manifest.get("status") != "selecting" or manifest.get("test_read") is not False:
                raise FileExistsError(f"refusing to overwrite completed {OUT}")
        else:
            OUT.mkdir(parents=True)
            manifest = {
                "status": "selecting",
                "experiment": "banking_known_recipe_geometry_search",
                "dataset": DATASET,
                "kirs": list(KIRS),
                "seeds": list(SEEDS),
                "coverage_targets": list(COVERAGE_TARGETS),
                "protocol": "shared_known_labels_textoir_source",
                "source": "textoir/data/banking",
                "recipe_order": list(RECIPE_ORDER),
                "geometry_grid": {
                    "k": list(coverage.K_VALUES),
                    "distance": list(coverage.DISTANCES),
                    "lambda": list(coverage.LAMBDA_VALUES),
                    "boundary_quantile": list(coverage.BOUNDARY_QUANTILES),
                    "rule": ["nearest_sphere", "normalized_union"],
                    "fusion": list(coverage.FUSIONS),
                    "fusion_weight": list(coverage.FUSION_WEIGHTS),
                },
                "selection": "Known-dev only; fixed pre-registered coverage targets; no pseudo-OOS",
                "real_oos_used_for_selection": False,
                "pseudo_oos_used": False,
                "test_used_for_selection": False,
                "test_read": False,
                "prior_test_results_exist": True,
                "device": "cuda",
            }
        dump(OUT / "MANIFEST.json", manifest)
        for target in COVERAGE_TARGETS:
            select_one_target(target, manifest)
        manifest["status"] = "all_choices_locked"
        manifest["selection_complete"] = True
        manifest["test_read"] = False
        dump(OUT / "MANIFEST.json", manifest)
        print(f"SELECTION_LOCKED targets={len(COVERAGE_TARGETS)} units={len(COVERAGE_TARGETS) * len(KIRS) * len(SEEDS)}")
    else:
        if not OUT.exists():
            raise FileNotFoundError(f"missing selection root: {OUT}")
        evaluate()
        print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
