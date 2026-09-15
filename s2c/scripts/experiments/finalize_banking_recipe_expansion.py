"""Lock and evaluate the expanded standard-Banking77 recipe search.

Selection is a two-stage Known-only contract: the recipe is selected from
Known-dev macro F1, and the Gate geometry is selected from Known-dev utility
at a fixed 90% Known coverage.  The final evaluator is invoked only after all
locks exist.  No test rows are opened by the selection stage.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.experiments import finalize_historical_known_coverage as coverage  # noqa: E402
from tools.eval.run_historical_trainable_full_pipeline import (  # noqa: E402
    MODEL_ROOT,
    _RacalGateEncoder,
)


ART = ROOT.parent / "artifacts" / "s2c" / "analysis" / "kir_sensitivity_known_only"
DATA_ROOT = ART / "data"
EXPANSION = ART / "banking_recipe_expansion"
OUT = ART / "banking_recipe_expansion_final"
DATASET = "banking77"
KIRS = (0.50, 0.75)
SEEDS = (13, 42, 87)
COVERAGE_TARGET = 0.90


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read(path: Path) -> object:
    return json.loads(path.read_text())


def tag(kir: float, seed: int) -> str:
    return f"kir{round(kir * 100):02d}_seed{seed}"


def record_for(kir: float, seed: int, recipe: str) -> dict:
    if recipe in {"last1", "last2", "last4"}:
        path = ART / "training" / f"banking77_{tag(kir, seed)}_{recipe}.json"
    else:
        path = EXPANSION / f"banking77_{tag(kir, seed)}_{recipe}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = dict(read(path))
    if not (
        value.get("dataset") == DATASET
        and float(value["kir"]) == kir
        and int(value["seed"]) == seed
        and value.get("test_read") is False
        and value.get("real_oos_used") is False
        and value.get("pseudo_oos_used") is False
        and Path(value["checkpoint"]).is_file()
    ):
        raise ValueError(f"record is not a pure Known-only training record: {path}")
    return {"path": str(path), **value}


def load_known(kir: float, seed: int) -> tuple[list[dict], list[dict]]:
    folder = DATA_ROOT / DATASET / tag(kir, seed) / "gate"
    train = list(read(folder / "train.json"))
    validation = [row for row in list(read(folder / "val.json")) if row["intent"] in {r["intent"] for r in train}]
    if not train or not validation or any(int(row["label"]) != 0 for row in train + validation):
        raise ValueError(f"invalid Known-only view: {folder}")
    return train, validation


def preference(row: dict) -> tuple[int, int, int, int, int, int]:
    """Deterministic simplicity tie-break after Known-dev utility."""
    return (
        int(row["distance"] == "mahalanobis_diag"),
        int(row["boundary"] == "mean_std_1.0"),
        int(row["rule"] == "normalized_union"),
        int(row["fusion"] == "none"),
        int(float(row["fusion_weight"]) == 0.0),
        -int(row["k"]),
    )


def lock_selection() -> None:
    if OUT.exists():
        manifest = dict(read(OUT / "MANIFEST.json"))
        if manifest.get("status") not in {"selecting", "locked"} or manifest.get("test_read") is not False:
            raise FileExistsError(f"refusing to overwrite completed selection: {OUT}")
    else:
        OUT.mkdir(parents=True)
    manifest = {
        "status": "selecting",
        "experiment": "banking_known_only_recipe_expansion_final",
        "dataset": DATASET,
        "source": "textoir/data/banking",
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "coverage_target": COVERAGE_TARGET,
        "recipe_selection": "expanded recipe winner by Known-dev macro F1 across seeds",
        "geometry_selection": (
            "K=1-only diagnostic family; Known-dev utility at fixed 90% "
            "Known coverage, with simplicity tie-break"
        ),
        "geometry_search_grid": {
            "k": [1, 2, 3, 4, 5],
            "distance": list(coverage.DISTANCES),
            "boundary": [f"mean_std_{value}" for value in coverage.LAMBDA_VALUES]
            + [f"center_quantile_{value}" for value in coverage.BOUNDARY_QUANTILES]
            + [f"intent_quantile_{value}" for value in coverage.BOUNDARY_QUANTILES],
            "rules": ["nearest_sphere", "normalized_union"],
            "fusions": list(coverage.FUSIONS),
        },
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_used_for_selection": False,
        "test_read": False,
        "prior_test_results_exist": True,
        "device": "cuda",
    }
    dump(OUT / "MANIFEST.json", manifest)
    coverage.COVERAGE_TARGET = COVERAGE_TARGET
    coverage.FIXED_THRESHOLD = None
    import torch

    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    locks = []
    for kir in KIRS:
        selection = dict(read(EXPANSION / f"selection_kir{round(kir * 100):02d}.json"))
        recipe = str(selection["winner"])
        groups = []
        record_group = []
        for seed in SEEDS:
            record = record_for(kir, seed, recipe)
            train, validation = load_known(kir, seed)
            encoder = _RacalGateEncoder(
                MODEL_ROOT / "all-MiniLM-L6-v2",
                Path(record["checkpoint"]),
                device,
            )
            train_values = encoder.encode([row["text"] for row in train], batch_size=256)
            validation_values = encoder.encode([row["text"] for row in validation], batch_size=256)
            candidates = coverage.search_known(
                train_values, validation_values, train, validation
            )
            dump(OUT / "validation" / f"{DATASET}_{tag(kir, seed)}_{recipe}.json", candidates)
            groups.append(candidates)
            record_group.append(record)
            del encoder
            torch.cuda.empty_cache()
        keys = [coverage.geometry_key(row) for row in groups[0]]
        assert all([coverage.geometry_key(row) for row in group] == keys for group in groups)
        aggregate = []
        for index, key in enumerate(keys):
            rows = [group[index] for group in groups]
            aggregate.append(
                {
                    "key": key,
                    "utility": float(np.mean([row["utility"] for row in rows])),
                    "wrong": float(np.mean([row["known_wrong_accept_rate"] for row in rows])),
                    "coverage": float(np.mean([row["known_coverage"] for row in rows])),
                    "rows": rows,
                }
            )
        k1 = [row for row in aggregate if int(row["key"][0]) == 1]
        if not k1:
            raise RuntimeError(f"no K=1 candidate for KIR={kir}")
        winner = max(k1, key=lambda row: (row["utility"], -row["wrong"], preference(dict(zip(coverage.GEOMETRY_FIELDS, row["key"], strict=True)))))
        geometry = dict(zip(coverage.GEOMETRY_FIELDS, winner["key"], strict=True))
        dump(OUT / "validation" / f"{DATASET}_kir{round(kir * 100):02d}_geometry_choice.json", {
            "recipe": recipe,
            "selection": "Known-dev only",
            "candidate_count": len(aggregate),
            "k1_candidate_count": len(k1),
            "winner": geometry,
            "utility_mean": winner["utility"],
            "wrong_accept_mean": winner["wrong"],
            "coverage_mean": winner["coverage"],
        })
        for record, selected in zip(record_group, winner["rows"], strict=True):
            lock = {
                "dataset": DATASET,
                "kir": kir,
                "seed": int(record["seed"]),
                "recipe_name": recipe,
                "recipe": record["recipe"],
                "epoch": int(record["epoch"]),
                "checkpoint": record["checkpoint"],
                "training_record": record["path"],
                "geometry": {
                    **geometry,
                    "coverage": COVERAGE_TARGET,
                    "threshold": float(selected["threshold"]),
                },
                "scoring_family": "coverage",
                "selection_contract": str(OUT / "MANIFEST.json"),
                "selection_evidence": {
                    "known_validation_f1": float(record["known_validation_f1"]),
                    "known_dev_coverage": float(selected["known_coverage"]),
                    "known_correct_accept_rate": float(selected["known_correct_accept_rate"]),
                    "known_wrong_accept_rate": float(selected["known_wrong_accept_rate"]),
                    "recipe_selection_source": str(EXPANSION / f"selection_kir{round(kir * 100):02d}.json"),
                    "k1_only": True,
                },
                "real_oos_used": False,
                "pseudo_oos_used": False,
                "test_used_for_selection": False,
                "test_read": False,
                "prior_test_results_exist": True,
            }
            locks.append(lock)
            dump(OUT / "locks" / f"{DATASET}_{tag(kir, int(record['seed']))}.json", lock)
        print(
            f"LOCK banking77 KIR={kir:.2f} recipe={recipe} "
            f"K=1/{geometry['distance']}/{geometry['boundary']} "
            f"utility={winner['utility']:.6f}",
            flush=True,
        )
    manifest.update(status="locked", lock_count=len(locks), test_read=False, test_used_for_selection=False)
    dump(OUT / "MANIFEST.json", manifest)
    dump(OUT / "selection_complete.json", {"status": "locked", "units": len(locks), "test_read": False, "test_used_for_selection": False})


def write_summary() -> None:
    rows = []
    for path in sorted((OUT / "locks").glob("*.json")):
        lock = dict(read(path))
        result = dict(read(OUT / "final" / DATASET / tag(float(lock["kir"]), int(lock["seed"])) / "Ours.json"))
        metrics = result["metrics"]
        rows.append({
            "dataset": DATASET,
            "kir": float(lock["kir"]),
            "seed": int(lock["seed"]),
            "recipe": lock["recipe_name"],
            "k": lock["geometry"]["k"],
            "distance": lock["geometry"]["distance"],
            "boundary": lock["geometry"]["boundary"],
            "rule": lock["geometry"]["rule"],
            "fusion": lock["geometry"]["fusion"],
            "fusion_weight": lock["geometry"]["fusion_weight"],
            "threshold": lock["geometry"]["threshold"],
            "oos_f1": 100 * metrics["oos_f1"],
            "known_f1": 100 * metrics["known_f1"],
            "accuracy": 100 * metrics["accuracy"],
            "known_recall": 100 * metrics["known_recall"],
            "false_accept_rate": 100 * metrics["false_accept_rate"],
            "oos_recall": 100 * metrics["oos_recall"],
        })
    fields = list(rows[0])
    with (OUT / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for kir in KIRS:
        group = [row for row in rows if row["kir"] == kir]
        if len(group) != len(SEEDS):
            raise ValueError(f"missing final rows for KIR={kir}")
        summary = {"dataset": DATASET, "kir": kir, "n": len(group), "recipe": group[0]["recipe"], "k": group[0]["k"], "distance": group[0]["distance"], "boundary": group[0]["boundary"], "rule": group[0]["rule"], "fusion": group[0]["fusion"], "fusion_weight": group[0]["fusion_weight"]}
        for metric in ("oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate", "oos_recall"):
            values = np.asarray([row[metric] for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
        summaries.append(summary)
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def evaluate() -> None:
    manifest = dict(read(OUT / "MANIFEST.json"))
    if manifest.get("status") != "locked" or manifest.get("test_read") is not False:
        raise RuntimeError("selection is not locked before evaluation")
    command = [
        sys.executable,
        str(ROOT / "scripts/experiments/evaluate_kir_ours.py"),
        "--datasets", DATASET,
        "--kirs", *[str(kir) for kir in KIRS],
        "--seeds", *[str(seed) for seed in SEEDS],
        "--selection-root", str(OUT),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    write_summary()
    manifest.update(status="complete", test_read=True, test_used_for_selection=False, evaluator=command)
    dump(OUT / "MANIFEST.json", manifest)
    print("BANKING_RECIPE_EXPANSION_FINAL_COMPLETE", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("select", "evaluate"), required=True)
    args = parser.parse_args()
    if args.stage == "select":
        lock_selection()
    else:
        evaluate()


if __name__ == "__main__":
    main()
