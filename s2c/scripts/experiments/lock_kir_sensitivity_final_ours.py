"""Lock the dense-KIR Ours contract before any test evaluation.

The representation recipe is selected from the existing seed-42 Known-dev
screening record for each dataset/KIR.  The selected recipe is then reused
for all three seeds.  Gate geometry is the fixed final-paper contract; no
test, real OOS, or pseudo-OOS row is read here.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.experiments import finalize_historical_known_coverage as coverage  # noqa: E402
from scripts.experiments.run_kir_sensitivity_known_only import KIRS, SEEDS  # noqa: E402
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder  # noqa: E402


ART = ROOT.parent / "artifacts" / "s2c" / "analysis" / "kir_sensitivity_known_only"
DATA_ROOT = ART / "data"
TRAINING_ROOT = ART / "training"
OUT = ART / "final_ours_selection"
MODEL_ROOT = ROOT.parent / "assets" / "models"

GEOMETRY = {
    "k": 1,
    "distance": "mahalanobis_diag",
    "boundary": "mean_std_1.0",
    "rule": "nearest_sphere",
    "fusion": "none",
    "fusion_weight": 0.0,
    "threshold": 1.0,
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def tag(kir: float, seed: int) -> str:
    return f"kir{round(kir * 100):02d}_seed{seed}"


def screening_path(dataset: str, kir: float) -> Path:
    prefix = "" if dataset == "banking77" else f"{dataset}_"
    return TRAINING_ROOT / f"{prefix}kir{round(kir * 100):02d}_screening.json"


def selection_source_path(dataset: str, kir: float) -> Path:
    """Resolve either the legacy screening record or canonical recipe search."""
    legacy = screening_path(dataset, kir)
    if legacy.is_file():
        return legacy
    canonical = TRAINING_ROOT / f"selection_kir{round(kir * 100):02d}.json"
    if canonical.is_file():
        return canonical
    return legacy


def choose_recipe(dataset: str, kir: float) -> str:
    path = selection_source_path(dataset, kir)
    payload = read(path)
    if path.name.startswith("selection_kir"):
        winner = payload.get("winner")
        if not winner:
            raise ValueError(f"canonical recipe selection has no winner: {path}")
        return str(winner)
    records = payload
    if not records:
        raise ValueError(f"empty Known-dev screening file: {path}")
    records = sorted(records, key=lambda row: (-float(row["known_validation_f1"]), str(row["name"])))
    record = records[0]
    if int(record["seed"]) != 42:
        raise ValueError(f"screening is not seed 42: {path}")
    if record.get("test_read") is not False or record.get("real_oos_used") is not False or record.get("pseudo_oos_used") is not False:
        raise ValueError(f"non-Known-only screening record: {path}")
    return str(record["name"])


def training_record(dataset: str, kir: float, seed: int, recipe: str) -> tuple[Path, dict]:
    path = TRAINING_ROOT / f"{dataset}_{tag(kir, seed)}_{recipe}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    record = read(path)
    if record.get("dataset") != dataset or float(record.get("kir")) != float(kir) or int(record.get("seed")) != int(seed):
        raise ValueError(f"training record identity mismatch: {path}")
    if record.get("name") != recipe or record.get("test_read") is not False:
        raise ValueError(f"training record is not the selected Known-only recipe: {path}")
    if record.get("real_oos_used") is not False or record.get("pseudo_oos_used") is not False:
        raise ValueError(f"OOS supervision in training record: {path}")
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return path, record


def known_views(dataset: str, kir: float, seed: int) -> tuple[list[dict], list[dict]]:
    folder = DATA_ROOT / dataset / tag(kir, seed) / "gate"
    train = read(folder / "train.json")
    validation = read(folder / "val.json")
    known = {str(row["intent"]) for row in train}
    if not train or not validation:
        raise ValueError(f"empty Known view: {folder}")
    if any(int(row.get("label", 0)) != 0 or str(row["intent"]) not in known for row in train + validation):
        raise ValueError(f"non-Known row in train/dev: {folder}")
    return train, validation


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("clinc150", "stackoverflow", "banking77"), default=["clinc150", "stackoverflow", "banking77"])
    parser.add_argument("--kirs", nargs="+", type=float, default=list(KIRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument("--banking-training-root", type=Path, default=None)
    parser.add_argument("--stackoverflow-training-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    datasets = tuple(args.datasets)
    kirs = tuple(args.kirs)
    seeds = tuple(args.seeds)
    if tuple(sorted(seeds)) != tuple(sorted(SEEDS)):
        raise ValueError("this campaign requires seeds {13,42,87}")

    data_root = args.data_root.resolve()
    training_root = args.training_root.resolve()
    banking_training_root = args.banking_training_root.resolve() if args.banking_training_root else None
    stackoverflow_training_root = args.stackoverflow_training_root.resolve() if args.stackoverflow_training_root else None
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    globals()["DATA_ROOT"] = data_root
    try:
        base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        base_commit = "unknown"
    manifest = {
        "status": "locking",
        "experiment": "ours_dense_kir_final_contract",
        "base_commit": base_commit,
        "datasets": list(datasets),
        "kirs": list(kirs),
        "seeds": list(seeds),
        "data_root": str(data_root),
        "dataset_variants": {"banking77": "standard_77_intent", "clinc150": "clinc150", "stackoverflow": "stackoverflow"},
        "geometry": GEOMETRY,
        "recipe_selection": "seed-42 Known-dev macro F1 winner per dataset/KIR, reused across seeds",
        "selection_supervision": "Known train/dev only",
        "real_oos_used": False,
        "pseudo_oos_used": False,
        "test_used_for_selection": False,
        "test_read": False,
        "test_previously_observed": True,
        "note": "Existing dense campaign artifacts include previously observed test evaluations; this lock stage itself never reads test.",
        "device": "cuda",
    }
    manifest["training_root"] = str(training_root)
    if banking_training_root is not None:
        manifest["banking_training_root"] = str(banking_training_root)
    if stackoverflow_training_root is not None:
        manifest["stackoverflow_training_root"] = str(stackoverflow_training_root)
    dump(out / "MANIFEST.json", manifest)

    coverage.COVERAGE_TARGET = 0.90
    coverage.FIXED_THRESHOLD = None
    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    rows = []
    for dataset in datasets:
        if dataset == "banking77" and banking_training_root:
            globals()["TRAINING_ROOT"] = banking_training_root
        elif dataset == "stackoverflow" and stackoverflow_training_root:
            globals()["TRAINING_ROOT"] = stackoverflow_training_root
        else:
            globals()["TRAINING_ROOT"] = training_root
        for kir in kirs:
            recipe = choose_recipe(dataset, kir)
            for seed in seeds:
                lock_path = out / "locks" / f"{dataset}_{tag(kir, seed)}.json"
                record_path, record = training_record(dataset, kir, seed, recipe)
                train, validation = known_views(dataset, kir, seed)
                if lock_path.is_file():
                    existing = read(lock_path)
                    if existing.get("geometry") != GEOMETRY or existing.get("recipe_name") != recipe:
                        raise ValueError(f"existing lock conflicts with final contract: {lock_path}")
                    lock = existing
                else:
                    encoder = _RacalGateEncoder(MODEL_ROOT / "all-MiniLM-L6-v2", Path(record["checkpoint"]), device)
                    try:
                        train_values = encoder.encode([row["text"] for row in train], batch_size=256)
                        validation_values = encoder.encode([row["text"] for row in validation], batch_size=256)
                    finally:
                        del encoder
                        torch.cuda.empty_cache()
                    _, scored = coverage._fit_and_score(train_values, train, validation_values, GEOMETRY)
                    accepted = np.asarray(scored["score"]) <= GEOMETRY["threshold"]
                    lock = {
                        "dataset": dataset,
                        "dataset_variant": "standard_77_intent" if dataset == "banking77" else dataset,
                        "protocol": "kir_sensitivity_known_only",
                        "kir": kir,
                        "seed": seed,
                        "recipe_name": recipe,
                        "recipe": record["recipe"],
                        "epoch": record["epoch"],
                        "checkpoint": record["checkpoint"],
                        "geometry": dict(GEOMETRY),
                        "selection_evidence": {
                            "screening_path": str(selection_source_path(dataset, kir)),
                            "training_record": str(record_path),
                            "known_validation_f1": float(record["known_validation_f1"]),
                            "known_dev_coverage": float(np.mean(accepted)),
                            "known_dev_wrong_accept_rate": 0.0,
                            "selection_rule": "fixed final-paper geometry; recipe selected by seed-42 Known-dev macro F1",
                        },
                        "real_oos_used": False,
                        "pseudo_oos_used": False,
                        "test_read": False,
                        "test_used_for_selection": False,
                        "test_previously_observed": True,
                    }
                    dump(lock_path, lock)
                rows.append({
                    "dataset": dataset,
                    "kir": kir,
                    "seed": seed,
                    "recipe": recipe,
                    **GEOMETRY,
                    "known_validation_f1": lock["selection_evidence"]["known_validation_f1"],
                    "known_dev_coverage": lock["selection_evidence"].get("known_dev_coverage"),
                    "checkpoint": lock["checkpoint"],
                    "test_read": lock["test_read"],
                })
                print(f"LOCKED {dataset}/{tag(kir, seed)} recipe={recipe}", flush=True)

    with (out / "known_selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest.update(status="locked", units=len(rows), lock_count=len(rows))
    dump(out / "MANIFEST.json", manifest)
    dump(out / "selection_complete.json", {
        "status": "locked",
        "units": len(rows),
        "test_read": False,
        "test_used_for_selection": False,
        "real_oos_used": False,
        "pseudo_oos_used": False,
    })
    print(f"LOCKS_COMPLETE {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
