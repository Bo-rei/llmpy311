"""Screen additional MiniLM recipes on standard Banking77 Known validation.

This is a training-only stage.  It never opens test rows and never creates a
pseudo-OOS view.  The resulting records are consumed by a later locked
geometry/evaluation stage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.experiments import search_historical_known_training as training  # noqa: E402


ART = ROOT.parent / "artifacts" / "s2c" / "analysis" / "kir_sensitivity_known_only"
OUT = ART / "banking_recipe_expansion"
CHECKPOINTS = OUT / "checkpoints"
DATA_ROOT = ART / "data"
DATASET = "banking77"
KIRS = (0.50, 0.75)
SEEDS = (13, 42, 87)
SCREENING_SEED = 42
TOP_N = 2

BASE = dict(
    layers=2,
    projection=256,
    projection_enabled=True,
    temperature=0.07,
    intra=0.1,
    inter=0.1,
    margin=0.2,
    lr=2e-5,
    projection_lr=2e-4,
    epochs=9,
)

# A compact, pre-registered factor search.  The existing last1/last2/last4
# records are included as controls; these names are new and therefore cannot
# collide with an earlier result bundle.
RECIPES = {
    "projection_only": {**BASE, "layers": 0},
    "last1_temp035": {**BASE, "layers": 1, "temperature": 0.035},
    "last2_temp035": {**BASE, "temperature": 0.035},
    "last2_temp014": {**BASE, "temperature": 0.14},
    "last2_intra025": {**BASE, "intra": 0.25},
    "last2_inter025": {**BASE, "inter": 0.25},
    "last2_margin04": {**BASE, "margin": 0.4},
    "last4_temp035": {**BASE, "layers": 4, "temperature": 0.035},
    "last4_epoch12": {**BASE, "layers": 4, "epochs": 12},
}

EXISTING_ART = ART / "training"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def existing_record(kir: float, seed: int, name: str) -> dict | None:
    path = EXISTING_ART / f"banking77_kir{round(kir * 100):02d}_seed{seed}_{name}.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if not (
        record.get("dataset") == DATASET
        and float(record.get("kir")) == kir
        and int(record.get("seed")) == seed
        and record.get("test_read") is False
        and record.get("real_oos_used") is False
        and record.get("pseudo_oos_used") is False
        and Path(record["checkpoint"]).is_file()
    ):
        return None
    return {"path": str(path), **record}


def record(kir: float, seed: int, name: str) -> dict:
    path = OUT / f"banking77_kir{round(kir * 100):02d}_seed{seed}_{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    return {"path": str(path), **value}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    training.DATA_ROOT = DATA_ROOT
    training.ART = CHECKPOINTS
    training.OUT = OUT
    training.RECIPES = RECIPES
    training.CELLS = [(DATASET, kir) for kir in KIRS]

    manifest = {
        "status": "running",
        "experiment": "banking_known_only_recipe_expansion",
        "dataset": DATASET,
        "source": "textoir/data/banking",
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "screening_seed": SCREENING_SEED,
        "top_n": TOP_N,
        "recipes": RECIPES,
        "selection": (
            "screen all new recipes plus existing Known-only controls on the "
            "Known dev split at seed 42; expand the top two per KIR to seeds "
            "13 and 87; select by mean Known-dev macro F1"
        ),
        "real_oos_used": False,
        "pseudo_oos_used": False,
        "test_read": False,
        "test_used_for_selection": False,
        "device": "cuda",
    }
    dump(OUT / "MANIFEST.json", manifest)

    import torch

    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    all_selection = {}
    for kir in KIRS:
        # Train only new recipes for the screening seed.  Existing controls
        # are read from their immutable Known-only records.
        candidates: dict[str, dict[int, dict]] = {}
        for name in RECIPES:
            candidates.setdefault(name, {})[SCREENING_SEED] = training.train_cell(
                DATASET, kir, SCREENING_SEED, name, device
            )
        for name in ("last1", "last2", "last4"):
            value = existing_record(kir, SCREENING_SEED, name)
            if value is not None:
                candidates.setdefault(name, {})[SCREENING_SEED] = value
        ranked = sorted(
            (
                {
                    "recipe": name,
                    "seed": SCREENING_SEED,
                    "known_validation_f1": float(values[SCREENING_SEED]["known_validation_f1"]),
                    "path": values[SCREENING_SEED]["path"],
                }
                for name, values in candidates.items()
            ),
            key=lambda row: (-row["known_validation_f1"], row["recipe"]),
        )
        selected_names = [row["recipe"] for row in ranked[:TOP_N]]
        dump(OUT / f"screening_kir{round(kir * 100):02d}.json", ranked)
        for name in selected_names:
            candidates.setdefault(name, {})
            for seed in (13, 87):
                if seed in candidates[name]:
                    continue
                if name in RECIPES:
                    candidates[name][seed] = training.train_cell(DATASET, kir, seed, name, device)
                else:
                    value = existing_record(kir, seed, name)
                    if value is None:
                        raise FileNotFoundError(f"missing existing control: {kir=} {seed=} {name=}")
                    candidates[name][seed] = value
        final_ranked = []
        for name in selected_names:
            values = [candidates[name][seed] for seed in SEEDS]
            final_ranked.append(
                {
                    "recipe": name,
                    "known_validation_f1_mean": sum(float(v["known_validation_f1"]) for v in values) / len(values),
                    "records": values,
                }
            )
        winner = max(final_ranked, key=lambda row: (row["known_validation_f1_mean"], -selected_names.index(row["recipe"])))
        all_selection[str(kir)] = {
            "screening": ranked,
            "expanded": final_ranked,
            "winner": winner["recipe"],
        }
        dump(OUT / f"selection_kir{round(kir * 100):02d}.json", all_selection[str(kir)])
        print(
            f"RECIPE_SELECTION banking77 KIR={kir:.2f} winner={winner['recipe']} "
            f"Known-dev-mean={winner['known_validation_f1_mean']:.6f}",
            flush=True,
        )

    manifest.update(status="validation_complete", selection=all_selection)
    dump(OUT / "MANIFEST.json", manifest)
    print("BANKING_RECIPE_EXPANSION_VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
