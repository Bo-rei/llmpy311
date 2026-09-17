"""Search Trainable MiniLM recipes on a TextOIR-aligned dataset.

Only Known train/dev labels are opened by this stage.  Test files are
materialized by the data-preparation stage but are intentionally not read
here.  The later selection stage chooses a recipe from the expanded seeds;
this script does not select a test winner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.experiments import search_historical_known_training as training  # noqa: E402


ART_ROOT = ROOT.parent / "artifacts" / "s2c" / "analysis" / "banking_textoir_aligned"
DATA_ROOT = ART_ROOT / "data"
OUT = ART_ROOT / "recipe_search_corrected"
CHECKPOINTS = OUT / "checkpoints"
DATASET = "banking77"
KIRS = (0.25, 0.50, 0.75)
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

# This is a bounded factor search.  Every variant keeps the same data,
# optimizer family, loss implementation and Known-only checkpoint criterion;
# only the named representation/training factor changes.
RECIPES = {
    "projection_only": {**BASE, "layers": 0},
    "last1": {**BASE, "layers": 1},
    "last2": {**BASE, "layers": 2},
    "last4": {**BASE, "layers": 4},
    "all6": {**BASE, "layers": 6},
    "no_projection": {**BASE, "projection_enabled": False},
    "projection128": {**BASE, "projection": 128},
    "temperature035": {**BASE, "temperature": 0.035},
    "temperature014": {**BASE, "temperature": 0.14},
    "intra025": {**BASE, "intra": 0.25},
    "inter025": {**BASE, "inter": 0.25},
    "margin04": {**BASE, "margin": 0.4},
    "lr_low": {**BASE, "lr": 1e-5, "projection_lr": 1e-4},
    "lr_high": {**BASE, "lr": 4e-5, "projection_lr": 4e-4},
    "epoch12": {**BASE, "epochs": 12},
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure(model: torch.nn.Module, recipe: dict, warmup: bool) -> torch.optim.Optimizer:
    """Apply the requested freeze contract before each training phase."""
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.projection.parameters():
        parameter.requires_grad_(bool(recipe["projection_enabled"]))
    # Projection-only warms up the adapter; a recipe without a projection
    # must instead expose its requested Transformer blocks immediately.
    if (not warmup or not recipe["projection_enabled"]) and int(recipe["layers"]) > 0:
        for layer in model.encoder.encoder.layer[-int(recipe["layers"]):]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
    groups = []
    encoder_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("encoder.") and parameter.requires_grad
    ]
    projection_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("projection.") and parameter.requires_grad
    ]
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": recipe["lr"]})
    if projection_parameters:
        groups.append({"params": projection_parameters, "lr": recipe["projection_lr"]})
    if not groups:
        raise ValueError(f"recipe has no trainable parameters: {recipe}")
    return torch.optim.AdamW(groups)


def train_cell(dataset: str, kir: float, seed: int, name: str, device: torch.device) -> dict:
    """Reuse the established training loop with the aligned roots and phase fix."""
    # The established loop is kept as the single loss/encoding implementation;
    # this wrapper only supplies the aligned data/checkpoint roots and the
    # correct projection-only/phase optimizer behavior.
    recipe = RECIPES[name]
    tag = f"{dataset}_kir{round(kir * 100):02d}_seed{seed}_{name}"
    record_path = OUT / f"{tag}.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("recipe") != recipe or not Path(record["checkpoint"]).is_file():
            raise RuntimeError(f"incompatible existing record: {record_path}")
        return record

    # Patch the small, self-contained loop's globals and freeze helper for this
    # isolated bundle; no historical artifact is overwritten.
    training.DATA_ROOT = DATA_ROOT
    training.ART = CHECKPOINTS
    training.OUT = OUT
    training.RECIPES = RECIPES
    original_configure = training.configure
    training.configure = configure
    try:
        record = training.train_cell(dataset, kir, seed, name, device)
    finally:
        training.configure = original_configure
    record.update(
        {
            "path": str(record_path),
            "protocol": "protocol_v2_textoir_v1",
            "dataset_variant": "standard_77_intent",
            "data_root": str(DATA_ROOT),
            "recipe_search": "Known train/dev macro-F1 checkpoint selection",
            "test_read": False,
            "test_used_for_selection": False,
            "real_oos_used": False,
            "pseudo_oos_used": False,
        }
    )
    dump(record_path, record)
    return record


def main() -> None:
    import argparse

    global DATASET, ART_ROOT, DATA_ROOT, OUT, CHECKPOINTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("banking77", "stackoverflow", "clinc150"), default=DATASET)
    parser.add_argument("--art-root", type=Path, default=ART_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Canonical prepared view root. Defaults to <art-root>/data.",
    )
    parser.add_argument("--kirs", nargs="+", type=float, default=list(KIRS))
    parser.add_argument("--screening-seed", type=int, default=SCREENING_SEED)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()
    DATASET = args.dataset
    ART_ROOT = args.art_root.resolve()
    DATA_ROOT = args.data_root.resolve() if args.data_root is not None else ART_ROOT / "data"
    OUT = ART_ROOT / "recipe_search_corrected"
    CHECKPOINTS = OUT / "checkpoints"
    kirs = tuple(args.kirs)
    screening_seed = int(args.screening_seed)
    top_n = int(args.top_n)
    if screening_seed not in SEEDS or top_n < 1 or top_n > len(RECIPES):
        raise ValueError("invalid screening seed or top-n")

    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    training.DATA_ROOT = DATA_ROOT
    training.ART = CHECKPOINTS
    training.OUT = OUT
    training.RECIPES = RECIPES
    manifest = {
        "status": "running",
        "experiment": "banking77_textoir_aligned_known_only_recipe_search",
        "dataset": DATASET,
        "dataset_variant": "standard_77_intent",
        "protocol": "protocol_v2_textoir_v1",
        "data_root": str(DATA_ROOT),
        "kirs": list(kirs),
        "seeds": list(SEEDS),
        "screening_seed": screening_seed,
        "top_n": top_n,
        "recipes": RECIPES,
        "selection": "screen recipes on Known-dev macro F1; expand top-N to remaining seeds; select by mean Known-dev macro F1",
        "representation_training": "Known train only; checkpoint selected on Known dev only",
        "real_oos_used_for_training": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_used_for_selection": False,
        "test_read": False,
        "device": "cuda",
    }
    dump(OUT / "MANIFEST.json", manifest)

    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    all_selection = {}
    for kir in kirs:
        screen = []
        for name in RECIPES:
            record = train_cell(DATASET, kir, screening_seed, name, device)
            screen.append(
                {
                    "recipe": name,
                    "seed": screening_seed,
                    "known_validation_f1": float(record["known_validation_f1"]),
                    "path": str(record["path"]),
                }
            )
        ranked = sorted(screen, key=lambda row: (-row["known_validation_f1"], row["recipe"]))
        selected_names = [row["recipe"] for row in ranked[:top_n]]
        expanded = []
        for name in selected_names:
            records = []
            for seed in SEEDS:
                records.append(train_cell(DATASET, kir, seed, name, device))
            mean = sum(float(row["known_validation_f1"]) for row in records) / len(records)
            expanded.append(
                {
                    "recipe": name,
                    "known_validation_f1_mean": mean,
                    "records": records,
                }
            )
        winner = max(
            expanded,
            key=lambda row: (row["known_validation_f1_mean"], -selected_names.index(row["recipe"])),
        )
        all_selection[str(kir)] = {
            "screening": ranked,
            "expanded": expanded,
            "winner": winner["recipe"],
            "selection_metric": "mean Known-dev macro F1 across seeds",
        }
        dump(OUT / f"selection_kir{round(kir * 100):02d}.json", all_selection[str(kir)])
        print(
            f"RECIPE_SELECTION banking77 KIR={kir:.2f} winner={winner['recipe']} "
            f"Known-dev-mean={winner['known_validation_f1_mean']:.6f}",
            flush=True,
        )

    manifest.update(status="validation_complete", selection=all_selection)
    dump(OUT / "MANIFEST.json", manifest)
    print("BANKING_TEXTOIR_ALIGNED_RECIPE_VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
