"""Lock the historical score-threshold=1 K=1 Mahalanobis control.

The representation recipe is taken from the existing Known-dev-only recipe
selection for each KIR.  The Gate geometry and threshold are a pre-declared
method-contract hypothesis; this stage reads only Known train/dev rows and
does not inspect test rows.
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
    KIRS,
    MODEL_ROOT,
    RECIPE_ROOT,
    SEEDS,
    load_known,
    read,
    tag,
)
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder  # noqa: E402


OUT = ART_ROOT / "fixed_k1_mahalanobis_threshold1_known_only"
GEOMETRY = {
    "k": 1,
    "distance": "mahalanobis_diag",
    "boundary": "mean_std_1.0",
    "rule": "nearest_sphere",
    "fusion": "none",
    "fusion_weight": 0.0,
    "threshold": 1.0,
}


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_record(kir: float, seed: int) -> tuple[str, dict]:
    selection = read(RECIPE_ROOT / f"selection_kir{round(kir * 100):02d}.json")
    recipe = str(selection["winner"])
    for expanded in selection["expanded"]:
        if str(expanded["recipe"]) == recipe:
            for record in expanded["records"]:
                if int(record["seed"]) == seed:
                    return recipe, record
    raise FileNotFoundError(f"missing selected recipe: KIR={kir}, seed={seed}")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    manifest = {
        "status": "locking",
        "experiment": "banking77_textoir_aligned_fixed_k1_mahalanobis_threshold1_known_only",
        "dataset": "banking77",
        "dataset_variant": "standard_77_intent",
        "protocol": "protocol_v2_textoir_v1",
        "data_root": str(DATA_ROOT),
        "recipe_root": str(RECIPE_ROOT),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "geometry": GEOMETRY,
        "recipe_selection": "existing Known-dev-only winner for each KIR",
        "threshold_selection": "fixed historical normalized score threshold s=1; not selected from test",
        "real_oos_used_for_training": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_read": False,
        "test_used_for_selection": False,
        "device": "cuda",
    }
    dump(OUT / "MANIFEST.json", manifest)
    rows = []
    for kir in KIRS:
        for seed in SEEDS:
            recipe, record = selected_record(kir, seed)
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
            accepted = output["score"] <= GEOMETRY["threshold"]
            truth = np.asarray([str(row["intent"]) for row in validation])
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
                "geometry": dict(GEOMETRY),
                "scoring_family": "coverage",
                "selection_evidence": {
                    "known_validation_f1": float(record["known_validation_f1"]),
                    "known_dev_coverage": float(np.mean(accepted)),
                    "known_dev_wrong_accept_rate": float(
                        np.mean(accepted & (owners != truth))
                    ),
                    "selection_rule": "fixed K=1 diagonal Mahalanobis mean+1sigma no fusion and historical score threshold 1",
                    "test_read": False,
                },
                "real_oos_used_for_training": False,
                "real_oos_used_for_selection": False,
                "pseudo_oos_used": False,
                "test_read": False,
                "test_used_for_selection": False,
            }
            dump(OUT / "locks" / f"banking77_{tag(kir, seed)}.json", lock)
            rows.append({
                "dataset": "banking77",
                "kir": kir,
                "seed": seed,
                "recipe": recipe,
                **GEOMETRY,
                "known_validation_f1": record["known_validation_f1"],
                "known_dev_coverage": float(np.mean(accepted)),
                "known_dev_wrong_accept_rate": float(
                    np.mean(accepted & (owners != truth))
                ),
            })
            del encoder
            torch.cuda.empty_cache()
    with (OUT / "known_selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest.update(status="locked", units=len(rows))
    dump(OUT / "MANIFEST.json", manifest)
    dump(OUT / "selection_complete.json", {
        "status": "locked",
        "units": len(rows),
        "test_read": False,
        "test_used_for_selection": False,
    })
    print("BANKING_TEXTOIR_ALIGNED_K1_MAHALANOBIS_THRESHOLD1_LOCKED", flush=True)


if __name__ == "__main__":
    main()
