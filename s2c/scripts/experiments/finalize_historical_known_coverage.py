"""Known-only coverage-controlled geometry search and one final test evaluation.

The acceptance coverage is fixed before evaluation.  It is not selected from
OOS rows, pseudo-OOS rows, or test rows.  The search is Gate-level; under the
historical evaluator the binary OOS decision is the same decision used by the
full cascade.
"""

import csv
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from scripts.experiments.search_historical_known_geometry import (
    _distance_matrix,
)
from scripts.experiments.search_historical_known_training import CELLS, OUT as TRAINING
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder
from tools.eval.run_historical_trainable_parameter_full_pipeline import (
    DATA_ROOT,
    DEFAULT_H1_ROOT,
    EXTERNAL_OOS_F1_BY_KIR,
    MODEL_ROOT,
    _metrics_from_output,
    write_csv,
)


OUT = ROOT / "results/analysis/historical_known_coverage90_v3"
SEEDS = (13, 42, 87)
COVERAGE_TARGET = 0.90
FIXED_THRESHOLD = None
ENCODE_BATCH_SIZE = 2048
K_VALUES = (1, 2, 3, 4, 5)
DISTANCES = ("euclidean", "mahalanobis_diag")
LAMBDA_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
BOUNDARY_QUANTILES = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
FUSION_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
FUSIONS = ("none", "ratio", "inverse_margin", "max_ratio")
KNOWN_WRONG_ACCEPT_PENALTY = 4.0
GEOMETRY_FIELDS = (
    "k",
    "distance",
    "boundary",
    "rule",
    "fusion",
    "fusion_weight",
)

RECIPES = {"existing"}


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _quantile(values, q):
    """Return a threshold accepting at least q of Known validation rows."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, q, method="higher"))


def _radius(detector, train_distances, train_intents, owners, boundary):
    kind, value = boundary.rsplit("_", 1)
    parameter = float(value)
    if boundary.startswith("mean_std_"):
        detector.radius_lambda = parameter
        detector._compute_radii()
        return np.asarray([sphere.radius for sphere in detector.spheres], dtype=np.float64)
    if boundary.startswith("center_quantile_"):
        assignment = np.argmin(
            np.where(train_intents[:, None] == owners[None, :], train_distances, np.inf),
            axis=1,
        )
        detector.radius_lambda = 3.0
        detector._compute_radii()
        return np.asarray(
            [
                np.quantile(train_distances[assignment == i, i], parameter)
                if np.any(assignment == i)
                else detector.spheres[i].radius
                for i in range(len(owners))
            ],
            dtype=np.float64,
        )
    if boundary.startswith("intent_quantile_"):
        return np.asarray(
            [
                np.quantile(train_distances[train_intents == name].min(axis=1), parameter)
                for name in owners
            ],
            dtype=np.float64,
        )
    raise ValueError(f"unsupported boundary {kind}: {boundary}")


def _score_from_distances(distances, detector, radius, owners, rule, fusion, weight):
    nearest = np.argmin(distances, axis=1)
    first = distances[np.arange(len(distances)), nearest]
    second = np.min(
        np.where(owners[None, :] != owners[nearest, None], distances, np.inf),
        axis=1,
    )
    normalized = distances / np.maximum(radius[None, :], 1e-12)
    ids = nearest if rule == "nearest_sphere" else np.argmin(normalized, axis=1)
    base = normalized[np.arange(len(distances)), ids]
    ratio = first / np.maximum(second, 1e-12)
    inverse_margin = first / np.maximum(second - first, 1e-12)
    if fusion == "none":
        score = base
    elif fusion == "ratio":
        score = base + weight * ratio
    elif fusion == "inverse_margin":
        score = base + weight * inverse_margin
    elif fusion == "max_ratio":
        score = np.maximum(base, weight * ratio)
    else:
        raise ValueError(fusion)
    return score, ids


def _fit_and_score(train_values, train_rows, values, config):
    train_intents = np.asarray([str(row["intent"]) for row in train_rows])
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=int(config["k"]),
        radius_method="mean_std",
        radius_lambda=1.0,
        distance_metric=str(config["distance"]),
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
        acceptance_mode="nearest_sphere",
    )
    detector.fit(train_values, train_intents)
    train_distances = _distance_matrix(detector, train_values)
    distances = _distance_matrix(detector, values)
    owners = np.asarray(
        [detector.cluster_to_intent[i] for i in range(len(detector.spheres))]
    )
    radius = _radius(detector, train_distances, train_intents, owners, config["boundary"])
    score, ids = _score_from_distances(
        distances,
        detector,
        radius,
        owners,
        str(config["rule"]),
        str(config["fusion"]),
        float(config["fusion_weight"]),
    )
    return detector, {"score": score, "nearest_cluster": ids}


def search_known(train_values, validation_values, train_rows, validation_rows):
    """Search geometry while fixing Known validation acceptance coverage."""
    train_intents = np.asarray([str(row["intent"]) for row in train_rows])
    truth = np.asarray([str(row["intent"]) for row in validation_rows])
    candidates = []
    for k in K_VALUES:
        detector = MultiSphereOOSDetector(
            center_mode="class_centroid_mixture",
            subcenters_per_intent=k,
            radius_method="mean_std",
            radius_lambda=1.0,
            distance_metric="euclidean",
            covariance_eps=1e-6,
            l2_normalize=True,
            random_state=42,
            acceptance_mode="nearest_sphere",
        )
        # Fit once per metric; the cached distances are reused for all rules.
        for metric in DISTANCES:
            detector.distance_metric = metric
            detector.fit(train_values, train_intents)
            train_distances = _distance_matrix(detector, train_values)
            validation_distances = _distance_matrix(detector, validation_values)
            owners = np.asarray(
                [detector.cluster_to_intent[i] for i in range(len(detector.spheres))]
            )
            nearest = np.argmin(validation_distances, axis=1)
            first = validation_distances[np.arange(len(validation_distances)), nearest]
            # Compute this once per detector rather than once per fusion candidate.
            second = np.min(
                np.where(
                    owners[None, :] != owners[nearest, None],
                    validation_distances,
                    np.inf,
                ),
                axis=1,
            )
            ratio = first / np.maximum(second, 1e-12)
            inverse_margin = first / np.maximum(second - first, 1e-12)
            center_assignment = np.argmin(
                np.where(
                    train_intents[:, None] == owners[None, :],
                    train_distances,
                    np.inf,
                ),
                axis=1,
            )
            intent_min_distances = {
                name: train_distances[train_intents == name].min(axis=1)
                for name in np.unique(train_intents)
            }
            boundaries = []
            for radius_lambda in LAMBDA_VALUES:
                boundary = f"mean_std_{radius_lambda}"
                detector.radius_lambda = radius_lambda
                detector._compute_radii()
                boundaries.append((boundary, np.asarray([sphere.radius for sphere in detector.spheres])))
            for quantile in BOUNDARY_QUANTILES:
                center_radius = np.asarray(
                    [
                        np.quantile(train_distances[center_assignment == i, i], quantile)
                        if np.any(center_assignment == i)
                        else detector.spheres[i].radius
                        for i in range(len(owners))
                    ]
                )
                boundaries.append((f"center_quantile_{quantile}", center_radius))
                intent_radius = np.asarray(
                    [np.quantile(intent_min_distances[name], quantile) for name in owners]
                )
                boundaries.append((f"intent_quantile_{quantile}", intent_radius))
            for boundary, radius in boundaries:
                normalized = validation_distances / np.maximum(radius[None, :], 1e-12)
                normalized_ids = np.argmin(normalized, axis=1)
                normalized_base = normalized[np.arange(len(normalized)), normalized_ids]
                nearest_base = first / np.maximum(radius[nearest], 1e-12)
                for rule in ("nearest_sphere", "normalized_union"):
                    ids = nearest if rule == "nearest_sphere" else normalized_ids
                    base = nearest_base if rule == "nearest_sphere" else normalized_base
                    correct = owners[ids] == truth
                    for fusion in FUSIONS:
                        weights = np.asarray(FUSION_WEIGHTS, dtype=np.float64)
                        if fusion == "none":
                            scores = np.repeat(base[:, None], len(weights), axis=1)
                        elif fusion == "ratio":
                            scores = base[:, None] + ratio[:, None] * weights[None, :]
                        elif fusion == "inverse_margin":
                            scores = base[:, None] + inverse_margin[:, None] * weights[None, :]
                        elif fusion == "max_ratio":
                            scores = np.maximum(base[:, None], ratio[:, None] * weights[None, :])
                        else:
                            raise ValueError(fusion)
                        thresholds = (
                            np.full(len(weights), float(FIXED_THRESHOLD), dtype=np.float64)
                            if FIXED_THRESHOLD is not None
                            else np.quantile(scores, COVERAGE_TARGET, axis=0, method="higher")
                        )
                        accepted = scores <= thresholds[None, :]
                        utilities = np.mean(
                            accepted
                            * np.where(
                                correct[:, None],
                                1.0,
                                -KNOWN_WRONG_ACCEPT_PENALTY,
                            ),
                            axis=0,
                        )
                        coverages = np.mean(accepted, axis=0)
                        correct_rates = np.mean(accepted & correct[:, None], axis=0)
                        wrong_rates = np.mean(accepted & ~correct[:, None], axis=0)
                        for index, weight in enumerate(FUSION_WEIGHTS):
                            candidates.append(
                                {
                                    "k": k,
                                    "distance": metric,
                                    "boundary": boundary,
                                    "rule": rule,
                                    "fusion": fusion,
                                    "fusion_weight": weight,
                                    "threshold": float(thresholds[index]),
                                    "utility": float(utilities[index]),
                                    "known_coverage": float(coverages[index]),
                                    "known_correct_accept_rate": float(correct_rates[index]),
                                    "known_wrong_accept_rate": float(wrong_rates[index]),
                                }
                            )
    # Every candidate has the same target coverage.  The utility only measures
    # the error profile of Known validation and never observes OOS labels.
    return candidates


def load_known(dataset, kir, seed):
    folder = DATA_ROOT / dataset / f"kir{round(kir * 100):02d}_seed{seed}" / "gate"
    train = read(folder / "train.json")
    known = {str(row["intent"]) for row in train}
    validation = [row for row in read(folder / "val.json") if str(row["intent"]) in known]
    assert train and validation
    assert all(int(row["label"]) == 0 for row in train + validation)
    return train, validation


def checkpoint_for(dataset, kir, seed, recipe):
    if recipe == "existing":
        return DEFAULT_H1_ROOT / dataset / f"kir{round(kir * 100):02d}_seed{seed}" / "trainable_k1" / "checkpoint.pt"
    tag = f"{dataset}_kir{round(kir * 100):02d}_seed{seed}_{recipe}"
    return Path(read(TRAINING / f"{tag}.json")["checkpoint"])


def geometry_key(row):
    return tuple(row[field] for field in GEOMETRY_FIELDS)


def main():
    global COVERAGE_TARGET, FIXED_THRESHOLD, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=float, default=COVERAGE_TARGET)
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--dataset", choices=("clinc150", "stackoverflow", "banking77_oos"))
    args = parser.parse_args()
    COVERAGE_TARGET = float(args.coverage)
    FIXED_THRESHOLD = args.fixed_threshold
    if not 0.0 < COVERAGE_TARGET <= 1.0:
        raise ValueError("--coverage must be in (0, 1]")
    target_cells = [
        cell for cell in CELLS if args.dataset is None or cell[0] == args.dataset
    ]
    if not target_cells:
        raise ValueError("the requested dataset has no registered cells")
    if FIXED_THRESHOLD is None:
        OUT = ROOT / f"results/analysis/historical_known_coverage{round(COVERAGE_TARGET * 100):02d}"
    else:
        OUT = ROOT / f"results/analysis/historical_known_fixed_threshold{round(FIXED_THRESHOLD * 100):03d}"
    if args.dataset is not None:
        OUT = OUT.with_name(OUT.name + f"_{args.dataset}")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    assert read(TRAINING / "MANIFEST.json")["status"] == "validation_complete"
    screened = read(TRAINING / "screening_selection.json")
    assert {(row["dataset"], row["kir"]) for row in screened} == set(CELLS)
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    manifest = {
        "status": "selecting",
        "device": "cuda",
        "protocol": "historical_v19_paper_main",
        "coverage_target": COVERAGE_TARGET,
        "fixed_threshold": FIXED_THRESHOLD,
        "selection": "mean Known validation utility at fixed Known coverage across three seeds",
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_used_for_selection": False,
        "test_read": False,
        "full_pipeline_verified": False,
    }
    dump(OUT / "MANIFEST.json", manifest)
    locks = []
    for dataset, kir in target_cells:
        recipe_names = ["existing"] + [
            row["name"] for row in screened if row["dataset"] == dataset and row["kir"] == kir
        ]
        choices = []
        for recipe in recipe_names:
            by_seed = {}
            checkpoints = []
            for seed in SEEDS:
                train, validation = load_known(dataset, kir, seed)
                checkpoint = checkpoint_for(dataset, kir, seed, recipe)
                checkpoints.append(str(checkpoint))
                encoder = _RacalGateEncoder(MODEL_ROOT / "all-MiniLM-L6-v2", checkpoint, device)
                train_values = encoder.encode(
                    [row["text"] for row in train], batch_size=ENCODE_BATCH_SIZE
                )
                validation_values = encoder.encode(
                    [row["text"] for row in validation], batch_size=ENCODE_BATCH_SIZE
                )
                candidates = search_known(
                    train_values,
                    validation_values,
                    train,
                    validation,
                )
                write_csv(
                    OUT / f"{dataset}_kir{round(kir * 100):02d}_seed{seed}_{recipe}.csv",
                    candidates,
                )
                by_seed[seed] = {geometry_key(row): row for row in candidates}
                del encoder
                torch.cuda.empty_cache()
            keys = list(by_seed[SEEDS[0]])
            assert all(set(by_seed[seed]) == set(keys) for seed in SEEDS)
            aggregate = []
            for key in keys:
                rows = [by_seed[seed][key] for seed in SEEDS]
                aggregate.append(
                    {
                        "key": key,
                        "utility": float(np.mean([row["utility"] for row in rows])),
                        "coverage": float(np.mean([row["known_coverage"] for row in rows])),
                        "wrong_accept": float(
                            np.mean([row["known_wrong_accept_rate"] for row in rows])
                        ),
                        "rows": rows,
                    }
                )
            winner = max(aggregate, key=lambda row: (row["utility"], -row["wrong_accept"]))
            config = dict(zip(GEOMETRY_FIELDS, winner["key"], strict=True))
            choices.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "recipe": recipe,
                    "config": config,
                    "utility": winner["utility"],
                    "known_coverage": winner["coverage"],
                    "checkpoints": checkpoints,
                    "validation_thresholds": [
                        row["threshold"] for row in winner["rows"]
                    ],
                }
            )
        cell_winner = max(choices, key=lambda row: (row["utility"], -row["known_coverage"]))
        locks.append(cell_winner)
        dump(OUT / f"{dataset}_kir{round(kir * 100):02d}_validation.json", choices)
        dump(OUT / "selection_lock.json", locks)
        print(f"LOCK {dataset} {kir}: {cell_winner}", flush=True)
    assert len(locks) == len(target_cells)
    manifest.update(status="all_choices_locked")
    dump(OUT / "MANIFEST.json", manifest)

    # No test path is touched until all nine per-cell decisions are immutable.
    results = []
    for lock in locks:
        config = lock["config"]
        for seed, checkpoint, threshold in zip(
            SEEDS,
            lock["checkpoints"],
            lock["validation_thresholds"],
            strict=True,
        ):
            train, _ = load_known(lock["dataset"], lock["kir"], seed)
            test = read(
                DATA_ROOT
                / lock["dataset"]
                / f"kir{round(lock['kir'] * 100):02d}_seed{seed}"
                / "gate"
                / "test.json"
            )
            encoder = _RacalGateEncoder(
                MODEL_ROOT / "all-MiniLM-L6-v2", Path(checkpoint), device
            )
            train_values = encoder.encode(
                [row["text"] for row in train], batch_size=ENCODE_BATCH_SIZE
            )
            test_values = encoder.encode(
                [row["text"] for row in test], batch_size=ENCODE_BATCH_SIZE
            )
            detector, output = _fit_and_score(train_values, train, test_values, config)
            metrics = _metrics_from_output(detector, output, test, threshold)
            results.append(
                {
                    "dataset": lock["dataset"],
                    "kir": lock["kir"],
                    "seed": seed,
                    "recipe": lock["recipe"],
                    "coverage_target": COVERAGE_TARGET,
                    **config,
                    "threshold": threshold,
                    **metrics,
                }
            )
            write_csv(OUT / "per_seed.csv", results)
            print(
                f"TEST {lock['dataset']}/{lock['kir']}/{seed} "
                f"OOS-F1={metrics['f1_u']:.6f}",
                flush=True,
            )
            del encoder
            torch.cuda.empty_cache()
    summary = []
    for lock in locks:
        group = [
            row
            for row in results
            if (row["dataset"], row["kir"]) == (lock["dataset"], lock["kir"])
        ]
        row = {
            "dataset": lock["dataset"],
            "kir": lock["kir"],
            "recipe": lock["recipe"],
            "coverage_target": COVERAGE_TARGET,
            **lock["config"],
        }
        for metric in (
            "f1_u",
            "f1_k",
            "accuracy",
            "known_recall",
            "false_accept_rate",
        ):
            row[f"{metric}_mean"] = float(np.mean([item[metric] for item in group]))
            row[f"{metric}_std"] = float(np.std([item[metric] for item in group]))
        row["baseline"] = EXTERNAL_OOS_F1_BY_KIR[lock["dataset"], lock["kir"]]
        row["delta_pp"] = 100.0 * row["f1_u_mean"] - row["baseline"]
        summary.append(row)
    write_csv(OUT / "summary.csv", summary)
    manifest.update(
        status="complete",
        test_read=True,
        completed_test_units=len(results),
        wins=sum(row["delta_pp"] > 0 for row in summary),
        std_ddof=0,
    )
    dump(OUT / "MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
