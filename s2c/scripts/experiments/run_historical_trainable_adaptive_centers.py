#!/usr/bin/env python3
"""H1 train-only adaptive centers; OOS-only validation selection then test confirmation."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.eval.run_historical_trainable_parameter_full_pipeline import (
    DATASETS, SEEDS, PAPER, DEFAULT_H1_ROOT, _encode_cell, _fit_detector,
    _run_pipeline, compute_metrics, write_csv,
)
from tools.eval.run_historical_trainable_extended_boundary_search import (
    _distance_matrix, _score_from_distances,
)

NAME = "historical_trainable_adaptive_centers"
LAMBDAS = (.25, .5, .75, 1., 1.25, 1.5, 2., 2.5, 3.)
THRESHOLDS = tuple(round(v / 100, 2) for v in range(70, 131, 5))
MODES = ("nearest_sphere", "normalized_union")
METRICS = ("oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate", "auroc", "aupr_oos")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assignments(bank):
    """Only fitted Known training arrays are available to the ranking function."""
    base = bank[1]
    rankings = {}
    stats = []
    for intent, ids in sorted(base.intent_to_clusters.items()):
        mask = base._train_cluster_labels == ids[0]
        x = base._train_embeddings[mask]
        diff = x - x.mean(axis=0)
        sse1 = float(np.sum(diff ** 2))
        labels2 = bank[2]._train_cluster_labels[mask]
        sse2 = float(np.sum((x - bank[2].kmeans.cluster_centers_[labels2]) ** 2))
        stats.append({"intent": intent, "n_train": len(x),
                      "mean_distance": float(np.linalg.norm(diff, axis=1).mean()),
                      "variance": sse1 / len(x),
                      "sse_reduction": (sse1 - sse2) / max(sse1, 1e-12)})
    intents = [s["intent"] for s in stats]
    strategies = {f"fixed_k{k}": {i: k for i in intents} for k in (1, 2, 3)}
    aliases = {}
    for ranking in ("mean_distance", "variance", "sse_reduction"):
        rankings[ranking] = sorted(stats, key=lambda s: (-s[ranking], s["intent"]))
        for percent in (10, 25, 50, 75, 100):
            selected = {s["intent"] for s in rankings[ranking][:int(np.ceil(len(intents) * percent / 100))]}
            for k in (2, 3):
                name = f"{ranking}_top{percent}_k{k}"
                if percent == 100:
                    aliases[name] = f"fixed_k{k}"
                else:
                    strategies[name] = {i: k if i in selected else 1 for i in intents}
    return strategies, stats, aliases


def mixed_detector(bank, assignment):
    """Reuse independent per-intent fits, preserving the native sphere geometry."""
    result = copy.copy(bank[1])
    result.spheres = []
    result.intent_to_clusters = {}
    result.cluster_to_intent = {}
    result.subcenters_overrides = dict(assignment)
    columns = []
    for intent, k in sorted(assignment.items()):
        result.intent_to_clusters[intent] = []
        for col in bank[k].intent_to_clusters[intent]:
            sid = len(result.spheres)
            result.spheres.append(replace(bank[k].spheres[col], cluster_id=sid))
            result.intent_to_clusters[intent].append(sid)
            result.cluster_to_intent[sid] = intent
            columns.append((k, col))
    result.intent_to_cluster = {i: ids[0] for i, ids in result.intent_to_clusters.items()}
    result.n_clusters = len(result.spheres)
    return result, columns


def workpoints(detector, output, rows, thresholds):
    """Count the actual multiclass confusion totals, without repeated string F1 calls."""
    labels = np.array([r["label"] for r in rows])
    known = labels == 0
    intents = sorted({r["intent"] for r in rows if r["label"] == 0})
    mapping = {intent: i for i, intent in enumerate(intents)}
    truth = np.array([mapping.get(r["intent"], -1) if r["label"] == 0 else -1 for r in rows])
    pred = np.array([mapping.get(detector.cluster_to_intent[int(i)], -2) for i in output["nearest_cluster"]])
    score = output["score"]
    auc = float(roc_auc_score(labels, score))
    ap = float(average_precision_score(labels, score))
    support = np.bincount(truth[known], minlength=len(intents))
    result = []
    for threshold in thresholds:
        reject = score > threshold
        tp = int(np.sum(reject & ~known))
        fp = int(np.sum(reject & known))
        fn = int(np.sum(~reject & ~known))
        correct = ~reject & known & (truth == pred)
        hits = np.bincount(truth[correct], minlength=len(intents))
        accepted_pred = pred[(~reject) & (pred >= 0)]
        predicted = np.bincount(accepted_pred, minlength=len(intents))
        f1 = 2 * hits / np.maximum(support + predicted, 1)
        result.append({"threshold": float(threshold), "oos_f1": 2 * tp / max(2 * tp + fp + fn, 1),
                       "known_f1": float(f1.mean()), "accuracy": float((correct.sum() + tp) / len(rows)),
                       "known_recall": float(np.mean(~reject[known])), "false_accept_rate": float(fn / (~known).sum()),
                       "false_accept_count": fn, "known_false_reject_count": fp,
                       "auroc": auc, "aupr_oos": ap})
    return result


def select(rows):
    """No test rows or auxiliary metrics affect selection; stable complexity tie break."""
    val = [r for r in rows if r["split"] == "val"]
    return dict(min(val, key=lambda r: (-r["oos_f1"], r["center_count"],
                    abs(r["radius_lambda"] - 1), abs(r["threshold"] - 1),
                    MODES.index(r["acceptance_mode"]), r["strategy"])))


def key(row):
    return row["strategy"], row["radius_lambda"], row["acceptance_mode"], row["threshold"]


def diagnostics(detector, output, rows, choice, split):
    labels = np.array([r["label"] for r in rows])
    score = output["score"]
    precision, recall, _ = precision_recall_curve(labels, score)
    ceiling = float(np.max(2 * precision * recall / np.maximum(precision + recall, 1e-12)))
    summary = {"split": split, "selected_oos_f1": choice["oos_f1"], "fixed_score_threshold_oracle_f1": ceiling,
               "oracle_is_posthoc_not_selection": True,
               "all_reject_oos_f1": float(2 * labels.sum() / (len(labels) + labels.sum())),
               "auroc": float(roc_auc_score(labels, score)), "aupr_oos": float(average_precision_score(labels, score))}
    errors = []
    for intent in sorted({r["intent"] for r in rows}):
        mask = np.array([r["intent"] == intent for r in rows])
        known = int(labels[mask][0]) == 0
        errors.append({"split": split, "intent": intent, "is_known": known, "count": int(mask.sum()),
                       "accepted_count": int(np.sum(score[mask] <= choice["threshold"])),
                       "rejected_count": int(np.sum(score[mask] > choice["threshold"])),
                       "near_boundary_count": int(np.sum(np.abs(score[mask] / choice["threshold"] - 1) <= .05)),
                       "median_score": float(np.median(score[mask]))})
    return summary, errors


def run_cell(dataset, seed, device, output_root, h1_root):
    cell = output_root / dataset / f"seed{seed}"
    cell.mkdir(parents=True, exist_ok=False)
    print(f"ENCODE {dataset} seed={seed} device={device}", flush=True)
    views, values = _encode_cell(dataset, seed, device, h1_root)
    if any(r["label"] != 0 for r in views["train"]):
        raise ValueError("Known-only train contract violated")
    bank = {k: _fit_detector(values["train"], views["train"], k, 1., MODES[0]) for k in (1, 2, 3)}
    strategies, stats, aliases = assignments(bank)
    dump(cell / "center_assignments.json", {"source": "train_known_only", "rankings": stats,
         "strategies": strategies, "aliases": aliases, "rounding": "ceil; descending rank; intent lexical ties"})
    val_dist = {k: _distance_matrix(d, values["val"]) for k, d in bank.items()}
    val_rows = []
    # Complete coarse search for every strategy before refining its validation winner.
    for name, assignment in strategies.items():
        d, columns = mixed_detector(bank, assignment)
        distances = np.column_stack([val_dist[k][:, col] for k, col in columns])
        local = []
        for lam in LAMBDAS:
            for b in bank.values():
                b.radius_lambda = lam
                b._compute_radii()
            d, _ = mixed_detector(bank, assignment)
            for mode in MODES:
                scores = _score_from_distances(distances, d, mode)
                local.extend({"dataset": dataset, "seed": seed, "split": "val", "strategy": name,
                              "center_count": len(d.spheres), "radius_lambda": lam, "acceptance_mode": mode,
                              "grid_stage": "coarse", **r} for r in workpoints(d, scores, views["val"], THRESHOLDS))
        winner = select(local)
        fine = [round(winner["threshold"] + step / 100, 2) for step in range(-4, 5)
                if .70 <= round(winner["threshold"] + step / 100, 2) <= 1.30
                and round(winner["threshold"] + step / 100, 2) not in THRESHOLDS]
        for b in bank.values():
            b.radius_lambda = winner["radius_lambda"]
            b._compute_radii()
        d, _ = mixed_detector(bank, assignment)
        scores = _score_from_distances(distances, d, winner["acceptance_mode"])
        local.extend({**winner, **r, "grid_stage": "fine"} for r in workpoints(d, scores, views["val"], fine))
        val_rows.extend(local)
    choices = []
    for scope, candidates in (("overall", val_rows), ("fixed", [r for r in val_rows if r["strategy"].startswith("fixed_")]),
                              ("adaptive", [r for r in val_rows if not r["strategy"].startswith("fixed_")])):
        choices.append({**select(candidates), "selection_scope": scope, "selection_objective": "validation_oos_f1_only",
                        "selection_guard": "none", "test_used_for_selection": False})
    write_csv(cell / "validation_workpoints.csv", val_rows)
    dump(cell / "selection_lock.json", {"choices": choices, "test_metrics_computed": False,
         "fine_rule": "each strategy coarse validation winner, same lambda/mode, threshold +/- .04 step .01"})
    print(f"LOCKED {dataset} seed={seed}: {choices[0]}", flush=True)
    # Test labels enter only after the full candidate list and all choices are frozen.
    test_dist = {k: _distance_matrix(d, values["test"]) for k, d in bank.items()}
    test_rows = []
    selected_test = []
    diag = []
    intent_errors = []
    for name, assignment in strategies.items():
        d, columns = mixed_detector(bank, assignment)
        distances = np.column_stack([test_dist[k][:, col] for k, col in columns])
        for lam in LAMBDAS:
            for b in bank.values():
                b.radius_lambda = lam
                b._compute_radii()
            d, _ = mixed_detector(bank, assignment)
            for mode in MODES:
                candidates = [r for r in val_rows if r["strategy"] == name and r["radius_lambda"] == lam and r["acceptance_mode"] == mode]
                scores = _score_from_distances(distances, d, mode)
                measured = workpoints(d, scores, views["test"], [r["threshold"] for r in candidates])
                test_rows.extend({**v, **m, "split": "test"} for v, m in zip(candidates, measured, strict=True))
    write_csv(cell / "test_workpoints.csv", test_rows)
    with tempfile.TemporaryDirectory(prefix="s2c_adaptive_direct_") as temp:
        direct_cache = {}
        for choice in choices:
            test = next(r for r in test_rows if key(r) == key(choice))
            for b in bank.values():
                b.radius_lambda = choice["radius_lambda"]
                b._compute_radii()
            d, columns = mixed_detector(bank, strategies[choice["strategy"]])
            d.radius_lambda = choice["radius_lambda"]
            d.acceptance_mode = choice["acceptance_mode"]
            record = {**test, "selection_scope": choice["selection_scope"], "validation_oos_f1": choice["oos_f1"],
                      "test_used_for_selection": False, "direct_pipeline_verified": False, "device": str(device)}
            if device.type == "cuda":
                if key(choice) not in direct_cache:
                    predictions, meta = _run_pipeline(dataset, seed, device, d, choice["threshold"], h1_root, Path(temp))
                    metrics, stages = compute_metrics(views["test"], predictions)
                    direct_cache[key(choice)] = {"full_pipeline_" + k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                    direct_cache[key(choice)]["direct_prediction_count"] = meta["prediction_count"]
                    del predictions
                record.update(direct_cache[key(choice)])
                record["gate_pipeline_oos_abs_delta"] = abs(record["full_pipeline_oos_f1"] - test["oos_f1"])
                if record["gate_pipeline_oos_abs_delta"] > 1e-10:
                    raise AssertionError("Gate/direct pipeline OOS mismatch")
                record["direct_pipeline_verified"] = True
            record["delta_oos_f1_pp"] = record["oos_f1"] * 100 - PAPER[dataset]["oos_f1"]
            selected_test.append(record)
            if choice["selection_scope"] == "overall":
                for split, dist, selected in (("val", val_dist, choice), ("test", test_dist, test)):
                    scores = _score_from_distances(np.column_stack([dist[k][:, col] for k, col in columns]), d, d.acceptance_mode)
                    ceiling, errors = diagnostics(d, scores, views[split], selected, split)
                    diag.append({"dataset": dataset, "seed": seed, **ceiling})
                    intent_errors.extend({"dataset": dataset, "seed": seed, **r} for r in errors)
    write_csv(cell / "selected_test.csv", selected_test)
    write_csv(cell / "ranking_diagnostics.csv", diag)
    write_csv(cell / "intent_errors.csv", intent_errors)
    print(f"DONE {dataset} seed={seed}: test={selected_test[0]['oos_f1']:.6f}", flush=True)
    return selected_test


def summarize(rows):
    result = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for scope in ("overall", "fixed", "adaptive"):
            group = [r for r in rows if r["dataset"] == dataset and r["selection_scope"] == scope]
            row = {"dataset": dataset, "selection_scope": scope, "seed_count": len(group), "paper_oos_f1": PAPER[dataset]["oos_f1"]}
            for metric in METRICS:
                numbers = [float(r[metric]) for r in group]
                row[metric + "_mean"] = float(np.mean(numbers))
                row[metric + "_std"] = float(np.std(numbers))
            beats = sum(float(r["delta_oos_f1_pp"]) > 0 for r in group)
            row.update({"delta_oos_f1_pp": row["oos_f1_mean"] * 100 - row["paper_oos_f1"],
                        "seeds_beating_paper": beats, "all_seeds_beat_paper": beats == len(group),
                        "only_one_seed_beats_paper": beats == 1,
                        "mean_beats_paper": row["oos_f1_mean"] * 100 > row["paper_oos_f1"],
                        "direct_pipeline_verified": all(str(r["direct_pipeline_verified"]).lower() == "true" for r in group), "std_ddof": 0})
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--seed", choices=SEEDS, type=int, action="append")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "analysis" / NAME)
    parser.add_argument("--h1-root", type=Path, default=DEFAULT_H1_ROOT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.zeros(1, device=device)  # Actual runtime check; never silently claim a CPU fallback is CUDA.
    started = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": NAME, "protocol": "historical_v19_paper_main", "evidence_level": "H1 controlled",
                "device": str(device), "datasets": args.dataset or list(DATASETS), "seeds": args.seed or list(SEEDS),
                "kir": .5, "representation": "existing_last2_minilm_384_256_384_residual_projection",
                "train_oos_used": False, "test_used_for_selection": False, "selection_objective": "validation_oos_f1_only",
                "selection_guard": "none", "lambdas": LAMBDAS, "thresholds": THRESHOLDS, "acceptance_modes": MODES,
                "radius_rule": "mean_std", "covariance_rule": "per_sphere_diagonal_variance_plus_1e-6",
                "downstream": "fixed cascade_full/gpu_kir50", "direct_pipeline_verified": False, "status": "running",
                "rerun_reason": "new train-only adaptive assignments; fixed baselines required under changed no-guard objective",
                "cuda_status": "runtime_tensor_verified" if device.type == "cuda" else "not_used; direct pipeline pending"}
    dump(args.output_root / "MANIFEST.json", manifest)
    selected = []
    for dataset in manifest["datasets"]:
        for seed in manifest["seeds"]:
            selected.extend(run_cell(dataset, seed, device, args.output_root, args.h1_root))
            write_csv(args.output_root / "selected_test_per_seed.csv", selected)
            write_csv(args.output_root / "selected_test_summary.csv", summarize(selected))
    manifest.update(status="complete", completed_unit_count=len(selected) // 3,
                    direct_pipeline_verified=all(r["direct_pipeline_verified"] for r in selected),
                    elapsed_seconds=time.time() - started)
    dump(args.output_root / "MANIFEST.json", manifest)
    print(json.dumps(summarize(selected), indent=2), flush=True)


if __name__ == "__main__":
    main()
