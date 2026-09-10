#!/usr/bin/env python3
"""Bounded CLINC H1 representation follow-up; freeze all choices before test."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from protocol_v2.experiments.racal_v1.representation import build_racal_model, encode_rows, set_seed  # noqa: E402
from protocol_v2.experiments.racal_v1.runner import _center_losses, _class_centers  # noqa: E402
from scripts.experiments.run_historical_trainable_adaptive_centers import (  # noqa: E402
    LAMBDAS, THRESHOLDS, MODES, dump, select, workpoints, diagnostics,
)
from tools.eval.run_historical_trainable_extended_boundary_search import _distance_matrix, _score_from_distances  # noqa: E402
from tools.eval.run_historical_trainable_parameter_full_pipeline import (  # noqa: E402
    _fit_detector, _run_pipeline, _encode_cell, DEFAULT_H1_ROOT, MODEL_ROOT, PAPER, write_csv,
    compute_metrics,
)
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder  # noqa: E402

NAME = "historical_trainable_checkpoint_selection"
# Fixed in advance, independent of any new test result. Longer last2 training,
# projection-only, and LoRA answer separate representation questions.
RECIPES = (
    ("last2_long", "last2_minilm_plus_projection", 2e-5),
    ("projection_only", "trainable_projection_only", 0.),
    ("lora_long", "lora_minilm_plus_projection", 2e-5),
)


def validation_boundary(train, val, views, seed, strategy):
    detector = _fit_detector(train, views["train"], 1, 1., MODES[0])
    distances = _distance_matrix(detector, val)
    candidates = []
    for lam in LAMBDAS:
        detector.radius_lambda = lam
        detector._compute_radii()
        for mode in MODES:
            output = _score_from_distances(distances, detector, mode)
            candidates.extend({"dataset": "clinc150", "seed": seed, "split": "val", "strategy": strategy,
                               "center_count": len(detector.spheres), "radius_lambda": lam,
                               "acceptance_mode": mode, **r} for r in workpoints(detector, output, views["val"], THRESHOLDS))
    return select(candidates)


def set_phase(model, warmup):
    # Restore the model's own freeze contract; never unfreeze the backbone of projection-only.
    if model.mode == "lora_minilm_plus_projection":
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("projection.") or (not warmup and "lora_" in name))
    else:
        model._apply_freeze_contract()
        if warmup:
            for p in model.encoder.parameters():
                p.requires_grad_(False)


def optimizer_for(model, backbone_lr):
    projection = [p for name, p in model.named_parameters() if name.startswith("projection.") and p.requires_grad]
    adapted = [p for name, p in model.named_parameters() if not name.startswith("projection.") and p.requires_grad]
    groups = [{"params": projection, "lr": 2e-4}]
    if adapted:
        groups.append({"params": adapted, "lr": backbone_lr})
    return torch.optim.AdamW(groups)


def completed_recipe(seed, recipe, artifact_root, output_root, epochs):
    """Reuse whole recipes only; optimizer/RNG state is not saved for partial training."""
    name, mode, backbone_lr = recipe
    run_dir = artifact_root / name / "clinc150" / f"kir50_seed{seed}" / "trainable_k1"
    history_path = output_root / f"{name}_seed{seed}_training_history.csv"
    entries = {p.name for p in run_dir.iterdir()} if run_dir.exists() else set()
    if not entries and not history_path.exists():
        return None  # Includes the empty directory left by a failed model factory.
    if entries != {"checkpoint.pt", "run_manifest.json"} or not history_path.is_file():
        raise ValueError(f"Partial nonempty recipe cannot resume: {run_dir}")
    result = json.loads((run_dir / "run_manifest.json").read_text())
    expected = {"dataset": "clinc150", "seed": seed, "strategy": name, "mode": mode,
                "split": "val", "backbone_lr": backbone_lr, "projection_lr": 2e-4,
                "oos_used_for_training": False, "test_used_for_selection": False,
                "checkpoint_objective": "validation_oos_f1_only"}
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Recipe manifest mismatch: {run_dir}")
    checkpoint_path = run_dir / "checkpoint.pt"
    if Path(result["checkpoint"]).resolve() != checkpoint_path.resolve():
        raise ValueError(f"Recipe checkpoint path mismatch: {run_dir}")
    with history_path.open(newline="") as stream:
        history = list(csv.DictReader(stream))
    if [int(r["epoch"]) for r in history] != list(range(1, epochs + 2)):
        raise ValueError(f"Incomplete or mismatched epoch history: {history_path}")
    best = None
    for row in history:
        identity = {"dataset": "clinc150", "seed": str(seed), "strategy": name, "mode": mode,
                    "split": "val", "phase": "warmup" if row["epoch"] == "1" else "finetune"}
        if any(row.get(k) != v for k, v in identity.items()) or not np.isfinite(float(row["oos_f1"])):
            raise ValueError(f"Recipe history mismatch: {history_path}")
        if best is None or float(row["oos_f1"]) > float(best["oos_f1"]) + 1e-12:
            best = row
    if any(str(result.get(k)) != v for k, v in best.items()):
        raise ValueError(f"Selected epoch does not match validation history: {run_dir}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state.get("mode") != mode or state.get("epoch") != result["epoch"] or not state.get("model"):
        raise ValueError(f"Checkpoint metadata mismatch: {checkpoint_path}")
    return result


def prepare_resume(artifact_root, output_root, manifest):
    """Validate every cell before allowing any new training or output writes."""
    prior = json.loads((output_root / "MANIFEST.json").read_text())
    expected = json.loads(json.dumps(manifest))
    if any(prior.get(k) != v for k, v in expected.items() if k not in {"status", "direct_pipeline_verified"}):
        raise ValueError("Resume configuration differs from the original run manifest")
    if prior.get("status") != "running" or prior.get("direct_pipeline_verified") is not False:
        raise ValueError("Resume requires an unfinished run before test evaluation")
    allowed = {"MANIFEST.json", "all_seeds_selection_lock.json", "candidate_validation.csv"}
    allowed.update(f"seed{seed}_selection_lock.json" for seed in manifest["seeds"])
    allowed.update(f"{name}_seed{seed}_training_history.csv"
                   for name, _, _ in RECIPES for seed in manifest["seeds"])
    if any(p.name not in allowed for p in output_root.iterdir()):
        raise ValueError("Resume refuses partial test results or unrecognized output files")
    if any(p.name not in {r[0] for r in RECIPES} for p in artifact_root.iterdir()):
        raise ValueError("Resume refuses selected/test artifacts or unrecognized recipe directories")
    return {(seed, recipe[0]): result for seed in manifest["seeds"] for recipe in RECIPES
            if (result := completed_recipe(seed, recipe, artifact_root, output_root, manifest["finetune_epochs"])) is not None}


def selection_lock(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing validation selection lock differs: {path}")
    else:
        dump(path, value)


def train_recipe(seed, recipe, views, device, artifact_root, output_root, epochs):
    name, mode, backbone_lr = recipe
    run_dir = artifact_root / name / "clinc150" / f"kir50_seed{seed}" / "trainable_k1"
    history_path = output_root / f"{name}_seed{seed}_training_history.csv"
    if history_path.exists() or (run_dir.exists() and any(run_dir.iterdir())):
        raise ValueError(f"Refusing to overwrite nonempty recipe: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / "all-MiniLM-L6-v2", local_files_only=True)
    model = build_racal_model(MODEL_ROOT / "all-MiniLM-L6-v2", mode, 256).to(device)
    labels = sorted({r["intent"] for r in views["train"]})
    label_map = {name: i for i, name in enumerate(labels)}
    targets = torch.tensor([label_map[r["intent"]] for r in views["train"]], dtype=torch.long, device=device)
    best = None
    best_state = None
    history = []
    start_time = time.time()
    for epoch in range(1, epochs + 2):  # One projection warmup, then eight full scheduled epochs.
        if epoch in (1, 2):
            set_phase(model, warmup=epoch == 1)
            optimizer = optimizer_for(model, backbone_lr)
        # Centers are detached reference targets; the training forward below retains gradients.
        refreshed = encode_rows(model, tokenizer, views["train"], device, 128, 256)
        centers_np, _, _ = _class_centers(refreshed, views["train"])
        centers = torch.tensor(centers_np, dtype=torch.float32, device=device)
        model.train()
        order = np.random.default_rng(seed + epoch * 7919).permutation(len(targets))
        losses = []
        for start in range(0, len(order), 64):
            indices = order[start:start + 64]
            tokens = tokenizer([views["train"][int(i)]["text"] for i in indices], padding=True,
                               truncation=True, max_length=256, return_tensors="pt").to(device)
            features = model(tokens)
            loss, _ = _center_losses(features, targets[torch.as_tensor(indices, device=device)], centers,
                                     .07, .1, .1, 1., .20)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        train = encode_rows(model, tokenizer, views["train"], device, 128, 256)
        val = encode_rows(model, tokenizer, views["val"], device, 128, 256)
        choice = validation_boundary(train, val, views, seed, name)
        record = {**choice, "epoch": epoch, "phase": "warmup" if epoch == 1 else "finetune",
                  "loss": float(np.mean(losses)), "mode": mode,
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "base_trainable_parameter_count": sum(p.numel() for n, p in model.named_parameters()
                                                        if p.requires_grad and not n.startswith("projection.") and "lora_" not in n),
                  "device": str(device), "elapsed_seconds": time.time() - start_time}
        history.append(record)
        # Exactly OOS F1; keep the earlier epoch on a tie.
        if best is None or choice["oos_f1"] > best["oos_f1"] + 1e-12:
            best = record
            best_state = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "mode": mode, "epoch": epoch, "label_map": label_map,
                          "freeze_report": model.freeze_report()}
        write_csv(history_path, history)
        print(f"TRAIN {name} seed={seed} epoch={epoch} val_oos={choice['oos_f1']:.6f} best={best['oos_f1']:.6f}", flush=True)
    torch.save(best_state, run_dir / "checkpoint.pt")
    result = {**best, "checkpoint": str(run_dir / "checkpoint.pt"), "oos_used_for_training": False,
              "test_used_for_selection": False, "checkpoint_objective": "validation_oos_f1_only",
              "backbone_lr": backbone_lr, "projection_lr": 2e-4}
    dump(run_dir / "run_manifest.json", result)
    del model, optimizer, best_state
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Reuse completed matching recipes before test; reject partial training")
    parser.add_argument("--artifact-root", type=Path, default=ROOT.parent / "artifacts/s2c/runs" / NAME)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/analysis" / NAME)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    args.artifact_root = args.artifact_root.resolve()
    args.output_root = args.output_root.resolve()
    manifest = {"experiment": NAME, "protocol": "historical_v19_paper_main", "evidence_level": "H1 controlled",
                "dataset": "clinc150", "kir": .5, "seeds": [13, 42, 87], "device": "cuda",
                "recipes": RECIPES, "finetune_epochs": args.epochs, "warmup_epochs": 1,
                "selection_objective": "validation_oos_f1_only", "selection_guard": "none",
                "train_oos_used": False, "test_used_for_selection": False, "test_access": "after_all_seed_recipe_locks",
                "boundary": "K1 diagonal Mahalanobis; predeclared lambda/threshold/mode grid",
                "lambdas": LAMBDAS, "thresholds": THRESHOLDS, "modes": MODES,
                "artifact_root": str(args.artifact_root), "status": "running", "direct_pipeline_verified": False}
    if args.resume:
        completed = prepare_resume(args.artifact_root, args.output_root, manifest)
    else:
        if args.output_root.exists() or args.artifact_root.exists():
            raise FileExistsError("Run roots already exist; use --resume for matching pre-test results")
        args.output_root.mkdir(parents=True)
        args.artifact_root.mkdir(parents=True)
        completed = {}
    torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.zeros(1, device=device)
    dump(args.output_root / "MANIFEST.json", manifest)
    selections = []
    candidates = []
    for seed in manifest["seeds"]:
        # Do not even load test rows during checkpoint selection.
        data = ROOT.parent / "assets/datasets/s2c/prepared/data/multidataset/v19/clinc150" / f"kir50_seed{seed}" / "gate"
        views = {split: json.loads((data / f"{split}.json").read_text()) for split in ("train", "val")}
        assert all(int(r["label"]) == 0 for r in views["train"])
        old_checkpoint = DEFAULT_H1_ROOT / "clinc150" / f"kir50_seed{seed}/trainable_k1/checkpoint.pt"
        encoder = _RacalGateEncoder(MODEL_ROOT / "all-MiniLM-L6-v2", old_checkpoint, device)
        values = {split: encoder.encode([r["text"] for r in rows], batch_size=128) for split, rows in views.items()}
        baseline = {**validation_boundary(values["train"], values["val"], views, seed, "existing_last2"),
                    "epoch": -1, "checkpoint": str(old_checkpoint), "mode": "last2_minilm_plus_projection",
                    "test_used_for_selection": False}
        del encoder, values
        torch.cuda.empty_cache()
        trials = [baseline]
        for recipe in RECIPES:
            result = completed.get((seed, recipe[0]))
            if result is None:
                result = train_recipe(seed, recipe, views, device, args.artifact_root, args.output_root, args.epochs)
            else:
                print(f"REUSE {recipe[0]} seed={seed} epoch={result['epoch']}", flush=True)
            trials.append(result)
        # Earlier candidate wins exact ties, preserving the existing checkpoint when possible.
        winner = max(trials, key=lambda r: r["oos_f1"])
        selections.append(winner)
        candidates.extend(trials)
        selection_lock(args.output_root / f"seed{seed}_selection_lock.json", {"candidates": trials, "selected": winner, "test_metrics_computed": False})
    selection_lock(args.output_root / "all_seeds_selection_lock.json", {"selected": selections, "test_metrics_computed": False})
    write_csv(args.output_root / "candidate_validation.csv", [{k: r.get(k, "") for k in
              ("dataset", "seed", "strategy", "epoch", "oos_f1", "known_f1", "accuracy", "known_recall", "radius_lambda", "threshold", "acceptance_mode", "checkpoint")} for r in candidates])
    # Copy only the locked winners into a fresh pipeline checkpoint layout.
    selected_root = args.artifact_root / "selected"
    results = []
    diagnostic_rows = []
    intent_errors = []
    with tempfile.TemporaryDirectory(prefix="s2c_checkpoint_direct_") as temp:
        for choice in selections:
            seed = choice["seed"]
            target = selected_root / "clinc150" / f"kir50_seed{seed}/trainable_k1/checkpoint.pt"
            target.parent.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(choice["checkpoint"], target)
            views, values = _encode_cell("clinc150", seed, device, selected_root)
            d = _fit_detector(values["train"], views["train"], 1, choice["radius_lambda"], choice["acceptance_mode"])
            output = _score_from_distances(_distance_matrix(d, values["test"]), d, d.acceptance_mode)
            gate = workpoints(d, output, views["test"], [choice["threshold"]])[0]
            predictions, meta = _run_pipeline("clinc150", seed, device, d, choice["threshold"], selected_root, Path(temp))
            metrics, stages = compute_metrics(views["test"], predictions)
            np.testing.assert_array_equal(output["score"] > choice["threshold"], [p["is_oos"] for p in predictions])
            known_count = len({r["intent"] for r in views["train"]})
            delta = abs(metrics["oos_f1"] - gate["oos_f1"])
            assert delta < 1e-10, "Gate/direct pipeline OOS mismatch"
            result = {"dataset": "clinc150", "seed": seed, "strategy": choice["strategy"], "epoch": choice["epoch"],
                      "radius_lambda": choice["radius_lambda"], "threshold": choice["threshold"], "acceptance_mode": choice["acceptance_mode"],
                      "validation_oos_f1": choice["oos_f1"], **gate,
                      "gate_f1_all": (known_count * gate["known_f1"] + gate["oos_f1"]) / (known_count + 1),
                      "device": "cuda", "direct_pipeline_verified": True,
                      "gate_pipeline_oos_abs_delta": delta, "test_used_for_selection": False,
                      "delta_oos_f1_pp": gate["oos_f1"] * 100 - PAPER["clinc150"]["oos_f1"],
                      **{"full_pipeline_" + k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                      "full_pipeline_f1_all": float(metrics["f1_all"]),
                      **{"full_pipeline_stage_" + k: v for k, v in stages.items()}}
            results.append(result)
            for split, selected in (("val", choice), ("test", gate)):
                scores = output if split == "test" else _score_from_distances(_distance_matrix(d, values[split]), d, d.acceptance_mode)
                diagnostic, errors = diagnostics(d, scores, views[split], selected, split)
                identity = {"dataset": "clinc150", "seed": seed, "strategy": choice["strategy"], "epoch": choice["epoch"]}
                diagnostic_rows.append({**identity, **diagnostic})
                intent_errors.extend({**identity, **r} for r in errors)
            del predictions
    write_csv(args.output_root / "selected_test_per_seed.csv", results)
    write_csv(args.output_root / "ranking_diagnostics.csv", diagnostic_rows)
    write_csv(args.output_root / "intent_errors.csv", intent_errors)
    summary = {"dataset": "clinc150", "seed_count": 3, "paper_oos_f1": PAPER["clinc150"]["oos_f1"],
               "direct_pipeline_verified": True, "std_ddof": 0}
    for metric in ("oos_f1", "known_recall", "false_accept_rate", "gate_f1_all", "full_pipeline_f1_all", "full_pipeline_known_macro_f1", "full_pipeline_overall_accuracy"):
        summary[metric + "_mean"] = float(np.mean([r[metric] for r in results]))
        summary[metric + "_std"] = float(np.std([r[metric] for r in results]))
    summary["delta_oos_f1_pp"] = summary["oos_f1_mean"] * 100 - summary["paper_oos_f1"]
    summary["seeds_beating_paper"] = sum(r["delta_oos_f1_pp"] > 0 for r in results)
    write_csv(args.output_root / "selected_test_summary.csv", [summary])
    manifest.update(status="complete", direct_pipeline_verified=True, completed_unit_count=9,
                    completed_direct_units=3, checkpoint_candidates_per_seed=4)
    dump(args.output_root / "MANIFEST.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
