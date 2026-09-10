#!/usr/bin/env python3
"""Known-only CLINC KIR=.50 recipe search with locked full-pipeline confirmation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from protocol_v2.experiments.racal_v1.representation import build_racal_model, encode_rows, set_seed
from protocol_v2.experiments.racal_v1.runner import _center_losses, _class_centers
from scripts.experiments.run_historical_trainable_checkpoint_selection import (
    optimizer_for, set_phase, validation_boundary,
)
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder
from tools.eval.run_historical_trainable_parameter_full_pipeline import (
    DEFAULT_H1_ROOT, MODEL_ROOT, _fit_detector, _metrics_from_output, _read_views,
    _vectorized_output, write_csv,
)
from scripts.experiments import run_historical_trainable_gate_search as geometry_search

NAME = "historical_trainable_gate_recipe_search"
DATASET = "clinc150"
SEEDS = (13, 42, 87)
RECIPES = (
    {"name": "last2_lr1e5", "mode": "last2_minilm_plus_projection", "backbone_lr": 1e-5, "temperature": .07, "intra": .1, "inter": .1, "margin": .20},
    {"name": "last2_temp10", "mode": "last2_minilm_plus_projection", "backbone_lr": 2e-5, "temperature": .10, "intra": .1, "inter": .1, "margin": .20},
    {"name": "last2_no_inter", "mode": "last2_minilm_plus_projection", "backbone_lr": 2e-5, "temperature": .07, "intra": .1, "inter": 0., "margin": .20},
    {"name": "lora_lr1e5", "mode": "lora_minilm_plus_projection", "backbone_lr": 1e-5, "temperature": .07, "intra": .1, "inter": .1, "margin": .20},
)


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def recipe_checkpoint(root: Path, recipe: dict[str, object], seed: int) -> Path:
    return root / str(recipe["name"]) / DATASET / f"kir50_seed{seed}" / "trainable_k1" / "checkpoint.pt"


def train_recipe(recipe: dict[str, object], seed: int, views: dict[str, list[dict[str, object]]], device: torch.device, artifact_root: Path, result_root: Path, epochs: int) -> dict[str, object]:
    name = str(recipe["name"])
    checkpoint_path = recipe_checkpoint(artifact_root, recipe, seed)
    history_path = result_root / f"{name}_seed{seed}_training_history.csv"
    run_dir = checkpoint_path.parent
    if run_dir.exists() or history_path.exists():
        raise FileExistsError(f"Refusing to overwrite recipe output: {run_dir}")
    run_dir.mkdir(parents=True)
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / "all-MiniLM-L6-v2", local_files_only=True)
    model = build_racal_model(MODEL_ROOT / "all-MiniLM-L6-v2", str(recipe["mode"]), 256).to(device)
    labels = sorted({str(row["intent"]) for row in views["train"]})
    label_map = {label: index for index, label in enumerate(labels)}
    targets = torch.tensor([label_map[str(row["intent"])] for row in views["train"]], dtype=torch.long, device=device)
    best = None
    best_state = None
    history = []
    for epoch in range(1, epochs + 2):
        set_phase(model, warmup=epoch == 1)
        optimizer = optimizer_for(model, float(recipe["backbone_lr"]))
        refreshed = encode_rows(model, tokenizer, views["train"], device, 128, 256)
        centers_np, _, _ = _class_centers(refreshed, views["train"])
        centers = torch.tensor(centers_np, dtype=torch.float32, device=device)
        model.train()
        order = np.random.default_rng(seed + epoch * 7919).permutation(len(targets))
        losses = []
        for start in range(0, len(order), 64):
            indices = order[start:start + 64]
            tokens = tokenizer([str(views["train"][int(index)]["text"]) for index in indices], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            features = model(tokens)
            loss, _ = _center_losses(features, targets[torch.as_tensor(indices, device=device)], centers, float(recipe["temperature"]), float(recipe["intra"]), float(recipe["inter"]), 1., float(recipe["margin"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_values = encode_rows(model, tokenizer, views["train"], device, 128, 256)
        val_values = encode_rows(model, tokenizer, views["val"], device, 128, 256)
        choice = validation_boundary(train_values, val_values, views, seed, name)
        record = {**choice, "epoch": epoch, "phase": "warmup" if epoch == 1 else "finetune", "mode": recipe["mode"], "loss": float(np.mean(losses)), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "device": str(device)}
        history.append(record)
        if best is None or float(choice["oos_f1"]) > float(best["oos_f1"]) + 1e-12:
            best = record
            best_state = {"model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, "mode": recipe["mode"], "epoch": epoch, "label_map": label_map, "freeze_report": model.freeze_report()}
        write_csv(history_path, history)
        print(f"TRAIN {name} seed={seed} epoch={epoch} val_oos={choice['oos_f1']:.6f} best={best['oos_f1']:.6f}", flush=True)
    torch.save(best_state, checkpoint_path)
    result = {**best, **recipe, "checkpoint": str(checkpoint_path), "oos_used_for_training": False, "test_used_for_selection": False, "checkpoint_objective": "validation_oos_f1_only"}
    dump(run_dir / "run_manifest.json", result)
    del model, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def gate_metrics(detector, values, rows, threshold):
    raw = _metrics_from_output(detector, _vectorized_output(detector, values), rows, threshold)
    return {"oos_f1": float(raw["f1_u"]), "known_f1": float(raw["f1_k"]), "accuracy": float(raw["accuracy"]), "known_recall": float(raw["known_recall"]), "false_accept_rate": float(raw["false_accept_rate"]), "auroc": float(raw["auroc"]), "aupr_oos": float(raw["aupr_oos"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--artifact-root", type=Path, default=ROOT.parent / "artifacts/s2c/runs" / NAME)
    parser.add_argument("--result-root", type=Path, default=ROOT / "results/analysis" / NAME)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    artifact_root = args.artifact_root.resolve(); result_root = args.result_root.resolve()
    if result_root.exists() or artifact_root.exists():
        raise FileExistsError("Recipe search roots already exist; use a fresh root")
    result_root.mkdir(parents=True); artifact_root.mkdir(parents=True)
    started = time.time()
    candidates = []
    for seed in SEEDS:
        views = _read_views(DATASET, seed)
        old_checkpoint = DEFAULT_H1_ROOT / DATASET / f"kir50_seed{seed}" / "trainable_k1" / "checkpoint.pt"
        encoder = _RacalGateEncoder(MODEL_ROOT / "all-MiniLM-L6-v2", old_checkpoint, device)
        old_values = {split: encoder.encode([str(row["text"]) for row in views[split]], batch_size=128) for split in ("train", "val")}
        del encoder
        old_choice = validation_boundary(old_values["train"], old_values["val"], views, seed, "existing_last2")
        candidates.append({**old_choice, "epoch": -1, "strategy": "existing_last2", "mode": "last2_minilm_plus_projection", "checkpoint": str(old_checkpoint), "test_used_for_selection": False})
        del old_values
        if device.type == "cuda": torch.cuda.empty_cache()
        for recipe in RECIPES:
            candidates.append(train_recipe(recipe, seed, views, device, artifact_root, result_root, args.epochs))
    locks = []
    for seed in SEEDS:
        trials = [row for row in candidates if int(row["seed"]) == seed]
        winner = min(trials, key=lambda row: (-float(row["oos_f1"]), int(row.get("epoch", -1)), row["strategy"]))
        locks.append({**winner, "selection_objective": "validation_oos_f1_only", "test_metrics_computed": False})
    dump(result_root / "selection_lock.json", {"choices": locks, "test_metrics_computed": False})
    write_csv(result_root / "candidate_validation.csv", [{key: row.get(key, "") for key in ("seed", "strategy", "epoch", "mode", "oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate", "radius_lambda", "threshold", "acceptance_mode", "checkpoint")} for row in candidates])
    selected_test = []
    for lock in locks:
        seed = int(lock["seed"]); views = _read_views(DATASET, seed); checkpoint = Path(str(lock["checkpoint"]))
        encoder = _RacalGateEncoder(MODEL_ROOT / "all-MiniLM-L6-v2", checkpoint, device)
        values = {split: encoder.encode([str(row["text"]) for row in views[split]], batch_size=128) for split in ("train", "test")}; del encoder
        detector = _fit_detector(values["train"], views["train"], 1, float(lock["radius_lambda"]), str(lock["acceptance_mode"]))
        gate = gate_metrics(detector, values["test"], views["test"], float(lock["threshold"]))
        pipeline = geometry_search._pipeline_confirm(detector, checkpoint, seed, device, views, float(lock["threshold"]))
        selected_test.append({**lock, "test_used_for_selection": False, **{f"gate_{key}": value for key, value in gate.items()}, **{f"pipeline_{key}": value for key, value in pipeline.items()}})
        del values
        if device.type == "cuda": torch.cuda.empty_cache()
    write_csv(result_root / "selected_test.csv", selected_test)
    dump(result_root / "selection_lock.json", {"choices": locks, "test_metrics_computed": True})
    summary = {"experiment": NAME, "protocol": "historical_v19_paper_main", "dataset": DATASET, "kir": .50, "seeds": list(SEEDS), "recipes": [recipe["name"] for recipe in RECIPES], "device": str(device), "completed_training_units": len(RECIPES) * len(SEEDS), "completed_confirmation_units": len(selected_test), "test_used_for_selection": False, "elapsed_seconds": time.time() - started}
    dump(result_root / "MANIFEST.json", summary)


if __name__ == "__main__":
    main()
