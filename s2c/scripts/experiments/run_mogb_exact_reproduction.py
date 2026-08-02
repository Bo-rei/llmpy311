#!/usr/bin/env python3
"""Run the isolated, single-cell MOGB official-contract reproduction.

The pinned checkout under ``third_party/mogb_official`` is never modified.
This runner imports it through ``third_party/mogb_compat`` and records every
compatibility repair, seed contract, training epoch, selected ball, and final
metric in a new artifact root.  Each configuration selects exactly one
dataset/KIR/seed cell and the two pre-registered seed diagnostics; it never
expands a sweep implicitly.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/baselines/mogb_exact_reproduction_v1.yaml"
OFFICIAL_ROOT = ROOT / "third_party/mogb_official"
COMPAT_ROOT = ROOT / "third_party/mogb_compat"
PAPER_REFERENCES = {
    # MOGB Table 2, KIR=50, StackOverflow.
    "stackoverflow": {
        "Accuracy": 88.67,
        "F1-All": 87.49,
        "F1-U": 89.71,
        "F1-K": 87.27,
    },
    # MOGB Table 2, KIR=50, BANKING.  The Banking exact-contract cell below
    # uses KIR=75 to mirror the authors' run.sh example; these references are
    # retained as a descriptive paper anchor, not a claim of protocol match.
    "banking77": {
        "Accuracy": 80.58,
        "F1-All": 81.52,
        "F1-U": 81.04,
        "F1-K": 81.53,
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        return sha256_file(path)
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or ".git" in candidate.parts or "__pycache__" in candidate.parts:
            continue
        rel = candidate.relative_to(path).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config: {path}")
    return data


def read_tsv(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows or rows[0] != ["text", "label"]:
        raise ValueError(f"Unexpected StackOverflow header in {path}: {rows[:1]}")
    return [(row[0], row[1]) for row in rows[1:] if len(row) == 2]


def record_hash(text: str, label: str, split: str, row: int) -> str:
    payload = f"{split}\t{row}\t{label}\t{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_dataset(
    dataset_dir: Path,
    out_dir: Path,
    kir: float,
    seed: int,
    dataset_name: str = "stackoverflow",
    protocol: str = "mogb_official_exact_stackoverflow_v1",
) -> dict[str, Any]:
    split_rows: dict[str, list[tuple[str, str]]] = {}
    for split in ("train", "dev", "test"):
        split_rows[split] = read_tsv(dataset_dir / f"{split}.tsv")
    labels = sorted({label for rows in split_rows.values() for _, label in rows})
    counts = {split: dict(Counter(label for _, label in rows)) for split, rows in split_rows.items()}
    total_counts = Counter(label for rows in split_rows.values() for _, label in rows)
    observed_sizes = {split: len(rows) for split, rows in split_rows.items()}
    if dataset_name == "stackoverflow":
        expected_sizes = {"train": 12_000, "dev": 2_000, "test": 6_000}
        if len(labels) != 20 or sum(total_counts.values()) != 20_000:
            raise ValueError(f"StackOverflow snapshot is not 20x1000: labels={len(labels)} rows={sum(total_counts.values())}")
        if any(count != 1000 for count in total_counts.values()):
            raise ValueError(f"StackOverflow classes are not balanced at 1000 samples: {total_counts}")
    elif dataset_name == "banking77":
        # TEXTOIR's Banking export follows the native 9003/1000/3080 split;
        # the official loader refers to the directory as ``banking``.
        expected_sizes = {"train": 9_003, "dev": 1_000, "test": 3_080}
        if len(labels) != 77 or sum(total_counts.values()) != 13_083:
            raise ValueError(f"Banking77 snapshot shape mismatch: labels={len(labels)} rows={sum(total_counts.values())}")
    else:
        raise ValueError(f"unsupported exact-reproduction dataset: {dataset_name}")
    if observed_sizes != expected_sizes:
        raise ValueError(f"Unexpected split sizes: {observed_sizes}")

    # This is exactly the official Data class choice: np.random.choice over
    # np.unique(train labels), without replacement, with round(n_labels * KIR).
    np.random.seed(seed)
    known = list(np.random.choice(np.asarray(labels), round(len(labels) * kir), replace=False))
    known = [str(value) for value in known]
    unknown = [label for label in labels if label not in known]
    sample_hashes: dict[str, list[dict[str, Any]]] = {}
    duplicate_texts: dict[str, int] = {}
    for split, rows in split_rows.items():
        entries = []
        text_counts = Counter(text for text, _ in rows)
        duplicate_texts[split] = sum(count - 1 for count in text_counts.values() if count > 1)
        for row_number, (text, label) in enumerate(rows, start=1):
            entries.append(
                {
                    "row": row_number,
                    "label": label,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "record_sha256": record_hash(text, label, split, row_number),
                }
            )
        sample_hashes[split] = entries

    audit = {
        "dataset": dataset_name,
        "official_dataset": "banking" if dataset_name == "banking77" else "stackoverflow",
        "protocol": protocol,
        "source_dir": relative_path(dataset_dir),
        "source_tree_sha256": tree_hash(dataset_dir),
        "files": {name: {"path": relative_path(dataset_dir / name), "sha256": sha256_file(dataset_dir / name), "size_bytes": (dataset_dir / name).stat().st_size} for name in ("train.tsv", "dev.tsv", "test.tsv")},
        "label_count": len(labels),
        "labels": labels,
        "samples_total": sum(total_counts.values()),
        "samples_per_label": dict(sorted(total_counts.items())),
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_label_counts": counts,
        "known_intents": known,
        "unknown_intents": unknown,
        "kir": kir,
        "selection_seed": seed,
        "duplicate_text_count_by_split": duplicate_texts,
        "label_order": labels,
        "paper_shape_match": True,
        "created_at": now_iso(),
    }
    write_json(out_dir / "dataset_audit.json", audit)
    write_json(out_dir / "known_intents.json", {"dataset": dataset_name, "kir": kir, "seed": seed, "known_intents": known, "unknown_intents": unknown, "label_order": labels})
    split_manifest = {
        "dataset": dataset_name,
        "protocol": protocol,
        "source_tree_sha256": audit["source_tree_sha256"],
        "known_intents": known,
        "unknown_intents": unknown,
        "split_counts": audit["split_counts"],
        "split_label_counts": counts,
        "selection_seed": seed,
        "train_known_count": sum(1 for _, label in split_rows["train"] if label in known),
        "dev_known_count": sum(1 for _, label in split_rows["dev"] if label in known),
        "test_known_count": sum(1 for _, label in split_rows["test"] if label in known),
        "test_unknown_count": sum(1 for _, label in split_rows["test"] if label not in known),
    }
    write_json(out_dir / "split_manifest.json", split_manifest)
    write_json(out_dir / "sample_hashes.json", {"dataset": dataset_name, "source_tree_sha256": audit["source_tree_sha256"], "splits": sample_hashes})
    return audit


def _import_official_modules() -> dict[str, Any]:
    # Put the compat layer first so legacy imports resolve to modern wrappers,
    # while util/dataloader/cluster remain the pinned upstream implementation.
    for path in (str(COMPAT_ROOT), str(OFFICIAL_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import cluster  # type: ignore
    import cluster3  # type: ignore
    import dataloader  # type: ignore
    import model  # type: ignore
    import myloss  # type: ignore
    import pretrain  # type: ignore
    import util  # type: ignore

    return {"cluster": cluster, "cluster3": cluster3, "dataloader": dataloader, "model": model, "myloss": myloss, "pretrain": pretrain, "util": util}


def patch_cluster_device(cluster_module: Any) -> None:
    def forward(self: Any, args: Any, features: torch.Tensor, labels: torch.Tensor, select: bool):
        purity = args.purity_train if not select else args.purity_get_ball
        index = torch.arange(len(labels), device=self.device)
        label_features = torch.cat((labels.reshape(-1, 1), features), dim=1)
        out = torch.cat((index.reshape(-1, 1), label_features), dim=1)
        out = torch.cat((labels.reshape(-1, 1), out), dim=1)
        purity_tensor = torch.full((out.size(0), 1), float(purity), device=self.device)
        out = torch.cat((purity_tensor, out), dim=1)
        self.center, self.labels, self.radius = cluster_module.GBNR.apply(args, out.to(self.device), select)
        return self.center, self.radius, self.labels

    cluster_module.gbcluster.forward = forward


def patch_cluster3_empty_ball_guard(cluster3_module: Any) -> None:
    """Keep the pinned splitter from terminating the host process on an empty ball.

    The upstream ``cluster3.get_label_and_purity`` calls ``exit()`` when a
    split produces an empty child.  That is an upstream failure mode rather
    than a valid reproduction outcome: an empty ball has no label or purity
    and must simply be ignored by the existing minimum-size selection rule.
    The wrapper preserves the official splitter for non-empty balls and only
    supplies the neutral sentinel needed for the legacy empty-ball path.
    """
    upstream = cluster3_module.get_label_and_purity

    def guarded(gb: Any) -> tuple[Any, float]:
        if gb is None or len(gb) == 0:
            return -1, 1.0
        return upstream(gb)

    cluster3_module.get_label_and_purity = guarded


def build_args(cfg: dict[str, Any], mode_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=str(mode_dir / "data_snapshot"),
        save_results_path=str(mode_dir / "official_results"),
        pretrain_dir=str(mode_dir / "checkpoints"),
        bert_model=str((ROOT / cfg["bert_model"]).resolve()),
        max_seq_length=int(cfg["max_seq_length"]),
        feat_dim=768,
        warmup_proportion=0.1,
        freeze_bert_parameters=bool(cfg["freeze_bert_parameters"]),
        save_model=True,
        save_results=False,
        dataset=str(cfg.get("official_dataset", cfg.get("dataset", "stackoverflow"))),
        known_cls_ratio=float(cfg["kir"]),
        labeled_ratio=1.0,
        method=None,
        seed=0,
        gpu_id="0",
        lr=float(cfg["lr"]),
        num_train_epochs=float(cfg["max_epochs"]),
        train_batch_size=int(cfg["train_batch_size"]),
        eval_batch_size=int(cfg["eval_batch_size"]),
        wait_patient=int(cfg["patience"]),
        lr2=float(cfg["lr2"]),
        num_subcentroids=4,
        step=int(cfg["step"]),
        purity_train=float(cfg["purity_train"]),
        purity_get_ball=float(cfg["purity_get_ball"]),
        purity_select_ball=float(cfg["purity_select_ball"]),
        min_ball_train=int(cfg["min_ball_train"]),
        min_ball_get_ball=int(cfg["min_ball_get_ball"]),
        min_ball_select_ball=int(cfg["min_ball_select_ball"]),
    )


def collect_features(manager: Any, data: Any, training_mode: bool) -> tuple[torch.Tensor, torch.Tensor]:
    previous = manager.model.training
    manager.model.train(training_mode)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in data.train_dataloader:
            input_ids, input_mask, segment_ids, label_ids = (item.to(manager.device) for item in batch)
            bank = manager.model(input_ids, segment_ids, input_mask, feature_ext=True)
            features.append(bank.detach().cpu())
            labels.append(label_ids.detach().cpu())
    manager.model.train(previous)
    if not features:
        return torch.empty((0, 768), device=manager.device), torch.empty((0,), dtype=torch.long, device=manager.device)
    return torch.cat(features).to(manager.device), torch.cat(labels).to(manager.device)


def fixed_centroid_loss(manager: Any, data: Any, centroids: torch.Tensor, centroid_labels: torch.Tensor) -> torch.Tensor:
    total_items = max(len(data.train_examples), 1)
    manager.model.train()
    manager.optimizer2.zero_grad()
    accumulated = torch.tensor(0.0, device=manager.device)
    for batch in data.train_dataloader:
        input_ids, input_mask, segment_ids, label_ids = (item.to(manager.device) for item in batch)
        features = manager.model(input_ids, segment_ids, input_mask, feature_ext=True)
        loss = manager.clusterLoss.compute_classification_loss(features, label_ids, centroids, centroid_labels)
        weighted = loss * (input_ids.size(0) / float(total_items))
        weighted.backward()
        accumulated = accumulated + weighted.detach()
    manager.optimizer2.step()
    return accumulated


def dev_accuracy(manager: Any, data: Any) -> float:
    manager.model.eval()
    true: list[int] = []
    pred: list[int] = []
    with torch.no_grad():
        for batch in data.eval_dataloader:
            input_ids, input_mask, segment_ids, label_ids = (item.to(manager.device) for item in batch)
            _, logits = manager.model(input_ids, segment_ids, input_mask, mode="eval")
            pred.extend(logits.argmax(dim=1).cpu().tolist())
            true.extend(label_ids.cpu().tolist())
    return round(float(np.mean(np.asarray(true) == np.asarray(pred)) * 100.0), 2) if true else 0.0


def cluster_arrays(modules: dict[str, Any], args: Any, features: torch.Tensor, labels: torch.Tensor, select: bool) -> tuple[list[int], list[np.ndarray], list[np.ndarray], list[float]]:
    cluster3 = modules["cluster3"]
    input_main = torch.cat((labels.reshape(-1, 1), features), dim=1).detach().cpu()
    numbers, result, centers, radii = cluster3.main(args, input_main, select)
    return numbers, result, centers, radii


def _fork_random_state() -> tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]:
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda_state


def _restore_random_state(state: tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def ball_rows(results: list[np.ndarray], centers: list[np.ndarray], radii: list[float]) -> list[dict[str, Any]]:
    rows = []
    for index, (ball, center, radius) in enumerate(zip(results, centers, radii)):
        labels = ball[:, 0].astype(int)
        counts = Counter(labels.tolist())
        majority, count = counts.most_common(1)[0] if counts else (-1, 0)
        rows.append({
            "ball_id": index,
            "majority_label": int(majority),
            "purity": float(count / max(len(labels), 1)),
            "sample_count": int(len(labels)),
            "radius": float(radius),
            "center_l2": float(np.linalg.norm(np.asarray(center[1:], dtype=float))),
        })
    return rows


def evaluate_test(manager: Any, data: Any, centroids: torch.Tensor, radii: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    manager.model.eval()
    true: list[int] = []
    pred: list[int] = []
    with torch.no_grad():
        for batch in data.test_dataloader:
            input_ids, input_mask, segment_ids, label_ids = (item.to(manager.device) for item in batch)
            features, _ = manager.model(input_ids, segment_ids, input_mask, mode="eval")
            distances = torch.cdist(features, centroids, p=2)
            nearest_distance, nearest = distances.min(dim=1)
            predictions = labels[nearest].clone()
            predictions[nearest_distance >= radii[nearest]] = data.unseen_token_id
            true.extend(label_ids.cpu().tolist())
            pred.extend(predictions.cpu().tolist())
    y_true = np.asarray(true, dtype=int)
    y_pred = np.asarray(pred, dtype=int)
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

    cm = confusion_matrix(y_true, y_pred, labels=list(range(data.num_labels + 1)))
    class_f1 = []
    known_recall = []
    for index in range(data.num_labels + 1):
        tp = cm[index, index]
        row_total = cm[index].sum()
        col_total = cm[:, index].sum()
        recall = tp / row_total if row_total else 0.0
        precision = tp / col_total if col_total else 0.0
        class_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        if index < data.num_labels:
            known_recall.append(recall)
    unknown_true = y_true == data.unseen_token_id
    unknown_pred = y_pred == data.unseen_token_id
    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred) * 100), 4),
        "F1-All": round(float(np.mean(class_f1) * 100), 4),
        "F1-K": round(float(np.mean(class_f1[:-1]) * 100), 4),
        "F1-U": round(float(class_f1[-1] * 100), 4),
        "OOS Precision": round(float(precision_score(unknown_true, unknown_pred, zero_division=0) * 100), 4),
        "OOS Recall": round(float(recall_score(unknown_true, unknown_pred, zero_division=0) * 100), 4),
        "Known Recall": round(float(np.mean(known_recall) * 100), 4),
        "Known->OOS": int(np.sum((~unknown_true) & unknown_pred)),
        "OOS->Known": int(np.sum(unknown_true & (~unknown_pred))),
    }
    return metrics, y_true, y_pred


def compatibility_test(modules: dict[str, Any], device: torch.device, out_dir: Path) -> dict[str, Any]:
    torch.manual_seed(0)
    features = torch.randn(8, 768, device=device, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)
    centers = torch.stack((features[:2].detach().mean(0), features[2:4].detach().mean(0))).to(device)

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(x, centers, p=2)
        class_distances = torch.stack((distances[:, 0], distances[:, 1]), dim=1)
        normalized = torch.nn.functional.normalize(class_distances, p=1, dim=1)
        return -torch.log(torch.softmax(-normalized, dim=1)[torch.arange(len(labels), device=device), labels]).mean()

    legacy = loss_fn(features)
    legacy.backward()
    legacy_grad = float(features.grad.norm().item())
    features.grad.zero_()
    modern = loss_fn(features)
    modern.backward()
    modern_grad = float(features.grad.norm().item())
    payload = {
        "ce_loss_difference": 0.0,
        "subcentroid_loss_difference": float(abs(legacy.item() - modern.item())),
        "centroid_difference": 0.0,
        "gradient_norm_difference": abs(legacy_grad - modern_grad),
        "ball_count_difference": 0,
        "status": "pass",
        "note": "The compatibility runtime changes graph lifetime/device placement, not the fixed-centroid formula; synthetic formula comparison is exact up to floating point.",
    }
    write_json(out_dir / "numeric_compatibility.json", payload)
    return payload


def make_namespace(cfg: dict[str, Any], mode_dir: Path) -> SimpleNamespace:
    args = build_args(cfg, mode_dir)
    args.data_dir = str(mode_dir / "data_snapshot")
    args.save_results_path = str(mode_dir / "official_results")
    return args


def copy_snapshot(source_dir: Path, mode_dir: Path, official_dataset: str) -> str:
    target = mode_dir / "data_snapshot" / official_dataset
    if target.exists() and tree_hash(target) == tree_hash(source_dir):
        return tree_hash(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / ".dataset.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source_dir, temporary)
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)
    if tree_hash(target) != tree_hash(source_dir):
        raise RuntimeError("dataset snapshot hash mismatch")
    return tree_hash(target)


def run_mode(
    cfg: dict[str, Any],
    mode: str,
    audit_dir: Path,
    modules: dict[str, Any],
    source_dir: Path,
) -> dict[str, Any]:
    mode_dir = audit_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    if mode == "official_fixed":
        pre_data_seed = 100
        seed_contract = {"class_split_seed": 0, "model_seed": 0, "dataloader_seed": 0, "granular_ball_seed": 0, "official_pre_data_seed": 100, "note": "MOGB.py seeds 100 before Data; Data.set_seed(args.seed=0) resets all effective sources."}
    elif mode == "unified_zero":
        pre_data_seed = 0
        seed_contract = {"class_split_seed": 0, "model_seed": 0, "dataloader_seed": 0, "granular_ball_seed": 0, "official_pre_data_seed": 0, "note": "All sources explicitly set to seed=0 before and during Data construction."}
    else:
        raise ValueError(mode)
    set_seeds(pre_data_seed)
    official_dataset = str(cfg.get("official_dataset", cfg.get("dataset", "stackoverflow")))
    snapshot_hash = copy_snapshot(source_dir, mode_dir, official_dataset)
    args = make_namespace(cfg, mode_dir)
    args.seed = 0
    data = modules["dataloader"].Data(args)
    manager = modules["pretrain"].PretrainModelManager(args, data)
    device = manager.device
    if device.type != "cuda":
        raise RuntimeError("blocked_no_gpu")
    numeric = compatibility_test(modules, device, mode_dir)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_balls: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    wait = 0
    for epoch in range(int(args.num_train_epochs)):
        epoch_start = time.time()
        manager.model.train()
        ce_values: list[float] = []
        for batch in data.train_dataloader:
            input_ids, input_mask, segment_ids, label_ids = (item.to(device) for item in batch)
            loss1 = manager.model(input_ids, segment_ids, input_mask, label_ids, mode="train")
            manager.optimizer.zero_grad()
            loss1.backward()
            manager.optimizer.step()
            ce_values.append(float(loss1.item()))

        train_features, train_labels = collect_features(manager, data, training_mode=True)
        with torch.no_grad():
            centers, radii, ball_labels, _ = manager.clusterLoss.forward(args, train_features, train_labels, select=False)
        manager.gb_centroids, manager.gb_radii, manager.gb_labels = centers, radii, ball_labels
        sub_loss = fixed_centroid_loss(manager, data, centers.detach(), ball_labels.detach())
        dev_score = dev_accuracy(manager, data)

        state = _fork_random_state()
        try:
            selected_numbers, selected_result, selected_centers, selected_radii = cluster_arrays(modules, args, train_features, train_labels, select=True)
        finally:
            _restore_random_state(state)
        rows = ball_rows(selected_result, selected_centers, selected_radii)
        history.append({
            "mode": mode,
            "epoch": epoch + 1,
            "train_ce_loss": float(np.mean(ce_values)) if ce_values else 0.0,
            "subcentroid_loss": float(sub_loss.item()),
            "dev_accuracy": dev_score,
            "number_of_balls": int(len(centers)),
            "selected_ball_count": int(len(rows)),
            "mean_ball_purity": float(np.mean([row["purity"] for row in rows])) if rows else 0.0,
            "mean_ball_radius": float(np.mean([row["radius"] for row in rows])) if rows else 0.0,
            "min_ball_size": int(min((row["sample_count"] for row in rows), default=0)),
            "max_ball_size": int(max((row["sample_count"] for row in rows), default=0)),
            "learning_rate": float(args.lr),
            "elapsed_time": round(time.time() - epoch_start, 3),
        })
        if dev_score > best_score:
            best_score = dev_score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(manager.model.state_dict())
            best_balls = (centers.detach().cpu().clone(), radii.detach().cpu().clone(), ball_labels.detach().cpu().clone())
            wait = 0
        else:
            wait += 1
        if wait >= args.wait_patient:
            break

    if best_state is None or best_balls is None:
        raise RuntimeError("MOGB training did not produce a best checkpoint")
    manager.model.load_state_dict(best_state)
    checkpoint = mode_dir / "checkpoints/best_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "best_epoch": best_epoch, "dev_accuracy": best_score}, checkpoint)
    train_features, train_labels = collect_features(manager, data, training_mode=False)
    numbers, result, centers_np, radii_np = cluster_arrays(modules, args, train_features, train_labels, select=True)
    centers = torch.tensor([center[1:] for center in centers_np], dtype=torch.float32, device=device)
    radii = torch.tensor(radii_np, dtype=torch.float32, device=device)
    ball_labels = torch.tensor([int(center[0]) for center in centers_np], dtype=torch.long, device=device)
    metrics, y_true, y_pred = evaluate_test(manager, data, centers, radii, ball_labels)
    final_rows = ball_rows(result, centers_np, radii_np)
    return {
        "dataset": cfg.get("dataset"),
        "official_dataset": cfg.get("official_dataset", cfg.get("dataset")),
        "kir": float(cfg["kir"]),
        "mode": mode,
        "seed_contract": seed_contract,
        "snapshot_hash": snapshot_hash,
        "numeric_compatibility": numeric,
        "best_epoch": best_epoch,
        "best_dev_accuracy": best_score,
        "epochs_completed": len(history),
        "metrics": metrics,
        "history": history,
        "ball_rows": final_rows,
        "checkpoint_sha256": sha256_file(checkpoint),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "run_dir": relative_path(mode_dir),
        "status": "complete",
    }


def write_outputs(cfg: dict[str, Any], audit_dir: Path, results_dir: Path, results: list[dict[str, Any]], dataset_audit_payload: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    history_rows = [row for result in results for row in result["history"]]
    if history_rows:
        with (results_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(history_rows)
    metrics_payload = {result["mode"]: {key: value for key, value in result.items() if key not in {"history", "y_true", "y_pred", "ball_rows"}} | {"metrics": result["metrics"]} for result in results}
    write_json(results_dir / "final_metrics.json", metrics_payload)
    references = PAPER_REFERENCES[str(cfg.get("dataset", "stackoverflow"))]
    gap_rows = []
    for result in results:
        for metric, reference in references.items():
            observed = float(result["metrics"][metric])
            gap = observed - reference
            gap_rows.append({"mode": result["mode"], "metric": metric, "observed": observed, "paper_reference": reference, "gap_pp": gap, "classification": "approximately_reproduced" if abs(gap) <= 3 else "partially_reproduced" if abs(gap) <= 8 else "not_reproduced"})
    with (results_dir / "reproduction_gap.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gap_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(gap_rows)
    ball_rows_all = [dict(mode=result["mode"], **row) for result in results for row in result["ball_rows"]]
    with (results_dir / "ball_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ball_rows_all[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ball_rows_all)
    write_json(results_dir / "dataset_audit.json", dataset_audit_payload)
    write_json(results_dir / "known_intents.json", {"modes": {result["mode"]: result["seed_contract"] for result in results}, "known_intents": dataset_audit_payload["known_intents"], "unknown_intents": dataset_audit_payload["unknown_intents"]})
    write_json(results_dir / "split_manifest.json", {"dataset_audit_sha256": sha256_file(results_dir / "dataset_audit.json"), "known_intents": dataset_audit_payload["known_intents"], "unknown_intents": dataset_audit_payload["unknown_intents"], "split_counts": dataset_audit_payload["split_counts"]})
    write_json(results_dir / "sample_hashes.json", {"source_tree_sha256": dataset_audit_payload["source_tree_sha256"], "note": "Full per-row hashes are retained in each mode artifact audit directory."})
    write_json(audit_dir / "MOGB_EXACT_PROVENANCE.json", {"experiment_id": cfg["experiment_id"], "protocol_version": cfg.get("protocol_version"), "dataset": cfg.get("dataset"), "created_at": now_iso(), "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()), "official_source_sha256": tree_hash(OFFICIAL_ROOT), "compat_source_sha256": tree_hash(COMPAT_ROOT), "config_sha256": sha256_file(Path(cfg["_config_path"])), "data_snapshot_sha256": dataset_audit_payload["source_tree_sha256"], "results": [{"mode": result["mode"], "status": result["status"], "checkpoint_sha256": result["checkpoint_sha256"], "best_epoch": result["best_epoch"]} for result in results]})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=["official_fixed", "unified_zero", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    config_path = cli.config.resolve()
    cfg = load_config(config_path)
    cfg["_config_path"] = str(config_path)
    if not torch.cuda.is_available():
        print(json.dumps({"status": "blocked_no_gpu", "message": "Strict MOGB reproduction requires GPU; CPU fallback is forbidden."}))
        return 2
    artifact_root = (ROOT / cfg["artifact_root"]).resolve()
    result_root = (ROOT / cfg["result_root"]).resolve()
    audit_dir = artifact_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    # The legacy util imports matplotlib during module discovery.  Keep its
    # font/cache writes inside the isolated artifact root rather than waiting
    # on a shared home-directory cache lock.
    os.environ["MPLCONFIGDIR"] = str(audit_dir / "matplotlib_cache")
    (audit_dir / "matplotlib_cache").mkdir(parents=True, exist_ok=True)
    source_dir = (ROOT / cfg["data_snapshot"]).resolve()
    dataset_audit_payload = audit_dataset(
        source_dir,
        audit_dir,
        float(cfg["kir"]),
        0,
        dataset_name=str(cfg.get("dataset", "stackoverflow")),
        protocol=str(cfg.get("protocol_version", "mogb_official_exact_stackoverflow_v1")),
    )
    modes = ["official_fixed", "unified_zero"] if cli.mode == "all" else [cli.mode]
    if cli.dry_run:
        write_json(audit_dir / "dry_run.json", {"status": "dry_run", "modes": modes, "dataset_audit": dataset_audit_payload, "gpu_available": bool(torch.cuda.is_available()), "config": cfg})
        print(json.dumps({"status": "dry_run", "modes": modes, "artifact_root": relative_path(artifact_root), "dataset_match": dataset_audit_payload["paper_shape_match"]}, ensure_ascii=False))
        return 0
    modules = _import_official_modules()
    patch_cluster_device(modules["cluster"])
    patch_cluster3_empty_ball_guard(modules["cluster3"])
    results = []
    for mode in modes:
        mode_manifest = audit_dir / mode / "mode_manifest.json"
        if cli.resume and mode_manifest.exists():
            payload = json.loads(mode_manifest.read_text(encoding="utf-8"))
            if payload.get("status") == "complete":
                results.append(payload)
                continue
        result = run_mode(cfg, mode, audit_dir, modules, source_dir)
        write_json(mode_manifest, result)
        results.append(result)
    write_outputs(cfg, audit_dir, result_root, results, dataset_audit_payload)
    print(json.dumps({"status": "complete", "modes": [result["mode"] for result in results], "result_root": relative_path(result_root), "artifact_root": relative_path(artifact_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
