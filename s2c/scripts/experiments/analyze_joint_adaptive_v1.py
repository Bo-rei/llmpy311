#!/usr/bin/env python3
"""Evaluate the trained K=1 checkpoint and post-hoc K=2 controls.

This is analysis only: it never changes the adaptive checkpoints and never
uses test labels for training or structure selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from protocol_v2.experiments.joint_adaptive_v1.runner import (  # noqa: E402
    CenterState,
    JointPrototypeModel,
    _model_path,
    evaluate_boundary,
    fit_boundary,
    _load_config,
    _load_views,
    _root,
    _normalize,
)
from protocol_v2.experiments.racal_v1.representation import choose_device, encode_rows  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_checkpoint(model_path: Path, checkpoint: Path, projection_hidden_dim: int, device: torch.device) -> tuple[JointPrototypeModel, CenterState]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    centers = np.asarray(payload["center_state"], dtype=np.float32)
    center_intents = tuple(str(x) for x in payload["center_intents"])
    parent_centers = np.asarray(payload["parent_centers"], dtype=np.float32)
    parent_intents = tuple(str(x) for x in payload["parent_intents"])
    state = CenterState(centers, center_intents, parent_centers, parent_intents, ())
    model = JointPrototypeModel(model_path, state, projection_hidden_dim).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, state


def _fixed_k2_state(train_values: np.ndarray, rows: list[dict[str, object]], parent: CenterState) -> CenterState:
    values = _normalize(train_values)
    centers: list[np.ndarray] = []
    intents: list[str] = []
    for name in parent.parent_intents:
        indices = np.asarray([i for i, row in enumerate(rows) if str(row["intent"]) == name], dtype=np.int64)
        fitted = KMeans(n_clusters=2, n_init=10, random_state=42).fit(values[indices])
        centers.extend(_normalize(fitted.cluster_centers_))
        intents.extend([name, name])
    return CenterState(np.asarray(centers, dtype=np.float32), tuple(intents), parent.parent_centers, parent.parent_intents, ())


def _no_parent_guard(boundary):
    # A very large parent radius makes the boundary equivalent to a union of
    # child spheres while retaining the same BoundaryState schema.
    return type(boundary)(boundary.centers, boundary.center_intents, boundary.radii, boundary.inv_diag_cov, boundary.parent_centers, boundary.parent_intents, np.full_like(boundary.parent_radii, 1.0e12), np.zeros_like(boundary.parent_inv_diag_cov))


def run(attempt: str, config_path: Path) -> dict[str, object]:
    import os

    os.environ["JOINT_ADAPTIVE_ATTEMPT"] = attempt
    from protocol_v2.runtime.paths import ProtocolV2Paths

    paths = ProtocolV2Paths.discover()
    config = _load_config(config_path)
    model_path = _model_path(paths, config)
    device = choose_device(str(config.get("device", "auto")))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rows: list[dict[str, object]] = []
    root = _root(paths)
    for seed in (13, 42, 87):
        views = _load_views(paths, seed)
        checkpoint = root / "runs" / f"seed_{seed}" / "joint_k1.pt"
        model, parent = _load_checkpoint(model_path, checkpoint, int(config["projection_hidden_dim"]), device)
        train_values = encode_rows(model.encoder, tokenizer, views.train, device, int(config["batch_size"]), int(config["max_length"]))
        test_values = encode_rows(model.encoder, tokenizer, views.test, device, int(config["batch_size"]), int(config["max_length"]))
        for method, state, guard in (
            ("joint_k1", parent, True),
            ("joint_fixed_k2_posthoc_union", _fixed_k2_state(train_values, views.train, parent), False),
            ("joint_fixed_k2_posthoc_parent_guard", _fixed_k2_state(train_values, views.train, parent), True),
        ):
            boundary = fit_boundary(train_embeddings=train_values, calibration_embeddings=train_values, rows=views.train, state=state, radius_lambda=float(config["radius_lambda"]))
            if not guard:
                boundary = _no_parent_guard(boundary)
            metrics, _ = evaluate_boundary(boundary, test_values, views.test, float(config["threshold"]))
            rows.append({"dataset": "stackoverflow", "kir": 0.50, "seed": seed, "method": method, "parent_guard": guard, "k_total": len(state.centers), **metrics})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = root / "analysis"
    _write_csv(out / "JOINT_ADAPTIVE_BASELINES.csv", rows)
    by_seed = {(int(row["seed"]), str(row["method"])): row for row in rows}
    paired = []
    for seed in (13, 42, 87):
        adaptive = json.loads((root / "runs" / f"seed_{seed}" / "run_manifest.json").read_text(encoding="utf-8"))["metrics"]
        k1 = by_seed[(seed, "joint_k1")]
        k2 = by_seed[(seed, "joint_fixed_k2_posthoc_union")]
        paired.append({"seed": seed, "adaptive_oos_f1": float(adaptive["oos_f1"]), "joint_k1_oos_f1": float(k1["oos_f1"]), "fixed_k2_union_oos_f1": float(k2["oos_f1"]), "adaptive_minus_k1_oos_f1": float(adaptive["oos_f1"]) - float(k1["oos_f1"]), "adaptive_minus_fixed_k2_oos_f1": float(adaptive["oos_f1"]) - float(k2["oos_f1"]), "adaptive_false_accept": float(adaptive["false_accept_rate"]), "k1_false_accept": float(k1["false_accept_rate"]), "fixed_k2_union_false_accept": float(k2["false_accept_rate"])})
    _write_csv(out / "JOINT_ADAPTIVE_PAIRED.csv", paired)
    return {"status": "complete", "rows": len(rows), "output": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="repair3")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.attempt, args.config.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
