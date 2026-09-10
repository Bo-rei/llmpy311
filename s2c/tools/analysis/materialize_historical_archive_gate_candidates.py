#!/usr/bin/env python3
"""Materialize validation-selected archive Gate detectors for pipeline replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from protocol_v2.experiments.racal_v1.boundary import detector_signature
from protocol_v2.experiments.racal_v1.representation import choose_device
from tools.analysis import search_historical_archive_gate_workpoints as search


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_archive_gate_search_seed42"
DEFAULT_OUTPUT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "historical_archive_gate_candidates_seed42"
DATASETS = ("clinc150", "stackoverflow", "banking77")
METHODS = ("frozen_k1", "partial_k1", "lora_k1")


def _selected(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one selected row: {path}")
    return rows[0]


def materialize(
    dataset: str,
    method: str,
    search_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, object]:
    selected_path = search_root / dataset / method / "selected_validation.csv"
    selected = _selected(selected_path)
    _, views, values = search._load_embeddings(dataset, method, device)
    detector = search._fit_detector(
        values["train"],
        views["train"],
        int(selected["k"]),
        float(selected["radius_lambda"]),
        selected["acceptance_mode"],
    )
    threshold = float(selected["threshold"])
    if threshold <= 0:
        raise ValueError(f"invalid selected threshold: {threshold}")
    for sphere in detector.spheres:
        sphere.radius = float(sphere.radius) * threshold
    signature = detector_signature(detector)
    signature.update(
        {
            "source_selected_validation": str(selected_path.resolve()),
            "effective_score_threshold": threshold,
            "radii_scaled_for_pipeline_threshold": True,
        }
    )
    candidate_root = output_root / dataset / method
    candidate_root.mkdir(parents=True, exist_ok=False)
    (candidate_root / "detector_signature.json").write_text(
        json.dumps(signature, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "s2c.historical_archive_gate_candidate.v1",
        "protocol": "historical_v19_archive_development_tuning",
        "dataset": dataset,
        "method": method,
        "seed": 42,
        "kir": 0.50,
        "device": str(device),
        "selected_validation": selected,
        "checkpoint_run": str(search._checkpoint_path(dataset, method).resolve()),
        "detector_signature": str((candidate_root / "detector_signature.json").resolve()),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "selection_uses_validation_oos_labels": True,
    }
    (candidate_root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=search.DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=search.MODEL_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    search.DATA_ROOT = args.data_root.resolve()
    search.MODEL_ROOT = args.model_root.resolve()
    device = choose_device(args.device)
    datasets = tuple(args.dataset or DATASETS)
    methods = tuple(args.method or METHODS)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifests = [
        materialize(dataset, method, args.search_root.resolve(), output_root, device)
        for dataset in datasets
        for method in methods
    ]
    summary = {
        "schema_version": "s2c.historical_archive_gate_candidate.summary.v1",
        "datasets": list(datasets),
        "methods": list(methods),
        "device": str(device),
        "completed_units": len(manifests),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }
    (output_root / "MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
