#!/usr/bin/env python3
"""Run the current cascade ablation with Known-only validation selection.

This is intentionally separate from ``run_trainable_paper_four_variants.py``:
that runner reproduces the earlier paper-shaped artifact whose auxiliary
thresholds were selected with OOS-labelled validation rows.  This runner uses
the locked Known-only Gate configurations and calibrates only the auxiliary
variants from Known validation coverage.

The run is evaluation-only.  It reuses completed Trainable MiniLM checkpoints
and the existing Router/Expert components; it does not write under
``../artifacts`` or export raw predictions/embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.experiments.finalize_historical_known_coverage import _score_from_distances  # noqa: E402
from tools.eval import run_historical_trainable_full_pipeline as full  # noqa: E402
from tools.eval import run_historical_trainable_parameter_full_pipeline as gate  # noqa: E402
from tools.eval.run_historical_trainable_extended_boundary_search import _distance_matrix  # noqa: E402
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector  # noqa: E402
from tools.eval.eval_smollm_cascade_v19 import (  # noqa: E402
    SmolLMCascadeEvaluator,
    SmolLMPrototypeGate,
    _fit_smollm_gate_centers,
)
from tools.eval.eval_minilm_cascade_v19 import (  # noqa: E402
    _predict_router_experts,
    _train_heads,
)
from tools.eval.run_trainable_paper_four_variants import (  # noqa: E402
    compose,
    downstream,
    metrics,
)


DATASETS = ("clinc150", "stackoverflow", "banking77_oos")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87)
VARIANTS = ("Ours", "Frozen MiniLM", "Without Gate", "Cascade-MiniLM", "Cascade-SmolLM")
KNOWN_LOCK = ROOT / "results/analysis/historical_known_coverage90_v3/selection_lock.json"
OUT = ROOT / "results/analysis/trainable_full_pipeline_ablation_known_only"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _views(
    dataset: str,
    kir: float,
    seed: int,
    splits: Sequence[str] = ("train", "val", "test"),
) -> dict[str, list[dict[str, Any]]]:
    root = gate.DATA_ROOT / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}"
    known = {str(value) for value in _read(root / "KNOWN_INTENTS.json")["known_intents"]}
    result: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        rows = []
        for raw in _read(root / "gate" / f"{split}.json"):
            row = dict(raw)
            row["text"] = str(row["text"])
            row["intent"] = str(row["intent"])
            row["label"] = int(row.get("label", 0 if row["intent"] in known else 1))
            rows.append(row)
        if not rows:
            raise ValueError(f"empty {dataset}/{split}/seed{seed}")
        result[split] = rows
    if "train" in result and not all(int(row["label"]) == 0 for row in result["train"]):
        raise AssertionError("Known training split contains OOS rows")
    return result


def _known_lock(dataset: str, kir: float, seed: int) -> tuple[dict[str, Any], Path]:
    locks = _read(KNOWN_LOCK)
    for lock in locks:
        if lock["dataset"] == dataset and float(lock["kir"]) == float(kir):
            index = list(lock["checkpoints"]).index(
                next(path for path in lock["checkpoints"] if f"seed{seed}_" in path)
            )
            config = dict(lock["config"])
            config["threshold"] = float(lock["validation_thresholds"][index])
            return config, Path(lock["checkpoints"][index])
    raise KeyError(f"no Known-only lock for {dataset}/{kir}")


def ours_spec(dataset: str, kir: float, seed: int) -> tuple[dict[str, Any], Path, str]:
    if dataset in {"clinc150", "banking77_oos"}:
        config, checkpoint = _known_lock(dataset, kir, seed)
        return config, checkpoint, "historical_known_coverage90_v3 lock"
    checkpoint = full.DEFAULT_H1_ROOT / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}" / "trainable_k1" / "checkpoint.pt"
    config = {
        "k": 1,
        "distance": "euclidean",
        "boundary": "mean_std_1.5",
        "rule": "normalized_union",
        "fusion_weight": 0.0,
        "threshold": 0.90,
    }
    source = _read(checkpoint.with_name("run_manifest.json"))
    if source.get("selection") != "known_only_validation_for_checkpoint":
        raise AssertionError(f"non-Known-only StackOverflow checkpoint: {checkpoint}")
    return config, checkpoint, "fixed Known-only StackOverflow contract"


def _known_quantile(scores: np.ndarray, rows: Sequence[Mapping[str, Any]], quantile: float = 0.90) -> float:
    values = np.asarray([score for score, row in zip(scores, rows, strict=True) if int(row["label"]) == 0], dtype=np.float64)
    if values.size == 0:
        raise ValueError("Known validation set is empty")
    return float(np.quantile(values, quantile, method="higher"))


def _replay_locked(
    train_values: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    values: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[MultiSphereOOSDetector, dict[str, np.ndarray]]:
    """Reconstruct the exact Known-only search score family, including fusion."""
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
    train_intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    detector.fit(train_values, train_intents)
    train_distances = _distance_matrix(detector, train_values)
    distances = _distance_matrix(detector, values)
    owners = np.asarray([detector.cluster_to_intent[i] for i in range(len(detector.spheres))])
    boundary = str(config["boundary"])
    parameter = float(boundary.rsplit("_", 1)[1])
    if boundary.startswith("mean_std_"):
        detector.radius_lambda = parameter
        detector._compute_radii()
        radius = np.asarray([sphere.radius for sphere in detector.spheres], dtype=np.float64)
    elif boundary.startswith("center_quantile_"):
        assignment = np.argmin(
            np.where(train_intents[:, None] == owners[None, :], train_distances, np.inf),
            axis=1,
        )
        detector.radius_lambda = 3.0
        detector._compute_radii()
        radius = np.asarray([
            np.quantile(train_distances[assignment == index, index], parameter)
            if np.any(assignment == index) else detector.spheres[index].radius
            for index in range(len(owners))
        ])
    elif boundary.startswith("intent_quantile_"):
        radius = np.asarray([
            np.quantile(train_distances[train_intents == name].min(axis=1), parameter)
            for name in owners
        ])
    else:
        raise ValueError(f"unsupported boundary: {boundary}")
    score, ids = _score_from_distances(
        distances,
        detector,
        radius,
        owners,
        str(config["rule"]),
        str(config.get("fusion", "none")),
        float(config.get("fusion_weight", 0.0)),
    )
    return detector, {"score": np.asarray(score), "nearest_cluster": np.asarray(ids)}


def _encode_trainable_splits(
    checkpoint: Path,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    encoder = full._RacalGateEncoder(full.MODEL_ROOT / "all-MiniLM-L6-v2", checkpoint, device)
    try:
        return {
            name: encoder.encode([str(row["text"]) for row in rows], batch_size=256)
            for name, rows in splits.items()
        }
    finally:
        del encoder
        torch.cuda.empty_cache()


def _encode_frozen_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(full.MODEL_ROOT / "all-MiniLM-L6-v2"), device=str(device))
    try:
        return {
            name: np.asarray(
                encoder.encode(
                    [str(row["text"]) for row in rows],
                    batch_size=256,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
            for name, rows in splits.items()
        }
    finally:
        del encoder
        torch.cuda.empty_cache()


def _fit_smollm_gate(
    pipeline: Any,
    train_rows: Sequence[Mapping[str, Any]],
) -> tuple[Any, SmolLMPrototypeGate, bool]:
    if pipeline.router_model is None:
        base = AutoModel.from_pretrained(full.MODEL_ROOT / "smollm135m", local_files_only=True).to(pipeline.device).eval()
        router_model = SimpleNamespace(base=base)
        owns_base = True
    else:
        router_model = pipeline.router_model
        owns_base = False
    embedder = SimpleNamespace(
        router_model=router_model,
        tokenizer=pipeline.tokenizer,
        device=pipeline.device,
        batch_size=64,
        max_length=64,
    )
    train_x = SmolLMCascadeEvaluator.embed_texts(embedder, [row["text"] for row in train_rows])
    centers, intents = _fit_smollm_gate_centers(train_x, train_rows, 1)
    return embedder, SmolLMPrototypeGate(centers, intents, 1.0), owns_base


def _smollm_score(
    embedder: Any,
    prototype: SmolLMPrototypeGate,
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    return np.asarray(
        prototype.predict(
            SmolLMCascadeEvaluator.embed_texts(embedder, [row["text"] for row in rows])
        )["score"],
        dtype=np.float64,
    )


def _release_smollm_gate(embedder: Any, owns_base: bool) -> None:
    if owns_base:
        del embedder.router_model.base
    del embedder
    torch.cuda.empty_cache()


def _recompute_cascade_minilm(
    dataset: str,
    kir: float,
    seed: int,
    out: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Recompute the homogeneous MiniLM cascade without re-running other variants."""
    views = _views(dataset, kir, seed, splits=("train", "val"))
    known_val = [row for row in views["val"] if int(row["label"]) == 0]
    ours_config, checkpoint, _ = ours_spec(dataset, kir, seed)

    frozen_x = _encode_frozen_splits(
        {"train": views["train"], "val": known_val},
        device,
    )
    frozen_train, frozen_val = frozen_x["train"], frozen_x["val"]
    frozen_config = dict(ours_config)
    _, frozen_val_out = _replay_locked(
        frozen_train,
        views["train"],
        frozen_val,
        frozen_config,
    )
    frozen_threshold = _known_quantile(frozen_val_out["score"], known_val)

    mini_router, mini_experts = _train_heads(views["train"], frozen_train, seed)

    # Test is loaded only after the Known-only threshold and classifier heads
    # have been fixed.
    test_rows = _views(dataset, kir, seed, splits=("test",))["test"]
    frozen_test = _encode_frozen_splits({"test": test_rows}, device)["test"]
    _, frozen_test_out = _replay_locked(
        frozen_train,
        views["train"],
        frozen_test,
        frozen_config,
    )
    domains, _, intents, _ = _predict_router_experts(
        mini_router,
        mini_experts,
        frozen_test,
    )
    mini_predictions = [
        dict(domain=domain, intent=intent)
        for domain, intent in zip(domains, intents, strict=True)
    ]
    variant_config = dict(frozen_config)
    variant_config["threshold"] = frozen_threshold
    return {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "variant": "Cascade-MiniLM",
        "selection": "Known validation 90% coverage quantile for frozen MiniLM Gate",
        "checkpoint": "Known-only MiniLM logistic heads",
        "gate_config": variant_config,
        "threshold": frozen_threshold,
        **metrics(
            test_rows,
            compose(mini_predictions, frozen_test_out["score"], frozen_threshold),
        ),
    }


def _rebuild_summary(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(output / "per_seed.csv", rows)
    summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for variant in VARIANTS:
                group = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and float(row["kir"]) == kir
                    and row["variant"] == variant
                ]
                if not group:
                    continue
                out_row = {
                    "dataset": dataset,
                    "kir": kir,
                    "variant": variant,
                    "seed_count": len(group),
                }
                for key in (
                    "known_f1",
                    "oos_f1",
                    "overall_accuracy",
                    "known_recall",
                    "false_accept_rate",
                ):
                    values = np.asarray([float(row[key]) for row in group], dtype=np.float64)
                    out_row[f"{key}_mean"] = float(values.mean())
                    out_row[f"{key}_std"] = float(values.std(ddof=0))
                summary.append(out_row)
    _write_csv(output / "summary.csv", summary)


def _replace_detector_path(detector: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(path)


def _run_cell(dataset: str, kir: float, seed: int, out: Path, device: torch.device) -> list[dict[str, Any]]:
    # Keep test rows out of the process until every validation-only calibration
    # decision for every variant has been made.
    views = _views(dataset, kir, seed, splits=("train", "val"))
    known_val = [row for row in views["val"] if int(row["label"]) == 0]
    ours, checkpoint, source = ours_spec(dataset, kir, seed)
    trainable_x = _encode_trainable_splits(
        checkpoint,
        {"train": views["train"], "val": known_val},
        device,
    )
    train_x, val_x = trainable_x["train"], trainable_x["val"]

    ours_config = dict(ours)
    ours_detector, _ = _replay_locked(train_x, views["train"], val_x, ours_config)
    ours_threshold = float(ours_config["threshold"])

    cell = out / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}"
    detector_path = cell / "ours_detector.json"
    _replace_detector_path(ours_detector, detector_path)
    full.KIR = float(kir)
    full.CASCADE_ROOT = (ROOT.parent / "artifacts/s2c/outputs/experiments/cascade_full" / f"gpu_kir{int(round(kir * 100)):02d}").resolve()
    pipeline = full._make_pipeline(dataset, seed, device, detector_path, full.DEFAULT_H1_ROOT, "trainable_k1")
    smol_embedder = None
    smol_owns_base = False
    try:
        # Complete every calibration decision before loading or encoding test.
        val_downstream = downstream(pipeline, known_val)

        # Frozen MiniLM uses the same locked geometry and the same fixed downstream.
        frozen_x = _encode_frozen_splits(
            {"train": views["train"], "val": known_val},
            device,
        )
        frozen_train = frozen_x["train"]
        frozen_val = frozen_x["val"]
        frozen_config = dict(ours_config)
        frozen_detector, frozen_val_out = _replay_locked(frozen_train, views["train"], frozen_val, frozen_config)
        frozen_threshold = _known_quantile(frozen_val_out["score"], known_val)
        frozen_variant_config = dict(frozen_config)
        frozen_variant_config["threshold"] = frozen_threshold

        # Without Gate: confidence rejection is calibrated on Known validation only.
        confidence_val = np.asarray([1.0 - float(pred["intent_prob"]) for pred in val_downstream], dtype=np.float64)
        without_threshold = _known_quantile(confidence_val, known_val)

        # Cascade-MiniLM is the homogeneous MiniLM cascade: its Gate and its
        # downstream heads both use the frozen MiniLM representation. Reusing
        # the Trainable Gate here would make the OOS decision identical to
        # Ours by construction and would no longer be the historical
        # Cascade-MiniLM variant.
        mini_router, mini_experts = _train_heads(views["train"], frozen_train, seed)

        # Cascade-SmolLM uses a SmolLM prototype Gate and Known-only threshold calibration.
        smol_embedder, smol_prototype, smol_owns_base = _fit_smollm_gate(pipeline, views["train"])
        smol_val_score = _smollm_score(smol_embedder, smol_prototype, known_val)
        smol_threshold = _known_quantile(smol_val_score, known_val)

        # Test is loaded only after the Known-only calibration phase above.
        views["test"] = _views(dataset, kir, seed, splits=("test",))["test"]
        test_rows = views["test"]
        test_trainable = _encode_trainable_splits(checkpoint, {"test": test_rows}, device)["test"]
        _, ours_test_out = _replay_locked(train_x, views["train"], test_trainable, ours_config)
        frozen_test_x = _encode_frozen_splits({"test": test_rows}, device)["test"]
        _, frozen_test_out = _replay_locked(frozen_train, views["train"], frozen_test_x, frozen_config)
        test_downstream = downstream(pipeline, test_rows)
        confidence_test = np.asarray([1.0 - float(pred["intent_prob"]) for pred in test_downstream], dtype=np.float64)
        domains, _, intents, _ = _predict_router_experts(mini_router, mini_experts, frozen_test_x)
        mini = [dict(domain=domain, intent=intent) for domain, intent in zip(domains, intents, strict=True)]
        smol_test_score = _smollm_score(smol_embedder, smol_prototype, test_rows)

        result: list[dict[str, Any]] = []
        result.append(dict(
            dataset=dataset,
            kir=kir,
            seed=seed,
            variant="Ours",
            selection="locked Known-only Gate config",
            checkpoint=str(checkpoint),
            gate_config=ours_config,
            threshold=ours_threshold,
            **metrics(test_rows, compose(test_downstream, ours_test_out["score"], ours_threshold)),
        ))
        result.append(dict(
            dataset=dataset,
            kir=kir,
            seed=seed,
            variant="Frozen MiniLM",
            selection="Known validation 90% coverage quantile",
            checkpoint="pretrained all-MiniLM-L6-v2",
            gate_config=frozen_variant_config,
            threshold=frozen_threshold,
            **metrics(test_rows, compose(test_downstream, frozen_test_out["score"], frozen_threshold)),
        ))
        result.append(dict(
            dataset=dataset,
            kir=kir,
            seed=seed,
            variant="Without Gate",
            selection="Known validation 90% coverage quantile",
            checkpoint="fixed Router/Expert",
            gate_config="intent confidence rejection",
            threshold=without_threshold,
            **metrics(test_rows, compose(test_downstream, confidence_test, without_threshold)),
        ))
        result.append(dict(
            dataset=dataset,
            kir=kir,
            seed=seed,
            variant="Cascade-MiniLM",
            selection="Known validation 90% coverage quantile for frozen MiniLM Gate",
            checkpoint="Known-only MiniLM logistic heads",
            gate_config=frozen_variant_config,
            threshold=frozen_threshold,
            **metrics(test_rows, compose(mini, frozen_test_out["score"], frozen_threshold)),
        ))
        result.append(dict(
            dataset=dataset,
            kir=kir,
            seed=seed,
            variant="Cascade-SmolLM",
            selection="Known validation 90% coverage quantile",
            checkpoint="pretrained/fixed SmolLM prototype Gate",
            gate_config="SmolLM prototype cosine support",
            threshold=smol_threshold,
            **metrics(test_rows, compose(test_downstream, smol_test_score, smol_threshold)),
        ))
        _write(cell / "config.json", {
            "dataset": dataset,
            "kir": kir,
            "seed": seed,
            "ours_source": source,
            "ours_checkpoint": str(checkpoint),
            "ours_config": ours_config,
            "selection_contract": "Known-only validation; 90% Known coverage for auxiliary thresholds",
            "real_oos_used_for_training": False,
            "real_oos_used_for_selection": False,
            "pseudo_oos_used": False,
            "test_used_for_selection": False,
            "variant_names": VARIANTS,
        })
        _write(cell / "results.json", result)
        return result
    finally:
        if smol_embedder is not None:
            _release_smollm_gate(smol_embedder, smol_owns_base)
        del pipeline
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--seed", choices=SEEDS, type=int, action="append")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true", help="Reuse completed cells with matching locked configurations")
    parser.add_argument(
        "--reconcile-cascade-minilm",
        action="store_true",
        help="Recompute the homogeneous Frozen-MiniLM Cascade-MiniLM variant in an existing run",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    datasets = tuple(args.dataset or DATASETS)
    seeds = tuple(args.seed or SEEDS)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this ablation")
    torch.set_num_threads(4)
    torch.zeros(1, device=device)
    if args.reconcile_cascade_minilm:
        if not (output / "MANIFEST.json").exists():
            raise SystemExit(f"Cannot reconcile missing run: {output}")
        rows: list[dict[str, Any]] = []
        for dataset in DATASETS:
            for kir in KIRS:
                for seed in SEEDS:
                    cell = output / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}"
                    result_path = cell / "results.json"
                    if not result_path.exists():
                        raise SystemExit(f"Cannot reconcile incomplete cell: {cell}")
                    cell_rows = _read(result_path)
                    replacement = _recompute_cascade_minilm(dataset, kir, seed, output, device)
                    cell_rows = [
                        replacement if row["variant"] == "Cascade-MiniLM" else row
                        for row in cell_rows
                    ]
                    _write(result_path, cell_rows)
                    rows.extend(cell_rows)
        _rebuild_summary(output, rows)
        manifest = _read(output / "MANIFEST.json")
        manifest["status"] = "complete"
        manifest["completed_units"] = len(rows)
        manifest["cascade_minilm_reconciled"] = True
        manifest["cascade_minilm_contract"] = (
            "Frozen MiniLM Gate and Known-only MiniLM logistic Router/Experts"
        )
        _write(output / "MANIFEST.json", manifest)
        print(json.dumps({"output_root": str(output), "completed_units": len(rows), "device": "cuda", "reconciled_variant": "Cascade-MiniLM"}, ensure_ascii=False), flush=True)
        return
    manifest = {
        "status": "running",
        "protocol": "historical_v19_paper_main__H1_current_full_pipeline_known_only_ablation",
        "datasets": list(datasets),
        "kirs": list(KIRS),
        "seeds": list(seeds),
        "variants": list(VARIANTS),
        "device": "cuda",
        "validation": "Known rows only; auxiliary thresholds use 90% Known coverage quantile",
        "real_oos_used_for_training": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_used_for_selection": False,
        "raw_predictions_written": False,
        "expected_units": len(datasets) * len(KIRS) * len(seeds) * len(VARIANTS),
    }
    _write(output / "MANIFEST.json", manifest)
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for kir in KIRS:
            for seed in seeds:
                cell = output / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}"
                if args.resume and (cell / "results.json").exists():
                    saved = _read(cell / "config.json")
                    config, checkpoint, _ = ours_spec(dataset, kir, seed)
                    if saved["ours_config"] != config or saved["ours_checkpoint"] != str(checkpoint):
                        raise ValueError(f"Stored cell differs from current selection lock: {cell}")
                    cell_rows = _read(cell / "results.json")
                    if {row["variant"] for row in cell_rows} != set(VARIANTS) or len(cell_rows) != len(VARIANTS):
                        raise ValueError(f"Incomplete stored cell: {cell}")
                else:
                    cell_rows = _run_cell(dataset, kir, seed, output, device)
                rows.extend(cell_rows)
                print(json.dumps(cell_rows, ensure_ascii=False), flush=True)
    _write_csv(output / "per_seed.csv", rows)
    _rebuild_summary(output, rows)
    manifest.update(status="complete", completed_units=len(rows))
    _write(output / "MANIFEST.json", manifest)
    print(json.dumps({"output_root": str(output), "completed_units": len(rows), "device": "cuda"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
