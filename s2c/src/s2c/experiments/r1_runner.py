"""Runner for the bounded R1 geometry-preserving CE-Recon pilot."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from s2c.data.hashing import atomic_write_json, atomic_write_text, sha256_file, sha256_json
from s2c.data.manifests import dataset_manifest_path, read_json
from s2c.experiments.geometry_preserving import (
    BETAS,
    DEFAULT_MODEL,
    DISTANCES,
    K_VALUES,
    REPRESENTATIONS,
    _encode,
    _safe,
    evaluate_gate,
    fixed_oos_buckets,
    geometry_metrics,
    git_patch_hash,
    load_checkpoint,
    load_bundle,
    train_representation,
    write_csv,
)
from s2c.experiments.mechanism_runner import E3Bundle
from s2c.runtime.paths import ProtocolV2Paths


R1_NAME = "r1_geometry_preserving_representation"
DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87)
KIR = 0.50


def r1_root(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / R1_NAME


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"R1 config must be a mapping: {path}")
    return payload


def _model_path(paths: ProtocolV2Paths, config: Mapping[str, Any]) -> Path:
    configured = Path(str(config["model_path"]))
    candidate = configured if configured.is_absolute() else (paths.project_root / configured).resolve()
    if not candidate.is_dir():
        candidate = (paths.project_root.parent / configured).resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"R1 local MiniLM model is unavailable: {candidate}")
    return candidate


def _e2_closeout(paths: ProtocolV2Paths) -> Path:
    return paths.run_root / "summaries" / "e2_closeout" / "E2_closeout_manifest.json"


def _registry_tree_hash(paths: ProtocolV2Paths) -> str:
    rows = []
    for path in sorted(paths.registry_root.glob("*/seed_*/kir_*.json")):
        rows.append((str(path.relative_to(paths.registry_root)), sha256_file(path)))
    return sha256_json(rows)


def freeze_provenance(paths: ProtocolV2Paths, config_path: Path) -> dict[str, Any]:
    root = r1_root(paths)
    root.mkdir(parents=True, exist_ok=True)
    patch_hash, patch_path = git_patch_hash(paths.project_root)
    config_hash = sha256_file(config_path)
    snapshot = {
        "schema_version": "s2c.r1_provenance.v1",
        "stage": "R1",
        "experiment": R1_NAME,
        "protocol_version": paths.dataset_version,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.project_root.parent, capture_output=True, text=True, check=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=paths.project_root.parent, capture_output=True, text=True, check=True).stdout.strip()),
        "code_patch_sha256": patch_hash,
        "code_patch": str(patch_path),
        "config_sha256": {str(config_path): config_hash},
        "e2_closeout_sha256": sha256_file(_e2_closeout(paths)),
        "canonical_manifest_sha256": {
            dataset: sha256_file(dataset_manifest_path(paths.manifest_root, dataset)) for dataset in DATASETS
        },
        "registry_tree_sha256": _registry_tree_hash(paths),
        "encoder": DEFAULT_MODEL,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "bootstrap_or_selection_seed": 20260725,
        "oos_used_for_selection": False,
    }
    atomic_write_json(root / "R1_PROVENANCE_SNAPSHOT.json", snapshot)
    return snapshot


def require_provenance(paths: ProtocolV2Paths) -> dict[str, Any]:
    path = r1_root(paths) / "R1_PROVENANCE_SNAPSHOT.json"
    if not path.is_file():
        raise RuntimeError("R1 provenance is not frozen; run the freeze command first")
    return read_json(path)


def _bundle_key(dataset: str, seed: int) -> str:
    return f"{dataset}/kir_0.50/seed_{seed}"


def _model_dir(paths: ProtocolV2Paths, dataset: str, seed: int, representation: str) -> Path:
    return r1_root(paths) / "checkpoints" / dataset / "kir_0.50" / f"seed_{seed}" / representation


def _beta_dir(paths: ProtocolV2Paths, dataset: str, beta: float) -> Path:
    return r1_root(paths) / "checkpoints" / "beta_selection" / dataset / f"beta_{beta:g}"


def _encode_representation(
    model_path: Path,
    checkpoint: Path,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    batch_size: int,
    max_length: int,
) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, _ = load_checkpoint(model_path, checkpoint, device)
    result = {
        split: _encode(model, tokenizer, split_rows, device, batch_size, max_length)
        for split, split_rows in rows.items()
    }
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _known_calibration_false_rejection(train: np.ndarray, calibration: np.ndarray, train_rows: Sequence[Mapping[str, Any]], threshold: float = 1.0) -> float:
    from s2c.experiments.geometry_preserving import fit_gate

    detector = fit_gate(train, train_rows, 1, "euclidean")
    scores = np.asarray(detector.predict_with_scores(calibration)["score"], dtype=float)
    return float(np.mean(scores > threshold))


def _teacher_geometry_row(bundle: Any, student: np.ndarray | None = None, calibration: np.ndarray | None = None, test: np.ndarray | None = None, seed: int = 42) -> dict[str, Any]:
    if student is None:
        student = bundle.train
    if calibration is None:
        calibration = bundle.calibration
    if test is None:
        test = bundle.test
    return geometry_metrics(
        bundle.train,
        bundle.calibration,
        bundle.test,
        student,
        calibration,
        test,
        bundle.views.train,
        bundle.views.calibration,
        bundle.views.test,
        seed,
    )


def _candidate_rows(paths: ProtocolV2Paths, config: dict[str, Any]) -> list[dict[str, Any]]:
    model_path = _model_path(paths, config)
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        bundle = load_bundle(paths, dataset, 42, KIR)
        for beta in BETAS:
            output = _beta_dir(paths, dataset, beta)
            manifest = train_representation(
                model_path=model_path,
                train_rows=bundle.views.train,
                calibration_rows=bundle.views.calibration,
                teacher_train=bundle.train,
                output_dir=output,
                method="ce_recon_geometry",
                seed=42,
                beta=beta,
                alpha=float(config["alpha"]),
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["learning_rate"]),
                max_length=int(config["max_length"]),
            )
            encoded = _encode_representation(
                model_path,
                Path(manifest["checkpoint"]),
                {"train": bundle.views.train, "calibration": bundle.views.calibration, "test": bundle.views.test},
                int(config["batch_size"]),
                int(config["max_length"]),
            )
            geometry = _teacher_geometry_row(bundle, encoded["train"], encoded["calibration"], encoded["test"], 42)
            coverage = _known_calibration_false_rejection(encoded["train"], encoded["calibration"], bundle.views.train)
            rows.append({
                "dataset": dataset,
                "seed": 42,
                "beta": beta,
                "known_validation_macro_f1": manifest["known_validation_macro_f1"],
                "known_calibration_false_rejection": coverage,
                "effective_rank": geometry["effective_rank"],
                "teacher_effective_rank": geometry["teacher_effective_rank"],
                "pairwise_distance_correlation": geometry["pairwise_distance_correlation"],
                "knn_neighborhood_preservation": geometry["knn_neighborhood_preservation"],
                "checkpoint": manifest["checkpoint"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "used_oos_for_selection": False,
            })
    return rows


def select_beta(paths: ProtocolV2Paths, config: dict[str, Any]) -> dict[str, Any]:
    root = r1_root(paths) / "summaries"
    selection_path = root / "R1_beta_selection.json"
    table_path = root / "R1_beta_selection.csv"
    if selection_path.is_file() and table_path.is_file():
        return read_json(selection_path)
    rows = _candidate_rows(paths, config)
    for field in ("known_validation_macro_f1", "known_calibration_false_rejection", "effective_rank", "pairwise_distance_correlation", "knn_neighborhood_preservation"):
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        mean, std = float(values.mean()), float(values.std())
        for row in rows:
            row[f"z_{field}"] = (float(row[field]) - mean) / std if std > 1e-12 else 0.0
    for row in rows:
        rank_collapse = max(0.0, (float(row["teacher_effective_rank"]) - float(row["effective_rank"])) / max(float(row["teacher_effective_rank"]), 1e-12))
        distortion = 1.0 - float(row["knn_neighborhood_preservation"])
        row["rank_collapse"] = rank_collapse
        row["neighborhood_distortion"] = distortion
    for field in ("rank_collapse", "neighborhood_distortion"):
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        mean, std = float(values.mean()), float(values.std())
        for row in rows:
            row[f"z_{field}"] = (float(row[field]) - mean) / std if std > 1e-12 else 0.0
    for row in rows:
        row["known_only_objective"] = (
            row["z_known_validation_macro_f1"]
            - row["z_known_calibration_false_rejection"]
            - row["z_rank_collapse"]
            - row["z_neighborhood_distortion"]
        )
    write_csv(table_path, rows)
    grouped = {}
    for beta in BETAS:
        selected = [row["known_only_objective"] for row in rows if float(row["beta"]) == float(beta)]
        grouped[str(beta)] = float(np.mean(selected))
    selected_beta = max(BETAS, key=lambda beta: (grouped[str(beta)], -beta))
    result = {
        "schema_version": "s2c.r1_beta_selection.v1",
        "selected_beta": selected_beta,
        "candidate_betas": list(BETAS),
        "objective": "z(known_validation_macro_f1)-z(known_calibration_false_rejection)-z(rank_collapse)-z(neighborhood_distortion)",
        "dataset_seed": "all datasets, seed=42",
        "scores_by_beta": grouped,
        "used_oos_for_selection": False,
        "test_used_for_selection": False,
        "table": str(table_path),
    }
    atomic_write_json(selection_path, result)
    return result


def _train_formal_models(paths: ProtocolV2Paths, config: dict[str, Any], selected_beta: float) -> list[dict[str, Any]]:
    model_path = _model_path(paths, config)
    manifests: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            bundle = load_bundle(paths, dataset, seed, KIR)
            ce_dir = _model_dir(paths, dataset, seed, "ce_recon")
            manifests.append(train_representation(
                model_path=model_path,
                train_rows=bundle.views.train,
                calibration_rows=bundle.views.calibration,
                teacher_train=bundle.train,
                output_dir=ce_dir,
                method="ce_recon",
                seed=seed,
                beta=0.0,
                alpha=float(config["alpha"]),
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["learning_rate"]),
                max_length=int(config["max_length"]),
            ) | {"dataset": dataset, "representation": "ce_recon", "r1_seed": seed})
            if seed == 42:
                geometry_dir = _beta_dir(paths, dataset, selected_beta)
                reference = read_json(geometry_dir / "training_manifest.json")
                manifests.append(reference | {"dataset": dataset, "representation": "ce_recon_geometry", "r1_seed": seed, "phase": "beta_selection_reference"})
            else:
                geometry_dir = _model_dir(paths, dataset, seed, "ce_recon_geometry")
                manifests.append(train_representation(
                    model_path=model_path,
                    train_rows=bundle.views.train,
                    calibration_rows=bundle.views.calibration,
                    teacher_train=bundle.train,
                    output_dir=geometry_dir,
                    method="ce_recon_geometry",
                    seed=seed,
                    beta=selected_beta,
                    alpha=float(config["alpha"]),
                    epochs=int(config["epochs"]),
                    batch_size=int(config["batch_size"]),
                    learning_rate=float(config["learning_rate"]),
                    max_length=int(config["max_length"]),
                ) | {"dataset": dataset, "representation": "ce_recon_geometry", "r1_seed": seed})
    return manifests


def _gate_row(
    paths: ProtocolV2Paths,
    bundle: E3Bundle,
    representation: str,
    seed: int,
    embeddings: dict[str, np.ndarray],
    buckets: np.ndarray,
    bucket_info: dict[str, float],
    k: int,
    distance: str,
    geometry: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    metrics = evaluate_gate(
        embeddings["train"],
        embeddings["calibration"],
        embeddings["test"],
        bundle.views.train,
        bundle.views.calibration,
        bundle.views.test,
        k,
        distance,
        buckets,
    )
    elapsed = time.perf_counter() - started
    metrics.update({
        "protocol_version": paths.dataset_version,
        "stage": "R1",
        "experiment_id": R1_NAME,
        "dataset": bundle.dataset,
        "kir": bundle.kir,
        "seed": seed,
        "representation": representation,
        "k": k,
        "distance": distance,
        "boundary": "mean_std",
        "threshold": 1.0,
        "status": status,
        "bucket_q20": bucket_info.get("q20"),
        "bucket_q80": bucket_info.get("q80"),
        "bucket_definition": bucket_info.get("source"),
        "representation_collision_rate": geometry.get("representation_collision_rate"),
        "effective_rank": geometry.get("effective_rank"),
        "teacher_effective_rank": geometry.get("teacher_effective_rank"),
        "knn_neighborhood_preservation": geometry.get("knn_neighborhood_preservation"),
        "pairwise_distance_correlation": geometry.get("pairwise_distance_correlation"),
        "scoring_seconds": elapsed,
        "samples_per_second": len(bundle.views.test) / max(elapsed, 1e-12),
    })
    return metrics


def run_pilot(paths: ProtocolV2Paths, config: dict[str, Any]) -> dict[str, Any]:
    require_provenance(paths)
    root = r1_root(paths)
    selected = select_beta(paths, config)
    selected_beta = float(selected["selected_beta"])
    manifests = _train_formal_models(paths, config, selected_beta)
    training_rows = [{**manifest, "stage": "R1"} for manifest in manifests]
    gate_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    model_path = _model_path(paths, config)
    for dataset in DATASETS:
        for seed in SEEDS:
            try:
                bundle = load_bundle(paths, dataset, seed, KIR)
                buckets, bucket_info = fixed_oos_buckets(bundle.train, bundle.test, bundle.views.test, [str(row["intent"]) for row in bundle.views.train])
                reps: dict[str, dict[str, np.ndarray]] = {"frozen_minilm": {"train": bundle.train, "calibration": bundle.calibration, "test": bundle.test}}
                for representation in ("ce_recon", "ce_recon_geometry"):
                    if representation == "ce_recon":
                        checkpoint = _model_dir(paths, dataset, seed, representation) / "encoder.pt"
                    elif seed == 42:
                        checkpoint = _beta_dir(paths, dataset, selected_beta) / "encoder.pt"
                    else:
                        checkpoint = _model_dir(paths, dataset, seed, representation) / "encoder.pt"
                    reps[representation] = _encode_representation(
                        model_path,
                        checkpoint,
                        {"train": bundle.views.train, "calibration": bundle.views.calibration, "test": bundle.views.test},
                        int(config["batch_size"]),
                        int(config["max_length"]),
                    )
                for representation, embeddings in reps.items():
                    geometry = geometry_metrics(
                        bundle.train,
                        bundle.calibration,
                        bundle.test,
                        embeddings["train"],
                        embeddings["calibration"],
                        embeddings["test"],
                        bundle.views.train,
                        bundle.views.calibration,
                        bundle.views.test,
                        seed,
                    )
                    geometry_rows.append({
                        "protocol_version": paths.dataset_version,
                        "stage": "R1",
                        "dataset": dataset,
                        "kir": KIR,
                        "seed": seed,
                        "representation": representation,
                        **geometry,
                    })
                    status = "reference_e2" if representation == "frozen_minilm" else "new_method_evidence"
                    for k in K_VALUES:
                        for distance in DISTANCES:
                            gate_rows.append(_gate_row(paths, bundle, representation, seed, embeddings, buckets, bucket_info, k, distance, geometry, status))
            except Exception as exc:
                failures.append({"dataset": dataset, "seed": seed, "error_type": type(exc).__name__, "error": str(exc)})
    summary_root = root / "summaries"
    write_csv(summary_root / "R1_training_summary.csv", training_rows)
    write_csv(summary_root / "R1_gate_summary.csv", gate_rows)
    write_csv(summary_root / "R1_geometry_analysis.csv", geometry_rows)
    write_csv(summary_root / "R1_effective_rank.csv", [{k: row.get(k) for k in ("dataset", "seed", "representation", "effective_rank", "teacher_effective_rank", "pairwise_distance_correlation", "knn_neighborhood_preservation")} for row in geometry_rows])
    write_csv(summary_root / "R1_near_oos_analysis.csv", [{k: row.get(k) for k in ("dataset", "seed", "representation", "k", "distance", "near_oos_f1", "medium_oos_f1", "far_oos_f1", "near_oos_recall", "near_false_accept_rate")} for row in gate_rows])
    k12: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for representation in REPRESENTATIONS:
                for distance in DISTANCES:
                    one = next((row for row in gate_rows if row["dataset"] == dataset and row["seed"] == seed and row["representation"] == representation and row["distance"] == distance and row["k"] == 1), None)
                    two = next((row for row in gate_rows if row["dataset"] == dataset and row["seed"] == seed and row["representation"] == representation and row["distance"] == distance and row["k"] == 2), None)
                    if one and two:
                        for metric in ("oos_f1", "near_oos_f1", "id_recall", "false_accept_rate", "false_reject_rate"):
                            k12.append({"dataset": dataset, "seed": seed, "representation": representation, "distance": distance, "metric": metric, "k1": one.get(metric), "k2": two.get(metric), "k2_minus_k1": float(two.get(metric, math.nan)) - float(one.get(metric, math.nan))})
    write_csv(summary_root / "R1_k1_k2_comparison.csv", k12)
    write_csv(summary_root / "R1_failed_or_invalid_runs.csv", failures)
    success = summarize_method_decision(summary_root, selected_beta, len(gate_rows), len(failures))
    atomic_write_json(summary_root / "R1_integrity.json", {"planned_gate_units": 108, "completed_gate_units": len(gate_rows), "failed_groups": len(failures), "selected_beta": selected_beta, "e4_to_e7_started": False})
    return success


def _mean_delta(rows: list[dict[str, Any]], representation: str, metric: str, k: int = 1) -> dict[str, float]:
    output: dict[str, float] = {}
    for dataset in DATASETS:
        current = [float(row[metric]) for row in rows if row["dataset"] == dataset and row["representation"] == representation and int(row["k"]) == k and math.isfinite(float(row.get(metric, math.nan)))]
        baseline = [float(row[metric]) for row in rows if row["dataset"] == dataset and row["representation"] == "ce_recon" and int(row["k"]) == k and math.isfinite(float(row.get(metric, math.nan)))]
        output[dataset] = float(np.mean(current) - np.mean(baseline)) if current and baseline else math.nan
    return output


def summarize_method_decision(summary_root: Path, selected_beta: float, completed: int, failed: int) -> dict[str, Any]:
    import pandas as pd

    gate_path = summary_root / "R1_gate_summary.csv"
    geometry_path = summary_root / "R1_geometry_analysis.csv"
    gate = pd.read_csv(gate_path) if gate_path.is_file() else pd.DataFrame()
    geometry = pd.read_csv(geometry_path) if geometry_path.is_file() else pd.DataFrame()
    decision = "insufficient_evidence"
    reasons: list[str] = []
    if completed == 108 and failed == 0 and not gate.empty:
        k1 = gate[gate["k"] == 1]
        for metric in ("oos_f1", "near_oos_f1"):
            if metric in k1:
                deltas = _mean_delta(k1.to_dict("records"), "ce_recon_geometry", metric, 1)
                reasons.append(f"{metric} delta vs CE-Recon: " + ", ".join(f"{d}={v:+.4f}" for d, v in deltas.items()))
        id_delta = _mean_delta(k1.to_dict("records"), "ce_recon_geometry", "id_recall", 1)
        reasons.append("ID Recall delta vs CE-Recon: " + ", ".join(f"{d}={v:+.4f}" for d, v in id_delta.items()))
        geo = geometry[geometry["representation"].isin(["ce_recon", "ce_recon_geometry"])].groupby("representation").mean(numeric_only=True)
        if {"ce_recon", "ce_recon_geometry"}.issubset(geo.index):
            reasons.append(f"effective-rank delta={geo.loc['ce_recon_geometry','effective_rank'] - geo.loc['ce_recon','effective_rank']:+.4f}")
            reasons.append(f"neighborhood-preservation delta={geo.loc['ce_recon_geometry','knn_neighborhood_preservation'] - geo.loc['ce_recon','knn_neighborhood_preservation']:+.4f}")
        decision = "pilot_complete_review_required"
    text = "\n".join([
        "# R1 method decision",
        "",
        "R1 is a bounded pilot, not a final method claim or an adaptive-K policy.",
        "",
        f"* Selected global beta (Known-only seed-42 selection): `{selected_beta:g}`",
        f"* Gate units: `{completed}/108`; failed groups: `{failed}`",
        "* Frozen rows are E2 read-only references; CE-Recon rows are protocol-migration controls; Geometry rows are new method evidence.",
        f"* Decision: `{decision}`",
        "",
        *[f"* {reason}" for reason in reasons],
        "",
        "## Gate",
        "",
        "No OOS sample was used for training, beta selection or checkpoint selection. K=2 is reported only as a structural diagnostic.",
        "",
        "## Next step",
        "",
        "Review this pilot against the pre-registered gates. Do not start R1_full, ADB, DA-ADB, MOGB or complete Pipeline automatically.",
        "",
    ])
    atomic_write_text(summary_root / "R1_method_decision.md", text)
    atomic_write_text(summary_root / "R1_CLOSEOUT.md", text.replace("# R1 method decision", "# R1 closeout"))
    return {"decision": decision, "selected_beta": selected_beta, "completed_gate_units": completed, "failed": failed}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "select-beta", "pilot"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = ProtocolV2Paths.discover()
    config = load_config(args.config)
    if args.command == "freeze":
        print(json.dumps(_safe(freeze_provenance(paths, args.config)), ensure_ascii=False, indent=2))
        return 0
    if args.command == "select-beta":
        require_provenance(paths)
        result = select_beta(paths, config)
        print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
        return 0
    result = run_pilot(paths, config)
    print(json.dumps(_safe(result), ensure_ascii=False, indent=2))
    return 0 if result.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
