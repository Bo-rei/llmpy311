#!/usr/bin/env python3
"""Build real embedding-space evidence for the S2C mechanism comparison.

The builder reuses completed protocol artifacts.  It materializes frozen and
Trainable MiniLM vectors in memory, reconstructs the existing K=1 Gate and
MOGB-Fair boundaries, and writes only aggregate tables plus figures.  No
training, threshold selection, K selection, UMAP fitting, or public raw
embedding export is performed here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2c-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from protocol_v2.data.hashing import sha256_file  # noqa: E402
from protocol_v2.experiments.mogb import (  # noqa: E402
    AdaptiveGranularBallClusterer,
    MOGBBoundary,
    make_mogb_boundaries,
    score_mogb_boundaries,
)
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector  # noqa: E402
from protocol_v2.experiments.racal_v1.boundary import fit_k1_detector  # noqa: E402
from protocol_v2.experiments.runner import _embedding_sha256  # noqa: E402
from protocol_v2.experiments.trainable_native_baselines_v1 import (  # noqa: E402
    _checkpoint_path,
    _load_embeddings,
)
from protocol_v2.gate.view_loader import GateViews, load_gate_views  # noqa: E402
from protocol_v2.runtime.paths import ProtocolV2Paths  # noqa: E402
from protocol_v2.experiments.partitions import normalize_for_detector  # noqa: E402


OUT = ROOT / "results/analysis/deep_geometry_mechanism_v1"
FIG = ROOT / "figures/deep_geometry_mechanism_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
SEEDS = (13, 42, 87)
KIR = 0.50
PLOT_SEED = 42
PLOT_MAX_POINTS = 1800
NEIGHBOR_K = 10
SURFACE_RADIAL_LEVELS = (0.75, 0.90, 0.99, 1.01, 1.15, 1.35)
SURFACE_DIRECTION_COUNT = 24
NATIVE_METHODS = ("msp", "energy", "knn", "lof")
NATIVE_METHOD_LABELS = {
    "msp": "MSP",
    "energy": "Energy",
    "knn": "kNN",
    "lof": "LOF",
}

DATASET_LABELS = {
    "clinc150": "CLINC150",
    "banking77": "Banking77",
    "stackoverflow": "StackOverflow",
}
TRANSITION_LABELS = {
    "both_correct": "Both correct",
    "trainable_only_correct": "Trainable only",
    "baseline_only_correct": "Baseline only",
    "both_wrong": "Both wrong",
}
TRANSITION_COLORS = {
    "both_correct": "#9E9E9E",
    "trainable_only_correct": "#0072B2",
    "baseline_only_correct": "#D55E00",
    "both_wrong": "#C44E52",
}
METHOD_LABELS = {
    "frozen_k1": "Frozen K=1",
    "trainable_k1": "Trainable K=1",
    "mogb_minilm": "MOGB-MiniLM",
}


@dataclass
class GeometryUnit:
    dataset: str
    seed: int
    views: GateViews
    frozen_train: np.ndarray
    frozen_test: np.ndarray
    trainable_train: np.ndarray
    trainable_test: np.ndarray
    frozen_detector: Any
    trainable_detector: Any
    k2_detector: Any
    mogb_clusterer: AdaptiveGranularBallClusterer
    mogb_boundaries: list[MOGBBoundary]
    native_outputs: dict[str, dict[str, np.ndarray]]
    frame: pd.DataFrame
    audit: dict[str, Any]
    source_paths: list[Path]


def sha256(path: Path) -> str:
    return sha256_file(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight")
    figure.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")


def sample_ids_hash(rows: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(row["sample_id"]) for row in rows).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def frozen_cache(
    paths: ProtocolV2Paths,
    dataset: str,
    seed: int,
    split: str,
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, Path, Path]:
    """Load the exact frozen cache whose ordered sample-id hash matches rows."""

    cache_dir = paths.embedding_cache_root / dataset / f"seed_{seed}" / "kir_0.50"
    expected_ids = sample_ids_hash(rows)
    matches: list[tuple[Path, Path]] = []
    for metadata_path in sorted(cache_dir.glob(f"{split}_*.json")):
        metadata = read_json(metadata_path)
        if metadata.get("sample_ids_sha256") != expected_ids:
            continue
        array_path = metadata_path.with_suffix(".npz")
        if array_path.is_file():
            matches.append((array_path, metadata_path))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one frozen cache for {dataset}/{seed}/{split}; "
            f"found {len(matches)} in {cache_dir}"
        )
    array_path, metadata_path = matches[0]
    metadata = read_json(metadata_path)
    with np.load(array_path, allow_pickle=False) as payload:
        values = np.asarray(payload["embeddings"], dtype=np.float32)
    if values.shape[0] != len(rows):
        raise ValueError(f"Frozen cache row count mismatch: {array_path}")
    if metadata.get("embedding_sha256") != _embedding_sha256(values):
        raise ValueError(f"Frozen cache content hash mismatch: {array_path}")
    return values, array_path, metadata_path


def e2_run_dir(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    return paths.run_root / (
        "e2_gate_core_dense/"
        f"protocol_v2_textoir_v1__{dataset}__kir_0.50__seed_{seed}__"
        "repr_frozen_minilm__k_1__dist_mahalanobis_diag__boundary_mean_std"
    )


def trainable_run_dir(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    return paths.run_root / (
        f"minilm_trainable_kir_sweep_v1/kir_0.50/runs/{dataset}/seed_{seed}"
    )


def mogb_run_dir(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    return paths.run_root / f"mogb_baseline_v1/{dataset}/kir_0.50/seed_{seed}/mogb_minilm"


def k2_run_dir(paths: ProtocolV2Paths, dataset: str, seed: int) -> Path:
    return paths.run_root / f"mogb_baseline_v1/{dataset}/kir_0.50/seed_{seed}/fixed_k2"


def native_run_dir(paths: ProtocolV2Paths, dataset: str, seed: int, method: str) -> Path:
    return paths.run_root / (
        f"native_baselines_trainable_v1/runs/{dataset}/kir_0.50/seed_{seed}/{method}"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def normalize_prediction_label(predicted_oos: np.ndarray, labels: Iterable[Any]) -> np.ndarray:
    output = np.asarray([str(value) for value in labels], dtype=object)
    output[np.asarray(predicted_oos, dtype=bool)] = "__oos__"
    return output


def open_correct(gold_oos: np.ndarray, true_intent: np.ndarray, predicted_oos: np.ndarray, predicted_label: np.ndarray) -> np.ndarray:
    return np.where(
        gold_oos.astype(bool),
        predicted_oos.astype(bool),
        (~predicted_oos.astype(bool)) & (predicted_label.astype(str) == true_intent.astype(str)),
    ).astype(bool)


def transition(left_correct: np.ndarray, right_correct: np.ndarray) -> np.ndarray:
    return np.select(
        [left_correct & right_correct, left_correct & ~right_correct, ~left_correct & right_correct],
        ["both_correct", "trainable_only_correct", "baseline_only_correct"],
        default="both_wrong",
    ).astype(object)


def compare_stored(
    generated_ids: list[str],
    generated_pred: np.ndarray,
    generated_score: np.ndarray,
    stored_rows: list[dict[str, Any]],
    score_key: str,
    pred_key: str,
) -> dict[str, Any]:
    stored_ids = [str(row["sample_id"]) for row in stored_rows]
    stored_pred = np.asarray([int(row[pred_key]) for row in stored_rows], dtype=np.int64)
    stored_score = np.asarray([float(row[score_key]) for row in stored_rows], dtype=np.float64)
    return {
        "stored_row_count": int(len(stored_rows)),
        "sample_id_order_exact": bool(generated_ids == stored_ids),
        "prediction_mismatch_count": int(np.sum(generated_pred != stored_pred)),
        "score_max_abs_delta": float(np.max(np.abs(generated_score - stored_score))),
    }


def neighbor_features(train_values: np.ndarray, train_labels: np.ndarray, test_values: np.ndarray, true_intent: np.ndarray, gold_oos: np.ndarray) -> dict[str, np.ndarray]:
    nearest = NearestNeighbors(n_neighbors=NEIGHBOR_K, metric="euclidean", n_jobs=-1)
    nearest.fit(train_values)
    distances, indices = nearest.kneighbors(test_values)
    labels = train_labels[indices]
    counts = np.stack([(labels == label).sum(axis=1) for label in np.unique(train_labels)], axis=1)
    majority_share = counts.max(axis=1) / float(NEIGHBOR_K)
    same_intent_share = np.full(len(test_values), np.nan, dtype=np.float64)
    known = ~gold_oos.astype(bool)
    if np.any(known):
        same_intent_share[known] = np.mean(labels[known] == true_intent[known, None], axis=1)
    return {
        "knn_mean_distance": distances.mean(axis=1),
        "knn_majority_share": majority_share,
        "knn_same_intent_share": same_intent_share,
    }


def build_frame(
    views: GateViews,
    frozen_test: np.ndarray,
    trainable_test: np.ndarray,
    frozen_output: dict[str, np.ndarray],
    trainable_output: dict[str, np.ndarray],
    k2_output: dict[str, np.ndarray],
    mogb_output: dict[str, np.ndarray],
    native_outputs: dict[str, dict[str, np.ndarray]],
    frozen_train: np.ndarray,
    trainable_train: np.ndarray,
) -> pd.DataFrame:
    gold_oos = np.asarray([int(row["label"]) for row in views.test], dtype=np.int64)
    true_intent = np.asarray([str(row["intent"]) for row in views.test], dtype=object)
    frozen_label = normalize_prediction_label(
        frozen_output["pred"],
        ["__unknown__"] * len(views.test),
    )
    trainable_label = normalize_prediction_label(
        trainable_output["pred"],
        ["__unknown__"] * len(views.test),
    )
    mogb_label = normalize_prediction_label(
        mogb_output["predicted_oos"],
        mogb_output["predicted_label"],
    )
    frozen_label[~frozen_output["pred"].astype(bool)] = np.asarray(
        [str(frozen_output["nearest_intent"][index]) for index in range(len(views.test))],
        dtype=object,
    )[~frozen_output["pred"].astype(bool)]
    trainable_label[~trainable_output["pred"].astype(bool)] = np.asarray(
        [str(trainable_output["nearest_intent"][index]) for index in range(len(views.test))],
        dtype=object,
    )[~trainable_output["pred"].astype(bool)]
    k2_label = normalize_prediction_label(
        k2_output["pred"],
        ["__unknown__"] * len(views.test),
    )
    k2_label[~k2_output["pred"].astype(bool)] = np.asarray(
        [str(k2_output["nearest_intent"][index]) for index in range(len(views.test))],
        dtype=object,
    )[~k2_output["pred"].astype(bool)]
    frozen_correct = open_correct(gold_oos, true_intent, frozen_output["pred"], frozen_label)
    trainable_correct = open_correct(gold_oos, true_intent, trainable_output["pred"], trainable_label)
    k2_correct = open_correct(gold_oos, true_intent, k2_output["pred"], k2_label)
    mogb_correct = open_correct(gold_oos, true_intent, mogb_output["predicted_oos"], mogb_label)
    frozen_train_norm = normalize_for_detector(frozen_train)
    trainable_train_norm = normalize_for_detector(trainable_train)
    frozen_test_norm = normalize_for_detector(frozen_test)
    trainable_test_norm = normalize_for_detector(trainable_test)
    frozen_knn = neighbor_features(frozen_train_norm, np.asarray([str(row["intent"]) for row in views.train], dtype=object), frozen_test_norm, true_intent, gold_oos)
    trainable_knn = neighbor_features(trainable_train_norm, np.asarray([str(row["intent"]) for row in views.train], dtype=object), trainable_test_norm, true_intent, gold_oos)
    frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(views.test), dtype=np.int64),
            "true_intent": true_intent,
            "gold_oos": gold_oos,
            "frozen_pred_oos": frozen_output["pred"],
            "trainable_pred_oos": trainable_output["pred"],
            "k2_pred_oos": k2_output["pred"],
            "mogb_pred_oos": mogb_output["predicted_oos"],
            "frozen_score": frozen_output["score"],
            "trainable_score": trainable_output["score"],
            "k2_score": k2_output["score"],
            "mogb_score": mogb_output["score"],
            "frozen_distance": frozen_output["distance"],
            "trainable_distance": trainable_output["distance"],
            "k2_distance": k2_output["distance"],
            "mogb_distance": mogb_output["distance"],
            "frozen_radius": frozen_output["radius"],
            "trainable_radius": trainable_output["radius"],
            "k2_radius": k2_output["radius"],
            "mogb_radius": mogb_output["radius"],
            "frozen_correct": frozen_correct,
            "trainable_correct": trainable_correct,
            "k2_correct": k2_correct,
            "mogb_correct": mogb_correct,
            "transition_trainable_frozen": transition(trainable_correct, frozen_correct),
            "transition_trainable_k2": transition(trainable_correct, k2_correct),
            "transition_trainable_mogb": transition(trainable_correct, mogb_correct),
            "embedding_movement_norm": np.linalg.norm(trainable_test_norm - frozen_test_norm, axis=1),
            "delta_score": trainable_output["score"] - frozen_output["score"],
            "delta_distance": trainable_output["distance"] - frozen_output["distance"],
            "delta_radius": trainable_output["radius"] - frozen_output["radius"],
            "frozen_knn_mean_distance": frozen_knn["knn_mean_distance"],
            "trainable_knn_mean_distance": trainable_knn["knn_mean_distance"],
            "frozen_knn_majority_share": frozen_knn["knn_majority_share"],
            "trainable_knn_majority_share": trainable_knn["knn_majority_share"],
            "frozen_knn_same_intent_share": frozen_knn["knn_same_intent_share"],
            "trainable_knn_same_intent_share": trainable_knn["knn_same_intent_share"],
            "frozen_nearest_intent": frozen_output["nearest_intent"],
            "trainable_nearest_intent": trainable_output["nearest_intent"],
            "k2_nearest_intent": k2_output["nearest_intent"],
            "mogb_nearest_ball": mogb_output["nearest_ball"],
        }
    )
    for method, output in native_outputs.items():
        frame[f"{method}_pred_oos"] = output["pred"]
        frame[f"{method}_score"] = output["score"]
        frame[f"{method}_correct"] = output["correct"]
        frame[f"transition_trainable_{method}"] = transition(trainable_correct, output["correct"])
    frame["gold_type"] = np.where(frame["gold_oos"].astype(bool), "OOS", "Known")
    return frame


def detector_output(detector: Any, values: np.ndarray) -> dict[str, np.ndarray]:
    output = detector.predict_with_scores(values)
    nearest_intent = np.asarray(
        [detector.cluster_to_intent[int(cluster)] for cluster in output["nearest_cluster"]],
        dtype=object,
    )
    return {
        "pred": np.asarray(output["pred"], dtype=np.int64),
        "score": np.asarray(output["score"], dtype=np.float64),
        "distance": np.asarray(output["distance"], dtype=np.float64),
        "radius": np.asarray(output["radius"], dtype=np.float64),
        "nearest_intent": nearest_intent,
        "nearest_cluster": np.asarray(output["nearest_cluster"], dtype=np.int64),
    }


def fit_k2_detector(train: np.ndarray, views: GateViews) -> MultiSphereOOSDetector:
    """Recreate the frozen fixed-K=2 detector used by the stored component run."""

    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=2,
        radius_method="mean_std",
        radius_lambda=1.0,
        distance_metric="euclidean",
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
        acceptance_mode="nearest_sphere",
    )
    detector.fit(np.asarray(train), np.asarray([str(row["intent"]) for row in views.train], dtype=object))
    return detector


def native_output(
    paths: ProtocolV2Paths,
    dataset: str,
    seed: int,
    method: str,
    views: GateViews,
) -> tuple[dict[str, np.ndarray], Path, dict[str, Any]]:
    """Load a same-Trainable-MiniLM native detector without recomputing it."""

    prediction_path = native_run_dir(paths, dataset, seed, method) / "predictions/test.jsonl"
    rows = load_jsonl(prediction_path)
    generated_ids = [str(row["sample_id"]) for row in views.test]
    stored_ids = [str(row["sample_id"]) for row in rows]
    if generated_ids != stored_ids:
        raise ValueError(f"Native {method} sample order mismatch: {prediction_path}")
    predicted_oos = np.asarray([int(row["predicted_is_oos"]) for row in rows], dtype=np.int64)
    score = np.asarray([float(row["oos_score"]) for row in rows], dtype=np.float64)
    nearest_intent = np.asarray(
        [str(row.get("nearest_known_intent", "__unknown__")) for row in rows],
        dtype=object,
    )
    gold_oos = np.asarray([int(row["label"]) for row in views.test], dtype=np.int64)
    true_intent = np.asarray([str(row["intent"]) for row in views.test], dtype=object)
    predicted_label = nearest_intent.copy()
    predicted_label[predicted_oos.astype(bool)] = "__oos__"
    correct = open_correct(gold_oos, true_intent, predicted_oos, predicted_label)
    manifest_path = native_run_dir(paths, dataset, seed, method) / "manifest.json"
    return (
        {
            "pred": predicted_oos,
            "score": score,
            "nearest_intent": nearest_intent,
            "predicted_label": predicted_label,
            "correct": correct,
        },
        prediction_path,
        read_json(manifest_path),
    )


def build_unit(paths: ProtocolV2Paths, dataset: str, seed: int, device: str) -> GeometryUnit:
    paths.require_experiment_admission(dataset)
    views = load_gate_views(paths, dataset, seed, KIR)
    frozen_train, frozen_train_path, frozen_train_meta = frozen_cache(paths, dataset, seed, "train_known", views.train)
    frozen_test, frozen_test_path, frozen_test_meta = frozen_cache(paths, dataset, seed, "test_combined", views.test)
    checkpoint = _checkpoint_path(paths, dataset, KIR, seed)
    trainable_train, _, trainable_test, trainable_info = _load_embeddings(
        paths,
        views,
        checkpoint,
        device,
        batch_size=256,
        max_length=256,
    )
    frozen_detector = fit_k1_detector(frozen_train, views.train, "mahalanobis_diag")
    trainable_detector = fit_k1_detector(trainable_train, views.train, "mahalanobis_diag")
    k2_detector = fit_k2_detector(frozen_train, views)
    frozen_output = detector_output(frozen_detector, frozen_test)
    trainable_output = detector_output(trainable_detector, trainable_test)
    k2_output = detector_output(k2_detector, frozen_test)
    normalized_frozen_train = normalize_for_detector(frozen_train)
    normalized_frozen_test = normalize_for_detector(frozen_test)
    mogb_clusterer = AdaptiveGranularBallClusterer(seed=seed)
    mogb_clusterer.fit(
        normalized_frozen_train,
        np.asarray([str(row["intent"]) for row in views.train], dtype=object),
    )
    mogb_boundaries = make_mogb_boundaries(mogb_clusterer, boundary="mean", distance="euclidean")
    mogb_output = score_mogb_boundaries(normalized_frozen_test, mogb_boundaries, distance="euclidean")
    native_outputs: dict[str, dict[str, np.ndarray]] = {}
    native_paths: dict[str, Path] = {}
    for method in NATIVE_METHODS:
        output, prediction_path, native_manifest = native_output(paths, dataset, seed, method, views)
        native_outputs[method] = output
        native_paths[method] = prediction_path
        if native_manifest.get("test_used_for_selection") is not False:
            raise ValueError(f"Native {method} manifest does not prove Known-only selection: {native_run_dir(paths, dataset, seed, method)}")
    frame = build_frame(
        views,
        frozen_test,
        trainable_test,
        frozen_output,
        trainable_output,
        k2_output,
        mogb_output,
        native_outputs,
        frozen_train,
        trainable_train,
    )
    generated_ids = [str(row["sample_id"]) for row in views.test]
    trainable_rows = load_jsonl(trainable_run_dir(paths, dataset, seed) / "predictions.jsonl")
    frozen_rows = load_jsonl(e2_run_dir(paths, dataset, seed) / "predictions/test.jsonl")
    k2_rows = load_tsv(k2_run_dir(paths, dataset, seed) / "predictions.tsv").to_dict("records")
    mogb_rows = load_tsv(mogb_run_dir(paths, dataset, seed) / "predictions.tsv").to_dict("records")
    source_paths = [
        frozen_train_path,
        frozen_train_meta,
        frozen_test_path,
        frozen_test_meta,
        checkpoint,
        checkpoint.parent / "training_manifest.json",
        trainable_run_dir(paths, dataset, seed) / "predictions.jsonl",
        e2_run_dir(paths, dataset, seed) / "manifest.json",
        e2_run_dir(paths, dataset, seed) / "predictions/test.jsonl",
        k2_run_dir(paths, dataset, seed) / "manifest.json",
        k2_run_dir(paths, dataset, seed) / "predictions.tsv",
        mogb_run_dir(paths, dataset, seed) / "manifest.json",
        mogb_run_dir(paths, dataset, seed) / "inputs.json",
        mogb_run_dir(paths, dataset, seed) / "balls.jsonl",
        views.export_root / "export_manifest.json",
        *native_paths.values(),
        *(native_run_dir(paths, dataset, seed, method) / "manifest.json" for method in NATIVE_METHODS),
    ]
    audit = {
        "dataset": dataset,
        "seed": seed,
        "train_rows": len(views.train),
        "test_rows": len(views.test),
        "frozen_embedding_sha256": _embedding_sha256(frozen_test),
        "trainable_embedding_sha256": trainable_info["test_embedding_sha256"],
        "frozen_replay": compare_stored(
            generated_ids,
            frozen_output["pred"],
            frozen_output["score"],
            frozen_rows,
            "oos_score",
            "predicted_is_oos",
        ),
        "trainable_replay": compare_stored(
            generated_ids,
            trainable_output["pred"],
            trainable_output["score"],
            trainable_rows,
            "oos_score",
            "predicted_is_oos",
        ),
        "k2_replay": compare_stored(
            generated_ids,
            k2_output["pred"],
            k2_output["score"],
            k2_rows,
            "normalized_score",
            "predicted_is_oos",
        ),
        "mogb_replay": compare_stored(
            generated_ids,
            mogb_output["predicted_oos"],
            mogb_output["score"],
            mogb_rows,
            "normalized_score",
            "predicted_is_oos",
        ),
        "native_replay": {
            method: compare_stored(
                generated_ids,
                native_outputs[method]["pred"],
                native_outputs[method]["score"],
                load_jsonl(native_paths[method]),
                "oos_score",
                "predicted_is_oos",
            )
            for method in NATIVE_METHODS
        },
        "test_used_for_selection": False,
        "test_labels_used_post_hoc": True,
    }
    return GeometryUnit(
        dataset=dataset,
        seed=seed,
        views=views,
        frozen_train=frozen_train,
        frozen_test=frozen_test,
        trainable_train=trainable_train,
        trainable_test=trainable_test,
        frozen_detector=frozen_detector,
        trainable_detector=trainable_detector,
        k2_detector=k2_detector,
        mogb_clusterer=mogb_clusterer,
        mogb_boundaries=mogb_boundaries,
        native_outputs=native_outputs,
        frame=frame,
        audit=audit,
        source_paths=source_paths,
    )


def transition_summary(units: list[GeometryUnit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit in units:
        for comparison, column in (
            ("trainable_vs_frozen", "transition_trainable_frozen"),
            ("trainable_vs_mogb", "transition_trainable_mogb"),
        ):
            for (gold_type, name), group in unit.frame.groupby(["gold_type", column], sort=True):
                rows.append(
                    {
                        "dataset": unit.dataset,
                        "seed": unit.seed,
                        "comparison": comparison,
                        "gold_type": gold_type,
                        "transition": name,
                        "count": len(group),
                        "rate_within_gold_type": len(group) / max(int((unit.frame["gold_type"] == gold_type).sum()), 1),
                        "mean_embedding_movement_norm": group["embedding_movement_norm"].mean(),
                        "median_embedding_movement_norm": group["embedding_movement_norm"].median(),
                        "mean_delta_score": group["delta_score"].mean(),
                        "mean_delta_distance": group["delta_distance"].mean(),
                        "mean_delta_radius": group["delta_radius"].mean(),
                    }
                )
    return pd.DataFrame(rows)


def neighborhood_summary(units: list[GeometryUnit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit in units:
        frame = unit.frame
        for gold_type, group in frame.groupby("gold_type", sort=True):
            rows.append(
                {
                    "dataset": unit.dataset,
                    "seed": unit.seed,
                    "gold_type": gold_type,
                    "count": len(group),
                    "frozen_knn_majority_share_mean": group["frozen_knn_majority_share"].mean(),
                    "trainable_knn_majority_share_mean": group["trainable_knn_majority_share"].mean(),
                    "frozen_knn_same_intent_share_mean": group["frozen_knn_same_intent_share"].mean(),
                    "trainable_knn_same_intent_share_mean": group["trainable_knn_same_intent_share"].mean(),
                    "frozen_score_mean": group["frozen_score"].mean(),
                    "trainable_score_mean": group["trainable_score"].mean(),
                    "frozen_correct_rate": group["frozen_correct"].mean(),
                    "trainable_correct_rate": group["trainable_correct"].mean(),
                }
            )
    return pd.DataFrame(rows)


def boundary_summary(units: list[GeometryUnit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit in units:
        for method, detector, count in (
            ("frozen_k1", unit.frozen_detector, len(unit.frozen_detector.spheres)),
            ("trainable_k1", unit.trainable_detector, len(unit.trainable_detector.spheres)),
        ):
            radii = np.asarray([float(sphere.radius) for sphere in detector.spheres], dtype=np.float64)
            rows.append(
                {
                    "dataset": unit.dataset,
                    "seed": unit.seed,
                    "method": method,
                    "center_count": count,
                    "mean_radius": radii.mean(),
                    "median_radius": np.median(radii),
                    "min_radius": radii.min(),
                    "max_radius": radii.max(),
                    "boundary_distance": detector.distance_metric,
                    "boundary_rule": "mean_std",
                }
            )
        selected = unit.mogb_clusterer.selected_balls
        radii = np.asarray([float(ball.radius) for ball in selected], dtype=np.float64)
        rows.append(
            {
                "dataset": unit.dataset,
                "seed": unit.seed,
                "method": "mogb_minilm",
                "center_count": len(selected),
                "mean_radius": radii.mean(),
                "median_radius": np.median(radii),
                "min_radius": radii.min(),
                "max_radius": radii.max(),
                "boundary_distance": "euclidean",
                "boundary_rule": "mean",
            }
        )
    return pd.DataFrame(rows)


def intent_multimodality(units: list[GeometryUnit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit in units:
        frame = unit.frame
        train_labels = np.asarray([str(row["intent"]) for row in unit.views.train], dtype=object)
        train_norm = normalize_for_detector(unit.frozen_train)
        for intent in sorted(np.unique(train_labels).tolist()):
            points = train_norm[train_labels == intent]
            if len(points) < 20:
                continue
            center = points.mean(axis=0, keepdims=True)
            inertia_one = float(np.sum((points - center) ** 2))
            km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(points)
            ratio = inertia_one / max(float(km.inertia_), 1e-12)
            test_known = frame[(frame["gold_oos"] == 0) & frame["true_intent"].eq(intent)]
            if test_known.empty:
                continue
            rows.append(
                {
                    "dataset": unit.dataset,
                    "seed": unit.seed,
                    "intent": intent,
                    "train_count": len(points),
                    "two_mode_inertia_ratio": ratio,
                    "frozen_known_correct_rate": test_known["frozen_correct"].mean(),
                    "trainable_known_correct_rate": test_known["trainable_correct"].mean(),
                    "mogb_known_correct_rate": test_known["mogb_correct"].mean(),
                    "trainable_minus_mogb_known_gap_pp": 100.0 * (test_known["trainable_correct"].mean() - test_known["mogb_correct"].mean()),
                }
            )
    return pd.DataFrame(rows)


def acceptance_overlap_summary(units: list[GeometryUnit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit in units:
        frame = unit.frame
        for gold_type, group in frame.groupby("gold_type", sort=True):
            frozen_accept = ~group["frozen_pred_oos"].astype(bool)
            k2_accept = ~group["k2_pred_oos"].astype(bool)
            mogb_accept = ~group["mogb_pred_oos"].astype(bool)
            rows.append(
                {
                    "dataset": unit.dataset,
                    "seed": unit.seed,
                    "gold_type": gold_type,
                    "count": len(group),
                    "frozen_k1_accept_rate": frozen_accept.mean(),
                    "frozen_k2_accept_rate": k2_accept.mean(),
                    "mogb_accept_rate": mogb_accept.mean(),
                    "k2_added_acceptance_rate": (k2_accept & ~frozen_accept).mean(),
                    "k2_removed_acceptance_rate": (frozen_accept & ~k2_accept).mean(),
                    "mogb_added_acceptance_rate": (mogb_accept & ~frozen_accept).mean(),
                    "mogb_removed_acceptance_rate": (frozen_accept & ~mogb_accept).mean(),
                    "k2_added_known_count": int((k2_accept & ~frozen_accept & group["gold_type"].eq("Known")).sum()),
                    "k2_added_oos_count": int((k2_accept & ~frozen_accept & group["gold_type"].eq("OOS")).sum()),
                }
            )
    return pd.DataFrame(rows)


def spearman_corr(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    x = left[valid].rank(method="average").to_numpy(dtype=np.float64)
    y = right[valid].rank(method="average").to_numpy(dtype=np.float64)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def detector_rank_transfer_summary(units: list[GeometryUnit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit in units:
        frame = unit.frame
        for method in NATIVE_METHODS:
            transition_column = f"transition_trainable_{method}"
            for gold_type, group in frame.groupby("gold_type", sort=True):
                transitions = group[transition_column]
                rows.append(
                    {
                        "dataset": unit.dataset,
                        "seed": unit.seed,
                        "gold_type": gold_type,
                        "method": method,
                        "count": len(group),
                        "score_spearman_with_s2c": spearman_corr(group["trainable_score"], group[f"{method}_score"]),
                        "s2c_only_correct_rate": (transitions == "trainable_only_correct").mean(),
                        "baseline_only_correct_rate": (transitions == "baseline_only_correct").mean(),
                        "both_correct_rate": (transitions == "both_correct").mean(),
                        "both_wrong_rate": (transitions == "both_wrong").mean(),
                    }
                )
    return pd.DataFrame(rows)


def error_state(gold_type: pd.Series, predicted_oos: pd.Series, correct: pd.Series) -> pd.Series:
    values = np.full(len(gold_type), "", dtype=object)
    known = gold_type.eq("Known").to_numpy()
    oos = ~known
    predicted = predicted_oos.astype(bool).to_numpy()
    is_correct = correct.astype(bool).to_numpy()
    values[known & is_correct] = "Known correct"
    values[known & predicted] = "Known rejected"
    values[known & ~predicted & ~is_correct] = "Known wrong intent"
    values[oos & predicted] = "OOS rejected"
    values[oos & ~predicted] = "OOS accepted"
    return pd.Series(values, index=gold_type.index, dtype=object)


def multi_method_error_transitions(units: list[GeometryUnit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = ["frozen_k1", "k2", "mogb", *NATIVE_METHODS]
    for unit in units:
        frame = unit.frame
        s2c_state = error_state(frame["gold_type"], frame["trainable_pred_oos"], frame["trainable_correct"])
        for method in methods:
            pred_column = "frozen_pred_oos" if method == "frozen_k1" else f"{method}_pred_oos"
            correct_column = "frozen_correct" if method == "frozen_k1" else f"{method}_correct"
            baseline_state = error_state(frame["gold_type"], frame[pred_column], frame[correct_column])
            for gold_type in ("Known", "OOS"):
                mask = frame["gold_type"].eq(gold_type)
                pairs = pd.DataFrame({"s2c_state": s2c_state[mask], "baseline_state": baseline_state[mask]})
                counts = pairs.value_counts().rename("count").reset_index()
                total = max(int(mask.sum()), 1)
                for row in counts.to_dict("records"):
                    rows.append(
                        {
                            "dataset": unit.dataset,
                            "seed": unit.seed,
                            "gold_type": gold_type,
                            "baseline": method,
                            "s2c_state": row["s2c_state"],
                            "baseline_state": row["baseline_state"],
                            "count": int(row["count"]),
                            "rate_within_gold_type": int(row["count"]) / total,
                        }
                    )
    return pd.DataFrame(rows)


def mogb_ball_risk_summary(units: list[GeometryUnit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit in units:
        frame = unit.frame
        balls = {int(ball.ball_id): ball for ball in unit.mogb_clusterer.selected_balls}
        for boundary in unit.mogb_boundaries:
            ball_id = int(boundary.ball_id)
            assigned = frame[frame["mogb_nearest_ball"].eq(ball_id)]
            known = assigned[assigned["gold_type"].eq("Known")]
            oos = assigned[assigned["gold_type"].eq("OOS")]
            ball = balls[ball_id]
            rows.append(
                {
                    "dataset": unit.dataset,
                    "seed": unit.seed,
                    "ball_id": ball_id,
                    "support": int(ball.sample_count),
                    "purity": float(ball.purity),
                    "depth": int(ball.depth),
                    "radius": float(boundary.radius),
                    "test_assigned_count": len(assigned),
                    "test_known_count": len(known),
                    "test_oos_count": len(oos),
                    "known_reject_rate": float(known["mogb_pred_oos"].mean()) if len(known) else np.nan,
                    "oos_false_accept_rate": float((~oos["mogb_pred_oos"].astype(bool)).mean()) if len(oos) else np.nan,
                    "known_correct_rate": float(known["mogb_correct"].mean()) if len(known) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _boundary_distance_matrix(
    values: np.ndarray,
    centers: list[np.ndarray],
    inv_diag_covariances: list[np.ndarray | None],
    distance: str,
) -> np.ndarray:
    """Return high-dimensional distances without changing the coordinate frame."""

    points = np.asarray(values, dtype=np.float64)
    distances: list[np.ndarray] = []
    for center, inv_diag_covariance in zip(centers, inv_diag_covariances, strict=True):
        diff = points - np.asarray(center, dtype=np.float64)
        if distance == "mahalanobis_diag":
            if inv_diag_covariance is None:
                raise ValueError("Missing diagonal covariance for Mahalanobis boundary")
            distances.append(
                np.sqrt(np.sum(diff * diff * np.asarray(inv_diag_covariance, dtype=np.float64), axis=1))
            )
        elif distance == "euclidean":
            distances.append(np.linalg.norm(diff, axis=1))
        else:
            raise ValueError(f"Unknown boundary distance: {distance}")
    return np.stack(distances, axis=1)


def _detector_geometry(detector: Any, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    spheres = list(detector.spheres)
    centers = [np.asarray(sphere.center, dtype=np.float64) for sphere in spheres]
    radii = np.asarray([float(sphere.radius) for sphere in spheres], dtype=np.float64)
    inv_diag_covariances = [sphere.inv_diag_cov for sphere in spheres]
    normalized = detector._normalize_embeddings(np.asarray(values, dtype=np.float64))
    distances = _boundary_distance_matrix(
        normalized,
        centers,
        inv_diag_covariances,
        detector.distance_metric,
    )
    return distances, radii, [str(sphere.intent_name) for sphere in spheres]


def _mogb_geometry(values: np.ndarray, boundaries: list[MOGBBoundary]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    centers = [np.asarray(boundary.center, dtype=np.float64) for boundary in boundaries]
    radii = np.asarray([float(boundary.radius) for boundary in boundaries], dtype=np.float64)
    distances = _boundary_distance_matrix(
        np.asarray(values, dtype=np.float64),
        centers,
        [None] * len(boundaries),
        "euclidean",
    )
    return distances, radii, [str(boundary.label) for boundary in boundaries]


def _local_geometry_frame(unit: GeometryUnit, method: str) -> pd.DataFrame:
    """Build a sample-level local margin table in a method's native frame."""

    if method == "trainable_k1":
        distances, radii, labels = _detector_geometry(unit.trainable_detector, unit.trainable_test)
        score = unit.frame["trainable_score"].to_numpy(dtype=np.float64)
        predicted_oos = unit.frame["trainable_pred_oos"].to_numpy(dtype=np.int64)
        correct = unit.frame["trainable_correct"].to_numpy(dtype=bool)
    elif method == "mogb":
        distances, radii, labels = _mogb_geometry(
            normalize_for_detector(unit.frozen_test),
            unit.mogb_boundaries,
        )
        score = unit.frame["mogb_score"].to_numpy(dtype=np.float64)
        predicted_oos = unit.frame["mogb_pred_oos"].to_numpy(dtype=np.int64)
        correct = unit.frame["mogb_correct"].to_numpy(dtype=bool)
    else:
        raise ValueError(f"Unknown local geometry method: {method}")

    ratios = distances / np.maximum(radii[None, :], 1e-12)
    order = np.argsort(distances, axis=1)
    nearest = order[:, 0]
    second = ratios[np.arange(len(ratios)), order[:, 1]] if ratios.shape[1] > 1 else np.full(len(order), np.nan)
    true_intent = unit.frame["true_intent"].astype(str).to_numpy()
    gold_type = unit.frame["gold_type"].astype(str).to_numpy()
    true_ratio = np.full(len(order), np.nan, dtype=np.float64)
    wrong_ratio = np.full(len(order), np.nan, dtype=np.float64)
    for intent in np.unique(true_intent[gold_type == "Known"]):
        own = np.asarray([label == intent for label in labels], dtype=bool)
        other = ~own
        rows = (gold_type == "Known") & (true_intent == intent)
        if not np.any(rows) or not np.any(own):
            continue
        true_ratio[rows] = ratios[rows][:, own].min(axis=1)
        if np.any(other):
            wrong_ratio[rows] = ratios[rows][:, other].min(axis=1)

    margin = np.where(
        gold_type == "Known",
        wrong_ratio - true_ratio,
        second - ratios[np.arange(len(ratios)), nearest],
    )
    state = error_state(
        unit.frame["gold_type"],
        pd.Series(predicted_oos),
        pd.Series(correct),
    ).to_numpy(dtype=object)
    return pd.DataFrame(
        {
            "sample_index": unit.frame["sample_index"].to_numpy(dtype=np.int64),
            "gold_type": gold_type,
            "score": score,
            "margin": margin,
            "state": state,
            "nearest_label": np.asarray(labels, dtype=object)[nearest],
            "nearest_ratio": ratios[np.arange(len(ratios)), nearest],
        }
    )


def local_boundary_margin_summary(units: list[GeometryUnit]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for unit in units:
        for method in ("trainable_k1", "mogb"):
            local = _local_geometry_frame(unit, method)
            for (gold_type, state), group in local.groupby(["gold_type", "state"], sort=True):
                rows.append(
                    {
                        "dataset": unit.dataset,
                        "seed": unit.seed,
                        "method": method,
                        "gold_type": gold_type,
                        "state": state,
                        "count": len(group),
                        "score_median": float(group["score"].median()),
                        "score_q25": float(group["score"].quantile(0.25)),
                        "score_q75": float(group["score"].quantile(0.75)),
                        "margin_median": float(group["margin"].median()),
                        "margin_q25": float(group["margin"].quantile(0.25)),
                        "margin_q75": float(group["margin"].quantile(0.75)),
                    }
                )
    return pd.DataFrame(rows)


def _geometry_prediction(
    values: np.ndarray,
    centers: list[np.ndarray],
    radii: np.ndarray,
    inv_diag_covariances: list[np.ndarray | None],
    distance: str,
) -> tuple[np.ndarray, np.ndarray]:
    distances = _boundary_distance_matrix(values, centers, inv_diag_covariances, distance)
    nearest = np.argmin(distances, axis=1)
    score = distances[np.arange(len(distances)), nearest] / np.maximum(radii[nearest], 1e-12)
    predicted_oos = (distances[np.arange(len(distances)), nearest] > radii[nearest]).astype(np.int64)
    return predicted_oos, score


def acceptance_surface_summary(units: list[GeometryUnit]) -> pd.DataFrame:
    """Probe acceptance geometry around each frozen K=2 sphere surface.

    This is a deterministic geometric probe.  It uses no test labels and is
    not a threshold or model-selection step.
    """

    rows: list[dict[str, Any]] = []
    for unit in units:
        frozen_spheres = list(unit.frozen_detector.spheres)
        k1_centers = [np.asarray(sphere.center, dtype=np.float64) for sphere in frozen_spheres]
        k1_radii = np.asarray([float(sphere.radius) for sphere in frozen_spheres], dtype=np.float64)
        k1_cov = [sphere.inv_diag_cov for sphere in frozen_spheres]
        k2_spheres = list(unit.k2_detector.spheres)
        k2_centers = [np.asarray(sphere.center, dtype=np.float64) for sphere in k2_spheres]
        k2_radii = np.asarray([float(sphere.radius) for sphere in k2_spheres], dtype=np.float64)
        mogb_centers = [np.asarray(boundary.center, dtype=np.float64) for boundary in unit.mogb_boundaries]
        mogb_radii = np.asarray([float(boundary.radius) for boundary in unit.mogb_boundaries], dtype=np.float64)
        rng = np.random.default_rng(1701 + unit.seed + DATASETS.index(unit.dataset) * 10007)
        for sphere in k2_spheres:
            directions = rng.normal(size=(SURFACE_DIRECTION_COUNT, len(sphere.center)))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
            radial_scale = directions
            points = np.concatenate(
                [
                    np.asarray(sphere.center, dtype=np.float64)[None, :]
                    + float(level) * float(sphere.radius) * radial_scale
                    for level in SURFACE_RADIAL_LEVELS
                ],
                axis=0,
            )
            method_predictions = {
                "frozen_k1": _geometry_prediction(points, k1_centers, k1_radii, k1_cov, "mahalanobis_diag"),
                "frozen_k2": _geometry_prediction(points, k2_centers, k2_radii, [None] * len(k2_centers), "euclidean"),
                "mogb": _geometry_prediction(points, mogb_centers, mogb_radii, [None] * len(mogb_centers), "euclidean"),
            }
            for level_index, level in enumerate(SURFACE_RADIAL_LEVELS):
                start = level_index * SURFACE_DIRECTION_COUNT
                stop = start + SURFACE_DIRECTION_COUNT
                for method, (predicted_oos, score) in method_predictions.items():
                    accepted = ~predicted_oos[start:stop].astype(bool)
                    rows.append(
                        {
                            "dataset": unit.dataset,
                            "seed": unit.seed,
                            "source_center_id": int(sphere.cluster_id),
                            "source_intent": str(sphere.intent_name),
                            "source_boundary": "frozen_k2",
                            "radial_level": float(level),
                            "method": method,
                            "direction_count": SURFACE_DIRECTION_COUNT,
                            "acceptance_rate": float(np.mean(accepted)),
                            "score_median": float(np.median(score[start:stop])),
                        }
                    )
    return pd.DataFrame(rows)


def stable_sample(frame: pd.DataFrame, column: str, limit: int, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    pieces: list[pd.DataFrame] = []
    for index, (_, group) in enumerate(frame.groupby(column, sort=True)):
        take = min(len(group), max(1, limit // max(frame[column].nunique(), 1)))
        pieces.append(group.sample(n=take, random_state=seed + index))
    selected = pd.concat(pieces, ignore_index=False)
    if len(selected) < limit:
        remaining = frame.drop(selected.index, errors="ignore")
        selected = pd.concat([selected, remaining.sample(n=min(len(remaining), limit - len(selected)), random_state=seed)], ignore_index=False)
    return selected.head(limit)


def projected_ellipse(
    ax: Any,
    pca: PCA,
    center: np.ndarray,
    inv_diag_cov: np.ndarray | None,
    radius: float,
    color: str,
    linestyle: str,
    alpha: float,
    linewidth: float,
) -> None:
    dim = len(center)
    if inv_diag_cov is None:
        covariance = np.eye(dim, dtype=np.float64)
    else:
        covariance = np.diag(1.0 / np.clip(np.asarray(inv_diag_cov, dtype=np.float64), 1e-12, None))
    projected = pca.components_ @ covariance @ pca.components_.T
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 1e-12, None)
    direction = eigenvectors[:, order[0]]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    widths = 2.0 * float(radius) * np.sqrt(eigenvalues)
    location = pca.transform(np.asarray(center, dtype=np.float64).reshape(1, -1))[0]
    ax.add_patch(
        Ellipse(
            location,
            width=float(widths[0]),
            height=float(widths[1]),
            angle=angle,
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
    )


def plot_transition_points(ax: Any, coordinates: np.ndarray, frame: pd.DataFrame, transition_column: str, limit: int, seed: int) -> None:
    work = frame.copy()
    selected = stable_sample(work, transition_column, limit, seed)
    indices = selected["sample_index"].to_numpy(dtype=np.int64)
    for name, label in TRANSITION_LABELS.items():
        mask = selected[transition_column].eq(name).to_numpy()
        if not np.any(mask):
            continue
        ax.scatter(
            coordinates[indices[mask], 0],
            coordinates[indices[mask], 1],
            s=8,
            alpha=0.44 if name != "both_correct" else 0.16,
            color=TRANSITION_COLORS[name],
            edgecolors="none",
            rasterized=True,
            label=label,
        )


def pca_for_boundary(points: np.ndarray, centers: list[np.ndarray], extra_centers: list[np.ndarray]) -> PCA:
    arrays = [points[:: max(1, len(points) // 2500)], np.asarray(centers, dtype=np.float64)]
    if extra_centers:
        arrays.append(np.asarray(extra_centers, dtype=np.float64))
    basis = np.vstack(arrays)
    return PCA(n_components=2, random_state=0).fit(basis)


def plot_embedding_overlay(units: dict[str, GeometryUnit]) -> None:
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(12.8, 14.0), squeeze=False)
    for row, dataset in enumerate(DATASETS):
        unit = units[dataset]
        frame = unit.frame
        frozen_test_norm = normalize_for_detector(unit.frozen_test)
        trainable_test_norm = normalize_for_detector(unit.trainable_test)
        frozen_centers = [np.asarray(sphere.center, dtype=np.float64) for sphere in unit.frozen_detector.spheres]
        trainable_centers = [np.asarray(sphere.center, dtype=np.float64) for sphere in unit.trainable_detector.spheres]
        mogb_centers = [np.asarray(boundary.center, dtype=np.float64) for boundary in unit.mogb_boundaries]
        frozen_pca = pca_for_boundary(frozen_test_norm, frozen_centers, mogb_centers)
        trainable_pca = pca_for_boundary(trainable_test_norm, trainable_centers, [])
        frozen_xy = frozen_pca.transform(frozen_test_norm)
        trainable_xy = trainable_pca.transform(trainable_test_norm)
        left = axes[row, 0]
        right = axes[row, 1]
        plot_transition_points(left, frozen_xy, frame, "transition_trainable_mogb", PLOT_MAX_POINTS, 100 + row)
        plot_transition_points(right, trainable_xy, frame, "transition_trainable_mogb", PLOT_MAX_POINTS, 200 + row)
        for sphere in unit.frozen_detector.spheres:
            projected_ellipse(left, frozen_pca, np.asarray(sphere.center), sphere.inv_diag_cov, float(sphere.radius), "#333333", "--", 0.22, 0.75)
        for boundary in unit.mogb_boundaries:
            projected_ellipse(left, frozen_pca, boundary.center, None, float(boundary.radius), "#D55E00", ":", 0.07, 0.55)
        important = sorted(
            unit.mogb_boundaries,
            key=lambda boundary: int(unit.mogb_clusterer.balls[int(boundary.ball_id)].sample_count),
        )[-12:]
        for boundary in important:
            projected_ellipse(left, frozen_pca, boundary.center, None, float(boundary.radius), "#D55E00", ":", 0.38, 1.0)
        for sphere in unit.trainable_detector.spheres:
            projected_ellipse(right, trainable_pca, np.asarray(sphere.center), sphere.inv_diag_cov, float(sphere.radius), "#0072B2", "-", 0.25, 0.8)
        left.set_title(f"{DATASET_LABELS[dataset]} — Frozen space: K=1 + MOGB balls")
        right.set_title(f"{DATASET_LABELS[dataset]} — Trainable space: K=1")
        for axis in (left, right):
            axis.set_xlabel("PCA-1")
            axis.set_ylabel("PCA-2")
            axis.grid(alpha=0.14)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=6, label=label)
        for name, label in TRANSITION_LABELS.items()
        for color in [TRANSITION_COLORS[name]]
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#333333", linestyle="--", label="Frozen K=1 projected boundary"),
            Line2D([0], [0], color="#D55E00", linestyle=":", label="MOGB selected-ball boundary"),
            Line2D([0], [0], color="#0072B2", linestyle="-", label="Trainable K=1 projected boundary"),
        ]
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=8)
    fig.suptitle("Real 384-D embedding geometry and projected acceptance boundaries (KIR=0.50, seed=42)", y=0.99, fontsize=13)
    save_figure(fig, "real_space_boundary_overlay")
    plt.close(fig)


def plot_movement(units: dict[str, GeometryUnit]) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16.0, 5.2), squeeze=False)
    for col, dataset in enumerate(DATASETS):
        unit = units[dataset]
        frozen = normalize_for_detector(unit.frozen_test)
        trainable = normalize_for_detector(unit.trainable_test)
        pca = PCA(n_components=2, random_state=0).fit(np.vstack([frozen, trainable]))
        start = pca.transform(frozen)
        end = pca.transform(trainable)
        ax = axes[0, col]
        selected = stable_sample(unit.frame, "transition_trainable_frozen", 600, 500 + col)
        indices = selected["sample_index"].to_numpy(dtype=np.int64)
        for name, label in TRANSITION_LABELS.items():
            mask = selected["transition_trainable_frozen"].eq(name).to_numpy()
            if not np.any(mask):
                continue
            color = TRANSITION_COLORS[name]
            ax.quiver(
                start[indices[mask], 0],
                start[indices[mask], 1],
                end[indices[mask], 0] - start[indices[mask], 0],
                end[indices[mask], 1] - start[indices[mask], 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
                alpha=0.34 if name != "both_correct" else 0.12,
                width=0.002,
                headwidth=3.5,
                headlength=4.5,
                rasterized=True,
            )
            ax.scatter(start[indices[mask], 0], start[indices[mask], 1], s=7, color=color, alpha=0.25, marker="o", rasterized=True)
            ax.scatter(end[indices[mask], 0], end[indices[mask], 1], s=8, color=color, alpha=0.45, marker="^", rasterized=True)
        movement = unit.frame["embedding_movement_norm"]
        s2c_only = unit.frame["transition_trainable_frozen"].eq("trainable_only_correct")
        both_wrong = unit.frame["transition_trainable_frozen"].eq("both_wrong")
        annotation = (
            f"mean |Δz|={movement.mean():.3f}\n"
            f"Trainable-only: {s2c_only.sum()}\n"
            f"Both wrong: {both_wrong.sum()}"
        )
        ax.text(0.03, 0.97, annotation, transform=ax.transAxes, va="top", fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78})
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Joint PCA-1")
        ax.set_ylabel("Joint PCA-2")
        ax.grid(alpha=0.14)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TRANSITION_COLORS[name], markersize=6, label=f"{label} (Frozen→Trainable)")
        for name, label in TRANSITION_LABELS.items()
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="#555555", linestyle="None", label="Frozen start", markerfacecolor="none"),
            Line2D([0], [0], marker="^", color="#555555", linestyle="None", label="Trainable end"),
        ]
    )
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.925), fontsize=8)
    fig.suptitle("Same-sample representation movement and correctness flips", y=0.99, fontsize=13)
    save_figure(fig, "frozen_trainable_movement")
    plt.close(fig)


def plot_neighborhoods(units: dict[str, GeometryUnit]) -> None:
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(12.8, 13.2), squeeze=False)
    for row, dataset in enumerate(DATASETS):
        unit = units[dataset]
        for col, (representation, x_column) in enumerate(
            (("Frozen", "frozen_knn_majority_share"), ("Trainable", "trainable_knn_majority_share"))
        ):
            ax = axes[row, col]
            selected = stable_sample(unit.frame, "transition_trainable_frozen", PLOT_MAX_POINTS, 700 + row * 10 + col)
            for name, label in TRANSITION_LABELS.items():
                points = selected[selected["transition_trainable_frozen"].eq(name)]
                if points.empty:
                    continue
                ax.scatter(
                    points[x_column],
                    points[f"{representation.lower()}_score"],
                    s=9,
                    alpha=0.42 if name != "both_correct" else 0.14,
                    color=TRANSITION_COLORS[name],
                    rasterized=True,
                    label=label,
                )
            ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(bottom=min(0.4, float(selected[f"{representation.lower()}_score"].min()) - 0.03))
            ax.set_title(f"{DATASET_LABELS[dataset]} — {representation}")
            ax.set_xlabel("10-NN training-label majority share")
            ax.set_ylabel("Normalized OOS score")
            ax.grid(alpha=0.14)
            known = unit.frame[unit.frame["gold_type"].eq("Known")]
            oos = unit.frame[unit.frame["gold_type"].eq("OOS")]
            ax.text(
                0.03,
                0.04,
                f"Known NN share={known[x_column].mean():.2f}\nOOS NN share={oos[x_column].mean():.2f}",
                transform=ax.transAxes,
                fontsize=7,
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TRANSITION_COLORS[name], markersize=6, label=label)
        for name, label in TRANSITION_LABELS.items()
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=8)
    fig.suptitle("Local-neighborhood purity versus gate failure: representation-level evidence", y=0.99, fontsize=13)
    save_figure(fig, "local_neighborhood_failure_map")
    plt.close(fig)


def plot_decomposition(units: dict[str, GeometryUnit]) -> None:
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(16.0, 9.0), squeeze=False)
    for col, dataset in enumerate(DATASETS):
        frame = units[dataset].frame
        selected = stable_sample(frame, "transition_trainable_frozen", 1400, 900 + col)
        for row, (x_column, xlabel) in enumerate(
            (("delta_distance", "Δ nearest distance (Trainable − Frozen)"), ("delta_radius", "Δ selected radius (Trainable − Frozen)"))
        ):
            ax = axes[row, col]
            for name, label in TRANSITION_LABELS.items():
                points = selected[selected["transition_trainable_frozen"].eq(name)]
                if points.empty:
                    continue
                ax.scatter(
                    points[x_column],
                    points["delta_score"],
                    s=8,
                    alpha=0.40 if name != "both_correct" else 0.12,
                    color=TRANSITION_COLORS[name],
                    rasterized=True,
                )
            ax.axhline(0, color="#333333", linewidth=0.75, linestyle="--")
            ax.axvline(0, color="#333333", linewidth=0.75, linestyle="--")
            ax.set_title(DATASET_LABELS[dataset])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Δ normalized OOS score")
            ax.grid(alpha=0.14)
            correlation = np.corrcoef(selected[x_column].to_numpy(float), selected["delta_score"].to_numpy(float))[0, 1]
            ax.text(0.03, 0.96, f"Pearson r={correlation:.2f}", transform=ax.transAxes, va="top", fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78})
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TRANSITION_COLORS[name], markersize=6, label=label)
        for name, label in TRANSITION_LABELS.items()
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=8)
    fig.suptitle("Score change decomposition: representation distance versus fitted boundary radius", y=0.99, fontsize=13)
    save_figure(fig, "boundary_component_decomposition")
    plt.close(fig)


def plot_multimodality(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16.0, 5.0), squeeze=False)
    for col, dataset in enumerate(DATASETS):
        ax = axes[0, col]
        sub = data[data["dataset"].eq(dataset)]
        ax.scatter(
            sub["two_mode_inertia_ratio"],
            sub["trainable_minus_mogb_known_gap_pp"],
            s=24,
            color="#0072B2",
            alpha=0.75,
            edgecolors="white",
            linewidths=0.35,
        )
        if len(sub):
            for _, point in sub.nlargest(min(5, len(sub)), "two_mode_inertia_ratio").iterrows():
                ax.annotate(str(point["intent"]), (point["two_mode_inertia_ratio"], point["trainable_minus_mogb_known_gap_pp"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Two-mode inertia ratio (train Known intent)")
        ax.set_ylabel("Trainable − MOGB Known correct gap (pp)")
        ax.grid(alpha=0.14)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.suptitle("Intent multimodality and the Trainable–MOGB Known-coverage gap", y=0.99, fontsize=13)
    save_figure(fig, "intent_multimodality_error_gap")
    plt.close(fig)


def _overlay_state(left_pred: pd.Series, right_pred: pd.Series) -> pd.Series:
    left_accept = ~left_pred.astype(bool)
    right_accept = ~right_pred.astype(bool)
    return pd.Series(
        np.select(
            [left_accept & right_accept, ~left_accept & right_accept, left_accept & ~right_accept],
            ["both_accept", "right_added", "left_only"],
            default="both_reject",
        ),
        index=left_pred.index,
    )


def _plot_boundary_set(
    ax: Any,
    pca: PCA,
    detector: Any,
    color: str,
    linestyle: str,
    alpha: float,
    linewidth: float,
) -> None:
    for sphere in detector.spheres:
        projected_ellipse(
            ax,
            pca,
            np.asarray(sphere.center),
            sphere.inv_diag_cov,
            float(sphere.radius),
            color,
            linestyle,
            alpha,
            linewidth,
        )


def _plot_mogb_boundaries(ax: Any, pca: PCA, boundaries: list[MOGBBoundary], alpha: float = 0.35) -> None:
    for boundary in boundaries:
        projected_ellipse(ax, pca, boundary.center, None, float(boundary.radius), "#D55E00", ":", alpha, 0.65)


def plot_acceptance_overlap(units: dict[str, GeometryUnit]) -> None:
    """Show which samples are added or removed by the counterfactual region."""

    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(13.6, 14.0), squeeze=False)
    for row, dataset in enumerate(DATASETS):
        unit = units[dataset]
        frame = unit.frame.copy()
        frozen = normalize_for_detector(unit.frozen_test)
        frozen_centers = [np.asarray(sphere.center) for sphere in unit.frozen_detector.spheres]
        k2_centers = [np.asarray(sphere.center) for sphere in unit.k2_detector.spheres]
        mogb_centers = [np.asarray(boundary.center) for boundary in unit.mogb_boundaries]
        pca = pca_for_boundary(frozen, frozen_centers + k2_centers, mogb_centers)
        coordinates = pca.transform(frozen)
        for col, (right_column, right_label) in enumerate(
            (("k2_pred_oos", "Frozen K=1 → Frozen K=2"), ("mogb_pred_oos", "Frozen K=1 → MOGB"))
        ):
            ax = axes[row, col]
            frame["overlay_state"] = _overlay_state(frame["frozen_pred_oos"], frame[right_column])
            selected = stable_sample(frame, "overlay_state", PLOT_MAX_POINTS, 1200 + row * 10 + col)
            indices = selected["sample_index"].to_numpy(dtype=np.int64)
            ax.scatter(coordinates[indices, 0], coordinates[indices, 1], s=5, color="#BDBDBD", alpha=0.12, rasterized=True)
            for gold_type, marker, color in (("Known", "o", "#0072B2"), ("OOS", "^", "#C44E52")):
                for state, edge in (("right_added", color), ("left_only", "#D55E00")):
                    subset = selected[(selected["gold_type"] == gold_type) & (selected["overlay_state"] == state)]
                    if subset.empty:
                        continue
                    ids = subset["sample_index"].to_numpy(dtype=np.int64)
                    ax.scatter(
                        coordinates[ids, 0],
                        coordinates[ids, 1],
                        s=13,
                        marker=marker,
                        color=edge,
                        alpha=0.72,
                        edgecolors="white",
                        linewidths=0.25,
                        rasterized=True,
                    )
            _plot_boundary_set(ax, pca, unit.frozen_detector, "#333333", "--", 0.26, 0.7)
            if right_column == "k2_pred_oos":
                _plot_boundary_set(ax, pca, unit.k2_detector, "#0072B2", "-", 0.22, 0.7)
            else:
                _plot_mogb_boundaries(ax, pca, unit.mogb_boundaries, alpha=0.18)
            known_added = int(((frame["gold_type"] == "Known") & (frame["overlay_state"] == "right_added")).sum())
            oos_added = int(((frame["gold_type"] == "OOS") & (frame["overlay_state"] == "right_added")).sum())
            known_removed = int(((frame["gold_type"] == "Known") & (frame["overlay_state"] == "left_only")).sum())
            ax.text(
                0.03,
                0.97,
                f"right-added Known={known_added}\nright-added OOS={oos_added}\nleft-only Known={known_removed}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
            ax.set_title(f"{DATASET_LABELS[dataset]} — {right_label}")
            ax.set_xlabel("Frozen PCA-1")
            ax.set_ylabel("Frozen PCA-2")
            ax.grid(alpha=0.12)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", markersize=6, label="Known added/removed"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#C44E52", markersize=6, label="OOS added/removed"),
        Line2D([0], [0], color="#333333", linestyle="--", label="Frozen K=1"),
        Line2D([0], [0], color="#0072B2", linestyle="-", label="K=2"),
        Line2D([0], [0], color="#D55E00", linestyle=":", label="MOGB balls"),
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(handles=handles, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.965), fontsize=8)
    fig.suptitle("Counterfactual acceptance regions: coverage gain versus open-space risk", y=0.995, fontsize=13)
    save_figure(fig, "acceptance_overlap_risk")
    plt.close(fig)


def plot_mogb_ball_risk(ball_data: pd.DataFrame, units: dict[str, GeometryUnit]) -> None:
    """Map MOGB ball support to Known rejection and OOS contamination."""

    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(13.6, 14.0), squeeze=False)
    for row, dataset in enumerate(DATASETS):
        unit = units[dataset]
        frozen = normalize_for_detector(unit.frozen_test)
        frozen_centers = [np.asarray(sphere.center) for sphere in unit.frozen_detector.spheres]
        mogb_centers = [np.asarray(boundary.center) for boundary in unit.mogb_boundaries]
        pca = pca_for_boundary(frozen, frozen_centers, mogb_centers)
        coordinates = pca.transform(frozen)
        subset = ball_data[(ball_data["dataset"] == dataset) & (ball_data["seed"] == PLOT_SEED)]
        ball_by_id = {int(row_value["ball_id"]): row_value for row_value in subset.to_dict("records")}
        for col, metric in enumerate(("known_reject_rate", "oos_false_accept_rate")):
            ax = axes[row, col]
            ax.scatter(coordinates[:, 0], coordinates[:, 1], s=4, color="#BDBDBD", alpha=0.08, rasterized=True)
            values = subset[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            vmax = max(float(np.max(finite)) if len(finite) else 1.0, 1e-6)
            norm = Normalize(vmin=0.0, vmax=vmax)
            cmap = plt.get_cmap("magma" if metric == "known_reject_rate" else "viridis")
            finite_rows = subset[subset[metric].notna()].sort_values(metric, ascending=False)
            highlighted_ids = set(finite_rows.head(24)["ball_id"].astype(int).tolist())
            for boundary in unit.mogb_boundaries:
                ball_id = int(boundary.ball_id)
                risk_row = ball_by_id.get(ball_id)
                value = float(risk_row[metric]) if risk_row and pd.notna(risk_row[metric]) else np.nan
                highlighted = ball_id in highlighted_ids
                color = cmap(norm(value)) if highlighted and np.isfinite(value) else "#BDBDBD"
                support = int(risk_row["support"]) if risk_row else 1
                projected_ellipse(
                    ax,
                    pca,
                    boundary.center,
                    None,
                    float(boundary.radius),
                    color,
                    "-",
                    0.82 if highlighted and np.isfinite(value) else 0.06,
                    0.65 + min(1.5, np.log1p(support) / 4.0) if highlighted else 0.35,
                )
                location = pca.transform(np.asarray(boundary.center).reshape(1, -1))[0]
                if highlighted:
                    ax.scatter(location[0], location[1], s=8 + 2.5 * np.sqrt(max(support, 1)), color=color, edgecolors="white", linewidths=0.25, alpha=0.82)
            ax.set_title(f"{DATASET_LABELS[dataset]} — MOGB {metric.replace('_', ' ')}")
            ax.set_xlabel("Frozen PCA-1")
            ax.set_ylabel("Frozen PCA-2")
            ax.grid(alpha=0.12)
            ax.text(
                0.03,
                0.04,
                f"selected balls={len(subset)}\nmedian purity={subset['purity'].median():.3f}\nmedian support={subset['support'].median():.0f}",
                transform=ax.transAxes,
                va="bottom",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
            colorbar_axis = fig.add_axes([0.46 if col == 0 else 0.94, 0.12 + (len(DATASETS) - row - 1) * 0.28, 0.012, 0.18])
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis)
    fig.tight_layout(rect=[0, 0, 0.91, 0.95])
    fig.suptitle("MOGB ball mechanism: local support, Known coverage loss and OOS contamination", y=0.995, fontsize=13)
    save_figure(fig, "mogb_ball_support_risk_map")
    plt.close(fig)


def plot_detector_rank_transfer(units: dict[str, GeometryUnit]) -> None:
    """Compare native detector ranking with S2C on the identical representation."""

    fig, axes = plt.subplots(len(DATASETS), len(NATIVE_METHODS), figsize=(17.0, 12.8), squeeze=False)
    for row, dataset in enumerate(DATASETS):
        frame = units[dataset].frame.copy()
        for col, method in enumerate(NATIVE_METHODS):
            ax = axes[row, col]
            transition_column = f"transition_trainable_{method}"
            frame["native_rank"] = frame[f"{method}_score"].rank(pct=True)
            frame["s2c_rank"] = frame["trainable_score"].rank(pct=True)
            selected = stable_sample(frame, transition_column, 1400, 1600 + row * 10 + col)
            for name in TRANSITION_LABELS:
                for gold_type, marker in (("Known", "o"), ("OOS", "^")):
                    points = selected[(selected[transition_column] == name) & (selected["gold_type"] == gold_type)]
                    if points.empty:
                        continue
                    ax.scatter(
                        points["native_rank"],
                        points["s2c_rank"],
                        s=8,
                        marker=marker,
                        color=TRANSITION_COLORS[name],
                        alpha=0.34 if name != "both_correct" else 0.11,
                        edgecolors="none",
                        rasterized=True,
                    )
            ax.plot([0.0, 1.0], [0.0, 1.0], color="#555555", linestyle="--", linewidth=0.7)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            rho = spearman_corr(frame["trainable_score"], frame[f"{method}_score"])
            s2c_only = (frame[transition_column] == "trainable_only_correct").mean()
            baseline_only = (frame[transition_column] == "baseline_only_correct").mean()
            ax.text(
                0.04,
                0.96,
                f"ρ={rho:.2f}\nS2C-only={100*s2c_only:.1f}%\nbase-only={100*baseline_only:.1f}%",
                transform=ax.transAxes,
                va="top",
                fontsize=6.7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80},
            )
            ax.set_title(f"{DATASET_LABELS[dataset]} — {NATIVE_METHOD_LABELS[method]}")
            ax.set_xlabel("Native detector score percentile")
            ax.set_ylabel("S2C score percentile")
            ax.grid(alpha=0.12)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TRANSITION_COLORS[name], markersize=6, label=label)
        for name, label in TRANSITION_LABELS.items()
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="#555555", linestyle="None", label="Known"),
            Line2D([0], [0], marker="^", color="#555555", linestyle="None", label="OOS"),
        ]
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.legend(handles=handles, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.965), fontsize=8)
    fig.suptitle("Same representation, different detector ranking: why native baselines lose different samples", y=0.995, fontsize=13)
    save_figure(fig, "detector_rank_transfer")
    plt.close(fig)


def plot_multi_method_error_transitions(units: dict[str, GeometryUnit]) -> None:
    """Show the complete S2C-to-baseline state transition matrix for StackOverflow."""

    unit = units["stackoverflow"]
    frame = unit.frame
    methods = ("frozen_k1", "k2", "mogb", *NATIVE_METHODS)
    method_labels = {
        "frozen_k1": "Frozen K=1",
        "k2": "Frozen K=2",
        "mogb": "MOGB",
        **NATIVE_METHOD_LABELS,
    }
    states = ("Known correct", "Known rejected", "Known wrong intent", "OOS rejected", "OOS accepted")
    short_states = ("K correct", "K reject", "K wrong", "OOS reject", "OOS accept")
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.8), squeeze=False)
    last_image = None
    for axis, method in zip(axes.ravel(), methods):
        s2c = error_state(frame["gold_type"], frame["trainable_pred_oos"], frame["trainable_correct"])
        pred_column = "frozen_pred_oos" if method == "frozen_k1" else f"{method}_pred_oos"
        correct_column = "frozen_correct" if method == "frozen_k1" else f"{method}_correct"
        baseline = error_state(frame["gold_type"], frame[pred_column], frame[correct_column])
        matrix = pd.crosstab(s2c, baseline).reindex(index=states, columns=states, fill_value=0).to_numpy(dtype=np.float64)
        matrix /= max(len(frame), 1)
        last_image = axis.imshow(matrix * 100.0, cmap="Blues", vmin=0.0, vmax=max(25.0, float(np.max(matrix * 100.0))))
        for i in range(len(states)):
            for j in range(len(states)):
                value = matrix[i, j] * 100.0
                if value < 0.05:
                    continue
                axis.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7, color="white" if value > 8 else "#222222")
        axis.set_xticks(range(len(states)), short_states, rotation=35, ha="right", fontsize=7)
        axis.set_yticks(range(len(states)), short_states, fontsize=7)
        axis.set_xlabel(f"{method_labels[method]} state")
        axis.set_ylabel("S2C state")
        axis.set_title(method_labels[method])
    axes.ravel()[-1].axis("off")
    fig.tight_layout(rect=[0, 0, 0.90, 0.92])
    if last_image is not None:
        colorbar_axis = fig.add_axes([0.925, 0.22, 0.018, 0.56])
        fig.colorbar(last_image, cax=colorbar_axis, label="Share of all test samples (%)")
    fig.suptitle("StackOverflow/KIR=.50/seed42: complete error-state transitions from S2C to each baseline", y=0.995, fontsize=13)
    save_figure(fig, "multi_method_error_transitions")
    plt.close(fig)


LOCAL_STATE_COLORS = {
    "Known correct": "#0072B2",
    "Known rejected": "#D55E00",
    "Known wrong intent": "#C44E52",
    "OOS rejected": "#2A9D8F",
    "OOS accepted": "#9467BD",
}


def plot_local_boundary_competition(units: dict[str, GeometryUnit]) -> None:
    """Show why a sample is rejected, accepted, or assigned to a wrong intent."""

    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(14.0, 13.5), squeeze=False)
    for row, dataset in enumerate(DATASETS):
        for col, (method, title) in enumerate(
            (("trainable_k1", "Trainable K=1"), ("mogb", "MOGB"))
        ):
            ax = axes[row, col]
            local = _local_geometry_frame(units[dataset], method)
            selected = stable_sample(local, "state", PLOT_MAX_POINTS, 2100 + row * 10 + col)
            for state, color in LOCAL_STATE_COLORS.items():
                points = selected[selected["state"].eq(state)]
                if points.empty:
                    continue
                marker = "o" if state.startswith("Known") else "^"
                ax.scatter(
                    points["score"],
                    points["margin"],
                    s=9,
                    marker=marker,
                    color=color,
                    alpha=0.48 if state not in {"Known correct", "OOS rejected"} else 0.22,
                    edgecolors="none",
                    rasterized=True,
                )
            finite_score = local["score"].to_numpy(dtype=np.float64)
            finite_margin = local["margin"].dropna().to_numpy(dtype=np.float64)
            x_max = max(1.25, float(np.quantile(finite_score, 0.995)) * 1.05)
            y_low = float(np.quantile(finite_margin, 0.005))
            y_high = float(np.quantile(finite_margin, 0.995))
            if y_low == y_high:
                y_low, y_high = y_low - 0.05, y_high + 0.05
            y_pad = max(0.03, 0.08 * (y_high - y_low))
            ax.set_xlim(left=0.0, right=x_max)
            ax.set_ylim(y_low - y_pad, y_high + y_pad)
            ax.axvline(1.0, color="#333333", linestyle="--", linewidth=0.8)
            ax.axhline(0.0, color="#777777", linestyle=":", linewidth=0.7)
            counts = local["state"].value_counts()
            annotation = (
                f"K correct={int(counts.get('Known correct', 0))}\n"
                f"K reject={int(counts.get('Known rejected', 0))}\n"
                f"K wrong={int(counts.get('Known wrong intent', 0))}\n"
                f"OOS accept={int(counts.get('OOS accepted', 0))}"
            )
            ax.text(
                0.03,
                0.97,
                annotation,
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
            ax.set_title(f"{DATASET_LABELS[dataset]} — {title}")
            ax.set_xlabel("Nearest normalized distance / radius")
            ax.set_ylabel("Known: wrong − true ratio; OOS: 2nd − 1st ratio")
            ax.grid(alpha=0.14)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=6, label=state)
        for state, color in LOCAL_STATE_COLORS.items()
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#333333", linestyle="--", label="Acceptance boundary = 1"),
            Line2D([0], [0], color="#777777", linestyle=":", label="Identity tie = 0"),
        ]
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=8)
    fig.suptitle(
        "Local decision geometry: rejection margin and intent competition in the native high-dimensional rule",
        y=0.995,
        fontsize=13,
    )
    save_figure(fig, "local_boundary_competition_margin")
    plt.close(fig)


def plot_acceptance_surface(surface: pd.DataFrame) -> None:
    """Visualize union/open-space exposure outside the frozen K=1 surface."""

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(16.0, 10.8), squeeze=False)
    method_labels = {
        "frozen_k1": "Frozen K=1",
        "frozen_k2": "Frozen K=2",
        "mogb": "MOGB",
    }
    method_colors = {
        "frozen_k1": "#333333",
        "frozen_k2": "#0072B2",
        "mogb": "#D55E00",
    }
    for col, dataset in enumerate(DATASETS):
        subset = surface[surface["dataset"].eq(dataset) & surface["seed"].eq(PLOT_SEED)]
        top = axes[0, col]
        for method in method_labels:
            grouped = (
                subset[subset["method"].eq(method)]
                .groupby("radial_level", as_index=False)["acceptance_rate"]
                .agg(["mean", "std"])
                .reset_index()
            )
            if grouped.empty:
                continue
            top.plot(
                grouped["radial_level"],
                grouped["mean"],
                marker="o",
                linewidth=1.4,
                markersize=4,
                color=method_colors[method],
                label=method_labels[method],
            )
            spread = grouped["std"].fillna(0.0).to_numpy(dtype=np.float64)
            mean = grouped["mean"].to_numpy(dtype=np.float64)
            x = grouped["radial_level"].to_numpy(dtype=np.float64)
            top.fill_between(x, np.clip(mean - spread, 0.0, 1.0), np.clip(mean + spread, 0.0, 1.0), color=method_colors[method], alpha=0.08)
        top.axvline(1.0, color="#555555", linestyle="--", linewidth=0.8)
        top.set_ylim(-0.02, 1.02)
        top.set_title(DATASET_LABELS[dataset])
        top.set_xlabel("Radial level relative to Frozen K=2 sphere")
        top.set_ylabel("Accepted fraction of directions")
        top.grid(alpha=0.14)
        top.text(
            0.03,
            0.05,
            "synthetic boundary probe\nno test labels",
            transform=top.transAxes,
            fontsize=7,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )

        bottom = axes[1, col]
        level = 0.90
        level_data = subset[subset["radial_level"].eq(level)]
        pivot = level_data.pivot_table(
            index="source_intent",
            columns="method",
            values="acceptance_rate",
            aggfunc="mean",
        )
        if "frozen_k1" not in pivot:
            bottom.axis("off")
            continue
        for method in ("frozen_k2", "mogb"):
            if method not in pivot:
                pivot[method] = np.nan
        pivot["k2_delta"] = pivot["frozen_k2"] - pivot["frozen_k1"]
        pivot["mogb_delta"] = pivot["mogb"] - pivot["frozen_k1"]
        selected = pivot.reindex(pivot["k2_delta"].abs().sort_values(ascending=False).head(10).index)
        y = np.arange(len(selected), dtype=np.float64)
        bottom.barh(y - 0.17, selected["k2_delta"], height=0.32, color="#0072B2", label="K=2 − K=1")
        bottom.barh(y + 0.17, selected["mogb_delta"], height=0.32, color="#D55E00", label="MOGB − K=1")
        bottom.axvline(0.0, color="#333333", linewidth=0.75)
        bottom.set_yticks(y, [str(index) for index in selected.index], fontsize=6)
        bottom.set_xlabel("Acceptance-rate difference at radial level 0.90")
        bottom.set_title("K=2 interior coverage by source intent")
        bottom.grid(axis="x", alpha=0.14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=8)
    fig.suptitle(
        "Acceptance-region geometry around K=2 centers: local-mode coverage versus K=1 and MOGB",
        y=0.995,
        fontsize=13,
    )
    save_figure(fig, "acceptance_surface_exposure")
    plt.close(fig)


def main() -> None:
    configure_plot()
    paths = ProtocolV2Paths.discover()
    units: list[GeometryUnit] = []
    plot_units: dict[str, GeometryUnit] = {}
    audits: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"[deep-geometry] {dataset} seed={seed}", flush=True)
            unit = build_unit(paths, dataset, seed, device="auto")
            units.append(unit)
            audits.append(unit.audit)
            if seed == PLOT_SEED:
                plot_units[dataset] = unit
    if set(plot_units) != set(DATASETS):
        raise RuntimeError(f"Missing plot units: {sorted(set(DATASETS) - set(plot_units))}")
    OUT.mkdir(parents=True, exist_ok=True)
    transition = transition_summary(units)
    neighborhoods = neighborhood_summary(units)
    boundaries = boundary_summary(units)
    multimodality = intent_multimodality([unit for unit in units if unit.seed == PLOT_SEED])
    acceptance = acceptance_overlap_summary(units)
    rank_transfer = detector_rank_transfer_summary(units)
    error_transitions = multi_method_error_transitions(units)
    ball_risk = mogb_ball_risk_summary(units)
    local_margin = local_boundary_margin_summary(units)
    surface = acceptance_surface_summary(units)
    atomic_csv(transition, OUT / "movement_transition_summary.csv")
    atomic_csv(neighborhoods, OUT / "neighborhood_summary.csv")
    atomic_csv(boundaries, OUT / "boundary_summary.csv")
    atomic_csv(multimodality, OUT / "intent_multimodality_error_gap.csv")
    atomic_csv(acceptance, OUT / "acceptance_overlap_summary.csv")
    atomic_csv(rank_transfer, OUT / "detector_rank_transfer_summary.csv")
    atomic_csv(error_transitions, OUT / "multi_method_error_transitions.csv")
    atomic_csv(ball_risk, OUT / "mogb_ball_risk_summary.csv")
    atomic_csv(local_margin, OUT / "local_boundary_margin_summary.csv")
    atomic_csv(surface, OUT / "acceptance_surface_summary.csv")
    atomic_json(audits, OUT / "reconstruction_audit.json")
    plot_embedding_overlay(plot_units)
    plot_movement(plot_units)
    plot_neighborhoods(plot_units)
    plot_decomposition(plot_units)
    plot_multimodality(multimodality)
    plot_acceptance_overlap(plot_units)
    plot_mogb_ball_risk(ball_risk, plot_units)
    plot_detector_rank_transfer(plot_units)
    plot_multi_method_error_transitions(plot_units)
    plot_local_boundary_competition(plot_units)
    plot_acceptance_surface(surface)
    source_paths = sorted({path for unit in units for path in unit.source_paths})
    manifest = {
        "analysis": "deep_geometry_mechanism_pack_v1",
        "protocol_version": paths.dataset_version,
        "analysis_only": True,
        "datasets": list(DATASETS),
        "kir": KIR,
        "seeds": list(SEEDS),
        "plot_seed": PLOT_SEED,
        "neighbor_k": NEIGHBOR_K,
        "projection": "PCA-2D-for-visualization-only",
        "selection_used_test_oos": False,
        "test_labels_used_post_hoc": True,
        "raw_embeddings_exported": False,
        "sample_ids_exported": False,
        "umap_used": False,
        "figure_export_formats": ["png_preview_600dpi", "svg", "pdf"],
        "figure_count": len(list(FIG.glob("*.png"))),
        "source_hashes": {
            str(path.relative_to(ROOT.parent)): sha256(path)
            for path in source_paths
            if path.is_file()
        },
        "outputs": {
            "movement_transition_summary": str((OUT / "movement_transition_summary.csv").relative_to(ROOT)),
            "neighborhood_summary": str((OUT / "neighborhood_summary.csv").relative_to(ROOT)),
            "boundary_summary": str((OUT / "boundary_summary.csv").relative_to(ROOT)),
            "intent_multimodality_error_gap": str((OUT / "intent_multimodality_error_gap.csv").relative_to(ROOT)),
            "reconstruction_audit": str((OUT / "reconstruction_audit.json").relative_to(ROOT)),
            "acceptance_overlap_summary": str((OUT / "acceptance_overlap_summary.csv").relative_to(ROOT)),
            "detector_rank_transfer_summary": str((OUT / "detector_rank_transfer_summary.csv").relative_to(ROOT)),
            "multi_method_error_transitions": str((OUT / "multi_method_error_transitions.csv").relative_to(ROOT)),
            "mogb_ball_risk_summary": str((OUT / "mogb_ball_risk_summary.csv").relative_to(ROOT)),
            "local_boundary_margin_summary": str((OUT / "local_boundary_margin_summary.csv").relative_to(ROOT)),
            "acceptance_surface_summary": str((OUT / "acceptance_surface_summary.csv").relative_to(ROOT)),
        },
        "figures": [str(path.relative_to(ROOT)) for path in sorted(FIG.glob("*.png"))],
    }
    atomic_json(manifest, OUT / "MANIFEST.json")
    print(json.dumps({"status": "ok", "units": len(units), "figures": len(manifest["figures"]), "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
