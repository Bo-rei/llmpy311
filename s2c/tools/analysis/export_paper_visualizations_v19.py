#!/usr/bin/env python3
"""Export paper-ready visualization evidence for v19 pipeline results.

The script reads saved experiment artifacts only. It does not synthesize
predictions, scores, or embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "pipeline"
    / "ablations"
    / "latest_strongest_v19"
    / "paper_mainline_proto_kir50_20260427"
)
DATASET_SLUGS = {
    "CLINC150": "clinc150",
    "BANKING77-OOS": "banking77_oos",
    "StackOverflow": "stackoverflow",
}
KIND_ORDER = ["known", "heldout_unknown", "native_oos"]
KIND_LABELS = {
    "known": "Known samples",
    "heldout_unknown": "Unknown samples",
    "native_oos": "OOS samples",
}
UMAP_PANEL_TITLES = [
    "Known samples and centroids",
    "Known vs held-out unknown samples",
    "Known vs OOS samples",
]
UMAP_LEGEND_LABELS = {
    "known": "Known samples",
    "centroids": "Cluster Centroids",
    "heldout_unknown": "Unknown samples",
    "native_oos": "OOS samples",
}
KIND_COLORS = {
    "known": "#4C78A8",
    "heldout_unknown": "#F58518",
    "native_oos": "#C44E52",
}
UMAP_PARAMS = {
    "n_components": 2,
    "n_neighbors": 30,
    "min_dist": 0.08,
    "metric": "cosine",
}
QUALITATIVE_NOTE = (
    "UMAP is used only for qualitative visualization; quantitative evaluation is based on OOS F1, "
    "Acc, Known F1, and gate score distributions."
)


@dataclass
class DetectorCenters:
    centers: np.ndarray
    radii: np.ndarray
    labels: List[str]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sample_kind(prediction: Dict[str, Any]) -> str:
    if not bool(prediction.get("is_oos", False)):
        return "known"
    if str(prediction.get("true_intent")) == "oos":
        return "native_oos"
    return "heldout_unknown"


def prediction_score(prediction: Dict[str, Any]) -> float:
    score = prediction.get("gate_score")
    if score is not None:
        return float(score)
    distance = float(prediction["gate_distance"])
    radius = float(prediction["gate_radius"])
    return distance / radius


def load_gate_score_rows(predictions_path: Path) -> List[Dict[str, Any]]:
    rows = []
    for prediction in load_json(predictions_path):
        rows.append({"kind": sample_kind(prediction), "score": prediction_score(prediction)})
    return rows


def load_detector_centers(detector_path: Path) -> DetectorCenters:
    detector = load_json(detector_path)
    spheres = detector.get("spheres", [])
    centers = np.asarray([sphere["center"] for sphere in spheres], dtype=np.float64)
    radii = np.asarray([sphere["radius"] for sphere in spheres], dtype=np.float64)
    labels = [str(sphere.get("intent_name", sphere.get("cluster_id", idx))) for idx, sphere in enumerate(spheres)]
    return DetectorCenters(centers=centers, radii=radii, labels=labels)


def select_known_intents(
    predictions: Sequence[Dict[str, Any]],
    count: int,
    seed: int,
    min_support: int = 5,
) -> List[str]:
    support = Counter(str(row["true_intent"]) for row in predictions if not bool(row.get("is_oos", False)))
    candidates = sorted(intent for intent, size in support.items() if size >= min_support)
    if len(candidates) <= count:
        return candidates
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(candidates, size=count, replace=False).tolist())


def compute_space_metrics(
    embeddings: np.ndarray,
    kinds: Sequence[str],
    centers: DetectorCenters,
) -> List[Dict[str, Any]]:
    diff = embeddings[:, None, :] - centers.centers[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = distances[np.arange(len(embeddings)), nearest_indices]
    nearest_ratios = nearest_distances / centers.radii[nearest_indices]
    rows = []
    for kind in KIND_ORDER:
        mask = np.asarray([item == kind for item in kinds], dtype=bool)
        if not mask.any():
            continue
        rows.append(
            {
                "kind": kind,
                "count": int(mask.sum()),
                "mean_nearest_center_distance": float(np.mean(nearest_distances[mask])),
                "median_nearest_center_distance": float(np.median(nearest_distances[mask])),
                "mean_distance_radius_ratio": float(np.mean(nearest_ratios[mask])),
                "median_distance_radius_ratio": float(np.median(nearest_ratios[mask])),
            }
        )
    return rows


def error_breakdown_row(dataset: str, eval_results_path: Path) -> Dict[str, Any]:
    metrics = load_json(eval_results_path).get("metrics", {})
    breakdown = metrics["cascade_error_breakdown"]
    known = breakdown["known"]
    oos = breakdown["oos"]
    return {
        "dataset": dataset,
        "gate_false_reject": int(known.get("gate_false_reject", 0)),
        "router_wrong_dispatch": int(known.get("router_error_given_gate_pass", 0)),
        "expert_wrong_classification": int(known.get("expert_error_given_router_correct", 0)),
        "oos_false_accept": int(oos.get("gate_false_accept", 0)),
    }


def dataset_dir(root: Path, slug: str) -> Path:
    return root / slug / "kir50_seed42" / "full_anchor"


def run_manifest(root: Path, slug: str) -> Dict[str, Any]:
    return load_json(dataset_dir(root, slug) / "run_manifest.json")


def detector_path(root: Path, slug: str) -> Path:
    path = Path(str(run_manifest(root, slug)["gate_detector_path"]))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def predictions_path(root: Path, slug: str) -> Path:
    return dataset_dir(root, slug) / "predictions.json"


def eval_results_path(root: Path, slug: str) -> Path:
    return dataset_dir(root, slug) / "eval_results.json"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _sample_indices(predictions: Sequence[Dict[str, Any]], max_points: int, seed: int) -> np.ndarray:
    if len(predictions) <= max_points:
        return np.arange(len(predictions))
    rng = np.random.default_rng(seed)
    by_kind: Dict[str, List[int]] = {kind: [] for kind in KIND_ORDER}
    for idx, prediction in enumerate(predictions):
        by_kind[sample_kind(prediction)].append(idx)
    selected = []
    per_kind = max(1, max_points // max(1, sum(bool(v) for v in by_kind.values())))
    for indices in by_kind.values():
        if not indices:
            continue
        take = min(len(indices), per_kind)
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    if len(selected) < max_points:
        remaining = np.setdiff1d(np.arange(len(predictions)), np.asarray(selected), assume_unique=False)
        take = min(len(remaining), max_points - len(selected))
        selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    return np.asarray(sorted(selected[:max_points]), dtype=int)


def _encode_texts(texts: Sequence[str], encoder_path: Path, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(encoder_path))
    return np.asarray(
        model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def _project_2d(matrix: np.ndarray, seed: int, require_umap: bool = False) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    try:
        os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
        from umap import UMAP

        reducer = UMAP(**UMAP_PARAMS, random_state=seed)
        params = dict(UMAP_PARAMS)
        params["random_state"] = seed
        return np.asarray(reducer.fit_transform(matrix), dtype=np.float32), "UMAP", params
    except Exception:
        if require_umap:
            raise
        reducer = PCA(n_components=2, random_state=seed)
        return (
            np.asarray(reducer.fit_transform(matrix), dtype=np.float32),
            "PCA fallback",
            {"n_components": 2, "random_state": seed},
        )


def _filter_clean_clinc_predictions(
    predictions: Sequence[Dict[str, Any]],
    selected_intents: Sequence[str],
    max_oos_per_kind: int,
    seed: int,
) -> List[Dict[str, Any]]:
    selected = set(selected_intents)
    known = [row for row in predictions if not bool(row.get("is_oos", False)) and str(row["true_intent"]) in selected]
    heldout = [row for row in predictions if sample_kind(row) == "heldout_unknown"]
    native_oos = [row for row in predictions if sample_kind(row) == "native_oos"]
    rng = np.random.default_rng(seed)

    def sample_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(rows) <= max_oos_per_kind:
            return rows
        indices = rng.choice(np.arange(len(rows)), size=max_oos_per_kind, replace=False)
        return [rows[int(idx)] for idx in sorted(indices)]

    return known + sample_rows(heldout) + sample_rows(native_oos)


def _selected_centers(centers: DetectorCenters, selected_intents: Sequence[str]) -> DetectorCenters:
    selected = set(selected_intents)
    indices = [idx for idx, label in enumerate(centers.labels) if label in selected]
    return DetectorCenters(
        centers=centers.centers[indices],
        radii=centers.radii[indices],
        labels=[centers.labels[idx] for idx in indices],
    )


def _write_space_metrics(output_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    json_path = output_dir / "clinc150_space_metrics.json"
    csv_path = output_dir / "clinc150_space_metrics.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return {"json": str(json_path), "csv": str(csv_path)}


def _with_suffix_path(path: Path, suffix: str, extension: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{extension}")


def _save_figure_atomic(fig: plt.Figure, output_path: Path, dpi: int = 300) -> Dict[str, str]:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_png = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    original_dpi = fig.dpi
    fig.set_dpi(dpi)
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).save(tmp_png)
    fig.set_dpi(original_dpi)
    tmp_png.replace(output_path)
    return {"png": str(output_path)}


def _set_shared_embedding_limits(axes: Sequence[plt.Axes], xy: np.ndarray) -> None:
    mins = np.nanmin(xy, axis=0)
    maxs = np.nanmax(xy, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pad = spans * 0.055
    for ax in axes:
        ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
        ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
        ax.locator_params(axis="both", nbins=4)
        ax.grid(False)


def _embedding_legend_handles(include_decision: bool = False) -> List[Line2D]:
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7.0, markerfacecolor=KIND_COLORS["known"],
               markeredgecolor="none", alpha=0.70, label=UMAP_LEGEND_LABELS["known"]),
        Line2D([0], [0], marker="*", linestyle="None", markersize=12.0, markerfacecolor="black",
               markeredgecolor="black", label=UMAP_LEGEND_LABELS["centroids"]),
        Line2D([0], [0], marker="x", linestyle="None", markersize=9.0, markeredgecolor=KIND_COLORS["heldout_unknown"],
               markerfacecolor="none", label=UMAP_LEGEND_LABELS["heldout_unknown"]),
        Line2D([0], [0], marker="x", linestyle="None", markersize=9.0, markeredgecolor=KIND_COLORS["native_oos"],
               markerfacecolor="none", label=UMAP_LEGEND_LABELS["native_oos"]),
    ]
    if include_decision:
        handles.extend(
            [
                Line2D([0], [0], marker="o", linestyle="None", markersize=8.0, markerfacecolor="none",
                       markeredgecolor="black", markeredgewidth=1.1, label="Known false rejected"),
                Line2D([0], [0], marker="s", linestyle="None", markersize=7.0, markerfacecolor="none",
                       markeredgecolor="black", markeredgewidth=1.1, label="Unknown/OOS false accepted"),
            ]
        )
    return handles


def _draw_clean_umap_panels(
    output_path: Path,
    sample_xy: np.ndarray,
    center_xy: np.ndarray,
    kinds: Sequence[str],
    method: str,
    include_decision: bool = False,
    final_gate_decisions: Sequence[str] | None = None,
    show_titles: bool = False,
) -> Dict[str, Any]:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharex=True, sharey=True)
    panels = [
        {"known"},
        {"known", "heldout_unknown"},
        {"known", "native_oos"},
    ]
    kind_array = np.asarray(kinds)
    for ax, visible_kinds, title in zip(axes, panels, UMAP_PANEL_TITLES):
        for kind in KIND_ORDER:
            if kind not in visible_kinds:
                continue
            mask = kind_array == kind
            if not mask.any():
                continue
            ax.scatter(
                sample_xy[mask, 0],
                sample_xy[mask, 1],
                s=13 if kind == "known" else 18,
                marker="o" if kind == "known" else "x",
                c=KIND_COLORS[kind],
                alpha=0.70 if kind == "known" else 0.78,
                linewidths=0.55,
            )
        ax.scatter(center_xy[:, 0], center_xy[:, 1], c="black", marker="*", s=52, linewidths=0.35)
        if include_decision and final_gate_decisions is not None:
            decision_array = np.asarray(final_gate_decisions)
            visible_mask = np.asarray([kind in visible_kinds for kind in kind_array])
            known_false_reject = visible_mask & (kind_array == "known") & (decision_array == "oos")
            oos_false_accept = visible_mask & (kind_array != "known") & (decision_array != "oos")
            if known_false_reject.any():
                ax.scatter(
                    sample_xy[known_false_reject, 0],
                    sample_xy[known_false_reject, 1],
                    s=58,
                    marker="o",
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=5,
                )
            if oos_false_accept.any():
                ax.scatter(
                    sample_xy[oos_false_accept, 0],
                    sample_xy[oos_false_accept, 1],
                    s=52,
                    marker="s",
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=5,
                )
        if show_titles:
            ax.set_title(title)
        ax.set_xlabel("UMAP dim1")
        ax.tick_params(length=2.5, width=0.6)
    axes[0].set_ylabel("UMAP dim2")
    _set_shared_embedding_limits(axes, np.vstack([sample_xy, center_xy]))
    fig.legend(
        handles=_embedding_legend_handles(include_decision=include_decision),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=6 if include_decision else 4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.1,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    paths = _save_figure_atomic(fig, output_path)
    plt.close(fig)
    return paths


def _draw_clean_umap_single_panel(
    output_path: Path,
    sample_xy: np.ndarray,
    center_xy: np.ndarray,
    kinds: Sequence[str],
    method: str,
) -> Dict[str, Any]:
    previous_rc = dict(plt.rcParams)
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "legend.fontsize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    fig, ax = plt.subplots(figsize=(18.5, 6.8))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.145, top=0.80)
    kind_array = np.asarray(kinds)

    known_mask = kind_array == "known"
    if known_mask.any():
        ax.scatter(
            sample_xy[known_mask, 0],
            sample_xy[known_mask, 1],
            s=24,
            marker="o",
            c=KIND_COLORS["known"],
            alpha=0.38,
            linewidths=0.0,
            label=KIND_LABELS["known"],
        )

    for kind, label in [
        ("heldout_unknown", KIND_LABELS["heldout_unknown"]),
        ("native_oos", KIND_LABELS["native_oos"]),
    ]:
        mask = kind_array == kind
        if not mask.any():
            continue
        ax.scatter(
            sample_xy[mask, 0],
            sample_xy[mask, 1],
            s=82,
            marker="x",
            c=KIND_COLORS[kind],
            alpha=0.92,
            linewidths=1.8,
            label=label,
        )

    ax.scatter(
        center_xy[:, 0],
        center_xy[:, 1],
        c="black",
        marker="*",
        s=280,
        linewidths=0.75,
        label="Known-intent centroids",
        zorder=6,
    )
    _set_shared_embedding_limits([ax], np.vstack([sample_xy, center_xy]))
    ax.grid(True, color="#D0D0D0", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel("UMAP dimension 1" if method == "UMAP" else "Projection dimension 1", labelpad=10)
    ax.set_ylabel("UMAP dimension 2" if method == "UMAP" else "Projection dimension 2", labelpad=10)
    handles, labels = ax.get_legend_handles_labels()
    order = [
        KIND_LABELS["known"],
        "Known-intent centroids",
        KIND_LABELS["heldout_unknown"],
        KIND_LABELS["native_oos"],
    ]
    by_label = {label: handle for handle, label in zip(handles, labels)}
    ax.legend(
        [by_label[label] for label in order if label in by_label],
        [label for label in order if label in by_label],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=True,
        handletextpad=0.7,
        columnspacing=1.4,
        labelspacing=0.45,
        markerscale=1.15,
        borderaxespad=0.0,
    )

    paths = _save_figure_atomic(fig, output_path)
    plt.close(fig)
    plt.rcParams.update(previous_rc)
    return paths


def _draw_clean_umap_decision_single_panel(
    output_path: Path,
    sample_xy: np.ndarray,
    center_xy: np.ndarray,
    kinds: Sequence[str],
    method: str,
    final_gate_decisions: Sequence[str],
) -> Dict[str, Any]:
    previous_rc = dict(plt.rcParams)
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "legend.fontsize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    fig, ax = plt.subplots(figsize=(18.5, 6.8))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.145, top=0.80)
    kind_array = np.asarray(kinds)
    decision_array = np.asarray(final_gate_decisions)
    accepted = decision_array != "oos"
    rejected = decision_array == "oos"

    if accepted.any():
        ax.scatter(
            sample_xy[accepted, 0],
            sample_xy[accepted, 1],
            s=26,
            marker="o",
            c=KIND_COLORS["known"],
            alpha=0.40,
            linewidths=0.0,
            label="Gate accepted (ID path)",
        )
    if rejected.any():
        ax.scatter(
            sample_xy[rejected, 0],
            sample_xy[rejected, 1],
            s=82,
            marker="x",
            c=KIND_COLORS["native_oos"],
            alpha=0.88,
            linewidths=1.8,
            label="Gate rejected (OOS)",
        )

    known_false_reject = (kind_array == "known") & rejected
    unknown_false_accept = (kind_array != "known") & accepted
    if known_false_reject.any():
        ax.scatter(
            sample_xy[known_false_reject, 0],
            sample_xy[known_false_reject, 1],
            s=96,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Known false rejected",
            zorder=7,
        )
    if unknown_false_accept.any():
        ax.scatter(
            sample_xy[unknown_false_accept, 0],
            sample_xy[unknown_false_accept, 1],
            s=88,
            marker="s",
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Unknown/OOS false accepted",
            zorder=7,
        )

    ax.scatter(
        center_xy[:, 0],
        center_xy[:, 1],
        c="black",
        marker="*",
        s=280,
        linewidths=0.75,
        label="Known-intent centroids",
        zorder=6,
    )
    _set_shared_embedding_limits([ax], np.vstack([sample_xy, center_xy]))
    ax.grid(True, color="#D0D0D0", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel("UMAP dimension 1" if method == "UMAP" else "Projection dimension 1", labelpad=10)
    ax.set_ylabel("UMAP dimension 2" if method == "UMAP" else "Projection dimension 2", labelpad=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=True,
        handletextpad=0.7,
        columnspacing=1.4,
        labelspacing=0.45,
        markerscale=1.15,
        borderaxespad=0.0,
    )

    paths = _save_figure_atomic(fig, output_path)
    plt.close(fig)
    plt.rcParams.update(previous_rc)
    return paths


def _decision_counts(sampled: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "known_false_rejected": 0,
        "heldout_unknown_false_accepted": 0,
        "native_oos_false_accepted": 0,
    }
    for row in sampled:
        kind = sample_kind(row)
        final_gate_decision = str(row.get("final_gate_decision", ""))
        if kind == "known" and final_gate_decision == "oos":
            counts["known_false_rejected"] += 1
        elif kind == "heldout_unknown" and final_gate_decision != "oos":
            counts["heldout_unknown_false_accepted"] += 1
        elif kind == "native_oos" and final_gate_decision != "oos":
            counts["native_oos_false_accepted"] += 1
    return counts


def plot_clinc_clean_umap(
    root: Path,
    output_path: Path,
    selected_intent_count: int,
    seed: int,
    batch_size: int,
    max_oos_per_kind: int,
) -> Dict[str, Any]:
    slug = "clinc150"
    predictions = load_json(predictions_path(root, slug))
    selected_intents = select_known_intents(predictions, count=selected_intent_count, seed=seed, min_support=20)
    sampled = _filter_clean_clinc_predictions(predictions, selected_intents, max_oos_per_kind, seed)
    texts = [row["text"] for row in sampled]
    kinds = [sample_kind(row) for row in sampled]

    manifest = run_manifest(root, slug)
    encoder_path = Path(str(manifest.get("gate_encoder_path", "all-MiniLM-L6-v2")))
    if not encoder_path.is_absolute():
        encoder_path = PROJECT_ROOT / encoder_path
    embeddings = _encode_texts(texts, encoder_path=encoder_path, batch_size=batch_size)
    all_centers = load_detector_centers(detector_path(root, slug))
    centers = _selected_centers(all_centers, selected_intents)
    if len(centers.labels) == 0:
        raise ValueError(f"No detector centers matched selected intents: {selected_intents}")

    all_points = np.vstack([embeddings, centers.centers])
    projected, method, projection_params = _project_2d(all_points, seed=seed, require_umap=True)
    sample_xy = projected[: len(sampled)]
    center_xy = projected[len(sampled) :]

    legacy_png = _with_suffix_path(output_path, "_legacy_3panel", ".png")
    single_panel_png = _with_suffix_path(output_path, "_singlepanel", ".png")
    paper_png = _with_suffix_path(output_path, "_paper", ".png")
    legacy_paths = _draw_clean_umap_panels(legacy_png, sample_xy, center_xy, kinds, method)
    readable_paths = _draw_clean_umap_panels(output_path, sample_xy, center_xy, kinds, method)
    single_panel_paths = _draw_clean_umap_single_panel(single_panel_png, sample_xy, center_xy, kinds, method)
    paper_paths = _draw_clean_umap_panels(paper_png, sample_xy, center_xy, kinds, method)

    final_gate_decisions = [str(row.get("final_gate_decision", "")) for row in sampled]
    can_draw_decision = all(value != "" for value in final_gate_decisions)
    decision_counts = _decision_counts(sampled) if can_draw_decision else {}
    decision_paths: Dict[str, str] = {}
    if can_draw_decision:
        decision_png = _with_suffix_path(output_path, "_decision", ".png")
        decision_paths = _draw_clean_umap_decision_single_panel(
            decision_png,
            sample_xy,
            center_xy,
            kinds,
            method,
            final_gate_decisions=final_gate_decisions,
        )

    metric_rows = compute_space_metrics(embeddings, kinds, centers)
    metric_paths = _write_space_metrics(output_path.parent, metric_rows)
    return {
        "figure": str(output_path),
        "legacy_outputs": legacy_paths,
        "readable_outputs": readable_paths,
        "single_panel_outputs": single_panel_paths,
        "paper_outputs": paper_paths,
        "decision_outputs": decision_paths,
        "projection": method,
        "projection_parameters": projection_params,
        "source_data_path": str(predictions_path(root, slug)),
        "selected_known_intents": selected_intents,
        "known_intent_count": len(selected_intents),
        "sample_count": len(sampled),
        "known_intent_center_count": len(centers.labels),
        "space_metrics": metric_paths,
        "boundaries_shown": False,
        "boundary_note": (
            "Local high-dimensional gate radii are not drawn in the UMAP plane because UMAP is nonlinear; "
            "drawing radius circles after projection would be visually suggestive but not geometrically valid."
        ),
        "decision_labels_shown": bool(decision_paths),
        "decision_counts": decision_counts,
        "decision_label_note": (
            "Decision view recolors all focused samples by final_gate_decision and outlines false decisions when present."
            if decision_paths
            else "Decision view was not generated because final_gate_decision was unavailable."
        ),
        "latex_caption": (
            "CLINC150 gate-embedding UMAP visualization. The panels show known samples, known-intent "
            "centroids, held-out unknown samples, and dataset-provided OOS samples under the same "
            "two-dimensional projection."
        ),
        "note": QUALITATIVE_NOTE,
    }


def plot_clinc_umap(root: Path, output_path: Path, max_points: int, seed: int, batch_size: int) -> Dict[str, Any]:
    slug = "clinc150"
    predictions = load_json(predictions_path(root, slug))
    indices = _sample_indices(predictions, max_points=max_points, seed=seed)
    sampled = [predictions[int(idx)] for idx in indices]
    texts = [row["text"] for row in sampled]
    kinds = [sample_kind(row) for row in sampled]
    accepted = [str(row.get("final_gate_decision")) != "oos" for row in sampled]

    manifest = run_manifest(root, slug)
    encoder_path = Path(str(manifest.get("gate_encoder_path", "all-MiniLM-L6-v2")))
    if not encoder_path.is_absolute():
        encoder_path = PROJECT_ROOT / encoder_path
    embeddings = _encode_texts(texts, encoder_path=encoder_path, batch_size=batch_size)
    centers = load_detector_centers(detector_path(root, slug))
    all_points = np.vstack([embeddings, centers.centers])
    projected, method, _ = _project_2d(all_points, seed=seed)
    sample_xy = projected[: len(sampled)]
    center_xy = projected[len(sampled) :]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for kind in KIND_ORDER:
        mask = np.asarray([item == kind for item in kinds])
        if not mask.any():
            continue
        for is_accepted, marker, label_suffix in [(True, "o", "accepted"), (False, "x", "rejected")]:
            style_mask = mask & np.asarray([value == is_accepted for value in accepted])
            if not style_mask.any():
                continue
            ax.scatter(
                sample_xy[style_mask, 0],
                sample_xy[style_mask, 1],
                s=13 if is_accepted else 18,
                marker=marker,
                alpha=0.55,
                linewidths=0.5,
                c=KIND_COLORS[kind],
                label=f"{KIND_LABELS[kind]} ({label_suffix})",
            )
    ax.scatter(center_xy[:, 0], center_xy[:, 1], c="black", marker="*", s=40, linewidths=0.4, label="Known-intent centroids")
    ax.set_xlabel("Projection dim. 1")
    ax.set_ylabel("Projection dim. 2")
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return {
        "figure": str(output_path),
        "projection": method,
        "sample_count": len(sampled),
        "known_intent_center_count": len(centers.labels),
    }


def _hist_density(ax: plt.Axes, scores: Sequence[float], label: str, color: str, bins: np.ndarray) -> None:
    if not scores:
        return
    density, edges = np.histogram(np.asarray(scores, dtype=float), bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.plot(centers, density, color=color, linewidth=1.8, label=label)
    ax.fill_between(centers, density, color=color, alpha=0.12)


def plot_gate_score_distribution(root: Path, slug: str, title: str, output_path: Path) -> Dict[str, Any]:
    rows = load_gate_score_rows(predictions_path(root, slug))
    by_kind = {kind: [row["score"] for row in rows if row["kind"] == kind] for kind in KIND_ORDER}
    all_scores = [row["score"] for row in rows]
    upper = min(max(all_scores) * 1.02, np.percentile(all_scores, 99.5) * 1.2)
    bins = np.linspace(0.0, max(1.8, float(upper)), 80)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for kind in KIND_ORDER:
        _hist_density(ax, by_kind[kind], KIND_LABELS[kind], KIND_COLORS[kind], bins)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="Threshold")
    ax.set_xlabel("Gate score")
    ax.set_ylabel("Density")
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=9,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0.3,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return {"figure": str(output_path), "counts": {kind: len(values) for kind, values in by_kind.items()}}


def plot_multi_dataset_gate_comparison(root: Path, output_path: Path) -> Dict[str, Any]:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), sharex=True, sharey=True)
    summaries = {}
    max_score = 0.0
    dataset_rows = {}
    for name, slug in DATASET_SLUGS.items():
        rows = load_gate_score_rows(predictions_path(root, slug))
        dataset_rows[name] = rows
        max_score = max(max_score, float(np.percentile([row["score"] for row in rows], 99.0)))
    bins = np.linspace(0.0, max(1.8, max_score * 1.15), 70)

    for ax, (name, rows) in zip(axes, dataset_rows.items()):
        known = [row["score"] for row in rows if row["kind"] == "known"]
        oos = [row["score"] for row in rows if row["kind"] != "known"]
        _hist_density(ax, known, "Known", KIND_COLORS["known"], bins)
        _hist_density(ax, oos, "Unknown / OOS samples", KIND_COLORS["heldout_unknown"], bins)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("distance / radius")
        summaries[name] = {"known": len(known), "oos": len(oos)}
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return {"figure": str(output_path), "counts": summaries}


def plot_error_breakdown(root: Path, output_path: Path) -> Dict[str, Any]:
    rows = [error_breakdown_row(name, eval_results_path(root, slug)) for name, slug in DATASET_SLUGS.items()]
    labels = [row["dataset"] for row in rows]
    series = [
        ("Gate false reject", "gate_false_reject", "#4E79A7"),
        ("Router wrong dispatch", "router_wrong_dispatch", "#F28E2B"),
        ("Expert wrong classification", "expert_wrong_classification", "#59A14F"),
        ("OOS false accept", "oos_false_accept", "#E15759"),
    ]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for label, key, color in series:
        values = np.asarray([row[key] for row in rows], dtype=float)
        ax.bar(x, values, bottom=bottom, label=label, color=color, width=0.62)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error count")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return {"figure": str(output_path), "rows": rows}


def plot_clinc_center_region(root: Path, output_path: Path, max_points: int, seed: int, batch_size: int) -> Dict[str, Any]:
    slug = "clinc150"
    predictions = load_json(predictions_path(root, slug))
    indices = _sample_indices(predictions, max_points=max_points, seed=seed)
    sampled = [predictions[int(idx)] for idx in indices]
    texts = [row["text"] for row in sampled]
    kinds = [sample_kind(row) for row in sampled]
    scores = np.asarray([prediction_score(row) for row in sampled], dtype=float)
    manifest = run_manifest(root, slug)
    encoder_path = Path(str(manifest.get("gate_encoder_path", "all-MiniLM-L6-v2")))
    if not encoder_path.is_absolute():
        encoder_path = PROJECT_ROOT / encoder_path
    embeddings = _encode_texts(texts, encoder_path=encoder_path, batch_size=batch_size)
    centers = load_detector_centers(detector_path(root, slug))
    all_points = np.vstack([embeddings, centers.centers])
    projected, method, _ = _project_2d(all_points, seed=seed)
    sample_xy = projected[: len(sampled)]
    center_xy = projected[len(sampled) :]

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    accepted_mask = scores <= 1.0
    for kind in KIND_ORDER:
        kind_mask = np.asarray([item == kind for item in kinds])
        if not kind_mask.any():
            continue
        ax.scatter(
            sample_xy[kind_mask & accepted_mask, 0],
            sample_xy[kind_mask & accepted_mask, 1],
            s=13,
            alpha=0.45,
            c=KIND_COLORS[kind],
            marker="o",
            label=f"{KIND_LABELS[kind]} inside",
        )
        ax.scatter(
            sample_xy[kind_mask & ~accepted_mask, 0],
            sample_xy[kind_mask & ~accepted_mask, 1],
            s=18,
            alpha=0.62,
            c=KIND_COLORS[kind],
            marker="x",
            label=f"{KIND_LABELS[kind]} outside",
        )
    ax.scatter(center_xy[:, 0], center_xy[:, 1], c="black", marker="*", s=42, label="Known-intent centroids")
    ax.set_xlabel("Projection dim. 1")
    ax.set_ylabel("Projection dim. 2")
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return {"figure": str(output_path), "projection": method, "sample_count": len(sampled)}


def _paper_manifest_keys(value: Any) -> Any:
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            clean_key = str(key).replace("native_oos", "provided_oos_samples")
            clean[clean_key] = _paper_manifest_keys(item)
        return clean
    if isinstance(value, list):
        return [_paper_manifest_keys(item) for item in value]
    return value


def export_figures(root: Path, output_dir: Path, max_points: int, seed: int, batch_size: int) -> Dict[str, Any]:
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_root": str(root),
        "figures": {
            "clinc150_clean_umap_3panel": plot_clinc_clean_umap(
                root,
                output_dir / "clinc150_clean_umap_3panel.png",
                selected_intent_count=8,
                seed=seed,
                batch_size=batch_size,
                max_oos_per_kind=260,
            ),
            "clinc150_umap_gate_space": plot_clinc_umap(
                root, output_dir / "clinc150_umap_gate_space.png", max_points, seed, batch_size
            ),
            "clinc150_gate_score_distribution": plot_gate_score_distribution(
                root, "clinc150", "CLINC150 KIR50 Gate Score Distribution", output_dir / "clinc150_gate_score_distribution.png"
            ),
            "multi_dataset_gate_comparison": plot_multi_dataset_gate_comparison(
                root, output_dir / "multi_dataset_gate_comparison.png"
            ),
            "pipeline_error_breakdown": plot_error_breakdown(root, output_dir / "pipeline_error_breakdown.png"),
            "clinc150_center_region": plot_clinc_center_region(
                root, output_dir / "clinc150_center_region.png", max_points, seed, batch_size
            ),
        },
    }
    manifest = _paper_manifest_keys(manifest)
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper visualization figures for v19")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "figures" / "paper_v19"))
    parser.add_argument("--max_points", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    manifest = export_figures(
        root=Path(args.root),
        output_dir=Path(args.output_dir),
        max_points=args.max_points,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(json.dumps({"figures": manifest["figures"], "output_dir": args.output_dir}, ensure_ascii=False))


if __name__ == "__main__":
    main()
