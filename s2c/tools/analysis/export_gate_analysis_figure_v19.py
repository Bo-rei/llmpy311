#!/usr/bin/env python3
"""Export the ACL-style gate analysis figure from saved v19 artifacts.

This script does not train or alter models. It reads the frozen CLINC150 KIR50
pipeline predictions and detector artifact, uses saved normalized gate scores
when present, and only recomputes MiniLM embeddings for visualization.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "pipeline"
    / "frozen_prototype_gate"
    / "prototype_gate_frozen_2026-04-09"
    / "prototype_gate_pipeline_frozen"
)
FIGURE_DIR = PROJECT_ROOT / "figures"
SEED = 20260317
DATASET = "CLINC150"
KIR = "0.50"

KIND_ORDER = ["known", "heldout_unknown", "oos_intent"]
KIND_LABELS = {
    "known": "Known",
    "heldout_unknown": "Held-out unknown",
    "oos_intent": "OOS intent",
}
KIND_COLORS = {
    "known": "#4C78A8",
    "heldout_unknown": "#F58518",
    "oos_intent": "#C44E52",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sample_kind(row: Dict[str, Any]) -> str:
    if not bool(row.get("is_oos", False)):
        return "known"
    if str(row.get("true_intent")) == "oos":
        return "oos_intent"
    return "heldout_unknown"


def normalized_score(row: Dict[str, Any]) -> Tuple[float, str]:
    if row.get("gate_score") is not None:
        return float(row["gate_score"]), "saved_gate_score"
    distance = float(row["gate_distance"])
    radius = float(row["gate_radius"])
    return distance / max(radius, 1e-12), "gate_distance_div_gate_radius"


def detector_path(run_dir: Path) -> Path:
    manifest = load_json(run_dir / "run_manifest.json")
    path = Path(str(manifest["gate_detector_path"]))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def gate_encoder_path(run_dir: Path) -> Path:
    eval_results = load_json(run_dir / "eval_results.json")
    path = Path(str(eval_results.get("config", {}).get("gate_encoder_path", "all-MiniLM-L6-v2")))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_detector_centers(path: Path) -> Tuple[np.ndarray, List[str]]:
    detector = load_json(path)
    spheres = detector.get("spheres", [])
    if not spheres:
        raise ValueError(f"No spheres found in detector: {path}")
    centers = np.asarray([sphere["center"] for sphere in spheres], dtype=np.float32)
    labels = [str(sphere.get("intent_name", sphere.get("cluster_id", idx))) for idx, sphere in enumerate(spheres)]
    return centers, labels


def select_known_intents(rows: Sequence[Dict[str, Any]], count: int) -> List[str]:
    support = Counter(str(row["true_intent"]) for row in rows if sample_kind(row) == "known")
    candidates = sorted(intent for intent, size in support.items() if size >= 20)
    rng = np.random.default_rng(SEED)
    if len(candidates) <= count:
        return candidates
    return sorted(rng.choice(candidates, size=count, replace=False).tolist())


def sample_for_embedding(rows: Sequence[Dict[str, Any]], selected_intents: Sequence[str]) -> List[int]:
    selected = set(selected_intents)
    by_kind: Dict[str, List[int]] = {kind: [] for kind in KIND_ORDER}
    for idx, row in enumerate(rows):
        kind = sample_kind(row)
        if kind == "known" and str(row["true_intent"]) not in selected:
            continue
        by_kind[kind].append(idx)

    rng = np.random.default_rng(SEED)
    sampled: List[int] = []
    limits = {"known": 320, "heldout_unknown": 260, "oos_intent": 260}
    for kind in KIND_ORDER:
        indices = by_kind[kind]
        if len(indices) <= limits[kind]:
            sampled.extend(indices)
            continue
        chosen = rng.choice(np.asarray(indices), size=limits[kind], replace=False)
        sampled.extend(int(i) for i in chosen.tolist())
    return sorted(sampled)


def encode_texts(texts: Sequence[str], encoder_path: Path) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(encoder_path))
    embeddings = model.encode(
        list(texts),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def project_tsne(matrix: np.ndarray) -> np.ndarray:
    perplexity = min(30, max(5, (len(matrix) - 1) // 4))
    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=SEED,
        metric="euclidean",
    )
    return np.asarray(reducer.fit_transform(matrix), dtype=np.float32)


def plot_embedding_panel(
    ax: plt.Axes,
    sample_xy: np.ndarray,
    center_xy: np.ndarray,
    kinds: Sequence[str],
) -> None:
    kind_array = np.asarray(kinds)
    for kind in KIND_ORDER:
        mask = kind_array == kind
        if not mask.any():
            continue
        ax.scatter(
            sample_xy[mask, 0],
            sample_xy[mask, 1],
            s=16 if kind == "known" else 28,
            marker="o" if kind == "known" else "x",
            c=KIND_COLORS[kind],
            alpha=0.62 if kind == "known" else 0.86,
            linewidths=0.9 if kind != "known" else 0.0,
        )
    ax.scatter(
        center_xy[:, 0],
        center_xy[:, 1],
        c="#111111",
        marker="*",
        s=88,
        linewidths=0.4,
        zorder=6,
    )
    ax.set_title("(a) Gate embedding visualization", loc="left", fontweight="bold")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(False)
    ax.tick_params(length=2.5, width=0.7)


def density_curve(scores: Sequence[float], bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    density, edges = np.histogram(np.asarray(scores, dtype=np.float64), bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, density


def plot_score_panel(ax: plt.Axes, rows: Sequence[Dict[str, Any]]) -> None:
    scores_by_kind: Dict[str, List[float]] = {kind: [] for kind in KIND_ORDER}
    all_scores: List[float] = []
    for row in rows:
        score, _ = normalized_score(row)
        scores_by_kind[sample_kind(row)].append(score)
        all_scores.append(score)

    upper = max(1.6, float(np.percentile(np.asarray(all_scores), 99.2)) * 1.08)
    bins = np.linspace(0.0, upper, 74)
    for kind in KIND_ORDER:
        x, y = density_curve(scores_by_kind[kind], bins)
        ax.plot(x, y, color=KIND_COLORS[kind], linewidth=2.1, label=KIND_LABELS[kind])
        ax.fill_between(x, y, color=KIND_COLORS[kind], alpha=0.13)
    ax.axvline(1.0, color="#111111", linestyle="--", linewidth=1.6, label="Threshold")
    ax.set_title("(b) Normalized boundary score distribution", loc="left", fontweight="bold")
    ax.set_xlabel("Normalized boundary score $s(e)$")
    ax.set_ylabel("Density")
    ax.grid(False)
    ax.tick_params(length=2.5, width=0.7)


def write_plot_data(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    sampled_indices: Sequence[int],
    sample_xy: np.ndarray,
) -> None:
    xy_by_index = {int(idx): sample_xy[pos] for pos, idx in enumerate(sampled_indices)}
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "kir",
        "sample_id",
        "sample_type",
        "text",
        "true_intent",
        "true_domain",
        "is_oos",
        "gate_pred",
        "final_gate_decision",
        "normalized_boundary_score",
        "score_source",
        "gate_distance",
        "gate_radius",
        "plotted_in_embedding_panel",
        "tsne_x",
        "tsne_y",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            score, source = normalized_score(row)
            xy = xy_by_index.get(idx)
            writer.writerow(
                {
                    "dataset": DATASET,
                    "kir": KIR,
                    "sample_id": idx,
                    "sample_type": sample_kind(row),
                    "text": row.get("text", ""),
                    "true_intent": row.get("true_intent", ""),
                    "true_domain": row.get("true_domain", ""),
                    "is_oos": bool(row.get("is_oos", False)),
                    "gate_pred": row.get("gate_pred", ""),
                    "final_gate_decision": row.get("final_gate_decision", ""),
                    "normalized_boundary_score": f"{score:.10f}",
                    "score_source": source,
                    "gate_distance": row.get("gate_distance", ""),
                    "gate_radius": row.get("gate_radius", ""),
                    "plotted_in_embedding_panel": xy is not None,
                    "tsne_x": "" if xy is None else f"{float(xy[0]):.8f}",
                    "tsne_y": "" if xy is None else f"{float(xy[1]):.8f}",
                }
            )


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for kind in KIND_ORDER:
        values = np.asarray([normalized_score(row)[0] for row in rows if sample_kind(row) == kind], dtype=np.float64)
        summary[kind] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "pct_le_1": float((values <= 1.0).mean() * 100.0),
        }
    return summary


def main() -> None:
    run_dir = DEFAULT_RUN_DIR
    predictions_path = run_dir / "predictions.json"
    rows = load_json(predictions_path)

    centers, center_labels = load_detector_centers(detector_path(run_dir))
    selected_intents = select_known_intents(rows, count=10)
    center_mask = np.asarray([label in set(selected_intents) for label in center_labels], dtype=bool)
    selected_centers = centers[center_mask]
    if selected_centers.size == 0:
        raise ValueError("No detector centers matched the selected known intents.")

    sampled_indices = sample_for_embedding(rows, selected_intents)
    sampled_rows = [rows[idx] for idx in sampled_indices]
    embeddings = encode_texts([str(row["text"]) for row in sampled_rows], gate_encoder_path(run_dir))
    projection = project_tsne(np.vstack([embeddings, selected_centers]))
    sample_xy = projection[: len(sampled_rows)]
    center_xy = projection[len(sampled_rows) :]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    plot_embedding_panel(axes[0], sample_xy, center_xy, [sample_kind(row) for row in sampled_rows])
    plot_score_panel(axes[1], rows)
    fig.suptitle("CLINC150 KIR=0.50 multi-centroid boundary gate", y=1.035, fontsize=12.5)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6, markerfacecolor=KIND_COLORS["known"], markeredgecolor="none", label="Known"),
        Line2D([0], [0], marker="*", linestyle="None", markersize=9, markerfacecolor="#111111", markeredgecolor="#111111", label="Known-intent centroids"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=7, markeredgecolor=KIND_COLORS["heldout_unknown"], markerfacecolor="none", label="Held-out unknown"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=7, markeredgecolor=KIND_COLORS["oos_intent"], markerfacecolor="none", label="OOS intent"),
        Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.5, label="Threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.075), ncol=5, frameon=False, columnspacing=1.0, handletextpad=0.45)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0), w_pad=2.2)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / "gate_analysis.pdf"
    png_path = FIGURE_DIR / "gate_analysis.png"
    csv_path = FIGURE_DIR / "gate_score_distribution.csv"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_plot_data(csv_path, rows, sampled_indices, sample_xy)

    metadata = {
        "dataset": DATASET,
        "kir": KIR,
        "seed": SEED,
        "run_dir": str(run_dir),
        "predictions_path": str(predictions_path),
        "detector_path": str(detector_path(run_dir)),
        "gate_encoder_path": str(gate_encoder_path(run_dir)),
        "selected_known_intents": selected_intents,
        "embedding_sample_count": len(sampled_rows),
        "selected_centroid_count": int(selected_centers.shape[0]),
        "projection": "t-SNE",
        "figure_pdf": str(pdf_path),
        "figure_png": str(png_path),
        "plot_data_csv": str(csv_path),
        "summary": summarize(rows),
    }
    (FIGURE_DIR / "gate_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
