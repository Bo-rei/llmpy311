#!/usr/bin/env python3
"""Generate single-panel UMAP figure with ALL known intents."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "pipeline"
    / "ablations"
    / "latest_strongest_v19"
    / "paper_mainline_proto_kir50_20260427"
)
OUTPUT_DIR = PROJECT_ROOT / "figures" / "paper_v19"

KIND_COLORS = {
    "known": "#4C78A8",
    "heldout_unknown": "#F58518",
    "native_oos": "#C44E52",
}
KIND_LABELS = {
    "known": "Known samples",
    "heldout_unknown": "Unknown samples",
    "native_oos": "OOS samples",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_kind(prediction: dict) -> str:
    if not bool(prediction.get("is_oos", False)):
        return "known"
    if str(prediction.get("true_intent")) == "oos":
        return "native_oos"
    return "heldout_unknown"


def _set_shared_embedding_limits(ax, points: np.ndarray):
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_pad = (x_max - x_min) * 0.03
    y_pad = (y_max - y_min) * 0.03
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def main():
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from collections import Counter

    slug = "clinc150"
    dataset_dir = ROOT / slug / "kir50_seed42" / "full_anchor"
    predictions = load_json(dataset_dir / "predictions.json")
    run_manifest = load_json(dataset_dir / "run_manifest.json")

    # Load detector centers
    det_path = Path(str(run_manifest["gate_detector_path"]))
    if not det_path.is_absolute():
        det_path = ROOT / det_path
    detector = load_json(det_path)
    spheres = detector.get("spheres", [])
    centers = np.asarray([s["center"] for s in spheres], dtype=np.float64)
    center_intents = [s.get("intent_name", "") for s in spheres]

    # Use ALL known intents (not just 8)
    support = Counter(str(p["true_intent"]) for p in predictions if not bool(p.get("is_oos", False)))
    selected = sorted(intent for intent, size in support.items() if size >= 5)
    selected_set = set(selected)
    print(f"Using {len(selected)} known intents")

    # Filter predictions - use all known intents
    known = [p for p in predictions if not bool(p.get("is_oos", False)) and str(p.get("true_intent", "")) in selected_set]
    oos_heldout = [p for p in predictions if sample_kind(p) == "heldout_unknown"]
    oos_native = [p for p in predictions if sample_kind(p) == "native_oos"]

    # Subsample to keep balanced
    rng = np.random.default_rng(20260427)
    max_per_kind = 400
    if len(known) > max_per_kind:
        known = list(rng.choice(known, size=max_per_kind, replace=False))
    if len(oos_heldout) > max_per_kind:
        oos_heldout = list(rng.choice(oos_heldout, size=max_per_kind, replace=False))
    if len(oos_native) > max_per_kind:
        oos_native = list(rng.choice(oos_native, size=max_per_kind, replace=False))

    sampled = list(known) + list(oos_heldout) + list(oos_native)
    texts = [p["text"] for p in sampled]
    kinds = [sample_kind(p) for p in sampled]
    print(f"Total samples: {len(sampled)} (known={len(known)}, unknown={len(oos_heldout)}, oos={len(oos_native)})")

    # Encode
    encoder_path = Path(str(run_manifest.get("gate_encoder_path", "all-MiniLM-L6-v2")))
    if not encoder_path.is_absolute():
        encoder_path = PROJECT_ROOT / encoder_path
    print(f"Loading encoder from {encoder_path}")
    encoder = SentenceTransformer(str(encoder_path))
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    print(f"Encoded {len(embeddings)} samples")

    # Select centers for all selected intents
    selected_centers_mask = [c in selected_set for c in center_intents]
    selected_centers = centers[selected_centers_mask]
    print(f"Selected {len(selected_centers)} centers")

    # UMAP
    all_points = np.vstack([embeddings, selected_centers])
    reducer = UMAP(n_components=2, n_neighbors=80, min_dist=0.02, metric="cosine", random_state=20260427)
    projected = reducer.fit_transform(all_points)
    sample_xy = projected[:len(embeddings)]
    center_xy = projected[len(embeddings):]
    print("UMAP done")

    # Plot
    kind_array = np.asarray(kinds)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    # Known samples
    mask = kind_array == "known"
    ax.scatter(
        sample_xy[mask, 0], sample_xy[mask, 1],
        s=50, marker="o", c=KIND_COLORS["known"],
        alpha=0.80, linewidths=0.0, label=KIND_LABELS["known"],
    )

    # Unknown samples
    mask = kind_array == "heldout_unknown"
    ax.scatter(
        sample_xy[mask, 0], sample_xy[mask, 1],
        s=45, marker="x", c=KIND_COLORS["heldout_unknown"],
        alpha=0.90, linewidths=1.6, label=KIND_LABELS["heldout_unknown"],
    )

    # OOS samples
    mask = kind_array == "native_oos"
    ax.scatter(
        sample_xy[mask, 0], sample_xy[mask, 1],
        s=45, marker="x", c=KIND_COLORS["native_oos"],
        alpha=0.90, linewidths=1.6, label=KIND_LABELS["native_oos"],
    )

    # Centroids
    ax.scatter(
        center_xy[:, 0], center_xy[:, 1],
        c="black", marker="*", s=280, linewidths=0.6, edgecolors="white", label="Centroids", zorder=5,
    )

    _set_shared_embedding_limits(ax, np.vstack([sample_xy, center_xy]))

    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.tick_params(length=2.5, width=0.6)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Legend below figure
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5, markeredgecolor=KIND_COLORS["known"], markerfacecolor=KIND_COLORS["known"], label=KIND_LABELS["known"]),
        Line2D([0], [0], marker="*", linestyle="None", markersize=7, markeredgecolor="black", markerfacecolor="black", label="Centroids"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=6, markeredgecolor=KIND_COLORS["heldout_unknown"], markerfacecolor="none", label=KIND_LABELS["heldout_unknown"]),
        Line2D([0], [0], marker="x", linestyle="None", markersize=6, markeredgecolor=KIND_COLORS["native_oos"], markerfacecolor="none", label=KIND_LABELS["native_oos"]),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.1,
    )

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    # Save to NEW filename (don't overwrite existing)
    png_path = OUTPUT_DIR / "gate_embedding_umap_all_intents.png"
    eps_path = OUTPUT_DIR / "gate_embedding_umap_all_intents.eps"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(eps_path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {eps_path}")


if __name__ == "__main__":
    main()
