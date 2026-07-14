#!/usr/bin/env python3
"""Generate single-panel UMAP figure from existing 3-panel data."""

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
    "known": "#0000FC",
    "heldout_unknown": "#4F5D75",
    "native_oos": "#C0392B",
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

    # Select 8 known intents
    support = Counter(str(p["true_intent"]) for p in predictions if not bool(p.get("is_oos", False)))
    candidates = sorted(intent for intent, size in support.items() if size >= 20)
    rng = np.random.default_rng(20260427)
    selected = sorted(rng.choice(candidates, size=8, replace=False).tolist())
    selected_set = set(selected)

    # Filter and subsample
    sampled = []
    for p in predictions:
        true_intent = str(p.get("true_intent", ""))
        is_oos = bool(p.get("is_oos", False))
        if not is_oos and true_intent not in selected_set:
            continue
        sampled.append(p)

    max_oos = 260
    oos_heldout = [p for p in sampled if sample_kind(p) == "heldout_unknown"]
    oos_native = [p for p in sampled if sample_kind(p) == "native_oos"]
    known = [p for p in sampled if sample_kind(p) == "known"]

    if len(oos_heldout) > max_oos:
        oos_heldout = list(rng.choice(oos_heldout, size=max_oos, replace=False))
    if len(oos_native) > max_oos:
        oos_native = list(rng.choice(oos_native, size=max_oos, replace=False))

    sampled = list(known) + list(oos_heldout) + list(oos_native)
    texts = [p["text"] for p in sampled]
    kinds = [sample_kind(p) for p in sampled]

    # Encode
    encoder_path = Path(str(run_manifest.get("gate_encoder_path", "all-MiniLM-L6-v2")))
    if not encoder_path.is_absolute():
        encoder_path = PROJECT_ROOT / encoder_path
    print(f"Loading encoder from {encoder_path}")
    encoder = SentenceTransformer(str(encoder_path))
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    print(f"Encoded {len(embeddings)} samples")

    # Select centers
    selected_centers_mask = [c in selected_set for c in center_intents]
    selected_centers = centers[selected_centers_mask]

    # UMAP - tuned for better cluster separation
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
        c="#F1C40F", marker="*", s=280, linewidths=0.6, edgecolors="black", label="Centroids", zorder=5,
    )

    # Draw radius circles (smaller, blue color matching known samples)
    umap_scale = 0.06
    umap_range = np.ptp(np.vstack([sample_xy, center_xy]), axis=0).mean()
    base_radius = umap_range * umap_scale

    for i, (cx, cy) in enumerate(center_xy):
        circle = plt.Circle((cx, cy), base_radius, color=KIND_COLORS["known"], fill=False, linestyle="--", linewidth=0.8, alpha=0.4)
        ax.add_patch(circle)

    # Set axis limits with more padding (wider horizontally, taller vertically)
    all_xy = np.vstack([sample_xy, center_xy])
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - x_range * 0.08, x_max + x_range * 0.08)
    ax.set_ylim(y_min - y_range * 0.10, y_max + y_range * 0.10)

    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.set_aspect("equal")
    ax.tick_params(length=2.5, width=0.6)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Legend below figure (centroid marker slightly larger)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5, markeredgecolor=KIND_COLORS["known"], markerfacecolor=KIND_COLORS["known"], label=KIND_LABELS["known"]),
        Line2D([0], [0], marker="*", linestyle="None", markersize=10, markeredgecolor="black", markerfacecolor="#F1C40F", label="Centroids"),
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

    # Save PNG and EPS
    png_path = OUTPUT_DIR / "gate_embedding_umap_clean_legacy_3panel.png"
    eps_path = OUTPUT_DIR / "gate_embedding_umap_clean_legacy_3panel.eps"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(eps_path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {eps_path}")


if __name__ == "__main__":
    main()
