#!/usr/bin/env python3
"""Generate single-panel UMAP figure combining all sample types."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

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
OUTPUT = PROJECT_ROOT / "figures" / "paper_v19" / "gate_embedding_umap_clean_legacy_3panel.png"

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


def main():
    from sentence_transformers import SentenceTransformer
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

    # Select known intents (8 intents)
    support = Counter(str(p["true_intent"]) for p in predictions if not bool(p.get("is_oos", False)))
    candidates = sorted(intent for intent, size in support.items() if size >= 20)
    rng = np.random.default_rng(20260427)
    selected = sorted(rng.choice(candidates, size=8, replace=False).tolist())
    selected_set = set(selected)

    # Filter predictions
    sampled = []
    for p in predictions:
        true_intent = str(p.get("true_intent", ""))
        is_oos = bool(p.get("is_oos", False))
        if not is_oos and true_intent not in selected_set:
            continue
        sampled.append(p)

    # Subsample OOS if too many
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

    # Encode texts
    encoder_path = Path(str(run_manifest.get("gate_encoder_path", "all-MiniLM-L6-v2")))
    if not encoder_path.is_absolute():
        encoder_path = PROJECT_ROOT / encoder_path
    print(f"Loading encoder from {encoder_path}")
    encoder = SentenceTransformer(str(encoder_path))
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    print(f"Encoded {len(embeddings)} samples")

    # Select centers for selected intents
    selected_centers_mask = [c in selected_set for c in center_intents]
    selected_centers = centers[selected_centers_mask]
    print(f"Selected {len(selected_centers)} centers")

    # UMAP projection
    all_points = np.vstack([embeddings, selected_centers])
    reducer = UMAP(n_components=2, n_neighbors=30, min_dist=0.08, metric="cosine", random_state=20260427)
    projected = reducer.fit_transform(all_points)
    sample_xy = projected[:len(embeddings)]
    center_xy = projected[len(embeddings):]
    print("UMAP done")

    # Plot
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    kind_array = np.asarray(kinds)

    # Known samples
    mask = kind_array == "known"
    ax.scatter(sample_xy[mask, 0], sample_xy[mask, 1], s=20, marker="o", c=KIND_COLORS["known"], alpha=0.5, linewidths=0.0, label=KIND_LABELS["known"], zorder=2)

    # Unknown samples
    mask = kind_array == "heldout_unknown"
    ax.scatter(sample_xy[mask, 0], sample_xy[mask, 1], s=40, marker="x", c=KIND_COLORS["heldout_unknown"], alpha=0.85, linewidths=1.5, label=KIND_LABELS["heldout_unknown"], zorder=3)

    # OOS samples
    mask = kind_array == "native_oos"
    ax.scatter(sample_xy[mask, 0], sample_xy[mask, 1], s=40, marker="x", c=KIND_COLORS["native_oos"], alpha=0.85, linewidths=1.5, label=KIND_LABELS["native_oos"], zorder=3)

    # Centroids
    ax.scatter(center_xy[:, 0], center_xy[:, 1], c="black", marker="*", s=200, linewidths=0.75, label="Centroids", zorder=5)

    ax.set_xlabel("UMAP dim1")
    ax.set_ylabel("UMAP dim2")
    ax.grid(True, color="#D0D0D0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=True, handletextpad=0.5, columnspacing=1.0, labelspacing=0.3)

    fig.tight_layout()
    fig.savefig(str(OUTPUT), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
