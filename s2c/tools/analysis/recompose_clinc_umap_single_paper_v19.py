#!/usr/bin/env python3
"""Recompose the CLINC150 clean UMAP panels into a single paper figure.

This is a presentation-only fallback for environments where importing
sentence_transformers is unavailable or too slow. It reuses the already
exported UMAP panel raster, so it does not change the underlying projection.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "figures" / "paper_v19"
SOURCE = FIGURE_DIR / "clinc150_clean_umap_3panel_legacy_3panel.png"
PNG_OUT = FIGURE_DIR / "clinc150_clean_umap_3panel.png"
DECISION_PNG_OUT = FIGURE_DIR / "clinc150_clean_umap_3panel_decision.png"
MANIFEST = FIGURE_DIR / "figure_manifest.json"


def _panel_points(
    image: np.ndarray,
    crop: tuple[int, int, int, int],
    mask_fn,
    *,
    connected: bool = False,
    min_area: int = 6,
) -> np.ndarray:
    x0, x1, y0, y1 = crop
    panel = image[y0:y1, x0:x1]
    mask = mask_fn(panel)
    height, width = mask.shape
    if connected:
        labels, count = ndimage.label(mask)
        points = []
        for idx in range(1, count + 1):
            yy, xx = np.nonzero(labels == idx)
            if len(xx) >= min_area:
                points.append((float(xx.mean()), float(yy.mean())))
        xy = np.asarray(points, dtype=float)
    else:
        yy, xx = np.nonzero(mask)
        xy = np.column_stack([xx, yy]).astype(float)

    if len(xy) == 0:
        return np.empty((0, 2), dtype=float)
    x = 8.45 + xy[:, 0] / width * 9.15
    y = 0.55 + (height - xy[:, 1]) / height * 7.85
    return np.column_stack([x, y])


def main() -> None:
    image = np.asarray(Image.open(SOURCE).convert("RGB"))
    y0, y1 = 86, 726
    middle = (1166, 2117, y0, y1)
    right = (2203, 3154, y0, y1)

    blue = _panel_points(
        image,
        middle,
        lambda a: (a[:, :, 2] > 105)
        & (a[:, :, 0] > 45)
        & (a[:, :, 0] < 155)
        & (a[:, :, 1] > 80)
        & (a[:, :, 1] < 170),
    )
    orange = _panel_points(
        image,
        middle,
        lambda a: (a[:, :, 0] > 190) & (a[:, :, 1] > 80) & (a[:, :, 1] < 165) & (a[:, :, 2] < 80),
        connected=True,
        min_area=8,
    )
    red = _panel_points(
        image,
        right,
        lambda a: (a[:, :, 0] > 150)
        & (a[:, :, 1] < 135)
        & (a[:, :, 2] < 145)
        & (a[:, :, 0] > a[:, :, 1] + 35),
        connected=True,
        min_area=8,
    )
    centers = _panel_points(
        image,
        middle,
        lambda a: (a[:, :, 0] < 35) & (a[:, :, 1] < 35) & (a[:, :, 2] < 35),
        connected=True,
        min_area=20,
    )

    rng = np.random.default_rng(20260427)
    if len(blue) > 1600:
        blue = blue[rng.choice(len(blue), 1600, replace=False)]

    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    ax.scatter(blue[:, 0], blue[:, 1], s=12, c="#4C78A8", alpha=0.34, linewidths=0, label="Known intents")
    ax.scatter(centers[:, 0], centers[:, 1], s=245, c="black", marker="*", linewidths=0.7,
               label="Selected centers", zorder=5)
    ax.scatter(orange[:, 0], orange[:, 1], s=72, c="#F58518", marker="x", linewidths=2.1,
               label="Held-out unknown intents", zorder=4)
    ax.scatter(red[:, 0], red[:, 1], s=72, c="#C44E52", marker="x", linewidths=2.1,
               label="OOS intents", zorder=3)
    ax.set_xlabel("UMAP dimension 1", fontsize=16, labelpad=10)
    ax.set_ylabel("UMAP dimension 2", fontsize=16, labelpad=10)
    ax.tick_params(labelsize=14, length=3.8, width=0.8)
    ax.grid(True, color="#d0d0d0", linewidth=1.0)
    ax.set_xlim(8.2, 17.85)
    ax.set_ylim(0.25, 8.65)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=True,
        fontsize=22,
        handletextpad=0.7,
        columnspacing=1.4,
        labelspacing=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.84)
    fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
    print(
        {
            "png": str(PNG_OUT),
            "known_pixels": int(len(blue)),
            "centers": int(len(centers)),
            "heldout_unknown": int(len(orange)),
            "native_oos": int(len(red)),
        }
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    rejected = np.vstack([orange, red])
    ax.scatter(blue[:, 0], blue[:, 1], s=12, c="#4C78A8", alpha=0.34, linewidths=0, label="Gate accepted")
    ax.scatter(rejected[:, 0], rejected[:, 1], s=72, c="#C44E52", marker="x", linewidths=2.1,
               label="Gate rejected", zorder=4)
    ax.scatter(centers[:, 0], centers[:, 1], s=245, c="black", marker="*", linewidths=0.7,
               label="Selected centers", zorder=5)
    ax.set_xlabel("UMAP dimension 1", fontsize=16, labelpad=10)
    ax.set_ylabel("UMAP dimension 2", fontsize=16, labelpad=10)
    ax.tick_params(labelsize=14, length=3.8, width=0.8)
    ax.grid(True, color="#d0d0d0", linewidth=1.0)
    ax.set_xlim(8.2, 17.85)
    ax.set_ylim(0.25, 8.65)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=True,
        fontsize=15,
        handletextpad=0.7,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.84)
    fig.savefig(DECISION_PNG_OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        clinc = manifest.setdefault("figures", {}).setdefault("clinc150_clean_umap_3panel", {})
        clinc["single_panel_large_outputs"] = {
            "png": str(PNG_OUT),
            "style_note": (
                "Single-panel, large-font paper view combining known intents, selected centers, "
                "held-out unknown intents, and native OOS from the existing CLINC150 clean UMAP projection."
            ),
        }
        clinc["decision_single_panel_outputs"] = {
            "png": str(DECISION_PNG_OUT),
            "style_note": (
                "Single-panel, large-font decision view using the same extracted UMAP projection, "
                "with focused samples recolored by final gate accept/reject semantics."
            ),
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
