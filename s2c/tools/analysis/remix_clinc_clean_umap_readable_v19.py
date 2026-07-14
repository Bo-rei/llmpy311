#!/usr/bin/env python3
"""Remix the existing CLINC150 clean UMAP panels into one readable paper figure.

This is a lightweight fallback for environments where importing torch to
re-encode samples is unavailable. It uses only the already-exported panel image
as source evidence and does not recompute embeddings or predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = PROJECT_ROOT / "figures" / "paper_v19"
SOURCE = PAPER_DIR / "clinc150_clean_umap_3panel.png"
OUTPUT = PAPER_DIR / "clinc150_clean_umap_singlepanel_readable.png"
MANIFEST = PAPER_DIR / "figure_manifest.json"

PANEL_BOXES = {
    "known": (128, 86, 1082, 725),
    "heldout": (1165, 86, 2119, 725),
    "native_oos": (2202, 86, 3156, 725),
}

COLORS = {
    "known": "#4C78A8",
    "heldout": "#F58518",
    "native_oos": "#C44E52",
    "centers": "black",
}
X_RANGE = (8.5, 17.5)
Y_RANGE = (0.9, 8.3)


def _component_centers(mask: np.ndarray, min_pixels: int, max_pixels: int) -> np.ndarray:
    """Return connected-component centroids in image coordinates."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    centers: list[tuple[float, float, int]] = []
    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if seen[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        seen[start_y, start_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        size = len(pixels)
        if min_pixels <= size <= max_pixels:
            arr = np.asarray(pixels, dtype=float)
            centers.append((float(arr[:, 1].mean()), float(arr[:, 0].mean()), size))
    if not centers:
        return np.empty((0, 2), dtype=float)
    return np.asarray([(x, y) for x, y, _ in centers], dtype=float)


def _to_axes(points: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    if len(points) == 0:
        return points
    x_norm = (points[:, 0] + left - left) / max(right - left, 1)
    y_norm = 1.0 - (points[:, 1] + top - top) / max(bottom - top, 1)
    x = X_RANGE[0] + x_norm * (X_RANGE[1] - X_RANGE[0])
    y = Y_RANGE[0] + y_norm * (Y_RANGE[1] - Y_RANGE[0])
    return np.column_stack([x, y])


def _sample_pixels(mask: np.ndarray, box: tuple[int, int, int, int], stride: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=float)
    points = np.column_stack([xs, ys]).astype(float)
    return _to_axes(points[::stride], box)


def _crop_mask(image: np.ndarray, box: tuple[int, int, int, int], kind: str) -> np.ndarray:
    left, top, right, bottom = box
    crop = image[top:bottom, left:right, :]
    r = crop[:, :, 0].astype(int)
    g = crop[:, :, 1].astype(int)
    b = crop[:, :, 2].astype(int)
    if kind == "known":
        return (b > 130) & (g > 95) & (r < 170) & (b > r + 25)
    if kind == "heldout":
        return (r > 190) & (g > 90) & (g < 190) & (b < 130)
    if kind == "native_oos":
        return (r > 165) & (g < 150) & (b < 170) & (r > g + 35)
    if kind == "centers":
        return (r < 35) & (g < 35) & (b < 35)
    raise ValueError(kind)


def _collect_points(image: np.ndarray) -> dict[str, np.ndarray]:
    known_mask = _crop_mask(image, PANEL_BOXES["known"], "known")
    heldout_mask = _crop_mask(image, PANEL_BOXES["heldout"], "heldout")
    native_mask = _crop_mask(image, PANEL_BOXES["native_oos"], "native_oos")
    center_mask = _crop_mask(image, PANEL_BOXES["known"], "centers")

    return {
        "known": _sample_pixels(known_mask, PANEL_BOXES["known"], stride=9),
        "heldout": _to_axes(_component_centers(heldout_mask, min_pixels=10, max_pixels=3000), PANEL_BOXES["heldout"]),
        "native_oos": _to_axes(_component_centers(native_mask, min_pixels=10, max_pixels=3000), PANEL_BOXES["native_oos"]),
        "centers": _to_axes(_component_centers(center_mask, min_pixels=35, max_pixels=450), PANEL_BOXES["known"]),
    }


def _legend_handles() -> Iterable[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="None", markersize=5.0, markerfacecolor=COLORS["known"],
               markeredgecolor="none", alpha=0.42, label="Known intents"),
        Line2D([0], [0], marker="*", linestyle="None", markersize=15.0, markerfacecolor="black",
               markeredgecolor="black", label="Selected centers"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=9.0, markeredgecolor=COLORS["heldout"],
               markerfacecolor="none", markeredgewidth=1.8, label="Held-out unknown intents"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=9.0, markeredgecolor=COLORS["native_oos"],
               markerfacecolor="none", markeredgewidth=1.8, label="OOS intents"),
    ]


def main() -> None:
    image = np.asarray(Image.open(SOURCE).convert("RGB"))
    points = _collect_points(image)

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 15,
            "axes.labelsize": 15,
            "legend.fontsize": 20,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    ax.scatter(points["known"][:, 0], points["known"][:, 1], s=10, c=COLORS["known"], alpha=0.30, linewidths=0)
    ax.scatter(points["heldout"][:, 0], points["heldout"][:, 1], s=58, c=COLORS["heldout"], marker="x",
               linewidths=1.8, alpha=0.95)
    ax.scatter(points["native_oos"][:, 0], points["native_oos"][:, 1], s=58, c=COLORS["native_oos"], marker="x",
               linewidths=1.8, alpha=0.92)
    ax.scatter(points["centers"][:, 0], points["centers"][:, 1], s=235, c="black", marker="*", linewidths=0.7, zorder=6)

    ax.set_xlim(X_RANGE[0] - 0.25, X_RANGE[1] + 0.25)
    ax.set_ylim(Y_RANGE[0] - 0.25, Y_RANGE[1] + 0.25)
    ax.set_xlabel("UMAP dimension 1", labelpad=10)
    ax.set_ylabel("UMAP dimension 2", labelpad=10)
    ax.grid(True, color="#D0D0D0", linewidth=0.95)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#CFCFCF")
        spine.set_linewidth(0.95)
    ax.legend(
        handles=list(_legend_handles()),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=2,
        frameon=True,
        handletextpad=0.75,
        columnspacing=1.4,
        labelspacing=0.45,
        borderpad=0.35,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    counts = {key: int(len(value)) for key, value in points.items()}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        figures = manifest.setdefault("figures", {})
        figures["clinc150_clean_umap_singlepanel_readable"] = {
            "figure": str(OUTPUT),
            "outputs": {"png": str(OUTPUT)},
            "source_figure": str(SOURCE),
            "remix_note": (
                "Readable single-panel remix of the existing three-panel CLINC150 clean UMAP export; "
                "no embeddings, predictions, or detector outputs are recomputed."
            ),
            "extracted_visual_counts": counts,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        {
            "png": str(OUTPUT),
            "counts": counts,
        }
    )


if __name__ == "__main__":
    main()
