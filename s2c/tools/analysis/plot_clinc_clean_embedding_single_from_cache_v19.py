#!/usr/bin/env python3
"""Draw a readable single-panel CLINC150 clean embedding figure from cached MiniLM embeddings."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis.export_paper_visualizations_v19 import (
    DEFAULT_ROOT,
    KIND_COLORS,
    _filter_clean_clinc_predictions,
    _project_2d,
    _selected_centers,
    detector_path,
    load_detector_centers,
    load_json,
    predictions_path,
    run_manifest,
    sample_kind,
    select_known_intents,
)


OUTPUT = PROJECT_ROOT / "figures" / "paper_v19" / "clinc150_clean_umap_single_paper.png"
CACHE = PROJECT_ROOT / "cache" / "gate_embeddings" / "test.pt"
MANIFEST = PROJECT_ROOT / "figures" / "paper_v19" / "figure_manifest.json"


def _embedding_lookup(cache: Dict[str, object]) -> Dict[str, List[np.ndarray]]:
    lookup: Dict[str, List[np.ndarray]] = {}
    for prefix in ("known", "oos"):
        embeddings = cache[f"{prefix}_embeddings"].detach().cpu().numpy()
        texts = cache[f"{prefix}_texts"]
        for text, embedding in zip(texts, embeddings):
            lookup.setdefault(str(text), []).append(np.asarray(embedding, dtype=np.float32))
    return lookup


def _points_from_cache(sampled: List[dict]) -> np.ndarray:
    cache = torch.load(CACHE, map_location="cpu")
    lookup = _embedding_lookup(cache)
    rows = []
    missing = []
    for row in sampled:
        bucket = lookup.get(str(row["text"]))
        if not bucket:
            missing.append(str(row["text"]))
            continue
        rows.append(bucket.pop(0))
    if missing:
        raise ValueError(f"Missing cached embeddings for {len(missing)} sampled rows")
    return np.asarray(rows, dtype=np.float32)


def _project(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, str, dict]:
    try:
        return _project_2d(matrix, seed=seed)
    except Exception:
        reducer = PCA(n_components=2, random_state=seed)
        return (
            np.asarray(reducer.fit_transform(matrix), dtype=np.float32),
            "PCA fallback",
            {"n_components": 2, "random_state": seed},
        )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6.2, markerfacecolor=KIND_COLORS["known"],
               markeredgecolor="none", alpha=0.38, label="Known intents"),
        Line2D([0], [0], marker="*", linestyle="None", markersize=16.0, markerfacecolor="black",
               markeredgecolor="black", label="Selected centers"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=9.2, markeredgecolor=KIND_COLORS["heldout_unknown"],
               markerfacecolor="none", markeredgewidth=1.9, label="Held-out unknown intents"),
        Line2D([0], [0], marker="x", linestyle="None", markersize=9.2, markeredgecolor=KIND_COLORS["native_oos"],
               markerfacecolor="none", markeredgewidth=1.9, label="OOS intents"),
    ]


def _draw(output: Path, sample_xy: np.ndarray, center_xy: np.ndarray, kinds: List[str], method: str) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.labelsize": 16,
            "legend.fontsize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    fig, ax = plt.subplots(figsize=(13.8, 6.4))
    kind_array = np.asarray(kinds)
    for kind, label, marker, size, alpha, zorder in [
        ("known", "Known intents", "o", 18, 0.36, 2),
        ("heldout_unknown", "Held-out unknown intents", "x", 72, 0.94, 4),
        ("native_oos", "OOS intents", "x", 72, 0.92, 3),
    ]:
        mask = kind_array == kind
        if not mask.any():
            continue
        ax.scatter(
            sample_xy[mask, 0],
            sample_xy[mask, 1],
            s=size,
            marker=marker,
            c=KIND_COLORS[kind],
            alpha=alpha,
            linewidths=0.0 if marker == "o" else 2.1,
            label=label,
            zorder=zorder,
        )
    ax.scatter(center_xy[:, 0], center_xy[:, 1], s=245, c="black", marker="*", linewidths=0.7,
               label="Selected centers", zorder=6)

    xy = np.vstack([sample_xy, center_xy])
    mins = np.nanmin(xy, axis=0)
    maxs = np.nanmax(xy, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pad = spans * 0.065
    ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))
    axis_name = "UMAP" if method == "UMAP" else "Projection"
    ax.set_xlabel(f"{axis_name} dimension 1", labelpad=10)
    ax.set_ylabel(f"{axis_name} dimension 2", labelpad=10)
    ax.grid(True, color="#D0D0D0", linewidth=0.95)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CFCFCF")
        spine.set_linewidth(0.95)
    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=True,
        handletextpad=0.7,
        columnspacing=1.4,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.155, top=0.80)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> None:
    seed = 20260427
    predictions = load_json(predictions_path(DEFAULT_ROOT, "clinc150"))
    selected_intents = select_known_intents(predictions, count=8, seed=seed, min_support=20)
    sampled = _filter_clean_clinc_predictions(predictions, selected_intents, max_oos_per_kind=260, seed=seed)
    kinds = [sample_kind(row) for row in sampled]
    embeddings = _points_from_cache(sampled)

    manifest = run_manifest(DEFAULT_ROOT, "clinc150")
    encoder_path = str(manifest.get("gate_encoder_path", CACHE))
    all_centers = load_detector_centers(detector_path(DEFAULT_ROOT, "clinc150"))
    centers = _selected_centers(all_centers, selected_intents)
    all_points = np.vstack([embeddings, centers.centers])
    projected, method, params = _project(all_points, seed=seed)
    sample_xy = projected[: len(sampled)]
    center_xy = projected[len(sampled) :]

    _draw(OUTPUT, sample_xy, center_xy, kinds, method)
    result = {
        "png": str(OUTPUT),
        "projection": method,
        "projection_parameters": params,
        "source_cache": str(CACHE),
        "gate_encoder_path": encoder_path,
        "sample_count": len(sampled),
        "kind_counts": {kind: int(kinds.count(kind)) for kind in ("known", "heldout_unknown", "native_oos")},
        "selected_known_intents": selected_intents,
    }
    if MANIFEST.exists():
        figure_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        clinc = figure_manifest.setdefault("figures", {}).setdefault("clinc150_clean_umap_3panel", {})
        clinc["single_panel_large_outputs"] = {"png": result["png"]}
        clinc["single_panel_large_note"] = (
            "Readable single-panel paper view generated from cached MiniLM gate embeddings and detector centers."
        )
        clinc["single_panel_large_projection"] = {
            "projection": method,
            "projection_parameters": params,
            "source_cache": str(CACHE),
            "kind_counts": result["kind_counts"],
        }
        MANIFEST.write_text(json.dumps(figure_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(result)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
