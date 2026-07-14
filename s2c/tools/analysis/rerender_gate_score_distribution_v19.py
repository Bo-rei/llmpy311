#!/usr/bin/env python3
"""Re-render gate_score_distribution_clean.png with ACL-standard formatting.

Fixes:
  1. Font sizes scaled to ~10-11pt (conference body text)
  2. Whitespace trimmed via bbox_inches='tight' + pad_inches=0.01
  3. Smaller figsize for single-column LaTeX (no manual scaling needed)
  4. Thicker density curves (linewidth=2.0)
  5. Legend in upper-left, compact, not covering peaks
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
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
OUTPUT_DIR = PROJECT_ROOT / "figures" / "paper_v19"
SLUG = "clinc150"

# ---------------------------------------------------------------------------
# Color & label config  (same as original)
# ---------------------------------------------------------------------------
KIND_ORDER = ["known", "heldout_unknown", "native_oos"]
KIND_LABELS = {
    "known": "Known intents",
    "heldout_unknown": "Unknown intents",
    "native_oos": "OOS intents",
}
KIND_COLORS = {
    "known": "#0000FC",
    "heldout_unknown": "#4F5D75",
    "native_oos": "#C0392B",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_kind(pred: Dict[str, Any]) -> str:
    if not bool(pred.get("is_oos", False)):
        return "known"
    if str(pred.get("true_intent")) == "oos":
        return "native_oos"
    return "heldout_unknown"


def prediction_score(pred: Dict[str, Any]) -> float:
    score = pred.get("gate_score")
    if score is not None:
        return float(score)
    distance = float(pred["gate_distance"])
    radius = float(pred["gate_radius"])
    return distance / radius


def load_gate_score_rows(predictions_path: Path) -> List[Dict[str, Any]]:
    rows = []
    for pred in load_json(predictions_path):
        rows.append({"kind": sample_kind(pred), "score": prediction_score(pred)})
    return rows


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_gate_score_distribution(
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    # ---- ACL-standard font config ----
    plt.rcParams.update({
        # Base font size ~10pt (matches LaTeX \normalsize at 10pt)
        "font.family": "DejaVu Sans",
        "font.size": 10,
        # Axes
        "axes.titlesize": 11,
        "axes.labelsize": 11,       # axis labels ~11pt
        "axes.linewidth": 0.8,
        # Ticks
        "xtick.labelsize": 10,      # tick labels ~10pt
        "ytick.labelsize": 10,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        # Legend
        "legend.fontsize": 10,      # legend text ~10pt
        "legend.frameon": False,
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        # PDF/vector font embedding
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ---- Bin scores ----
    by_kind = {kind: [r["score"] for r in rows if r["kind"] == kind] for kind in KIND_ORDER}
    all_scores = [r["score"] for r in rows]
    upper = min(max(all_scores) * 1.02, np.percentile(all_scores, 99.5) * 1.2)
    bins = np.linspace(0.0, max(1.8, float(upper)), 80)

    # ---- Figure: compact for single-column LaTeX ----
    # Standard ACL single-column width is ~3.3in (or ~5.5in for full page).
    # figsize=(4.5, 2.8) fills a single column nicely without scaling.
    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    # ---- Draw density curves ----
    for kind in KIND_ORDER:
        scores = by_kind[kind]
        if not scores:
            continue
        density, edges = np.histogram(np.asarray(scores, dtype=float), bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.plot(
            centers, density,
            color=KIND_COLORS[kind],
            linewidth=2.0,          # thicker for print legibility
            label=KIND_LABELS[kind],
        )
        ax.fill_between(centers, density, color=KIND_COLORS[kind], alpha=0.12)

    # ---- Threshold line ----
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.4, label="Threshold")

    # ---- Labels ----
    ax.set_xlabel("Gate score")
    ax.set_ylabel("Density")

    # ---- Legend: upper-left, compact, no frame ----
    # Use small padding so it doesn't eat into the data region
    ax.legend(
        loc="upper left",
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.2,
        borderaxespad=0.3,
        borderpad=0.3,
    )

    # ---- Tight layout + aggressive crop ----
    fig.tight_layout(pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    root = DEFAULT_ROOT
    predictions_path = root / SLUG / "kir50_seed42" / "full_anchor" / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found: {predictions_path}")

    rows = load_gate_score_rows(predictions_path)
    print(f"Loaded {len(rows)} rows: "
          f"{sum(1 for r in rows if r['kind'] == 'known')} known, "
          f"{sum(1 for r in rows if r['kind'] == 'heldout_unknown')} held-out unknown, "
          f"{sum(1 for r in rows if r['kind'] == 'native_oos')} OOS")

    output_path = OUTPUT_DIR / "gate_score_distribution_clean.png"
    plot_gate_score_distribution(rows, output_path)

    # Also save PDF for LaTeX vector inclusion
    pdf_path = OUTPUT_DIR / "gate_score_distribution_clean.pdf"
    plot_gate_score_distribution(rows, pdf_path)


if __name__ == "__main__":
    main()
