"""Build the single-axis Ours-only dense-KIR sensitivity figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT.parent / "artifacts" / "s2c" / "analysis" / "kir_sensitivity_known_only" / "final_ours_selection"
OUT = ROOT / "results" / "analysis" / "kir_sensitivity_known_only" / "final_ours"
FIG = ROOT / "figures" / "kir_sensitivity_known_only" / "final_ours"
KIRS = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90)
SEEDS = (13, 42, 87)
INTENT_COUNTS = {"clinc150": 150, "stackoverflow": 20, "banking77": 77}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ART)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--figure-dir", type=Path, default=FIG)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    rows = []
    expected = {(dataset, kir, seed) for dataset in INTENT_COUNTS for kir in KIRS for seed in SEEDS}
    for path in sorted((artifact_root / "final").glob("*/*/Ours.json")):
        result = read(path)
        dataset = str(result["dataset"])
        kir = float(result["kir"])
        seed = int(result["seed"])
        if (dataset, kir, seed) not in expected:
            raise ValueError(f"unexpected result identity: {path}")
        if result.get("method") != "Ours" or result.get("full_pipeline") is not True:
            raise ValueError(f"not a full-pipeline Ours result: {path}")
        selection = result.get("selection", {})
        if selection.get("test_read") is not False or selection.get("test_used_for_selection") is not False:
            raise ValueError(f"test participated in selection: {path}")
        if selection.get("real_oos_used") is not False or selection.get("pseudo_oos_used") is not False:
            raise ValueError(f"OOS supervision participated in selection: {path}")
        metrics = result["metrics"]
        rows.append({
            "dataset": dataset,
            "kir": kir,
            "actual_kir": len(read(artifact_root.parent / "data" / dataset / f"kir{round(kir * 100):02d}_seed{seed}" / "known_labels.json")) / INTENT_COUNTS[dataset],
            "known_intent_count": len(read(artifact_root.parent / "data" / dataset / f"kir{round(kir * 100):02d}_seed{seed}" / "known_labels.json")),
            "intent_count": INTENT_COUNTS[dataset],
            "seed": seed,
            "oos_f1": 100.0 * float(metrics["oos_f1"]),
            "known_f1": 100.0 * float(metrics["known_f1"]),
            "accuracy": 100.0 * float(metrics["accuracy"]),
            "known_recall": 100.0 * float(metrics["known_recall"]),
            "false_accept_rate": 100.0 * float(metrics["false_accept_rate"]),
            "source": str(path),
        })
    actual = {(r["dataset"], float(r["kir"]), int(r["seed"])) for r in rows}
    missing = sorted(expected - actual)
    if missing:
        raise RuntimeError(f"missing {len(missing)} full-pipeline Ours cells; first={missing[:3]}")
    rows.sort(key=lambda r: (r["dataset"], r["kir"], r["seed"]))
    write_csv(args.output_dir / "per_seed.csv", rows)

    summary = []
    for dataset in INTENT_COUNTS:
        for kir in KIRS:
            group = [r for r in rows if r["dataset"] == dataset and r["kir"] == kir]
            if len(group) != len(SEEDS):
                raise RuntimeError(f"incomplete group {dataset}/{kir}")
            out = {"dataset": dataset, "kir": kir, "seed_count": len(group), "known_intent_count": group[0]["known_intent_count"], "actual_kir": group[0]["actual_kir"]}
            for metric in ("oos_f1", "known_f1", "accuracy", "known_recall", "false_accept_rate"):
                values = np.asarray([r[metric] for r in group], dtype=float)
                out[f"{metric}_mean"] = float(np.mean(values))
                out[f"{metric}_std"] = float(np.std(values, ddof=0))
            summary.append(out)
    write_csv(args.output_dir / "summary.csv", summary)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"clinc150": "#1f77b4", "stackoverflow": "#d62728", "banking77": "#2ca02c"}
    labels = {"clinc150": "CLINC150", "stackoverflow": "StackOverflow", "banking77": "Banking77"}
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for dataset in ("clinc150", "stackoverflow", "banking77"):
        points = [r for r in summary if r["dataset"] == dataset]
        ax.errorbar(
            [r["kir"] for r in points],
            [r["oos_f1_mean"] for r in points],
            yerr=[r["oos_f1_std"] for r in points],
            marker="o",
            linewidth=1.7,
            markersize=4,
            capsize=2,
            color=colors[dataset],
            label=labels[dataset],
        )
    ax.set_xlabel("Known Intent Ratio (KIR)")
    ax.set_ylabel("OOS F1 (%)")
    ax.set_xticks(KIRS)
    ax.set_xlim(0.07, 0.93)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.figure_dir / f"ours_oos_f1.{extension}", dpi=300)
    plt.close(fig)

    manifest = {
        "status": "complete",
        "experiment": "ours_dense_kir_final_contract",
        "artifact_root": str(artifact_root),
        "datasets": list(INTENT_COUNTS),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "units": len(rows),
        "summary_points": len(summary),
        "metric": "OOS F1 from the same full-pipeline predictions",
        "aggregation": "mean and population std (ddof=0) over seeds",
        "selection": "Known-dev-only; no real or pseudo OOS; test not used for selection",
        "figure": str(args.figure_dir / "ours_oos_f1.png"),
        "test_previously_observed": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"FIGURE_COMPLETE units={len(rows)} summary_points={len(summary)}", flush=True)


if __name__ == "__main__":
    main()
