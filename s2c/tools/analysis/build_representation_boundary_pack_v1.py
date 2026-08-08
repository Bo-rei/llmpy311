"""Summarize existing representation-training and K=1/K=2 results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "representation_boundary_pack_v1"
FIG = ROOT / "figures" / "representation_boundary_pack_v1"


def normalize_dataset(value: str) -> str:
    return "banking77" if str(value).startswith("banking77") else str(value)


def load_gate() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "representation" / "representation_fixed_results.csv")
    df["dataset"] = df["dataset"].map(normalize_dataset)
    return df[df["k_variant"].isin(["k1", "k2"])].copy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "representation", "k_variant"], as_index=False)
        .agg(
            n_seeds=("data_seed", "nunique"),
            oos_f1_mean=("oos_f1", "mean"),
            oos_f1_std=("oos_f1", "std"),
            near_oos_f1_mean=("near_oos_f1", "mean"),
            near_oos_f1_std=("near_oos_f1", "std"),
            medium_oos_f1_mean=("medium_oos_f1", "mean"),
            far_oos_f1_mean=("far_oos_f1", "mean"),
            id_recall_mean=("id_recall", "mean"),
            id_recall_std=("id_recall", "std"),
            false_accept_mean=("oos_recall", "mean"),
            false_accept_std=("oos_recall", "std"),
            auroc_mean=("auroc", "mean"),
            aupr_oos_mean=("aupr_oos", "mean"),
        )
    )


def effects(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index=["dataset", "representation", "data_seed"], columns="k_variant", values=["oos_f1", "near_oos_f1", "id_recall", "oos_recall"])
    rows = []
    for (dataset, representation), group in wide.groupby(level=[0, 1]):
        for metric in ["oos_f1", "near_oos_f1", "id_recall", "oos_recall"]:
            deltas = group[metric]["k2"] - group[metric]["k1"]
            rows.append(
                {
                    "dataset": dataset,
                    "representation": representation,
                    "metric": metric,
                    "n_seeds": int(deltas.notna().sum()),
                    "k2_minus_k1_mean": float(deltas.mean()),
                    "k2_minus_k1_std": float(deltas.std()),
                    "wins": int((deltas > 0).sum()),
                    "ties": int((deltas == 0).sum()),
                    "losses": int((deltas < 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def load_geometry() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / "representation" / "representation_geometry_summary.csv")
    df["dataset"] = df["dataset"].map(normalize_dataset)
    return df


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, dataset in zip(axes, sorted(summary["dataset"].unique())):
        part = summary[summary["dataset"].eq(dataset)]
        for rep in sorted(part["representation"].unique()):
            rows = part[part["representation"].eq(rep)].set_index("k_variant").reindex(["k1", "k2"])
            ax.plot([1, 2], rows[f"{metric}_mean"] * 100, marker="o", label=rep)
        ax.set_title(dataset)
        ax.set_xticks([1, 2], ["K=1", "K=2"])
        ax.grid(alpha=.25)
        ax.set_xlabel("center count")
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Representation training: {ylabel} and center count")
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def plot_geometry(geometry: pd.DataFrame) -> None:
    metrics = [("effective_rank", "Effective rank"), ("relative_separation", "Relative separation"), ("same_intent_alignment", "Same-intent alignment")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric, label) in zip(axes, metrics):
        grouped = geometry.groupby("representation")[metric].mean().sort_values()
        ax.bar(grouped.index, grouped.values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"][: len(grouped)])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Representation geometry diagnostics (existing runs)")
    fig.tight_layout()
    fig.savefig(FIG / "representation_geometry_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_gate()
    summary = summarize(df)
    delta = effects(df)
    geometry = load_geometry()
    summary.to_csv(OUT / "representation_k1_k2_summary.csv", index=False)
    delta.to_csv(OUT / "representation_k2_minus_k1.csv", index=False)
    geometry.to_csv(OUT / "representation_geometry.csv", index=False)
    plot_metric(summary, "oos_f1", "OOS F1 (%)", "representation_oos_f1_k1_k2.png")
    plot_metric(summary, "near_oos_f1", "Near-OOS F1 (%)", "representation_near_oos_f1_k1_k2.png")
    plot_metric(summary, "id_recall", "Known Recall (%)", "representation_known_recall_k1_k2.png")
    plot_geometry(geometry)
    print(f"summary={len(summary)} rows; effects={len(delta)} rows; figures={len(list(FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
