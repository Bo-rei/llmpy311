"""Create a comparison/statistics/visualization pack from completed s2c runs.

This is analysis-only. It never reads raw text, never changes a run, and never
uses test results to select a configuration. All comparisons retain their
protocol/supervision scope in the output.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "experiment_evidence_pack_v2"
FIG = ROOT / "figures" / "experiment_evidence_pack_v2"

METHOD_LABELS = {
    "single_centroid": "Single centroid",
    "fixed_k2": "Fixed K=2",
    "random_partition": "Random partition",
    "mogb_minilm": "MOGB-MiniLM",
    "mogb_partition_ours_boundary": "MOGB partition + s2c boundary",
    "ours_partition_mogb_boundary": "s2c partition + MOGB boundary",
}

COLORS = {
    "Single centroid": "#1f77b4",
    "Fixed K=2": "#ff7f0e",
    "Random partition": "#2ca02c",
    "MOGB-MiniLM": "#d62728",
    "MOGB partition + s2c boundary": "#9467bd",
    "s2c partition + MOGB boundary": "#8c564b",
    "Trainable MiniLM K=1": "#e377c2",
}


def _bootstrap_ci(values: pd.Series, seed: int = 20260725, n: int = 10_000) -> tuple[float, float]:
    values = values.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = __import__("numpy").random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return float(__import__("numpy").quantile(samples, 0.025)), float(__import__("numpy").quantile(samples, 0.975))


def read_fair_matrix() -> pd.DataFrame:
    path = ROOT / "results" / "mogb" / "fair_matrix.csv"
    df = pd.read_csv(path)
    df = df[df["method"].isin(METHOD_LABELS)].copy()
    df["method_label"] = df["method"].map(METHOD_LABELS)
    return df


def summarize_fair(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["dataset", "kir", "method", "method_label"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            oos_f1_mean=("oos_f1", "mean"),
            oos_f1_std=("oos_f1", "std"),
            known_recall_mean=("id_recall", "mean"),
            known_recall_std=("id_recall", "std"),
            f1_all_mean=("f1_all", "mean"),
            f1_all_std=("f1_all", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            false_accept_mean=("false_accept_rate", "mean"),
            false_accept_std=("false_accept_rate", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
        )
    )
    return grouped


def paired_effects(df: pd.DataFrame) -> pd.DataFrame:
    reference = df[df["method"].eq("single_centroid")].set_index(["dataset", "kir", "seed"])
    rows: list[dict[str, object]] = []
    for method in [m for m in METHOD_LABELS if m != "single_centroid"]:
        candidate = df[df["method"].eq(method)].set_index(["dataset", "kir", "seed"])
        common = reference.index.intersection(candidate.index)
        for dataset, kir in sorted({(idx[0], idx[1]) for idx in common}):
            idx = [x for x in common if x[0] == dataset and x[1] == kir]
            for metric, higher_is_better in [
                ("oos_f1", True),
                ("f1_all", True),
                ("id_recall", True),
                ("false_accept_rate", False),
            ]:
                deltas = pd.Series([candidate.loc[key, metric] - reference.loc[key, metric] for key in idx])
                wins = int((deltas > 1e-12 if higher_is_better else deltas < -1e-12).sum())
                ties = int((deltas.abs() <= 1e-12).sum())
                losses = len(deltas) - wins - ties
                low, high = _bootstrap_ci(deltas, seed=20260725 + len(rows))
                rows.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "metric": metric,
                        "n_pairs": len(deltas),
                        "mean_delta": float(deltas.mean()),
                        "median_delta": float(deltas.median()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "reference": "single_centroid",
                        "seed": "paired same split/seed",
                    }
                )
    return pd.DataFrame(rows)


def trainable_context() -> pd.DataFrame:
    path = ROOT / "results" / "analysis" / "minilm_trainable_kir_sweep_v1" / "mean_std.csv"
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "dataset": df["dataset"],
            "kir": df["kir"],
            "method": "Trainable MiniLM K=1",
            "method_label": "Trainable MiniLM K=1",
            "n_seeds": df["n_seeds"],
            "oos_f1_mean": df["oos_f1_mean"],
            "oos_f1_std": df["oos_f1_std"],
            "known_recall_mean": df["known_recall_mean"],
            "known_recall_std": df["known_recall_std"],
            "f1_all_mean": df["f1_all_mean"],
            "f1_all_std": df["f1_all_std"],
            "accuracy_mean": df["accuracy_mean"],
            "accuracy_std": df["accuracy_std"],
            "false_accept_mean": df["false_accept_rate_mean"],
            "false_accept_std": df["false_accept_rate_std"],
            "auroc_mean": df["auroc_mean"],
            "auroc_std": df["auroc_std"],
        }
    )


def plot_oos(summary: pd.DataFrame, trainable: pd.DataFrame) -> None:
    for dataset in sorted(summary["dataset"].unique()):
        fig, ax = plt.subplots(figsize=(9, 5.2))
        part = summary[summary["dataset"].eq(dataset)]
        for method in part["method_label"].unique():
            rows = part[part["method_label"].eq(method)].sort_values("kir")
            ax.errorbar(
                rows["kir"], rows["oos_f1_mean"] * 100, yerr=rows["oos_f1_std"].fillna(0) * 100,
                marker="o", label=method, color=COLORS.get(method), capsize=2,
            )
        tr = trainable[trainable["dataset"].eq(dataset)].sort_values("kir")
        ax.errorbar(tr["kir"], tr["oos_f1_mean"] * 100, yerr=tr["oos_f1_std"] * 100,
                    marker="D", linestyle="--", label="Trainable MiniLM K=1", color=COLORS["Trainable MiniLM K=1"], capsize=2)
        ax.set(title=f"{dataset}: OOS F1 across KIR (fair components)", xlabel="KIR", ylabel="OOS F1 (%)")
        ax.set_xticks(sorted(summary["kir"].unique()))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / f"{dataset}_oos_f1_by_kir.png", dpi=180)
        plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame) -> None:
    for dataset in sorted(summary["dataset"].unique()):
        fig, ax = plt.subplots(figsize=(8, 5.4))
        part = summary[summary["dataset"].eq(dataset)]
        for _, row in part.iterrows():
            ax.scatter(row["known_recall_mean"] * 100, row["oos_f1_mean"] * 100,
                       color=COLORS.get(row["method_label"]), s=45, alpha=.85)
            if row["kir"] == .50:
                ax.annotate(f"{row['method_label']} (.50)", (row["known_recall_mean"] * 100, row["oos_f1_mean"] * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set(title=f"{dataset}: OOS F1 vs Known Recall trade-off", xlabel="Known Recall (%)", ylabel="OOS F1 (%)")
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(FIG / f"{dataset}_oos_known_tradeoff.png", dpi=180)
        plt.close(fig)


def plot_delta_heatmap(effects: pd.DataFrame) -> None:
    metrics = {"oos_f1": "OOS F1", "f1_all": "F1-All", "id_recall": "Known Recall", "false_accept_rate": "False Accept"}
    for dataset in sorted(effects["dataset"].unique()):
        for metric, label in metrics.items():
            part = effects[(effects["dataset"].eq(dataset)) & (effects["metric"].eq(metric))]
            if part.empty:
                continue
            pivot = part.pivot(index="method_label", columns="kir", values="mean_delta") * 100
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            im = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="RdYlGn" if metric != "false_accept_rate" else "RdYlGn_r", vmin=-30, vmax=30)
            ax.set_xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), pivot.index)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.iloc[i, j]
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
            ax.set(title=f"{dataset}: {label} delta vs Single centroid (pp)", xlabel="KIR")
            fig.colorbar(im, ax=ax, label="delta (pp)")
            fig.tight_layout()
            fig.savefig(FIG / f"{dataset}_delta_{metric}.png", dpi=180)
            plt.close(fig)


def cascade_decomposition() -> pd.DataFrame:
    path = ROOT / "results" / "pipeline" / "cascade_error_decomposition_summary.csv"
    df = pd.read_csv(path)
    df.to_csv(OUT / "cascade_error_decomposition.csv", index=False)
    for dataset in sorted(df["dataset"].unique()):
        part = df[(df["dataset"].eq(dataset)) & (df["gate"].isin(["frozen_k1", "ce_recon_selected_k", "best_controlled_baseline"]))]
        if part.empty:
            continue
        stages = ["known_rejected_by_gate", "oos_accepted_by_gate", "known_wrong_domain", "known_wrong_expert"]
        pivot = part[part["stage"].isin(stages)].pivot(index="gate", columns="stage", values="rate_mean").reindex(columns=stages)
        fig, ax = plt.subplots(figsize=(9, 4.7))
        pivot.mul(100).plot(kind="bar", ax=ax, color=["#d62728", "#ff9896", "#9467bd", "#8c564b"])
        ax.set_title(f"{dataset}: cascade error decomposition")
        ax.set_ylabel("rate (%)")
        ax.set_xlabel("Gate variant")
        ax.legend(fontsize=8, labels=["Known rejected by Gate", "OOS accepted by Gate", "Known wrong domain", "Known wrong expert"])
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(FIG / f"{dataset}_cascade_error_decomposition.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fair = read_fair_matrix()
    summary = summarize_fair(fair)
    effects = paired_effects(fair)
    trainable = trainable_context()
    summary.to_csv(OUT / "fair_method_kir_summary.csv", index=False)
    effects.to_csv(OUT / "fair_paired_effects.csv", index=False)
    trainable.to_csv(OUT / "trainable_k1_kir_context.csv", index=False)
    plot_oos(summary, trainable)
    plot_tradeoff(summary)
    plot_delta_heatmap(effects)
    cascade_decomposition()
    manifest = {
        "source_fair_matrix": "results/mogb/fair_matrix.csv",
        "source_trainable": "results/analysis/minilm_trainable_kir_sweep_v1/mean_std.csv",
        "bootstrap_seed": 20260725,
        "bootstrap_resamples": 10000,
        "notes": [
            "Fair rows use same protocol, split and seed pairing across methods.",
            "Trainable rows are context only: 3 seeds and a different representation regime.",
            "No test result is used to select a method or parameter.",
        ],
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"fair summary={len(summary)} rows; paired effects={len(effects)} rows; figures={len(list(FIG.glob('*.png')))}")


if __name__ == "__main__":
    main()
