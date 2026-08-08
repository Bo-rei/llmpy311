"""Build a protocol-layered summary without ranking incompatible contracts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "unified_layered_summary_v1"


def _normalise_dataset(value: str) -> str:
    value = str(value).lower()
    return "banking77" if value.startswith("banking77") else value


def trainable_rows() -> pd.DataFrame:
    src = ROOT / "results" / "analysis" / "minilm_trainable_kir_sweep_v1" / "mean_std.csv"
    df = pd.read_csv(src)
    return pd.DataFrame(
        {
            "dataset": df["dataset"],
            "kir": df["kir"],
            "method": "Trainable MiniLM K=1",
            "layer": "current_protocol",
            "representation": "last2_minilm_plus_projection",
            "partition": "single_centroid",
            "distance": "mahalanobis_diag",
            "oos_f1": df["oos_f1_mean"],
            "oos_f1_std": df["oos_f1_std"],
            "known_macro_f1": df["f1_k_mean"],
            "known_recall": df["known_recall_mean"],
            "f1_all": df["f1_all_mean"],
            "accuracy": df["accuracy_mean"],
            "scope": "protocol_v2_textoir_v1 / 3 seeds / Known-only",
        }
    )


def frozen_rows() -> pd.DataFrame:
    src = ROOT / "results" / "gate_only" / "kir_k_fixed_mean_std.csv"
    df = pd.read_csv(src)
    df = df[df["distance"].eq("mahalanobis_diag") & df["k_gate"].isin([1, 2])].copy()
    df["dataset"] = df["dataset"].map(_normalise_dataset)
    df["method"] = "Frozen MiniLM K=" + df["k_gate"].astype(int).astype(str)
    return pd.DataFrame(
        {
            "dataset": df["dataset"],
            "kir": df["kir"].astype(float) / 100.0,
            "method": df["method"],
            "layer": "current_protocol",
            "representation": "frozen_minilm",
            "partition": "fixed_kmeans",
            "distance": df["distance"],
            "oos_f1": df["test_oos_f1_mean"],
            "oos_f1_std": df["test_oos_f1_std"],
            "known_macro_f1": pd.NA,
            "known_recall": df["test_id_recall_mean"],
            "f1_all": pd.NA,
            "accuracy": pd.NA,
            "scope": "protocol_v2_textoir_v1 / 3 seeds / Gate-only",
        }
    )


def mogb_rows() -> pd.DataFrame:
    src = ROOT / "results" / "mogb" / "fair_matrix.csv"
    df = pd.read_csv(src)
    names = {
        "single_centroid": "Frozen single centroid",
        "fixed_k2": "Frozen fixed K=2",
        "random_partition": "Frozen random partition K=2",
        "mogb_minilm": "MOGB-MiniLM",
        "mogb_partition_ours_boundary": "MOGB partition + s2c boundary",
        "ours_partition_mogb_boundary": "s2c partition + MOGB boundary",
    }
    df = df[df["method"].isin(names)].copy()
    grouped = (
        df.groupby(["dataset", "kir", "method"], as_index=False)
        .agg(
            oos_f1=("oos_f1", "mean"),
            oos_f1_std=("oos_f1", "std"),
            known_macro_f1=("f1_k", "mean"),
            known_recall=("id_recall", "mean"),
            f1_all=("f1_all", "mean"),
            accuracy=("accuracy", "mean"),
        )
    )
    return pd.DataFrame(
        {
            "dataset": grouped["dataset"],
            "kir": grouped["kir"],
            "method": grouped["method"].map(names),
            "layer": "current_protocol",
            "representation": "frozen_minilm",
            "partition": grouped["method"].map(
                {
                    "single_centroid": "single",
                    "fixed_k2": "fixed_kmeans",
                    "random_partition": "random_balanced",
                    "mogb_minilm": "mogb_adaptive",
                    "mogb_partition_ours_boundary": "mogb_adaptive",
                    "ours_partition_mogb_boundary": "fixed_kmeans",
                }
            ),
            "distance": "mixed_by_method",
            "oos_f1": grouped["oos_f1"],
            "oos_f1_std": grouped["oos_f1_std"],
            "known_macro_f1": grouped["known_macro_f1"],
            "known_recall": grouped["known_recall"],
            "f1_all": grouped["f1_all"],
            "accuracy": grouped["accuracy"],
            "scope": "protocol_v2_textoir_v1 / frozen MiniLM / 5 seeds",
        }
    )


def external_rows() -> pd.DataFrame:
    src = ROOT / "results" / "final_baselines" / "summary.csv"
    df = pd.read_csv(src)
    df = df[df["method"].isin(["ADB", "DA-ADB", "BRAK", "MOGB-official (strict single-cell)", "DCLOOS-official (reduced-budget recovered)"])].copy()
    return pd.DataFrame(
        {
            "dataset": df["dataset"],
            "kir": df["kir"],
            "method": df["method"],
            "layer": "external_compatibility",
            "representation": df["training_regime"],
            "partition": "method_specific",
            "distance": "method_specific",
            "oos_f1": df["oos_f1"],
            "oos_f1_std": pd.NA,
            "known_macro_f1": df["known_macro_f1"],
            "known_recall": df["known_recall"],
            "f1_all": df["f1_all"],
            "accuracy": df["accuracy"],
            "scope": df["scope"].fillna("external compatibility"),
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = pd.concat([trainable_rows(), frozen_rows(), mogb_rows(), external_rows()], ignore_index=True)
    out = out.sort_values(["dataset", "kir", "layer", "method"], kind="stable")
    out.to_csv(OUT / "all_layers.csv", index=False)
    (OUT / "README.md").write_text(
        "# 分层汇总\n\n"
        "该目录只合并可追溯的轻量统计，不把 current protocol、MOGB fair component、"
        "external compatibility 和 fulltex 历史 Cascade 混成一个公平排名。\n\n"
        "- `current_protocol`：protocol_v2_textoir_v1 下的 Trainable/Frozen/MOGB fair rows。\n"
        "- `external_compatibility`：ADB、DA-ADB、BRAK、MOGB strict、DCLOOS reduced 等兼容性行，"
        "表示、监督或 seed 合同不同。\n"
        "- `fulltex.tex` 历史 Cascade 不在 CSV 中，需参见 `docs/analysis/MINILM_TRAINABLE_VS_FULLTEX_AND_BASELINES_V1.md`。\n",
        encoding="utf-8",
    )
    print(f"wrote {len(out)} rows to {OUT / 'all_layers.csv'}")


if __name__ == "__main__":
    main()
