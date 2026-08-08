"""Create protocol-separated CSV summaries and figures from existing results.

This is an analysis-only layer. It does not train models or modify historical
artifacts, and it labels incompatible external results instead of pooling them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2c-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "active_experiment_dashboard_v1"
FIG = ROOT / "figures" / "active_experiment_dashboard_v1"
REPORT = ROOT / "docs" / "analysis" / "ACTIVE_EXPERIMENT_REPORT.md"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_name(value: object) -> str:
    return {"banking77_oos": "banking77", "oos": "unknown_pool"}.get(str(value), str(value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def build_gate() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "results/gate_only/kir_k_fixed_mean_std.csv"
    raw = read_csv(path)
    if raw.empty:
        return raw, raw
    frame = raw.loc[raw["phase"].eq("fixed")].copy()
    frame["dataset"] = frame["dataset"].map(dataset_name)
    frame["kir"] = frame["kir"].astype(float) / 100.0
    frame["method"] = "fixed_k" + frame["k_gate"].astype(int).astype(str)
    frame["scope"] = "gate_only_frozen_minilm"
    frame["protocol"] = "protocol_v2_textoir_v1"
    frame["source"] = str(path.relative_to(ROOT))
    frame = frame.rename(columns={"seed_count": "n", "test_oos_f1_mean": "oos_f1", "test_oos_f1_std": "oos_f1_std", "test_id_recall_mean": "known_recall", "test_id_recall_std": "known_recall_std", "test_auroc_mean": "auroc", "test_auroc_std": "auroc_std", "test_aupr_oos_mean": "aupr_oos", "test_aupr_oos_std": "aupr_oos_std"})
    cols = ["dataset", "kir", "method", "scope", "protocol", "n", "source", "oos_f1", "oos_f1_std", "known_recall", "known_recall_std", "auroc", "auroc_std", "aupr_oos", "aupr_oos_std", "distance", "k_gate"]
    return frame[cols], raw


def build_representation() -> pd.DataFrame:
    path = ROOT / "results/representation/representation_fixed_results.csv"
    frame = read_csv(path)
    if frame.empty:
        return frame
    frame = frame.loc[(frame["status"] == "complete") & frame["k_variant"].isin(["k1", "k2"])].copy()
    frame["dataset"] = frame["dataset"].map(dataset_name)
    frame["kir"] = frame["kir"].astype(float) / 100.0
    frame["method"] = "representation_" + frame["representation"].astype(str)
    frame["scope"] = "gate_only_representation_control"
    frame["protocol"] = "protocol_v2_textoir_v1"
    frame["source"] = str(path.relative_to(ROOT))
    return frame.rename(columns={"data_seed": "seed", "id_recall": "known_recall"})


def build_mogb() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "results/mogb/fair_matrix.csv"
    raw = read_csv(path)
    if raw.empty:
        return raw, raw
    raw["dataset"] = raw["dataset"].map(dataset_name)
    raw["scope"] = "fair_frozen_minilm_component"
    raw["protocol"] = "protocol_v2_textoir_v1"
    raw["source"] = str(path.relative_to(ROOT))
    summary = raw.groupby(["dataset", "kir", "method"], as_index=False).agg(n=("seed", "count"), oos_f1=("oos_f1", "mean"), oos_f1_std=("oos_f1", "std"), f1_all=("f1_all", "mean"), f1_all_std=("f1_all", "std"), f1_k=("f1_k", "mean"), f1_k_std=("f1_k", "std"), accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"), known_recall=("id_recall", "mean"), known_recall_std=("id_recall", "std"), false_accept_rate=("false_accept_rate", "mean"), false_accept_rate_std=("false_accept_rate", "std"), false_reject_rate=("false_reject_rate", "mean"), false_reject_rate_std=("false_reject_rate", "std"), auroc=("auroc", "mean"), auroc_std=("auroc", "std"), aupr_oos=("aupr_oos", "mean"), aupr_oos_std=("aupr_oos", "std"))
    summary["scope"] = "fair_frozen_minilm_component"
    summary["protocol"] = "protocol_v2_textoir_v1"
    summary["source"] = str(path.relative_to(ROOT))
    return summary, raw


def build_baselines() -> pd.DataFrame:
    path = ROOT / "results/final_baselines/summary.csv"
    frame = read_csv(path)
    if frame.empty:
        return frame
    frame["dataset"] = frame["dataset"].map(dataset_name)
    frame["source"] = str(path.relative_to(ROOT))
    def classify(value: object) -> str:
        text = str(value)
        if "protocol_v2_fair" in text:
            return "fair_frozen_minilm_component"
        if "modernized" in text:
            return "external_compatibility_single_cell"
        if "official" in text:
            return "official_or_negative_reproduction"
        if "brak" in text:
            return "known_only_control"
        return "historical_or_other"
    frame["comparability"] = frame["scope"].map(classify)
    return frame


def trainable_row(path: Path, row: pd.Series, scope: str) -> dict[str, object]:
    def get(*names: str) -> object:
        for name in names:
            if name in row.index:
                return row[name]
        return np.nan
    return {"dataset": "stackoverflow", "kir": 0.50, "method": str(row.get("method", row.get("variant", "unknown"))), "scope": scope, "protocol": "protocol_v2_textoir_v1", "n": get("n_seeds", "seed_count"), "source": str(path.relative_to(ROOT)), "oos_f1": get("oos_f1_mean"), "oos_f1_std": get("oos_f1_std"), "f1_all": get("f1_all_mean"), "f1_all_std": get("f1_all_std"), "f1_k": get("f1_k_mean"), "f1_k_std": get("f1_k_std"), "accuracy": get("accuracy_mean"), "accuracy_std": get("accuracy_std"), "known_recall": get("known_recall_mean"), "known_recall_std": get("known_recall_std"), "false_accept_rate": get("false_acceptance_mean", "false_accept_rate_mean"), "false_accept_rate_std": get("false_acceptance_std", "false_accept_rate_std"), "false_reject_rate": get("false_reject_rate_mean"), "false_reject_rate_std": get("false_reject_rate_std"), "auroc": get("auroc_mean"), "auroc_std": get("auroc_std"), "aupr_oos": get("aupr_oos_mean"), "aupr_oos_std": get("aupr_oos_std")}


def build_trainable() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    inputs = [(ROOT / "results/diagnostics/racal_v1/RACAL_V1_STAGE1_MEAN_STD.csv", "gate_only_trainable_minilm"), (ROOT / "results/diagnostics/racal_v1/stage2_fixed_k2/RACAL_V1_STAGE2_MEAN_STD.csv", "gate_only_trainable_minilm"), (ROOT / "results/diagnostics/joint_adaptive_multicenter_v1/pilot_summary.csv", "diagnostic_pilot"), (ROOT / "results/diagnostics/joint_adaptive_multicenter_contract_repair_v1/summary.csv", "diagnostic_pilot"), (ROOT / "results/diagnostics/consistency_gate_v1/summary.csv", "diagnostic_pilot")]
    for path, scope in inputs:
        frame = read_csv(path)
        for _, row in frame.iterrows():
            item = trainable_row(path, row, scope)
            if "variant" in frame.columns:
                item["method"] = "consistency_" + str(row["variant"])
            rows.append(item)
    return pd.DataFrame(rows)


def build_trainable_lambda_deltas() -> pd.DataFrame:
    """Load the completed Known-only lambda/K control without recomputation."""
    path = ROOT / "results/diagnostics/minilm_trainable_lambda_control_v1/k_delta_by_lambda.csv"
    frame = read_csv(path)
    if frame.empty:
        return frame
    frame["source"] = str(path.relative_to(ROOT))
    return frame


def build_latex_metric_audit() -> pd.DataFrame:
    """Materialize the historical paper-table override as explicit evidence."""
    path = ROOT.parent / "artifacts/s2c/outputs/paper_results/stackoverflow/kir50_seed42/full_anchor/eval_results.json"
    summary_path = ROOT.parent / "artifacts/s2c/outputs/paper_results/ablation_summary.csv"
    payload = read_json(path)
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    primary = metrics.get("primary_metrics", {}) if isinstance(metrics, dict) else {}
    audit = read_csv(summary_path)
    row = audit.loc[
        (audit.get("dataset", pd.Series(dtype=str)).astype(str).str.upper() == "STACKOVERFLOW")
        & audit.get("variant", pd.Series(dtype=str)).astype(str).eq("full_anchor")
        & audit.get("kir_tag", pd.Series(dtype=str)).astype(str).eq("kir50_seed42")
    ]
    paper_oos = float(row["oos_f1"].iloc[0]) if not row.empty and pd.notna(row["oos_f1"].iloc[0]) else np.nan
    paper_acc = float(row["overall_accuracy"].iloc[0]) if not row.empty and pd.notna(row["overall_accuracy"].iloc[0]) else np.nan
    return pd.DataFrame([
        {"dataset": "stackoverflow", "kir": 0.50, "metric": "oos_f1", "paper_table_value": paper_oos, "json_metrics_value": metrics.get("oos_f1"), "raw_primary_value": primary.get("oos_f1"), "override_source": row["metric_override_source"].iloc[0] if not row.empty else "", "source": str(path.relative_to(ROOT.parent)), "interpretation": "历史表面值；不作为当前 Gate-only 公平排名"},
        {"dataset": "stackoverflow", "kir": 0.50, "metric": "overall_accuracy", "paper_table_value": paper_acc, "json_metrics_value": metrics.get("overall_accuracy"), "raw_primary_value": primary.get("overall_accuracy"), "override_source": row["metric_override_source"].iloc[0] if not row.empty else "", "source": str(path.relative_to(ROOT.parent)), "interpretation": "历史表面值；不作为当前 Gate-only 公平排名"},
    ])


def build_current_cascade() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT.parent / "artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/cascade_summary.csv"
    raw = read_csv(path)
    if raw.empty:
        return raw, raw
    raw["dataset"] = raw["dataset"].map(dataset_name)
    raw["kir"] = 0.50
    raw["scope"] = "current_protocol_cascade_kir50"
    raw["protocol"] = "protocol_v2_textoir_v1"
    raw["source"] = str(path.relative_to(ROOT.parent))
    summary = raw.groupby(["dataset", "gate"], as_index=False).agg(
        n=("kir_seed", "count"),
        oos_f1=("oos_f1", "mean"),
        oos_f1_std=("oos_f1", "std"),
        known_macro_f1=("known_macro_f1", "mean"),
        known_macro_f1_std=("known_macro_f1", "std"),
        accuracy=("overall_accuracy", "mean"),
        accuracy_std=("overall_accuracy", "std"),
        known_recall=("id_recall", "mean"),
        known_recall_std=("id_recall", "std"),
        false_accept_rate=("oos_false_accept_rate", "mean"),
        false_accept_rate_std=("oos_false_accept_rate", "std"),
        false_reject_rate=("known_false_reject_rate", "mean"),
        false_reject_rate_std=("known_false_reject_rate", "std"),
        router_error_rate=("router_error_rate", "mean"),
        expert_error_rate=("expert_error_rate", "mean"),
    )
    summary["kir"] = 0.50
    summary["scope"] = "current_protocol_cascade_kir50"
    summary["protocol"] = "protocol_v2_textoir_v1"
    summary["source"] = str(path.relative_to(ROOT.parent))
    return summary, raw


def build_k_selection(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize oracle and Known-recall-constrained K choices.

    This is descriptive analysis of the already completed fixed-K sweep.  The
    ``oracle_best_k`` column must not be used as a validation selection rule.
    ``safe_best_k`` is only a diagnostic reference using a predeclared 1 pp
    Known-recall tolerance.
    """
    if raw.empty:
        return pd.DataFrame()
    frame = raw.loc[raw["phase"].eq("fixed")].copy()
    frame["dataset"] = frame["dataset"].map(dataset_name)
    frame["kir"] = frame["kir"].astype(float) / 100.0
    rows: list[dict[str, object]] = []
    for (dataset, kir, distance), group in frame.groupby(["dataset", "kir", "distance"]):
        group = group.sort_values("k_gate").copy()
        base = group.loc[group["k_gate"].eq(1)]
        if base.empty:
            continue
        base_row = base.iloc[0]
        oracle = group.sort_values(["test_oos_f1_mean", "k_gate"], ascending=[False, True]).iloc[0]
        safe = group.loc[group["test_id_recall_mean"] >= float(base_row["test_id_recall_mean"]) - 0.01]
        safe_best = safe.sort_values(["test_oos_f1_mean", "k_gate"], ascending=[False, True]).iloc[0] if not safe.empty else None
        rows.append({
            "dataset": dataset,
            "kir": float(kir),
            "distance": distance,
            "k1_oos_f1": float(base_row["test_oos_f1_mean"]),
            "k1_known_recall": float(base_row["test_id_recall_mean"]),
            "oracle_best_k": int(oracle["k_gate"]),
            "oracle_best_oos_f1": float(oracle["test_oos_f1_mean"]),
            "oracle_oos_delta_pp": (float(oracle["test_oos_f1_mean"]) - float(base_row["test_oos_f1_mean"])) * 100.0,
            "oracle_known_recall_delta_pp": (float(oracle["test_id_recall_mean"]) - float(base_row["test_id_recall_mean"])) * 100.0,
            "safe_best_k_1pp": int(safe_best["k_gate"]) if safe_best is not None else np.nan,
            "safe_best_oos_f1": float(safe_best["test_oos_f1_mean"]) if safe_best is not None else np.nan,
            "safe_best_oos_delta_pp": (float(safe_best["test_oos_f1_mean"]) - float(base_row["test_oos_f1_mean"])) * 100.0 if safe_best is not None else np.nan,
            "safe_best_known_recall_delta_pp": (float(safe_best["test_id_recall_mean"]) - float(base_row["test_id_recall_mean"])) * 100.0 if safe_best is not None else np.nan,
            "safe_candidate_count": int(len(safe)),
            "selection_note": "descriptive_test_oracle; not a formal validation selection rule",
        })
    return pd.DataFrame(rows)


def build_component_gaps(baselines: pd.DataFrame) -> pd.DataFrame:
    """Compute fair frozen-MiniLM component deltas against Single centroid."""
    if baselines.empty:
        return pd.DataFrame()
    fair = baselines.loc[
        baselines["comparability"].eq("fair_frozen_minilm_component")
        & baselines["kir"].astype(str).eq("0.5")
    ].copy()
    if fair.empty:
        return fair
    metrics = ["oos_f1", "f1_all", "known_macro_f1", "known_recall", "accuracy"]
    rows: list[dict[str, object]] = []
    for dataset, group in fair.groupby("dataset"):
        base = group.loc[group["method"].eq("Single centroid")]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, row in group.iterrows():
            item: dict[str, object] = {
                "dataset": dataset,
                "kir": 0.50,
                "method": row["method"],
                "baseline_method": "Single centroid",
                "protocol": row.get("scope", "unknown"),
                "source": row["source"],
            }
            for metric in metrics:
                value = pd.to_numeric(row.get(metric), errors="coerce")
                base_value = pd.to_numeric(base_row.get(metric), errors="coerce")
                item[f"{metric}_delta_pp"] = (float(value) - float(base_value)) * 100.0 if pd.notna(value) and pd.notna(base_value) else np.nan
            rows.append(item)
    return pd.DataFrame(rows)


def plot_k_sweep(raw: pd.DataFrame) -> str:
    gate = raw.copy()
    gate["dataset"] = gate["dataset"].map(dataset_name)
    gate["kir"] = gate["kir"].astype(float) / 100.0
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharey=True)
    colors = {25: "#1b9e77", 50: "#d95f02", 75: "#7570b3"}
    # ``normalize_dataset`` maps the legacy BANKING77-OOS label to
    # ``banking77``; use the normalized name here so the panel is populated.
    for i, dataset in enumerate(["clinc150", "banking77", "stackoverflow"]):
        for j, distance in enumerate(["euclidean", "mahalanobis_diag"]):
            ax = axes[i, j]
            sub = gate[(gate["dataset"] == dataset) & (gate["distance"] == distance)]
            for kir, group in sub.groupby("kir"):
                group = group.sort_values("k_gate")
                # ``kir`` was normalized to [0, 1] above; display the
                # requested percentage rather than truncating it to 0/1.
                pct = int(round(float(kir) * 100))
                ax.errorbar(group["k_gate"], group["test_oos_f1_mean"] * 100, yerr=group["test_oos_f1_std"] * 100, marker="o", capsize=3, color=colors.get(pct, "#444444"), label=f"KIR {pct}%")
            ax.set_title(f"{dataset} / {distance}")
            ax.set_xlabel("K")
            ax.set_ylabel("OOS F1 (%)")
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.grid(alpha=0.25)
            if i == 0 and j == 1:
                ax.legend(fontsize=8)
    fig.suptitle("固定多中心：K 与 KIR 对 OOS F1 的影响", fontsize=14)
    fig.tight_layout()
    path = FIG / "k_sweep_oos_f1.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_trainable(frame: pd.DataFrame) -> str:
    names = ["frozen_k1", "trainable_k1", "trainable_fixed_k2"]
    labels = ["Frozen K=1", "Trainable K=1", "Trainable K=2"]
    metrics = [("oos_f1", "OOS F1 (%)", "oos_f1_std"), ("f1_all", "F1-All (%)", "f1_all_std"), ("known_recall", "Known Recall (%)", "known_recall_std"), ("false_accept_rate", "False acceptance (%)", "false_accept_rate_std")]
    sub = frame.drop_duplicates("method").set_index("method").reindex(names)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    for ax, (metric, title, error) in zip(axes, metrics):
        ax.bar(np.arange(3), sub[metric].to_numpy(float) * 100, yerr=sub[error].to_numpy(float) * 100, capsize=4, color=["#8da0cb", "#1b9e77", "#d95f02"])
        ax.set_xticks(np.arange(3), labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("StackOverflow KIR=0.50：表示训练改善 K=1，但固定 K=2 失效", fontsize=14)
    fig.tight_layout()
    path = FIG / "trainable_k1_k2_tradeoff.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_representation(frame: pd.DataFrame) -> str:
    data = frame.loc[frame["kir"].eq(0.50)].copy()
    reps = ["frozen", "ce", "supcon", "ce_recon", "geometry_ce_recon"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, dataset in zip(axes, ["clinc150", "banking77", "stackoverflow"]):
        stats = data[data["dataset"] == dataset].groupby(["representation", "k_gate"], as_index=False).agg(mean=("oos_f1", "mean"), std=("oos_f1", "std"))
        x = np.arange(len(reps))
        for j, k in enumerate([1, 2]):
            vals, errs = [], []
            for rep in reps:
                hit = stats[(stats["representation"] == rep) & (stats["k_gate"] == k)]
                vals.append(float(hit["mean"].iloc[0]) * 100 if not hit.empty else np.nan)
                errs.append(float(hit["std"].iloc[0]) * 100 if not hit.empty else 0.0)
            ax.bar(x + (j - 0.5) * 0.36, vals, 0.36, yerr=errs, capsize=3, label=f"K={k}")
        ax.set_title(dataset)
        ax.set_xticks(x, [r.replace("geometry_", "geom_") for r in reps], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if dataset == "clinc150":
            ax.set_ylabel("OOS F1 (%)")
            ax.legend()
    fig.suptitle("MiniLM 表示训练与 K 的交互", fontsize=14)
    fig.tight_layout()
    path = FIG / "representation_k_interaction.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_baselines(frame: pd.DataFrame) -> str:
    data = frame[(frame["dataset"] == "stackoverflow") & (frame["kir"].astype(str) == "0.5")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, kind, title in [(axes[0], "fair_frozen_minilm_component", "同协议冻结 MiniLM 组件"), (axes[1], "external_compatibility_single_cell", "外部兼容性单格（不可混入主排名）")]:
        sub = data[data["comparability"] == kind].drop_duplicates("method")
        if sub.empty:
            ax.set_axis_off()
            continue
        x = np.arange(len(sub))
        ax.bar(x, sub["oos_f1"] * 100, color="#1b9e77")
        ax.set_xticks(x, sub["method"], rotation=45, ha="right")
        ax.set_ylabel("OOS F1 (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("StackOverflow KIR=0.50：结果与可比性分层", fontsize=14)
    fig.tight_layout()
    path = FIG / "stackoverflow_baseline_layers.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_stackoverflow_tradeoff(baselines: pd.DataFrame, trainable: pd.DataFrame) -> str:
    """Plot the OOS-vs-known tradeoff without pooling incompatible contracts."""
    rows: list[dict[str, object]] = []
    fair = baselines[
        (baselines["dataset"] == "stackoverflow")
        & (baselines["kir"].astype(str) == "0.5")
        & baselines["oos_f1"].notna()
        & baselines["known_macro_f1"].notna()
    ].copy()
    for _, row in fair.iterrows():
        rows.append({
            "method": str(row["method"]),
            "oos_f1": float(row["oos_f1"]),
            "known_f1": float(row["known_macro_f1"]),
            "layer": "当前/兼容结果",
        })
    for _, row in trainable[trainable["method"].isin(["frozen_k1", "trainable_k1", "trainable_fixed_k2"])].drop_duplicates("method").iterrows():
        rows.append({
            "method": str(row["method"]),
            "oos_f1": float(row["oos_f1"]),
            "known_f1": float(row["f1_k"]),
            "layer": "当前 Trainable/Frozen",
        })
    data = pd.DataFrame(rows).drop_duplicates("method")
    colors = {
        "当前/兼容结果": "#377eb8",
        "当前 Trainable/Frozen": "#1b9e77",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer, group in data.groupby("layer"):
        ax.scatter(group["known_f1"] * 100, group["oos_f1"] * 100, s=70, alpha=0.88, color=colors[layer], label=layer)
        for _, row in group.iterrows():
            short = str(row["method"]).replace("MOGB partition + s2c boundary", "MOGB-part+s2c").replace("s2c partition + MOGB boundary", "s2c-part+MOGB")
            ax.annotate(short, (row["known_f1"] * 100, row["oos_f1"] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
    # The paper-table point is intentionally separate: its source and metric
    # contract are audited in WHY_TRAINABLE_MINILM_BELOW_LATEX.md.
    ax.scatter([75.48], [89.71], marker="*", s=180, facecolors="none", edgecolors="#d95f02", linewidths=1.5, label="fulltex 表面值（不可比）")
    ax.annotate("fulltex Ours\n(历史表面值)", (75.48, 89.71), xytext=(6, -20), textcoords="offset points", fontsize=8, color="#a63603")
    ax.set_xlabel("Known Macro-F1 (%)")
    ax.set_ylabel("OOS F1 (%)")
    ax.set_title("StackOverflow KIR=0.50：Known/OOS 权衡与历史表面值")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = FIG / "stackoverflow_known_oos_tradeoff.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_current_cascade(frame: pd.DataFrame) -> str:
    data = frame.copy()
    names = list(data["gate"].drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("oos_f1", "OOS F1 (%)"), ("known_macro_f1", "Known Macro-F1 (%)"), ("known_recall", "Known Recall (%)")]
    palette = {"frozen_k1": "#8da0cb", "frozen_selected_k": "#66c2a5", "ce_recon_selected_k": "#fc8d62", "best_controlled_baseline": "#e78ac3"}
    for ax, (metric, title) in zip(axes, metrics):
        for name in names:
            sub = data[data["gate"] == name]
            if sub.empty:
                continue
            value = float(sub[metric].iloc[0]) * 100
            error_col = f"{metric}_std"
            error = float(sub[error_col].iloc[0]) * 100 if error_col in sub and pd.notna(sub[error_col].iloc[0]) else 0.0
            x = names.index(name)
            ax.bar(x, value, yerr=error, capsize=4, color=palette.get(name, "#999999"))
        ax.set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("当前 protocol_v2_textoir_v1：Gate 变化传递到完整 Cascade 的结果", fontsize=14)
    fig.tight_layout()
    path = FIG / "current_cascade_gate_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_adaptive() -> str:
    labels, attempted, accepted = ["RC-AMBL", "Joint adaptive", "Contract repair"], [6, 3, 3], [0, 0, 0]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(3)
    ax.bar(x, attempted, color="#d9d9d9", label="候选 split")
    ax.bar(x, accepted, color="#1b9e77", label="接受 split")
    for i in range(3):
        ax.text(i, attempted[i] + 0.1, "最终平均 K_y=1.0", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("split count")
    ax.set_title("StackOverflow：风险门与共同训练 pilot 的结构选择")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIG / "adaptive_decision_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_k_selection(frame: pd.DataFrame) -> str:
    """Visualize dataset/KIR-dependent K choices without calling them selected."""
    if frame.empty:
        return ""
    datasets = ["clinc150", "banking77", "stackoverflow"]
    kirs = [0.25, 0.50, 0.75]
    distances = ["euclidean", "mahalanobis_diag"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [(0, "oracle_best_k", "测试集最优 K（仅描述性）", "K"), (1, "safe_best_k_1pp", "Known Recall 允许下降 1pp 时的安全 K（描述性）", "K"), (2, "oracle_oos_delta_pp", "oracle K 相对 K=1 的 OOS F1 差值", "百分点"), (3, "oracle_known_recall_delta_pp", "oracle K 相对 K=1 的 Known Recall 差值", "百分点")]
    for panel_index, column, title, unit in panels:
        ax = axes.flat[panel_index]
        matrix = np.full((len(datasets), len(kirs)), np.nan)
        for i, dataset in enumerate(datasets):
            for j, kir in enumerate(kirs):
                vals = frame.loc[(frame["distance"] == distances[panel_index % 2]) & frame["dataset"].eq(dataset) & np.isclose(frame["kir"], kir), column]
                if not vals.empty:
                    matrix[i, j] = float(vals.iloc[0])
        image = ax.imshow(matrix, cmap="RdYlGn" if "delta" in column else "viridis", aspect="auto")
        ax.set_xticks(range(len(kirs)), [f"KIR {int(k * 100)}%" for k in kirs])
        ax.set_yticks(range(len(datasets)), datasets)
        ax.set_title(f"{title}\n({distances[panel_index % 2]})", fontsize=10)
        for i in range(len(datasets)):
            for j in range(len(kirs)):
                value = matrix[i, j]
                if np.isfinite(value):
                    text = f"{value:.1f}" if "delta" in column else f"{int(round(value))}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=unit)
    fig.suptitle("固定 K 消融：最优 K 随数据集、KIR 和距离变化（不作为正式选 K）", fontsize=13)
    fig.tight_layout()
    path = FIG / "k_selection_tradeoff.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def plot_component_gaps(frame: pd.DataFrame) -> str:
    """Show fair frozen-MiniLM component deltas against the single centroid."""
    if frame.empty:
        return ""
    datasets = ["clinc150", "banking77", "stackoverflow"]
    methods = ["Random partition", "Fixed K=2", "MOGB-MiniLM", "MOGB partition + s2c boundary", "s2c partition + MOGB boundary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, dataset in zip(axes, datasets):
        sub = frame.loc[(frame["dataset"] == dataset) & frame["method"].isin(methods)].copy()
        sub = sub.set_index("method").reindex(methods).dropna(subset=["oos_f1_delta_pp"])
        x = np.arange(len(sub))
        ax.bar(x, sub["oos_f1_delta_pp"].to_numpy(float), color=["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"][:len(sub)])
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks(x, [str(v).replace("MOGB partition + s2c boundary", "MOGB-part+s2c").replace("s2c partition + MOGB boundary", "s2c-part+MOGB") for v in sub.index], rotation=42, ha="right", fontsize=8)
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.25)
        if dataset == "clinc150":
            ax.set_ylabel("相对 Single centroid 的 OOS F1 差值（百分点）")
    fig.suptitle("同一冻结 MiniLM 下：分簇与边界组件相对单中心的变化", fontsize=14)
    fig.tight_layout()
    path = FIG / "fair_component_gaps.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(ROOT))


def fmt(value: object) -> str:
    try:
        if pd.isna(value):
            return "未记录"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "未记录"


def build_report(trainable: pd.DataFrame, figures: list[str], lambda_deltas: pd.DataFrame) -> str:
    names = ["frozen_k1", "trainable_k1", "trainable_fixed_k2"]
    lines = [
        "# 当前 s2c 实验证据总览（active_experiment_dashboard_v1）",
        "",
        "> 本报告只汇总已有轻量 CSV，不重新训练、不覆盖历史 artifact。结果先按实验合同分层，再讨论性能；外部兼容性数字不自动并入同协议主排名。",
        "",
        "中文集中版：[`EXPERIMENT_COMPARISON_ZH.md`](EXPERIMENT_COMPARISON_ZH.md)。",
        "",
        "## 1. 总体判断",
        "",
        "当前项目已经完成大量固定 K、KIR、随机分簇、聚类诊断、MiniLM 表示、MOGB 组件和端到端兼容性实验。当前主要缺口不是没有数字，而是数字属于不同实验合同。现有证据支持：",
        "",
        "- 固定 K 的收益依赖数据集和 KIR，不存在跨数据集统一最优 K；",
        "- Banking77 在部分条件下从多中心获益，CLINC150 收益小且不稳定，StackOverflow 固定多中心持续过覆盖；",
        "- Known-only Trainable MiniLM 对 K=1 有稳定正收益，但没有自动修复 K=2 的多球 false acceptance；",
        "- 训练参与式 joint-adaptive pilot 已真正执行，但候选 split 全部被 Known calibration 安全门拒绝，最终 K_y=1；",
        "- MOGB 官方 BERT、ADB/DA-ADB 和 DCLOOS 当前主要是严格负复现或兼容性证据，不能和冻结 MiniLM fair matrix 直接排名。",
        "",
        "## 2. StackOverflow 当前最关键对照",
        "",
        "| 方法 | OOS F1 | F1-All | Known Recall | False acceptance | 协议层 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in names:
        hit = trainable[trainable["method"].eq(name)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        lines.append(f"| {name} | {fmt(row['oos_f1'])} | {fmt(row['f1_all'])} | {fmt(row['known_recall'])} | {fmt(row['false_accept_rate'])} | 当前 Gate |")
    lines += [
        "",
        "Trainable K=1 相对 Frozen K=1 的 OOS F1 提升约 9.42 个百分点，说明表示适配本身有效；Trainable K=2 的 OOS F1 大幅下降并伴随 false acceptance 上升，说明当前瓶颈是多球接受区域组合，而不是单纯缺少训练。",
        "",
        "## 3. K/KIR 与数据集差异",
        "",
        "`k_sweep_oos_f1.png` 展示固定 Gate-only 结果：CLINC150 通常在 K=2 附近达到局部峰值后回落；Banking77 在部分距离下随 K 增大受益但 Known 覆盖有代价；StackOverflow 从 K>1 开始显著退化，且 KIR 越高过覆盖风险越明显。",
        "",
        "这说明多中心是否有效不是单一方法属性，而是数据集语义结构、表示空间、半径规则和接受区域并集共同决定的结果。",
        "",
        "`k_selection_tradeoff.png` 和 `k_selection_summary.csv` 将测试集 oracle 最优 K 与 Known Recall 约束下的诊断 K 分开记录；这些结果只用于解释 KIR/数据集异质性，不能被当作正式验证选 K。",
        "`fair_component_gaps.png` 和 `fair_component_gaps.csv` 则在同一冻结 MiniLM、同一 KIR=0.50 下，以 Single centroid 为基准分解随机分簇、固定 K=2、MOGB 粒球和边界替换的增量。",
        "",
        "## 4. MiniLM 表示实验",
        "",
        "表示对照图同时画 Frozen、CE、SupCon、CE-Recon 及其 K=1/K=2 结果，用于区分表示训练能否改善单中心 OOS 排序，以及这种改善能否传递到固定多中心边界。当前已有结果显示前者较明确，后者不稳定。",
        "",
        "## 5. MOGB 与外部基线",
        "",
        "同协议 MOGB fair matrix 可用于组件归因；官方 BERT MOGB 未复现论文参考数值；ADB/DA-ADB 只有单 seed compatibility artifact；DCLOOS 使用 pseudo-OOS 与外部 OOS，且正式运行未完成。因此不能用一张柱状图宣称我的方法已经超过 MOGB/DCLOOS。",
        "",
        "`stackoverflow_known_oos_tradeoff.png` 将 Known Macro-F1 与 OOS F1 放在同一坐标系，并把 `fulltex.tex` 的历史表面值单独标为不可比点；它不是当前协议的 SOTA 排名。",
        "`historical_latex_metric_audit.csv` 逐项记录论文表值、JSON 覆盖值和 raw `primary_metrics`，用于防止历史 override 被误当成当前可复算结果。",
        "`current_cascade_gate_comparison.png` 只显示当前 36-unit Cascade 的 Gate 对照，不与历史论文表混合。",
        "",
        "## 6. KIR=0.50 方法分层对照",
        "",
        "`KIR50_METHOD_COMPARISON_V1.md` 将 Trainable K=1/K=2、冻结 MiniLM 组件和 ADB/DA-ADB/BRAK 兼容结果放在同一张分层表中。StackOverflow 的 Trainable K=1 为 86.71%，高于同协议 Frozen Single centroid 76.55%、MOGB-MiniLM 72.92% 和 MOGB partition+s2c boundary 79.25%；ADB/DA-ADB 分别为 89.47%/90.90%，但属于 BERT/不同训练合同的兼容单格，不能直接视为公平超越或落后。",
        "详见 `docs/analysis/KIR50_METHOD_COMPARISON_V1.md`、`kir50_method_layers.png` 和 `kir50_method_tradeoff.png`。",
        "",
        "## 7. Trainable MiniLM 的 λ/K 受控分析",
        "",
        "`minilm_trainable_lambda_control_v1` 在同一 checkpoint 上评价 λ={0.50,0.75,1.00,1.25,1.50,2.00}，选择规则只使用 Known calibration。",
    ]
    if not lambda_deltas.empty:
        for _, row in lambda_deltas.loc[lambda_deltas["selection"].eq("known_only_selected")].sort_values("dataset").iterrows():
            lines.append(
                f"- {row['dataset']}：K=2−K=1 OOS F1 `{float(row['k2_minus_k1_test_oos_f1_mean']) * 100:+.2f}pp`，"
                f"Known Recall `{float(row['k2_minus_k1_test_known_recall_mean']) * 100:+.2f}pp`，"
                f"false acceptance `{float(row['k2_minus_k1_test_false_accept_rate_mean']) * 100:+.2f}pp`。"
            )
    lines += [
        "这说明 StackOverflow 的 K=2 退化不是 λ=1 单点设置造成，Trainable MiniLM 的主要收益仍属于 K=1 表示和分数排序。",
        "详见 `docs/analysis/MINILM_TRAINABLE_LAMBDA_CONTROL_V1.md` 和 `trainable_lambda_k_interaction.png`。",
        "",
        "## 8. 自适应多中心的实际结果",
        "",
        "`adaptive_decision_summary.png` 显示 RC-AMBL、joint adaptive 和 contract repair 的候选 split 均被拒绝，最终平均 K_y=1。当前实现已经包含候选分裂、共同训练、Known-only calibration 选择和安全回退，但 StackOverflow 上没有找到安全的新增中心。",
        "",
        "## 9. 下一步实验优先级",
        "",
        "1. 对已有 Trainable/Frozen/MOGB fair rows 做逐数据集、逐 KIR、逐 seed 的主表与置信区间汇总；",
        "2. 对 MOGB fair matrix 做逐意图 false-accept/false-reject 归因，不继续盲目扩大官方 BERT 复现；",
        "3. 在统一监督条件、数据划分和随机种子后，再把 ADB、DA-ADB、DCLOOS 纳入正式比较；",
        "4. 接入完整 Cascade 前先冻结 Gate 候选，分别验证 Frozen K=1 与 Trainable K=1 的下游传递；",
        "5. 继续多中心前保留 Trainable K=1 为安全基线，新增中心必须通过 Known-only 风险门。",
        "",
        "## 10. 图表文件",
        "",
    ]
    lines.extend(f"- `{figure}`" for figure in figures)
    lines.extend(["", "历史 R1/R1-full 中已被 contract audit 标记为 superseded 或 exploratory 的几何和 test-defined near-OOS 结果不进入本报告正式结论。", ""])
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    gate, gate_raw = build_gate()
    representation = build_representation()
    mogb, mogb_raw = build_mogb()
    baselines = build_baselines()
    trainable = build_trainable()
    latex_audit = build_latex_metric_audit()
    cascade, cascade_raw = build_current_cascade()
    lambda_deltas = build_trainable_lambda_deltas()
    if not gate.empty:
        atomic_csv(gate, OUT / "gate_k_sweep.csv")
        atomic_csv(gate_raw, OUT / "gate_k_sweep_raw.csv")
    if not representation.empty:
        atomic_csv(representation, OUT / "representation_rows.csv")
    if not mogb.empty:
        atomic_csv(mogb, OUT / "mogb_component_summary.csv")
        atomic_csv(mogb_raw, OUT / "mogb_component_rows.csv")
    if not baselines.empty:
        atomic_csv(baselines, OUT / "baseline_layers.csv")
    if not trainable.empty:
        atomic_csv(trainable, OUT / "trainable_and_adaptive_rows.csv")
    atomic_csv(latex_audit, OUT / "historical_latex_metric_audit.csv")
    if not cascade.empty:
        atomic_csv(cascade, OUT / "current_cascade_summary.csv")
        atomic_csv(cascade_raw, OUT / "current_cascade_rows.csv")
    if not lambda_deltas.empty:
        atomic_csv(lambda_deltas, OUT / "trainable_lambda_deltas.csv")
    k_selection = build_k_selection(gate_raw)
    component_gaps = build_component_gaps(baselines)
    atomic_csv(k_selection, OUT / "k_selection_summary.csv")
    atomic_csv(component_gaps, OUT / "fair_component_gaps.csv")
    overview = pd.concat([gate, mogb, trainable], ignore_index=True, sort=False)
    atomic_csv(overview, OUT / "experiment_overview.csv")
    figures = []
    if not gate.empty:
        figures.append(plot_k_sweep(gate_raw))
    if not trainable.empty:
        figures.append(plot_trainable(trainable))
    if not representation.empty:
        figures.append(plot_representation(representation))
    if not baselines.empty:
        figures.append(plot_baselines(baselines))
        figures.append(plot_stackoverflow_tradeoff(baselines, trainable))
    if not cascade.empty:
        figures.append(plot_current_cascade(cascade))
    if not k_selection.empty:
        figures.append(plot_k_selection(k_selection))
    if not component_gaps.empty:
        figures.append(plot_component_gaps(component_gaps))
    lambda_figure = FIG / "trainable_lambda_k_interaction.png"
    if lambda_figure.is_file():
        figures.append(str(lambda_figure.relative_to(ROOT)))
    for comparison_figure in (FIG / "kir50_method_layers.png", FIG / "kir50_method_tradeoff.png"):
        if comparison_figure.is_file():
            figures.append(str(comparison_figure.relative_to(ROOT)))
    figures.append(plot_adaptive())
    source_paths = ["results/gate_only/kir_k_fixed_mean_std.csv", "results/representation/representation_fixed_results.csv", "results/mogb/fair_matrix.csv", "results/final_baselines/summary.csv", "results/analysis/kir50_method_comparison_v1/rows.csv", "results/diagnostics/racal_v1/RACAL_V1_STAGE1_MEAN_STD.csv", "results/diagnostics/racal_v1/stage2_fixed_k2/RACAL_V1_STAGE2_MEAN_STD.csv", "results/diagnostics/joint_adaptive_multicenter_v1/pilot_summary.csv", "results/diagnostics/joint_adaptive_multicenter_contract_repair_v1/summary.csv", "results/diagnostics/consistency_gate_v1/summary.csv", "results/diagnostics/minilm_trainable_lambda_control_v1/k_delta_by_lambda.csv", "../artifacts/s2c/outputs/paper_results/stackoverflow/kir50_seed42/full_anchor/eval_results.json", "../artifacts/s2c/outputs/paper_results/ablation_summary.csv", "../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/cascade_summary.csv"]
    sources = {p: sha256(ROOT / p) for p in source_paths if (ROOT / p).is_file()}
    manifest = {"schema": "s2c.active_experiment_dashboard_v1", "existing_results_only": True, "sources": sources, "figures": figures}
    atomic_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), OUT / "DASHBOARD_MANIFEST.json")
    atomic_text(build_report(trainable, figures, lambda_deltas), REPORT)
    print(json.dumps({"overview_rows": len(overview), "figures": figures, "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
