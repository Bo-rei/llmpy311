#!/usr/bin/env python3
"""Audit compatible external-baseline cells without mixing contracts.

This report is intentionally separate from the current fair Gate matrix.  ADB
and DA-ADB consume the legacy TextOIR BERT runner, while the S2C rows consume
the protocol_v2 MiniLM Gate runner.  The script derives binary OOS metrics from
the saved per-sample predictions and marks invalid semantic runs explicitly.
It never uses the external ``results.csv`` as the source of truth when that
file disagrees with ``y_true.npy``/``y_pred.npy``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = ROOT.parent / "artifacts/s2c/external/adb_protocol_v2_probe/stackoverflow"
FAIR_PER_SEED = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
OUTPUT_ROOT = ROOT / "results/analysis/archive/analysis/comparison_atlas_v1"
FIGURE_ROOT = ROOT / "figures/archive/analysis/comparison_atlas_v1"

_CJK_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
if _CJK_FONT.is_file():
    font_manager.fontManager.addfont(str(_CJK_FONT))
    _CJK_FAMILY = font_manager.FontProperties(fname=str(_CJK_FONT)).get_name()
else:
    _CJK_FAMILY = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = [_CJK_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

METHOD_LABELS = {
    "trainable_k1": "S2C Trainable K=1",
    "single_centroid": "S2C Frozen K=1",
    "fixed_k2": "S2C Frozen K=2",
    "random_partition": "S2C Random K=2",
    "mogb_minilm": "MOGB-MiniLM",
    "mogb_partition_ours_boundary": "MOGB partition + S2C boundary",
    "ours_partition_mogb_boundary": "S2C partition + MOGB boundary",
    "ADB": "ADB (same protocol cell)",
    "DA-ADB": "DA-ADB (adapted; invalid)",
}


def _safe_f1_all(y_true: np.ndarray, y_pred: np.ndarray, oos_label: int) -> float:
    return float(f1_score(y_true, y_pred, labels=list(range(oos_label + 1)), average="macro", zero_division=0))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    oos_label = int(np.max(y_true))
    known = y_true != oos_label
    oos = ~known
    predicted_oos = y_pred == oos_label
    known_recall = float(np.mean(~predicted_oos[known]))
    false_accept = float(np.mean(~predicted_oos[oos]))
    false_reject = float(np.mean(predicted_oos[known]))
    return {
        "oos_f1": float(f1_score(oos, predicted_oos, zero_division=0)),
        "f1_all": _safe_f1_all(y_true, y_pred, oos_label),
        "f1_k": float(f1_score(y_true[known], y_pred[known], labels=list(range(oos_label)), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "known_recall": known_recall,
        "false_accept_rate": false_accept,
        "false_reject_rate": false_reject,
        "auroc": np.nan,
        "aupr_oos": np.nan,
    }


def _audit_run(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    method = str(manifest.get("method", run_dir.parts[-5] if len(run_dir.parts) >= 5 else "unknown"))
    seed = int(manifest.get("seed", 0))
    y_true_paths = sorted(run_dir.rglob("y_true.npy"))
    y_pred_paths = sorted(run_dir.rglob("y_pred.npy"))
    row: dict[str, object] = {
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seed": seed,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "layer": "same_protocol_external_compatibility",
        "contract": "protocol_v2_split + legacy TextOIR BERT runner",
        "source_run_dir": str(run_dir.relative_to(ROOT.parent)),
        "manifest_status": manifest.get("status"),
        "run_return_code": manifest.get("return_code"),
        "valid_semantic_metrics": False,
        "invalid_reason": "missing_predictions",
    }
    if y_true_paths and y_pred_paths:
        y_true = np.load(y_true_paths[0])
        y_pred = np.load(y_pred_paths[0])
        if y_true.shape == y_pred.shape and y_true.size:
            metrics = _metrics(y_true, y_pred)
            row.update(metrics)
            row["valid_semantic_metrics"] = bool(np.unique(y_pred).size > 1 and np.isfinite(y_pred).all())
            row["invalid_reason"] = "" if row["valid_semantic_metrics"] else "all_class_prediction_or_nonfinite"
            row["n_test"] = int(y_true.size)
            row["n_known"] = int(np.sum(y_true != np.max(y_true)))
            row["n_oos"] = int(np.sum(y_true == np.max(y_true)))
        else:
            row["invalid_reason"] = "prediction_shape_mismatch"
    return row


def build(
    output_dir: Path = OUTPUT_ROOT,
    figure_dir: Path = FIGURE_ROOT,
    external_root: Path = EXTERNAL_ROOT,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    external_rows = [_audit_run(p.parent) for p in sorted(external_root.glob("*/kir50/*/run_manifest.json"))]
    external = pd.DataFrame(external_rows)
    external.to_csv(output_dir / "external_same_protocol_cells.csv", index=False)

    fair = pd.read_csv(FAIR_PER_SEED)
    fair = fair[(fair.dataset == "stackoverflow") & (fair.kir == 0.50) & fair.method.isin(METHOD_LABELS)].copy()
    fair["layer"] = "current_protocol_v2_fair_gate"
    fair["contract"] = "protocol_v2 frozen/trainable MiniLM Gate"
    fair["valid_semantic_metrics"] = True
    fair["manifest_status"] = "complete"
    fair["invalid_reason"] = ""
    fair["source_run_dir"] = "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
    fair["method_label"] = fair.method.map(METHOD_LABELS)
    cols = [
        "dataset", "kir", "seed", "method", "method_label", "layer", "contract",
        "source_run_dir", "manifest_status", "valid_semantic_metrics", "invalid_reason",
        "oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate",
        "false_reject_rate", "auroc", "aupr_oos",
    ]
    for col in cols:
        if col not in external:
            external[col] = np.nan
    combined = pd.concat([fair[cols], external[cols]], ignore_index=True)
    combined.to_csv(output_dir / "stackoverflow_kir50_external_and_fair_cells.csv", index=False)

    valid = combined[combined.valid_semantic_metrics == True].copy()  # noqa: E712
    valid["oos_f1_pct"] = valid.oos_f1 * 100
    valid["f1_all_pct"] = valid.f1_all * 100
    valid["known_recall_pct"] = valid.known_recall * 100
    value_columns = ["oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate", "accuracy"]
    summary = valid.groupby(["method", "method_label", "layer"], as_index=False).agg(
        **{f"{column}_mean": (column, "mean") for column in value_columns},
        **{f"{column}_std": (column, "std") for column in value_columns},
    )
    summary.to_csv(output_dir / "stackoverflow_kir50_external_summary.csv", index=False)

    pair_metrics = ["oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate", "accuracy"]
    pair_rows: list[dict[str, object]] = []
    paired = valid[valid.method.isin(["ADB", "trainable_k1"])].pivot(index="seed", columns="method", values=pair_metrics)
    for metric in pair_metrics:
        if (metric, "ADB") not in paired or (metric, "trainable_k1") not in paired:
            continue
        delta = paired[(metric, "trainable_k1")] - paired[(metric, "ADB")]
        for seed, value in delta.items():
            pair_rows.append({"seed": int(seed), "metric": metric, "trainable_minus_adb": float(value), "delta_pp": float(value * 100.0)})
    paired_effects = pd.DataFrame(pair_rows)
    paired_effects.to_csv(output_dir / "adb_vs_trainable_paired_effects.csv", index=False)
    paired_summary = paired_effects.groupby("metric", as_index=False)["delta_pp"].agg(["mean", "std"]).reset_index()
    paired_summary.to_csv(output_dir / "adb_vs_trainable_paired_effects_summary.csv", index=False)

    # A single-cell/seed audit plot, with invalid DA-ADB kept in a separate panel.
    plot_methods = ["single_centroid", "trainable_k1", "fixed_k2", "random_partition", "mogb_minilm", "mogb_partition_ours_boundary", "ADB"]
    plot = valid[valid.method.isin(plot_methods)].copy()
    means = plot.groupby(["method", "method_label"], as_index=False)[["oos_f1", "f1_all", "known_recall", "false_accept_rate"]].mean()
    means = means.set_index("method").reindex([m for m in plot_methods if m in means.method.values]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(means))
    axes[0].bar(x - 0.18, means.oos_f1 * 100, 0.36, label="OOS F1")
    axes[0].bar(x + 0.18, means.f1_all * 100, 0.36, label="F1-All")
    axes[0].set_ylabel("百分比")
    axes[0].set_title("StackOverflow KIR=0.50：有效结果")
    axes[0].set_xticks(x, means.method_label, rotation=35, ha="right")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].scatter(means.false_accept_rate * 100, means.known_recall * 100, s=55)
    for _, r in means.iterrows():
        axes[1].annotate(r.method_label, (r.false_accept_rate * 100, r.known_recall * 100), fontsize=7, xytext=(4, 3), textcoords="offset points")
    axes[1].set_xlabel("False acceptance (%)")
    axes[1].set_ylabel("Known Recall (%)")
    axes[1].set_title("接受风险与 Known 覆盖工作点")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "stackoverflow_external_single_cell_comparison.png", dpi=180)
    plt.close(fig)

    # A paired effect plot is more informative than a standalone ranking: it
    # shows whether the apparent advantage survives seed matching.
    if not paired_summary.empty:
        labels = {"oos_f1": "OOS F1", "f1_all": "F1-All", "known_recall": "Known Recall", "false_accept_rate": "False acceptance", "false_reject_rate": "False rejection", "accuracy": "Accuracy"}
        plot_summary = paired_summary.copy()
        plot_summary["label"] = plot_summary.metric.map(labels)
        fig, ax = plt.subplots(figsize=(8.0, 4.4))
        x = np.arange(len(plot_summary))
        ax.errorbar(x, plot_summary["mean"], yerr=plot_summary["std"].fillna(0), fmt="o", capsize=4, color="#2c7fb8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, plot_summary["label"], rotation=25, ha="right")
        ax.set_ylabel("Trainable K=1 - ADB（百分点）")
        ax.set_title("StackOverflow/KIR=0.50：同 seed 配对差值")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figure_dir / "adb_vs_trainable_paired_delta.png", dpi=180)
        plt.close(fig)

    invalid = combined[combined.valid_semantic_metrics != True]  # noqa: E712
    invalid.to_csv(output_dir / "external_invalid_semantic_runs.csv", index=False)
    payload = {
        "schema_version": 1,
        "tool": "build_external_single_cell_comparison_v1.py",
        "external_root": str(external_root),
        "external_rows": int(len(external)),
        "fair_rows": int(len(fair)),
        "valid_rows": int(len(valid)),
        "invalid_rows": int(len(invalid)),
        "invalid_methods": sorted(set(invalid.method.astype(str))) if not invalid.empty else [],
        "metric_source": "y_true.npy/y_pred.npy; external results.csv is not trusted when inconsistent",
        "auroc_and_aupr": "not available from hard predictions; recorded as NaN",
        "paired_effect_rows": int(len(paired_effects)),
    }
    (output_dir / "EXTERNAL_COMPARISON_MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# StackOverflow 同协议外部基线单格对比",
        "",
        "> 当前表只用于审计同一 StackOverflow/KIR=0.50 数据合同下的兼容运行，不把不同监督条件混成一个 SOTA 排名。",
        "",
        "## 结果边界",
        "",
        "- S2C 与 MOGB-MiniLM 行来自 protocol_v2_textoir_v1 的统一 fair Gate 矩阵。",
        "- ADB 行来自 TextOIR BERT 兼容 runner，但输入 split 根由 protocol_v2 导出，属于同数据/同 KIR 单格参考，不等同于 MiniLM fair Gate。",
        "- DA-ADB 适配尝试即使进程返回 0，也必须以逐样本预测审计为准；全类预测或 NaN 训练不能作为有效结果。",
        "",
        "## 当前可用观察",
        "",
        "- ADB 的同协议 seed 结果可以作为外部单格参照；它与 S2C 的差异同时包含 BERT 表示、端到端训练和边界训练，不能归因于单一边界组件。",
        "- MOGB-MiniLM 的低 false acceptance 伴随极低 Known Recall，说明其保守拒识工作点不能只看 OOS F1。",
        "- DA-ADB 若出现 NaN 或单类预测，必须保留为 invalid，不得引用官方 results.csv 中与预测不一致的数字。",
        "",
        "## 机器可读输出",
        "",
        "- `stackoverflow_kir50_external_and_fair_cells.csv`：公平行和外部兼容行，含 layer/contract/status。",
        "- `external_invalid_semantic_runs.csv`：无效运行及原因。",
        "- `adb_vs_trainable_paired_effects.csv`：三个 seed 的配对差值。",
        "- `stackoverflow_external_single_cell_comparison.png`：有效单格工作点图。",
        "- `adb_vs_trainable_paired_delta.png`：同 seed 差值及标准差。",
    ]
    (output_dir / "STACKOVERFLOW_EXTERNAL_COMPARISON_V1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.figure_dir, args.external_root.resolve()), ensure_ascii=False, indent=2))
