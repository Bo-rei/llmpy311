#!/usr/bin/env python3
"""Retrospective score-operating-point diagnostic for current baselines.

The thresholds here are selected from test labels only to place methods on a
common Known-Recall axis.  They are explicitly diagnostic and never replace
the Known-only threshold/radius protocol.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from protocol_v2.evaluation.metrics import compute_binary_oos_metrics


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1"
OUT_ROOT = ROOT / "results" / "analysis" / "operating_point_diagnostic_v1"
FIG_ROOT = ROOT / "figures" / "operating_point_diagnostic_v1"
TARGETS = (0.75, 0.85, 0.90, 0.95)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metric(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    labels = np.asarray([int(row["gold_is_oos"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["oos_score"]) for row in rows], dtype=np.float64)
    value = compute_binary_oos_metrics(labels, scores, threshold)
    predicted = (scores > threshold).astype(np.int64)
    return {**value, "known_recall": float(value["id_recall"]), "threshold": float(threshold), "f1_binary": float(f1_score(labels, predicted, zero_division=0))}


def _candidate_thresholds(rows: list[dict[str, Any]]) -> np.ndarray:
    scores = np.asarray([float(row["oos_score"]) for row in rows], dtype=np.float64)
    values = np.unique(scores[np.isfinite(scores)])
    if values.size > 500:
        values = np.quantile(values, np.linspace(0.0, 1.0, 501))
    return np.unique(np.concatenate((values, [0.0, 1.0])))


def _collect() -> pd.DataFrame:
    items: list[tuple[str, Path, dict[str, Any]]] = []
    # Trainable predictions use a compact stage-specific path.
    for stage in ("minilm_trainable_kir_sweep_v1", "minilm_trainable_kir_sweep_extension_v1"):
        for path in ARTIFACT_ROOT.glob(f"{stage}/kir_*/runs/*/seed_*/predictions.jsonl"):
            rel = path.relative_to(ARTIFACT_ROOT)
            kir = float(rel.parts[1].split("_")[1])
            dataset, seed = rel.parts[3], int(rel.parts[4].split("_")[1])
            items.append(("trainable_k1", path, {"dataset": dataset, "kir": kir, "seed": seed, "protocol_version": "protocol_v2_textoir_v1"}))
    # Native baselines store one test JSONL per manifest.
    for path in ARTIFACT_ROOT.glob("native_baselines_v1/**/predictions/test.jsonl"):
        manifest = json.loads((path.parent.parent / "manifest.json").read_text(encoding="utf-8"))
        config = manifest["config"]
        items.append((str(config["method"]), path, {"dataset": str(config["dataset"]), "kir": float(config["kir"]), "seed": int(config["seed"]), "protocol_version": str(config["protocol_version"])}))
    result: list[dict[str, Any]] = []
    for method, path, meta in items:
        rows = _rows(path)
        for target in TARGETS:
            known_scores = np.asarray([float(row["oos_score"]) for row in rows if int(row["gold_is_oos"]) == 0], dtype=np.float64)
            if not known_scores.size:
                raise RuntimeError(f"No Known rows in diagnostic input: {path}")
            # Retrospective only: use the test Known-score quantile to place
            # each method at a common nominal recall.  This is not a tuning
            # rule and must never enter the formal protocol.
            threshold = float(np.quantile(known_scores, target, method="linear"))
            selected = _metric(rows, threshold)
            result.append({**meta, "method": method, "target_known_recall": target, "diagnostic_threshold": selected["threshold"], "achieved_known_recall": selected["known_recall"], "oos_f1": selected["oos_f1"], "oos_precision": selected["oos_precision"], "oos_recall": selected["oos_recall"], "false_accept_rate": selected["false_accept_rate"], "false_reject_rate": selected["false_reject_rate"], "auroc": selected["auroc"], "aupr_oos": selected["aupr_oos"], "source_path": os.path.relpath(path, ROOT)})
    frame = pd.DataFrame(result)
    if frame.empty:
        raise RuntimeError("No trainable/native prediction files found for operating-point diagnostic")
    return frame.sort_values(["dataset", "kir", "target_known_recall", "method", "seed"]).reset_index(drop=True)


def _plot(frame: pd.DataFrame) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    methods = ["trainable_k1", "msp", "energy", "knn", "lof"]
    labels = {"trainable_k1": "Trainable K=1", "msp": "MSP", "energy": "Energy", "knn": "kNN", "lof": "LOF"}
    colours = {"trainable_k1": "#d62728", "msp": "#1f77b4", "energy": "#2ca02c", "knn": "#9467bd", "lof": "#ff7f0e"}
    for dataset in sorted(frame["dataset"].unique()):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        for method in methods:
            sub = frame[(frame["dataset"] == dataset) & (frame["kir"] == 0.5) & (frame["method"] == method)].groupby("target_known_recall", as_index=False)[["achieved_known_recall", "oos_f1", "false_accept_rate"]].mean()
            if sub.empty:
                continue
            axes[0].plot(sub["achieved_known_recall"] * 100, sub["oos_f1"] * 100, marker="o", color=colours[method], label=labels[method])
            axes[1].plot(sub["achieved_known_recall"] * 100, sub["false_accept_rate"] * 100, marker="o", color=colours[method], label=labels[method])
            threshold_summary = frame[(frame["dataset"] == dataset) & (frame["kir"] == 0.5) & (frame["method"] == method)].groupby("target_known_recall", as_index=False)["diagnostic_threshold"].mean()
            axes[2].plot(threshold_summary["target_known_recall"] * 100, threshold_summary["diagnostic_threshold"], marker="o", color=colours[method], label=labels[method])
        axes[0].set_title(f"{dataset} KIR=0.50: OOS F1 at matched Known Recall")
        axes[1].set_title(f"{dataset} KIR=0.50: false acceptance")
        axes[2].set_title(f"{dataset}: diagnostic threshold drift")
        axes[0].set_xlabel("Achieved Known Recall (%)")
        axes[0].set_ylabel("OOS F1 (%)")
        axes[1].set_xlabel("Achieved Known Recall (%)")
        axes[1].set_ylabel("False Accept (%)")
        axes[2].set_xlabel("Target Known Recall (%)")
        axes[2].set_ylabel("Test-derived threshold")
        for axis in axes:
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        fig.savefig(FIG_ROOT / f"{dataset}_operating_points.png", dpi=180)
        plt.close(fig)


def _report(frame: pd.DataFrame) -> None:
    sub = frame[(frame["kir"] == 0.5) & (frame["target_known_recall"] == 0.85)]
    lines = [
        "# Known Recall 对齐的阈值工作点诊断 V1",
        "",
        "> 这是事后诊断，不是正式调参结果。阈值使用 test 标签仅用于把不同 score 标度放到同一 Known Recall 横轴，不能用于论文主结果或后续模型选择。",
        "",
        "## 为什么需要这项诊断",
        "",
        "原生 MSP/Energy/kNN/LOF 的 Known-only conformal 阈值默认保留约 95% Known 样本；当前 Trainable 固定 threshold=1 的 Known Recall 明显更低。因此直接比较 OOS F1 会把不同拒识工作点混在一起。",
        "",
        "## KIR=0.50、目标 Known Recall≈85%",
        "",
        "| 数据集 | 方法 | 实际 Known Recall | OOS F1 | False Accept | False Reject |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in sub.groupby(["dataset", "method"], as_index=False).mean(numeric_only=True).iterrows():
        lines.append(f"| {row['dataset']} | {row['method']} | {row['achieved_known_recall']*100:.2f}% | {row['oos_f1']*100:.2f}% | {row['false_accept_rate']*100:.2f}% | {row['false_reject_rate']*100:.2f}% |")
    lines += [
        "",
        "## 结论边界",
        "",
        "1. Trainable 的 OOS F1 优势不能只解释为阈值造成；需要看同一 Known Recall 工作点的曲线。",
        "2. 若对齐后 Trainable 仍保持更高 OOS F1，说明表示/score 排序本身更有利；若优势消失，说明主要是当前 threshold=1 导致拒识工作点更激进。",
        "3. 该诊断不改变正式 protocol_v2_textoir_v1 的 Known-only 选择规则。",
        "",
        "- 数据：`results/analysis/operating_point_diagnostic_v1/per_seed_targets.csv`",
        "- 图：`figures/operating_point_diagnostic_v1/`",
    ]
    (ROOT / "docs" / "analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "analysis" / "OPERATING_POINT_DIAGNOSTIC_V1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global OUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    OUT_ROOT = args.output_root.resolve()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = _collect()
    frame.to_csv(OUT_ROOT / "per_seed_targets.csv", index=False)
    summary = frame.groupby(["dataset", "kir", "method", "target_known_recall"], as_index=False)[["achieved_known_recall", "oos_f1", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"]].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
    summary.reset_index().to_csv(OUT_ROOT / "summary.csv", index=False)
    _plot(frame)
    _report(frame)
    print(json.dumps({"rows": len(frame), "run_root": str(ARTIFACT_ROOT), "targets": list(TARGETS)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
