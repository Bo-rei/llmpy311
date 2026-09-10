#!/usr/bin/env python3
"""Audit and visualize the three-seed ADB BERT compatibility cells.

This is analysis-only.  It reads audited y_true/y_pred arrays from the isolated
ADB runtime and pairs them with the existing protocol_v2 Trainable-K1 rows on
the same dataset/seed/KIR=.50 cells.  It never selects a model or threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = ROOT.parent / "artifacts/s2c/external/adb_gpu_runtime_v2"
DEFAULT_FAIR = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
DEFAULT_OUTPUT = ROOT / "results/analysis/archive/analysis/adb_cross_dataset_v1"
DEFAULT_FIGURES = ROOT / "figures/archive/analysis/adb_cross_dataset_v1"
DATASET_MAP = {"banking": "banking77", "oos": "clinc150", "stackoverflow": "stackoverflow"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("prediction arrays are empty or shape-mismatched")
    oos_label = int(np.max(y_true))
    known = y_true != oos_label
    oos = ~known
    predicted_oos = y_pred == oos_label
    return {
        "oos_f1": float(f1_score(oos, predicted_oos, zero_division=0)),
        "f1_all": float(f1_score(y_true, y_pred, labels=list(range(oos_label + 1)), average="macro", zero_division=0)),
        "f1_k": float(f1_score(y_true[known], y_pred[known], labels=list(range(oos_label)), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "known_recall": float(np.mean(~predicted_oos[known])),
        "false_accept_rate": float(np.mean(~predicted_oos[oos])),
        "false_reject_rate": float(np.mean(predicted_oos[known])),
    }


def read_fair(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "trainable_k1" or abs(float(row["kir"]) - 0.50) > 1e-9:
                continue
            key = (row["dataset"], int(row["seed"]))
            lookup[key] = {k: float(row[k]) for k in ("oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate")}
    return lookup


def audit_runs(external_roots: list[Path], fair_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    fair = read_fair(fair_path)
    rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    manifest_paths: list[Path] = []
    for external_root in external_roots:
        manifest_paths.extend(external_root.glob("*/ADB/kir50/*/run_manifest.json"))
    for manifest_path in sorted(set(manifest_paths)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_key = manifest.get("dataset") or manifest_path.parts[-5]
        dataset = DATASET_MAP.get(str(dataset_key), str(dataset_key))
        seed = int(manifest.get("seed", manifest_path.parts[-2]))
        y_true_paths = sorted(manifest_path.parent.rglob("y_true.npy"))
        y_pred_paths = sorted(manifest_path.parent.rglob("y_pred.npy"))
        row: dict[str, object] = {
            "dataset": dataset,
            "runtime_dataset": dataset_key,
            "kir": 0.50,
            "seed": seed,
            "method": "ADB",
            "contract": "protocol_v2 split + TextOIR BERT compatibility",
            "run_manifest": str(manifest_path.relative_to(ROOT.parent)),
            "manifest_status": manifest.get("status"),
            "valid_semantic_metrics": False,
            "invalid_reason": "missing_predictions",
        }
        if len(y_true_paths) != 1 or len(y_pred_paths) != 1:
            rows.append(row)
            continue
        y_true = np.load(y_true_paths[0])
        y_pred = np.load(y_pred_paths[0])
        try:
            row.update(metrics(y_true, y_pred))
            row["n_test"] = int(y_true.size)
            row["n_known"] = int(np.sum(y_true != np.max(y_true)))
            row["n_oos"] = int(np.sum(y_true == np.max(y_true)))
            row["valid_semantic_metrics"] = bool(np.unique(y_pred).size > 1 and np.isfinite(y_pred).all())
            row["invalid_reason"] = "" if row["valid_semantic_metrics"] else "all_class_prediction_or_nonfinite"
        except (TypeError, ValueError) as exc:
            row["invalid_reason"] = str(exc)
        rows.append(row)
        manifests.append(
            {
                "path": str(manifest_path.relative_to(ROOT.parent)),
                "sha256": sha256(manifest_path),
                "status": manifest.get("status"),
                "artifact_audit": manifest.get("artifact_audit", {}),
            }
        )
    paired: list[dict[str, object]] = []
    for row in rows:
        if not row.get("valid_semantic_metrics"):
            continue
        fair_row = fair.get((str(row["dataset"]), int(row["seed"])))
        if fair_row is None:
            row["paired_trainable"] = False
            continue
        row["paired_trainable"] = True
        for metric in ("oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate"):
            paired.append(
                {
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "metric": metric,
                    "trainable_minus_adb": fair_row[metric] - float(row[metric]),
                    "delta_pp": 100.0 * (fair_row[metric] - float(row[metric])),
                }
            )
    return rows, paired, manifests


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_report(rows: list[dict[str, object]], paired: list[dict[str, object]]) -> str:
    valid = [r for r in rows if r.get("valid_semantic_metrics")]
    lines = [
        "# ADB 跨数据集三 seed 外部合同对比",
        "",
        "本报告审计 ADB 在当前 protocol_v2 导出 split 上的 BERT/TextOIR 兼容运行，并与相同 dataset×seed×KIR=.50 的 S2C Trainable K=1 做配对。它不把 BERT 外部合同并入 MiniLM fair 主排名。",
        "",
        f"有效单元：`{len(valid)}/{len(rows)}`；配对差值：`{len(paired)}` 行；指标从 `y_true.npy/y_pred.npy` 重算。",
        "",
        "## ADB 逐 seed 结果",
        "",
        "| 数据集 | seed | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | False Acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(valid, key=lambda item: (str(item["dataset"]), int(item["seed"]))):
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {100*float(row['oos_f1']):.2f} | {100*float(row['f1_all']):.2f} | {100*float(row['f1_k']):.2f} | {100*float(row['accuracy']):.2f} | {100*float(row['known_recall']):.2f} | {100*float(row['false_accept_rate']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## S2C Trainable K=1 − ADB 配对均值（百分点）",
            "",
            "| 数据集 | 指标 | 均值 | 标准差 |",
            "|---|---|---:|---:|",
        ]
    )
    grouped: dict[tuple[str, str], list[float]] = {}
    for item in paired:
        grouped.setdefault((str(item["dataset"]), str(item["metric"])), []).append(float(item["delta_pp"]))
    metric_labels = {
        "oos_f1": "OOS F1",
        "f1_all": "F1-All",
        "f1_k": "F1-K",
        "accuracy": "Accuracy",
        "known_recall": "Known Recall",
        "false_accept_rate": "False Acceptance",
        "false_reject_rate": "False Rejection",
    }
    for (dataset, metric), values in sorted(grouped.items()):
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        lines.append(f"| {dataset} | {metric_labels.get(metric, metric)} | {mean:.2f} | {std:.2f} |")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- ADB 使用端到端 BERT/TextOIR 训练；S2C Trainable K=1 使用 Known-only MiniLM 适配。配对差值只能说明同数据 split 下的工作点差异，不能分离 backbone、训练目标和边界学习的贡献。",
            "- DA-ADB、DCLOOS 和 MOGB 官方论文值不在本报告中重新排名；它们分别有 invalid、额外 OOS 监督或官方数据/环境合同差异。",
            "- 下一步可在本报告基础上补充 ADB 的 KIR 曲线，但当前不使用测试结果选参。",
            "",
            "机器可读输出：`results/analysis/archive/analysis/adb_cross_dataset_v1/`；图表：`figures/archive/analysis/adb_cross_dataset_v1/`。",
        ]
    )
    return "\n".join(lines) + "\n"


def plot(rows: list[dict[str, object]], paired: list[dict[str, object]], figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if r.get("valid_semantic_metrics")]
    datasets = sorted({str(r["dataset"]) for r in valid})
    adb_means = {d: np.mean([float(r["oos_f1"]) for r in valid if r["dataset"] == d]) * 100 for d in datasets}
    trainable_by_dataset: dict[str, list[float]] = {}
    for row in paired:
        if row["metric"] == "oos_f1":
            trainable_by_dataset.setdefault(str(row["dataset"]), []).append(float(row["delta_pp"]) + 100 * float(next(r["oos_f1"] for r in valid if r["dataset"] == row["dataset"] and int(r["seed"]) == int(row["seed"]))))
    trainable_means = {d: np.mean(trainable_by_dataset[d]) if d in trainable_by_dataset else np.nan for d in datasets}
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, [adb_means[d] for d in datasets], 0.36, label="ADB / BERT")
    ax.bar(x + 0.18, [trainable_means[d] for d in datasets], 0.36, label="S2C Trainable K=1 / MiniLM")
    ax.set_xticks(x, datasets)
    ax.set_ylabel("OOS F1 (%)")
    ax.set_title("KIR=0.50：ADB 与 S2C Trainable 的同 split 外部参照")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "adb_vs_trainable_oos_f1.png", dpi=180)
    plt.close(fig)

    # Each dataset's paired OOS delta is more useful than an absolute ranking.
    deltas = {d: [float(r["delta_pp"]) for r in paired if r["dataset"] == d and r["metric"] == "oos_f1"] for d in datasets}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([deltas[d] for d in datasets], tick_labels=datasets, showmeans=True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("S2C Trainable K=1 − ADB (pp)")
    ax.set_title("同 seed 配对 OOS F1 差值（外部合同参照）")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "adb_vs_trainable_paired_oos_delta.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-root",
        type=Path,
        action="append",
        default=None,
        help="ADB runtime root; repeat to combine separate dataset roots",
    )
    parser.add_argument("--fair", type=Path, default=DEFAULT_FAIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    external_roots = [
        p.resolve()
        for p in (args.external_root or [
            DEFAULT_EXTERNAL_ROOT,
            ROOT.parent / "artifacts/s2c/external/adb_gpu_runtime_v1",
        ])
    ]
    rows, paired, manifests = audit_runs(external_roots, args.fair.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "adb_per_seed.csv", rows)
    write_csv(args.output_dir / "adb_trainable_paired_effects.csv", paired)
    valid = [r for r in rows if r.get("valid_semantic_metrics")]
    summaries: list[dict[str, object]] = []
    for dataset in sorted({str(r["dataset"]) for r in valid}):
        group = [r for r in valid if r["dataset"] == dataset]
        entry: dict[str, object] = {"dataset": dataset, "n_seeds": len(group)}
        for metric in ("oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_accept_rate", "false_reject_rate"):
            values = np.asarray([float(r[metric]) for r in group])
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        summaries.append(entry)
    write_csv(args.output_dir / "adb_summary.csv", summaries)
    paired_summary: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    for item in paired:
        grouped.setdefault((str(item["dataset"]), str(item["metric"])), []).append(float(item["delta_pp"]))
    for (dataset, metric), values in sorted(grouped.items()):
        paired_summary.append(
            {
                "dataset": dataset,
                "metric": metric,
                "mean_delta_pp": float(np.mean(values)),
                "std_delta_pp": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "n_pairs": len(values),
            }
        )
    write_csv(args.output_dir / "adb_trainable_paired_summary.csv", paired_summary)
    plot(rows, paired, args.figure_dir)
    manifest = {
        "schema_version": 1,
        "analysis": "adb_cross_dataset_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "external_roots": [str(p) for p in external_roots],
        "fair_source": str(args.fair),
        "fair_source_sha256": sha256(args.fair),
        "run_manifest_count": len(manifests),
        "valid_units": len(valid),
        "paired_effect_rows": len(paired),
        "run_manifests": manifests,
        "selection_used_test_oos": False,
        "metric_source": "y_true.npy/y_pred.npy",
        "outputs": ["adb_per_seed.csv", "adb_summary.csv", "adb_trainable_paired_effects.csv", "adb_trainable_paired_summary.csv"],
    }
    atomic_text(args.output_dir / "ADB_CROSS_DATASET_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    atomic_text(args.output_dir / "ADB_CROSS_DATASET_REPORT.md", build_report(rows, paired))
    print(json.dumps({"status": "ok", "rows": len(rows), "valid": len(valid), "paired": len(paired)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
