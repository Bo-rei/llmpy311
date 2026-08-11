#!/usr/bin/env python3
"""汇总 ADB BERT/TextOIR 外部合同在三个 KIR 下的可审计结果。

该脚本只读取隔离 runtime 的 y_true/y_pred 和 protocol_v2 Trainable-K1 的
逐 seed CSV。它不选择阈值、模型或超参数，也不把 BERT 外部合同并入 MiniLM
fair 主排名。
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
DEFAULT_EXTERNAL_ROOTS = [
    ROOT.parent / "artifacts/s2c/external/adb_gpu_runtime_v2",
    ROOT.parent / "artifacts/s2c/external/adb_gpu_runtime_v1",
    ROOT.parent / "artifacts/s2c/external/adb_gpu_runtime_v3",
]
DEFAULT_FAIR = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
DEFAULT_OUTPUT = ROOT / "results/analysis/adb_kir_sensitivity_v2"
DEFAULT_FIGURES = ROOT / "figures/adb_kir_sensitivity_v2"
DATASET_MAP = {"banking": "banking77", "oos": "clinc150", "stackoverflow": "stackoverflow"}
METRICS = (
    "oos_f1",
    "f1_all",
    "f1_k",
    "accuracy",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
)


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
        "f1_all": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(oos_label + 1)),
                average="macro",
                zero_division=0,
            )
        ),
        "f1_k": float(
            f1_score(
                y_true[known],
                y_pred[known],
                labels=list(range(oos_label)),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "known_recall": float(np.mean(~predicted_oos[known])),
        "false_accept_rate": float(np.mean(~predicted_oos[oos])),
        "false_reject_rate": float(np.mean(predicted_oos[known])),
    }


def read_fair(path: Path) -> dict[tuple[str, float, int], dict[str, float]]:
    lookup: dict[tuple[str, float, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "trainable_k1":
                continue
            key = (row["dataset"], float(row["kir"]), int(row["seed"]))
            lookup[key] = {metric: float(row[metric]) for metric in METRICS}
    return lookup


def infer_dataset(manifest_path: Path, manifest: dict[str, object]) -> str:
    path_text = "/".join(manifest_path.parts)
    for name in ("clinc150", "banking77", "stackoverflow"):
        if name in path_text:
            return name
    return DATASET_MAP.get(str(manifest.get("dataset")), str(manifest.get("dataset")))


def infer_kir(manifest_path: Path, manifest: dict[str, object]) -> float:
    if manifest.get("known_cls_ratio") is not None:
        return float(manifest["known_cls_ratio"])
    for part in manifest_path.parts:
        if part.startswith("kir"):
            return float(part.removeprefix("kir").replace("_", "."))
    raise ValueError(f"Cannot infer KIR from {manifest_path}")


def discover_runs(
    roots: list[Path], fair_path: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    fair = read_fair(fair_path)
    rows: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    paths: set[Path] = set()
    for root in roots:
        paths.update(root.rglob("run_manifest.json"))
    for manifest_path in sorted(paths):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("method") != "ADB":
            continue
        dataset = infer_dataset(manifest_path, manifest)
        kir = infer_kir(manifest_path, manifest)
        if kir not in (0.25, 0.50, 0.75):
            continue
        seed = int(manifest.get("seed", manifest_path.parent.name.removeprefix("seed_")))
        y_true_paths = sorted(manifest_path.parent.rglob("y_true.npy"))
        y_pred_paths = sorted(manifest_path.parent.rglob("y_pred.npy"))
        row: dict[str, object] = {
            "dataset": dataset,
            "kir": kir,
            "seed": seed,
            "method": "ADB",
            "contract": "protocol_v2 split + TextOIR BERT compatibility",
            "run_manifest": str(manifest_path.relative_to(ROOT.parent)),
            "manifest_status": manifest.get("status"),
            "valid_semantic_metrics": False,
            "invalid_reason": "missing_predictions",
        }
        if len(y_true_paths) == 1 and len(y_pred_paths) == 1:
            try:
                y_true = np.load(y_true_paths[0])
                y_pred = np.load(y_pred_paths[0])
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
                "split_sha256": manifest.get("split_sha256", {}),
            }
        )
        if row["valid_semantic_metrics"]:
            fair_row = fair.get((dataset, kir, seed))
            if fair_row is not None:
                for metric in METRICS:
                    paired.append(
                        {
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "metric": metric,
                            "trainable_minus_adb": fair_row[metric] - float(row[metric]),
                            "delta_pp": 100.0 * (fair_row[metric] - float(row[metric])),
                        }
                    )
    return rows, paired, manifests


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    valid = [row for row in rows if row.get("valid_semantic_metrics")]
    output: list[dict[str, object]] = []
    keys = sorted({(str(row["dataset"]), float(row["kir"])) for row in valid})
    for dataset, kir in keys:
        group = [row for row in valid if row["dataset"] == dataset and float(row["kir"]) == kir]
        entry: dict[str, object] = {"dataset": dataset, "kir": kir, "method": "ADB", "n_seeds": len(group)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group])
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        output.append(entry)
    return output


def paired_summary(paired: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], list[float]] = {}
    for row in paired:
        grouped.setdefault((str(row["dataset"]), float(row["kir"]), str(row["metric"])), []).append(float(row["delta_pp"]))
    output: list[dict[str, object]] = []
    for (dataset, kir, metric), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "kir": kir,
                "metric": metric,
                "mean_delta_pp": float(np.mean(values)),
                "std_delta_pp": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "n_pairs": len(values),
            }
        )
    return output


def plot(summary: list[dict[str, object]], pair_summary: list[dict[str, object]], figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    datasets = ["clinc150", "banking77", "stackoverflow"]
    metrics_to_plot = [("oos_f1", "OOS F1 (%)"), ("f1_all", "F1-All (%)"), ("known_recall", "Known Recall (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics_to_plot):
        for dataset in datasets:
            group = sorted([row for row in summary if row["dataset"] == dataset], key=lambda row: float(row["kir"]))
            x = [float(row["kir"]) for row in group]
            y = [100.0 * float(row[f"{metric}_mean"]) for row in group]
            ax.plot(x, y, marker="o", label=dataset)
        ax.set_xlabel("KIR")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0.25, 0.50, 0.75])
        ax.grid(alpha=0.2)
        ax.set_title(f"ADB {ylabel}")
    axes[-1].legend(fontsize=7)
    fig.suptitle("ADB BERT/TextOIR：跨数据集 KIR 敏感性")
    fig.tight_layout()
    fig.savefig(figures / "adb_kir_curves.png", dpi=180)
    plt.close(fig)

    rows = [row for row in pair_summary if row["metric"] == "oos_f1"]
    matrix = np.full((len(datasets), 3), np.nan)
    for row in rows:
        matrix[datasets.index(str(row["dataset"])), [0.25, 0.50, 0.75].index(float(row["kir"]))] = float(row["mean_delta_pp"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix))), vmax=max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix))))
    ax.set_xticks(range(3), ["0.25", "0.50", "0.75"])
    ax.set_yticks(range(len(datasets)), datasets)
    ax.set_xlabel("KIR")
    ax.set_title("S2C Trainable K=1 − ADB OOS F1（百分点）")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="百分点")
    fig.tight_layout()
    fig.savefig(figures / "trainable_minus_adb_oos_f1_heatmap.png", dpi=180)
    plt.close(fig)


def build_report(
    summary: list[dict[str, object]],
    pair_summary: list[dict[str, object]],
    valid_count: int,
    total_count: int,
    figure_dir: Path,
) -> str:
    labels = {"oos_f1": "OOS F1", "f1_all": "F1-All", "known_recall": "Known Recall"}
    seed_counts = sorted({int(row["n_seeds"]) for row in summary})
    seed_label = ", ".join(str(value) for value in seed_counts) if seed_counts else "0"
    expected_units = 3 * 3 * max(seed_counts, default=0)
    lines = [
        "# ADB 跨 KIR 三数据集外部合同分析",
        "",
        "本报告只使用 protocol_v2_textoir_v1 的固定 split、隔离 TextOIR BERT runtime 和已审计 y_true/y_pred。",
        "ADB 是 BERT/TextOIR 外部合同；S2C Trainable K=1 是 Known-only MiniLM。所有差值仅用于机制与工作点分析，",
        "不进入 MiniLM fair 主排名，也不用于测试集选参。",
        "",
        f"有效单元：`{valid_count}/{total_count}`；当前每个 dataset×KIR 的 seed 数为 `{seed_label}`，"
        f"预期规模为 3 数据集×3 KIR×{max(seed_counts, default=0)} seed = `{expected_units}`。",
        "",
        "## ADB 绝对结果",
        "",
        "| 数据集 | KIR | OOS F1 | F1-All | Known Recall | False Acceptance | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {float(row['kir']):.2f} | {100*float(row['oos_f1_mean']):.2f}±{100*float(row['oos_f1_std']):.2f} | {100*float(row['f1_all_mean']):.2f}±{100*float(row['f1_all_std']):.2f} | {100*float(row['known_recall_mean']):.2f}±{100*float(row['known_recall_std']):.2f} | {100*float(row['false_accept_rate_mean']):.2f}±{100*float(row['false_accept_rate_std']):.2f} | {row['n_seeds']} |"
        )
    lines.extend(["", "## S2C Trainable K=1 − ADB 配对差值", "", "| 数据集 | KIR | 指标 | 均值(pp) | 标准差(pp) | n |", "|---|---:|---|---:|---:|---:|"])
    for row in pair_summary:
        if row["metric"] not in labels:
            continue
        lines.append(f"| {row['dataset']} | {float(row['kir']):.2f} | {labels[str(row['metric'])]} | {float(row['mean_delta_pp']):.2f} | {float(row['std_delta_pp']):.2f} | {row['n_pairs']} |")
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- KIR 变化下，Trainable 与 ADB 的差异依赖数据集；不能把 StackOverflow 的接近结果外推到 CLINC150 或 Banking77。",
            "- CLINC150 的 Trainable OOS F1 接近 ADB，但 F1-All 与 Known Recall 低于 ADB；综合指标不是全面领先。",
            "- Banking77 和 StackOverflow 上 Trainable 的 F1-All 高于 ADB，但 Known Recall 可能低于或接近 ADB，仍需报告工作点权衡。",
            "- 该报告不能证明 Trainable 超过论文中的完整 ADB，也不能和 MOGB 论文值或 DCLOOS 外部 OOS 监督结果直接排名。",
            "",
            f"图表：`{figure_dir}`；机器可读结果由调用方指定输出目录。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, action="append", default=None)
    parser.add_argument("--fair", type=Path, default=DEFAULT_FAIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    roots = [p.resolve() for p in (args.external_root or DEFAULT_EXTERNAL_ROOTS)]
    rows, paired, manifests = discover_runs(roots, args.fair.resolve())
    summaries = summarize(rows)
    pair_summaries = paired_summary(paired)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "adb_per_seed.csv", rows)
    write_csv(args.output_dir / "adb_summary.csv", summaries)
    write_csv(args.output_dir / "adb_trainable_paired_effects.csv", paired)
    write_csv(args.output_dir / "adb_trainable_paired_summary.csv", pair_summaries)
    plot(summaries, pair_summaries, args.figure_dir)
    valid_count = sum(bool(row.get("valid_semantic_metrics")) for row in rows)
    manifest = {
        "schema_version": 1,
        "analysis": "adb_kir_sensitivity_v2",
        "protocol_version": "protocol_v2_textoir_v1",
        "external_roots": [str(p) for p in roots],
        "fair_source": str(args.fair.resolve()),
        "fair_source_sha256": sha256(args.fair.resolve()),
        "run_manifest_count": len(manifests),
        "valid_units": valid_count,
        "paired_effect_rows": len(paired),
        "run_manifests": manifests,
        "selection_used_test_oos": False,
        "metric_source": "y_true.npy/y_pred.npy",
        "outputs": ["adb_per_seed.csv", "adb_summary.csv", "adb_trainable_paired_effects.csv", "adb_trainable_paired_summary.csv"],
    }
    atomic_text(args.output_dir / "ADB_KIR_SENSITIVITY_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    atomic_text(
        args.output_dir / "ADB_KIR_SENSITIVITY_REPORT.md",
        build_report(summaries, pair_summaries, valid_count, len(rows), args.figure_dir),
    )
    print(json.dumps({"status": "ok", "rows": len(rows), "valid": valid_count, "paired": len(paired)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
