#!/usr/bin/env python3
"""对 protocol_v2 Trainable-K1 与 ADB BERT 外部合同做配对统计。

该工具只读取已经完成的逐 seed 摘要，不训练、不调阈值，也不把不同 backbone
合并成同条件排名。每个 dataset×KIR×metric 只有五个配对 seed，因此结果用于
稳定性和工作点解释，而不是替代完整多方法显著性研究。
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
from scipy.stats import binomtest

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAIR = ROOT / "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"
DEFAULT_ADB = ROOT / "results/analysis/adb_kir_sensitivity_v2/adb_per_seed.csv"
DEFAULT_OUTPUT = ROOT / "results/analysis/adb_paired_inference_v1"
DEFAULT_FIGURES = ROOT / "figures/adb_paired_inference_v1"
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
METRICS = (
    ("oos_f1", "OOS F1"),
    ("f1_all", "F1-All"),
    ("known_recall", "Known Recall"),
    ("false_accept_rate", "False Acceptance"),
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


def read_rows(path: Path, *, method: str | None = None) -> dict[tuple[str, float, int], dict[str, float]]:
    output: dict[tuple[str, float, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if method is not None and row.get("method") != method:
                continue
            key = (row["dataset"], float(row["kir"]), int(row["seed"]))
            output[key] = {metric: float(row[metric]) for metric, _ in METRICS}
    return output


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repetitions: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def infer(
    fair: dict[tuple[str, float, int], dict[str, float]],
    adb: dict[tuple[str, float, int], dict[str, float]],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            keys = sorted(key for key in fair if key[0] == dataset and key[1] == kir and key in adb)
            if not keys:
                continue
            for metric, metric_label in METRICS:
                deltas = np.asarray([fair[key][metric] - adb[key][metric] for key in keys], dtype=float)
                wins = int(np.sum(deltas > 1e-12))
                ties = int(np.sum(np.isclose(deltas, 0.0, atol=1e-12)))
                losses = int(deltas.size - wins - ties)
                ci_low, ci_high = bootstrap_ci(deltas, rng, bootstrap_repetitions)
                std = float(np.std(deltas, ddof=1)) if deltas.size > 1 else float("nan")
                p_value = float(binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue) if wins + losses else 1.0
                rows.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "metric": metric,
                        "metric_label": metric_label,
                        "n_pairs": int(deltas.size),
                        "mean_delta": float(np.mean(deltas)),
                        "median_delta": float(np.median(deltas)),
                        "std_delta": std,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "cohen_dz": float(np.mean(deltas) / std) if np.isfinite(std) and std > 0 else float("nan"),
                        "sign_test_p": p_value,
                        "seeds": "|".join(str(key[2]) for key in keys),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["dataset", "kir", "metric"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot(rows: list[dict[str, object]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    colors = {"clinc150": "#1f77b4", "banking77": "#ff7f0e", "stackoverflow": "#2ca02c"}
    for ax, (metric, label) in zip(axes.flat, METRICS):
        for dataset in DATASETS:
            group = sorted([r for r in rows if r["dataset"] == dataset and r["metric"] == metric], key=lambda r: float(r["kir"]))
            x = np.asarray([float(r["kir"]) for r in group])
            y = 100.0 * np.asarray([float(r["mean_delta"]) for r in group])
            low = y - 100.0 * np.asarray([float(r["ci95_low"]) for r in group])
            high = 100.0 * np.asarray([float(r["ci95_high"]) for r in group]) - y
            ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3, color=colors[dataset], label=dataset)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"Trainable-K1 − ADB：{label}")
        ax.set_ylabel("百分点")
        ax.set_xticks(KIRS)
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("KIR")
    axes[1, 1].set_xlabel("KIR")
    axes[0, 1].legend(fontsize=8, loc="best")
    fig.suptitle("五 seed 配对效应与 95% bootstrap CI（ADB 为 BERT/TextOIR 外部合同）")
    fig.tight_layout()
    fig.savefig(figure_dir / "trainable_minus_adb_paired_forest.png", dpi=190)
    plt.close(fig)


def report(rows: list[dict[str, object]], fair_path: Path, adb_path: Path, repetitions: int, seed: int) -> str:
    lines = [
        "# Trainable-K1 与 ADB 五 seed 配对推断",
        "",
        "本报告只对相同 protocol_v2 registry、相同 dataset×KIR×seed 做配对统计。ADB 使用 BERT/TextOIR，",
        "Trainable-K1 使用 Known-only MiniLM，因此 CI 和胜负只描述工作点差异，不构成同 backbone SOTA 排名。",
        "",
        f"bootstrap：`{repetitions}` 次；RNG seed：`{seed}`；输入 seed：`13,42,87,100,123`。",
        "",
        "| 数据集 | KIR | 指标 | 均值差(pp) | 95% CI(pp) | 胜/平/负 | sign-test p | Cohen dz |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {float(row['kir']):.2f} | {row['metric_label']} | "
            f"{100*float(row['mean_delta']):.2f} | [{100*float(row['ci95_low']):.2f}, {100*float(row['ci95_high']):.2f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {float(row['sign_test_p']):.4f} | {float(row['cohen_dz']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- OOS F1 的正差值表示 Trainable-K1 高于 ADB，负差值表示 ADB 更高。",
            "- Known Recall 和 false acceptance 必须与 OOS F1 一起解释；只看 OOS F1 会隐藏拒识/覆盖工作点。",
            "- 五个 seed 的配对 CI 只量化当前固定 split 与两个训练合同的差异，不能消除 BERT/MiniLM、训练目标和监督差异。",
            "",
            f"fair source SHA256：`{sha256(fair_path)}`",
            f"ADB source SHA256：`{sha256(adb_path)}`",
            "图：`figures/adb_paired_inference_v1/trainable_minus_adb_paired_forest.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair", type=Path, default=DEFAULT_FAIR)
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    args = parser.parse_args()
    fair_path = args.fair.resolve()
    adb_path = args.adb.resolve()
    fair = read_rows(fair_path, method="trainable_k1")
    adb = read_rows(adb_path, method="ADB")
    rows = infer(fair, adb, bootstrap_repetitions=args.bootstrap_repetitions, bootstrap_seed=args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_inference.csv", rows)
    plot(rows, args.figure_dir.resolve())
    manifest = {
        "schema_version": 1,
        "analysis": "adb_paired_inference_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "fair_source": str(fair_path),
        "fair_source_sha256": sha256(fair_path),
        "adb_source": str(adb_path),
        "adb_source_sha256": sha256(adb_path),
        "row_count": len(rows),
        "n_pairs_per_cell": 5,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "selection_used_test_oos": False,
    }
    atomic_text(args.output_dir / "ADB_PAIRED_INFERENCE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    atomic_text(args.output_dir / "ADB_PAIRED_INFERENCE_REPORT.md", report(rows, fair_path, adb_path, args.bootstrap_repetitions, args.bootstrap_seed))
    print(json.dumps({"status": "ok", "rows": len(rows), "pairs_per_cell": 5}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
