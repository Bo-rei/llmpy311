#!/usr/bin/env python3
"""Summarize current-protocol DA-ADB cells against matched S2C Trainable K=1.

The external DA-ADB runs use the same protocol_v2 StackOverflow split roots and
seed-specific Known lists as the matched S2C rows, but retain a separate
BERT/TextOIR contract.  This script therefore reports paired descriptive
differences, never a pooled SOTA ranking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts" / "s2c"
RUN_ROOT = (
    ARTIFACTS
    / "external"
    / "da_adb_gpu_runtime_v1"
    / "stackoverflow"
    / "DA-ADB"
    / "kir_0.50"
)
S2C_PER_SEED = ROOT / "results" / "analysis" / "cross_protocol_tradeoff_v1" / "per_seed.csv"
OUT = ROOT / "results" / "analysis" / "da_adb_current_protocol_summary_v1"
FIG = ROOT / "figures" / "da_adb_current_protocol_summary_v1"
REPORT = ROOT / "docs" / "analysis" / "DA_ADB_CURRENT_PROTOCOL_SUMMARY_V1.md"
SEEDS = (42, 87, 100)
BOOTSTRAP_SEED = 20260810


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_dir(run: Path) -> Path:
    dirs = sorted(
        p
        for p in (run / "textoir_outputs" / "open_intent_detection").glob("*")
        if (p / "y_true.npy").is_file() and (p / "y_pred.npy").is_file()
    )
    if len(dirs) != 1:
        raise RuntimeError(f"expected one prediction directory under {run}, got {dirs}")
    return dirs[0]


def da_metrics(run: Path) -> dict:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    pdir = prediction_dir(run)
    y_true = np.load(pdir / "y_true.npy")
    y_pred = np.load(pdir / "y_pred.npy")
    unknown = int(manifest["unknown_label_id"])
    known_labels = list(range(unknown))
    known = y_true != unknown
    oos = ~known
    pred_oos = y_pred == unknown
    return {
        "method": "DA-ADB",
        "contract": "BERT/TextOIR external compatibility",
        "seed": int(manifest["seed"]),
        "run_manifest": str((run / "run_manifest.json").resolve()),
        "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
        "split_train_sha256": manifest["split_sha256"]["train"],
        "split_dev_sha256": manifest["split_sha256"]["dev"],
        "split_test_sha256": manifest["split_sha256"]["test"],
        "known_labels": "|".join(manifest["known_labels"]),
        "oos_f1": float(f1_score(oos, pred_oos, zero_division=0)),
        "f1_all": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_known": float(
            f1_score(y_true, y_pred, labels=known_labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "known_recall": float(((y_pred[known] != unknown).sum() / known.sum())),
        "false_acceptance": float(((y_pred[oos] != unknown).sum() / oos.sum())),
        "false_rejection": float(((y_pred[known] == unknown).sum() / known.sum())),
        "n_samples": int(y_true.size),
        "predicted_label_count": int(np.unique(y_pred).size),
        "finite": bool(np.isfinite(y_true).all() and np.isfinite(y_pred).all()),
        "shape_equal": bool(y_true.shape == y_pred.shape),
        "y_true_sha256": sha256_file(pdir / "y_true.npy"),
        "y_pred_sha256": sha256_file(pdir / "y_pred.npy"),
    }


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=20_000,
        method="percentile",
        rng=rng,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def main() -> int:
    da_rows = [da_metrics(RUN_ROOT / f"seed_{seed}") for seed in SEEDS]
    s2c = pd.read_csv(S2C_PER_SEED)
    s2c = s2c[
        (s2c["dataset"] == "stackoverflow")
        & (s2c["kir"] == 0.50)
        & (s2c["method"] == "trainable_k1")
        & (s2c["seed"].isin(SEEDS))
    ].copy()
    if set(s2c["seed"].astype(int)) != set(SEEDS):
        raise RuntimeError("matched S2C Trainable K=1 rows are incomplete")
    s2c = s2c.set_index("seed")
    metrics = [
        "oos_f1",
        "f1_all",
        "f1_known",
        "accuracy",
        "known_recall",
        "false_acceptance",
        "false_rejection",
    ]
    rows = []
    paired = []
    for da in da_rows:
        seed = int(da["seed"])
        s = s2c.loc[seed]
        rows.append({"method": "DA-ADB", "contract": da["contract"], "seed": seed, **{m: da[m] for m in metrics}})
        rows.append(
            {
                "method": "S2C-Trainable-K1",
                "contract": "protocol_v2 Known-only MiniLM fair",
                "seed": seed,
                **{
                    m: float(
                        s["f1_k"] if m == "f1_known" else
                        s["false_accept_rate"] if m == "false_acceptance" else
                        s["false_reject_rate"] if m == "false_rejection" else
                        s[m]
                    )
                    for m in metrics
                },
            }
        )
        paired.append(
            {
                "seed": seed,
                **{
                    f"trainable_minus_da_{m}": (
                        float(
                            s["f1_k"] if m == "f1_known" else
                            s["false_accept_rate"] if m == "false_acceptance" else
                            s["false_reject_rate"] if m == "false_rejection" else
                            s[m]
                        ) - da[m]
                    )
                    for m in metrics
                },
            }
        )
    per_seed = pd.DataFrame(rows)
    pair_df = pd.DataFrame(paired)
    summary_rows = []
    for method, group in per_seed.groupby("method", sort=False):
        row = {"method": method, "contract": group["contract"].iloc[0], "n_seeds": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    pair_summary = {
        "comparison": "S2C-Trainable-K1 minus DA-ADB; same seed, different training contract",
        "n_seeds": len(pair_df),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": 20_000,
    }
    for metric in metrics:
        values = pair_df[f"trainable_minus_da_{metric}"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(values)
        pair_summary[f"{metric}_mean_delta"] = float(values.mean())
        pair_summary[f"{metric}_std_delta"] = float(values.std(ddof=1))
        pair_summary[f"{metric}_ci95_low"] = lo
        pair_summary[f"{metric}_ci95_high"] = hi
        pair_summary[f"{metric}_trainable_wins"] = int(np.sum(values > 0))
        pair_summary[f"{metric}_da_adb_wins"] = int(np.sum(values < 0))
        pair_summary[f"{metric}_ties"] = int(np.sum(values == 0))
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(OUT / "per_seed.csv", index=False)
    summary.to_csv(OUT / "summary_mean_std_ci.csv", index=False)
    pair_df.to_csv(OUT / "paired_deltas.csv", index=False)
    (OUT / "paired_summary.json").write_text(
        json.dumps(pair_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "experiment": "da_adb_current_protocol_summary_v1",
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seeds": list(SEEDS),
        "source_da_runs": [row["run_manifest"] for row in da_rows],
        "source_s2c_per_seed": str(S2C_PER_SEED.resolve()),
        "source_s2c_per_seed_sha256": sha256_file(S2C_PER_SEED),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": 20_000,
        "ranking_policy": "paired descriptive comparison only; no cross-contract SOTA ranking",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"DA-ADB": "#d95f02", "S2C-Trainable-K1": "#1b9e77"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(SEEDS))
    for ax, metric, title in zip(axes, ["oos_f1", "f1_all"], ["OOS F1", "F1-All"]):
        for method in ["DA-ADB", "S2C-Trainable-K1"]:
            g = per_seed[per_seed["method"] == method].sort_values("seed")
            ax.plot(x, g[metric] * 100, marker="o", linewidth=2, label=method, color=colors[method])
        ax.set_xticks(x, [str(seed) for seed in SEEDS])
        ax.set_xlabel("seed")
        ax.set_ylabel("percent")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("StackOverflow protocol_v2: DA-ADB vs S2C Trainable K=1", fontsize=13)
    fig.savefig(FIG / "current_protocol_seed_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for method in ["DA-ADB", "S2C-Trainable-K1"]:
        g = per_seed[per_seed["method"] == method].sort_values("seed")
        ax.scatter(g["false_acceptance"] * 100, g["oos_f1"] * 100, s=70, label=method, color=colors[method])
        for _, row in g.iterrows():
            ax.annotate(str(int(row["seed"])), (row["false_acceptance"] * 100, row["oos_f1"] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("false acceptance (%)")
    ax.set_ylabel("OOS F1 (%)")
    ax.set_title("Known/OOS operating-point comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIG / "current_protocol_tradeoff.png", dpi=180)
    plt.close(fig)

    da_summary = summary[summary.method == "DA-ADB"].iloc[0]
    s2c_summary = summary[summary.method == "S2C-Trainable-K1"].iloc[0]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# DA-ADB 当前协议三 seed 汇总 V1

## 运行范围

- 数据集：StackOverflow；KIR=`0.50`；seed=`42, 87, 100`。
- DA-ADB：BERT/TextOIR 外部兼容合同，使用当前 protocol_v2 的独立 split 根和 seed-specific Known 列表。
- S2C：同一 seed 的 Known-only MiniLM Trainable K=1 fair 行。
- 测试 OOS 只用于最终评价，不参与训练、epoch、阈值或参数选择。
- 两者仍不是同骨干/同训练合同，本报告只作配对描述和合同审计。

## 当前协议结果

| 方法 | OOS F1 | F1-All | F1-Known | Accuracy | Known Recall | FA | FR |
|---|---:|---:|---:|---:|---:|---:|---:|
| DA-ADB | {da_summary['oos_f1_mean']*100:.2f} ± {da_summary['oos_f1_std']*100:.2f} | {da_summary['f1_all_mean']*100:.2f} ± {da_summary['f1_all_std']*100:.2f} | {da_summary['f1_known_mean']*100:.2f} ± {da_summary['f1_known_std']*100:.2f} | {da_summary['accuracy_mean']*100:.2f} ± {da_summary['accuracy_std']*100:.2f} | {da_summary['known_recall_mean']*100:.2f} ± {da_summary['known_recall_std']*100:.2f} | {da_summary['false_acceptance_mean']*100:.2f} ± {da_summary['false_acceptance_std']*100:.2f} | {da_summary['false_rejection_mean']*100:.2f} ± {da_summary['false_rejection_std']*100:.2f} |
| S2C Trainable K=1 | {s2c_summary['oos_f1_mean']*100:.2f} ± {s2c_summary['oos_f1_std']*100:.2f} | {s2c_summary['f1_all_mean']*100:.2f} ± {s2c_summary['f1_all_std']*100:.2f} | {s2c_summary['f1_known_mean']*100:.2f} ± {s2c_summary['f1_known_std']*100:.2f} | {s2c_summary['accuracy_mean']*100:.2f} ± {s2c_summary['accuracy_std']*100:.2f} | {s2c_summary['known_recall_mean']*100:.2f} ± {s2c_summary['known_recall_std']*100:.2f} | {s2c_summary['false_acceptance_mean']*100:.2f} ± {s2c_summary['false_acceptance_std']*100:.2f} | {s2c_summary['false_rejection_mean']*100:.2f} ± {s2c_summary['false_rejection_std']*100:.2f} |

## 配对差值：S2C Trainable K=1 − DA-ADB

| 指标 | 平均差值 | 95% bootstrap CI | S2C 胜出/DA-ADB 胜出/平局 |
|---|---:|---:|---:|
"""
        + "\n".join(
            f"| {metric} | {pair_summary[f'{metric}_mean_delta']*100:+.2f}pp | [{pair_summary[f'{metric}_ci95_low']*100:+.2f}, {pair_summary[f'{metric}_ci95_high']*100:+.2f}]pp | {pair_summary[f'{metric}_trainable_wins']}/{pair_summary[f'{metric}_da_adb_wins']}/{pair_summary[f'{metric}_ties']} |"
            for metric in metrics
        )
        + f"""

## 解释边界

当前协议下 DA-ADB 三个 seed 的 OOS F1 分别为
`{', '.join(f'{x:.2f}%' for x in per_seed.loc[per_seed.method == 'DA-ADB', 'oos_f1'].mul(100))}`，
S2C Trainable K=1 分别为
`{', '.join(f'{x:.2f}%' for x in per_seed.loc[per_seed.method == 'S2C-Trainable-K1', 'oos_f1'].mul(100))}`。
因此旧兼容单格 `90.90%` 不能代表当前协议 DA-ADB；当前三 seed 均值也不能与历史论文表或 DCLOOS reduced 结果直接排名。

当前证据只支持：在这三个 StackOverflow protocol_v2 工作点上，S2C Trainable K=1 的 OOS F1 和 F1-All 更高，
而 DA-ADB 的结果方差较大；这仍同时包含 MiniLM/BERT 表示、训练目标和外部适配层差异。

## 证据文件

- `results/analysis/da_adb_current_protocol_summary_v1/per_seed.csv`
- `results/analysis/da_adb_current_protocol_summary_v1/summary_mean_std_ci.csv`
- `results/analysis/da_adb_current_protocol_summary_v1/paired_deltas.csv`
- `figures/da_adb_current_protocol_summary_v1/current_protocol_seed_metrics.png`
- `figures/da_adb_current_protocol_summary_v1/current_protocol_tradeoff.png`
- 旧/新合同拆分：`docs/analysis/DA_ADB_CONTRACT_COMPARISON_V1.md`
""",
        encoding="utf-8",
    )
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
