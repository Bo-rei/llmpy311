#!/usr/bin/env python3
"""Summarise the current-protocol StackOverflow Cascade bridge.

This is an analysis-only tool.  It reads the six completed rows produced by
``run_protocol_v2_cascade_bridge_v1.py`` and never retrains or overwrites any
historical Gate, MOGB, E2/E3, or R1 artifact.
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

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1" / "cascade_bridge_v1"
OUT = ROOT / "results" / "analysis" / "cascade_bridge_v1"
FIG = ROOT / "figures" / "cascade_bridge_v1"
REPORT = ROOT / "docs" / "analysis" / "CASCADE_BRIDGE_V1.md"
METRICS = [
    "oos_f1",
    "f1_all",
    "f1_k",
    "accuracy",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
    "auroc",
    "aupr_oos",
]

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_metrics() -> pd.DataFrame:
    payload = json.loads((ARTIFACT_ROOT / "metrics.json").read_text(encoding="utf-8"))
    rows = payload.get("metrics", [])
    frame = pd.DataFrame(rows)
    required = {"seed", "gate_variant", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"cascade metrics missing columns: {sorted(missing)}")
    expected = {(seed, variant) for seed in (13, 42, 87) for variant in ("frozen_k1", "trainable_k1")}
    observed = {(int(row.seed), str(row.gate_variant)) for row in frame.itertuples()}
    if observed != expected or len(frame) != len(expected):
        raise RuntimeError(f"expected six unique seed/variant rows, observed={sorted(observed)}")
    for column in METRICS:
        frame[column] = frame[column].astype(float)
    frame["seed"] = frame["seed"].astype(int)
    return frame.sort_values(["seed", "gate_variant"]).reset_index(drop=True)


def build_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = frame.groupby("gate_variant", as_index=False).agg(
        n_seeds=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
        **{f"{metric}_std": (metric, "std") for metric in METRICS},
    )
    paired = frame.pivot(index="seed", columns="gate_variant", values=METRICS)
    rows = []
    for metric in METRICS:
        delta = paired[metric]["trainable_k1"] - paired[metric]["frozen_k1"]
        rows.append(
            {
                "metric": metric,
                "reference": "frozen_k1",
                "candidate": "trainable_k1",
                "n_pairs": int(delta.size),
                "mean_delta": float(delta.mean()),
                "std_delta": float(delta.std(ddof=1)),
                "wins": int((delta > 0).sum()),
                "ties": int((delta == 0).sum()),
                "losses": int((delta < 0).sum()),
            }
        )
    return summary, pd.DataFrame(rows)


def plot_metric_bars(summary: pd.DataFrame) -> None:
    labels = ["OOS F1", "F1-All", "Known Recall", "FA rate", "FR rate"]
    keys = ["oos_f1_mean", "f1_all_mean", "known_recall_mean", "false_accept_rate_mean", "false_reject_rate_mean"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = {"frozen_k1": "#718096", "trainable_k1": "#2b6cb0"}
    for index, variant in enumerate(("frozen_k1", "trainable_k1")):
        row = summary.loc[summary["gate_variant"].eq(variant)].iloc[0]
        values = [float(row[key]) for key in keys]
        errors = [float(row[key.replace("_mean", "_std")]) for key in keys]
        positions = x + (index - 0.5) * width
        ax.bar(positions, values, width, yerr=errors, capsize=3, label=variant, color=colors[variant])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("metric (mean ± std)")
    ax.set_title("StackOverflow/KIR=.50/current-protocol Cascade bridge")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(FIG / "frozen_vs_trainable_cascade_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pair_deltas(paired: pd.DataFrame) -> None:
    display = paired.loc[paired["metric"].isin(["oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate"])].copy()
    labels = {"oos_f1": "OOS F1", "f1_all": "F1-All", "known_recall": "Known Recall", "false_accept_rate": "FA rate", "false_reject_rate": "FR rate"}
    display["label"] = display["metric"].map(labels)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    colors = ["#c53030" if value < 0 else "#2b6cb0" for value in display["mean_delta"]]
    ax.bar(display["label"], display["mean_delta"], color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Trainable K=1 − Frozen K=1")
    ax.set_title("Paired current-protocol Cascade effect (three seeds)")
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(FIG / "trainable_minus_frozen_cascade_effect.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(frame: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    def val(variant: str, metric: str) -> str:
        row = summary.loc[summary["gate_variant"].eq(variant)].iloc[0]
        return f"{float(row[f'{metric}_mean']) * 100:.2f}% ± {float(row[f'{metric}_std']) * 100:.2f}pp"

    delta = paired.set_index("metric")["mean_delta"]
    return f"""# 当前协议 Cascade 桥接实验（V1）

更新时间：2026-08-10
实验阶段：`cascade_bridge_v1`
范围：`protocol_v2_textoir_v1`、StackOverflow、KIR=0.50、seeds={{13,42,87}}。

## 这次补的是什么

这是一次**当前数据协议内的下游桥接实验**：同一份 `train_known` 训练 SmolLM Expert，使用
`calibration_known` 选择 checkpoint，然后分别接入已有的 Frozen K=1 和 Trainable K=1 Gate。
StackOverflow 在当前快照中只有一个 domain，因此 Router 是显式的常量路由，不伪造多领域 Router 结果。

它解决了此前的合同断点：Trainable Gate 不再直接拼接旧 v19 Router/Expert。它不重训 Gate，不修改
E2/E3/R1/MOGB artifacts，也不是 fulltex 历史 Cascade 的复现。

## 结果（均值 ± seed 标准差）

| Gate | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | FA | FR | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen K=1 | {val('frozen_k1','oos_f1')} | {val('frozen_k1','f1_all')} | {val('frozen_k1','f1_k')} | {val('frozen_k1','accuracy')} | {val('frozen_k1','known_recall')} | {val('frozen_k1','false_accept_rate')} | {val('frozen_k1','false_reject_rate')} | {val('frozen_k1','auroc')} |
| Trainable K=1 | {val('trainable_k1','oos_f1')} | {val('trainable_k1','f1_all')} | {val('trainable_k1','f1_k')} | {val('trainable_k1','accuracy')} | {val('trainable_k1','known_recall')} | {val('trainable_k1','false_accept_rate')} | {val('trainable_k1','false_reject_rate')} | {val('trainable_k1','auroc')} |

Trainable K=1 相对 Frozen K=1 的配对均值差值：OOS F1 `{delta['oos_f1'] * 100:+.2f}pp`、
F1-All `{delta['f1_all'] * 100:+.2f}pp`、Known Recall `{delta['known_recall'] * 100:+.2f}pp`、
FA `{delta['false_accept_rate'] * 100:+.2f}pp`、FR `{delta['false_reject_rate'] * 100:+.2f}pp`、
Accuracy `{delta['accuracy'] * 100:+.2f}pp`。

## 与旧结果和外部基线的关系

| 结果层 | 能否直接和本次表格混排 | 原因 |
|---|---|---|
| 当前 Trainable/Frozen K=1 Cascade | 可以，在本报告内部 | 同一 TEXTOIR protocol、同一 views、同一 Expert checkpoint/seed |
| 当前 fair Gate matrix | 只能作 Gate-only 参照 | 没有 Router/Expert；本次只改变下游层 |
| fulltex 历史 `Ours` | 不可以直接排名 | 是旧合同的完整 Gate–Router–Expert Cascade |
| MOGB-MiniLM-Fair | 只能作同协议 Gate 组件参照 | 没有同一当前协议下的 Cascade Expert |
| MOGB 官方 BERT 兼容复现 | 不可以混成同协议主表 | 数据快照、BERT/训练损失、旧环境和论文配置未完全可恢复 |
| ADB/DCLOOS | 不能直接作为本表 SOTA 排名 | 训练监督、backbone 或外部 OOS 条件不同 |

## 机制解读

1. Trainable K=1 的收益仍主要来自 Gate 的分数排序和较低的 OOS 误接收；在当前 Cascade 中，
   同一 Expert 只在 Gate 接受的 Known 样本上工作，因此下游训练没有把旧 v19 数据差异混入结果。
2. Frozen K=1 的主要损失仍是 Gate false acceptance 较高，而不是 Expert 本身完全失效。
3. 该实验没有证明 Trainable K=1 已经超过 MOGB 论文或 DCLOOS。它只证明：在同一当前协议、
   同一 Expert 和相同已知/未知测试集合下，Trainable Gate 的下游桥接优于 Frozen Gate。

## 可复现性与限制

- `metrics.json` 中 `test_used_for_selection=false`、`oos_used_for_training=false`。
- 每个 seed 记录 train/calibration/test sample-id hash 和 Expert checkpoint SHA256。
- 完整模型 checkpoint 保留在 artifacts，轻量汇总和图在 `results/analysis/cascade_bridge_v1/`、
  `figures/cascade_bridge_v1/`。
- 只有三个 StackOverflow seed，尚未覆盖 CLINC150/Banking77，也没有 Router 多 domain 训练。
- 旧 fulltex Cascade、MOGB 官方论文数字和 DCLOOS 外部 OOS 结果仍需要单独的统一协议桥接，不能从本实验外推。

## 证据文件

- `results/analysis/cascade_bridge_v1/per_seed.csv`
- `results/analysis/cascade_bridge_v1/summary_mean_std.csv`
- `results/analysis/cascade_bridge_v1/paired_effects.csv`
- `figures/cascade_bridge_v1/frozen_vs_trainable_cascade_metrics.png`
- `figures/cascade_bridge_v1/trainable_minus_frozen_cascade_effect.png`
- `../artifacts/s2c/runs/protocol_v2_textoir_v1/cascade_bridge_v1/PROVENANCE.json`
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame = load_metrics()
    summary, paired = build_summary(frame)
    atomic_csv(frame, OUT / "per_seed.csv")
    atomic_csv(summary, OUT / "summary_mean_std.csv")
    atomic_csv(paired, OUT / "paired_effects.csv")
    plot_metric_bars(summary)
    plot_pair_deltas(paired)
    atomic_text(REPORT, build_report(frame, summary, paired))
    manifest = {
        "stage": "cascade_bridge_v1",
        "protocol": "protocol_v2_textoir_v1",
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seeds": [13, 42, 87],
        "completed_rows": int(len(frame)),
        "test_used_for_selection": False,
        "oos_used_for_training": False,
        "source_metrics_sha256": sha256_file(ARTIFACT_ROOT / "metrics.json"),
        "source_provenance_sha256": sha256_file(ARTIFACT_ROOT / "PROVENANCE.json"),
        "summary_paths": [
            str((OUT / "per_seed.csv").relative_to(ROOT)),
            str((OUT / "summary_mean_std.csv").relative_to(ROOT)),
            str((OUT / "paired_effects.csv").relative_to(ROOT)),
        ],
        "figure_paths": [
            str((FIG / "frozen_vs_trainable_cascade_metrics.png").relative_to(ROOT)),
            str((FIG / "trainable_minus_frozen_cascade_effect.png").relative_to(ROOT)),
        ],
    }
    atomic_text(OUT / "MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({"stage": manifest["stage"], "completed_rows": len(frame), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
