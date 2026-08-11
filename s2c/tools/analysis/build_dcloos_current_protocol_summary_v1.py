#!/usr/bin/env python3
"""Summarize a fixed-registry DCLOOS cell without hiding contract differences.

The DCLOOS row uses the current StackOverflow train/dev/test snapshot and the
same protocol Known-label file, but retains DCLOOS's pseudo/external-OOS
supervision.  It is therefore an adapted external-supervision reference, not a
Known-only fair ranking row.
"""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts" / "s2c"
RUN = ARTIFACTS / "external" / "dcloos_stackoverflow_kir050_seed42_fixed_registry_v1"
S2C = ROOT / "results" / "analysis" / "cross_protocol_tradeoff_v1" / "per_seed.csv"
OUT = ROOT / "results" / "analysis" / "dcloos_current_protocol_summary_v1"
FIG = ROOT / "figures" / "dcloos_current_protocol_summary_v1"
REPORT = ROOT / "docs" / "analysis" / "DCLOOS_CURRENT_PROTOCOL_SUMMARY_V1.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dcloos_row(run_root: Path) -> dict[str, object]:
    manifest_path = run_root / "run_manifest.json"
    metrics_path = run_root / "metrics.json"
    pred_path = run_root / "predictions.npz"
    for path in (manifest_path, metrics_path, pred_path):
        if not path.is_file():
            raise RuntimeError(f"incomplete DCLOOS artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    arrays = np.load(pred_path)
    y_true = arrays["y_true"].astype(int)
    y_pred = arrays["y_pred"].astype(int)
    unknown = int(metrics["known_class_count"])
    known = y_true != unknown
    oos = ~known
    pred_oos = y_pred == unknown
    known_labels = list(range(unknown))
    return {
        "method": "DCLOOS-fixed-registry",
        "contract": "current split + fixed Known list + BERT + pseudo/external OOS",
        "scope": "adapted_external_supervision_reference",
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seed": 42,
        "oos_f1": float(f1_score(oos, pred_oos, zero_division=0)),
        "oos_precision": float(((oos & pred_oos).sum() / max(pred_oos.sum(), 1))),
        "oos_recall": float(((oos & pred_oos).sum() / max(oos.sum(), 1))),
        "f1_all": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_k": float(f1_score(y_true, y_pred, labels=known_labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "known_recall": float(((y_pred[known] != unknown).sum() / max(known.sum(), 1))),
        "false_acceptance": float(((y_pred[oos] != unknown).sum() / max(oos.sum(), 1))),
        "false_rejection": float(((y_pred[known] == unknown).sum() / max(known.sum(), 1))),
        "n_samples": int(y_true.size),
        "known_class_count": unknown,
        "known_labels_file": manifest.get("known_labels_file"),
        "run_manifest": str(manifest_path.resolve()),
        "run_manifest_sha256": sha256(manifest_path),
        "metrics_sha256": sha256(metrics_path),
        "predictions_sha256": sha256(pred_path),
        "negative_corpus_sha256": manifest.get("negative_corpus_sha256"),
        "best_epoch": metrics.get("best_epoch"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=RUN,
        help="Completed fixed-registry DCLOOS artifact root (must contain final metrics.json).",
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    dcloos = dcloos_row(run_root)
    source = pd.read_csv(S2C)
    source = source[(source.dataset == "stackoverflow") & (source.kir == 0.50) & (source.seed == 42)].copy()
    wanted = ["single_centroid", "random_partition", "fixed_k2", "mogb_minilm", "mogb_partition_ours_boundary", "trainable_k1"]
    source = source[source.method.isin(wanted)]
    rows = []
    for _, row in source.iterrows():
        rows.append({
            "method": row["method"],
            "contract": row.get("supervision", "protocol_v2 Known-only") + " / " + row.get("representation", ""),
            "scope": "same_protocol_known_only_gate",
            "dataset": row["dataset"],
            "kir": row["kir"],
            "seed": int(row["seed"]),
            "oos_f1": row["oos_f1"],
            "oos_precision": row["oos_precision"],
            "oos_recall": row["oos_recall"],
            "f1_all": row["f1_all"],
            "f1_k": row["f1_k"],
            "accuracy": row["accuracy"],
            "known_recall": row["known_recall"],
            "false_acceptance": row["false_accept_rate"],
            "false_rejection": row["false_reject_rate"],
            "n_samples": None,
        })
    rows.append(dcloos)
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "per_method.csv", index=False)
    metrics = ["oos_f1", "f1_all", "f1_k", "accuracy", "known_recall", "false_acceptance", "false_rejection"]
    frame[frame.scope == "same_protocol_known_only_gate"].to_csv(OUT / "known_only_reference_rows.csv", index=False)
    frame[frame.scope != "same_protocol_known_only_gate"].to_csv(OUT / "external_supervision_rows.csv", index=False)
    summary = frame.groupby(["method", "scope"], as_index=False)[metrics].mean()
    summary.to_csv(OUT / "summary.csv", index=False)
    contracts = frame[["method", "scope", "contract"]].drop_duplicates()
    contracts.to_csv(OUT / "contract_matrix.csv", index=False)
    manifest = {
        "experiment": "dcloos_current_protocol_summary_v1",
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seed": 42,
        "dcloos_run": str(run_root),
        "dcloos_manifest_sha256": dcloos["run_manifest_sha256"],
        "s2c_source": str(S2C.resolve()),
        "s2c_source_sha256": sha256(S2C),
        "ranking_policy": "DCLOOS remains separate because pseudo/external OOS supervision differs; no pooled SOTA ranking",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    order = wanted + ["DCLOOS-fixed-registry"]
    plot = frame.set_index("method").reindex(order).reset_index()
    colors = {"DCLOOS-fixed-registry": "#7570b3", "trainable_k1": "#1b9e77"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(order))
    for ax, metric, title in zip(axes, ["oos_f1", "f1_all"], ["OOS F1", "F1-All"]):
        values = plot[metric].to_numpy(dtype=float) * 100
        bars = ax.bar(x, values, color=[colors.get(m, "#56b4e9") for m in order])
        ax.set_xticks(x, [m.replace("_", "\\n") for m in order], rotation=0, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("percent")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("StackOverflow KIR=0.50 seed=42: DCLOOS and current S2C references", fontsize=13)
    fig.savefig(FIG / "dcloos_vs_s2c_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for _, row in plot.iterrows():
        color = colors.get(row["method"], "#56b4e9")
        ax.scatter(row["false_acceptance"] * 100, row["oos_f1"] * 100, s=75, color=color)
        ax.annotate(row["method"], (row["false_acceptance"] * 100, row["oos_f1"] * 100), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("false acceptance (%)")
    ax.set_ylabel("OOS F1 (%)")
    ax.set_title("Known/OOS operating points (contract-aware)")
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "dcloos_tradeoff.png", dpi=180)
    plt.close(fig)

    trainable = frame[frame.method == "trainable_k1"].iloc[0]
    dcl = frame[frame.method == "DCLOOS-fixed-registry"].iloc[0]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# DCLOOS 当前 StackOverflow 单格汇总 V1

## 运行合同

- 数据集：StackOverflow；KIR=`0.50`；seed=`42`。
- 正样本 train/dev/test 与 `protocol_v2_textoir_v1` 的 StackOverflow registry export 对齐。
- Known 列表由同一 registry 的 `known_labels.json` 固定注入，避免 DCLOOS 上游再次随机抽类。
- DCLOOS 仍使用 BERT、pseudo-OOS 与外部 SQuAD OOS；因此它是 `adapted_external_supervision_reference`，不是 Known-only fair 行。
- 外部 SQuAD 快照 SHA256：`{dcl['negative_corpus_sha256']}`。
- 最佳 epoch（按 DCLOOS validation 规则）：`{dcl['best_epoch']}`。

## 当前单格结果

| 方法 | 合同 | OOS F1 | F1-All | F1-Known | Accuracy | Known Recall | FA | FR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S2C Trainable K=1 | Known-only MiniLM Gate | {trainable['oos_f1']*100:.2f} | {trainable['f1_all']*100:.2f} | {trainable['f1_k']*100:.2f} | {trainable['accuracy']*100:.2f} | {trainable['known_recall']*100:.2f} | {trainable['false_acceptance']*100:.2f} | {trainable['false_rejection']*100:.2f} |
| DCLOOS fixed-registry | BERT + pseudo/external OOS | {dcl['oos_f1']*100:.2f} | {dcl['f1_all']*100:.2f} | {dcl['f1_k']*100:.2f} | {dcl['accuracy']*100:.2f} | {dcl['known_recall']*100:.2f} | {dcl['false_acceptance']*100:.2f} | {dcl['false_rejection']*100:.2f} |

其余当前协议 Gate 参考行见 `results/analysis/dcloos_current_protocol_summary_v1/per_method.csv`；它们只用于显示同一测试工作点的覆盖—拒识位置，不构成把不同训练合同强行合并的 SOTA 排名。

## 解释边界

1. 这个单格可以回答“在同一 StackOverflow registry 工作点上，DCLOOS 外部未知监督与 S2C Known-only Gate 的运行点如何不同”。
2. 它不能回答“谁在相同监督条件下 SOTA”，因为 DCLOOS 使用外部 SQuAD 和 pseudo-OOS，而 S2C/MOGB-Fair 只用 Known 数据。
3. 若 DCLOOS 的 OOS F1 较高，应同时检查 F1-Known、Known Recall 和 false rejection，不能只看 OOS F1。
4. 历史 DCLOOS reduced `KIR=.75/seed=888` 仍保留为另一合同，不能覆盖或替换本单格。

## 文件

- `results/analysis/dcloos_current_protocol_summary_v1/per_method.csv`
- `results/analysis/dcloos_current_protocol_summary_v1/contract_matrix.csv`
- `figures/dcloos_current_protocol_summary_v1/dcloos_vs_s2c_metrics.png`
- `figures/dcloos_current_protocol_summary_v1/dcloos_tradeoff.png`
- `results/analysis/dcloos_current_protocol_summary_v1/MANIFEST.json`
""",
        encoding="utf-8",
    )
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
