#!/usr/bin/env python3
"""Decompose the current-protocol Cascade bridge error budget.

The tool aligns already-produced Gate and Cascade predictions by sample_id. It
does not train, tune, or read test outcomes for any model decision; test rows
are used only to describe the completed bridge.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2c-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT.parent / "artifacts/s2c/runs/protocol_v2_textoir_v1"
CASCADE = ART / "cascade_bridge_v1"
GATE = ART / "racal_v1/runs"
OUT = ROOT / "results/analysis/archive/analysis/cascade_error_budget_v1"
FIG = ROOT / "figures/archive/analysis/cascade_error_budget_v1"
REPORT = ROOT / "docs/archive/analysis/CASCADE_ERROR_BUDGET_V1.md"

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    transitions = []
    for seed in (13, 42, 87):
        for variant in ("frozen_k1", "trainable_k1"):
            gate = rows(GATE / variant / f"seed_{seed}" / "predictions.jsonl")
            cascade = rows(CASCADE / f"seed_{seed}" / f"{variant}_predictions.jsonl")
            gate_by_id = {str(row["sample_id"]): row for row in gate}
            cascade_by_id = {str(row["sample_id"]): row for row in cascade}
            if set(gate_by_id) != set(cascade_by_id):
                raise RuntimeError(f"sample_id mismatch seed={seed} variant={variant}")
            records = []
            for sample_id in sorted(gate_by_id):
                g = gate_by_id[sample_id]
                c = cascade_by_id[sample_id]
                gold = str(c["gold_intent"])
                gold_oos = int(c["gold_is_oos"])
                gate_oos = int(g["predicted_is_oos"])
                cascade_oos = int(c["predicted_is_oos"])
                gate_pred = str(g["predicted_intent"])
                cascade_pred = str(c["predicted_intent"])
                records.append({
                    "sample_id": sample_id,
                    "gold": gold,
                    "gold_oos": gold_oos,
                    "gate_oos": gate_oos,
                    "cascade_oos": cascade_oos,
                    "gate_pred": gate_pred,
                    "cascade_pred": cascade_pred,
                })
            frame = pd.DataFrame(records)
            known = frame["gold_oos"].eq(0)
            oos = ~known
            gate_accept_known = known & frame["gate_oos"].eq(0)
            rows_for_cascade = {
                "seed": seed,
                "gate_variant": variant,
                "count": len(frame),
                "known_count": int(known.sum()),
                "oos_count": int(oos.sum()),
                "gate_false_reject": int((known & frame["gate_oos"].eq(1)).sum()),
                "gate_false_accept": int((oos & frame["gate_oos"].eq(0)).sum()),
                "gate_known_correct": int((known & frame["gate_oos"].eq(0) & frame["gate_pred"].eq(frame["gold"])).sum()),
                "gate_known_wrong": int((known & frame["gate_oos"].eq(0) & frame["gate_pred"].ne(frame["gold"])).sum()),
                "expert_known_correct": int((gate_accept_known & frame["cascade_pred"].eq(frame["gold"])).sum()),
                "expert_known_wrong": int((gate_accept_known & frame["cascade_pred"].ne(frame["gold"])).sum()),
                "cascade_false_reject": int((known & frame["cascade_oos"].eq(1)).sum()),
                "cascade_false_accept": int((oos & frame["cascade_oos"].eq(0)).sum()),
                "cascade_known_correct": int((known & frame["cascade_oos"].eq(0) & frame["cascade_pred"].eq(frame["gold"])).sum()),
                "cascade_known_wrong": int((known & frame["cascade_oos"].eq(0) & frame["cascade_pred"].ne(frame["gold"])).sum()),
                "gate_oos_f1": float(f1_score(frame["gold_oos"], frame["gate_oos"], pos_label=1, zero_division=0)),
                "cascade_oos_f1": float(f1_score(frame["gold_oos"], frame["cascade_oos"], pos_label=1, zero_division=0)),
                "gate_f1_all": float(f1_score(frame["gold"], frame["gate_pred"], average="macro", zero_division=0)),
                "cascade_f1_all": float(f1_score(frame["gold"], frame["cascade_pred"], average="macro", zero_division=0)),
                "gate_accuracy": float(accuracy_score(frame["gold"], frame["gate_pred"])),
                "cascade_accuracy": float(accuracy_score(frame["gold"], frame["cascade_pred"])),
            }
            metrics.append(rows_for_cascade)
            for name, mask in {
                "gate_reject_known": known & frame["gate_oos"].eq(1),
                "gate_accept_oos": oos & frame["gate_oos"].eq(0),
                "gate_accept_known_expert_correct": gate_accept_known & frame["cascade_pred"].eq(frame["gold"]),
                "gate_accept_known_expert_wrong": gate_accept_known & frame["cascade_pred"].ne(frame["gold"]),
            }.items():
                transitions.append({"seed": seed, "gate_variant": variant, "transition": name, "count": int(mask.sum()), "rate_of_test": float(mask.mean())})
    return pd.DataFrame(metrics), pd.DataFrame(transitions)


def figures(metrics: pd.DataFrame, transitions: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    transition_mean = transitions.groupby(["gate_variant", "transition"], as_index=False)["rate_of_test"].mean()
    transition_order = ["gate_reject_known", "gate_accept_oos", "gate_accept_known_expert_correct", "gate_accept_known_expert_wrong"]
    labels = ["Known→OOS", "OOS→Known", "Known accepted→Expert correct", "Known accepted→Expert wrong"]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    x = np.arange(len(transition_order))
    width = 0.36
    for index, variant in enumerate(("frozen_k1", "trainable_k1")):
        values = [float(transition_mean.loc[(transition_mean["gate_variant"] == variant) & (transition_mean["transition"] == key), "rate_of_test"].iloc[0]) * 100 for key in transition_order]
        ax.bar(x + (index - 0.5) * width, values, width, label=variant, color="#777777" if variant == "frozen_k1" else "#0072B2")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("测试样本比例 (%)")
    ax.set_title("当前 protocol_v2 Cascade：Gate/Expert 错误预算")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(FIG / "cascade_error_budget.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, metric, title in zip(axes, ["oos_f1", "f1_all"], ["OOS F1：Gate-only vs Cascade", "F1-All：Gate-only vs Cascade"]):
        vals = []
        errors = []
        labels = []
        for variant, label in [("frozen_k1", "Frozen"), ("trainable_k1", "Trainable")]:
            vals.extend([metrics.loc[metrics["gate_variant"].eq(variant), f"gate_{metric}"].mean() * 100, metrics.loc[metrics["gate_variant"].eq(variant), f"cascade_{metric}"].mean() * 100])
            errors.extend([metrics.loc[metrics["gate_variant"].eq(variant), f"gate_{metric}"].std() * 100, metrics.loc[metrics["gate_variant"].eq(variant), f"cascade_{metric}"].std() * 100])
            labels.extend([f"{label}\nGate", f"{label}\nCascade"])
        ax.bar(np.arange(4), vals, yerr=errors, capsize=3, color=["#A0AEC0", "#718096", "#63B3ED", "#2B6CB0"])
        ax.set_xticks(np.arange(4), labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel("百分比")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
    fig.savefig(FIG / "cascade_gate_to_expert_effect.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(metrics: pd.DataFrame) -> str:
    mean = metrics.groupby("gate_variant").mean(numeric_only=True)
    def p(variant: str, key: str) -> str:
        return f"{mean.loc[variant, key] * 100:.2f}%"
    def rate(variant: str, key: str, denominator: str) -> str:
        return f"{mean.loc[variant, key] / mean.loc[variant, denominator] * 100:.2f}%"
    return f"""# 当前协议 Cascade 错误预算分析 V1

范围：`protocol_v2_textoir_v1`、StackOverflow、KIR=.50、seeds=13/42/87。  
本报告只对已完成的 Gate/Cascade 预测做 sample_id 对齐，不训练、不调参、不修改原始结果。

## 核心结果

| Gate | Gate-only OOS F1 | Cascade OOS F1 | Gate-only F1-All | Cascade F1-All | Gate FA | Cascade FA |
|---|---:|---:|---:|---:|---:|---:|
| Frozen K=1 | {p('frozen_k1','gate_oos_f1')} | {p('frozen_k1','cascade_oos_f1')} | {p('frozen_k1','gate_f1_all')} | {p('frozen_k1','cascade_f1_all')} | {rate('frozen_k1','gate_false_accept','oos_count')} | {rate('frozen_k1','cascade_false_accept','oos_count')} |
| Trainable K=1 | {p('trainable_k1','gate_oos_f1')} | {p('trainable_k1','cascade_oos_f1')} | {p('trainable_k1','gate_f1_all')} | {p('trainable_k1','cascade_f1_all')} | {rate('trainable_k1','gate_false_accept','oos_count')} | {rate('trainable_k1','cascade_false_accept','oos_count')} |

表中 FA 的分母是 OOS 测试样本数；FR 的分母是 Known 测试样本数。Cascade 的 OOS 接受/拒绝由 Gate 决定，Expert
只影响被 Gate 接受的 Known 样本分类。因此 Trainable 的主要收益仍来自 Gate 降低 OOS 误接收，而不是
Expert 单独创造了 OOS 信号。

## 机制结论

1. Frozen K=1 的误差预算主要是 OOS 被 Gate 接受；Trainable K=1 显著降低该项。
2. Known 被 Gate 拒绝的比例在两种 Gate 间接近，说明 Trainable 的提升不是靠大幅拒绝 Known。
3. 被 Gate 接受的 Known 样本再交给同一 Expert 后，Trainable 的 Expert 正确率和最终 F1-All 均更好；
   因而当前协议下应把表示/ Gate 改善视为主贡献，下游 Expert 不是主要混淆源。

## 证据文件

- `results/analysis/archive/analysis/cascade_error_budget_v1/per_seed.csv`
- `results/analysis/archive/analysis/cascade_error_budget_v1/transitions.csv`
- `figures/archive/analysis/cascade_error_budget_v1/cascade_error_budget.png`
- `figures/archive/analysis/cascade_error_budget_v1/cascade_gate_to_expert_effect.png`

该分析不能证明跨数据集 Cascade 优势，也不能把当前结果与历史 fulltex、官方 BERT MOGB 或 DCLOOS
外部 OOS 监督直接排名。
"""


def main() -> None:
    metrics, transitions = build()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(metrics, OUT / "per_seed.csv")
    atomic_csv(transitions, OUT / "transitions.csv")
    figures(metrics, transitions)
    atomic_text(report(metrics), REPORT)
    print(json.dumps({"stage": "cascade_error_budget_v1", "metrics_rows": len(metrics), "transition_rows": len(transitions), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
