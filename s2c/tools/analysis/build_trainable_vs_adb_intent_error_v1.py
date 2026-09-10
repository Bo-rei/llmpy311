#!/usr/bin/env python3
"""Build intent-level, sample-aligned Trainable-K1 versus ADB diagnostics.

This is a post-hoc analysis of completed predictions.  It does not fit a
model, choose a threshold, or export text.  Known errors are grouped by their
gold intent; OOS false accepts are grouped by the predicted Known intent.
Those two views answer different questions and are kept in separate columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis.build_trainable_vs_adb_cross_dataset_error_budget_v1 import (  # noqa: E402
    ADB_SOURCE,
    DATASETS,
    FAIR_SOURCE,
    KIRS,
    SEEDS,
    classify,
    load_adb,
    load_sources,
    load_trainable,
    load_view,
    resolve_project_path,
    sha256,
)


OUT_ROOT = Path("results/analysis/archive/analysis/trainable_vs_adb_intent_error_v1")
FIG_ROOT = Path("figures/archive/analysis/trainable_vs_adb_intent_error_v1")
REPORT = Path("docs/archive/analysis/TRAINABLE_VS_ADB_INTENT_ERROR_V1.md")


def _safe_rate(value: int, denominator: int) -> float:
    return float(value) / max(1, int(denominator))


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    fair, adb = load_sources()
    fair_by_key = {
        tuple(row[key] for key in ("dataset", "kir", "seed")): row
        for _, row in fair.iterrows()
    }
    adb_by_key = {
        tuple(row[key] for key in ("dataset", "kir", "seed")): row
        for _, row in adb.iterrows()
    }

    intent_rows: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                key = (dataset, kir, seed)
                view = load_view(dataset, kir, seed)
                trainable, train_manifest, train_pred_path = load_trainable(
                    fair_by_key[key], view
                )
                adb_true, adb_pred, adb_manifest, adb_test_path, adb_pred_root = load_adb(
                    adb_by_key[key], view
                )
                if len(trainable) != len(adb_true) or len(view) != len(trainable):
                    raise ValueError(f"Prediction count mismatch: {key}")

                known_rows = [
                    str(row["evaluation_label"])
                    for row in view
                    if str(row["evaluation_label"]) != "oos"
                ]
                intents = sorted(set(known_rows))
                n_oos = sum(str(row["evaluation_label"]) == "oos" for row in view)
                by_intent: dict[str, dict[str, int]] = {
                    intent: {
                        "n_known": 0,
                        "trainable_correct": 0,
                        "trainable_wrong": 0,
                        "trainable_rejected": 0,
                        "adb_correct": 0,
                        "adb_wrong": 0,
                        "adb_rejected": 0,
                        "trainable_oos_false_accept": 0,
                        "adb_oos_false_accept": 0,
                    }
                    for intent in intents
                }

                for view_row, train_row, adb_gold, adb_prediction in zip(
                    view, trainable, adb_true, adb_pred
                ):
                    gold = str(view_row["evaluation_label"])
                    train_pred = str(train_row["predicted_intent"])
                    train_oos = bool(int(train_row["predicted_is_oos"]))
                    adb_oos = adb_prediction == "oos"
                    if gold == "oos":
                        if not train_oos and train_pred in by_intent:
                            by_intent[train_pred]["trainable_oos_false_accept"] += 1
                        if not adb_oos and adb_prediction in by_intent:
                            by_intent[adb_prediction]["adb_oos_false_accept"] += 1
                        continue

                    if gold not in by_intent:
                        raise ValueError(f"Unknown intent in view: {dataset}/{kir}/{seed}/{gold}")
                    values = by_intent[gold]
                    values["n_known"] += 1
                    train_state = classify(False, train_oos, gold, train_pred)
                    adb_state = classify(False, adb_oos, gold, adb_prediction)
                    state_suffix = {
                        "known_correct": "correct",
                        "known_wrong": "wrong",
                        "known_rejected": "rejected",
                    }
                    values[f"trainable_{state_suffix[train_state]}"] += 1
                    values[f"adb_{state_suffix[adb_state]}"] += 1

                for intent in intents:
                    values = by_intent[intent]
                    n_known = values["n_known"]
                    train_rej_rate = _safe_rate(values["trainable_rejected"], n_known)
                    adb_rej_rate = _safe_rate(values["adb_rejected"], n_known)
                    train_fa_rate = _safe_rate(
                        values["trainable_oos_false_accept"], n_oos
                    )
                    adb_fa_rate = _safe_rate(values["adb_oos_false_accept"], n_oos)
                    intent_rows.append(
                        {
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "intent": intent,
                            "n_known": n_known,
                            "n_oos": n_oos,
                            **values,
                            "trainable_known_rejected_rate": train_rej_rate,
                            "adb_known_rejected_rate": adb_rej_rate,
                            "delta_known_rejected_pp": 100.0 * (train_rej_rate - adb_rej_rate),
                            "trainable_oos_false_accept_rate": train_fa_rate,
                            "adb_oos_false_accept_rate": adb_fa_rate,
                            "delta_oos_false_accept_pp": 100.0 * (train_fa_rate - adb_fa_rate),
                            "delta_known_acceptance_pp": -100.0 * (train_rej_rate - adb_rej_rate),
                        }
                    )

                alignment.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "seed": seed,
                        "n_test": len(view),
                        "n_intents": len(intents),
                        "trainable_predictions_sha256": sha256(train_pred_path),
                        "trainable_manifest_sha256": sha256(
                            train_pred_path.parent / "run_manifest.json"
                        ),
                        "adb_test_sha256": sha256(adb_test_path),
                        "adb_manifest_sha256": sha256(
                            resolve_project_path(str(adb_by_key[key]["run_manifest"]))
                        ),
                        "adb_prediction_sha256": sha256(adb_pred_root / "y_pred.npy"),
                        "aligned": True,
                    }
                )

    raw = pd.DataFrame(intent_rows)
    mean = (
        raw.groupby(["dataset", "kir", "intent"], as_index=False)
        .agg(
            n_known=("n_known", "mean"),
            n_oos=("n_oos", "mean"),
            delta_known_rejected_pp=("delta_known_rejected_pp", "mean"),
            delta_known_rejected_std_pp=("delta_known_rejected_pp", "std"),
            delta_oos_false_accept_pp=("delta_oos_false_accept_pp", "mean"),
            delta_oos_false_accept_std_pp=("delta_oos_false_accept_pp", "std"),
            trainable_known_rejected_rate=("trainable_known_rejected_rate", "mean"),
            adb_known_rejected_rate=("adb_known_rejected_rate", "mean"),
            trainable_oos_false_accept_rate=("trainable_oos_false_accept_rate", "mean"),
            adb_oos_false_accept_rate=("adb_oos_false_accept_rate", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    return raw, mean, alignment


def make_figures(mean: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 9, "axes.titlesize": 11, "figure.dpi": 140, "font.family": "DejaVu Sans"}
    )

    # Each panel is one dataset; columns are intents sorted by mean Known
    # rejection delta.  This makes broad versus concentrated effects visible.
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        frame = mean[mean.dataset.eq(dataset)].copy()
        order = (
            frame.groupby("intent")["delta_known_rejected_pp"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        pivot = frame.pivot(index="kir", columns="intent", values="delta_known_rejected_pp").reindex(
            index=KIRS, columns=order
        )
        values = pivot.to_numpy()
        vmax = max(1.0, float(np.nanmax(np.abs(values))))
        image = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{dataset}: Trainable−ADB Known rejection (pp)")
        ax.set_yticks(range(len(KIRS)), [f"KIR={kir:.2f}" for kir in KIRS])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=5)
        ax.axhline(-0.5, color="black", linewidth=0.4)
        fig.colorbar(image, ax=ax, fraction=0.012, pad=0.01, label="pp")
    fig.savefig(out / "known_rejection_delta_by_intent.png", dpi=180)
    plt.close(fig)

    # A single compact mechanism view: each point is one intent×KIR row.
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    colors = {"clinc150": "#4472c4", "banking77": "#ed7d31", "stackoverflow": "#70ad47"}
    for dataset in DATASETS:
        frame = mean[mean.dataset.eq(dataset)]
        ax.scatter(
            frame["delta_known_rejected_pp"],
            frame["delta_oos_false_accept_pp"],
            s=np.clip(frame["n_known"].to_numpy() / 2.0, 8, 60),
            alpha=0.52,
            label=dataset,
            color=colors[dataset],
            edgecolors="none",
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Trainable−ADB Known rejection (pp; positive = more Known rejected)")
    ax.set_ylabel("Trainable−ADB OOS false accept (pp; positive = more OOS accepted)")
    ax.set_title("Intent-level error budget: coverage versus open-space risk")
    ax.legend(frameon=False)
    fig.savefig(out / "intent_error_budget_scatter.png", dpi=180)
    plt.close(fig)

    # Rank the largest OOS acceptor changes, retaining only aggregate intent IDs.
    top = mean.sort_values("delta_oos_false_accept_pp", ascending=False).groupby("dataset", sort=False).head(8)
    bottom = mean.sort_values("delta_oos_false_accept_pp", ascending=True).groupby("dataset", sort=False).head(8)
    combined = pd.concat(
        [
            top.assign(group="largest increase"),
            bottom.assign(group="largest decrease"),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        frame = combined[combined.dataset.eq(dataset)].copy()
        frame["label"] = frame["intent"].astype(str) + "\nKIR=" + frame["kir"].map(lambda x: f"{x:.2f}")
        frame = frame.sort_values("delta_oos_false_accept_pp")
        ax.barh(frame["label"], frame["delta_oos_false_accept_pp"], color=np.where(frame["delta_oos_false_accept_pp"] >= 0, "#c00000", "#4472c4"))
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_title(dataset)
        ax.set_xlabel("Δ OOS false accept (pp)")
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("Largest intent-level OOS false-accept changes")
    fig.savefig(out / "top_intent_oos_acceptor_changes.png", dpi=180)
    plt.close(fig)


def write_report(raw: pd.DataFrame, mean: pd.DataFrame, alignment: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    dataset_lines = []
    for dataset in DATASETS:
        frame = mean[mean.dataset.eq(dataset)]
        dataset_lines.append(
            f"- **{dataset}**：意图级 Known 误拒差值均值 `{frame.delta_known_rejected_pp.mean():+.2f}pp`；"
            f"OOS 误接收差值均值 `{frame.delta_oos_false_accept_pp.mean():+.2f}pp`；"
            f"按意图×KIR 观察到 `{len(frame)}` 个聚合点。"
        )
    text = "\n".join(
        [
            "# Trainable-K1 与 ADB 意图级错误归因 V1",
            "",
            "更新时间：2026-08-10",
            "",
            "本报告只读取已经完成且通过 sample-id、文本、标签和 manifest 对齐审计的 Trainable-K1 与 ADB 预测。",
            "Known 错误按真实 Known intent 聚合；OOS 误接收按预测吸收 OOS 的 Known intent 聚合。两者语义不同，不能合并解释。",
            "本分析不训练、不调阈值、不选择 checkpoint，也不把 BERT/TextOIR ADB 与 MiniLM Trainable 混成 SOTA 排名。",
            "",
            "## 覆盖范围",
            "",
            f"- {len(raw):,} 条 intent×dataset×KIR×seed 记录；{len(mean):,} 条 intent×dataset×KIR 聚合记录。",
            f"- 对齐单元：{len(alignment)} 个 dataset×KIR×seed；全部 `aligned=true`。",
            "- 输入：Trainable/ADB 逐样本预测和 protocol test view；轻量输出不包含文本。",
            "",
            "## 主要观察",
            "",
            *dataset_lines,
            "",
            "- 如果点落在散点图右上方，Trainable 同时增加 Known 误拒和 OOS 误接收，属于更保守但有额外开放空间风险的意图。",
            "- 如果点落在左下方，Trainable 同时恢复 Known 覆盖并减少 OOS 误接收，是最有利的工作点变化。",
            "- 该图用于解释平均差异由哪些意图贡献，不用于从测试结果选择意图、中心数或阈值。",
            "",
            "## 产物",
            "",
            "- `results/analysis/archive/analysis/trainable_vs_adb_intent_error_v1/intent_error_per_seed.csv`",
            "- `results/analysis/archive/analysis/trainable_vs_adb_intent_error_v1/intent_error_mean.csv`",
            "- `results/analysis/archive/analysis/trainable_vs_adb_intent_error_v1/ALIGNMENT.json`",
            "- `figures/archive/analysis/trainable_vs_adb_intent_error_v1/known_rejection_delta_by_intent.png`",
            "- `figures/archive/analysis/trainable_vs_adb_intent_error_v1/intent_error_budget_scatter.png`",
            "- `figures/archive/analysis/trainable_vs_adb_intent_error_v1/top_intent_oos_acceptor_changes.png`",
            "",
            f"Manifest SHA256：`{manifest['manifest_sha256']}`。",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=FIG_ROOT)
    args = parser.parse_args()
    raw, mean, alignment = build()
    args.output_root.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_root / "intent_error_per_seed.csv", index=False)
    mean.to_csv(args.output_root / "intent_error_mean.csv", index=False)
    alignment_path = args.output_root / "ALIGNMENT.json"
    alignment_path.write_text(json.dumps(alignment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_figures(mean, args.figure_root)
    manifest = {
        "analysis": "trainable_vs_adb_intent_error_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "source_fair": str(FAIR_SOURCE),
        "source_adb": str(ADB_SOURCE),
        "source_fair_sha256": sha256(FAIR_SOURCE),
        "source_adb_sha256": sha256(ADB_SOURCE),
        "raw_rows": int(len(raw)),
        "mean_rows": int(len(mean)),
        "aligned_units": int(len(alignment)),
        "output_root": str(args.output_root),
        "figure_root": str(args.figure_root),
        "no_training": True,
        "no_test_selection": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(manifest_payload).hexdigest()
    (args.output_root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(raw, mean, alignment, manifest)
    print(json.dumps({"status": "ok", "raw_rows": len(raw), "mean_rows": len(mean), "aligned_units": len(alignment)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
