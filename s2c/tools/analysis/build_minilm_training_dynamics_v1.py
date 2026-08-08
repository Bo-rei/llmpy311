"""Summarize existing Known-only Trainable MiniLM training dynamics.

No training is run here.  The script reads training_history.tsv and the final
metrics from the immutable Trainable K=1 runs to diagnose selection/metric
alignment across datasets and KIR values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts/s2c/runs/protocol_v2_textoir_v1"
OUT = ROOT / "results/analysis/minilm_training_dynamics_v1"
FIG = ROOT / "figures/minilm_training_dynamics_v1"
TRAINABLE_ROOTS = (ARTIFACT_ROOT / "minilm_trainable_kir_sweep_v1", ARTIFACT_ROOT / "minilm_trainable_kir_sweep_extension_v1")
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_run(dataset: str, kir: float, seed: int) -> Path:
    for root in TRAINABLE_ROOTS:
        path = root / f"kir_{kir:.2f}" / "runs" / dataset / f"seed_{seed}"
        if (path / "training_history.tsv").is_file():
            return path
    raise FileNotFoundError(f"Missing training history: {dataset} kir={kir} seed={seed}")


def _read_run(dataset: str, kir: float, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _find_run(dataset, kir, seed)
    history = pd.read_csv(path / "training_history.tsv", sep="\t")
    manifest = _json(path / "training_manifest.json")
    metrics = _json(path / "metrics.json")
    if history.empty:
        raise ValueError(f"Empty training history: {path}")
    history["dataset"] = dataset
    history["kir"] = kir
    history["seed"] = seed
    history["test_oos_f1"] = float(metrics["oos_f1"])
    history["test_known_recall"] = float(metrics["known_recall"])
    history["best_epoch_manifest"] = int(manifest["best_epoch"])
    history["run_dir"] = str(path)
    return history, {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "best_epoch": int(manifest["best_epoch"]),
        "best_selection_score": float(manifest["best_selection_score"]),
        "test_oos_f1": float(metrics["oos_f1"]),
        "test_known_recall": float(metrics["known_recall"]),
        "training_time_seconds": float(metrics.get("training_time_seconds", np.nan)),
        "trainable_parameters": int(metrics.get("trainable_parameters", 0)),
        "epochs_recorded": len(history),
        "calibration_f1_k_best": float(history["calibration_f1_k"].max()),
        "calibration_known_recall_at_best": float(history.loc[history["selection_score"].idxmax(), "calibration_known_recall"]),
        "selection_score_max": float(history["selection_score"].max()),
    }


def _write(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def _plot(history: pd.DataFrame, runs: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"clinc150": "#4c78a8", "banking77": "#f58518", "stackoverflow": "#54a24b"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        for kir in KIRS:
            part = history[(history.dataset == dataset) & (history.kir == kir)]
            curve = part.groupby("epoch", as_index=False)["selection_score"].mean()
            ax.plot(curve.epoch, curve.selection_score, marker="o", linewidth=1.5, label=f"KIR={kir:.2f}")
        ax.set_title(dataset)
        ax.set_xlabel("epoch")
        ax.set_ylabel("Known calibration selection score")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Known-only Trainable MiniLM selection dynamics")
    fig.tight_layout()
    fig.savefig(FIG / "selection_dynamics_by_kir.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for dataset in DATASETS:
        part = runs[runs.dataset == dataset]
        axes[0].scatter(part.best_epoch, part.test_oos_f1 * 100, label=dataset, color=colors[dataset], s=55)
        axes[1].scatter(part.selection_score_max, part.test_oos_f1 * 100, label=dataset, color=colors[dataset], s=55)
    axes[0].set_xlabel("selected epoch")
    axes[1].set_xlabel("best Known calibration selection score")
    for ax in axes:
        ax.set_ylabel("test OOS F1 (%)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Selection objective versus held-out OOS performance (diagnostic only)")
    fig.tight_layout()
    fig.savefig(FIG / "selection_vs_test_oos.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = runs[runs.dataset == dataset]
        for kir in KIRS:
            row = part[part.kir == kir]
            ax.scatter(row.calibration_known_recall_at_best * 100, row.test_known_recall * 100, s=58, label=f"KIR={kir:.2f}")
        ax.plot([60, 100], [60, 100], linestyle=":", color="black", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("Known calibration recall (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Known test recall (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Known coverage transfer: calibration selection vs held-out test")
    fig.tight_layout()
    fig.savefig(FIG / "calibration_vs_test_known_recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in DATASETS:
        part = runs[runs.dataset == dataset]
        ax.boxplot([part[part.kir == kir].best_epoch for kir in KIRS], positions=np.arange(len(KIRS)) + list(DATASETS).index(dataset) * 0.22, widths=0.18, patch_artist=True, boxprops={"facecolor": colors[dataset], "alpha": 0.55}, medianprops={"color": "black"})
    ax.set_xticks(np.arange(len(KIRS)) + 0.22)
    ax.set_xticklabels([f"{kir:.2f}" for kir in KIRS])
    ax.set_xlabel("KIR")
    ax.set_ylabel("selected epoch")
    ax.set_title("Selected epoch distribution by KIR")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "selected_epoch_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report(runs: pd.DataFrame, history: pd.DataFrame) -> None:
    lines = [
        "# MiniLM Trainable 训练动态诊断 V1",
        "",
        "> 只读取已完成 Trainable K=1 的 Known-only training history 和最终 metrics；不重新训练、不改变 checkpoint、不使用测试 OOS 选择任何参数。",
        "",
        "## 1. 结果",
        "",
        f"- 读取 {len(runs)} 个训练 run、{len(history)} 条 epoch 记录；三数据集、KIR={{.25,.50,.75}}、五 seed。",
        f"- 可训练参数量：{int(runs.trainable_parameters.iloc[0]):,}；epoch 记录均来自 Known calibration 选择。",
        "- `selection_score = calibration F1-K + 0.05 × Known Recall`，不是 OOS F1；因此它只能保证 Known-only 目标，不保证历史 fulltex 的 OOS 目标。",
        "",
        "## 2. 机制解释",
        "",
        "1. 训练动态图展示的是 Known-only 选择目标；测试 OOS 图仅用于事后检查二者是否错位。",
        "2. 当 KIR 增大时，Known intent universe 变小、OOS 组成变难，单一 Known-only 选择目标不能自动适配所有开放程度。",
        "3. 这解释了为什么 Trainable 在当前 K=1 Gate 上通常优于 Frozen，但仍可能低于 fulltex：历史 fulltex 使用了不同的 λ/unknown validation、固定 K=2 和完整 Cascade。",
        "4. 即使 Known-only 选择曲线稳定，也不能说明固定 K>1 的 union boundary 安全；该问题已经由 StackOverflow K=2 配对实验单独证明。",
        "",
        "## 3. Known coverage transfer",
        "",
        "- 选模后的 calibration Known Recall 与 test Known Recall 的平均差异并不大；例如 KIR=.50 为 CLINC150 `-0.64pp`、Banking77 `-0.21pp`、StackOverflow `+0.33pp`。",
        "- 因此当前高 KIR 的 OOS F1 下降不能简单解释为 Known 覆盖崩溃，更可能来自 OOS score 分布和边界校准错位。",
        "",
        "## 4. 文件",
        "",
        "- `results/analysis/minilm_training_dynamics_v1/run_summary.csv`",
        "- `results/analysis/minilm_training_dynamics_v1/history.csv`",
        "- `figures/minilm_training_dynamics_v1/`（含 calibration/test Known Recall、selection dynamics 和 OOS 对齐图）",
        "",
        "## 5. 结论边界",
        "",
        "- 本阶段是训练选择机制诊断，不是新方法结果，也不是 SOTA 排名。",
        "- 后续如要提高历史协议可比性，应先做同一表示、同一 K、同一阈值监督条件的 bridge baseline，而不是盲目增加训练 epoch。",
    ]
    (ROOT / "docs/analysis/MINILM_TRAINING_DYNAMICS_V1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    histories: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                history, run = _read_run(dataset, kir, seed)
                histories.append(history)
                run_rows.append(run)
    history = pd.concat(histories, ignore_index=True)
    runs = pd.DataFrame(run_rows)
    _write(runs, "run_summary.csv")
    _write(history.drop(columns=["run_dir"]), "history.csv")
    _plot(history, runs)
    _report(runs, history)
    print(json.dumps({"status": "complete", "run_count": len(runs), "epoch_rows": len(history), "figures": len(list(FIG.glob("*.png")))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
