"""Analyze threshold sensitivity and radius stability for existing K=1 runs.

All threshold curves are descriptive post-hoc diagnostics.  No threshold is
selected from test OOS and no existing run artifact is modified.
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
OUT = ROOT / "results/analysis/threshold_radius_stability_v1"
FIG = ROOT / "figures/threshold_radius_stability_v1"
TRAINABLE_ROOTS = (ARTIFACT_ROOT / "minilm_trainable_kir_sweep_v1", ARTIFACT_ROOT / "minilm_trainable_kir_sweep_extension_v1")
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
THRESHOLDS = (0.75, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trainable_dir(dataset: str, kir: float, seed: int) -> Path:
    for root in TRAINABLE_ROOTS:
        path = root / f"kir_{kir:.2f}" / "runs" / dataset / f"seed_{seed}"
        if (path / "predictions.jsonl").is_file():
            return path
    raise FileNotFoundError(f"Trainable run missing: {dataset} kir={kir} seed={seed}")


def _frozen_dir(dataset: str, kir: float, seed: int) -> Path:
    name = (
        f"protocol_v2_textoir_v1__{dataset}__kir_{kir:.2f}__seed_{seed}__"
        "repr_frozen_minilm__k_1__dist_mahalanobis_diag__boundary_mean_std"
    )
    path = ARTIFACT_ROOT / "e2_gate_core_dense" / name
    if not (path / "predictions/test.jsonl").is_file():
        raise FileNotFoundError(f"Frozen E2 run missing: {path}")
    return path


def _predictions(path: Path, dataset: str, kir: float, seed: int, representation: str) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append({
                "dataset": dataset,
                "kir": kir,
                "seed": seed,
                "representation": representation,
                "gold_is_oos": int(item["gold_is_oos"]),
                "score": float(item["oos_score"]),
                "radius": float(item.get("radius", np.nan)),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"Empty predictions: {path}")
    return frame


def _metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    # predicted_is_oos=1 means reject; gold_is_oos=1 means OOS.
    predicted_oos = frame.score > threshold
    gold_oos = frame.gold_is_oos.astype(bool)
    tp = int((predicted_oos & gold_oos).sum())
    fp = int((predicted_oos & ~gold_oos).sum())
    fn = int((~predicted_oos & gold_oos).sum())
    known_count = int((~gold_oos).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "oos_f1": f1,
        "oos_precision": precision,
        "oos_recall": recall,
        "known_recall": 1.0 - fp / max(known_count, 1),
        "false_accept_rate": fp / max(known_count, 1),
        "false_reject_rate": fn / max(int(gold_oos.sum()), 1),
    }


def _radius_row(path: Path, dataset: str, kir: float, seed: int, representation: str, predictions: pd.DataFrame) -> dict[str, Any]:
    detector = path / "detector_signature.json"
    all_radii: np.ndarray
    if detector.is_file():
        payload = _json(detector)
        all_radii = np.asarray([float(sphere["radius"]) for sphere in payload.get("spheres", [])], dtype=float)
    else:
        all_radii = predictions.radius.to_numpy(dtype=float) if "radius" in predictions else np.array([], dtype=float)
    all_radii = all_radii[np.isfinite(all_radii)]
    assigned = predictions["radius"].to_numpy(dtype=float) if "radius" in predictions else np.array([], dtype=float)
    assigned = assigned[np.isfinite(assigned)]
    return {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "representation": representation,
        "n_centers_or_unique_radii": int(len(all_radii) if detector.is_file() else len(np.unique(assigned))),
        "radius_mean": float(np.mean(all_radii)) if len(all_radii) else np.nan,
        "radius_std": float(np.std(all_radii)) if len(all_radii) else np.nan,
        "radius_cv": float(np.std(all_radii) / max(np.mean(all_radii), 1e-12)) if len(all_radii) else np.nan,
        "radius_min": float(np.min(all_radii)) if len(all_radii) else np.nan,
        "radius_max": float(np.max(all_radii)) if len(all_radii) else np.nan,
        "assigned_radius_mean": float(np.mean(assigned)) if len(assigned) else np.nan,
        "assigned_radius_std": float(np.std(assigned)) if len(assigned) else np.nan,
        "assigned_radius_cv": float(np.std(assigned) / max(np.mean(assigned), 1e-12)) if len(assigned) else np.nan,
    }


def _write(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    # ``seed`` is an experimental identifier, not a metric.  Never average it
    # into the grouped summary (which would otherwise produce misleading values
    # such as seed=73 for five seeds).
    excluded = set(keys) | {"seed"}
    numeric = [col for col in frame.columns if col not in excluded]
    means = frame.groupby(keys, as_index=False)[numeric].mean()
    std = frame.groupby(keys, as_index=False)[numeric].std().rename(columns={col: f"{col}_std" for col in numeric})
    return means.merge(std, on=keys, how="left")


def _plot(thresholds: pd.DataFrame, radii: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"frozen": "#4c78a8", "trainable": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = thresholds[(thresholds.dataset == dataset) & (thresholds.kir == 0.50)]
        for representation in ("frozen", "trainable"):
            part = subset[subset.representation == representation].groupby("threshold", as_index=False).oos_f1.mean()
            ax.plot(part.threshold, part.oos_f1 * 100, marker="o", color=colors[representation], label=representation)
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("threshold (diagnostic only)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Threshold sensitivity at KIR=0.50: Frozen vs Trainable K=1")
    fig.tight_layout()
    fig.savefig(FIG / "threshold_sensitivity_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary = radii[radii.kir == 0.50]
    for representation, color in colors.items():
        part = summary[summary.representation == representation].groupby("dataset", as_index=False).radius_cv.mean()
        axes[0].bar(np.arange(len(part)) + (-0.18 if representation == "frozen" else 0.18), part.radius_cv, width=0.35, color=color, label=representation)
        part2 = summary[summary.representation == representation].groupby("dataset", as_index=False).assigned_radius_cv.mean()
        axes[1].bar(np.arange(len(part2)) + (-0.18 if representation == "frozen" else 0.18), part2.assigned_radius_cv, width=0.35, color=color, label=representation)
    for ax, title in zip(axes, ("All fitted radii CV", "Assigned-test-radius CV")):
        ax.set_xticks(np.arange(len(DATASETS)))
        ax.set_xticklabels(DATASETS)
        ax.set_title(title)
        ax.set_ylabel("coefficient of variation")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("KIR=0.50 radius stability")
    fig.tight_layout()
    fig.savefig(FIG / "radius_stability_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Known Recall and OOS F1 at the diagnostic threshold grid.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    subset = thresholds[thresholds.kir == 0.50]
    for representation, color in colors.items():
        part = subset[subset.representation == representation].groupby("threshold", as_index=False)[["known_recall", "oos_f1"]].mean()
        axes[0].plot(part.threshold, part.known_recall * 100, marker="o", color=color, label=representation)
        axes[1].plot(part.threshold, part.oos_f1 * 100, marker="o", color=color, label=representation)
    for ax, title, ylabel in zip(axes, ("Known Recall", "OOS F1"), ("Known Recall (%)", "OOS F1 (%)")):
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("threshold")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Threshold trade-off at KIR=0.50 (diagnostic, not selected)")
    fig.tight_layout()
    fig.savefig(FIG / "threshold_tradeoff_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report(threshold_mean: pd.DataFrame, radius_mean: pd.DataFrame) -> None:
    central = threshold_mean[threshold_mean.kir == 0.50]
    lines = [
        "# Threshold 与半径稳定性诊断 V1",
        "",
        "> 本报告只做已完成 K=1 预测的事后诊断。阈值曲线使用测试 score 计算，但不用于选择正式阈值、不修改任何 run。",
        "",
        "## 1. 主要发现",
        "",
        "1. Trainable 与 Frozen 的 score 标度不同，因此固定 threshold=1 对两种表示并非完全等价的工作点。",
        "2. 这可以解释部分 Trainable 与历史 fulltex 的差距，但不能解释 StackOverflow 固定 K=2 的 union false acceptance；后者是独立的多中心问题。",
        "3. 半径 CV 只描述估计稳定性；稳定半径不等于 OOS 方向正确，必须和 Known Recall、false acceptance 一起看。",
        "",
        "## 2. KIR=0.50 诊断性工作点",
        "",
        "| 数据集 | 表示 | threshold=1 OOS F1 | threshold 网格最佳 OOS F1（oracle diagnostic） | 对应 threshold |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for representation in ("frozen", "trainable"):
            part = central[(central.dataset == dataset) & (central.representation == representation)]
            row = part.loc[part.oos_f1.idxmax()]
            at_one = part[part.threshold == 1.0].iloc[0]
            lines.append(f"| {dataset} | {representation} | {at_one.oos_f1*100:.2f} | {row.oos_f1*100:.2f} | {row.threshold:.2f} |")
    lines.extend(
        [
            "",
            "## 3. 证据文件",
            "",
            "- `results/analysis/threshold_radius_stability_v1/threshold_sensitivity_mean_std.csv`",
            "- `results/analysis/threshold_radius_stability_v1/radius_stability_mean_std.csv`",
            "- `figures/threshold_radius_stability_v1/`",
            "",
            "## 4. 结论边界",
            "",
            "- 网格最佳 threshold 只作为 score 标度敏感性诊断，不能写成正式调参结果；正式协议仍是 threshold=1。",
            "- 若要提高当前 Trainable 与历史结果的可比性，下一步应预注册 Known-only 的 threshold/半径校准规则，再在新的独立验证池上运行，而不是读取 test oracle。",
        ]
    )
    (ROOT / "docs/analysis/THRESHOLD_RADIUS_STABILITY_V1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    threshold_rows: list[dict[str, Any]] = []
    radius_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                train_path = _trainable_dir(dataset, kir, seed)
                frozen_path = _frozen_dir(dataset, kir, seed)
                train_pred = _predictions(train_path / "predictions.jsonl", dataset, kir, seed, "trainable")
                frozen_pred = _predictions(frozen_path / "predictions/test.jsonl", dataset, kir, seed, "frozen")
                radius_rows.append(_radius_row(train_path, dataset, kir, seed, "trainable", train_pred))
                radius_rows.append(_radius_row(frozen_path, dataset, kir, seed, "frozen", frozen_pred))
                for representation, pred in (("trainable", train_pred), ("frozen", frozen_pred)):
                    for threshold in THRESHOLDS:
                        threshold_rows.append({"dataset": dataset, "kir": kir, "seed": seed, "representation": representation, "threshold": threshold, **_metrics(pred, threshold)})
    thresholds = pd.DataFrame(threshold_rows)
    radii = pd.DataFrame(radius_rows)
    threshold_mean = _aggregate(thresholds, ["dataset", "kir", "representation", "threshold"])
    radius_mean = _aggregate(radii, ["dataset", "kir", "representation"])
    _write(thresholds, "threshold_sensitivity_per_seed.csv")
    _write(threshold_mean, "threshold_sensitivity_mean_std.csv")
    _write(radii, "radius_stability_per_seed.csv")
    _write(radius_mean, "radius_stability_mean_std.csv")
    _plot(thresholds, radii)
    _report(threshold_mean, radius_mean)
    print(json.dumps({"status": "complete", "threshold_rows": len(thresholds), "radius_rows": len(radii), "figures": len(list(FIG.glob("*.png")))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
