"""Analyze score, radius and error geometry for existing Frozen/Trainable K=1 runs.

This is an analysis-only stage.  It never selects a checkpoint, radius, threshold
or K from test outcomes and does not write into any experiment artifact root.
Only aggregate numeric diagnostics are exported; sample identifiers are retained
solely in memory while reading prediction files.
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
OUT = ROOT / "results/analysis/minilm_boundary_diagnostics_v1"
FIG = ROOT / "figures/minilm_boundary_diagnostics_v1"
TRAINABLE_ROOTS = (
    ARTIFACT_ROOT / "minilm_trainable_kir_sweep_v1",
    ARTIFACT_ROOT / "minilm_trainable_kir_sweep_extension_v1",
)
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any], name: str) -> float:
    combined = payload.get("combined", payload)
    if name == "known_recall":
        name = "id_recall"
    return float(combined[name])


def _read_predictions(path: Path, dataset: str, kir: float, seed: int, representation: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "seed": seed,
                    "representation": representation,
                    "sample_id": str(item.get("sample_id", "")),
                    "gold_is_oos": int(item.get("gold_is_oos", 0)),
                    "predicted_is_oos": int(item.get("predicted_is_oos", 1)),
                    "score": float(item["oos_score"]),
                    "distance": float(item.get("distance", np.nan)),
                    "radius": float(item.get("radius", np.nan)),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"Prediction file is empty: {path}")
    frame["accepted_known"] = frame["predicted_is_oos"] == 0
    frame["score_margin"] = 1.0 - frame["score"]
    return frame


def _trainable_dir(dataset: str, kir: float, seed: int) -> Path:
    for root in TRAINABLE_ROOTS:
        candidate = root / f"kir_{kir:.2f}" / "runs" / dataset / f"seed_{seed}"
        if (candidate / "predictions.jsonl").is_file():
            return candidate
    raise FileNotFoundError(f"Trainable run not found: {dataset} kir={kir} seed={seed}")


def _frozen_dir(dataset: str, kir: float, seed: int) -> Path:
    name = (
        f"protocol_v2_textoir_v1__{dataset}__kir_{kir:.2f}__seed_{seed}__"
        "repr_frozen_minilm__k_1__dist_mahalanobis_diag__boundary_mean_std"
    )
    path = ARTIFACT_ROOT / "e2_gate_core_dense" / name
    if not (path / "predictions/test.jsonl").is_file():
        raise FileNotFoundError(f"Frozen E2 run not found: {path}")
    return path


def _run_summary(run_dir: Path, representation: str, dataset: str, kir: float, seed: int) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    payload = _json(metrics_path)
    prediction_path = run_dir / "predictions.jsonl" if representation == "trainable" else run_dir / "predictions/test.jsonl"
    predictions = _read_predictions(prediction_path, dataset, kir, seed, representation)
    summary: dict[str, Any] = {
        "dataset": dataset,
        "kir": kir,
        "seed": seed,
        "representation": representation,
        "oos_f1": _metric(payload, "oos_f1"),
        "known_recall": _metric(payload, "known_recall"),
        "false_accept_rate": _metric(payload, "false_accept_rate"),
        "false_reject_rate": _metric(payload, "false_reject_rate"),
        "auroc": _metric(payload, "auroc"),
        "aupr_oos": _metric(payload, "aupr_oos"),
        "f1_k": float(payload.get("f1_k", np.nan)) if representation == "trainable" else np.nan,
        "f1_all": float(payload.get("f1_all", np.nan)) if representation == "trainable" else np.nan,
    }
    for group_name, group in (("known", predictions[predictions.gold_is_oos == 0]), ("oos", predictions[predictions.gold_is_oos == 1])):
        if group.empty:
            continue
        scores = group.score.to_numpy(dtype=float)
        radii = group.radius.to_numpy(dtype=float)
        summary.update(
            {
                f"{group_name}_score_mean": float(np.mean(scores)),
                f"{group_name}_score_median": float(np.median(scores)),
                f"{group_name}_score_p90": float(np.quantile(scores, 0.90)),
                f"{group_name}_score_p95": float(np.quantile(scores, 0.95)),
                f"{group_name}_score_p99": float(np.quantile(scores, 0.99)),
                f"{group_name}_radius_mean": float(np.nanmean(radii)),
                f"{group_name}_radius_std": float(np.nanstd(radii)),
                f"{group_name}_radius_cv": float(np.nanstd(radii) / max(np.nanmean(radii), 1e-12)),
                f"{group_name}_accepted_rate": float(group.accepted_known.mean()),
                f"{group_name}_score_leq_1_rate": float((group.score <= 1.0).mean()),
            }
        )
    summary["score_gap_median_oos_minus_known"] = summary["oos_score_median"] - summary["known_score_median"]
    summary["radius_gap_oos_minus_known"] = summary["oos_radius_mean"] - summary["known_radius_mean"]
    return summary


def _collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                trainable = _trainable_dir(dataset, kir, seed)
                frozen = _frozen_dir(dataset, kir, seed)
                summaries.append(_run_summary(trainable, "trainable", dataset, kir, seed))
                summaries.append(_run_summary(frozen, "frozen", dataset, kir, seed))
                score_frames.append(_read_predictions(trainable / "predictions.jsonl", dataset, kir, seed, "trainable"))
                score_frames.append(_read_predictions(frozen / "predictions/test.jsonl", dataset, kir, seed, "frozen"))
    return pd.DataFrame(summaries), pd.concat(score_frames, ignore_index=True)


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in summary.columns if column not in {"dataset", "kir", "representation", "seed"}]
    grouped = summary.groupby(["dataset", "kir", "representation"], as_index=False)
    means = grouped[numeric].mean().add_suffix("_mean")
    means = means.rename(columns={"dataset_mean": "dataset", "kir_mean": "kir", "representation_mean": "representation"})
    std = grouped[numeric].std().add_suffix("_std")
    std = std.rename(columns={"dataset_std": "dataset", "kir_std": "kir", "representation_std": "representation"})
    return means.merge(std, on=["dataset", "kir", "representation"], how="left")


def _write(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def _plot(summary: pd.DataFrame, scores: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"frozen": "#4c78a8", "trainable": "#d62728"}
    # Test score distributions at the central KIR.  This is diagnostic only;
    # the threshold was fixed before these plots were generated.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = scores[(scores.dataset == dataset) & (scores.kir == 0.50)]
        for representation in ("frozen", "trainable"):
            for is_oos, label in ((0, "Known"), (1, "OOS")):
                values = subset[(subset.representation == representation) & (subset.gold_is_oos == is_oos)].score
                ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.8, color=colors[representation], linestyle="-" if is_oos == 0 else "--", label=f"{representation} {label}")
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("normalized OOS score")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("density")
    axes[-1].legend(fontsize=7, frameon=False)
    fig.suptitle("KIR=0.50 test score distributions: Frozen vs Trainable K=1")
    fig.tight_layout()
    fig.savefig(FIG / "score_distributions_kir050.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Median score gap and selected-radius gap by KIR.
    med = summary.pivot_table(index=["dataset", "kir"], columns="representation", values="score_gap_median_oos_minus_known")
    med["delta_trainable_minus_frozen"] = med["trainable"] - med["frozen"]
    radius = summary.pivot_table(index=["dataset", "kir"], columns="representation", values="radius_gap_oos_minus_known")
    radius["delta_trainable_minus_frozen"] = radius["trainable"] - radius["frozen"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, table, title, ylabel in ((axes[0], med, "Median OOS-known score gap", "score gap"), (axes[1], radius, "Assigned-radius OOS-known gap", "radius gap")):
        values = table["delta_trainable_minus_frozen"].unstack("dataset")
        values.plot(kind="bar", ax=ax, width=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("KIR")
        ax.set_ylabel(ylabel)
        ax.legend(title="dataset", frameon=False)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Trainable minus Frozen change in diagnostic geometry")
    fig.tight_layout()
    fig.savefig(FIG / "geometry_gap_trainable_minus_frozen.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # OOS F1 versus false acceptance, with each point a dataset/KIR mean.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = summary[summary.dataset == dataset]
        for representation in ("frozen", "trainable"):
            part = subset[subset.representation == representation]
            ax.plot(part.false_accept_rate * 100, part.oos_f1 * 100, marker="o", label=representation, color=colors[representation])
            for _, row in part.iterrows():
                ax.annotate(f"{row.kir:.2f}", (row.false_accept_rate * 100, row.oos_f1 * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(dataset)
        ax.grid(alpha=0.25)
        ax.set_xlabel("false acceptance (%)")
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(frameon=False)
    fig.suptitle("KIR trade-off: OOS F1 versus false acceptance")
    fig.tight_layout()
    fig.savefig(FIG / "oos_f1_false_acceptance_kir.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report(summary: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    kir50 = summary[summary["kir"] == 0.50].groupby(["dataset", "representation"], as_index=False).mean(numeric_only=True)
    lines = [
        "# MiniLM 表示—边界诊断 V1",
        "",
        "> 本报告只读取已经完成的 Frozen E2 K=1 与 Trainable K=1 预测和 metrics，做 test-score 机制诊断；不重新训练、不选择阈值、不选择 checkpoint，也不把诊断结果当作调参依据。",
        "",
        "## 1. 范围与数据来源",
        "",
        "- 三个数据集、KIR={0.25,0.50,0.75}、seeds={13,42,87,100,123}、K=1、diagonal Mahalanobis、mean+std、threshold=1。",
        "- Trainable 读取 `minilm_trainable_kir_sweep_v1` 与其五 seed 扩展；Frozen 读取相同 E2 单元。",
        "- 逐样本 score 只在内存中用于分组统计，导出的 CSV 不包含文本；test OOS 仅用于事后解释。",
        "",
        "## 2. 诊断指标",
        "",
        "- `score_gap_median_oos_minus_known`：OOS 与 Known 的中位 normalized score 差；越大通常表示排序分离更强。",
        "- `radius_gap_oos_minus_known`：被最近中心选中的半径差；只描述半径与样本分配关系，不表示训练时使用了 OOS。",
        "- false acceptance/rejection、OOS F1 和 Known Recall 直接来自各 run 的正式 metrics。",
        "",
        "## 3. 当前可见的机制解释",
        "",
        "1. Trainable K=1 的收益首先体现在 score 排序：它改变了 Known/OOS score 分布，而不是恢复多中心边界。",
        "2. 当前训练目标只优化 Known 分类、类内紧致和类间 margin；它没有直接约束 OOS 误接受，因此不能保证 K>1 的并集安全。",
        "3. checkpoint 只按 Known calibration 的 `F1-K + 0.05×Known Recall` 选择，历史 fulltex 的 λ 则使用了不同的 OOS/unknown validation 合同。",
        "4. KIR 增大后，Known intent 覆盖减少而 OOS 组成变难；表示适配收益因数据集而异，不能用一个全局 epoch 或阈值解释。",
        "",
        "## 4. KIR=0.50 的数值证据",
        "",
    ]
    for dataset in DATASETS:
        frozen = kir50[(kir50.dataset == dataset) & (kir50.representation == "frozen")].iloc[0]
        trainable = kir50[(kir50.dataset == dataset) & (kir50.representation == "trainable")].iloc[0]
        lines.append(
            f"- {dataset}：median score gap（OOS−Known）由 Frozen "
            f"{frozen.score_gap_median_oos_minus_known:.3f} 变为 Trainable "
            f"{trainable.score_gap_median_oos_minus_known:.3f}；OOS median score "
            f"{frozen.oos_score_median:.3f}→{trainable.oos_score_median:.3f}；"
            f"false acceptance {frozen.false_accept_rate * 100:.2f}%→{trainable.false_accept_rate * 100:.2f}%。"
        )
    lines.extend(
        [
        "",
        "## 4. 证据文件",
        "",
        "- `results/analysis/minilm_boundary_diagnostics_v1/run_summary.csv`：45×2 个逐 seed 诊断摘要。",
        "- `results/analysis/minilm_boundary_diagnostics_v1/summary_mean_std.csv`：dataset×KIR×representation 汇总。",
        "- `results/analysis/minilm_boundary_diagnostics_v1/score_quantiles.csv`：Known/OOS score 分位数。",
        "- `figures/minilm_boundary_diagnostics_v1/`：score 分布、几何差值和 OOS F1—false acceptance 图。",
        "",
        "## 5. 当前结论边界",
        "",
        "- 这些图解释为什么 Trainable 能改善当前 K=1，但不能证明它已经达到 fulltex 历史结果或 SOTA。",
        "- 这些图也不能把 K=1 的表示收益外推为固定 K=2/多中心收益；StackOverflow 的多球 false acceptance 仍需单独处理。",
        "- 下一步应优先做 calibration coverage、半径稳定性和历史 fulltex/当前协议的逐组件桥接，而不是继续盲目扩展 K。",
        ]
    )
    (ROOT / "docs/analysis/MINILM_BOUNDARY_DIAGNOSTICS_V1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    summary, scores = _collect()
    agg = _aggregate(summary)
    _write(summary, "run_summary.csv")
    _write(agg, "summary_mean_std.csv")
    quantile_rows: list[dict[str, Any]] = []
    for keys, group in scores.groupby(["dataset", "kir", "seed", "representation", "gold_is_oos"]):
        values = group["score"].to_numpy(dtype=float)
        quantile_rows.append(
            {
                "dataset": keys[0],
                "kir": keys[1],
                "seed": keys[2],
                "representation": keys[3],
                "gold_is_oos": keys[4],
                "n": len(values),
                "score_mean": float(np.mean(values)),
                "score_p10": float(np.quantile(values, 0.10)),
                "score_p50": float(np.quantile(values, 0.50)),
                "score_p90": float(np.quantile(values, 0.90)),
                "score_p95": float(np.quantile(values, 0.95)),
                "score_p99": float(np.quantile(values, 0.99)),
                "score_leq_1_rate": float(np.mean(values <= 1.0)),
            }
        )
    _write(pd.DataFrame(quantile_rows), "score_quantiles.csv")
    _plot(summary, scores)
    _report(summary, agg)
    print(json.dumps({"status": "complete", "summary_rows": len(summary), "score_rows": len(scores), "output": str(OUT), "figures": len(list(FIG.glob("*.png")))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
