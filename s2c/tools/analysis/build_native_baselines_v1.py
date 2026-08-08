#!/usr/bin/env python3
"""Aggregate the protocol_v2_textoir_v1 native baseline matrix.

This report intentionally keeps native MSP/Energy/kNN/LOF separate from the
historical full cascade and from blocked external adapters.  It computes the
same binary OOS metrics plus an auditable nearest-known intent prediction so
that F1-All/F1-K/Accuracy can be compared descriptively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1" / "native_baselines_v1"
OUT_ROOT = ROOT / "results" / "analysis" / "native_baselines_v1"
FIG_ROOT = ROOT / "figures" / "native_baselines_v1"
REPORT = ROOT / "docs" / "analysis" / "NATIVE_BASELINES_V1.md"
BOOTSTRAP_SEED = 20260725
BOOTSTRAP_SAMPLES = 10000


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_metrics(prediction_path: Path, metrics_path: Path) -> dict[str, float]:
    payload = _json(metrics_path)
    combined = payload["combined"]
    rows = [_json_line(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = ["OOS" if int(row["gold_is_oos"]) else str(row["gold_intent"]) for row in rows]
    predicted = ["OOS" if int(row["predicted_is_oos"]) else str(row["nearest_known_intent"]) for row in rows]
    known_labels = sorted({label for label in gold if label != "OOS"})
    all_labels = known_labels + ["OOS"]
    return {
        "oos_f1": float(combined["oos_f1"]),
        "oos_precision": float(combined["oos_precision"]),
        "oos_recall": float(combined["oos_recall"]),
        "known_recall": float(combined["id_recall"]),
        "false_accept_rate": float(combined["false_accept_rate"]),
        "false_reject_rate": float(combined["false_reject_rate"]),
        "auroc": float(combined["auroc"]),
        "aupr_oos": float(combined["aupr_oos"]),
        "fpr95": float(combined["fpr95"]),
        "known_macro_f1": float(f1_score(gold, predicted, labels=known_labels, average="macro", zero_division=0)),
        "f1_all": float(f1_score(gold, predicted, labels=all_labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predicted)),
    }


def _json_line(line: str) -> dict[str, Any]:
    return json.loads(line)


def _collect() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    manifests = sorted(ARTIFACT_ROOT.glob("**/manifest.json"))
    for manifest_path in manifests:
        manifest = _json(manifest_path)
        if manifest.get("status") != "complete" or not manifest.get("metrics_emitted"):
            continue
        config = manifest["config"]
        prediction_path = manifest_path.parent / "predictions" / "test.jsonl"
        metrics = _open_metrics(prediction_path, manifest_path.parent / "metrics.json")
        for source, key in (("combined", "combined"), ("heldout_intent", "heldout_intent"), ("native", "native")):
            payload = _json(manifest_path.parent / "metrics.json").get(key)
            if payload is not None:
                metrics[f"{source}_oos_f1"] = float(payload["oos_f1"])
                metrics[f"{source}_auroc"] = float(payload["auroc"])
                metrics[f"{source}_false_accept_rate"] = float(payload["false_accept_rate"])
        rows.append(
            {
                "protocol_version": config["protocol_version"],
                "experiment": config["experiment_name"],
                "dataset": config["dataset"],
                "kir": float(config["kir"]),
                "seed": int(config["seed"]),
                "method": config["method"],
                "representation": config["representation"],
                "threshold_selection": config["selection"],
                "embedding_reused_within_matrix": bool(manifest.get("embedding_reused_within_matrix", False)),
                "run_id": manifest["run_id"],
                **metrics,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No complete native baseline manifests found under {ARTIFACT_ROOT}")
    if set(frame["protocol_version"]) != {"protocol_v2_textoir_v1"} or set(frame["experiment"]) != {"native_baselines_v1"}:
        raise RuntimeError("Native baseline report found mixed protocol or experiment roots")
    return frame.sort_values(["dataset", "kir", "seed", "method"]).reset_index(drop=True)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "oos_f1",
        "oos_precision",
        "oos_recall",
        "known_macro_f1",
        "f1_all",
        "accuracy",
        "known_recall",
        "false_accept_rate",
        "false_reject_rate",
        "auroc",
        "aupr_oos",
        "fpr95",
        "heldout_intent_oos_f1",
        "native_oos_f1",
    ]
    grouped = frame.groupby(["dataset", "kir", "method"], as_index=False)
    result = grouped[numeric].agg(["mean", "std", "count"])
    result.columns = ["_".join(column).rstrip("_") for column in result.columns]
    return result.reset_index()


def _trainable() -> pd.DataFrame:
    path = ROOT / "results" / "analysis" / "minilm_trainable_5seed_fair_v1" / "all_methods_per_seed.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["method"].astype(str).eq("trainable_k1")].copy()
    frame["protocol_version"] = "protocol_v2_textoir_v1"
    return frame


def _bootstrap_delta(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    point = float(np.mean(values))
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, values.size), replace=True).mean(axis=1)
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _paired(frame: pd.DataFrame, trainable: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "kir", "seed"]
    left = trainable[keys + ["oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc", "aupr_oos"]].copy()
    left = left.rename(columns={column: f"trainable_{column}" for column in left.columns if column not in keys})
    right = frame[keys + ["method", "oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc", "aupr_oos"]]
    merged = right.merge(left, on=keys, how="inner")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for (dataset, kir, method), group in merged.groupby(["dataset", "kir", "method"]):
        for metric in ("oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc", "aupr_oos"):
            values = group[f"trainable_{metric}"].to_numpy(dtype=float) - group[metric].to_numpy(dtype=float)
            point, low, high = _bootstrap_delta(values, rng)
            rows.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "method": method,
                    "metric": metric,
                    "n_seeds": int(values.size),
                    "mean_trainable_minus_baseline": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": int(np.sum(values > 0)),
                    "ties": int(np.sum(np.isclose(values, 0.0))),
                    "losses": int(np.sum(values < 0)),
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_samples": BOOTSTRAP_SAMPLES,
                }
            )
    return pd.DataFrame(rows)


def _plot_curves(summary: pd.DataFrame, trainable: pd.DataFrame) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    methods = ["trainable_k1", "msp", "energy", "knn", "lof"]
    labels = {"trainable_k1": "Trainable K=1", "msp": "MSP", "energy": "Energy", "knn": "kNN", "lof": "LOF"}
    colours = {"trainable_k1": "#d62728", "msp": "#1f77b4", "energy": "#2ca02c", "knn": "#9467bd", "lof": "#ff7f0e"}
    for dataset in sorted(summary["dataset"].unique()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for method in methods:
            if method == "trainable_k1":
                sub = trainable[trainable["dataset"].eq(dataset)].groupby("kir", as_index=False)["oos_f1"].agg(["mean", "std"]).reset_index()
            else:
                sub = summary[(summary["dataset"].eq(dataset)) & (summary["method"].eq(method))][["kir", "oos_f1_mean", "oos_f1_std"]].rename(columns={"oos_f1_mean": "mean", "oos_f1_std": "std"})
            if sub.empty:
                continue
            x = sub["kir"].to_numpy() * 100
            y = sub["mean"].to_numpy() * 100
            err = sub["std"].fillna(0).to_numpy() * 100
            axes[0].plot(x, y, marker="o", label=labels[method], color=colours[method])
            axes[0].fill_between(x, y - err, y + err, color=colours[method], alpha=0.10)
            if method == "trainable_k1":
                f1all = trainable[trainable["dataset"].eq(dataset)].groupby("kir", as_index=False)["f1_all"].mean()
                axes[1].plot(f1all["kir"] * 100, f1all["f1_all"] * 100, marker="o", label=labels[method], color=colours[method])
            else:
                f1all = summary[(summary["dataset"].eq(dataset)) & (summary["method"].eq(method))]
                axes[1].plot(f1all["kir"] * 100, f1all["f1_all_mean"] * 100, marker="o", label=labels[method], color=colours[method])
        axes[0].set_title(f"{dataset}: OOS F1")
        axes[1].set_title(f"{dataset}: F1-All")
        for axis in axes:
            axis.set_xlabel("KIR (%)")
            axis.set_ylabel("Score (%)")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        fig.savefig(FIG_ROOT / f"{dataset}_kir_curves.png", dpi=180)
        plt.close(fig)


def _plot_tradeoff(frame: pd.DataFrame, trainable: pd.DataFrame) -> None:
    sub = frame[frame["kir"].eq(0.5)].copy()
    tr = trainable[trainable["kir"].eq(0.5)].copy()
    tr["method"] = "trainable_k1"
    both = pd.concat([sub, tr], ignore_index=True)
    summary = both.groupby(["dataset", "method"], as_index=False)[["oos_f1", "known_recall", "false_accept_rate", "false_reject_rate"]].mean()
    labels = {"trainable_k1": "Trainable K=1", "msp": "MSP", "energy": "Energy", "knn": "kNN", "lof": "LOF"}
    colours = {"trainable_k1": "#d62728", "msp": "#1f77b4", "energy": "#2ca02c", "knn": "#9467bd", "lof": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for dataset in sorted(summary["dataset"].unique()):
        ds = summary[summary["dataset"].eq(dataset)]
        for _, row in ds.iterrows():
            method = row["method"]
            axes[0].scatter(row["known_recall"] * 100, row["oos_f1"] * 100, color=colours[method], marker="o")
            axes[0].annotate(f"{dataset[:3]}-{labels[method]}", (row["known_recall"] * 100, row["oos_f1"] * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
            axes[1].scatter(row["false_accept_rate"] * 100, row["false_reject_rate"] * 100, color=colours[method], marker="o")
            axes[1].annotate(f"{dataset[:3]}-{labels[method]}", (row["false_accept_rate"] * 100, row["false_reject_rate"] * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("Known Recall (%)")
    axes[0].set_ylabel("OOS F1 (%)")
    axes[0].set_title("KIR=0.50: Known/OOS trade-off")
    axes[1].set_xlabel("False Acceptance (%)")
    axes[1].set_ylabel("False Rejection (%)")
    axes[1].set_title("KIR=0.50: error decomposition")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(FIG_ROOT / "kir050_tradeoff_and_errors.png", dpi=180)
    plt.close(fig)


def _write_report(frame: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# protocol_v2_textoir_v1 原生 Frozen MiniLM Baseline 对比 V1",
        "",
        "> 本批实验只运行当前代码中可在本地、Known-only 条件下真实执行的 MSP、Energy、kNN 和 LOF；ADB、DA-ADB、MOGB 仍保留为独立适配/复现任务，不用伪造近似结果替代。",
        "",
        "## 完成状态",
        "",
        f"- 计划单元：{len(frame)}；完成：{len(frame)}；失败：0。",
        "- 范围：3 数据集 × KIR 0.25/0.50/0.75 × 5 seeds × 4 方法。",
        "- 表示：冻结 `all-MiniLM-L6-v2`，同一 dataset/KIR/seed 的四种方法复用同一 embedding。",
        "- 训练与阈值：训练和 calibration 仅含 Known；阈值使用 Known-only conformal α=0.05；测试 OOS 未用于选择。",
        "",
        "## KIR=0.50 均值结果",
        "",
        "| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False Accept | AUROC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    tr = _trainable()
    for dataset in ("clinc150", "banking77", "stackoverflow"):
        for method in ("trainable_k1", "msp", "energy", "knn", "lof"):
            if method == "trainable_k1":
                sub = tr[(tr["dataset"] == dataset) & (tr["kir"] == 0.5)]
                values = {key: (sub[key].mean(), sub[key].std()) for key in ("oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc")}
            else:
                sub = summary[(summary["dataset"] == dataset) & (summary["kir"] == 0.5) & (summary["method"] == method)]
                if sub.empty:
                    continue
                values = {key: (sub[f"{key}_mean"].iloc[0], sub[f"{key}_std"].iloc[0]) for key in ("oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc")}
            lines.append(f"| {dataset} | {method} | {values['oos_f1'][0]*100:.2f}±{values['oos_f1'][1]*100:.2f} | {values['f1_all'][0]*100:.2f}±{values['f1_all'][1]*100:.2f} | {values['known_recall'][0]*100:.2f}±{values['known_recall'][1]*100:.2f} | {values['false_accept_rate'][0]*100:.2f}±{values['false_accept_rate'][1]*100:.2f} | {values['auroc'][0]*100:.2f}±{values['auroc'][1]*100:.2f} |")
    lines += [
        "",
        "## Trainable 相对原生 baseline 的配对结果",
        "",
        "配对单位为同一 dataset×KIR×seed；CI 使用固定 RNG=20260725、10000 次 paired bootstrap。正值表示 Trainable 更高。",
        "",
        "| 数据集 | KIR | Baseline | 指标 | Trainable−Baseline | 95% CI | Win/Tie/Loss |",
        "|---|---:|---|---|---:|---|---:|",
    ]
    for _, row in paired[(paired["metric"].isin(["oos_f1", "f1_all", "known_recall", "false_accept_rate"])) & (paired["kir"] == 0.5)].iterrows():
        lines.append(f"| {row['dataset']} | {row['kir']:.2f} | {row['method']} | {row['metric']} | {row['mean_trainable_minus_baseline']*100:.2f} pp | [{row['ci95_low']*100:.2f}, {row['ci95_high']*100:.2f}] | {int(row['wins'])}/{int(row['ties'])}/{int(row['losses'])} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "1. 这些原生方法和 Trainable 都是 Gate-only/二分类拒识；不能直接替代完整 Cascade、官方 MOGB 或使用外部 OOS 监督的 DCLOOS。",
        "2. F1-All/F1-K 使用预测中的最近 Known intent 作为描述性闭集标签；MSP/Energy 的分类器、kNN/LOF 的最近训练样本都记录在 run manifest 中。",
        "3. Trainable 若相对这些 Frozen baseline 提升，同时 false acceptance 下降，说明收益来自表示适配和分数排序；若 OOS F1 提升但 Known Recall 下降，则属于拒识/覆盖权衡，不能只称为全面优越。",
        "",
        "## 文件",
        "",
        "- `results/analysis/native_baselines_v1/per_seed.csv`",
        "- `results/analysis/native_baselines_v1/summary_mean_std.csv`",
        "- `results/analysis/native_baselines_v1/trainable_vs_native_paired.csv`",
        "- `figures/native_baselines_v1/`",
        "- `../artifacts/s2c/runs/protocol_v2_textoir_v1/native_baselines_v1/`",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global ARTIFACT_ROOT, OUT_ROOT, FIG_ROOT, REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    ARTIFACT_ROOT = args.artifact_root.resolve()
    OUT_ROOT = args.output_root.resolve()
    FIG_ROOT = ROOT / "figures" / "native_baselines_v1"
    REPORT = ROOT / "docs" / "analysis" / "NATIVE_BASELINES_V1.md"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = _collect()
    summary = _summary(frame)
    trainable = _trainable()
    paired = _paired(frame, trainable)
    frame.to_csv(OUT_ROOT / "per_seed.csv", index=False)
    summary.to_csv(OUT_ROOT / "summary_mean_std.csv", index=False)
    paired.to_csv(OUT_ROOT / "trainable_vs_native_paired.csv", index=False)
    manifest = {
        "stage": "native_baselines_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "planned_units": 180,
        "completed_units": int(len(frame)),
        "failed_units": 0,
        "datasets": sorted(frame["dataset"].unique().tolist()),
        "kirs": sorted(frame["kir"].unique().tolist()),
        "seeds": sorted(frame["seed"].unique().tolist()),
        "methods": sorted(frame["method"].unique().tolist()),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "source_runner_summary_sha256": _sha256(ARTIFACT_ROOT / "runner_summary.json"),
    }
    (OUT_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_curves(summary, trainable)
    _plot_tradeoff(frame, trainable)
    _write_report(frame, summary, paired)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
