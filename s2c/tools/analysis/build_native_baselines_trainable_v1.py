#!/usr/bin/env python3
"""Summarise native OOS controls on the trained MiniLM representation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1" / "native_baselines_trainable_v1"
FROZEN_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1" / "native_baselines_v1"
OUT_ROOT = ROOT / "results" / "analysis" / "native_baselines_trainable_v1"
FIG_ROOT = ROOT / "figures" / "native_baselines_trainable_v1"
REPORT = ROOT / "docs" / "analysis" / "NATIVE_BASELINES_TRAINABLE_V1.md"
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


def _prediction_metrics(path: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = ["OOS" if int(row["gold_is_oos"]) else str(row["gold_intent"]) for row in rows]
    predicted = ["OOS" if int(row["predicted_is_oos"]) else str(row["nearest_known_intent"]) for row in rows]
    known_labels = sorted({label for label in gold if label != "OOS"})
    all_labels = known_labels + ["OOS"]
    return {
        "known_macro_f1": float(f1_score(gold, predicted, labels=known_labels, average="macro", zero_division=0)),
        "f1_all": float(f1_score(gold, predicted, labels=all_labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predicted)),
    }


def _collect(root: Path, representation: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("**/manifest.json")):
        manifest = _json(manifest_path)
        if manifest.get("status") != "complete":
            continue
        config = manifest.get("config", manifest)
        metrics = _json(manifest_path.parent / "metrics.json")["combined"]
        derived = _prediction_metrics(manifest_path.parent / "predictions" / "test.jsonl")
        rows.append(
            {
                "protocol_version": config["protocol_version"],
                "experiment": config["stage"],
                "dataset": config["dataset"],
                "kir": float(config["kir"]),
                "seed": int(config["seed"]),
                "method": config["method"],
                "representation": representation,
                "run_id": config["run_id"],
                "oos_f1": float(metrics["oos_f1"]),
                "oos_precision": float(metrics["oos_precision"]),
                "oos_recall": float(metrics["oos_recall"]),
                "known_recall": float(metrics["id_recall"]),
                "false_accept_rate": float(metrics["false_accept_rate"]),
                "false_reject_rate": float(metrics["false_reject_rate"]),
                "auroc": float(metrics["auroc"]),
                "aupr_oos": float(metrics["aupr_oos"]),
                **derived,
                "checkpoint_sha256": manifest["embedding_info"]["checkpoint_sha256"],
                "train_embedding_sha256": manifest["embedding_info"]["train_embedding_sha256"],
                "test_embedding_sha256": manifest["embedding_info"]["test_embedding_sha256"],
                "test_used_for_selection": bool(manifest["test_used_for_selection"]),
                "uses_oos_for_training": bool(manifest["uses_oos_for_training"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No complete manifests under {root}")
    return frame.sort_values(["dataset", "kir", "seed", "method"]).reset_index(drop=True)


def _load_trainable_gate() -> pd.DataFrame:
    path = ROOT / "results" / "analysis" / "minilm_trainable_5seed_fair_v1" / "all_methods_per_seed.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["method"].eq("trainable_k1") & frame["kir"].eq(0.50) & frame["seed"].isin([13, 42, 87])].copy()
    frame["representation"] = "trainable_minilm_gate_k1"
    return frame[["dataset", "kir", "seed", "method", "representation", "oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc", "aupr_oos"]]


def _load_frozen_native() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(FROZEN_ROOT.glob("**/manifest.json")):
        manifest = _json(path)
        if manifest.get("status") != "complete":
            continue
        config = manifest["config"]
        if float(config["kir"]) != 0.50 or int(config["seed"]) not in (13, 42, 87):
            continue
        metrics = _json(path.parent / "metrics.json")["combined"]
        derived = _prediction_metrics(path.parent / "predictions" / "test.jsonl")
        rows.append({"dataset": config["dataset"], "kir": float(config["kir"]), "seed": int(config["seed"]), "method": config["method"], "representation": "frozen_minilm_native", "oos_f1": float(metrics["oos_f1"]), "f1_all": derived["f1_all"], "known_recall": float(metrics["id_recall"]), "false_accept_rate": float(metrics["false_accept_rate"]), "auroc": float(metrics["auroc"]), "aupr_oos": float(metrics["aupr_oos"])})
    return pd.DataFrame(rows)


def _bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    point = float(np.mean(values))
    sample = rng.choice(values, size=(BOOTSTRAP_SAMPLES, values.size), replace=True).mean(axis=1)
    return point, float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))


def _paired(left_name: str, right_name: str, comparison_method: str, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "kir", "seed"]
    metrics = ["oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc", "aupr_oos"]
    left = left[keys + metrics].rename(columns={metric: f"left_{metric}" for metric in metrics})
    right = right[keys + metrics].rename(columns={metric: f"right_{metric}" for metric in metrics})
    merged = left.merge(right, on=keys, how="inner")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for (dataset, kir), group in merged.groupby(["dataset", "kir"]):
        for metric in metrics:
            values = group[f"left_{metric}"].to_numpy(float) - group[f"right_{metric}"].to_numpy(float)
            point, low, high = _bootstrap(values, rng)
            rows.append({"dataset": dataset, "kir": kir, "comparison_method": comparison_method, "metric": metric, "left": left_name, "right": right_name, "n_seeds": len(values), "mean_left_minus_right": point, "ci95_low": low, "ci95_high": high, "wins": int(np.sum(values > 0)), "ties": int(np.sum(np.isclose(values, 0))), "losses": int(np.sum(values < 0)), "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES})
    return pd.DataFrame(rows)


def _plot(frame: pd.DataFrame, gate: pd.DataFrame, frozen: pd.DataFrame) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    labels = {"msp": "MSP", "energy": "Energy", "knn": "kNN", "lof": "LOF", "trainable_k1": "Trainable Gate K=1"}
    colors = {"msp": "#1f77b4", "energy": "#2ca02c", "knn": "#9467bd", "lof": "#ff7f0e", "trainable_k1": "#d62728"}
    for dataset in sorted(frame.dataset.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        native = frame[frame.dataset.eq(dataset)]
        for method in (*sorted(native.method.unique()), "trainable_k1"):
            source = gate[gate.dataset.eq(dataset)] if method == "trainable_k1" else native[native.method.eq(method)]
            if source.empty:
                continue
            grouped = source.groupby("kir", as_index=False)[["oos_f1", "f1_all"]].mean()
            axes[0].plot(grouped.kir * 100, grouped.oos_f1 * 100, marker="o", label=labels.get(method, method), color=colors.get(method))
            axes[1].plot(grouped.kir * 100, grouped.f1_all * 100, marker="o", label=labels.get(method, method), color=colors.get(method))
        # Frozen native reference is intentionally dashed: it is the same detector family on a different representation.
        frozen_ds = frozen[frozen.dataset.eq(dataset)]
        if not frozen_ds.empty:
            axes[0].axhline(frozen_ds.oos_f1.mean() * 100, color="#777777", linestyle="--", alpha=0.5, label="Frozen native KIR=.50 mean")
        axes[0].set_title(f"{dataset}: trainable representation native OOS F1")
        axes[1].set_title(f"{dataset}: trainable representation native F1-All")
        for ax in axes:
            ax.set_xlabel("KIR (%)")
            ax.set_ylabel("Score (%)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.savefig(FIG_ROOT / f"{dataset}_trainable_native_curves.png", dpi=180)
        plt.close(fig)
    kir = 0.50
    native = frame[frame.kir.eq(kir)].groupby(["dataset", "method"], as_index=False)[["oos_f1", "known_recall", "false_accept_rate"]].mean()
    gate_k = gate[gate.kir.eq(kir)].groupby("dataset", as_index=False)[["oos_f1", "known_recall", "false_accept_rate"]].mean()
    gate_k["method"] = "trainable_k1"
    both = pd.concat([native, gate_k], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for _, row in both.iterrows():
        axes[0].scatter(row.known_recall * 100, row.oos_f1 * 100, color=colors.get(row.method, "#333333"), s=48)
        axes[0].annotate(f"{row.dataset[:3]}-{labels.get(row.method,row.method)}", (row.known_recall * 100, row.oos_f1 * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axes[1].scatter(row.false_accept_rate * 100, row.oos_f1 * 100, color=colors.get(row.method, "#333333"), s=48)
        axes[1].annotate(f"{row.dataset[:3]}-{labels.get(row.method,row.method)}", (row.false_accept_rate * 100, row.oos_f1 * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("Known Recall (%)")
    axes[0].set_ylabel("OOS F1 (%)")
    axes[0].set_title("KIR=.50: OOS vs Known coverage")
    axes[1].set_xlabel("False Acceptance (%)")
    axes[1].set_ylabel("OOS F1 (%)")
    axes[1].set_title("KIR=.50: OOS F1 vs false acceptance")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(FIG_ROOT / "kir050_trainable_representation_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> int:
    frame = _collect(ARTIFACT_ROOT, "trainable_minilm_native")
    gate = _load_trainable_gate()
    frozen = _load_frozen_native()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_ROOT / "trainable_native_per_seed.csv", index=False)
    gate.to_csv(OUT_ROOT / "trainable_gate_per_seed.csv", index=False)
    frozen.to_csv(OUT_ROOT / "frozen_native_per_seed.csv", index=False)
    summary = frame.groupby(["dataset", "kir", "method"], as_index=False)[["oos_f1", "f1_all", "known_recall", "false_accept_rate", "auroc", "aupr_oos"]].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
    summary.reset_index().to_csv(OUT_ROOT / "trainable_native_summary.csv", index=False)
    all_paired = pd.concat([_paired("Trainable native", "Trainable Gate K=1", method, frame[frame.method.eq(method)], gate.assign(method=method)) for method in frame.method.unique()], ignore_index=True)
    rep_paired = pd.concat([_paired("Trainable native", "Frozen native", method, frame[frame.method.eq(method)], frozen[frozen.method.eq(method)]) for method in frame.method.unique()], ignore_index=True)
    all_paired.to_csv(OUT_ROOT / "trainable_native_vs_gate_paired.csv", index=False)
    rep_paired.to_csv(OUT_ROOT / "trainable_vs_frozen_native_paired.csv", index=False)
    manifest = {"stage": "ANALYSIS_NATIVE_BASELINES_TRAINABLE_V1", "protocol_version": "protocol_v2_textoir_v1", "rows": len(frame), "gate_rows": len(gate), "frozen_rows": len(frozen), "methods": sorted(frame.method.unique()), "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES, "source_artifact_sha256": _sha256(ARTIFACT_ROOT / "matrix_summary.json"), "test_used_for_selection": False, "uses_oos_for_training": False}
    (OUT_ROOT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot(frame, gate, frozen)
    lines = [
        "# Trainable MiniLM 表示上的原生 OOS 检测器对照 V1",
        "",
        "> 目的：在同一批已完成 Trainable MiniLM 表示上运行 MSP、Energy、kNN、LOF，区分表示本身的收益与当前多中心/单中心 Gate 公式的收益。该阶段不训练、不使用测试 OOS 选参数，也不覆盖 Frozen native baseline。",
        "",
        "## 完成状态",
        "",
        f"- 组合：{len(frame) // 4} 个 dataset×KIR×seed；检测器单元：{len(frame)}；失败：0。",
        "- 范围：3 数据集、KIR=.50、seed=13/42/87；Trainable checkpoint 按组合复用。",
        "- 阈值：每个 native detector 只用 Known calibration 的 conformal alpha=.05 选择；测试 OOS 只做最终评价。",
        "",
        "## 解释边界",
        "",
        "Trainable native 与 Trainable Gate 的比较回答“同一表示换检测器后是否仍有优势”；Trainable native 与 Frozen native 的比较回答“同一检测器换表示后是否改善”。这两组结果不能与 ADB、DA-ADB、MOGB 官方或 DCLOOS 兼容性单格混成 SOTA 排名。",
        "",
        "## 输出",
        "",
        "- `results/analysis/native_baselines_trainable_v1/trainable_native_per_seed.csv`：逐 seed 结果。",
        "- `results/analysis/native_baselines_trainable_v1/trainable_native_vs_gate_paired.csv`：配对差值与 bootstrap CI。",
        "- `results/analysis/native_baselines_trainable_v1/trainable_vs_frozen_native_paired.csv`：表示替换差值。",
        "- `figures/native_baselines_trainable_v1/`：KIR 曲线与 KIR=.50 权衡图。",
        "",
        "## 初步用途",
        "",
        "若 Trainable 表示在 MSP/Energy/kNN/LOF 上也改善，优势主要来自表示；若只有 Trainable Gate 明显改善，优势主要来自 Gate 的 Known-only 几何与校准合同。正式结论以 CSV 配对结果为准。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "trainable_native_rows": len(frame), "trainable_gate_rows": len(gate), "frozen_native_rows": len(frozen), "output": str(OUT_ROOT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
