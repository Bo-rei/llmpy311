#!/usr/bin/env python3
"""Build cross-dataset, sample-aligned Trainable-K1 vs ADB error budgets.

This is an aggregate-only, post-hoc analysis of already completed artifacts.
It verifies the exact protocol test text order for each dataset/KIR/seed before
counting five mutually exclusive states.  No test result is used for tuning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
ARTIFACT_ROOT = Path("../artifacts/s2c")
FAIR_SOURCE = Path("results/analysis/cross_protocol_tradeoff_v1/per_seed.csv")
ADB_SOURCE = Path("results/analysis/adb_kir_sensitivity_v2/adb_per_seed.csv")
DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
STATES = ("known_correct", "known_wrong", "known_rejected", "oos_correct_rejected", "oos_false_accept")
STATE_LABELS = {
    "known_correct": "Known correct",
    "known_wrong": "Known wrong intent",
    "known_rejected": "Known rejected",
    "oos_correct_rejected": "OOS correctly rejected",
    "oos_false_accept": "OOS false accept",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    candidate = PROJECT_ROOT / ".." / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def classify(gold_is_oos: bool, pred_is_oos: bool, gold: str, predicted: str) -> str:
    if gold_is_oos:
        return "oos_correct_rejected" if pred_is_oos else "oos_false_accept"
    if pred_is_oos:
        return "known_rejected"
    return "known_correct" if predicted == gold else "known_wrong"


def load_view(dataset: str, kir: float, seed: int) -> list[dict[str, Any]]:
    path = Path("data/views/protocol_v2_textoir_v1") / dataset / f"seed_{seed}" / f"kir_{kir:.2f}" / "test_combined.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty view: {path}")
    return rows


def load_trainable(row: pd.Series, view: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    metrics_path = resolve_project_path(str(row["metrics_path"]))
    run_dir = metrics_path.parent
    predictions_path = run_dir / "predictions.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = load_json(manifest_path)
    view_ids = [str(x["sample_id"]) for x in view]
    pred_ids = [str(x["sample_id"]) for x in predictions]
    if pred_ids != view_ids:
        raise ValueError(f"Trainable sample order mismatch: {row.dataset}/{row.kir}/{row.seed}")
    if len(predictions) != len(view):
        raise ValueError("Trainable count mismatch")
    split = manifest.get("split_validation", {})
    if split.get("test_sample_ids_sha256") not in (None, ids_hash(view_ids)):
        raise ValueError(f"Trainable split hash mismatch: {row.dataset}/{row.kir}/{row.seed}")
    return predictions, manifest, predictions_path


def find_test_file(data_root: Path, expected_sha: str) -> Path:
    candidates = [path for path in data_root.rglob("test.tsv") if sha256(path) == expected_sha]
    if len(candidates) != 1:
        raise ValueError(f"Expected one matching ADB test snapshot under {data_root}, got {len(candidates)}")
    return candidates[0]


def load_adb(adb_row: pd.Series, view: list[dict[str, Any]]) -> tuple[list[str], list[str], dict[str, Any], Path, Path]:
    manifest_path = resolve_project_path(str(adb_row["run_manifest"]))
    manifest = load_json(manifest_path)
    data_root = Path(str(manifest["data_root"]))
    test_path = find_test_file(data_root, str(manifest["split_sha256"]["test"]))
    with test_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        body = list(reader)
    if reader.fieldnames != ["text", "label"] or len(body) != len(view):
        raise ValueError(f"Malformed ADB test snapshot: {test_path}")
    adb_text = [str(x["text"]) for x in body]
    adb_label = [str(x["label"]) for x in body]
    view_text = [str(x["text"]) for x in view]
    view_label = [str(x["evaluation_label"]) for x in view]
    if adb_text != view_text:
        raise ValueError(f"ADB/protocol text order mismatch: {adb_row.dataset}/{adb_row.kir}/{adb_row.seed}")
    if adb_label != view_label:
        raise ValueError(f"ADB/protocol label order mismatch: {adb_row.dataset}/{adb_row.kir}/{adb_row.seed}")
    directories = manifest.get("artifact_audit", {}).get("prediction_directories", [])
    if len(directories) != 1:
        raise ValueError("ADB prediction directory count is not one")
    prediction_root = Path(directories[0])
    y_true = np.load(prediction_root / "y_true.npy")
    y_pred = np.load(prediction_root / "y_pred.npy")
    known_labels = [str(x) for x in manifest["known_labels"]]
    unknown_id = int(manifest["unknown_label_id"])
    true_labels: list[str] = []
    pred_labels: list[str] = []
    for index, (true_id, pred_id) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
        true_id, pred_id = int(true_id), int(pred_id)
        true_label = "oos" if true_id == unknown_id else known_labels[true_id]
        if true_label != adb_label[index]:
            raise ValueError(f"ADB y_true mismatch: {adb_row.dataset}/{adb_row.kir}/{adb_row.seed}/{index}")
        true_labels.append(true_label)
        pred_labels.append("oos" if pred_id == unknown_id else known_labels[pred_id])
    return true_labels, pred_labels, manifest, test_path, prediction_root


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    fair = pd.read_csv(FAIR_SOURCE)
    fair = fair[fair.method.eq("trainable_k1")].copy()
    adb = pd.read_csv(ADB_SOURCE)
    adb = adb[adb.method.eq("ADB") & adb.valid_semantic_metrics.astype(bool)].copy()
    keys = ["dataset", "kir", "seed"]
    expected = {(dataset, kir, seed) for dataset in DATASETS for kir in KIRS for seed in SEEDS}
    for name, frame in (("Trainable", fair), ("ADB", adb)):
        got = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
        if got != expected or frame.duplicated(keys).any():
            raise ValueError(f"{name} coverage/duplicates invalid: {len(got)}")
    return fair, adb


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    fair, adb = load_sources()
    fair_by_key = {tuple(row[key] for key in ("dataset", "kir", "seed")): row for _, row in fair.iterrows()}
    adb_by_key = {tuple(row[key] for key in ("dataset", "kir", "seed")): row for _, row in adb.iterrows()}
    per_seed: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            for seed in SEEDS:
                key = (dataset, kir, seed)
                view = load_view(dataset, kir, seed)
                trainable, train_manifest, train_pred_path = load_trainable(fair_by_key[key], view)
                adb_true, adb_pred, adb_manifest, adb_test_path, adb_pred_root = load_adb(adb_by_key[key], view)
                states_train: list[str] = []
                states_adb: list[str] = []
                for index, (view_row, train_row, adb_gold, adb_prediction) in enumerate(zip(view, trainable, adb_true, adb_pred)):
                    gold = str(view_row["evaluation_label"])
                    gold_is_oos = gold == "oos"
                    train_pred = str(train_row["predicted_intent"])
                    train_oos = bool(int(train_row["predicted_is_oos"]))
                    train_state = classify(gold_is_oos, train_oos, gold, train_pred)
                    adb_state = classify(gold_is_oos, adb_prediction == "oos", gold, adb_prediction)
                    states_train.append(train_state)
                    states_adb.append(adb_state)
                    transitions.append({"dataset": dataset, "kir": kir, "seed": seed, "trainable_state": train_state, "adb_state": adb_state, "count": 1})
                for method, states in (("Trainable-K1", states_train), ("ADB", states_adb)):
                    for state in STATES:
                        count = int(sum(value == state for value in states))
                        per_seed.append({
                            "dataset": dataset,
                            "kir": kir,
                            "seed": seed,
                            "method": method,
                            "state": state,
                            "count": count,
                            "n_test": len(view),
                            "n_known": int(sum(str(x["evaluation_label"]) != "oos" for x in view)),
                            "n_oos": int(sum(str(x["evaluation_label"]) == "oos" for x in view)),
                            "rate_all": count / len(view),
                            "rate_known": count / max(1, int(sum(str(x["evaluation_label"]) != "oos" for x in view))) if state.startswith("known_") else np.nan,
                            "rate_oos": count / max(1, int(sum(str(x["evaluation_label"]) == "oos" for x in view))) if state.startswith("oos_") else np.nan,
                        })
                alignment.append({
                    "dataset": dataset,
                    "kir": kir,
                    "seed": seed,
                    "n_test": len(view),
                    "test_sample_ids_sha256": ids_hash([str(x["sample_id"]) for x in view]),
                    "protocol_view_sha256": sha256(Path("data/views/protocol_v2_textoir_v1") / dataset / f"seed_{seed}" / f"kir_{kir:.2f}" / "test_combined.jsonl"),
                    "trainable_predictions_sha256": sha256(train_pred_path),
                    "trainable_manifest_sha256": sha256(train_pred_path.parent / "run_manifest.json"),
                    "adb_test_sha256": sha256(adb_test_path),
                    "adb_manifest_sha256": sha256(resolve_project_path(str(adb_by_key[key]["run_manifest"]))),
                    "adb_prediction_sha256": sha256(adb_pred_root / "y_pred.npy"),
                    "aligned": True,
                })
    return pd.DataFrame(per_seed), pd.DataFrame(transitions), alignment


def aggregate(per_seed: pd.DataFrame, transitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = per_seed.groupby(["dataset", "kir", "method", "state"], as_index=False).agg(
        count_mean=("count", "mean"), count_std=("count", "std"), rate_all_mean=("rate_all", "mean"), rate_all_std=("rate_all", "std"), rate_known_mean=("rate_known", "mean"), rate_oos_mean=("rate_oos", "mean"), n_seeds=("seed", "nunique")
    )
    key_cols = ["dataset", "kir", "state"]
    train = grouped[grouped.method.eq("Trainable-K1")][key_cols + ["rate_all_mean", "rate_known_mean", "rate_oos_mean"]].rename(columns={"rate_all_mean": "trainable_rate_all", "rate_known_mean": "trainable_rate_known", "rate_oos_mean": "trainable_rate_oos"})
    adb = grouped[grouped.method.eq("ADB")][key_cols + ["rate_all_mean", "rate_known_mean", "rate_oos_mean"]].rename(columns={"rate_all_mean": "adb_rate_all", "rate_known_mean": "adb_rate_known", "rate_oos_mean": "adb_rate_oos"})
    delta = train.merge(adb, on=key_cols, validate="one_to_one")
    for metric in ("rate_all", "rate_known", "rate_oos"):
        delta[f"delta_{metric}"] = delta[f"trainable_{metric}"] - delta[f"adb_{metric}"]
    transition_summary = transitions.groupby(["dataset", "kir", "trainable_state", "adb_state"], as_index=False).agg(count=("count", "sum"))
    transition_summary["rate_all"] = transition_summary["count"] / transition_summary.groupby(["dataset", "kir"])["count"].transform("sum")
    return grouped, delta, transition_summary


def figures(grouped: pd.DataFrame, delta: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140, "font.family": "DejaVu Sans"})
    for metric, label in (("delta_rate_known", "Trainable - ADB: Known rejection (pp)"), ("delta_rate_oos", "Trainable - ADB: OOS false accept (pp)")):
        values = delta[delta.state.isin(["known_rejected", "oos_false_accept"])].copy()
        state = "known_rejected" if metric == "delta_rate_known" else "oos_false_accept"
        pivot = values[values.state.eq(state)].pivot(index="dataset", columns="kir", values=metric).reindex(index=DATASETS, columns=KIRS) * 100
        vmax = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
        fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
        im = ax.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(KIRS)), [f"{x:.2f}" for x in KIRS])
        ax.set_yticks(range(len(DATASETS)), DATASETS)
        ax.set_xlabel("KIR")
        ax.set_title(label)
        for i in range(len(DATASETS)):
            for j in range(len(KIRS)):
                ax.text(j, i, f"{pivot.iloc[i, j]:+.1f}", ha="center", va="center")
        fig.colorbar(im, ax=ax, label="percentage points")
        fig.savefig(out / f"{state}_delta_heatmap.png", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = grouped[grouped.dataset.eq(dataset) & grouped.state.isin(["known_rejected", "oos_false_accept"])].copy()
        for method, color, marker in (("Trainable-K1", "#1f77b4", "o"), ("ADB", "#d62728", "s")):
            x = subset[(subset.method == method) & subset.state.eq("known_rejected")].set_index("kir").rate_known_mean.reindex(KIRS) * 100
            y = subset[(subset.method == method) & subset.state.eq("oos_false_accept")].set_index("kir").rate_oos_mean.reindex(KIRS) * 100
            ax.plot(x, y, marker=marker, color=color, label=method)
            for kir, xv, yv in zip(KIRS, x, y):
                ax.annotate(f"{kir:.2f}", (xv, yv), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(dataset)
        ax.set_xlabel("Known rejection (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS false accept (%)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Known coverage vs open-space risk across KIR")
    fig.savefig(out / "known_rejection_vs_oos_accept_frontier.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = grouped[(grouped.dataset == dataset) & grouped.state.isin(["known_rejected", "oos_false_accept"])].copy()
        for method, color in (("Trainable-K1", "#1f77b4"), ("ADB", "#d62728")):
            for state, hatch in (("known_rejected", ""), ("oos_false_accept", "//")):
                row = subset[(subset.method == method) & subset.state.eq(state)]
                vals = row.set_index("kir").rate_all_mean.reindex(KIRS).to_numpy() * 100
                xpos = np.arange(len(KIRS)) + (-0.18 if method == "Trainable-K1" else 0.18)
                ax.bar(xpos, vals, width=0.32, color=color, hatch=hatch, alpha=0.85, label=f"{method} / {state}" if dataset == DATASETS[0] else None)
        ax.set_title(dataset)
        ax.set_xticks(np.arange(len(KIRS)), [".25", ".50", ".75"])
        ax.set_xlabel("KIR")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Share of all test samples (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Error-budget composition by dataset and KIR")
    fig.savefig(out / "error_budget_by_dataset_kir.png", bbox_inches="tight")
    plt.close(fig)


def report(grouped: pd.DataFrame, delta: pd.DataFrame, alignments: list[dict[str, Any]], aligned_rows: int, path: Path) -> None:
    def value(dataset: str, kir: float, method: str, state: str, col: str) -> float:
        row = grouped[(grouped.dataset == dataset) & (grouped.kir == kir) & (grouped.method == method) & (grouped.state == state)]
        return float(row.iloc[0][col] * 100)

    lines = [
        "# Trainable-K1 与 ADB 跨数据集逐样本错误预算 V1",
        "",
        "更新时间：2026-08-10",
        "",
        "本报告只读取已完成的 45 个 Trainable-K1 与 45 个 ADB artifact，覆盖三个数据集、三个 KIR 和五个 seed。每个单元先验证 ADB 测试快照与运行 manifest 的 SHA256，再验证文本顺序和 protocol view 完全一致，最后按同一测试样本统计五种互斥错误状态。",
        "",
        "## 合同边界",
        "",
        "Trainable-K1 是 Known-only MiniLM 表示适配；ADB 是 BERT/TextOIR 外部兼容实现。样本级对齐保证错误预算比较可信，但不能消除 backbone、训练目标和边界实现差异，因此本报告不是跨骨干 SOTA 排名。",
        "",
        "## 关键结果",
        "",
        "| 数据集 | KIR | Trainable Known 拒绝 | ADB Known 拒绝 | Trainable OOS 误接收 | ADB OOS 误接收 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for kir in KIRS:
            lines.append(f"| {dataset} | {kir:.2f} | {value(dataset, kir, 'Trainable-K1', 'known_rejected', 'rate_known_mean'):.2f}% | {value(dataset, kir, 'ADB', 'known_rejected', 'rate_known_mean'):.2f}% | {value(dataset, kir, 'Trainable-K1', 'oos_false_accept', 'rate_oos_mean'):.2f}% | {value(dataset, kir, 'ADB', 'oos_false_accept', 'rate_oos_mean'):.2f}% |")
    lines += [
        "",
        "## 解释",
        "",
        "- Trainable 的优势并非跨所有工作点都表现为更低 OOS 误接收；其主要表现是减少部分 Known false rejection，同时在部分数据集/KIR 增加一定 OOS false acceptance。",
        "- 需要同时阅读 OOS F1、F1-All、Known Recall 和错误预算；只看单个 OOS F1 会掩盖工作点交换。",
        "- 数据集差异明显：StackOverflow 的 Trainable/ADB 差异主要是覆盖—拒识平衡；Banking77 和 CLINC150 的方向随 KIR 改变，不能用一个全局机制解释。",
        "- 本分析不使用测试 OOS 调参，不改变任何 checkpoint、registry、阈值或历史 artifact；逐样本结果仅在本地分析脚本中读取，输出只保留聚合计数。",
        "",
        "## 产物",
        "",
        "- 结果：`results/analysis/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/`",
        "- 图表：`figures/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/`",
        f"- 对齐单元：45/45；各单元使用其 protocol test count，共 {aligned_rows:,} 条记录。",
        "",
        "## 下一步",
        "",
        "继续围绕已完成的 fair matrix 做表示几何、边界和错误预算可视化；若要宣称超过 ADB/MOGB/DCLOOS，仍需统一 backbone、训练监督和评估合同，不能由本报告单独推出。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed, transitions, alignments = build()
    grouped, delta, _ = aggregate(per_seed, transitions)
    per_seed.to_csv(args.output_dir / "per_seed_error_budget.csv", index=False)
    grouped.to_csv(args.output_dir / "state_summary.csv", index=False)
    delta.to_csv(args.output_dir / "state_deltas.csv", index=False)
    transitions.groupby(["dataset", "kir", "trainable_state", "adb_state"], as_index=False).agg(count=("count", "sum")).to_csv(args.output_dir / "state_transition_counts.csv", index=False)
    figures(grouped, delta, args.figure_dir)
    report(grouped, delta, alignments, len(transitions), Path("docs/archive/analysis/TRAINABLE_VS_ADB_CROSS_DATASET_ERROR_BUDGET_V1.md"))
    manifest = {
        "schema_version": 1,
        "analysis": "trainable_vs_adb_cross_dataset_error_budget_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "scope": {"datasets": list(DATASETS), "kirs": list(KIRS), "seeds": list(SEEDS)},
        "source_hashes": {"fair_source": sha256(FAIR_SOURCE), "adb_source": sha256(ADB_SOURCE)},
        "alignment": alignments,
        "aligned_units": len(alignments),
        "aligned_rows": int(len(transitions)),
        "raw_text_exported": False,
        "sample_level_predictions_exported": False,
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "aligned_units": len(alignments), "aligned_rows": len(transitions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
