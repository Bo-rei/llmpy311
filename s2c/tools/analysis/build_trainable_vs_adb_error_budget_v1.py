#!/usr/bin/env python3
"""Build a sample-aligned error budget for Trainable-K1 versus ADB.

The analysis is intentionally post-hoc and read-only.  It proves text-order
and sample-id alignment through the per-seed protocol data snapshots, then
exports only aggregate state transitions (never raw text or sample-level
predictions).
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


ROOT = Path("../artifacts/s2c")
DATASET = "stackoverflow"
KIR = 0.50
SEEDS = (42, 87, 100)
TRAINABLE_DIRS = {
    42: ROOT / "runs/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_v1/kir_0.50/runs/stackoverflow/seed_42",
    87: ROOT / "runs/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_v1/kir_0.50/runs/stackoverflow/seed_87",
    100: ROOT / "runs/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_extension_v1/kir_0.50/runs/stackoverflow/seed_100",
}
ADB_ROOT = ROOT / "external/adb_gpu_runtime_v1/stackoverflow/ADB/kir50"
ADB_DATA_ROOTS = {
    42: ROOT / "external/adb_protocol_v2_probe_data_v1/stackoverflow",
    87: ROOT / "external/adb_protocol_v2_probe_data_v1_seed87/stackoverflow",
    100: ROOT / "external/adb_protocol_v2_probe_data_v1_seed100/stackoverflow",
}
VIEW_ROOT = Path("data/views/protocol_v2_textoir_v1/stackoverflow")

STATE_ORDER = (
    "known_correct",
    "known_wrong",
    "known_rejected",
    "oos_correct_rejected",
    "oos_false_accept",
)
STATE_LABELS = {
    "known_correct": "Known 正确分类",
    "known_wrong": "Known 错意图",
    "known_rejected": "Known 被拒绝",
    "oos_correct_rejected": "OOS 正确拒绝",
    "oos_false_accept": "OOS 误接收",
}
STATE_FIG_LABELS = {
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


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def classify(gold_is_oos: bool, predicted_is_oos: bool, gold: str, predicted: str) -> str:
    if gold_is_oos:
        return "oos_correct_rejected" if predicted_is_oos else "oos_false_accept"
    if predicted_is_oos:
        return "known_rejected"
    return "known_correct" if predicted == gold else "known_wrong"


def _view_path(seed: int) -> Path:
    return VIEW_ROOT / f"seed_{seed}/kir_0.50/test_combined.jsonl"


def load_view(seed: int) -> list[dict[str, Any]]:
    path = _view_path(seed)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 6000:
        raise ValueError(f"Unexpected protocol test count for seed={seed}: {len(rows)}")
    return rows


def load_trainable(seed: int, view: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = TRAINABLE_DIRS[seed]
    rows = [json.loads(line) for line in (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = load_json(root / "run_manifest.json")
    if len(rows) != len(view):
        raise ValueError(f"Trainable count mismatch seed={seed}")
    view_ids = [str(row["sample_id"]) for row in view]
    pred_ids = [str(row["sample_id"]) for row in rows]
    if pred_ids != view_ids:
        raise ValueError(f"Trainable sample order mismatch seed={seed}")
    split = manifest.get("split_validation", {})
    if split.get("test_sample_ids_sha256") not in (None, ids_hash(view_ids)):
        raise ValueError(f"Trainable manifest sample hash mismatch seed={seed}")
    return rows, manifest


def load_adb(seed: int, view: list[dict[str, Any]]) -> tuple[list[str], list[str], dict[str, Any]]:
    run_root = ADB_ROOT / f"seed{seed}"
    manifest = load_json(run_root / "run_manifest.json")
    data_root = ADB_DATA_ROOTS[seed]
    test_path = data_root / "test.tsv"
    if sha256(test_path) != manifest["split_sha256"]["test"]:
        raise ValueError(f"ADB test snapshot hash mismatch seed={seed}")
    with test_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        body = list(reader)
    if reader.fieldnames != ["text", "label"] or len(body) != len(view):
        raise ValueError(f"Malformed ADB test snapshot seed={seed}")
    adb_texts = [str(row["text"]) for row in body]
    adb_labels = [str(row["label"]) for row in body]
    view_texts = [str(row["text"]) for row in view]
    view_labels = [str(row["evaluation_label"]) for row in view]
    if list(adb_texts) != view_texts:
        raise ValueError(f"ADB/protocol text order mismatch seed={seed}")
    expected = [label if label != "oos" else "oos" for label in view_labels]
    if list(adb_labels) != expected:
        raise ValueError(f"ADB/protocol evaluation label mismatch seed={seed}")
    prediction_dirs = manifest["artifact_audit"]["prediction_directories"]
    if len(prediction_dirs) != 1:
        raise ValueError(f"Expected one ADB prediction directory seed={seed}")
    prediction_root = Path(prediction_dirs[0])
    y_true = np.load(prediction_root / "y_true.npy")
    y_pred = np.load(prediction_root / "y_pred.npy")
    if len(y_true) != len(view) or len(y_pred) != len(view):
        raise ValueError(f"ADB prediction count mismatch seed={seed}")
    known_labels = [str(x) for x in manifest["known_labels"]]
    unknown_id = int(manifest["unknown_label_id"])
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    for index, (true_id, pred_id) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
        true_id = int(true_id)
        pred_id = int(pred_id)
        expected_label = str(adb_labels[index])
        true_label = "oos" if true_id == unknown_id else known_labels[true_id]
        predicted_label = "oos" if pred_id == unknown_id else known_labels[pred_id]
        if true_label != expected_label:
            raise ValueError(f"ADB y_true mismatch seed={seed}, row={index}")
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
    return true_labels, predicted_labels, manifest


def build_seed(seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    view = load_view(seed)
    trainable, train_manifest = load_trainable(seed, view)
    adb_true, adb_pred, adb_manifest = load_adb(seed, view)
    rows: list[dict[str, Any]] = []
    for index, (view_row, train_row, adb_gold, adb_prediction) in enumerate(zip(view, trainable, adb_true, adb_pred)):
        gold = str(view_row["evaluation_label"])
        gold_is_oos = gold == "oos"
        train_pred = str(train_row["predicted_intent"])
        train_is_oos = bool(int(train_row["predicted_is_oos"]))
        train_state = classify(gold_is_oos, train_is_oos, gold, train_pred)
        adb_is_oos = adb_prediction == "oos"
        adb_state = classify(gold_is_oos, adb_is_oos, gold, adb_prediction)
        rows.append({
            "dataset": DATASET,
            "kir": KIR,
            "seed": seed,
            "row_index": index,
            "sample_id": str(view_row["sample_id"]),
            "gold_is_oos": int(gold_is_oos),
            "trainable_state": train_state,
            "adb_state": adb_state,
            "trainable_predicted_is_oos": int(train_is_oos),
            "adb_predicted_is_oos": int(adb_is_oos),
        })
    frame = pd.DataFrame(rows)
    alignment = {
        "seed": seed,
        "aligned": True,
        "count": len(frame),
        "test_sample_ids_sha256": ids_hash(frame["sample_id"].tolist()),
        "protocol_view_sha256": sha256(_view_path(seed)),
        "trainable_predictions_sha256": sha256(TRAINABLE_DIRS[seed] / "predictions.jsonl"),
        "trainable_metrics_sha256": sha256(TRAINABLE_DIRS[seed] / "metrics.json"),
        "adb_test_sha256": sha256(ADB_DATA_ROOTS[seed] / "test.tsv"),
        "adb_run_manifest_sha256": sha256(ADB_ROOT / f"seed{seed}/run_manifest.json"),
        "trainable_contract": train_manifest.get("representation_mode"),
        "adb_contract": adb_manifest.get("contract", "BERT/TextOIR compatibility"),
    }
    return frame, alignment


def state_counts(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for state in STATE_ORDER:
        subset = frame[frame[column].eq(state)]
        rows.append({
            "method": "Trainable-K1" if column == "trainable_state" else "ADB",
            "state": state,
            "state_label": STATE_LABELS[state],
            "count": int(len(subset)),
            "rate_all": float(len(subset) / len(frame)),
            "rate_known": float(len(subset) / max(1, int((frame.gold_is_oos == 0).sum()))) if state.startswith("known_") else np.nan,
            "rate_oos": float(len(subset) / max(1, int((frame.gold_is_oos == 1).sum()))) if state.startswith("oos_") else np.nan,
        })
    return pd.DataFrame(rows)


def transition_counts(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.groupby(["trainable_state", "adb_state"], dropna=False).size().reset_index(name="count")
    data["trainable_state_label"] = data["trainable_state"].map(STATE_LABELS)
    data["adb_state_label"] = data["adb_state"].map(STATE_LABELS)
    data["rate_all"] = data["count"] / len(frame)
    return data


def draw_figures(counts: pd.DataFrame, transitions: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140})
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    pivot = counts.pivot(index="state", columns="method", values="rate_all").reindex(STATE_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(STATE_ORDER))
    width = 0.38
    for offset, method, color in [(-width / 2, "Trainable-K1", "#1f77b4"), (width / 2, "ADB", "#d62728")]:
        ax.bar(x + offset, pivot[method].to_numpy() * 100, width, label=method, color=color)
    ax.set_xticks(x, [STATE_FIG_LABELS[s] for s in STATE_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Share of all test samples (%)")
    ax.set_title("StackOverflow/KIR=0.50: error-budget states (3 seeds)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "state_error_budget.png", bbox_inches="tight")
    plt.close(fig)

    matrix = transitions.groupby(["trainable_state", "adb_state"], as_index=False)["count"].sum().pivot(index="trainable_state", columns="adb_state", values="count").fillna(0)
    matrix = matrix.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(STATE_ORDER)), [STATE_FIG_LABELS[s] for s in STATE_ORDER], rotation=25, ha="right")
    ax.set_yticks(range(len(STATE_ORDER)), [STATE_FIG_LABELS[s] for s in STATE_ORDER])
    ax.set_xlabel("ADB state")
    ax.set_ylabel("Trainable-K1 state")
    ax.set_title("Same-sample state transitions: Trainable-K1 vs ADB")
    for i in range(len(STATE_ORDER)):
        for j in range(len(STATE_ORDER)):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Sample count")
    fig.savefig(out_dir / "state_transition_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    trade = counts[counts.state.isin(["known_rejected", "oos_false_accept"])].pivot(index="state", columns="method", values="rate_all").reindex(["known_rejected", "oos_false_accept"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    x = np.arange(len(trade.index))
    for offset, method, color in [(-width / 2, "Trainable-K1", "#1f77b4"), (width / 2, "ADB", "#d62728")]:
        ax.bar(x + offset, trade[method].to_numpy() * 100, width, label=method, color=color)
    ax.set_xticks(x, ["Known rejected", "OOS false accept"])
    ax.set_ylabel("Share of all test samples (%)")
    ax.set_title("Open-space error-budget trade-off")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "known_reject_oos_accept_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def write_report(counts: pd.DataFrame, transitions: pd.DataFrame, alignments: list[dict[str, Any]], out: Path) -> None:
    def rate(method: str, state: str) -> float:
        row = counts[(counts.method == method) & (counts.state == state)]
        return float(row.iloc[0].rate_all * 100) if len(row) else float("nan")

    lines = [
        "# Trainable-K1 与 ADB 逐样本错误预算 V1",
        "",
        "更新时间：2026-08-10",
        "",
        "本分析只读取已完成 artifact，不训练、不调参、不修改历史结果。范围固定为 StackOverflow、KIR=0.50、seed={42,87,100}。ADB 使用 BERT/TextOIR 兼容合同，Trainable-K1 使用 Known-only MiniLM 合同；因此这里回答的是同一测试样本上的错误预算差异，不是同骨干 SOTA 排名。",
        "",
        "## 对齐结论",
        "",
        "- 三个 seed 均通过：ADB 独立测试快照 SHA256 与运行 manifest 一致；文本顺序与对应 protocol view 完全一致；Trainable prediction 的 sample_id 顺序与 protocol view 完全一致。",
        "- 每个 seed 对齐 6000 个测试样本，共 18000 条对齐记录；未导出原始文本或逐样本公开预测。",
        "",
        "## 错误预算（3 seed 汇总）",
        "",
        "| 状态 | Trainable-K1 | ADB | 含义 |",
        "|---|---:|---:|---|",
        f"| Known 正确分类 | {rate('Trainable-K1', 'known_correct'):.2f}% | {rate('ADB', 'known_correct'):.2f}% | 已知样本被正确分类 |",
        f"| Known 错意图 | {rate('Trainable-K1', 'known_wrong'):.2f}% | {rate('ADB', 'known_wrong'):.2f}% | 接受为 Known 但意图错误 |",
        f"| Known 被拒绝 | {rate('Trainable-K1', 'known_rejected'):.2f}% | {rate('ADB', 'known_rejected'):.2f}% | Known false rejection |",
        f"| OOS 正确拒绝 | {rate('Trainable-K1', 'oos_correct_rejected'):.2f}% | {rate('ADB', 'oos_correct_rejected'):.2f}% | OOS 正确判为未知 |",
        f"| OOS 误接收 | {rate('Trainable-K1', 'oos_false_accept'):.2f}% | {rate('ADB', 'oos_false_accept'):.2f}% | OOS false acceptance |",
        "",
        "## 当前可解释结论",
        "",
        f"1. Trainable-K1 在这三个 seed 上把 Known 被拒绝从 ADB 的 {rate('ADB', 'known_rejected') / 0.5:.2f}%（条件于 3000 个 Known 样本）降到 {rate('Trainable-K1', 'known_rejected') / 0.5:.2f}%，但 OOS 误接收从 {rate('ADB', 'oos_false_accept') / 0.5:.2f}% 升到 {rate('Trainable-K1', 'oos_false_accept') / 0.5:.2f}%。因此当前 StackOverflow 证据是“少一些 Known 误拒、略多一些 OOS 误接收”的工作点交换，不是无代价优势。",
        "2. ADB 与 Trainable 的差异同时包含 backbone、训练目标和边界实现差异；本分析不能把差异归因给 MiniLM 训练或 ADB 边界中的单一因素。",
        "3. 逐样本对齐只证明错误预算比较可信，不改变任何阈值、checkpoint、registry 或 test 选择流程；不能把三 seed StackOverflow 结果外推为跨数据集结论。",
        "",
        "## 产物",
        "",
        "- 机器可读：`results/analysis/archive/analysis/trainable_vs_adb_error_budget_v1/`",
        "- 图表：`figures/archive/analysis/trainable_vs_adb_error_budget_v1/`",
        "- 逐状态转移仅保留聚合计数；原始文本和逐样本预测不进入公开结果。",
        "",
        "## 对齐来源",
        "",
    ]
    for item in alignments:
        lines.append(f"- seed={item['seed']}: test_sample_ids_sha256={item['test_sample_ids_sha256']}; adb_test_sha256={item['adb_test_sha256']}; aligned={item['aligned']}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/archive/analysis/trainable_vs_adb_error_budget_v1"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/archive/analysis/trainable_vs_adb_error_budget_v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    alignments: list[dict[str, Any]] = []
    for seed in SEEDS:
        frame, alignment = build_seed(seed)
        frames.append(frame)
        alignments.append(alignment)
    all_rows = pd.concat(frames, ignore_index=True)
    counts = pd.concat([state_counts(frame, "trainable_state") for frame in frames] + [state_counts(frame, "adb_state") for frame in frames], ignore_index=True)
    counts = counts.groupby(["method", "state", "state_label"], as_index=False).agg(count=("count", "sum"), rate_all=("rate_all", "mean"), rate_known=("rate_known", "mean"), rate_oos=("rate_oos", "mean"))
    counts.to_csv(args.output_dir / "state_counts.csv", index=False)
    transitions = transition_counts(all_rows)
    transitions.to_csv(args.output_dir / "state_transition_counts.csv", index=False)
    # The aggregate row contains only state counts and hashes, not sample text.
    seed_rows = []
    for seed, frame in zip(SEEDS, frames):
        seed_rows.append({"dataset": DATASET, "kir": KIR, "seed": seed, "n_test": len(frame), "n_known": int((frame.gold_is_oos == 0).sum()), "n_oos": int((frame.gold_is_oos == 1).sum()), **{f"trainable_{state}": int((frame.trainable_state == state).sum()) for state in STATE_ORDER}, **{f"adb_{state}": int((frame.adb_state == state).sum()) for state in STATE_ORDER}})
    pd.DataFrame(seed_rows).to_csv(args.output_dir / "per_seed_error_budget.csv", index=False)
    draw_figures(counts, transitions, args.figure_dir)
    write_report(counts, transitions, alignments, Path("docs/archive/analysis/TRAINABLE_VS_ADB_ERROR_BUDGET_V1.md"))
    manifest = {
        "schema_version": 1,
        "analysis": "trainable_vs_adb_error_budget_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "scope": {"dataset": DATASET, "kir": KIR, "seeds": list(SEEDS)},
        "alignment": alignments,
        "n_aligned_rows": int(len(all_rows)),
        "state_order": list(STATE_ORDER),
        "raw_text_exported": False,
        "sample_level_predictions_exported": False,
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "n_aligned_rows": len(all_rows), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
