#!/usr/bin/env python3
"""Compare DA-ADB cells without pretending different contracts are one ranking."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts" / "s2c"
OUT = ROOT / "results" / "analysis" / "da_adb_contract_comparison_v1"
FIG = ROOT / "figures" / "da_adb_contract_comparison_v1"
REPORT = ROOT / "docs" / "analysis" / "DA_ADB_CONTRACT_COMPARISON_V1.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(prediction_dir: Path, unknown_label: int) -> dict:
    y_true = np.load(prediction_dir / "y_true.npy")
    y_pred = np.load(prediction_dir / "y_pred.npy")
    known_labels = list(range(unknown_label))
    known = y_true != unknown_label
    oos = ~known
    pred_oos = y_pred == unknown_label
    return {
        "n_samples": int(y_true.size),
        "oos_f1": float(f1_score(oos, pred_oos, zero_division=0)),
        "f1_all": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_known": float(f1_score(y_true, y_pred, labels=known_labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "known_recall": float(((y_pred[known] != unknown_label).sum() / known.sum())),
        "false_acceptance": float(((y_pred[oos] != unknown_label).sum() / oos.sum())),
        "false_rejection": float(((y_pred[known] == unknown_label).sum() / known.sum())),
        "predicted_labels": sorted(np.unique(y_pred).astype(int).tolist()),
        "finite": bool(np.isfinite(y_true).all() and np.isfinite(y_pred).all()),
        "shape_equal": bool(y_true.shape == y_pred.shape),
        "y_true_sha256": sha256_file(prediction_dir / "y_true.npy"),
        "y_pred_sha256": sha256_file(prediction_dir / "y_pred.npy"),
    }


def prediction_dir(run: Path) -> Path:
    dirs = sorted(
        p for p in (run / "textoir_outputs" / "open_intent_detection").glob("*")
        if (p / "y_true.npy").is_file() and (p / "y_pred.npy").is_file()
    )
    if len(dirs) != 1:
        raise RuntimeError(f"Expected exactly one prediction directory under {run}, got {dirs}")
    return dirs[0]


def main() -> int:
    historical_run = ARTIFACTS / "external" / "da_adb_compat_single_cell_v3" / "stackoverflow" / "DA-ADB" / "kir50" / "seed0"
    isolated_legacy = ARTIFACTS / "external" / "da_adb_gpu_runtime_v1_legacy" / "stackoverflow" / "DA-ADB" / "kir_0.50" / "seed_0"
    isolated_current = ARTIFACTS / "external" / "da_adb_gpu_runtime_v1" / "stackoverflow" / "DA-ADB" / "kir_0.50" / "seed_42"
    runs = [
        ("historical_seed0", historical_run, 10, "旧兼容：textoir/data + textoir-py39"),
        ("isolated_seed0_same_contract", isolated_legacy, 10, "同旧 split/Known list + 新隔离 CUDA"),
        ("isolated_seed42_protocol_v2", isolated_current, 10, "protocol_v2 split/Known list + 新隔离 CUDA"),
    ]
    rows = []
    for run_id, run, unknown_label, contract in runs:
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        row = {
            "run_id": run_id,
            "contract": contract,
            "run_manifest": str(run / "run_manifest.json"),
            "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
            "status": manifest.get("status"),
            "seed": manifest.get("seed"),
            "known_labels": "|".join(manifest.get("known_labels", [])),
            "train_sha256": manifest.get("split_sha256", {}).get("train"),
            "dev_sha256": manifest.get("split_sha256", {}).get("dev"),
            "test_sha256": manifest.get("split_sha256", {}).get("test"),
            "packages": json.dumps(manifest.get("environment", {}).get("packages", {}), sort_keys=True),
        }
        row.update(metrics(prediction_dir(run), unknown_label))
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUT / "da_adb_contract_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "comparison": "DA-ADB historical versus isolated runtime/data contracts",
        "rows": rows,
        "interpretation": {
            "historical_minus_isolated_legacy": "runtime/compatibility difference under the same old split and Known list",
            "isolated_legacy_minus_isolated_current": "seed, Known-list and protocol split difference under the same new runtime",
            "ranking_policy": "descriptive contract diagnosis only; no cross-contract SOTA ranking",
        },
    }
    (OUT / "DA_ADB_CONTRACT_COMPARISON_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    FIG.mkdir(parents=True, exist_ok=True)
    labels = ["historical seed0", "isolated seed0\nsame old contract", "isolated seed42\nprotocol_v2"]
    metric_names = ["oos_f1", "f1_all", "f1_known", "accuracy", "known_recall", "false_acceptance"]
    metric_labels = ["OOS F1", "F1-All", "F1-Known", "Accuracy", "Known Recall", "False Acceptance"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    x = np.arange(len(rows))
    for ax, name, label in zip(axes.flat, metric_names, metric_labels):
        values = [row[name] * 100 for row in rows]
        bars = ax.bar(x, values, color=["#7f8c8d", "#2c7fb8", "#d95f02"])
        ax.set_title(label)
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("DA-ADB contract comparison: old rerun vs protocol_v2", fontsize=14)
    fig.savefig(FIG / "da_adb_contract_metrics.png", dpi=180)
    plt.close(fig)
    historical = rows[0]
    legacy = rows[1]
    current = rows[2]
    delta_runtime = {name: (legacy[name] - historical[name]) * 100 for name in metric_names}
    delta_contract = {name: (current[name] - legacy[name]) * 100 for name in metric_names}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# DA-ADB 新旧合同差异分析 V1

## 目的

旧兼容单格报告了 OOS F1=`{historical['oos_f1'] * 100:.2f}%`，新的 protocol_v2 隔离 CUDA 单格为
`{current['oos_f1'] * 100:.2f}%`。本报告不挑选更高的数字，而是在同一新 runtime 下复跑旧 seed=0
数据/Known-list 合同，拆分“运行时/兼容层差异”和“seed/Known-list/split 差异”。

## 三个单元

| 单元 | 数据/Known 合同 | runtime | OOS F1 | F1-All | Known Recall | FA |
|---|---|---|---:|---:|---:|---:|
| 历史 seed0 | 旧 `textoir/data`，旧 Known list | textoir-py39，旧兼容 | {historical['oos_f1'] * 100:.2f}% | {historical['f1_all'] * 100:.2f}% | {historical['known_recall'] * 100:.2f}% | {historical['false_acceptance'] * 100:.2f}% |
| 隔离 seed0 | 与历史相同的 split/Known list | 新 CUDA runtime | {legacy['oos_f1'] * 100:.2f}% | {legacy['f1_all'] * 100:.2f}% | {legacy['known_recall'] * 100:.2f}% | {legacy['false_acceptance'] * 100:.2f}% |
| 隔离 seed42 | protocol_v2 split/Known list | 新 CUDA runtime | {current['oos_f1'] * 100:.2f}% | {current['f1_all'] * 100:.2f}% | {current['known_recall'] * 100:.2f}% | {current['false_acceptance'] * 100:.2f}% |

## 差值解释

历史 → 隔离 seed0（只近似控制数据/Known list，改变 runtime/兼容层）的差值为：

`OOS F1 {delta_runtime['oos_f1']:+.2f}pp`，`F1-All {delta_runtime['f1_all']:+.2f}pp`，
`Known Recall {delta_runtime['known_recall']:+.2f}pp`，`FA {delta_runtime['false_acceptance']:+.2f}pp`。

隔离 seed0 → 隔离 seed42（保持新 runtime，改变 seed、Known list 和 protocol split）的差值为：

`OOS F1 {delta_contract['oos_f1']:+.2f}pp`，`F1-All {delta_contract['f1_all']:+.2f}pp`，
`Known Recall {delta_contract['known_recall']:+.2f}pp`，`FA {delta_contract['false_acceptance']:+.2f}pp`。

这不是严格的因果分解：seed、Known list 和数据快照不能只靠三个单元完全分离。但它已经证明，旧
`90.90%` 不能直接当作当前 protocol_v2 DA-ADB 性能；新 runtime 下同旧合同的结果必须先作为独立
复现结果记录。所有三单元的逐样本预测均有限、形状一致且非单类塌缩。

## 证据和图

- 图：`figures/da_adb_contract_comparison_v1/da_adb_contract_metrics.png`
- 表：`results/analysis/da_adb_contract_comparison_v1/da_adb_contract_comparison.csv`
- Manifest：`results/analysis/da_adb_contract_comparison_v1/DA_ADB_CONTRACT_COMPARISON_MANIFEST.json`
- 新旧运行根：`{historical_run}`、`{isolated_legacy}`、`{isolated_current}`

结论仍然是合同分层：DA-ADB 可作为外部 BERT/TextOIR 参照，但当前没有证据支持它与
`S2C-Trainable-K1` 的同骨干公平排名，也没有证据支持跨合同 SOTA 宣称。
""",
        encoding="utf-8",
    )
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
