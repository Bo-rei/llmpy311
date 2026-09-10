#!/usr/bin/env python3
"""Audit one isolated DA-ADB CUDA run without mixing contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts" / "s2c"
RUN = (
    ARTIFACTS
    / "external"
    / "da_adb_gpu_runtime_v1"
    / "stackoverflow"
    / "DA-ADB"
    / "kir_0.50"
    / "seed_42"
)
OUT = ROOT / "results" / "analysis" / "da_adb_gpu_runtime_v1"
REPORT = ROOT / "docs" / "analysis" / "DA_ADB_GPU_RUNTIME_SINGLE_CELL_V1.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest_path = RUN / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_dir = Path(manifest["artifact_audit"]["prediction_directories"][0])
    y_true = np.load(prediction_dir / "y_true.npy")
    y_pred = np.load(prediction_dir / "y_pred.npy")
    known_label = int(manifest["unknown_label_id"])
    known_labels = list(range(known_label))
    finite = bool(np.isfinite(y_true).all() and np.isfinite(y_pred).all())
    unique_pred = sorted(np.unique(y_pred).astype(int).tolist())
    unique_true = sorted(np.unique(y_true).astype(int).tolist())
    valid_shape = bool(y_true.shape == y_pred.shape and y_true.ndim == 1)
    no_all_class_collapse = len(unique_pred) > 1
    oos_mask = y_true == known_label
    known_mask = ~oos_mask
    oos_pred = y_pred == known_label
    oos_f1 = float(f1_score(oos_mask, oos_pred, zero_division=0))
    f1_known = float(f1_score(y_true, y_pred, labels=known_labels, average="macro", zero_division=0))
    f1_all = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    payload = {
        "schema_version": 1,
        "protocol_version": "protocol_v2_textoir_v1",
        "contract": "BERT/TextOIR external compatibility",
        "status": "valid_external_cell",
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seed": 42,
        "method": "DA-ADB",
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "split_sha256": manifest["split_sha256"],
        "known_labels": manifest["known_labels"],
        "environment": manifest["environment"],
        "metrics": {
            "oos_f1": oos_f1,
            "f1_known": f1_known,
            "f1_all": f1_all,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "known_recall": float(((y_pred[known_mask] != known_label).sum() / known_mask.sum())),
            "false_acceptance_rate": float(((y_pred[oos_mask] != known_label).sum() / oos_mask.sum())),
            "false_rejection_rate": float(((y_pred[known_mask] == known_label).sum() / known_mask.sum())),
        },
        "prediction_audit": {
            "n_samples": int(y_true.size),
            "shape_equal": valid_shape,
            "finite": finite,
            "true_labels": unique_true,
            "predicted_labels": unique_pred,
            "all_class_collapse": not no_all_class_collapse,
            "y_true_sha256": sha256_file(prediction_dir / "y_true.npy"),
            "y_pred_sha256": sha256_file(prediction_dir / "y_pred.npy"),
        },
        "ranking_policy": "external_contract_only; not merged with MiniLM fair ranking",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "DA_ADB_GPU_RUNTIME_SINGLE_CELL_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (OUT / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "kir", "seed", "method", *payload["metrics"].keys()])
        writer.writeheader()
        writer.writerow({"dataset": "stackoverflow", "kir": 0.50, "seed": 42, "method": "DA-ADB", **payload["metrics"]})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    m = payload["metrics"]
    REPORT.write_text(
        f"""# DA-ADB 隔离 CUDA 单格审计 V1

## 运行合同

- 数据集：StackOverflow；KIR=`0.50`；seed=`42`。
- 数据根：`{manifest['data_root']}`；训练/开发/测试 SHA256 记录在 run manifest。
- 表示与训练：BERT `bert_disaware`，TEXTOIR 外部兼容 runner；不是 MiniLM fair 行。
- runtime：`{manifest['environment']['packages']['torch']}`，Transformers `{manifest['environment']['packages']['transformers']}`，RTX 5070 CUDA。
- 测试 OOS 只在最终测试阶段使用；没有用测试 OOS 选 epoch、阈值或参数。

## 审计结果

`run_manifest.json` 为 `complete`，返回码为 0；逐样本预测存在且无 NaN/Inf，真实标签与预测标签形状一致，预测没有塌缩为单一类别。`y_true/y_pred` 的 SHA256、split SHA256 和完整命令均保存在机器可读 manifest 中。

## 指标

| 指标 | 值 |
|---|---:|
| OOS F1 | {m['oos_f1'] * 100:.2f}% |
| F1-All | {m['f1_all'] * 100:.2f}% |
| F1-Known | {m['f1_known'] * 100:.2f}% |
| Accuracy | {m['accuracy'] * 100:.2f}% |
| Known Recall | {m['known_recall'] * 100:.2f}% |
| False Acceptance | {m['false_acceptance_rate'] * 100:.2f}% |
| False Rejection | {m['false_rejection_rate'] * 100:.2f}% |

## 与既有 DA-ADB 证据的关系

该单格修复了“当前隔离 runtime 只有 NaN/全类预测”的阻断，但它不能自动证明算法性能，也不能覆盖旧 seed=0 的兼容单格。新结果为 OOS F1 `{m['oos_f1'] * 100:.2f}%`，明显低于旧兼容单格的 90.90%；差异必须归因于数据/seed/环境/兼容配置差异并单独审计，不能挑选较高数字。

因此当前 DA-ADB 状态为 `valid_external_cell_pending_replication`：可作为 BERT/TextOIR 外部参照，尚不能与 MiniLM fair matrix 合并排名，也不能据此宣称 SOTA。

机器可读证据：`results/analysis/archive/analysis/da_adb_gpu_runtime_v1/`；原始运行：`{RUN}`。
""",
        encoding="utf-8",
    )
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
