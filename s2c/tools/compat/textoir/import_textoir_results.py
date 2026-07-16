#!/usr/bin/env python3
"""将上游 TextOIR 预测导入为 s2c 可审计的 JSONL/JSON 产物。

导入过程不信任上游汇总 CSV 作为唯一真值：它会重新读取 ``y_true.npy``
和 ``y_pred.npy``，校验与原始 test.tsv 的行级对齐，然后统一复算 accuracy、
Known macro-F1、Open/OOS F1 和 macro-F1。上游 CSV 仅作为对照 provenance 保留。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from ._common import read_tsv, sha256_file, write_json
except ImportError:  # Direct script execution.
    from _common import read_tsv, sha256_file, write_json


def locate_prediction_dir(run_dir: Path) -> Path:
    """要求每个 run 唯一对应一个同时包含 y_true/y_pred 的上游输出目录。"""

    candidates = sorted((run_dir / "textoir_outputs" / "open_intent_detection").glob("*"))
    matches = [
        path
        for path in candidates
        if (path / "y_true.npy").is_file() and (path / "y_pred.npy").is_file()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one prediction directory under {run_dir}, found {len(matches)}")
    return matches[0]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, unknown_id: int) -> dict:
    """按 TextOIR 的 K 个 Known 类 + 1 个 unknown 类重新计算多类指标。

    ``known_macro_f1`` 只平均 Known 类；``open_oos_f1`` 只取 unknown 类；
    ``macro_f1`` 才是 K+1 个类别的平均。这些指标不与 s2c Gate-only 的二分 OOS F1 混表。
    """

    labels = range(unknown_id + 1)
    per_label_f1 = {}
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label_f1[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    known = [score for label, score in per_label_f1.items() if label != unknown_id]
    return {
        "samples": int(y_true.size),
        "accuracy": float(np.mean(y_true == y_pred)),
        "known_macro_f1": float(np.mean(known)) if known else None,
        "open_oos_f1": per_label_f1.get(unknown_id),
        "macro_f1": float(np.mean(list(per_label_f1.values()))) if per_label_f1 else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Defaults to RUN_DIR/imported")
    return parser.parse_args()


def main() -> int:
    """验证已完成 run，重算指标，并导出逐样本预测。"""

    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "imported").resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Cannot import run with status={manifest.get('status')!r}")

    prediction_dir = locate_prediction_dir(run_dir)
    y_true = np.load(prediction_dir / "y_true.npy", allow_pickle=False).reshape(-1)
    y_pred = np.load(prediction_dir / "y_pred.npy", allow_pickle=False).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Prediction shape mismatch: {y_true.shape} != {y_pred.shape}")

    test_path = Path(manifest["textoir_root"]) / "data" / manifest["dataset"] / "test.tsv"
    test_rows = read_tsv(test_path)
    if len(test_rows) != y_true.size:
        raise ValueError(f"Test/prediction length mismatch: {len(test_rows)} != {y_true.size}")

    # unknown_id 固定为 Known label list 之后的第 K+1 类。不从
    # y_pred 的最大值反推，因为某次预测可能恰好没有输出 unknown。
    known_labels = list(manifest["known_labels"])
    unknown_id = int(manifest["unknown_label_id"])
    valid_ids = set(range(unknown_id + 1))
    observed_ids = set(map(int, y_true)) | set(map(int, y_pred))
    if not observed_ids <= valid_ids:
        raise ValueError(f"Predictions contain invalid label ids: {sorted(observed_ids - valid_ids)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (row, truth, prediction) in enumerate(zip(test_rows, y_true, y_pred)):
            truth_id = int(truth)
            prediction_id = int(prediction)
            # 通过原始 TSV 和冻结的 known-label list 独立重建 truth id。
            # 任何一行不匹配都说明预测顺序或协议已漂移，必须终止导入。
            expected_truth = known_labels.index(row["label"]) if row["label"] in known_labels else unknown_id
            if truth_id != expected_truth:
                raise ValueError(
                    f"Prediction alignment failed at test row {index}: {truth_id} != {expected_truth}"
                )
            true_is_oos = truth_id == unknown_id
            if not true_is_oos:
                oos_source = "known"
            elif manifest["dataset"] == "oos" and row["label"] == "oos":
                oos_source = "native_oos"
            else:
                oos_source = "heldout_unknown"
            record = {
                "sample_id": f"textoir:{manifest['dataset']}:test:{index:06d}",
                "text": row["text"],
                "source_intent": row["label"],
                "true_label_id": truth_id,
                "predicted_label_id": prediction_id,
                "true_intent": known_labels[truth_id] if truth_id < unknown_id else manifest["unknown_label"],
                "predicted_intent": known_labels[prediction_id] if prediction_id < unknown_id else manifest["unknown_label"],
                "true_is_oos": true_is_oos,
                "predicted_is_oos": prediction_id == unknown_id,
                "oos_source": oos_source,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "dataset": manifest["dataset"],
        "method": manifest["method"],
        "known_cls_ratio": manifest["known_cls_ratio"],
        "seed": manifest["seed"],
        "known_labels": known_labels,
        "split_sha256": sha256_file(test_path),
        "prediction_counts": dict(sorted(Counter(map(int, y_pred)).items())),
        "metrics": metrics(y_true.astype(int), y_pred.astype(int), unknown_id),
        "predictions": str(prediction_path),
    }
    results_csv = run_dir / "results" / "results.csv"
    if results_csv.is_file():
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        summary["upstream_result"] = rows[-1] if rows else None
    write_json(output_dir / "import_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
