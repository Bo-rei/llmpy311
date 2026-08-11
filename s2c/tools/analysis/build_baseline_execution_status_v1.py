#!/usr/bin/env python3
"""Record external-baseline execution status without mixing contracts.

This is an analysis/provenance report.  It does not train a model and never
promotes a legacy compatibility cell into a protocol_v2 result.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts" / "s2c"
OUT = ROOT / "results" / "analysis" / "baseline_execution_status_v1"
DOC = ROOT / "docs" / "analysis" / "BASELINE_EXECUTION_STATUS_V1.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def manifest_status(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing_manifest", "detail": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "invalid_manifest", "detail": f"{path}: {exc}"}
    return {
        "status": payload.get("status", "unknown"),
        "return_code": payload.get("return_code"),
        "detail": str(path),
        "artifact_complete": payload.get("artifact_audit", {}).get("complete"),
        "data_root": payload.get("data_root"),
        "known_label_count": len(payload.get("known_labels", [])),
        "environment": payload.get("environment", {}).get("packages"),
        "preflight": payload.get("method_preflight", {}).get("complete"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data_rows = []
    for seed in (42, 87, 100):
        export = ROOT / "data" / "exports" / "protocol_v2_textoir_v1" / "adb" / "stackoverflow" / f"seed_{seed}" / "kir_0.50"
        source_manifest = export / "export_manifest.json"
        metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
        data_rows.append(
            {
                "dataset": "stackoverflow",
                "kir": 0.5,
                "seed": seed,
                "export_dir": str(export),
                "source_export_manifest_sha256": sha256_file(source_manifest),
                "registry_sha256": metadata.get("registry_sha256"),
                "train_sha256": next(item["sha256"] for item in metadata["files"] if item["relative_path"] == "train.tsv"),
                "dev_sha256": next(item["sha256"] for item in metadata["files"] if item["relative_path"] == "dev.tsv"),
                "test_sha256": next(item["sha256"] for item in metadata["files"] if item["relative_path"] == "test.tsv"),
                "materialized_root": str(ARTIFACTS / "external" / "adb_protocol_v1" / "stackoverflow" / f"seed_{seed}" / "data"),
            }
        )

    with (OUT / "protocol_data_hashes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data_rows[0]))
        writer.writeheader()
        writer.writerows(data_rows)

    attempt_root = ARTIFACTS / "external" / "adb_protocol_v1" / "stackoverflow" / "ADB" / "kir50"
    rows = []
    first = manifest_status(attempt_root / "seed42" / "run_manifest.json")
    rows.append(
        {
            "method": "ADB",
            "dataset": "stackoverflow",
            "kir": 0.5,
            "seed": 42,
            "attempt": "seed42",
            **first,
            "contract": "protocol_v2_data_adapter",
            "result_usable": False,
        }
    )
    for attempt in ("seed42_attempt2", "seed42_attempt3"):
        rows.append(
            {
                "method": "ADB",
                "dataset": "stackoverflow",
                "kir": 0.5,
                "seed": 42,
                "attempt": attempt,
                **manifest_status(attempt_root / attempt / "run_manifest.json"),
                "contract": "protocol_v2_data_adapter",
                "result_usable": False,
            }
        )
    rows.append(
        {
            "method": "DA-ADB",
            "dataset": "stackoverflow",
            "kir": 0.5,
            "seed": 42,
            "attempt": "not_started_shared_preflight_blocker",
            "status": "not_started",
            "return_code": None,
            "detail": "Shares the ADB environment probe; no same-protocol metric was started.",
            "artifact_complete": False,
            "data_root": None,
            "known_label_count": None,
            "environment": None,
            "preflight": False,
            "contract": "protocol_v2_data_adapter",
            "result_usable": False,
        }
    )
    compat = [
        ("ADB", ARTIFACTS / "external" / "adb_compat_single_cell_v2" / "stackoverflow" / "ADB" / "kir50" / "seed0" / "run_manifest.json"),
        ("DA-ADB", ARTIFACTS / "external" / "da_adb_compat_single_cell_v3" / "stackoverflow" / "DA-ADB" / "kir50" / "seed0" / "run_manifest.json"),
    ]
    for method, path in compat:
        status = manifest_status(path)
        rows.append(
            {
                "method": method,
                "dataset": "stackoverflow",
                "kir": 0.5,
                "seed": 0,
                "attempt": "legacy_compat_single_cell",
                **status,
                "contract": "legacy_textoir_compatibility",
                "result_usable": bool(status.get("artifact_complete")),
            }
        )

    fields = [
        "method", "dataset", "kir", "seed", "attempt", "status", "return_code", "detail",
        "artifact_complete", "data_root", "known_label_count", "environment", "preflight",
        "contract", "result_usable",
    ]
    with (OUT / "attempt_status.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_version": "protocol_v2_textoir_v1",
        "scope": "external_baseline execution status and contract separation",
        "inputs": [
            str(ROOT / "tools" / "compat" / "textoir" / "run_external_textoir.py"),
            str(OUT / "protocol_data_hashes.csv"),
            str(ROOT / "docs" / "archive" / "mogb_reproduction" / "mogb_integration" / "ADB_DAADB_AUDIT.md"),
        ],
        "protocol_data_seed_count": len(data_rows),
        "attempt_count": len(rows),
        "usable_protocol_v2_metrics": sum(int(row["result_usable"] and row["contract"] == "protocol_v2_data_adapter") for row in rows),
        "legacy_compatibility_metrics": sum(int(row["result_usable"] and row["contract"] == "legacy_textoir_compatibility") for row in rows),
        "status": "same_protocol_baseline_blocked_after_data_adapter",
    }
    write_json(OUT / "MANIFEST.json", payload)

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(
        """# 外部基线同协议运行状态 V1

> 这是执行状态和合同审计，不是新的 SOTA 排名。旧兼容单格数字与 protocol_v2 结果严格分层。

## 已完成

- StackOverflow/KIR=0.50 的 protocol_v2 ADB 导出已经从 `data/exports/protocol_v2_textoir_v1/adb/` 物化到独立 artifact 数据根。
- seed=42、87、100 的 train/dev/test 和 Known labels 均复制前后 SHA256 一致；Known 列表与 registry 完全一致。
- `run_external_textoir.py` 新增显式 `--data-root` 和 `--known-labels-file`，外部 runner 不再必须读取 `textoir/data`。

## ADB/DA-ADB 当前状态

| 方法 | 同协议状态 | 说明 |
|---|---|---|
| ADB | blocked_preflight | 第一次已进入训练前数据加载，但本地 BERT 只有 `model.safetensors`；转换到隔离 `pytorch_model.bin` 后，后续 CUDA/torch 环境探针在 90 秒内超时。没有可用同协议指标。 |
| DA-ADB | not_started_shared_blocker | 与 ADB 共用同一旧版 torch/Transformers/CUDA 兼容层；为避免重复制造不完整运行，没有启动训练。 |
| ADB/DA-ADB 旧兼容单格 | legacy_compatibility_only | 仍保留 seed=0 的历史 TextOIR TSV 结果，不是 protocol_v2 五 seed 公平结果。 |

## 旧兼容数字的边界

旧 ADB 单格为 OOS F1 89.47、F1-All 87.63；旧 DA-ADB 单格为 OOS F1 90.90、F1-All 89.23。它们使用独立 TextOIR snapshot、BERT 和单 seed，不能与当前 Trainable/Frozen 五 seed fair rows 合并排名，也不能被称为当前 protocol_v2 的正式 SOTA 对比。

## 当前实验结论

同协议数据适配已经完成，但同协议 ADB/DA-ADB 尚未产出有效指标。因此当前最可靠的比较仍是：Trainable K=1、Frozen/fixed-K、MOGB frozen-MiniLM 组件和原生检测器的同协议分析；ADB/DA-ADB 只作为合同不同的兼容性参照。当前不能据此宣称超过 ADB 或 DA-ADB。

## 下一步

1. 在独立、可验证的旧版环境中修复 CUDA/torch 探针或改用已验证的 CPU/GPU runtime；
2. 先完成 StackOverflow/KIR=0.50/seed=42 的 ADB 和 DA-ADB 单格；
3. 只有单格通过完整 artifact audit 后，才扩展 seed=87、100；
4. 同协议结果达到 3 seed 后，才加入既有性能热图和 Pareto 图。

详细执行记录：`results/analysis/baseline_execution_status_v1/attempt_status.csv`、`protocol_data_hashes.csv`、`MANIFEST.json`。
""",
        encoding="utf-8",
    )
    print(DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
