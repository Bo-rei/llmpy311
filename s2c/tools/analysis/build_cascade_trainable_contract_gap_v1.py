#!/usr/bin/env python3
"""Audit the data-contract gap between Trainable K=1 and the old Cascade.

This is an analysis-only tool.  It does not train or evaluate a model.  The
purpose is to prevent a Trainable MiniLM Gate trained on the TEXTOIR protocol
from being silently combined with Router/Expert checkpoints trained on the
legacy ``v19`` prepared data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c"
RACAL_ROOT = ARTIFACT_ROOT / "runs" / "protocol_v2_textoir_v1" / "racal_v1" / "runs" / "trainable_k1"
V19_ROOT = ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19"
CASCADE_ROOT = ARTIFACT_ROOT / "outputs" / "experiments" / "cascade_full" / "gpu_kir50"
OUT = ROOT / "results" / "analysis" / "cascade_trainable_contract_gap_v1"
FIG = ROOT / "figures" / "cascade_trainable_contract_gap_v1"
REPORT = ROOT / "docs" / "analysis" / "CASCADE_TRAINABLE_CONTRACT_GAP_V1.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_values(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def row_hashes(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    source_ids = [str(row.get("source_id", "")) for row in rows]
    return {
        "sample_id_sha256": sha256_values(sample_ids) if all(sample_ids) else None,
        "source_id_sha256": sha256_values(source_ids) if all(source_ids) else None,
    }


def collect_v2(seed: int = 42) -> dict[str, Any]:
    manifest_path = RACAL_ROOT / f"seed_{seed}" / "training_manifest.json"
    payload = load_json(manifest_path)
    input_payload = payload["input"]
    view = input_payload["view_contract"]
    return {
        "layer": "trainable_gate",
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seed": seed,
        "protocol": "protocol_v2_textoir_v1",
        "data_root": "protocol_v2 canonical/views (manifest-bound)",
        "train_count": int(view["train_count"]),
        "calibration_count": int(view["calibration_count"]),
        "test_count": int(view["test_count"]),
        "test_known_count": int(view["test_known_count"]),
        "test_oos_count": int(view["test_oos_count"]),
        "train_sample_id_sha256": input_payload["train_sample_ids_sha256"],
        "calibration_sample_id_sha256": input_payload["calibration_sample_ids_sha256"],
        "test_sample_id_sha256": input_payload["test_sample_ids_sha256"],
        "source_id_sha256": None,
        "router_expert_training_contract": "not included; Gate-only checkpoint",
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
    }


def collect_v19(seed: int = 42) -> dict[str, Any]:
    root = V19_ROOT / "stackoverflow" / f"kir50_seed{seed}" / "gate"
    split_rows = {name: load_json(root / f"{name}.json") for name in ("train", "val", "test")}
    hashes = {name: row_hashes(rows) for name, rows in split_rows.items()}
    test_labels = [int(row.get("label", 0)) for row in split_rows["test"]]
    return {
        "layer": "legacy_cascade_components",
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seed": seed,
        "protocol": "v19_prepared_data",
        "data_root": str(root.resolve()),
        "train_count": len(split_rows["train"]),
        "calibration_count": len(split_rows["val"]),
        "test_count": len(split_rows["test"]),
        "test_known_count": sum(label == 0 for label in test_labels),
        "test_oos_count": sum(label == 1 for label in test_labels),
        "train_sample_id_sha256": hashes["train"]["sample_id_sha256"],
        "calibration_sample_id_sha256": hashes["val"]["sample_id_sha256"],
        "test_sample_id_sha256": hashes["test"]["sample_id_sha256"],
        "source_id_sha256": hashes["test"]["source_id_sha256"],
        "router_expert_training_contract": "Router/Expert checkpoints trained under v19 component plan",
        "checkpoint_sha256": None,
    }


def collect_component_manifest(seed: int = 42) -> dict[str, Any]:
    payload = load_json(CASCADE_ROOT / "component_plan.json")
    row = next(item for item in payload["plans"] if item["dataset"] == "stackoverflow" and int(item["seed"]) == seed)
    return {
        "dataset": "stackoverflow",
        "seed": seed,
        "protocol": payload.get("protocol"),
        "source": row.get("source"),
        "data_root": row.get("data_root"),
        "router": row.get("router"),
        "experts": row.get("experts"),
        "router_exists": Path(row["router"]).is_file(),
        "experts_exists": Path(row["experts"]).is_dir(),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    text_lines: list[str] = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_counts(rows: list[dict[str, Any]]) -> None:
    labels = ["Train", "Calibration/Val", "Test", "Test Known", "Test OOS"]
    v2 = rows[0]
    v19 = rows[1]
    values_v2 = [v2["train_count"], v2["calibration_count"], v2["test_count"], v2["test_known_count"], v2["test_oos_count"]]
    values_v19 = [v19["train_count"], v19["calibration_count"], v19["test_count"], v19["test_known_count"], v19["test_oos_count"]]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x = range(len(labels))
    width = 0.36
    ax.bar([i - width / 2 for i in x], values_v2, width, label="protocol_v2 Trainable checkpoint", color="#2b6cb0")
    ax.bar([i + width / 2 for i in x], values_v19, width, label="v19 Cascade components", color="#c53030")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("row count")
    ax.set_title("StackOverflow/KIR=.50/seed42: Trainable Gate vs legacy Cascade contract")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    for values, offset in ((values_v2, -width / 2), (values_v19, width / 2)):
        for index, value in enumerate(values):
            ax.text(index + offset, value + max(values) * 0.015, str(value), ha="center", va="bottom", fontsize=8)
    fig.savefig(FIG / "split_count_contract_gap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(v2: dict[str, Any], v19: dict[str, Any], component: dict[str, Any]) -> str:
    return f"""# Trainable Gate 与旧 Cascade 的数据合同断点（V1）

更新时间：2026-08-10  
分析阶段：`cascade_trainable_contract_gap_v1`  
目的：阻止将 `protocol_v2_textoir_v1` 的 Trainable Gate 与旧 `v19` Router/Expert 直接混合。

## 结论

当前不能直接生成“Trainable MiniLM 完整 Cascade”结果。原因不是 Trainable checkpoint 缺失，而是
现有完整 Cascade 的下游组件和数据不是同一协议：Trainable checkpoint 使用 TEXTOIR canonical/views，
而 Router/Expert 来自 v19 prepared data。若直接替换 Gate encoder，会同时改变输入样本、Known 列表、
split 和下游训练分布，结果无法归因。

## 证据

| 项目 | protocol_v2 Trainable | v19 Cascade components |
|---|---:|---:|
| train 行数 | {v2['train_count']} | {v19['train_count']} |
| calibration/val 行数 | {v2['calibration_count']} | {v19['calibration_count']} |
| test 行数 | {v2['test_count']} | {v19['test_count']} |
| test Known 行数 | {v2['test_known_count']} | {v19['test_known_count']} |
| test OOS 行数 | {v2['test_oos_count']} | {v19['test_oos_count']} |
| 数据协议 | `{v2['protocol']}` | `{v19['protocol']}` |
| Trainable test sample hash | `{v2['test_sample_id_sha256']}` | `{v19['test_sample_id_sha256'] or '无 sample_id 字段'}` |
| v19 test source_id hash | — | `{v19['source_id_sha256']}` |

v19 的 gate JSON 只有 `text/intent/domain/split/label/source_id/title/question/tag` 等字段，没有当前
protocol 的 `sample_id`。即使文本数量接近，也不能视为同一测试集合。

下游组件清单：

- component plan：`{component['protocol']}`
- source：`{component['source']}`
- Router exists：`{component['router_exists']}`
- Expert root exists：`{component['experts_exists']}`

## 对当前结果的影响

因此当前结果必须分为：

1. Trainable K=1：当前协议的 Gate-only 结果；
2. Frozen/CE-Recon Cascade：旧 v19 下游组件的 Cascade 结果；
3. 历史 fulltex Ours：历史旧合同的完整 Cascade；
4. 外部 MOGB/ADB/DCLOOS：各自独立监督和数据合同。

不能把第 1 项直接和第 2 项合成同协议端到端排名。

## 下一项真正需要运行的实验

若要回答“Trainable 表示是否改善完整 Cascade”，必须先在当前 `protocol_v2_textoir_v1` 的同一
registry/views 上重新训练 Router/Expert，然后固定这些下游 checkpoint，比较：

```text
Frozen K=1 Cascade
Trainable K=1 Cascade
```

首轮建议 StackOverflow/KIR=.50/seeds={13,42,87}，之后再扩展其他数据集。训练和评价期间不得读取
旧 v19 test 文件，也不能复用 v19 Router/Expert 冒充同协议结果。

## 产物

- `results/analysis/archive/analysis/cascade_trainable_contract_gap_v1/contract_comparison.csv`
- `results/analysis/archive/analysis/cascade_trainable_contract_gap_v1/component_manifest.json`
- `figures/archive/analysis/cascade_trainable_contract_gap_v1/split_count_contract_gap.png`

这是合同审计，不是模型失败结论；在下游组件迁移完成前，Trainable 的 Gate-only 结果仍然是当前
最可靠的自有候选。
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    v2 = collect_v2()
    v19 = collect_v19()
    component = collect_component_manifest()
    rows = [v2, v19]
    write_csv(rows, OUT / "contract_comparison.csv")
    atomic_write(OUT / "component_manifest.json", json.dumps(component, ensure_ascii=False, indent=2) + "\n")
    plot_counts(rows)
    atomic_write(REPORT, build_report(v2, v19, component))
    manifest = {
        "analysis_id": "cascade_trainable_contract_gap_v1",
        "protocol": "protocol_v2_textoir_v1",
        "sources": {
            "trainable_manifest": str(RACAL_ROOT / "seed_42" / "training_manifest.json"),
            "trainable_manifest_sha256": sha256_file(RACAL_ROOT / "seed_42" / "training_manifest.json"),
            "legacy_v19_gate_train": str(V19_ROOT / "stackoverflow" / "kir50_seed42" / "gate" / "train.json"),
            "legacy_v19_gate_test": str(V19_ROOT / "stackoverflow" / "kir50_seed42" / "gate" / "test.json"),
            "component_plan": str(CASCADE_ROOT / "component_plan.json"),
            "component_plan_sha256": sha256_file(CASCADE_ROOT / "component_plan.json"),
        },
        "test_used_for_selection": False,
        "outputs": [
            "contract_comparison.csv",
            "component_manifest.json",
            "split_count_contract_gap.png",
            str(REPORT.relative_to(ROOT)),
        ],
    }
    atomic_write(OUT / "MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "rows": len(rows), "report": str(REPORT), "figure": str(FIG / "split_count_contract_gap.png")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
