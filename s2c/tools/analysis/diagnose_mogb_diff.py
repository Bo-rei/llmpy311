#!/usr/bin/env python3
"""Localize the MOGB reproduction discrepancy without launching training.

The four requested combinations are represented explicitly.  Existing strict
StackOverflow/Banking artifacts are read as the modern-code/current-data
observation (D); the other combinations are recorded as blocked when the
required upstream materials are absent or the pinned checkout fails before
data loading.  This is a diagnostic artifact, not a new MOGB experiment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "third_party/mogb_official"
COMPAT = ROOT / "third_party/mogb_compat"
CURRENT_DATA = ROOT / "data/sources/textoir/dffe2b1b848a069a6808f8089b4cb9bd16e2062b/stackoverflow"
STRICT_ROOT = ROOT.parent / "artifacts/s2c/external/mogb_exact_reproduction_v1/audit"
OUT = ROOT / "results/diagnostics/mogb_diff"
DOC = ROOT / "docs/archive/mogb_reproduction/MOGB_DIAGNOSIS.md"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or ".git" in candidate.parts or "__pycache__" in candidate.parts:
            continue
        digest.update(candidate.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_failure() -> str:
    if not (OFFICIAL / "utils").exists():
        return "MOGB.py:3 import utils -> pinned checkout has no utils package"
    if not (OFFICIAL / "requirements.txt").is_file():
        return "preflight requirements -> requirements.txt missing"
    return "not statically blocked"


def current_contract() -> dict[str, Any]:
    audit = load_json(STRICT_ROOT / "dataset_audit.json")
    known = load_json(STRICT_ROOT / "known_intents.json")
    sample_hashes = file_hash(STRICT_ROOT / "sample_hashes.json")
    files = audit.get("files", {})
    return {
        "data_tree_sha256": audit.get("source_tree_sha256"),
        "file_sha256": {name: value.get("sha256") for name, value in sorted(files.items())},
        "label_order": audit.get("label_order", []),
        "label_order_sha256": hash_json(audit.get("label_order", [])),
        "known_intents": known.get("known_intents", audit.get("known_intents", [])),
        "known_intents_sha256": hash_json(known.get("known_intents", audit.get("known_intents", []))),
        "sample_ids_sha256": sample_hashes,
        "sample_id_status": "row_record_hashes_only; raw sample IDs not retained in strict artifact",
    }


def modern_observation() -> dict[str, Any]:
    manifest = load_json(STRICT_ROOT / "official_fixed/mode_manifest.json")
    provenance = load_json(STRICT_ROOT / "MOGB_EXACT_PROVENANCE.json")
    contract = current_contract()
    history = manifest.get("history", [])
    ball_rows = manifest.get("ball_rows", [])
    return {
        "status": "observed_existing_artifact",
        "data_source": "current_textoir_snapshot",
        "code_source": "modern_compatibility_runner",
        **contract,
        "token_ids_status": "not_retained",
        "initial_bert_embeddings_status": "not_retained",
        "epoch_ce_loss": [{"epoch": row.get("epoch"), "value": row.get("train_ce_loss")} for row in history],
        "epoch_subcentroid_loss": [{"epoch": row.get("epoch"), "value": row.get("subcentroid_loss")} for row in history],
        "ball_count": len(ball_rows),
        "per_intent_ball_count": dict(sorted(__import__("collections").Counter(str(row.get("majority_label")) for row in ball_rows).items())),
        "ball_sizes": [row.get("sample_count") for row in ball_rows],
        "ball_radii": [row.get("radius") for row in ball_rows],
        "best_checkpoint_sha256": provenance.get("results", [{}])[0].get("checkpoint_sha256"),
        "test_predictions_status": "not_retained; final metrics and ball manifest retained",
        "first_failure": None,
        "notes": "No training was launched by this diagnostic; values come from the frozen strict single-cell artifact.",
    }


def blocked_observation(group: str, data_source: str, code_source: str, reason: str, contract: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "status": "blocked_preflight",
        "data_source": data_source,
        "code_source": code_source,
        **(contract or {
            "data_tree_sha256": None,
            "file_sha256": {},
            "label_order": [],
            "label_order_sha256": None,
            "known_intents": [],
            "known_intents_sha256": None,
            "sample_ids_sha256": None,
            "sample_id_status": "unavailable",
        }),
        "token_ids_status": "unavailable",
        "initial_bert_embeddings_status": "unavailable",
        "epoch_ce_loss": [],
        "epoch_subcentroid_loss": [],
        "ball_count": None,
        "per_intent_ball_count": None,
        "ball_sizes": None,
        "ball_radii": None,
        "best_checkpoint_sha256": None,
        "test_predictions_status": "unavailable",
        "first_failure": reason,
        "notes": f"Combination {group} was not launched; no upstream material was substituted.",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    contract = current_contract()
    original_data_candidates = [
        OFFICIAL / "data",
        OFFICIAL / "dataset",
        ROOT.parent / "MOGB/data",
        ROOT.parent / "third_party/MOGB/data",
    ]
    original_data = next((path for path in original_data_candidates if path.is_dir()), None)
    failure = source_failure()
    rows = []
    observations: dict[str, dict[str, Any]] = {}
    observations["A_original_code_original_data"] = blocked_observation(
        "A", "original_accompanying_data", "pinned_original_code",
        "original accompanying MOGB data directory not found; earliest runnable comparison is unavailable",
        None,
    ) if original_data is None else blocked_observation("A", str(original_data), "pinned_original_code", failure, None)
    observations["B_original_code_textoir_data"] = blocked_observation(
        "B", "current_textoir_snapshot", "pinned_original_code", failure, contract
    )
    observations["C_modern_code_original_data"] = blocked_observation(
        "C", "original_accompanying_data", "modern_compatibility_runner",
        "original accompanying MOGB data directory not found; compatibility runner was not given a substitute",
        None,
    ) if original_data is None else blocked_observation("C", str(original_data), "modern_compatibility_runner", "not launched by no-new-training contract", None)
    observations["D_modern_code_textoir_data"] = modern_observation()

    fieldnames = [
        "group", "status", "data_source", "code_source", "data_tree_sha256", "label_order_sha256",
        "known_intents_sha256", "sample_ids_sha256", "token_ids_status", "initial_embeddings_status",
        "epoch_ce_loss_status", "epoch_subcentroid_loss_status", "ball_count", "per_intent_ball_count_status",
        "ball_size_status", "ball_radius_status", "best_checkpoint_sha256", "test_predictions_status",
        "first_failure", "notes",
    ]
    for group, observation in observations.items():
        rows.append({
            "group": group,
            "status": observation["status"],
            "data_source": observation["data_source"],
            "code_source": observation["code_source"],
            "data_tree_sha256": observation.get("data_tree_sha256"),
            "label_order_sha256": observation.get("label_order_sha256"),
            "known_intents_sha256": observation.get("known_intents_sha256"),
            "sample_ids_sha256": observation.get("sample_ids_sha256"),
            "token_ids_status": observation.get("token_ids_status"),
            "initial_embeddings_status": observation.get("initial_bert_embeddings_status"),
            "epoch_ce_loss_status": "available" if observation.get("epoch_ce_loss") else "unavailable",
            "epoch_subcentroid_loss_status": "available" if observation.get("epoch_subcentroid_loss") else "unavailable",
            "ball_count": observation.get("ball_count"),
            "per_intent_ball_count_status": "available" if observation.get("per_intent_ball_count") is not None else "unavailable",
            "ball_size_status": "available" if observation.get("ball_sizes") is not None else "unavailable",
            "ball_radius_status": "available" if observation.get("ball_radii") is not None else "unavailable",
            "best_checkpoint_sha256": observation.get("best_checkpoint_sha256"),
            "test_predictions_status": observation.get("test_predictions_status"),
            "first_failure": observation.get("first_failure"),
            "notes": observation.get("notes"),
        })
    with (OUT / "stage_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    first = {
        "classification": "public_code_not_reproduced_under_available_materials",
        "fixed_cell": "StackOverflow/KIR=0.50/seed=0",
        "original_source_commit": "5b689e2a03de0d86ec41212825e5db8d7f0e5c02",
        "original_data_found": original_data is not None,
        "pinned_code_earliest_static_failure": failure,
        "observed_group": "D_modern_code_textoir_data",
        "observed_group_note": "D is an existing strict artifact, not a new run; its paper gap remains not_reproduced_strict.",
        "groups": observations,
        "reasoning": [
            "The original accompanying dataset/sample draw is not available locally, so A and C cannot be executed without violating the no-substitution contract.",
            "The pinned checkout fails before training because MOGB.py imports a missing utils package; B is not a valid original-code run on current data.",
            "The existing modern/current strict artifact proves the compatibility path can converge, but it cannot isolate data versus implementation without A/C.",
        ],
    }
    (OUT / "first_divergence.json").write_text(json.dumps(first, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    document = f"""# MOGB 四组合差异诊断

## 范围

固定目标为 StackOverflow、KIR=0.50、seed=0。本诊断只读取现有
strict artifact 和 pinned third-party checkout，不启动新的 MOGB 训练，不改写
`../artifacts`。

## 组合状态

| 组合 | 代码 | 数据 | 状态 |
|---|---|---|---|
| A | pinned original | original accompanying data | {observations['A_original_code_original_data']['status']} |
| B | pinned original | current TextOIR snapshot | {observations['B_original_code_textoir_data']['status']} |
| C | modern compatibility | original accompanying data | {observations['C_modern_code_original_data']['status']} |
| D | modern compatibility | current TextOIR snapshot | {observations['D_modern_code_textoir_data']['status']} |

## 首次分叉

`first_divergence.json` 和 `stage_comparison.csv` 保存了每一组合的来源哈希、标签/Known
列表哈希以及可用的训练和粒球字段。原始 MOGB 配套数据未在本地材料中发现，因此不能把
TextOIR 数据伪装成 A/C；pinned checkout 的静态最早失败点是
`MOGB.py:3` 导入缺失的 `utils` 包。D 使用已有 strict artifact，未重新执行。

## 结论

最终分类：**public_code_not_reproduced_under_available_materials**。

这不是对数据或代码单独归因的结论：缺失 A/C 意味着无法通过四组合实验隔离数据契约与
兼容实现。现有 D 仍应保持 `not_reproduced_strict`，不能与论文数字直接混合，也不能由此
继续添加非必要兼容补丁。

## 证据位置

* `results/diagnostics/mogb_diff/stage_comparison.csv`
* `results/diagnostics/mogb_diff/first_divergence.json`
* `third_party/mogb_official/ORIGINAL_SOURCE.md`
* `third_party/mogb_official/PATCH_LOG.md`
* `../artifacts/s2c/external/mogb_exact_reproduction_v1/audit/`
"""
    DOC.write_text(document, encoding="utf-8")
    print(json.dumps({"classification": first["classification"], "groups": {key: value["status"] for key, value in observations.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
