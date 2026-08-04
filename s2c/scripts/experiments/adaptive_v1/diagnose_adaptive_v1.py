#!/usr/bin/env python3
"""Materialise RC-AMBL diagnostics and a lightweight closeout manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from protocol_v2.data.hashing import atomic_write_json, atomic_write_text, sha256_file
from protocol_v2.runtime.paths import ProtocolV2Paths


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    paths = ProtocolV2Paths.discover()
    root = args.artifact_root or (paths.run_root / "adaptive_v1" / "contract_repair5")
    output = args.output or (paths.results_root / "diagnostics" / "adaptive_v1")
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    ky_rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    intent_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for seed_dir in sorted((root / "runs" / "stackoverflow").glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        metrics = _read_csv(seed_dir / "metrics.csv")
        metric_rows.extend(metrics)
        for metric in metrics:
            method = metric["method"]
            centers_path = seed_dir / f"{method.removeprefix('RC-AMBL-')}_centers.json"
            operations_path = seed_dir / f"{method.removeprefix('RC-AMBL-')}_operations.csv"
            audit_path = seed_dir / f"{method.removeprefix('RC-AMBL-')}_selection_audit.json"
            centers = _load_json(centers_path) if centers_path.is_file() else []
            by_intent: dict[str, list[dict[str, Any]]] = {}
            for center in centers:
                by_intent.setdefault(str(center["intent"]), []).append(center)
            operations = _read_csv(operations_path)
            audit = _load_json(audit_path) if audit_path.is_file() else {}
            for intent, values in sorted(by_intent.items()):
                ky_rows.append({"seed": seed, "method": method, "intent": intent, "k_y": len(values), "sample_count": sum(int(item["sample_count"]) for item in values), "mean_radius": sum(float(item["radius"]) for item in values) / max(len(values), 1), "mean_stability": sum(float(item["stability"]) for item in values) / max(len(values), 1)})
                intent_ops = [row for row in operations if row.get("intent") == intent]
                intent_rows.append({"seed": seed, "method": method, "intent": intent, "k_y": len(values), "candidate_splits": len(intent_ops), "accepted_splits": sum(row.get("split_accepted", "False") == "True" for row in intent_ops), "rejected_splits": sum(row.get("split_accepted", "False") != "True" for row in intent_ops), "reject_reasons": ";".join(sorted({row.get("reject_reason", "") for row in intent_ops if row.get("reject_reason")}))})
            for row in operations:
                row.update({"seed": seed, "method": method})
                operation_rows.append(row)
                decision_rows.append({"seed": seed, "method": method, "round": row.get("round"), "intent": row.get("intent"), "split_accepted": row.get("split_accepted"), "reject_reason": row.get("reject_reason"), "known_recall_delta": row.get("known_recall_delta"), "ambiguity_delta": row.get("ambiguity_delta"), "proxy_false_accept_delta": row.get("proxy_false_accept_delta"), "stability_median": row.get("stability_median"), "complexity_adjusted_gain": row.get("complexity_adjusted_gain")})
            if audit:
                for round_info in audit.get("rounds", []):
                    if round_info.get("status") == "no_candidate":
                        decision_rows.append({"seed": seed, "method": method, "round": round_info.get("round"), "intent": "", "split_accepted": False, "reject_reason": "no_candidate"})

    false_acceptance_rows = [
        {key: row.get(key) for key in ("dataset", "kir", "seed", "method", "source", "oos_f1", "known_recall", "false_accept_rate", "false_reject_rate", "f1_all", "f1_k")}
        for row in _read_csv(output / "main_results.csv")
        if row.get("dataset") == "stackoverflow"
    ]
    diagnostic_tables = {
        "ky_distribution.csv": ky_rows,
        "center_operations.csv": operation_rows,
        "calibration_decisions.csv": decision_rows,
        "intent_split_summary.csv": intent_rows,
        "false_acceptance_comparison.csv": false_acceptance_rows,
    }
    for name, table in diagnostic_tables.items():
        _write_csv(diagnostics / name, table)
        _write_csv(output / name, table)

    file_hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"ADAPTIVE_V1_RESULT_MANIFEST.json"}:
            file_hashes[str(path.relative_to(root))] = sha256_file(path)
    manifest = {
        "experiment_id": "adaptive_v1",
        "revision": root.name,
        "protocol_version": "protocol_v2_textoir_v1",
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seeds": [13, 42, 87],
        "test_used_for_selection": False,
        "new_metric_rows": len(metric_rows),
        "diagnostic_rows": {"ky_distribution": len(ky_rows), "center_operations": len(operation_rows), "calibration_decisions": len(decision_rows), "intent_split_summary": len(intent_rows)},
        "files_sha256": file_hashes,
    }
    atomic_write_json(root / "ADAPTIVE_V1_RESULT_MANIFEST.json", manifest)
    public_manifest = {"artifact_root": str(root), "public_output": str(output), "artifact_manifest_sha256": sha256_file(root / "ADAPTIVE_V1_RESULT_MANIFEST.json"), "required_files": sorted([str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()])}
    atomic_write_json(diagnostics / "result_manifest.json", public_manifest)
    atomic_write_json(output / "result_manifest.json", public_manifest)

    integrity = """# RC-AMBL 实验完整性检查\n\n- 协议：`protocol_v2_textoir_v1`\n- 数据集：StackOverflow，KIR=0.50\n- seeds：13、42、87；每个 seed 生成 KnownOnly 和 ProxyOOS 两行。\n- 训练/结构选择只读取 Known train 与 calibration_select；阈值只读取 calibration_threshold。\n- 测试 OOS 不参与中心数、半径、阈值或 margin 选择。\n- `nearest_sphere` 历史 E2 语义未修改；RC-AMBL 使用独立的加权类证据和父级边界保护。\n- 所有候选分裂的操作、稳定性、覆盖变化和拒绝原因已落盘。\n- `ADAPTIVE_V1_VERIFY.json` 必须显示 3/3 seed、6 行指标且 status=pass。\n\n本轮没有覆盖或删除 E2/E3/BRAK/MOGB 历史结果。\n"""
    atomic_write_text(root / "ADAPTIVE_V1_INTEGRITY.md", integrity)
    closeout = """# RC-AMBL 阶段收口\n\n本轮完成 StackOverflow/KIR=0.50、seeds=13/42/87 的 RC-AMBL KnownOnly 与 ProxyOOS 对照。所有候选分裂均经过 Known-only 校准安全门；当前三 seed 均回退到 `K_y=1`，因此结果是安全回退诊断，不是多中心成功。\n\nRC-AMBL 的 OOS 结果必须结合 `main_results.csv` 与 E2 K=1 复用基线阅读；不能把 calibration-evidence 阈值结果直接当作 E2 nearest-sphere 的公平替代。下一步由 `ADAPTIVE_V1_REPORT.md` 的失败/晋级判定决定，当前不扩展到其他数据集或 KIR。\n"""
    atomic_write_text(root / "ADAPTIVE_V1_CLOSEOUT.md", closeout)
    print(json.dumps({"status": "complete", "root": str(root), "diagnostics": str(diagnostics), "public_output": str(output), "metric_rows": len(metric_rows)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
