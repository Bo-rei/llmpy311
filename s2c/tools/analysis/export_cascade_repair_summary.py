#!/usr/bin/env python3
"""把完整 Cascade 修复目录导出为可读汇总和 provenance。

该脚本只聚合已经生成的 ``eval_results.json``，不重新选择 Gate、K 或阈值，
也不重新推理。这样论文表格可以从一个小 CSV 生成，而无需手工抄写逐样本结果。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("evaluations/*/kir*_seed*/**/eval_results.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        relative = result_path.relative_to(root).parts
        dataset, kir_seed, gate = relative[1], relative[2], relative[3]
        decomposition = metrics["cascade_error_breakdown"]
        known = decomposition["known"]
        oos = decomposition["oos"]
        rows.append(
            {
                "dataset": dataset,
                "kir_seed": kir_seed,
                "gate": gate,
                "oos_f1": metrics["oos_f1"],
                "known_macro_f1": metrics["known_macro_f1"],
                "overall_accuracy": metrics["overall_accuracy"],
                "id_recall": metrics["gate_id_recall"],
                "oos_false_accept_rate": oos["gate_false_accept_rate"],
                "known_false_reject_rate": known["gate_false_reject_rate"],
                "router_error_rate": known.get("router_error_rate_given_gate_pass", 0.0),
                "expert_error_rate": known.get("expert_error_rate_given_router_correct", 0.0),
                "result_path": str(result_path),
                "result_sha256": _sha256(result_path),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--protocol",
        default="representative_cascade_repair_fixed_downstream",
        help="写入 provenance 的协议名，不影响汇总逻辑。",
    )
    args = parser.parse_args()
    root = args.input_root.resolve()
    output_dir = (args.output_dir or root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _metrics_rows(root)
    if not rows:
        raise SystemExit(f"no eval_results.json found under {root / 'evaluations'}")

    fieldnames = list(rows[0].keys())
    summary_path = output_dir / "cascade_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    decomposition_path = output_dir / "cascade_error_decomposition.csv"
    decomposition_fields = [
        "dataset",
        "kir_seed",
        "gate",
        "stage",
        "count",
        "rate",
    ]
    with decomposition_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=decomposition_fields)
        writer.writeheader()
        for row in rows:
            result = json.loads(Path(row["result_path"]).read_text(encoding="utf-8"))
            for stage, count in result["cascade_error_decomposition_sample_level"]["counts"].items():
                writer.writerow(
                    {
                        "dataset": row["dataset"],
                        "kir_seed": row["kir_seed"],
                        "gate": row["gate"],
                        "stage": stage,
                        "count": count,
                        "rate": result["cascade_error_decomposition_sample_level"]["rates"][stage],
                    }
                )

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=Path(__file__).resolve().parents[2], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    provenance = {
        "protocol": str(args.protocol),
        "input_root": str(root),
        "unit_count": len(rows),
        "datasets": sorted({row["dataset"] for row in rows}),
        "gates": sorted({row["gate"] for row in rows}),
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "python": sys.executable,
        "summary_path": str(summary_path),
        "decomposition_path": str(decomposition_path),
        "used_test_for_selection": False,
        "used_oos_for_expert_training": False,
    }
    (output_dir / "repair_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
