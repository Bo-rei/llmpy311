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
import statistics
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


def _seed_from_kir_seed(value: str) -> int:
    """从 ``kir50_seed42`` 提取 data seed，避免汇总表依赖路径位置。"""

    marker = "_seed"
    if marker not in value:
        raise ValueError(f"cannot parse seed from kir_seed={value!r}")
    return int(value.rsplit(marker, 1)[1])


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """统一写出小型分析表；空表也保留表头，方便下游审计。"""

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(rows: list[dict[str, Any]], field: str) -> tuple[float, float]:
    values = [float(row[field]) for row in rows]
    if not values:
        raise ValueError(f"cannot summarize empty rows for {field}")
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成同 seed 对齐的 Gate 比较表。

    参照始终是同一 dataset/seed 的 ``frozen_k1``。这一步只做配对差值，
    不回看测试集选择任何 Gate、K 或阈值。
    """

    references = {
        (row["dataset"], row["kir_seed"]): row
        for row in rows
        if row["gate"] == "frozen_k1"
    }
    metric_names = [
        "oos_f1",
        "known_macro_f1",
        "overall_accuracy",
        "id_recall",
        "oos_false_accept_rate",
        "known_false_reject_rate",
        "router_error_rate",
        "expert_error_rate",
    ]
    output: list[dict[str, Any]] = []
    for row in rows:
        reference = references.get((row["dataset"], row["kir_seed"]))
        if reference is None:
            raise ValueError(
                f"missing frozen_k1 reference for {row['dataset']}/{row['kir_seed']}"
            )
        paired = {
            "dataset": row["dataset"],
            "kir_seed": row["kir_seed"],
            "seed": _seed_from_kir_seed(row["kir_seed"]),
            "gate": row["gate"],
            **{name: row[name] for name in metric_names},
        }
        for name in metric_names:
            paired[f"delta_vs_frozen_k1_{name}"] = float(row[name]) - float(reference[name])
        output.append(paired)
    return output


def _summary_rows(paired_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 dataset/Gate 汇总均值、标准差和 seed-level 配对差。"""

    metric_names = [
        "oos_f1",
        "known_macro_f1",
        "overall_accuracy",
        "id_recall",
        "oos_false_accept_rate",
        "known_false_reject_rate",
        "router_error_rate",
        "expert_error_rate",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in paired_rows:
        grouped.setdefault((str(row["dataset"]), str(row["gate"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (dataset, gate), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "dataset": dataset,
            "gate": gate,
            "seed_count": len(group),
        }
        for name in metric_names:
            mean, std = _mean_std(group, name)
            delta_mean, delta_std = _mean_std(group, f"delta_vs_frozen_k1_{name}")
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
            row[f"delta_vs_frozen_k1_{name}_mean"] = delta_mean
            row[f"delta_vs_frozen_k1_{name}_std"] = delta_std
        output.append(row)
    return output


def _error_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 dataset/Gate 聚合六阶段错误分解，不混合不同样本规模。"""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        result = json.loads(Path(row["result_path"]).read_text(encoding="utf-8"))
        sample = result["cascade_error_decomposition_sample_level"]
        for stage, count in sample["counts"].items():
            item = {
                "dataset": row["dataset"],
                "gate": row["gate"],
                "stage": stage,
                "count": int(count),
                "rate": float(sample["rates"][stage]),
            }
            grouped.setdefault((item["dataset"], item["gate"], stage), []).append(item)
    output: list[dict[str, Any]] = []
    for (dataset, gate, stage), group in sorted(grouped.items()):
        count_mean, count_std = _mean_std(group, "count")
        rate_mean, rate_std = _mean_std(group, "rate")
        output.append(
            {
                "dataset": dataset,
                "gate": gate,
                "stage": stage,
                "seed_count": len(group),
                "count_mean": count_mean,
                "count_std": count_std,
                "rate_mean": rate_mean,
                "rate_std": rate_std,
                "count_total": sum(int(item["count"]) for item in group),
            }
        )
    return output


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
    _write_csv(summary_path, rows, fieldnames)

    decomposition_path = output_dir / "cascade_error_decomposition.csv"
    decomposition_fields = [
        "dataset",
        "kir_seed",
        "gate",
        "stage",
        "count",
        "rate",
    ]
    decomposition_rows: list[dict[str, Any]] = []
    for row in rows:
        result = json.loads(Path(row["result_path"]).read_text(encoding="utf-8"))
        sample = result["cascade_error_decomposition_sample_level"]
        for stage, count in sample["counts"].items():
            decomposition_rows.append(
                {
                    "dataset": row["dataset"],
                    "kir_seed": row["kir_seed"],
                    "gate": row["gate"],
                    "stage": stage,
                    "count": count,
                    "rate": sample["rates"][stage],
                }
            )
    _write_csv(decomposition_path, decomposition_rows, decomposition_fields)

    paired_rows = _paired_rows(rows)
    paired_fields = list(paired_rows[0].keys())
    paired_path = output_dir / "cascade_gate_by_seed.csv"
    _write_csv(paired_path, paired_rows, paired_fields)

    summary_rows = _summary_rows(paired_rows)
    summary_fields = list(summary_rows[0].keys())
    summary_by_gate_path = output_dir / "cascade_gate_summary.csv"
    _write_csv(summary_by_gate_path, summary_rows, summary_fields)

    error_summary_rows = _error_summary_rows(rows)
    error_summary_path = output_dir / "cascade_error_decomposition_summary.csv"
    _write_csv(error_summary_path, error_summary_rows, list(error_summary_rows[0].keys()))

    # Banking77 的 CE-Recon 取舍需要单独呈现：OOS 拒识率上升通常伴随
    # Known false reject 上升。把配对差值落成小表，避免论文解释依赖手工抄数。
    banking_tradeoff = []
    for row in paired_rows:
        if row["dataset"] != "banking77_oos":
            continue
        banking_tradeoff.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "gate": row["gate"],
                "oos_f1": row["oos_f1"],
                "known_macro_f1": row["known_macro_f1"],
                "id_recall": row["id_recall"],
                "oos_false_accept_rate": row["oos_false_accept_rate"],
                "known_false_reject_rate": row["known_false_reject_rate"],
                "delta_oos_f1": row["delta_vs_frozen_k1_oos_f1"],
                "delta_known_macro_f1": row["delta_vs_frozen_k1_known_macro_f1"],
                "delta_id_recall": row["delta_vs_frozen_k1_id_recall"],
                "delta_oos_false_accept_rate": row[
                    "delta_vs_frozen_k1_oos_false_accept_rate"
                ],
                "delta_known_false_reject_rate": row[
                    "delta_vs_frozen_k1_known_false_reject_rate"
                ],
                "interpretation": (
                    "OOS rejection improves while ID recall/known macro-F1 trade off"
                    if row["gate"] == "ce_recon_selected_k"
                    else "paired reference or controlled comparison"
                ),
            }
        )
    banking_path = output_dir / "cascade_banking_tradeoff.csv"
    _write_csv(banking_path, banking_tradeoff, list(banking_tradeoff[0].keys()))

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
        "paired_summary_path": str(paired_path),
        "gate_summary_path": str(summary_by_gate_path),
        "error_summary_path": str(error_summary_path),
        "banking_tradeoff_path": str(banking_path),
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
