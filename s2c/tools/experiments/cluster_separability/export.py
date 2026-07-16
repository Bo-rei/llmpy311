#!/usr/bin/env python3
"""将已有多簇可分性产物汇总为论文表格与完整性审计。

该 exporter 对原始实验结果严格只读：它只从已存在的 ``eval_results.json``
和 ``run_manifest.json`` 复制指标，并报告 canonical 网格中缺失的单元。它不用
其他 seed 补缺失值，不在空单元上做均值，也不从文件路径推测一个未记录指标。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
KIRS = (25, 50, 75)
K_VALUES = (1, 2, 3, 4, 5)
DISTANCES = ("euclidean", "mahalanobis_diag")
DATA_SEEDS = (13, 42, 87)
GRID_PHASES = ("fixed", "tuned")

MATRIX_FIELDS = (
    "phase",
    "dataset",
    "kir",
    "data_seed",
    "distance",
    "k_gate",
    "method",
    "config_hash",
    "guard_violation",
    "eval_results",
    "run_manifest",
    "validation_oos_f1",
    "validation_id_recall",
    "validation_fpr95",
    "test_oos_precision",
    "test_oos_recall",
    "test_oos_f1",
    "test_id_recall",
    "test_oos_rejection",
    "test_auroc",
    "test_aupr_oos",
    "test_fpr95",
    "run_elapsed_seconds",
    "test_scoring_seconds",
    "test_samples_per_second",
    "process_peak_rss_mb",
)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _value(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping.get(key)
    return "" if value is None else value


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def _unit_row(root: Path, eval_path: Path) -> dict[str, Any] | None:
    """将一个实验单元展平为一行；产物损坏时返回 None 交给缺失审计。"""

    unit_dir = eval_path.parent
    manifest_path = unit_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        evaluation = _load_json(eval_path)
        manifest = _load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    config_value = manifest.get("config", {})
    config = config_value if isinstance(config_value, Mapping) else {}
    validation_value = evaluation.get("validation", {})
    validation = validation_value if isinstance(validation_value, Mapping) else {}
    test_value = evaluation.get("test", {})
    test = test_value if isinstance(test_value, Mapping) else {}
    threshold_path = unit_dir / "threshold_selection.json"
    threshold = _load_json(threshold_path) if threshold_path.is_file() else {}
    timing_path = unit_dir / "timing.json"
    timing = _load_json(timing_path) if timing_path.is_file() else {}
    selected_boundary = threshold.get("selected", threshold.get("selected_operating_point", {}))

    relative_parts = unit_dir.relative_to(root).parts
    # 早期 baseline manifest 未必存储 phase，因此允许从规范目录的第一层推断。
    # 显式 config/evaluation 字段始终优先，避免移动目录后改写实验语义。
    inferred_phase = relative_parts[0] if relative_parts else ""
    phase = str(config.get("phase", evaluation.get("phase", inferred_phase)))
    row = {
        "phase": phase,
        "dataset": _value(config, "dataset"),
        "kir": _value(config, "kir"),
        "data_seed": _value(config, "data_seed"),
        "distance": _value(config, "distance"),
        "k_gate": _value(config, "k_gate"),
        "method": _value(config, "method"),
        "config_hash": _value(manifest, "config_hash"),
        "guard_violation": _value(selected_boundary, "guard_violation")
        if isinstance(selected_boundary, Mapping) else "",
        "eval_results": str(eval_path.relative_to(root)),
        "run_manifest": str(manifest_path.relative_to(root)),
        "run_elapsed_seconds": _value(manifest, "elapsed_seconds"),
        "test_scoring_seconds": _value(timing, "test_scoring_seconds"),
        "test_samples_per_second": _value(timing, "test_samples_per_second"),
        "process_peak_rss_mb": _value(timing, "process_peak_rss_mb"),
    }
    for key in ("oos_f1", "id_recall", "fpr95"):
        row[f"validation_{key}"] = _value(validation, key)
    for key in (
        "oos_precision",
        "oos_recall",
        "oos_f1",
        "id_recall",
        "oos_rejection",
        "auroc",
        "aupr_oos",
        "fpr95",
    ):
        row[f"test_{key}"] = _value(test, key)
    return row


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows = [row for path in sorted(root.rglob("eval_results.json")) if (row := _unit_row(root, path))]
    return sorted(
        rows,
        key=lambda row: (
            str(row["phase"]),
            str(row["dataset"]),
            int(row["kir"] or -1),
            int(row["data_seed"] or -1),
            str(row["distance"]),
            int(row["k_gate"] or -1),
            str(row["method"]),
        ),
    )


def _grid_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, str, int] | None:
    try:
        return (
            str(row["phase"]),
            str(row["dataset"]),
            int(row["kir"]),
            int(row["data_seed"]),
            str(row["distance"]),
            int(row["k_gate"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def build_missing_cells(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """对 fixed/tuned 各 270 个 canonical 单元做精确集合差审计。"""

    present = {key for row in rows if (key := _grid_key(row)) is not None}
    missing: dict[str, list[dict[str, Any]]] = {}
    for phase in GRID_PHASES:
        cells = []
        for dataset, kir, k_gate, distance, seed in product(
            DATASETS, KIRS, K_VALUES, DISTANCES, DATA_SEEDS
        ):
            key = (phase, dataset, kir, seed, distance, k_gate)
            if key not in present:
                cells.append(
                    {
                        "dataset": dataset,
                        "kir": kir,
                        "data_seed": seed,
                        "distance": distance,
                        "k_gate": k_gate,
                    }
                )
        missing[phase] = cells
    expected_per_phase = len(DATASETS) * len(KIRS) * len(K_VALUES) * len(DISTANCES) * len(DATA_SEEDS)
    return {
        "expected_per_phase": expected_per_phase,
        "present": {phase: expected_per_phase - len(missing[phase]) for phase in GRID_PHASES},
        "missing_count": {phase: len(missing[phase]) for phase in GRID_PHASES},
        "missing": missing,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_figure_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
        and item.suffix.lower() in {".png", ".pdf", ".svg"}
        # TextOIR attempt 会复制 upstream 自带的示例图；它们不是本研究生成的
        # paper figure，且外部协议必须与 s2c 主表隔离，因此不进入本 manifest。
        and "textoir_protocol" not in item.relative_to(root).parts
    ):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "format": path.suffix.lower().lstrip("."),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def build_data_protocol_audit(
    root: Path,
    rows: Iterable[Mapping[str, Any]],
    missing: Mapping[str, Any],
) -> dict[str, Any]:
    """生成跨产物的实验协议索引，不复制 TextOIR 的大体量 split 审计。

    这里记录的是论文汇总时最容易被误用的约束：OOS 正类与分数方向、K/阈值
    的选择数据，以及 TextOIR 与 s2c 不可直接混表。TextOIR 自身的标签列表和
    split hash 仍保留在独立审计文件中，本索引只保存相对路径和文件哈希。
    """

    source = list(rows)
    baseline_methods = sorted(
        {
            str(row["method"])
            for row in source
            if row.get("phase") == "baselines" and row.get("method")
        }
    )
    textoir_audit = root / "textoir_protocol" / "data_protocol_audit.json"
    textoir_reference: dict[str, Any] = {
        "separate_protocol": True,
        "direct_numeric_merge_with_s2c_forbidden": True,
        "audit_path": None,
        "audit_sha256": None,
    }
    if textoir_audit.is_file():
        textoir_reference.update(
            {
                "audit_path": str(textoir_audit.relative_to(root)),
                "audit_sha256": _sha256(textoir_audit),
            }
        )

    return {
        "schema_version": 1,
        "experiment_family": "cluster_separability_v19",
        "s2c_gate_protocol": {
            "oos_is_positive_class": True,
            "oos_score_direction": "higher_is_more_oos",
            "selection_split": "validation",
            "test_used_for_k_or_threshold_selection": False,
            "covariance_scope": "per_cluster",
            "covariance_eps": 1e-6,
            "gate_only_results_are_not_end_to_end_intent_metrics": True,
            "controlled_baseline_methods": baseline_methods,
            "canonical_grid": {
                "expected_per_phase": missing.get("expected_per_phase"),
                "present": missing.get("present", {}),
                "missing_count": missing.get("missing_count", {}),
            },
        },
        "textoir_protocol": textoir_reference,
    }


def _mean_std_rows(rows: Iterable[Mapping[str, Any]], phase: str) -> list[dict[str, Any]]:
    """仅在相同 dataset/KIR/distance/K 内跨 data seed 汇总 mean±std。"""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("phase") == phase
            and row.get("k_gate") != ""
            and int(row["k_gate"]) in K_VALUES
        ):
            key = (row["dataset"], int(row["kir"]), row["distance"], int(row["k_gate"]))
            grouped[key].append(row)
    output = []
    metrics = ("test_oos_f1", "test_id_recall", "test_auroc", "test_aupr_oos", "test_fpr95")
    for (dataset, kir, distance, k_gate), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "phase": phase, "dataset": dataset, "kir": kir,
            "distance": distance, "k_gate": k_gate, "seed_count": len(items),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) != ""]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else ""
        output.append(result)
    return output


def _selected_k_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按 dataset/KIR/seed/distance 在 tuned validation 结果中选一个 dataset-level K。

    先排除违反 ID-recall guard 的 K；仅当所有 K 都违反时才在全部候选中选择。
    并列顺序与 Runner 一致：OOS F1、FPR95、ID recall，最后优先较小 K。
    任何 test 指标都不参与 selected K。
    """

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") == "tuned" and row.get("k_gate") != "":
            grouped[(row["dataset"], int(row["kir"]), int(row["data_seed"]), row["distance"])].append(row)
    selected = []
    for key, items in sorted(grouped.items()):
        eligible = [item for item in items if item.get("guard_violation") is not True]
        pool = eligible or items
        winner = max(
            pool,
            key=lambda item: (
                float(item["validation_oos_f1"]),
                -float(item["validation_fpr95"]),
                float(item["validation_id_recall"]),
                -int(item["k_gate"]),
            ),
        )
        selected.append(
            {
                "dataset": key[0], "kir": key[1], "data_seed": key[2], "distance": key[3],
                "selected_k": int(winner["k_gate"]),
                "validation_oos_f1": winner["validation_oos_f1"],
                "validation_id_recall": winner["validation_id_recall"],
                "validation_fpr95": winner["validation_fpr95"],
                "guard_violation": winner.get("guard_violation", ""),
            }
        )
    return selected


def _gate_baseline_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """构造同协议的 Gate-only 逐 seed 对比表。

    置信度与邻域方法来自 ``baselines`` phase。几何对照统一读取 tuned phase：
    K=1 分别表示 Euclidean/局部对角 Mahalanobis 单中心；历史 K=2 和
    validation-selected K 使用 Mahalanobis。只有存在受控 Baseline 时才生成该表，
    避免把一个仅含几何行的残缺文件误当成完整对比实验。
    """

    source = [dict(row) for row in rows]
    controlled = [dict(row) for row in source if row.get("phase") == "baselines"]
    if not controlled:
        return []

    tuned = [row for row in source if row.get("phase") == "tuned"]
    by_key = {
        (
            row.get("dataset"),
            int(row["kir"]),
            int(row["data_seed"]),
            row.get("distance"),
            int(row["k_gate"]),
        ): row
        for row in tuned
        if row.get("kir") != "" and row.get("data_seed") != "" and row.get("k_gate") != ""
    }
    geometric: list[dict[str, Any]] = []
    groups = sorted(
        {
            (row["dataset"], int(row["kir"]), int(row["data_seed"]))
            for row in controlled
        }
    )
    fixed_specs = (
        ("euclidean", 1, "euclidean_centroid"),
        ("mahalanobis_diag", 1, "diag_mahalanobis_centroid"),
        ("mahalanobis_diag", 2, "multisphere_k2"),
    )
    for dataset, kir, seed in groups:
        for distance, k_gate, method in fixed_specs:
            candidate = by_key.get((dataset, kir, seed, distance, k_gate))
            if candidate is not None:
                copied = dict(candidate)
                copied["method"] = method
                geometric.append(copied)

    # selected K 严格复用 exporter 的 validation-only 选择规则；test 指标不参与。
    selected = {
        (row["dataset"], int(row["kir"]), int(row["data_seed"]), row["distance"]): int(row["selected_k"])
        for row in _selected_k_rows(tuned)
        if row["distance"] == "mahalanobis_diag"
    }
    for dataset, kir, seed in groups:
        k_gate = selected.get((dataset, kir, seed, "mahalanobis_diag"))
        if k_gate is None:
            continue
        candidate = by_key.get((dataset, kir, seed, "mahalanobis_diag", k_gate))
        if candidate is not None:
            copied = dict(candidate)
            copied["method"] = "multisphere_selected_k"
            geometric.append(copied)

    return sorted(
        [*controlled, *geometric],
        key=lambda row: (
            str(row["dataset"]),
            int(row["kir"]),
            int(row["data_seed"]),
            str(row["method"]),
        ),
    )


def _gate_baseline_mean_std(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """在相同 dataset/KIR/method 内跨 data seed 汇总 mean 和 sample std。"""

    metrics = (
        "validation_oos_f1",
        "validation_id_recall",
        "validation_fpr95",
        "test_oos_precision",
        "test_oos_recall",
        "test_oos_f1",
        "test_id_recall",
        "test_oos_rejection",
        "test_auroc",
        "test_aupr_oos",
        "test_fpr95",
        "run_elapsed_seconds",
        "test_scoring_seconds",
        "test_samples_per_second",
        "process_peak_rss_mb",
    )
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["kir"]), str(row["method"]))].append(row)

    output: list[dict[str, Any]] = []
    for (dataset, kir, method), items in sorted(grouped.items()):
        result: dict[str, Any] = {
            "dataset": dataset,
            "kir": kir,
            "method": method,
            "seed_count": len(items),
            "data_seeds": "|".join(str(value) for value in sorted({int(row["data_seed"]) for row in items})),
            "distance": "|".join(sorted({str(row["distance"]) for row in items if row.get("distance") != ""})),
            "k_gate_values": "|".join(sorted({str(row["k_gate"]) for row in items if row.get("k_gate") != ""}, key=int)),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in items if row.get(metric) != ""]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else ""
        output.append(result)
    return output


def export_artifacts(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """导出原始矩阵、跨种子汇总、selected-K 频次和图像 hash manifest。"""

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"experiment root does not exist: {root}")
    output_dir = (output_dir or root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(root)
    _write_csv(output_dir / "experiment_matrix.csv", rows, MATRIX_FIELDS)

    missing = build_missing_cells(rows)
    (output_dir / "missing_cells.json").write_text(
        json.dumps(missing, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )

    protocol_audit = build_data_protocol_audit(root, rows, missing)
    (output_dir / "data_protocol_audit.json").write_text(
        json.dumps(protocol_audit, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    produced = ["experiment_matrix.csv", "missing_cells.json", "data_protocol_audit.json"]
    phase_outputs = {
        "fixed": "kir_k_fixed_boundary.csv",
        "tuned": "kir_k_tuned_boundary.csv",
    }
    for phase, filename in phase_outputs.items():
        # K=8 是单独的过度碎片化 stress control，不混入 canonical K=1..5 主表。
        selected = [
            row
            for row in rows
            if row["phase"] == phase
            and row.get("k_gate") != ""
            and int(row["k_gate"]) in K_VALUES
        ]
        if selected:
            _write_csv(output_dir / filename, selected, MATRIX_FIELDS)
            produced.append(filename)

    baseline_rows = _gate_baseline_rows(rows)
    if baseline_rows:
        _write_csv(output_dir / "gate_baseline_by_seed.csv", baseline_rows, MATRIX_FIELDS)
        baseline_summary = _gate_baseline_mean_std(baseline_rows)
        _write_csv(
            output_dir / "gate_baseline_summary.csv",
            baseline_summary,
            baseline_summary[0].keys(),
        )
        produced.extend(("gate_baseline_by_seed.csv", "gate_baseline_summary.csv"))

    stress_rows = [
        row
        for row in rows
        if row.get("phase") == "fixed" and row.get("k_gate") != "" and int(row["k_gate"]) == 8
    ]
    if stress_rows:
        _write_csv(output_dir / "k8_stress_control.csv", stress_rows, MATRIX_FIELDS)
        produced.append("k8_stress_control.csv")

    for phase in ("fixed", "tuned"):
        summaries = _mean_std_rows(rows, phase)
        if summaries:
            filename = f"kir_k_{phase}_mean_std.csv"
            _write_csv(output_dir / filename, summaries, summaries[0].keys())
            produced.append(filename)

    selected_k = _selected_k_rows(rows)
    if selected_k:
        _write_csv(output_dir / "selected_k_summary.csv", selected_k, selected_k[0].keys())
        frequencies = Counter(
            (row["dataset"], row["kir"], row["distance"], row["selected_k"])
            for row in selected_k
        )
        frequency_rows = [
            {"dataset": key[0], "kir": key[1], "distance": key[2], "selected_k": key[3], "seed_frequency": count}
            for key, count in sorted(frequencies.items())
        ]
        _write_csv(output_dir / "selected_k_frequency.csv", frequency_rows, frequency_rows[0].keys())
        produced.extend(("selected_k_summary.csv", "selected_k_frequency.csv"))

    figures = build_figure_manifest(root)
    (output_dir / "figure_manifest.json").write_text(
        json.dumps({"figures": figures}, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    produced.append("figure_manifest.json")
    return {"source_rows": len(rows), "produced": produced, "figures": len(figures)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Canonical cluster-separability experiment root")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(export_artifacts(args.root, args.output_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
