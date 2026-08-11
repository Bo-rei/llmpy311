"""Create the unified S2C comparison index and mechanism evidence bundle.

The builder is post-hoc only.  It reuses completed five-seed fair summaries,
existing paired inference, and existing mechanism figures.  The only new plots
are compact entry-point views generated from those CSVs; no training or test
threshold selection happens here.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from unified_prediction_contract_v1 import (
    PROTOCOL_VERSION,
    build_contract,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "unified_comparison_v1"
CONTRACT_OUT = ROOT / "results" / "analysis" / "unified_prediction_contract_v1"
LOCAL_CONTRACT_OUT = ROOT.parent / "artifacts" / "s2c" / "analysis" / "unified_prediction_contract_v1"
FIGURES = ROOT / "figures" / "unified_comparison_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | None:
    if value in (None, "", "NA", "nan", "NaN"):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def text_number(value: Any) -> str:
    value = number(value)
    return "" if value is None else f"{value:.10g}"


def method_backbone(method: str) -> str:
    if method == "trainable_k1" or method.startswith("Native-"):
        return "MiniLM-trainable"
    if method.startswith("ADB") or method.startswith("DA-ADB"):
        return "BERT"
    if method.startswith("DCLOOS"):
        return "BERT"
    return "MiniLM-frozen"


def fair_summary_rows() -> list[dict[str, Any]]:
    path = ROOT / "results" / "analysis" / "cross_protocol_tradeoff_v1" / "summary_mean_std.csv"
    rows = []
    for source in read_csv(path):
        method = source.get("method", "")
        for metric in (
            "oos_f1",
            "f1_all",
            "f1_k",
            "accuracy",
            "known_recall",
            "false_accept_rate",
            "false_reject_rate",
            "auroc",
            "aupr_oos",
        ):
            source.setdefault(f"{metric}_mean", source.get(metric, ""))
        rows.append(
            {
                **source,
                "method_id": method,
                "method_label": source.get("method_label", method),
                "contract_layer": "same_protocol_fair",
                "comparison_scope": "same-protocol fair comparison",
                "backbone": method_backbone(method),
                "supervision_type": "Known-only",
                "split_scope": "protocol_v2_textoir_v1 / 3 datasets / 3 KIR / 5 seeds",
                "status": "complete_final_metrics",
                "test_used_for_selection": "False",
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def _aggregate(rows: list[dict[str, str]], group_fields: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        result: dict[str, Any] = dict(zip(group_fields, key))
        result["n_seeds"] = len(members)
        for metric in metrics:
            values = [number(member.get(metric)) for member in members]
            values = [value for value in values if value is not None]
            result[f"{metric}_mean"] = sum(values) / len(values) if values else None
            if len(values) > 1:
                mean = result[f"{metric}_mean"]
                result[f"{metric}_std"] = (
                    sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                ) ** 0.5
            else:
                result[f"{metric}_std"] = None
        output.append(result)
    return output


def native_summary_rows() -> list[dict[str, Any]]:
    path = ROOT / "results" / "analysis" / "native_baselines_trainable_v1" / "trainable_native_per_seed.csv"
    metrics = [
        "oos_f1",
        "f1_all",
        "known_macro_f1",
        "known_recall",
        "false_accept_rate",
        "false_reject_rate",
        "accuracy",
        "auroc",
        "aupr_oos",
    ]
    labels = {
        "msp": "Native-MSP-Trainable-MiniLM",
        "energy": "Native-Energy-Trainable-MiniLM",
        "knn": "Native-kNN-Trainable-MiniLM",
        "lof": "Native-LOF-Trainable-MiniLM",
    }
    rows = []
    for source in _aggregate(read_csv(path), ["dataset", "kir", "method"], metrics):
        method = source["method"]
        source.update(
            {
                "method_id": f"native_{method}",
                "method_label": labels.get(method, method),
                "contract_layer": "same_protocol_native_backbone_control",
                "comparison_scope": "same representation detector control",
                "backbone": "MiniLM-trainable",
                "supervision_type": "Known-only",
                "split_scope": "protocol_v2_textoir_v1 / 3 seeds / native detector",
                "status": "complete_final_metrics",
                "test_used_for_selection": "False",
                "source": str(path.relative_to(ROOT)),
                "f1_k_mean": source.get("known_macro_f1_mean"),
                "f1_k_std": source.get("known_macro_f1_std"),
            }
        )
        rows.append(source)
    return rows


def adb_summary_rows() -> list[dict[str, Any]]:
    path = ROOT / "results" / "analysis" / "adb_kir_sensitivity_v1" / "adb_summary.csv"
    rows = []
    for source in read_csv(path):
        rows.append(
            {
                **source,
                "method_id": "ADB-external-BERT",
                "method_label": "ADB external BERT",
                "contract_layer": "external_backbone",
                "comparison_scope": "external backbone reference; not fair-pooled",
                "backbone": "BERT",
                "supervision_type": "Known-only-external-runtime",
                "split_scope": "protocol_v2 split / 3 seeds / TextOIR BERT compatibility",
                "status": "complete_external_final_metrics",
                "test_used_for_selection": "not_declared_external",
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def da_adb_summary_rows() -> list[dict[str, Any]]:
    path = ROOT / "results" / "analysis" / "da_adb_gpu_runtime_v1" / "metrics.csv"
    rows = []
    for source in read_csv(path):
        rows.append(
            {
                "dataset": source.get("dataset"),
                "kir": source.get("kir"),
                "method_id": "DA-ADB-external-BERT",
                "method_label": "DA-ADB external BERT",
                "contract_layer": "external_backbone",
                "comparison_scope": "external single-cell reference; not fair-pooled",
                "backbone": "BERT",
                "supervision_type": "Known-only-external-runtime",
                "n_seeds": 1,
                "oos_f1_mean": source.get("oos_f1"),
                "f1_all_mean": source.get("f1_all"),
                "f1_k_mean": source.get("f1_known"),
                "accuracy_mean": source.get("accuracy"),
                "known_recall_mean": source.get("known_recall"),
                "false_accept_rate_mean": source.get("false_acceptance_rate"),
                "false_reject_rate_mean": source.get("false_rejection_rate"),
                "seed_scope": source.get("seed"),
                "split_scope": "protocol_v2 split / one isolated external cell",
                "status": "complete_external_single_cell",
                "test_used_for_selection": "not_declared_external",
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def dcloos_reference_rows() -> list[dict[str, Any]]:
    path = ROOT / "results" / "final_baselines" / "summary.csv"
    rows = []
    for source in read_csv(path):
        if source.get("method") != "DCLOOS-official (reduced-budget recovered)":
            continue
        rows.append(
            {
                "dataset": "oos+squad",
                "kir": source.get("kir"),
                "method_id": "DCLOOS-reduced",
                "method_label": "DCLOOS reduced reference",
                "contract_layer": "different_supervision",
                "comparison_scope": "different supervision reference; no fair ranking",
                "backbone": "BERT",
                "supervision_type": "pseudo-OOS plus external-OOS",
                "n_seeds": 1,
                "oos_f1_mean": source.get("oos_f1"),
                "f1_all_mean": source.get("f1_all"),
                "f1_k_mean": source.get("known_macro_f1"),
                "accuracy_mean": source.get("accuracy"),
                "known_recall_mean": source.get("known_recall"),
                "false_accept_rate_mean": None,
                "false_reject_rate_mean": None,
                "split_scope": "official external-OOS reduced single cell / seed 888",
                "status": "complete_recovered_intermediate_prediction",
                "test_used_for_selection": "different_supervision; validation-best intermediate",
                "source": str(path.relative_to(ROOT)),
            }
        )
    return rows


def build_method_summary() -> list[dict[str, Any]]:
    rows = fair_summary_rows() + native_summary_rows() + adb_summary_rows() + da_adb_summary_rows() + dcloos_reference_rows()
    preferred_fields = [
        "method_id",
        "method_label",
        "contract_layer",
        "comparison_scope",
        "backbone",
        "supervision_type",
        "dataset",
        "kir",
        "n_seeds",
        "oos_f1_mean",
        "oos_f1_std",
        "f1_all_mean",
        "f1_all_std",
        "f1_k_mean",
        "f1_k_std",
        "accuracy_mean",
        "accuracy_std",
        "known_recall_mean",
        "known_recall_std",
        "false_accept_rate_mean",
        "false_accept_rate_std",
        "false_reject_rate_mean",
        "false_reject_rate_std",
        "auroc_mean",
        "auroc_std",
        "aupr_oos_mean",
        "aupr_oos_std",
        "split_scope",
        "status",
        "test_used_for_selection",
        "source",
    ]
    for row in rows:
        for field in preferred_fields:
            row.setdefault(field, "")
    return sorted(rows, key=lambda row: (str(row.get("contract_layer")), str(row.get("dataset")), float(row.get("kir") or 0), str(row.get("method_label"))))


def paired_statistics() -> list[dict[str, Any]]:
    source = ROOT / "results" / "analysis" / "statistical_stability_v1" / "paired_effects.csv"
    manifest_path = ROOT / "results" / "analysis" / "statistical_stability_v1" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    rows = []
    for row in read_csv(source):
        rows.append(
            {
                **row,
                "protocol_version": PROTOCOL_VERSION,
                "comparison_contract": "same_protocol_fair",
                "paired_unit": "dataset × KIR × seed",
                "bootstrap_replicates": manifest.get("bootstrap_repetitions", 10000),
                "bootstrap_seed": manifest.get("bootstrap_seed", 20260725),
                "source": str(source.relative_to(ROOT)),
                "test_usage": "post-hoc only; no selection or tuning",
            }
        )
    return rows


def contract_summary(groups: list[Mapping[str, Any]], alignment: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for group in groups:
        grouped[
            (
                str(group["method"]),
                str(group["contract_layer"]),
                str(group["backbone"]),
                str(group["supervision_type"]),
                str(group["selection_audit"]),
            )
        ].append(group)
    aligned_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in alignment:
        aligned_by_method[str(row["method"])].append(row)
    result = []
    for key, members in sorted(grouped.items()):
        method, layer, backbone, supervision, selection_audit = key
        result.append(
            {
                "method": method,
                "contract_layer": layer,
                "backbone": backbone,
                "supervision_type": supervision,
                "selection_audit": selection_audit,
                "n_runs": len(members),
                "n_rows": sum(int(member["row_count"]) for member in members),
                "n_cells": len({(member["dataset"], member["kir"], member["seed"]) for member in members}),
                "aligned_cells": sum(row["status"] == "aligned" for row in aligned_by_method[method]),
                "non_aligned_cells": sum(row["status"] != "aligned" for row in aligned_by_method[method]),
                "score_available": all(bool(member["score_available"]) for member in members),
                "source_paths": " | ".join(sorted({str(member["source_path"]) for member in members})),
            }
        )
    return result


def _plot_fair_heatmap(summary: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    rows = [row for row in summary if row.get("contract_layer") == "same_protocol_fair"]
    methods = sorted({str(row.get("method_label")) for row in rows})
    cells = sorted({f"{row.get('dataset')}\nKIR={float(row.get('kir')):.2f}" for row in rows})
    values = np.full((len(methods), len(cells)), np.nan)
    for row in rows:
        cell = f"{row.get('dataset')}\nKIR={float(row.get('kir')):.2f}"
        value = number(row.get("oos_f1_mean"))
        if value is not None:
            values[methods.index(str(row.get("method_label"))), cells.index(cell)] = value
    fig, ax = plt.subplots(figsize=(14, max(5, len(methods) * 0.55)))
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(cells)), cells, rotation=45, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_title("Same-protocol fair OOS F1: method × dataset × KIR")
    fig.colorbar(image, ax=ax, label="OOS F1")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_fair_pareto(summary: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    rows = [row for row in summary if row.get("contract_layer") == "same_protocol_fair" and float(row.get("kir")) == 0.5]
    datasets = sorted({str(row.get("dataset")) for row in rows})
    fig, axes = plt.subplots(1, max(1, len(datasets)), figsize=(5 * max(1, len(datasets)), 4.5), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        subset = [row for row in rows if row.get("dataset") == dataset]
        for row in subset:
            x = number(row.get("known_recall_mean"))
            y = number(row.get("oos_f1_mean"))
            if x is None or y is None:
                continue
            axis.scatter(x, y, s=38)
            axis.annotate(str(row["method_label"]).replace(" ", "\n"), (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        axis.set_title(dataset)
        axis.set_xlabel("Known Recall")
        axis.set_ylabel("OOS F1")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    fig.suptitle("Same-protocol fair Pareto view at KIR=0.50")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_contract_layers(summary: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    counts: dict[str, int] = defaultdict(int)
    for row in summary:
        counts[str(row["contract_layer"])] += int(row["n_runs"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(counts)
    ax.bar(labels, [counts[label] for label in labels], color=["#4472c4", "#70ad47", "#ed7d31", "#a5a5a5"][: len(labels)])
    ax.set_ylabel("normalized method-run groups")
    ax.set_title("Evidence layers retained by the unified contract")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def build_figures(summary: list[dict[str, Any]], contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    generated = [
        ("unified_fair_oos_f1_heatmap", FIGURES / "fair_oos_f1_heatmap.png", "performance", "same_protocol_fair", ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv", _plot_fair_heatmap, summary),
        ("unified_fair_pareto_kir050", FIGURES / "fair_pareto_kir050.png", "performance", "same_protocol_fair", ROOT / "results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv", _plot_fair_pareto, summary),
        ("unified_contract_layers", FIGURES / "contract_layer_counts.png", "contract", "all", OUT / "contract_summary.csv", _plot_contract_layers, contracts),
    ]
    manifest: list[dict[str, Any]] = []
    for figure_id, path, kind, layer, source, builder, data in generated:
        created = builder(data, path)
        manifest.append(
            {
                "figure_id": figure_id,
                "path": str(path.relative_to(ROOT)),
                "kind": kind,
                "contract_layer": layer,
                "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
                "source_sha256": sha256_file(source) if source.exists() else None,
                "status": "generated" if created else "unavailable_matplotlib",
            }
        )
    reused = [
        ("fair_pareto_existing", "figures/cross_protocol_tradeoff_v1/pareto_oos_f1_known_recall.png", "performance", "same_protocol_fair", "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"),
        ("fair_error_decomposition", "figures/cross_protocol_tradeoff_v1/error_decomposition_kir050.png", "error", "same_protocol_fair", "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"),
        ("fair_kir_curves", "figures/cross_protocol_tradeoff_v1/kir_curves_oos_f1_f1_all.png", "performance", "same_protocol_fair", "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"),
        ("fair_seed_variance", "figures/cross_protocol_tradeoff_v1/seed_variance_oos_f1_kir050.png", "robustness", "same_protocol_fair", "results/analysis/cross_protocol_tradeoff_v1/per_seed.csv"),
        ("mechanism_pack_heatmap", "figures/experimental_mechanism_pack_v3/oos_f1_method_kir_heatmap.png", "performance", "same_protocol_fair", "results/analysis/experimental_mechanism_pack_v3/method_summary.csv"),
        ("mechanism_pack_paired_effects", "figures/experimental_mechanism_pack_v3/trainable_paired_effects_heatmap.png", "paired_statistics", "same_protocol_fair", "results/analysis/experimental_mechanism_pack_v3/method_summary.csv"),
        ("mechanism_closure_dashboard", "figures/mechanism_closure_v1/mechanism_closure_dashboard.png", "mechanism", "same_protocol_fair", "results/analysis/mechanism_closure_v1/MANIFEST.json"),
        ("mechanism_closure_mogb_risk", "figures/mechanism_closure_v1/mogb_risk_workpoints.png", "mechanism", "same_protocol_fair", "results/analysis/mechanism_closure_v1/MANIFEST.json"),
        ("comparison_atlas_layers", "figures/comparison_atlas_v2/current_gate_and_cascade_layers.png", "contract", "layered", "results/analysis/comparison_atlas_v2/MANIFEST.json"),
        ("comparison_atlas_cascade_seed", "figures/comparison_atlas_v2/cascade_bridge_seed_effect.png", "contract", "different_system_level", "results/analysis/comparison_atlas_v2/MANIFEST.json"),
        ("statistical_forest", "figures/statistical_stability_v1/paired_oos_f1_forest.png", "paired_statistics", "same_protocol_fair", "results/analysis/statistical_stability_v1/MANIFEST.json"),
        ("statistical_rank_heatmap", "figures/statistical_stability_v1/method_rank_heatmap_oos_f1.png", "performance", "same_protocol_fair", "results/analysis/statistical_stability_v1/MANIFEST.json"),
        ("representation_geometry", "figures/representation_geometry_visuals_v1/geometry_tradeoff.png", "representation", "mechanism", "results/analysis/representation_geometry_visuals_v1/MANIFEST.json"),
        ("error_quadrants", "figures/representation_geometry_visuals_v1/stackoverflow_error_quadrants.png", "error", "mechanism", "results/analysis/representation_geometry_visuals_v1/MANIFEST.json"),
    ]
    for figure_id, relative, kind, layer, source_relative in reused:
        path = ROOT / relative
        source = ROOT / source_relative
        manifest.append(
            {
                "figure_id": figure_id,
                "path": relative,
                "kind": kind,
                "contract_layer": layer,
                "source": source_relative,
                "source_sha256": sha256_file(source) if source.exists() and source.is_file() else None,
                "status": "reused" if path.exists() else "missing_existing_evidence",
            }
        )
    return manifest


def write_report(summary: list[dict[str, Any]], contracts: list[dict[str, Any]], figures: list[dict[str, Any]], contract_result: Mapping[str, Any]) -> None:
    fair = [row for row in summary if row.get("contract_layer") == "same_protocol_fair"]
    fair_methods = sorted({str(row.get("method_label")) for row in fair})
    aligned = sum(row["status"] == "aligned" for row in contract_result["alignment"])
    non_aligned = len(contract_result["alignment"]) - aligned
    figure_status = defaultdict(int)
    for figure in figures:
        figure_status[figure["status"]] += 1
    lines = [
        "# S2C 统一对比实验与机制报告 V1",
        "",
        "> 本报告是 analysis-only 收口。它统一已有结果的行级合同、统计入口和证据索引；没有新增模型、元学习、自适应 K、椭球边界或复杂聚合，也没有重跑 E2/E3/MOGB/DCLOOS。",
        "",
        "## 1. 结论边界",
        "",
        f"同协议 Known-only fair 层覆盖 `{len(fair)}` 个 dataset × KIR × method 汇总行，方法为：{', '.join(fair_methods)}。它是唯一可以直接做 Trainable-K1 与 Frozen/MOGB 组件的公平主层。",
        f"统一 prediction contract 已写入 `{CONTRACT_OUT.relative_to(ROOT)}`；本地逐样本压缩 JSONL 位于 `{LOCAL_CONTRACT_OUT.relative_to(ROOT.parent)}`，只含匿名 `sample_id`，不含原始文本或 embedding。共 `{contract_result['schema']['row_count']}` 行、`{contract_result['schema']['group_count']}` 个运行组。",
        f"对齐审计中 `{aligned}` 个方法-cell 完全匹配参考样本序列，`{non_aligned}` 个被保留为外部/失败对齐，不会被补成伪 fair row。",
        "",
        "## 2. 性能现象",
        "",
        "`method_summary.csv` 保留 OOS F1、F1-All、F1-K、Accuracy、Known Recall、false acceptance/rejection、AUROC 和 AUPR-OOS 的均值/标准差。`paired_statistics.csv` 复用同一 `dataset × KIR × seed` 配对单位的 10,000 次 bootstrap、95% CI、win/tie/loss 和 effect size；它只用于事后分析，不用于阈值或方法选择。",
        "",
        "当前 fair 层可以回答的是：Known-only MiniLM 表示下，S2C-Trainable-K1 与 Frozen/KIR/MOGB 组件的 coverage–open-space 工作点差异。它不能回答“超过完整 MOGB、完整 DCLOOS 或历史 fulltex SOTA”。",
        "",
        "## 3. 样本级错误转移与分数层",
        "",
        "统一行级字段保留 `true_label/is_true_oos/predicted_label/predicted_oos/accepted_known`，因此 Known false rejection、OOS false acceptance、wrong-intent 和共同正确/共同错误可以按匿名 sample_id 做后续 paired set 分析。Gate 行额外保留 nearest intent/distance、center/radius/normalized distance；MOGB 行额外保留 ball id/count/size/radius/purity。",
        "已有的 score overlap、错误预算、KIR 曲线、Pareto、seed 稳定性、MOGB risk workpoint 和 error-aware UMAP 被统一收录到 `figure_manifest.json`。UMAP/geometry 只用于解释，未进入训练或参数选择。",
        "",
        "## 4. 表示、边界与 MOGB 机制",
        "",
        "本收口把表示控制和边界控制分开：native MiniLM MSP/Energy/kNN/LOF 是同一 trainable representation 上的检测器控制；MOGB-MiniLM 与 partition/boundary swap 是同协议组件归因；S2C-Trainable-K1 是 Known-only 适配后的单中心 Gate。它们不与 BERT 外部方法混为一个排名。",
        "",
        "MOGB 的逐样本 ball 字段来自 `balls.jsonl` 和 `ball_statistics.json`。selected ball 缺类、tiny-ball、radius/purity 风险继续由现有 MOGB 机制图和 `docs/analysis/MOGB_OPERATING_POINT_VISUALS_V1.md` 解释，本阶段不复制或覆盖 MOGB 原始 artifact。",
        "",
        "## 5. 监督差异与阻塞项",
        "",
        "ADB 的完整外部 BERT 标签结果可被 sample_id 对齐，但其 score 语义和 test-selection 声明不属于同一 MiniLM fair 合同，因此只进入 `external_backbone` 层。DA-ADB 保留已有 current-protocol 外部单元/汇总，不进入 fair paired table。",
        "DCLOOS 分为 `official_timeout_no_final_metrics`、`reduced_complete_different_supervision` 和 `official_timeout_no_final_metrics`。reduced 单元含 pseudo-OOS 与外部 SQuAD OOS，且来自 validation-best intermediate prediction；它可以作为监督强度参考，不能与 Known-only 方法直接排名。",
        "KNNCL、OpenMax、DOC、DeepUnk 等传统 TextOIR 路线若只有历史 fulltex 或没有当前最终 metrics，则登记为 blocked/historical，不用中间预测填表。详见 `unified_prediction_contract_v1/blocked_methods.csv`。",
        "",
        "## 6. 成本、失败场景和后续停止条件",
        "",
        "当前交付优先完成可审计主矩阵、逐样本合同、配对统计和机制索引；没有在 baseline 合同未闭合前扩展 low-resource、threshold robustness 或新的复杂模型。后续只有在同一 split、Known list、seed、评估器和 final metrics 全部闭合后，才可把外部方法提升为 fair 主表。",
        "",
        "## 7. 机器可读入口",
        "",
        f"- 汇总：`{(OUT / 'method_summary.csv').relative_to(ROOT)}`、`{(OUT / 'paired_statistics.csv').relative_to(ROOT)}`。",
        f"- 合同：`{(OUT / 'contract_summary.csv').relative_to(ROOT)}`、`{(OUT / 'prediction_schema_manifest.json').relative_to(ROOT)}`。",
        f"- 图索引：`{(OUT / 'figure_manifest.json').relative_to(ROOT)}`；图状态统计：{dict(figure_status)}。",
        f"- 对齐：`{(CONTRACT_OUT / 'alignment.csv').relative_to(ROOT)}`、`{(CONTRACT_OUT / 'failed_alignment.csv').relative_to(ROOT)}`。",
        "- 现有证据入口：`docs/analysis/EXPERIMENT_COMPARISON_OVERVIEW_V2.md`、`docs/analysis/EXPERIMENT_VISUAL_EVIDENCE_BUNDLE_V2.md`、`docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md`、`docs/对比实验/MOGB_DCLOOS_对比结果报告.md`。",
    ]
    (ROOT / "docs" / "analysis" / "UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    contract_result = build_contract(ROOT, CONTRACT_OUT, LOCAL_CONTRACT_OUT)
    summary = build_method_summary()
    contracts = contract_summary(contract_result["groups"], contract_result["alignment"])
    paired = paired_statistics()
    summary_fields = sorted({key for row in summary for key in row})
    write_csv(OUT / "method_summary.csv", summary, summary_fields)
    write_csv(
        OUT / "contract_summary.csv",
        contracts,
        list(contracts[0].keys()) if contracts else ["method"],
    )
    write_csv(
        OUT / "paired_statistics.csv",
        paired,
        list(paired[0].keys()) if paired else ["comparison_method"],
    )
    write_csv(
        OUT / "failed_or_blocked_methods.csv",
        [
            {
                "kind": "blocked_method",
                **row,
            }
            for row in contract_result["blocked"]
        ]
        + [
            {
                "kind": "alignment",
                "method": row["method"],
                "status": row["status"],
                "contract_layer": row["contract_layer"],
                "dataset": row["dataset"],
                "kir": row["kir"],
                "seed": row["seed"],
                "reason": row["reason"],
                "source": row["source_path"],
                "final_metrics_available": "unknown",
                "include_in_unified_rows": False,
            }
            for row in contract_result["alignment"]
            if row["status"] != "aligned"
        ],
        [
            "kind",
            "method",
            "status",
            "contract_layer",
            "dataset",
            "kir",
            "seed",
            "reason",
            "source",
            "final_metrics_available",
            "include_in_unified_rows",
        ],
    )
    schema = json.loads((CONTRACT_OUT / "schema_manifest.json").read_text(encoding="utf-8"))
    schema.update(
        {
            "comparison_output": str(OUT.relative_to(ROOT)),
            "method_summary": str((OUT / "method_summary.csv").relative_to(ROOT)),
            "paired_statistics": str((OUT / "paired_statistics.csv").relative_to(ROOT)),
            "contract_summary": str((OUT / "contract_summary.csv").relative_to(ROOT)),
            "test_usage": "analysis-only; no threshold or method selection",
        }
    )
    (OUT / "prediction_schema_manifest.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figures = build_figures(summary, contracts)
    write_csv(
        OUT / "figure_sources.csv",
        figures,
        ["figure_id", "path", "kind", "contract_layer", "source", "source_sha256", "status"],
    )
    (OUT / "figure_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "unified_figure_manifest_v1",
                "protocol_version": PROTOCOL_VERSION,
                "test_usage": "analysis-only",
                "figures": figures,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(summary, contracts, figures, contract_result)
    print(json.dumps({"summary_rows": len(summary), "contract_groups": len(contract_result["groups"]), "figures": len(figures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
