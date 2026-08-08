"""Build a five-seed, protocol-aware MiniLM comparison pack.

This script only reads completed artifacts.  It joins the 45 Trainable K=1
runs (three datasets, three KIR values, five seeds) with the corresponding
Frozen K=1 E2 runs and the existing five-seed MOGB fair matrix.  The joins are
keyed by dataset/KIR/seed; no test result is used to select a method.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1"
TRAINABLE_ROOTS = [
    ARTIFACTS / "minilm_trainable_kir_sweep_v1",
    ARTIFACTS / "minilm_trainable_kir_sweep_extension_v1",
]
E2_ROOT = ARTIFACTS / "e2_gate_core_dense"
FAIR_PATH = ROOT / "results" / "mogb" / "fair_matrix.csv"
OUT = ROOT / "results" / "analysis" / "minilm_trainable_5seed_fair_v1"
FIG = ROOT / "figures" / "minilm_trainable_5seed_fair_v1"
REPORT = ROOT / "docs" / "analysis" / "MINILM_TRAINABLE_5SEED_FAIR_COMPARISON_V1.md"

DATASETS = ("clinc150", "banking77", "stackoverflow")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87, 100, 123)
METRICS = (
    "oos_f1",
    "f1_all",
    "f1_k",
    "accuracy",
    "known_recall",
    "false_accept_rate",
    "false_reject_rate",
    "auroc",
    "aupr_oos",
)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_trainable() -> pd.DataFrame:
    rows: list[dict] = []
    for root in TRAINABLE_ROOTS:
        for path in sorted(root.glob("kir_*/runs/*/seed_*/metrics.json")):
            payload = _read_json(path)
            payload["source_stage"] = root.name
            payload["metrics_path"] = str(path.relative_to(ROOT.parent))
            rows.append(payload)
    frame = pd.DataFrame(rows)
    required = {"dataset", "kir", "seed", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Trainable metrics missing columns: {missing}")
    frame = frame.loc[
        frame["dataset"].isin(DATASETS)
        & frame["kir"].astype(float).isin(KIRS)
        & frame["seed"].astype(int).isin(SEEDS)
    ].copy()
    frame = frame.drop_duplicates(["dataset", "kir", "seed"], keep=False)
    expected = len(DATASETS) * len(KIRS) * len(SEEDS)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} unique trainable rows, got {len(frame)}")
    frame["method"] = "trainable_k1"
    frame["representation"] = "last2_minilm_plus_projection"
    frame["distance"] = "mahalanobis_diag"
    frame["boundary"] = "mean_std"
    frame["supervision"] = "known_only_train_and_calibration"
    frame["protocol_version"] = "protocol_v2_textoir_v1"
    frame["test_used_for_selection"] = frame["test_used_for_selection"].astype(bool)
    frame["oos_used_for_training"] = frame["oos_used_for_training"].astype(bool)
    if frame["test_used_for_selection"].any() or frame["oos_used_for_training"].any():
        raise ValueError("Trainable rows violate the selection/training contract")
    return frame.sort_values(["dataset", "kir", "seed"]).reset_index(drop=True)


def _parse_e2_name(name: str) -> dict[str, object] | None:
    parts = name.split("__")
    if len(parts) < 8:
        return None
    values: dict[str, object] = {}
    for part in parts:
        if part.startswith("protocol_v2_textoir_v1"):
            continue
        if part.startswith("repr_"):
            values["representation"] = part.removeprefix("repr_")
        elif part.startswith("kir_"):
            values["kir"] = float(part.removeprefix("kir_"))
        elif part.startswith("seed_"):
            values["seed"] = int(part.removeprefix("seed_"))
        elif part.startswith("k_"):
            values["k"] = int(part.removeprefix("k_"))
        elif part.startswith("dist_"):
            values["distance"] = part.removeprefix("dist_")
        elif part.startswith("boundary_"):
            values["boundary"] = part.removeprefix("boundary_")
        elif part in DATASETS:
            values["dataset"] = part
    return values if len(values) == 7 else None


def read_frozen_e2() -> pd.DataFrame:
    rows: list[dict] = []
    for path in E2_ROOT.iterdir():
        if not path.is_dir() or not path.name.endswith("__boundary_mean_std"):
            continue
        spec = _parse_e2_name(path.name)
        if spec is None:
            continue
        if (
            spec["dataset"] not in DATASETS
            or float(spec["kir"]) not in KIRS
            or int(spec["seed"]) not in SEEDS
            or int(spec["k"]) != 1
            or spec["distance"] != "mahalanobis_diag"
        ):
            continue
        metrics = _read_json(path / "metrics.json")["combined"]
        # E2 calls Known Recall ``id_recall``; normalize that name here so
        # the five-seed join uses the same schema as Trainable metrics.
        metric_aliases = {"known_recall": "id_recall"}
        row = {**spec, **{metric: metrics.get(metric_aliases.get(metric, metric)) for metric in METRICS}}
        row["method"] = "frozen_k1"
        row["representation"] = "frozen_minilm"
        row["supervision"] = "none"
        row["protocol_version"] = "protocol_v2_textoir_v1"
        row["metrics_path"] = str((path / "metrics.json").relative_to(ROOT.parent))
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected = len(DATASETS) * len(KIRS) * len(SEEDS)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} Frozen E2 rows, got {len(frame)}")
    return frame.sort_values(["dataset", "kir", "seed"]).reset_index(drop=True)


def read_fair_matrix() -> pd.DataFrame:
    frame = pd.read_csv(FAIR_PATH)
    frame["dataset"] = frame["dataset"].replace({"banking77_oos": "banking77"})
    frame = frame.loc[
        frame["dataset"].isin(DATASETS)
        & frame["kir"].isin(KIRS)
        & frame["seed"].isin(SEEDS)
    ].copy()
    details = {
        "single_centroid": ("frozen_minilm", "euclidean", "ours_mean_std", "frozen_known_only"),
        "fixed_k2": ("frozen_minilm", "euclidean", "ours_mean_std", "frozen_known_only"),
        "random_partition": ("frozen_minilm", "euclidean", "ours_mean_std", "frozen_known_only"),
        "mogb_minilm": ("frozen_minilm", "euclidean", "mogb_mean", "frozen_known_only"),
        "mogb_partition_ours_boundary": ("frozen_minilm", "mahalanobis_diag", "mean_std", "frozen_known_only"),
        "ours_partition_mogb_boundary": ("frozen_minilm", "euclidean", "mogb_mean", "frozen_known_only"),
    }
    frame["representation"] = frame["method"].map(lambda x: details[str(x)][0])
    frame["distance"] = frame["method"].map(lambda x: details[str(x)][1])
    frame["boundary"] = frame["method"].map(lambda x: details[str(x)][2])
    frame["supervision"] = frame["method"].map(lambda x: details[str(x)][3])
    frame["protocol_version"] = "protocol_v2_textoir_v1"
    return frame.sort_values(["dataset", "kir", "seed", "method"]).reset_index(drop=True)


def historical_fulltex() -> pd.DataFrame:
    values = {
        0.25: {
            "clinc150": (71.75, 95.01, 90.45),
            "stackoverflow": (73.61, 94.47, 91.04),
            "banking77": (75.83, 93.99, 89.07),
        },
        0.50: {
            "clinc150": (79.95, 91.96, 86.78),
            "stackoverflow": (75.48, 89.71, 85.54),
            "banking77": (74.90, 88.23, 78.98),
        },
        0.75: {
            "clinc150": (66.32, 87.10, 79.83),
            "stackoverflow": (80.18, 75.57, 81.32),
            "banking77": (70.28, 86.49, 77.84),
        },
    }
    rows = []
    for kir, datasets in values.items():
        for dataset, (f1_k, oos_f1, accuracy) in datasets.items():
            rows.append(
                {
                    "dataset": dataset,
                    "kir": kir,
                    "method": "fulltex_historical_ours",
                    "f1_k_percent": f1_k,
                    "oos_f1_percent": oos_f1,
                    "accuracy_percent": accuracy,
                    "source": "s2c/fulltex.tex:main_results_all",
                }
            )
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, group_cols: list[str], prefix: str = "") -> pd.DataFrame:
    agg = frame.groupby(group_cols, as_index=False).agg(
        n_seeds=("seed", "count"),
        **{f"{prefix}{metric}_mean": (metric, "mean") for metric in METRICS},
        **{f"{prefix}{metric}_std": (metric, "std") for metric in METRICS},
    )
    return agg


def write_csv(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def make_figures(trainable: pd.DataFrame, frozen: pd.DataFrame, fair: pd.DataFrame, historical: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {0.25: "#4c78a8", 0.50: "#f58518", 0.75: "#54a24b"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        for kir in KIRS:
            t = trainable.loc[(trainable.dataset == dataset) & (trainable.kir == kir)]
            f = frozen.loc[(frozen.dataset == dataset) & (frozen.kir == kir)]
            ax.errorbar([kir], [t.oos_f1.mean() * 100], yerr=[t.oos_f1.std() * 100], marker="o", color=colors[kir], capsize=3)
            ax.errorbar([kir], [f.oos_f1.mean() * 100], yerr=[f.oos_f1.std() * 100], marker="s", color=colors[kir], alpha=0.55, capsize=3)
        ax.set_title(dataset)
        ax.set_xticks(KIRS)
        ax.set_xlabel("KIR")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(
        [plt.Line2D([0], [0], marker="o", color="black", linestyle=""), plt.Line2D([0], [0], marker="s", color="black", linestyle="")],
        ["Trainable K=1", "Frozen K=1"],
        frameon=False,
    )
    fig.suptitle("Five-seed Trainable vs Frozen K=1 across KIR (same Mahalanobis contract)")
    fig.tight_layout()
    fig.savefig(FIG / "trainable_vs_frozen_5seed_kir.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    method_order = ["trainable_k1", "single_centroid", "fixed_k2", "random_partition", "mogb_minilm", "mogb_partition_ours_boundary", "ours_partition_mogb_boundary"]
    method_labels = {
        "trainable_k1": "Trainable K=1",
        "single_centroid": "Frozen single",
        "fixed_k2": "Frozen fixed K=2",
        "random_partition": "Random partition",
        "mogb_minilm": "MOGB partition",
        "mogb_partition_ours_boundary": "MOGB partition + s2c boundary",
        "ours_partition_mogb_boundary": "s2c partition + MOGB boundary",
    }
    trainable_mean = aggregate(trainable, ["dataset", "kir"])
    trainable_mean["method"] = "trainable_k1"
    trainable_mean = trainable_mean.rename(columns={"oos_f1_mean": "oos_f1", "f1_all_mean": "f1_all", "known_recall_mean": "known_recall"})
    fair_mean = fair.groupby(["dataset", "kir", "method"], as_index=False).agg(oos_f1=("oos_f1", "mean"), f1_all=("f1_all", "mean"), known_recall=("id_recall", "mean"))
    context = pd.concat([trainable_mean[["dataset", "kir", "method", "oos_f1", "f1_all", "known_recall"]], fair_mean], ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = context[context.dataset == dataset]
        for method in method_order:
            part = sub[sub.method == method].sort_values("kir")
            if part.empty:
                continue
            style = {"trainable_k1": ("*", 130, "#d62728"), "single_centroid": ("o", 45, "#1f77b4"), "fixed_k2": ("s", 45, "#2ca02c"), "random_partition": ("^", 45, "#9467bd"), "mogb_minilm": ("X", 45, "#ff7f0e"), "mogb_partition_ours_boundary": ("D", 45, "#8c564b"), "ours_partition_mogb_boundary": ("P", 45, "#17becf")}[method]
            ax.plot(part.kir, part.oos_f1 * 100, marker=style[0], markersize=8 if method == "trainable_k1" else 5, linewidth=2 if method == "trainable_k1" else 1, label=method_labels[method], color=style[2])
        ax.set_title(dataset)
        ax.set_xticks(KIRS)
        ax.set_xlabel("KIR")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Five-seed method/KIR comparison: Trainable, fixed-K, random and MOGB components")
    fig.tight_layout()
    fig.savefig(FIG / "all_methods_5seed_kir.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = context[context.dataset == dataset]
        for method in method_order:
            part = sub[sub.method == method]
            if part.empty:
                continue
            ax.scatter(part.known_recall * 100, part.oos_f1 * 100, s=90 if method == "trainable_k1" else 38, marker="*" if method == "trainable_k1" else "o", label=method_labels[method])
        ax.set_title(dataset)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[1].set_xlabel("Known Recall (%)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle("OOS F1 vs Known Recall trade-off (five-seed means)")
    fig.tight_layout()
    fig.savefig(FIG / "all_methods_known_oos_tradeoff_5seed.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Historical values are deliberately shown as a separate dashed reference,
    # not as a same-contract method ranking.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        t = trainable_mean[trainable_mean.dataset == dataset].sort_values("kir")
        h = historical[historical.dataset == dataset].sort_values("kir")
        ax.plot(t.kir, t.oos_f1 * 100, marker="o", linewidth=2, label="Trainable K=1 (current contract)")
        ax.plot(h.kir, h.oos_f1_percent, marker="x", linestyle="--", linewidth=1.8, label="fulltex Ours (historical contract)")
        ax.set_title(dataset)
        ax.set_xticks(KIRS)
        ax.set_xlabel("KIR")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("OOS F1 (%)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Historical fulltex vs current Trainable (protocol-mismatch reference)")
    fig.tight_layout()
    fig.savefig(FIG / "trainable_vs_fulltex_reference.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(trainable: pd.DataFrame, frozen: pd.DataFrame, fair: pd.DataFrame, historical: pd.DataFrame, comparison: pd.DataFrame) -> None:
    trainable_mean = aggregate(trainable, ["dataset", "kir"])
    fair_mean = fair.groupby(["dataset", "kir", "method"], as_index=False).agg(oos_f1=("oos_f1", "mean"), f1_all=("f1_all", "mean"), known_recall=("id_recall", "mean"), false_accept_rate=("false_accept_rate", "mean"))
    lines = [
        "# MiniLM Trainable 五 seed 公平对比与可视化 V1",
        "",
        "> 本报告把 Trainable K=1 补齐到与 MOGB 公平矩阵相同的五个 seed（13/42/87/100/123），用于方法和机制对比。它不把历史 `fulltex.tex` 数字伪装成同协议结果，也不把 MOGB 组件适配称为官方复现。",
        "",
        "## 1. 完成状态",
        "",
        "- Trainable K=1：45/45；新增扩展：18/18；失败、缺失、重复、无效指标：0。",
        "- 范围：CLINC150、Banking77、StackOverflow；KIR=0.25/0.50/0.75；五个 seed；仅 K=1。",
        "- 训练：MiniLM 最后两层 + residual projection；Known train 训练，Known calibration 选 checkpoint。",
        "- 测试 OOS 未用于训练、checkpoint、阈值或边界选择。",
        "- 新增 artifact：`../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_extension_v1/`。",
        "",
        "## 2. Trainable K=1 五 seed 结果",
        "",
        "| 数据集 | KIR | OOS F1 | F1-All | Known Recall | False Accept | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in trainable_mean.sort_values(["dataset", "kir"]).iterrows():
        lines.append(f"| {row.dataset} | {row.kir:.2f} | {row.oos_f1_mean*100:.2f}±{row.oos_f1_std*100:.2f} | {row.f1_all_mean*100:.2f}±{row.f1_all_std*100:.2f} | {row.known_recall_mean*100:.2f}±{row.known_recall_std*100:.2f} | {row.false_accept_rate_mean*100:.2f}±{row.false_accept_rate_std*100:.2f} | {row.auroc_mean*100:.2f}±{row.auroc_std*100:.2f} |")
    lines += [
        "",
        "## 3. 与同一 E2 Frozen K=1 的配对差值",
        "",
        "| 数据集 | KIR | Trainable−Frozen OOS F1 | Trainable−Frozen Known Recall | Trainable−Frozen False Accept |",
        "|---|---:|---:|---:|---:|",
    ]
    paired = comparison.groupby(["dataset", "kir"], as_index=False).agg(
        oos_f1_delta_pp=("oos_f1_delta_pp", "mean"),
        known_recall_delta_pp=("known_recall_delta_pp", "mean"),
        false_accept_delta_pp=("false_accept_delta_pp", "mean"),
    )
    for _, row in paired.sort_values(["dataset", "kir"]).iterrows():
        lines.append(f"| {row.dataset} | {row.kir:.2f} | {row.oos_f1_delta_pp:+.2f} pp | {row.known_recall_delta_pp:+.2f} pp | {row.false_accept_delta_pp:+.2f} pp |")
    lines += [
        "",
        "## 4. 与 Frozen/MOGB 组件的同协议上下文",
        "",
        "MOGB 公平矩阵固定 Frozen MiniLM，但其组件使用欧氏距离/平均半径或 MOGB 分区；Trainable 使用最后两层适配 + 对角马氏距离/mean+std。因此下面是协议分层比较，不是无条件排名。",
        "",
        "| 数据集 | KIR | 方法 | OOS F1 | F1-All | Known Recall | False Accept |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    names = {"trainable_k1": "Trainable K=1", "single_centroid": "Frozen single", "fixed_k2": "Frozen fixed K=2", "random_partition": "Random partition", "mogb_minilm": "MOGB partition", "mogb_partition_ours_boundary": "MOGB partition + s2c boundary", "ours_partition_mogb_boundary": "s2c partition + MOGB boundary"}
    context = pd.concat([
        trainable_mean.assign(method="trainable_k1", oos_f1=trainable_mean.oos_f1_mean, f1_all=trainable_mean.f1_all_mean, known_recall=trainable_mean.known_recall_mean, false_accept_rate=trainable_mean.false_accept_rate_mean),
        fair_mean,
    ], ignore_index=True)
    for _, row in context.sort_values(["dataset", "kir", "method"]).iterrows():
        lines.append(f"| {row.dataset} | {row.kir:.2f} | {names.get(row.method, row.method)} | {row.oos_f1*100:.2f} | {row.f1_all*100:.2f} | {row.known_recall*100:.2f} | {row.false_accept_rate*100:.2f} |")
    lines += [
        "",
        "## 5. 与 fulltex.tex 的历史数字",
        "",
        "这些数字来自 `s2c/fulltex.tex` 的 `tab:main_results_all`，历史 `Ours` 使用冻结 MiniLM、固定 K_y=2、数据集相关 λ，并且正文说明 unknown/OOS validation 参与 λ 学习；当前 Trainable 是 Known-only、Gate-only、K=1。因此这里只做参考，不作公平排名。",
        "",
        "| 数据集 | KIR | 当前 Trainable OOS F1 | fulltex Ours OOS F1 | 差值（仅描述） |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in historical.sort_values(["dataset", "kir"]).iterrows():
        current = trainable_mean[(trainable_mean.dataset == row.dataset) & (trainable_mean.kir == row.kir)].iloc[0].oos_f1_mean * 100
        lines.append(f"| {row.dataset} | {row.kir:.2f} | {current:.2f} | {row.oos_f1_percent:.2f} | {current-row.oos_f1_percent:+.2f} pp |")
    lines += [
        "",
        "## 6. 当前可以确认的机制结论",
        "",
        "1. Trainable K=1 的收益最稳定地来自单中心分数排序，而不是自动恢复多中心：在此前同一 checkpoint 的 K=2 控制中，StackOverflow OOS F1 下降约 19pp，false acceptance 上升约 34pp。",
        "2. KIR 增大后 Banking77 和 StackOverflow 的 OOS F1 下降，说明 Known/OOS 几何重叠是主要瓶颈；这不是简单增加训练 epoch 能解决的。",
        "3. Trainable 在同协议下通常优于 Frozen K=1，但 Banking77 高 KIR 退化，说明微调改变了类内方差和边界校准，收益具有数据集依赖性。",
        "4. MOGB 组件的部分 OOS F1 提升常伴随 Known Recall/F1-All 下降，必须同时看拒识收益和已知类覆盖，不能只按 OOS F1 排名。",
        "5. `fulltex.tex` 的高分不能直接作为当前 Trainable 的失败证据；先前历史方法和当前协议在 K、λ、调参监督、数据快照及系统层级上都不同。",
        "",
        "## 7. 图表与数据文件",
        "",
        "- `results/analysis/minilm_trainable_5seed_fair_v1/`：五 seed 逐单元、均值、配对差值和历史参照 CSV。",
        "- `figures/minilm_trainable_5seed_fair_v1/trainable_vs_frozen_5seed_kir.png`：Trainable/Frozen KIR 曲线。",
        "- `figures/minilm_trainable_5seed_fair_v1/all_methods_5seed_kir.png`：Trainable、固定 K、随机分区和 MOGB 组件曲线。",
        "- `figures/minilm_trainable_5seed_fair_v1/all_methods_known_oos_tradeoff_5seed.png`：Known Recall–OOS F1 权衡。",
        "- `figures/minilm_trainable_5seed_fair_v1/trainable_vs_fulltex_reference.png`：历史 fulltex 参照图。",
        "",
        "## 8. 仍然不能声称什么",
        "",
        "- 不能声称 Trainable 已达到 SOTA；ADB/DA-ADB/DCLOOS 仍不是同协议五 seed 主表。",
        "- 不能声称 K=2 或 MOGB 组件是普遍更优；StackOverflow 的固定多中心退化仍然存在。",
        "- 不能把历史 `fulltex.tex` 的 OOS validation 调参结果与当前严格 Known-only 结果直接做绝对排名。",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    trainable = read_trainable()
    frozen = read_frozen_e2()
    fair = read_fair_matrix()
    historical = historical_fulltex()
    keys = ["dataset", "kir", "seed"]
    metric_columns = keys + list(METRICS)
    comparison = trainable[metric_columns].merge(frozen[metric_columns], on=keys, how="inner", suffixes=("_trainable", "_frozen"), validate="one_to_one")
    for metric in METRICS:
        comparison[f"{metric}_delta"] = comparison[f"{metric}_trainable"] - comparison[f"{metric}_frozen"]
    comparison["method"] = "frozen_k1"
    comparison["oos_f1_delta_pp"] = comparison["oos_f1_delta"] * 100
    comparison["known_recall_delta_pp"] = comparison["known_recall_delta"] * 100
    comparison["false_accept_delta_pp"] = comparison["false_accept_rate_delta"] * 100
    trainable_mean = aggregate(trainable, ["dataset", "kir"])
    fair_mean = fair.groupby(["dataset", "kir", "method"], as_index=False).agg(oos_f1=("oos_f1", "mean"), f1_all=("f1_all", "mean"), known_recall=("id_recall", "mean"), false_accept_rate=("false_accept_rate", "mean"))
    write_csv(trainable, "trainable_per_seed.csv")
    write_csv(trainable_mean, "trainable_mean_std.csv")
    write_csv(frozen, "frozen_e2_per_seed.csv")
    write_csv(fair, "mogb_fair_per_seed.csv")
    write_csv(fair_mean, "mogb_fair_mean_std.csv")
    write_csv(comparison, "trainable_vs_frozen_paired.csv")
    write_csv(historical, "fulltex_historical_reference.csv")
    rows = []
    for _, row in trainable_mean.iterrows():
        h = historical[(historical.dataset == row.dataset) & (historical.kir == row.kir)].iloc[0]
        rows.append({"dataset": row.dataset, "kir": row.kir, "trainable_oos_f1_mean": row.oos_f1_mean, "fulltex_oos_f1": h.oos_f1_percent / 100.0, "descriptive_delta_pp": row.oos_f1_mean * 100 - h.oos_f1_percent, "source": "protocol_mismatch_reference_only"})
    write_csv(pd.DataFrame(rows), "trainable_vs_fulltex_reference.csv")
    # One table carrying all methods is convenient for downstream plots and tables.
    all_methods = pd.concat([trainable.assign(method="trainable_k1"), fair], ignore_index=True, sort=False)
    write_csv(all_methods, "all_methods_per_seed.csv")
    make_figures(trainable, frozen, fair, historical)
    write_report(trainable, frozen, fair, historical, comparison)
    print(json.dumps({"status": "complete", "trainable_units": len(trainable), "frozen_units": len(frozen), "fair_units": len(fair), "summary_rows": len(trainable_mean), "report": str(REPORT), "figures": [str(p) for p in sorted(FIG.glob("*.png"))]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
