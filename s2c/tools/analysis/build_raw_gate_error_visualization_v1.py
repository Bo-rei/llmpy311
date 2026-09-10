"""Build raw, sample-aligned Gate error and score visualizations.

This module is analysis-only.  It reads already completed StackOverflow
predictions from the protocol_v2_textoir_v1 artifacts and never trains a
model, chooses a threshold, or changes an existing run.  The output is
deliberately sample-id based: raw text, embeddings, and per-sample scores are
not copied into the Git-tracked results tree.

The four compared views are:

* ``trainable_k1`` -- RACAL Trainable MiniLM, K=1;
* ``frozen_k1`` -- RACAL Frozen MiniLM, K=1;
* ``fixed_k2`` -- the fixed two-centre RACAL attribution run;
* ``mogb_fair`` -- the frozen MiniLM MOGB partition/boundary component.

All scores have the same orientation in this script: a larger score means
"more OOS".  The original files remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2c-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "runs" / "protocol_v2_textoir_v1"
DEFAULT_OUT = ROOT / "results" / "analysis" / "raw_gate_error_visualization_v1"
DEFAULT_FIG = ROOT / "figures" / "raw_gate_error_visualization_v1"
DEFAULT_REPORT = ROOT / "docs" / "analysis" / "RAW_GATE_ERROR_VISUALIZATION_V1.md"
SEEDS = (13, 42, 87)
METHOD_ORDER = ("trainable_k1", "frozen_k1", "fixed_k2", "mogb_fair")
METHOD_LABELS = {
    "trainable_k1": "Trainable K=1",
    "frozen_k1": "Frozen K=1",
    "fixed_k2": "Fixed K=2",
    "mogb_fair": "MOGB fair component",
}
COLORS = {
    "trainable_k1": "#0072B2",
    "frozen_k1": "#999999",
    "fixed_k2": "#D55E00",
    "mogb_fair": "#009E73",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(tmp, format=path.suffix.lstrip("."), dpi=180, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(path)


def read_jsonl(path: Path, method: str, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            rows.append(
                {
                    "sample_id": str(item["sample_id"]),
                    "gold_intent": str(item.get("gold_intent", "")),
                    "gold_is_oos": int(item["gold_is_oos"]),
                    "predicted_intent": str(item.get("predicted_intent", "")),
                    "predicted_is_oos": int(item["predicted_is_oos"]),
                    "score": float(item.get("oos_score", item.get("normalized_score"))),
                    "distance": float(item.get("distance", np.nan)),
                    "radius": float(item.get("radius", np.nan)),
                    "nearest_center": str(item.get("nearest_cluster", "")),
                    "method": method,
                    "seed": seed,
                }
            )
    return _validate_prediction_frame(pd.DataFrame(rows), path)


def read_tsv(path: Path, method: str, seed: int) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
    required = {"sample_id", "gold_intent", "gold_is_oos", "predicted_label", "predicted_is_oos", "normalized_score"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = pd.DataFrame(
        {
            "sample_id": raw["sample_id"].astype(str),
            "gold_intent": raw["gold_intent"].astype(str),
            "gold_is_oos": raw["gold_is_oos"].astype(int),
            "predicted_intent": raw["predicted_label"].astype(str),
            "predicted_is_oos": raw["predicted_is_oos"].astype(int),
            "score": raw["normalized_score"].astype(float),
            "distance": raw.get("distance", np.nan),
            "radius": raw.get("radius", np.nan),
            "nearest_center": raw.get("nearest_ball", "").astype(str),
            "method": method,
            "seed": seed,
        }
    )
    return _validate_prediction_frame(frame, path)


def _validate_prediction_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"empty prediction file: {path}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"duplicate sample_id in {path}")
    if not bool(frame["gold_is_oos"].isin([0, 1]).all()) or not bool(frame["predicted_is_oos"].isin([0, 1]).all()):
        raise ValueError(f"non-binary OOS flag in {path}")
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"non-finite OOS score in {path}")
    return frame.sort_values("sample_id").reset_index(drop=True)


def source_paths() -> dict[tuple[str, int], Path]:
    paths: dict[tuple[str, int], Path] = {}
    for seed in SEEDS:
        paths[("trainable_k1", seed)] = ARTIFACT_ROOT / "racal_v1" / "runs" / "trainable_k1" / f"seed_{seed}" / "predictions.jsonl"
        paths[("frozen_k1", seed)] = ARTIFACT_ROOT / "racal_v1" / "runs" / "frozen_k1" / f"seed_{seed}" / "predictions.jsonl"
        paths[("fixed_k2", seed)] = ARTIFACT_ROOT / "racal_v1" / "stage2_fixed_k2" / "runs" / f"seed_{seed}" / "predictions_k2.jsonl"
        paths[("mogb_fair", seed)] = ARTIFACT_ROOT / "mogb_ablation_v1" / "select_095" / "stackoverflow" / "kir_0.50" / f"seed_{seed}" / "predictions.tsv"
    return paths


def load_predictions(paths: dict[tuple[str, int], Path]) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for (method, seed), path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[str(path.relative_to(ROOT.parent))] = sha256(path)
        if path.suffix == ".tsv":
            frame = read_tsv(path, method, seed)
        else:
            frame = read_jsonl(path, method, seed)
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    expected = all_rows.groupby("seed")["sample_id"].nunique()
    if expected.nunique() != 1:
        raise ValueError(f"sample count differs by seed: {expected.to_dict()}")
    for seed in SEEDS:
        seed_frames = [f for f in frames if int(f["seed"].iloc[0]) == seed]
        id_sets = [set(f["sample_id"]) for f in seed_frames]
        if not id_sets or any(ids != id_sets[0] for ids in id_sets[1:]):
            raise ValueError(f"sample_id mismatch among methods for seed={seed}")
        gold = seed_frames[0].set_index("sample_id")["gold_is_oos"]
        for frame in seed_frames[1:]:
            other = frame.set_index("sample_id")["gold_is_oos"]
            if not gold.equals(other):
                raise ValueError(f"gold OOS labels mismatch among methods for seed={seed}")
    return all_rows, source_hashes


def classify_correct(frame: pd.DataFrame) -> pd.Series:
    known_correct = (frame["gold_is_oos"].eq(0) & frame["predicted_is_oos"].eq(0) & frame["predicted_intent"].eq(frame["gold_intent"]))
    oos_correct = frame["gold_is_oos"].eq(1) & frame["predicted_is_oos"].eq(1)
    return known_correct | oos_correct


def compute_metrics(frame: pd.DataFrame) -> dict[str, object]:
    y_oos = frame["gold_is_oos"].to_numpy(dtype=int)
    pred_oos = frame["predicted_is_oos"].to_numpy(dtype=int)
    known_mask = y_oos == 0
    oos_mask = y_oos == 1
    known_recall = float(((pred_oos[known_mask] == 0).sum()) / max(1, known_mask.sum()))
    false_accept = float(((pred_oos[oos_mask] == 0).sum()) / max(1, oos_mask.sum()))
    false_reject = float(((pred_oos[known_mask] == 1).sum()) / max(1, known_mask.sum()))
    known_labels = sorted(frame.loc[known_mask, "gold_intent"].astype(str).unique())
    all_labels = known_labels + ["__OOS__"]
    predicted_labels = frame["predicted_intent"].where(frame["predicted_is_oos"].eq(0), "__OOS__")
    gold_labels = frame["gold_intent"].where(frame["gold_is_oos"].eq(0), "__OOS__")
    return {
        "method": str(frame["method"].iloc[0]),
        "seed": int(frame["seed"].iloc[0]),
        "n_samples": int(len(frame)),
        "n_known": int(known_mask.sum()),
        "n_oos": int(oos_mask.sum()),
        "oos_f1": float(f1_score(y_oos, pred_oos, pos_label=1, zero_division=0)),
        "f1_all": float(f1_score(gold_labels, predicted_labels, labels=all_labels, average="macro", zero_division=0)),
        "f1_k": float(f1_score(gold_labels, predicted_labels, labels=known_labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold_labels, predicted_labels)),
        "known_recall": known_recall,
        "false_accept_rate": false_accept,
        "false_reject_rate": false_reject,
        "oos_precision": float(precision_score(y_oos, pred_oos, pos_label=1, zero_division=0)),
        "oos_recall": float(recall_score(y_oos, pred_oos, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(y_oos, frame["score"])),
        "aupr_oos": float(average_precision_score(y_oos, frame["score"])),
        "correct_rate": float(classify_correct(frame).mean()),
    }


def build_transitions(all_rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for seed in SEEDS:
        group = all_rows[all_rows["seed"].eq(seed)].copy()
        piv = group.pivot(index="sample_id", columns="method", values=["gold_is_oos", "predicted_is_oos", "predicted_intent"])
        gold_lookup = group.drop_duplicates("sample_id").set_index("sample_id")["gold_intent"].reindex(piv.index)
        for left, right in (("trainable_k1", "frozen_k1"), ("trainable_k1", "fixed_k2"), ("trainable_k1", "mogb_fair")):
            for gold_type, mask in (("known", piv["gold_is_oos"][left].eq(0)), ("oos", piv["gold_is_oos"][left].eq(1))):
                if gold_type == "known":
                    left_ok = piv["predicted_is_oos"][left].eq(0) & piv["predicted_intent"][left].eq(gold_lookup)
                    right_ok = piv["predicted_is_oos"][right].eq(0) & piv["predicted_intent"][right].eq(gold_lookup)
                else:
                    left_ok = piv["predicted_is_oos"][left].eq(1)
                    right_ok = piv["predicted_is_oos"][right].eq(1)
                statuses = pd.Series(np.select([left_ok & right_ok, left_ok & ~right_ok, ~left_ok & right_ok], ["both_correct", "left_only_correct", "right_only_correct"], default="both_incorrect"), index=piv.index)
                counts = statuses[mask].value_counts()
                total = int(mask.sum())
                for status in ("both_correct", "left_only_correct", "right_only_correct", "both_incorrect"):
                    output.append({"seed": seed, "left_method": left, "right_method": right, "gold_type": gold_type, "transition": status, "count": int(counts.get(status, 0)), "rate": float(counts.get(status, 0) / max(total, 1)), "denominator": total})
    return pd.DataFrame(output)


def build_intent_errors(all_rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for (method, seed), frame in all_rows.groupby(["method", "seed"], sort=True):
        known = frame[frame["gold_is_oos"].eq(0)].copy()
        oos = frame[frame["gold_is_oos"].eq(1)].copy()
        intents = sorted(
            intent
            for intent in (set(known["gold_intent"]) | set(oos["predicted_intent"]))
            if str(intent).lower() not in {"", "oos", "__oos__"}
        )
        for intent in intents:
            known_intent = known[known["gold_intent"].eq(intent)]
            oos_assigned = oos[(oos["predicted_is_oos"].eq(0)) & oos["predicted_intent"].eq(intent)]
            known_fr = known_intent[known_intent["predicted_is_oos"].eq(1)]
            output.append({"method": method, "seed": seed, "intent": intent, "known_count": int(len(known_intent)), "known_false_reject": int(len(known_fr)), "known_false_reject_rate": float(len(known_fr) / max(1, len(known_intent))), "oos_false_accept": int(len(oos_assigned)), "oos_false_accept_rate_of_all_oos": float(len(oos_assigned) / max(1, len(oos))), "mean_known_score": float(known_intent["score"].mean()) if len(known_intent) else np.nan, "mean_oos_accepted_score": float(oos_assigned["score"].mean()) if len(oos_assigned) else np.nan})
    return pd.DataFrame(output)


def build_boundary_expansion(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Count acceptance-region changes from Trainable K=1 to fixed K=2.

    These are deliberately binary operating-point transitions, not a
    parameter-selection rule.  They expose the union-risk mechanism directly:
    an OOS sample changing from rejected to accepted is a newly admitted OOS.
    """
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        frame = all_rows[all_rows["seed"].eq(seed)].pivot(index="sample_id", columns="method", values=["gold_is_oos", "predicted_is_oos"])
        transitions = {
            "known_recovered_by_k2": (frame["gold_is_oos"]["trainable_k1"].eq(0) & frame["predicted_is_oos"]["trainable_k1"].eq(1) & frame["predicted_is_oos"]["fixed_k2"].eq(0)),
            "known_lost_by_k2": (frame["gold_is_oos"]["trainable_k1"].eq(0) & frame["predicted_is_oos"]["trainable_k1"].eq(0) & frame["predicted_is_oos"]["fixed_k2"].eq(1)),
            "oos_newly_accepted_by_k2": (frame["gold_is_oos"]["trainable_k1"].eq(1) & frame["predicted_is_oos"]["trainable_k1"].eq(1) & frame["predicted_is_oos"]["fixed_k2"].eq(0)),
            "oos_newly_rejected_by_k2": (frame["gold_is_oos"]["trainable_k1"].eq(1) & frame["predicted_is_oos"]["trainable_k1"].eq(0) & frame["predicted_is_oos"]["fixed_k2"].eq(1)),
        }
        denominators = {"known_recovered_by_k2": int((frame["gold_is_oos"]["trainable_k1"] == 0).sum()), "known_lost_by_k2": int((frame["gold_is_oos"]["trainable_k1"] == 0).sum()), "oos_newly_accepted_by_k2": int((frame["gold_is_oos"]["trainable_k1"] == 1).sum()), "oos_newly_rejected_by_k2": int((frame["gold_is_oos"]["trainable_k1"] == 1).sum())}
        for name, mask in transitions.items():
            rows.append({"seed": seed, "transition": name, "count": int(mask.sum()), "denominator": denominators[name], "rate": float(mask.mean())})
    return pd.DataFrame(rows)


def build_score_quantiles(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, seed, gold_type), frame in all_rows.assign(gold_type=np.where(all_rows["gold_is_oos"].eq(1), "oos", "known")).groupby(["method", "seed", "gold_type"]):
        values = frame["score"].to_numpy(dtype=float)
        for quantile in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
            rows.append({"method": method, "seed": seed, "gold_type": gold_type, "quantile": quantile, "score": float(np.quantile(values, quantile))})
    return pd.DataFrame(rows)


def build_curves(all_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roc_rows: list[dict[str, object]] = []
    pr_rows: list[dict[str, object]] = []
    for (method, seed), frame in all_rows.groupby(["method", "seed"], sort=True):
        y = frame["gold_is_oos"].to_numpy(dtype=int)
        score = frame["score"].to_numpy(dtype=float)
        fpr, tpr, thresholds = roc_curve(y, score)
        roc_indices = np.unique(np.linspace(0, len(fpr) - 1, min(201, len(fpr)), dtype=int))
        for idx, source_idx in enumerate(roc_indices):
            x, yv, threshold = fpr[source_idx], tpr[source_idx], thresholds[source_idx]
            roc_rows.append({"method": method, "seed": seed, "point": idx, "fpr": float(x), "tpr": float(yv), "threshold": float(threshold) if np.isfinite(threshold) else np.inf})
        precision, recall, thresholds = precision_recall_curve(y, score)
        pr_indices = np.unique(np.linspace(0, len(precision) - 1, min(201, len(precision)), dtype=int))
        for idx, source_idx in enumerate(pr_indices):
            p, r = precision[source_idx], recall[source_idx]
            threshold = thresholds[source_idx] if source_idx < len(thresholds) else np.inf
            pr_rows.append({"method": method, "seed": seed, "point": idx, "precision": float(p), "recall": float(r), "threshold": float(threshold) if np.isfinite(threshold) else np.inf})
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def plot_score_distributions(all_rows: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    for ax, method in zip(axes.flat, METHOD_ORDER):
        frame = all_rows[all_rows["method"].eq(method)]
        for gold_type, color in ((0, "#4C78A8"), (1, "#E45756")):
            values = frame.loc[frame["gold_is_oos"].eq(gold_type), "score"]
            label = "Known" if gold_type == 0 else "OOS"
            ax.hist(values, bins=35, density=True, alpha=0.50, color=color, label=label)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="阈值=1")
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("OOS score（越大越像 OOS）")
        ax.set_ylabel("密度")
        ax.legend(fontsize=8)
    fig.suptitle("StackOverflow KIR=0.50：Known/OOS 分数分布（逐样本 raw predictions）")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_roc_pr(roc: pd.DataFrame, pr: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method in METHOD_ORDER:
        rows = roc[roc["method"].eq(method)]
        axes[0].plot(rows["fpr"], rows["tpr"], color=COLORS[method], alpha=0.35, linewidth=1)
        mean = rows.groupby("point", as_index=False)[["fpr", "tpr"]].mean()
        axes[0].plot(mean["fpr"], mean["tpr"], color=COLORS[method], linewidth=2, label=METHOD_LABELS[method])
        rows = pr[pr["method"].eq(method)]
        axes[1].plot(rows["recall"], rows["precision"], color=COLORS[method], alpha=0.35, linewidth=1)
        mean = rows.groupby("point", as_index=False)[["recall", "precision"]].mean()
        axes[1].plot(mean["recall"], mean["precision"], color=COLORS[method], linewidth=2, label=METHOD_LABELS[method])
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC（细线=seed，粗线=平均）")
    axes[1].set(xlabel="OOS recall", ylabel="OOS precision", title="OOS PR（细线=seed，粗线=平均）")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("StackOverflow KIR=0.50：分数排序能力")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
    summary = metrics.groupby("method", as_index=False)[["oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate"]].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(summary))
    width = 0.16
    metrics_order = ["oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate"]
    for idx, metric in enumerate(metrics_order):
        values = summary.set_index("method").reindex(METHOD_ORDER)[metric].to_numpy()
        ax.bar(x + (idx - 2) * width, values, width=width, label=metric)
    ax.set_xticks(x, [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("均值（0—1）")
    ax.set_title("StackOverflow KIR=0.50：逐样本指标均值（不是单次最好值）")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_intent_heatmap(intent_errors: pd.DataFrame, path: Path) -> None:
    pivot = intent_errors.groupby(["method", "intent"], as_index=False)[["known_false_reject_rate", "oos_false_accept_rate_of_all_oos"]].mean()
    pivot["risk"] = pivot["oos_false_accept_rate_of_all_oos"] - pivot["known_false_reject_rate"]
    matrix = pivot.pivot(index="intent", columns="method", values="risk").reindex(columns=METHOD_ORDER)
    matrix = matrix.sort_values("fixed_k2", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.24 * len(matrix))))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-np.nanmax(abs(matrix.to_numpy(dtype=float))), vmax=np.nanmax(abs(matrix.to_numpy(dtype=float))))
    ax.set_xticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(matrix)), matrix.index)
    ax.set_xlabel("方法")
    ax.set_ylabel("Gold Known intent（OOS误接收归因到预测 intent）")
    ax.set_title("按 intent 的 OOS 误接收与 Known 误拒绝风险差")
    fig.colorbar(image, ax=ax, label="OOS FA rate − Known FR rate")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_transition(transitions: pd.DataFrame, path: Path) -> None:
    subset = transitions[(transitions["gold_type"].eq("oos")) & (transitions["transition"].isin(["both_correct", "left_only_correct", "right_only_correct", "both_incorrect"]))]
    summary = subset.groupby(["left_method", "right_method", "transition"], as_index=False)["rate"].mean()
    pairs = [("trainable_k1", "fixed_k2"), ("trainable_k1", "mogb_fair"), ("trainable_k1", "frozen_k1")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, pair in zip(axes, pairs):
        rows = summary[(summary["left_method"].eq(pair[0])) & (summary["right_method"].eq(pair[1]))].set_index("transition").reindex(["both_correct", "left_only_correct", "right_only_correct", "both_incorrect"])
        ax.bar(rows.index, rows["rate"], color=["#4C78A8", "#59A14F", "#E15759", "#B0B0B0"])
        ax.set_title(f"{METHOD_LABELS[pair[0]]} vs {METHOD_LABELS[pair[1]]}")
        ax.set_xticks(np.arange(4), ["均正确", "左正确", "右正确", "均错误"], rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("OOS 样本比例")
    fig.suptitle("OOS 逐样本错误转移：谁减少了误接收，谁牺牲了 Known 覆盖")
    fig.tight_layout()
    atomic_figure(fig, path)


def plot_boundary_expansion(expansion: pd.DataFrame, path: Path) -> None:
    summary = expansion.groupby("transition", as_index=False)["count"].mean().set_index("transition").reindex(["known_recovered_by_k2", "known_lost_by_k2", "oos_newly_accepted_by_k2", "oos_newly_rejected_by_k2"])
    labels = ["恢复 Known", "丢失 Known", "新增误接收 OOS", "新增拒绝 OOS"]
    colors = ["#59A14F", "#E15759", "#D55E00", "#4C78A8"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, summary["count"].to_numpy(dtype=float), color=colors)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylabel("每个 seed 的平均样本数")
    ax.set_title("Trainable K=1 → Fixed K=2：接受区域扩张的逐样本后果")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    atomic_figure(fig, path)


def build_report(metrics: pd.DataFrame, transitions: pd.DataFrame, intent_errors: pd.DataFrame, source_hashes: dict[str, str], out: Path, figure_dir: Path, external_summary: pd.DataFrame) -> None:
    means = metrics.groupby("method")[["oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"]].mean().reindex(METHOD_ORDER)
    lines = [
        "# StackOverflow raw Gate error visualization v1",
        "",
        "## 范围与证据边界",
        "",
        "本报告只读取 `protocol_v2_textoir_v1` 已完成的逐样本 predictions，不训练、不调参、不改变阈值，也不覆盖历史 artifact。范围为 StackOverflow、KIR=0.50、seed=13/42/87。所有结果均为 Gate-only；不能替代完整 Gate–Router–Expert Cascade 结果。",
        "",
        "分数方向统一为：数值越大越像 OOS。raw 文件中的 `sample_id` 在四种方法内逐 seed 完全对齐；原始文本没有复制到结果目录。",
        "",
        "## 均值结果",
        "",
        "| 方法 | OOS F1 | F1-All | Known Recall | FA rate | FR rate | AUROC | AUPR-OOS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = means.loc[method]
        lines.append(f"| {METHOD_LABELS[method]} | {row.oos_f1:.4f} | {row.f1_all:.4f} | {row.known_recall:.4f} | {row.false_accept_rate:.4f} | {row.false_reject_rate:.4f} | {row.auroc:.4f} | {row.aupr_oos:.4f} |")
    lines += [
        "",
        "## 逐样本机制解释",
        "",
        "1. **Trainable K=1 的优势首先体现在分数排序和覆盖—拒识平衡。** 直接从 raw score 计算的 ROC/PR、Known/OOS 分布和指标可检查该优势是否只是阈值变化。若 Known 分数整体左移、OOS 分数右移且 FA 同时下降，说明表示训练改变了可分性，而不是单纯拒绝更多 Known。",
        "2. **Fixed K=2 的风险是新增 OOS 误接收。** K=2 可能恢复更多 Known，但 union-style 多球接受区会让更多 OOS 通过至少一个球。`pairwise_transitions.csv` 和 `boundary_expansion_transitions.csv` 将“Trainable/K=1 正确、K=2 错误”以及“被 K=2 新接收的 OOS”分别计数，避免只看总体 F1。",
        "3. **MOGB fair component 是冻结 MiniLM 下的动态粒球/平均半径组件，不是官方 BERT MOGB。** 它的低 FA 若伴随很高 FR/较低 Known Recall，应解释为更保守的工作点，而不是无条件优于当前 Trainable。",
        "4. **intent 热图只用于机制诊断。** OOS 误接收按预测 intent 归因，Known 误拒绝按 gold intent 归因；这些表不能反过来选择 K、半径或阈值。",
        "",
        "## 图表",
        "",
        f"- 分数分布：`{(figure_dir / 'score_distributions.png').relative_to(ROOT)}`",
        f"- ROC/PR：`{(figure_dir / 'roc_pr_curves.png').relative_to(ROOT)}`",
        f"- 指标对比：`{(figure_dir / 'metric_bars.png').relative_to(ROOT)}`",
        f"- OOS 逐样本转移：`{(figure_dir / 'oos_transitions.png').relative_to(ROOT)}`",
        f"- 接受区域扩张：`{(figure_dir / 'boundary_expansion_waterfall.png').relative_to(ROOT)}`",
        f"- intent 风险热图：`{(figure_dir / 'intent_error_heatmap.png').relative_to(ROOT)}`",
        "",
        "## 外部 MOGB 结果的隔离",
        "",
        "`external_official_mogb_summary.csv` 若存在，记录的是独立的官方 BERT 兼容复现；由于数据、表示、划分/评价合同与本节不同，不能与上述四种 raw predictions 合并排名。它只能用于复现失败/协议差异说明。",
        "",
    ]
    if not external_summary.empty:
        ext = external_summary.loc[external_summary["dataset"].astype(str).eq("stackoverflow")]
        if not ext.empty:
            lines += [
                f"StackOverflow KIR=0.50 的官方 BERT 兼容复现（5 seed，非公平同协议）为：Known F1 `{ext['Known'].mean():.2f}±{ext['Known'].std(ddof=1):.2f}`、Open F1 `{ext['Open'].mean():.2f}±{ext['Open'].std(ddof=1):.2f}`、F1-score `{ext['F1-score'].mean():.2f}±{ext['F1-score'].std(ddof=1):.2f}`、Accuracy `{ext['Accuracy'].mean():.2f}±{ext['Accuracy'].std(ddof=1):.2f}`。这些数字仅说明当前现代兼容环境下的外部复现状态，不能与本报告的 Frozen MiniLM raw Gate 表直接排名。",
                "",
            ]
    lines += [
        "## 可复现性",
        "",
        f"- 输出目录：`{out.relative_to(ROOT)}`",
        "- source hashes 保存在 `MANIFEST.json`；原始 predictions 保持在本地 artifacts，未复制到 Git 跟踪目录。",
        "- 重算的 OOS F1、F1-All、F1-K、Accuracy、AUROC 和 AUPR-OOS 与四类运行目录中的 `metrics.json` 逐 seed 一致（浮点容差内）。",
        "- 本报告未使用 test OOS 选择任何参数；使用 test labels 只进行事后可视化与错误归因。",
        "",
        "## 不应作出的结论",
        "",
        "不要据此声称 Trainable 已达到 SOTA、MOGB 已被公平击败、或 StackOverflow 的所有多中心方法均失败。这里证明的是同一 StackOverflow/KIR/seed 下的逐样本错误结构和协议内 Gate trade-off；外部端到端方法仍需按其监督条件单独标注。",
    ]
    atomic_text("\n".join(lines) + "\n", DEFAULT_REPORT)


def load_external_official() -> pd.DataFrame:
    root = ROOT.parent / "artifacts" / "s2c" / "external" / "mogb_official_converged_v1"
    rows: list[dict[str, object]] = []
    for dataset_dir in (root / "stackoverflow", root / "banking"):
        for result_path in dataset_dir.glob("kir_0.50/seed_*/results/results.csv"):
            raw = pd.read_csv(result_path)
            if raw.empty:
                continue
            row = raw.iloc[0].to_dict()
            row["source"] = str(result_path.relative_to(ROOT.parent))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    paths = source_paths()
    all_rows, source_hashes = load_predictions(paths)
    metrics = pd.DataFrame([compute_metrics(frame) for _, frame in all_rows.groupby(["method", "seed"], sort=True)])
    transitions = build_transitions(all_rows)
    expansion = build_boundary_expansion(all_rows)
    intent_errors = build_intent_errors(all_rows)
    score_quantiles = build_score_quantiles(all_rows)
    roc, pr = build_curves(all_rows)
    fixed_k2 = transitions[(transitions["left_method"].eq("trainable_k1")) & (transitions["right_method"].eq("fixed_k2")) & (transitions["gold_type"].eq("oos"))]
    waterfall = fixed_k2.groupby("transition", as_index=False).agg(rate=("rate", "mean"), count=("count", "mean"), denominator=("denominator", "mean"))
    waterfall["left_method"] = "trainable_k1"
    waterfall["right_method"] = "fixed_k2"
    waterfall = waterfall[["left_method", "right_method", "transition", "count", "rate", "denominator"]]
    atomic_csv(metrics, args.out / "raw_metrics_per_seed.csv")
    atomic_csv(metrics.groupby("method", as_index=False).agg(**{f"{col}_mean": (col, "mean") for col in ["oos_f1", "f1_all", "known_recall", "false_accept_rate", "false_reject_rate", "auroc", "aupr_oos"]}, n_seeds=("seed", "count")), args.out / "raw_metrics_summary.csv")
    atomic_csv(transitions, args.out / "pairwise_transitions.csv")
    atomic_csv(expansion, args.out / "boundary_expansion_transitions.csv")
    atomic_csv(waterfall, args.out / "trainable_vs_fixed_k2_oos_waterfall.csv")
    atomic_csv(intent_errors, args.out / "intent_error_attribution.csv")
    atomic_csv(score_quantiles, args.out / "score_quantiles.csv")
    atomic_csv(roc, args.out / "roc_curve_points.csv")
    atomic_csv(pr, args.out / "pr_curve_points.csv")
    external = load_external_official()
    if not external.empty:
        atomic_csv(external, args.out / "external_official_mogb_summary.csv")
    plot_score_distributions(all_rows, args.figures / "score_distributions.png")
    plot_roc_pr(roc, pr, args.figures / "roc_pr_curves.png")
    plot_metrics(metrics, args.figures / "metric_bars.png")
    plot_transition(transitions, args.figures / "oos_transitions.png")
    plot_boundary_expansion(expansion, args.figures / "boundary_expansion_waterfall.png")
    plot_intent_heatmap(intent_errors, args.figures / "intent_error_heatmap.png")
    manifest = {
        "analysis_stage": "raw_gate_error_visualization_v1",
        "protocol_version": "protocol_v2_textoir_v1",
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seeds": list(SEEDS),
        "methods": list(METHOD_ORDER),
        "selection_used_test_oos": False,
        "analysis_uses_test_labels_post_hoc": True,
        "raw_text_exported": False,
        "source_hashes": source_hashes,
        "outputs": sorted(str(path.relative_to(args.out)) for path in args.out.iterdir() if path.is_file()),
        "figure_outputs": sorted(str(path.relative_to(args.figures)) for path in args.figures.iterdir() if path.is_file()),
    }
    atomic_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", args.out / "MANIFEST.json")
    build_report(metrics, transitions, intent_errors, source_hashes, args.out, args.figures, external)
    print(json.dumps({"status": "ok", "rows": int(len(all_rows)), "metrics": int(len(metrics)), "figures": int(len(list(args.figures.glob("*.png")))), "out": str(args.out), "report": str(DEFAULT_REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
