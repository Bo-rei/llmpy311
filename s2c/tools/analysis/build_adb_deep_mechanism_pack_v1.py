#!/usr/bin/env python3
"""Build native ADB boundary evidence for the current StackOverflow cell.

This bundle complements the same-protocol S2C/MOGB geometry figures with the
actual ADB decision variables: BERT test features, nearest-centroid distance,
learned radius, and the normalized distance used by ADB's open-set rule.  The
builder writes only aggregate tables and figures inside the repository.  The
numeric ADB arrays remain in the external run artifact and are never copied to
the public result directory.

The S2C-versus-ADB panels are descriptive paired comparisons across backbones;
they do not treat MiniLM and BERT representations as the same coordinate
system.  PCA is used for visualization only.  Acceptance is audited in the
native normalized-distance space, where the exact ADB boundary is 1.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2c-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


ROOT = Path(__file__).resolve().parents[2]
REPO_PARENT = ROOT.parent
DEFAULT_ANALYSIS = (
    REPO_PARENT
    / "artifacts"
    / "s2c"
    / "external"
    / "adb_mechanism_v1"
    / "stackoverflow"
    / "ADB"
    / "kir_0.50"
    / "seed_42"
    / "textoir_outputs"
    / "open_intent_detection"
    / "ADB_stackoverflow_0.5_1.0_bert_42"
    / "analysis"
)
DEFAULT_OUT = ROOT / "results" / "analysis" / "adb_deep_mechanism_v1"
DEFAULT_FIG = ROOT / "figures" / "adb_deep_mechanism_v1"
VIEW_PATH = (
    ROOT
    / "data"
    / "views"
    / "protocol_v2_textoir_v1"
    / "stackoverflow"
    / "seed_42"
    / "kir_0.50"
    / "test_combined.jsonl"
)
S2C_PER_SEED = ROOT / "results" / "analysis" / "cross_protocol_tradeoff_v1" / "per_seed.csv"
ADB_PER_SEED = ROOT / "results" / "analysis" / "adb_kir_sensitivity_v2" / "adb_per_seed.csv"
ADB_CANONICAL_RUN = (
    REPO_PARENT
    / "artifacts"
    / "s2c"
    / "external"
    / "adb_gpu_runtime_v1"
    / "stackoverflow"
    / "ADB"
    / "kir50"
    / "seed42"
)
SEED = 42
THRESHOLD = 1.0

STATE_ORDER = (
    "known_correct",
    "known_wrong",
    "known_rejected",
    "oos_correct_rejected",
    "oos_false_accept",
)
STATE_LABELS = {
    "known_correct": "Known correct",
    "known_wrong": "Known wrong intent",
    "known_rejected": "Known rejected",
    "oos_correct_rejected": "OOS rejected",
    "oos_false_accept": "OOS false accept",
}
STATE_COLORS = {
    "known_correct": "#0072B2",
    "known_wrong": "#56B4E9",
    "known_rejected": "#D55E00",
    "oos_correct_rejected": "#009E73",
    "oos_false_accept": "#CC79A7",
}
TRANSITION_LABELS = {
    "both_correct": "Both correct",
    "s2c_only_correct": "S2C only correct",
    "adb_only_correct": "ADB only correct",
    "both_wrong": "Both wrong",
}
TRANSITION_COLORS = {
    "both_correct": "#999999",
    "s2c_only_correct": "#0072B2",
    "adb_only_correct": "#D55E00",
    "both_wrong": "#C44E52",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_or_parent(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    for base in (ROOT, REPO_PARENT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (ROOT / path).resolve()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 120,
        }
    )


def save_figure(figure: plt.Figure, name: str, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / f"{name}.png", dpi=450, bbox_inches="tight")
    plt.close(figure)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def state_for(gold_oos: bool, pred_oos: bool, true_label: str, pred_label: str) -> str:
    if gold_oos:
        return "oos_correct_rejected" if pred_oos else "oos_false_accept"
    if pred_oos:
        return "known_rejected"
    return "known_correct" if pred_label == true_label else "known_wrong"


def open_correct(gold_oos: np.ndarray, pred_oos: np.ndarray, true_label: np.ndarray, pred_label: np.ndarray) -> np.ndarray:
    return np.where(
        gold_oos,
        pred_oos,
        (~pred_oos) & (pred_label.astype(str) == true_label.astype(str)),
    ).astype(bool)


def load_s2c_predictions(view: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(S2C_PER_SEED)
    rows = frame[
        (frame["dataset"] == "stackoverflow")
        & (frame["kir"] == 0.5)
        & (frame["seed"] == SEED)
        & (frame["method_label"] == "Trainable K=1")
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one S2C Trainable K=1 row, got {len(rows)}")
    metrics_path = resolve_repo_or_parent(str(rows.iloc[0]["metrics_path"]))
    prediction_path = metrics_path.parent / "predictions.jsonl"
    records = load_jsonl(prediction_path)
    view_ids = [str(row["sample_id"]) for row in view]
    record_ids = [str(row["sample_id"]) for row in records]
    if record_ids != view_ids:
        raise RuntimeError("S2C sample_id order does not match the protocol view")
    true_labels = np.asarray(
        ["oos" if int(row["gold_is_oos"]) else str(row["gold_intent"]) for row in records],
        dtype=object,
    )
    pred_labels = np.asarray(
        ["oos" if int(row["predicted_is_oos"]) else str(row["predicted_intent"]) for row in records],
        dtype=object,
    )
    view_labels = np.asarray([str(row["evaluation_label"]) for row in view], dtype=object)
    if not np.array_equal(true_labels, view_labels):
        raise RuntimeError("S2C true labels do not match the protocol view")
    return (
        true_labels == "oos",
        pred_labels == "oos",
        pred_labels,
        {
            "metrics_path": str(metrics_path),
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "test_used_for_selection": bool(rows.iloc[0]["test_used_for_selection"]),
        },
    )


def load_adb(analysis_dir: Path, view: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = analysis_dir / "analysis_manifest.json"
    run_manifest_path = analysis_dir.parents[3] / "run_manifest.json"
    required = (
        "y_true.npy",
        "y_pred.npy",
        "native_pred.npy",
        "test_features.npy",
        "nearest_label.npy",
        "nearest_distance.npy",
        "radius.npy",
        "oos_score.npy",
    )
    missing = [name for name in required if not (analysis_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"ADB native analysis is incomplete: {missing}")
    if not manifest_path.is_file() or not run_manifest_path.is_file():
        raise FileNotFoundError("ADB analysis/run manifest is missing")
    analysis_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    y_true = np.load(analysis_dir / "y_true.npy", allow_pickle=False).astype(np.int64)
    y_pred = np.load(analysis_dir / "y_pred.npy", allow_pickle=False).astype(np.int64)
    native_pred = np.load(analysis_dir / "native_pred.npy", allow_pickle=False).astype(np.int64)
    features = np.load(analysis_dir / "test_features.npy", allow_pickle=False).astype(np.float32)
    nearest_label = np.load(analysis_dir / "nearest_label.npy", allow_pickle=False).astype(np.int64)
    nearest_distance = np.load(analysis_dir / "nearest_distance.npy", allow_pickle=False).astype(np.float64)
    radius = np.load(analysis_dir / "radius.npy", allow_pickle=False).astype(np.float64)
    normalized_distance = np.load(analysis_dir / "oos_score.npy", allow_pickle=False).astype(np.float64)
    expected = len(view)
    arrays = (y_true, y_pred, native_pred, features, nearest_label, nearest_distance, radius, normalized_distance)
    if any(array.shape[0] != expected for array in arrays):
        raise RuntimeError("ADB native analysis row count does not match the protocol view")
    if not np.array_equal(y_pred, native_pred):
        raise RuntimeError("ADB hook native_pred differs from the saved native y_pred")
    known_labels = np.asarray([str(value) for value in run_manifest["known_labels"]], dtype=object)
    unknown_id = int(run_manifest["unknown_label_id"])
    safe_true_ids = np.minimum(y_true, len(known_labels) - 1)
    true_labels = np.where(y_true == unknown_id, "oos", known_labels[safe_true_ids])
    nearest_names = known_labels[nearest_label]
    pred_labels = np.where(native_pred == unknown_id, "oos", nearest_names)
    view_labels = np.asarray([str(row["evaluation_label"]) for row in view], dtype=object)
    if not np.array_equal(true_labels, view_labels):
        raise RuntimeError("ADB y_true does not match the protocol view")
    gold_oos = true_labels == "oos"
    pred_oos = pred_labels == "oos"
    states = np.asarray(
        [state_for(bool(gold), bool(pred), str(true), str(output)) for gold, pred, true, output in zip(gold_oos, pred_oos, true_labels, pred_labels)],
        dtype=object,
    )
    frame = pd.DataFrame(
        {
            "sample_index": np.arange(expected, dtype=np.int64),
            "true_label": true_labels,
            "gold_oos": gold_oos,
            "adb_pred_label": pred_labels,
            "adb_pred_oos": pred_oos,
            "adb_state": states,
            "nearest_intent": nearest_names,
            "nearest_distance": nearest_distance,
            "radius": radius,
            "normalized_distance": normalized_distance,
        }
    )
    frame.attrs["features"] = features
    frame.attrs["known_labels"] = known_labels
    return frame, {
        "analysis_manifest": str(manifest_path),
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "run_manifest": str(run_manifest_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "known_labels": known_labels.tolist(),
        "unknown_label_id": unknown_id,
        "sample_count": expected,
        "feature_shape": list(features.shape),
        "score_definition": analysis_manifest["score_definition"],
        "test_used_for_selection": "not_declared_external",
    }


def build_intent_risk(frame: pd.DataFrame, known_labels: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for intent_id, intent in enumerate(known_labels):
        known = frame[(~frame["gold_oos"]) & (frame["true_label"] == intent)]
        assigned_oos = frame[frame["gold_oos"] & (frame["nearest_intent"] == intent)]
        rows.append(
            {
                "intent_id": intent_id,
                "intent": intent,
                "radius": float(frame.loc[frame["nearest_intent"] == intent, "radius"].iloc[0]),
                "known_test_count": int(len(known)),
                "known_rejected_count": int(known["adb_pred_oos"].sum()),
                "known_recall": float((~known["adb_pred_oos"]).mean()) if len(known) else np.nan,
                "known_median_normalized_distance": float(known["normalized_distance"].median()) if len(known) else np.nan,
                "oos_assigned_count": int(len(assigned_oos)),
                "oos_false_accept_count": int((~assigned_oos["adb_pred_oos"]).sum()),
                "oos_false_accept_rate_when_assigned": float((~assigned_oos["adb_pred_oos"]).mean()) if len(assigned_oos) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_transition_tables(frame: pd.DataFrame, s2c_gold_oos: np.ndarray, s2c_pred_oos: np.ndarray, s2c_pred_label: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    s2c_state = np.asarray(
        [state_for(bool(gold), bool(pred), str(true), str(output)) for gold, pred, true, output in zip(s2c_gold_oos, s2c_pred_oos, frame["true_label"], s2c_pred_label)],
        dtype=object,
    )
    frame["s2c_pred_label"] = s2c_pred_label
    frame["s2c_pred_oos"] = s2c_pred_oos
    frame["s2c_state"] = s2c_state
    table = (
        frame.groupby(["adb_state", "s2c_state"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    table["fraction_of_test"] = table["count"] / len(frame)
    adb_correct = open_correct(frame["gold_oos"].to_numpy(), frame["adb_pred_oos"].to_numpy(), frame["true_label"].to_numpy(), frame["adb_pred_label"].to_numpy())
    s2c_correct = open_correct(s2c_gold_oos, s2c_pred_oos, frame["true_label"].to_numpy(), s2c_pred_label)
    transition = np.select(
        [adb_correct & s2c_correct, ~adb_correct & s2c_correct, adb_correct & ~s2c_correct],
        ["both_correct", "s2c_only_correct", "adb_only_correct"],
        default="both_wrong",
    )
    frame["correctness_transition"] = transition
    correctness = (
        frame.groupby(["gold_oos", "correctness_transition"], observed=False)
        .size()
        .rename("count")
        .reset_index()
    )
    correctness["fraction_within_gold_type"] = correctness.groupby("gold_oos")["count"].transform(lambda values: values / values.sum())
    return table, correctness


def plot_native_boundary(frame: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    bins = np.linspace(0, max(2.5, float(frame["normalized_distance"].quantile(0.995))), 40)
    for gold_oos, label, color in ((False, "Known", "#0072B2"), (True, "OOS", "#D55E00")):
        values = frame.loc[frame["gold_oos"] == gold_oos, "normalized_distance"]
        ax.hist(values, bins=bins, density=True, alpha=0.45, label=label, color=color)
    ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1, label="ADB boundary = 1")
    ax.set_title("Native ADB score separation")
    ax.set_xlabel("nearest distance / learned radius")
    ax.set_ylabel("density")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    positions = np.arange(len(STATE_ORDER))
    values = [frame.loc[frame["adb_state"] == state, "normalized_distance"] for state in STATE_ORDER]
    ax.boxplot(values, positions=positions, patch_artist=True, showfliers=False, widths=0.6, boxprops={"facecolor": "#D9D9D9"}, medianprops={"color": "black"})
    for position, state in zip(positions, STATE_ORDER):
        ax.scatter(position, np.median(frame.loc[frame["adb_state"] == state, "normalized_distance"]), color=STATE_COLORS[state], zorder=3, s=25)
    ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions, [STATE_LABELS[state].replace(" ", "\n", 1) for state in STATE_ORDER], rotation=0)
    ax.set_ylabel("normalized distance")
    ax.set_title("Error states relative to the exact boundary")

    ax = axes[1, 0]
    for state in STATE_ORDER:
        subset = frame[frame["adb_state"] == state]
        ax.scatter(subset["nearest_distance"], subset["radius"], s=7, alpha=0.42, color=STATE_COLORS[state], label=STATE_LABELS[state])
    limit = max(float(frame["nearest_distance"].quantile(0.995)), float(frame["radius"].max()))
    ax.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1, label="distance = radius")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("nearest centroid distance")
    ax.set_ylabel("learned radius")
    ax.set_title("Native acceptance geometry")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    ax = axes[1, 1]
    medians = [float(frame.loc[frame["adb_state"] == state, "normalized_distance"].median()) for state in STATE_ORDER]
    counts = [int((frame["adb_state"] == state).sum()) for state in STATE_ORDER]
    bars = ax.bar(np.arange(len(STATE_ORDER)), medians, color=[STATE_COLORS[state] for state in STATE_ORDER])
    for bar, median, count in zip(bars, medians, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{median:.2f}\n(n={count})", ha="center", va="bottom", fontsize=8)
    ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(STATE_ORDER)), [STATE_LABELS[state].replace(" ", "\n", 1) for state in STATE_ORDER])
    ax.set_ylabel("median normalized distance")
    ax.set_title("Where the boundary places each error type")
    save_figure(fig, "adb_native_distance_radius", figure_dir)


def plot_pca_projection(frame: pd.DataFrame, figure_dir: Path) -> None:
    features = np.asarray(frame.attrs["features"], dtype=np.float32)
    projection = PCA(n_components=2, random_state=SEED).fit_transform(features)
    rng = np.random.default_rng(SEED)
    keep = np.arange(len(frame))
    if len(keep) > 2400:
        keep = np.sort(rng.choice(keep, size=2400, replace=False))
    centroids_path = DEFAULT_ANALYSIS.parent / "centroids.npy"
    if centroids_path.is_file():
        centroids = np.load(centroids_path, allow_pickle=False).astype(np.float32)
        centroid_projection = PCA(n_components=2, random_state=SEED).fit(features).transform(centroids)
    else:
        centroid_projection = None
    fig, ax = plt.subplots(figsize=(9.5, 7.5), constrained_layout=True)
    for state in STATE_ORDER:
        subset = frame.iloc[keep][frame.iloc[keep]["adb_state"].to_numpy() == state]
        indices = subset["sample_index"].to_numpy(dtype=int)
        ax.scatter(projection[indices, 0], projection[indices, 1], s=8, alpha=0.42, color=STATE_COLORS[state], label=STATE_LABELS[state])
    if centroid_projection is not None:
        ax.scatter(centroid_projection[:, 0], centroid_projection[:, 1], marker="X", s=65, facecolor="white", edgecolor="black", linewidth=0.9, label="ADB centroid")
        for index, (x_value, y_value) in enumerate(centroid_projection):
            ax.text(x_value, y_value, str(index), fontsize=7, ha="center", va="center")
    ax.set_title("ADB BERT feature space: error states and centroids")
    ax.set_xlabel("PCA-1 (visualization only)")
    ax.set_ylabel("PCA-2 (visualization only)")
    ax.text(0.01, 0.01, "Acceptance is not decided in this 2-D projection; exact rule is normalized distance ≤ 1 in native space.", transform=ax.transAxes, fontsize=8, color="#444444")
    ax.legend(frameon=False, fontsize=8, loc="best")
    save_figure(fig, "adb_centroid_pca_error_map", figure_dir)


def plot_intent_risk(risk: pd.DataFrame, figure_dir: Path) -> None:
    ordered = risk.sort_values(["known_recall", "known_test_count"], na_position="last").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), constrained_layout=True)
    ax = axes[0]
    colors = ["#D55E00" if value < 0.8 else "#0072B2" for value in ordered["known_recall"]]
    ax.barh(np.arange(len(ordered)), ordered["known_recall"], color=colors)
    ax.set_yticks(np.arange(len(ordered)), ordered["intent"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Known recall within intent")
    ax.set_title("ADB Known under-coverage by intent")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    ax = axes[1]
    valid = ordered[ordered["oos_assigned_count"] > 0].copy()
    scatter = ax.scatter(valid["radius"], valid["known_recall"], s=20 + 2 * valid["oos_assigned_count"], c=valid["oos_false_accept_rate_when_assigned"].fillna(0), cmap="magma", vmin=0, vmax=max(0.3, float(valid["oos_false_accept_rate_when_assigned"].max())), alpha=0.85, edgecolor="white", linewidth=0.5)
    for _, row in valid.iterrows():
        ax.text(row["radius"], row["known_recall"], str(int(row["intent_id"])), fontsize=7, ha="center", va="center")
    ax.set_xlabel("learned radius")
    ax.set_ylabel("Known recall")
    ax.set_title("Radius, coverage and OOS contamination")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label("OOS false-accept rate when assigned")
    fig.suptitle("ADB intent-level boundary risk (numbers identify Known intent order)", fontsize=12)
    save_figure(fig, "adb_intent_radius_risk", figure_dir)


def plot_paired_transition(frame: pd.DataFrame, figure_dir: Path) -> None:
    states = list(STATE_ORDER)
    counts = pd.crosstab(frame["adb_state"], frame["s2c_state"]).reindex(index=states, columns=states, fill_value=0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    image = ax.imshow(counts.to_numpy(), cmap="Blues")
    for row in range(len(states)):
        for col in range(len(states)):
            value = int(counts.iloc[row, col])
            ax.text(col, row, str(value), ha="center", va="center", fontsize=8, color="white" if value > counts.to_numpy().max() * 0.55 else "black")
    ax.set_xticks(range(len(states)), [STATE_LABELS[state] for state in states], rotation=45, ha="right")
    ax.set_yticks(range(len(states)), [STATE_LABELS[state] for state in states])
    ax.set_xlabel("S2C Trainable K=1 state")
    ax.set_ylabel("ADB state")
    ax.set_title("Same-sample error-state transition")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03, label="count")

    ax = axes[0, 1]
    transition_order = tuple(TRANSITION_LABELS)
    values = [int((frame["correctness_transition"] == name).sum()) for name in transition_order]
    bars = ax.bar(np.arange(len(values)), values, color=[TRANSITION_COLORS[name] for name in transition_order])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(values)), [TRANSITION_LABELS[name] for name in transition_order], rotation=25, ha="right")
    ax.set_ylabel("test samples")
    ax.set_title("Correctness transition across backbones")

    ax = axes[1, 0]
    for name in transition_order:
        subset = frame[frame["correctness_transition"] == name]
        ax.hist(subset["normalized_distance"], bins=np.linspace(0, max(2.5, float(frame["normalized_distance"].quantile(0.995))), 35), histtype="step", linewidth=1.5, color=TRANSITION_COLORS[name], label=TRANSITION_LABELS[name], density=True)
    ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("ADB normalized distance")
    ax.set_ylabel("density")
    ax.set_title("ADB boundary position of paired outcomes")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    gold_types = (False, True)
    x = np.arange(2)
    width = 0.2
    for offset, name in enumerate(transition_order):
        values = []
        for gold_oos in gold_types:
            subset = frame[frame["gold_oos"] == gold_oos]
            values.append(float((subset["correctness_transition"] == name).mean()))
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=TRANSITION_LABELS[name], color=TRANSITION_COLORS[name])
    ax.set_xticks(x, ["Known", "OOS"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction within gold type")
    ax.set_title("Where the paired difference occurs")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save_figure(fig, "adb_s2c_native_error_transition", figure_dir)


def plot_rank_transfer(frame: pd.DataFrame, figure_dir: Path) -> float:
    adb_rank = minmax_scale(frame["normalized_distance"].to_numpy(), feature_range=(0, 1)).ravel()
    s2c_rank = minmax_scale(frame["s2c_oos_score"].to_numpy(), feature_range=(0, 1)).ravel()
    frame["adb_score_rank"] = adb_rank
    frame["s2c_score_rank"] = s2c_rank
    spearman = float(pd.Series(adb_rank).corr(pd.Series(s2c_rank), method="spearman"))
    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
    for name in tuple(TRANSITION_LABELS):
        subset = frame[frame["correctness_transition"] == name]
        ax.scatter(subset["adb_score_rank"], subset["s2c_score_rank"], s=8, alpha=0.38, color=TRANSITION_COLORS[name], label=TRANSITION_LABELS[name])
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("ADB normalized-distance percentile")
    ax.set_ylabel("S2C OOS-score percentile")
    ax.set_title(f"Paired score ordering across BERT and MiniLM (Spearman ρ={spearman:.3f})")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.02, 0.02, "Percentile ranks are comparable across score scales; raw feature coordinates are not.", transform=ax.transAxes, fontsize=8, color="#444444")
    save_figure(fig, "adb_s2c_score_rank_transfer", figure_dir)
    return spearman


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    out_dir = args.out_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    if not VIEW_PATH.is_file():
        raise FileNotFoundError(VIEW_PATH)
    configure_plot()
    view = load_jsonl(VIEW_PATH)
    adb_frame, adb_audit = load_adb(analysis_dir, view)
    canonical_prediction_path = (
        ADB_CANONICAL_RUN
        / "textoir_outputs"
        / "open_intent_detection"
        / "ADB_stackoverflow_0.5_1.0_bert_42"
        / "y_pred.npy"
    )
    if canonical_prediction_path.is_file():
        canonical_prediction = np.load(canonical_prediction_path, allow_pickle=False)
        probe_prediction = np.load(analysis_dir / "y_pred.npy", allow_pickle=False)
        adb_audit["canonical_main_prediction_path"] = str(canonical_prediction_path)
        adb_audit["canonical_main_prediction_sha256"] = sha256_file(canonical_prediction_path)
        adb_audit["canonical_main_prediction_mismatch_count"] = int(
            np.sum(canonical_prediction != probe_prediction)
        )
        adb_audit["mechanism_probe_is_main_metric_run"] = bool(
            np.array_equal(canonical_prediction, probe_prediction)
        )
    else:
        adb_audit["canonical_main_prediction_path"] = str(canonical_prediction_path)
        adb_audit["canonical_main_prediction_mismatch_count"] = None
        adb_audit["mechanism_probe_is_main_metric_run"] = False
    s2c_gold_oos, s2c_pred_oos, s2c_pred_label, s2c_audit = load_s2c_predictions(view)
    transition_table, correctness_table = build_transition_tables(adb_frame, s2c_gold_oos, s2c_pred_oos, s2c_pred_label)
    adb_frame["s2c_oos_score"] = np.nan
    metrics_path = resolve_repo_or_parent(str(pd.read_csv(S2C_PER_SEED).query("dataset == 'stackoverflow' and kir == 0.5 and seed == 42 and method_label == 'Trainable K=1'").iloc[0]["metrics_path"]))
    prediction_records = load_jsonl(metrics_path.parent / "predictions.jsonl")
    if all("oos_score" in record for record in prediction_records):
        adb_frame["s2c_oos_score"] = [float(record["oos_score"]) for record in prediction_records]
    else:
        adb_frame["s2c_oos_score"] = s2c_pred_oos.astype(float)
    if not np.isfinite(adb_frame["s2c_oos_score"]).all():
        raise RuntimeError("S2C predictions do not contain a usable oos_score")

    known_labels = [str(value) for value in adb_frame.attrs["known_labels"]]
    risk = build_intent_risk(adb_frame, known_labels)
    state_summary = (
        adb_frame.groupby("adb_state", observed=False)["normalized_distance"]
        .agg(count="size", mean="mean", median="median", p90=lambda values: values.quantile(0.9))
        .reindex(STATE_ORDER)
        .reset_index()
    )
    state_summary["fraction_of_test"] = state_summary["count"] / len(adb_frame)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(adb_frame.drop(columns=[], errors="ignore"), out_dir / "adb_sample_mechanism.csv")
    atomic_csv(risk, out_dir / "adb_intent_boundary_risk.csv")
    atomic_csv(transition_table, out_dir / "adb_s2c_state_transition.csv")
    atomic_csv(correctness_table, out_dir / "adb_s2c_correctness_transition.csv")
    atomic_csv(state_summary, out_dir / "adb_native_state_summary.csv")

    plot_native_boundary(adb_frame, figure_dir)
    plot_pca_projection(adb_frame, figure_dir)
    plot_intent_risk(risk, figure_dir)
    plot_paired_transition(adb_frame, figure_dir)
    spearman = plot_rank_transfer(adb_frame, figure_dir)

    source_paths = [
        VIEW_PATH,
        S2C_PER_SEED,
        ADB_PER_SEED,
        canonical_prediction_path,
        ADB_CANONICAL_RUN / "run_manifest.json",
        analysis_dir / "analysis_manifest.json",
        analysis_dir.parents[3] / "run_manifest.json",
    ]
    for name in ("y_true.npy", "y_pred.npy", "native_pred.npy", "test_features.npy", "nearest_label.npy", "nearest_distance.npy", "radius.npy", "oos_score.npy"):
        source_paths.append(analysis_dir / name)
    source_hashes = {str(path): sha256_file(path) for path in source_paths if path.is_file()}
    figures = sorted(str(path.relative_to(ROOT)) for path in figure_dir.glob("*.png"))
    manifest = {
        "analysis": "adb_deep_mechanism_pack_v1",
        "analysis_only": True,
        "dataset": "stackoverflow",
        "kir": 0.5,
        "seed": SEED,
        "protocol_view": str(VIEW_PATH.relative_to(ROOT)),
        "contract": "external_backbone_reference",
        "backbone": "BERT",
        "method": "ADB",
        "native_boundary": "nearest_centroid_distance / learned_radius <= 1.0",
        "projection": "PCA-2D-for-visualization-only",
        "raw_text_saved": False,
        "raw_embeddings_exported": False,
        "sample_ids_exported": False,
        "test_used_for_selection": False,
        "mechanism_probe_not_substituted_for_main_metrics": True,
        "figures": figures,
        "outputs": {
            "sample_mechanism": str((out_dir / "adb_sample_mechanism.csv").relative_to(ROOT)),
            "intent_boundary_risk": str((out_dir / "adb_intent_boundary_risk.csv").relative_to(ROOT)),
            "state_transition": str((out_dir / "adb_s2c_state_transition.csv").relative_to(ROOT)),
            "correctness_transition": str((out_dir / "adb_s2c_correctness_transition.csv").relative_to(ROOT)),
            "native_state_summary": str((out_dir / "adb_native_state_summary.csv").relative_to(ROOT)),
        },
        "audits": {
            "adb": adb_audit,
            "s2c": s2c_audit,
            "sample_order_exact": True,
            "true_label_order_exact": True,
            "score_rank_spearman": spearman,
            "adb_saved_pred_exact": True,
        },
        "source_hashes": source_hashes,
    }
    atomic_json(manifest, out_dir / "MANIFEST.json")
    print(json.dumps({"out_dir": str(out_dir), "figure_count": len(figures), "score_rank_spearman": spearman}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
