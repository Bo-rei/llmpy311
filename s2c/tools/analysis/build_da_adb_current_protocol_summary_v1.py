#!/usr/bin/env python3
"""Summarize current-protocol DA-ADB cells against matched S2C Trainable K=1.

The external DA-ADB runs use the same protocol_v2 StackOverflow split roots and
seed-specific Known lists as the matched S2C rows, but retain a separate
BERT/TextOIR contract.  This script therefore reports paired descriptive
differences, never a pooled SOTA ranking.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT.parent / "artifacts" / "s2c"
RUN_ROOT = (
    ARTIFACTS
    / "external"
    / "da_adb_gpu_runtime_v1"
    / "stackoverflow"
    / "DA-ADB"
    / "kir_0.50"
)
S2C_PER_SEED = ROOT / "results" / "analysis" / "cross_protocol_tradeoff_v1" / "per_seed.csv"
ADB_PER_SEED = ROOT / "results" / "analysis" / "adb_kir_sensitivity_v2" / "adb_per_seed.csv"
OUT = ROOT / "results" / "analysis" / "da_adb_current_protocol_summary_v1"
FIG = ROOT / "figures" / "da_adb_current_protocol_summary_v1"
REPORT = ROOT / "docs" / "archive" / "analysis" / "DA_ADB_CURRENT_PROTOCOL_SUMMARY_V1.md"
SEEDS = (42, 87, 100)
BOOTSTRAP_SEED = 20260810
STATES = (
    "known_correct",
    "known_wrong",
    "known_rejected",
    "oos_correct_rejected",
    "oos_false_accept",
)
STATE_LABELS = {
    "known_correct": "Known correct",
    "known_wrong": "Known wrong",
    "known_rejected": "Known rejected",
    "oos_correct_rejected": "OOS correctly rejected",
    "oos_false_accept": "OOS false accept",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (ROOT, ROOT.parent):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / path).resolve()


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def load_view(seed: int) -> list[dict]:
    path = ROOT / "data" / "views" / "protocol_v2_textoir_v1" / "stackoverflow" / f"seed_{seed}" / "kir_0.50" / "test_combined.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty protocol view: {path}")
    return rows


def classify(gold_is_oos: bool, pred_is_oos: bool, gold: str, predicted: str) -> str:
    if gold_is_oos:
        return "oos_correct_rejected" if pred_is_oos else "oos_false_accept"
    if pred_is_oos:
        return "known_rejected"
    return "known_correct" if predicted == gold else "known_wrong"


def find_test_file(data_root: Path, expected_sha: str) -> Path:
    candidates = [
        path
        for path in data_root.rglob("test.tsv")
        if sha256_file(path) == expected_sha
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one test.tsv matching {expected_sha}, got {candidates}")
    return candidates[0]


def load_test_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["text", "label"]:
            raise RuntimeError(f"unexpected test.tsv fields: {path}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"empty test.tsv: {path}")
    return rows


def check_external_view(
    body: list[dict[str, str]],
    view: list[dict],
    method: str,
) -> tuple[bool, bool]:
    if len(body) != len(view):
        raise RuntimeError(f"{method} test count mismatch: {len(body)} != {len(view)}")
    text_match = [str(row["text"]) for row in body] == [str(row["text"]) for row in view]
    label_match = [str(row["label"]) for row in body] == [str(row["evaluation_label"]) for row in view]
    if not text_match or not label_match:
        raise RuntimeError(
            f"{method} protocol view mismatch: text={text_match}, label={label_match}"
        )
    return text_match, label_match


def load_s2c_predictions(row: pd.Series, view: list[dict]) -> tuple[list[str], dict]:
    metrics_path = resolve_path(str(row["metrics_path"]))
    prediction_path = metrics_path.parent / "predictions.jsonl"
    records = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    view_ids = [str(item["sample_id"]) for item in view]
    record_ids = [str(item["sample_id"]) for item in records]
    if record_ids != view_ids:
        raise RuntimeError(f"S2C sample order mismatch for seed {row['seed']}")
    predicted = [
        "oos" if bool(int(item["predicted_is_oos"])) else str(item["predicted_intent"])
        for item in records
    ]
    labels = [
        "oos" if bool(int(item["gold_is_oos"])) else str(item["gold_intent"])
        for item in records
    ]
    view_labels = [str(item["evaluation_label"]) for item in view]
    if labels != view_labels:
        raise RuntimeError(f"S2C true-label order mismatch for seed {row['seed']}")
    run_manifest = metrics_path.parent / "run_manifest.json"
    return predicted, {
        "sample_id_order_match": True,
        "true_label_match": True,
        "prediction_sha256": sha256_file(prediction_path),
        "run_manifest_sha256": sha256_file(run_manifest) if run_manifest.is_file() else None,
        "test_used_for_selection": bool(row["test_used_for_selection"]),
        "test_sample_ids_sha256": ids_hash(view_ids),
    }


def load_adb_predictions(row: pd.Series, view: list[dict]) -> tuple[list[str], dict]:
    manifest_path = resolve_path(str(row["run_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_path = find_test_file(Path(manifest["data_root"]), str(manifest["split_sha256"]["test"]))
    body = load_test_tsv(test_path)
    text_match, label_match = check_external_view(body, view, "ADB")
    prediction_dirs = manifest.get("artifact_audit", {}).get("prediction_directories", [])
    if len(prediction_dirs) != 1:
        raise RuntimeError(f"ADB prediction directory count is not one: {manifest_path}")
    prediction_root = Path(prediction_dirs[0])
    y_true = np.load(prediction_root / "y_true.npy")
    y_pred = np.load(prediction_root / "y_pred.npy")
    if y_true.shape != y_pred.shape or y_true.size != len(view):
        raise RuntimeError(f"ADB prediction shape mismatch: {manifest_path}")
    known_labels = [str(item) for item in manifest["known_labels"]]
    unknown_id = int(manifest["unknown_label_id"])
    labels = []
    predicted = []
    for index, (true_id, pred_id) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
        true_label = "oos" if int(true_id) == unknown_id else known_labels[int(true_id)]
        if true_label != str(view[index]["evaluation_label"]):
            raise RuntimeError(f"ADB y_true mismatch at row {index}: {manifest_path}")
        labels.append(true_label)
        predicted.append("oos" if int(pred_id) == unknown_id else known_labels[int(pred_id)])
    return predicted, {
        "sample_id_order_match": True,
        "true_label_match": True,
        "text_order_match": text_match,
        "label_order_match": label_match,
        "prediction_sha256": sha256_file(prediction_root / "y_pred.npy"),
        "run_manifest_sha256": sha256_file(manifest_path),
        "test_split_sha256": str(manifest["split_sha256"]["test"]),
        "test_used_for_selection": "not_declared_external",
    }


def load_da_predictions(run: Path, view: list[dict]) -> tuple[list[str], dict]:
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_dir = prediction_dir_for_run(run)
    test_path = find_test_file(Path(manifest["data_root"]), str(manifest["split_sha256"]["test"]))
    body = load_test_tsv(test_path)
    text_match, label_match = check_external_view(body, view, "DA-ADB")
    y_true = np.load(prediction_dir / "y_true.npy")
    y_pred = np.load(prediction_dir / "y_pred.npy")
    if y_true.shape != y_pred.shape or y_true.size != len(view):
        raise RuntimeError(f"DA-ADB prediction shape mismatch: {manifest_path}")
    known_labels = [str(item) for item in manifest["known_labels"]]
    unknown_id = int(manifest["unknown_label_id"])
    labels = []
    predicted = []
    for index, (true_id, pred_id) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
        true_label = "oos" if int(true_id) == unknown_id else known_labels[int(true_id)]
        if true_label != str(view[index]["evaluation_label"]):
            raise RuntimeError(f"DA-ADB y_true mismatch at row {index}: {manifest_path}")
        labels.append(true_label)
        predicted.append("oos" if int(pred_id) == unknown_id else known_labels[int(pred_id)])
    return predicted, {
        "sample_id_order_match": True,
        "true_label_match": True,
        "text_order_match": text_match,
        "label_order_match": label_match,
        "prediction_sha256": sha256_file(prediction_dir / "y_pred.npy"),
        "run_manifest_sha256": sha256_file(manifest_path),
        "test_split_sha256": str(manifest["split_sha256"]["test"]),
        "test_used_for_selection": "not_declared_external",
    }


def prediction_dir_for_run(run: Path) -> Path:
    dirs = sorted(
        p
        for p in (run / "textoir_outputs" / "open_intent_detection").glob("*")
        if (p / "y_true.npy").is_file() and (p / "y_pred.npy").is_file()
    )
    if len(dirs) != 1:
        raise RuntimeError(f"expected one prediction directory under {run}, got {dirs}")
    return dirs[0]


def prediction_dir(run: Path) -> Path:
    dirs = sorted(
        p
        for p in (run / "textoir_outputs" / "open_intent_detection").glob("*")
        if (p / "y_true.npy").is_file() and (p / "y_pred.npy").is_file()
    )
    if len(dirs) != 1:
        raise RuntimeError(f"expected one prediction directory under {run}, got {dirs}")
    return dirs[0]


def da_metrics(run: Path) -> dict:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    pdir = prediction_dir(run)
    y_true = np.load(pdir / "y_true.npy")
    y_pred = np.load(pdir / "y_pred.npy")
    unknown = int(manifest["unknown_label_id"])
    known_labels = list(range(unknown))
    known = y_true != unknown
    oos = ~known
    pred_oos = y_pred == unknown
    return {
        "method": "DA-ADB",
        "contract": "BERT/TextOIR external compatibility",
        "seed": int(manifest["seed"]),
        "run_manifest": str((run / "run_manifest.json").resolve()),
        "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
        "split_train_sha256": manifest["split_sha256"]["train"],
        "split_dev_sha256": manifest["split_sha256"]["dev"],
        "split_test_sha256": manifest["split_sha256"]["test"],
        "known_labels": "|".join(manifest["known_labels"]),
        "oos_f1": float(f1_score(oos, pred_oos, zero_division=0)),
        "f1_all": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_known": float(
            f1_score(y_true, y_pred, labels=known_labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "known_recall": float(((y_pred[known] != unknown).sum() / known.sum())),
        "false_acceptance": float(((y_pred[oos] != unknown).sum() / oos.sum())),
        "false_rejection": float(((y_pred[known] == unknown).sum() / known.sum())),
        "n_samples": int(y_true.size),
        "predicted_label_count": int(np.unique(y_pred).size),
        "finite": bool(np.isfinite(y_true).all() and np.isfinite(y_pred).all()),
        "shape_equal": bool(y_true.shape == y_pred.shape),
        "y_true_sha256": sha256_file(pdir / "y_true.npy"),
        "y_pred_sha256": sha256_file(pdir / "y_pred.npy"),
    }


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=20_000,
        method="percentile",
        rng=rng,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def build_external_mechanism_views(
    s2c: pd.DataFrame,
    da_runs: dict[int, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aggregate-only paired views for S2C, ADB, and DA-ADB.

    The external methods are never merged into the fair MiniLM ranking.  This
    view only counts aligned test rows after verifying the source text/label
    order against the protocol view.  It intentionally exports no sample IDs
    or raw text.
    """
    adb = pd.read_csv(ADB_PER_SEED)
    adb = adb[
        adb["method"].eq("ADB")
        & adb["valid_semantic_metrics"].astype(bool)
        & adb["dataset"].eq("stackoverflow")
        & adb["kir"].eq(0.50)
        & adb["seed"].isin(SEEDS)
    ].copy()
    if set(adb["seed"].astype(int)) != set(SEEDS) or adb.duplicated("seed").any():
        raise RuntimeError("ADB StackOverflow KIR=.50 three-seed rows are incomplete")
    s2c = s2c[
        s2c["dataset"].eq("stackoverflow")
        & s2c["kir"].eq(0.50)
        & s2c["method"].eq("trainable_k1")
        & s2c["seed"].isin(SEEDS)
    ].copy()
    if set(s2c["seed"].astype(int)) != set(SEEDS) or s2c.duplicated("seed").any():
        raise RuntimeError("S2C StackOverflow KIR=.50 three-seed rows are incomplete")

    s2c_by_seed = {int(row["seed"]): row for _, row in s2c.iterrows()}
    adb_by_seed = {int(row["seed"]): row for _, row in adb.iterrows()}
    transition_rows: list[dict] = []
    state_rows: list[dict] = []
    intent_rows: list[dict] = []
    alignment_rows: list[dict] = []

    for seed in SEEDS:
        view = load_view(seed)
        s2c_pred, s2c_audit = load_s2c_predictions(s2c_by_seed[seed], view)
        adb_pred, adb_audit = load_adb_predictions(adb_by_seed[seed], view)
        da_pred, da_audit = load_da_predictions(da_runs[seed], view)
        gold = [str(item["evaluation_label"]) for item in view]
        state_by_method = {
            "S2C-Trainable-K1": [
                classify(label == "oos", pred == "oos", label, pred)
                for label, pred in zip(gold, s2c_pred)
            ],
            "ADB": [
                classify(label == "oos", pred == "oos", label, pred)
                for label, pred in zip(gold, adb_pred)
            ],
            "DA-ADB": [
                classify(label == "oos", pred == "oos", label, pred)
                for label, pred in zip(gold, da_pred)
            ],
        }
        n_known = sum(label != "oos" for label in gold)
        n_oos = sum(label == "oos" for label in gold)
        for method, states in state_by_method.items():
            for state in STATES:
                count = int(sum(item == state for item in states))
                state_rows.append(
                    {
                        "dataset": "stackoverflow",
                        "kir": 0.50,
                        "seed": seed,
                        "method": method,
                        "state": state,
                        "count": count,
                        "n_test": len(gold),
                        "n_known": n_known,
                        "n_oos": n_oos,
                        "rate_all": count / len(gold),
                        "rate_known": count / n_known if state.startswith("known_") else np.nan,
                        "rate_oos": count / n_oos if state.startswith("oos_") else np.nan,
                    }
                )

        comparisons = {
            "S2C→ADB": (state_by_method["S2C-Trainable-K1"], state_by_method["ADB"]),
            "S2C→DA-ADB": (state_by_method["S2C-Trainable-K1"], state_by_method["DA-ADB"]),
            "ADB→DA-ADB": (state_by_method["ADB"], state_by_method["DA-ADB"]),
        }
        for comparison, (left, right) in comparisons.items():
            for left_state in STATES:
                for right_state in STATES:
                    count = int(
                        sum(
                            a == left_state and b == right_state
                            for a, b in zip(left, right)
                        )
                    )
                    transition_rows.append(
                        {
                            "dataset": "stackoverflow",
                            "kir": 0.50,
                            "seed": seed,
                            "comparison": comparison,
                            "from_state": left_state,
                            "to_state": right_state,
                            "count": count,
                            "rate_all": count / len(gold),
                        }
                    )

        intents = sorted({label for label in gold if label != "oos"})
        by_intent = {
            intent: {
                "n_known": 0,
                "n_oos": n_oos,
                **{
                    f"{method_key}_{state}": 0
                    for method_key in ("s2c", "adb", "da_adb")
                    for state in ("known_correct", "known_wrong", "known_rejected")
                },
                **{
                    f"{method_key}_oos_false_accept": 0
                    for method_key in ("s2c", "adb", "da_adb")
                },
            }
            for intent in intents
        }
        method_predictions = {
            "s2c": s2c_pred,
            "adb": adb_pred,
            "da_adb": da_pred,
        }
        for index, label in enumerate(gold):
            if label != "oos":
                by_intent[label]["n_known"] += 1
                for method_key, predictions in method_predictions.items():
                    state = classify(False, predictions[index] == "oos", label, predictions[index])
                    by_intent[label][f"{method_key}_{state}"] += 1
            else:
                for method_key, predictions in method_predictions.items():
                    predicted = predictions[index]
                    if predicted != "oos" and predicted in by_intent:
                        by_intent[predicted][f"{method_key}_oos_false_accept"] += 1
        for intent, values in by_intent.items():
            row = {
                "dataset": "stackoverflow",
                "kir": 0.50,
                "seed": seed,
                "intent": intent,
                **values,
            }
            for method_key in ("s2c", "adb", "da_adb"):
                known_denominator = max(1, values["n_known"])
                oos_denominator = max(1, n_oos)
                row[f"{method_key}_known_rejected_rate"] = (
                    values[f"{method_key}_known_rejected"] / known_denominator
                )
                row[f"{method_key}_oos_false_accept_rate"] = (
                    values[f"{method_key}_oos_false_accept"] / oos_denominator
                )
            row["adb_minus_s2c_known_rejected_pp"] = 100.0 * (
                row["adb_known_rejected_rate"] - row["s2c_known_rejected_rate"]
            )
            row["da_adb_minus_s2c_known_rejected_pp"] = 100.0 * (
                row["da_adb_known_rejected_rate"] - row["s2c_known_rejected_rate"]
            )
            row["adb_minus_s2c_oos_false_accept_pp"] = 100.0 * (
                row["adb_oos_false_accept_rate"] - row["s2c_oos_false_accept_rate"]
            )
            row["da_adb_minus_s2c_oos_false_accept_pp"] = 100.0 * (
                row["da_adb_oos_false_accept_rate"] - row["s2c_oos_false_accept_rate"]
            )
            intent_rows.append(row)

        alignment_rows.append(
            {
                "dataset": "stackoverflow",
                "kir": 0.50,
                "seed": seed,
                "n_test": len(view),
                "n_known": n_known,
                "n_oos": n_oos,
                "test_sample_ids_sha256": s2c_audit["test_sample_ids_sha256"],
                "s2c_prediction_sha256": s2c_audit["prediction_sha256"],
                "s2c_manifest_sha256": s2c_audit["run_manifest_sha256"],
                "s2c_sample_id_order_match": s2c_audit["sample_id_order_match"],
                "s2c_true_label_match": s2c_audit["true_label_match"],
                "s2c_test_used_for_selection": s2c_audit["test_used_for_selection"],
                "adb_prediction_sha256": adb_audit["prediction_sha256"],
                "adb_manifest_sha256": adb_audit["run_manifest_sha256"],
                "adb_text_order_match": adb_audit["text_order_match"],
                "adb_label_order_match": adb_audit["label_order_match"],
                "adb_true_label_match": adb_audit["true_label_match"],
                "adb_test_used_for_selection": adb_audit["test_used_for_selection"],
                "da_adb_prediction_sha256": da_audit["prediction_sha256"],
                "da_adb_manifest_sha256": da_audit["run_manifest_sha256"],
                "da_adb_text_order_match": da_audit["text_order_match"],
                "da_adb_label_order_match": da_audit["label_order_match"],
                "da_adb_true_label_match": da_audit["true_label_match"],
                "da_adb_test_used_for_selection": da_audit["test_used_for_selection"],
                "aligned": True,
            }
        )

    transitions = pd.DataFrame(transition_rows)
    state_composition = pd.DataFrame(state_rows)
    intent_per_seed = pd.DataFrame(intent_rows)
    alignment = pd.DataFrame(alignment_rows)
    return transitions, state_composition, intent_per_seed, alignment


def make_mechanism_figures(
    transitions: pd.DataFrame,
    state_composition: pd.DataFrame,
    intent_per_seed: pd.DataFrame,
) -> None:
    """Create the external-backbone error mechanism figures."""
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 9, "axes.titlesize": 11, "figure.dpi": 140, "font.family": "DejaVu Sans"}
    )
    state_ticks = [STATE_LABELS[state] for state in STATES]

    # Three transition matrices expose whether a baseline changes the type of
    # error, not only the final metric.  Percentages are over all 6,000 rows.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, comparison in zip(axes, ["S2C→ADB", "S2C→DA-ADB", "ADB→DA-ADB"]):
        frame = transitions[transitions["comparison"].eq(comparison)]
        pivot = frame.groupby(["from_state", "to_state"], as_index=False)["count"].sum().pivot(
            index="from_state", columns="to_state", values="count"
        ).reindex(index=STATES, columns=STATES, fill_value=0)
        values = pivot.to_numpy(dtype=float)
        total = values.sum()
        ax.imshow(values / max(1.0, total) * 100.0, cmap="YlGnBu", vmin=0, aspect="auto")
        ax.set_xticks(range(len(STATES)), state_ticks, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(len(STATES)), state_ticks, fontsize=7)
        ax.set_xlabel("to")
        ax.set_ylabel("from")
        ax.set_title(comparison)
        for i in range(len(STATES)):
            for j in range(len(STATES)):
                value = values[i, j] / max(1.0, total) * 100.0
                if value > 0.05:
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7)
    fig.suptitle("External-backbone error transitions: StackOverflow KIR=.50", fontsize=13)
    fig.savefig(FIG / "external_error_transition_heatmaps.png", dpi=220)
    plt.close(fig)

    # The state signature shows the coverage/rejection trade-off alongside the
    # transition matrices, while retaining the three-seed aggregate contract.
    composition = (
        state_composition.groupby(["method", "state"], as_index=False)["count"].sum()
    )
    totals = composition.groupby("method")["count"].transform("sum")
    composition["rate"] = composition["count"] / totals * 100.0
    pivot = composition.pivot(index="method", columns="state", values="rate").reindex(
        index=["S2C-Trainable-K1", "ADB", "DA-ADB"], columns=STATES, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    bottom = np.zeros(len(pivot))
    palette = ["#3a923a", "#c7a008", "#d95f02", "#377eb8", "#984ea3"]
    for state, color in zip(STATES, palette):
        values = pivot[state].to_numpy(dtype=float)
        ax.bar(pivot.index, values, bottom=bottom, label=STATE_LABELS[state], color=color)
        bottom += values
    ax.set_ylabel("share of all test samples (%)")
    ax.set_title("External method error signature: StackOverflow KIR=.50")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(0, 100)
    fig.savefig(FIG / "external_error_signature.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Intent fingerprint: positive values mean the external method contributes
    # more of that error type than S2C.  OOS false accepts are assigned to the
    # predicted Known intent, so this panel is not a gold-intent error rate.
    summary = (
        intent_per_seed.groupby(["intent"], as_index=False)
        .agg(
            adb_known_mean=("adb_minus_s2c_known_rejected_pp", "mean"),
            adb_known_std=("adb_minus_s2c_known_rejected_pp", "std"),
            da_known_mean=("da_adb_minus_s2c_known_rejected_pp", "mean"),
            da_known_std=("da_adb_minus_s2c_known_rejected_pp", "std"),
            adb_oos_mean=("adb_minus_s2c_oos_false_accept_pp", "mean"),
            adb_oos_std=("adb_minus_s2c_oos_false_accept_pp", "std"),
            da_oos_mean=("da_adb_minus_s2c_oos_false_accept_pp", "mean"),
            da_oos_std=("da_adb_minus_s2c_oos_false_accept_pp", "std"),
        )
        .sort_values("da_known_mean")
    )
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True, sharex=True)
    for ax, prefix, title, ylabel in (
        (
            axes[0],
            "known",
            "Known rejection difference",
            "external − S2C Known rejection (pp)",
        ),
        (
            axes[1],
            "oos",
            "OOS false-accept allocation difference",
            "external − S2C OOS false accept (pp)",
        ),
    ):
        ax.errorbar(
            x - 0.06,
            summary[f"adb_{prefix}_mean"],
            yerr=summary[f"adb_{prefix}_std"].fillna(0),
            fmt="o",
            capsize=3,
            color="#d95f02",
            label="ADB",
        )
        ax.errorbar(
            x + 0.06,
            summary[f"da_{prefix}_mean"],
            yerr=summary[f"da_{prefix}_std"].fillna(0),
            fmt="s",
            capsize=3,
            color="#7570b3",
            label="DA-ADB",
        )
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x, summary["intent"], rotation=50, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Intent-level external error fingerprint: StackOverflow KIR=.50", fontsize=13)
    fig.savefig(FIG / "external_intent_error_fingerprint.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    da_rows = [da_metrics(RUN_ROOT / f"seed_{seed}") for seed in SEEDS]
    s2c = pd.read_csv(S2C_PER_SEED)
    s2c = s2c[
        (s2c["dataset"] == "stackoverflow")
        & (s2c["kir"] == 0.50)
        & (s2c["method"] == "trainable_k1")
        & (s2c["seed"].isin(SEEDS))
    ].copy()
    if set(s2c["seed"].astype(int)) != set(SEEDS):
        raise RuntimeError("matched S2C Trainable K=1 rows are incomplete")
    s2c_all = s2c.copy()
    s2c = s2c.set_index("seed")
    metrics = [
        "oos_f1",
        "f1_all",
        "f1_known",
        "accuracy",
        "known_recall",
        "false_acceptance",
        "false_rejection",
    ]
    rows = []
    paired = []
    for da in da_rows:
        seed = int(da["seed"])
        s = s2c.loc[seed]
        rows.append({"method": "DA-ADB", "contract": da["contract"], "seed": seed, **{m: da[m] for m in metrics}})
        rows.append(
            {
                "method": "S2C-Trainable-K1",
                "contract": "protocol_v2 Known-only MiniLM fair",
                "seed": seed,
                **{
                    m: float(
                        s["f1_k"] if m == "f1_known" else
                        s["false_accept_rate"] if m == "false_acceptance" else
                        s["false_reject_rate"] if m == "false_rejection" else
                        s[m]
                    )
                    for m in metrics
                },
            }
        )
        paired.append(
            {
                "seed": seed,
                **{
                    f"trainable_minus_da_{m}": (
                        float(
                            s["f1_k"] if m == "f1_known" else
                            s["false_accept_rate"] if m == "false_acceptance" else
                            s["false_reject_rate"] if m == "false_rejection" else
                            s[m]
                        ) - da[m]
                    )
                    for m in metrics
                },
            }
        )
    per_seed = pd.DataFrame(rows)
    pair_df = pd.DataFrame(paired)
    summary_rows = []
    for method, group in per_seed.groupby("method", sort=False):
        row = {"method": method, "contract": group["contract"].iloc[0], "n_seeds": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_ci(values)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    pair_summary = {
        "comparison": "S2C-Trainable-K1 minus DA-ADB; same seed, different training contract",
        "n_seeds": len(pair_df),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": 20_000,
    }
    for metric in metrics:
        values = pair_df[f"trainable_minus_da_{metric}"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(values)
        pair_summary[f"{metric}_mean_delta"] = float(values.mean())
        pair_summary[f"{metric}_std_delta"] = float(values.std(ddof=1))
        pair_summary[f"{metric}_ci95_low"] = lo
        pair_summary[f"{metric}_ci95_high"] = hi
        pair_summary[f"{metric}_trainable_wins"] = int(np.sum(values > 0))
        pair_summary[f"{metric}_da_adb_wins"] = int(np.sum(values < 0))
        pair_summary[f"{metric}_ties"] = int(np.sum(values == 0))

    da_runs = {
        int(row["seed"]): Path(row["run_manifest"]).parent
        for row in da_rows
    }
    transitions, state_composition, intent_per_seed, alignment = build_external_mechanism_views(
        s2c_all,
        da_runs,
    )
    transition_summary = (
        transitions.groupby(["dataset", "kir", "comparison", "from_state", "to_state"], as_index=False)
        .agg(count=("count", "sum"), n_seeds=("seed", "nunique"))
    )
    transition_summary["rate_all"] = transition_summary["count"] / transition_summary.groupby(
        ["dataset", "kir", "comparison"]
    )["count"].transform("sum")
    state_summary = (
        state_composition.groupby(["dataset", "kir", "method", "state"], as_index=False)
        .agg(
            count_mean=("count", "mean"),
            count_std=("count", "std"),
            rate_all_mean=("rate_all", "mean"),
            rate_known_mean=("rate_known", "mean"),
            rate_oos_mean=("rate_oos", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    intent_summary = (
        intent_per_seed.groupby(["dataset", "kir", "intent"], as_index=False)
        .agg(
            n_known=("n_known", "mean"),
            n_oos=("n_oos", "mean"),
            adb_minus_s2c_known_rejected_pp=("adb_minus_s2c_known_rejected_pp", "mean"),
            adb_minus_s2c_known_rejected_std_pp=("adb_minus_s2c_known_rejected_pp", "std"),
            da_adb_minus_s2c_known_rejected_pp=("da_adb_minus_s2c_known_rejected_pp", "mean"),
            da_adb_minus_s2c_known_rejected_std_pp=("da_adb_minus_s2c_known_rejected_pp", "std"),
            adb_minus_s2c_oos_false_accept_pp=("adb_minus_s2c_oos_false_accept_pp", "mean"),
            adb_minus_s2c_oos_false_accept_std_pp=("adb_minus_s2c_oos_false_accept_pp", "std"),
            da_adb_minus_s2c_oos_false_accept_pp=("da_adb_minus_s2c_oos_false_accept_pp", "mean"),
            da_adb_minus_s2c_oos_false_accept_std_pp=("da_adb_minus_s2c_oos_false_accept_pp", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(OUT / "per_seed.csv", index=False)
    summary.to_csv(OUT / "summary_mean_std_ci.csv", index=False)
    pair_df.to_csv(OUT / "paired_deltas.csv", index=False)
    (OUT / "paired_summary.json").write_text(
        json.dumps(pair_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    transitions.to_csv(OUT / "external_error_transitions_per_seed.csv", index=False)
    transition_summary.to_csv(OUT / "external_error_transitions_summary.csv", index=False)
    state_composition.to_csv(OUT / "external_state_composition_per_seed.csv", index=False)
    state_summary.to_csv(OUT / "external_state_composition_summary.csv", index=False)
    intent_per_seed.to_csv(OUT / "external_intent_error_per_seed.csv", index=False)
    intent_summary.to_csv(OUT / "external_intent_error_summary.csv", index=False)
    alignment.to_csv(OUT / "external_alignment_audit.csv", index=False)
    make_mechanism_figures(transitions, state_composition, intent_per_seed)
    manifest = {
        "schema_version": 2,
        "experiment": "da_adb_current_protocol_summary_v1",
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seeds": list(SEEDS),
        "source_da_runs": [row["run_manifest"] for row in da_rows],
        "source_s2c_per_seed": str(S2C_PER_SEED.resolve()),
        "source_s2c_per_seed_sha256": sha256_file(S2C_PER_SEED),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": 20_000,
        "ranking_policy": "paired descriptive comparison only; no cross-contract SOTA ranking",
        "mechanism_scope": {
            "methods": ["S2C-Trainable-K1", "ADB", "DA-ADB"],
            "unit": "StackOverflow/KIR=.50/seed={42,87,100}",
            "selection_used_test_oos": False,
            "external_selection_audit": "not_declared_external",
            "raw_text_exported": False,
            "sample_ids_exported": False,
        },
        "mechanism_outputs": [
            "external_alignment_audit.csv",
            "external_error_transitions_per_seed.csv",
            "external_error_transitions_summary.csv",
            "external_state_composition_per_seed.csv",
            "external_state_composition_summary.csv",
            "external_intent_error_per_seed.csv",
            "external_intent_error_summary.csv",
            "../figures/da_adb_current_protocol_summary_v1/external_error_transition_heatmaps.png",
            "../figures/da_adb_current_protocol_summary_v1/external_error_signature.png",
            "../figures/da_adb_current_protocol_summary_v1/external_intent_error_fingerprint.png",
        ],
        "alignment_audit": alignment.to_dict(orient="records"),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"DA-ADB": "#d95f02", "S2C-Trainable-K1": "#1b9e77"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(SEEDS))
    for ax, metric, title in zip(axes, ["oos_f1", "f1_all"], ["OOS F1", "F1-All"]):
        for method in ["DA-ADB", "S2C-Trainable-K1"]:
            g = per_seed[per_seed["method"] == method].sort_values("seed")
            ax.plot(x, g[metric] * 100, marker="o", linewidth=2, label=method, color=colors[method])
        ax.set_xticks(x, [str(seed) for seed in SEEDS])
        ax.set_xlabel("seed")
        ax.set_ylabel("percent")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("StackOverflow protocol_v2: DA-ADB vs S2C Trainable K=1", fontsize=13)
    fig.savefig(FIG / "current_protocol_seed_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for method in ["DA-ADB", "S2C-Trainable-K1"]:
        g = per_seed[per_seed["method"] == method].sort_values("seed")
        ax.scatter(g["false_acceptance"] * 100, g["oos_f1"] * 100, s=70, label=method, color=colors[method])
        for _, row in g.iterrows():
            ax.annotate(str(int(row["seed"])), (row["false_acceptance"] * 100, row["oos_f1"] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("false acceptance (%)")
    ax.set_ylabel("OOS F1 (%)")
    ax.set_title("Known/OOS operating-point comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(FIG / "current_protocol_tradeoff.png", dpi=180)
    plt.close(fig)

    da_summary = summary[summary.method == "DA-ADB"].iloc[0]
    s2c_summary = summary[summary.method == "S2C-Trainable-K1"].iloc[0]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# DA-ADB 当前协议三 seed 汇总 V1

## 运行范围

- 数据集：StackOverflow；KIR=`0.50`；seed=`42, 87, 100`。
- DA-ADB：BERT/TextOIR 外部兼容合同，使用当前 protocol_v2 的独立 split 根和 seed-specific Known 列表。
- S2C：同一 seed 的 Known-only MiniLM Trainable K=1 fair 行。
- 测试 OOS 只用于最终评价，不参与训练、epoch、阈值或参数选择。
- 两者仍不是同骨干/同训练合同，本报告只作配对描述和合同审计。

## 当前协议结果

| 方法 | OOS F1 | F1-All | F1-Known | Accuracy | Known Recall | FA | FR |
|---|---:|---:|---:|---:|---:|---:|---:|
| DA-ADB | {da_summary['oos_f1_mean']*100:.2f} ± {da_summary['oos_f1_std']*100:.2f} | {da_summary['f1_all_mean']*100:.2f} ± {da_summary['f1_all_std']*100:.2f} | {da_summary['f1_known_mean']*100:.2f} ± {da_summary['f1_known_std']*100:.2f} | {da_summary['accuracy_mean']*100:.2f} ± {da_summary['accuracy_std']*100:.2f} | {da_summary['known_recall_mean']*100:.2f} ± {da_summary['known_recall_std']*100:.2f} | {da_summary['false_acceptance_mean']*100:.2f} ± {da_summary['false_acceptance_std']*100:.2f} | {da_summary['false_rejection_mean']*100:.2f} ± {da_summary['false_rejection_std']*100:.2f} |
| S2C Trainable K=1 | {s2c_summary['oos_f1_mean']*100:.2f} ± {s2c_summary['oos_f1_std']*100:.2f} | {s2c_summary['f1_all_mean']*100:.2f} ± {s2c_summary['f1_all_std']*100:.2f} | {s2c_summary['f1_known_mean']*100:.2f} ± {s2c_summary['f1_known_std']*100:.2f} | {s2c_summary['accuracy_mean']*100:.2f} ± {s2c_summary['accuracy_std']*100:.2f} | {s2c_summary['known_recall_mean']*100:.2f} ± {s2c_summary['known_recall_std']*100:.2f} | {s2c_summary['false_acceptance_mean']*100:.2f} ± {s2c_summary['false_acceptance_std']*100:.2f} | {s2c_summary['false_rejection_mean']*100:.2f} ± {s2c_summary['false_rejection_std']*100:.2f} |

## 配对差值：S2C Trainable K=1 − DA-ADB

| 指标 | 平均差值 | 95% bootstrap CI | S2C 胜出/DA-ADB 胜出/平局 |
|---|---:|---:|---:|
"""
        + "\n".join(
            f"| {metric} | {pair_summary[f'{metric}_mean_delta']*100:+.2f}pp | [{pair_summary[f'{metric}_ci95_low']*100:+.2f}, {pair_summary[f'{metric}_ci95_high']*100:+.2f}]pp | {pair_summary[f'{metric}_trainable_wins']}/{pair_summary[f'{metric}_da_adb_wins']}/{pair_summary[f'{metric}_ties']} |"
            for metric in metrics
        )
        + f"""

## 解释边界

当前协议下 DA-ADB 三个 seed 的 OOS F1 分别为
`{', '.join(f'{x:.2f}%' for x in per_seed.loc[per_seed.method == 'DA-ADB', 'oos_f1'].mul(100))}`，
S2C Trainable K=1 分别为
`{', '.join(f'{x:.2f}%' for x in per_seed.loc[per_seed.method == 'S2C-Trainable-K1', 'oos_f1'].mul(100))}`。
因此旧兼容单格 `90.90%` 不能代表当前协议 DA-ADB；当前三 seed 均值也不能与历史论文表或 DCLOOS reduced 结果直接排名。

当前证据只支持：在这三个 StackOverflow protocol_v2 工作点上，S2C Trainable K=1 的 OOS F1 和 F1-All 更高，
而 DA-ADB 的结果方差较大；这仍同时包含 MiniLM/BERT 表示、训练目标和外部适配层差异。

## 样本级机制证据

外部方法的深层图不把 BERT embedding 与 MiniLM embedding 强行放在同一坐标系，而是沿着同一测试样本追踪错误状态：

- `external_error_transition_heatmaps.png`：S2C、ADB、DA-ADB 之间的五状态错误转移；
- `external_error_signature.png`：三种方法的 Known coverage、Known wrong、OOS rejection 和 OOS false accept 构成；
- `external_intent_error_fingerprint.png`：按 Known 真实 intent 和 OOS 被吸收的预测 intent 定位错误集中在哪些局部区域。

这些图只能解释“哪些样本状态和 intent 上外部方法付出更多代价”，不能把差异归因成单一的表示、损失或边界原因；外部运行的 test-selection 字段没有声明，因此仍保留在 external-backbone 层。

## 证据文件

- `results/analysis/da_adb_current_protocol_summary_v1/per_seed.csv`
- `results/analysis/da_adb_current_protocol_summary_v1/summary_mean_std_ci.csv`
- `results/analysis/da_adb_current_protocol_summary_v1/paired_deltas.csv`
- `figures/da_adb_current_protocol_summary_v1/current_protocol_seed_metrics.png`
- `figures/da_adb_current_protocol_summary_v1/current_protocol_tradeoff.png`
- `figures/da_adb_current_protocol_summary_v1/external_error_transition_heatmaps.png`
- `figures/da_adb_current_protocol_summary_v1/external_error_signature.png`
- `figures/da_adb_current_protocol_summary_v1/external_intent_error_fingerprint.png`
- `results/analysis/da_adb_current_protocol_summary_v1/external_alignment_audit.csv`
- 旧/新合同拆分：`docs/archive/analysis/DA_ADB_CONTRACT_COMPARISON_V1.md`
""",
        encoding="utf-8",
    )
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
