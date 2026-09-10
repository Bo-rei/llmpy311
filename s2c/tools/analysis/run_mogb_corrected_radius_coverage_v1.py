#!/usr/bin/env python3
"""Known-only radius coverage attribution for corrected-loss BERT-MOGB.

This is a pure post-hoc boundary audit on top of an existing corrected-loss
MOGB checkpoint.  It never retrains BERT, never re-selects epochs via test,
and never uses test OOS to choose any radius work point.

Workflow:
1. Load the registered corrected-loss checkpoint and official data snapshot.
2. Rebuild selected balls/centers/radii with the exact reproduction helpers.
3. Verify the default replay reproduces the stored metrics and selected-ball
   summary within float tolerance.
4. Use only dev/eval Known samples to calibrate global radius multipliers for
   target Known coverage {0.80, 0.85, 0.90, 0.95}.
5. Evaluate the frozen test set once per pre-registered work point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments import run_mogb_exact_reproduction as exact  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "baselines" / "mogb_corrected_subcentroid_loss_v1.yaml"
RESULT_ROOT = ROOT / "results" / "analysis" / "mogb_corrected_radius_coverage_v1"
FIGURE_ROOT = ROOT / "figures" / "mogb_corrected_radius_coverage_v1"
REPORT_PATH = ROOT / "docs" / "analysis" / "MOGB_CORRECTED_RADIUS_COVERAGE_V1.md"
TARGET_COVERAGES = (0.80, 0.85, 0.90, 0.95)
FLOAT_RTOL = 1e-6
FLOAT_ATOL = 1e-6

# The analysis figures are Chinese-first.  Register the runtime font explicitly
# because isolated matplotlib caches do not always discover system TTC files.
_CJK_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
if _CJK_FONT.exists():
    font_manager.fontManager.addfont(str(_CJK_FONT))
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Unifont", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class RuntimeBundle:
    cfg: dict[str, Any]
    modules: dict[str, Any]
    manager: Any
    data: Any
    mode_dir: Path
    manifest: dict[str, Any]
    checkpoint_path: Path


@dataclass(frozen=True)
class ReconstructionCandidate:
    replay_seed: int
    numbers: list[int]
    result: list[np.ndarray]
    centers_np: list[np.ndarray]
    radii_np: list[float]
    balls: pd.DataFrame
    train_score: float
    score_breakdown: dict[str, float]


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_ratio_quantile(ratios: np.ndarray, target: float) -> float:
    """Return the smallest multiplier whose empirical coverage reaches target."""
    if not 0.0 < target <= 1.0:
        raise ValueError(f"invalid target coverage: {target}")
    if ratios.size == 0:
        raise ValueError("empty dev ratio table")
    ordered = np.sort(np.asarray(ratios, dtype=np.float64))
    index = max(0, math.ceil(target * ordered.size) - 1)
    return float(ordered[index])


def compute_coverage(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(scores, dtype=np.float64) <= threshold))


def collect_features_from_loader(manager: Any, loader: Any) -> tuple[torch.Tensor, torch.Tensor]:
    previous = manager.model.training
    manager.model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, input_mask, segment_ids, label_ids = (item.to(manager.device) for item in batch)
            feature_bank, _ = manager.model(input_ids, segment_ids, input_mask, mode="eval")
            features.append(feature_bank.detach().cpu())
            labels.append(label_ids.detach().cpu())
    manager.model.train(previous)
    if not features:
        return torch.empty((0, 768), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def score_with_radii(
    manager: Any,
    loader: Any,
    centroids: torch.Tensor,
    radii: torch.Tensor,
    labels: torch.Tensor,
    unseen_token_id: int,
) -> dict[str, Any]:
    manager.model.eval()
    true: list[int] = []
    pred: list[int] = []
    nearest_ball: list[int] = []
    nearest_distance: list[float] = []
    assigned_radius: list[float] = []
    accepted: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, input_mask, segment_ids, label_ids = (item.to(manager.device) for item in batch)
            features, _ = manager.model(input_ids, segment_ids, input_mask, mode="eval")
            distances = torch.cdist(features, centroids, p=2)
            nearest_values, nearest = distances.min(dim=1)
            radius_values = radii[nearest]
            predictions = labels[nearest].clone()
            accepted_mask = nearest_values < radius_values
            predictions[~accepted_mask] = unseen_token_id
            true.extend(label_ids.cpu().tolist())
            pred.extend(predictions.cpu().tolist())
            nearest_ball.extend(nearest.cpu().tolist())
            nearest_distance.extend(nearest_values.cpu().tolist())
            assigned_radius.extend(radius_values.cpu().tolist())
            accepted.extend(accepted_mask.to(torch.int64).cpu().tolist())
    y_true = np.asarray(true, dtype=np.int64)
    y_pred = np.asarray(pred, dtype=np.int64)
    metrics, _, _ = exact.evaluate_test(manager, _LoaderProxy(loader, unseen_token_id), centroids, radii, labels)
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "nearest_ball": np.asarray(nearest_ball, dtype=np.int64),
        "nearest_distance": np.asarray(nearest_distance, dtype=np.float64),
        "assigned_radius": np.asarray(assigned_radius, dtype=np.float64),
        "accepted": np.asarray(accepted, dtype=np.int64),
    }


class _LoaderProxy:
    """Minimal proxy to satisfy exact.evaluate_test's data contract."""

    def __init__(self, loader: Any, unseen_token_id: int) -> None:
        self.test_dataloader = loader
        self.unseen_token_id = unseen_token_id
        self.num_labels = unseen_token_id


def replay_ball_rows(results: list[np.ndarray], centers: list[np.ndarray], radii: list[float]) -> pd.DataFrame:
    return pd.DataFrame(exact.ball_rows(results, centers, radii)).sort_values("ball_id").reset_index(drop=True)


def compare_ball_rows(reference_rows: list[dict[str, Any]], observed: pd.DataFrame) -> dict[str, Any]:
    reference = pd.DataFrame(reference_rows).sort_values("ball_id").reset_index(drop=True)
    if len(reference) != len(observed):
        return {"passed": False, "reason": "ball_count_mismatch", "reference": len(reference), "observed": len(observed)}
    key_columns = ["ball_id", "majority_label", "sample_count"]
    if not reference[key_columns].equals(observed[key_columns]):
        return {"passed": False, "reason": "ball_identity_mismatch"}
    numeric_deltas = {}
    for column in ("purity", "radius", "center_l2"):
        delta = float(np.max(np.abs(reference[column].to_numpy(dtype=float) - observed[column].to_numpy(dtype=float))))
        numeric_deltas[column] = delta
    passed = all(delta <= 1e-6 for delta in numeric_deltas.values())
    return {"passed": passed, "reason": "ok" if passed else "numeric_mismatch", **numeric_deltas}


def per_ball_attribution(
    workpoint: str,
    nearest_ball: np.ndarray,
    accepted: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dev_nearest_ball: np.ndarray,
    dev_distances: np.ndarray,
    dev_radii: np.ndarray,
    balls: pd.DataFrame,
    multiplier: float,
    unseen_token_id: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dev_true_known = np.ones_like(dev_nearest_ball, dtype=bool)
    del dev_true_known
    for row in balls.to_dict(orient="records"):
        ball_id = int(row["ball_id"])
        dev_mask = dev_nearest_ball == ball_id
        test_mask = nearest_ball == ball_id
        rows.append(
            {
                "workpoint": workpoint,
                "ball_id": ball_id,
                "majority_label": int(row["majority_label"]),
                "sample_count": int(row["sample_count"]),
                "purity": float(row["purity"]),
                "default_radius": float(row["radius"]),
                "effective_radius": float(row["radius"] * multiplier),
                "dev_assigned_known_count": int(np.sum(dev_mask)),
                "dev_assigned_known_covered_count": int(np.sum(dev_mask & (dev_distances <= dev_radii * multiplier))),
                "dev_assigned_known_coverage": float(np.mean(dev_distances[dev_mask] <= dev_radii[dev_mask] * multiplier)) if np.any(dev_mask) else math.nan,
                "test_assigned_count": int(np.sum(test_mask)),
                "test_predicted_known_count": int(np.sum(test_mask & (accepted == 1))),
                "test_predicted_oos_count": int(np.sum(test_mask & (accepted == 0))),
                "test_false_accept_count": int(np.sum(test_mask & (accepted == 1) & (y_true == unseen_token_id))),
                "test_known_true_count": int(np.sum(test_mask & (y_true != unseen_token_id))),
                "test_known_rejected_count": int(np.sum(test_mask & (accepted == 0) & (y_true != unseen_token_id))),
            }
        )
    return pd.DataFrame(rows)


def reconstruction_objective(reference_rows: list[dict[str, Any]], observed: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Train-side similarity objective; never uses test metrics."""
    reference = pd.DataFrame(reference_rows).sort_values(["majority_label", "sample_count", "radius"]).reset_index(drop=True)
    observed = observed.sort_values(["majority_label", "sample_count", "radius"]).reset_index(drop=True)
    reference_counts = reference["sample_count"].to_numpy(dtype=np.float64)
    observed_counts = observed["sample_count"].to_numpy(dtype=np.float64)
    max_len = max(len(reference_counts), len(observed_counts))
    padded_ref = np.pad(reference_counts, (0, max_len - len(reference_counts)))
    padded_obs = np.pad(observed_counts, (0, max_len - len(observed_counts)))
    count_penalty = abs(len(reference) - len(observed))
    size_l1 = float(np.abs(padded_ref - padded_obs).sum())
    radius_mean_penalty = abs(reference["radius"].mean() - observed["radius"].mean()) if len(observed) else float("inf")
    purity_mean_penalty = abs(reference["purity"].mean() - observed["purity"].mean()) if len(observed) else float("inf")
    label_hist_ref = reference["majority_label"].value_counts().sort_index()
    label_hist_obs = observed["majority_label"].value_counts().sort_index()
    label_union = sorted(set(label_hist_ref.index).union(label_hist_obs.index))
    label_hist_penalty = float(
        sum(abs(int(label_hist_ref.get(label, 0)) - int(label_hist_obs.get(label, 0))) for label in label_union)
    )
    total = count_penalty * 1_000_000.0 + label_hist_penalty * 10_000.0 + size_l1 + radius_mean_penalty * 1_000.0 + purity_mean_penalty * 1_000.0
    return total, {
        "count_penalty": float(count_penalty),
        "label_hist_penalty": label_hist_penalty,
        "sample_count_l1": size_l1,
        "radius_mean_penalty": float(radius_mean_penalty),
        "purity_mean_penalty": float(purity_mean_penalty),
    }


def reconstruct_selected_balls(
    bundle: RuntimeBundle,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
) -> ReconstructionCandidate:
    args = exact.make_namespace(bundle.cfg, bundle.mode_dir)
    candidate_seeds = list(range(32)) + [42, 87, 100, 123]
    candidates: list[ReconstructionCandidate] = []
    for replay_seed in candidate_seeds:
        random.seed(replay_seed)
        np.random.seed(replay_seed)
        torch.manual_seed(replay_seed)
        numbers, result, centers_np, radii_np = exact.cluster_arrays(bundle.modules, args, train_features, train_labels, select=True)
        balls = replay_ball_rows(result, centers_np, radii_np)
        score, breakdown = reconstruction_objective(bundle.manifest["ball_rows"], balls)
        candidates.append(
            ReconstructionCandidate(
                replay_seed=replay_seed,
                numbers=numbers,
                result=result,
                centers_np=centers_np,
                radii_np=radii_np,
                balls=balls,
                train_score=score,
                score_breakdown=breakdown,
            )
        )
    candidates.sort(key=lambda item: (item.train_score, item.replay_seed))
    return candidates[0]


def build_runtime() -> RuntimeBundle:
    if not torch.cuda.is_available():
        raise RuntimeError("blocked_no_gpu")
    cfg = exact.load_config(CONFIG_PATH)
    cfg["_config_path"] = str(CONFIG_PATH)
    result_root = (ROOT / cfg["result_root"]).resolve()
    metrics_payload = json.loads((result_root / "final_metrics.json").read_text(encoding="utf-8"))
    mode_payload = metrics_payload["official_fixed"]
    mode_dir = Path(mode_payload["run_dir"])
    manifest = json.loads((mode_dir / "mode_manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = mode_dir / "checkpoints" / "best_checkpoint.pt"
    os.environ["MPLCONFIGDIR"] = str(RESULT_ROOT / "matplotlib_cache")
    (RESULT_ROOT / "matplotlib_cache").mkdir(parents=True, exist_ok=True)
    modules = exact._import_official_modules()
    exact.patch_cluster_device(modules["cluster"])
    exact.patch_cluster3_empty_ball_guard(modules["cluster3"])
    pre_data_seed = manifest["seed_contract"]["official_pre_data_seed"]
    exact.set_seeds(int(pre_data_seed))
    args = exact.make_namespace(cfg, mode_dir)
    args.seed = 0
    data = modules["dataloader"].Data(args)
    manager = modules["pretrain"].PretrainModelManager(args, data)
    exact.configure_subcentroid_loss(manager, cfg)
    checkpoint = torch.load(checkpoint_path, map_location=manager.device)
    manager.model.load_state_dict(checkpoint["model_state_dict"])
    return RuntimeBundle(
        cfg=cfg,
        modules=modules,
        manager=manager,
        data=data,
        mode_dir=mode_dir,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
    )


def run_analysis() -> dict[str, Any]:
    bundle = build_runtime()
    manager = bundle.manager
    data = bundle.data
    started = time.time()

    train_features, train_labels = exact.collect_features(manager, data, training_mode=False)
    candidate = reconstruct_selected_balls(bundle, train_features, train_labels)
    centers_np = candidate.centers_np
    radii_np = candidate.radii_np
    observed_balls = candidate.balls
    ball_check = compare_ball_rows(bundle.manifest["ball_rows"], observed_balls)
    strict_replay = bool(ball_check["passed"])
    replay_status = "strict_selected_ball_replay" if strict_replay else "deterministic_fixed_checkpoint_reconstruction"

    centroids = torch.tensor([center[1:] for center in centers_np], dtype=torch.float32, device=manager.device)
    default_radii = torch.tensor(radii_np, dtype=torch.float32, device=manager.device)
    ball_labels = torch.tensor([int(center[0]) for center in centers_np], dtype=torch.long, device=manager.device)

    default_eval = score_with_radii(manager, data.test_dataloader, centroids, default_radii, ball_labels, data.unseen_token_id)
    reference_metrics = bundle.manifest["metrics"]
    metric_deltas = {
        key: abs(float(default_eval["metrics"][key]) - float(reference_metrics[key]))
        for key in ("Accuracy", "F1-All", "F1-K", "F1-U", "Known Recall", "OOS Precision", "OOS Recall")
    }
    metric_replay_equivalent = max(metric_deltas.values()) <= 1e-6

    dev_features, dev_labels = collect_features_from_loader(manager, data.eval_dataloader)
    dev_features = dev_features.to(manager.device)
    dev_distances = torch.cdist(dev_features, centroids, p=2)
    dev_nearest_distance, dev_nearest_ball = dev_distances.min(dim=1)
    dev_assigned_radius = default_radii[dev_nearest_ball]
    dev_ratios = (dev_nearest_distance / torch.clamp(dev_assigned_radius, min=1e-12)).detach().cpu().numpy()

    workpoint_rows: list[dict[str, Any]] = []
    per_ball_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []

    workpoints = [("default_mean", math.nan, 1.0)]
    for coverage in TARGET_COVERAGES:
        multiplier = stable_ratio_quantile(dev_ratios, coverage)
        workpoints.append((f"dev_known_coverage_{coverage:.2f}", coverage, multiplier))
        threshold_rows.append(
            {
                "workpoint": f"dev_known_coverage_{coverage:.2f}",
                "target_dev_known_coverage": coverage,
                "radius_multiplier": multiplier,
                "achieved_dev_known_coverage": compute_coverage(dev_ratios, multiplier),
            }
        )

    for workpoint, target, multiplier in workpoints:
        scaled_radii = default_radii * float(multiplier)
        test_eval = score_with_radii(manager, data.test_dataloader, centroids, scaled_radii, ball_labels, data.unseen_token_id)
        metrics = test_eval["metrics"]
        workpoint_rows.append(
            {
                "workpoint": workpoint,
                "target_dev_known_coverage": target,
                "radius_multiplier": multiplier,
                "achieved_dev_known_coverage": compute_coverage(dev_ratios, multiplier),
                "selected_ball_count": int(len(observed_balls)),
                "best_epoch": int(bundle.manifest["best_epoch"]),
                **metrics,
            }
        )
        per_ball = per_ball_attribution(
            workpoint=workpoint,
            nearest_ball=test_eval["nearest_ball"],
            accepted=test_eval["accepted"],
            y_true=test_eval["y_true"],
            y_pred=test_eval["y_pred"],
            dev_nearest_ball=dev_nearest_ball.detach().cpu().numpy(),
            dev_distances=dev_nearest_distance.detach().cpu().numpy(),
            dev_radii=dev_assigned_radius.detach().cpu().numpy(),
            balls=observed_balls,
            multiplier=float(multiplier),
            unseen_token_id=data.unseen_token_id,
        )
        per_ball_frames.append(per_ball)

    workpoint_frame = pd.DataFrame(workpoint_rows)
    threshold_frame = pd.DataFrame(threshold_rows)
    per_ball_frame = pd.concat(per_ball_frames, ignore_index=True)

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_csv(RESULT_ROOT / "workpoint_metrics.csv", workpoint_frame)
    atomic_csv(RESULT_ROOT / "dev_known_radius_thresholds.csv", threshold_frame)
    atomic_csv(RESULT_ROOT / "per_ball_radius_attribution.csv", per_ball_frame)
    atomic_csv(RESULT_ROOT / "selected_ball_replay.csv", observed_balls)
    atomic_json(
        RESULT_ROOT / "replay_check.json",
        {
            "selected_ball_replay": ball_check,
            "replay_status": replay_status,
            "replay_seed": candidate.replay_seed,
            "train_side_reconstruction_score": candidate.train_score,
            "train_side_reconstruction_breakdown": candidate.score_breakdown,
            "default_metric_replay_equivalent": metric_replay_equivalent,
            "default_metric_max_abs_delta": max(metric_deltas.values()),
            "default_metric_abs_delta": metric_deltas,
            "reference_checkpoint_sha256": bundle.manifest["checkpoint_sha256"],
            "checkpoint_sha256": sha256_file(bundle.checkpoint_path),
        },
    )

    plot_results(workpoint_frame, per_ball_frame)
    report = build_report(
        workpoint_frame,
        threshold_frame,
        per_ball_frame,
        bundle,
        ball_check,
        metric_deltas,
        replay_status,
        candidate.replay_seed,
    )
    atomic_text(REPORT_PATH, report)

    manifest = {
        "experiment_id": "mogb_corrected_radius_coverage_v1",
        "status": "complete",
        "replay_status": replay_status,
        "strict_selected_ball_replay": strict_replay,
        "replay_seed": candidate.replay_seed,
        "default_metric_replay_equivalent": metric_replay_equivalent,
        "dataset": bundle.cfg["dataset"],
        "kir": float(bundle.cfg["kir"]),
        "seed": int(bundle.cfg["seed"]),
        "source_experiment_id": bundle.cfg["experiment_id"],
        "source_mode": "official_fixed",
        "selected_ball_count": int(len(observed_balls)),
        "reference_selected_ball_count": int(len(bundle.manifest["ball_rows"])),
        "workpoints": [row["workpoint"] for row in workpoint_rows],
        "target_coverages": TARGET_COVERAGES,
        "test_used_for_selection": False,
        "dev_known_sample_count": int(len(dev_ratios)),
        "elapsed_seconds": time.time() - started,
        "script_sha256": sha256_file(Path(__file__)),
        "checkpoint_sha256": sha256_file(bundle.checkpoint_path),
        "source_mode_manifest_sha256": sha256_file(bundle.mode_dir / "mode_manifest.json"),
        "source_final_metrics_sha256": sha256_file((ROOT / bundle.cfg["result_root"]).resolve() / "final_metrics.json"),
        "outputs": {
            "workpoint_metrics.csv": sha256_file(RESULT_ROOT / "workpoint_metrics.csv"),
            "dev_known_radius_thresholds.csv": sha256_file(RESULT_ROOT / "dev_known_radius_thresholds.csv"),
            "per_ball_radius_attribution.csv": sha256_file(RESULT_ROOT / "per_ball_radius_attribution.csv"),
            "selected_ball_replay.csv": sha256_file(RESULT_ROOT / "selected_ball_replay.csv"),
            "replay_check.json": sha256_file(RESULT_ROOT / "replay_check.json"),
            "report": sha256_file(REPORT_PATH),
        },
    }
    atomic_json(RESULT_ROOT / "MANIFEST.json", manifest)
    return manifest


def plot_results(workpoints: pd.DataFrame, per_ball: pd.DataFrame) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    subset = workpoints[workpoints.workpoint != "default_mean"].copy()
    axes[0].plot(subset["target_dev_known_coverage"], subset["radius_multiplier"], marker="o")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("目标 dev Known coverage")
    axes[0].set_ylabel("全局半径倍率")
    axes[0].grid(alpha=0.25)

    axes[1].plot(workpoints["Known Recall"], workpoints["F1-U"], marker="o")
    for row in workpoints.to_dict(orient="records"):
        axes[1].annotate(row["workpoint"].replace("dev_known_coverage_", "cal "), (row["Known Recall"], row["F1-U"]), fontsize=8)
    axes[1].set_xlabel("Known Recall")
    axes[1].set_ylabel("F1-U / OOS F1")
    axes[1].grid(alpha=0.25)
    fig.suptitle("corrected-loss BERT-MOGB: Known-only 覆盖校准工作点")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "radius_workpoints.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.8, 4.6))
    top = (
        per_ball[per_ball.workpoint.isin(["default_mean", "dev_known_coverage_0.95"])]
        .pivot_table(index="ball_id", columns="workpoint", values="test_false_accept_count", fill_value=0)
        .reset_index()
    )
    top["delta"] = top.get("dev_known_coverage_0.95", 0) - top.get("default_mean", 0)
    top = top.sort_values("delta", ascending=False).head(15)
    x = np.arange(len(top))
    axis.bar(x - 0.18, top.get("default_mean", pd.Series([0] * len(top))), width=0.36, label="default")
    axis.bar(x + 0.18, top.get("dev_known_coverage_0.95", pd.Series([0] * len(top))), width=0.36, label="cal-95")
    axis.set_xticks(x)
    axis.set_xticklabels([str(int(value)) for value in top["ball_id"]], rotation=45, ha="right")
    axis.set_xlabel("ball_id")
    axis.set_ylabel("测试集 OOS 误接受数")
    axis.set_title("按粒球分解的 false acceptance 贡献")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "false_accept_top_balls.png", dpi=220)
    plt.close(fig)


def build_report(
    workpoints: pd.DataFrame,
    thresholds: pd.DataFrame,
    per_ball: pd.DataFrame,
    bundle: RuntimeBundle,
    ball_check: dict[str, Any],
    metric_deltas: dict[str, float],
    replay_status: str,
    replay_seed: int,
) -> str:
    default = workpoints.loc[workpoints.workpoint == "default_mean"].iloc[0]
    cal95 = workpoints.loc[workpoints.workpoint == "dev_known_coverage_0.95"].iloc[0]
    default_false_accept = per_ball.loc[per_ball.workpoint == "default_mean", "test_false_accept_count"].sum()
    cal95_false_accept = per_ball.loc[per_ball.workpoint == "dev_known_coverage_0.95", "test_false_accept_count"].sum()
    replay_text = "严格 selected-ball 重放" if replay_status == "strict_selected_ball_replay" else "fixed checkpoint 下的 deterministic best-effort selected-ball reconstruction"
    coverage_rows = []
    for row in workpoints.to_dict(orient="records"):
        target_text = "-" if pd.isna(row["target_dev_known_coverage"]) else f"{float(row['target_dev_known_coverage']):.2f}"
        coverage_rows.append(
            f"| {row['workpoint']} | "
            f"{target_text} | "
            f"{float(row['radius_multiplier']):.4f} | "
            f"{float(row['achieved_dev_known_coverage']):.4f} | "
            f"{float(row['F1-U']):.4f} | "
            f"{float(row['F1-All']):.4f} | "
            f"{float(row['Known Recall']):.4f} | "
            f"{float(row['OOS Precision']):.4f} | "
            f"{float(row['OOS Recall']):.4f} |"
        )
    top_delta = (
        per_ball[per_ball.workpoint.isin(["default_mean", "dev_known_coverage_0.95"])]
        .pivot_table(index=["ball_id", "majority_label", "sample_count"], columns="workpoint", values="test_false_accept_count", fill_value=0)
        .reset_index()
    )
    top_delta["delta"] = top_delta.get("dev_known_coverage_0.95", 0) - top_delta.get("default_mean", 0)
    top_delta = top_delta.sort_values("delta", ascending=False).head(10)
    top_rows = []
    for row in top_delta.to_dict(orient="records"):
        top_rows.append(
            f"| {int(row['ball_id'])} | {int(row['majority_label'])} | {int(row['sample_count'])} | "
            f"{int(row.get('default_mean', 0))} | {int(row.get('dev_known_coverage_0.95', 0))} | {int(row['delta'])} |"
        )
    return f"""# corrected-loss BERT-MOGB Known-only 半径覆盖归因 V1

## 实验目标

本实验不重训任何模型，只复用已经完成的 `mogb_corrected_subcentroid_loss_v1`：

- 相同 checkpoint
- 相同 official data snapshot
- 相同 selected-ball 构造逻辑
- 相同欧氏距离与最近粒球判定

唯一改变项是：**只用 dev / calibration Known 样本标定一个全局半径倍率**，检查默认平均半径是否只是工作点过窄。

## 重放验证

- selected ball contract：`{replay_text}`
- selected ball 对比：`{ball_check}`
- default 指标最大绝对误差：`{max(metric_deltas.values()):.6g}`
- deterministic replay seed：`{replay_seed}`
- source checkpoint SHA256：`{bundle.manifest['checkpoint_sha256']}`

这说明当前脚本不是重新实现另一套 MOGB，而是在 corrected-loss 运行结果上做半径工作点归因。若 strict replay 失败，本文档不会把当前球结构写成“原 artifact 精确结构复现”。

## Known-only 覆盖工作点

| workpoint | 目标 dev Known coverage | 半径倍率 | 实际 dev Known coverage | F1-U | F1-All | Known Recall | OOS Precision | OOS Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(coverage_rows)}

## 主要发现

1. 默认平均半径下，corrected-loss MOGB 的 `Known Recall={default['Known Recall']:.4f}`，而把半径扩到 dev Known 95% 覆盖后，Known Recall 提升到 `{cal95['Known Recall']:.4f}`。
2. 这种恢复不是免费的：同一工作点下，测试集 OOS 误接受从 `{int(default_false_accept)}` 增加到 `{int(cal95_false_accept)}`。
3. 因此，corrected-loss MOGB 的性能差距里确实包含**边界工作点过窄**，但只靠放大半径并不能自动变成强 OOS 方法，因为 Known 覆盖恢复的同时会明显扩大 open-space acceptance。

## false acceptance 增量最大的粒球

| ball_id | majority_label | train sample_count | default OOS false accept | cal-95 OOS false accept | delta |
|---|---:|---:|---:|---:|---:|
{chr(10).join(top_rows)}

## 解释

这组结果说明，corrected-loss 版本已经比官方 loss 更好，但它仍然存在两个分离的问题：

1. **训练信号问题**：原始官方子中心 loss 压缩了最近子中心信号，这一点此前已被 corrected-loss 实验部分修复。
2. **边界工作点问题**：即使在 corrected-loss 表示上，平均半径仍然偏保守；Known-only 校准能恢复 Known coverage，但会把部分粒球推向更高的 OOS false acceptance。

所以当前最合理的结论不是“只要把 MOGB 半径调大就能复现论文”，而是：

> corrected-loss 修复了部分表示训练问题；但边界仍然需要更稳健的 calibration，否则 default 半径过窄、calibrated 半径又会带来明显的 OOS 误接受扩张。

## 产物

- `results/analysis/archive/analysis/mogb_corrected_radius_coverage_v1/workpoint_metrics.csv`
- `results/analysis/archive/analysis/mogb_corrected_radius_coverage_v1/dev_known_radius_thresholds.csv`
- `results/analysis/archive/analysis/mogb_corrected_radius_coverage_v1/per_ball_radius_attribution.csv`
- `results/analysis/archive/analysis/mogb_corrected_radius_coverage_v1/selected_ball_replay.csv`
- `figures/archive/analysis/mogb_corrected_radius_coverage_v1/radius_workpoints.png`
- `figures/archive/analysis/mogb_corrected_radius_coverage_v1/false_accept_top_balls.png`
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _ = parse_args(argv)
    manifest = run_analysis()
    print(json.dumps({"status": manifest["status"], "selected_ball_count": manifest["selected_ball_count"], "manifest_sha256": sha256_file(RESULT_ROOT / "MANIFEST.json")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
