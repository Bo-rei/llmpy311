#!/usr/bin/env python3
"""准备完整 KIR50 Cascade 矩阵所需的 CE-Recon detector 与线性 Gate。

Gate-only v19 已经保存了 Frozen MiniLM detector，v21 已经保存了每个 KIR50
seed 的 CE-Recon embeddings，但 v21 的研究输出没有把 detector 序列化为
Cascade 可直接加载的 JSON，也没有保存 sklearn 线性分类头。本脚本只补这两类
“适配层”产物，不重新选择 test 阈值、不使用 OOS 训练表示，也不覆盖旧结果。

输出根目录：
    ../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/gates/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gate.multi_sphere_oos_detector import MultiSphereOOSDetector  # noqa: E402
from src.runtime import WorkspacePaths  # noqa: E402
from tools.experiments.cluster_separability.baselines import (  # noqa: E402
    C_CANDIDATES,
    _fixed_k1_guard,
    _linear_oos_scores,
    select_operating_point,
)
from tools.experiments.cluster_separability.protocol import compute_binary_oos_metrics  # noqa: E402

PATHS = WorkspacePaths.discover(PROJECT_ROOT)
DATA_ROOT = PATHS.prepared_data_root / "multidataset" / "v19"
ARTIFACT_ROOT = PATHS.artifact_root / "outputs" / "experiments"
V19_ROOT = ARTIFACT_ROOT / "cluster_separability_v19"
ADAPT_ROOT = ARTIFACT_ROOT / "minilm_representation_analysis" / "adaptation"
OUTPUT_ROOT = ARTIFACT_ROOT / "cascade_full" / "gpu_kir50" / "gates"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
SEEDS = (13, 42, 87)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(dataset: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    root = DATA_ROOT / dataset / f"kir50_seed{seed}" / "gate"
    return {split: json.loads((root / f"{split}.json").read_text()) for split in ("train", "val", "test")}


def _selected_k(dataset: str, seed: int) -> int:
    table = np.genfromtxt(
        V19_ROOT / "selected_k_summary.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    matches = [
        row
        for row in table
        if str(row["dataset"]) == dataset
        and int(row["kir"]) == 50
        and int(row["data_seed"]) == seed
        and str(row["distance"]) == "mahalanobis_diag"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one selected K for {dataset}/kir50/seed{seed}, got {len(matches)}")
    return int(matches[0]["selected_k"])


def _embedding_cache(dataset: str, seed: int, split: str) -> np.ndarray:
    """读取与当前 gate split hash 对齐的 Frozen MiniLM cache。"""

    split_path = DATA_ROOT / dataset / f"kir50_seed{seed}" / "gate" / f"{split}.json"
    data_hash = _sha256_file(split_path)
    cache_dir = V19_ROOT / "embedding_cache" / dataset / f"kir50_seed{seed}"
    candidates = []
    for sidecar in cache_dir.glob(f"{split}_*.json"):
        payload = json.loads(sidecar.read_text())
        if payload.get("data_hash") == data_hash:
            candidates.append(sidecar.with_suffix(".npz"))
    if len(candidates) != 1 or not candidates[0].is_file():
        raise FileNotFoundError(f"cannot resolve unique MiniLM cache for {split_path}: {candidates}")
    return np.asarray(np.load(candidates[0])["embeddings"], dtype=np.float32)


def _adapted_embeddings(dataset: str, seed: int) -> dict[str, np.ndarray]:
    path = ADAPT_ROOT / dataset / f"kir50_seed{seed}" / "ce_recon" / "embeddings.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing CE-Recon embeddings: {path}")
    archive = np.load(path)
    return {split: np.asarray(archive[split], dtype=np.float32) for split in ("train", "val", "test")}


def _prepare_detector(dataset: str, seed: int, embeddings: dict[str, np.ndarray], rows: dict[str, list[dict[str, Any]]]) -> Path:
    selected = _selected_k(dataset, seed)
    output = OUTPUT_ROOT / dataset / f"kir50_seed{seed}" / "ce_recon_selected_k.detector.json"
    if output.is_file():
        return output
    intents = np.asarray([str(row["intent"]) for row in rows["train"]], dtype=object)
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=selected,
        radius_method="mean_std",
        radius_lambda=1.0,
        distance_metric="mahalanobis_diag",
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
    )
    detector.fit(embeddings["train"], intents)
    output.parent.mkdir(parents=True, exist_ok=True)
    detector.save(output)
    manifest = {
        "protocol": "cascade_full_ce_recon_detector_adapter",
        "dataset": dataset,
        "kir": 50,
        "data_seed": seed,
        "representation": "ce_recon",
        "selected_k": selected,
        "distance": "mahalanobis_diag",
        "covariance_scope": "per_cluster",
        "radius_method": "mean_std",
        "radius_lambda": 1.0,
        "threshold": 1.0,
        "used_oos_for_training": False,
        "source_embeddings": str((ADAPT_ROOT / dataset / f"kir50_seed{seed}" / "ce_recon" / "embeddings.npz").resolve()),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return output


def _prepare_baseline(dataset: str, seed: int, embeddings: dict[str, np.ndarray], rows: dict[str, list[dict[str, Any]]]) -> Path:
    output = OUTPUT_ROOT / dataset / f"kir50_seed{seed}" / "best_linear_baseline.pkl"
    if output.is_file():
        return output
    labels = {split: np.asarray([int(row["label"]) for row in rows[split]], dtype=np.int64) for split in rows}
    intents = sorted({str(row["intent"]) for row in rows["train"]})
    intent_to_id = {intent: index for index, intent in enumerate(intents)}
    train_y = np.asarray([intent_to_id[str(row["intent"])] for row in rows["train"]], dtype=np.int64)
    val_known = labels["val"] == 0
    val_known_y = np.asarray(
        [intent_to_id[str(row["intent"])] for row in rows["val"] if int(row["label"]) == 0],
        dtype=np.int64,
    )
    fitted: dict[float, LogisticRegression] = {}
    c_rows = []
    for c_value in C_CANDIDATES:
        classifier = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=2000, random_state=42)
        classifier.fit(embeddings["train"], train_y)
        known_f1 = float(f1_score(val_known_y, classifier.predict(embeddings["val"][val_known]), average="macro", zero_division=0))
        fitted[float(c_value)] = classifier
        c_rows.append({"C": float(c_value), "known_validation_macro_f1": known_f1})
    selected_c = max(c_rows, key=lambda row: (row["known_validation_macro_f1"], -row["C"]))
    classifier = fitted[selected_c["C"]]
    guard, fixed_recall = _fixed_k1_guard(
        embeddings["train"],
        np.asarray([str(row["intent"]) for row in rows["train"]], dtype=object),
        rows["val"],
        embeddings["val"],
        "euclidean",
    )
    method_rows = []
    for method in ("msp", "energy", "entropy"):
        scores, _ = _linear_oos_scores(classifier, embeddings["val"], method)
        selected, _ = select_operating_point(labels["val"], scores, guard)
        method_rows.append({"method": method, "operating_point": selected})
    selected_method = max(
        method_rows,
        key=lambda row: (
            float(row["operating_point"]["metrics"]["oos_f1"]),
            -float(row["operating_point"]["metrics"]["fpr95"]),
            float(row["operating_point"]["metrics"]["id_recall"]),
        ),
    )
    operating = selected_method["operating_point"]
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "kir": 50,
        "data_seed": seed,
        "method": selected_method["method"],
        "threshold": float(operating["threshold"]),
        "classifier": classifier,
        "selection_path": "validation_only_current_sklearn",
        "selection": {"C": selected_c, "methods": method_rows, "id_recall_guard": guard, "fixed_k1_validation_id_recall": fixed_recall},
        "test_used_for_selection": False,
        "external_hard_negative_used_for_selection": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    output.with_suffix(".json").write_text(
        json.dumps({key: value for key, value in payload.items() if key != "classifier"}, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    datasets = tuple(args.dataset or DATASETS)
    seeds = tuple(args.seed or SEEDS)
    units = []
    for dataset in datasets:
        for seed in seeds:
            root = OUTPUT_ROOT / dataset / f"kir50_seed{seed}"
            units.append({
                "dataset": dataset,
                "seed": seed,
                "frozen_k1": str((V19_ROOT / "fixed" / dataset / f"kir50_seed{seed}" / "mahalanobis_diag" / "k1" / "detector.json").resolve()),
                "frozen_selected_k": str((V19_ROOT / "fixed" / dataset / f"kir50_seed{seed}" / "mahalanobis_diag" / f"k{_selected_k(dataset, seed)}" / "detector.json").resolve()),
                "ce_recon_detector": str((root / "ce_recon_selected_k.detector.json").resolve()),
                "baseline": str((root / "best_linear_baseline.pkl").resolve()),
            })
    missing = []
    for unit in units:
        for key in ("frozen_k1", "frozen_selected_k"):
            if not Path(unit[key]).is_file():
                missing.append(unit[key])
        if args.execute:
            rows = _rows(unit["dataset"], int(unit["seed"]))
            frozen = {split: _embedding_cache(unit["dataset"], int(unit["seed"]), split) for split in rows}
            adapted = _adapted_embeddings(unit["dataset"], int(unit["seed"]))
            unit["ce_recon_detector"] = str(_prepare_detector(unit["dataset"], int(unit["seed"]), adapted, rows).resolve())
            unit["baseline"] = str(_prepare_baseline(unit["dataset"], int(unit["seed"]), frozen, rows).resolve())
        else:
            for key in ("ce_recon_detector", "baseline"):
                if not Path(unit[key]).is_file():
                    missing.append(unit[key])
    payload = {
        "schema_version": 1,
        "protocol": "cascade_full_kir50_gate_adapters",
        "execute": bool(args.execute),
        "datasets": list(datasets),
        "seeds": list(seeds),
        "units": units,
        "missing_before_or_after_run": missing,
        "status": "complete" if not missing else "missing_inputs_or_outputs",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "gate_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
