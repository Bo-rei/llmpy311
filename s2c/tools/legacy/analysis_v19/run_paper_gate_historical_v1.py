#!/usr/bin/env python3
"""Replay explicit frozen-MiniLM Gate contracts on archived v19 data.

The paper contains two different fixed-boundary contracts that must not be
mixed: its main Cascade description uses ``K_y=2`` and lambda=0.5 on CLINC
(lambda=1 elsewhere), whereas its controlled K ablation fixes lambda=1 for
every dataset.  This runner requires the caller to name which contract is
being replayed.  It intentionally does not tune on the archived validation
split.

The output is a Gate-only, standard OOS-positive F1 replay.  It is never a
claim to reproduce the paper's full Gate -> Router -> Expert Cascade.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = PROJECT_ROOT.parent / "archives" / "submissions" / "s2c-submission"
ARCHIVE_SRC = ARCHIVE_ROOT / "src"
for import_root in (ARCHIVE_ROOT, ARCHIVE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

MultiSphereOOSDetector = import_module(
    "gate.multi_sphere_oos_detector"
).MultiSphereOOSDetector


CONTRACTS = {
    "controlled_k_ablation": {
        "description": "Paper controlled K table: lambda=1.0 for every dataset.",
        "lambda_by_dataset": {
            "clinc150": 1.0,
            "stackoverflow": 1.0,
            "banking77": 1.0,
            "banking77_oos": 1.0,
        },
    },
    "paper_main_gate_geometry": {
        "description": "Paper main Cascade geometry: K_y=2, CLINC lambda=0.5; this is Gate-only here.",
        "lambda_by_dataset": {
            "clinc150": 0.5,
            "stackoverflow": 1.0,
            "banking77": 1.0,
            "banking77_oos": 1.0,
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split(path: Path, known: set[str]) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["intent"] = str(item["intent"])
        item["text"] = str(item["text"])
        item["label"] = int(item.get("label", 0 if item["intent"] in known else 1))
        normalized.append(item)
    return normalized


def _metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0
    )
    known = labels == 0
    oos = labels == 1
    return {
        "oos_precision": float(precision),
        "oos_recall": float(recall),
        "oos_f1": float(f1),
        "known_recall": float(np.mean(predictions[known] == 0)) if np.any(known) else 0.0,
        "oos_rejection": float(np.mean(predictions[oos] == 1)) if np.any(oos) else 0.0,
        "false_accept_rate": float(np.mean(predictions[oos] == 0)) if np.any(oos) else 0.0,
        "false_reject_rate": float(np.mean(predictions[known] == 1)) if np.any(known) else 0.0,
        "accuracy": float(np.mean(predictions == labels)),
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "confusion": {
            "true_known_pred_known": int(np.sum((labels == 0) & (predictions == 0))),
            "true_known_pred_oos": int(np.sum((labels == 0) & (predictions == 1))),
            "true_oos_pred_known": int(np.sum((labels == 1) & (predictions == 0))),
            "true_oos_pred_oos": int(np.sum((labels == 1) & (predictions == 1))),
        },
        "standard_oos_f1_definition": "sklearn f1_score with OOS as positive class",
    }


def run(
    dataset: str,
    data_root: Path,
    model_path: Path,
    output_dir: Path,
    seed: int,
    contract: str,
    device: str,
) -> dict[str, Any]:
    started = time.time()
    if contract not in CONTRACTS:
        raise ValueError(f"Unknown replay contract: {contract}")
    contract_spec = CONTRACTS[contract]
    radius_lambda = float(contract_spec["lambda_by_dataset"][dataset])
    manifest = json.loads((data_root / "KNOWN_INTENTS.json").read_text(encoding="utf-8"))
    known = {str(value) for value in manifest["known_intents"]}
    views = {
        split: _load_split(data_root / "gate" / f"{split}.json", known)
        for split in ("train", "val", "test")
    }
    if any(row["label"] != 0 for row in views["train"]):
        raise ValueError("Historical paper Gate requires Known-only train rows")

    encoder = SentenceTransformer(str(model_path), device=device)
    embeddings = {
        split: np.asarray(
            encoder.encode(
                [row["text"] for row in rows],
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ),
            dtype=np.float32,
        )
        for split, rows in views.items()
    }
    outputs: dict[str, Any] = {
        "schema_version": "s2c.historical_protocol_v1.paper_gate.v2",
        "dataset": dataset,
        "data_root": str(data_root.resolve()),
        "model_path": str(model_path.resolve()),
        "model_config_sha256": _sha256(model_path / "config.json"),
        "data_manifest": manifest,
        "data_split_counts": {
            split: {
                "total": len(rows),
                "known": int(sum(row["label"] == 0 for row in rows)),
                "oos": int(sum(row["label"] == 1 for row in rows)),
            }
            for split, rows in views.items()
        },
        "replay_contract": {
            "name": contract,
            "description": contract_spec["description"],
        },
        "paper_settings": {
            "embedding": "all-MiniLM-L6-v2",
            "l2_normalize": True,
            "center_construction": "per-intent KMeans",
            "distance": "diagonal Mahalanobis",
            "radius": "mean + lambda * std",
            "lambda": radius_lambda,
            "threshold": 1.0,
            "seed": int(seed),
            "selection": "fixed contract lambda; no validation/test selection",
        },
        "runs": {},
    }
    test_labels = np.asarray([row["label"] for row in views["test"]], dtype=np.int64)
    test_texts = [row["text"] for row in views["test"]]
    for k in (1, 2):
        detector = MultiSphereOOSDetector(
            n_clusters=None,
            radius_method="mean_std",
            radius_lambda=radius_lambda,
            center_mode="class_centroid_mixture",
            distance_metric="mahalanobis_diag",
            l2_normalize=True,
            subcenters_per_intent=k,
            random_state=int(seed),
        )
        detector.fit(
            embeddings["train"],
            np.asarray([row["intent"] for row in views["train"]], dtype=object),
        )
        scored = detector.predict_with_scores(embeddings["test"])
        result = _metrics(test_labels, scored["pred"], scored["score"])
        run_dir = output_dir / f"k{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": f"{contract}_gate_k{k}",
            "k_per_intent": int(k),
            "dataset": dataset,
            "seed": int(seed),
            "settings": outputs["paper_settings"],
            "metrics": result,
            "score_summary": {
                "known_p50": float(np.percentile(scored["score"][test_labels == 0], 50)),
                "oos_p50": float(np.percentile(scored["score"][test_labels == 1], 50)),
            },
            "n_spheres": len(detector.spheres),
            "test_text_sha256": hashlib.sha256("\n".join(test_texts).encode()).hexdigest(),
            "elapsed_seconds": time.time() - started,
            "status": "complete",
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "predictions.json").write_text(
            json.dumps(
                [
                    {"label": int(label), "prediction": int(pred), "score": float(score)}
                    for label, pred, score in zip(test_labels, scored["pred"], scored["score"])
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        outputs["runs"][f"k{k}"] = payload
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(
            set().union(*(spec["lambda_by_dataset"] for spec in CONTRACTS.values()))
        ),
    )
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contract", choices=sorted(CONTRACTS), default="controlled_k_ablation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(
        args.dataset,
        args.data_root.resolve(),
        args.model_path.resolve(),
        args.output_dir.resolve(),
        args.seed,
        args.contract,
        args.device,
    )
    print(json.dumps({"dataset": args.dataset, "runs": {key: value["metrics"] for key, value in result["runs"].items()}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
