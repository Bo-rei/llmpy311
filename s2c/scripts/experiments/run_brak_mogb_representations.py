#!/usr/bin/env python3
"""Run the registered BRAK comparison on three MOGB-related representations.

This is a small, post-reproduction diagnostic.  It reuses the same
StackOverflow/KIR=0.50/seed=0 protocol views as the strict MOGB cell and
evaluates fixed K=1..5 plus the existing Known-only BRAK selector on:

* frozen MiniLM;
* the initial MOGB BERT feature head;
* the best strict-MOGB trained hierarchical BERT checkpoint.

No test metric is used for selecting K.  The script does not modify the
official checkout and keeps its arrays under the ignored external artifact
root rather than committing embeddings or predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protocol_v2.experiments.brak import (  # noqa: E402
    BRAKSelection,
    evaluate_intent_candidates,
    selection_rows,
)
from protocol_v2.experiments.partitions import normalize_for_detector  # noqa: E402
from protocol_v2.runtime.paths import ProtocolV2Paths  # noqa: E402
from scripts.experiments.run_brak_pilot import _evaluate_detector  # noqa: E402
from scripts.experiments.run_brak_pilot import _fixed_detector as _fixed_detector_existing  # noqa: E402
from scripts.experiments.run_brak_pilot import _selected_detector as _selected_detector_existing  # noqa: E402
from scripts.experiments.run_mogb_exact_reproduction import (  # noqa: E402
    _import_official_modules,
    build_args,
    load_config,
    patch_cluster_device,
    sha256_file,
    set_seeds,
    tree_hash,
)


EXPERIMENT_ID = "brak_mogb_representation_v1"
MOGB_EXACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "external" / "mogb_exact_reproduction_v1"
DEFAULT_CONFIG = ROOT / "configs" / "baselines" / "mogb_exact_reproduction_v1.yaml"
REPRESENTATIONS = ("frozen_minilm", "mogb_initial_bert", "mogb_trained_hierarchical_bert")
KS = (1, 2, 3, 4, 5)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        atomic_write(path, "\n")
        return
    fields = sorted({key for row in rows for key in row})
    with path.with_suffix(path.suffix + ".tmp").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(path.suffix + ".tmp").replace(path)


def _text_hash(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _read_tsv(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        # Match the pinned MOGB DataProcessor exactly: quotechar=None keeps
        # literal quote characters in StackOverflow titles.
        rows = list(csv.reader(handle, delimiter="\t", quotechar=None))
    if not rows or rows[0] != ["text", "label"]:
        raise ValueError(f"Unexpected StackOverflow TSV header: {path}")
    return [(row[0], row[1]) for row in rows[1:] if len(row) == 2]


def _build_seed0_views(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    source = MOGB_EXACT_ROOT / "audit" / "official_fixed" / "data_snapshot" / "stackoverflow"
    if not source.is_dir():
        raise FileNotFoundError(f"Missing immutable strict-MOGB StackOverflow snapshot: {source}")
    splits = {name: _read_tsv(source / f"{name}.tsv") for name in ("train", "dev", "test")}
    labels = sorted({label for rows in splits.values() for _, label in rows})
    rng = np.random.RandomState(0)
    known = [str(value) for value in rng.choice(np.asarray(labels), round(len(labels) * 0.50), replace=False)]
    known_set = set(known)

    def rows_for(split: str, *, include_unknown: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row_number, (text, intent) in enumerate(splits[split], start=1):
            is_known = intent in known_set
            if split != "test" and not is_known:
                continue
            if split == "test" and not include_unknown and not is_known:
                continue
            rows.append(
                {
                    "text": text,
                    "intent": intent,
                    "label": 0 if is_known else 1,
                    "sample_id": hashlib.sha256(f"{split}\t{row_number}\t{intent}\t{text}".encode()).hexdigest(),
                }
            )
        return rows

    views = SimpleNamespace(
        train=rows_for("train", include_unknown=False),
        calibration=rows_for("dev", include_unknown=False),
        test=rows_for("test", include_unknown=True),
    )
    audit = {
        "source_dir": str(source),
        "known_intents": known,
        "unknown_intents": [label for label in labels if label not in known_set],
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "known_view_counts": {"train": len(views.train), "calibration": len(views.calibration), "test": len(views.test)},
        "source_tree_sha256": tree_hash(source),
    }
    return views, audit


def _encode_loader(manager: Any, loader: Any) -> np.ndarray:
    manager.model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, input_mask, segment_ids, _ = (item.to(manager.device) for item in batch)
            values = manager.model(input_ids, segment_ids, input_mask, feature_ext=True)
            chunks.append(values.detach().cpu().numpy().astype(np.float32, copy=False))
    if not chunks:
        return np.empty((0, 768), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _align_by_text(encoded: np.ndarray, examples: list[Any], rows: list[dict[str, Any]], split: str) -> np.ndarray:
    if encoded.shape[0] != len(examples):
        raise ValueError(f"{split}: encoded/example length mismatch {encoded.shape[0]} != {len(examples)}")
    positions: dict[str, deque[int]] = defaultdict(deque)
    for index, example in enumerate(examples):
        positions[str(example.text_a)].append(index)
    ordered: list[int] = []
    for row in rows:
        text = str(row["text"])
        if not positions[text]:
            raise ValueError(f"{split}: protocol row text not found in MOGB Data examples")
        ordered.append(positions[text].popleft())
    if len(ordered) != len(rows):
        raise ValueError(f"{split}: alignment length mismatch")
    return encoded[np.asarray(ordered, dtype=np.int64)]


def _load_mogb_features(cfg: dict[str, Any], representation: str, views: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mode_dir = MOGB_EXACT_ROOT / "audit" / "official_fixed"
    data_snapshot = mode_dir / "data_snapshot"
    if not data_snapshot.is_dir():
        raise FileNotFoundError(f"Missing strict MOGB data snapshot: {data_snapshot}")
    modules = _import_official_modules()
    patch_cluster_device(modules["cluster"])
    set_seeds(0)
    args = build_args(cfg, mode_dir)
    args.data_dir = str(data_snapshot)
    args.seed = 0
    data = modules["dataloader"].Data(args)
    manager = modules["pretrain"].PretrainModelManager(args, data)
    checkpoint = None
    if representation == "mogb_trained_hierarchical_bert":
        checkpoint = mode_dir / "checkpoints" / "best_checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing strict MOGB checkpoint: {checkpoint}")
        payload = torch.load(checkpoint, map_location=manager.device, weights_only=False)
        manager.model.load_state_dict(payload["model_state_dict"])

    train_raw = _encode_loader(manager, data.train_dataloader)
    calibration_raw = _encode_loader(manager, data.eval_dataloader)
    test_raw = _encode_loader(manager, data.test_dataloader)
    train = normalize_for_detector(_align_by_text(train_raw, data.train_examples, views.train, "train"))
    calibration = normalize_for_detector(_align_by_text(calibration_raw, data.eval_examples, views.calibration, "calibration"))
    test = normalize_for_detector(_align_by_text(test_raw, data.test_examples, views.test, "test"))
    return train, calibration, test, {
        "source": representation,
        "model_device": str(manager.device),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "train_embedding_sha256": _text_hash(train),
        "calibration_embedding_sha256": _text_hash(calibration),
        "test_embedding_sha256": _text_hash(test),
    }


def _load_frozen_minilm(views: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    model_path = ROOT.parent / "assets" / "models" / "all-MiniLM-L6-v2"
    encoder = SentenceTransformer(str(model_path), device="cuda")

    def encode(rows: list[dict[str, Any]]) -> np.ndarray:
        values = encoder.encode(
            [str(row["text"]) for row in rows],
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return normalize_for_detector(np.asarray(values, dtype=np.float32))

    train, calibration, test = encode(views.train), encode(views.calibration), encode(views.test)
    return train, calibration, test, {
        "source": "local_all-MiniLM-L6-v2",
        "model_path": str(model_path),
        "train_embedding_sha256": _text_hash(train),
        "calibration_embedding_sha256": _text_hash(calibration),
        "test_embedding_sha256": _text_hash(test),
    }


def _selection_for_representation(
    train: np.ndarray,
    train_rows: list[dict[str, Any]],
    calibration: np.ndarray,
    calibration_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, BRAKSelection], list[dict[str, Any]]]:
    train_intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    calibration_intents = np.asarray([str(row["intent"]) for row in calibration_rows], dtype=object)
    selections: dict[str, BRAKSelection] = {}
    diagnostics: list[dict[str, Any]] = []
    for intent in sorted(set(train_intents.tolist())):
        proper = train[train_intents == intent]
        target = calibration[calibration_intents == intent]
        other = calibration[calibration_intents != intent]
        selected = evaluate_intent_candidates(
            intent,
            proper,
            target,
            other,
            max_k=5,
            seed=seed,
            distance="mahalanobis_diag",
            covariance_eps=1e-6,
            bootstrap_repeats=5,
            alpha=1.0,
            beta=1.0,
            gamma=0.25,
            eta=0.01,
            delta=0.02,
            min_improvement=0.01,
        )
        selections[intent] = selected
        diagnostics.extend(selection_rows(selected))
    return selections, diagnostics


def _run_representation(
    representation: str,
    train: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    views: Any,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    fixed_metrics: dict[int, dict[str, float]] = {}
    for k in KS:
        detector = _fixed_detector_existing(train, views.train, k, seed)
        metrics = _evaluate_detector(detector, test, views.test)
        fixed_metrics[k] = metrics
        summary.append({"representation": representation, "method": f"fixed_k{k}", "selection_source": "predeclared_fixed_k", **metrics})
    selections, diagnostics = _selection_for_representation(train, views.train, calibration, views.calibration, seed)
    detector = _selected_detector_existing(train, views.train, selections, seed)
    metrics = _evaluate_detector(detector, test, views.test)
    summary.append(
        {
            "representation": representation,
            "method": "brak",
            "selection_source": "proper_train_and_known_calibration_only",
            "selected_k_mean": float(np.mean([item.selected_k for item in selections.values()])),
            "selected_k_median": float(np.median([item.selected_k for item in selections.values()])),
            **metrics,
        }
    )
    for row in diagnostics:
        row.update({"representation": representation})
    distribution = {
        "representation": representation,
        "selected_k_counts": {
            str(k): int(sum(item.selected_k == k for item in selections.values())) for k in KS
        },
        "selected_k_values": {intent: int(item.selected_k) for intent, item in sorted(selections.items())},
        "fixed_k_oos_f1": {str(k): float(fixed_metrics[k]["oos_f1"]) for k in KS},
    }
    return summary, diagnostics, distribution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("BRAK MOGB representation comparison requires the same GPU contract as MOGB")
    cfg = load_config(args.config.resolve())
    paths = ProtocolV2Paths.discover()
    output_root = ROOT / "results" / "mogb_exact_reproduction" / "brak_mogb_representation"
    artifact_root = MOGB_EXACT_ROOT / "brak_mogb_representation"
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    views, split_audit = _build_seed0_views(cfg)
    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": paths.dataset_version,
        "dataset": "stackoverflow",
        "kir": 0.50,
        "seed": 0,
        "candidate_k": list(KS),
        "distance": "mahalanobis_diag",
        "radius": "mean_std_lambda_1.0",
        "selection_data": "proper_train_and_known_calibration_only",
        "test_used_for_selection": False,
        "mogb_exact_artifact_root": str(MOGB_EXACT_ROOT),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "split_audit": split_audit,
    }
    write_json(artifact_root / "BRAK_MOGB_PROVENANCE.json", provenance)
    all_summary: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    representation_meta: dict[str, Any] = {"frozen_minilm": {"source": "protocol_v2_frozen_cache"}}
    for representation in REPRESENTATIONS:
        if representation == "frozen_minilm":
            train, calibration, test, meta = _load_frozen_minilm(views)
            representation_meta[representation] = meta
        else:
            train, calibration, test, meta = _load_mogb_features(cfg, representation, views)
            representation_meta[representation] = meta
        summary, diagnostics, distribution = _run_representation(representation, train, calibration, test, views, 0)
        all_summary.extend(summary)
        all_diagnostics.extend(diagnostics)
        distributions.append(distribution)
    write_csv(output_root / "brak_summary.csv", all_summary)
    write_csv(output_root / "brak_selection_diagnostics.csv", all_diagnostics)
    write_json(output_root / "selected_k_distribution.json", distributions)
    write_json(output_root / "representation_metadata.json", representation_meta)
    provenance["representations"] = representation_meta
    write_json(artifact_root / "BRAK_MOGB_PROVENANCE.json", provenance)
    print(json.dumps({"status": "complete", "summary_rows": len(all_summary), "representations": list(REPRESENTATIONS), "output": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
