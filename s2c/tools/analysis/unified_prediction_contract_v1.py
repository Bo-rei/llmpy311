"""Build and audit a row-level prediction contract for the current S2C evidence.

This module deliberately stays in the analysis layer.  It reads completed local
artifacts and emits a compressed, local-only JSONL file; it does not train,
select a threshold, or mutate an experiment run directory.  Incomplete external
baselines are represented in the public blocker table instead of being turned
into synthetic prediction rows.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


PROTOCOL_VERSION = "protocol_v2_textoir_v1"
SCHEMA_VERSION = "unified_prediction_contract_v1"
OOS_LABEL = "__oos__"

BASE_FIELDS = [
    "protocol_version",
    "dataset",
    "kir",
    "seed",
    "method",
    "backbone",
    "supervision_type",
    "split",
    "sample_id",
    "true_label",
    "is_true_oos",
    "predicted_label",
    "predicted_oos",
    "accepted_known",
    "oos_score",
    "confidence",
    "run_id",
    "registry_sha256",
    "canonical_manifest_sha256",
]
GATE_FIELDS = [
    "nearest_intent",
    "nearest_distance",
    "boundary_score",
    "center_id",
    "radius",
    "normalized_distance",
]
MOGB_FIELDS = [
    "ball_id",
    "ball_count",
    "ball_size",
    "ball_radius",
    "ball_purity",
]
DCLOOS_FIELDS = ["oos_logit", "oos_probability"]
ALL_FIELDS = BASE_FIELDS + GATE_FIELDS + MOGB_FIELDS + DCLOOS_FIELDS

DATASET_ALIASES = {
    "banking": "banking77",
    "banking77": "banking77",
    "clinc": "clinc150",
    "clinc150": "clinc150",
    "oos": "clinc150",
    "stackoverflow": "stackoverflow",
}

MOGB_METHOD_LABELS = {
    "fixed_k2": "S2C-Frozen-K2",
    "mogb_minilm": "MOGB-MiniLM",
    "mogb_partition_ours_boundary": "MOGB-partition-S2C-boundary",
    "ours_partition_mogb_boundary": "S2C-partition-MOGB-boundary",
    "random_partition": "S2C-random-K2",
    "single_centroid": "S2C-Frozen-K1",
}

NATIVE_METHOD_LABELS = {
    "msp": "Native-MSP-Trainable-MiniLM",
    "energy": "Native-Energy-Trainable-MiniLM",
    "knn": "Native-kNN-Trainable-MiniLM",
    "lof": "Native-LOF-Trainable-MiniLM",
}


def normalise_dataset(value: Any) -> str:
    """Return the protocol dataset name used by the current checkout."""

    value = str(value).strip().lower()
    return DATASET_ALIASES.get(value, value)


def normalise_oos_label(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    return OOS_LABEL if value.strip().lower() in {"oos", "__oos__", "<unk>", "unknown"} else value


def _number(value: Any) -> float | int | None:
    if value is None or value == "" or value == "NA":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _boolean_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "yes", "y", "1"}:
            return 1
        if value in {"false", "no", "n", "0"}:
            return 0
    return int(bool(value))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_from_manifest(manifest: Mapping[str, Any], name: str) -> str | None:
    """Read a named source hash while preferring explicit input hash fields."""

    input_data = manifest.get("input", {})
    input_hashes = input_data.get("input_hashes", {}) if isinstance(input_data, Mapping) else {}
    top_hashes = manifest.get("input_hashes", {})
    candidates: list[Any] = []
    if name == "registry":
        candidates.extend(
            [
                input_hashes.get("registry"),
                input_data.get("registry_sha256") if isinstance(input_data, Mapping) else None,
                top_hashes.get("registry_sha256") if isinstance(top_hashes, Mapping) else None,
                manifest.get("registry_sha256"),
            ]
        )
    if name == "canonical_manifest":
        candidates.extend(
            [
                input_hashes.get("canonical_manifest"),
                input_data.get("canonical_manifest_sha256")
                if isinstance(input_data, Mapping)
                else None,
                top_hashes.get("canonical_manifest_sha256")
                if isinstance(top_hashes, Mapping)
                else None,
                manifest.get("canonical_manifest_sha256"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, str) and re.fullmatch(r"[0-9a-fA-F]{64}", candidate):
            return candidate
    return None


def _empty_row() -> dict[str, Any]:
    return {field: None for field in ALL_FIELDS}


def _base_row(
    *,
    dataset: str,
    kir: float,
    seed: int,
    method: str,
    backbone: str,
    supervision_type: str,
    sample_id: str,
    true_label: str,
    is_true_oos: int,
    predicted_label: str | None,
    predicted_oos: int | None,
    oos_score: Any,
    confidence: Any,
    run_id: str,
    registry_sha256: str | None,
    canonical_manifest_sha256: str | None,
) -> dict[str, Any]:
    row = _empty_row()
    row.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": normalise_dataset(dataset),
            "kir": float(kir),
            "seed": int(seed),
            "method": method,
            "backbone": backbone,
            "supervision_type": supervision_type,
            "split": "test",
            "sample_id": str(sample_id),
            "true_label": normalise_oos_label(true_label),
            "is_true_oos": int(is_true_oos),
            "predicted_label": normalise_oos_label(predicted_label),
            "predicted_oos": None if predicted_oos is None else int(predicted_oos),
            "accepted_known": None if predicted_oos is None else int(not bool(predicted_oos)),
            "oos_score": _number(oos_score),
            "confidence": _number(confidence),
            "run_id": run_id,
            "registry_sha256": registry_sha256,
            "canonical_manifest_sha256": canonical_manifest_sha256,
        }
    )
    return row


def validate_row(row: Mapping[str, Any]) -> list[str]:
    """Validate one normalized row without requiring an optional score."""

    errors: list[str] = []
    missing = [field for field in BASE_FIELDS + ["sample_id"] if field not in row]
    if missing:
        errors.append("missing_fields=" + ",".join(missing))
        return errors
    for field_name in ("protocol_version", "dataset", "method", "backbone", "supervision_type", "split", "sample_id", "run_id"):
        if row.get(field_name) in (None, ""):
            errors.append(f"empty_{field_name}")
    if row.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("protocol_version_mismatch")
    if row.get("split") != "test":
        errors.append("split_not_test")
    for field_name in ("is_true_oos", "predicted_oos", "accepted_known"):
        value = row.get(field_name)
        if value is None:
            errors.append(f"missing_{field_name}")
            continue
        if value is not None and value not in (0, 1, False, True):
            errors.append(f"{field_name}_not_boolean")
    predicted_oos = row.get("predicted_oos")
    accepted_known = row.get("accepted_known")
    if predicted_oos is not None and accepted_known is not None and accepted_known != int(not bool(predicted_oos)):
        errors.append("accepted_known_inconsistent")
    for field_name in ("kir", "oos_score", "confidence", "nearest_distance", "boundary_score", "radius", "normalized_distance", "ball_radius", "ball_purity", "oos_logit", "oos_probability"):
        value = row.get(field_name)
        if value is not None:
            try:
                if not math.isfinite(float(value)):
                    errors.append(f"{field_name}_not_finite")
            except (TypeError, ValueError):
                errors.append(f"{field_name}_not_numeric")
    return errors


def validate_rows(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Strictly validate a small group, including exact sample-id uniqueness/order."""

    errors: list[str] = []
    seen: set[str] = set()
    previous: str | None = None
    for index, row in enumerate(rows):
        errors.extend(f"row_{index}:{error}" for error in validate_row(row))
        sample_id = row.get("sample_id")
        if sample_id in seen:
            errors.append(f"row_{index}:duplicate_sample_id")
        if previous is not None and sample_id == previous:
            errors.append(f"row_{index}:adjacent_duplicate_sample_id")
        if sample_id is not None:
            seen.add(str(sample_id))
        previous = sample_id
    return errors


def _sample_digest_update(state: dict[str, Any], sample_id: str) -> None:
    encoded = str(sample_id).encode("utf-8")
    state["sample_hash"].update(encoded)
    state["sample_hash"].update(b"\n")
    digest = hashlib.sha256(encoded).digest()
    state["xor_digest"] = bytes(a ^ b for a, b in zip(state["xor_digest"], digest))
    state["sum_digest"] = (state["sum_digest"] + int.from_bytes(digest, "big")) % (1 << 256)


@dataclass
class GroupAudit:
    dataset: str
    kir: float
    seed: int
    method: str
    run_id: str
    backbone: str
    supervision_type: str
    source_path: str
    contract_layer: str
    score_available: bool
    selection_audit: str
    count: int = 0
    invalid_rows: int = 0
    first_sample_id: str | None = None
    last_sample_id: str | None = None
    label_hash: Any = field(default_factory=hashlib.sha256)
    sample_hash: Any = field(default_factory=hashlib.sha256)
    xor_digest: bytes = field(default_factory=lambda: bytes(32))
    sum_digest: int = 0

    def add(self, row: Mapping[str, Any]) -> None:
        sample_id = str(row["sample_id"])
        if self.first_sample_id is None:
            self.first_sample_id = sample_id
        self.last_sample_id = sample_id
        self.count += 1
        _sample_digest_update(self.__dict__, sample_id)
        label_payload = f"{row.get('true_label')}\t{row.get('is_true_oos')}\n".encode("utf-8")
        self.label_hash.update(label_payload)
        errors = validate_row(row)
        self.invalid_rows += int(bool(errors))

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "kir": self.kir,
            "seed": self.seed,
            "method": self.method,
            "run_id": self.run_id,
            "backbone": self.backbone,
            "supervision_type": self.supervision_type,
            "source_path": self.source_path,
            "contract_layer": self.contract_layer,
            "selection_audit": self.selection_audit,
            "score_available": self.score_available,
            "row_count": self.count,
            "invalid_rows": self.invalid_rows,
            "first_sample_id": self.first_sample_id,
            "last_sample_id": self.last_sample_id,
            "sample_id_sequence_sha256": self.sample_hash.hexdigest(),
            "sample_id_multiset_xor": self.xor_digest.hex(),
            "sample_id_multiset_sum": f"{self.sum_digest:064x}",
            "true_label_sequence_sha256": self.label_hash.hexdigest(),
        }


def _meta_from_row(row: Mapping[str, Any]) -> tuple[str, float, int, str, str, str]:
    return (
        str(row["dataset"]),
        float(row["kir"]),
        int(row["seed"]),
        str(row["method"]),
        str(row["run_id"]),
        str(row["backbone"]),
    )


def _path_manifest(path: Path) -> Path | None:
    candidate = path.parent / "run_manifest.json"
    return candidate if candidate.exists() else None


def _parse_trainable_location(path: Path, manifest: Mapping[str, Any]) -> tuple[str, float, int]:
    dataset = normalise_dataset(manifest.get("dataset", ""))
    kir = _number(manifest.get("kir"))
    seed = _integer(manifest.get("seed"))
    if not dataset:
        match = re.search(r"/runs/([^/]+)/seed_(\d+)/predictions\.jsonl$", str(path))
        if match:
            dataset, seed = normalise_dataset(match.group(1)), int(match.group(2))
    if kir is None:
        match = re.search(r"/kir_(\d+\.\d+)/runs/", str(path))
        if match:
            kir = float(match.group(1))
    if seed is None:
        match = re.search(r"/seed_(\d+)/predictions\.jsonl$", str(path))
        if match:
            seed = int(match.group(1))
    if not dataset or kir is None or seed is None:
        raise ValueError(f"cannot parse Trainable location: {path}")
    return dataset, float(kir), int(seed)


def _trainable_run_paths(artifacts: Path) -> list[Path]:
    roots = [
        artifacts / "runs" / PROTOCOL_VERSION / "minilm_trainable_kir_sweep_v1",
        artifacts / "runs" / PROTOCOL_VERSION / "minilm_trainable_kir_sweep_extension_v1",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(root.rglob("predictions.jsonl"))
    return sorted(paths)


def iter_trainable_rows(artifacts: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    seen: set[tuple[str, float, int]] = set()
    for prediction_path in _trainable_run_paths(artifacts):
        manifest_path = _path_manifest(prediction_path)
        signature_path = prediction_path.parent / "detector_signature.json"
        if manifest_path is None or not signature_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("test_used_for_selection") is True:
            continue
        dataset, kir, seed = _parse_trainable_location(prediction_path, manifest)
        key = (dataset, kir, seed)
        if key in seen:
            continue
        seen.add(key)
        signature = _read_json(signature_path)
        center_to_intent = {
            str(item.get("cluster_id")): item.get("intent")
            for item in signature.get("spheres", [])
            if item.get("cluster_id") is not None
        }
        hashes = {
            "registry_sha256": _hash_from_manifest(manifest, "registry"),
            "canonical_manifest_sha256": _hash_from_manifest(manifest, "canonical_manifest"),
        }
        run_id = str(manifest.get("run_id") or f"s2c_trainable_k1__{dataset}__kir_{kir:.2f}__seed_{seed}")
        method = "S2C-Trainable-K1"
        meta = {
            "source_path": str(prediction_path),
            "contract_layer": "same_protocol_fair",
            "score_available": True,
            "selection_audit": "manifest:test_used_for_selection=false;selection=known_calibration_only",
        }
        with prediction_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                predicted_oos = _boolean_int(raw.get("predicted_is_oos"))
                row = _base_row(
                    dataset=dataset,
                    kir=kir,
                    seed=seed,
                    method=method,
                    backbone="MiniLM-trainable",
                    supervision_type="Known-only",
                    sample_id=raw.get("sample_id"),
                    true_label=raw.get("gold_intent"),
                    is_true_oos=_boolean_int(raw.get("gold_is_oos")),
                    predicted_label=raw.get("predicted_intent"),
                    predicted_oos=predicted_oos,
                    oos_score=raw.get("oos_score"),
                    confidence=None,
                    run_id=run_id,
                    registry_sha256=hashes["registry_sha256"],
                    canonical_manifest_sha256=hashes["canonical_manifest_sha256"],
                )
                row.update(
                    {
                        "nearest_intent": center_to_intent.get(str(raw.get("nearest_cluster"))),
                        "nearest_distance": _number(raw.get("distance")),
                        "boundary_score": _number(raw.get("oos_score")),
                        "center_id": _integer(raw.get("nearest_cluster")),
                        "radius": _number(raw.get("radius")),
                        "normalized_distance": _number(raw.get("oos_score")),
                    }
                )
                yield row, {**meta, "run_id": run_id}


def _frozen_prediction_paths(artifacts: Path) -> list[Path]:
    root = artifacts / "runs" / PROTOCOL_VERSION / "e2_gate_core_dense"
    return sorted(root.rglob("predictions/test.jsonl")) if root.exists() else []


def _parse_frozen_location(path: Path, manifest: Mapping[str, Any]) -> tuple[str, float, int, int, str]:
    config = manifest.get("config", {})
    dataset = normalise_dataset(config.get("dataset"))
    kir = _number(config.get("kir"))
    seed = _integer(config.get("seed"))
    k_gate = _integer(config.get("k_gate"))
    distance = str(config.get("distance", ""))
    if dataset and kir is not None and seed is not None and k_gate is not None:
        return dataset, float(kir), int(seed), int(k_gate), distance
    match = re.search(
        r"__([^_]+(?:77|150)?)__kir_(\d+\.\d+)__seed_(\d+).*__k_(\d+)__dist_([^_]+)",
        str(path),
    )
    if not match:
        raise ValueError(f"cannot parse Frozen location: {path}")
    return normalise_dataset(match.group(1)), float(match.group(2)), int(match.group(3)), int(match.group(4)), match.group(5)


def iter_frozen_rows(artifacts: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    selected_kirs = {0.25, 0.5, 0.75}
    selected_k = {1, 2}
    seen: set[tuple[str, float, int, int]] = set()
    for prediction_path in _frozen_prediction_paths(artifacts):
        manifest_path = prediction_path.parent.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("test_used_for_selection") is True:
            continue
        try:
            dataset, kir, seed, k_gate, distance = _parse_frozen_location(prediction_path, manifest)
        except ValueError:
            continue
        if round(kir, 2) not in selected_kirs or k_gate not in selected_k or distance != "mahalanobis_diag":
            continue
        key = (dataset, kir, seed, k_gate)
        if key in seen:
            continue
        seen.add(key)
        run_id = str(manifest.get("run_id") or prediction_path.parent.parent.name)
        hashes = {
            "registry_sha256": _hash_from_manifest(manifest, "registry"),
            "canonical_manifest_sha256": _hash_from_manifest(manifest, "canonical_manifest"),
        }
        meta = {
            "source_path": str(prediction_path),
            "contract_layer": "same_protocol_fair",
            "score_available": True,
            "selection_audit": "manifest:test_used_for_selection=false;selection=fixed_boundary_known_only_calibration",
            "run_id": run_id,
        }
        with prediction_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                predicted_oos = _boolean_int(raw.get("predicted_is_oos"))
                row = _base_row(
                    dataset=dataset,
                    kir=kir,
                    seed=seed,
                    method=f"S2C-Frozen-Gate-K{k_gate}",
                    backbone="MiniLM-frozen",
                    supervision_type="Known-only",
                    sample_id=raw.get("sample_id"),
                    true_label=raw.get("gold_intent"),
                    is_true_oos=_boolean_int(raw.get("gold_is_oos")),
                    predicted_label=raw.get("nearest_known_intent") if not predicted_oos else OOS_LABEL,
                    predicted_oos=predicted_oos,
                    oos_score=raw.get("oos_score"),
                    confidence=None,
                    run_id=run_id,
                    registry_sha256=hashes["registry_sha256"],
                    canonical_manifest_sha256=hashes["canonical_manifest_sha256"],
                )
                row.update(
                    {
                        "nearest_intent": raw.get("nearest_known_intent"),
                        "nearest_distance": _number(raw.get("distance")),
                        "boundary_score": _number(raw.get("oos_score")),
                        "center_id": _integer(raw.get("nearest_cluster")),
                        "radius": _number(raw.get("radius")),
                        "normalized_distance": _number(raw.get("oos_score")),
                    }
                )
                yield row, meta


def _mogb_prediction_paths(artifacts: Path) -> list[Path]:
    root = artifacts / "runs" / PROTOCOL_VERSION / "mogb_baseline_v1"
    return sorted(root.rglob("predictions.tsv")) if root.exists() else []


def _parse_mogb_location(path: Path, row: Mapping[str, Any]) -> tuple[str, float, int, str]:
    dataset = normalise_dataset(row.get("dataset"))
    kir = float(row["kir"])
    seed = int(row["seed"])
    method = str(row["method"])
    return dataset, kir, seed, method


def iter_mogb_rows(artifacts: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for prediction_path in _mogb_prediction_paths(artifacts):
        manifest_path = prediction_path.parent / "manifest.json"
        stats_path = prediction_path.parent / "ball_statistics.json"
        balls_path = prediction_path.parent / "balls.jsonl"
        if manifest_path is None or not stats_path.exists() or not balls_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("test_used_for_selection") is True:
            continue
        stats = _read_json(stats_path)
        ball_map: dict[int, dict[str, Any]] = {}
        with balls_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ball = json.loads(line)
                    if ball.get("ball_id") is not None:
                        ball_map[int(ball["ball_id"])] = ball
        hashes = {
            "registry_sha256": _hash_from_manifest(manifest, "registry"),
            "canonical_manifest_sha256": _hash_from_manifest(manifest, "canonical_manifest"),
        }
        with prediction_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for raw in reader:
                dataset, kir, seed, source_method = _parse_mogb_location(prediction_path, raw)
                method = MOGB_METHOD_LABELS.get(source_method, f"MOGB-{source_method}")
                predicted_oos = _boolean_int(raw.get("predicted_is_oos"))
                ball_id = _integer(raw.get("nearest_ball"))
                ball = ball_map.get(ball_id, {})
                run_id = str(
                    manifest.get("run_id")
                    or f"mogb__{dataset}__kir_{kir:.2f}__seed_{seed}__{source_method}"
                )
                row = _base_row(
                    dataset=dataset,
                    kir=kir,
                    seed=seed,
                    method=method,
                    backbone="MiniLM-frozen",
                    supervision_type="Known-only",
                    sample_id=raw.get("sample_id"),
                    true_label=raw.get("gold_intent"),
                    is_true_oos=_boolean_int(raw.get("gold_is_oos")),
                    predicted_label=raw.get("predicted_label"),
                    predicted_oos=predicted_oos,
                    oos_score=raw.get("normalized_score"),
                    confidence=None,
                    run_id=run_id,
                    registry_sha256=hashes["registry_sha256"],
                    canonical_manifest_sha256=hashes["canonical_manifest_sha256"],
                )
                row.update(
                    {
                        "boundary_score": _number(raw.get("normalized_score")),
                        "ball_id": ball_id,
                        "ball_count": _integer(stats.get("selected_balls")),
                        "ball_size": _integer(ball.get("sample_count")),
                        "ball_radius": _number(ball.get("radius")),
                        "ball_purity": _number(ball.get("purity")),
                    }
                )
                yield row, {
                    "source_path": str(prediction_path),
                    "contract_layer": "same_protocol_fair",
                    "score_available": True,
                    "selection_audit": "manifest:test_used_for_selection=false;Known-only",
                    "run_id": run_id,
                }


def _native_prediction_paths(artifacts: Path) -> list[Path]:
    root = artifacts / "runs" / PROTOCOL_VERSION / "native_baselines_trainable_v1"
    return sorted(
        path for path in root.rglob("test.jsonl") if path.parent.name == "predictions"
    ) if root.exists() else []


def _parse_native_location(path: Path, manifest: Mapping[str, Any]) -> tuple[str, float, int, str]:
    dataset = normalise_dataset(manifest.get("dataset"))
    kir = float(manifest["kir"])
    seed = int(manifest["seed"])
    method = str(manifest["method"]).lower()
    return dataset, kir, seed, method


def iter_native_rows(
    artifacts: Path, trainable_hashes: Mapping[tuple[str, float, int], Mapping[str, str | None]]
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for prediction_path in _native_prediction_paths(artifacts):
        manifest_path = prediction_path.parent.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete" or manifest.get("test_used_for_selection") is True:
            continue
        dataset, kir, seed, source_method = _parse_native_location(prediction_path, manifest)
        method = NATIVE_METHOD_LABELS.get(source_method, f"Native-{source_method}-Trainable-MiniLM")
        hashes = trainable_hashes.get((dataset, kir, seed), {})
        run_id = str(manifest.get("run_id") or f"native__{dataset}__kir_{kir:.2f}__seed_{seed}__{source_method}")
        meta = {
            "source_path": str(prediction_path),
            "contract_layer": "same_protocol_native_backbone_control",
            "score_available": True,
            "selection_audit": "manifest:test_used_for_selection=false;uses_oos_for_calibration=false",
            "run_id": run_id,
        }
        with prediction_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                predicted_oos = _boolean_int(raw.get("predicted_is_oos"))
                predicted_label = raw.get("nearest_known_intent") if not predicted_oos else OOS_LABEL
                row = _base_row(
                    dataset=dataset,
                    kir=kir,
                    seed=seed,
                    method=method,
                    backbone="MiniLM-trainable",
                    supervision_type="Known-only",
                    sample_id=raw.get("sample_id"),
                    true_label=raw.get("gold_intent"),
                    is_true_oos=_boolean_int(raw.get("gold_is_oos")),
                    predicted_label=predicted_label,
                    predicted_oos=predicted_oos,
                    oos_score=raw.get("oos_score"),
                    confidence=None,
                    run_id=run_id,
                    registry_sha256=hashes.get("registry_sha256"),
                    canonical_manifest_sha256=hashes.get("canonical_manifest_sha256"),
                )
                row["boundary_score"] = _number(raw.get("oos_score"))
                yield row, meta


def _adb_priority(path: Path) -> tuple[int, str]:
    match = re.search(r"adb_gpu_runtime_v(\d+b?)", str(path))
    version = match.group(1) if match else "0"
    order = {"4b": 0, "3": 1, "2": 2, "1": 3, "4": 4}
    return order.get(version, 9), str(path)


def _adb_export_dir(root: Path, dataset: str, seed: int, kir: float) -> Path | None:
    export_dataset = {
        "banking": "banking77",
        "banking77": "banking77",
        "oos": "clinc150",
        "clinc150": "clinc150",
        "stackoverflow": "stackoverflow",
    }.get(dataset, dataset)
    base = root / "data" / "exports" / PROTOCOL_VERSION / "adb" / export_dataset / f"seed_{seed}"
    candidates = [base / f"kir_{kir:.2f}", base / f"kir_{kir:.1f}", base / f"kir{int(kir * 100)}"]
    return next((candidate for candidate in candidates if (candidate / "sample_ids.json").exists()), None)


def _adb_prediction_dir(manifest: Mapping[str, Any]) -> Path | None:
    audit = manifest.get("artifact_audit", {})
    directories = audit.get("prediction_directories", []) if isinstance(audit, Mapping) else []
    for value in directories:
        path = Path(str(value))
        if (path / "y_true.npy").exists() and (path / "y_pred.npy").exists():
            return path
    return None


def _labels_from_export(export_dir: Path, unknown_id: int | None = None) -> tuple[dict[int, str], int]:
    label_map = _read_json(export_dir / "label_map.json")
    inverse = {int(value): str(key) for key, value in label_map.items()}
    known_labels = _read_json(export_dir / "known_labels.json")
    if isinstance(known_labels, Mapping):
        known_count = len(known_labels)
    else:
        known_count = len(known_labels)
    return inverse, unknown_id if unknown_id is not None else max(inverse, default=known_count - 1) + 1


def iter_adb_rows(root: Path, artifacts: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    manifests = sorted(artifacts.glob("external/adb_gpu_runtime_v*/**/run_manifest.json"), key=_adb_priority)
    selected: dict[tuple[str, float, int], Path] = {}
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete" or not manifest.get("artifact_audit", {}).get("complete"):
            continue
        dataset = normalise_dataset(manifest.get("dataset"))
        kir = _number(manifest.get("known_cls_ratio"))
        seed = _integer(manifest.get("seed"))
        if kir is None or seed is None:
            continue
        selected.setdefault((dataset, float(kir), seed), manifest_path)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a project dependency
        return
    for key, manifest_path in sorted(selected.items()):
        dataset, kir, seed = key
        manifest = _read_json(manifest_path)
        prediction_dir = _adb_prediction_dir(manifest)
        export_dir = _adb_export_dir(root, dataset, seed, kir)
        if prediction_dir is None or export_dir is None:
            continue
        sample_ids = _read_json(export_dir / "sample_ids.json").get("test", [])
        y_true = np.load(prediction_dir / "y_true.npy")
        y_pred = np.load(prediction_dir / "y_pred.npy")
        if len(sample_ids) != len(y_true) or len(y_true) != len(y_pred):
            continue
        inverse, unknown_id = _labels_from_export(export_dir, _integer(manifest.get("unknown_label_id")))
        export_manifest = _read_json(export_dir / "export_manifest.json")
        run_id = f"adb_external__{dataset}__kir_{kir:.2f}__seed_{seed}__{manifest_path.parts[-5]}"
        meta = {
            "source_path": str(prediction_dir / "y_pred.npy"),
            "contract_layer": "external_backbone",
            "score_available": False,
            "selection_audit": "external_runtime_manifest_does_not_declare_test_selection;Known-only training contract retained as external",
            "run_id": run_id,
        }
        for sample_id, true_id, pred_id in zip(sample_ids, y_true.tolist(), y_pred.tolist()):
            true_id = int(true_id)
            pred_id = int(pred_id)
            true_oos = int(true_id == unknown_id)
            predicted_oos = int(pred_id == unknown_id)
            row = _base_row(
                dataset=dataset,
                kir=kir,
                seed=seed,
                method="ADB-external-BERT",
                backbone="BERT",
                supervision_type="Known-only-external-runtime",
                sample_id=sample_id,
                true_label=OOS_LABEL if true_oos else inverse.get(true_id, f"label_{true_id}"),
                is_true_oos=true_oos,
                predicted_label=OOS_LABEL if predicted_oos else inverse.get(pred_id, f"label_{pred_id}"),
                predicted_oos=predicted_oos,
                oos_score=None,
                confidence=None,
                run_id=run_id,
                registry_sha256=export_manifest.get("registry_sha256"),
                canonical_manifest_sha256=export_manifest.get("canonical_manifest_sha256"),
            )
            yield row, meta


def iter_all_rows(root: Path, artifacts: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    yield from iter_trainable_rows(artifacts)
    yield from iter_frozen_rows(artifacts)
    trainable_hashes: dict[tuple[str, float, int], dict[str, str | None]] = {}
    for prediction_path in _trainable_run_paths(artifacts):
        manifest_path = _path_manifest(prediction_path)
        if manifest_path is None:
            continue
        manifest = _read_json(manifest_path)
        try:
            dataset, kir, seed = _parse_trainable_location(prediction_path, manifest)
        except ValueError:
            continue
        trainable_hashes.setdefault(
            (dataset, kir, seed),
            {
                "registry_sha256": _hash_from_manifest(manifest, "registry"),
                "canonical_manifest_sha256": _hash_from_manifest(manifest, "canonical_manifest"),
            },
        )
    yield from iter_native_rows(artifacts, trainable_hashes)
    yield from iter_mogb_rows(artifacts)
    yield from iter_adb_rows(root, artifacts)


def blocked_method_rows(root: Path, artifacts: Path) -> list[dict[str, Any]]:
    external = artifacts / "external"
    return [
        {
            "method": "TextOIR-KNNCL",
            "status": "not_completed_in_current_protocol",
            "contract_layer": "planned_traditional_detector",
            "dataset": "all",
            "kir": "all",
            "seed": "all",
            "reason": "No complete final-metrics prediction artifact was found; do not fill with historical fulltex rows.",
            "source": "docs/analysis/BASELINE_EXECUTION_STATUS_V1.md",
            "final_metrics_available": False,
            "include_in_unified_rows": False,
        },
        {
            "method": "TextOIR-OpenMax-DOC-DeepUnk",
            "status": "not_completed_in_current_protocol",
            "contract_layer": "planned_traditional_detector",
            "dataset": "all",
            "kir": "all",
            "seed": "all",
            "reason": "Historical fulltex comparison is retained as historical evidence, not a current row-level contract.",
            "source": "docs/analysis/MINILM_TRAINABLE_VS_FULLTEX_AND_BASELINES_V1.md",
            "final_metrics_available": False,
            "include_in_unified_rows": False,
        },
        {
            "method": "DA-ADB",
            "status": "external_summary_only",
            "contract_layer": "external_backbone",
            "dataset": "stackoverflow",
            "kir": 0.5,
            "seed": "42|87|100",
            "reason": "Existing current-protocol BERT/DA-ADB summaries are not row-level aligned with the MiniLM fair matrix; keep separate.",
            "source": "docs/analysis/DA_ADB_CURRENT_PROTOCOL_SUMMARY_V1.md",
            "final_metrics_available": True,
            "include_in_unified_rows": False,
        },
        {
            "method": "DCLOOS-current",
            "status": "official_timeout_no_final_metrics",
            "contract_layer": "different_supervision",
            "dataset": "stackoverflow",
            "kir": 0.5,
            "seed": 42,
            "reason": "Fixed-registry run was interrupted; manifest explicitly excludes intermediate predictions from final metrics.",
            "source": str(external / "dcloos_stackoverflow_kir050_seed42_fixed_registry_v1" / "run_manifest.json"),
            "final_metrics_available": False,
            "include_in_unified_rows": False,
        },
        {
            "method": "DCLOOS-reduced",
            "status": "reduced_complete_different_supervision",
            "contract_layer": "different_supervision",
            "dataset": "oos+squad",
            "kir": 0.75,
            "seed": 888,
            "reason": "Recovered intermediate prediction with pseudo-OOS plus external SQuAD OOS; not a Known-only test contract.",
            "source": str(external / "dcloos_official_oos_kir75_seed888_reduced_v2" / "recovery_manifest.json"),
            "final_metrics_available": True,
            "include_in_unified_rows": False,
        },
        {
            "method": "DCLOOS-official",
            "status": "official_timeout_no_final_metrics",
            "contract_layer": "different_supervision",
            "dataset": "oos+squad",
            "kir": 0.75,
            "seed": 888,
            "reason": "Official single-cell process hit the runtime ceiling; intermediate predictions are excluded.",
            "source": str(external / "dcloos_official_single_cell_v1" / "run_manifest.json"),
            "final_metrics_available": False,
            "include_in_unified_rows": False,
        },
    ]


def _group_key(group: Mapping[str, Any]) -> tuple[str, float, int]:
    return str(group["dataset"]), float(group["kir"]), int(group["seed"])


def build_alignment(groups: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, float, int], list[Mapping[str, Any]]] = defaultdict(list)
    for group in groups:
        by_cell[_group_key(group)].append(group)
    rows: list[dict[str, Any]] = []
    for cell, cell_groups in sorted(by_cell.items()):
        reference = next((g for g in cell_groups if g["method"] == "S2C-Trainable-K1"), cell_groups[0])
        for group in sorted(cell_groups, key=lambda item: (item["method"], item["run_id"])):
            sequence_match = group["sample_id_sequence_sha256"] == reference["sample_id_sequence_sha256"]
            multiset_match = (
                group["sample_id_multiset_xor"] == reference["sample_id_multiset_xor"]
                and group["sample_id_multiset_sum"] == reference["sample_id_multiset_sum"]
                and group["row_count"] == reference["row_count"]
            )
            label_match = group["true_label_sequence_sha256"] == reference["true_label_sequence_sha256"]
            if sequence_match and multiset_match and label_match:
                status = "aligned"
                reason = "sample_id order/set and true-label sequence match reference"
            elif group["contract_layer"] == "external_backbone":
                status = "external_contract_unaligned"
                reason = "external backbone row contract differs; mismatch retained without fair pooling"
            else:
                status = "failed_alignment"
                reason = "sample_id or true-label sequence mismatch"
            rows.append(
                {
                    "dataset": cell[0],
                    "kir": cell[1],
                    "seed": cell[2],
                    "reference_method": reference["method"],
                    "method": group["method"],
                    "run_id": group["run_id"],
                    "contract_layer": group["contract_layer"],
                    "backbone": group["backbone"],
                    "supervision_type": group["supervision_type"],
                    "sample_count": group["row_count"],
                    "reference_count": reference["row_count"],
                    "sample_id_order_match": sequence_match,
                    "sample_id_set_match": multiset_match,
                    "true_label_match": label_match,
                    "score_available": group["score_available"],
                    "selection_audit": group["selection_audit"],
                    "status": status,
                    "reason": reason,
                    "source_path": group["source_path"],
                }
            )
    return rows


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_contract(root: Path, public_out: Path, local_out: Path) -> dict[str, Any]:
    """Normalize completed artifacts and write the contract audit outputs."""

    artifacts = root.parent / "artifacts" / "s2c"
    public_out.mkdir(parents=True, exist_ok=True)
    local_out.mkdir(parents=True, exist_ok=True)
    prediction_path = local_out / "predictions.jsonl.gz"
    groups: dict[tuple[str, float, int, str, str], GroupAudit] = {}
    row_count = 0
    invalid_count = 0
    with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
        for row, meta in iter_all_rows(root, artifacts):
            errors = validate_row(row)
            invalid_count += int(bool(errors))
            row_count += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            key = (
                str(row["dataset"]),
                float(row["kir"]),
                int(row["seed"]),
                str(row["method"]),
                str(row["run_id"]),
            )
            group = groups.get(key)
            if group is None:
                group = GroupAudit(
                    dataset=str(row["dataset"]),
                    kir=float(row["kir"]),
                    seed=int(row["seed"]),
                    method=str(row["method"]),
                    run_id=str(row["run_id"]),
                    backbone=str(row["backbone"]),
                    supervision_type=str(row["supervision_type"]),
                    source_path=str(meta["source_path"]),
                    contract_layer=str(meta["contract_layer"]),
                    score_available=bool(meta["score_available"]),
                    selection_audit=str(meta["selection_audit"]),
                )
                groups[key] = group
            group.add(row)
    group_rows = [groups[key].as_dict() for key in sorted(groups)]
    alignment = build_alignment(group_rows)
    failed_alignment = [row for row in alignment if row["status"] != "aligned"]
    write_csv(
        public_out / "alignment.csv",
        alignment,
        list(alignment[0].keys()) if alignment else ["status"],
    )
    write_csv(
        public_out / "failed_alignment.csv",
        failed_alignment,
        list(failed_alignment[0].keys()) if failed_alignment else ["status"],
    )
    blocked = blocked_method_rows(root, artifacts)
    write_csv(
        public_out / "blocked_methods.csv",
        blocked,
        [
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
    schema_manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "row_fields": ALL_FIELDS,
        "base_fields": BASE_FIELDS,
        "method_specific_fields": {
            "gate": GATE_FIELDS,
            "mogb": MOGB_FIELDS,
            "dcloos": DCLOOS_FIELDS,
        },
        "row_level_artifact": str(prediction_path),
        "row_level_artifact_policy": "local-only compressed JSONL; no raw text or embeddings",
        "row_count": row_count,
        "group_count": len(group_rows),
        "invalid_row_count": invalid_count,
        "aligned_rows": sum(row["status"] == "aligned" for row in alignment),
        "failed_alignment_rows": len(failed_alignment),
        "score_available_groups": sum(bool(row["score_available"]) for row in group_rows),
        "selection_policy": "test OOS is never used for selection; source manifests are audited and external undeclared fields remain external",
        "source_families": ["S2C-Trainable-K1", "native MiniLM controls", "MOGB fair components", "ADB external BERT"],
        "blocked_method_count": len(blocked),
    }
    (public_out / "schema_manifest.json").write_text(
        json.dumps(schema_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Unified prediction contract audit V1",
        "",
        f"- protocol: `{PROTOCOL_VERSION}`",
        f"- normalized rows: `{row_count}`",
        f"- method runs: `{len(group_rows)}`",
        f"- aligned method-cell rows: `{schema_manifest['aligned_rows']}`",
        f"- non-aligned/external rows: `{schema_manifest['failed_alignment_rows']}`",
        f"- invalid rows: `{invalid_count}`",
        "",
        "## Contract boundary",
        "",
        "Trainable K=1、native MiniLM controls 和 MOGB fair components are read from completed current-protocol artifacts. ADB rows are retained as an external BERT contract and have nullable OOS scores because its runtime artifact exposes final labels rather than the S2C score semantics. DCLOOS and unavailable traditional detectors are listed in `blocked_methods.csv`; no intermediate prediction is promoted to a final metric.",
        "",
        "## Audit interpretation",
        "",
        "`alignment.csv` compares each method-cell with S2C-Trainable-K1 using deterministic ordered sample-id and label digests. `aligned` means the row sequence and labels match; `external_contract_unaligned` is intentionally not a fair-comparison failure claim, but a boundary against pooling different backbone or supervision contracts.",
        "",
        "The compressed row-level artifact is local-only and contains anonymous sample IDs, not raw text or embeddings.",
    ]
    (public_out / "contract_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"schema": schema_manifest, "groups": group_rows, "alignment": alignment, "blocked": blocked}


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    build_contract(
        repo_root,
        repo_root / "results" / "analysis" / "unified_prediction_contract_v1",
        repo_root.parent / "artifacts" / "s2c" / "analysis" / "unified_prediction_contract_v1",
    )
