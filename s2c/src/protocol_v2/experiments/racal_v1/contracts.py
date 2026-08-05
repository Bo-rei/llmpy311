"""Configuration, provenance and data-contract helpers for RACAL-v1."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from protocol_v2.data.hashing import sha256_file, sha256_json
from protocol_v2.experiments.mechanism_runner import E3Bundle


STAGE = "racal_v1"
DATASET = "stackoverflow"
KIR = 0.50
FORMAL_SEEDS = (13, 42, 87)


@dataclass(frozen=True)
class RacalConfig:
    protocol_version: str
    model_path: str
    dataset: str
    kir: float
    seeds: tuple[int, ...]
    warmup_epochs: int
    finetune_epochs: int
    patience: int
    batch_size: int
    max_length: int
    projection_hidden_dim: int
    projection_lr: float
    backbone_lr: float
    temperature: float
    intra_weight: float
    inter_weight: float
    classification_weight: float
    inter_margin: float
    radius_lambda: float
    threshold: float
    device: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RacalConfig":
        required = ("protocol_version", "model_path", "dataset", "kir", "seeds")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"RACAL config is missing keys: {missing}")
        if str(payload["dataset"]).lower() != DATASET:
            raise ValueError("RACAL-v1 first stage is restricted to StackOverflow")
        if abs(float(payload["kir"]) - KIR) > 1e-12:
            raise ValueError("RACAL-v1 first stage is restricted to KIR=0.50")
        seeds = tuple(int(seed) for seed in payload["seeds"])
        if not seeds or any(seed not in FORMAL_SEEDS for seed in seeds):
            raise ValueError(f"RACAL seeds must be drawn from {FORMAL_SEEDS}: {seeds}")
        return cls(
            protocol_version=str(payload["protocol_version"]),
            model_path=str(payload["model_path"]),
            dataset=DATASET,
            kir=KIR,
            seeds=seeds,
            warmup_epochs=int(payload.get("warmup_epochs", 1)),
            finetune_epochs=int(payload.get("finetune_epochs", 3)),
            patience=int(payload.get("patience", 1)),
            batch_size=int(payload.get("batch_size", 64)),
            max_length=int(payload.get("max_length", 256)),
            projection_hidden_dim=int(payload.get("projection_hidden_dim", 256)),
            projection_lr=float(payload.get("projection_lr", 2e-4)),
            backbone_lr=float(payload.get("backbone_lr", 2e-5)),
            temperature=float(payload.get("temperature", 0.07)),
            intra_weight=float(payload.get("intra_weight", 0.1)),
            inter_weight=float(payload.get("inter_weight", 0.1)),
            classification_weight=float(payload.get("classification_weight", 1.0)),
            inter_margin=float(payload.get("inter_margin", 0.20)),
            radius_lambda=float(payload.get("radius_lambda", 1.0)),
            threshold=float(payload.get("threshold", 1.0)),
            device=str(payload.get("device", "auto")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": STAGE,
            "protocol_version": self.protocol_version,
            "model_path": self.model_path,
            "dataset": self.dataset,
            "kir": self.kir,
            "seeds": list(self.seeds),
            "warmup_epochs": self.warmup_epochs,
            "finetune_epochs": self.finetune_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "projection_hidden_dim": self.projection_hidden_dim,
            "projection_lr": self.projection_lr,
            "backbone_lr": self.backbone_lr,
            "temperature": self.temperature,
            "intra_weight": self.intra_weight,
            "inter_weight": self.inter_weight,
            "classification_weight": self.classification_weight,
            "inter_margin": self.inter_margin,
            "radius_lambda": self.radius_lambda,
            "threshold": self.threshold,
            "device": self.device,
            "selection": "known_calibration_only",
            "test_used_for_selection": False,
            "oos_used_for_training": False,
            "unsupported_in_stage": [
                "adaptive_center_activation",
                "fixed_k2",
                "proxy_oos",
                "parent_guard",
                "rc_ambl_energy_gap_threshold",
            ],
        }


def load_config(path: Path) -> RacalConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"RACAL config must be a mapping: {path}")
    return RacalConfig.from_mapping(payload)


def rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(str(row["sample_id"]) for row in rows).encode()).hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def validate_bundle(bundle: E3Bundle) -> dict[str, Any]:
    train, calibration, test = bundle.views.train, bundle.views.calibration, bundle.views.test
    id_sets = {name: {str(row["sample_id"]) for row in rows} for name, rows in (("train", train), ("calibration", calibration), ("test", test))}
    overlaps = {
        "train_calibration": sorted(id_sets["train"] & id_sets["calibration"]),
        "train_test": sorted(id_sets["train"] & id_sets["test"]),
        "calibration_test": sorted(id_sets["calibration"] & id_sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"RACAL split sample IDs overlap: {overlaps}")
    if any(int(row["label"]) != 0 for row in train + calibration):
        raise ValueError("RACAL train/calibration must contain Known rows only")
    if len(bundle.train) != len(train) or len(bundle.calibration) != len(calibration) or len(bundle.test) != len(test):
        raise ValueError("RACAL embedding/view row counts are inconsistent")
    return {
        "train_count": len(train),
        "calibration_count": len(calibration),
        "test_count": len(test),
        "test_known_count": int(sum(int(row["label"]) == 0 for row in test)),
        "test_oos_count": int(sum(int(row["label"]) == 1 for row in test)),
        "train_sample_ids_sha256": rows_hash(train),
        "calibration_sample_ids_sha256": rows_hash(calibration),
        "test_sample_ids_sha256": rows_hash(test),
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "test_used_for_selection": False,
    }


def git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=project_root, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    nested = project_root / "third_party" / "mogb_official"
    nested_status = ""
    nested_head = None
    if (nested / ".git").exists() or (nested / "HEAD").exists():
        nested_status = subprocess.run(["git", "status", "--short"], cwd=nested, check=True, capture_output=True, text=True).stdout.strip()
        nested_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=nested, check=True, capture_output=True, text=True).stdout.strip()
    return {
        "base_commit": run("rev-parse", "HEAD"),
        "git_dirty": bool(run("status", "--short")),
        "project_status": run("status", "--short"),
        "third_party_mogb_head": nested_head,
        "third_party_mogb_dirty": bool(nested_status),
        "third_party_mogb_status": nested_status,
        "third_party_note": "Read-only historical audit checkout; not modified by RACAL-v1.",
    }


def model_file_hashes(model_path: Path) -> dict[str, str]:
    names = ("config.json", "tokenizer.json", "model.safetensors", "pytorch_model.bin")
    found = {name: sha256_file(model_path / name) for name in names if (model_path / name).is_file()}
    if "config.json" not in found or not ({"model.safetensors", "pytorch_model.bin"} & found.keys()):
        raise FileNotFoundError(f"Incomplete local MiniLM model: {model_path}")
    return found


def provenance_payload(project_root: Path, config_path: Path, config: RacalConfig, model_path: Path) -> dict[str, Any]:
    state = git_state(project_root)
    return {
        "schema_version": "s2c.racal_v1.provenance.v1",
        "stage": STAGE,
        "protocol_version": config.protocol_version,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_hash": sha256_json(config.as_dict()),
        "model_path": str(model_path),
        "model_files": model_file_hashes(model_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git": state,
        "historical_artifacts_immutable": True,
        "test_used_for_selection": False,
        "oos_used_for_training": False,
    }


def reference_run_name(seed: int) -> str:
    return (
        f"protocol_v2_textoir_v1__stackoverflow__kir_0.50__seed_{seed}__"
        "repr_frozen_minilm__k_1__dist_mahalanobis_diag__boundary_mean_std"
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)
