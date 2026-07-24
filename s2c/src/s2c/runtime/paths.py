"""Path rules for protocol_v2 without changing legacy v19 path semantics."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _discover_project(start: Path | None = None) -> Path:
    explicit = _env_path("S2C_PROJECT_ROOT")
    if explicit is not None:
        return explicit
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Unable to locate s2c project root; set S2C_PROJECT_ROOT explicitly.")


@dataclass(frozen=True)
class ProtocolV2Paths:
    """Resolve the new data layer while leaving ``WorkspacePaths`` legacy-only."""

    project_root: Path
    data_root: Path
    artifacts_root: Path
    results_root: Path
    legacy_root: Path
    textoir_import_root: Path | None
    # The rejected TEXTOIR-derived candidate remains at ``protocol_v2`` for
    # historical audit only.  New commands must be safe by default and resolve
    # the admitted official reconstruction unless callers deliberately select
    # a different version through ``S2C_DATASET_VERSION``.
    dataset_version: str = "protocol_v2_official_v1"

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProtocolV2Paths":
        project = _discover_project(start)
        workspace = project.parent
        return cls(
            project_root=project,
            data_root=_env_path("S2C_DATA_ROOT") or project / "data",
            artifacts_root=_env_path("S2C_ARTIFACTS_ROOT") or workspace / "artifacts" / "s2c",
            results_root=_env_path("S2C_RESULTS_ROOT") or project / "results",
            legacy_root=workspace / "assets" / "datasets" / "s2c",
            textoir_import_root=_env_path("S2C_TEXTOIR_IMPORT_ROOT"),
            dataset_version=os.environ.get("S2C_DATASET_VERSION", "protocol_v2_official_v1"),
        )

    @property
    def protocol_root(self) -> Path:
        return self.data_root / "canonical" / self.dataset_version

    @property
    def registry_root(self) -> Path:
        return self.data_root / "registries" / self.dataset_version

    @property
    def view_root(self) -> Path:
        return self.data_root / "views" / self.dataset_version

    @property
    def export_root(self) -> Path:
        return self.data_root / "exports" / self.dataset_version

    @property
    def manifest_root(self) -> Path:
        return self.data_root / "manifests" / self.dataset_version

    @property
    def run_root(self) -> Path:
        return self.artifacts_root / "runs" / self.dataset_version

    @property
    def embedding_cache_root(self) -> Path:
        return self.artifacts_root / "cache" / "embeddings" / self.dataset_version

    @property
    def experiment_admission_path(self) -> Path:
        """唯一的正式实验准入开关；数据审计未通过时默认拒绝。"""
        return self.project_root / "configs" / "data" / "protocol_v2_admission.json"

    def require_experiment_admission(self, dataset: str | None = None) -> dict[str, object]:
        """Fail closed before any training, embedding or evaluation writes.

        该检查不替代逐文件数据验证；它只防止来源裁决尚未通过时，旧的 candidate
        registry/view 被误当作正式数据继续 resume。
        """
        path = self.experiment_admission_path
        if not path.is_file():
            raise RuntimeError(f"Missing protocol_v2 experiment admission record: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        admitted_versions = payload.get("admitted_dataset_versions")
        version_allowed = not admitted_versions or self.dataset_version in admitted_versions
        dataset_statuses = payload.get("dataset_admission", {})
        dataset_allowed = (
            dataset is None
            or not dataset_statuses
            or dataset_statuses.get(dataset) == "admitted"
        )
        if payload.get("status") not in {"admitted", "partially_admitted"} or not version_allowed or not dataset_allowed:
            reason = payload.get("reason", "data provenance has not admitted formal experiments")
            target = f" dataset={dataset}" if dataset is not None else ""
            raise RuntimeError(
                f"{self.dataset_version} formal experiments are blocked{target}: {reason}"
            )
        return payload

    def require_textoir_import_root(self, override: Path | None = None) -> Path:
        """Return a TEXTOIR root only for the explicit import command."""
        candidate = (override or self.textoir_import_root or self.project_root.parent / "textoir").resolve()
        data = candidate / "data"
        if not data.is_dir():
            raise FileNotFoundError(f"TEXTOIR import source is unavailable: {data}")
        return candidate

    def reject_textoir_runtime_path(self, path: Path) -> None:
        """Fail closed if a non-import command receives a TEXTOIR data path."""
        resolved = path.resolve()
        textoir_data = (self.project_root.parent / "textoir" / "data").resolve()
        if resolved == textoir_data or textoir_data in resolved.parents:
            raise ValueError(f"protocol_v2 runtime data must not point to textoir/data: {resolved}")
