"""Workspace-relative paths for the active s2c project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolve active project paths without machine-specific defaults."""

    project_root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "WorkspacePaths":
        here = (start or Path(__file__)).resolve()
        project_root = next(
            (candidate for candidate in (here, *here.parents) if (candidate / "src").is_dir() and (candidate / "configs").is_dir()),
            None,
        )
        if project_root is None:
            raise RuntimeError("Unable to locate the s2c project root")
        return cls(project_root=project_root)

    @property
    def workspace_root(self) -> Path:
        # Prefer the nearest parent containing the canonical asset layers so
        # the active ``<workspace>/s2c`` layout remains relocatable.
        for candidate in (self.project_root.parent, *self.project_root.parents):
            if (candidate / "assets").is_dir() and (candidate / "artifacts").is_dir():
                return candidate
        return self.project_root.parent

    @property
    def config_root(self) -> Path:
        return self.project_root / "configs"

    @property
    def model_root(self) -> Path:
        return self.workspace_root / "assets" / "models"

    @property
    def dataset_root(self) -> Path:
        return self.workspace_root / "assets" / "datasets" / "s2c"

    @property
    def artifact_root(self) -> Path:
        return self.workspace_root / "artifacts" / "s2c"

    @property
    def prepared_data_root(self) -> Path:
        return self.dataset_root / "prepared" / "data"

    @property
    def source_data_root(self) -> Path:
        return self.dataset_root / "source"

    @property
    def smollm135m(self) -> Path:
        return self.model_root / "smollm135m"

    @property
    def smollm17b(self) -> Path:
        return self.model_root / "smollm17b"

    @property
    def minilm(self) -> Path:
        return self.model_root / "all-MiniLM-L6-v2"
