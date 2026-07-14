"""The only supported command-line entrypoint for active s2c workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.runtime import ArtifactRegistry, WorkspacePaths, load_profile


@dataclass(frozen=True)
class WorkflowPlan:
    action: str
    command: tuple[str, ...]
    cwd: Path

    def as_dict(self) -> dict[str, object]:
        return {"action": self.action, "command": list(self.command), "cwd": str(self.cwd)}


def _ratio(kir: int) -> str:
    return f"{kir / 100:.2f}"


def _build_plan(args: argparse.Namespace, paths: WorkspacePaths) -> WorkflowPlan:
    profile = load_profile(paths.config_root / "profiles.yaml", args.dataset)
    base = (sys.executable, "-m")
    common = ("--datasets", profile.dataset, "--kir_values", _ratio(args.kir), "--seed", str(args.seed))
    if args.action == "prepare":
        command: Sequence[str] = (
            *base,
            "scripts.data.active.rebuild_multi_dataset_v19",
            *common,
            "--output_root",
            str(paths.prepared_data_root / "multidataset" / "v19"),
        )
        if profile.stackoverflow_known_selection_strategy:
            command = (*command, "--stackoverflow_known_selection_strategy", profile.stackoverflow_known_selection_strategy)
    elif args.action == "train":
        command = (
            *base,
            "tools.analysis.run_multi_dataset_training_v19",
            *common,
            "--data_root_base",
            str(paths.prepared_data_root / "multidataset" / "v19"),
            "--artifact_root",
            str(paths.artifact_root / "outputs" / "experiments" / "multi_dataset_v19"),
            "--model_path",
            str(paths.smollm135m),
            "--device",
            args.device,
        )
    elif args.action == "evaluate":
        command = (
            *base,
            "tools.analysis.run_multi_dataset_benchmark_v19",
            *common,
            "--data_root_base",
            str(paths.prepared_data_root / "multidataset" / "v19"),
            "--artifact_root",
            str(paths.artifact_root / "outputs" / "experiments" / "multi_dataset_v19"),
            "--model_path",
            str(paths.smollm135m),
            "--device",
            args.device,
        )
    else:
        registry = ArtifactRegistry.load(paths.config_root / "artifacts.yaml")
        registry.require_runnable(args.anchor)
        command = (
            *base,
            "tools.analysis.run_structure_backbone_ablation_v19",
            "--datasets",
            profile.dataset,
            "--kir_values",
            _ratio(args.kir),
            "--model_path",
            str(paths.smollm135m),
            "--device",
            args.device,
        )
    return WorkflowPlan(action=args.action, command=tuple(command), cwd=paths.project_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical s2c research workflow CLI")
    parser.add_argument("action", choices=("prepare", "train", "evaluate", "ablate"))
    parser.add_argument("--dataset", choices=("clinc150", "banking77_oos", "snips", "stackoverflow"), required=True)
    parser.add_argument("--kir", choices=(25, 50, 75), type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor", default="clinc150-kir50-frozen")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = WorkspacePaths.discover(Path.cwd())
    plan = _build_plan(args, paths)
    if args.dry_run:
        print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=2))
        return 0
    return subprocess.run(plan.command, cwd=plan.cwd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
