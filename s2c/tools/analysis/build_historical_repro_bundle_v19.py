#!/usr/bin/env python3
"""Build the historical reproduction bundle index and status summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis.historical_best_pipeline_v19 import HISTORICAL_BEST_PIPELINE


VERIFIED_MAINCHAIN = {
    "tools/analysis/run_prototype_gate_frozen_baseline_v19.py",
    "tools/eval/eval_system_pipeline_v19.py",
    "tools/analysis/historical_best_pipeline_v19.py",
    "tools/analysis/replay_historical_chain_v19.py",
}

FROZEN_DEPENDENCY = {
    "outputs/experiments/archive/sweeps/2026-03-23/gate_l2_mix2_train/gate_l2_mix2_true_lambda_1p6/detector.json",
    "outputs/experiments/archive/sweeps/2026-03-23/pipeline_phase3_proto_eval/pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json",
}


def classify_file_status(relpath: str) -> str:
    """Classify one file or artifact in the historical bundle."""

    if relpath in VERIFIED_MAINCHAIN:
        return "VERIFIED_MAINCHAIN"
    if relpath in FROZEN_DEPENDENCY:
        return "FROZEN_DEPENDENCY"
    if relpath.startswith("tools/analysis/run_multi_dataset") or relpath.startswith(
        "data/multidataset/"
    ):
        return "REFERENCE_ONLY"
    return "REFERENCE_ONLY"


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_bundle_summary(replay_root: Path) -> Dict[str, object]:
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    replay_manifest = {}
    replay_manifest_path = replay_root / "replay_manifest.json"
    if replay_manifest_path.exists():
        replay_manifest = _load_json(replay_manifest_path)

    tracked_items: List[str] = [
        "tools/analysis/run_prototype_gate_frozen_baseline_v19.py",
        "tools/eval/eval_system_pipeline_v19.py",
        "tools/analysis/historical_best_pipeline_v19.py",
        "tools/analysis/replay_historical_chain_v19.py",
        strict["reference_eval_results"],
        strict["frozen_detector_path"],
    ]
    return {
        "strict_replay": strict,
        "replay_manifest_path": str(replay_manifest_path) if replay_manifest_path.exists() else None,
        "replay_manifest": replay_manifest,
        "file_status": {
            item: classify_file_status(str(item))
            for item in tracked_items
        },
    }


def render_index(summary: Dict[str, object]) -> str:
    strict = summary["strict_replay"]
    file_status = summary["file_status"]
    replay_manifest = summary.get("replay_manifest", {})
    lines = [
        "# Historical Reproduction Bundle",
        "",
        "## Strict Historical Protocol",
        "",
        f"- `data_root`: `{strict['data_root']}`",
        f"- `known_intents_path`: `{strict['known_intents_path']}`",
        f"- `gate_encoder_path`: `{strict['gate_encoder_path']}`",
        f"- `reference_eval_results`: `{strict['reference_eval_results']}`",
        "",
        "## Replay Status",
        "",
    ]
    for stage in replay_manifest.get("stages", []):
        lines.append(f"- `{stage['name']}`: `{stage['status']}`")
    lines.extend(
        [
            "",
        "## File Status",
        "",
        ]
    )
    for relpath, status in file_status.items():
        lines.append(f"- `{status}`: `{relpath}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical reproduction bundle")
    parser.add_argument(
        "--output_dir",
        default="docs/historical_repro_bundle",
        help="Output directory for the bundle index.",
    )
    parser.add_argument(
        "--replay_root",
        default="outputs/reports/historical_replay_20260414",
        help="Replay root containing replay_manifest.json.",
    )
    args = parser.parse_args()

    replay_root = PROJECT_ROOT / args.replay_root
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_bundle_summary(replay_root)

    summary_path = replay_root / "bundle_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    index_path = output_dir / "INDEX.md"
    index_path.write_text(render_index(summary), encoding="utf-8")

    print(f"bundle_index: {index_path}")
    print(f"bundle_summary: {summary_path}")


if __name__ == "__main__":
    main()
