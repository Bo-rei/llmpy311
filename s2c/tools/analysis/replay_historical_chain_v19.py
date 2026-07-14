#!/usr/bin/env python3
"""Strict historical chain replay helper for CLINC150@KIR50."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis.historical_best_pipeline_v19 import HISTORICAL_BEST_PIPELINE


def build_strict_eval_command(output_dir: str) -> List[str]:
    """Build the strict historical eval command without generalized paths."""

    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools/eval/eval_system_pipeline_v19.py"),
        "--data_root",
        str(strict["data_root"]),
        "--data_root_scope",
        "all",
        "--gate_encoder_path",
        str(strict["gate_encoder_path"]),
        "--output_dir",
        str(output_dir),
    ]


def build_replay_manifest(output_root: Path, device: str = "cuda") -> Dict[str, object]:
    """Build the baseline strict replay manifest."""

    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    return {
        "mode": "strict_historical_replay",
        "data_root": strict["data_root"],
        "known_intents_path": strict["known_intents_path"],
        "gate_encoder_path": strict["gate_encoder_path"],
        "reference_eval_results": strict["reference_eval_results"],
        "frozen_eval_results": strict["frozen_eval_results"],
        "target_metrics": strict["target_metrics"],
        "stages": [
            {
                "name": "truth_freeze",
                "status": "pending",
            },
            {
                "name": "data_validation",
                "status": "pending",
            },
            {
                "name": "frozen_eval_replay",
                "status": "pending",
                "command": [
                    sys.executable,
                    str(
                        PROJECT_ROOT
                        / "tools/analysis/run_prototype_gate_frozen_baseline_v19.py"
                    ),
                    "--data_root",
                    str(strict["data_root"]),
                    "--device",
                    str(device),
                    "--output_dir",
                    str(output_root / "frozen_eval"),
                ],
            },
            {
                "name": "metric_comparison",
                "status": "pending",
            }
        ],
    }


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_historical_data(strict: Dict[str, object]) -> Dict[str, int]:
    data = _load_json(PROJECT_ROOT / strict["data_root"] / "gate" / "test.json")
    id_count = sum(
        1 for row in data if row.get("intent") != "oos" and row.get("label") != 1
    )
    oos_count = len(data) - id_count
    return {"total": len(data), "id_count": id_count, "oos_count": oos_count}


def compare_metrics(reference_metrics: Dict[str, float], candidate_metrics: Dict[str, float]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for key, value in reference_metrics.items():
        deltas[key] = float(candidate_metrics.get(key, 0.0)) - float(value)
    return deltas


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the strict historical chain")
    parser.add_argument(
        "--mode",
        default="report_only",
        choices=["report_only", "full"],
        help="Write replay manifest only or execute the strict frozen replay.",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/reports/historical_replay_20260414",
        help="Directory for replay manifests and future stage outputs.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Execution device for strict replay. User requested GPU execution.",
    )
    args = parser.parse_args()

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = build_replay_manifest(output_root, device=str(args.device))
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()

    manifest["stages"][0]["status"] = "completed"
    manifest["stages"][0]["truth_anchors"] = {
        "reference_eval_results": strict["reference_eval_results"],
        "frozen_eval_results": strict["frozen_eval_results"],
    }

    data_validation = validate_historical_data(strict)
    manifest["stages"][1]["status"] = "completed"
    manifest["stages"][1]["observed_counts"] = data_validation

    if args.mode == "full":
        frozen_cmd = manifest["stages"][2]["command"]
        _run(frozen_cmd)
        manifest["stages"][2]["status"] = "completed"

        replay_eval_path = output_root / "frozen_eval" / "eval_results.json"
        reference_metrics = _load_json(
            PROJECT_ROOT / strict["reference_eval_results"]
        )["metrics"]
        replay_metrics = _load_json(replay_eval_path)["metrics"]
        manifest["stages"][3]["status"] = "completed"
        manifest["stages"][3]["reference_metrics"] = reference_metrics
        manifest["stages"][3]["replay_metrics"] = replay_metrics
        manifest["stages"][3]["metric_deltas"] = compare_metrics(
            strict["target_metrics"], replay_metrics
        )
    else:
        manifest["stages"][2]["status"] = "skipped"
        manifest["stages"][3]["status"] = "skipped"

    manifest_path = output_root / "replay_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"replay_manifest: {manifest_path}")


if __name__ == "__main__":
    main()
