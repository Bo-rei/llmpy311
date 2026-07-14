#!/usr/bin/env python3
"""Train SNIPS expert as a single-domain expert and emit all_results.json.

The legacy batch expert trainer validates domains against CLINC domain names,
which breaks for SNIPS. This script runs single-domain expert training using
train_expert_v19.py with domain=snips and writes an aggregate result file
compatible with the orchestrator expectation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERT_SCRIPT = (
    PROJECT_ROOT
    / "tools"
    / "train"
    / "train_expert_v19.py"
)


def run_command(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    # Legacy trainer checks this value and exits early if it is not 'bo'.
    env["CONDA_DEFAULT_ENV"] = "bo"
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SNIPS single-domain expert")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    expert_output = output_root / "snips"
    aggregate_file = output_root / "all_results.json"

    if args.skip_existing and aggregate_file.exists() and aggregate_file.stat().st_size > 0:
        return

    cmd = [
        sys.executable,
        str(EXPERT_SCRIPT),
        "--domain",
        "snips",
        "--model_path",
        str(args.model_path),
        "--data_dir",
        str(args.data_dir),
        "--output_dir",
        str(args.output_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--weight_decay",
        str(args.weight_decay),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--max_length",
        str(args.max_length),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--patience",
        str(args.patience),
    ]
    run_command(cmd)

    result_file = expert_output / "results.json"
    if not result_file.exists():
        raise FileNotFoundError(f"Missing expected SNIPS expert result: {result_file}")

    payload = {"snips": json.load(open(result_file, "r", encoding="utf-8"))}
    with open(aggregate_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
