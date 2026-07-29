"""
Batch Expert training — HiLSA-MoE v19
=======================================
Sequentially trains Stage-2 Expert models for all 10 domains using
train_expert_v19.py.  Each domain is trained in a subprocess so that GPU
memory is fully released between runs.

Usage:
    # Train all 10 domains with defaults
    python tools/train/train_all_experts_v19.py

    # Resume: skip domains whose outputs already exist
    python tools/train/train_all_experts_v19.py --skip_existing

    # Train a subset
    python tools/train/train_all_experts_v19.py --domains banking credit_cards

    # Override training hyper-parameters (forwarded to train_expert_v19.py)
    python tools/train/train_all_experts_v19.py --epochs 10 --batch_size 16

Exit codes:
    0 — all domains completed successfully
    1 — environment check failed
    2 — one or more domains failed (partial results may exist)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from legacy.runtime import WorkspacePaths  # noqa: E402

PATHS = WorkspacePaths.discover(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DOMAINS = [
    "auto_and_commute",
    "banking",
    "credit_cards",
    "home",
    "kitchen_and_dining",
    "meta",
    "small_talk",
    "travel",
    "utility",
    "work",
]

EXPERT_SCRIPT = (
    PROJECT_ROOT
    / "tools"
    / "train"
    / "train_expert_v19.py"
)
PYTHON = sys.executable


def main():
    # ------------------------------------------------------------------
    # Environment self-check (R3.1)
    # ------------------------------------------------------------------
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "bo":
        log.error(
            f"Expected conda env 'bo', got '{conda_env}'. "
            "Please activate: conda activate bo"
        )
        sys.exit(1)

    pythonpath = os.environ.get("PYTHONPATH", "")
    if str(PROJECT_ROOT) not in pythonpath:
        log.warning(
            f"PYTHONPATH does not include {PROJECT_ROOT}. "
            f"Run this module from {PROJECT_ROOT} with: python -m tools.train.train_all_experts_v19"
        )

    # Experts-lock check (R3.2)
    lock_file = PROJECT_ROOT / ".no_experts_run"
    if lock_file.exists():
        log.error("Experts training locked by policy (.no_experts_run exists). Exit 2.")
        sys.exit(2)

    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train all 10 domain Experts sequentially (v19)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DEFAULT_DOMAINS,
        help="Subset of domains to train (default: the historical CLINC domains)",
    )
    parser.add_argument("--model_path", default=str(PATHS.smollm135m))
    parser.add_argument("--data_dir", default=str(PATHS.prepared_data_root / "v19/experts"))
    parser.add_argument("--output_dir", default=str(PATHS.artifact_root / "outputs/experts_v19"))
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
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip domains whose output directory already exists",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)

    log.info(f"=== Batch Expert Training: {len(args.domains)} domains ===")
    log.info(f"Domains: {args.domains}")
    log.info(f"Output root: {output_root}")

    results: dict[str, str] = {}

    for i, domain in enumerate(args.domains, 1):
        domain_out = output_root / domain
        log.info(f"--- [{i}/{len(args.domains)}] Domain: {domain} ---")

        if args.skip_existing and domain_out.exists():
            log.info(f"  Skipping {domain} (output exists at {domain_out})")
            results[domain] = "SKIPPED"
            continue

        cmd = [
            PYTHON,
            str(EXPERT_SCRIPT),
            "--domain",
            domain,
            "--model_path",
            args.model_path,
            "--data_dir",
            args.data_dir,
            "--output_dir",
            args.output_dir,
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

        log.info(f"  CMD: {' '.join(str(c) for c in cmd)}")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        try:
            proc = subprocess.run(cmd, env=env, check=True)
            results[domain] = "OK"
            log.info(f"  ✓ {domain} completed successfully.")
        except subprocess.CalledProcessError as exc:
            results[domain] = f"FAILED (exit={exc.returncode})"
            log.error(
                f"  ✗ {domain} FAILED with exit code {exc.returncode}. "
                "Continuing with next domain."
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=== Batch training summary ===")
    any_failed = False
    for domain, status in results.items():
        log.info(f"  {domain:25s}: {status}")
        if status.startswith("FAILED"):
            any_failed = True

    # Aggregate results across completed domains
    import json

    aggregate = {}
    for domain in args.domains:
        res_file = output_root / domain / "results.json"
        if res_file.exists():
            with open(res_file) as f:
                aggregate[domain] = json.load(f)

    if aggregate:
        agg_path = output_root / "all_results.json"
        with open(agg_path, "w") as f:
            json.dump(aggregate, f, indent=2)
        log.info(f"Aggregate results saved → {agg_path}")

        # Print per-domain test accuracy summary
        log.info("Per-domain test accuracy (from results.json):")
        for domain, res in aggregate.items():
            acc = res.get("test_acc", "N/A")
            log.info(
                f"  {domain:25s}: test_acc={acc:.4f}"
                if isinstance(acc, float)
                else f"  {domain}: {acc}"
            )

    if any_failed:
        sys.exit(2)


if __name__ == "__main__":
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    main()
