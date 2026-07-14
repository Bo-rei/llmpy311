#!/usr/bin/env python3
"""Download and mirror the StackOverflow 20k source used by v19."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.active.rebuild_multi_dataset_v19 import (
    STACKOVERFLOW_RAW_FILES,
    STACKOVERFLOW_RAW_SOURCE_REPO,
    sync_stackoverflow_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download StackOverflow 20k source for v19")
    parser.add_argument("--output_root", default="stackoverflow_origin")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_base = "https://raw.githubusercontent.com/jacoxu/StackOverflow/master/rawText"
    for filename in STACKOVERFLOW_RAW_FILES.values():
        destination = output_root / filename
        if args.force or not destination.exists():
            urlretrieve(f"{raw_base}/{filename}", destination)

    manifest = sync_stackoverflow_source(output_root, force=bool(args.force))
    logging.getLogger(__name__).info(
        "Prepared StackOverflow source mirror from %s -> %s (%d retained rows)",
        STACKOVERFLOW_RAW_SOURCE_REPO,
        manifest["records_path"],
        int(manifest["retained_record_count"]),
    )


if __name__ == "__main__":
    main()
