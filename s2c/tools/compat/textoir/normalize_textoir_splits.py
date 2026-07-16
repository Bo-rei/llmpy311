#!/usr/bin/env python3
"""将不可变的 TextOIR TSV 分割导出为 s2c 可审计的 canonical JSONL。

此脚本只做字段归一化与稳定 sample_id 编号，不重抽样、不重新选 known label、
不改变行顺序。同时保存源/目标 hash，便于确认导入前后的数据身份。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from ._common import (
        DATASETS,
        SPLITS,
        default_textoir_root,
        read_tsv,
        sha256_file,
        write_json,
    )
except ImportError:  # Direct script execution.
    from _common import (
        DATASETS,
        SPLITS,
        default_textoir_root,
        read_tsv,
        sha256_file,
        write_json,
    )


def normalize_dataset(textoir_root: Path, output_dir: Path, dataset: str) -> dict:
    """逐行保序转换一个数据集的 train/dev/test，并返回 hash manifest。"""

    dataset_manifest = {"dataset": dataset, "splits": {}}
    for split in SPLITS:
        source = textoir_root / "data" / dataset / f"{split}.tsv"
        rows = read_tsv(source)
        destination = output_dir / dataset / f"{split}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        label_counts = Counter()
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows):
                record = {
                    "sample_id": f"textoir:{dataset}:{split}:{index:06d}",
                    "dataset": dataset,
                    "split": split,
                    "text": row["text"],
                    "intent": row["label"],
                }
                encoded = (
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                handle.write(encoded.decode("utf-8"))
                digest.update(encoded)
                label_counts[row["label"]] += 1
        dataset_manifest["splits"][split] = {
            "source": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "output": str(destination.resolve()),
            "output_sha256": digest.hexdigest(),
            "samples": len(rows),
            "label_counts": dict(sorted(label_counts.items())),
        }
    return dataset_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textoir-root", type=Path, default=default_textoir_root())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    textoir_root = args.textoir_root.resolve()
    manifest = {
        "schema_version": 1,
        "source_repository": str(textoir_root),
        "datasets": [
            normalize_dataset(textoir_root, args.output_dir, dataset)
            for dataset in args.datasets
        ],
    }
    manifest_path = args.output_dir / "split_manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
