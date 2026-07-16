#!/usr/bin/env python3
"""在不 import TextOIR 的情况下审计上游代码与数据协议。

审计产物记录 Git remote/commit/cleanliness、方法注册状态、每个 TSV 的 SHA256
与标签计数。它是 TextOIR 结果的 provenance 入口，不负责训练或改动上游工作树。
"""

from __future__ import annotations

import ast
import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from ._common import (
        DATASETS,
        EXPECTED_METHODS,
        METHOD_CONTRACTS,
        SPLITS,
        benchmark_labels,
        default_textoir_root,
        git_output,
        read_tsv,
        sha256_file,
        write_json,
    )
except ImportError:  # Direct script execution.
    from _common import (
        DATASETS,
        EXPECTED_METHODS,
        METHOD_CONTRACTS,
        SPLITS,
        benchmark_labels,
        default_textoir_root,
        git_output,
        read_tsv,
        sha256_file,
        write_json,
    )


def build_audit(textoir_root: Path, datasets: tuple[str, ...]) -> dict:
    """构建一份可机读的 TextOIR 代码/数据快照。"""

    root = textoir_root.resolve()
    if not (root / ".git").exists():
        raise FileNotFoundError(f"TEXTOIR Git repository not found: {root}")

    method_root = root / "open_intent_detection" / "methods"
    config_root = root / "open_intent_detection" / "configs"
    method_map_source = method_root / "__init__.py"
    # 用 AST 检查 method_map，而不 import methods。后者会立即加载上游
    # optional Llama/peft 依赖，使“协议审计”不必要地变成“环境兼容测试”。
    method_map_tree = ast.parse(
        method_map_source.read_text(encoding="utf-8"), filename=str(method_map_source)
    )
    registered_methods: set[str] = set()
    for node in method_map_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "method_map" for target in node.targets
        ) and isinstance(node.value, ast.Dict):
            registered_methods = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            break
    methods = {}
    for method in EXPECTED_METHODS:
        contract = METHOD_CONTRACTS[method]
        example_path = root / "open_intent_detection" / "examples" / f"run_{method}.sh"
        example_text = example_path.read_text(encoding="utf-8") if example_path.is_file() else ""
        expected_cli = {
            "method": f"--method '{method}'",
            "backbone": f"--backbone '{contract['backbone']}'",
            "config": f"--config_file_name '{contract['config']}'",
            "loss": f"--loss_fct '{contract['loss']}'",
            "pretrain": "--pretrain" if contract.get("pretrain") else None,
            "save_model": "--save_model" if contract.get("save_model") else None,
        }
        checks = {
            name: token in example_text if token is not None else True
            for name, token in expected_cli.items()
        }
        methods[method] = {
            "registered": method in registered_methods,
            "config_file": (config_root / f"{method}.py").is_file(),
            "example_file": example_path.is_file(),
            "example_sha256": sha256_file(example_path) if example_path.is_file() else None,
            "runner_contract": contract,
            "example_contract_checks": checks,
            "contract_matches_example": all(checks.values()),
        }

    dataset_audit = {}
    for dataset in datasets:
        labels = benchmark_labels(root, dataset)
        splits = {}
        for split in SPLITS:
            path = root / "data" / dataset / f"{split}.tsv"
            rows = read_tsv(path)
            counts = Counter(row["label"] for row in rows)
            splits[split] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "samples": len(rows),
                "labels": len(counts),
                "label_counts": dict(sorted(counts.items())),
            }
        dataset_audit[dataset] = {
            "benchmark_label_count": len(labels),
            "benchmark_labels": labels,
            "splits": splits,
        }

    status = git_output(root, "status", "--porcelain")
    return {
        "schema_version": 1,
        "repository": {
            "path": str(root),
            "origin": git_output(root, "remote", "get-url", "origin"),
            "head": git_output(root, "rev-parse", "HEAD"),
            "branch": git_output(root, "branch", "--show-current"),
            "clean": not bool(status),
            "status_porcelain": status.splitlines(),
        },
        "methods": methods,
        "methods_complete": all(
            row["registered"]
            and row["config_file"]
            and row["example_file"]
            and row["contract_matches_example"]
            for row in methods.values()
        ),
        "datasets": dataset_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textoir-root", type=Path, default=default_textoir_root())
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--require-clean", action="store_true", help="Fail if the TEXTOIR worktree is dirty"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(args.textoir_root, tuple(args.datasets))
    if args.output:
        write_json(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if not audit["methods_complete"]:
        return 2
    if args.require_clean and not audit["repository"]["clean"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
