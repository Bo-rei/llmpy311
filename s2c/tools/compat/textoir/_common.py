"""TextOIR 兼容工具共用的轻量 I/O 与 provenance 函数。

此处刻意不 import TextOIR Python 包，以免旧版 torch/transformers 依赖污染 s2c
主环境。与上游代码的交互只通过文件、Git 命令和独立子进程完成。
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("banking", "oos", "stackoverflow")
SPLITS = ("train", "dev", "test")
EXPECTED_METHODS = ("MSP", "DOC", "ADB", "OpenMax", "KNNCL", "DA-ADB")
METHOD_CONTRACTS = {
    "MSP": {"backbone": "bert", "config": "MSP", "loss": "CrossEntropyLoss"},
    "DOC": {
        "backbone": "bert_doc",
        "config": "DOC",
        "loss": "Binary_CrossEntropyLoss",
    },
    "ADB": {
        "backbone": "bert",
        "config": "ADB",
        "loss": "CrossEntropyLoss",
        "pretrain": True,
        "save_model": True,
    },
    "OpenMax": {
        "backbone": "bert",
        "config": "OpenMax",
        "loss": "CrossEntropyLoss",
    },
    "KNNCL": {"backbone": "bert_knncl", "config": "KNNCL", "loss": "KNNCLoss"},
    "DA-ADB": {
        "backbone": "bert_disaware",
        "config": "DA-ADB",
        "loss": "CrossEntropyLoss",
        "pretrain": True,
    },
}


def default_textoir_root() -> Path:
    return Path(__file__).resolve().parents[4] / "textoir"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """先写临时文件再原子替换，避免中断时留下半个 JSON manifest。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["text", "label"]:
            raise ValueError(f"Expected TSV header 'text\\tlabel' in {path}, got {reader.fieldnames}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            text = row.get("text")
            label = row.get("label")
            if text is None or label is None:
                raise ValueError(f"Malformed row {line_number} in {path}")
            rows.append({"text": text, "label": label})
    return rows


def benchmark_labels(textoir_root: Path, dataset: str) -> list[str]:
    """用 AST 读取上游 benchmark label 字面量，不执行其模块顶层代码。"""

    source = textoir_root / "open_intent_detection" / "dataloaders" / "__init__.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "benchmark_labels"
            for target in node.targets
        ):
            labels = ast.literal_eval(node.value)
            if dataset not in labels:
                raise KeyError(f"Dataset {dataset!r} is absent from benchmark_labels")
            return list(labels[dataset])
    raise ValueError(f"Could not find literal benchmark_labels in {source}")


def ensure_datasets(values: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(values)
    invalid = sorted(set(selected) - set(DATASETS))
    if invalid:
        raise ValueError(f"Unsupported datasets: {', '.join(invalid)}")
    return selected
