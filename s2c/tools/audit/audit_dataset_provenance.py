"""三方数据来源审计工具。

这个脚本只读取官方快照、TEXTOIR 和 s2c 数据，不训练模型、不生成 embedding，
也不改写任何原始数据。输出使用文本哈希和短预览，避免把完整数据集复制到 Git。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    """只做可复现的 Unicode/空白规范化，不做语义改写。"""
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value.casefold()).strip()


def text_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def preview(value: str, limit: int = 180) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def record(
    dataset: str,
    source: str,
    split: str,
    index: int,
    text: str,
    label: str,
    *,
    compare_label: str | None = None,
    task_label: str | None = None,
) -> dict[str, object]:
    """统一记录格式；compare_label 用于追踪原始意图，task_label 保留任务标签。"""
    return {
        "dataset": dataset,
        "source": source,
        "split": split,
        "index": index,
        "text": text,
        "label": label,
        "compare_label": compare_label if compare_label is not None else label,
        "task_label": task_label if task_label is not None else label,
    }


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_clinc_json(root: Path, source: str) -> list[dict[str, object]]:
    """读取 CLINC 的原始 JSON；``source`` 不能靠路径猜测。"""
    payload = read_json(root / "data" / "data_full.json")
    assert isinstance(payload, dict)
    rows: list[dict[str, object]] = []
    for split, values in payload.items():
        for index, value in enumerate(values):
            rows.append(record("clinc150", source, split, index, value[0], value[1]))
    return rows


def load_clinc_textoir(root: Path, flavor: str = "oos") -> list[dict[str, object]]:
    """TEXTOIR 的 oos 快照保留官方内容；clinc 快照是另一套协议，单独标记。"""
    rows: list[dict[str, object]] = []
    directory = root / "data" / flavor
    for file_split, canonical in (("train", "train"), ("dev", "val"), ("test", "test")):
        for index, value in enumerate(read_tsv(directory / f"{file_split}.tsv")):
            label = value["label"]
            split = "oos_all" if label == "oos" else canonical
            rows.append(record("clinc150", f"textoir_{flavor}", split, index, value["text"], label))
    return rows


def load_banking_file(root: Path, split: str) -> list[dict[str, object]]:
    texts = read_lines(root / split / "seq.in")
    labels = read_lines(root / split / "label")
    if len(texts) != len(labels):
        raise ValueError(f"Banking split length mismatch: {root}/{split}")
    return [record("banking77", "s2c_banking_standard", split, i, text, label)
            for i, (text, label) in enumerate(zip(texts, labels))]


def load_banking_official(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train", "test"):
        with (root / "banking_data" / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            for index, value in enumerate(csv.DictReader(handle)):
                rows.append(record("banking77", "official_banking", split, index, value["text"], value["category"]))
    return rows


def load_banking_textoir(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for file_split, canonical in (("train", "train"), ("dev", "dev"), ("test", "test")):
        for index, value in enumerate(read_tsv(root / "data" / "banking" / f"{file_split}.tsv")):
            rows.append(record("banking77", "textoir_banking", canonical, index, value["text"], value["label"]))
    return rows


def load_banking_oos(root: Path) -> list[dict[str, object]]:
    """保留 id-oos 的 label_original；ood-oos 没有可追踪的原始意图。"""
    rows: list[dict[str, object]] = []
    base = root / "banking77_oos"
    for split in ("train", "valid", "test"):
        texts = read_lines(base / split / "seq.in")
        labels = read_lines(base / split / "label")
        for i, (text, label) in enumerate(zip(texts, labels)):
            rows.append(record("banking77_oos", "s2c_banking_oos_raw", split, i, text, label))
    for source in ("id-oos", "ood-oos"):
        for split in ("train", "valid", "test"):
            text_path = base / source / split / "seq.in"
            if not text_path.exists():
                continue
            texts = read_lines(text_path)
            labels = read_lines(base / source / split / "label")
            original_path = base / source / split / "label_original"
            original = read_lines(original_path) if original_path.exists() else labels
            for i, (text, task_label, original_label) in enumerate(zip(texts, labels, original)):
                rows.append(record("banking77_oos", "s2c_banking_oos_raw", f"{source}_{split}", i,
                                   text, original_label, compare_label=original_label,
                                   task_label=task_label))
    return rows


STACK_LABELS = [
    "wordpress", "oracle", "svn", "apache", "excel", "matlab", "visual-studio",
    "cocoa", "osx", "bash", "spring", "hibernate", "scala", "sharepoint", "ajax",
    "qt", "drupal", "linq", "haskell", "magento",
]

# `jacoxu/StackOverflow` fixes the 20,000-title content reference, but its
# README is not a data licence.  The digest records exactly the upstream text
# inspected during adjudication without copying that README into the audit.
STACK_UPSTREAM_README_SHA256 = "3d51882c2a3ec462309fb04afa0f3039858881303ffa577d3cf300e1e8d72b4d"
STACK_UPSTREAM_LICENSE_FILE_STATUS = "not_present_http_404"


def load_stack_official(root: Path) -> list[dict[str, object]]:
    titles = read_lines(root / "title_StackOverflow.txt")
    labels = read_lines(root / "label_StackOverflow.txt")
    return [record("stackoverflow", "official_stackoverflow", "raw", i, text, STACK_LABELS[int(label) - 1])
            for i, (text, label) in enumerate(zip(titles, labels))]


def load_stack_textoir(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for file_split in ("train", "dev", "test"):
        for index, value in enumerate(read_tsv(root / "data" / "stackoverflow" / f"{file_split}.tsv")):
            rows.append(record("stackoverflow", "textoir_stackoverflow", file_split, index, value["text"], value["label"]))
    return rows


def load_stack_s2c(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (root / "records.jsonl").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            value = json.loads(line)
            rows.append(record("stackoverflow", "s2c_stackoverflow_raw", value.get("source_split", "raw"),
                               index, value["title"], value["intent"]))
    return rows


def load_prepared_directory(base: Path, dataset: str) -> list[dict[str, object]]:
    """读取一个历史 Gate 输入快照。

    历史 prepared 数据是派生实验输入，不是 canonical raw source。这里保留它的
    ``source_split``，用于证明历史实验究竟使用了什么，而不是把 KIR 子集误当作
    原始数据缺失。
    """
    rows: list[dict[str, object]] = []
    for file_split in ("train", "val", "test"):
        path = base / f"{file_split}.json"
        if not path.exists():
            continue
        for index, value in enumerate(read_json(path)):
            intent = str(value.get("intent", ""))
            is_oos = int(value.get("label", 0)) == 1
            compare_label = intent if intent and intent != "oos" else "oos"
            rows.append(record(dataset, f"s2c_prepared_{dataset}_{base.parent.name}",
                               value.get("source_split", file_split), index, value["text"],
                               compare_label, compare_label=compare_label,
                               task_label="oos" if is_oos else "known"))
    return rows


def load_protocol_version(root: Path, dataset: str, version: str) -> list[dict[str, object]]:
    """读取 versioned canonical；它是审计对象，不自动获得官方地位。"""
    path = root / "canonical" / version / dataset / "records.jsonl"
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            value = json.loads(line)
            rows.append(record(
                dataset,
                f"s2c_{version}",
                str(value["original_split"]),
                index,
                str(value["text"]),
                str(value["intent"]),
            ))
    return rows


def load_protocol_candidate(root: Path, dataset: str) -> list[dict[str, object]]:
    """兼容旧调用：读取被阻断的 TEXTOIR candidate。"""
    return load_protocol_version(root, dataset, "protocol_v2")


def prepared_directories(root: Path, dataset: str) -> list[Path]:
    """返回真实存在的历史输入目录；不假定 KIR 或 seed 的固定集合。"""
    slug = {"clinc150": "clinc150", "banking77": "banking77", "banking77_oos": "banking77_oos", "stackoverflow": "stackoverflow"}[dataset]
    dataset_root = root / slug
    return sorted(path / "gate" for path in dataset_root.glob("kir*_seed*") if (path / "gate").is_dir())


def json_values(value: object) -> Iterable[str]:
    """递归抽取 manifest 中的字符串，仅用于统计历史 data_root 引用。"""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from json_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from json_values(child)


def historical_reference_counts(artifacts_root: Path, prepared_root: Path) -> Counter[str]:
    """统计已有结果 manifest 对每个 prepared 输入目录的引用次数。

    只读取 JSON manifest；绝不扫描 checkpoint、embedding 或逐样本分数。
    """
    counts: Counter[str] = Counter()
    if not artifacts_root.exists():
        return counts
    for path in artifacts_root.rglob("*.json"):
        # ``predictions.json`` 等逐样本产物可能非常大，且不属于输入 provenance。
        # 只读取命名明确的 manifest，并对异常大文件保守跳过。
        if "manifest" not in path.name.lower() or path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            payload = read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for candidate in json_values(payload):
            try:
                resolved = Path(candidate).resolve()
            except OSError:
                continue
            if prepared_root in resolved.parents:
                counts[str(resolved)] += 1
    return counts


def exact_key(row: dict[str, object]) -> tuple[str, str]:
    """原始文本与标签的记录键：不得隐式清理任何字符。"""
    return str(row["text"]), str(row["compare_label"])


def normalized_label_key(row: dict[str, object]) -> tuple[str, str]:
    """仅供诊断格式差异的规范化键，不能替代精确记录键。"""
    return normalize_text(str(row["text"])), str(row["compare_label"])


def text_key(row: dict[str, object]) -> str:
    return normalize_text(str(row["text"]))


def scope_rows(rows: list[dict[str, object]], scope: str | None) -> list[dict[str, object]]:
    return rows if scope in (None, "all") else [row for row in rows if row["split"] == scope]


def intent_count_rate(left: list[dict[str, object]], right: list[dict[str, object]]) -> float | None:
    if not left:
        return None
    lc, rc = Counter(str(x["compare_label"]) for x in left), Counter(str(x["compare_label"]) for x in right)
    if not lc:
        return None
    return sum(min(count, rc.get(label, 0)) / max(count, rc.get(label, 0)) for label, count in lc.items()) / len(lc)


def compare(left: list[dict[str, object]], right: list[dict[str, object]],
            dataset: str, comparison: str, scope: str = "all") -> dict[str, object]:
    lrows, rrows = scope_rows(left, scope), scope_rows(right, scope)
    left_exact, right_exact = Counter(exact_key(x) for x in lrows), Counter(exact_key(x) for x in rrows)
    left_normalized_labels = Counter(normalized_label_key(x) for x in lrows)
    right_normalized_labels = Counter(normalized_label_key(x) for x in rrows)
    left_texts, right_texts = Counter(text_key(x) for x in lrows), Counter(text_key(x) for x in rrows)
    exact = sum((left_exact & right_exact).values())
    normalized_label = sum((left_normalized_labels & right_normalized_labels).values())
    norm = sum((left_texts & right_texts).values())
    # 预索引 split，避免对每个共同记录再次全表扫描。
    left_splits: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    right_splits: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in lrows:
        left_splits[exact_key(row)][str(row["split"])] += 1
    for row in rrows:
        right_splits[exact_key(row)][str(row["split"])] += 1
    split_same = sum(
        sum((left_splits[k] & right_splits[k]).values())
        for k in left_exact.keys() & right_exact.keys()
    )
    intents_l = {str(x["compare_label"]) for x in lrows}
    intents_r = {str(x["compare_label"]) for x in rrows}
    return {
        "dataset": dataset,
        "comparison": comparison,
        "scope": scope,
        "left_count": len(lrows),
        "right_count": len(rrows),
        "exact_record_match_count": exact,
        "exact_record_match_rate": exact / len(lrows) if lrows else None,
        "normalized_text_and_label_match_count": normalized_label,
        "normalized_text_and_label_match_rate": normalized_label / len(lrows) if lrows else None,
        "normalized_text_match_count": norm,
        "normalized_text_match_rate": norm / len(lrows) if lrows else None,
        "label_match_rate": normalized_label / norm if norm else None,
        "split_match_rate": split_same / exact if exact else None,
        "intent_set_match_rate": len(intents_l & intents_r) / len(intents_l) if intents_l else None,
        "per_class_count_match_rate": intent_count_rate(lrows, rrows),
    }


def representative(rows: Iterable[dict[str, object]], wanted: tuple[str, str]) -> dict[str, object] | None:
    for row in rows:
        if exact_key(row) == wanted:
            return row
    return None


def sample_row(row: dict[str, object], side: str, other_source: str) -> dict[str, object]:
    return {
        "dataset": row["dataset"],
        "side": side,
        "source": row["source"],
        "other_source": other_source,
        "split": row["split"],
        "index": row["index"],
        "compare_label": row["compare_label"],
        "task_label": row["task_label"],
        "text_sha256": text_hash(str(row["text"])),
        "text_preview": preview(str(row["text"])),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def difference_rows(left: list[dict[str, object]], right: list[dict[str, object]],
                    dataset: str, comparison: str, missing: bool) -> list[dict[str, object]]:
    lk, rk = Counter(exact_key(x) for x in left), Counter(exact_key(x) for x in right)
    source_rows = left if missing else right
    other = right if missing else left
    other_source = str(other[0]["source"]) if other else ""
    source_by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in source_rows:
        source_by_key[exact_key(row)].append(row)
    rows: list[dict[str, object]] = []
    for wanted, count in ((lk - rk) if missing else (rk - lk)).items():
        for row in source_by_key[wanted][:count]:
            value = sample_row(row, "left" if missing else "right", other_source)
            value.update({"dataset": dataset, "comparison": comparison})
            rows.append(value)
    return rows


def conflict_rows(left: list[dict[str, object]], right: list[dict[str, object]],
                  dataset: str, comparison: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lm, rm = defaultdict(list), defaultdict(list)
    for row in left:
        lm[text_key(row)].append(row)
    for row in right:
        rm[text_key(row)].append(row)
    labels, splits = [], []
    for text in sorted(set(lm) & set(rm)):
        ll, rr = {str(x["compare_label"]) for x in lm[text]}, {str(x["compare_label"]) for x in rm[text]}
        if ll != rr:
            labels.append({
                "dataset": dataset, "comparison": comparison, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text_preview": preview(next(iter(lm[text]))["text"]),
                "left_labels": "|".join(sorted(ll)), "right_labels": "|".join(sorted(rr)),
                "left_splits": "|".join(sorted({str(x["split"]) for x in lm[text]})),
                "right_splits": "|".join(sorted({str(x["split"]) for x in rm[text]})),
            })
        for label in sorted(ll & rr):
            ls, rs = Counter(str(x["split"]) for x in lm[text] if str(x["compare_label"]) == label), Counter(str(x["split"]) for x in rm[text] if str(x["compare_label"]) == label)
            if ls != rs:
                splits.append({
                    "dataset": dataset, "comparison": comparison,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "text_preview": preview(next(x for x in lm[text] if str(x["compare_label"]) == label)["text"]),
                    "compare_label": label,
                    "left_splits": "|".join(sorted(ls)), "right_splits": "|".join(sorted(rs)),
                })
    return labels, splits


def normalized_only_match_rows(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
    dataset: str,
    comparison: str,
) -> list[dict[str, object]]:
    """列出文本仅在 Unicode/空白规范化后才匹配的记录。

    这既证明来源存在格式差异，也避免把这类差异伪装成精确一致。每个规范化
    文本-标签组合至多记录一对代表样本，完整差异仍可由 missing/extra CSV 追溯。
    """
    left_by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    right_by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in left:
        left_by_key[normalized_label_key(row)].append(row)
    for row in right:
        right_by_key[normalized_label_key(row)].append(row)
    rows: list[dict[str, object]] = []
    for wanted in sorted(left_by_key.keys() & right_by_key.keys()):
        left_raw = {exact_key(row) for row in left_by_key[wanted]}
        right_raw = {exact_key(row) for row in right_by_key[wanted]}
        if left_raw & right_raw:
            continue
        left_row, right_row = left_by_key[wanted][0], right_by_key[wanted][0]
        rows.append({
            "dataset": dataset,
            "comparison": comparison,
            "label": wanted[1],
            "normalized_text_sha256": hashlib.sha256(wanted[0].encode("utf-8")).hexdigest(),
            "left_text_sha256": hashlib.sha256(str(left_row["text"]).encode("utf-8")).hexdigest(),
            "right_text_sha256": hashlib.sha256(str(right_row["text"]).encode("utf-8")).hexdigest(),
            "left_text_preview": preview(str(left_row["text"])),
            "right_text_preview": preview(str(right_row["text"])),
            "left_split": left_row["split"],
            "right_split": right_row["split"],
        })
    return rows


def source_license_rows(root: Path, textoir: Path, official_banking: Path, official_stack: Path) -> list[dict[str, object]]:
    return [
        # 输出中只保存逻辑路径，避免把本机绝对路径写入可提交的审计结果。
        {"dataset": "clinc150", "source": "official_clinc", "url": "https://github.com/clinc/oos-eval", "revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de", "file": "official_snapshot/clinc150/LICENSE", "sha256": sha256(root / "clinc150" / "LICENSE"), "license": "CC BY 3.0", "status": "verified", "notes": "官方仓库 LICENSE 明确写明 Attribution 3.0 Unported。"},
        {"dataset": "clinc150", "source": "textoir_repository", "url": "https://github.com/thuiar/TEXTOIR.git", "revision": "dffe2b1b848a069a6808f8089b4cb9bd16e2062b", "file": "textoir/LICENSE", "sha256": sha256(textoir / "LICENSE"), "license": "MIT (repository code; data-specific license not stated)", "status": "partial", "notes": "TEXTOIR 仓库代码有 MIT，但 data/ 未提供独立数据许可证。"},
        {"dataset": "clinc150", "source": "s2c_clinc_raw", "url": "", "revision": "", "file": "assets/datasets/s2c/source/clinc150/data/data_full.json", "sha256": sha256(root / "clinc150" / "data" / "data_full.json"), "license": "inherits official CC BY 3.0", "status": "verified_inherited", "notes": "s2c raw 文件与固定官方 data_full.json 完全一致；不代表历史 prepared split 已通过审计。"},
        {"dataset": "banking77", "source": "official_banking", "url": "https://github.com/PolyAI-LDN/task-specific-datasets", "revision": "57ec275d8078af65b7731c2a98be812d844a6d6b", "file": "official_snapshot/task-specific-datasets/LICENSE", "sha256": sha256(official_banking / "LICENSE"), "license": "CC BY 4.0", "status": "verified", "notes": "PolyAI 官方仓库 LICENSE 为 Creative Commons Attribution 4.0。"},
        {"dataset": "banking77", "source": "textoir_banking_data", "url": "https://github.com/thuiar/TEXTOIR.git", "revision": "dffe2b1b848a069a6808f8089b4cb9bd16e2062b", "file": "textoir/data/banking/", "sha256": "", "license": "MIT (repository code; data-specific license not stated)", "status": "partial", "notes": "内容与官方 Banking77 一致，但 TEXTOIR 未提供独立 data/ 许可证。"},
        {"dataset": "banking77", "source": "s2c_banking_standard", "url": "", "revision": "", "file": "assets/datasets/s2c/source/banking77/", "sha256": "", "license": "inherits official CC BY 4.0", "status": "verified_inherited", "notes": "s2c standard 文件与官方 train/test 内容一致；valid 是本地派生切分。"},
        {"dataset": "banking77_oos", "source": "s2c_banking_oos_raw", "url": "", "revision": "", "file": "assets/datasets/s2c/source/banking77_oos/", "sha256": "", "license": "unknown", "status": "blocked", "notes": "id-oos/ood-oos 没有可核验的上游 commit、许可证或生成说明。"},
        {"dataset": "banking77_oos", "source": "textoir_banking_data", "url": "https://github.com/thuiar/TEXTOIR.git", "revision": "dffe2b1b848a069a6808f8089b4cb9bd16e2062b", "file": "textoir/data/banking/", "sha256": "", "license": "standard Banking77 only; OOS extension absent", "status": "not_applicable", "notes": "TEXTOIR 提供标准 Banking77，不提供当前 s2c id-oos/ood-oos 扩展。"},
        {"dataset": "stackoverflow", "source": "official_stackoverflow", "url": "https://github.com/jacoxu/StackOverflow", "revision": "7c207f51e649fff9e4736610b9d44431bb7ccf00", "file": "official_snapshot/stackoverflow_raw/title_StackOverflow.txt ; official_snapshot/stackoverflow_raw/label_StackOverflow.txt", "sha256": sha256(official_stack / "title_StackOverflow.txt") + ";" + sha256(official_stack / "label_StackOverflow.txt"), "license": "not stated", "status": "blocked", "notes": f"README SHA256={STACK_UPSTREAM_README_SHA256}；仅要求致谢 Kaggle，未标识具体数据集或 data dump；LICENSE endpoint HTTP 404；语料没有 post ID/date，不能将通用 CC BY-SA 政策映射到每条记录。"},
        {"dataset": "stackoverflow", "source": "textoir_repository", "url": "https://github.com/thuiar/TEXTOIR.git", "revision": "dffe2b1b848a069a6808f8089b4cb9bd16e2062b", "file": "textoir/data/stackoverflow/", "sha256": "", "license": "unknown", "status": "blocked", "notes": "TEXTOIR 数据文件随仓库提供，但没有独立数据许可证。"},
        {"dataset": "stackoverflow", "source": "s2c_stackoverflow_raw", "url": "https://github.com/jacoxu/StackOverflow", "revision": "", "file": "assets/datasets/s2c/source/stackoverflow/records.jsonl", "sha256": "", "license": "not stated", "status": "blocked", "notes": "s2c raw 内容可与公开原始文件匹配，但许可证不明确，因此不能进入 canonical protocol。"},
    ]


def write_historical_audit(
    output: Path,
    artifacts_root: Path,
    prepared_root: Path,
    sources: dict[str, dict[str, list[dict[str, object]]]],
) -> None:
    """Recompute legacy-input evidence without treating it as a raw source.

    This reference scan is intentionally separate from the three-way raw-data
    comparison.  It may traverse a large immutable artifact tree, so a later
    source-only refresh can reuse the prior inventory instead of repeating it.
    """

    historical_rows: list[dict[str, object]] = []
    historical_metrics: list[dict[str, object]] = []
    manifest_references = historical_reference_counts(artifacts_root, prepared_root)
    for dataset in ("clinc150", "banking77_oos", "stackoverflow"):
        for gate_directory in prepared_directories(prepared_root, dataset):
            base = gate_directory.parent
            variant = base.name
            manifest_path = base / "MANIFEST.json"
            rows = load_prepared_directory(gate_directory, dataset)
            manifest_hash = sha256(manifest_path) if manifest_path.exists() else ""
            historical_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "prepared_relative_path": str(base.relative_to(prepared_root)),
                    "gate_record_count": len(rows),
                    "manifest_sha256": manifest_hash,
                    "artifact_manifest_reference_count": manifest_references.get(str(base.resolve()), 0),
                    "source_status": "legacy_derived" if dataset != "banking77_oos" else "legacy_blocked_unverified",
                }
            )
            if dataset in sources:
                for comparison, left_name in (
                    ("official_vs_historical_prepared", "official"),
                    ("textoir_vs_historical_prepared", "textoir"),
                ):
                    metric = compare(sources[dataset][left_name], rows, dataset, comparison)
                    metric.update({"variant": variant, "prepared_relative_path": str(base.relative_to(prepared_root))})
                    historical_metrics.append(metric)
            else:
                historical_metrics.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "prepared_relative_path": str(base.relative_to(prepared_root)),
                        "comparison": "not_comparable_to_standard_banking77",
                        "status": "blocked_unverified",
                        "reason": "BANKING77-OOS is a legacy extension; official and TEXTOIR only provide standard Banking77.",
                    }
                )
    write_csv(output / "historical_input_inventory.csv", historical_rows)
    write_csv(output / "historical_comparison_metrics.csv", historical_metrics)


def require_reusable_historical_audit(output: Path) -> None:
    """Fail closed when a source-only refresh lacks prior legacy evidence."""

    required = [output / "historical_input_inventory.csv", output / "historical_comparison_metrics.csv"]
    missing_historical = [str(path) for path in required if not path.is_file()]
    if missing_historical:
        raise FileNotFoundError(
            "Cannot reuse historical audit because required evidence is missing: " + ", ".join(missing_historical)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "data_audit")
    parser.add_argument("--official-clinc-root", type=Path, default=Path("../assets/datasets/s2c/source/clinc150"))
    parser.add_argument("--official-banking-root", type=Path, default=Path("/tmp/provenance_sources/task-specific-datasets"))
    parser.add_argument("--official-stack-root", type=Path, default=Path("/tmp/provenance_sources/stackoverflow_raw"))
    parser.add_argument("--textoir-root", type=Path, default=Path("../textoir"))
    parser.add_argument("--s2c-source-root", type=Path, default=Path("../assets/datasets/s2c/source"))
    parser.add_argument("--s2c-prepared-root", type=Path, default=Path("../assets/datasets/s2c/prepared/data/multidataset/v19"))
    parser.add_argument("--s2c-data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("../artifacts/s2c/outputs/experiments"),
        help="仅扫描历史 JSON manifest，统计实际 prepared 输入引用。",
    )
    parser.add_argument(
        "--reuse-historical-audit",
        action="store_true",
        help="复用已有历史 prepared 输入审计，避免为 source/license refresh 递归扫描大型 artifact 树。",
    )
    args = parser.parse_args()
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    official_clinc, official_banking, official_stack = map(resolve, (args.official_clinc_root, args.official_banking_root, args.official_stack_root))
    textoir, s2c_source, prepared, s2c_data, artifacts, output = map(
        resolve,
        (
            args.textoir_root,
            args.s2c_source_root,
            args.s2c_prepared_root,
            args.s2c_data_root,
            args.artifacts_root,
            args.output,
        ),
    )
    output.mkdir(parents=True, exist_ok=True)

    sources: dict[str, dict[str, list[dict[str, object]]]] = {
        "clinc150": {
            "official": load_clinc_json(official_clinc, "official_clinc"),
            "textoir": load_clinc_textoir(textoir, "oos"),
            "s2c_raw": load_clinc_json(s2c_source / "clinc150", "s2c_clinc_raw"),
            "s2c_candidate": load_protocol_candidate(s2c_data, "clinc150"),
        },
        "banking77": {
            "official": load_banking_official(official_banking),
            "textoir": load_banking_textoir(textoir),
            "s2c_raw": [
                x
                for split in ("train", "valid", "test")
                for x in load_banking_file(s2c_source / "banking77", split)
            ],
            "s2c_candidate": load_protocol_candidate(s2c_data, "banking77"),
        },
        "stackoverflow": {
            "official": load_stack_official(official_stack),
            "textoir": load_stack_textoir(textoir),
            "s2c_raw": load_stack_s2c(s2c_source / "stackoverflow"),
            "s2c_candidate": load_protocol_candidate(s2c_data, "stackoverflow"),
        },
    }
    # Reconstructed canonical data remains a fourth audited object.  It is
    # compared back to official raw data before admission, rather than being
    # silently trusted because its directory name says “official”.
    official_version = "protocol_v2_official_v1"
    for dataset in ("clinc150", "banking77"):
        canonical = s2c_data / "canonical" / official_version / dataset / "records.jsonl"
        if canonical.is_file():
            sources[dataset]["s2c_official_reconstruction"] = load_protocol_version(
                s2c_data, dataset, official_version
            )

    summary: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    extra: list[dict[str, object]] = []
    conflicts: list[dict[str, object]] = []
    split_conflicts: list[dict[str, object]] = []
    normalized_only: list[dict[str, object]] = []
    # 对历史 prepared 快照也做同一套比较。它们不是 raw source，
    # 但必须进入审计结果，避免把“原始内容一致”和“历史实验切分可复现”混为一谈。
    pair_specs = {
        "official_vs_textoir": ("official", "textoir"),
        "official_vs_s2c_raw": ("official", "s2c_raw"),
        "textoir_vs_s2c_raw": ("textoir", "s2c_raw"),
        "official_vs_s2c_candidate": ("official", "s2c_candidate"),
        "textoir_vs_s2c_candidate": ("textoir", "s2c_candidate"),
    }
    for dataset, variants in sources.items():
        active_pairs = dict(pair_specs)
        if "s2c_official_reconstruction" in variants:
            active_pairs["official_vs_s2c_official_reconstruction"] = (
                "official",
                "s2c_official_reconstruction",
            )
        for comparison, (left_name, right_name) in active_pairs.items():
            if left_name not in variants or right_name not in variants:
                summary.append({"dataset": dataset, "comparison": comparison, "scope": "unavailable", "status": "missing_source"})
                continue
            left, right = variants[left_name], variants[right_name]
            scopes = ["all"] + sorted({str(x["split"]) for x in left} & {str(x["split"]) for x in right})
            for scope in scopes:
                row = compare(left, right, dataset, comparison, scope)
                row["status"] = "complete"
                summary.append(row)
            missing.extend(difference_rows(left, right, dataset, comparison, True))
            extra.extend(difference_rows(left, right, dataset, comparison, False))
            label_rows, split_rows = conflict_rows(left, right, dataset, comparison)
            conflicts.extend(label_rows)
            split_conflicts.extend(split_rows)
            normalized_only.extend(normalized_only_match_rows(left, right, dataset, comparison))

    write_csv(output / "official_vs_textoir.csv", [x for x in summary if x["comparison"] == "official_vs_textoir"])
    write_csv(output / "official_vs_s2c.csv", [x for x in summary if str(x["comparison"]).startswith("official_vs_s2c")])
    write_csv(output / "textoir_vs_s2c.csv", [x for x in summary if str(x["comparison"]).startswith("textoir_vs_s2c")])
    write_csv(output / "comparison_metrics.csv", summary)
    write_csv(output / "normalized_text_matches.csv", normalized_only)
    write_csv(output / "missing_samples.csv", missing)
    write_csv(output / "extra_samples.csv", extra)
    write_csv(output / "label_conflicts.csv", conflicts)
    write_csv(output / "split_conflicts.csv", split_conflicts)
    write_csv(output / "source_license_report.csv", source_license_rows(s2c_source, textoir, official_banking, official_stack))

    # 历史实验实际读取的是 prepared/v19，而不是上面的 raw 或 protocol_v2 candidate。
    # 单独输出这部分，防止把 KIR 子集造成的样本数差异误报为 raw source 冲突。
    if args.reuse_historical_audit:
        require_reusable_historical_audit(output)
    else:
        write_historical_audit(output, artifacts, prepared, sources)

    def all_scope(dataset: str, comparison: str) -> dict[str, object]:
        """取一个完整数据集的比对行；缺失来源不允许回退到部分 split。"""
        for row in summary:
            if row["dataset"] == dataset and row["comparison"] == comparison and row["scope"] == "all":
                return row
        raise RuntimeError(f"Missing all-scope comparison: {dataset}/{comparison}")

    clinc_textoir = all_scope("clinc150", "official_vs_textoir")
    clinc_s2c_raw = all_scope("clinc150", "official_vs_s2c_raw")
    clinc_candidate = all_scope("clinc150", "official_vs_s2c_candidate")
    clinc_reconstruction = all_scope("clinc150", "official_vs_s2c_official_reconstruction")
    banking_textoir = all_scope("banking77", "official_vs_textoir")
    banking_s2c_raw = all_scope("banking77", "official_vs_s2c_raw")
    banking_candidate = all_scope("banking77", "official_vs_s2c_candidate")
    banking_reconstruction = all_scope("banking77", "official_vs_s2c_official_reconstruction")
    stack_textoir = all_scope("stackoverflow", "official_vs_textoir")
    stack_s2c_raw = all_scope("stackoverflow", "official_vs_s2c_raw")
    stack_candidate = all_scope("stackoverflow", "official_vs_s2c_candidate")

    decisions = {
        "clinc150": {
            "dataset": "clinc150", "decision": "reconstructed_from_official",
            "dataset_version": official_version,
            "official_source": "https://github.com/clinc/oos-eval:data/data_full.json",
            "official_revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
            "official_file_sha256": sha256(official_clinc / "data" / "data_full.json"),
            "textoir_commit": "dffe2b1b848a069a6808f8089b4cb9bd16e2062b",
            "textoir_match_rate": clinc_textoir["exact_record_match_rate"],
            "textoir_normalized_text_match_rate": clinc_textoir["normalized_text_match_rate"],
            "s2c_match_rate": clinc_s2c_raw["exact_record_match_rate"],
            "s2c_protocol_v2_candidate_match_rate": clinc_candidate["exact_record_match_rate"],
            "s2c_official_reconstruction_match_rate": clinc_reconstruction["exact_record_match_rate"],
            "split_match": bool(clinc_reconstruction["split_match_rate"] == 1.0),
            "split_match_rate": clinc_reconstruction["split_match_rate"],
            "license_status": "verified", "canonical_source": "official_clinc/data/data_full.json",
            "protocol_v2_candidate_status": "rejected_textoir_candidate",
            "canonical_rebuild_status": "materialized_and_validated",
            "rejected_sources": ["textoir_oos_snapshot_as_raw_canonical", "historical_s2c_prepared_v19"],
            "rejection_reasons": [
                "TEXTOIR content matches the official corpus but collapses native-OOS sub-splits into test.",
                "The existing s2c/data candidate was imported from that TEXTOIR snapshot, so it is not an official-raw canonical dataset.",
                "Historical prepared snapshots are KIR-derived experiment inputs, not raw sources.",
            ],
            "legacy_result_policy": "Historical results remain traceable only. protocol_v2_official_v1 uses the official data_full.json reconstruction.",
        },
        "banking77": {
            "dataset": "banking77", "decision": "reconstructed_from_official",
            "dataset_version": official_version,
            "official_source": "https://github.com/PolyAI-LDN/task-specific-datasets:banking_data/{train,test}.csv",
            "official_revision": "57ec275d8078af65b7731c2a98be812d844a6d6b",
            "official_file_sha256": sha256(official_banking / "banking_data" / "train.csv") + ";" + sha256(official_banking / "banking_data" / "test.csv"),
            "textoir_commit": "dffe2b1b848a069a6808f8089b4cb9bd16e2062b",
            "textoir_match_rate": banking_textoir["exact_record_match_rate"],
            "textoir_normalized_text_match_rate": banking_textoir["normalized_text_match_rate"],
            "s2c_match_rate": banking_s2c_raw["exact_record_match_rate"],
            "s2c_protocol_v2_candidate_match_rate": banking_candidate["exact_record_match_rate"],
            "s2c_official_reconstruction_match_rate": banking_reconstruction["exact_record_match_rate"],
            "split_match": bool(banking_reconstruction["split_match_rate"] == 1.0),
            "split_match_rate": banking_reconstruction["split_match_rate"],
            "license_status": "verified", "canonical_source": "official_banking/banking_data",
            "protocol_v2_candidate_status": "rejected_textoir_candidate",
            "canonical_rebuild_status": "materialized_and_validated",
            "calibration_derivation": "stratified_sha256_rank_v1; official train only; target 1000",
            "rejected_sources": ["textoir_banking_snapshot_as_raw_canonical", "legacy_banking77_oos_extension"],
            "rejection_reasons": [
                "TEXTOIR creates dev by moving 1,000 official-train records; the conversion rule must be versioned separately from the raw source.",
                "The existing s2c/data candidate was imported from that TEXTOIR snapshot, not directly from the official raw source.",
                "BANKING77-OOS is a different, unverified legacy extension and cannot be merged with standard Banking77.",
            ],
            "legacy_result_policy": "Use official standard Banking77 plus the recorded calibration derivation. Legacy BANKING77-OOS remains incomparable.",
        },
        "banking77_oos": {
            "dataset": "banking77_oos", "decision": "blocked_unverified",
            "official_source": "https://github.com/PolyAI-LDN/task-specific-datasets (base Banking77 only; no official OOS extension found)",
            "official_revision": "57ec275d8078af65b7731c2a98be812d844a6d6b",
            "official_file_sha256": sha256(official_banking / "banking_data" / "train.csv") + ";" + sha256(official_banking / "banking_data" / "test.csv"),
            "textoir_commit": "dffe2b1b848a069a6808f8089b4cb9bd16e2062b",
            # OOS 扩展没有可接受的官方对照；用 null 表示“不可裁决”，
            # 不把部分 base 文本重合误写成整体匹配率。
            "textoir_match_rate": None, "s2c_match_rate": None, "split_match": False,
            "license_status": "blocked", "canonical_source": None,
            "rejected_sources": ["s2c_banking_oos_raw", "s2c_prepared_banking77_oos_kir50_seed42"],
            "rejection_reasons": ["No traceable official OOS-extension source or license", "Known base records contain text-label conflicts against official Banking77", "TEXTOIR has standard Banking77 but not this OOS extension"],
            "legacy_result_policy": "Do not start protocol_v2 or claim official/TEXTOIR comparability; historical BANKING77-OOS results remain legacy-only.",
        },
        "stackoverflow": {
            "dataset": "stackoverflow", "decision": "blocked_unverified",
            "dataset_version": "stackoverflow_source_unverified_7c207f51e649",
            "official_source": "https://github.com/jacoxu/StackOverflow:rawText/{title,label}_StackOverflow.txt",
            "official_revision": "7c207f51e649fff9e4736610b9d44431bb7ccf00",
            "official_file_sha256": sha256(official_stack / "title_StackOverflow.txt") + ";" + sha256(official_stack / "label_StackOverflow.txt"),
            "upstream_readme_sha256": STACK_UPSTREAM_README_SHA256,
            "upstream_license_file_status": STACK_UPSTREAM_LICENSE_FILE_STATUS,
            "textoir_commit": "dffe2b1b848a069a6808f8089b4cb9bd16e2062b",
            "textoir_match_rate": stack_textoir["exact_record_match_rate"],
            "textoir_normalized_text_match_rate": stack_textoir["normalized_text_match_rate"],
            "s2c_match_rate": stack_s2c_raw["exact_record_match_rate"],
            "s2c_protocol_v2_candidate_match_rate": stack_candidate["exact_record_match_rate"],
            "split_match": bool(stack_candidate["split_match_rate"] == 1.0),
            "split_match_rate": stack_candidate["split_match_rate"],
            "license_status": "blocked", "canonical_source": None,
            "license_assessment": "The upstream README asks users to acknowledge Kaggle but does not identify a Kaggle dataset, data-dump revision, post identifiers, or redistribution license. The general Stack Overflow CC BY-SA policy cannot be assigned per record without source post metadata.",
            "protocol_v2_candidate_status": "blocked",
            "rejected_sources": ["official_stackoverflow", "textoir_stackoverflow", "s2c_stackoverflow_raw", "s2c_protocol_v2_candidate"],
            "rejection_reasons": [
                "Raw content matches the public repository, but neither the upstream repository nor TEXTOIR declares a clear data license.",
                "The historical s2c corpus deduplicates to 19,980 rows, whereas the TEXTOIR candidate has 20,000 rows.",
            ],
            "legacy_result_policy": "Do not include StackOverflow in canonical protocol_v2, formal experiments, or TEXTOIR-comparable claims until source and license are independently verified.",
        },
    }
    for dataset, decision in decisions.items():
        decision_path = output / dataset / "dataset_decision.json"
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    snapshot = {
        "audit_scope": "three_way_data_provenance",
        "generated_by": "s2c/tools/audit/audit_dataset_provenance.py",
        "generated_at": "2026-07-22",
        "training_run": False,
        "embedding_run": False,
        "official_sources": {"clinc_commit": "828f8093932c8fe6ca7936c3d2e52903b1c523de", "banking_commit": "57ec275d8078af65b7731c2a98be812d844a6d6b", "stackoverflow_ref": "7c207f51e649fff9e4736610b9d44431bb7ccf00"},
        "textoir_commit": "dffe2b1b848a069a6808f8089b4cb9bd16e2062b",
        "comparison_files": sorted(p.name for p in output.glob("*.csv")),
        "decision_files": {k: str(Path(k) / "dataset_decision.json") for k in decisions},
        "decisions": {k: v["decision"] for k, v in decisions.items()},
        "protocol_v2_status": "partially_admitted_official_v1_clinc150_banking77_only",
        "experiment_admission": {
            "training": "clinc150_and_banking77_only_after_materialized_inputs",
            "embedding_generation": "clinc150_and_banking77_only_after_materialized_inputs",
            "mogb_or_dcl_reproduction": "clinc150_and_banking77_only_after_materialized_inputs",
            "textoir_fair_comparability_claim": False,
        },
        "historical_reference_audit": "reused_existing_inventory" if args.reuse_historical_audit else "recomputed",
    }
    (output / "audit_manifest.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary_rows": len(summary), "missing": len(missing), "extra": len(extra), "label_conflicts": len(conflicts), "split_conflicts": len(split_conflicts), "decisions": snapshot["decisions"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
