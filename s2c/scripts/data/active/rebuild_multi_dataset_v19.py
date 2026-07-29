#!/usr/bin/env python3
"""Rebuild multi-dataset v19 protocol artifacts.

This script expands the CLINC150-only v19 data protocol to four datasets:

* CLINC150
* BANKING77-OOS
* SNIPS
* STACKOVERFLOW

All outputs follow the existing v19 downstream contract:

* gate/train|val|test.json
* router/train|val|test.json
* experts/<domain>/train|val|test.json
* KNOWN_INTENTS.json
* MANIFEST.json

The goal is to keep the model stack untouched while making the data protocol
deterministic, auditable, and compatible with the current pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from legacy.runtime import WorkspacePaths


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATHS = WorkspacePaths.discover(PROJECT_ROOT)
LOGGER = logging.getLogger(__name__)

CLINC_DATA_ROOT = PATHS.source_data_root / "clinc150" / "data"
BANKING_DATA_ROOT = PATHS.source_data_root / "banking77_oos"
SNIPS_DATA_ROOT = PATHS.source_data_root / "snips"
STACKOVERFLOW_DATA_ROOT = PATHS.source_data_root / "stackoverflow"
CLINC_OOS_ROOT = CLINC_DATA_ROOT

DEFAULT_OUTPUT_ROOT = PATHS.prepared_data_root / "multidataset" / "v19"
STACKOVERFLOW_KAGGLE_DATASET = "jacoxu/StackOverflow (Kaggle-derived 20k subset)"
STACKOVERFLOW_RAW_SOURCE_REPO = "https://github.com/jacoxu/StackOverflow"
STACKOVERFLOW_RAW_FILES = {
    "titles": "title_StackOverflow.txt",
    "labels": "label_StackOverflow.txt",
}
STACKOVERFLOW_INTENTS: List[str] = [
    "wordpress",
    "oracle",
    "svn",
    "apache",
    "excel",
    "matlab",
    "visual-studio",
    "cocoa",
    "osx",
    "bash",
    "spring",
    "hibernate",
    "scala",
    "sharepoint",
    "ajax",
    "qt",
    "drupal",
    "linq",
    "haskell",
    "magento",
]
STACKOVERFLOW_DOMAIN_TO_INTENTS: Dict[str, List[str]] = {"stackoverflow": list(STACKOVERFLOW_INTENTS)}
STACKOVERFLOW_DOMAIN_FALLBACKS: Dict[str, List[str]] = {}
STACKOVERFLOW_KNOWN_INTENTS_BY_KIR: Dict[float, List[str]] = {
    0.25: STACKOVERFLOW_INTENTS[:5],
    0.50: STACKOVERFLOW_INTENTS[:10],
    0.75: STACKOVERFLOW_INTENTS[:15],
}
STACKOVERFLOW_KNOWN_SELECTION_CHOICES = ("nested_prefix", "seeded_random")


@dataclass(frozen=True)
class DatasetBundle:
    """Container for one rebuilt dataset run."""

    dataset: str
    dataset_slug: str
    kir: float
    seed: int
    output_root: Path
    known_intents: List[str]
    unknown_intents: List[str]
    gate: Dict[str, List[Dict[str, Any]]]
    router: Dict[str, List[Dict[str, Any]]]
    experts: Dict[str, Dict[str, List[Dict[str, Any]]]]
    manifest: Dict[str, Any]


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _load_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file.read().splitlines()]


def _stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _allocate_counts(total: int, weights: Sequence[float]) -> List[int]:
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if len(weights) == 0:
        raise ValueError("weights must be non-empty")

    normalized = [max(float(weight), 0.0) for weight in weights]
    weight_sum = float(sum(normalized))
    if weight_sum <= 0.0:
        raise ValueError("weights must sum to a positive value")

    raw = [float(total) * weight / weight_sum for weight in normalized]
    counts = [int(value) for value in raw]
    remainder = int(total - sum(counts))
    if remainder > 0:
        order = sorted(
            range(len(raw)),
            key=lambda idx: (raw[idx] - counts[idx], -idx),
            reverse=True,
        )
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts


def _shuffle_copy(items: Sequence[Any], seed: int) -> List[Any]:
    shuffled = list(items)
    rng = random.Random(int(seed))
    rng.shuffle(shuffled)
    return shuffled


def _split_items(
    items: Sequence[Any],
    weights: Sequence[float],
    seed: int,
) -> List[List[Any]]:
    if len(items) == 0:
        return [[] for _ in weights]

    shuffled = _shuffle_copy(items, seed)
    counts = _allocate_counts(len(shuffled), weights)

    splits: List[List[Any]] = []
    cursor = 0
    for count in counts:
        next_cursor = cursor + count
        splits.append(list(shuffled[cursor:next_cursor]))
        cursor = next_cursor
    return splits


def _make_record(
    text: str,
    intent: str,
    domain: str,
    split: str,
    label: int,
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "text": str(text),
        "intent": str(intent),
        "domain": str(domain),
        "split": str(split),
        "label": int(label),
    }
    for key, value in extra.items():
        payload[str(key)] = value
    return payload


def _balanced_known_selection(
    domain_to_intents: Dict[str, List[str]],
    known_count: int,
    seed: int,
) -> List[str]:
    """Match the existing v19 ratio selection logic for CLINC150."""
    rng = random.Random(int(seed))
    domains = sorted(domain_to_intents.keys())
    all_intents = [intent for domain in domains for intent in domain_to_intents[domain]]
    total = len(all_intents)
    if known_count <= 0 or known_count >= total:
        raise ValueError(f"known_count must be in (0, {total}), got {known_count}")

    ratio = known_count / total
    quotas: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []

    for domain in domains:
        size = len(domain_to_intents[domain])
        raw = size * ratio
        base = int(raw)
        quotas[domain] = min(base, size)
        remainders.append((raw - base, domain))

    assigned = sum(quotas.values())
    remaining = known_count - assigned

    if remaining > 0:
        for _, domain in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if remaining <= 0:
                break
            capacity = len(domain_to_intents[domain]) - quotas[domain]
            if capacity <= 0:
                continue
            quotas[domain] += 1
            remaining -= 1

    if remaining > 0:
        for domain in domains:
            if remaining <= 0:
                break
            capacity = len(domain_to_intents[domain]) - quotas[domain]
            if capacity <= 0:
                continue
            step = min(capacity, remaining)
            quotas[domain] += step
            remaining -= step

    if sum(quotas.values()) != known_count:
        raise RuntimeError("Failed to allocate exact known-intent quota")

    known: List[str] = []
    for domain in domains:
        candidates = sorted(domain_to_intents[domain])
        picked = rng.sample(candidates, quotas[domain]) if quotas[domain] > 0 else []
        known.extend(picked)

    return sorted(known)


def _all_intents_from_domain_map(domain_map: Dict[str, List[str]]) -> List[str]:
    intents = [intent for values in domain_map.values() for intent in values]
    return sorted(dict.fromkeys(intents))


def _normalize_stackoverflow_title(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _parse_stackoverflow_primary_tag(raw_tags: str) -> Optional[str]:
    tags = [tag.strip() for tag in str(raw_tags).split("|") if tag.strip()]
    if not tags:
        return None
    return tags[0]


def _resolve_stackoverflow_intents(
    tag_counts: Dict[str, int],
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    resolved: Dict[str, List[str]] = {}
    replacements: Dict[str, str] = {}
    used_tags: set[str] = set()

    for domain_name, intents in STACKOVERFLOW_DOMAIN_TO_INTENTS.items():
        resolved[domain_name] = []
        fallbacks = list(STACKOVERFLOW_DOMAIN_FALLBACKS.get(domain_name, []))
        for intent_name in intents:
            selected = None
            if tag_counts.get(intent_name, 0) >= STACKOVERFLOW_TARGET_PER_INTENT and intent_name not in used_tags:
                selected = intent_name
            else:
                for fallback_name in fallbacks:
                    if (
                        tag_counts.get(fallback_name, 0) >= STACKOVERFLOW_TARGET_PER_INTENT
                        and fallback_name not in used_tags
                    ):
                        selected = fallback_name
                        replacements[intent_name] = fallback_name
                        break

            if selected is None:
                raise RuntimeError(
                    f"Unable to resolve StackOverflow intent for domain={domain_name} base_intent={intent_name}. "
                    f"Need at least {STACKOVERFLOW_TARGET_PER_INTENT} samples."
                )

            used_tags.add(selected)
            resolved[domain_name].append(selected)

    return resolved, replacements


def _detect_stackoverflow_csv_splits(stackoverflow_root: Path) -> Optional[Dict[str, Path]]:
    candidates = {
        "train": [stackoverflow_root / "train.csv"],
        "valid": [stackoverflow_root / "valid.csv", stackoverflow_root / "val.csv"],
        "test": [stackoverflow_root / "test.csv"],
    }
    resolved: Dict[str, Path] = {}
    for split_name, options in candidates.items():
        for option in options:
            if option.exists():
                resolved[split_name] = option
                break
        if split_name not in resolved:
            return None
    return resolved


def _load_stackoverflow_raw_titles_and_labels(stackoverflow_root: Path) -> List[Dict[str, Any]]:
    titles_path = stackoverflow_root / STACKOVERFLOW_RAW_FILES["titles"]
    labels_path = stackoverflow_root / STACKOVERFLOW_RAW_FILES["labels"]
    if not titles_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Missing StackOverflow raw titles/labels. "
            f"Expected {titles_path} and {labels_path}."
        )

    titles = _load_lines(titles_path)
    raw_labels = _load_lines(labels_path)
    if len(titles) != len(raw_labels):
        raise RuntimeError(
            f"StackOverflow raw titles/labels length mismatch: {len(titles)} vs {len(raw_labels)}"
        )

    rows: List[Dict[str, Any]] = []
    for idx, (title, raw_label) in enumerate(zip(titles, raw_labels)):
        title = str(title).strip()
        if not title:
            continue
        label_idx = int(str(raw_label).strip()) - 1
        if label_idx < 0 or label_idx >= len(STACKOVERFLOW_INTENTS):
            raise ValueError(f"Unexpected StackOverflow label index: {raw_label}")
        rows.append(
            {
                "id": idx,
                "title": title,
                "question": "",
                "tag": STACKOVERFLOW_INTENTS[label_idx],
            }
        )
    return rows


def _write_stackoverflow_split_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Title", "Body", "Tag"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Title": str(row["title"]),
                    "Body": str(row.get("question", "")),
                    "Tag": str(row["tag"]),
                }
            )


def _materialize_stackoverflow_csv_splits_from_raw(stackoverflow_root: Path) -> Dict[str, Path]:
    rows = _load_stackoverflow_raw_titles_and_labels(stackoverflow_root)
    intent_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        intent_to_rows[str(row["tag"])].append(row)

    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for intent_name in STACKOVERFLOW_INTENTS:
        intent_rows = intent_to_rows.get(intent_name, [])
        if len(intent_rows) == 0:
            raise RuntimeError(f"Missing rows for StackOverflow intent: {intent_name}")
        train_split, valid_split, test_split = _split_items(
            intent_rows,
            (0.6, 0.1, 0.3),
            seed=_stable_int(f"stackoverflow_raw_split::{intent_name}"),
        )
        train_rows.extend(train_split)
        valid_rows.extend(valid_split)
        test_rows.extend(test_split)

    csv_paths = {
        "train": stackoverflow_root / "train.csv",
        "valid": stackoverflow_root / "valid.csv",
        "test": stackoverflow_root / "test.csv",
    }
    _write_stackoverflow_split_csv(csv_paths["train"], train_rows)
    _write_stackoverflow_split_csv(csv_paths["valid"], valid_rows)
    _write_stackoverflow_split_csv(csv_paths["test"], test_rows)
    return csv_paths


def _normalize_stackoverflow_single_label_row(
    row: Dict[str, Any],
    *,
    split_name: str,
    row_id: int,
) -> Optional[Dict[str, Any]]:
    title = str(row.get("Title", row.get("title", ""))).strip()
    if not title:
        return None

    raw_tag = (
        row.get("Tag")
        or row.get("tag")
        or row.get("Label")
        or row.get("label")
        or row.get("Tags")
        or row.get("tags")
    )
    tag = str(raw_tag).strip()
    if not tag:
        return None
    if tag not in STACKOVERFLOW_INTENTS:
        return None

    body = str(row.get("Body", row.get("body", row.get("question", "")))).strip()
    id_raw = row.get("Id", row.get("id", row_id))
    try:
        record_id = int(id_raw)
    except (TypeError, ValueError):
        record_id = int(_stable_int(f"{split_name}:{tag}:{title}") % 10_000_000_000)

    return {
        "id": record_id,
        "title": title,
        "question": body,
        "tag": tag,
        "intent": tag,
        "domain": "stackoverflow",
        "source_split": split_name,
    }


def sync_stackoverflow_source(
    stackoverflow_root: Path,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    records_path = stackoverflow_root / "records.jsonl"
    manifest_path = stackoverflow_root / "SOURCE_MANIFEST.json"
    if not force and records_path.exists() and manifest_path.exists():
        cached_manifest = _load_json(manifest_path)
        if (
            cached_manifest.get("domain_policy") == "single_domain"
            and cached_manifest.get("intent_policy") == "single_label_tag"
            and list(cached_manifest.get("intent_universe", [])) == list(STACKOVERFLOW_INTENTS)
        ):
            return cached_manifest

    stackoverflow_root.mkdir(parents=True, exist_ok=True)
    csv_splits = _detect_stackoverflow_csv_splits(stackoverflow_root)
    source_format = "official_csv_splits"
    if csv_splits is None:
        csv_splits = _materialize_stackoverflow_csv_splits_from_raw(stackoverflow_root)
        source_format = "raw_titles_stratified_to_csv_splits"

    selected_records: List[Dict[str, Any]] = []
    split_counts: Dict[str, int] = {}
    filtered_counts = {"empty_title_or_tag": 0, "unknown_tag": 0, "deduplicated_titles": 0}
    for split_name, csv_path in csv_splits.items():
        row_count = 0
        with open(csv_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row_id, row in enumerate(reader):
                row_count += 1
                normalized = _normalize_stackoverflow_single_label_row(
                    row,
                    split_name=split_name,
                    row_id=row_id,
                )
                if normalized is None:
                    raw_tag = (
                        row.get("Tag")
                        or row.get("tag")
                        or row.get("Label")
                        or row.get("label")
                        or row.get("Tags")
                        or row.get("tags")
                    )
                    if raw_tag:
                        filtered_counts["unknown_tag"] += 1
                    else:
                        filtered_counts["empty_title_or_tag"] += 1
                    continue
                selected_records.append(normalized)
        split_counts[split_name] = int(row_count)

    split_priority = {"train": 0, "valid": 1, "test": 2}
    best_by_title: Dict[str, Dict[str, Any]] = {}
    for row in selected_records:
        normalized_title = _normalize_stackoverflow_title(str(row["title"]))
        existing = best_by_title.get(normalized_title)
        if existing is None:
            best_by_title[normalized_title] = row
            continue
        existing_key = (
            int(split_priority.get(str(existing["source_split"]), 99)),
            int(existing["id"]),
        )
        candidate_key = (
            int(split_priority.get(str(row["source_split"]), 99)),
            int(row["id"]),
        )
        if candidate_key < existing_key:
            best_by_title[normalized_title] = row
        filtered_counts["deduplicated_titles"] += 1
    selected_records = list(best_by_title.values())

    selected_records.sort(
        key=lambda row: (
            str(row["source_split"]),
            str(row["intent"]),
            int(row["id"]),
        )
    )

    with open(records_path, "w", encoding="utf-8") as file:
        for row in selected_records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_intent_counts = {intent_name: 0 for intent_name in STACKOVERFLOW_INTENTS}
    split_intent_counts: Dict[str, Dict[str, int]] = {
        split_name: {intent_name: 0 for intent_name in STACKOVERFLOW_INTENTS}
        for split_name in sorted(split_counts.keys())
    }
    for row in selected_records:
        intent_name = str(row["intent"])
        source_split = str(row["source_split"])
        per_intent_counts[intent_name] += 1
        split_intent_counts[source_split][intent_name] += 1

    manifest = {
        "dataset_repo": STACKOVERFLOW_KAGGLE_DATASET,
        "source_repo": STACKOVERFLOW_RAW_SOURCE_REPO,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_splits": split_counts,
        "records_path": str(records_path.resolve()),
        "retained_record_count": len(selected_records),
        "filters": filtered_counts,
        "text_fields": ["Title"],
        "text_policy": "title_only",
        "intent_policy": "single_label_tag",
        "domain_policy": "single_domain",
        "source_format": source_format,
        "intent_universe": list(STACKOVERFLOW_INTENTS),
        "selected_tag_counts": per_intent_counts,
        "split_intent_counts": split_intent_counts,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _load_stackoverflow_records(stackoverflow_root: Path) -> List[Dict[str, Any]]:
    records_path = stackoverflow_root / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(
            f"Missing StackOverflow records mirror: {records_path}. "
            "Run the StackOverflow download step first."
        )

    rows: List[Dict[str, Any]] = []
    with open(records_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _select_stackoverflow_known_intents(
    all_intents: Sequence[str],
    kir: float,
    seed: int,
    strategy: str = "seeded_random",
) -> List[str]:
    rounded_kir = round(float(kir), 2)
    normalized_strategy = str(strategy).strip().lower()
    if normalized_strategy not in STACKOVERFLOW_KNOWN_SELECTION_CHOICES:
        raise ValueError(
            f"Unsupported StackOverflow known selection strategy: {strategy}. "
            f"Expected one of {STACKOVERFLOW_KNOWN_SELECTION_CHOICES}."
        )

    if normalized_strategy == "nested_prefix":
        if rounded_kir not in STACKOVERFLOW_KNOWN_INTENTS_BY_KIR:
            raise ValueError(f"Unsupported StackOverflow KIR: {kir}")
        requested = list(STACKOVERFLOW_KNOWN_INTENTS_BY_KIR[rounded_kir])
    else:
        known_count = int(round(len(all_intents) * float(kir)))
        if known_count <= 0 or known_count >= len(all_intents):
            raise ValueError(f"StackOverflow KIR must select a non-empty proper subset: {kir}")
        requested = sorted(random.Random(int(seed)).sample([str(x) for x in all_intents], known_count))

    available = set(str(intent) for intent in all_intents)
    missing = [intent_name for intent_name in requested if intent_name not in available]
    if missing:
        raise ValueError(f"StackOverflow known intents missing from source pool: {missing}")
    return requested


def _write_split_files(base_dir: Path, splits: Dict[str, List[Dict[str, Any]]]) -> None:
    for split_name, records in splits.items():
        _write_json(base_dir / f"{split_name}.json", records)


def _write_expert_domain(
    experts_root: Path,
    domain_name: str,
    domain_splits: Dict[str, List[Dict[str, Any]]],
    intent_to_label: Dict[str, int],
) -> None:
    domain_dir = experts_root / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)

    for split_name, records in domain_splits.items():
        _write_json(domain_dir / f"{split_name}.json", records)

    _write_json(domain_dir / "intent_map.json", {int(label): intent for intent, label in intent_to_label.items()})


def _audit_bundle(bundle: DatasetBundle) -> Dict[str, Any]:
    """Light-weight audit that mirrors the v19 data invariants."""
    gate_train = bundle.gate.get("train", [])
    gate_val = bundle.gate.get("val", [])
    gate_test = bundle.gate.get("test", [])
    router_train = bundle.router.get("train", [])
    router_val = bundle.router.get("val", [])
    router_test = bundle.router.get("test", [])

    gate_train_oos = sum(int(row["label"]) for row in gate_train)
    gate_val_oos = sum(int(row["label"]) for row in gate_val)
    gate_test_oos = sum(int(row["label"]) for row in gate_test)

    router_oos = {
        split: sum(
            1
            for row in records
            if str(row.get("intent", "")) == "oos" or str(row.get("domain", "")) == "unknown"
        )
        for split, records in {
            "train": router_train,
            "val": router_val,
            "test": router_test,
        }.items()
    }

    gate_train_texts = {row["text"] for row in gate_train}
    gate_val_texts = {row["text"] for row in gate_val}
    gate_test_texts = {row["text"] for row in gate_test}
    train_val_overlap = gate_train_texts & gate_val_texts
    train_test_overlap = gate_train_texts & gate_test_texts
    val_test_overlap = gate_val_texts & gate_test_texts
    gate_isolation_ok = len(train_val_overlap) == 0 and len(train_test_overlap) == 0

    expert_oos_violations: Dict[str, int] = {}
    for domain_name, domain_splits in bundle.experts.items():
        count = 0
        for split_records in domain_splits.values():
            for row in split_records:
                if str(row.get("intent", "")) == "oos" or str(row.get("domain", "")) != domain_name:
                    count += 1
        expert_oos_violations[domain_name] = count

    audit = {
        "dataset": bundle.dataset,
        "dataset_slug": bundle.dataset_slug,
        "kir": float(bundle.kir),
        "seed": int(bundle.seed),
        "gate": {
            "train_size": len(gate_train),
            "val_size": len(gate_val),
            "test_size": len(gate_test),
            "train_oos_count": gate_train_oos,
            "val_oos_count": gate_val_oos,
            "test_oos_count": gate_test_oos,
            "text_isolation_ok": gate_isolation_ok,
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap),
        },
        "router": {
            "train_size": len(router_train),
            "val_size": len(router_val),
            "test_size": len(router_test),
            "train_oos_count": router_oos["train"],
            "val_oos_count": router_oos["val"],
            "test_oos_count": router_oos["test"],
            "train_label_set": sorted({int(row["label"]) for row in router_train}),
            "val_label_set": sorted({int(row["label"]) for row in router_val}),
            "test_label_set": sorted({int(row["label"]) for row in router_test}),
        },
        "experts": {
            "domain_count": len(bundle.experts),
            "oos_violations": expert_oos_violations,
        },
        "passed": bool(
            gate_train_oos == 0
            and all(value == 0 for value in router_oos.values())
            and gate_isolation_ok
            and all(value == 0 for value in expert_oos_violations.values())
        ),
    }
    return audit


def _finalize_bundle(bundle: DatasetBundle) -> Dict[str, Any]:
    """Persist bundle payloads to disk and write manifests."""
    gate_dir = bundle.output_root / "gate"
    router_dir = bundle.output_root / "router"
    experts_dir = bundle.output_root / "experts"
    gate_dir.mkdir(parents=True, exist_ok=True)
    router_dir.mkdir(parents=True, exist_ok=True)
    experts_dir.mkdir(parents=True, exist_ok=True)

    _write_split_files(gate_dir, bundle.gate)
    _write_split_files(router_dir, bundle.router)
    _write_json(router_dir / "domain_map.json", bundle.manifest.get("domain_map", {}))

    for domain_name, domain_splits in bundle.experts.items():
        intent_to_label = bundle.manifest["experts"][domain_name]["intent_to_label"]
        _write_expert_domain(experts_dir, domain_name, domain_splits, intent_to_label)

    _write_json(bundle.output_root / "KNOWN_INTENTS.json", bundle.manifest["known_intents_manifest"])
    _write_json(bundle.output_root / "MANIFEST.json", bundle.manifest)

    audit = _audit_bundle(bundle)
    _write_json(bundle.output_root / "AUDIT.json", audit)
    return audit


def build_clinc_bundle(
    clinc_root: Path,
    kir: float,
    seed: int,
    output_root: Path,
) -> DatasetBundle:
    raw = _load_json(clinc_root / "data_full.json")
    domains = _load_json(clinc_root / "domains.json")
    domain_to_intents = {str(domain): [str(intent) for intent in intents] for domain, intents in domains.items()}
    all_intents = _all_intents_from_domain_map(domain_to_intents)

    known_count = int(round(len(all_intents) * float(kir)))
    known_count = max(1, min(len(all_intents) - 1, known_count))
    known_intents = _balanced_known_selection(domain_to_intents, known_count, seed)
    known_set = set(known_intents)
    unknown_intents = sorted([intent for intent in all_intents if intent not in known_set])

    domain_map = {str(domain): idx for idx, domain in enumerate(sorted(domain_to_intents.keys()))}
    intent_to_domain = {intent: domain for domain, intents in domain_to_intents.items() for intent in intents}

    gate_train: List[Dict[str, Any]] = []
    gate_val: List[Dict[str, Any]] = []
    gate_test: List[Dict[str, Any]] = []
    router_splits = {"train": [], "val": [], "test": []}
    domain_to_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"train": [], "val": [], "test": []}
    )

    for split_name in ("train", "val", "test"):
        for text, intent in raw[split_name]:
            text = str(text)
            intent = str(intent)
            domain = str(intent_to_domain[intent])
            is_known = intent in known_set
            if split_name == "train":
                if is_known:
                    gate_train.append(_make_record(text, intent, domain, split_name, 0, source_split=split_name))
                    router_splits[split_name].append(
                        _make_record(text, intent, domain, split_name, int(domain_map[domain]), source_split=split_name)
                    )
                    domain_to_records[domain][split_name].append(
                        _make_record(text, intent, domain, split_name, 0, source_split=split_name)
                    )
            elif split_name == "val":
                if is_known:
                    gate_val.append(_make_record(text, intent, domain, split_name, 0, source_split=split_name))
                    router_splits[split_name].append(
                        _make_record(text, intent, domain, split_name, int(domain_map[domain]), source_split=split_name)
                    )
                    domain_to_records[domain][split_name].append(
                        _make_record(text, intent, domain, split_name, 0, source_split=split_name)
                    )
                else:
                    gate_val.append(_make_record(text, intent, "unknown", split_name, 1, source_split=split_name))
            else:
                if is_known:
                    gate_test.append(_make_record(text, intent, domain, split_name, 0, source_split=split_name))
                    router_splits[split_name].append(
                        _make_record(text, intent, domain, split_name, int(domain_map[domain]), source_split=split_name)
                    )
                    domain_to_records[domain][split_name].append(
                        _make_record(text, intent, domain, split_name, 0, source_split=split_name)
                    )
                else:
                    gate_test.append(_make_record(text, intent, "unknown", split_name, 1, source_split=split_name))

    for split_name in ("val",):
        for text, intent in raw.get("oos_val", []):
            gate_val.append(_make_record(text, "oos", "unknown", "oos_val", 1, source_split="oos_val"))
    for text, intent in raw.get("oos_test", []):
        gate_test.append(_make_record(text, "oos", "unknown", "oos_test", 1, source_split="oos_test"))
    for text, intent in raw.get("oos_train", []):
        # Keep the original CLINC protocol for completeness; not used by train splits.
        pass

    experts: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    expert_meta: Dict[str, Dict[str, Any]] = {}
    for domain in sorted(domain_to_intents.keys()):
        intents = [intent for intent in known_intents if intent_to_domain[intent] == domain]
        intent_to_label = {intent: idx for idx, intent in enumerate(sorted(intents))}
        domain_splits = {"train": [], "val": [], "test": []}
        for split_name in ("train", "val", "test"):
            for row in domain_to_records[domain][split_name]:
                row = dict(row)
                row["label"] = int(intent_to_label[row["intent"]])
                domain_splits[split_name].append(row)
        experts[domain] = domain_splits
        expert_meta[domain] = {
            "intent_to_label": intent_to_label,
            "label_to_intent": {int(label): intent for intent, label in intent_to_label.items()},
            "train_size": len(domain_splits["train"]),
            "val_size": len(domain_splits["val"]),
            "test_size": len(domain_splits["test"]),
        }

    manifest = {
        "dataset": "CLINC150",
        "dataset_slug": "clinc150",
        "kir": float(kir),
        "seed": int(seed),
        "source_roots": {
            "clinc_root": str(clinc_root),
            "clinc_oos_root": str(CLINC_OOS_ROOT),
        },
        "known_intents_manifest": {
            "dataset": "CLINC150",
            "ratio": float(kir),
            "seed": int(seed),
            "known_intents": known_intents,
            "unknown_intents": unknown_intents,
            "known_count": len(known_intents),
            "unknown_count": len(unknown_intents),
            "intent_universe": len(all_intents),
            "selection_protocol": {
                "method": "domain_balanced_largest_remainder",
                "deterministic": True,
            },
        },
        "split_counts": {
            "gate": {name: len(records) for name, records in {"train": gate_train, "val": gate_val, "test": gate_test}.items()},
            "router": {name: len(records) for name, records in router_splits.items()},
            "experts": {
                domain: {name: len(records) for name, records in domain_splits.items()}
                for domain, domain_splits in experts.items()
            },
        },
        "domains": sorted(domain_map.keys()),
        "domain_map": domain_map,
        "experts": expert_meta,
        "protocol": {
            "gate_train": "known_only",
            "gate_val": "known_plus_clinc_oos",
            "gate_test": "known_plus_unknown_plus_clinc_oos",
            "router": "known_only",
            "experts": "known_only",
        },
    }

    return DatasetBundle(
        dataset="CLINC150",
        dataset_slug="clinc150",
        kir=float(kir),
        seed=int(seed),
        output_root=output_root,
        known_intents=known_intents,
        unknown_intents=unknown_intents,
        gate={"train": gate_train, "val": gate_val, "test": gate_test},
        router=router_splits,
        experts=experts,
        manifest=manifest,
    )


def _read_banking_split(root: Path, split: str) -> List[Dict[str, Any]]:
    texts = _load_lines(root / split / "seq.in")
    labels = _load_lines(root / split / "label")
    if len(texts) != len(labels):
        raise ValueError(f"BANKING77-OOS split mismatch for {split}: {len(texts)} texts vs {len(labels)} labels")
    return [
        _make_record(text=text, intent=label, domain="banking", split=split, label=0, source_split=split)
        for text, label in zip(texts, labels)
    ]


def _read_banking_oos_split(root: Path, split: str, source: str) -> List[Dict[str, Any]]:
    texts = _load_lines(root / source / split / "seq.in")
    if source == "id-oos":
        labels = _load_lines(root / source / split / "label_original")
        if len(texts) != len(labels):
            raise ValueError(f"BANKING77-OOS {source}/{split} mismatch: {len(texts)} texts vs {len(labels)} labels")
        return [
            _make_record(text=text, intent=label, domain="unknown", split=f"{source}_{split}", label=1, source_split=f"{source}_{split}")
            for text, label in zip(texts, labels)
        ]

    labels = _load_lines(root / source / split / "label")
    if len(texts) != len(labels):
        raise ValueError(f"BANKING77-OOS {source}/{split} mismatch: {len(texts)} texts vs {len(labels)} labels")
    return [
        _make_record(text=text, intent="oos", domain="unknown", split=f"{source}_{split}", label=1, source_split=f"{source}_{split}")
        for text, label in zip(texts, labels)
    ]


def build_banking_bundle(
    banking_root: Path,
    kir: float,
    seed: int,
    output_root: Path,
) -> DatasetBundle:
    base_train = _read_banking_split(banking_root, "train")
    base_val = _read_banking_split(banking_root, "valid")
    base_test = _read_banking_split(banking_root, "test")

    base_intents = sorted({row["intent"] for row in base_train + base_val + base_test})
    known_count = int(round(len(base_intents) * float(kir)))
    known_count = max(1, min(len(base_intents) - 1, known_count))
    known_intents = sorted(random.Random(int(seed)).sample(base_intents, known_count))
    known_set = set(known_intents)

    id_oos_val = _read_banking_oos_split(banking_root, "valid", "id-oos")
    id_oos_test = _read_banking_oos_split(banking_root, "test", "id-oos")
    ood_oos_val = _read_banking_oos_split(banking_root, "valid", "ood-oos")
    ood_oos_test = _read_banking_oos_split(banking_root, "test", "ood-oos")

    unknown_base_intents = sorted([intent for intent in base_intents if intent not in known_set])
    unknown_intents = sorted(
        set(unknown_base_intents)
        | {row["intent"] for row in id_oos_val + id_oos_test}
        | {"oos"}
    )

    gate_train = [dict(row) for row in base_train if row["intent"] in known_set]
    for row in gate_train:
        row["label"] = 0
        row["split"] = "train"

    gate_val = []
    gate_test = []
    router_splits = {"train": [], "val": [], "test": []}
    domain_to_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"train": [], "val": [], "test": []}
    )

    def _append_known(rows: List[Dict[str, Any]], split_name: str, dest_gate: List[Dict[str, Any]]) -> None:
        for row in rows:
            row = dict(row)
            row["split"] = split_name
            if row["intent"] in known_set:
                row["label"] = 0
                dest_gate.append(row)
                router_splits[split_name].append(_make_record(
                    text=row["text"],
                    intent=row["intent"],
                    domain="banking",
                    split=split_name,
                    label=0,
                    source_split=split_name,
                ))
                domain_to_records["banking"][split_name].append(
                    _make_record(
                        text=row["text"],
                        intent=row["intent"],
                        domain="banking",
                        split=split_name,
                        label=0,
                        source_split=split_name,
                    )
                )
            else:
                row["label"] = 1
                row["domain"] = "unknown"
                dest_gate.append(row)

    _append_known(base_val, "val", gate_val)
    _append_known(base_test, "test", gate_test)

    for row in id_oos_val:
        gate_val.append(row)
    for row in id_oos_test:
        gate_test.append(row)
    for row in ood_oos_val:
        gate_val.append(row)
    for row in ood_oos_test:
        gate_test.append(row)

    router_splits["train"] = [
        _make_record(text=row["text"], intent=row["intent"], domain="banking", split="train", label=0, source_split="train")
        for row in base_train
        if row["intent"] in known_set
    ]
    router_splits["val"] = [
        _make_record(text=row["text"], intent=row["intent"], domain="banking", split="val", label=0, source_split="val")
        for row in base_val
        if row["intent"] in known_set
    ]
    router_splits["test"] = [
        _make_record(text=row["text"], intent=row["intent"], domain="banking", split="test", label=0, source_split="test")
        for row in base_test
        if row["intent"] in known_set
    ]

    intent_to_label = {intent: idx for idx, intent in enumerate(known_intents)}
    expert_splits = {"banking": {"train": [], "val": [], "test": []}}
    for split_name, rows in router_splits.items():
        expert_splits["banking"][split_name] = [
            _make_record(
                text=row["text"],
                intent=row["intent"],
                domain="banking",
                split=split_name,
                label=int(intent_to_label[row["intent"]]),
                source_split=split_name,
            )
            for row in rows
        ]

    domain_map = {"banking": 0}
    manifest = {
        "dataset": "BANKING77-OOS",
        "dataset_slug": "banking77_oos",
        "kir": float(kir),
        "seed": int(seed),
        "source_roots": {
            "banking_root": str(banking_root),
        },
        "known_intents_manifest": {
            "dataset": "BANKING77-OOS",
            "ratio": float(kir),
            "seed": int(seed),
            "known_intents": known_intents,
            "unknown_intents": unknown_intents,
            "known_count": len(known_intents),
            "unknown_count": len(unknown_intents),
            "intent_universe": len(base_intents),
            "selection_protocol": {
                "method": "uniform_random_sample",
                "deterministic": True,
                "notes": "Known intents are sampled from the 50 in-domain BANKING77 intents; unknown intents include the held-out in-domain intents plus id-oos labels.",
            },
        },
        "split_counts": {
            "gate": {name: len(records) for name, records in {"train": gate_train, "val": gate_val, "test": gate_test}.items()},
            "router": {name: len(records) for name, records in router_splits.items()},
            "experts": {domain: {name: len(records) for name, records in splits.items()} for domain, splits in expert_splits.items()},
        },
        "domains": ["banking"],
        "domain_map": domain_map,
        "experts": {
            "banking": {
                "intent_to_label": intent_to_label,
                "label_to_intent": {int(label): intent for intent, label in intent_to_label.items()},
                "train_size": len(expert_splits["banking"]["train"]),
                "val_size": len(expert_splits["banking"]["val"]),
                "test_size": len(expert_splits["banking"]["test"]),
            }
        },
        "protocol": {
            "gate_train": "known_only",
            "gate_val": "known_plus_unknown_plus_native_oos",
            "gate_test": "known_plus_unknown_plus_native_oos",
            "router": "known_only",
            "experts": "known_only",
        },
        "oos_sources": {
            "id_oos": "id-oos/{valid,test}",
            "ood_oos": "ood-oos/{valid,test}",
        },
    }

    return DatasetBundle(
        dataset="BANKING77-OOS",
        dataset_slug="banking77_oos",
        kir=float(kir),
        seed=int(seed),
        output_root=output_root,
        known_intents=known_intents,
        unknown_intents=unknown_intents,
        gate={"train": gate_train, "val": gate_val, "test": gate_test},
        router=router_splits,
        experts=expert_splits,
        manifest=manifest,
    )


def build_banking77_standard_bundle(
    banking77_root: Path,
    kir: float,
    seed: int,
    output_root: Path,
) -> DatasetBundle:
    """Build standard Banking77 bundle (77 intents, TextOIR protocol).

    Unlike Banking77-OOS, this uses the full 77-intent Banking77 dataset
    with train/dev/test splits matching TextOIR. Unknown intents (not selected
    as known via KIR) are treated as OOS for gate evaluation.
    """
    base_train = _read_banking_split(banking77_root, "train")
    base_val = _read_banking_split(banking77_root, "valid")
    base_test = _read_banking_split(banking77_root, "test")

    base_intents = sorted({row["intent"] for row in base_train + base_val + base_test})
    known_count = int(round(len(base_intents) * float(kir)))
    known_count = max(1, min(len(base_intents) - 1, known_count))
    known_intents = sorted(random.Random(int(seed)).sample(base_intents, known_count))
    known_set = set(known_intents)
    unknown_intents = sorted([intent for intent in base_intents if intent not in known_set])

    gate_train = [dict(row) for row in base_train if row["intent"] in known_set]
    for row in gate_train:
        row["label"] = 0
        row["split"] = "train"

    gate_val = []
    gate_test = []
    router_splits = {"train": [], "val": [], "test": []}
    domain_to_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"train": [], "val": [], "test": []}
    )

    def _append_known(rows: List[Dict[str, Any]], split_name: str, dest_gate: List[Dict[str, Any]]) -> None:
        for row in rows:
            row = dict(row)
            row["split"] = split_name
            if row["intent"] in known_set:
                row["label"] = 0
                dest_gate.append(row)
                router_splits[split_name].append(_make_record(
                    text=row["text"],
                    intent=row["intent"],
                    domain="banking",
                    split=split_name,
                    label=0,
                    source_split=split_name,
                ))
                domain_to_records["banking"][split_name].append(
                    _make_record(
                        text=row["text"],
                        intent=row["intent"],
                        domain="banking",
                        split=split_name,
                        label=0,
                        source_split=split_name,
                    )
                )
            else:
                row["label"] = 1
                row["domain"] = "unknown"
                dest_gate.append(row)

    _append_known(base_val, "val", gate_val)
    _append_known(base_test, "test", gate_test)

    router_splits["train"] = [
        _make_record(text=row["text"], intent=row["intent"], domain="banking", split="train", label=0, source_split="train")
        for row in base_train
        if row["intent"] in known_set
    ]
    router_splits["val"] = [
        _make_record(text=row["text"], intent=row["intent"], domain="banking", split="val", label=0, source_split="val")
        for row in base_val
        if row["intent"] in known_set
    ]
    router_splits["test"] = [
        _make_record(text=row["text"], intent=row["intent"], domain="banking", split="test", label=0, source_split="test")
        for row in base_test
        if row["intent"] in known_set
    ]

    intent_to_label = {intent: idx for idx, intent in enumerate(known_intents)}
    expert_splits = {"banking": {"train": [], "val": [], "test": []}}
    for split_name, rows in router_splits.items():
        expert_splits["banking"][split_name] = [
            _make_record(
                text=row["text"],
                intent=row["intent"],
                domain="banking",
                split=split_name,
                label=int(intent_to_label[row["intent"]]),
                source_split=split_name,
            )
            for row in rows
        ]

    domain_map = {"banking": 0}
    manifest = {
        "dataset": "BANKING77",
        "dataset_slug": "banking77",
        "kir": float(kir),
        "seed": int(seed),
        "source_roots": {
            "banking77_root": str(banking77_root),
        },
        "known_intents_manifest": {
            "dataset": "BANKING77",
            "ratio": float(kir),
            "seed": int(seed),
            "known_intents": known_intents,
            "unknown_intents": unknown_intents,
            "known_count": len(known_intents),
            "unknown_count": len(unknown_intents),
            "intent_universe": len(base_intents),
            "selection_protocol": {
                "method": "uniform_random_sample",
                "deterministic": True,
                "notes": "Standard Banking77 with 77 intents. Known intents sampled uniformly; unknown intents serve as OOS for gate evaluation.",
            },
        },
        "split_counts": {
            "gate": {name: len(records) for name, records in {"train": gate_train, "val": gate_val, "test": gate_test}.items()},
            "router": {name: len(records) for name, records in router_splits.items()},
            "experts": {domain: {name: len(records) for name, records in splits.items()} for domain, splits in expert_splits.items()},
        },
        "domains": ["banking"],
        "domain_map": domain_map,
        "experts": {
            "banking": {
                "intent_to_label": intent_to_label,
                "label_to_intent": {int(label): intent for intent, label in intent_to_label.items()},
                "train_size": len(expert_splits["banking"]["train"]),
                "val_size": len(expert_splits["banking"]["val"]),
                "test_size": len(expert_splits["banking"]["test"]),
            }
        },
        "protocol": {
            "gate_train": "known_only",
            "gate_val": "known_plus_unknown",
            "gate_test": "known_plus_unknown",
            "router": "known_only",
            "experts": "known_only",
        },
    }

    return DatasetBundle(
        dataset="BANKING77",
        dataset_slug="banking77",
        kir=float(kir),
        seed=int(seed),
        output_root=output_root,
        known_intents=known_intents,
        unknown_intents=unknown_intents,
        gate={"train": gate_train, "val": gate_val, "test": gate_test},
        router=router_splits,
        experts=expert_splits,
        manifest=manifest,
    )


def _flatten_snips_sample(sample: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for item in sample.get("data", []):
        pieces.append(str(item.get("text", "")))
    return "".join(pieces).strip()


def _load_snips_intent_pool(raw_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    intent_to_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_texts: set[str] = set()
    skipped_duplicates = 0
    for intent_dir in sorted(raw_root.iterdir()):
        if not intent_dir.is_dir():
            continue
        intent_name = intent_dir.name
        candidate_files = sorted(intent_dir.glob("train_*.json")) + sorted(intent_dir.glob("validate_*.json"))
        for json_file in candidate_files:
            if json_file.name.endswith("_full.json"):
                continue
            payload = _load_json(json_file)
            if not isinstance(payload, dict):
                continue
            for key, samples in payload.items():
                if str(key) != intent_name or not isinstance(samples, list):
                    continue
                for idx, sample in enumerate(samples):
                    if not isinstance(sample, dict):
                        continue
                    text = _flatten_snips_sample(sample)
                    if not text:
                        continue
                    if text in seen_texts:
                        skipped_duplicates += 1
                        continue
                    seen_texts.add(text)
                    intent_to_samples[intent_name].append(
                        {
                            "text": text,
                            "intent": intent_name,
                            "domain": "snips",
                            "split": json_file.stem,
                            "label": 0,
                            "source_split": json_file.stem,
                            "sample_index": idx,
                        }
                    )
    if len(intent_to_samples) == 0:
        raise RuntimeError(f"No SNIPS samples found under {raw_root}")
    if skipped_duplicates > 0:
        LOGGER.warning("SNIPS loader skipped %d duplicate texts during flattening.", skipped_duplicates)
    return intent_to_samples


def build_snips_bundle(
    snips_root: Path,
    clinc_oos_root: Path,
    kir: float,
    seed: int,
    output_root: Path,
) -> DatasetBundle:
    intent_to_samples = _load_snips_intent_pool(snips_root)
    all_intents = sorted(intent_to_samples.keys())
    known_count = int(round(len(all_intents) * float(kir)))
    known_count = max(1, min(len(all_intents) - 1, known_count))
    known_intents = sorted(random.Random(int(seed)).sample(all_intents, known_count))
    known_set = set(known_intents)
    unknown_intents = sorted([intent for intent in all_intents if intent not in known_set])

    gate_train: List[Dict[str, Any]] = []
    gate_val: List[Dict[str, Any]] = []
    gate_test: List[Dict[str, Any]] = []
    router_splits = {"train": [], "val": [], "test": []}
    domain_to_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"train": [], "val": [], "test": []}
    )

    for intent in all_intents:
        samples = intent_to_samples[intent]
        intent_seed = int(seed) + _stable_int(f"snips::{intent}")
        if intent in known_set:
            train_samples, val_samples, test_samples = _split_items(samples, (0.8, 0.1, 0.1), intent_seed)
            for split_name, split_samples in zip(("train", "val", "test"), (train_samples, val_samples, test_samples)):
                for row in split_samples:
                    record = _make_record(
                        text=row["text"],
                        intent=intent,
                        domain="snips",
                        split=split_name,
                        label=0,
                        source_split=row["source_split"],
                    )
                    if split_name == "train":
                        gate_train.append(record)
                    elif split_name == "val":
                        gate_val.append(record)
                    else:
                        gate_test.append(record)
                    router_splits[split_name].append(
                        _make_record(
                            text=row["text"],
                            intent=intent,
                            domain="snips",
                            split=split_name,
                            label=0,
                            source_split=row["source_split"],
                        )
                    )
                    domain_to_records["snips"][split_name].append(
                        _make_record(
                            text=row["text"],
                            intent=intent,
                            domain="snips",
                            split=split_name,
                            label=0,
                            source_split=row["source_split"],
                        )
                    )
        else:
            val_samples, test_samples = _split_items(samples, (0.5, 0.5), intent_seed)
            for row in val_samples:
                gate_val.append(
                    _make_record(
                        text=row["text"],
                        intent=intent,
                        domain="unknown",
                        split="val",
                        label=1,
                        source_split=row["source_split"],
                    )
                )
            for row in test_samples:
                gate_test.append(
                    _make_record(
                        text=row["text"],
                        intent=intent,
                        domain="unknown",
                        split="test",
                        label=1,
                        source_split=row["source_split"],
                    )
                )

    clinc_oos_payload = _load_json(clinc_oos_root / "data_oos_plus.json")
    for row in clinc_oos_payload.get("oos_val", []):
        if not isinstance(row, list) or len(row) < 1:
            continue
        gate_val.append(
            _make_record(
                text=str(row[0]),
                intent="oos",
                domain="unknown",
                split="clinc_oos_val",
                label=1,
                source_split="clinc_oos_val",
            )
        )
    for row in clinc_oos_payload.get("oos_test", []):
        if not isinstance(row, list) or len(row) < 1:
            continue
        gate_test.append(
            _make_record(
                text=str(row[0]),
                intent="oos",
                domain="unknown",
                split="clinc_oos_test",
                label=1,
                source_split="clinc_oos_test",
            )
        )

    intent_to_label = {intent: idx for idx, intent in enumerate(known_intents)}
    expert_splits = {"snips": {"train": [], "val": [], "test": []}}
    for split_name in ("train", "val", "test"):
        for row in router_splits[split_name]:
            expert_splits["snips"][split_name].append(
                _make_record(
                    text=row["text"],
                    intent=row["intent"],
                    domain="snips",
                    split=split_name,
                    label=int(intent_to_label[row["intent"]]),
                    source_split=row.get("source_split", split_name),
                )
            )

    domain_map = {"snips": 0}
    manifest = {
        "dataset": "SNIPS",
        "dataset_slug": "snips",
        "kir": float(kir),
        "seed": int(seed),
        "source_roots": {
            "snips_root": str(snips_root),
            "clinc_oos_root": str(clinc_oos_root),
        },
        "known_intents_manifest": {
            "dataset": "SNIPS",
            "ratio": float(kir),
            "seed": int(seed),
            "known_intents": known_intents,
            "unknown_intents": unknown_intents,
            "known_count": len(known_intents),
            "unknown_count": len(unknown_intents),
            "intent_universe": len(all_intents),
            "selection_protocol": {
                "method": "uniform_random_sample",
                "deterministic": True,
                "notes": "Known intents are sampled from the 7 SNIPS intents; unknown intents are held-out intents plus injected CLINC OOS examples.",
            },
        },
        "split_counts": {
            "gate": {name: len(records) for name, records in {"train": gate_train, "val": gate_val, "test": gate_test}.items()},
            "router": {name: len(records) for name, records in router_splits.items()},
            "experts": {domain: {name: len(records) for name, records in splits.items()} for domain, splits in expert_splits.items()},
        },
        "domains": ["snips"],
        "domain_map": domain_map,
        "experts": {
            "snips": {
                "intent_to_label": intent_to_label,
                "label_to_intent": {int(label): intent for intent, label in intent_to_label.items()},
                "train_size": len(expert_splits["snips"]["train"]),
                "val_size": len(expert_splits["snips"]["val"]),
                "test_size": len(expert_splits["snips"]["test"]),
            }
        },
        "protocol": {
            "gate_train": "known_only",
            "gate_val": "known_plus_unknown_plus_clinc_oos",
            "gate_test": "known_plus_unknown_plus_clinc_oos",
            "router": "known_only",
            "experts": "known_only",
            "known_split_ratios": [0.8, 0.1, 0.1],
            "unknown_split_ratios": [0.5, 0.5],
        },
        "oos_sources": {
            "clinc_oos_val": "clinc_data_origin/data/data_oos_plus.json::oos_val",
            "clinc_oos_test": "clinc_data_origin/data/data_oos_plus.json::oos_test",
        },
    }

    return DatasetBundle(
        dataset="SNIPS",
        dataset_slug="snips",
        kir=float(kir),
        seed=int(seed),
        output_root=output_root,
        known_intents=known_intents,
        unknown_intents=unknown_intents,
        gate={"train": gate_train, "val": gate_val, "test": gate_test},
        router=router_splits,
        experts=expert_splits,
        manifest=manifest,
    )


def build_stackoverflow_bundle(
    stackoverflow_root: Path,
    kir: float,
    seed: int,
    output_root: Path,
    known_selection_strategy: str = "seeded_random",
) -> DatasetBundle:
    source_manifest = sync_stackoverflow_source(stackoverflow_root)
    raw_records = _load_stackoverflow_records(stackoverflow_root)
    all_intents = [str(intent) for intent in source_manifest["intent_universe"]]
    known_intents = _select_stackoverflow_known_intents(
        all_intents,
        kir,
        seed=seed,
        strategy=known_selection_strategy,
    )
    known_set = set(known_intents)
    unknown_intents = sorted([intent for intent in all_intents if intent not in known_set])

    split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    split_intent_rows: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        source_split: defaultdict(list) for source_split in ("train", "valid", "test")
    }
    for row in raw_records:
        intent_name = str(row["intent"])
        if intent_name not in all_intents:
            continue
        source_split = str(row["source_split"])
        if source_split not in split_intent_rows:
            continue
        split_intent_rows[source_split][intent_name].append(dict(row))

    domain_name = "stackoverflow"
    domain_map = {domain_name: 0}
    gate_train: List[Dict[str, Any]] = []
    gate_val: List[Dict[str, Any]] = []
    gate_test: List[Dict[str, Any]] = []
    router_splits = {"train": [], "val": [], "test": []}
    domain_to_records: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        domain_name: {"train": [], "val": [], "test": []}
    }

    for source_split, rows_by_intent in split_intent_rows.items():
        normalized_split = split_map[source_split]
        for intent_name, rows in rows_by_intent.items():
            for row in rows:
                text = str(row["title"]).strip()
                if intent_name in known_set:
                    record = _make_record(
                        text=text,
                        intent=intent_name,
                        domain=domain_name,
                        split=normalized_split,
                        label=0,
                        source_split=source_split,
                        source_id=int(row["id"]),
                        title=str(row["title"]),
                        question=str(row.get("question", "")),
                        tag=str(row["tag"]),
                    )
                    if normalized_split == "train":
                        gate_train.append(record)
                    elif normalized_split == "val":
                        gate_val.append(record)
                    else:
                        gate_test.append(record)

                    router_record = _make_record(
                        text=text,
                        intent=intent_name,
                        domain=domain_name,
                        split=normalized_split,
                        label=0,
                        source_split=source_split,
                        source_id=int(row["id"]),
                    )
                    router_splits[normalized_split].append(router_record)
                    domain_to_records[domain_name][normalized_split].append(
                        _make_record(
                            text=text,
                            intent=intent_name,
                            domain=domain_name,
                            split=normalized_split,
                            label=0,
                            source_split=source_split,
                            source_id=int(row["id"]),
                        )
                    )
                elif normalized_split in {"val", "test"}:
                    heldout_split = (
                        "heldout_oos_valid" if normalized_split == "val" else "heldout_oos_test"
                    )
                    target = gate_val if normalized_split == "val" else gate_test
                    target.append(
                        _make_record(
                            text=text,
                            intent=intent_name,
                            domain="unknown",
                            split=heldout_split,
                            label=1,
                            source_split=heldout_split,
                            source_id=int(row["id"]),
                            title=str(row["title"]),
                            question=str(row.get("question", "")),
                            tag=str(row["tag"]),
                        )
                    )

    intent_to_label = {intent: idx for idx, intent in enumerate(sorted(known_intents))}
    expert_splits = {"train": [], "val": [], "test": []}
    for split_name in ("train", "val", "test"):
        for row in domain_to_records[domain_name][split_name]:
            expert_row = dict(row)
            expert_row["label"] = int(intent_to_label[row["intent"]])
            expert_splits[split_name].append(expert_row)

    experts: Dict[str, Dict[str, List[Dict[str, Any]]]] = {domain_name: expert_splits}
    expert_meta: Dict[str, Dict[str, Any]] = {
        domain_name: {
            "intent_to_label": intent_to_label,
            "label_to_intent": {int(label): intent for intent, label in intent_to_label.items()},
            "train_size": len(expert_splits["train"]),
            "val_size": len(expert_splits["val"]),
            "test_size": len(expert_splits["test"]),
        }
    }

    manifest = {
        "dataset": "STACKOVERFLOW",
        "dataset_slug": "stackoverflow",
        "kir": float(kir),
        "seed": int(seed),
        "source_roots": {
            "stackoverflow_root": str(stackoverflow_root),
            "kaggle_dataset": STACKOVERFLOW_KAGGLE_DATASET,
        },
        "known_intents_manifest": {
            "dataset": "STACKOVERFLOW",
            "ratio": float(kir),
            "seed": int(seed),
            "known_intents": known_intents,
            "unknown_intents": unknown_intents,
            "known_count": len(known_intents),
            "unknown_count": len(unknown_intents),
            "intent_universe": len(all_intents),
            "selection_protocol": {
                "method": f"official_split_single_domain_{known_selection_strategy}_kir",
                "deterministic": True,
                "seed": int(seed),
                "text_field": "title",
                "intent_policy": "single_label_tag",
                "source_split_policy": "preserve_existing_or_materialize_12k_2k_6k",
            },
        },
        "split_counts": {
            "gate": {name: len(records) for name, records in {"train": gate_train, "val": gate_val, "test": gate_test}.items()},
            "router": {name: len(records) for name, records in router_splits.items()},
            "experts": {
                domain_name: {name: len(records) for name, records in domain_splits.items()}
                for domain_name, domain_splits in experts.items()
            },
        },
        "domains": [domain_name],
        "domain_map": domain_map,
        "experts": expert_meta,
        "text_field": "title",
        "intent_policy": "single_label_tag",
        "oos_sources": {
            "heldout_oos": "heldout_oos_{valid,test}",
        },
        "protocol": {
            "gate_train": "known_only",
            "gate_val": "known_plus_heldout_oos",
            "gate_test": "known_plus_heldout_oos",
            "router": "known_only",
            "experts": "known_only",
        },
    }

    return DatasetBundle(
        dataset="STACKOVERFLOW",
        dataset_slug="stackoverflow",
        kir=float(kir),
        seed=int(seed),
        output_root=output_root,
        known_intents=known_intents,
        unknown_intents=unknown_intents,
        gate={"train": gate_train, "val": gate_val, "test": gate_test},
        router=router_splits,
        experts=experts,
        manifest=manifest,
    )


def _build_output_root(base_root: Path, dataset_slug: str, kir: float, seed: int) -> Path:
    kir_tag = f"kir{int(round(float(kir) * 100)):02d}"
    return base_root / dataset_slug / f"{kir_tag}_seed{int(seed)}"


def _build_bundle(
    dataset: str,
    kir: float,
    seed: int,
    output_root: Path,
    clinc_root: Path,
    banking_root: Path,
    snips_root: Path,
    stackoverflow_root: Path,
    clinc_oos_root: Path,
    stackoverflow_known_selection_strategy: str,
    banking77_root: Optional[Path] = None,
) -> DatasetBundle:
    dataset_key = dataset.strip().upper()
    if dataset_key == "CLINC150":
        return build_clinc_bundle(clinc_root=clinc_root, kir=kir, seed=seed, output_root=output_root)
    if dataset_key == "BANKING77-OOS":
        return build_banking_bundle(banking_root=banking_root, kir=kir, seed=seed, output_root=output_root)
    if dataset_key == "BANKING77":
        if banking77_root is None:
            raise ValueError("BANKING77 requires --banking77_root")
        return build_banking77_standard_bundle(banking77_root=banking77_root, kir=kir, seed=seed, output_root=output_root)
    if dataset_key == "SNIPS":
        return build_snips_bundle(
            snips_root=snips_root,
            clinc_oos_root=clinc_oos_root,
            kir=kir,
            seed=seed,
            output_root=output_root,
        )
    if dataset_key == "STACKOVERFLOW":
        return build_stackoverflow_bundle(
            stackoverflow_root=stackoverflow_root,
            kir=kir,
            seed=seed,
            output_root=output_root,
            known_selection_strategy=stackoverflow_known_selection_strategy,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild multi-dataset v19 protocol artifacts")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CLINC150", "BANKING77-OOS", "SNIPS"],
        help="Datasets to rebuild. Defaults to all supported datasets.",
    )
    parser.add_argument(
        "--kir_values",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="Known-intent ratios to rebuild.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--clinc_root", default=str(CLINC_DATA_ROOT))
    parser.add_argument("--banking_root", default=str(BANKING_DATA_ROOT))
    parser.add_argument("--snips_root", default=str(SNIPS_DATA_ROOT))
    parser.add_argument("--stackoverflow_root", default=str(STACKOVERFLOW_DATA_ROOT))
    parser.add_argument(
        "--stackoverflow_known_selection_strategy",
        default="seeded_random",
        choices=STACKOVERFLOW_KNOWN_SELECTION_CHOICES,
        help="Known-intent selection strategy for StackOverflow KIR splits.",
    )
    parser.add_argument("--clinc_oos_root", default=str(CLINC_OOS_ROOT))
    parser.add_argument("--banking77_root", default=str(PATHS.source_data_root / "banking77"),
                        help="Root directory for standard Banking77 dataset (TextOIR format).")
    parser.add_argument(
        "--index_out",
        default=None,
        help="Optional aggregate index JSON path. Defaults to <output_root>/index.json.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_roots = {
        "CLINC150": Path(args.clinc_root),
        "BANKING77-OOS": Path(args.banking_root),
        "BANKING77": Path(args.banking77_root),
        "SNIPS": Path(args.snips_root),
        "STACKOVERFLOW": Path(args.stackoverflow_root),
    }

    summary: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        dataset_key = dataset.strip().upper()
        if dataset_key not in dataset_roots:
            raise ValueError(f"Unsupported dataset in --datasets: {dataset}")
        for kir in args.kir_values:
            kir = float(kir)
            run_output_root = _build_output_root(output_root, dataset_key.lower().replace("-", "_"), kir, int(args.seed))
            LOGGER.info(
                "Rebuilding dataset=%s kir=%.2f seed=%d -> %s",
                dataset_key,
                kir,
                int(args.seed),
                run_output_root,
            )
            bundle = _build_bundle(
                dataset=dataset_key,
                kir=kir,
                seed=int(args.seed),
                output_root=run_output_root,
                clinc_root=Path(args.clinc_root),
                banking_root=Path(args.banking_root),
                snips_root=Path(args.snips_root),
                stackoverflow_root=Path(args.stackoverflow_root),
                clinc_oos_root=Path(args.clinc_oos_root),
                stackoverflow_known_selection_strategy=str(args.stackoverflow_known_selection_strategy),
                banking77_root=Path(args.banking77_root),
            )
            audit = _finalize_bundle(bundle)
            summary.append(
                {
                    "dataset": bundle.dataset,
                    "dataset_slug": bundle.dataset_slug,
                    "kir": float(bundle.kir),
                    "seed": int(bundle.seed),
                    "output_root": str(bundle.output_root),
                    "audit_passed": bool(audit["passed"]),
                    "split_counts": bundle.manifest["split_counts"],
                }
            )

    index_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "datasets": [str(dataset) for dataset in args.datasets],
        "kir_values": [float(value) for value in args.kir_values],
        "runs": summary,
    }
    index_path = Path(args.index_out) if args.index_out else output_root / "index.json"
    _write_json(index_path, index_payload)
    LOGGER.info("Wrote index -> %s", index_path)


if __name__ == "__main__":
    main()
