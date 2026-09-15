"""Materialize the canonical TextOIR-aligned standard Banking77 views.

The source exports already contain the frozen TextOIR benchmark-label order,
NumPy RandomState Known-intent selection, and the canonical train/dev/test
sample mapping.  This script only adapts their schema for the legacy
Gate->Router->Expert evaluator; it does not select a model or read test rows
for selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

SOURCE_ROOT = ROOT / "data" / "exports" / "protocol_v2_textoir_v1" / "k_plus_1_way" / "banking77"
ART_ROOT = ROOT.parent / "artifacts" / "s2c" / "analysis" / "banking_textoir_aligned"
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def tag(kir: float, seed: int) -> str:
    return f"kir{round(kir * 100):02d}_seed{seed}"


def adapt_gate_rows(rows: list[dict], split: str) -> list[dict]:
    result = []
    for row in rows:
        # The canonical export keeps the original intent in original_intent;
        # train/dev labels are Known intents, while test label==__oos__ marks
        # held-out-intent OOS.
        intent = str(row.get("original_intent", row["label"]))
        is_oos = split == "test" and str(row["label"]) == "__oos__"
        result.append(
            {
                "text": str(row["text"]),
                "intent": intent,
                "domain": "banking",
                "label": int(is_oos),
                "sample_id": str(row["sample_id"]),
            }
        )
    return result


def build_downstream(folder: Path, known: list[str]) -> None:
    train = list(read(folder / "gate" / "train.json"))
    validation = list(read(folder / "gate" / "val.json"))
    known_set = set(known)
    if not train or not validation:
        raise ValueError(f"empty Known split: {folder}")
    if any(row["intent"] not in known_set or int(row["label"]) != 0 for row in train + validation):
        raise ValueError(f"non-Known row in training/validation: {folder}")

    domains = sorted({row["domain"] for row in train})
    domain_map = {domain: index for index, domain in enumerate(domains)}
    dump(folder / "router" / "domain_map.json", domain_map)
    for domain in domains:
        intents = sorted({row["intent"] for row in train if row["domain"] == domain})
        intent_map = {intent: index for index, intent in enumerate(intents)}
        dump(folder / "experts" / domain / "intent_map.json", intent_map)
        for split in ("train", "val"):
            rows = list(read(folder / "gate" / f"{split}.json"))
            dump(
                folder / "experts" / domain / f"{split}.json",
                [{**row, "label": intent_map[row["intent"]]} for row in rows if row["domain"] == domain],
            )
    for split in ("train", "val"):
        rows = list(read(folder / "gate" / f"{split}.json"))
        dump(
            folder / "router" / f"{split}.json",
            [{**row, "label": domain_map[row["domain"]]} for row in rows],
        )


def build_cell(kir: float, seed: int, output_root: Path) -> dict:
    source = SOURCE_ROOT / f"seed_{seed}" / f"kir_{kir:.2f}"
    if not source.is_dir():
        raise FileNotFoundError(source)
    target = output_root / "data" / "banking77" / tag(kir, seed)
    target.mkdir(parents=True, exist_ok=True)

    known = list(read(source / "known_labels.json"))
    if len(known) != round(77 * kir) or len(set(known)) != len(known):
        raise ValueError(f"invalid Known list for {source}")
    dump(target / "known_labels.json", known)

    counts = {}
    oos_counts = {}
    for source_split, target_split in (("train", "train"), ("dev", "val"), ("test", "test")):
        source_rows = list(read(source / f"{source_split}.json"))
        rows = adapt_gate_rows(source_rows, source_split)
        if target_split != "test" and any(row["label"] != 0 for row in rows):
            raise ValueError(f"OOS supervision leaked into {target_split}: {source}")
        dump(target / "gate" / f"{target_split}.json", rows)
        counts[target_split] = len(rows)
        oos_counts[target_split] = sum(int(row["label"]) for row in rows)

    # Copy the exact canonical list and provenance, rather than regenerating
    # labels from a second RNG implementation.
    shutil.copy2(source / "label_map.json", target / "label_map.json")
    shutil.copy2(source / "sample_ids.json", target / "sample_ids.json")
    build_downstream(target, [str(item) for item in known])
    source_manifest = dict(read(source / "export_manifest.json"))
    manifest = {
        "dataset": "banking77",
        "dataset_variant": "standard_77_intent",
        "protocol": "protocol_v2_textoir_v1",
        "source_export": str(source),
        "source_export_manifest": str(source / "export_manifest.json"),
        "source_export_manifest_fields": {
            "canonical_manifest_sha256": source_manifest.get("canonical_manifest_sha256"),
            "canonical_sample_id_mapping_sha256": source_manifest.get("canonical_sample_id_mapping_sha256"),
            "protocol_version": source_manifest.get("protocol_version"),
            "seed": source_manifest.get("seed"),
            "kir": source_manifest.get("kir"),
        },
        "seed": seed,
        "kir": kir,
        "known_count": len(known),
        "actual_kir": len(known) / 77.0,
        "known_labels_file": str(target / "known_labels.json"),
        "intent_selection": "TextOIR benchmark_labels[banking] order + NumPy RandomState(seed).choice(replace=False)",
        "train_validation_supervision": "Known intents only",
        "test_oos_source": "held-out intents in canonical standard Banking77 test split",
        "counts": counts,
        "oos_counts": oos_counts,
        "real_oos_used_for_training": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_materialized": True,
        "test_used_for_selection": False,
    }
    dump(target / "view_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ART_ROOT)
    parser.add_argument("--kirs", nargs="+", type=float, default=list(KIRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    manifests = [build_cell(kir, seed, output_root) for kir in args.kirs for seed in args.seeds]
    dump(
        output_root / "MANIFEST.json",
        {
            "status": "data_prepared",
            "experiment": "banking77_textoir_aligned_known_only",
            "protocol": "protocol_v2_textoir_v1",
            "source_root": str(SOURCE_ROOT),
            "dataset": "banking77",
            "dataset_variant": "standard_77_intent",
            "kirs": list(args.kirs),
            "seeds": list(args.seeds),
            "units": manifests,
            "real_oos_used_for_training": False,
            "real_oos_used_for_selection": False,
            "pseudo_oos_used": False,
            "test_used_for_selection": False,
        },
    )
    print(f"Prepared {len(manifests)} TextOIR-aligned standard Banking77 views", flush=True)


if __name__ == "__main__":
    main()
