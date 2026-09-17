"""Materialize a canonical TextOIR export for the legacy cascade evaluator.

The source exports are already fixed by the protocol-v2 registry and use the
TextOIR benchmark label order plus NumPy RandomState intent selection.  This
adapter only supplies the legacy Gate/Router/Expert JSON schema; it does not
select a model and never reads test rows during selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "data" / "exports" / "protocol_v2_textoir_v1" / "k_plus_1_way"
DEFAULT_OUTPUT_ROOT = ROOT.parent / "artifacts" / "s2c" / "analysis" / "kir_sensitivity_known_only"
KIRS = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90)
SEEDS = (13, 42, 87)
INTENT_COUNTS = {"clinc150": 150, "stackoverflow": 20, "banking77": 77}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def tag(kir: float, seed: int) -> str:
    return f"kir{round(kir * 100):02d}_seed{seed}"


def adapt_rows(rows: list[dict], split: str, domain: str) -> list[dict]:
    result = []
    for row in rows:
        original = str(row.get("original_intent", row["label"]))
        is_oos = split == "test" and str(row["label"]) == "__oos__"
        result.append(
            {
                "text": str(row["text"]),
                "intent": original,
                "domain": domain,
                "label": int(is_oos),
                "sample_id": str(row["sample_id"]),
            }
        )
    return result


def build_downstream(folder: Path, known: list[str]) -> None:
    train = read(folder / "gate" / "train.json")
    validation = read(folder / "gate" / "val.json")
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
            rows = read(folder / "gate" / f"{split}.json")
            dump(
                folder / "experts" / domain / f"{split}.json",
                [{**row, "label": intent_map[row["intent"]]} for row in rows if row["domain"] == domain],
            )
    for split in ("train", "val"):
        rows = read(folder / "gate" / f"{split}.json")
        dump(folder / "router" / f"{split}.json", [{**row, "label": domain_map[row["domain"]]} for row in rows])


def build_cell(dataset: str, kir: float, seed: int, output_root: Path) -> dict:
    source = SOURCE_ROOT / dataset / f"seed_{seed}" / f"kir_{kir:.2f}"
    if not source.is_dir():
        raise FileNotFoundError(source)
    target = output_root / "data" / dataset / tag(kir, seed)
    target.mkdir(parents=True, exist_ok=True)
    known = list(read(source / "known_labels.json"))
    if len(set(known)) != len(known):
        raise ValueError(f"duplicate Known labels: {source}")

    counts = {}
    oos_counts = {}
    domain = dataset
    for source_split, target_split in (("train", "train"), ("dev", "val"), ("test", "test")):
        rows = adapt_rows(read(source / f"{source_split}.json"), source_split, domain)
        if target_split != "test" and any(row["label"] != 0 for row in rows):
            raise ValueError(f"OOS supervision leaked into {source}")
        dump(target / "gate" / f"{target_split}.json", rows)
        counts[target_split] = len(rows)
        oos_counts[target_split] = sum(int(row["label"]) for row in rows)
    build_downstream(target, known)
    for name in ("label_map.json", "sample_ids.json"):
        shutil.copy2(source / name, target / name)
    source_manifest = read(source / "export_manifest.json")
    manifest = {
        "dataset": dataset,
        "dataset_variant": "canonical_textoir_export",
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
        "known_labels_file": str(target / "known_labels.json"),
        "known_count": len(known),
        "actual_kir": len(known) / float(INTENT_COUNTS[dataset]),
        "intent_selection": "canonical TextOIR export; benchmark label order + NumPy RandomState",
        "train_validation_supervision": "Known intents only",
        "test_oos_source": "held-out intents and native OOS mapped to __oos__ in canonical export",
        "counts": counts,
        "oos_counts": oos_counts,
        "real_oos_used_for_training": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "test_materialized": True,
        "test_used_for_selection": False,
    }
    dump(target / "known_labels.json", known)
    dump(target / "view_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("clinc150", "stackoverflow", "banking77"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kirs", nargs="+", type=float, default=list(KIRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    manifests = [build_cell(args.dataset, kir, seed, output_root) for kir in args.kirs for seed in args.seeds]
    dump(
        output_root / f"{args.dataset}_textoir_aligned_manifest.json",
        {
            "status": "data_prepared",
            "experiment": "kir_sensitivity_textoir_aligned_known_only",
            "dataset": args.dataset,
            "protocol": "protocol_v2_textoir_v1",
            "source_root": str(SOURCE_ROOT / args.dataset),
            "kirs": list(args.kirs),
            "seeds": list(args.seeds),
            "units": manifests,
            "real_oos_used_for_training": False,
            "real_oos_used_for_selection": False,
            "pseudo_oos_used": False,
            "test_used_for_selection": False,
        },
    )
    print(f"Prepared {len(manifests)} canonical {args.dataset} views", flush=True)


if __name__ == "__main__":
    main()
