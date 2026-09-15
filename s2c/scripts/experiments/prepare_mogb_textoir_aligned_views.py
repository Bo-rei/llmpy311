"""Adapt canonical TextOIR exports to the pinned MOGB loader contract.

The adapter changes only field names and the train/dev/test directory layout.
The source rows, labels, Known intent lists and sample IDs remain those of the
canonical ``protocol_v2_textoir_v1/k_plus_1_way`` exports.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = ROOT / "data" / "exports" / "protocol_v2_textoir_v1" / "k_plus_1_way"
OUT = ROOT.parent / "artifacts" / "s2c" / "analysis" / "mogb_shared_official_aligned" / "data"
DATASETS = ("banking77", "stackoverflow", "clinc150")
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tag(kir: float, seed: int) -> str:
    return f"kir{round(kir * 100):02d}_seed{seed}"


def adapt_row(row: dict, known: set[str], split: str, domain: str) -> dict:
    intent = str(row.get("original_intent", row["label"]))
    text = str(row["text"])
    sample_id = row.get("sample_id")
    is_oos = str(row["label"]) == "__oos__" or intent not in known
    if split in ("train", "val") and is_oos:
        raise ValueError(f"OOS row leaked into {split}: {row}")
    result = {
        "text": text,
        "intent": intent,
        "domain": domain,
        "label": int(is_oos),
    }
    if sample_id is not None:
        result["sample_id"] = str(sample_id)
    return result


def main() -> None:
    if OUT.exists():
        manifest_path = OUT.parent / "MANIFEST.json"
        if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("status") == "complete":
            print("MOGB_TEXTOIR_ALIGNED_VIEWS_ALREADY_PREPARED")
            return
        raise FileExistsError(f"refusing to overwrite incomplete output: {OUT}")

    units = []
    source_hashes = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            for kir in KIRS:
                source = EXPORT_ROOT / dataset / f"seed_{seed}" / f"kir_{kir:.2f}"
                source_manifest = source / "export_manifest.json"
                known_path = source / "known_labels.json"
                if not source_manifest.is_file() or not known_path.is_file():
                    raise FileNotFoundError(source)
                known_list = json.loads(known_path.read_text(encoding="utf-8"))
                known = {str(label) for label in known_list}
                target = OUT / dataset / tag(kir, seed)
                for source_split, target_split in (("train", "train"), ("dev", "val"), ("test", "test")):
                    rows = json.loads((source / f"{source_split}.json").read_text(encoding="utf-8"))
                    adapted = [adapt_row(row, known, target_split, dataset) for row in rows]
                    if target_split in ("train", "val"):
                        assert all(row["label"] == 0 for row in adapted)
                    write_json(target / "gate" / f"{target_split}.json", adapted)
                write_json(target / "known_labels.json", known_list)
                write_json(target / "source_export_manifest.json", json.loads(source_manifest.read_text(encoding="utf-8")))
                test_rows = json.loads((target / "gate" / "test.json").read_text(encoding="utf-8"))
                train_rows = json.loads((target / "gate" / "train.json").read_text(encoding="utf-8"))
                val_rows = json.loads((target / "gate" / "val.json").read_text(encoding="utf-8"))
                unit = {
                    "dataset": dataset,
                    "dataset_variant": "standard_protocol_export",
                    "protocol": "protocol_v2_textoir_v1",
                    "kir": kir,
                    "seed": seed,
                    "source_export": str(source),
                    "source_export_manifest": str(source_manifest),
                    "source_export_manifest_sha256": sha256(source_manifest),
                    "known_labels": str(target / "known_labels.json"),
                    "known_count": len(known_list),
                    "counts": {
                        "train": len(train_rows),
                        "val": len(val_rows),
                        "test": len(test_rows),
                    },
                    "test_oos_count": sum(row["label"] for row in test_rows),
                    "path": str(target),
                    "real_oos_training": False,
                    "real_oos_validation": False,
                    "pseudo_oos_used": False,
                    "test_used_for_selection": False,
                }
                units.append(unit)
                print(f"PREPARED {dataset}/{tag(kir, seed)}", flush=True)

    manifest_path = OUT.parent / "MANIFEST.json"
    manifest = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "mogb_official_compatible_textoir_aligned_adapter",
        "protocol": "protocol_v2_textoir_v1",
        "source_root": str(EXPORT_ROOT),
        "output_root": str(OUT),
        "datasets": list(DATASETS),
        "kirs": list(KIRS),
        "seeds": list(SEEDS),
        "planned_units": len(DATASETS) * len(KIRS) * len(SEEDS),
        "completed_units": len(units),
        "units": units,
        "real_oos_training": False,
        "real_oos_validation": False,
        "pseudo_oos_used": False,
        "test_used_for_selection": False,
        "adaptation": "field/layout adapter only; canonical rows and intent split preserved",
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "units": len(units)}, indent=2))


if __name__ == "__main__":
    main()
