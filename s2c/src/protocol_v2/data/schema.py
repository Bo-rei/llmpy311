"""Protocol_v2 constants and small typed data contracts."""

from __future__ import annotations

from dataclasses import dataclass


CANONICAL_SCHEMA_VERSION = "protocol_v2.canonical.v1"
REGISTRY_SCHEMA_VERSION = "protocol_v2.registry.v1"
VIEW_SCHEMA_VERSION = "protocol_v2.views.v1"
EXPORT_SCHEMA_VERSION = "protocol_v2.exports.v1"
PROTOCOL_VERSION = "protocol_v2"
NATIVE_OOS_LABEL = "oos"

FORMAL_KIRS: tuple[float, ...] = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90)
FORMAL_SEEDS: tuple[int, ...] = tuple(range(10))
COMPATIBILITY_SEEDS: tuple[int, ...] = (13, 42, 87)
ALL_REGISTRY_SEEDS: tuple[int, ...] = FORMAL_SEEDS + COMPATIBILITY_SEEDS


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    textoir_directory: str
    source_splits: tuple[str, ...] = ("train", "dev", "test")
    native_oos_label: str | None = None
    public_data_tracking: bool = True


DATASET_SPECS: dict[str, DatasetSpec] = {
    "clinc150": DatasetSpec("clinc150", "oos", native_oos_label=NATIVE_OOS_LABEL),
    "banking77": DatasetSpec("banking77", "banking"),
    "stackoverflow": DatasetSpec("stackoverflow", "stackoverflow", public_data_tracking=False),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset]
    except KeyError as exc:
        allowed = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown protocol_v2 dataset {dataset!r}; expected one of: {allowed}") from exc


def format_kir(kir: float) -> str:
    return f"{kir:.2f}"

