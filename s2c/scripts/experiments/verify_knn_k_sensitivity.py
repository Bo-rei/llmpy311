"""Verify the complete four-k KNN sensitivity closeout."""

from __future__ import annotations

import csv
import json

from protocol_v2.runtime.paths import ProtocolV2Paths


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    paths = ProtocolV2Paths.discover()
    root = paths.run_root / "clmsg_v1" / "summary" / "knn_k_sensitivity_v1"
    integrity = json.loads((root / "integrity.json").read_text(encoding="utf-8"))
    rows = _read_csv(root / "all_runs.csv")
    paired_single = _read_csv(root / "paired_vs_single.csv")
    paired_k10 = _read_csv(root / "paired_vs_k10.csv")
    effects = _read_csv(root / "paired_effects.csv")
    cells = {(row["dataset"], row["kir"], row["seed"], row["k_neighbors"]) for row in rows}
    if integrity.get("status") != "complete" or integrity.get("test_used_for_selection") is not False:
        raise ValueError("KNN k-sensitivity integrity contract failed")
    if len(rows) != 180 or len(cells) != 180:
        raise ValueError("KNN k-sensitivity coverage is not 180/180 unique")
    if len(paired_single) != 180 or len(paired_k10) != 135 or len(effects) != 63:
        raise ValueError("KNN k-sensitivity paired/statistical coverage mismatch")
    if {int(row["k_neighbors"]) for row in rows} != {5, 10, 20, 30}:
        raise ValueError("KNN k-sensitivity has unexpected neighbour counts")
    result = {
        "status": "ok",
        "cells": len(rows),
        "unique_cells": len(cells),
        "new_k_cells": integrity["new_k_cells"],
        "existing_k10_cells": integrity["existing_k10_cells"],
        "reused_k10_cells": integrity["reused_k10_cells"],
        "paired_vs_single_rows": len(paired_single),
        "paired_vs_k10_rows": len(paired_k10),
        "effect_rows": len(effects),
        "test_used_for_selection": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
