"""Verify the 45-cell fixed KNN full-protocol baseline."""

from __future__ import annotations

import csv
import json

from protocol_v2.runtime.paths import ProtocolV2Paths


def main() -> int:
    paths = ProtocolV2Paths.discover()
    root = paths.run_root / "clmsg_v1" / "summary" / "knn_pareto_v1"
    integrity = json.loads((root / "integrity.json").read_text(encoding="utf-8"))
    with (root / "all_runs.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (root / "paired_effects.csv").open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    with (root / "significance.csv").open(newline="", encoding="utf-8") as handle:
        significance = list(csv.DictReader(handle))
    cells = {(row["dataset"], row["kir"], row["seed"]) for row in rows}
    if integrity.get("status") != "complete" or integrity.get("test_used_for_selection") is not False:
        raise ValueError("KNN Pareto integrity contract failed")
    if len(rows) != 45 or len(cells) != 45:
        raise ValueError("KNN Pareto cell coverage is not 45/45 unique")
    if len(paired) != 45 * 4 or len(significance) != 3 * 3 * 4:
        raise ValueError("KNN Pareto paired/statistical coverage mismatch")
    if any(row["method"] != "knn_only" for row in rows):
        raise ValueError("KNN Pareto summary contains a non-KNN method")
    result = {
        "status": "ok",
        "cells": len(rows),
        "unique_cells": len(cells),
        "paired_rows": len(paired),
        "significance_rows": len(significance),
        "fresh_cells": integrity["fresh_cells"],
        "reused_cells": integrity["reused_cells"],
        "test_used_for_selection": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
