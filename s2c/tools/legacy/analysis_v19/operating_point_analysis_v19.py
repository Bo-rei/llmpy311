from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def rank_operating_points(
    result_paths: Iterable[Path | str],
    *,
    min_gate_id_recall: Optional[float] = None,
    min_known_intent_accuracy: Optional[float] = None,
    sort_by: str = "oos_f1",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_path in result_paths:
        path = Path(raw_path)
        payload = _load_json(path)
        config = payload.get("config", {})
        metrics = payload.get("metrics", {})
        row = {
            "path": str(path),
            "low": config.get("semantic_uncertain_low"),
            "high": config.get("semantic_uncertain_high"),
            "threshold": config.get("semantic_gate_threshold"),
            "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
            "known_accuracy": float(metrics.get("known_accuracy", metrics.get("known_intent_accuracy", 0.0))),
            "oos_accuracy": float(metrics.get("oos_accuracy", metrics.get("gate_oos_rejection", 0.0))),
            "known_intent_accuracy": float(metrics.get("known_intent_accuracy", 0.0)),
            "known_macro_f1": float(metrics.get("known_macro_f1", 0.0)),
            "gate_id_recall": float(metrics.get("gate_id_recall", 0.0)),
            "gate_oos_rejection": float(metrics.get("gate_oos_rejection", 0.0)),
            "oos_f1": float(metrics.get("oos_f1", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
        }
        if min_gate_id_recall is not None and row["gate_id_recall"] < float(min_gate_id_recall):
            continue
        if min_known_intent_accuracy is not None and row["known_intent_accuracy"] < float(
            min_known_intent_accuracy
        ):
            continue
        rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row.get(sort_by, 0.0)),
            float(row.get("gate_oos_rejection", 0.0)),
            float(row.get("known_intent_accuracy", 0.0)),
        ),
        reverse=True,
    )
    return rows
