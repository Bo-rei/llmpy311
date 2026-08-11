#!/usr/bin/env python3
"""Audit v19 ablation outputs from per-sample prediction files.

This script deliberately treats prediction files as the metric source of truth.
It normalizes rows across current v19 evaluators, recomputes paper metrics, and
checks whether Cascade-MiniLM and Cascade-SmolLM share OOS decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

OOS_LABELS = {"__oos__", "oos", "unknown", "heldout_oos"}

DATASET_NAMES = {
    "clinc150": "CLINC150",
    "banking77_oos": "BANKING77-OOS",
    "stackoverflow": "STACKOVERFLOW",
}

PAPER_VARIANTS = {
    "full_anchor": "Full Pipeline",
    "full_pipeline": "Full Pipeline",
    "wo_gate": "w/o Gate",
    "wo_gate_confidence": "w/o Gate",
    "banking_wo_geometric_gate_expert_confidence": "w/o Gate",
    "cascade_minilm": "Cascade-MiniLM",
    "cascade_smollm": "Cascade-SmolLM",
}


@dataclass(frozen=True)
class NormalizedPrediction:
    dataset: str
    variant: str
    sample_id: str
    text: str | None
    gold_label: str
    pred_label: str
    gold_is_oos: bool
    pred_is_oos: bool
    gate_score: float | None
    gate_decision: str | None
    router_pred: str | None
    expert_pred: str | None
    confidence_score: float | None
    threshold: float | None
    gate_model_path: str | None
    router_model_path: str | None
    expert_model_path: str | None
    output_file_source: str

    def with_variant(self, **updates: Any) -> "NormalizedPrediction":
        return replace(self, **updates)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _is_oos_label(value: Any) -> bool:
    return str(value).strip().lower() in OOS_LABELS


def _manifest_for(prediction_path: Path) -> Dict[str, Any]:
    manifest_path = prediction_path.parent / "run_manifest.json"
    if manifest_path.exists():
        return dict(_load_json(manifest_path))
    anchor_path = prediction_path.parent / "anchor_metadata.json"
    if anchor_path.exists():
        return dict(_load_json(anchor_path))
    return {}


def _eval_for(prediction_path: Path) -> Dict[str, Any]:
    eval_path = prediction_path.parent / "eval_results.json"
    if eval_path.exists():
        return dict(_load_json(eval_path))
    return {}


def normalize_predictions(
    *,
    dataset: str,
    variant: str,
    prediction_path: Path,
) -> List[NormalizedPrediction]:
    rows = list(_load_json(prediction_path))
    manifest = _manifest_for(prediction_path)
    eval_payload = _eval_for(prediction_path)
    config = dict(eval_payload.get("config", {}))
    threshold = _float_or_none(
        manifest.get("selected_threshold")
        or manifest.get("router_confidence_threshold")
        or config.get("router_confidence_threshold")
        or eval_payload.get("cascade_smollm", {}).get("selected_threshold")
    )
    gate_model_path = (
        manifest.get("gate_model_path")
        or manifest.get("gate_detector_path")
        or config.get("gate_detector_path")
        or config.get("gate_encoder_path")
    )
    router_model_path = manifest.get("router_ckpt") or config.get("router_ckpt")
    expert_model_path = manifest.get("experts_root") or config.get("experts_root")

    normalized: List[NormalizedPrediction] = []
    for idx, row in enumerate(rows):
        gold_label = str(row.get("true_intent", row.get("intent_gold", row.get("gold_label", row.get("intent", "")))))
        gold_is_oos = bool(int(row.get("true_gate_label", row.get("label", 1 if _is_oos_label(gold_label) else 0))) == 1)
        if gold_is_oos:
            gold_label = "__oos__"
        pred_is_oos = bool(row.get("is_oos", row.get("pred_is_oos", False)))
        pred_label = "__oos__" if pred_is_oos else str(row.get("intent", row.get("pred_label", "")))
        normalized.append(
            NormalizedPrediction(
                dataset=dataset,
                variant=variant,
                sample_id=str(row.get("sample_id", row.get("id", idx))),
                text=row.get("text"),
                gold_label=gold_label,
                pred_label=pred_label,
                gold_is_oos=gold_is_oos,
                pred_is_oos=pred_is_oos,
                gate_score=_float_or_none(row.get("gate_score")),
                gate_decision=row.get("final_gate_decision") or ("oos" if pred_is_oos else "id"),
                router_pred=row.get("domain"),
                expert_pred=None if pred_is_oos else row.get("intent"),
                confidence_score=_float_or_none(row.get("intent_prob", row.get("no_gate_confidence"))),
                threshold=threshold,
                gate_model_path=None if gate_model_path is None else str(gate_model_path),
                router_model_path=None if router_model_path is None else str(router_model_path),
                expert_model_path=None if expert_model_path is None else str(expert_model_path),
                output_file_source=str(prediction_path),
            )
        )
    return normalized


def recompute_metrics(rows: Sequence[NormalizedPrediction]) -> tuple[Dict[str, float], Dict[str, int | float]]:
    gold_labels = [row.gold_label for row in rows]
    pred_labels = [row.pred_label for row in rows]
    gold_oos = np.asarray([int(row.gold_is_oos) for row in rows], dtype=np.int64)
    pred_oos = np.asarray([int(row.pred_is_oos) for row in rows], dtype=np.int64)

    known_labels = sorted({row.gold_label for row in rows if not row.gold_is_oos})
    known_true = [row.gold_label for row in rows if not row.gold_is_oos]
    known_pred = [row.pred_label for row in rows if not row.gold_is_oos]
    known_f1 = (
        float(f1_score(known_true, known_pred, labels=known_labels, average="macro", zero_division=0))
        if known_true
        else 0.0
    )
    precision, recall, oos_f1, _ = precision_recall_fscore_support(
        gold_oos,
        pred_oos,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    tp = int(np.sum((gold_oos == 1) & (pred_oos == 1)))
    fp = int(np.sum((gold_oos == 0) & (pred_oos == 1)))
    fn = int(np.sum((gold_oos == 1) & (pred_oos == 0)))
    tn = int(np.sum((gold_oos == 0) & (pred_oos == 0)))
    acc = float(np.mean([gold == pred for gold, pred in zip(gold_labels, pred_labels)])) if rows else 0.0
    return (
        {
            "known_f1": known_f1,
            "oos_f1": float(oos_f1),
            "acc": acc,
            "precision_oos": float(precision),
            "recall_oos": float(recall),
        },
        {
            "tp_oos": tp,
            "fp_oos": fp,
            "fn_oos": fn,
            "tn_oos": tn,
            "precision_oos": float(precision),
            "recall_oos": float(recall),
        },
    )


def compare_prediction_overlap(
    minilm_rows: Sequence[NormalizedPrediction],
    smollm_rows: Sequence[NormalizedPrediction],
) -> Dict[str, Any]:
    if len(minilm_rows) != len(smollm_rows):
        raise ValueError("Prediction files have different lengths")
    n = len(minilm_rows)
    same_binary = sum(a.pred_is_oos == b.pred_is_oos for a, b in zip(minilm_rows, smollm_rows))
    same_label = sum(a.pred_label == b.pred_label for a, b in zip(minilm_rows, smollm_rows))
    minilm_metrics, minilm_counts = recompute_metrics(minilm_rows)
    smollm_metrics, smollm_counts = recompute_metrics(smollm_rows)
    left_scores = [row.gate_score for row in minilm_rows]
    right_scores = [row.gate_score for row in smollm_rows]
    score_pairs = [
        (float(left), float(right))
        for left, right in zip(left_scores, right_scores)
        if left is not None and right is not None
    ]
    score_correlation = None
    if len(score_pairs) > 1:
        score_correlation = float(np.corrcoef([p[0] for p in score_pairs], [p[1] for p in score_pairs])[0, 1])
    return {
        "same_binary_rate": float(same_binary / n) if n else 0.0,
        "same_pred_label_rate": float(same_label / n) if n else 0.0,
        "gate_score_correlation": score_correlation,
        "minilm_oos_f1_raw": minilm_metrics["oos_f1"],
        "smollm_oos_f1_raw": smollm_metrics["oos_f1"],
        "raw_oos_f1_identical": bool(np.isclose(minilm_metrics["oos_f1"], smollm_metrics["oos_f1"])),
        "minilm_tp_oos": minilm_counts["tp_oos"],
        "minilm_fp_oos": minilm_counts["fp_oos"],
        "minilm_fn_oos": minilm_counts["fn_oos"],
        "smollm_tp_oos": smollm_counts["tp_oos"],
        "smollm_fp_oos": smollm_counts["fp_oos"],
        "smollm_fn_oos": smollm_counts["fn_oos"],
        "tp_fp_fn_identical": (
            minilm_counts["tp_oos"] == smollm_counts["tp_oos"]
            and minilm_counts["fp_oos"] == smollm_counts["fp_oos"]
            and minilm_counts["fn_oos"] == smollm_counts["fn_oos"]
        ),
        "minilm_gate_model_path": _single_value(row.gate_model_path for row in minilm_rows),
        "smollm_gate_model_path": _single_value(row.gate_model_path for row in smollm_rows),
    }


def _single_value(values: Iterable[str | None]) -> str | None:
    unique = sorted({str(value) for value in values if value})
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return "MULTIPLE: " + " | ".join(unique[:4])
    return None


def _prediction_path(root: Path, slug: str, kir_tag: str, variant: str) -> Path | None:
    candidates = [
        root / slug / kir_tag / variant / "predictions.json",
        root / slug / kir_tag / _legacy_variant_dir(variant) / "predictions.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _legacy_variant_dir(variant: str) -> str:
    if variant == "wo_gate_confidence":
        return "wo_gate"
    if variant == "full_pipeline":
        return "full_anchor"
    return variant


def _discover_kir_tags(root: Path, slug: str, kir_tags: Sequence[str] | None) -> List[str]:
    if kir_tags:
        return list(kir_tags)
    discovered = sorted(path.name for path in (root / slug).glob("kir*_seed*") if path.is_dir())
    return discovered or ["kir50_seed42"]


def discover_prediction_groups(root: Path, kir_tags: Sequence[str] | None = None) -> Dict[tuple[str, str, str], Path]:
    groups: Dict[tuple[str, str, str], Path] = {}
    for slug, dataset in DATASET_NAMES.items():
        variants = ["full_anchor", "wo_gate_confidence", "cascade_minilm", "cascade_smollm"]
        if slug == "banking77_oos":
            variants = [
                "full_anchor",
                "banking_wo_geometric_gate_expert_confidence",
                "cascade_minilm",
                "cascade_smollm",
            ]
        for kir_tag in _discover_kir_tags(root, slug, kir_tags):
            for variant in variants:
                path = _prediction_path(root, slug, kir_tag, variant)
                if path is not None:
                    groups[(dataset, kir_tag, PAPER_VARIANTS[variant])] = path
    return groups


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}"


def write_reports(root: Path, output_dir: Path, kir_tags: Sequence[str] | None = None) -> Dict[str, Path]:
    groups = discover_prediction_groups(root, kir_tags=kir_tags)
    all_rows: Dict[tuple[str, str, str], List[NormalizedPrediction]] = {
        key: normalize_predictions(dataset=key[0], variant=key[2], prediction_path=path)
        for key, path in groups.items()
    }

    metrics_rows: List[Dict[str, Any]] = []
    counts_rows: List[Dict[str, Any]] = []
    for (dataset, kir_tag, variant), rows in sorted(all_rows.items()):
        metrics, counts = recompute_metrics(rows)
        metrics_rows.append(
            {
                "dataset": dataset,
                "kir_tag": kir_tag,
                "variant": variant,
                "known_f1_raw": metrics["known_f1"],
                "oos_f1_raw": metrics["oos_f1"],
                "acc_raw": metrics["acc"],
                "precision_oos_raw": metrics["precision_oos"],
                "recall_oos_raw": metrics["recall_oos"],
                "known_f1_pct_2dp": _fmt_pct(metrics["known_f1"]),
                "oos_f1_pct_2dp": _fmt_pct(metrics["oos_f1"]),
                "acc_pct_2dp": _fmt_pct(metrics["acc"]),
            }
        )
        counts_rows.append({"dataset": dataset, "kir_tag": kir_tag, "variant": variant, **counts})

    overlap_rows: List[Dict[str, Any]] = []
    for dataset in sorted(DATASET_NAMES.values()):
        dataset_kirs = sorted({key[1] for key in all_rows if key[0] == dataset})
        for kir_tag in dataset_kirs:
            left = all_rows.get((dataset, kir_tag, "Cascade-MiniLM"))
            right = all_rows.get((dataset, kir_tag, "Cascade-SmolLM"))
            if left is not None and right is not None:
                overlap_rows.append({"dataset": dataset, "kir_tag": kir_tag, **compare_prediction_overlap(left, right)})

    _write_csv(
        output_dir / "ablation_metrics_recomputed.csv",
        metrics_rows,
        [
            "dataset",
            "kir_tag",
            "variant",
            "known_f1_raw",
            "oos_f1_raw",
            "acc_raw",
            "precision_oos_raw",
            "recall_oos_raw",
            "known_f1_pct_2dp",
            "oos_f1_pct_2dp",
            "acc_pct_2dp",
        ],
    )
    _write_csv(
        output_dir / "ablation_oos_confusion_counts.csv",
        counts_rows,
        ["dataset", "kir_tag", "variant", "tp_oos", "fp_oos", "fn_oos", "tn_oos", "precision_oos", "recall_oos"],
    )
    _write_normalized_predictions(output_dir, all_rows)
    _write_variant_audit(output_dir, groups, all_rows)
    _write_overlap_report(output_dir, overlap_rows)
    _write_final_report(output_dir, metrics_rows, overlap_rows)
    _write_latex_section(output_dir, metrics_rows, overlap_rows)
    return {
        "variant_audit": output_dir / "ablation_variant_audit.md",
        "metrics": output_dir / "ablation_metrics_recomputed.csv",
        "counts": output_dir / "ablation_oos_confusion_counts.csv",
        "overlap": output_dir / "ablation_prediction_overlap_report.md",
        "final_report": output_dir / "final_ablation_audit_report.md",
        "latex": output_dir / "updated_ablation_section.tex",
    }


def _write_normalized_predictions(
    output_dir: Path,
    all_rows: Dict[tuple[str, str, str], List[NormalizedPrediction]],
) -> None:
    pred_dir = output_dir / "normalized_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for (dataset, kir_tag, variant), rows in all_rows.items():
        slug = f"{dataset}_{kir_tag}_{variant}".replace("/", "_").replace(" ", "_").replace("-", "_")
        with (pred_dir / f"{slug}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _write_variant_audit(
    output_dir: Path,
    groups: Dict[tuple[str, str, str], Path],
    all_rows: Dict[tuple[str, str, str], List[NormalizedPrediction]],
) -> None:
    lines = ["# Ablation Variant Audit", "", "All paths below are resolved from prediction-file metadata.", ""]
    for key, path in sorted(groups.items()):
        rows = all_rows[key]
        first = rows[0] if rows else None
        lines.extend(
            [
                f"## {key[0]} / {key[1]} / {key[2]}",
                f"- kir_tag: `{key[1]}`",
                f"- variant: `{key[2]}`",
                f"- prediction file: `{path}`",
                f"- gate model/path: `{first.gate_model_path if first else None}`",
                f"- router model/path: `{first.router_model_path if first else None}`",
                f"- expert model/path: `{first.expert_model_path if first else None}`",
                f"- threshold: `{first.threshold if first else None}`",
                f"- sample count: `{len(rows)}`",
                "",
            ]
        )
    (output_dir / "ablation_variant_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _write_overlap_report(output_dir: Path, overlap_rows: Sequence[Dict[str, Any]]) -> None:
    lines = ["# Ablation Prediction Overlap Report", ""]
    for row in overlap_rows:
        lines.extend(
            [
                f"## {row['dataset']} / {row['kir_tag']}",
                f"- same_binary_rate: `{row['same_binary_rate']:.6f}`",
                f"- same_pred_label_rate: `{row['same_pred_label_rate']:.6f}`",
                f"- gate_score_correlation: `{row['gate_score_correlation']}`",
                f"- Cascade-MiniLM OOS F1 raw: `{row['minilm_oos_f1_raw']:.10f}`",
                f"- Cascade-SmolLM OOS F1 raw: `{row['smollm_oos_f1_raw']:.10f}`",
                f"- TP/FP/FN identical: `{row['tp_fp_fn_identical']}`",
                f"- MiniLM TP/FP/FN: `{row['minilm_tp_oos']}/{row['minilm_fp_oos']}/{row['minilm_fn_oos']}`",
                f"- SmolLM TP/FP/FN: `{row['smollm_tp_oos']}/{row['smollm_fp_oos']}/{row['smollm_fn_oos']}`",
                f"- MiniLM gate path: `{row['minilm_gate_model_path']}`",
                f"- SmolLM gate path: `{row['smollm_gate_model_path']}`",
                "",
            ]
        )
    (output_dir / "ablation_prediction_overlap_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_final_report(
    output_dir: Path,
    metrics_rows: Sequence[Dict[str, Any]],
    overlap_rows: Sequence[Dict[str, Any]],
) -> None:
    identical = [row for row in overlap_rows if row["same_binary_rate"] == 1.0 and row["tp_fp_fn_identical"]]
    lines = [
        "# Final Ablation Audit Report",
        "",
        "## Verdict",
        "- Current audited outputs are not sufficient to claim Cascade-SmolLM is an all-SmolLM cascade if its gate path still resolves to the MiniLM detector or encoder.",
        "- Existing prediction evidence shows identical OOS decisions between Cascade-MiniLM and Cascade-SmolLM wherever `same_binary_rate=1.0` and TP/FP/FN are identical.",
        "- Use the fresh SmolLM-gate evaluator before making paper claims about a homogeneous SmolLM cascade.",
        "",
        "## Recomputed Metrics",
        "",
        "| Dataset | Variant | Known F1 | OOS F1 | Acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['dataset']} {row['kir_tag']} | {row['variant']} | {row['known_f1_pct_2dp']} | {row['oos_f1_pct_2dp']} | {row['acc_pct_2dp']} |"
        )
    lines.extend(["", "## Overlap Finding", ""])
    if identical:
        for row in identical:
            lines.append(
                f"- {row['dataset']}: Cascade-MiniLM and Cascade-SmolLM have identical OOS decisions "
                f"at {row['kir_tag']} "
                f"(TP/FP/FN={row['minilm_tp_oos']}/{row['minilm_fp_oos']}/{row['minilm_fn_oos']})."
            )
    else:
        lines.append("- No dataset has fully identical OOS decisions after the audited run.")
    lines.extend(
        [
            "",
            "## Paper Recommendation",
            "- Include Known F1 in the ablation table because the method discusses OOS rejection and known-intent classification trade-offs.",
            "- Do not state `Cascade-SmolLM = SmolLM gate + SmolLM router/expert` unless the generated `ablation_variant_audit.md` shows a SmolLM gate path.",
            "- If a true SmolLM gate run is unavailable, rename the current row to reflect the real implementation, e.g. `MiniLM-Gate + SmolLM Downstream`.",
            "",
        ]
    )
    (output_dir / "final_ablation_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_latex_section(
    output_dir: Path,
    metrics_rows: Sequence[Dict[str, Any]],
    overlap_rows: Sequence[Dict[str, Any]],
) -> None:
    rows = "\n".join(
        (
            f"{row['dataset']} {row['kir_tag']} & {row['variant']} & {row['known_f1_pct_2dp']} & "
            f"{row['oos_f1_pct_2dp']} & {row['acc_pct_2dp']} \\\\"
        )
        for row in metrics_rows
    )
    invalid_note = any(row["same_binary_rate"] == 1.0 and row["tp_fp_fn_identical"] for row in overlap_rows)
    caveat = (
        "The audited Cascade-SmolLM outputs share the same OOS decisions as Cascade-MiniLM, "
        "so this row should not be described as a full SmolLM cascade until the SmolLM gate run is regenerated."
        if invalid_note
        else "The audited Cascade-SmolLM outputs use distinct OOS decisions from Cascade-MiniLM."
    )
    text = rf"""\subsection{{Ablation Studies}}
We use ablation studies to answer two questions: whether the front-end OOS gate is necessary, and whether the heterogeneous MiniLM--SmolLM configuration is more effective than homogeneous cascade variants. Full Pipeline denotes the MiniLM gate with SmolLM router and expert modules. The w/o Gate variant removes the front-end gate and restores OOS prediction only through validation-tuned downstream confidence. Cascade-MiniLM uses MiniLM-based gate, router, and expert modules. Cascade-SmolLM should denote a SmolLM-based gate with SmolLM router and expert modules; this label is valid only when the audit confirms that the gate is not the MiniLM detector.

\begin{{table}}[t]
\centering
\caption{{Ablation results recomputed from per-sample predictions.}}
\begin{{tabular}}{{llccc}}
\toprule
Dataset & Variant & Known F1 & OOS F1 & Acc \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}

The main conclusion is that front-end OOS rejection must be audited separately from known-intent classification. The Full Pipeline result should be interpreted through OOS F1 and Known F1 together rather than through accuracy alone.

{caveat} The cascade workflow decouples OOS rejection from known-intent prediction: OOS F1 reflects the front-end rejection behavior, while Known F1 reflects downstream multi-class classification and can be further improved by stronger router or expert classifiers. We therefore report Known F1 in the ablation table without overstating it as the sole optimization target of the method.
"""
    (output_dir / "updated_ablation_section.tex").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v19 ablation prediction outputs")
    parser.add_argument("root", help="Ablation root containing dataset/kir/variant predictions")
    parser.add_argument("--output_dir", default=None, help="Defaults to <root>/audit")
    parser.add_argument("--kir_tags", nargs="*", default=None, help="Optional KIR tags to audit, e.g. kir25_seed42 kir75_seed42")
    args = parser.parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "audit"
    outputs = write_reports(root, output_dir, kir_tags=args.kir_tags)
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
