from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.cluster_separability import export as exporter


def _write_unit(
    root: Path,
    *,
    phase: str,
    dataset: str = "clinc150",
    kir: int = 50,
    seed: int = 42,
    distance: str = "mahalanobis_diag",
    k_gate: int = 2,
    test_oos_f1: float = 0.81,
    validation_oos_f1: float = 0.79,
    method: str | None = None,
) -> Path:
    unit = root / phase / dataset / f"kir{kir}_seed{seed}" / distance / f"k{k_gate}"
    unit.mkdir(parents=True)
    config = {
        "phase": phase,
        "dataset": dataset,
        "kir": kir,
        "data_seed": seed,
        "distance": distance,
        "k_gate": k_gate,
    }
    if method is not None:
        config["method"] = method
    (unit / "run_manifest.json").write_text(
        json.dumps({"config": config, "config_hash": f"hash-{phase}-{k_gate}"}),
        encoding="utf-8",
    )
    (unit / "eval_results.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "validation": {"oos_f1": validation_oos_f1, "id_recall": 0.92, "fpr95": 0.2},
                "test": {
                    "oos_precision": 0.8,
                    "oos_recall": 0.82,
                    "oos_f1": test_oos_f1,
                    "id_recall": 0.9,
                    "oos_rejection": 0.82,
                    "auroc": 0.88,
                    "aupr_oos": 0.87,
                    "fpr95": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )
    return unit


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_copies_only_existing_fixed_and_tuned_results(tmp_path: Path):
    root = tmp_path / "experiments"
    fixed = _write_unit(root, phase="fixed", k_gate=1, test_oos_f1=0.71)
    _write_unit(root, phase="tuned", k_gate=2, test_oos_f1=0.83)
    # An eval result without its provenance manifest is incomplete and must not
    # become a paper result.
    orphan = root / "fixed" / "orphan"
    orphan.mkdir(parents=True)
    (orphan / "eval_results.json").write_text(
        json.dumps({"test": {"oos_f1": 1.0}}), encoding="utf-8"
    )

    result = exporter.export_artifacts(root)

    assert result["source_rows"] == 2
    assert len(_read_csv(root / "experiment_matrix.csv")) == 2
    fixed_rows = _read_csv(root / "kir_k_fixed_boundary.csv")
    tuned_rows = _read_csv(root / "kir_k_tuned_boundary.csv")
    assert fixed_rows[0]["test_oos_f1"] == "0.71"
    assert fixed_rows[0]["eval_results"] == str((fixed / "eval_results.json").relative_to(root))
    assert tuned_rows[0]["test_oos_f1"] == "0.83"
    assert not (root / "gate_baseline_summary.csv").exists()


def test_missing_cells_reports_canonical_grid_without_filling_results(tmp_path: Path):
    root = tmp_path / "experiments"
    _write_unit(root, phase="fixed", dataset="clinc150", kir=25, seed=13, distance="euclidean", k_gate=1)

    exporter.export_artifacts(root)
    missing = json.loads((root / "missing_cells.json").read_text(encoding="utf-8"))

    assert missing["expected_per_phase"] == 270
    assert missing["present"] == {"fixed": 1, "tuned": 0}
    assert missing["missing_count"] == {"fixed": 269, "tuned": 270}
    assert len(_read_csv(root / "experiment_matrix.csv")) == 1
    assert not (root / "kir_k_tuned_boundary.csv").exists()


def test_protocol_audit_keeps_textoir_separate_and_records_hash(tmp_path: Path):
    root = tmp_path / "experiments"
    _write_unit(root, phase="baselines", method="msp", k_gate=1)
    textoir_audit = root / "textoir_protocol" / "data_protocol_audit.json"
    textoir_audit.parent.mkdir(parents=True)
    textoir_audit.write_text('{"upstream_commit":"abc123"}', encoding="utf-8")

    exporter.export_artifacts(root)

    audit = json.loads((root / "data_protocol_audit.json").read_text(encoding="utf-8"))
    s2c_protocol = audit["s2c_gate_protocol"]
    assert s2c_protocol["oos_is_positive_class"] is True
    assert s2c_protocol["selection_split"] == "validation"
    assert s2c_protocol["test_used_for_k_or_threshold_selection"] is False
    assert s2c_protocol["controlled_baseline_methods"] == ["msp"]
    assert audit["textoir_protocol"]["separate_protocol"] is True
    assert audit["textoir_protocol"]["audit_path"] == (
        "textoir_protocol/data_protocol_audit.json"
    )
    assert len(audit["textoir_protocol"]["audit_sha256"]) == 64


def test_baseline_table_combines_controlled_and_comparable_geometry_rows(tmp_path: Path):
    root = tmp_path / "experiments"
    _write_unit(root, phase="baselines", method="msp", k_gate=1)
    _write_unit(root, phase="tuned", distance="euclidean", k_gate=1)
    _write_unit(root, phase="tuned", distance="mahalanobis_diag", k_gate=1)
    _write_unit(root, phase="tuned", distance="mahalanobis_diag", k_gate=2)
    _write_unit(
        root,
        phase="tuned",
        distance="mahalanobis_diag",
        k_gate=3,
        validation_oos_f1=0.95,
    )

    exporter.export_artifacts(root)

    rows = _read_csv(root / "gate_baseline_by_seed.csv")
    assert {row["method"] for row in rows} == {
        "msp",
        "euclidean_centroid",
        "diag_mahalanobis_centroid",
        "multisphere_k2",
        "multisphere_selected_k",
    }
    selected = next(row for row in rows if row["method"] == "multisphere_selected_k")
    assert selected["k_gate"] == "3"
    summary = _read_csv(root / "gate_baseline_summary.csv")
    assert {row["method"] for row in summary} == {row["method"] for row in rows}
    assert all(row["seed_count"] == "1" for row in summary)


def test_baseline_summary_reports_mean_and_std_across_data_seeds(tmp_path: Path):
    root = tmp_path / "experiments"
    for seed, score in ((13, 0.70), (42, 0.80), (87, 0.90)):
        _write_unit(
            root,
            phase="baselines",
            method="msp",
            seed=seed,
            test_oos_f1=score,
        )

    exporter.export_artifacts(root)

    row = _read_csv(root / "gate_baseline_summary.csv")[0]
    assert row["method"] == "msp"
    assert row["seed_count"] == "3"
    assert round(float(row["test_oos_f1_mean"]), 8) == 0.8
    assert round(float(row["test_oos_f1_std"]), 8) == 0.1


def test_baseline_export_includes_available_runtime_measurements(tmp_path: Path):
    root = tmp_path / "experiments"
    unit = _write_unit(root, phase="baselines", method="msp", k_gate=1)
    (unit / "timing.json").write_text(
        json.dumps(
            {
                "test_scoring_seconds": 0.25,
                "test_samples_per_second": 4000.0,
                "process_peak_rss_mb": 512.0,
            }
        ),
        encoding="utf-8",
    )

    exporter.export_artifacts(root)

    by_seed = _read_csv(root / "gate_baseline_by_seed.csv")[0]
    assert by_seed["test_scoring_seconds"] == "0.25"
    assert by_seed["test_samples_per_second"] == "4000.0"
    assert by_seed["process_peak_rss_mb"] == "512.0"
    summary = _read_csv(root / "gate_baseline_summary.csv")[0]
    assert summary["test_samples_per_second_mean"] == "4000.0"


def test_figure_manifest_contains_only_real_figures_with_hashes(tmp_path: Path):
    root = tmp_path / "experiments"
    root.mkdir()
    figure = root / "figures" / "near_oos.svg"
    figure.parent.mkdir()
    figure.write_text("<svg/>", encoding="utf-8")
    (figure.parent / "notes.txt").write_text("not a figure", encoding="utf-8")
    upstream_figure = (
        root
        / "textoir_protocol"
        / "attempt_0001"
        / "runtime_overlay"
        / "figs"
        / "upstream_example.png"
    )
    upstream_figure.parent.mkdir(parents=True)
    upstream_figure.write_bytes(b"not a paper figure")

    result = exporter.export_artifacts(root)
    manifest = json.loads((root / "figure_manifest.json").read_text(encoding="utf-8"))

    assert result["source_rows"] == 0
    assert manifest["figures"] == [
        {
            "bytes": 6,
            "format": "svg",
            "path": "figures/near_oos.svg",
            "sha256": "d4dc56669143034f31aa309635d4113d9ad76a02b1739da22c965ed2049be9e6",
        }
    ]
    assert _read_csv(root / "experiment_matrix.csv") == []


def test_k8_stress_is_separate_from_canonical_fixed_table(tmp_path: Path):
    root = tmp_path / "experiments"
    _write_unit(root, phase="fixed", k_gate=1)
    _write_unit(root, phase="fixed", k_gate=8)

    exporter.export_artifacts(root)

    fixed = _read_csv(root / "kir_k_fixed_boundary.csv")
    stress = _read_csv(root / "k8_stress_control.csv")
    assert [row["k_gate"] for row in fixed] == ["1"]
    assert [row["k_gate"] for row in stress] == ["8"]
