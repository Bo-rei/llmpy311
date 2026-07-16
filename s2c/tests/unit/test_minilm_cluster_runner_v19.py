from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.cluster_separability.protocol import compute_coverage_counts
from tools.experiments.cluster_separability import runner


def test_unified_cluster_separability_cli_lists_all_stages() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tools.experiments.cluster_separability", "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert all(name in completed.stdout for name in ("grid", "baseline", "analyze", "export"))


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "phase": "smoke",
        "dataset": "clinc150",
        "kir": 50,
        "data_seed": 42,
        "k_gate": 1,
        "distance": "euclidean",
        "encoder_path": tmp_path / "fake-minilm",
        "data_root": tmp_path / "data",
        "output_root": tmp_path / "outputs",
        "batch_size": 8,
        "dry_run": False,
        "resume": False,
        "grid": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_bundle(tmp_path: Path) -> tuple[argparse.Namespace, dict[str, np.ndarray]]:
    args = _args(tmp_path)
    root = Path(args.data_root) / "clinc150" / "kir50_seed42"
    train_rows = [
        {"text": f"a{i}", "intent": "a", "domain": "d", "label": 0, "source_split": "train"}
        for i in range(4)
    ] + [
        {"text": f"b{i}", "intent": "b", "domain": "d", "label": 0, "source_split": "train"}
        for i in range(4)
    ]
    evaluation_rows = [
        {"text": "a-val", "intent": "a", "domain": "d", "label": 0, "source_split": "val"},
        {"text": "b-val", "intent": "b", "domain": "d", "label": 0, "source_split": "val"},
        {"text": "z-val", "intent": "z", "domain": "unknown", "label": 1, "source_split": "val"},
        {"text": "oos-val", "intent": "oos", "domain": "unknown", "label": 1, "source_split": "oos_val"},
    ]
    _write_json(root / "KNOWN_INTENTS.json", {"known_intents": ["a", "b"], "unknown_intents": ["z"]})
    _write_json(root / "MANIFEST.json", {"dataset": "CLINC150", "kir": 0.5, "seed": 42})
    _write_json(root / "gate" / "train.json", train_rows)
    _write_json(root / "gate" / "val.json", evaluation_rows)
    _write_json(root / "gate" / "test.json", evaluation_rows)
    embeddings = {
        "train": np.asarray(
            [
                [1.0, 0.00], [1.0, 0.05], [0.95, 0.00], [0.95, 0.05],
                [0.0, 1.00], [0.05, 1.0], [0.0, 0.95], [0.05, 0.95],
            ],
            dtype=np.float32,
        ),
        "val": np.asarray([[1.0, 0.02], [0.02, 1.0], [-1.0, -1.0], [-0.7, -0.7]], dtype=np.float32),
        "test": np.asarray([[1.0, 0.02], [0.02, 1.0], [-1.0, -1.0], [-0.7, -0.7]], dtype=np.float32),
    }
    return args, embeddings


def test_canonical_grid_has_270_unique_units_and_dry_run_audits_inputs(tmp_path: Path):
    args = _args(tmp_path, phase="fixed", grid=True)
    units = runner._grid_arguments(args)

    identities = {
        (unit.dataset, unit.kir, unit.data_seed, unit.distance, unit.k_gate)
        for unit in units
    }
    assert len(units) == 270
    assert len(identities) == 270

    payload = runner._dry_run_payload(units[0])
    assert len(payload["required_inputs"]) == 5
    assert not any(item["exists"] for item in payload["required_inputs"])
    assert "validation_scores.parquet" in payload["would_write"]


def test_effective_k_is_capped_and_mahalanobis_covariance_is_per_cluster():
    embeddings = np.asarray(
        [
            [1.0, 0.0], [0.9, 0.1],
            [0.0, 1.0], [0.1, 0.9], [0.0, 0.8], [0.2, 1.0], [0.8, 0.8], [0.7, 0.9],
        ],
        dtype=np.float64,
    )
    intents = np.asarray(["small", "small", "large", "large", "large", "large", "large", "large"])
    detector = runner._build_detector(4, "mahalanobis_diag")
    detector.fit(embeddings, intents)

    assert len(detector.intent_to_clusters["small"]) == 2
    assert len(detector.intent_to_clusters["large"]) == 4
    normalized = detector._normalize_embeddings(embeddings)
    labels = np.asarray(detector._train_cluster_labels)
    for sphere in detector.spheres:
        points = normalized[labels == sphere.cluster_id]
        expected = 1.0 / (np.var(points - sphere.center, axis=0) + detector.covariance_eps)
        assert sphere.inv_diag_cov is not None
        np.testing.assert_allclose(sphere.inv_diag_cov, expected)


def test_score_frame_has_canonical_schema_and_distinct_intent_coverage():
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]], dtype=np.float64
    )
    intents = np.asarray(["a", "a", "b", "b"])
    detector = runner._build_detector(1, "euclidean")
    detector.fit(embeddings, intents)
    query = np.asarray([[1.0, 0.02], [0.0, 1.0], [-1.0, 0.02]], dtype=np.float64)
    rows = [
        {"text": "a", "intent": "a", "label": 0, "source_split": "test"},
        {"text": "z", "intent": "z", "label": 1, "source_split": "test"},
        {"text": "b", "intent": "b", "label": 0, "source_split": "test"},
    ]

    _, frame = runner._score_split(
        detector,
        rows,
        query,
        {"known_intents": ["a", "b"], "unknown_intents": ["z"]},
        "test",
        1.0,
    )

    required = {
        "sample_id", "true_binary_label", "true_intent", "oos_source", "source_split",
        "score", "prediction", "nearest_intent", "nearest_cluster", "distance", "radius",
        "coverage_count", "requested_k", "effective_k",
    }
    assert required <= set(frame.columns)
    distances = runner._all_sphere_distances(detector, query)
    expected = compute_coverage_counts(
        distances,
        np.asarray([sphere.radius for sphere in detector.spheres]),
        [str(sphere.intent_name) for sphere in detector.spheres],
    )
    assert frame["coverage_count"].tolist() == expected.tolist()


def test_run_writes_validation_scores_and_resume_requires_matching_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    args, embeddings = _synthetic_bundle(tmp_path)
    monkeypatch.setattr(runner, "_load_encoder", lambda _: object())

    def fake_embeddings(**kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        split = str(kwargs["split"])
        values = embeddings[split]
        return values, {
            "cache_key": f"synthetic-{split}",
            "embedding_hash": runner.hashlib.sha256(values.tobytes()).hexdigest(),
            "cache_hit": False,
        }

    monkeypatch.setattr(runner, "_load_or_encode", fake_embeddings)
    output = runner.run_unit(args)

    validation_path = output / "validation_scores.parquet"
    assert validation_path.is_file()
    validation = pd.read_parquet(validation_path)
    assert len(validation) == 4
    assert set(validation["source_split"]) == {"val", "oos_val"}
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"

    args.resume = True
    monkeypatch.setattr(
        runner,
        "_load_encoder",
        lambda _: (_ for _ in ()).throw(AssertionError("valid resume loaded encoder")),
    )
    assert runner.run_unit(args) == output

    # A changed input invalidates resume before any model or embedding work.
    train_path = Path(args.data_root) / "clinc150" / "kir50_seed42" / "gate" / "train.json"
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="valid resume loaded encoder"):
        runner.run_unit(args)
