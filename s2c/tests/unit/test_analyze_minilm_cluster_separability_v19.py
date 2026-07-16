from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.cluster_separability import analysis


def _unit_vector_for_distance(distance: float) -> list[float]:
    cosine = 1.0 - distance
    return [cosine, float(np.sqrt(max(0.0, 1.0 - cosine**2)))]


def test_validation_bucket_thresholds_ignore_known_examples():
    labels = [0, 1, 1, 1, 1, 1]
    distances = [99.0, 0.0, 0.25, 0.5, 0.75, 1.0]

    q20, q80 = analysis.validation_bucket_thresholds(labels, distances)

    assert q20 == pytest.approx(0.20)
    assert q80 == pytest.approx(0.80)


def test_near_far_uses_validation_quantiles_and_full_known_test_set():
    centers = np.asarray([[1.0, 0.0]])
    validation_labels = [0, 1, 1, 1, 1, 1]
    validation_embeddings = np.asarray(
        [_unit_vector_for_distance(value) for value in (0.9, 0.0, 0.25, 0.5, 0.75, 1.0)]
    )
    # Two known samples plus one OOS sample in each frozen validation bucket.
    test_embeddings = np.asarray(
        [_unit_vector_for_distance(value) for value in (0.05, 0.10, 0.10, 0.50, 0.90)]
    )
    test_scores = pd.DataFrame(
        {
            "true_binary_label": [0, 0, 1, 1, 1],
            "oos_source": ["known", "known", "heldout_unknown", "heldout_unknown", "heldout_unknown"],
            "score": [0.2, 0.3, 1.2, 1.2, 1.2],
        }
    )

    rows = analysis.compute_near_far_rows(
        validation_labels=validation_labels,
        validation_embeddings=validation_embeddings,
        test_scores=test_scores,
        test_embeddings=test_embeddings,
        centers=centers,
        metadata={"dataset": "toy"},
    )

    combined = {row["bucket"]: row for row in rows if row["oos_source"] == "combined"}
    assert set(combined) == {"near", "medium", "far"}
    assert all(row["validation_q20"] == pytest.approx(0.20) for row in combined.values())
    assert all(row["validation_q80"] == pytest.approx(0.80) for row in combined.values())
    assert [combined[name]["bucket_oos_count"] for name in ("near", "medium", "far")] == [1, 1, 1]
    assert all(row["known_test_count"] == 2 for row in combined.values())
    assert all(row["oos_f1"] == pytest.approx(1.0) for row in combined.values())


def test_overlap_counts_distinct_intent_multicoverage_from_saved_scores():
    scores = pd.DataFrame(
        {
            "true_binary_label": [0, 0, 1, 1, 1],
            "prediction": [0, 1, 0, 0, 1],
            "coverage_count": [1, 2, 2, 1, 0],
        }
    )

    row = analysis.compute_overlap_row(scores, {"dataset": "toy"})

    assert row["known_multi_coverage_rate"] == pytest.approx(0.5)
    assert row["oos_multi_coverage_rate"] == pytest.approx(1 / 3)
    assert row["false_accepted_oos_count"] == 2
    assert row["false_accepted_oos_multi_coverage_rate"] == pytest.approx(0.5)


def test_intent_pair_overlap_counts_gold_groups_and_false_accepts():
    detector_state = {
        "l2_normalize": False,
        "distance_metric": "euclidean",
        "spheres": [
            {"center": [0.0, 0.0], "radius": 1.1, "intent_name": "a", "inv_diag_cov": None},
            # 同一 intent 的第二个 sphere 必须先按 intent 合并，不能形成 (a, a) pair。
            {"center": [-0.2, 0.0], "radius": 0.3, "intent_name": "a", "inv_diag_cov": None},
            {"center": [1.0, 0.0], "radius": 1.1, "intent_name": "b", "inv_diag_cov": None},
        ],
    }
    embeddings = np.asarray([[0.5, 0.0], [0.8, 0.0], [3.0, 0.0]])
    scores = pd.DataFrame(
        {
            "true_binary_label": [0, 1, 1],
            "prediction": [0, 0, 1],
        }
    )

    rows = analysis.compute_intent_pair_overlap_rows(
        detector_state, embeddings, scores, {"dataset": "toy"}
    )

    assert len(rows) == 1
    assert (rows[0]["intent_a"], rows[0]["intent_b"]) == ("a", "b")
    assert rows[0]["joint_test_count"] == 2
    assert rows[0]["joint_known_count"] == 1
    assert rows[0]["joint_oos_count"] == 1
    assert rows[0]["joint_false_accepted_oos_count"] == 1

    mahalanobis_state = {
        **detector_state,
        "distance_metric": "mahalanobis_diag",
        "spheres": [
            {**sphere, "inv_diag_cov": [1.0, 1.0]}
            for sphere in detector_state["spheres"]
        ],
    }
    mahalanobis_rows = analysis.compute_intent_pair_overlap_rows(
        mahalanobis_state, embeddings, scores, {"dataset": "toy"}
    )
    assert mahalanobis_rows[0]["joint_test_count"] == 2


def test_geometry_diagnostics_do_not_require_umap_when_nonlinear_disabled(tmp_path: Path):
    train = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    test = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    rows = [
        {"label": 0, "intent": "a"},
        {"label": 1, "intent": "unknown", "source_split": "heldout_oos_test"},
        {"label": 1, "intent": "oos", "source_split": "clinc_oos_test"},
    ]

    diagnostics = analysis.create_geometry_outputs(
        train, test, rows, tmp_path, max_samples=3, nonlinear=False
    )

    assert diagnostics["embedding_dimension"] == 2
    assert diagnostics["figure_sample_count"] == 3
    assert diagnostics["tsne_figure"] is None
    assert diagnostics["umap_figure"] is None
    assert (tmp_path / "pca.png").is_file()
    assert (tmp_path / "pca_diagnostics.json").is_file()


def test_cluster_stability_subsamples_but_predicts_on_full_intent_support():
    # 两个簇之间留出足够间隔，任何 80% 子样本都应恢复相同的完整样本划分。
    cluster_a = np.asarray([[1.0, value] for value in (-0.03, -0.01, 0.01, 0.03, 0.05)])
    cluster_b = np.asarray([[-1.0, value] for value in (-0.03, -0.01, 0.01, 0.03, 0.05)])
    embeddings = np.vstack((cluster_a, cluster_b))
    rows = [{"intent": "intent_a"} for _ in range(len(embeddings))]

    result = analysis.compute_intent_stability_rows(
        rows,
        embeddings,
        requested_k=2,
        repeats=4,
        subsample_fraction=0.8,
        random_seed=7,
        metadata={"dataset": "toy"},
    )

    assert len(result) == 1
    assert result[0]["support"] == 10
    assert result[0]["subsample_size"] == 8
    assert result[0]["pair_count"] == 6
    assert result[0]["ari_median"] == pytest.approx(1.0)
    assert result[0]["ari_iqr"] == pytest.approx(0.0)


def test_representation_controls_select_thresholds_from_validation_only():
    train = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=float
    )
    train_rows = [
        {"intent": "a", "label": 0},
        {"intent": "a", "label": 0},
        {"intent": "b", "label": 0},
        {"intent": "b", "label": 0},
    ]
    val = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    val_rows = [
        {"intent": "a", "label": 0},
        {"intent": "b", "label": 0},
        {"intent": "unknown", "label": 1},
        {"intent": "unknown", "label": 1},
    ]
    test_rows = list(val_rows)

    first = analysis.compute_representation_control_rows(
        train_rows=train_rows,
        validation_rows=val_rows,
        test_rows=test_rows,
        train_embeddings=train,
        validation_embeddings=val,
        test_embeddings=val,
        id_recall_guard=0.5,
        metadata={"dataset": "toy"},
    )
    # 改变 test embedding 不得反向改变 validation 上冻结的阈值或 norm 方向。
    second = analysis.compute_representation_control_rows(
        train_rows=train_rows,
        validation_rows=val_rows,
        test_rows=test_rows,
        train_embeddings=train,
        validation_embeddings=val,
        test_embeddings=val * 3.0,
        id_recall_guard=0.5,
        metadata={"dataset": "toy"},
    )

    by_name_1 = {row["control"]: row for row in first}
    by_name_2 = {row["control"]: row for row in second}
    assert set(by_name_1) == {
        "l2_k1_centroid",
        "pca_remove_pc1_k1_centroid",
        "raw_norm_only",
        "l2_euclidean_cosine_equivalence",
    }
    for name in ("l2_k1_centroid", "pca_remove_pc1_k1_centroid", "raw_norm_only"):
        assert by_name_1[name]["threshold"] == pytest.approx(by_name_2[name]["threshold"])
        assert by_name_1[name]["score_direction"] == by_name_2[name]["score_direction"]
    equivalent = by_name_1["l2_euclidean_cosine_equivalence"]
    assert equivalent["nearest_intent_agreement"] == pytest.approx(1.0)
    assert equivalent["max_distance_identity_error"] < 1e-10


def test_hard_intent_correlations_report_spearman_and_bootstrap_ci():
    rows = []
    for index in range(1, 9):
        rows.append(
            {
                "dataset": "toy",
                "kir": 50,
                "distance": "euclidean",
                "intent": f"intent_{index}",
                "multimodality_eligible": True,
                "multimodality_score": float(index),
                "false_reject_improvement_k1_minus_k2": float(index),
                "oos_false_accept_delta_k2_minus_k1": float(-index),
                "known_multi_coverage_delta_k2_minus_k1": float(index),
                "intent_f1_delta_k2_minus_k1": float(index),
            }
        )

    correlations = analysis.compute_hard_intent_correlations(
        rows, bootstrap_repeats=100, random_seed=7
    )

    by_metric = {row["target_metric"]: row for row in correlations}
    assert set(by_metric) == {
        "false_reject_improvement_k1_minus_k2",
        "oos_false_accept_delta_k2_minus_k1",
        "known_multi_coverage_delta_k2_minus_k1",
        "intent_f1_delta_k2_minus_k1",
    }
    assert by_metric["false_reject_improvement_k1_minus_k2"]["spearman_rho"] == pytest.approx(1.0)
    assert by_metric["oos_false_accept_delta_k2_minus_k1"]["spearman_rho"] == pytest.approx(-1.0)
    assert all(row["sample_count"] == 8 for row in correlations)
    assert all(row["bootstrap_valid_repeats"] > 0 for row in correlations)


def test_selected_k_lookup_is_specific_to_seed_and_distance(tmp_path: Path):
    pd.DataFrame.from_records(
        [
            {"dataset": "clinc150", "kir": 50, "data_seed": 42, "distance": "euclidean", "selected_k": 2},
            {"dataset": "clinc150", "kir": 50, "data_seed": 42, "distance": "mahalanobis_diag", "selected_k": 3},
            {"dataset": "clinc150", "kir": 50, "data_seed": 87, "distance": "euclidean", "selected_k": 4},
        ]
    ).to_csv(tmp_path / "selected_k_summary.csv", index=False)

    assert analysis._selected_k(tmp_path, "clinc150", 50, 42, "euclidean") == 2
    assert analysis._selected_k(tmp_path, "clinc150", 50, 42, "mahalanobis_diag") == 3


def test_stability_grid_writes_resumable_unit_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(analysis, "DATASETS", ("toy",))
    monkeypatch.setattr(analysis, "DATA_SEEDS", (42,))
    monkeypatch.setattr(analysis, "STABILITY_K_VALUES", (2,))
    source = tmp_path / "fixed" / "toy" / "kir50_seed42" / "euclidean" / "k2"
    source.mkdir(parents=True)
    data_root = tmp_path / "data"
    (data_root / "gate").mkdir(parents=True)
    rows = [{"intent": "a"} for _ in range(4)] + [{"intent": "b"} for _ in range(4)]
    (data_root / "gate" / "train.json").write_text(json.dumps(rows), encoding="utf-8")
    cache_key = "1234567890abcdef-rest"
    cache = tmp_path / "embedding_cache" / "toy" / "kir50_seed42"
    cache.mkdir(parents=True)
    embeddings = np.asarray(
        [[1.0, -0.1], [1.0, 0.0], [1.0, 0.1], [0.9, 0.0], [-1.0, -0.1], [-1.0, 0.0], [-1.0, 0.1], [-0.9, 0.0]]
    )
    np.savez_compressed(cache / "train_1234567890abcdef.npz", embeddings=embeddings)
    (source / "run_manifest.json").write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "embedding_cache": {"train": {"cache_key": cache_key, "sample_count": 8}},
            }
        ),
        encoding="utf-8",
    )

    first = analysis.run_stability_grid(tmp_path, repeats=3)
    assert first["unit_count"] == 1
    assert first["intent_rows"] == 2
    unit_manifest = tmp_path / "analysis" / "stability" / "toy" / "kir50_seed42" / "k2" / "stability_manifest.json"
    assert unit_manifest.is_file()

    # 第二次必须读取完整单元，而不是重新拟合。
    monkeypatch.setattr(
        analysis,
        "compute_intent_stability_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("resume should skip fitting")),
    )
    resumed = analysis.run_stability_grid(tmp_path, repeats=3, resume=True)
    assert resumed["unit_count"] == 1
