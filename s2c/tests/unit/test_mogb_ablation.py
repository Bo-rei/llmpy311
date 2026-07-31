from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.experiments.run_mogb_ablation import (
    _materialize_variant_config,
    _run_ablation,
    load_config,
    run_one,
    run_one_loaded,
    safe_variant_slug,
)
from scripts.experiments.run_mogb_ablation_sweep import (
    _apply_sharding,
    _group_rows_by_cell,
    _merge_config,
    build_partition_variants,
    build_specifications,
    build_sweep_rows,
    main as sweep_main,
)
from scripts.experiments.run_mogb_fair import _run_method


def _table() -> tuple[np.ndarray, list[dict[str, object]], np.ndarray, list[dict[str, object]]]:
    a_points = [[1.0, 0.0], [0.98, 0.20], [0.95, -0.20], [0.97, 0.12], [0.96, -0.10]]
    b_points = [[0.0, 1.0], [0.20, 0.98], [-0.20, 0.95], [0.10, 0.97], [-0.12, 0.96]]
    train = np.asarray(a_points + a_points + b_points + b_points, dtype=np.float64)
    train_rows = [
        {"sample_id": f"a{i}", "intent": "a", "label": 0} for i in range(1, 11)
    ] + [
        {"sample_id": f"b{i}", "intent": "b", "label": 0} for i in range(1, 11)
    ]
    test = np.asarray([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]], dtype=np.float64)
    test_rows = [
        {"sample_id": "t1", "intent": "a", "label": 0},
        {"sample_id": "t2", "intent": "b", "label": 0},
        {"sample_id": "o1", "intent": "oos", "label": 1},
    ]
    return train, train_rows, test, test_rows


def test_safe_variant_slug_normalizes_unsafe_input() -> None:
    assert safe_variant_slug(" Default Mean/Std ") == "default-mean-std"
    assert safe_variant_slug("...") == "default"


def test_default_ablation_matches_core_mogb_minilm_path() -> None:
    train, train_rows, test, test_rows = _table()
    config = {
        "distance": "euclidean",
        "boundary": "mean",
        "purity_train": 0.90,
        "purity_get_ball": 1.00,
        "purity_select_ball": 0.90,
        "min_ball_train": 10,
        "min_ball_get_ball": 5,
        "min_ball_select_ball": 10,
    }
    output, balls, details = _run_ablation(config, train, train_rows, test, test_rows, seed=42)
    core_output, core_balls, core_details = _run_method(
        "mogb_minilm",
        train,
        train_rows,
        test,
        test_rows,
        42,
    )
    assert np.allclose(output["score"], core_output["score"])
    assert np.array_equal(output["predicted_oos"], core_output["predicted_oos"])
    assert np.array_equal(output["nearest_ball"], core_output["nearest_ball"])
    assert np.array_equal(output["predicted_label"], core_output["predicted_label"])
    assert balls == core_balls
    assert details["cluster_count"] == core_details["cluster_count"]


def test_run_one_persists_manifest_with_ablation_parameters(tmp_path: Path, monkeypatch) -> None:
    train, train_rows, test, test_rows = _table()

    def fake_load_cached_inputs(paths, dataset, seed, kir):  # noqa: ANN001
        del paths, dataset, seed, kir
        return (
            SimpleNamespace(train=train_rows, test=test_rows),
            train,
            test,
            {"registry_sha256": "r", "canonical_manifest_sha256": "c"},
        )

    monkeypatch.setattr("scripts.experiments.run_mogb_ablation._load_cached_inputs", fake_load_cached_inputs)
    paths = SimpleNamespace(project_root=tmp_path, run_root=tmp_path / "runs")
    config = _materialize_variant_config(
        load_config(Path("configs/baselines/mogb_ablation_v1.yaml")),
        dataset="stackoverflow",
        kir=0.50,
        seed=42,
        variant="Unsafe Variant/Name",
        distance="mahalanobis_diag",
        boundary="mean_std",
        purity_get_ball=0.95,
        purity_select_ball=0.95,
        min_ball_get_ball=5,
        min_ball_select_ball=5,
    )
    run_dir = run_one(paths, config)
    manifest = __import__("json").loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    written = __import__("json").loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "unsafe-variant-name"
    assert manifest["test_used_for_selection"] is False
    assert manifest["ablation_parameters"]["distance"] == "mahalanobis_diag"
    assert manifest["ablation_parameters"]["boundary"] == "mean_std"
    assert manifest["ablation_parameters"]["purity_get_ball"] == 0.95
    assert manifest["ablation_parameters"]["min_ball_get_ball"] == 5
    assert manifest["ablation_parameters"]["min_ball_select_ball"] == 5
    assert written["output_variant"] == "unsafe-variant-name"


def test_run_one_loaded_matches_run_one_output(tmp_path: Path, monkeypatch) -> None:
    train, train_rows, test, test_rows = _table()

    def fake_load_cached_inputs(paths, dataset, seed, kir):  # noqa: ANN001
        del paths, dataset, seed, kir
        return (
            SimpleNamespace(train=train_rows, test=test_rows),
            train,
            test,
            {"registry_sha256": "r", "canonical_manifest_sha256": "c"},
        )

    monkeypatch.setattr("scripts.experiments.run_mogb_ablation._load_cached_inputs", fake_load_cached_inputs)
    paths = SimpleNamespace(project_root=tmp_path, run_root=tmp_path / "runs")
    config = _materialize_variant_config(
        load_config(Path("configs/baselines/mogb_ablation_v1.yaml")),
        dataset="stackoverflow",
        kir=0.50,
        seed=42,
        variant="Batch Equality",
        distance="euclidean",
        boundary="mean",
    )
    standard_dir = run_one(paths, config)
    loaded = fake_load_cached_inputs(paths, "stackoverflow", 42, 0.50)
    loaded_dir = run_one_loaded(paths, {**config, "output_variant": "batch-equality-loaded"}, loaded)
    standard_metrics = __import__("json").loads((standard_dir / "metrics.json").read_text(encoding="utf-8"))
    loaded_metrics = __import__("json").loads((loaded_dir / "metrics.json").read_text(encoding="utf-8"))
    assert standard_metrics["oos_f1"] == loaded_metrics["oos_f1"]
    assert standard_metrics["accuracy"] == loaded_metrics["accuracy"]
    assert standard_metrics["effective_cluster_count"] == loaded_metrics["effective_cluster_count"]


def test_sweep_builds_expected_12_specs_and_540_cells() -> None:
    assert len(build_partition_variants()) == 10
    assert len(build_specifications()) == 12
    rows = build_sweep_rows(
        ("clinc150", "banking77", "stackoverflow"),
        (0.25, 0.50, 0.75),
        (13, 42, 87, 100, 123),
    )
    assert len(rows) == 540
    assert len({row["variant"] for row in rows}) == 12


def test_sharding_and_grouping_preserve_expected_structure() -> None:
    rows = build_sweep_rows(("stackoverflow",), (0.50,), (42, 87))
    shards = [_apply_sharding(rows, index, 3) for index in range(3)]
    assert sum(len(shard) for shard in shards) == len(rows)
    assert {id(row) for shard in shards for row in shard} == {id(row) for row in rows}
    for shard in shards:
        assert len(shard) % 12 == 0
        shard_groups = _group_rows_by_cell(shard)
        assert all(len(cell_rows) == 12 for cell_rows in shard_groups.values())
    grouped = _group_rows_by_cell(rows)
    assert set(grouped) == {("stackoverflow", 0.50, 42), ("stackoverflow", 0.50, 87)}
    assert all(len(cell_rows) == 12 for cell_rows in grouped.values())


def test_merge_config_promotes_variant_and_keeps_base_defaults() -> None:
    base = load_config(Path("configs/baselines/mogb_ablation_v1.yaml"))
    config = _merge_config(
        base,
        {
            "dataset": "stackoverflow",
            "kir": 0.50,
            "seed": 42,
            "variant": "get_085",
            "distance": "euclidean",
            "boundary": "mean",
            "purity_get_ball": 0.85,
            "purity_select_ball": None,
            "min_ball_get_ball": None,
            "min_ball_select_ball": None,
        },
    )
    assert config["output_variant"] == "get_085"
    assert config["purity_get_ball"] == 0.85
    assert config["purity_select_ball"] == 0.90
    assert config["min_ball_get_ball"] == 5


def test_sweep_groups_into_45_protocol_cells_with_12_variants_each() -> None:
    rows = build_sweep_rows(
        ("clinc150", "banking77", "stackoverflow"),
        (0.25, 0.50, 0.75),
        (13, 42, 87, 100, 123),
    )
    grouped: dict[tuple[str, float, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), float(row["kir"]), int(row["seed"]))].append(row)
    assert len(grouped) == 45
    expected_variants = Counter(spec["variant"] for spec in build_specifications())
    for cell_rows in grouped.values():
        assert len(cell_rows) == 12
        assert Counter(str(row["variant"]) for row in cell_rows) == expected_variants


def test_sweep_rows_use_unique_variant_distance_boundary_combinations_per_protocol_cell() -> None:
    rows = build_sweep_rows(
        ("stackoverflow",),
        (0.50,),
        (42,),
    )
    seen = set()
    for row in rows:
        key = (
            str(row["variant"]),
            str(row["distance"]),
            str(row["boundary"]),
            row.get("purity_get_ball"),
            row.get("purity_select_ball"),
            row.get("min_ball_get_ball"),
            row.get("min_ball_select_ball"),
        )
        assert key not in seen
        seen.add(key)
    assert len(seen) == 12


def test_sweep_loads_cached_inputs_once_per_protocol_cell(tmp_path: Path, monkeypatch) -> None:
    train, train_rows, test, test_rows = _table()
    calls: list[tuple[str, int, float]] = []

    def fake_load_cached_inputs(paths, dataset, seed, kir):  # noqa: ANN001
        del paths
        calls.append((dataset, seed, kir))
        return (
            SimpleNamespace(train=train_rows, test=test_rows),
            train,
            test,
            {"registry_sha256": "r", "canonical_manifest_sha256": "c"},
        )

    def fake_run_one_loaded(paths, config, loaded, *, output_dir=None, overwrite=False):  # noqa: ANN001
        del loaded, overwrite
        run_dir = output_dir / str(config["output_variant"]) / str(config["dataset"]) / f"kir_{config['kir']:.2f}" / f"seed_{config['seed']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return run_dir

    monkeypatch.setattr("scripts.experiments.run_mogb_ablation_sweep._load_cached_inputs", fake_load_cached_inputs)
    monkeypatch.setattr("scripts.experiments.run_mogb_ablation_sweep.run_one_loaded", fake_run_one_loaded)
    output_dir = tmp_path / "mogb_ablation_v1"
    exit_code = sweep_main(
        [
            "--config",
            "configs/baselines/mogb_ablation_v1.yaml",
            "--datasets",
            "stackoverflow",
            "--kirs",
            "0.50",
            "--seeds",
            "42",
            "87",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    assert calls == [("stackoverflow", 42, 0.5), ("stackoverflow", 87, 0.5)]
    manifests = sorted(output_dir.glob("*/stackoverflow/kir_0.50/seed_*/manifest.json"))
    assert len(manifests) == 24
    assert "default" not in {path.parts[-5] for path in manifests}
    resume_exit = sweep_main(
        [
            "--config",
            "configs/baselines/mogb_ablation_v1.yaml",
            "--datasets",
            "stackoverflow",
            "--kirs",
            "0.50",
            "--seeds",
            "42",
            "87",
            "--output-dir",
            str(output_dir),
            "--resume",
        ]
    )
    assert resume_exit == 0
    assert calls == [("stackoverflow", 42, 0.5), ("stackoverflow", 87, 0.5)]
