from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.experiments.run_mogb_fair import _run_method
from scripts.experiments.run_mogb_fixed_k_mean_ablation import (
    load_config,
    run_fixed_k_mean,
)
from scripts.experiments.run_mogb_fixed_k_mean_ablation_sweep import (
    DEFAULT_NEW_K_VALUES,
    apply_shard,
    build_rows,
    group_rows,
    main as sweep_main,
)


def _table() -> tuple[np.ndarray, list[dict[str, object]], np.ndarray, list[dict[str, object]]]:
    a = [[1.0, 0.0], [0.98, 0.20], [0.95, -0.20], [0.97, 0.12], [0.96, -0.10]]
    b = [[0.0, 1.0], [0.20, 0.98], [-0.20, 0.95], [0.10, 0.97], [-0.12, 0.96]]
    train = np.asarray(a + a + b + b, dtype=np.float64)
    train_rows = [
        {"sample_id": f"a{i}", "intent": "a", "label": 0} for i in range(10)
    ] + [{"sample_id": f"b{i}", "intent": "b", "label": 0} for i in range(10)]
    test = np.asarray([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]], dtype=np.float64)
    test_rows = [
        {"sample_id": "t1", "intent": "a", "label": 0},
        {"sample_id": "t2", "intent": "b", "label": 0},
        {"sample_id": "o1", "intent": "oos", "label": 1},
    ]
    return train, train_rows, test, test_rows


def test_k2_matches_existing_ours_partition_mogb_boundary() -> None:
    train, train_rows, test, test_rows = _table()
    output, balls, details = run_fixed_k_mean(train, train_rows, test, k=2)
    reference, reference_balls, reference_details = _run_method(
        "ours_partition_mogb_boundary", train, train_rows, test, test_rows, 42
    )
    for field in ("score", "distance", "radius"):
        assert np.allclose(output[field], reference[field], atol=1e-12, rtol=0.0)
    for field in ("predicted_oos", "nearest_ball", "predicted_label"):
        assert np.array_equal(output[field], reference[field])
    assert balls == reference_balls
    assert details["cluster_count"] == reference_details["cluster_count"]


def test_plan_has_135_new_and_45_reference_units() -> None:
    rows = build_rows(
        ("clinc150", "banking77", "stackoverflow"),
        (0.25, 0.50, 0.75),
        (13, 42, 87, 100, 123),
    )
    groups = group_rows(rows)
    assert DEFAULT_NEW_K_VALUES == (1, 3, 4)
    assert len(rows) == 135
    assert len(groups) == 45
    assert all({int(row["k"]) for row in group} == {1, 3, 4} for group in groups.values())


def test_plan_uses_registered_k_values() -> None:
    rows = build_rows(("stackoverflow",), (0.50,), (42,), (1, 4))
    assert [int(row["k"]) for row in rows] == [1, 4]
    assert all(row["partition"] == "per_intent_kmeans" for row in rows)


def test_sharding_keeps_complete_protocol_cells() -> None:
    rows = build_rows(("stackoverflow",), (0.50,), (13, 42, 87))
    shards = [apply_shard(rows, index, 2) for index in range(2)]
    assert sum(len(shard) for shard in shards) == len(rows)
    assert all(len(shard) % 3 == 0 for shard in shards)
    assert {id(row) for shard in shards for row in shard} == {id(row) for row in rows}


def test_sweep_loads_each_protocol_cell_once(tmp_path: Path, monkeypatch) -> None:
    train, train_rows, test, test_rows = _table()
    calls: list[tuple[str, int, float]] = []

    def fake_load(paths, dataset, seed, kir):  # noqa: ANN001
        del paths
        calls.append((dataset, seed, kir))
        return (
            SimpleNamespace(train=train_rows, test=test_rows),
            train,
            test,
            {"registry_sha256": "r", "canonical_manifest_sha256": "c"},
        )

    def fake_run(paths, config, loaded, *, output_dir=None, overwrite=False):  # noqa: ANN001
        del paths, loaded, overwrite
        run_dir = (
            output_dir
            / f"fixed_k{config['k']}"
            / str(config["dataset"])
            / f"kir_{config['kir']:.2f}"
            / f"seed_{config['seed']}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return run_dir

    monkeypatch.setattr(
        "scripts.experiments.run_mogb_fixed_k_mean_ablation_sweep._load_cached_inputs",
        fake_load,
    )
    monkeypatch.setattr(
        "scripts.experiments.run_mogb_fixed_k_mean_ablation_sweep.run_one_loaded", fake_run
    )
    monkeypatch.setattr(
        "scripts.experiments.run_mogb_fixed_k_mean_ablation_sweep.validate_reference",
        lambda *args, **kwargs: {"run_dir": "reference", "inputs": {}, "run_id": "ref"},
    )
    output = tmp_path / "fixed"
    code = sweep_main(
        [
            "--config",
            "configs/baselines/mogb_fixed_k_mean_ablation_v1.yaml",
            "--datasets",
            "stackoverflow",
            "--kirs",
            "0.50",
            "--seeds",
            "42",
            "87",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert calls == [("stackoverflow", 42, 0.50), ("stackoverflow", 87, 0.50)]
    assert len(list(output.glob("fixed_k*/stackoverflow/kir_0.50/seed_*/manifest.json"))) == 6
    assert load_config(Path("configs/baselines/mogb_fixed_k_mean_ablation_v1.yaml"))["test_used_for_selection"] is False
