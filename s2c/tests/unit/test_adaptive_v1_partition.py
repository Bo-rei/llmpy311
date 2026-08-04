import numpy as np

from protocol_v2.experiments.adaptive_v1.partition import bootstrap_split_stability, pca_median_split


def test_pca_split_is_deterministic_and_balanced():
    x = np.column_stack([np.arange(20, dtype=float), np.zeros(20)])
    a, axis_a, info_a = pca_median_split(x)
    b, axis_b, info_b = pca_median_split(x)
    assert np.array_equal(a, b)
    assert np.allclose(axis_a, axis_b)
    assert tuple(np.bincount(a)) == (10, 10)
    assert info_a == info_b


def test_bootstrap_stability_is_finite():
    rng = np.random.default_rng(7)
    x = np.vstack([rng.normal(-2, .1, (20, 3)), rng.normal(2, .1, (20, 3))])
    result = bootstrap_split_stability(x, seed=42, repeats=5)
    assert 0.0 <= result["median"] <= 1.0
