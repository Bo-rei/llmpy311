from __future__ import annotations

import math

from tools.experiments.cluster_separability.v21_cluster_casebook import (
    _aligned_jsd,
    _cluster_keywords,
)


def test_casebook_handles_empty_keyword_cluster() -> None:
    keywords, distribution = _cluster_keywords(["the and", "of the"])
    assert keywords == []
    assert distribution.size == 0


def test_aligned_jsd_is_symmetric_and_defined() -> None:
    left = ["reset card password", "reset card"]
    right = ["transfer money", "transfer funds"]
    value = _aligned_jsd(left, right)
    assert math.isfinite(value)
    assert value == _aligned_jsd(right, left)

