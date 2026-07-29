from __future__ import annotations

import torch

from protocol_v2.experiments.minilm_training import _supcon_loss, _sha256_ids


def test_supcon_is_finite_for_singleton_batch() -> None:
    features = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    loss = _supcon_loss(features, labels)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_supcon_is_finite_with_positive_pairs() -> None:
    features = torch.randn(4, 4)
    labels = torch.tensor([0, 0, 1, 1])
    loss = _supcon_loss(features, labels)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_sample_id_hash_is_order_sensitive() -> None:
    rows = [{"sample_id": "a"}, {"sample_id": "b"}]
    assert _sha256_ids(rows) != _sha256_ids(list(reversed(rows)))
