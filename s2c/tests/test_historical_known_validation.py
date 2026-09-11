import numpy as np
from scripts.experiments.run_historical_known_validation import choose_threshold


def test_known_f1_rejects_wrong_class_tail_and_keeps_smallest_tie():
    score, threshold = choose_threshold(
        np.array([.5, .6, .8]), np.array(['a', 'b', 'b']), ['a', 'b', 'a'])
    assert threshold == .6
    assert np.isclose(score, (2/3 + 1)/2)
