import numpy as np
from tools.eval.final_prediction_metrics import prediction_metrics


def test_oos_false_accepts_reduce_known_f1():
    result = prediction_metrics(['a','b','__oos__','__oos__'], ['a','b','a','__oos__'], ['a','b'])
    assert np.isclose(result['known_f1'], 5/6)
    assert np.isclose(result['oos_f1'], 2/3)
    assert result['oos_false_accepts'] == 1
    assert result['known_recall'] == 1
    assert result['false_accept_rate'] == .5


def test_known_classification_error_does_not_change_binary_detection():
    result = prediction_metrics(['a','b','__oos__'], ['b','a','__oos__'], ['a','b'])
    assert result['oos_f1'] == 1
    assert result['known_f1'] == 0
    assert result['accuracy'] == 1/3
