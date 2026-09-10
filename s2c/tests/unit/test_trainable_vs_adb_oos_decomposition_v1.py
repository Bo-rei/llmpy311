from tools.analysis.build_trainable_vs_adb_oos_decomposition_v1 import decompose_counts


def test_oos_precision_counts_known_rejected_as_false_positive() -> None:
    metrics = decompose_counts(
        n_known=100,
        n_oos=100,
        oos_correct=80,
        oos_false_accept=20,
        known_rejected=10,
        known_correct=85,
        known_wrong=5,
    )
    assert metrics["oos_precision"] == 80 / 90
    assert metrics["oos_recall"] == 0.8
    assert metrics["known_acceptance"] == 0.9
    assert metrics["oos_f1"] == 2 * (80 / 90) * 0.8 / ((80 / 90) + 0.8)


def test_oos_false_accept_changes_recall_not_oos_precision_denominator() -> None:
    base = decompose_counts(100, 100, 80, 20, 10, 85, 5)
    changed = decompose_counts(100, 100, 80, 0, 10, 85, 5)
    assert changed["oos_precision"] == base["oos_precision"]
    assert changed["oos_recall"] == base["oos_recall"]
