from scripts.experiments.select_banking_known_holdout import candidates, pick


def test_bounded_geometry_contract():
    configs = candidates()
    assert len(configs) == 96
    assert {c['k'] for c in configs} == {1,3}
    assert {c['coverage'] for c in configs} == {.8,.9,.95}


def test_one_se_selects_simpler_candidate_when_indistinguishable():
    configs = [dict(k=1,fusion='none'),dict(k=3,fusion='inverse_margin')]
    folds = [[dict(balanced_accuracy=a),dict(balanced_accuracy=b)]
             for a,b in ((.8,.75),(.8,.9),(.8,.9))]
    selected,_,_ = pick(folds,configs)
    assert selected == 0
