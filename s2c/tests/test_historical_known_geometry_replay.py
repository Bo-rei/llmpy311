import numpy as np
from scripts.experiments.search_historical_known_geometry import search
from scripts.experiments.finalize_historical_known_search import replay


def test_scoring_reconstruction_matches_validation_candidates():
    rng = np.random.default_rng(18)
    centers = rng.normal(size=(3, 8))
    train = np.concatenate([c + .2*rng.normal(size=(12,8)) for c in centers])
    values = np.concatenate([c + .4*rng.normal(size=(3,8)) for c in centers])
    labels = np.repeat(['a','b','c'],12)
    truth = np.repeat(['a','b','c'],3)
    rows = [{'intent':name} for name in labels]
    _, candidates = search(train,values,labels,truth)
    for boundary in ('mean_std_1.0','center_quantile_0.9','intent_quantile_0.9'):
        for rule in ('nearest_sphere','normalized_union'):
            config = next(r for r in candidates if r['k']==2 and r['distance']=='mahalanobis_diag'
                          and r['boundary']==boundary and r['rule']==rule
                          and r['fusion_weight']==.5 and r['threshold']==1.5)
            detector, output = replay(train,rows,values,config)
            predicted = np.array([detector.cluster_to_intent[int(i)] for i in output['nearest_cluster']])
            accepted = output['score'] <= config['threshold']
            utility = np.mean(accepted * np.where(predicted==truth,1.,-4.))
            assert np.isclose(utility,config['utility'])
            assert np.isclose(accepted.mean(),config['known_coverage'])


def test_coverage_selection_and_replay_agree(monkeypatch):
    from scripts.experiments import finalize_historical_known_coverage as coverage
    monkeypatch.setattr(coverage, 'K_VALUES', (1, 2))
    monkeypatch.setattr(coverage, 'LAMBDA_VALUES', (1.,))
    monkeypatch.setattr(coverage, 'BOUNDARY_QUANTILES', (.9,))
    monkeypatch.setattr(coverage, 'FUSION_WEIGHTS', (.5,))
    monkeypatch.setattr(coverage, 'COVERAGE_TARGET', .9)
    monkeypatch.setattr(coverage, 'FIXED_THRESHOLD', None)
    rng = np.random.default_rng(18)
    centers = rng.normal(size=(3, 8))
    train = np.concatenate([c + .2*rng.normal(size=(12, 8)) for c in centers])
    values = np.concatenate([c + .4*rng.normal(size=(10, 8)) for c in centers])
    rows = [{'intent': name, 'label': 0} for name in np.repeat(['a', 'b', 'c'], 12)]
    val_rows = [{'intent': name, 'label': 0} for name in np.repeat(['a', 'b', 'c'], 10)]
    candidates = coverage.search_known(train, values, rows, val_rows)
    for config in candidates:
        _, output = coverage._fit_and_score(train, rows, values, config)
        accepted = output['score'] <= config['threshold']
        assert np.isclose(accepted.mean(), config['known_coverage'])
        assert .9 <= accepted.mean() < 1.
