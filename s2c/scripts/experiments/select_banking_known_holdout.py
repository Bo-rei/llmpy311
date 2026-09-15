"""Select one Banking geometry from retrained Known-intent holdouts, never test."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.run_kir_sensitivity_known_only import ART, dump

OUT = ART.parent / 'banking_known_holdout'
KIRS = (.25, .5, .75)
SEEDS = (13, 42, 87)


def candidates():
    return [dict(k=k, distance=d, boundary=b, rule=r, fusion=f,
                 fusion_weight=0. if f == 'none' else 1., coverage=q)
            for k, d, b, r, f, q in itertools.product(
                (1, 3), ('mahalanobis_diag', 'euclidean'),
                ('mean_std_1.0', 'center_quantile_0.9'),
                ('normalized_union', 'nearest_sphere'), ('none', 'inverse_margin'),
                (.8, .9, .95))]


def pick(rows, configs):
    scores = np.asarray([[r['balanced_accuracy'] for r in fold] for fold in rows])
    means = scores.mean(axis=0)
    best = int(means.argmax())
    se = scores[:, best].std(ddof=1) / np.sqrt(len(scores))
    eligible = np.flatnonzero(means >= means[best] - se)
    # One-standard-error rule: fewer centers, no fusion, then validation score.
    chosen = min(eligible, key=lambda i: (configs[i]['k'],
                 configs[i]['fusion'] != 'none', -means[i], i))
    return int(chosen), means.tolist(), float(se)


def main():
    import torch
    from scripts.experiments import search_historical_known_training as training
    from scripts.experiments.finalize_historical_known_coverage import _fit_and_score, _quantile
    from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder

    OUT.mkdir(parents=True, exist_ok=True)
    configs = candidates()
    contract = dict(protocol='banking77_shared_textoir_known_intent_holdout',
        dataset='banking77', data_root=str(ART / 'data'), kirs=KIRS, seeds=SEEDS,
        inner_seed=42, folds=3, recipe=training.RECIPES['last2'], candidates=configs,
        selection='mean fold balanced detection accuracy; one-SE prefers K1/no fusion',
        checkpoint_selection='retained-intent Known dev macro F1',
        real_oos_training=False, real_oos_validation=False, pseudo_oos_validation=True,
        test_used_for_selection=False, test_previously_observed=True,
        known_split='shared manifest; Python sample, not TextOIR native NumPy seed sampling')
    lock_path = OUT / 'selection_contract.json'
    if lock_path.exists():
        assert json.loads(lock_path.read_text()) == json.loads(json.dumps(contract))
    else:
        dump(lock_path, contract)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    fold_results = []
    training.OUT = OUT / 'training_records'
    training.ART = OUT / 'checkpoints'
    training.OUT.mkdir(exist_ok=True)
    training.ART.mkdir(exist_ok=True)
    for kir in KIRS:
        tag = f'kir{round(kir * 100):02d}_seed42'
        source = ART / 'data/banking77' / tag / 'gate'
        train, dev = [json.loads((source / f'{s}.json').read_text()) for s in ('train', 'val')]
        assert all(r['label'] == 0 for r in train + dev)
        names = np.asarray(sorted({r['intent'] for r in train}))
        names = np.random.RandomState(314159).permutation(names)
        for fold, held_names in enumerate(np.array_split(names, 3)):
            held = set(held_names.tolist())
            dataset = f'banking77_inner{fold}'
            fold_dir = OUT / 'inner_data' / dataset / tag / 'gate'
            retained_train = [r for r in train if r['intent'] not in held]
            retained_dev = [r for r in dev if r['intent'] not in held]
            held_dev = [r for r in dev if r['intent'] in held]
            dump(fold_dir / 'train.json', retained_train)
            dump(fold_dir / 'val.json', retained_dev)
            dump(fold_dir / 'holdout.json', held_dev)
            dump(fold_dir / 'MANIFEST.json', dict(source=str(source), held_intents=sorted(held),
                excluded_from_encoder_training=True, excluded_from_checkpoint_selection=True))
            training.DATA_ROOT = OUT / 'inner_data'
            record = training.train_cell(dataset, kir, 42, 'last2', device)
            result_path = OUT / f'kir{round(kir*100):02d}_fold{fold}.json'
            if result_path.exists():
                fold_results.append(json.loads(result_path.read_text())['candidates'])
                continue
            encoder = _RacalGateEncoder(training.MODEL_ROOT / 'all-MiniLM-L6-v2',
                                         Path(record['checkpoint']), device)
            rows = retained_dev + held_dev
            tv, vv = [encoder.encode([r['text'] for r in rr], batch_size=128)
                      for rr in (retained_train, rows)]
            scored = []
            for config in configs:
                _, result = _fit_and_score(tv, retained_train, vv, config)
                scores = result['score']
                threshold = _quantile(scores[:len(retained_dev)], config['coverage'])
                kr = float((scores[:len(retained_dev)] <= threshold).mean())
                recall = float((scores[len(retained_dev):] > threshold).mean())
                scored.append(dict(config=config, threshold=threshold, known_recall=kr,
                    held_intent_recall=recall, balanced_accuracy=(kr + recall) / 2))
            dump(result_path, dict(checkpoint=record, candidates=scored, test_read=False))
            fold_results.append(scored)
            del encoder
            torch.cuda.empty_cache()
    index, means, se = pick(fold_results, configs)
    selected = configs[index]
    k1_ids = [i for i,c in enumerate(configs) if c['k'] == 1]
    k1_index, _, _ = pick([[fold[i] for i in k1_ids] for fold in fold_results],
                         [configs[i] for i in k1_ids])
    control = configs[k1_ids[k1_index]]
    dump(OUT / 'geometry_selection.json', dict(config=selected, selected_index=index,
         mean_scores=means, best_standard_error=se, rule=contract['selection'],
         diagnostic_k1=control, diagnostic_k1_role='validation-selected ablation, never replace primary by test score',
         test_read=False))
    # Fit final representation on every outer Known class; reuse only identical recipes/data.
    for kir in KIRS:
        for seed in SEEDS:
            tag = f'kir{round(kir*100):02d}_seed{seed}'
            existing = ART / 'training' / f'banking77_{tag}_last2.json'
            if existing.exists():
                record = json.loads(existing.read_text())
                assert record['recipe'] == training.RECIPES['last2']
            else:
                training.DATA_ROOT = ART / 'data'
                record = training.train_cell('banking77', kir, seed, 'last2', device)
            source = ART / 'data/banking77' / tag / 'gate'
            tr, dv = [json.loads((source / f'{s}.json').read_text()) for s in ('train', 'val')]
            encoder = _RacalGateEncoder(training.MODEL_ROOT / 'all-MiniLM-L6-v2', Path(record['checkpoint']), device)
            tv, vv = [encoder.encode([r['text'] for r in rr], batch_size=128) for rr in (tr, dv)]
            _, result = _fit_and_score(tv, tr, vv, selected)
            geometry = {**selected, 'threshold': _quantile(result['score'], selected['coverage'])}
            dump(OUT / 'locks' / f'banking77_{tag}.json', dict(checkpoint=record['checkpoint'],
                geometry=geometry, scoring_family='coverage', selection_contract=str(lock_path),
                geometry_selection=str(OUT / 'geometry_selection.json'),
                real_oos_used=False, pseudo_oos_validation=True, test_read=False))
            _, result = _fit_and_score(tv, tr, vv, control)
            control_geometry = {**control, 'threshold': _quantile(result['score'], control['coverage'])}
            dump(OUT / 'k1_control/locks' / f'banking77_{tag}.json', dict(checkpoint=record['checkpoint'],
                geometry=control_geometry, scoring_family='coverage', selection_contract=str(lock_path),
                role='ablation_only', real_oos_used=False, pseudo_oos_validation=True, test_read=False))
            del encoder
            torch.cuda.empty_cache()
    dump(OUT / 'selection_complete.json', dict(status='locked', units=9, test_read=False))


if __name__ == '__main__':
    main()
