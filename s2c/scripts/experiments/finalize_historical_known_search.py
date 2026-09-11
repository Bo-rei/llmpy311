"""Select shared per-cell geometry on Known validation, then evaluate test once."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.search_historical_known_geometry import search, OUT as GEOMETRY
from scripts.experiments.search_historical_known_training import OUT as TRAINING, CELLS
from tools.eval.run_historical_trainable_parameter_full_pipeline import (
    DATA_ROOT, MODEL_ROOT, DEFAULT_H1_ROOT, EXTERNAL_OOS_F1_BY_KIR, write_csv, _metrics_from_output)
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder
from tools.eval.run_historical_trainable_extended_boundary_search import _distance_matrix
from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector

OUT = ROOT / 'results/analysis/historical_known_final'
SEEDS = (13, 42, 87)
CONFIG_FIELDS = ('k', 'distance', 'boundary', 'rule', 'fusion_weight', 'threshold')


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def replay(train_values, train, values, config):
    """Reconstruct exactly the scoring family used by the validation search."""
    detector = MultiSphereOOSDetector(center_mode='class_centroid_mixture',
        subcenters_per_intent=int(config['k']), radius_method='mean_std', radius_lambda=1.,
        distance_metric=config['distance'], covariance_eps=1e-6, l2_normalize=True,
        random_state=42, acceptance_mode='nearest_sphere')
    intents = np.array([r['intent'] for r in train])
    detector.fit(train_values, intents)
    td, vd = _distance_matrix(detector, train_values), _distance_matrix(detector, values)
    owners = np.array([detector.cluster_to_intent[i] for i in range(len(detector.spheres))])
    boundary = config['boundary']
    parameter = float(boundary.rsplit('_', 1)[1])
    if boundary.startswith('mean_std_'):
        detector.radius_lambda = parameter
        detector._compute_radii()
        radius = np.array([s.radius for s in detector.spheres])
    elif boundary.startswith('center_quantile_'):
        assignment = np.argmin(np.where(intents[:, None] == owners[None, :], td, np.inf), axis=1)
        # The validation grid's fallback was lambda=3 (last mean/std candidate).
        detector.radius_lambda = 3.
        detector._compute_radii()
        radius = np.array([np.quantile(td[assignment == i, i], parameter)
            if np.any(assignment == i) else detector.spheres[i].radius for i in range(len(owners))])
    elif boundary.startswith('intent_quantile_'):
        radius = np.array([np.quantile(td[intents == name].min(axis=1), parameter) for name in owners])
    else:
        raise ValueError(boundary)
    nearest = vd.argmin(axis=1)
    first = vd[np.arange(len(vd)), nearest]
    second = np.min(np.where(owners[None, :] != owners[nearest, None], vd, np.inf), axis=1)
    normalized = vd / np.maximum(radius, 1e-12)
    ids = nearest if config['rule'] == 'nearest_sphere' else normalized.argmin(axis=1)
    score = normalized[np.arange(len(vd)), ids] + float(config['fusion_weight']) * first / np.maximum(second, 1e-12)
    return detector, dict(score=score, nearest_cluster=ids)


def load_known(dataset, kir, seed):
    folder = DATA_ROOT / dataset / f'kir{round(kir*100):02d}_seed{seed}' / 'gate'
    train = read(folder / 'train.json')
    known = {r['intent'] for r in train}
    val = [r for r in read(folder / 'val.json') if r['intent'] in known]
    assert all(int(r['label']) == 0 for r in train + val)
    return train, val


def main():
    assert read(TRAINING/'MANIFEST.json')['status'] == 'validation_complete'
    assert read(GEOMETRY/'MANIFEST.json')['status'] == 'validation_complete'
    screened = read(TRAINING/'screening_selection.json')
    assert {(r['dataset'], r['kir']) for r in screened} == set(CELLS)
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    torch.zeros(1, device=device)
    manifest = dict(status='selecting', device='cuda', protocol='historical_v19_paper_main',
        real_oos_used_for_selection=False, pseudo_oos_used=False, test_used_for_selection=False,
        selection='mean Known selective utility across three seeds; shared recipe and geometry per dataset/KIR',
        full_pipeline_verified=False)
    dump(OUT/'MANIFEST.json', manifest)
    locks = []
    for dataset, kir in CELLS:
        recipes = ['existing'] + [r['name'] for r in screened if r['dataset'] == dataset and r['kir'] == kir]
        choices = []
        for name in recipes:
            groups, checkpoints = [], []
            for seed in SEEDS:
                tag = f'{dataset}_kir{round(kir*100):02d}_seed{seed}'
                train, val = load_known(dataset, kir, seed)
                if name == 'existing':
                    checkpoint = DEFAULT_H1_ROOT/dataset/f'kir{round(kir*100):02d}_seed{seed}/trainable_k1/checkpoint.pt'
                    with (GEOMETRY/f'{tag}_candidates.csv').open() as stream:
                        candidates = list(csv.DictReader(stream))
                    for c in candidates:
                        c['k'] = int(c['k'])
                        for key in ('fusion_weight','threshold','utility','known_coverage','correct_accept_rate'):
                            c[key] = float(c[key])
                else:
                    checkpoint = Path(read(TRAINING/f'{tag}_{name}.json')['checkpoint'])
                    encoder = _RacalGateEncoder(MODEL_ROOT/'all-MiniLM-L6-v2', checkpoint, device)
                    arrays = [encoder.encode([r['text'] for r in rows], batch_size=256) for rows in (train,val)]
                    _, candidates = search(*arrays, np.array([r['intent'] for r in train]), np.array([r['intent'] for r in val]))
                    write_csv(OUT/f'{tag}_{name}_validation.csv', candidates)
                    del encoder
                    torch.cuda.empty_cache()
                groups.append(candidates)
                checkpoints.append(str(checkpoint))
            assert all(len(g) == len(groups[0]) for g in groups)
            utility = np.mean([[r['utility'] for r in group] for group in groups], axis=0)
            idx = int(np.argmax(utility))
            config = {k: groups[0][idx][k] for k in CONFIG_FIELDS}
            assert all({k:g[idx][k] for k in CONFIG_FIELDS} == config for g in groups)
            choices.append(dict(dataset=dataset, kir=kir, recipe=name, config=config,
                                utility=float(utility[idx]), checkpoints=checkpoints))
        winner = max(choices, key=lambda r:r['utility'])
        locks.append(winner)
        dump(OUT/f'{dataset}_kir{round(kir*100):02d}_validation.json', choices)
        dump(OUT/'selection_lock.json', locks)
        print(f'LOCK {dataset} {kir}: {winner}', flush=True)
    assert len(locks) == 9
    manifest.update(status='all_choices_locked')
    dump(OUT/'MANIFEST.json', manifest)
    results=[]
    # This is the first test access in this search campaign.
    for lock in locks:
        for seed, checkpoint in zip(SEEDS, lock['checkpoints'], strict=True):
            dataset, kir = lock['dataset'], lock['kir']
            train, _ = load_known(dataset,kir,seed)
            test = read(DATA_ROOT/dataset/f'kir{round(kir*100):02d}_seed{seed}/gate/test.json')
            encoder = _RacalGateEncoder(MODEL_ROOT/'all-MiniLM-L6-v2',Path(checkpoint),device)
            tv, ev = [encoder.encode([r['text'] for r in rows], batch_size=256) for rows in (train,test)]
            detector, output = replay(tv,train,ev,lock['config'])
            metrics = _metrics_from_output(detector,output,test,lock['config']['threshold'])
            results.append(dict(dataset=dataset,kir=kir,seed=seed,recipe=lock['recipe'],**lock['config'],**metrics))
            write_csv(OUT/'per_seed.csv',results)
            print(f'TEST {dataset}/{kir}/{seed} OOS-F1={metrics["f1_u"]:.6f}',flush=True)
            del encoder
            torch.cuda.empty_cache()
    summary=[]
    for lock in locks:
        group=[r for r in results if (r['dataset'],r['kir'])==(lock['dataset'],lock['kir'])]
        row=dict(dataset=lock['dataset'],kir=lock['kir'],recipe=lock['recipe'],**lock['config'])
        for metric in ('f1_u','f1_k','accuracy','known_recall','false_accept_rate'):
            row[metric+'_mean']=float(np.mean([r[metric] for r in group]))
            row[metric+'_std']=float(np.std([r[metric] for r in group]))
        row['baseline']=EXTERNAL_OOS_F1_BY_KIR[lock['dataset'],lock['kir']]
        row['delta_pp']=100*row['f1_u_mean']-row['baseline']
        summary.append(row)
    write_csv(OUT/'summary.csv',summary)
    manifest.update(status='complete',wins=sum(r['delta_pp']>0 for r in summary),completed_test_units=27,std_ddof=0)
    dump(OUT/'MANIFEST.json',manifest)


if __name__ == '__main__':
    main()
