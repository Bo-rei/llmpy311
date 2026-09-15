"""Prepare shared BANKING77 views and screen Known-only MiniLM recipes.

This stage never evaluates test. Baseline and full-pipeline finalization are
separate pending stages; training completion is not a final result.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
ART = ROOT.parent / 'artifacts/s2c/analysis/kir_sensitivity_known_only'
KIRS = (.25, .5, .75, .1, .2, .3, .4, .6, .7, .8, .9)
SEEDS = (13, 42, 87)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def banking_views(source, root):
    splits = {}
    for split in ('train', 'dev', 'test'):
        with (source / f'{split}.tsv').open() as stream:
            splits[split] = list(csv.DictReader(stream, delimiter='\t'))
    labels = sorted({r['label'] for r in splits['train']})
    assert len(labels) == 77
    assert all(r['label'] in labels for rows in splits.values() for r in rows)
    units = []
    for kir in KIRS:
        for seed in SEEDS:
            known = sorted(random.Random(seed).sample(labels, round(77 * kir)))
            folder = root / 'banking77' / f'kir{round(100*kir):02d}_seed{seed}'
            dump(folder / 'known_labels.json', known)
            counts = {}
            for split, rows in splits.items():
                selected = [dict(text=r['text'], intent=r['label'], domain='banking',
                                 label=int(r['label'] not in known)) for r in rows
                            if split == 'test' or r['label'] in known]
                name = 'val' if split == 'dev' else split
                dump(folder / 'gate' / f'{name}.json', selected)
                export = folder / 'textoir/banking' / f'{split}.tsv'
                export.parent.mkdir(parents=True, exist_ok=True)
                with export.open('w') as stream:
                    writer = csv.writer(stream, delimiter='\t')
                    writer.writerow(('text', 'label'))
                    writer.writerows((r['text'], r['intent']) for r in selected)
                counts[name] = len(selected)
            units.append(dict(dataset='banking77', kir=kir, seed=seed,
                              known_count=len(known), actual_kir=len(known)/77,
                              counts=counts, path=str(folder)))
    return units


def other_views(root):
    from scripts.data.active import rebuild_multi_dataset_v19 as builder
    source = ROOT.parent / 'assets/datasets/s2c/source'
    records = builder._load_stackoverflow_records(source / 'stackoverflow')
    universe = json.loads((source / 'stackoverflow/SOURCE_MANIFEST.json').read_text())['intent_universe']
    units = []
    for dataset in ('clinc150', 'stackoverflow'):
        for kir in KIRS:
            for seed in SEEDS:
                folder = root / dataset / f'kir{round(100*kir):02d}_seed{seed}'
                if dataset == 'clinc150':
                    bundle = builder.build_clinc_bundle(source / 'clinc150/data', kir, seed, folder)
                    known, splits = bundle.known_intents, bundle.gate
                else:
                    known = builder._select_stackoverflow_known_intents(universe, kir, seed)
                    splits = {'train': [], 'val': [], 'test': []}
                    for r in records:
                        split = {'valid':'val', 'val':'val', 'train':'train', 'test':'test'}[r['source_split']]
                        if split != 'train' or r['intent'] in known:
                            splits[split].append(dict(text=r['title'].strip(), intent=r['intent'],
                                domain='stackoverflow', label=int(r['intent'] not in known)))
                dump(folder / 'known_labels.json', known)
                for split, rows in splits.items():
                    selected = [r for r in rows if split == 'test' or r['intent'] in known]
                    dump(folder / 'gate' / f'{split}.json', selected)
                    runtime = 'oos' if dataset == 'clinc150' else dataset
                    export = folder / 'textoir' / runtime / f'{"dev" if split == "val" else split}.tsv'
                    export.parent.mkdir(parents=True, exist_ok=True)
                    with export.open('w') as stream:
                        writer = csv.writer(stream, delimiter='\t')
                        writer.writerow(('text', 'label'))
                        writer.writerows((r['text'], r['intent']) for r in selected)
                units.append(dict(dataset=dataset, kir=kir, seed=seed, path=str(folder)))
    return units


def downstream_views(folder):
    known = json.loads((folder / 'known_labels.json').read_text())
    train = json.loads((folder / 'gate/train.json').read_text())
    domains = sorted({r['domain'] for r in train})
    domain_map = {d:i for i,d in enumerate(domains)}
    dump(folder / 'router/domain_map.json', domain_map)
    for domain in domains:
        intents = sorted({r['intent'] for r in train if r['domain'] == domain})
        mapping = {name:i for i,name in enumerate(intents)}
        dump(folder / 'experts' / domain / 'intent_map.json', mapping)
        for split in ('train','val'):
            rows = json.loads((folder / 'gate' / f'{split}.json').read_text())
            dump(folder / 'experts' / domain / f'{split}.json',
                 [{**r, 'label':mapping[r['intent']]} for r in rows if r['domain']==domain])
    for split in ('train','val'):
        rows = json.loads((folder / 'gate' / f'{split}.json').read_text())
        assert all(r['intent'] in known and r['label']==0 for r in rows)
        dump(folder / 'router' / f'{split}.json', [{**r,'label':domain_map[r['domain']]} for r in rows])


def coverage_geometry(datasets, kirs):
    """Restore the historical 90% Known coverage selection on the new data."""
    import numpy as np
    import torch
    from scripts.experiments import finalize_historical_known_coverage as coverage
    from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder

    output = ART / 'coverage_repair'
    contract = dict(coverage_target=0.90, seeds=SEEDS,
        selection='shared recipe/geometry by mean Known utility; per-seed Known quantile threshold',
        source='scripts/experiments/finalize_historical_known_coverage.py',
        reason='dense KIR runner used unrestricted utility instead of historical coverage calibration',
        real_oos_used=False, pseudo_oos_used=False, test_read=False,
        prior_test_results_exist=True,
        stackoverflow_policy='fixed K1/euclidean/mean_std_1.5/normalized_union/threshold0.9')
    dump(output / 'contract.json', contract)
    coverage.COVERAGE_TARGET = 0.90
    coverage.FIXED_THRESHOLD = None
    grid = {name: getattr(coverage, name) for name in
            ('K_VALUES', 'DISTANCES', 'LAMBDA_VALUES', 'BOUNDARY_QUANTILES', 'FUSIONS', 'FUSION_WEIGHTS')}
    torch.set_num_threads(4)
    device = torch.device('cuda')
    for dataset in datasets:
        for name, value in grid.items():
            setattr(coverage, name, value)
        coverage.FIXED_THRESHOLD = None
        dataset_contract = dict(contract)
        if dataset == 'stackoverflow':
            # The historical final StackOverflow method used this fixed rule.
            coverage.K_VALUES = (1,)
            coverage.DISTANCES = ('euclidean',)
            coverage.LAMBDA_VALUES = (1.5,)
            coverage.BOUNDARY_QUANTILES = ()
            coverage.FUSIONS = ('none',)
            coverage.FUSION_WEIGHTS = (0.,)
            coverage.FIXED_THRESHOLD = .9
            dataset_contract.update(coverage_target=None, fixed_threshold=.9,
                selection='shared recipe by mean Known utility; historical fixed StackOverflow geometry')
        for kir in kirs:
            tag = f'{dataset}_kir{round(kir*100):02d}'
            if all((output / 'locks' / f'{tag}_seed{s}.json').exists() for s in SEEDS):
                continue
            screening = ART / 'training' / f'{tag}_screening.json'
            if not screening.exists() and dataset == 'banking77':
                screening = ART / 'training' / f'kir{round(kir*100):02d}_screening.json'
            recipes = json.loads(screening.read_text())
            choices = []
            for recipe in recipes:
                groups, records = [], []
                for seed in SEEDS:
                    unit = f'{tag}_seed{seed}'
                    record = json.loads((ART / 'training' / f'{unit}_{recipe["name"]}.json').read_text())
                    records.append(record)
                    cached = output / 'validation' / f'{unit}_{recipe["name"]}.json'
                    if cached.exists():
                        candidates = json.loads(cached.read_text())
                    else:
                        folder = ART / 'data' / dataset / f'kir{round(kir*100):02d}_seed{seed}/gate'
                        train, val = [json.loads((folder / f'{s}.json').read_text()) for s in ('train', 'val')]
                        assert train and val and all(r['label'] == 0 for r in train + val)
                        encoder = _RacalGateEncoder(ROOT.parent / 'assets/models/all-MiniLM-L6-v2',
                                                   Path(record['checkpoint']), device)
                        arrays = [encoder.encode([r['text'] for r in rows], batch_size=256)
                                  for rows in (train, val)]
                        candidates = coverage.search_known(*arrays, train, val)
                        if dataset == 'stackoverflow':
                            candidates = [c for c in candidates if c['rule'] == 'normalized_union']
                        dump(cached, candidates)
                        del encoder
                        torch.cuda.empty_cache()
                    groups.append(candidates)
                    print(f'KNOWN {unit}/{recipe["name"]}: {len(candidates)} candidates', flush=True)
                keys = [coverage.geometry_key(c) for c in groups[0]]
                assert all([coverage.geometry_key(c) for c in g] == keys for g in groups)
                utilities = np.mean([[c['utility'] for c in g] for g in groups], axis=0)
                wrong = np.mean([[c['known_wrong_accept_rate'] for c in g] for g in groups], axis=0)
                index = max(range(len(keys)), key=lambda i: (utilities[i], -wrong[i]))
                choices.append(dict(records=records, geometry=[g[index] for g in groups],
                                    utility=float(utilities[index]),
                                    coverage=float(np.mean([g[index]['known_coverage'] for g in groups]))))
            winner = max(choices, key=lambda c: (c['utility'], -c['coverage']))
            dump(output / f'{tag}_selection.json', choices)
            for record, geometry in zip(winner['records'], winner['geometry']):
                dump(output / 'locks' / f'{tag}_seed{record["seed"]}.json',
                     {**record, **dataset_contract, 'geometry': geometry, 'scoring_family': 'coverage'})
            print(f'LOCK {tag}: shared Known utility={winner["utility"]:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'prepare-other', 'train', 'geometry', 'geometry-coverage'), required=True)
    parser.add_argument('--kirs', nargs='+', type=float, default=list(KIRS))
    parser.add_argument('--datasets', nargs='+', choices=('banking77', 'clinc150', 'stackoverflow'),
                        default=['banking77'])
    args = parser.parse_args()
    if args.stage in ('geometry', 'geometry-coverage'):
        coverage_geometry(args.datasets, args.kirs)
        return
    if args.stage == 'prepare-other':
        units = other_views(ART / 'data')
        for folder in (ART / 'data').glob('*/kir*'):
            downstream_views(folder)
        dump(ART / 'other_data_manifest.json', dict(units=units, status='data_prepared'))
        print(f'Prepared {len(units)} CLINC150/StackOverflow views and downstream training views')
        return
    if args.stage == 'prepare':
        units = banking_views(ROOT.parent / 'textoir/data/banking', ART / 'data')
        dump(ART / 'MANIFEST.json', dict(status='banking_data_prepared',
             protocol='kir_sensitivity_known_only', source='textoir/data/banking',
             seeds=SEEDS, kirs=KIRS, units=units, final_units_completed=0,
             real_oos_used_for_selection=False, pseudo_oos_used=False,
             test_used_for_selection=False))
        print(f'Prepared {len(units)} BANKING77 shared views', flush=True)
        return
    from scripts.experiments import search_historical_known_training as training
    import torch
    training.DATA_ROOT = ART / 'data'
    training.ART = ART / 'checkpoints'
    training.OUT = ART / 'training'
    training.OUT.mkdir(parents=True, exist_ok=True)
    training.ART.mkdir(parents=True, exist_ok=True)
    dump(ART / 'training_contract.json', dict(recipes=training.RECIPES,
         selection='Known validation macro F1; stable recipe order; earliest epoch',
         screening_seed=42, expansion_seeds=[13,87], test_read=False))
    torch.set_num_threads(4)
    device = torch.device('cuda')
    torch.zeros(1, device=device)
    for dataset, kir in ((d, k) for d in args.datasets for k in KIRS):
        records = [training.train_cell(dataset, kir, 42, name, device)
                   for name in training.RECIPES]
        winners = sorted(records, key=lambda r: -r['known_validation_f1'])[:2]
        dump(ART / 'training' / f'{dataset}_kir{round(kir*100):02d}_screening.json', winners)
        for seed in (13,87):
            for winner in winners:
                training.train_cell(dataset, kir, seed, winner['name'], device)
    dump(ART / 'training_complete.json', dict(status='banking_training_complete',
         final_units_completed=0, test_read=False))


if __name__ == '__main__':
    main()
