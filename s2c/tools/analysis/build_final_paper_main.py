"""Export only manifest-backed final predictions; leave missing cells unavailable."""
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.experiments.run_kir_sensitivity_known_only import dump
from tools.eval.final_prediction_metrics import prediction_metrics

OUT = ROOT / 'results/final_paper_main'
ALIGNED_ART = ROOT.parent / 'artifacts' / 's2c' / 'analysis' / 'banking_textoir_aligned'
SELECTION = ALIGNED_ART / 'fixed_k1_mahalanobis_threshold1_known_only'
MOGB = ALIGNED_ART.parent / 'mogb_shared_official_aligned'
KIRS = (.25,.5,.75)
SEEDS = (13,42,87)
DATASETS = ('banking77','clinc150','stackoverflow')
MOGB_PUBLIC = {
    'stackoverflow': {0.25: 94.42, 0.50: 89.71, 0.75: 75.52},
    'banking77': {0.25: 88.29, 0.50: 81.04, 0.75: 71.27},
}
MOGB_PUBLIC_SOURCE = 'https://ojs.aaai.org/index.php/AAAI/article/view/34630/36785'


def csv_write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_banking():
    source = ROOT.parent / 'textoir/data/banking'
    raw = {}
    for split in ('train','dev','test'):
        with (source/f'{split}.tsv').open() as stream:
            raw[split] = list(csv.DictReader(stream, delimiter='\t'))
    labels = sorted({r['label'] for r in raw['train']})
    assert len(labels) == 77
    audit = []
    for kir in KIRS:
        for seed in SEEDS:
            view = ALIGNED_ART/'data/banking77'/f'kir{round(kir*100):02d}_seed{seed}'
            known = json.loads((view/'known_labels.json').read_text())
            native = np.random.RandomState(seed).choice(labels, round(77*kir), replace=False).tolist()
            counts = {}
            for split, rows in raw.items():
                expected = [(r['text'],r['label']) for r in rows if split == 'test' or r['label'] in known]
                name = 'val' if split == 'dev' else split
                actual = json.loads((view/'gate'/f'{name}.json').read_text())
                # The canonical exporter preserves rows but may use a different
                # order for the test view than the raw TextOIR TSV.  Audit exact
                # membership/content, not an incidental serialization order.
                assert Counter((r['text'],r['intent']) for r in actual) == Counter(expected)
                assert all(r['label'] == int(r['intent'] not in known) for r in actual)
                export_root = (ROOT/'data'/'exports'/'protocol_v2_textoir_v1'
                               /'k_plus_1_way'/'banking77'
                               /f'seed_{seed}'/f'kir_{kir:.2f}')
                exported = json.loads((export_root/f'{split}.json').read_text())
                assert Counter(
                    (r['text'],r.get('original_intent',r['label'])) for r in exported
                ) == Counter(expected)
                assert all(
                    r['label'] == (r.get('original_intent', r['label'])
                                   if r.get('original_intent', r['label']) in known else '__oos__')
                    for r in exported
                )
                counts[name] = dict(known=sum(r['label']==0 for r in actual), oos=sum(r['label']==1 for r in actual))
            audit.append(dict(kir=kir,seed=seed,source=str(source),known_labels=known,
                native_textoir_known_labels=native, native_sampling_match=set(known)==set(native),
                shared_textoir_export_exact_match=True, counts=counts))
    return dict(dataset='banking77', intent_count=77, native_oos=False,
        oos_source='held-out Banking intents only', cells=audit,
        alignment='TextOIR source and shared known_labels override; not native NumPy split sampling',
        historical_excluded='banking77_oos')


def references():
    source = ROOT.parent/'textoir/open_intent_detection/results/results.md'
    rows = []
    methods = ('MSP','OpenMax','DOC','DeepUnk','KNNCL','ADB','DA-ADB')
    for line in source.read_text().splitlines():
        p = [v.strip() for v in line.split('|')[1:-1]]
        if len(p) != 8 or p[0] != 'banking' or p[1] not in methods:
            continue
        if float(p[3]) != 1.0:
            continue
        rows.append(dict(dataset='banking77',method=p[1],kir=float(p[2]),
            known_f1=float(p[4]),oos_f1=float(p[5]),accuracy=float(p[7]),
            source=str(source),comparison='reported reference; not shared-seed rerun',
            supervision='not yet audited per method',backbone='not yet audited per method'))
    return rows


def main():
    import argparse
    global OUT, SELECTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-root', type=Path, default=SELECTION)
    parser.add_argument('--output-root', type=Path, default=OUT)
    args = parser.parse_args()
    SELECTION, OUT = args.selection_root, args.output_root
    selection_manifest_path = SELECTION/'MANIFEST.json'
    contract = json.loads(selection_manifest_path.read_text()) if selection_manifest_path.exists() else {}
    selection_description = contract.get(
        'selection',
        'Ours uses Known train/dev selection and a pre-declared fixed geometry; test deferred.',
    )
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT/'dataset_audit.json', audit_banking())
    records, status = [], []
    for method in ('Ours','MOGB-official-compatible'):
        for dataset in (('banking77',) if method == 'Ours' else DATASETS):
            for kir in KIRS:
                for seed in SEEDS:
                    tag = f'kir{round(kir*100):02d}_seed{seed}'
                    if method == 'Ours':
                        path = SELECTION/'final'/dataset/tag/'Ours.json'
                        predictions = path.with_suffix('.npz')
                    else:
                        path = MOGB/dataset/tag/'result.json'
                        predictions = path.parent/'predictions.npz'
                    manifest_path = path.parent/'MANIFEST.json'
                    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                    if not path.exists():
                        status.append(dict(method=method,dataset=dataset,kir=kir,seed=seed,
                            status=manifest.get('status','pending'),reason=manifest.get('failure',''),source=str(manifest_path)))
                        continue
                    if method == 'MOGB-official-compatible' and manifest.get('status') != 'complete':
                        status.append(dict(method=method,dataset=dataset,kir=kir,seed=seed,
                            status=manifest.get('status','pending'),
                            reason='result exists without a complete cell manifest',
                            source=str(manifest_path)))
                        continue
                    if not predictions.exists():
                        status.append(dict(method=method,dataset=dataset,kir=kir,seed=seed,
                            status='incomplete',reason='prediction array is missing',source=str(path)))
                        continue
                    source = json.loads(path.read_text())
                    if method == 'Ours':
                        assert source.get('full_pipeline') is True
                    else:
                        # MOGB is its own end-to-end open-intent classifier;
                        # it is not the Ours Gate->Router->Expert cascade.
                        assert source.get('task', 'end_to_end_open_intent_classifier') == (
                            'end_to_end_open_intent_classifier')
                    data_root = ALIGNED_ART if method == 'Ours' else MOGB
                    data = data_root/'data'/dataset/tag
                    known = json.loads((data/'known_labels.json').read_text())
                    values = np.load(predictions, allow_pickle=False)
                    test = json.loads((data/'gate/test.json').read_text())
                    expected = np.asarray([r['intent'] if r['intent'] in known else '__oos__' for r in test])
                    assert np.array_equal(expected, values['y_true']), f'Prediction alignment: {path}'
                    metrics = prediction_metrics(values['y_true'],values['y_pred'],known)
                    for key in ('oos_f1','known_f1','accuracy'):
                        assert abs(metrics[key]-source['metrics'][key]) < 1e-10
                    record = dict(dataset=dataset,method=method,kir=kir,seed=seed,
                                  source=str(path),predictions=str(predictions))
                    record.update({k:(v*100 if not isinstance(v,int) else v) for k,v in metrics.items()})
                    if 'score' in values:
                        for label, mask in [('known', expected!='__oos__'),('oos',expected=='__oos__')]:
                            for q in (.1,.5,.9):
                                record[f'{label}_score_q{int(q*100)}'] = float(np.quantile(values['score'][mask],q))
                    records.append(record)
                    status.append(dict(method=method,dataset=dataset,kir=kir,seed=seed,status='complete',reason='',source=str(path)))
    csv_write(OUT/'per_seed.csv',records, None if records else ['dataset','method','kir','seed','oos_f1','known_f1','accuracy','source'])
    csv_write(OUT/'run_status.csv',status)
    summaries = []
    for method in ('Ours','MOGB-official-compatible'):
        for dataset in (('banking77',) if method == 'Ours' else DATASETS):
            for kir in KIRS:
                group = [r for r in records if (r['method'],r['dataset'],r['kir']) == (method,dataset,kir)]
                row = dict(method=method,dataset=dataset,kir=kir,n=len(group),status='complete' if len(group)==3 else 'unavailable')
                if len(group)==3:
                    for metric in ('oos_f1','known_f1','accuracy','known_recall','false_accept_rate','oos_recall'):
                        v = [r[metric] for r in group]
                        row[metric+'_mean'],row[metric+'_std'] = float(np.mean(v)),float(np.std(v))
                summaries.append(row)
    csv_write(OUT/'summary.csv',summaries)
    refs = references()
    csv_write(OUT/'banking_reported_references.csv',refs)
    comparison = []
    for row in summaries:
        if row['n'] != 3:
            continue
        method = row['method']
        comparison.append(dict(
            dataset=row['dataset'], method=method, kir=row['kir'], n=row['n'],
            oos_f1_mean=row['oos_f1_mean'], oos_f1_std=row['oos_f1_std'],
            known_f1_mean=row['known_f1_mean'], known_f1_std=row['known_f1_std'],
            accuracy_mean=row['accuracy_mean'], accuracy_std=row['accuracy_std'],
            known_recall_mean=row['known_recall_mean'],
            false_accept_rate_mean=row['false_accept_rate_mean'],
            source=('full-pipeline Ours; shared canonical split'
                    if method == 'Ours' else
                    'official MOGB-compatible; shared canonical split'),
            backbone=('MiniLM' if method == 'Ours' else 'BERT-base-uncased'),
            supervision='Known-only train/dev; test deferred',
            comparison_class='shared_split_three_seed',
            eligible_for_aligned_main='true',
        ))
    # Reported TextOIR rows stay visible but are explicitly ineligible for a
    # shared-seed main ranking because their original split/seed contract is
    # not the same as the canonical exports.
    for row in refs:
        comparison.append(dict(
            dataset=row['dataset'], method='TextOIR-' + row['method'], kir=row['kir'],
            n='', oos_f1_mean=row['oos_f1'], oos_f1_std='',
            known_f1_mean=row['known_f1'], known_f1_std='',
            accuracy_mean=row['accuracy'], accuracy_std='',
            known_recall_mean='', false_accept_rate_mean='', source=row['source'],
            backbone='BERT/TextOIR', supervision='source-specific; not yet audited',
            comparison_class='reported_reference', eligible_for_aligned_main='false',
        ))
    csv_write(OUT/'comparison.csv', comparison)
    public_rows = []
    for dataset, by_kir in MOGB_PUBLIC.items():
        for kir, value in by_kir.items():
            public_rows.append(dict(
                dataset=dataset, method='MOGB-public-paper', kir=kir,
                oos_f1=value, metric='F1-U', source=MOGB_PUBLIC_SOURCE,
                source_table='AAAI 2025 Table 2',
                comparison_class='published_reference',
                eligible_for_aligned_main='false',
                note='Published MOGB contract; not a matched-seed rerun under this canonical split',
            ))
    csv_write(OUT/'mogb_public_comparison.csv', public_rows)
    dump(OUT/'config_lock.json',dict(selection_contract=str(selection_manifest_path),
        status='locked' if (SELECTION/'selection_complete.json').exists() else 'pending',
        selection=contract,
        per_seed_lock_root=str(SELECTION/'locks')))
    complete = sum(r['status']=='complete' for r in status)
    dump(OUT/'MANIFEST.json',dict(status='complete' if complete==36 else 'incomplete',
        completed_units=complete,planned_units=36,protocol='shared_known_labels_textoir_source',
        dataset_audit=str(OUT/'dataset_audit.json'),selection_root=str(SELECTION),mogb_root=str(MOGB),
        primary_banking_source='summary.csv; only complete three-seed Ours rows',
        historical_banking77_oos_excluded=True,test_used_for_selection=False,
        test_previously_observed=True,std_ddof=0,units='percent except counts and scores',
        reference_policy='reported baselines are separate; shared-seed superiority requires matched predictions',
        comparison_csv=str(OUT/'comparison.csv'),
        mogb_public_comparison_csv=str(OUT/'mogb_public_comparison.csv'),
        mogb_public_source=MOGB_PUBLIC_SOURCE))
    lines = ['# Final experiment comparison','',
        '| Dataset | Method | KIR | n | OOS F1 | Known F1 | Accuracy |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        cells = [f"{r[m+'_mean']:.2f}±{r[m+'_std']:.2f}" if r['n']==3 else 'unavailable'
                 for m in ('oos_f1','known_f1','accuracy')]
        lines.append(f"| {r['dataset']} | {r['method']} | {r['kir']} | {r['n']} | " + ' | '.join(cells)+' |')
    lines += ['', 'Reported Banking77 references use different seed/split contracts; see banking_reported_references.csv.',
              'MOGB is eligible only after all 27 official-compatible cells complete with shared predictions.',
              selection_description]
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=complete,planned=36,output=str(OUT))))


if __name__ == '__main__':
    main()
