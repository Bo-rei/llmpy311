"""Known-only calibration of existing H1 K1 checkpoints; test after global lock."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.eval.run_historical_trainable_parameter_full_pipeline import (
    DATA_ROOT, MODEL_ROOT, DEFAULT_H1_ROOT, _fit_detector, _metrics_from_output,
    EXTERNAL_OOS_F1_BY_KIR, write_csv,
)
from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder
from tools.eval.run_historical_trainable_extended_boundary_search import _distance_matrix, _score_from_distances

OUT = ROOT / 'results/analysis/historical_known_validation'


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def choose_threshold(scores, predicted, truth):
    """Maximize Known-class macro F1; smallest threshold wins exact ties."""
    labels = sorted(set(truth))
    candidates = np.unique(scores)
    best = (-1., 0.)
    for threshold in candidates:
        pred = np.where(scores <= threshold, predicted, '__oos__')
        score = f1_score(truth, pred, labels=labels, average='macro', zero_division=0)
        if score > best[0] + 1e-12:
            best = (float(score), float(threshold))
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    torch.zeros(1, device=device)
    manifest = dict(protocol='historical_v19_paper_main', evidence='H1 Gate-only',
                    status='running', device='cuda', datasets=['clinc150'],
                    kirs=[.25, .5, .75], seeds=[13, 42, 87],
                    checkpoint_selection='existing known_only_validation_for_checkpoint',
                    primary='known_macro_f1', secondary='known_coverage_95',
                    k=1, radius_lambda=1., acceptance_mode='nearest_sphere',
                    validation_oos_used=False, test_used_for_selection=False,
                    test_access='after_all_nine_calibration_locks',
                    rerun_reason='new Known-only calibration replacing fixed threshold 1')
    dump('MANIFEST.json', manifest)
    locked = []
    for kir in manifest['kirs']:
        for seed in manifest['seeds']:
            tag = f'kir{round(kir*100):02d}_seed{seed}'
            data = DATA_ROOT / 'clinc150' / tag / 'gate'
            checkpoint = DEFAULT_H1_ROOT / 'clinc150' / tag / 'trainable_k1/checkpoint.pt'
            source = json.loads(checkpoint.with_name('run_manifest.json').read_text())
            assert source['selection'] == 'known_only_validation_for_checkpoint'
            train = json.loads((data / 'train.json').read_text())
            val = [r for r in json.loads((data / 'val.json').read_text()) if int(r['label']) == 0]
            assert train and val and all(int(r['label']) == 0 for r in train)
            encoder = _RacalGateEncoder(MODEL_ROOT / 'all-MiniLM-L6-v2', checkpoint, device)
            detector = _fit_detector(encoder.encode([r['text'] for r in train]), train, 1, 1., 'nearest_sphere')
            output = _score_from_distances(_distance_matrix(detector, encoder.encode([r['text'] for r in val])), detector, 'nearest_sphere')
            pred = np.array([detector.cluster_to_intent[int(i)] for i in output['nearest_cluster']])
            score, threshold = choose_threshold(output['score'], pred, [r['intent'] for r in val])
            thresholds = dict(known_macro_f1=threshold, known_coverage_95=float(np.quantile(output['score'], .95, method='higher')))
            state = OUT / f'{tag}_detector.json'
            detector.save(state)
            locked.append(dict(dataset='clinc150', kir=kir, seed=seed, checkpoint=str(checkpoint),
                               detector=str(state), thresholds=thresholds, validation_known_f1=score,
                               validation_known_count=len(val)))
            dump('selection_lock.json', locked)
            print(f'LOCK {tag} {thresholds}', flush=True)
            del encoder
            torch.cuda.empty_cache()
    # No test file is opened above this point.
    results = []
    for choice in locked:
        from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
        detector = MultiSphereOOSDetector()
        detector.load(Path(choice['detector']))
        detector.cluster_to_intent = {int(k): v for k, v in detector.cluster_to_intent.items()}
        encoder = _RacalGateEncoder(MODEL_ROOT / 'all-MiniLM-L6-v2', Path(choice['checkpoint']), device)
        tag = f"kir{round(choice['kir']*100):02d}_seed{choice['seed']}"
        rows = json.loads((DATA_ROOT / 'clinc150' / tag / 'gate/test.json').read_text())
        output = _score_from_distances(_distance_matrix(detector, encoder.encode([r['text'] for r in rows])), detector, 'nearest_sphere')
        for method, threshold in choice['thresholds'].items():
            metrics = _metrics_from_output(detector, output, rows, threshold)
            results.append(dict(dataset='clinc150', kir=choice['kir'], seed=choice['seed'], method=method,
                                threshold=threshold, **metrics))
            print(f'TEST {tag} {method} OOS={metrics["f1_u"]:.5f}', flush=True)
        del encoder
        torch.cuda.empty_cache()
    write_csv(OUT / 'per_seed.csv', results)
    summaries = []
    for kir in manifest['kirs']:
        for method in ('known_macro_f1', 'known_coverage_95'):
            group = [r for r in results if r['kir'] == kir and r['method'] == method]
            row = dict(dataset='clinc150', kir=kir, method=method, seeds=3, std_ddof=0)
            for metric in ('f1_u', 'f1_k', 'accuracy', 'known_recall', 'false_accept_rate', 'auroc'):
                row[metric+'_mean'] = float(np.mean([r[metric] for r in group]))
                row[metric+'_std'] = float(np.std([r[metric] for r in group]))
            row['external_reference'] = EXTERNAL_OOS_F1_BY_KIR['clinc150', kir]
            row['delta_pp'] = 100*row['f1_u_mean'] - row['external_reference']
            summaries.append(row)
    write_csv(OUT / 'summary.csv', summaries)
    manifest.update(status='complete', completed_units=9, full_pipeline_verified=False)
    dump('MANIFEST.json', manifest)


if __name__ == '__main__':
    main()
