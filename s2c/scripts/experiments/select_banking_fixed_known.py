"""Lock a simple, pure Known-dev Banking control before any test evaluation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.run_kir_sensitivity_known_only import ART, dump

OUT = ART.parent / 'banking_fixed_known'


def main():
    import torch
    from scripts.experiments.finalize_historical_known_coverage import _fit_and_score, _quantile
    from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder
    from scripts.experiments.search_historical_known_training import RECIPES

    config = dict(k=1, distance='mahalanobis_diag', boundary='mean_std_1.0',
                  rule='normalized_union', fusion='none', fusion_weight=0., coverage=.9)
    contract = dict(dataset='banking77', kirs=[.25,.5,.75], seeds=[13,42,87],
        config=config, recipe=RECIPES['last2'],
        selection='fixed K1/diagonal Mahalanobis/no fusion; 90% Known-dev coverage',
        justification='user-requested simple control; geometry fixed before this evaluation',
        real_oos_used=False, pseudo_oos_used=False, test_used_for_selection=False,
        test_previously_observed=True, role='prespecified_control_not_test_selected_winner')
    path = OUT / 'selection_contract.json'
    if path.exists():
        assert json.loads(path.read_text()) == contract
    else:
        dump(path, contract)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.zeros(1, device=device)
    for kir in contract['kirs']:
        for seed in contract['seeds']:
            tag = f'kir{round(100*kir):02d}_seed{seed}'
            lock_path = OUT/'locks'/f'banking77_{tag}.json'
            if lock_path.exists():
                continue
            record_path = ART/'training'/f'banking77_{tag}_last2.json'
            if not record_path.exists():
                # Outer encoders there were trained on all Known classes; only
                # the separate geometry selector used intent holdouts.
                record_path = ART.parent/'banking_known_holdout/training_records'/f'banking77_{tag}_last2.json'
            record = json.loads(record_path.read_text())
            assert record['recipe'] == RECIPES['last2']
            assert record['dataset'] == 'banking77' and record['kir'] == kir and record['seed'] == seed
            assert record['real_oos_used'] is False and record['pseudo_oos_used'] is False
            assert record['test_read'] is False
            source = ART/'data/banking77'/tag/'gate'
            train, dev = [json.loads((source/f'{name}.json').read_text()) for name in ('train','val')]
            known = {row['intent'] for row in train}
            assert all(row['label'] == 0 and row['intent'] in known for row in train+dev)
            encoder = _RacalGateEncoder(ROOT.parent/'assets/models/all-MiniLM-L6-v2',
                                        Path(record['checkpoint']), device)
            tv, vv = [encoder.encode([row['text'] for row in rows], batch_size=128)
                      for rows in (train, dev)]
            _, result = _fit_and_score(tv, train, vv, config)
            threshold = _quantile(result['score'], config['coverage'])
            dump(lock_path, dict(checkpoint=record['checkpoint'], training_record=str(record_path),
                geometry={**config,'threshold':threshold}, scoring_family='coverage',
                selection_contract=str(path), real_oos_used=False, pseudo_oos_used=False,
                test_read=False, test_previously_observed=True,
                known_dev_coverage=float((result['score'] <= threshold).mean())))
            print(f'LOCKED banking77/{tag} CUDA threshold={threshold:.6f}', flush=True)
            del encoder
            torch.cuda.empty_cache()
    dump(OUT/'selection_complete.json', dict(status='locked', units=9, test_read=False))


if __name__ == '__main__':
    main()
