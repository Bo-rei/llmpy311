"""Evaluate the locked Gate and matched Router/Experts after all selections."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.experiments.run_kir_sensitivity_known_only import ART,KIRS,SEEDS,dump


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=['banking77','clinc150','stackoverflow'])
    parser.add_argument('--kirs',nargs='+',type=float,default=list(KIRS))
    parser.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    parser.add_argument('--gate-only',action='store_true',
                        help='Write Gate-only final test artifacts without requiring downstream components')
    parser.add_argument('--selection-root', type=Path, default=ART/'coverage_repair')
    parser.add_argument('--data-root', type=Path, default=ART/'data')
    parser.add_argument('--components-root', type=Path, default=ART/'components')
    args=parser.parse_args()
    selection_root=args.selection_root
    data_root=args.data_root
    components_root=args.components_root
    import numpy as np
    import torch
    from sklearn.metrics import f1_score,accuracy_score
    from tools.eval.final_prediction_metrics import prediction_metrics
    from scripts.experiments.finalize_historical_known_search import replay
    from tools.eval.run_historical_trainable_full_pipeline import _RacalGateEncoder
    from legacy.pipeline.system_pipeline import HiLSAMoEV19Pipeline,PipelinePaths
    datasets=tuple(args.datasets)
    kirs=tuple(args.kirs)
    seeds=tuple(args.seeds)
    cells=[(d,k,s) for d in datasets for k in kirs for s in seeds]
    for d,k,s in cells:
        tag=f'kir{round(k*100):02d}_seed{s}'
        assert (selection_root/'locks'/f'{d}_{tag}.json').exists(), 'Missing Gate selection'
        if not args.gate_only:
            assert (components_root/d/tag/'selection_complete.json').exists(), 'Missing downstream selection'
    device=torch.device('cuda')
    torch.set_num_threads(4)
    models=ROOT.parent/'assets/models'
    for d,k,s in cells:
        tag=f'kir{round(k*100):02d}_seed{s}'
        result=selection_root/'final'/d/tag/('Ours_gate.json' if args.gate_only else 'Ours.json')
        if result.exists():continue
        lock=json.loads((selection_root/'locks'/f'{d}_{tag}.json').read_text())
        data=data_root/d/tag
        train=json.loads((data/'gate/train.json').read_text())
        test=json.loads((data/'gate/test.json').read_text())
        print(f'LOAD {d}/{tag}',flush=True)
        encoder=_RacalGateEncoder(models/'all-MiniLM-L6-v2',Path(lock['checkpoint']),device)
        tv,ev=[encoder.encode([r['text'] for r in rows],batch_size=256) for rows in (train,test)]
        print(f'GATE {d}/{tag} n_test={len(test)}',flush=True)
        if lock.get('scoring_family') == 'coverage':
            from scripts.experiments.finalize_historical_known_coverage import _fit_and_score
            _,out=_fit_and_score(tv,train,ev,lock['geometry'])
        else:
            _,out=replay(tv,train,ev,lock['geometry'])
        rejected=out['score']>lock['geometry']['threshold']
        if args.gate_only:
            truth=np.asarray([int(r['label']) == 1 for r in test],dtype=bool)
            metrics=dict(
                oos_f1=float(f1_score(truth,rejected)),
                gate_accuracy=float((truth == rejected).mean()),
                known_recall=float((~rejected[~truth]).mean()),
                false_accept_rate=float((~rejected[truth]).mean()),
            )
            result.parent.mkdir(parents=True,exist_ok=True)
            dump(result,dict(dataset=d,kir=k,seed=s,method='Ours-Gate',metrics=metrics,
                             selection=lock,full_pipeline=False,test_read=True))
            np.savez_compressed(result.with_suffix('.npz'),
                                y_true=truth.astype(np.int64),y_pred=rejected.astype(np.int64))
            print(d,tag,metrics,flush=True)
            del encoder
            torch.cuda.empty_cache()
            continue
        component=json.loads((components_root/d/tag/'selection_complete.json').read_text())
        paths=PipelinePaths(model_path=models/'smollm135m',gate_encoder_path=models/'all-MiniLM-L6-v2',
            gate_detector_path=data/'unused_detector.json',router_ckpt_path=Path(component['router']),
            experts_root=Path(component['experts']),experts_data_root=data/'experts',
            router_data_path=data/'router/train.json',gate_train_path=data/'gate/train.json')
        pipeline=HiLSAMoEV19Pipeline(paths,device='cuda',max_length=64,semantic_gate_enabled=False,gate_mode='multisphere')
        pipeline._load_tokenizer();pipeline._load_router();pipeline._load_domain_mapping();pipeline._index_experts()
        accepted=np.flatnonzero(~rejected)
        pred=np.full(len(test),'__oos__',dtype=object)
        if len(accepted):
            texts=[test[i]['text'] for i in accepted]
            route=pipeline._router_predict(texts,batch_size=32)
            domains=[pipeline.domain_id_to_name[int(i)] for i in route['domain_ids']]
            for domain in sorted(set(domains)):
                ids=[i for i,name in enumerate(domains) if name==domain]
                outputs=pipeline._expert_predict_group(domain,[texts[i] for i in ids],batch_size=32)
                for i,name in zip(ids,outputs['intent_names']):pred[accepted[i]]=name
        print(f'CASCADE {d}/{tag} accepted={len(accepted)}',flush=True)
        gold=np.array(['__oos__' if r['label'] else r['intent'] for r in test])
        known=sorted({r['intent'] for r in train})
        truth=gold=='__oos__'
        assert np.array_equal(pred == '__oos__', rejected), 'Cascade OOS decisions differ from Gate'
        metrics=prediction_metrics(gold,pred,known)
        result.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(result.with_suffix('.npz'),y_true=gold.astype(str),y_pred=pred.astype(str),
                            score=out['score'])
        dump(result,dict(dataset=d,kir=k,seed=s,method='Ours',metrics=metrics,selection=lock,
                         components=component,data_root=str(data),full_pipeline=True,device=str(device),
                         predictions=str(result.with_suffix('.npz')),test_used_for_selection=False))
        print(d,tag,metrics,flush=True)
        del pipeline,encoder
        torch.cuda.empty_cache()


if __name__=='__main__':main()
