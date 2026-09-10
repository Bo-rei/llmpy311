"""Evaluate the paper's four structural/backbone variants around current Ours."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import argparse
import csv
import json
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoModel
from sklearn.metrics import f1_score
from tools.eval import run_historical_trainable_parameter_full_pipeline as gate
from tools.eval import run_historical_trainable_full_pipeline as full
from tools.eval.eval_minilm_cascade_v19 import _train_heads, _predict_router_experts
from tools.eval.eval_smollm_cascade_v19 import SmolLMCascadeEvaluator, SmolLMPrototypeGate, _fit_smollm_gate_centers

OUT = ROOT / 'results/analysis/trainable_paper_four_variants'
PAPER_GEOMETRY = False
DATASETS = ('clinc150', 'stackoverflow', 'banking77_oos')
VARIANTS = ('Ours', 'Without Gate', 'Cascade-MiniLM', 'Cascade-SmolLM')


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')


def ours_config(dataset, kir):
    checkpoint = full.DEFAULT_H1_ROOT / dataset / f'kir{int(kir*100)}_seed42/trainable_k1/checkpoint.pt'
    lam, threshold, mode = 1., 1., 'nearest_sphere'
    if PAPER_GEOMETRY:
        return dict(checkpoint=str(checkpoint), k=2, radius_lambda=.5 if dataset == 'clinc150' else 1., threshold=1., acceptance_mode='normalized_union')
    if dataset == 'clinc150' and kir == .5:
        lock = full.read_json(ROOT/'results/analysis/historical_trainable_gate_recipe_search/selection_lock.json')
        chosen = next(r for r in lock['choices'] if r['seed']==42)
        checkpoint = Path(chosen['checkpoint'])
        lam,threshold,mode = chosen['radius_lambda'],chosen['threshold'],chosen['acceptance_mode']
    elif dataset == 'stackoverflow' and kir == .5:
        lam,threshold = 2.,.85
    elif dataset == 'banking77_oos':
        lam,threshold,mode = { .25:(.5,.95,'nearest_sphere'), .5:(.75,.95,'normalized_union'), .75:(.75,1.,'normalized_union')}[kir]
    return dict(checkpoint=str(checkpoint),k=1,radius_lambda=lam,threshold=threshold,acceptance_mode=mode)


def downstream(pipeline, rows):
    texts=[r['text'] for r in rows]
    route=pipeline._router_predict(texts,batch_size=128)
    domains=[pipeline.domain_id_to_name[i] for i in route['domain_ids']]
    preds=[None]*len(rows)
    for domain in sorted(set(domains)):
        ids=[i for i,d in enumerate(domains) if d==domain]
        expert=pipeline._expert_predict_group(domain,[texts[i] for i in ids],batch_size=128)
        for j,i in enumerate(ids):
            preds[i]=dict(domain=domain,intent=expert['intent_names'][j],intent_prob=expert['intent_probs'][j],domain_prob=route['domain_probs'][i])
    return preds


def compose(down, scores, threshold):
    return [dict(p,is_oos=bool(s>threshold),gate_pred=int(s>threshold),gate_score=float(s)) for p,s in zip(down,scores,strict=True)]


def metrics(rows, predictions):
    m,_=full.compute_metrics(rows,predictions)
    known=sorted({r['intent'] for r in rows if r['label']==0})
    truth=['__oos__' if r['label'] else r['intent'] for r in rows]
    pred=['__oos__' if p['is_oos'] else p['intent'] for p in predictions]
    # Compute directly on every test sample; C is intent count, not sample count.
    m['known_f1']=float(f1_score(truth,pred,labels=known,average='macro',zero_division=0))
    return {k:100*float(m[k]) for k in ('known_f1','oos_f1','overall_accuracy','f1_all','known_recall','false_accept_rate')}


def select(rows, down, scores):
    sweep=[]
    for t in np.linspace(.2,.95,16):
        m=metrics(rows,compose(down,scores,t))
        sweep.append(dict(threshold=float(t),**m))
    best=max(sweep,key=lambda r:(r['f1_all'],r['overall_accuracy'],-r['threshold']))
    return best['threshold'],sweep


def run_cell(dataset,kir):
    cell=OUT/dataset/f'kir{int(kir*100)}_seed42'
    final=cell/'results.json'
    if final.exists():
        return full.read_json(final)
    torch.manual_seed(42); np.random.seed(42)
    device=torch.device('cuda')
    gate.KIR=full.KIR=kir
    full.CASCADE_ROOT=ROOT.parent/'artifacts/s2c/outputs/experiments/cascade_full'/f'gpu_kir{int(kir*100)}'
    cfg=ours_config(dataset,kir)
    save(cell/'config.json',dict(dataset=dataset,kir=kir,seed=42,ours=cfg,variant_names=VARIANTS,threshold_selection='validation full macro F1; grid .2:.05:.95',single_domain_smollm_gate='pretrained SmolLM backbone because router is constant',test_used_for_selection=False))
    views=gate._read_views(dataset,42)
    encoder=full._RacalGateEncoder(full.MODEL_ROOT/'all-MiniLM-L6-v2',Path(cfg['checkpoint']),device)
    xs={s:encoder.encode([r['text'] for r in views[s]],batch_size=128) for s in ('train','val','test')}
    detector=gate._fit_detector(xs['train'],views['train'],1,cfg['radius_lambda'],cfg['acceptance_mode'])
    scores={s:gate._vectorized_output(detector,xs[s])['score'] for s in ('val','test')}
    del encoder
    with tempfile.TemporaryDirectory(prefix='s2c_four_variants_') as tmp:
        path=Path(tmp)/'detector.json'
        gate._write_detector(path,detector,cfg['threshold'])
        pipeline=full._make_pipeline(dataset,42,device,path,full.DEFAULT_H1_ROOT)
        if dataset=='clinc150' and kir==.5:
            pipeline.gate_encoder=full._RacalGateEncoder(full.MODEL_ROOT/'all-MiniLM-L6-v2',Path(cfg['checkpoint']),device)
        direct=pipeline.predict_batch([r['text'] for r in views['test']],batch_size=128)
        ours=metrics(views['test'],direct)
        assert np.array_equal([p['is_oos'] for p in direct],scores['test']>cfg['threshold'])
        result=[dict(dataset=dataset,kir=kir,seed=42,variant='Ours',**ours)]
        # Run the same Router/Expert on all samples, including gate rejects.
        downs={s:downstream(pipeline,views[s]) for s in ('val','test')}
        confidence={s:np.array([1-p['intent_prob'] for p in downs[s]]) for s in downs}
        # Original Without Gate uses intent-confidence rejection.
        t,sweep=select(views['val'],downs['val'],confidence['val'])
        save(cell/'without_gate_selection.json',dict(threshold=t,sweep=sweep,score='1-expert confidence'))
        result.append(dict(dataset=dataset,kir=kir,seed=42,variant='Without Gate',**metrics(views['test'],compose(downs['test'],confidence['test'],t))))
        # Preserve the new MiniLM Gate; replace only downstream with the paper heads.
        router,experts=_train_heads(views['train'],xs['train'],42)
        d,dp,i,ip=_predict_router_experts(router,experts,xs['test'])
        mini=[dict(domain=a,intent=b) for a,b in zip(d,i,strict=True)]
        result.append(dict(dataset=dataset,kir=kir,seed=42,variant='Cascade-MiniLM',**metrics(views['test'],compose(mini,scores['test'],cfg['threshold']))))
        # Reuse the original SmolLM prototype-Gate pooling and center construction.
        if pipeline.router_model is None:
            base=AutoModel.from_pretrained(full.MODEL_ROOT/'smollm135m',local_files_only=True).to(device).eval()
            router_model=SimpleNamespace(base=base)
        else:
            router_model=pipeline.router_model
        embedder=SimpleNamespace(router_model=router_model,tokenizer=pipeline.tokenizer,device=device,batch_size=64,max_length=64)
        smol={s:SmolLMCascadeEvaluator.embed_texts(embedder,[r['text'] for r in views[s]]) for s in ('train','val')}
        centers,intents=_fit_smollm_gate_centers(smol['train'],views['train'],1)
        prototype=SmolLMPrototypeGate(centers,intents,1.)
        val_scores=prototype.predict(smol['val'])['score']
        t,sweep=select(views['val'],downs['val'],val_scores)
        save(cell/'smollm_selection.json',dict(threshold=t,sweep=sweep,score='1-max cosine support'))
        test_x=SmolLMCascadeEvaluator.embed_texts(embedder,[r['text'] for r in views['test']])
        result.append(dict(dataset=dataset,kir=kir,seed=42,variant='Cascade-SmolLM',**metrics(views['test'],compose(downs['test'],prototype.predict(test_x)['score'],t))))
    save(final,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    del pipeline,router_model,embedder
    torch.cuda.empty_cache()
    return result


def main():
    global OUT, PAPER_GEOMETRY
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=DATASETS,action='append')
    parser.add_argument('--paper-geometry',action='store_true')
    parser.add_argument('--output-root',type=Path,default=None)
    args=parser.parse_args()
    PAPER_GEOMETRY=bool(args.paper_geometry)
    if args.output_root is not None:
        OUT=args.output_root.resolve()
    rows=[]
    for dataset in args.dataset or DATASETS:
        for kir in (.25,.5,.75):
            rows.extend(run_cell(dataset,kir))
    OUT.mkdir(parents=True,exist_ok=True)
    # Rebuild from every completed cell, supporting restart between cells.
    rows=[r for p in sorted(OUT.glob('*/kir*_seed42/results.json')) for r in full.read_json(p)]
    with (OUT/'summary.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(OUT/'MANIFEST.json',dict(completed_units=len(rows),expected_units=36,device='cuda',seed=42,variants=VARIANTS,paper_geometry=PAPER_GEOMETRY,protocol='current H1 paper-geometry four-variant ablation' if PAPER_GEOMETRY else 'current H1 matched four-variant ablation',test_used_for_selection=False,new_evaluation=True))


if __name__=='__main__':
    main()
