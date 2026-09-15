"""Summarize completed full-pipeline/native results; never fill missing cells."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT.parent/'artifacts/s2c/analysis/kir_sensitivity_known_only'
OUT=ROOT/'results/analysis/kir_sensitivity_known_only/coverage_repair'


def write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    import argparse
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-root',type=Path,default=ART/'coverage_repair')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output_dir
    from sklearn.metrics import f1_score,accuracy_score
    rows=[]
    for path in (args.selection_root/'final').glob('*/*/Ours.json'):
        r=json.loads(path.read_text())
        assert r['full_pipeline']
        rows.append(dict(dataset=r['dataset'],kir=r['kir'],seed=r['seed'],method='Ours',
                         **r['metrics'],source=str(path),
                         recipe=r['selection']['name'],
                         geometry=json.dumps(r['selection']['geometry'],sort_keys=True),
                         checkpoint=r['selection']['checkpoint'],
                         validation_selection=r['selection']['selection']))
    seen=set()
    for path in (ART/'baselines').glob('*/*/*/final_predictions.npz'):
        manifest=json.loads(path.with_name('run_manifest.json').read_text())
        dataset=path.parents[2].name
        key=(dataset,float(manifest['known_cls_ratio']),int(manifest['seed']),manifest['method'])
        if key in seen:raise ValueError(f'Duplicate final baseline result: {key}')
        seen.add(key)
        v=np.load(path,allow_pickle=False)
        gold,pred=v['y_true'],v['y_pred']; oos=int(v['oos_id'])
        truth=gold==oos; rejected=pred==oos
        rows.append(dict(dataset=dataset,kir=key[1],seed=key[2],method=key[3],
            oos_f1=float(f1_score(truth,rejected)),
            known_f1=float(f1_score(gold,pred,labels=list(range(oos)),average='macro',zero_division=0)),
            accuracy=float(accuracy_score(gold,pred)),known_recall=float((~rejected[~truth]).mean()),
            false_accept_rate=float((~rejected[truth]).mean()),source=str(path)))
    write(OUT/'per_seed.csv',rows)
    summaries=[]
    for dataset,kir,method in sorted({(r['dataset'],r['kir'],r['method']) for r in rows}):
        group=[r for r in rows if (r['dataset'],r['kir'],r['method'])==(dataset,kir,method)]
        if sorted(r['seed'] for r in group)!=[13,42,87]:continue
        row=dict(dataset=dataset,kir=kir,method=method,seed_count=3)
        for metric in ('oos_f1','known_f1','accuracy','known_recall','false_accept_rate'):
            values=[100*r[metric] for r in group]
            row[metric+'_mean']=float(np.mean(values));row[metric+'_std']=float(np.std(values,ddof=0))
        summaries.append(row)
    write(OUT/'summary.csv',summaries)
    comparisons=[]
    for ours in [r for r in summaries if r['method']=='Ours']:
        baselines=[r for r in summaries if r['dataset']==ours['dataset'] and r['kir']==ours['kir'] and r['method']!='Ours']
        if len(baselines)!=4:continue
        best=max(baselines,key=lambda r:r['oos_f1_mean'])
        comparisons.append(dict(dataset=ours['dataset'],kir=ours['kir'],
            ours_oos_f1_mean=ours['oos_f1_mean'],ours_oos_f1_std=ours['oos_f1_std'],
            strongest_baseline=best['method'],baseline_oos_f1=best['oos_f1_mean'],
            delta_pp=ours['oos_f1_mean']-best['oos_f1_mean']))
    write(OUT/'baseline_deltas.csv',comparisons)
    write(OUT/'main_table_candidates.csv',[r for r in comparisons if r['kir'] in (.25,.5,.75)])
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'MANIFEST.json').write_text(json.dumps(dict(status='complete' if len(rows)==495 else 'partial',
        completed_units=len(rows),expected_units=495,complete_summary_points=len(summaries),std_ddof=0),indent=2))
    if summaries:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figures=ROOT/'figures/kir_sensitivity_known_only'
        if args.selection_root != ART:figures=figures/args.selection_root.name
        figures.mkdir(parents=True,exist_ok=True)
        for metric in ('oos_f1','known_f1'):
            fig,axes=plt.subplots(1,3,figsize=(13,3.5),sharey=True)
            for ax,dataset in zip(axes,('clinc150','stackoverflow','banking77')):
                for method in ('DOC','KNNCL','ADB','DA-ADB','Ours'):
                    points=sorted([r for r in summaries if r['dataset']==dataset and r['method']==method],key=lambda r:r['kir'])
                    if points:ax.errorbar([r['kir'] for r in points],[r[metric+'_mean'] for r in points],
                        yerr=[r[metric+'_std'] for r in points],marker='o',capsize=2,label=method)
                ax.set(title=dataset,xlabel='KIR');ax.grid(alpha=.2)
            axes[0].set_ylabel(metric+' (%)');axes[-1].legend(fontsize=8)
            fig.suptitle('Mean ± standard deviation, n=3'+(' (partial)' if len(rows)!=495 else ''))
            fig.tight_layout()
            for extension in ('pdf','png'):fig.savefig(figures/f'{metric}.{extension}',dpi=200)
            plt.close(fig)
    print(f'{len(rows)}/495 final units; {len(summaries)}/165 complete mean/std points')


if __name__=='__main__':main()
