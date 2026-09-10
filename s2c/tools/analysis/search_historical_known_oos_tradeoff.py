"""Reconcile Known F1 and select existing workpoints using validation only."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/analysis/historical_trainable_oos_priority_search'
DATA = ROOT.parent / 'assets/datasets/s2c/prepared/data/multidataset/v19'


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def count(dataset, kir):
    counts = [len(json.loads((DATA / dataset / f'kir{int(kir*100)}_seed{s}' / 'KNOWN_INTENTS.json').read_text())['known_intents']) for s in (13,42,87)]
    assert len(set(counts)) == 1
    return counts[0]


def key(row):
    return (int(row['k']), float(row['radius_lambda']), float(row['threshold']), row['acceptance_mode'])


def main():
    # Extract every baseline numerically, including the best value in each column.
    table = (ROOT / 'fulltex.tex').read_text().split('\\multirow{8}{*}{$0.25$}')[1].split('\\bottomrule')[0]
    kir = .25
    baseline = defaultdict(list)
    for line in table.splitlines():
        match = re.search(r'\\multirow\{8\}\{\*\}\{\$(0\.\d+)\$\}', line)
        if match:
            kir = float(match[1])
        cells = line.split('&')
        if len(cells) != 11 or 'Ours' in cells[1]:
            continue
        values = [float(re.search(r'\d+\.\d+', cell)[0]) for cell in cells[2:]]
        for index, dataset in enumerate(('clinc150','stackoverflow','banking77_oos')):
            baseline[dataset,kir].append(values[index*3:index*3+3])
    maxima = {key: [max(v[i] for v in rows) for i in range(3)] for key,rows in baseline.items()}
    corrected = []
    for row in read(ROOT / 'results/analysis/historical_paper_ablation/current_h1_full_pipeline_summary.csv'):
        d, kir = row['dataset'], float(row['kir'])
        n = count(d,kir)
        fk = ((n+1)*float(row['f1_all_mean'])-float(row['oos_f1_mean']))/n
        vals = (fk,float(row['oos_f1_mean']),float(row['accuracy_mean']))
        corrected.append(dict(dataset=d,kir=kir,configuration=row['configuration'],known_f1=fk,legacy_known_subset_f1=float(row['known_macro_f1_mean']),oos_f1=vals[1],accuracy=vals[2],**{f'delta_{m}_vs_external': v-b for m,v,b in zip(('known','oos','acc'),vals,maxima[d,kir])}))
    write('corrected_current_metrics.csv',corrected)
    choices, audits = [], []
    sources = [(ROOT/'results/analysis/historical_trainable_parameter_full_pipeline',.5,d) for d in ('clinc150','stackoverflow','banking77_oos')]
    sources += [(OUT/f'kir{int(k*100)}',k,'banking77_oos') for k in (.25,.5,.75)]
    for source,kir,d in sources:
        n = count(d,kir)
        val, test = defaultdict(list),defaultdict(list)
        for r in read(source/'gate_workpoints.csv'):
            if r['dataset']==d and r['split']=='val':
                val[key(r)].append(r)
        for r in read(source/'full_pipeline_candidates_test.csv'):
            if r['dataset']==d:
                test[key(r)].append(r)
        vm = {k: (mean(float(r['oos_f1']) for r in rs),mean(float(r['known_f1']) for r in rs)) for k,rs in val.items() if len(rs)==3}
        peak = max(v[0] for v in vm.values())
        def metrics(k):
            rs=test[k]
            assert len(rs)==3
            return (100*mean(((n+1)*float(r['f1_all'])-float(r['oos_f1']))/n for r in rs),100*mean(float(r['oos_f1']) for r in rs),100*mean(float(r['overall_accuracy']) for r in rs))
        base = metrics((1,1.,1.,'nearest_sphere'))
        for slack in (0,.25,.5,1.,2.):
            eligible = [k for k,v in vm.items() if v[0]>=peak-slack/100]
            selected = max(eligible,key=lambda k:(vm[k][1],vm[k][0],-k[0],-abs(k[2]-1),k[3]))
            vals=metrics(selected)
            choices.append(dict(dataset=d,kir=kir,source=source.relative_to(ROOT).as_posix(),oos_validation_slack_pp=slack,configuration=str(selected),val_oos=100*vm[selected][0],val_gate_known=100*vm[selected][1],known_f1=vals[0],oos_f1=vals[1],accuracy=vals[2],delta_known_vs_k1=vals[0]-base[0],delta_oos_vs_k1=vals[1]-base[1],delta_acc_vs_k1=vals[2]-base[2],selection='validation_gate_metrics_only_test_full_pipeline_confirmation'))
        candidates=[metrics(k) for k in test]
        audits.append(dict(dataset=d,kir=kir,source=source.relative_to(ROOT).as_posix(),candidate_count=len(candidates),all_metric_winners=sum(all(v>b for v,b in zip(values,maxima[d,kir])) for values in candidates),baseline_known=maxima[d,kir][0],baseline_oos=maxima[d,kir][1],baseline_acc=maxima[d,kir][2],max_test_known=max(v[0] for v in candidates),test_search='posthoc_diagnostic_not_model_selection'))
    write('validation_tradeoff_choices.csv',choices)
    write('corrected_candidate_audit.csv',audits)
    lines = ['# Known F1 口径修正与 OOS–Known 配置筛选','',
             '此前三个工作点“全面 SOTA”的判断撤回：legacy known_macro_f1 仅在真实 Known 子集上计算，排除了 OOS 误接收造成的 Known false positives。', '',
             '本文采用全测试集上各 Known 类的 macro F1。由 evaluator 的显式标签集合可精确还原：Known F1 = ((C+1) × F1-All − OOS F1) / C，C 是 Known 类数量；三个 seed 的 C 已检查一致。OOS F1 和 Accuracy 不变。','',
             '## 当前九组结果（百分数）','',
             '| 数据集 | KIR | 全测试集 Known F1 | 旧 Known 子集 F1 | OOS F1 | Acc |',
             '|---|---:|---:|---:|---:|---:|']
    for r in corrected:
        lines.append(f"| {r['dataset']} | {r['kir']} | {r['known_f1']:.2f} | {r['legacy_known_subset_f1']:.2f} | {r['oos_f1']:.2f} | {r['accuracy']:.2f} |")
    lines += ['', '## 已有候选筛查', '',
              f"共检查 {sum(r['candidate_count'] for r in audits)} 个 dataset×KIR×配置×来源均值记录（两个 Banking KIR=.50 网格有重叠），三项超过论文其他 baseline 的记录为 0。该筛查只是测试集后验诊断，不用来选模型。", '',
              '## 验证集选择的折中配置', '',
              '先保留验证集 OOS F1 距最优值不超过 0/0.25/0.5/1/2 pp 的候选，再选验证集 Gate Known F1 最大者。五种容差全部保留，未按测试结果选择容差。表中展示 0.25 pp 容差；完整 30 行见 CSV。验证集使用最近中心意图预测，测试使用已保存的固定 Router/Expert replay，尚无本轮新的逐配置 GPU 直接确认。', '',
              '| 数据集 | KIR | 配置 (K, λ, t, mode) | Known F1 | OOS F1 | Acc | ΔKnown / ΔOOS vs 原始K1 |',
              '|---|---:|---|---:|---:|---:|---:|']
    for r in choices:
        if r['oos_validation_slack_pp']==.25:
            lines.append(f"| {r['dataset']} | {r['kir']} | {r['configuration']} | {r['known_f1']:.2f} | {r['oos_f1']:.2f} | {r['accuracy']:.2f} | {r['delta_known_vs_k1']:+.2f} / {r['delta_oos_vs_k1']:+.2f} |")
    lines += ['', '原始K1参照是同一来源、相同checkpoint和下游的 K=1、λ=1、t=1、nearest_sphere。两个 Banking KIR=.50 来源的下游 replay 有小幅差异，各自对齐自己的参照。', '',
              'CLINC和StackOverflow的KIR=.25/.75未在这些来源中保存完整边界候选网格，本次没有为这些单元捏造新的选择结果。所有结果属于H1与历史论文数值比较，尚不能宣称九个条件全面领先。', '',
              '来源与输出：', '',
              '- [九组口径修正](../../results/analysis/historical_trainable_oos_priority_search/corrected_current_metrics.csv)',
              '- [全部验证集折中选择](../../results/analysis/historical_trainable_oos_priority_search/validation_tradeoff_choices.csv)',
              '- [候选覆盖与逐指标上限诊断](../../results/analysis/historical_trainable_oos_priority_search/corrected_candidate_audit.csv)', '']
    (ROOT/'docs/analysis/historical_known_oos_tradeoff.md').write_text('\n'.join(lines))
    print(f'Corrected {len(corrected)} groups; audited {sum(r["candidate_count"] for r in audits)} candidate records; selected {len(choices)} validation workpoints.')


if __name__ == '__main__':
    main()
