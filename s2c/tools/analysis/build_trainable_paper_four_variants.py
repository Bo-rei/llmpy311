"""Render the actual four-variant ablation in the paper's table layout."""
import csv
import json
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/analysis/trainable_paper_four_variants'
REPORT=ROOT/'docs/analysis/trainable_paper_four_variants.md'
DATASETS=('clinc150','stackoverflow','banking77_oos')
VARIANTS=('Ours','Without Gate','Cascade-MiniLM','Cascade-SmolLM')


def build(input_root=OUT, report_path=REPORT, paper_geometry=False):
    records=[r for p in sorted(input_root.glob('*/kir*_seed42/results.json')) for r in json.loads(p.read_text())]
    indexed={(r['dataset'],r['kir'],r['variant']):r for r in records}
    assert len(indexed)==len(records), 'duplicate cells'
    lines=['# Trainable Gate 完整系统：论文四变体消融','',
           f'新评估完成 {len(records)}/36 个单元。三个数据集 × KIR=.25/.50/.75 × 四个变体，seed=42。主表使用论文布局：每个数据集两列 Acc、OOS F1；下表所有已填数字均来自本轮运行。','',
           '| KIR | Variant | CLINC Acc | CLINC OOS F1 | StackOverflow Acc | StackOverflow OOS F1 | BANKING77-OOS Acc | BANKING77-OOS OOS F1 |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    for kir in (.25,.5,.75):
        for variant in VARIANTS:
            cells=[]
            for dataset in DATASETS:
                r=indexed.get((dataset,kir,variant))
                cells+=['待完成','待完成'] if r is None else [f"{r['overall_accuracy']:.2f}",f"{r['oos_f1']:.2f}"]
            lines.append(f'| {kir:.2f} | {variant} | '+ ' | '.join(cells)+' |')
    lines+=['','## 四行实际含义','',
            '- Ours：' + ('Trainable MiniLM Gate + 固定 SmolLM Router/Expert，使用论文几何 K=2、CLINC lambda=.5、其余 lambda=1、threshold=1、normalized boundary。' if paper_geometry else '当前 Trainable MiniLM Gate + 固定 SmolLM LoRA Router/Expert，逐数据集/KIR使用此前已确定的主配置，见各cell config.json。'),
            '- Without Gate：移除几何 Gate，所有样本进入原 Router/Expert；用下游 Expert 意图置信度拒识。这遵循原论文执行代码的 intent_confidence 分支；论文文字的 Router rejection 在单域数据中没有可用置信信息。',
            '- Cascade-MiniLM：保留当前 Trainable MiniLM Gate，复用论文 MiniLM cascade 的域分类头和域内意图分类头（Known train 上 logistic regression），替换 SmolLM 下游。表示取当前 Trainable MiniLM 的输出；三阶段均为 MiniLM。',
            '- Cascade-SmolLM：Gate改为 SmolLM mean-pooled/L2-normalized 意图原型余弦分数，复用论文原型 Gate 实现；下游仍是相同 SmolLM Router/Expert。CLINC使用router backbone；单域常量router没有模型参数，使用预训练SmolLM backbone。',
            '',
            'Without Gate和Cascade-SmolLM的拒识分数阈值在验证集上按完整macro F1选择，搜索分数阈值为0.20至0.95，步长0.05。所有阈值与验证集分数单独保存。Ours/Cascade-MiniLM共用已确定的Gate及边界，所以OOS F1相同是设计预期，Known分类和Acc可以不同。',
            '',
            '本表属于当前H1同数据、同下游条件下的结构/模型替换实验。旧论文四行历史数字未参与本表填充或参数选择。',
            '', '## 补充完整指标','',
            '| Dataset | KIR | Variant | Known F1 | OOS F1 | Acc | Known Recall | False Acceptance |',
            '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['dataset']} | {r['kir']:.2f} | {r['variant']} | "+' | '.join(f"{r[k]:.2f}" for k in ('known_f1','oos_f1','overall_accuracy','known_recall','false_accept_rate'))+' |')
    source_rel = input_root.resolve().relative_to(ROOT).as_posix()
    lines+=['','Known F1直接在全测试集上对Known类求macro F1，保留真实OOS引起的Known false positives。全部四行同一数据/KIR/seed；这是单seed结果，不是三个seed平均。','',
            f'[源表](../../{source_rel}/summary.csv)','']
    report_path.write_text('\n'.join(lines))
    if records:
        with (input_root/'summary.csv').open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(f'{len(records)}/36 cells rendered: {report_path}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-root',type=Path,default=OUT)
    parser.add_argument('--report',type=Path,default=REPORT)
    parser.add_argument('--paper-geometry',action='store_true')
    args=parser.parse_args()
    build(args.input_root.resolve(),args.report.resolve(),args.paper_geometry)
