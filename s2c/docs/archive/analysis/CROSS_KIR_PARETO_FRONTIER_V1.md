# 跨 KIR 多指标 Pareto 前沿 V1

本分析读取当前 `protocol_v2_textoir_v1` 已完成的 315 个 fair Gate 行（7 个方法×3 数据集×3 KIR×5 seed），只做五 seed 均值/标准差和后验多指标归因，不重训、不调参、不使用测试标签选择方法。

输入：`/home/bo/bo01/llmpy311/s2c/results/analysis/cross_protocol_tradeoff_v1/per_seed.csv`；SHA256：`31ccdd433b38b86ec92b0cd81b460114a7fe1e7f12b94d4a5d031aab7d703145`。Pareto 目标为最大化 OOS F1、F1-All、Known Recall，同时最小化 false acceptance。

## 解读

- 星形点表示在同一 dataset×KIR 单元的四指标均值上未被其他 fair 方法同时支配；它不是统计显著性结论。
- 误差线表示五个正式 seed 的标准差，用于观察稳定性，不用于选择测试最优点。
- Pareto 前沿可以区分“单项 OOS F1 高”与“Known 分类、OOS 拒识和误接收同时平衡”。

## Trainable-K1 状态

| 数据集 | KIR | Trainable 是否在 Pareto 前沿 | Pareto 方法 |
|---|---:|---|---|
| banking77 | 0.25 | 是 | Frozen K=2, MOGB-MiniLM, MOGB part. + S2C bound., Random K=2, Frozen K=1, Trainable K=1 |
| banking77 | 0.50 | 是 | Frozen K=2, MOGB-MiniLM, MOGB part. + S2C bound., S2C part. + MOGB bound., Random K=2, Frozen K=1, Trainable K=1 |
| banking77 | 0.75 | 是 | Frozen K=2, MOGB-MiniLM, MOGB part. + S2C bound., S2C part. + MOGB bound., Random K=2, Frozen K=1, Trainable K=1 |
| clinc150 | 0.25 | 是 | Frozen K=2, MOGB-MiniLM, MOGB part. + S2C bound., S2C part. + MOGB bound., Random K=2, Frozen K=1, Trainable K=1 |
| clinc150 | 0.50 | 是 | Frozen K=2, MOGB-MiniLM, MOGB part. + S2C bound., S2C part. + MOGB bound., Random K=2, Frozen K=1, Trainable K=1 |
| clinc150 | 0.75 | 是 | Frozen K=2, MOGB-MiniLM, MOGB part. + S2C bound., S2C part. + MOGB bound., Random K=2, Frozen K=1, Trainable K=1 |
| stackoverflow | 0.25 | 是 | MOGB-MiniLM, MOGB part. + S2C bound., Random K=2, Frozen K=1, Trainable K=1 |
| stackoverflow | 0.50 | 是 | MOGB-MiniLM, MOGB part. + S2C bound., Random K=2, Frozen K=1, Trainable K=1 |
| stackoverflow | 0.75 | 是 | MOGB-MiniLM, MOGB part. + S2C bound., Random K=2, Frozen K=1, Trainable K=1 |

## 输出

- `results/analysis/archive/analysis/cross_kir_pareto_frontier_v1/pareto_points.csv`：每个方法的均值、标准差和 Pareto 标记。
- `results/analysis/archive/analysis/cross_kir_pareto_frontier_v1/pareto_summary.csv`：每个 dataset×KIR 的前沿方法。
- `figures/archive/analysis/cross_kir_pareto_frontier_v1/cross_kir_pareto_frontier.png`：9 个工作点的多指标前沿图。
- `figures/archive/analysis/cross_kir_pareto_frontier_v1/pareto_cell_counts.png`：方法进入 Pareto 前沿的工作点数量。

## 边界

该分析只说明当前统一 MiniLM Known-only Gate 合同下的多指标工作点关系；MOGB 官方 BERT、ADB、DA-ADB 和 DCLOOS 的外部监督/骨干合同不在此图中，不据此宣称跨方法 SOTA。
