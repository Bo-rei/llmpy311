# Known F1 口径修正与 OOS–Known 配置筛选

此前三个工作点“全面 SOTA”的判断撤回：legacy known_macro_f1 仅在真实 Known 子集上计算，排除了 OOS 误接收造成的 Known false positives。

本文采用全测试集上各 Known 类的 macro F1。由 evaluator 的显式标签集合可精确还原：Known F1 = ((C+1) × F1-All − OOS F1) / C，C 是 Known 类数量；三个 seed 的 C 已检查一致。OOS F1 和 Accuracy 不变。

## 当前九组结果（百分数）

| 数据集 | KIR | 全测试集 Known F1 | 旧 Known 子集 F1 | OOS F1 | Acc |
|---|---:|---:|---:|---:|---:|
| clinc150 | 0.25 | 74.89 | 78.99 | 95.10 | 91.19 |
| clinc150 | 0.5 | 82.45 | 86.53 | 91.93 | 88.02 |
| clinc150 | 0.75 | 78.03 | 79.15 | 80.80 | 79.56 |
| stackoverflow | 0.25 | 80.07 | 84.64 | 95.38 | 91.58 |
| stackoverflow | 0.5 | 83.45 | 85.23 | 89.96 | 86.94 |
| stackoverflow | 0.75 | 84.85 | 86.17 | 75.16 | 81.92 |
| banking77_oos | 0.25 | 60.42 | 85.30 | 92.58 | 86.91 |
| banking77_oos | 0.5 | 65.26 | 71.19 | 91.89 | 85.96 |
| banking77_oos | 0.75 | 69.67 | 79.63 | 86.60 | 79.44 |

## 已有候选筛查

共检查 3636 个 dataset×KIR×配置×来源均值记录（两个 Banking KIR=.50 网格有重叠），三项超过论文其他 baseline 的记录为 0。该筛查只是测试集后验诊断，不用来选模型。

## 验证集选择的折中配置

先保留验证集 OOS F1 距最优值不超过 0/0.25/0.5/1/2 pp 的候选，再选验证集 Gate Known F1 最大者。五种容差全部保留，未按测试结果选择容差。表中展示 0.25 pp 容差；完整 30 行见 CSV。验证集使用最近中心意图预测，测试使用已保存的固定 Router/Expert replay，尚无本轮新的逐配置 GPU 直接确认。

| 数据集 | KIR | 配置 (K, λ, t, mode) | Known F1 | OOS F1 | Acc | ΔKnown / ΔOOS vs 原始K1 |
|---|---:|---|---:|---:|---:|---:|
| clinc150 | 0.5 | (1, 2.0, 0.95, 'nearest_sphere') | 80.01 | 90.42 | 86.10 | +3.22 / +0.93 |
| stackoverflow | 0.5 | (1, 1.0, 0.95, 'nearest_sphere') | 82.65 | 89.15 | 86.14 | -0.14 / +0.36 |
| banking77_oos | 0.5 | (1, 0.75, 0.95, 'normalized_union') | 67.45 | 91.53 | 85.59 | +2.47 / +3.06 |
| banking77_oos | 0.25 | (1, 1.0, 0.9, 'nearest_sphere') | 66.14 | 95.69 | 92.03 | +5.72 / +3.12 |
| banking77_oos | 0.5 | (1, 0.75, 0.95, 'normalized_union') | 67.46 | 91.53 | 85.59 | +2.47 / +3.06 |
| banking77_oos | 0.75 | (1, 1.0, 0.95, 'nearest_sphere') | 70.25 | 88.40 | 81.56 | +0.58 / +1.79 |

原始K1参照是同一来源、相同checkpoint和下游的 K=1、λ=1、t=1、nearest_sphere。两个 Banking KIR=.50 来源的下游 replay 有小幅差异，各自对齐自己的参照。

CLINC和StackOverflow的KIR=.25/.75未在这些来源中保存完整边界候选网格，本次没有为这些单元捏造新的选择结果。所有结果属于H1与历史论文数值比较，尚不能宣称九个条件全面领先。

来源与输出：

- [九组口径修正](../../results/analysis/historical_trainable_oos_priority_search/corrected_current_metrics.csv)
- [全部验证集折中选择](../../results/analysis/historical_trainable_oos_priority_search/validation_tradeoff_choices.csv)
- [候选覆盖与逐指标上限诊断](../../results/analysis/historical_trainable_oos_priority_search/corrected_candidate_audit.csv)
