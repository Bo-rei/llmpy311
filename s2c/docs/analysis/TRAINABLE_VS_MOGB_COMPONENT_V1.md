# Trainable K=1 与 MOGB 组件五 seed 对比 V1

> 这是同一数据划分、KIR、seed 下的组件级比较，不是 MOGB 官方 BERT 论文结果的严格复现排名。MOGB 行来自 Frozen MiniLM fair matrix，表示和监督条件与 Trainable 不同。

## 1. 比较范围

- Trainable：Known-only、最后两层 MiniLM + projection、K=1、diagonal Mahalanobis、mean+std。
- `mogb_partition_ours_boundary`：Frozen MiniLM + MOGB 动态分区 + s2c 边界，是最接近的边界合同对照。
- `mogb_minilm`：Frozen MiniLM + MOGB 分区/欧氏距离/平均半径。
- `fixed_k2`：Frozen MiniLM + 固定 K=2；作为多中心失败参考。
- 每个比较均按 dataset×KIR×seed 配对，bootstrap 10,000 次，仅用于统计描述。

## 2. KIR=0.50 的均值

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False Accept |
|---|---|---:|---:|---:|---:|
| clinc150 | Trainable K=1 | 90.44 | 81.82 | 74.44 | 3.69 |
| clinc150 | MOGB partition + s2c boundary | 85.56 | 64.60 | 53.34 | 2.50 |
| clinc150 | MOGB MiniLM | 81.32 | 44.95 | 31.57 | 0.90 |
| clinc150 | Frozen fixed K=2 | 89.20 | 80.09 | 75.14 | 6.44 |
| banking77 | Trainable K=1 | 83.56 | 81.67 | 82.21 | 15.74 |
| banking77 | MOGB partition + s2c boundary | 79.40 | 64.81 | 52.18 | 3.50 |
| banking77 | MOGB MiniLM | 74.99 | 48.60 | 33.41 | 1.09 |
| banking77 | Frozen fixed K=2 | 75.46 | 76.16 | 82.55 | 29.05 |
| stackoverflow | Trainable K=1 | 87.67 | 86.55 | 83.89 | 9.34 |
| stackoverflow | MOGB partition + s2c boundary | 79.25 | 63.34 | 50.39 | 1.86 |
| stackoverflow | MOGB MiniLM | 72.92 | 43.30 | 27.09 | 0.79 |
| stackoverflow | Frozen fixed K=2 | 63.53 | 72.76 | 86.89 | 47.17 |

## 3. 机制解读

1. Trainable K=1 在 KIR=.50 的 OOS F1 高于 MOGB partition+s2c boundary，但 MOGB 的 false acceptance 更低；MOGB 主要通过拒绝更多 Known 换取保守拒识。
2. 例如 StackOverflow：Trainable 为 OOS F1 87.67%、Known Recall 83.89%、false acceptance 9.34%；MOGB partition+s2c boundary 为 79.25%、50.39%、1.86%。这不是单纯的 OOS F1 胜负，而是“覆盖—拒识”工作点不同。
3. CLINC150 和 Banking77 也表现出相同趋势：MOGB 组件通常降低 false acceptance，但 Known Recall/F1-All 显著下降；Trainable 保留了更高的 Known 覆盖。
4. 因此当前可支持的解释是：Trainable 的优势来自可训练表示带来的 score 分离和更平衡的工作点，不是已经证明自己超过了完整 MOGB 或所有端到端基线。

## 4. 证据文件

- `results/analysis/trainable_vs_mogb_component_v1/aggregate.csv`
- `results/analysis/trainable_vs_mogb_component_v1/paired_effects.csv`
- `figures/trainable_vs_mogb_component_v1/`

## 5. 结论边界

- 不能把 MOGB fair component 结果称为官方 MOGB 完整复现；官方 BERT 结果另有复现状态。
- 不能只用 OOS F1 选择方法；Known Recall、F1-All 和 false acceptance 必须同时报告。
- 下一步应把相同监督条件下的强基线和完整 Cascade 单独桥接，而不是把不同协议的数字合成一个 SOTA 排名。
