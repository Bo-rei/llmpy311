# Frozen MiniLM Gate vs Trainable MiniLM Gate 消融

这组消融只改变 Gate 表示：Frozen all-MiniLM-L6-v2 与当前 Trainable MiniLM；KIR、K=1、对角 Mahalanobis、mean+1×std、threshold=1、Router、Expert、数据和 seed 均保持一致。

论文中的 Without Gate / Cascade-MiniLM / Cascade-SmolLM 36 个 artifact 仍作为历史论文设置参照；本报告新增的是更直接回答当前方法改动的 Frozen-vs-Trainable Gate 对照。

Known F1 使用全测试集 Known 类 macro F1，包含 OOS 被接受为 Known 造成的 precision 损失；旧报告中的 Known 子集指标不用于本表。

| 数据集 | KIR | Gate | Known F1 | OOS F1 | Accuracy | Known Recall | False Acceptance |
|---|---:|---|---:|---:|---:|---:|---:|
| banking77_oos | 0.25 | frozen_k1 | 60.57±2.75 | 91.58±0.65 | 85.33±1.07 | 86.39 | 14.00 |
| banking77_oos | 0.25 | trainable_k1 | 62.83±2.41 | 92.58±0.89 | 86.91±1.36 | 84.38 | 12.01 |
| banking77_oos | 0.50 | frozen_k1 | 62.43±1.06 | 84.82±1.18 | 76.75±1.30 | 88.83 | 23.66 |
| banking77_oos | 0.50 | trainable_k1 | 65.86±1.29 | 88.47±0.67 | 81.25±0.72 | 85.47 | 16.93 |
| banking77_oos | 0.75 | frozen_k1 | 67.93±0.97 | 83.55±1.77 | 76.16±1.68 | 89.89 | 23.91 |
| banking77_oos | 0.75 | trainable_k1 | 70.09±0.25 | 86.60±0.42 | 79.44±0.38 | 86.58 | 17.54 |
| clinc150 | 0.25 | frozen_k1 | 72.87±1.90 | 94.29±0.29 | 89.83±0.17 | 73.68 | 4.66 |
| clinc150 | 0.25 | trainable_k1 | 75.39±1.65 | 95.10±0.30 | 91.19±0.27 | 73.13 | 2.97 |
| clinc150 | 0.50 | frozen_k1 | 74.94±0.94 | 88.02±0.82 | 83.08±0.95 | 73.67 | 7.06 |
| clinc150 | 0.50 | trainable_k1 | 76.98±0.92 | 89.50±0.68 | 84.96±0.81 | 73.44 | 4.11 |
| clinc150 | 0.75 | frozen_k1 | 76.43±0.29 | 78.95±0.91 | 77.76±0.61 | 74.81 | 8.96 |
| clinc150 | 0.75 | trainable_k1 | 78.05±0.38 | 80.80±0.66 | 79.56±0.51 | 73.83 | 4.36 |
| stackoverflow | 0.25 | frozen_k1 | 79.20±4.91 | 93.85±2.00 | 89.30±3.00 | 83.07 | 6.54 |
| stackoverflow | 0.25 | trainable_k1 | 82.61±0.13 | 95.38±1.02 | 91.58±0.92 | 83.56 | 3.83 |
| stackoverflow | 0.50 | frozen_k1 | 75.50±2.37 | 79.02±3.78 | 76.74±3.08 | 83.22 | 23.52 |
| stackoverflow | 0.50 | trainable_k1 | 83.38±1.60 | 88.78±2.21 | 85.89±1.80 | 83.71 | 7.10 |
| stackoverflow | 0.75 | frozen_k1 | 78.90±1.42 | 63.13±4.40 | 75.17±1.94 | 83.59 | 31.01 |
| stackoverflow | 0.75 | trainable_k1 | 84.24±0.91 | 75.16±2.36 | 81.92±0.96 | 83.11 | 9.21 |

## Trainable − Frozen 配对差值

| 数据集 | KIR | Δ Known F1 | Δ OOS F1 | Δ Accuracy | Δ Known Recall | Δ False Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| clinc150 | 0.25 | +2.52 | +0.80 | +1.36 | -0.56 | -1.68 |
| clinc150 | 0.50 | +2.04 | +1.47 | +1.88 | -0.24 | -2.94 |
| clinc150 | 0.75 | +1.62 | +1.84 | +1.81 | -0.98 | -4.60 |
| stackoverflow | 0.25 | +3.41 | +1.52 | +2.28 | +0.49 | -2.72 |
| stackoverflow | 0.50 | +7.88 | +9.76 | +9.15 | +0.49 | -16.43 |
| stackoverflow | 0.75 | +5.35 | +12.03 | +6.75 | -0.48 | -21.80 |
| banking77_oos | 0.25 | +2.26 | +1.00 | +1.58 | -2.01 | -1.99 |
| banking77_oos | 0.50 | +3.43 | +3.65 | +4.50 | -3.37 | -6.73 |
| banking77_oos | 0.75 | +2.16 | +3.05 | +3.28 | -3.31 | -6.37 |

## 读法

如果 Trainable 的 OOS F1 上升且 False Acceptance 下降，同时 Known F1/Accuracy 没有同步大幅下降，可以把收益归因到 Gate 表示适配在当前固定下游链路上的作用。若 OOS 改善伴随 Known Recall 下降，则应报告为 coverage–rejection trade-off，而不是三个指标全面改善。

## 证据范围

- 3 个数据集 × 3 个 KIR × 3 个 seed × 2 个 Gate = 54 个 CUDA full-pipeline 单元。
- Router/Expert 使用对应 KIR 的同一固定组件；没有重新训练下游模型。
- Trainable 和 Frozen 使用同样的 K=1、lambda=1、nearest-sphere detector contract。
- fulltex.tex 未修改；论文四变体历史 artifact 不与本表混合排名。

结果入口：

- [逐 seed 结果](../../results/analysis/historical_gate_ablation/per_seed.csv)
- [汇总](../../results/analysis/historical_gate_ablation/summary.csv)
- [Trainable−Frozen 差值](../../results/analysis/historical_gate_ablation/paired_delta.csv)
