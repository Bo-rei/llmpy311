# protocol_v2_textoir_v1 原生 Frozen MiniLM Baseline 对比 V1

> 本批实验只运行当前代码中可在本地、Known-only 条件下真实执行的 MSP、Energy、kNN 和 LOF；ADB、DA-ADB、MOGB 仍保留为独立适配/复现任务，不用伪造近似结果替代。

## 完成状态

- 计划单元：180；完成：180；失败：0。
- 范围：3 数据集 × KIR 0.25/0.50/0.75 × 5 seeds × 4 方法。
- 表示：冻结 `all-MiniLM-L6-v2`，同一 dataset/KIR/seed 的四种方法复用同一 embedding。
- 训练与阈值：训练和 calibration 仅含 Known；阈值使用 Known-only conformal α=0.05；测试 OOS 未用于选择。

## KIR=0.50 均值结果

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False Accept | AUROC |
|---|---|---:|---:|---:|---:|---:|
| clinc150 | trainable_k1 | 90.44±0.44 | 81.82±0.92 | 74.44±0.34 | 3.69±0.76 | 95.33±0.55 |
| clinc150 | msp | 85.58±1.18 | 83.36±0.95 | 95.46±0.62 | 22.97±1.99 | 95.25±0.25 |
| clinc150 | energy | 84.29±1.38 | 81.79±0.45 | 95.77±0.50 | 25.12±2.24 | 95.14±0.22 |
| clinc150 | knn | 73.94±3.01 | 71.76±1.70 | 95.74±0.38 | 39.63±4.03 | 92.12±0.59 |
| clinc150 | lof | 66.03±5.30 | 69.34±2.15 | 96.13±0.38 | 49.28±5.95 | 91.07±0.68 |
| banking77 | trainable_k1 | 83.56±1.83 | 81.67±1.05 | 82.21±0.48 | 15.74±3.43 | 90.14±1.11 |
| banking77 | msp | 65.44±2.16 | 76.83±0.59 | 93.71±0.47 | 48.36±2.51 | 88.52±1.21 |
| banking77 | energy | 61.11±6.25 | 74.52±1.67 | 93.21±1.09 | 52.81±7.12 | 86.06±1.69 |
| banking77 | knn | 60.85±2.35 | 74.54±0.46 | 94.04±0.52 | 53.69±2.58 | 89.12±0.98 |
| banking77 | lof | 44.11±5.11 | 70.91±0.93 | 96.61±0.44 | 70.65±4.36 | 86.19±1.49 |
| stackoverflow | trainable_k1 | 87.67±1.66 | 86.55±1.30 | 83.89±0.36 | 9.34±3.26 | 92.35±1.49 |
| stackoverflow | msp | 53.48±7.16 | 72.91±2.17 | 95.28±0.44 | 61.50±7.26 | 88.89±2.13 |
| stackoverflow | energy | 47.08±7.39 | 70.63±1.92 | 95.61±0.57 | 67.62±6.59 | 88.77±1.81 |
| stackoverflow | knn | 61.33±9.68 | 69.94±4.13 | 94.87±0.80 | 52.89±10.49 | 89.44±1.71 |
| stackoverflow | lof | 34.76±2.41 | 62.13±1.89 | 95.59±0.15 | 78.01±1.84 | 80.25±2.01 |

## Trainable 相对原生 baseline 的配对结果

配对单位为同一 dataset×KIR×seed；CI 使用固定 RNG=20260725、10000 次 paired bootstrap。正值表示 Trainable 更高。

| 数据集 | KIR | Baseline | 指标 | Trainable−Baseline | 95% CI | Win/Tie/Loss |
|---|---:|---|---|---:|---|---:|
| banking77 | 0.50 | energy | oos_f1 | 22.45 pp | [18.29, 26.61] | 5/0/0 |
| banking77 | 0.50 | energy | f1_all | 7.16 pp | [5.99, 8.32] | 5/0/0 |
| banking77 | 0.50 | energy | known_recall | -11.00 pp | [-12.17, -10.20] | 0/0/5 |
| banking77 | 0.50 | energy | false_accept_rate | -37.06 pp | [-41.73, -32.53] | 0/0/5 |
| banking77 | 0.50 | knn | oos_f1 | 22.71 pp | [20.52, 24.90] | 5/0/0 |
| banking77 | 0.50 | knn | f1_all | 7.14 pp | [6.28, 8.15] | 5/0/0 |
| banking77 | 0.50 | knn | known_recall | -11.83 pp | [-12.57, -10.96] | 0/0/5 |
| banking77 | 0.50 | knn | false_accept_rate | -37.95 pp | [-41.08, -34.82] | 0/0/5 |
| banking77 | 0.50 | lof | oos_f1 | 39.46 pp | [36.36, 42.36] | 5/0/0 |
| banking77 | 0.50 | lof | f1_all | 10.76 pp | [10.30, 11.22] | 5/0/0 |
| banking77 | 0.50 | lof | known_recall | -14.39 pp | [-15.01, -13.86] | 0/0/5 |
| banking77 | 0.50 | lof | false_accept_rate | -54.91 pp | [-57.18, -52.55] | 0/0/5 |
| banking77 | 0.50 | msp | oos_f1 | 18.13 pp | [16.56, 20.06] | 5/0/0 |
| banking77 | 0.50 | msp | f1_all | 4.85 pp | [4.08, 5.80] | 5/0/0 |
| banking77 | 0.50 | msp | known_recall | -11.50 pp | [-12.22, -10.91] | 0/0/5 |
| banking77 | 0.50 | msp | false_accept_rate | -32.62 pp | [-35.59, -30.22] | 0/0/5 |
| clinc150 | 0.50 | energy | oos_f1 | 6.15 pp | [4.90, 7.40] | 5/0/0 |
| clinc150 | 0.50 | energy | f1_all | 0.03 pp | [-1.06, 0.98] | 3/0/2 |
| clinc150 | 0.50 | energy | known_recall | -21.32 pp | [-21.67, -20.97] | 0/0/5 |
| clinc150 | 0.50 | energy | false_accept_rate | -21.43 pp | [-23.45, -19.40] | 0/0/5 |
| clinc150 | 0.50 | knn | oos_f1 | 16.50 pp | [13.76, 18.85] | 5/0/0 |
| clinc150 | 0.50 | knn | f1_all | 10.05 pp | [8.21, 11.32] | 5/0/0 |
| clinc150 | 0.50 | knn | known_recall | -21.30 pp | [-21.54, -20.97] | 0/0/5 |
| clinc150 | 0.50 | knn | false_accept_rate | -35.94 pp | [-39.18, -32.35] | 0/0/5 |
| clinc150 | 0.50 | lof | oos_f1 | 24.41 pp | [20.81, 29.09] | 5/0/0 |
| clinc150 | 0.50 | lof | f1_all | 12.48 pp | [10.79, 14.25] | 5/0/0 |
| clinc150 | 0.50 | lof | known_recall | -21.69 pp | [-22.17, -21.31] | 0/0/5 |
| clinc150 | 0.50 | lof | false_accept_rate | -45.58 pp | [-50.92, -41.26] | 0/0/5 |
| clinc150 | 0.50 | msp | oos_f1 | 4.86 pp | [3.76, 6.14] | 5/0/0 |
| clinc150 | 0.50 | msp | f1_all | -1.54 pp | [-2.73, -0.44] | 1/0/4 |
| clinc150 | 0.50 | msp | known_recall | -21.01 pp | [-21.48, -20.53] | 0/0/5 |
| clinc150 | 0.50 | msp | false_accept_rate | -19.28 pp | [-21.41, -17.49] | 0/0/5 |
| stackoverflow | 0.50 | energy | oos_f1 | 40.59 pp | [35.93, 45.25] | 5/0/0 |
| stackoverflow | 0.50 | energy | f1_all | 15.92 pp | [14.37, 17.64] | 5/0/0 |
| stackoverflow | 0.50 | energy | known_recall | -11.72 pp | [-12.39, -11.17] | 0/0/5 |
| stackoverflow | 0.50 | energy | false_accept_rate | -58.28 pp | [-61.43, -54.67] | 0/0/5 |
| stackoverflow | 0.50 | knn | oos_f1 | 26.34 pp | [19.61, 33.58] | 5/0/0 |
| stackoverflow | 0.50 | knn | f1_all | 16.61 pp | [13.34, 20.15] | 5/0/0 |
| stackoverflow | 0.50 | knn | known_recall | -10.97 pp | [-11.58, -10.31] | 0/0/5 |
| stackoverflow | 0.50 | knn | false_accept_rate | -43.55 pp | [-50.67, -36.43] | 0/0/5 |
| stackoverflow | 0.50 | lof | oos_f1 | 52.91 pp | [52.28, 53.68] | 5/0/0 |
| stackoverflow | 0.50 | lof | f1_all | 24.42 pp | [22.57, 26.63] | 5/0/0 |
| stackoverflow | 0.50 | lof | known_recall | -11.70 pp | [-12.01, -11.39] | 0/0/5 |
| stackoverflow | 0.50 | lof | false_accept_rate | -68.67 pp | [-70.05, -67.73] | 0/0/5 |
| stackoverflow | 0.50 | msp | oos_f1 | 34.19 pp | [29.05, 37.79] | 5/0/0 |
| stackoverflow | 0.50 | msp | f1_all | 13.64 pp | [12.42, 15.18] | 5/0/0 |
| stackoverflow | 0.50 | msp | known_recall | -11.39 pp | [-11.91, -10.86] | 0/0/5 |
| stackoverflow | 0.50 | msp | false_accept_rate | -52.16 pp | [-54.81, -48.47] | 0/0/5 |

## 解释边界

1. 这些原生方法和 Trainable 都是 Gate-only/二分类拒识；不能直接替代完整 Cascade、官方 MOGB 或使用外部 OOS 监督的 DCLOOS。
2. F1-All/F1-K 使用预测中的最近 Known intent 作为描述性闭集标签；MSP/Energy 的分类器、kNN/LOF 的最近训练样本都记录在 run manifest 中。
3. Trainable 若相对这些 Frozen baseline 提升，同时 false acceptance 下降，说明收益来自表示适配和分数排序；若 OOS F1 提升但 Known Recall 下降，则属于拒识/覆盖权衡，不能只称为全面优越。

## 文件

- `results/analysis/native_baselines_v1/per_seed.csv`
- `results/analysis/native_baselines_v1/summary_mean_std.csv`
- `results/analysis/native_baselines_v1/trainable_vs_native_paired.csv`
- `figures/native_baselines_v1/`
- `../artifacts/s2c/runs/protocol_v2_textoir_v1/native_baselines_v1/`
