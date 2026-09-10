# ADB 跨数据集三 seed 外部合同对比

本报告审计 ADB 在当前 protocol_v2 导出 split 上的 BERT/TextOIR 兼容运行，并与相同 dataset×seed×KIR=.50 的 S2C Trainable K=1 做配对。它不把 BERT 外部合同并入 MiniLM fair 主排名。

有效单元：`9/9`；配对差值：`63` 行；指标从 `y_true.npy/y_pred.npy` 重算。

## ADB 逐 seed 结果

| 数据集 | seed | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | False Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 42 | 72.69 | 77.93 | 90.68 | 74.29 | 89.93 | 37.31 |
| banking77 | 87 | 75.65 | 79.96 | 90.98 | 76.62 | 89.08 | 32.69 |
| banking77 | 100 | 74.57 | 76.66 | 88.79 | 74.84 | 88.75 | 34.04 |
| clinc150 | 42 | 89.52 | 86.30 | 94.39 | 87.51 | 91.42 | 14.43 |
| clinc150 | 87 | 90.41 | 85.49 | 92.86 | 88.11 | 90.00 | 12.12 |
| clinc150 | 100 | 90.10 | 86.22 | 93.98 | 88.11 | 90.49 | 12.93 |
| stackoverflow | 42 | 85.95 | 84.25 | 87.77 | 84.85 | 79.30 | 9.03 |
| stackoverflow | 87 | 87.02 | 85.35 | 88.69 | 86.02 | 80.87 | 8.23 |
| stackoverflow | 100 | 89.12 | 87.38 | 89.67 | 88.23 | 82.17 | 5.30 |

## S2C Trainable K=1 − ADB 配对均值（百分点）

| 数据集 | 指标 | 均值 | 标准差 |
|---|---|---:|---:|
| banking77 | Accuracy | 7.24 | 0.95 |
| banking77 | F1-All | 3.68 | 0.36 |
| banking77 | F1-K | -8.35 | 0.64 |
| banking77 | False Acceptance | -19.89 | 3.15 |
| banking77 | False Rejection | 7.13 | 0.79 |
| banking77 | Known Recall | -7.13 | 0.79 |
| banking77 | OOS F1 | 9.79 | 1.72 |
| clinc150 | Accuracy | -0.60 | 0.46 |
| clinc150 | F1-All | -4.14 | 1.25 |
| clinc150 | F1-K | -11.99 | 1.60 |
| clinc150 | False Acceptance | -9.37 | 0.23 |
| clinc150 | False Rejection | 16.28 | 1.07 |
| clinc150 | Known Recall | -16.28 | 1.07 |
| clinc150 | OOS F1 | 0.35 | 0.14 |
| stackoverflow | Accuracy | 0.87 | 0.82 |
| stackoverflow | F1-All | 1.03 | 0.46 |
| stackoverflow | F1-K | -2.16 | 0.67 |
| stackoverflow | False Acceptance | 0.66 | 1.78 |
| stackoverflow | False Rejection | -2.90 | 1.43 |
| stackoverflow | Known Recall | 2.90 | 1.43 |
| stackoverflow | OOS F1 | 0.85 | 0.87 |

## 解释边界

- ADB 使用端到端 BERT/TextOIR 训练；S2C Trainable K=1 使用 Known-only MiniLM 适配。配对差值只能说明同数据 split 下的工作点差异，不能分离 backbone、训练目标和边界学习的贡献。
- DA-ADB、DCLOOS 和 MOGB 官方论文值不在本报告中重新排名；它们分别有 invalid、额外 OOS 监督或官方数据/环境合同差异。
- 下一步可在本报告基础上补充 ADB 的 KIR 曲线，但当前不使用测试结果选参。

机器可读输出：`results/analysis/archive/analysis/adb_cross_dataset_v1/`；图表：`figures/archive/analysis/adb_cross_dataset_v1/`。
