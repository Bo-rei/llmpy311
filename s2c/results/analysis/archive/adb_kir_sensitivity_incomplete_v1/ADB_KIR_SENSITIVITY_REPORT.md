# ADB 跨 KIR 三数据集外部合同分析

本报告只使用 protocol_v2_textoir_v1 的固定 split、隔离 TextOIR BERT runtime 和已审计 y_true/y_pred。
ADB 是 BERT/TextOIR 外部合同；S2C Trainable K=1 是 Known-only MiniLM。所有差值仅用于机制与工作点分析，
不进入 MiniLM fair 主排名，也不用于测试集选参。

有效单元：`27/27`；目标为 3 数据集×3 KIR×3 seed。

## ADB 绝对结果

| 数据集 | KIR | OOS F1 | F1-All | Known Recall | False Acceptance | n |
|---|---:|---:|---:|---:|---:|---:|
| banking77 | 0.25 | 83.83±2.46 | 69.23±1.74 | 87.98±1.37 | 24.96±3.66 | 3 |
| banking77 | 0.50 | 74.30±1.50 | 78.18±1.67 | 89.25±0.61 | 34.68±2.37 | 3 |
| banking77 | 0.75 | 65.64±3.06 | 85.82±0.36 | 90.33±0.29 | 36.67±4.15 | 3 |
| clinc150 | 0.25 | 91.53±1.08 | 76.88±1.38 | 90.85±0.95 | 13.68±1.99 | 3 |
| clinc150 | 0.50 | 90.01±0.45 | 86.00±0.44 | 90.64±0.72 | 13.16±1.18 | 3 |
| clinc150 | 0.75 | 85.09±1.16 | 88.54±0.46 | 91.65±0.06 | 17.05±2.02 | 3 |
| stackoverflow | 0.25 | 93.36±0.32 | 83.21±1.26 | 83.64±1.71 | 7.69±0.17 | 3 |
| stackoverflow | 0.50 | 87.36±1.61 | 85.66±1.59 | 80.78±1.44 | 7.52±1.97 | 3 |
| stackoverflow | 0.75 | 73.08±0.74 | 85.85±0.46 | 80.22±0.50 | 8.24±2.07 | 3 |

## S2C Trainable K=1 − ADB 配对差值

| 数据集 | KIR | 指标 | 均值(pp) | 标准差(pp) | n |
|---|---:|---|---:|---:|---:|
| banking77 | 0.25 | F1-All | 10.99 | 2.60 | 3 |
| banking77 | 0.25 | Known Recall | -6.67 | 1.33 | 3 |
| banking77 | 0.25 | OOS F1 | 8.86 | 2.37 | 3 |
| banking77 | 0.50 | F1-All | 3.68 | 0.36 | 3 |
| banking77 | 0.50 | Known Recall | -7.13 | 0.79 | 3 |
| banking77 | 0.50 | OOS F1 | 9.79 | 1.72 | 3 |
| banking77 | 0.75 | F1-All | -1.21 | 0.69 | 3 |
| banking77 | 0.75 | Known Recall | -7.90 | 0.60 | 3 |
| banking77 | 0.75 | OOS F1 | 4.55 | 3.12 | 3 |
| clinc150 | 0.25 | F1-All | 3.08 | 1.42 | 3 |
| clinc150 | 0.25 | Known Recall | -17.11 | 2.04 | 3 |
| clinc150 | 0.25 | OOS F1 | 3.73 | 0.26 | 3 |
| clinc150 | 0.50 | F1-All | -4.14 | 1.25 | 3 |
| clinc150 | 0.50 | Known Recall | -16.28 | 1.07 | 3 |
| clinc150 | 0.50 | OOS F1 | 0.35 | 0.14 | 3 |
| clinc150 | 0.75 | F1-All | -5.45 | 0.39 | 3 |
| clinc150 | 0.75 | Known Recall | -16.55 | 1.13 | 3 |
| clinc150 | 0.75 | OOS F1 | -2.20 | 0.78 | 3 |
| stackoverflow | 0.25 | F1-All | 5.48 | 0.75 | 3 |
| stackoverflow | 0.25 | Known Recall | 1.00 | 1.68 | 3 |
| stackoverflow | 0.25 | OOS F1 | 2.74 | 0.36 | 3 |
| stackoverflow | 0.50 | F1-All | 1.03 | 0.46 | 3 |
| stackoverflow | 0.50 | Known Recall | 2.90 | 1.43 | 3 |
| stackoverflow | 0.50 | OOS F1 | 0.85 | 0.87 | 3 |
| stackoverflow | 0.75 | F1-All | 1.04 | 0.48 | 3 |
| stackoverflow | 0.75 | Known Recall | 3.88 | 1.08 | 3 |
| stackoverflow | 0.75 | OOS F1 | 1.66 | 2.14 | 3 |

## 结论边界

- KIR 变化下，Trainable 与 ADB 的差异依赖数据集；不能把 StackOverflow 的接近结果外推到 CLINC150 或 Banking77。
- CLINC150 的 Trainable OOS F1 接近 ADB，但 F1-All 与 Known Recall 低于 ADB；综合指标不是全面领先。
- Banking77 和 StackOverflow 上 Trainable 的 F1-All 高于 ADB，但 Known Recall 可能低于或接近 ADB，仍需报告工作点权衡。
- 该报告不能证明 Trainable 超过论文中的完整 ADB，也不能和 MOGB 论文值或 DCLOOS 外部 OOS 监督结果直接排名。

图表：`figures/archive/adb_kir_sensitivity_incomplete_v1/`；机器可读结果：`results/analysis/archive/adb_kir_sensitivity_incomplete_v1/`。
