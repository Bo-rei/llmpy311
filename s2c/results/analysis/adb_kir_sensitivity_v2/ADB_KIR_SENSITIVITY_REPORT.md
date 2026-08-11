# ADB 跨 KIR 三数据集外部合同分析

本报告只使用 protocol_v2_textoir_v1 的固定 split、隔离 TextOIR BERT runtime 和已审计 y_true/y_pred。
ADB 是 BERT/TextOIR 外部合同；S2C Trainable K=1 是 Known-only MiniLM。所有差值仅用于机制与工作点分析，
不进入 MiniLM fair 主排名，也不用于测试集选参。

有效单元：`45/45`；当前每个 dataset×KIR 的 seed 数为 `5`，预期规模为 3 数据集×3 KIR×5 seed = `45`。

## ADB 绝对结果

| 数据集 | KIR | OOS F1 | F1-All | Known Recall | False Acceptance | n |
|---|---:|---:|---:|---:|---:|---:|
| banking77 | 0.25 | 82.80±2.28 | 68.58±2.19 | 87.47±1.32 | 26.41±3.31 | 5 |
| banking77 | 0.50 | 74.97±1.92 | 78.79±1.53 | 89.14±0.67 | 33.67±2.71 | 5 |
| banking77 | 0.75 | 66.27±2.34 | 85.89±0.35 | 90.35±0.27 | 35.82±3.21 | 5 |
| clinc150 | 0.25 | 91.60±0.88 | 76.61±1.15 | 90.98±0.91 | 13.58±1.67 | 5 |
| clinc150 | 0.50 | 89.39±0.91 | 85.49±0.77 | 90.84±0.61 | 14.35±1.83 | 5 |
| clinc150 | 0.75 | 84.76±1.35 | 88.52±0.64 | 91.54±0.17 | 17.49±2.30 | 5 |
| stackoverflow | 0.25 | 93.14±1.75 | 83.12±3.05 | 83.52±1.98 | 8.03±2.74 | 5 |
| stackoverflow | 0.50 | 87.21±1.17 | 85.81±1.15 | 81.37±1.33 | 8.27±1.80 | 5 |
| stackoverflow | 0.75 | 73.70±1.00 | 85.99±0.39 | 80.84±0.94 | 8.12±1.55 | 5 |

## S2C Trainable K=1 − ADB 配对差值

| 数据集 | KIR | 指标 | 均值(pp) | 标准差(pp) | n |
|---|---:|---|---:|---:|---:|
| banking77 | 0.25 | F1-All | 9.93 | 3.26 | 5 |
| banking77 | 0.25 | Known Recall | -5.63 | 1.76 | 5 |
| banking77 | 0.25 | OOS F1 | 8.85 | 1.77 | 5 |
| banking77 | 0.50 | F1-All | 2.89 | 1.11 | 5 |
| banking77 | 0.50 | Known Recall | -6.93 | 1.12 | 5 |
| banking77 | 0.50 | OOS F1 | 8.60 | 2.04 | 5 |
| banking77 | 0.75 | F1-All | -1.76 | 0.90 | 5 |
| banking77 | 0.75 | Known Recall | -7.94 | 0.43 | 5 |
| banking77 | 0.75 | OOS F1 | 2.39 | 3.70 | 5 |
| clinc150 | 0.25 | F1-All | 3.08 | 1.46 | 5 |
| clinc150 | 0.25 | Known Recall | -17.35 | 1.54 | 5 |
| clinc150 | 0.25 | OOS F1 | 3.57 | 0.69 | 5 |
| clinc150 | 0.50 | F1-All | -3.67 | 1.26 | 5 |
| clinc150 | 0.50 | Known Recall | -16.40 | 0.79 | 5 |
| clinc150 | 0.50 | OOS F1 | 1.05 | 0.97 | 5 |
| clinc150 | 0.75 | F1-All | -5.61 | 1.03 | 5 |
| clinc150 | 0.75 | Known Recall | -16.42 | 0.87 | 5 |
| clinc150 | 0.75 | OOS F1 | -1.91 | 1.49 | 5 |
| stackoverflow | 0.25 | F1-All | 4.52 | 1.93 | 5 |
| stackoverflow | 0.25 | Known Recall | 0.77 | 1.75 | 5 |
| stackoverflow | 0.25 | OOS F1 | 2.34 | 1.13 | 5 |
| stackoverflow | 0.50 | F1-All | 0.75 | 0.84 | 5 |
| stackoverflow | 0.50 | Known Recall | 2.52 | 1.18 | 5 |
| stackoverflow | 0.50 | OOS F1 | 0.46 | 1.30 | 5 |
| stackoverflow | 0.75 | F1-All | 1.29 | 0.56 | 5 |
| stackoverflow | 0.75 | Known Recall | 3.27 | 1.18 | 5 |
| stackoverflow | 0.75 | OOS F1 | 2.60 | 2.10 | 5 |

## 结论边界

- KIR 变化下，Trainable 与 ADB 的差异依赖数据集；不能把 StackOverflow 的接近结果外推到 CLINC150 或 Banking77。
- CLINC150 的 Trainable OOS F1 接近 ADB，但 F1-All 与 Known Recall 低于 ADB；综合指标不是全面领先。
- Banking77 和 StackOverflow 上 Trainable 的 F1-All 高于 ADB，但 Known Recall 可能低于或接近 ADB，仍需报告工作点权衡。
- 该报告不能证明 Trainable 超过论文中的完整 ADB，也不能和 MOGB 论文值或 DCLOOS 外部 OOS 监督结果直接排名。

图表：`figures/adb_kir_sensitivity_v2`；机器可读结果由调用方指定输出目录。
