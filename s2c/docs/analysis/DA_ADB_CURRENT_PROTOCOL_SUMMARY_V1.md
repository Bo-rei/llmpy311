# DA-ADB 当前协议三 seed 汇总 V1

## 运行范围

- 数据集：StackOverflow；KIR=`0.50`；seed=`42, 87, 100`。
- DA-ADB：BERT/TextOIR 外部兼容合同，使用当前 protocol_v2 的独立 split 根和 seed-specific Known 列表。
- S2C：同一 seed 的 Known-only MiniLM Trainable K=1 fair 行。
- 测试 OOS 只用于最终评价，不参与训练、epoch、阈值或参数选择。
- 两者仍不是同骨干/同训练合同，本报告只作配对描述和合同审计。

## 当前协议结果

| 方法 | OOS F1 | F1-All | F1-Known | Accuracy | Known Recall | FA | FR |
|---|---:|---:|---:|---:|---:|---:|---:|
| DA-ADB | 72.48 ± 6.24 | 74.02 ± 3.13 | 74.18 ± 2.86 | 72.23 ± 5.12 | 75.97 ± 4.10 | 29.07 ± 11.19 | 24.03 ± 4.10 |
| S2C Trainable K=1 | 88.21 ± 1.72 | 86.69 ± 1.45 | 86.54 ± 1.42 | 87.23 ± 1.72 | 83.68 ± 0.28 | 8.18 ± 3.36 | 16.32 ± 0.28 |

## 配对差值：S2C Trainable K=1 − DA-ADB

| 指标 | 平均差值 | 95% bootstrap CI | S2C 胜出/DA-ADB 胜出/平局 |
|---|---:|---:|---:|
| oos_f1 | +15.73pp | [+10.78, +19.70]pp | 3/0/0 |
| f1_all | +12.67pp | [+10.73, +13.74]pp | 3/0/0 |
| f1_known | +12.36pp | [+10.72, +13.44]pp | 3/0/0 |
| accuracy | +15.00pp | [+11.13, +17.53]pp | 3/0/0 |
| known_recall | +7.71pp | [+3.37, +10.57]pp | 3/0/0 |
| false_acceptance | -20.89pp | [-28.73, -12.90]pp | 0/3/0 |
| false_rejection | -7.71pp | [-10.57, -3.37]pp | 0/3/0 |

## 解释边界

当前协议下 DA-ADB 三个 seed 的 OOS F1 分别为
`70.82%, 67.22%, 79.38%`，
S2C Trainable K=1 分别为
`87.55%, 86.92%, 90.16%`。
因此旧兼容单格 `90.90%` 不能代表当前协议 DA-ADB；当前三 seed 均值也不能与历史论文表或 DCLOOS reduced 结果直接排名。

当前证据只支持：在这三个 StackOverflow protocol_v2 工作点上，S2C Trainable K=1 的 OOS F1 和 F1-All 更高，
而 DA-ADB 的结果方差较大；这仍同时包含 MiniLM/BERT 表示、训练目标和外部适配层差异。

## 证据文件

- `results/analysis/da_adb_current_protocol_summary_v1/per_seed.csv`
- `results/analysis/da_adb_current_protocol_summary_v1/summary_mean_std_ci.csv`
- `results/analysis/da_adb_current_protocol_summary_v1/paired_deltas.csv`
- `figures/da_adb_current_protocol_summary_v1/current_protocol_seed_metrics.png`
- `figures/da_adb_current_protocol_summary_v1/current_protocol_tradeoff.png`
- 旧/新合同拆分：`docs/analysis/DA_ADB_CONTRACT_COMPARISON_V1.md`
