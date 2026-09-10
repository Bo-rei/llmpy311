# S2C 与 MOGB-Fair 工作点和排序能力归因 V1

> 本阶段只重放已冻结的 45 个同 split 配对单元。默认阈值结果是正式已完成结果；oracle threshold 和
> matched-known-recall 使用 test 标签，只作为事后机制诊断，严禁用于选择模型、阈值或论文主结果。

## 核心结论

1. 默认 OOS F1 上 S2C 胜出 `44/45` 个单元，阈值无关 AUROC 上胜出 `45/45` 个单元；
   两者同时胜出的单元为 `44/45`。因此当前优势不只是阈值偏移，还包含分数排序质量差异。
2. 45 单元平均 AUROC 差值为 `+7.93pp`，AUPR-OOS 差值为
   `+12.00pp`；默认 OOS F1 差值为 `+12.36pp`。
3. 事后最优阈值下，S2C 相对 MOGB 的 OOS F1 上限仍相差
   `+6.43pp`。这说明仅移动 MOGB 默认 mean-radius 工作点不能关闭全部差距。
4. MOGB 的默认工作点更保守，但它的 default-to-oracle gap 平均为
   `7.12pp`；S2C 为
   `1.19pp`。该差距量化了两者的校准损失。
5. 在事后匹配约 80% Known coverage 时，S2C 与 MOGB 的 OOS F1/false acceptance 见下表；这只用于
   判断同覆盖前沿，不构成新的正式指标。

## 45 单元总体均值

| method            |   oos_f1 |   auroc |   aupr_oos |   oracle_oos_f1 |   default_to_oracle_gap |   known_recall |
|:------------------|---------:|--------:|-----------:|----------------:|------------------------:|---------------:|
| MOGB-MiniLM-Fair  |   0.7339 |  0.8519 |     0.7769 |          0.8051 |                  0.0712 |         0.3121 |
| S2C Trainable K=1 |   0.8575 |  0.9312 |     0.8970 |          0.8694 |                  0.0119 |         0.8022 |

## 事后匹配 80% Known coverage

| method            |   oos_f1 |   false_accept_rate |   known_recall |
|:------------------|---------:|--------------------:|---------------:|
| MOGB-MiniLM-Fair  |   0.7494 |              0.2620 |         0.8006 |
| S2C Trainable K=1 |   0.8605 |              0.0732 |         0.8006 |

## 每数据集与 KIR 的配对差值（百分点）

| dataset       |   kir |   aupr_oos |   auroc |   oos_f1 |   oracle_oos_f1 |
|:--------------|------:|-----------:|--------:|---------:|----------------:|
| banking77     | +0.25 |      +3.08 |   +5.39 |    +1.35 |           +1.28 |
| banking77     | +0.50 |      +9.91 |   +6.58 |    +8.57 |           +4.16 |
| banking77     | +0.75 |     +22.18 |   +9.13 |   +21.02 |          +11.63 |
| clinc150      | +0.25 |      +2.42 |   +5.28 |    +2.78 |           +1.26 |
| clinc150      | +0.50 |      +8.17 |   +7.69 |    +9.12 |           +4.70 |
| clinc150      | +0.75 |     +17.24 |   +8.57 |   +16.92 |           +8.45 |
| stackoverflow | +0.25 |      +5.17 |   +8.57 |    +6.04 |           +4.13 |
| stackoverflow | +0.50 |     +10.70 |   +7.58 |   +14.75 |           +6.55 |
| stackoverflow | +0.75 |     +29.14 |  +12.58 |   +30.72 |          +15.67 |

## 六张机制图

1. `threshold_free_delta_heatmap.png`：AUROC、AUPR-OOS 和分数分离度的配对差值。
2. `default_vs_oracle_oos_f1.png`：默认阈值与事后 OOS F1 上限。
3. `matched_known_recall_frontier.png`：KIR=.50 相同 Known coverage 下的 OOS F1 前沿。
4. `score_quantiles_kir050.png`：Known/OOS 分数的10%--90%分位区间。
5. `calibration_gap_heatmap.png`：默认工作点离事后最优阈值的距离。
6. `ranking_vs_workpoint.png`：AUROC优势与默认OOS F1优势是否同步。

## 解释边界

- `S2C-Trainable-K1` 与 `MOGB-MiniLM-Fair` 共享 dataset/KIR/seed 和 evaluator，但表示与边界合同不同。
- 该结果解释当前公平组件差距；不能替代完整 BERT MOGB、历史 Cascade、ADB/DA-ADB 或 DCLOOS 主表。
- oracle 和 matched-coverage 结果是测试敏感性分析，不能进入任何参数选择或正式方法声明。
- Manifest SHA256：`500bf78e3c2347d8d96a6df1f2804b72a851e112acdc9ed4171b971f60f19d58`。
