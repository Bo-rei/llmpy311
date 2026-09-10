# MOGB逐粒球开放空间风险归因 V1

更新时间：2026-08-09  
活动协议：`protocol_v2_textoir_v1`

## 1. 分析问题

本阶段回答：当前 `S2C-Trainable-K1` 为什么在同协议矩阵中优于 MOGB-Fair？差距究竟来自少数危险粒球、普遍的小球碎片化，还是边界工作点？

分析复用三数据集、KIR=0.25/0.50/0.75、五seed的90个已冻结方法单元。`MOGB-Fair` 与 `MOGB partition + S2C boundary` 使用完全相同的自适应粒球；唯一差异是距离与边界。测试标签仅用于事后错误归因，不参与任何选择或训练。

## 2. 完整性

- 方法单元：`90/90`；
- 逐粒球行：`8930`；
- 数据集×KIR×seed：45个；
- 每个逐样本 `nearest_ball` 都属于 selected ball；
- 从 predictions 重算的 OOS assignment、false acceptance、Known false rejection 与 `balls.jsonl` 存储计数逐球完全一致；
- 两种方法的 selected ball ID、标签、训练支持、深度和纯度逐单元完全一致。

## 3. 数据集级结果

| dataset       | method_label                  |   selected_balls |   tiny_support_lt20_ratio |   top10pct_false_accept_share |   top10pct_known_false_reject_share |   oos_f1 |   f1_all |   known_recall |   false_accept_rate |   false_reject_rate |   trainable_minus_mogb_oos_f1 |   trainable_minus_mogb_f1_all |
|:--------------|:------------------------------|-----------------:|--------------------------:|------------------------------:|------------------------------------:|---------:|---------:|---------------:|--------------------:|--------------------:|------------------------------:|------------------------------:|
| banking77     | MOGB partition + S2C boundary |          93.2667 |                    0.5071 |                        0.9245 |                              0.435  |   0.7486 |   0.6486 |         0.5324 |              0.0386 |              0.4676 |                        0.0643 |                        0.1658 |
| banking77     | MOGB-Fair                     |          93.2667 |                    0.5071 |                        0.9682 |                              0.2379 |   0.7098 |   0.4896 |         0.3405 |              0.0112 |              0.6595 |                        0.1031 |                        0.3248 |
| clinc150      | MOGB partition + S2C boundary |         136.467  |                    0.3614 |                        0.9365 |                              0.5138 |   0.8389 |   0.6479 |         0.5387 |              0.0221 |              0.4613 |                        0.056  |                        0.1669 |
| clinc150      | MOGB-Fair                     |         136.467  |                    0.3614 |                        0.984  |                              0.2364 |   0.7988 |   0.4633 |         0.3263 |              0.008  |              0.6737 |                        0.0961 |                        0.3514 |
| stackoverflow | MOGB partition + S2C boundary |          67.9333 |                    0.4917 |                        0.9021 |                              0.7107 |   0.7447 |   0.6213 |         0.4952 |              0.019  |              0.5048 |                        0.1201 |                        0.2503 |
| stackoverflow | MOGB-Fair                     |          67.9333 |                    0.4917 |                        0.8898 |                              0.3333 |   0.6932 |   0.4349 |         0.2696 |              0.008  |              0.7304 |                        0.1717 |                        0.4367 |

上表中 `Trainable-minus-MOGB` 是同 dataset×KIR×seed 配对后的均值。它不是对官方 BERT MOGB 论文的排名，只解释当前 Frozen MiniLM 公平组件。

## 4. 主要机制结论

1. **粒球错误高度集中。** 每个单元中最高风险10%的粒球承担了远高于其数量占比的 OOS false acceptance 或 Known false rejection。因此整体结果不是“所有粒球都同样差”，而是少数边界承担了主要开放空间风险。
2. **纯度不能解释开放空间风险。** 被选择的粒球纯度几乎全部为1；纯度只说明 Known 标签一致，无法说明粒球是否朝 OOS 密集方向扩张。
3. **训练支持和半径主要解释 Known 误拒，而不是稳定解释 OOS 误接收。** 两者与 Known false rejection 的逐单元 Spearman 通常为中等正相关；与 OOS false acceptance 的相关性明显更弱且受边界影响。它们不能单独作为安全删球规则。
4. **边界是主要放大器之一。** 在同一粒球分区上替换 S2C 边界会显著恢复 Known coverage/F1-All，但也改变 OOS false acceptance 的集中位置；这验证了“分区结构”和“边界工作点”必须分开分析。
5. **Trainable K1 的优势不是简单拒绝更多样本。** MOGB-Fair 的主要代价是高 Known false rejection；S2C边界混合版恢复覆盖后仍落后，说明当前差距同时涉及表示排序、粒球结构与局部边界。

## 5. 结构变量与错误的逐单元Spearman

| dataset       | method                       | predictor         | outcome            |   mean |   std |   count |
|:--------------|:-----------------------------|:------------------|:-------------------|-------:|------:|--------:|
| banking77     | mogb_minilm                  | structural_radius | known_false_reject |  0.261 | 0.103 |      15 |
| banking77     | mogb_minilm                  | structural_radius | oos_false_accept   |  0.151 | 0.124 |      15 |
| banking77     | mogb_minilm                  | train_support     | known_false_reject |  0.43  | 0.137 |      15 |
| banking77     | mogb_minilm                  | train_support     | oos_false_accept   |  0.081 | 0.109 |      15 |
| banking77     | mogb_partition_ours_boundary | structural_radius | known_false_reject |  0.661 | 0.053 |      15 |
| banking77     | mogb_partition_ours_boundary | structural_radius | oos_false_accept   |  0.217 | 0.111 |      15 |
| banking77     | mogb_partition_ours_boundary | train_support     | known_false_reject |  0.527 | 0.095 |      15 |
| banking77     | mogb_partition_ours_boundary | train_support     | oos_false_accept   |  0.243 | 0.117 |      15 |
| clinc150      | mogb_minilm                  | structural_radius | known_false_reject |  0.438 | 0.063 |      15 |
| clinc150      | mogb_minilm                  | structural_radius | oos_false_accept   |  0.193 | 0.074 |      15 |
| clinc150      | mogb_minilm                  | train_support     | known_false_reject |  0.56  | 0.08  |      15 |
| clinc150      | mogb_minilm                  | train_support     | oos_false_accept   |  0.151 | 0.077 |      15 |
| clinc150      | mogb_partition_ours_boundary | structural_radius | known_false_reject |  0.618 | 0.041 |      15 |
| clinc150      | mogb_partition_ours_boundary | structural_radius | oos_false_accept   |  0.265 | 0.101 |      15 |
| clinc150      | mogb_partition_ours_boundary | train_support     | known_false_reject |  0.519 | 0.053 |      15 |
| clinc150      | mogb_partition_ours_boundary | train_support     | oos_false_accept   |  0.215 | 0.055 |      15 |
| stackoverflow | mogb_minilm                  | structural_radius | known_false_reject |  0.536 | 0.189 |      15 |
| stackoverflow | mogb_minilm                  | structural_radius | oos_false_accept   |  0.255 | 0.177 |      15 |
| stackoverflow | mogb_minilm                  | train_support     | known_false_reject |  0.462 | 0.135 |      15 |
| stackoverflow | mogb_minilm                  | train_support     | oos_false_accept   |  0.219 | 0.117 |      15 |
| stackoverflow | mogb_partition_ours_boundary | structural_radius | known_false_reject |  0.601 | 0.106 |      15 |
| stackoverflow | mogb_partition_ours_boundary | structural_radius | oos_false_accept   |  0.332 | 0.144 |      15 |
| stackoverflow | mogb_partition_ours_boundary | train_support     | known_false_reject |  0.593 | 0.089 |      15 |
| stackoverflow | mogb_partition_ours_boundary | train_support     | oos_false_accept   |  0.449 | 0.131 |      15 |

这些相关性是描述性机制分析，不用于选择粒球。特别是 test OOS false acceptance 绝不能作为正式删球规则。

## 6. 小粒球比例

| dataset       | method_label                  |   tiny_support_lt_20 |
|:--------------|:------------------------------|---------------------:|
| banking77     | MOGB partition + S2C boundary |             0.540386 |
| banking77     | MOGB-Fair                     |             0.540386 |
| clinc150      | MOGB partition + S2C boundary |             0.376649 |
| clinc150      | MOGB-Fair                     |             0.376649 |
| stackoverflow | MOGB partition + S2C boundary |             0.504416 |
| stackoverflow | MOGB-Fair                     |             0.504416 |

小粒球是明显现象，但不是主要错误承担者：支持量较大的 Q3/Q4 粒球贡献了更多 OOS false acceptance 和 Known false rejection。同样的粒球分区换边界后，错误预算仍显著改变，因此不能把失败简单归因于 tiny cluster。

## 7. 图表

- `figures/archive/analysis/mogb_ball_risk_attribution_v1/top10pct_ball_error_concentration.png`
- `figures/archive/analysis/mogb_ball_risk_attribution_v1/ball_support_radius_risk.png`
- `figures/archive/analysis/mogb_ball_risk_attribution_v1/support_quartile_error_budget.png`
- `figures/archive/analysis/mogb_ball_risk_attribution_v1/ball_risk_vs_trainable_gap.png`
- `figures/archive/analysis/mogb_ball_risk_attribution_v1/intent_ball_risk_heatmap.png`

## 8. 研究边界

- 这是对已完成预测的分析，不是新的模型训练；
- `MOGB-Fair` 是 Frozen MiniLM 组件适配，不是官方 BERT MOGB；
- test标签只用于解释，不用于调粒球、半径或阈值；
- 结果支持“纯度导向粒球不等价于OOS风险导向结构”，但不能据此宣布官方MOGB算法无效。

结果manifest SHA256：`aca7b154c60aedbe964c4582c9bfa0d5acd02ab0cd6640ae24a59ff64e047e28`。
