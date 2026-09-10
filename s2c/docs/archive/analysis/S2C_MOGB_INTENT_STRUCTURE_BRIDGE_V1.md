# S2C 与 MOGB-Fair 逐意图结构--收益桥接 V1

> 本报告联合两份已冻结的 analysis-only 证据：逐 intent 错误归因与 MOGB selected-ball 结构。
> test错误只用于事后解释，不能删球、选K、调阈值或生成自适应方法。

## 核心结论

1. 共桥接 `1850` 条 intent×seed 记录。S2C 相对 MOGB 的逐intent平均 Known
   拒绝率降低 `46.70pp`，同时每个intent平均新增 OOS
   误接收率 `0.19pp`。
2. `ball_count=0` 的行明确表示 MOGB selected-ball 阶段遗漏该注册 Known intent，其占比为
   `1.41%`；它是覆盖缺陷，但不是全部差距。
3. 每个 dataset/KIR 中，正向 Known 恢复 intent 的比例和恢复集中度见表。该分析区分“广泛恢复”与
   “少数高风险 intent 主导”，避免只看全局均值。
4. selected-ball数量与 Known恢复率的九个 dataset×KIR Spearman rho 范围为
   `-0.382` 到 `+0.445`。这说明自适应粒球复杂度与S2C优势的关系具有
   数据集/KIR依赖性，不能把更多粒球直接解释为更好的开放集结构。
5. 样本数恢复--代价图只衡量 Gate 覆盖，不等于完整 Known 分类正确率；完整五状态分类证据仍以
   `TRAINABLE_MOGB_OPEN_INTENT_TRANSITIONS_V1.md` 为准。

## dataset×KIR 汇总

| dataset       |   kir |   known_recovery_count_mean |   known_recovery_count_std |   extra_oos_accept_count_mean |   extra_oos_accept_count_std |   coverage_tradeoff_net_count_mean |   positive_recovery_intent_ratio_mean |   missing_selected_ball_ratio_mean |   mean_ball_count |
|:--------------|------:|----------------------------:|---------------------------:|------------------------------:|-----------------------------:|-----------------------------------:|--------------------------------------:|-----------------------------------:|------------------:|
| banking77     |  0.25 |                      318.20 |                      19.07 |                        204.80 |                        72.53 |                             113.40 |                                100.00 |                               0.00 |              2.28 |
| banking77     |  0.50 |                      741.80 |                      30.34 |                        228.60 |                        46.49 |                             513.20 |                                100.00 |                               0.00 |              2.50 |
| banking77     |  0.75 |                     1244.60 |                      30.55 |                        143.20 |                        34.33 |                            1101.40 |                                100.00 |                               1.72 |              2.44 |
| clinc150      |  0.25 |                      416.40 |                      26.29 |                        117.60 |                        44.93 |                             298.80 |                                100.00 |                               0.00 |              1.76 |
| clinc150      |  0.50 |                      964.60 |                      28.43 |                         96.20 |                        26.20 |                             868.40 |                                100.00 |                               0.80 |              1.80 |
| clinc150      |  0.75 |                     1543.00 |                      19.60 |                         74.60 |                        10.24 |                            1468.40 |                                100.00 |                               1.43 |              1.85 |
| stackoverflow |  0.25 |                      775.20 |                      76.24 |                        131.00 |                        87.76 |                             644.20 |                                100.00 |                               0.00 |              7.48 |
| stackoverflow |  0.50 |                     1704.00 |                      74.61 |                        256.40 |                        90.44 |                            1447.60 |                                100.00 |                               4.00 |              6.98 |
| stackoverflow |  0.75 |                     2832.40 |                      93.39 |                        122.00 |                        83.39 |                            2710.40 |                                100.00 |                              10.67 |              6.44 |

## KIR=0.50 按 MOGB 每intent粒球数分组

| dataset       | ball_count_group   |   intent_seed_rows |   known_recovery_rate_mean |   extra_oos_accept_rate_mean |   coverage_tradeoff_net_count_mean |
|:--------------|:-------------------|-------------------:|---------------------------:|-----------------------------:|-----------------------------------:|
| banking77     | 1                  |                 41 |                      43.11 |                         0.34 |                              12.00 |
| banking77     | 2                  |                 57 |                      48.73 |                         0.35 |                              13.96 |
| banking77     | 3+                 |                 92 |                      51.39 |                         0.43 |                              13.89 |
| clinc150      | 0                  |                  3 |                      73.33 |                         0.02 |                              21.33 |
| clinc150      | 1                  |                156 |                      39.55 |                         0.03 |                              10.81 |
| clinc150      | 2                  |                141 |                      45.22 |                         0.05 |                              12.00 |
| clinc150      | 3+                 |                 75 |                      44.13 |                         0.04 |                              11.99 |
| stackoverflow | 0                  |                  2 |                      83.00 |                         0.35 |                             238.50 |
| stackoverflow | 1                  |                  3 |                      71.89 |                         1.33 |                             175.67 |
| stackoverflow | 2                  |                  3 |                      68.44 |                         0.98 |                             176.00 |
| stackoverflow | 3+                 |                 42 |                      53.64 |                         0.84 |                             135.86 |

## 恢复集中度

| dataset       |   kir |   positive_recovery_intent_ratio |   top10pct_recovery_share |   top20pct_recovery_share |   recovery_gini |
|:--------------|------:|---------------------------------:|--------------------------:|--------------------------:|----------------:|
| banking77     |  0.25 |                           100.00 |                     15.37 |                     29.37 |            0.16 |
| banking77     |  0.50 |                           100.00 |                     15.76 |                     29.32 |            0.16 |
| banking77     |  0.75 |                           100.00 |                     15.01 |                     28.23 |            0.15 |
| clinc150      |  0.25 |                           100.00 |                     16.81 |                     31.31 |            0.20 |
| clinc150      |  0.50 |                           100.00 |                     17.81 |                     30.89 |            0.21 |
| clinc150      |  0.75 |                           100.00 |                     17.33 |                     30.88 |            0.21 |
| stackoverflow |  0.25 |                           100.00 |                     27.11 |                     27.11 |            0.10 |
| stackoverflow |  0.50 |                           100.00 |                     14.47 |                     26.86 |            0.11 |
| stackoverflow |  0.75 |                           100.00 |                     17.95 |                     26.79 |            0.12 |

## 六张机制图

1. `intent_known_recovery_distribution.png`：逐intent Known覆盖恢复的全分布。
2. `ball_count_recovery_tradeoff.png`：粒球数分组后的 Known恢复与新增OOS误收。
3. `ball_count_vs_recovery.png`：结构复杂度与恢复幅度散点。
4. `known_recovery_concentration.png`：Top intents承担多少恢复贡献。
5. `top_intent_recovery_kir050.png`：三个数据集恢复最大的具体intent及其平均粒球数。
6. `known_recovery_vs_oos_cost.png`：九个dataset×KIR的样本数收益--代价。

## 解释边界

- MOGB-Fair 使用 Frozen MiniLM；S2C 使用 Known-only Trainable MiniLM K=1，因此结构与表示同时变化。
- 该桥接说明 MOGB 的 adaptive balls 在哪些intent上形成覆盖风险，不证明完整BERT MOGB无效。
- 所有 test-defined intent风险和相关性只用于解释，不得成为新的ball selector。
- Manifest SHA256：`2b139c5b23fea877e5c01cf1ba73ed8e65b6c9f3088c72fdc77d968b45ccc0f9`。
