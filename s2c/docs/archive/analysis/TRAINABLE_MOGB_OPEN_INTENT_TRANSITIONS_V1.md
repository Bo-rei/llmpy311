# S2C Trainable K1 与 MOGB-Fair 开放意图结果转移分析 V1

更新时间：2026-08-09  
活动协议：`protocol_v2_textoir_v1`

## 1. 问题与合同

本阶段不再只比较 Known/OOS 二分类 Gate，而是把每条测试样本分成五种互斥结果：Known 正确、Known 错类、Known 被拒、OOS 正确拒绝、OOS 误接收。比较范围为三数据集、KIR=0.25/0.50/0.75、五个正式 seed，共45个逐样本配对单元。

`S2C-Trainable-K1` 是当前 Known-only、最后两层 MiniLM+projection、单中心 Gate；`MOGB-Fair` 是相同 TEXTOIR split 上冻结 MiniLM 的自适应粒球组件。后者不是官方 BERT MOGB。本分析只读取冻结预测，test标签只用于事后归因。

## 2. 完整性

- 配对单元：45/45；Banking77/CLINC150/StackOverflow 每方法测试样本数分别为 `3080`/`5700`/`6000`；
- 总预测读取量：`443400`行；逐配对 sample_id、gold_intent、Known/OOS标记完全一致；
- 五状态转移组合：25种，非零汇总组合 `108`；
- score-bin Known 样本累计计数：`101750`；
- 每单元重新计算 F1-All，并与冻结汇总核对。

## 3. 跨数据集差距来源

| dataset       |   f1_all_delta |   known_gain |   oos_gain |   recovery_from_reject |   recovery_from_wrong |
|:--------------|---------------:|-------------:|-----------:|-----------------------:|----------------------:|
| banking77     |         0.3248 |       0.4628 |    -0.1411 |               738.7333 |                0.2000 |
| clinc150      |         0.3514 |       0.4117 |    -0.0285 |               963.8667 |                1.2000 |
| stackoverflow |         0.4367 |       0.5616 |    -0.0653 |              1734.0667 |                2.5333 |

`known_gain` 和 `oos_gain` 分别是同一配对单元中 S2C-only correct 减去 MOGB-only correct，再除以对应 Known/OOS 样本数。结果直接显示：S2C 相对 MOGB-Fair 的主要优势来自恢复被 MOGB 边界拒绝的 Known 样本，而不是通过额外接受 OOS 换取表面指标。

## 4. KIR=0.50 的配对结果

| dataset       |   f1_all_delta_mean |   net_known_correct_gain_rate |   net_oos_correct_gain_rate |   known_trainable_recovers_mogb_reject_mean |   known_trainable_fixes_mogb_wrong_mean |
|:--------------|--------------------:|------------------------------:|----------------------------:|--------------------------------------------:|----------------------------------------:|
| banking77     |              0.3307 |                        0.4676 |                     -0.1465 |                                    713.8000 |                                  0.4000 |
| clinc150      |              0.3686 |                        0.4218 |                     -0.0279 |                                    953.8000 |                                  1.0000 |
| stackoverflow |              0.4325 |                        0.5606 |                     -0.0855 |                                   1681.0000 |                                  2.2000 |

`known_trainable_recovers_mogb_reject` 是 S2C 正确分类、但 MOGB 判成 OOS 的 Known 样本数；`known_trainable_fixes_mogb_wrong` 是 S2C 修正 MOGB Known 错类的数量。前者明显更大，说明两者差距首先是 MOGB 边界覆盖问题，其次才是粒球之间的 Known 分类竞争。

## 5. 机制结论

1. **F1-All 优势主要来自 Known 覆盖恢复。** MOGB-Fair 的平均 false acceptance 很低，但大量 Known 样本落在所有平均半径粒球之外；S2C 的单中心边界恢复了这些样本。
2. **表示排序仍然重要。** S2C 不只把 MOGB 拒绝的样本强行接收；它需要同时给出正确 Known intent。意图级正确率与 MOGB score-bin 曲线显示，很多 MOGB score>1 的样本仍可被 S2C 正确分类。
3. **MOGB 并非所有错误都来自边界。** 仍存在双方都接受但 MOGB 选错 Known intent、S2C 修正的样本；因此子中心表示/最近球竞争也是次级差距来源。
4. **不能据此比较官方论文 SOTA。** 当前结论只适用于 MOGB-Fair 组件合同；官方 BERT MOGB 的公开代码兼容复现仍需单独解释数据、训练目标和半径合同差异。

## 6. 图表

- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/open_intent_transition_heatmaps_kir050.png`
- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/paired_correctness_gain_decomposition.png`
- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/known_recovery_by_mogb_score_bin.png`
- `figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/top_intent_known_recovery_kir050.png`

意图图只显示至少在3/5个seed中进入Known集合的意图，避免偶然类别划分产生的单seed极值；完整CSV保留全部意图并记录`n_seeds`。

## 7. 研究边界

- 没有重新训练、调阈值、改split或覆盖历史artifact；
- test标签没有参与模型、边界、K或工作点选择；
- 不输出原始文本或逐样本ID到Git轻量结果；
- 该结果用于解释“为什么当前同协议 S2C 行优于 MOGB-Fair”，不能写成超过完整 MOGB、ADB、DA-ADB或DCLOOS。
