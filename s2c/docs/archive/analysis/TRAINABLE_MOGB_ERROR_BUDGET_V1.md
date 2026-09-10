# Trainable-K1 与 MOGB-Fair 错误预算分析（V1）

> 本报告是 analysis-only 证据，不训练模型、不选择参数、不使用测试 OOS 调参，也不修改原始逐样本 artifact。

## 1. 核心问题

这里比较的是同一 `protocol_v2_textoir_v1`、同一 registry、同一 sample_id 集合下的 `S2C-Trainable-K1` 与 `MOGB-MiniLM-Fair`。因此它回答的是：在相同冻结/训练后表示和评价合同中，两者的错误预算为什么不同；不是对完整 BERT MOGB 论文的排名。

## 2. 完整性审计

- 源行数：45（3 数据集 × 3 KIR × 5 seeds）；
- dataset×KIR 汇总组：9；
- `net_known_correct_gain + net_oos_correct_gain = net_total_correct_gain` 不一致数：0；
- 选择参数：否；所有行均为冻结结果的事后归因。

## 3. 机制结论

1. Trainable-K1 的净收益主要来自恢复 MOGB 拒绝的 Known 样本；这不是少数意图的偶然现象，而是三个数据集和三个 KIR 上都存在的广泛覆盖差异。
2. Trainable-K1 通常会损失一部分原本被 MOGB 正确拒绝的 OOS 样本，但恢复的 Known 正确量更大，因此 F1-All 和 Known 分类工作点显著改善。
3. MOGB-Fair 的低 false acceptance 伴随大规模 Known false rejection；它更像保守的拒识工作点，而不是全面更强的开放意图分类器。
4. StackOverflow/KIR=.50 的净 Known 正确恢复约 1,682 个，而净 OOS 正确拒绝减少约 256 个；这直接解释了为什么当前方法的 F1-All 从约 43.30% 提升到约 86.55%。

## 4. 可视化

- `figures/archive/analysis/trainable_mogb_error_budget_v1/error_budget_decomposition.svg`：每个 dataset×KIR 的 Known 恢复、OOS 正确拒绝变化和 F1-All 增量；
- 原始五状态转移图：`figures/archive/analysis/trainable_mogb_open_intent_transitions_v1/open_intent_transition_heatmaps_kir050.png`；
- 配对工作点图：`figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/matched_known_recall_frontier.png`。

## 5. 结论边界

该分析支持“当前 Trainable-K1 在相同 MiniLM Gate 合同下比 MOGB-Fair 更平衡”的结论；不能支持“当前方法超过完整 MOGB 论文方法”或“所有监督条件下达到 SOTA”。MOGB 官方 BERT 复现和 DCLOOS 外部监督仍需独立合同报告。

机器可读结果：`results/analysis/archive/analysis/trainable_mogb_error_budget_v1/`。
