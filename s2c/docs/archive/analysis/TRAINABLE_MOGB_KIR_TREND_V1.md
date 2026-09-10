# Trainable-K1 与 MOGB-Fair 的 KIR 趋势分析（V1）

> 本报告只使用已完成的 protocol_v2 五 seed汇总，不重新训练、不使用测试集选参，也不将 MOGB 论文 BERT 数字混入同合同结果。

## 1. 结果

- Trainable-K1 相对 MOGB-Fair 的 OOS F1、F1-All 和 Known Recall 差值在三个数据集上均为正；
- 随 KIR 从 0.25 增加到 0.75，三项优势整体扩大；
- StackOverflow 的增幅最明显：OOS F1 差值从 +6.04pp 增加到 +30.72pp，F1-All 从 +33.57pp 增加到 +54.19pp；
- Banking77 的 OOS F1 差值从 +1.35pp 增加到 +21.02pp，但 false acceptance 差值也从 +8.83pp 增加到 +18.84pp，说明 Trainable 的覆盖恢复伴随更高的开放空间风险；
- CLINC150 的 OOS F1 差值从 +2.78pp 增加到 +16.92pp，Known Recall 差值从 +36.53pp 增加到 +45.92pp。

## 2. 机制解释

MOGB-Fair 的平均半径和粒球边界在 Known 类比例较低时已经偏保守；KIR 增大后，MOGB 需要覆盖更多 Known intent，但仍保留同样的保守边界逻辑，因此 Known false rejection 的差距扩大。Trainable-K1 的 Known-only 表示适配改善了单中心分数排序和覆盖，因而差距随 KIR 放大。

这不是“Trainable 在所有方面都更好”：false acceptance 在部分数据集会增加，尤其 Banking77；正确结论是 Trainable-K1 在当前合同下取得了更平衡的 Known/OOS 工作点。

## 3. 可视化与数据

- `figures/archive/analysis/trainable_mogb_kir_trend_v1/kir_trend.svg`：OOS F1、F1-All、Known Recall 的三面板趋势图；
- `results/analysis/archive/analysis/trainable_mogb_kir_trend_v1/kir_trend.csv`：Trainable、MOGB 绝对值和逐 KIR 差值；
- 同合同五 seed 主表：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`。

## 4. 边界

本趋势只证明当前 protocol_v2 中 Trainable-K1 相对冻结 MiniLM 的 MOGB 组件对照更稳定；不能外推为超过完整 BERT MOGB 论文、ADB 或 DCLOOS。
