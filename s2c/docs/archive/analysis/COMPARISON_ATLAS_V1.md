# 当前实验对比图谱 V1

> 本报告只做实验整理、对比和机制分析，不提出新方法，也不把不同数据合同的数字混合排名。

## 1. 当前到底比较哪些方法

当前主分析对象是 `protocol_v2_textoir_v1` 下的 3 数据集 × 3 KIR × 7 方法 × 5 seeds。
历史 `fulltex.tex` 的 `Ours` 是完整 Gate–Router–Expert Cascade；当前 `S2C Trainable K=1` 是 Known-only 训练后的 Gate-only 方法，两者不能直接合并成一个排名。
MOGB 公平行是冻结 MiniLM 组件；MOGB BERT 单格属于现代兼容复现，外部 ADB/DA-ADB/DCLOOS 属于兼容或不同监督合同。

## 2. 当前公平矩阵的总体结果

- S2C Trainable K=1：平均 OOS F1 **85.75%**，F1-All **83.36%**。
- Frozen K=1：平均 OOS F1 **78.64%**，F1-All **78.30%**。
- MOGB-MiniLM：平均 OOS F1 **73.39%**，F1-All **46.26%**，Known Recall **31.21%**。
- Trainable K=1 在当前公平矩阵的 OOS F1 9 个设置中 8 个第一，F1-All 9 个设置全部第一；这只是当前自有/组件矩阵结论，不是对完整官方基线的 SOTA 宣称。

## 3. 为什么当前自有方法在公平矩阵中更好

1. **表示层**：Trainable K=1 改善了单中心的 Known/OOS 分数分离；相对 Frozen K=1，结果表中的 delta 热图直接显示每个数据集和 KIR 的提升。
2. **决策层**：它使用单中心边界，避免固定多个球的接受区域并集快速扩大；因此相比 Frozen K=2，通常能减少 OOS 误接受，同时只付出有限 Known Recall。
3. **工作点层**：MOGB-MiniLM 的 false acceptance 很低，但 false rejection 极高；它更像保守拒识器，而不是完整分类性能更好的替代方案。

## 4. 固定多中心为什么受限

固定 K=2 的风险图把 OOS F1、false acceptance 和 Known Recall 的变化放在同一张图中。StackOverflow 的主要问题是新增中心扩大了接受区域：false acceptance 上升明显，而 OOS F1 下降；这不是简单的中心数不足。

## 5. MOGB 组件归因

- MOGB-MiniLM → MOGB partition + S2C boundary：隔离粒球划分与边界规则的变化。
- MOGB partition + S2C boundary → S2C partition + MOGB boundary：隔离 MOGB 平均半径/欧氏边界的影响。
- 这些组件行可以说明当前差距来自 Known 覆盖、半径和接受区域工作点，但不能替代 MOGB 官方 BERT 端到端复现。

## 6. 历史论文结果和外部基线边界

历史 `fulltex.tex` 的完整 Cascade 在旧协议下 9/9 个 dataset×KIR 单元的 OOS F1 高于表内基线；这证明旧论文主结果强，但不能直接证明当前 Gate-only Trainable K=1 已超过历史 Cascade。
ADB、DA-ADB 和 DCLOOS 的旧兼容数字只作参考，不进入当前公平主排名。StackOverflow/KIR=.50 的 ADB seed=42/87/100 已形成同数据合同的 BERT/TextOIR 外部单格参照；DA-ADB 兼容运行出现 NaN/全类预测并被标记无效。

## 7. 本轮图和机器可读结果

- `figures/archive/analysis/comparison_atlas_v1/fair_oos_f1_heatmap.png`：当前公平 OOS F1 热图。
- `figures/archive/analysis/comparison_atlas_v1/fair_f1_all_heatmap.png`：当前公平 F1-All 热图。
- `figures/archive/analysis/comparison_atlas_v1/trainable_minus_frozen_delta_heatmap.png`：Trainable 相对 Frozen 的提升。
- `figures/archive/analysis/comparison_atlas_v1/fixed_k2_risk_attribution.png`：固定双中心风险归因。
- `figures/archive/analysis/comparison_atlas_v1/mogb_component_tradeoff.png`：MOGB/S2C 组件工作点。
- `figures/archive/analysis/comparison_atlas_v1/fair_pareto_oos_known_recall.png`：OOS F1–Known Recall 权衡。
- `figures/archive/analysis/comparison_atlas_v1/seed_stability_oos_f1.png`：五 seed 稳定性。
- `docs/analysis/STACKOVERFLOW_EXTERNAL_COMPARISON_V1.md`：ADB 三 seed 与 DA-ADB 逐样本有效性审计。
- `figures/archive/analysis/comparison_atlas_v1/stackoverflow_external_single_cell_comparison.png`：同协议单格外部参照。

## 8. 结论边界和下一步

当前最可靠结论是：Trainable K=1 是当前统一 Known-only Gate 矩阵中最平衡的自有方法；固定 K>1 在 StackOverflow 有明显 boundary-union 风险；MOGB-MiniLM 的低误接受伴随严重 Known 拒绝。
仍缺少同一监督/数据合同下的 DA-ADB、DCLOOS 多 seed 主表，以及作者数据合同闭合后的官方 MOGB 复现。
DCLOOS 的外部 SQuAD 负样本来源已经找到，当前有一个 reduced-budget 兼容单元；默认预算超时，且
reduced 单元使用 pseudo-OOS/外部 OOS，不能进入 Known-only fair 主排名。下一步应优先维护外部合同
分层与逐样本错误分析，而不是继续堆 K 或新增损失。

## 9. DCLOOS 当前隔离 smoke（2026-08-10）

本轮又用当前 StackOverflow/KIR=.50/seed=42 train/dev/test 做了一个隔离 DCLOOS smoke。审计发现：
protocol registry 固定了 10 个 Known intent，而上游 DCLOOS 会在这 10 个标签中按 `seed=42` 再抽 5 个，
所以这不是同 Known list 的公平单元。完整 1 epoch smoke 能产出 metrics/predictions，但 OOS F1=0、F1-All=8.31%，
属于低预算收敛诊断，不能与 S2C、MOGB 或 ADB 排名。

这次运行同时把上游标量 `.item()` 和 schedule 越界等兼容问题记录在 overlay 中。完整合同说明见
`docs/archive/analysis/DCLOOS_CONTRACT_STATUS_V1.md`；当前结论是：DCLOOS 链路已可审计，但公平结果尚未完成，
不能用 reduced/smoke 数字宣称 SOTA。
