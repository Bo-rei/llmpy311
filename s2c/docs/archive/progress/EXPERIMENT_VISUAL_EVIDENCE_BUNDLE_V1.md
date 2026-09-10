# 当前实验与可视化证据总览 V1

> **口径提示（2026-08-09）**：本文件保留多批次图表索引，其中部分早期段落使用三 seed 或旧汇总。当前五 seed 同协议的权威数字和排名以 [`EXPERIMENT_ANALYSIS_MASTER_V1.md`](EXPERIMENT_ANALYSIS_MASTER_V1.md) 为准；本文件中的早期数字不得覆盖该主入口。

更新时间：2026-08-08  
活动协议：`protocol_v2_textoir_v1`  
证据类型：已完成实验与 analysis-only 后验分析；不新增训练、不覆盖历史 artifact。

## 先看结论

当前最强的自有 Gate 工作点是 **Trainable MiniLM + K=1**。它的优势主要来自表示适配后的 score separation 和覆盖—拒识平衡，而不是固定多中心本身。

固定 K>1 的问题在 StackOverflow 最清楚：增加中心确实恢复了一部分 Known，但接受区域并集同时吸收了大量 OOS。聚类 ARI 很高也不能保证 OOS 边界有效。

MOGB 的 Frozen MiniLM 组件更保守，能够降低 false acceptance，但牺牲大量 Known Recall；它不是当前 Trainable 的同一工作点。官方 BERT MOGB 和 DCLOOS 由于表示、训练监督或数据合同不同，仍然只能作为隔离的兼容性参照，不能混进同一 SOTA 排名。

## 1. 当前证据层次

| 层次 | 回答的问题 | 入口 |
|---|---|---|
| 性能层 | 哪个方法在同协议下更稳定 | [`VISUAL_EVIDENCE_CHAIN_V1.md`](VISUAL_EVIDENCE_CHAIN_V1.md) |
| 逐样本决策层 | K=2 到底新增了哪些错误 | [`RAW_GATE_ERROR_VISUALIZATION_V1.md`](RAW_GATE_ERROR_VISUALIZATION_V1.md) |
| 表示层 | Trainable 为什么改善 K=1 | [`REPRESENTATION_GEOMETRY_VISUALS_V1.md`](REPRESENTATION_GEOMETRY_VISUALS_V1.md) |
| 表示—边界交互 | 几何改善为何不能保证 K=2 | [`REPRESENTATION_BOUNDARY_PACK_V1.md`](REPRESENTATION_BOUNDARY_PACK_V1.md) |
| 配对效应与 intent 异质性 | Trainable 的优势是否在同一 seed 下稳定，以及哪些 intent 可能受益于多中心 | [`PAIRED_EFFECT_INTENT_HETEROGENEITY_V1.md`](PAIRED_EFFECT_INTENT_HETEROGENEITY_V1.md) |
| 基线层 | MOGB/ADB/DA-ADB/DCLOOS 的合同差异 | [`MINILM_TRAINABLE_VS_FULLTEX_AND_BASELINES_V1.md`](MINILM_TRAINABLE_VS_FULLTEX_AND_BASELINES_V1.md) |
| 外部基线执行状态 | protocol_v2 数据适配、ADB/DA-ADB preflight 与阻断 | [`BASELINE_EXECUTION_STATUS_V1.md`](BASELINE_EXECUTION_STATUS_V1.md) |
| 当前状态 | 所有阶段、重复规则和阻断 | [`CURRENT_STATUS.md`](../CURRENT_STATUS.md) |

## 2. 同协议性能结论

Trainable K=1 在 3 个数据集×3 个 KIR 的 9 个同协议单元中，OOS F1 排名第一 8 次、第二 1 次。这是当前 fair matrix 内的稳定性证据，不是 SOTA 声明。

跨 KIR 五 seed 均值：

| 数据集 | OOS F1 | F1-All | Known Recall | False Acceptance |
|---|---:|---:|---:|---:|
| CLINC150 | 89.48% | 81.47% | 74.40% | 3.65% |
| Banking77 | 81.29% | 81.44% | 82.16% | 15.23% |
| StackOverflow | 86.48% | 87.16% | 84.10% | 7.33% |

图表：

- `figures/visual_evidence_chain_v1/performance_heatmap.png`
- `figures/visual_evidence_chain_v1/pareto_tradeoff.png`
- `figures/visual_evidence_chain_v1/kir_curves.png`
- `figures/visual_evidence_chain_v1/method_rank_heatmap.png`
- `figures/visual_evidence_chain_v1/relative_trainable_heatmap.png`

## 3. StackOverflow 的逐样本机制

在 StackOverflow/KIR=.50、seed 13/42/87 上，四种方法的均值为：

| 方法 | OOS F1 | F1-All | Known Recall | False Acceptance |
|---|---:|---:|---:|---:|
| Trainable K=1 | 86.71% | 85.65% | 83.92% | 11.14% |
| Frozen K=1 | 77.29% | 78.60% | 83.71% | 26.54% |
| Fixed K=2 | 67.65% | 76.81% | 93.62% | 45.26% |
| MOGB fair component | 73.19% | 45.15% | 28.34% | 0.92% |

从 Trainable K=1 切到 Fixed K=2：

- 恢复 Known：平均约 297.3 个；
- 丢失 Known：平均约 6.3 个；
- 新增误接收 OOS：平均约 1,025.3 个。

因此 K=2 的主要失败不是“没有覆盖 Known”，而是新增 OOS 接受远大于恢复的 Known。对应图表：

- `figures/raw_gate_error_visualization_v1/boundary_expansion_waterfall.png`
- `figures/raw_gate_error_visualization_v1/oos_transitions.png`
- `figures/raw_gate_error_visualization_v1/intent_error_heatmap.png`
- `figures/representation_geometry_visuals_v1/stackoverflow_error_quadrants.png`

## 4. 表示几何为什么只能改善 K=1

CE/SupCon 相比 Frozen 提高了 purity、relative separation 和 same-intent alignment，同时降低 effective rank。这些变化能改善单中心 score separation，但不能直接约束多个局部球的接受并集。

StackOverflow 的 K=2 near-OOS F1 变化：

| 表示 | K=2 − K=1 Near-OOS F1 |
|---|---:|
| Frozen | −9.8pp |
| CE | −10.7pp |
| SupCon | −32.3pp |

这说明：

```text
更好的 Known 几何 ≠ 更安全的多中心开放边界
```

Trainable/Frozen 的 score-gap 图显示，Trainable 在三个数据集都扩大了 OOS−Known median score gap，并降低了 false acceptance；这支持 Trainable K=1 的主要收益来自分数排序，而不是单纯改变拒绝比例。

对应图表：

- `figures/representation_geometry_visuals_v1/geometry_tradeoff.png`
- `figures/representation_geometry_visuals_v1/near_oos_delta.png`
- `figures/representation_geometry_visuals_v1/score_gap_false_accept.png`
- `figures/representation_boundary_pack_v1/representation_geometry_summary.png`

## 5. MOGB 对比应该怎样解释

当前已有两种不同层级的 MOGB 证据，不能混写：

1. **MOGB MiniLM fair/component**：与当前 Frozen MiniLM 共用表示的组件比较。它更保守，false acceptance 较低，但 Known Recall/F1-All 显著下降。
2. **MOGB official BERT compatibility reproduction**：使用官方代码逻辑和 BERT 路径，但当前可用环境下没有复现论文数字，且数据/指标合同与当前 Gate-only 协议不同。

因此目前可以说：

- 在同一 Frozen MiniLM 组件口径下，Trainable K=1 的覆盖—拒识工作点更平衡；
- 不能说已经公平击败官方 MOGB；
- 不能把官方 BERT 兼容复现的数字与当前 MiniLM Gate 数字直接排名。

ADB、DA-ADB 和 DCLOOS 同样保存在外部/兼容层，尤其 DCLOOS 使用额外 pseudo-OOS 或开放域监督，不能与 Known-only Gate 直接视为同一监督条件。

当前同协议外部基线执行状态另见 [`BASELINE_EXECUTION_STATUS_V1.md`](BASELINE_EXECUTION_STATUS_V1.md)：
StackOverflow 三个 seed 的 protocol exports 已完成 SHA256 对齐，ADB 三个单元已经通过兼容运行并由
`y_true.npy/y_pred.npy` 重算；DA-ADB 仍因 NaN/全类预测无效。因此 ADB 仍只进入外部合同参照，DA-ADB
和历史兼容数字不进入当前 fair 主排名。

## 6. 当前实验回答了什么

已经回答：

1. Trainable MiniLM 是否改善当前 Gate：是，主要改善 K=1 score separation。
2. 固定 K=2 是否普遍更好：否，StackOverflow 明显失败，Banking77 仅条件性收益，CLINC150 接近中性。
3. 稳定聚类是否足够：否，StackOverflow ARI 约 0.926 仍可能大规模误接收 OOS。
4. MOGB fair 是否只是单指标更高：不是，它更像保守拒识工作点，Known 覆盖代价明显。
5. 表示训练能否自动修复多中心：不能，几何改善没有消除 acceptance-union 风险。

## 6.1 新增配对统计与 intent-level 证据

新增 [`PAIRED_EFFECT_INTENT_HETEROGENEITY_V1.md`](PAIRED_EFFECT_INTENT_HETEROGENEITY_V1.md) 和四张图：

- 同一 `dataset × KIR × seed` 下，Trainable K=1 相对 Frozen/MOGB 组件的 OOS F1 配对差值及 95% bootstrap CI；
- KIR=0.50 下 OOS F1、F1-All、Known Recall、false acceptance 的四指标差值热图；
- 三个数据集在不同 KIR/距离下，逐 intent×seed `safe_gain_oracle` 的比例；
- Euclidean 下 oracle-best-K 的 intent 分布。

该统计进一步说明：Trainable K=1 的优势在相同 seed 配对后仍然稳定；Banking77 的多中心潜在收益比例高于 StackOverflow，但 StackOverflow 在高 KIR 下 safe-gain 比例下降。后两项使用测试 oracle，仅用于解释异质性，不能作为正式 adaptive-K 选择器。

## 6.2 新增 MOGB 同工作点诊断

新增 [`MOGB_OPERATING_POINT_VISUALS_V1.md`](MOGB_OPERATING_POINT_VISUALS_V1.md) 和四张图。该阶段读取已有 315 个 fair per-seed 预测文件，在测试 Known Recall 约为 0.75/0.85/0.90/0.95 的事后工作点上比较 Trainable K=1、Frozen 单/双中心、随机划分和 MOGB 组件。

KIR=.50、Known Recall≈.85 的直接证据是：Trainable K=1 在 CLINC150/Banking77/StackOverflow 的 OOS F1 分别为 0.9164/0.8241/0.8705，false acceptance 分别为 0.0715/0.1964/0.1132；StackOverflow 固定 K=2 为 0.6692/0.4216，MOGB-MiniLM 为 0.6860/0.3913。该阶段只用于解释 coverage–open-space tradeoff，阈值按 test-known 分数事后对齐，不能进入正式选择或 SOTA 主表。

## 6.3 新增逐样本错误归因

新增 [`STACKOVERFLOW_ERROR_ATTRIBUTION_V2.md`](STACKOVERFLOW_ERROR_ATTRIBUTION_V2.md) 和四张图，按同一 `sample_id` 对齐 StackOverflow/KIR=.50 的 7 方法×5 seed 预测。它进一步给出固定 K=2 的 OOS 错误转移、MOGB 的 Known 过拒绝，以及具体吸收 OOS 的 intent。当前 score<=1 下，Trainable K=1 的 OOS F1/FA/FR 为 0.8767/0.0934/0.1611，Frozen K=2 为 0.6353/0.4717/0.1311，MOGB-MiniLM 为 0.7292/0.0079/0.7291。该阶段仍是冻结预测归因，不是调参或跨基线公平排名。

尚未回答：

1. 在完全相同的数据、Known 列表、指标和监督合同下，Trainable、MOGB、ADB、DA-ADB、DCLOOS 的最终统一排名。
2. 当前 Gate 候选进入完整 Router/Expert Cascade 后的端到端收益。
3. 是否存在一个新的边界规则，能在不扩大 StackOverflow false acceptance 的情况下利用 Banking77 的局部结构。

## 7. 目前不要作出的结论

- 不要写“已经达到 SOTA”。
- 不要写“多中心普遍有效”。
- 不要把 MOGB fair component 写成官方 MOGB 复现。
- 不要把 Gate-only 指标写成完整 Cascade 指标。
- 不要用 test-oracle 的最佳 K 或阈值作为正式方法选择依据。

## 6.4 跨数据集错误归因

新增 [`CROSS_DATASET_ERROR_ATTRIBUTION_V1.md`](CROSS_DATASET_ERROR_ATTRIBUTION_V1.md) 与四张图：

1. `trainable_oos_win_heatmap.png`：Trainable K=1 相对各比较方法的 OOS 正确率差异；
2. `false_acceptance_delta_vs_trainable.png`：比较方法 false acceptance 减 Trainable 的差值；
3. `false_rejection_delta_vs_trainable.png`：比较方法 false rejection 减 Trainable 的差值；
4. `cross_dataset_oos_f1_false_acceptance.png`：三数据集、三 KIR 下的 OOS F1–false acceptance 工作点。

该阶段对齐 315 个已有 fair run 和 1,890,000 条逐样本预测。Trainable K=1 相对 Frozen/Random 的 false acceptance 在全部九个 dataset×KIR 单元均更低；MOGB 组件虽然通常更少误接收 OOS，却以更高 Known false rejection 换取保守性。所有图和汇总都属于 frozen-prediction 机制分析，不是测试集调参结果，也不是 ADB/DA-ADB/DCLOOS 的统一排名。

## 6.5 跨数据集逐 intent 风险

新增 [`CROSS_DATASET_INTENT_RISK_VISUALS_V1.md`](CROSS_DATASET_INTENT_RISK_VISUALS_V1.md) 与四张图：

1. `intent_risk_scatter_kir050.png`：KIR=.50 下逐 intent 的 Known false rejection–OOS false acceptance 关系；
2. `oos_acceptance_concentration_kir050.png`：前 1/5 个 intent 对 OOS 误接收的集中度；
3. `intent_error_delta_scatter_kir050.png`：相对 Trainable 的逐 intent 错误变化；
4. `fixed_k2_oos_acceptor_rank_curve.png`：固定 K=2 的 OOS 吸收 intent 累计贡献。

这批证据把 StackOverflow 固定 K=2 的退化定位到少数高风险 intent：KIR=.50 时前五个 intent 贡献约 64.9% 的 false acceptance。MOGB 组件在相同图中表现为更高的 Known 误拒，而不是无代价地消除 OOS 风险。

## 6.6 五 seed 统计稳定性

新增 [`STATISTICAL_STABILITY_V1.md`](STATISTICAL_STABILITY_V1.md) 与三张图：

1. `paired_oos_f1_forest.png`：Trainable 相对各 fair 方法的配对五 seed OOS F1 差值及 95% bootstrap CI；
2. `method_rank_heatmap_oos_f1.png`：每个 dataset/KIR 的 OOS F1 平均排名；
3. `five_seed_oos_f1_kir_curves.png`：均值、标准差和 KIR 曲线。

这批统计显示 Trainable K=1 相对 Frozen K=1、Frozen K=2、Random K=2 在 9/9 个 dataset×KIR 组合中五 seed 全部胜出；按 OOS F1 平均排名，它在 8/9 个 dataset×KIR 单元排名第一。Banking77/KIR=.25 是唯一由 MOGB partition + ours 略占优势的工作点。

## 6.7 Gate→Cascade 配对桥接

新增 [`GATE_CASCADE_PAIRED_BRIDGE_V2.md`](GATE_CASCADE_PAIRED_BRIDGE_V2.md) 与三张图：

1. `gate_cascade_oos_f1_bridge_forest.png`：Gate-only Trainable K=1 与三 seed Cascade 条目的桥接差值；
2. `cascade_only_oos_accuracy_tradeoff.png`：Cascade 内部 OOS F1 与整体准确率的工作点；
3. `cascade_router_expert_error_decomposition.png`：Router 与 Expert 错误分解。

该阶段消费已有 45 行 Cascade/Gate 汇总，不重训、不改阈值。核心限制是 Trainable 行为 Gate-only、
其余行行为 Cascade，所以桥接森林图不是统一端到端排名。相对 Frozen K=1 Cascade，Trainable Gate
的 OOS F1 差值为 CLINC150 +2.41 pp、Banking77 −0.05 pp、StackOverflow +7.69 pp；同层 Cascade
中 CE-Recon selected-K 相对 Frozen K=1 的差值分别为 +1.97、+4.66、+8.60 pp。外部 ADB、DA-ADB、
DCLOOS 仍未进入同监督主表，不能由此宣称 SOTA。

## 6.8 同一 Trainable 表示下的检测器对照

新增 [`TRAINABLE_DETECTOR_MECHANISM_V1.md`](TRAINABLE_DETECTOR_MECHANISM_V1.md) 与三张图：

1. `trainable_detector_pareto.png`：同一 Trainable MiniLM 表示下 Gate、MSP、Energy、kNN、LOF 的覆盖—拒识工作点；
2. `trainable_detector_paired_effects.png`：Gate 相对原生检测器的 OOS F1、F1-All、Known Recall 配对差值；
3. `trainable_detector_error_balance.png`：false acceptance 与 false rejection 的构成。

12 个同表示 OOS F1 比较均由 Trainable Gate 在 3/3 seed 上胜出，但主要通过更低 false
acceptance 和更低 Known Recall 换取；这说明 Trainable 的优势不只来自 encoder，也来自 Gate
边界/校准工作点，同时不支持“所有指标全面优于原生检测器”的表述。

## 6.9 全指标 Fair Effect Landscape

新增 [`FAIR_EFFECT_LANDSCAPE_V1.md`](FAIR_EFFECT_LANDSCAPE_V1.md) 与三张图：

1. `trainable_oos_f1_effect_heatmap.png`：Trainable K=1 相对 Frozen、Random 和 MOGB 组件的
   OOS F1 配对优势，星号表示 95% paired bootstrap CI 不跨零；
2. `trainable_oos_fa_effect_landscape.png`：OOS F1 优势与 false-acceptance 风险降低量的
   联合工作点；
3. `trainable_multimetric_effect_summary.png`：OOS F1、F1-All、F1-K、Known Recall、错误率、
   AUROC 和 AUPR-OOS 的平均效应热图。

该阶段从已有统计源表中筛选 432 行（3 数据集 × 3 KIR × 6 比较 × 8 指标），没有训练或测试
集调参。它显示 Trainable K=1 在当前 fair matrix 中是较平衡的自有工作点；MOGB 组件更偏向
保守拒识并付出较高 Known false rejection。该结果仍不能替代同监督 ADB/DA-ADB/DCLOOS
主表，也不能写成 SOTA 结论。

## 6.10 KIR 敏感性分解

新增 [`KIR_SENSITIVITY_DECOMPOSITION_V1.md`](KIR_SENSITIVITY_DECOMPOSITION_V1.md) 和三张图：

1. `oos_f1_kir_sensitivity.png`：三数据集的 OOS F1 随 KIR 变化曲线；
2. `kir_endpoint_delta_heatmap.png`：KIR=.25→.75 的 OOS F1、F1-All、F1-K、Known Recall、
   false acceptance、false rejection 端点变化；
3. `coverage_oos_trajectory_by_kir.png`：每个方法在 Known Recall–OOS F1 平面上的 KIR 轨迹。

该分析量化了退化机制：StackOverflow Frozen K=2 的 OOS F1 端点下降约 43.0 pp，同时
false acceptance 增加约 42.2 pp；MOGB 组件误接收变化很小，却以明显的 Known Recall
下降换取保守拒识；Trainable K=1 的 Known Recall 随 KIR 基本稳定。所有图均消费已有五
seed summary，不产生新的模型证据。

## 8. 下一步

当前最合理的下一步仍是整理同协议 baseline 工作点和错误分解，而不是继续增加 K、损失项或 adaptive-K 规则：

1. 先将现有 7+6+4 张机制图和 CSV 统一到一份可审计 evidence bundle；
2. 在相同监督条件下整理 MOGB/ADB/DA-ADB/DCLOOS 的可比字段和不可比字段；当前 ADB/DA-ADB 先修复独立 runtime 后从 seed=42 单格开始；
3. 只有发现关键比较仍缺少真实运行证据时，才登记一个最小的补充实验；
4. 继续保持 E2/E3/R1、Trainable checkpoint 和历史基线不可覆盖。
## 6.11 OOS 误接收—误拒绝预算

新增 [`OOS_ERROR_BUDGET_V1.md`](OOS_ERROR_BUDGET_V1.md) 与三张图：

1. `oos_precision_recall_workpoints.png`：KIR=.50 的 OOS precision–recall 工作点；
2. `oos_error_budget_kir050.png`：OOS→Known 误接收和 Known→OOS 误拒绝的并列预算；
3. `trainable_oos_error_budget_effects.png`：Trainable 相对各冻结/组件方法的配对效应及 CI。

该阶段从已审计的 315 行 per-seed 指标重构 OOS precision/recall，并逐行核对 OOS F1，
没有训练、调参或测试选择。KIR=.50 的 StackOverflow 结果最能区分机制：Frozen K=2 的
false acceptance 为 47.17%，而 MOGB MiniLM 组件为 0.79% 但 Known false rejection 为
72.91%；Trainable K=1 为 9.34%/16.11%，因此是当前 fair matrix 内更平衡的工作点。
该图包用于错误预算解释，不构成完整 MOGB、ADB、DA-ADB、DCLOOS 或 SOTA 排名。
