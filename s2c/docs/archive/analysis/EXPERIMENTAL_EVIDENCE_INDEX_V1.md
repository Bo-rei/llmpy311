# s2c 实验结果与机制证据索引 V1

更新时间：2026-08-09
活动协议：`protocol_v2_textoir_v1`

## 1. 当前研究主线

当前阶段的重点是实验、对比和机制分析，不把尚未公平验证的数字写成 SOTA。所有结果必须区分：

- 表示：Frozen MiniLM、Trainable MiniLM、BERT；
- 训练监督：Known-only、伪 OOS、外部 OOS；
- 边界：单中心、固定多中心、MOGB 粒球组件；
- 系统层级：Gate-only 或完整 Gate–Router–Expert Cascade。

新增统一对比图谱入口：`docs/archive/analysis/COMPARISON_ATLAS_V1.md`。它只允许当前
`protocol_v2_textoir_v1` fair Gate 层直接排名，并将历史 Cascade、MOGB 官方现代兼容和外部兼容单格
单独列出。

## 2. 已完成实验地图

| 实验 | 主要问题 | 范围 | 当前状态 | 证据 |
|---|---|---|---|---|
| E2 固定 K 网格 | K、KIR、距离是否存在统一最优 | 3 数据集、11 KIR、5 seed、K=1..5、2 距离 | 1650/1650 | `docs/archive/analysis/EXPERIMENT_EVIDENCE_PACK_V2.md` |
| E3 KMeans/随机与稳定性 | 多中心收益是否来自真实结构 | 720 控制单元、诊断统计 | 完成 | `docs/archive/analysis/EXPERIMENT_EVIDENCE_PACK_V2.md` |
| RACAL Trainable K=1/K=2 | 训练表示能否修复固定多中心 | 三数据集、KIR=.50、多个 seed | 完成，K=1 有收益、K=2 不稳定 | `docs/archive/experiments/racal_v1/` |
| Trainable KIR sweep | K=1 表示收益是否跨开放程度稳定 | 45 个五 seed 单元 | 完成 | `docs/archive/analysis/MINILM_TRAINABLE_5SEED_FAIR_COMPARISON_V1.md` |
| 表示—边界诊断 | 收益来自 score 分离还是边界变化 | 90 run、443400 score | 完成 | `docs/archive/analysis/MINILM_BOUNDARY_DIAGNOSTICS_V1.md` |
| 训练动态诊断 | Known-only 选模是否与 OOS 错位 | 45 run、179 epoch | 完成 | `docs/archive/analysis/MINILM_TRAINING_DYNAMICS_V1.md` |
| 阈值/半径稳定性 | score 标度与半径估计是否解释 Trainable/fulltex 差距 | 810 阈值行、90 半径行 | 完成（仅诊断） | `docs/archive/analysis/THRESHOLD_RADIUS_STABILITY_V1.md` |
| 同协议方法权衡 | Trainable 与 Frozen/MOGB 组件的覆盖—拒识工作点如何不同 | 315 行、486 paired effects、4 图 | 完成（仅诊断） | `docs/archive/analysis/CROSS_PROTOCOL_TRADEOFF_V1.md` |
| Gate→Cascade 桥接 | Router/Expert 是否解释 Gate-only 与系统级结果差异 | 45 行、15 summary、3 图 | 完成（仅诊断） | `docs/archive/analysis/GATE_CASCADE_BRIDGE_V1.md` |
| Trainable→Cascade 合同审计 | 当前 Trainable Gate 能否直接复用旧 v19 Router/Expert | 1 个 StackOverflow/KIR=.50/seed42 合同对照、3 张轻量产物 | 完成；禁止直接混合 | `docs/archive/analysis/CASCADE_TRAINABLE_CONTRACT_GAP_V1.md` |
| 当前协议 Cascade bridge | 在同一 protocol_v2 Expert 下，Trainable K=1 Gate 的收益是否传递到 Cascade | StackOverflow/KIR=.50、3 seeds、6/6 评价行、2 张图、配对效应 | 完成；当前协议内部可比较 | `docs/archive/analysis/CASCADE_BRIDGE_V1.md` |
| 合同分层比较图谱 V2 | 现在究竟比较哪个“我的方法”，Cascade 如何与 fair Gate/MOGB/外部参考分层 | 10 条 StackOverflow/KIR=.50 合同行、3 张图、MOGB 论文差距上下文 | 完成；分析入口 | `docs/analysis/COMPARISON_ATLAS_V2.md` |
| Cascade Gate/Expert 误差预算 | Trainable 的 Cascade 增益来自 Gate 还是 Expert | 6 条 seed/variant 行、24 条样本转移行、2 张图 | 完成；机制分析 | `docs/archive/analysis/CASCADE_ERROR_BUDGET_V1.md` |
| Trainable/MOGB 组件配对 | 自有方法与 MOGB 组件的覆盖—拒识权衡 | 45 Trainable + 135 MOGB 行 | 完成 | `docs/archive/analysis/TRAINABLE_VS_MOGB_COMPONENT_V1.md` |
| MOGB 官方 BERT | 能否复现作者论文数字 | StackOverflow/Banking 单格 | `not_reproduced_strict` | `docs/对比实验/MOGB_DCLOOS_对比结果报告.md` |
| MOGB Known-only 校准与损失归因 | 论文差距来自默认工作点还是子中心训练信号 | 45 粒球单元、225 校准工作点、270 损失契约单元 | 45/45 等价；完成 | `docs/archive/analysis/MOGB_KNOWN_CALIBRATION_ATTRIBUTION_V1.md` |
| corrected-loss BERT-MOGB 半径覆盖 | 修正子中心损失后，剩余论文差距能否由mean-radius工作点解释 | StackOverflow/KIR=.50/seed0、1 checkpoint、5工作点、33球重建 | 完成；非严格ball replay | `docs/archive/analysis/MOGB_CORRECTED_RADIUS_COVERAGE_V1.md` |
| ADB/DA-ADB | 近期边界基线兼容性 | BERT 单格 | compatibility artifact | `docs/archive/analysis/KIR50_METHOD_COMPARISON_V1.md` |
| DCLOOS | 端到端伪 OOS/外部 OOS 方法 | reduced-budget 单格 | 条件不同，非公平主表 | `docs/archive/external_baselines/dcloos/DCLOOS_REPRODUCTION_REPORT.md` |
| Trainable detector control | 固定 Trainable MiniLM 表示后只替换检测器 | KIR=.50、3 数据集、3 seed、Gate/MSP/Energy/kNN/LOF | 完成，机制诊断 | `docs/archive/analysis/TRAINABLE_DETECTOR_MECHANISM_V1.md` |
| Fair effect landscape | 全指标配对效果是否支持“平衡工作点”解释 | 432 行、3 数据集、3 KIR、6 比较、8 指标、3 图 | 完成，analysis-only | `docs/archive/analysis/FAIR_EFFECT_LANDSCAPE_V1.md` |
| KIR 敏感性分解 | 随开放比例变化的性能退化来自误接收还是过拒绝 | 3 数据集、7 方法、6 指标、3 图 | 完成，analysis-only | `docs/archive/analysis/KIR_SENSITIVITY_DECOMPOSITION_V1.md` |
| 综合实验分析主入口 | 统一五 seed fair matrix 的正确排名、Pareto、热力图和 KIR 曲线 | 63 汇总行、7 方法、3 数据集、3 KIR、5 图 | 完成，analysis-only | `docs/archive/analysis/EXPERIMENT_ANALYSIS_MASTER_V1.md` |
| 可视化分析索引 | 将性能、表示、决策和外部合同图组织为可读证据链 | 6 张主图、多个附录图、中文索引 | 完成，analysis-only | `docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md` |

## 3. 当前自己的方法是什么

当前最可靠的可训练版本是：

```text
Known train
  → MiniLM 最后两层 + residual projection 训练
  → Known calibration 选择 checkpoint
  → K=1 diagonal Mahalanobis detector
  → mean + std radius，threshold=1
  → Gate-only OOS 评价
```

它不是完整的自适应多中心方法。固定 K=2 仍是独立的失败诊断；候选 split 的训练参与式实验已实现，
但在 StackOverflow 上没有通过 Known-only 安全门。

## 4. 五 seed 的核心数值

Trainable K=1 相对同一 E2 Frozen K=1 的 OOS F1 增量：

| 数据集 | KIR=.25 | KIR=.50 | KIR=.75 |
|---|---:|---:|---:|
| CLINC150 | +0.45pp | +1.12pp | +1.38pp |
| Banking77 | +2.47pp | +4.72pp | +6.94pp |
| StackOverflow | +5.06pp | +9.55pp | +10.50pp |

Known Recall 最大下降分别为 CLINC150 1.37pp、Banking77 2.90pp；StackOverflow 三个 KIR 均小幅上升。

## 5. 为什么 Trainable K=1 在当前协议下优于 MOGB 组件

KIR=.50 五 seed 均值如下：

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False Accept |
|---|---|---:|---:|---:|---:|
| CLINC150 | Trainable K=1 | 90.44 | 81.82 | 74.44 | 3.69 |
| CLINC150 | MOGB partition + s2c boundary | 85.56 | 64.60 | 53.34 | 2.50 |
| Banking77 | Trainable K=1 | 83.56 | 81.67 | 82.21 | 15.74 |
| Banking77 | MOGB partition + s2c boundary | 79.40 | 64.81 | 52.18 | 3.50 |
| StackOverflow | Trainable K=1 | 87.67 | 86.55 | 83.89 | 9.34 |
| StackOverflow | MOGB partition + s2c boundary | 79.25 | 63.34 | 50.39 | 1.86 |

因此当前能够支持的表述是：

> Trainable K=1 在相同划分和 seed 下提供了更平衡的 Known 覆盖—OOS 拒识工作点；MOGB 组件更保守，
> 但通过牺牲大量 Known Recall 换取更低 false acceptance。

这不是“无条件超过完整 MOGB”，因为 MOGB 组件使用 Frozen MiniLM，且不是作者 BERT 完整方法。

## 6. 为什么仍然低于或接近 fulltex 历史结果

`fulltex.tex` 历史表与当前 Trainable 不同：

1. 历史结果是完整 Gate–Router–Expert Cascade，当前结果是 Gate-only；
2. 历史使用固定 `K_y=2` 和数据集相关 λ；
3. 历史正文说明 unknown/OOS validation 参与 λ 学习，当前严格 Known-only；
4. 历史数据快照、Known 列表和 split 与当前 `protocol_v2_textoir_v1` 不完全相同；
5. 当前 Trainable 的 loss 优化 Known 中心分类/紧致/间隔，没有直接优化 OOS 风险。

所以当前结果低于历史数字，不应直接解释为“MiniLM 可训练失败”。

## 7. 视觉证据入口

- Trainable/Frozen KIR 曲线：`figures/archive/analysis/minilm_trainable_5seed_fair_v1/trainable_vs_frozen_5seed_kir.png`
- Score 分布：`figures/archive/analysis/minilm_boundary_diagnostics_v1/score_distributions_kir050.png`
- OOS F1—false acceptance：`figures/archive/analysis/minilm_boundary_diagnostics_v1/oos_f1_false_acceptance_kir.png`
- Trainable/MOGB Known Recall—OOS F1：`figures/archive/analysis/trainable_vs_mogb_component_v1/kir050_known_recall_oos_f1_components.png`
- Trainable/MOGB OOS F1 配对差值：`figures/archive/analysis/trainable_vs_mogb_component_v1/paired_oos_f1_delta_heatmap.png`
- Calibration→test Known Recall：`figures/archive/analysis/minilm_training_dynamics_v1/calibration_vs_test_known_recall.png`
- MOGB Known-only 工作点：`figures/archive/analysis/mogb_known_calibration_attribution_v1/known_recall_oos_f1_workpoints.png`
- MOGB 子中心损失信号：`figures/archive/analysis/mogb_known_calibration_attribution_v1/subcentroid_loss_signal.png`
- Trainable K=1 与校准 MOGB-Fair：`figures/archive/analysis/mogb_known_calibration_attribution_v1/trainable_k1_vs_mogb_cal80.png`

## 8. 下一步实验顺序

1. 把已完成的 threshold/半径诊断纳入监督条件与系统层级分层，不使用 test oracle 选择正式参数；
2. 在当前 `protocol_v2_textoir_v1` registry/views 上重新训练 Router/Expert，先做 StackOverflow/KIR=.50、
   seeds=13/42/87 的 Frozen K=1 vs Trainable K=1 Cascade 配对实验；
3. 完成下游合同桥接后，在统一监督条件、数据划分和随机种子下再把 ADB、DA-ADB、DCLOOS 纳入正式比较；
4. 只有在上述合同完全一致后，才扩展正式的 MOGB/DCLOOS 主表。

当前不应做：盲目增加 K、继续添加损失项、把官方 MOGB 负复现写成 MOGB 失败，或把 DCLOOS 的外部 OOS 监督结果与 Known-only 方法直接排名。

## 当前协议原生 OOD controls 与工作点诊断

`native_baselines_v1` 在同一 registry/view、冻结 MiniLM 和 Known-only calibration 下完成 MSP/Energy/kNN/LOF 的 180/180 矩阵，结果与训练表示分开记录：`docs/analysis/NATIVE_BASELINES_V1.md`、`results/analysis/native_baselines_v1/`。这些是原生控制，不是官方 ADB/DA-ADB/MOGB/DCLOOS 的替代品。

`operating_point_diagnostic_v1` 只做 retrospective matched-Known-Recall 对齐，用于解释为什么 Trainable 默认 OOS F1 较高但 Known Recall 较低，以及在共同工作点是否仍有 score-ranking 优势。它不选择正式阈值，不进入论文主结果：`docs/archive/analysis/OPERATING_POINT_DIAGNOSTIC_V1.md`、`results/analysis/archive/analysis/operating_point_diagnostic_v1/`。

## Trainable 与 MOGB frozen-component 归因

`trainable_vs_mogb_ablation_v1` 将 45 个 Trainable K=1 五 seed 单元与 180 个已完成 MOGB frozen-MiniLM 距离/半径组件行按 dataset×KIR×seed 配对，生成 324 个 bootstrap metric effects、180 个组件归因行和 3 张图。结果显示：MOGB 的 mean→mean+std 半径变化比 purity-get 阈值更能改变 OOS F1，但 Trainable 仍在相同工作点保留更高 Known Recall/F1-All。

证据入口：`docs/archive/analysis/TRAINABLE_VS_MOGB_ABLATION_V1.md`、`results/analysis/archive/analysis/trainable_vs_mogb_ablation_v1/`、`figures/archive/analysis/trainable_vs_mogb_ablation_v1/`。

## Trainable 表示上的原生 detector 归因

`native_baselines_trainable_v1` 在已完成的 Trainable MiniLM checkpoint 上运行 MSP、Energy、kNN、LOF，
范围为 3 数据集×KIR=.50×3 seed，共 36/36。它同时与 Frozen native 和 Trainable Gate K=1 配对，
回答“收益来自表示、检测器还是 Gate 几何”。

报告：`docs/archive/analysis/NATIVE_BASELINES_TRAINABLE_V1.md`；结果：`results/analysis/archive/analysis/native_baselines_trainable_v1/`；
可视化：`figures/archive/analysis/native_baselines_trainable_v1/`。

## MOGB 官方代码合同逐行审计

新增 `docs/archive/analysis/MOGB_CODE_CONTRACT_LINE_AUDIT_V1.md` 和
`results/analysis/archive/analysis/mogb_code_contract_line_audit_v1/evidence.csv`，将 pinned upstream 的 loss、设备、
递归随机拆分、selected-ball 过滤、平均半径和最近球推理逐行对应到已观测的 MOGB loss 压缩、Known
过拒和最终球状态不可重放问题。该阶段只做源代码归因，不产生新性能结果。

## 实验机制分析包 V3（2026-08-06）

基于已有 `minilm_trainable_5seed_fair_v1/all_methods_per_seed.csv` 的 315 行逐 seed 结果，新增 324 个
paired bootstrap effects、63 个方法汇总、63 个 Pareto 标记和四张可视化。该包只做轻量结果再分析，不读取
checkpoint、embedding 或原始文本，也不产生新的训练证据。入口：
`docs/archive/analysis/EXPERIMENTAL_MECHANISM_PACK_V3.md`；数据：`results/analysis/archive/analysis/experimental_mechanism_pack_v3/`；
图：`figures/archive/analysis/experimental_mechanism_pack_v3/`。
| OOS 误接收—误拒绝预算 | OOS F1 的 precision/recall 组成及 Trainable/MOGB 的风险取舍 | 315 per-seed 行、63 汇总、324 配对效应、3 图 | 完成，analysis-only | `docs/archive/analysis/OOS_ERROR_BUDGET_V1.md` |
| 方法对比地图 | 明确“我的方法”、固定多中心、MOGB 组件、自适应 pilot 和外部 baseline 的边界 | 7 行 fair matrix 定义、外部合同分层、StackOverflow KIR=.50 对照 | 完成，analysis-only | `docs/archive/analysis/METHOD_COMPARISON_MAP_V1.md` |
| 机制证据 V2 | 可视化 OOS F1、FA、Known 覆盖预算、表示几何与 K=2 风险 | 63 fair rows、54 acceptance-budget rows、4 图 | 完成，analysis-only | `docs/archive/analysis/MECHANISM_EVIDENCE_V2.md` |
| 历史 SOTA 合同分层 | 从 `fulltex.tex` 直接提取历史 Cascade 主表，明确 Ours 与 ADB/DA-ADB 等基线的历史差值，并与当前 Gate-only 结果分层 | 72 个历史单元、3 张历史对比图、MOGB 差距类别审计 | 完成，analysis-only | `docs/archive/analysis/HISTORICAL_SOTA_AND_CURRENT_COMPARISON_V1.md` |
| MOGB 复现差距 V2 | 审计官方 BERT 单格是否收敛、子中心损失是否有效、平均半径边界为何偏离论文工作点 | 2 个 exact 单格、92 epoch 轨迹、127 个粒球、10 个短跑兼容单元、5 张图 | 完成，analysis-only | `docs/archive/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md` |
| MOGB 缺类安全回补 | selected-ball 筛选遗漏 Known 类能解释多少过拒绝，以及回补是否关闭 S2C–MOGB 差距 | 45 粒球拟合、180 评分、16 受影响单元、4 图 | 完成，Known-only component attribution | `docs/archive/analysis/MOGB_SELECTED_CLASS_RESCUE_V1.md` |
| S2C--MOGB 排序/工作点归因 | 区分当前优势来自阈值校准还是分数排序，并在相同 Known coverage 与事后最优阈值下检查差距 | 45 配对、90方法单元、9,090阈值曲线、6图 | 完成，analysis-only；oracle仅诊断 | `docs/archive/analysis/S2C_MOGB_OPERATING_CURVE_ATTRIBUTION_V1.md` |
| S2C--MOGB 逐意图结构桥接 | 判断 Known 恢复是否由少数缺球/高粒球数 intent 主导，并关联粒球复杂度与 OOS 代价 | 1,850 intent×seed、45单元、663 intent汇总、90相关结果、6图 | 完成，analysis-only；test 仅事后解释 | `docs/archive/analysis/S2C_MOGB_INTENT_STRUCTURE_BRIDGE_V1.md` |
| 当前协议三数据集 Cascade 桥接 | 在同一 Known-only Expert 下比较 Frozen K=1 与 Trainable K=1 的下游迁移，区分 Gate 分数收益与 Expert 偶然性 | 18 个 dataset×seed×Gate 单元（StackOverflow 6 + CLINC150/Banking77 12）、逐样本 Gate→Cascade 错误预算、3 图 | 完成，analysis-only | `docs/archive/analysis/CASCADE_BRIDGE_CROSS_DATASET_V1.md` |
