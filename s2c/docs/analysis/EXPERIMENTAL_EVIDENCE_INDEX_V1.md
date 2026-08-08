# s2c 实验结果与机制证据索引 V1

更新时间：2026-08-06  
活动协议：`protocol_v2_textoir_v1`

## 1. 当前研究主线

当前阶段的重点是实验、对比和机制分析，不把尚未公平验证的数字写成 SOTA。所有结果必须区分：

- 表示：Frozen MiniLM、Trainable MiniLM、BERT；
- 训练监督：Known-only、伪 OOS、外部 OOS；
- 边界：单中心、固定多中心、MOGB 粒球组件；
- 系统层级：Gate-only 或完整 Gate–Router–Expert Cascade。

## 2. 已完成实验地图

| 实验 | 主要问题 | 范围 | 当前状态 | 证据 |
|---|---|---|---|---|
| E2 固定 K 网格 | K、KIR、距离是否存在统一最优 | 3 数据集、11 KIR、5 seed、K=1..5、2 距离 | 1650/1650 | `docs/analysis/EXPERIMENT_EVIDENCE_PACK_V2.md` |
| E3 KMeans/随机与稳定性 | 多中心收益是否来自真实结构 | 720 控制单元、诊断统计 | 完成 | `docs/analysis/EXPERIMENT_EVIDENCE_PACK_V2.md` |
| RACAL Trainable K=1/K=2 | 训练表示能否修复固定多中心 | 三数据集、KIR=.50、多个 seed | 完成，K=1 有收益、K=2 不稳定 | `docs/racal_v1/` |
| Trainable KIR sweep | K=1 表示收益是否跨开放程度稳定 | 45 个五 seed 单元 | 完成 | `docs/analysis/MINILM_TRAINABLE_5SEED_FAIR_COMPARISON_V1.md` |
| 表示—边界诊断 | 收益来自 score 分离还是边界变化 | 90 run、443400 score | 完成 | `docs/analysis/MINILM_BOUNDARY_DIAGNOSTICS_V1.md` |
| 训练动态诊断 | Known-only 选模是否与 OOS 错位 | 45 run、179 epoch | 完成 | `docs/analysis/MINILM_TRAINING_DYNAMICS_V1.md` |
| 阈值/半径稳定性 | score 标度与半径估计是否解释 Trainable/fulltex 差距 | 810 阈值行、90 半径行 | 完成（仅诊断） | `docs/analysis/THRESHOLD_RADIUS_STABILITY_V1.md` |
| 同协议方法权衡 | Trainable 与 Frozen/MOGB 组件的覆盖—拒识工作点如何不同 | 315 行、486 paired effects、4 图 | 完成（仅诊断） | `docs/analysis/CROSS_PROTOCOL_TRADEOFF_V1.md` |
| Gate→Cascade 桥接 | Router/Expert 是否解释 Gate-only 与系统级结果差异 | 45 行、15 summary、3 图 | 完成（仅诊断） | `docs/analysis/GATE_CASCADE_BRIDGE_V1.md` |
| Trainable/MOGB 组件配对 | 自有方法与 MOGB 组件的覆盖—拒识权衡 | 45 Trainable + 135 MOGB 行 | 完成 | `docs/analysis/TRAINABLE_VS_MOGB_COMPONENT_V1.md` |
| MOGB 官方 BERT | 能否复现作者论文数字 | StackOverflow/Banking 单格 | `not_reproduced_strict` | `docs/对比实验/MOGB_DCLOOS_对比结果报告.md` |
| ADB/DA-ADB | 近期边界基线兼容性 | BERT 单格 | compatibility artifact | `docs/analysis/KIR50_METHOD_COMPARISON_V1.md` |
| DCLOOS | 端到端伪 OOS/外部 OOS 方法 | reduced-budget 单格 | 条件不同，非公平主表 | `docs/archive/external_baselines/dcloos/DCLOOS_REPRODUCTION_REPORT.md` |

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

- Trainable/Frozen KIR 曲线：`figures/minilm_trainable_5seed_fair_v1/trainable_vs_frozen_5seed_kir.png`
- Score 分布：`figures/minilm_boundary_diagnostics_v1/score_distributions_kir050.png`
- OOS F1—false acceptance：`figures/minilm_boundary_diagnostics_v1/oos_f1_false_acceptance_kir.png`
- Trainable/MOGB Known Recall—OOS F1：`figures/trainable_vs_mogb_component_v1/kir050_known_recall_oos_f1_components.png`
- Trainable/MOGB OOS F1 配对差值：`figures/trainable_vs_mogb_component_v1/paired_oos_f1_delta_heatmap.png`
- Calibration→test Known Recall：`figures/minilm_training_dynamics_v1/calibration_vs_test_known_recall.png`

## 8. 下一步实验顺序

1. 把已完成的 threshold/半径诊断纳入监督条件与系统层级分层，不使用 test oracle 选择正式参数；
2. 在相同 Frozen/Trainable 表示下补齐最强边界基线的多 seed 小矩阵；
3. 冻结 Gate 候选后，再做完整 Cascade 的传递实验；
4. 只有在监督条件、split、seed 和指标完全一致后，才做正式的 MOGB/DCLOOS 主表。

当前不应做：盲目增加 K、继续添加损失项、把官方 MOGB 负复现写成 MOGB 失败，或把 DCLOOS 的外部 OOS 监督结果与 Known-only 方法直接排名。

## 当前协议原生 OOD controls 与工作点诊断

`native_baselines_v1` 在同一 registry/view、冻结 MiniLM 和 Known-only calibration 下完成 MSP/Energy/kNN/LOF 的 180/180 矩阵，结果与训练表示分开记录：`docs/analysis/NATIVE_BASELINES_V1.md`、`results/analysis/native_baselines_v1/`。这些是原生控制，不是官方 ADB/DA-ADB/MOGB/DCLOOS 的替代品。

`operating_point_diagnostic_v1` 只做 retrospective matched-Known-Recall 对齐，用于解释为什么 Trainable 默认 OOS F1 较高但 Known Recall 较低，以及在共同工作点是否仍有 score-ranking 优势。它不选择正式阈值，不进入论文主结果：`docs/analysis/OPERATING_POINT_DIAGNOSTIC_V1.md`、`results/analysis/operating_point_diagnostic_v1/`。

## Trainable 与 MOGB frozen-component 归因

`trainable_vs_mogb_ablation_v1` 将 45 个 Trainable K=1 五 seed 单元与 180 个已完成 MOGB frozen-MiniLM 距离/半径组件行按 dataset×KIR×seed 配对，生成 324 个 bootstrap metric effects、180 个组件归因行和 3 张图。结果显示：MOGB 的 mean→mean+std 半径变化比 purity-get 阈值更能改变 OOS F1，但 Trainable 仍在相同工作点保留更高 Known Recall/F1-All。

证据入口：`docs/analysis/TRAINABLE_VS_MOGB_ABLATION_V1.md`、`results/analysis/trainable_vs_mogb_ablation_v1/`、`figures/trainable_vs_mogb_ablation_v1/`。

## Trainable 表示上的原生 detector 归因

`native_baselines_trainable_v1` 在已完成的 Trainable MiniLM checkpoint 上运行 MSP、Energy、kNN、LOF，
范围为 3 数据集×KIR=.50×3 seed，共 36/36。它同时与 Frozen native 和 Trainable Gate K=1 配对，
回答“收益来自表示、检测器还是 Gate 几何”。

报告：`docs/analysis/NATIVE_BASELINES_TRAINABLE_V1.md`；结果：`results/analysis/native_baselines_trainable_v1/`；
可视化：`figures/native_baselines_trainable_v1/`。

## 实验机制分析包 V3（2026-08-06）

基于已有 `minilm_trainable_5seed_fair_v1/all_methods_per_seed.csv` 的 315 行逐 seed 结果，新增 324 个
paired bootstrap effects、63 个方法汇总、63 个 Pareto 标记和四张可视化。该包只做轻量结果再分析，不读取
checkpoint、embedding 或原始文本，也不产生新的训练证据。入口：
`docs/analysis/EXPERIMENTAL_MECHANISM_PACK_V3.md`；数据：`results/analysis/experimental_mechanism_pack_v3/`；
图：`figures/experimental_mechanism_pack_v3/`。
