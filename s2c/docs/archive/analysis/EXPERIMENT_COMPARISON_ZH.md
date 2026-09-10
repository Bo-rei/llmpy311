# s2c 当前实验与基线对比总报告

> **口径提示（2026-08-09）**：本文包含较早批次的三 seed/不同来源汇总。当前五 seed 同协议的权威数字、正确方向排名和外部基线合同边界以 [`EXPERIMENT_ANALYSIS_MASTER_V1.md`](EXPERIMENT_ANALYSIS_MASTER_V1.md) 为准；本文历史数字不得覆盖主入口。

更新时间：2026-08-06  
活动协议：`protocol_v2_textoir_v1`

本报告只汇总已经落盘的实验结果，不把不同数据版本、不同监督条件和不同系统层级的数字混成一个排名。原始结果、配置和 manifest 仍以各自路径为准。

## 1. 当前方法到底是什么

当前 s2c 的主 Gate 是：

```text
all-MiniLM-L6-v2
→ 384 维句向量（Frozen 或 Known-only Trainable）
→ L2 归一化
→ 每个 Known intent 内 KMeans 固定 K 分簇
→ 每个簇计算中心、对角 Mahalanobis 距离和 μ+λσ 半径
→ min(distance / radius) ≤ 1 接受 Known，否则 OOS
```

历史 `fulltex.tex` 描述的也是固定后处理多中心：MiniLM 不训练、KMeans 分簇、局部半径和最近球判定（见 `fulltex.tex:223-275`）。当前新增的 Trainable MiniLM 属于 RACAL-v1 控制实验，不是论文原方法的直接复现。

## 2. 最可靠的当前结果

StackOverflow、KIR=0.50、当前协议下：

| 方法 | 层级 | OOS F1 | F1-All | Known Recall | False Acceptance |
|---|---|---:|---:|---:|---:|
| Frozen K=1 | Gate-only | 77.29% | 78.60% | 83.71% | 26.54% |
| Trainable MiniLM K=1 | Gate-only | **86.71%** | **85.65%** | 83.92% | **11.14%** |
| Trainable fixed K=2 | Gate-only | 67.65% | 76.81% | 93.62% | 45.26% |
| CE-Recon selected-K | 当前 Cascade | 87.62% | — | 86.15% | 11.06% |

因此，Trainable MiniLM 对单中心 Gate 是真实改进：相对当前 Frozen K=1，OOS F1 提升约 9.42 个百分点，Known Recall 基本不变，false acceptance 降低约 15.40 个百分点。但它没有解决固定多中心的接受区域过覆盖。

数据来源：

- `results/diagnostics/racal_v1/RACAL_V1_STAGE1_MEAN_STD.csv`
- `results/diagnostics/racal_v1/stage2_fixed_k2/RACAL_V1_STAGE2_MEAN_STD.csv`
- `results/analysis/archive/analysis/active_experiment_dashboard_v1/current_cascade_summary.csv`

### 2.1 跨数据集 Trainable K=1 控制（KIR=0.50）

为回答“Trainable MiniLM 是否只在 StackOverflow 偶然有效”，已在 CLINC150、Banking77 上新增
13/42/87 三个 seed；Frozen 基线逐 seed 读取同一 protocol 的 E2 K=1、对角 Mahalanobis 单元，
StackOverflow 复用已完成的 RACAL-v1 同协议控制。该表不混用论文旧快照，也不使用测试 OOS 选 checkpoint。

| 数据集 | Frozen OOS F1 | Trainable OOS F1 | 差值 | Frozen Known Recall | Trainable Known Recall | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | 89.31% | 90.43% | +1.12 pp | 75.60% | 74.27% | -1.33 pp |
| Banking77 | 79.58% | 84.77% | +5.18 pp | 83.86% | 81.91% | -1.95 pp |
| StackOverflow | 77.29% | 86.71% | +9.42 pp | 83.71% | 83.92% | +0.21 pp |

这批结果说明 Trainable MiniLM 的 K=1 收益具有跨数据集一致的正向 OOS 变化，但 Known Recall
代价并不一致：StackOverflow 基本不损失，CLINC150/Banking77 有小幅下降。因此当前结论是“表示
适配值得继续做”，而不是“训练后必然优于 Frozen”。这也解释了为什么不能直接把 Trainable K=1
结果与 `fulltex.tex` 的历史 89.71% 做一对一排名：后者是旧快照下的完整 Cascade 表面值，而这里是
当前 TEXTOIR 快照、Gate-only、Known-only checkpoint 选择。

证据入口：[`MINILM_TRAINABLE_CONTROL_V1.md`](MINILM_TRAINABLE_CONTROL_V1.md)、
[`paired_deltas.csv`](../../results/diagnostics/minilm_trainable_control_v1/paired_deltas.csv)、
[`trainable_cross_dataset.png`](../../figures/archive/analysis/active_experiment_dashboard_v1/trainable_cross_dataset.png)。

## 3. 为什么当前 Trainable 结果低于 `fulltex.tex` 的 89.71%

这不是已证明的模型退化，而是实验合同不同：

1. `fulltex.tex` 的 `Ours=89.71` 是历史完整系统表面值；当前 Trainable 结果是 Gate-only。
2. 历史 Cascade 还包含语义 Gate、Router、Expert 和历史训练/阈值设置（`fulltex.tex:329`）。
3. 当前使用 `protocol_v2_textoir_v1` 的 TEXTOIR 快照；论文使用旧 StackOverflow 快照、旧 Known 列表和旧 split。
4. 历史 artifact 中 `metrics.oos_f1=0.8971`，但原始 `primary_metrics.oos_f1=0.5910`，并且 `ablation_summary.csv` 明确记录了 `main_table_ours` 指标覆盖。

所以 89.71% 不能作为当前 Trainable Gate 的直接可复算目标。详细审计见 [`WHY_TRAINABLE_MINILM_BELOW_LATEX.md`](WHY_TRAINABLE_MINILM_BELOW_LATEX.md) 和 `results/analysis/archive/analysis/active_experiment_dashboard_v1/historical_latex_metric_audit.csv`。

## 4. K 与 KIR 的真实规律

固定 K 消融不支持“统一最优 K”：

- CLINC150：通常 K=2 或 K=3 只有小幅收益，继续增加 K 往往回落。
- Banking77：测试集 oracle 结果常随 K 增大而上升，但 Known Recall 下降约 9–15 个百分点，说明 OOS 提升伴随明显误拒。
- StackOverflow：KIR=0.25、0.50、0.75 和两种距离下，测试集 oracle 最优基本都是 K=1。

新增分析文件：

- `results/analysis/archive/analysis/active_experiment_dashboard_v1/k_selection_summary.csv`
- `figures/archive/analysis/active_experiment_dashboard_v1/k_selection_tradeoff.png`

其中 `oracle_best_k` 只用于描述测试敏感性，不能用于正式选 K；`safe_best_k_1pp` 也只是预注册的 Known Recall 诊断，不是已经验证的选择器。

## 5. 与同协议冻结 MiniLM 基线的组件比较

StackOverflow、KIR=0.50、5 个 seed、冻结 MiniLM：

| 方法 | OOS F1 | Known F1 | Known Recall | 解释 |
|---|---:|---:|---:|---|
| Single centroid | 76.55% | 80.32% | 87.15% | 安全单中心基线 |
| Random partition | 75.88% | 80.20% | 87.64% | 增加中心本身没有收益 |
| Fixed K=2 | 63.53% | 73.68% | 86.89% | KMeans 多球明显退化 |
| MOGB-MiniLM | 72.92% | 40.34% | 27.09% | 动态粒球/平均半径组合不稳定 |
| MOGB partition + s2c boundary | **79.25%** | 61.75% | 50.39% | OOS F1 提升，但 Known 覆盖严重下降 |
| s2c partition + MOGB boundary | 73.35% | 62.61% | 54.49% | 仅换边界无法解决问题 |

这些数字说明：MOGB 的动态划分在 StackOverflow 上并非完全无效，但 OOS F1 的提升伴随大幅 Known 退化；当前 s2c 不能据此宣称“整体优于 MOGB”。组件差值见 [`fair_component_gaps.csv`](../../results/analysis/archive/analysis/active_experiment_dashboard_v1/fair_component_gaps.csv) 和 [`fair_component_gaps.png`](../../figures/archive/analysis/active_experiment_dashboard_v1/fair_component_gaps.png)。

## 6. 外部端到端基线的比较边界

当前已经有以下兼容性结果，但不能直接作为统一主排名：

| 方法 | StackOverflow OOS F1 | 状态 |
|---|---:|---|
| ADB | 89.47% | 单 seed、BERT 兼容性结果 |
| DA-ADB | 90.90% | 单 seed、BERT 兼容性结果 |
| DCLOOS reduced | 87.05% | 使用伪 OOS/外部 OOS，非公平当前协议 |
| MOGB official strict | 未复现论文 | StackOverflow 单格和 Banking 单格均未达到论文数字 |

因此目前没有证据证明当前方法已经超过 ADB、DA-ADB 或 DCLOOS；也没有证据证明这些结果与当前 Frozen MiniLM Gate 完全公平。DCLOOS 使用外部/伪 OOS 监督，必须单独标注。

## 7. 当前实验真正证明了什么

1. **表示层结论：** Known-only Trainable MiniLM 能稳定改善 K=1 Gate。
2. **多中心结论：** 表示改善不能自动转化为安全的 K>1；StackOverflow 的 K=2 主要表现为 false acceptance 爆炸。
3. **数据集结论：** Banking77 的多中心收益具有条件性，CLINC150 收益有限，StackOverflow 固定多中心持续失败。
4. **基线结论：** MOGB 组件值得比较，但当前公平矩阵尚不足以声称 s2c 全面领先；ADB/DA-ADB/DCLOOS 仍需统一监督条件和 split 后再排名。

## 8. 已生成的可视化

- [`k_sweep_oos_f1.png`](../../figures/archive/analysis/active_experiment_dashboard_v1/k_sweep_oos_f1.png)：KIR×K×距离的 OOS F1 曲线。
- [`k_selection_tradeoff.png`](../../figures/archive/analysis/active_experiment_dashboard_v1/k_selection_tradeoff.png)：oracle K、Known Recall 约束和数据集差异。
- [`trainable_k1_k2_tradeoff.png`](../../figures/archive/analysis/active_experiment_dashboard_v1/trainable_k1_k2_tradeoff.png)：Trainable K=1/K=2 的 OOS–Known 权衡。
- [`representation_k_interaction.png`](../../figures/archive/analysis/active_experiment_dashboard_v1/representation_k_interaction.png)：Frozen、CE、SupCon、CE-Recon 与 K 的交互。
- [`fair_component_gaps.png`](../../figures/archive/analysis/active_experiment_dashboard_v1/fair_component_gaps.png)：同一冻结 MiniLM 下各组件相对单中心的变化。
- [`stackoverflow_known_oos_tradeoff.png`](../../figures/archive/analysis/active_experiment_dashboard_v1/stackoverflow_known_oos_tradeoff.png)：当前方法、MOGB 组件和历史表面值的分层散点图。

## 8.1 跨数据集错误归因

新增 `docs/archive/analysis/CROSS_DATASET_ERROR_ATTRIBUTION_V1.md`。它在同一 `sample_id` 合同下复核了三数据集、三 KIR、五 seed 的 315 个已有 fair run，共 1,890,000 条预测记录。结果显示，Trainable K=1 相对 Frozen K=1、Frozen K=2 和 Random K=2 的 false acceptance 在全部九个 dataset×KIR 单元都更低，且 StackOverflow 的优势随 KIR 增大而扩大；这支持“Trainable K=1 的主要收益来自更好的 Known/OOS 分数分离，而非单纯加大拒识”的机制解释。

MOGB-MiniLM 与 MOGB partition + ours boundary 的 false acceptance 往往低于 Trainable，但其 false rejection 显著更高，尤其 MOGB-MiniLM 在三个数据集上都更保守。因此这批结果只能说明风险取舍和错误来源，不能替代同监督、同 split 的外部 baseline 实验，更不能据此宣称已经达到 SOTA。

### 8.2 逐 intent 机制图

新增 `docs/archive/analysis/CROSS_DATASET_INTENT_RISK_VISUALS_V1.md`。KIR=.50 时，StackOverflow Frozen K=2 的 `cocoa`、`sharepoint`、`osx`、`spring`、`scala` 五个 intent 合计贡献约 64.9% 的 OOS false acceptance；这说明多中心的主要损失来自少数边界吸收器。MOGB-MiniLM 的前五 intent 贡献约 83.7%，但它整体 Known false rejection 更高，因而不能只从误接收占比判断优劣。

逐 intent 差值图还显示：Frozen K=2 的错误点主要位于“Known 误拒没有增加、OOS 误接受增加”的方向；MOGB 点主要位于“Known 误拒大幅增加、OOS 误接受减少”的方向；Trainable K=1 处在两类风险之间更平衡。

### 8.3 五 seed 统计稳定性

新增 `docs/archive/analysis/STATISTICAL_STABILITY_V1.md`。在同一 `dataset × KIR × seed` 配对下，Trainable K=1 相对 Frozen K=1、Frozen K=2 和 Random K=2 的 OOS F1 在 9/9 个组合中五个 seed 全部胜出；相对 MOGB 组件在 8/9 个组合中配对 bootstrap CI 为正。按 OOS F1 的平均排名，Trainable 在 8/9 个 dataset×KIR 单元排名第一，唯一例外是 Banking77/KIR=.25。

这说明当前 Trainable 优势具有 seed 稳定性，但不能把五 seed 的同协议 fair 统计扩展为对 ADB、DA-ADB 或 DCLOOS 的 SOTA 结论；这些方法仍需在统一监督和 runtime 下单独完成。

## 9. 下一批实验顺序

为了继续以实验为主，不先重写方法，下一批应按以下顺序：

1. **表示诊断与校准：** 继续分析 Trainable/Frozen 的类内方差、半径、Known Recall 和阈值稳定性，不直接扩大 K。
2. **逐 intent MOGB 归因：** 对同一 split 输出粒球数、半径、Known false rejection 和 OOS false acceptance，解释 MOGB 与固定 K 的差异来源。
3. **统一外部基线小矩阵：** 先只做 KIR=0.50、同一 Known 列表和 3 个 seed，分别记录 ADB、DA-ADB、DCLOOS 的监督条件。
4. **完整 Cascade 对照：** Frozen K=1、Trainable K=1、当前最安全 Gate 使用同一 Router/Expert 和原始指标重算，确认 Gate 改进能否传递到端到端系统。

在上述对照完成前，不把任何单 seed 兼容性结果写成 SOTA，也不继续盲目扩大 K 或新增复杂损失。

## 10. 新增配对统计与 intent-level 分析

为避免只看汇总均值，新增 `PAIRED_EFFECT_INTENT_HETEROGENEITY_V1.md`：

- 对同一 `dataset × KIR × seed` 配对计算 Trainable K=1 与 Frozen/MOGB 组件的 OOS F1、F1-All、Known Recall、false acceptance 差值；
- 使用 10,000 次固定 RNG bootstrap 输出 95% CI、win/tie/loss；
- 统计每个 intent×seed 在测试 oracle 下的 `safe_gain_oracle`、`oracle-best-K` 和 Known/OOS 变化。

新增图表位于 `figures/archive/analysis/paired_effect_intent_heterogeneity_v1/`：

1. `paired_oos_f1_forest.png`：三数据集、三个 KIR 的配对 OOS F1 森林图；
2. `kir50_tradeoff_heatmap.png`：KIR=0.50 四项指标配对差值；
3. `safe_gain_heatmap.png`：逐 intent 多中心安全收益比例；
4. `oracle_best_k_distribution.png`：Euclidean 下 oracle-best-K 分布。

这批结果是后验机制证据，不新增训练，也不把 test-oracle 结果用于正式选择中心数。它显示 Trainable K=1 的跨 seed 收益具有一致方向，而多中心潜在收益集中在 Banking77 的部分 intent；StackOverflow 的安全收益比例在高 KIR 下下降，支持“多中心收益具有数据集和 intent 异质性”的判断。

## 11. StackOverflow 逐样本错误归因

新增 [`STACKOVERFLOW_ERROR_ATTRIBUTION_V2.md`](STACKOVERFLOW_ERROR_ATTRIBUTION_V2.md)，使用同一 `sample_id` 对齐 7 种方法、5 个 seed、共 210,000 条记录。它显示正式 score<=1 下，Trainable K=1 的 OOS F1/false acceptance/false rejection 为 `0.8767/0.0934/0.1611`；Frozen K=2 为 `0.6353/0.4717/0.1311`；MOGB-MiniLM 为 `0.7292/0.0079/0.7291`。

因此，Trainable K=1 的当前优势可以具体解释为：固定 K=2 主要新增 OOS 误接收，MOGB-MiniLM 主要新增 Known 误拒绝，而 Trainable K=1 在两类风险之间更平衡。固定 K=2 的误接收主要由 `cocoa、sharepoint、osx、spring、scala` 五个 intent 贡献。该结论仍限于 StackOverflow/KIR=.50 的已有组件结果，不代表已经完成外部基线公平排名。

## 12. Gate→Cascade 配对桥接

新增 `docs/archive/analysis/GATE_CASCADE_PAIRED_BRIDGE_V2.md`。这不是新模型实验，而是对已有 Gate/Cascade
结果的评价层审计：`trainable_k1_gate` 只有 Gate 指标，其他四类记录来自 Cascade。报告因此将
Trainable 与 Cascade 的比较称为 bridge，并将 CE-Recon selected-K、Frozen selected-K 与 Best
controlled 仅在 Cascade 内配对。

桥接结果显示 Trainable Gate 相对 Frozen K=1 Cascade 的 OOS F1 差值在 CLINC150、Banking77、
StackOverflow 分别为 +2.41、−0.05、+7.69 pp，false acceptance 下降约 3.44、10.20、12.38 pp。
同层 Cascade 中 CE-Recon selected-K 的 OOS F1 相对 Frozen K=1 分别为 +1.97、+4.66、+8.60 pp，
但 Banking77 Known macro-F1 下降约 3.25 pp。结果可用于解释 Gate 改善如何被 Router/Expert 传递，
不能用于外部 baseline 或 SOTA 排名。

## 13. 同一 Trainable 表示下的原生检测器对照

新增 `docs/archive/analysis/TRAINABLE_DETECTOR_MECHANISM_V1.md`。在 KIR=.50、三个数据集和三个
seed 中，Gate 与 MSP、Energy、kNN、LOF 共享同一 Trainable MiniLM checkpoint；检测器只
使用 Known calibration 进行自身校准。

Gate 的 OOS F1 相对四种检测器在 12/12 个比较中均为正，StackOverflow 的提升最大；但
Known Recall 同时下降约 11–21 pp。因而当前最准确的解释是：Gate 形成了更开放集导向的
coverage–rejection 工作点，原生检测器保留更多 Known、却误接受更多 OOS。该证据比单独
比较 Frozen 和 Trainable 更能区分“表示收益”和“边界/校准收益”，但仍不是外部 SOTA 排名。

## 14. 全指标 Fair Effect Landscape

新增 `docs/archive/analysis/FAIR_EFFECT_LANDSCAPE_V1.md`、
`results/analysis/archive/analysis/fair_effect_landscape_v1/` 和
`figures/archive/analysis/fair_effect_landscape_v1/`。该阶段复用五 seed 配对统计源表，不重跑模型。
源表共有 486 行，本阶段按预注册范围分析 432 行（3 数据集 × 3 KIR × 6 比较 × 8 指标）。

Trainable K=1 相对 Frozen K=1/K=2/Random K=2 的 OOS F1 平均优势约为 `+7.1/+10.2/+7.4`
个百分点，9 个 dataset×KIR 单元均有稳定正差异。相对 MOGB MiniLM、MOGB partition + ours
boundary、ours partition + MOGB boundary 的平均 OOS F1 优势约为 `+12.4/+8.0/+9.9` pp，
其中 8/9 单元的 CI 排除零；Banking77/KIR=.25 是局部例外。

多指标结果必须一起看：Trainable 相对 Frozen K=1 的 F1-All/F1-K/AUROC/AUPR-OOS 分别约
`+5.1/+5.0/+3.8/+5.4` pp，false acceptance 降低约 `14.3` pp，但 Known Recall 下降约
`3.3` pp。MOGB 组件则明显偏向保守拒识，Known false rejection 大幅增加。因此当前
Trainable 是 fair matrix 内较平衡的自有工作点，而不是在所有指标上无条件优于 MOGB。

这些图和数字不能替代同监督 ADB、DA-ADB、DCLOOS 的统一主表，也不能写成 SOTA 或官方
MOGB 复现结论。

## DCLOOS 目前为什么没有进入对比主表

DCLOOS 的官方代码和外部 SQuAD 负样本已经接入并完成隔离 smoke，但它在输入 train.tsv 上会重新
随机选择 Known 类别。当前 protocol registry 的 Known 列表与 DCLOOS 的随机列表不一致，而且 DCLOOS
额外使用 pseudo-OOS 和外部 OOS 监督。因此它不是与 S2C 同训练合同的方法。

1 epoch smoke 已输出完整指标，但 OOS F1=0、F1-All=8.31%，只能说明端到端链路可运行，不能说明
DCLOOS 性能。既有 reduced 单元也只能作为外部监督参考，不能与当前 Trainable K=1、MOGB-MiniLM
或历史 Cascade 混排名。具体输入列表、artifact 和兼容性 blocker 见
`docs/archive/analysis/DCLOOS_CONTRACT_STATUS_V1.md`。

## 15. KIR 敏感性分解

新增 `docs/archive/analysis/KIR_SENSITIVITY_DECOMPOSITION_V1.md`、
`results/analysis/archive/analysis/kir_sensitivity_decomposition_v1/` 和
`figures/archive/analysis/kir_sensitivity_decomposition_v1/`。该阶段使用已有五 seed summary，计算每个
`dataset × method × metric` 在 KIR=.25→.75 的端点变化和斜率，没有重复训练。

Trainable K=1 的 OOS F1 下降在 CLINC150/Banking77/StackOverflow 为 `−12.3/−23.0/−19.2`
pp；Frozen K=2 为 `−13.0/−26.5/−43.0` pp。StackOverflow 固定 K=2 还伴随约 `+42.2 pp`
false acceptance 增量，说明随着开放比例变化，固定多中心的接受区域风险快速扩大。

MOGB MiniLM 的 false acceptance 端点变化接近零，但 Known Recall 分别下降约 7.9、11.2、
11.4 pp；这进一步说明其当前组件结果是保守拒识工作点。Trainable K=1 的 Known Recall
端点变化约为 `+1.5/+0.6/−0.2` pp，更适合作为当前自有的平衡工作点。该趋势分析仍不等同
于外部基线公平排名或 SOTA 证据。
## 16. OOS precision/recall 与错误预算

新增 `docs/archive/analysis/OOS_ERROR_BUDGET_V1.md`、`results/analysis/archive/analysis/oos_error_budget_v1/` 和
`figures/archive/analysis/oos_error_budget_v1/`。这一阶段只消费 315 个已审计的同协议 fair per-seed 行，
按 `n_known/n_oos` 和 FA/FR 重构 OOS precision/recall，并检查重构 F1 与源表一致。

KIR=.50 的五 seed 均值显示：

| 数据集 | 方法 | OOS F1 | OOS Precision | OOS Recall | Known Recall | FA | FR |
|---|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | Trainable K=1 | 90.44 | 85.25 | 96.31 | 74.44 | 3.69 | 25.56 |
| Banking77 | Trainable K=1 | 83.56 | 82.93 | 84.26 | 82.21 | 15.74 | 17.79 |
| StackOverflow | Trainable K=1 | 87.67 | 84.91 | 90.66 | 83.89 | 9.34 | 16.11 |
| StackOverflow | Frozen K=2 | 63.53 | 80.09 | 52.83 | 86.89 | 47.17 | 13.11 |
| StackOverflow | MOGB MiniLM 组件 | 72.92 | 57.65 | 99.21 | 27.09 | 0.79 | 72.91 |

所以当前结果应解释为工作点差异：固定多中心的 StackOverflow 失败来自 OOS 过接收，
MOGB MiniLM 组件的低 FA 来自极强的 Known 过拒绝，Trainable K=1 在两类风险之间更平衡。
这仍不能替代同监督完整 MOGB、ADB、DA-ADB 和 DCLOOS 的公平主表，也不能声称 SOTA。
