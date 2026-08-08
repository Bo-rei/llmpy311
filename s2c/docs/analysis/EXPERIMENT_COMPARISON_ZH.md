# s2c 当前实验与基线对比总报告

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
- `results/analysis/active_experiment_dashboard_v1/current_cascade_summary.csv`

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
[`trainable_cross_dataset.png`](../../figures/active_experiment_dashboard_v1/trainable_cross_dataset.png)。

## 3. 为什么当前 Trainable 结果低于 `fulltex.tex` 的 89.71%

这不是已证明的模型退化，而是实验合同不同：

1. `fulltex.tex` 的 `Ours=89.71` 是历史完整系统表面值；当前 Trainable 结果是 Gate-only。
2. 历史 Cascade 还包含语义 Gate、Router、Expert 和历史训练/阈值设置（`fulltex.tex:329`）。
3. 当前使用 `protocol_v2_textoir_v1` 的 TEXTOIR 快照；论文使用旧 StackOverflow 快照、旧 Known 列表和旧 split。
4. 历史 artifact 中 `metrics.oos_f1=0.8971`，但原始 `primary_metrics.oos_f1=0.5910`，并且 `ablation_summary.csv` 明确记录了 `main_table_ours` 指标覆盖。

所以 89.71% 不能作为当前 Trainable Gate 的直接可复算目标。详细审计见 [`WHY_TRAINABLE_MINILM_BELOW_LATEX.md`](WHY_TRAINABLE_MINILM_BELOW_LATEX.md) 和 `results/analysis/active_experiment_dashboard_v1/historical_latex_metric_audit.csv`。

## 4. K 与 KIR 的真实规律

固定 K 消融不支持“统一最优 K”：

- CLINC150：通常 K=2 或 K=3 只有小幅收益，继续增加 K 往往回落。
- Banking77：测试集 oracle 结果常随 K 增大而上升，但 Known Recall 下降约 9–15 个百分点，说明 OOS 提升伴随明显误拒。
- StackOverflow：KIR=0.25、0.50、0.75 和两种距离下，测试集 oracle 最优基本都是 K=1。

新增分析文件：

- `results/analysis/active_experiment_dashboard_v1/k_selection_summary.csv`
- `figures/active_experiment_dashboard_v1/k_selection_tradeoff.png`

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

这些数字说明：MOGB 的动态划分在 StackOverflow 上并非完全无效，但 OOS F1 的提升伴随大幅 Known 退化；当前 s2c 不能据此宣称“整体优于 MOGB”。组件差值见 [`fair_component_gaps.csv`](../../results/analysis/active_experiment_dashboard_v1/fair_component_gaps.csv) 和 [`fair_component_gaps.png`](../../figures/active_experiment_dashboard_v1/fair_component_gaps.png)。

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

- [`k_sweep_oos_f1.png`](../../figures/active_experiment_dashboard_v1/k_sweep_oos_f1.png)：KIR×K×距离的 OOS F1 曲线。
- [`k_selection_tradeoff.png`](../../figures/active_experiment_dashboard_v1/k_selection_tradeoff.png)：oracle K、Known Recall 约束和数据集差异。
- [`trainable_k1_k2_tradeoff.png`](../../figures/active_experiment_dashboard_v1/trainable_k1_k2_tradeoff.png)：Trainable K=1/K=2 的 OOS–Known 权衡。
- [`representation_k_interaction.png`](../../figures/active_experiment_dashboard_v1/representation_k_interaction.png)：Frozen、CE、SupCon、CE-Recon 与 K 的交互。
- [`fair_component_gaps.png`](../../figures/active_experiment_dashboard_v1/fair_component_gaps.png)：同一冻结 MiniLM 下各组件相对单中心的变化。
- [`stackoverflow_known_oos_tradeoff.png`](../../figures/active_experiment_dashboard_v1/stackoverflow_known_oos_tradeoff.png)：当前方法、MOGB 组件和历史表面值的分层散点图。

## 9. 下一批实验顺序

为了继续以实验为主，不先重写方法，下一批应按以下顺序：

1. **表示诊断与校准：** 继续分析 Trainable/Frozen 的类内方差、半径、Known Recall 和阈值稳定性，不直接扩大 K。
2. **逐 intent MOGB 归因：** 对同一 split 输出粒球数、半径、Known false rejection 和 OOS false acceptance，解释 MOGB 与固定 K 的差异来源。
3. **统一外部基线小矩阵：** 先只做 KIR=0.50、同一 Known 列表和 3 个 seed，分别记录 ADB、DA-ADB、DCLOOS 的监督条件。
4. **完整 Cascade 对照：** Frozen K=1、Trainable K=1、当前最安全 Gate 使用同一 Router/Expert 和原始指标重算，确认 Gate 改进能否传递到端到端系统。

在上述对照完成前，不把任何单 seed 兼容性结果写成 SOTA，也不继续盲目扩大 K 或新增复杂损失。
