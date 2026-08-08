# 可训练 MiniLM 为什么低于 `fulltex.tex`：协议分层分析

更新时间：2026-08-06  
活动协议：`protocol_v2_textoir_v1`

## 先给结论

当前可训练 MiniLM 并不是“训练失败”，而是在**不同评价层级**下与 `fulltex.tex` 的历史结果比较：
当前运行的是 Known-only、Gate-only 的 Trainable K=1/K=2；`fulltex.tex` 的历史 `Ours` 表是完整
Gate→Router→Expert Cascade 的旧协议结果。两者不能直接当作同一方法的复现分数。因此当前
Trainable K=1 低于历史表，首要原因是协议和系统组成不同，其次才是表示训练本身的性能差异。

## 1. 代码中当前可训练 MiniLM 到底是什么

当前入口为 `src/protocol_v2/experiments/minilm_trainable_control_v1.py` 及其 KIR sweep 包装器。
它复用 `minilm_trainable_control_v1` 的训练逻辑：

1. 使用固定 `protocol_v2_textoir_v1` 的 canonical、registry 和 train/calibration split；
2. 从 `all-MiniLM-L6-v2` 初始化，只更新最后两层和 residual projection；
3. 只用 Known train 训练，并用 Known calibration 选 checkpoint；
4. 丢弃分类头，将 student embedding 送入当前 Gate；
5. 当前 sweep 的 Gate 是单中心 `K=1`、diagonal Mahalanobis、固定 `mean+std` 半径和同一阈值合同；
6. test OOS 只在配置冻结后评价，不用于训练或 checkpoint/参数选择。

因此它是“训练后的 MiniLM 表示 + Gate-only 评价”，不是 `fulltex.tex` 中的完整 Cascade。

## 2. 当前新实验：KIR sweep（Trainable K=1）

27/27 单元完成（3 数据集×3 KIR×3 seed），均值如下：

| 数据集 | KIR | OOS F1 | F1-All | Known Recall | False Accept |
|---|---:|---:|---:|---:|---:|
| CLINC150 | .25 | 94.94±0.30 | 79.01±0.95 | 73.33±0.61 | 3.61±0.54 |
| CLINC150 | .50 | 90.43±0.62 | 81.90±1.01 | 74.27±0.32 | 3.61±1.06 |
| CLINC150 | .75 | 82.55±0.42 | 82.43±0.36 | 74.57±0.55 | 4.06±0.50 |
| Banking77 | .25 | 92.17±0.78 | 79.27±2.29 | 81.93±0.75 | 9.45±1.61 |
| Banking77 | .50 | 84.77±0.71 | 82.31±0.83 | 81.91±0.17 | 13.46±1.22 |
| Banking77 | .75 | 69.53±3.37 | 84.20±1.07 | 82.61±0.42 | 18.33±5.89 |
| StackOverflow | .25 | 95.05±1.40 | 86.59±2.52 | 84.18±0.76 | 4.64±2.43 |
| StackOverflow | .50 | 86.71±0.96 | 85.65±0.37 | 83.92±0.39 | 11.14±2.02 |
| StackOverflow | .75 | 76.68±1.59 | 87.07±0.29 | 84.27±0.91 | 8.42±4.41 |

同距离 Frozen K=1 的 OOS F1 差值为：

| 数据集 | KIR=.25 | KIR=.50 | KIR=.75 |
|---|---:|---:|---:|
| CLINC150 | +0.64 pp | +2.41 pp | +3.59 pp |
| Banking77 | +0.59 pp | -0.05 pp | -14.02 pp |
| StackOverflow | +1.19 pp | +7.69 pp | +13.54 pp |

可见 Trainable K=1 在 CLINC150、StackOverflow 的三个 KIR 都改善 OOS 分数，但 Banking77 在
高 KIR 退化；这不是跨数据集 SOTA 结论。完整 CSV 和图在
`results/analysis/minilm_trainable_kir_sweep_v1/`、`figures/minilm_trainable_kir_sweep_v1/`。

## 3. 为什么会比 `fulltex.tex` 历史表低

### 3.1 比较的是 Gate-only 与完整 Cascade

`fulltex.tex` 的历史表 `tab:main_results_all` 报告的是 `Ours` 的旧系统级结果；代码中还有
SmolLM-135M Router/Expert 和 LoRA（见 `fulltex.tex:197--221`、`329`）。当前 Trainable 只替换/评估
MiniLM Gate，不训练或运行该 Router/Expert 链路，所以已知意图的 F1-All、Accuracy 和最终系统
OOS 结果都不在同一层。

例如 StackOverflow KIR=.50：历史 `Ours` 表为 OOS F1 `89.71`、Accuracy `85.54`；当前
Trainable K=1 为 OOS F1 `86.71±0.96`、F1-All `85.65±0.37`。这个约 3 pp 的差距不能归因于
“MiniLM 训练不如冻结”，因为历史数字包含不同 Cascade 组件。

### 3.2 数据、split、Known 列表和版本不完全相同

历史表使用旧数据/旧 split/旧实验实现；当前主协议使用固定 TEXTOIR snapshot、canonical、registry
和 Known-only calibration。当前 StackOverflow 测试池和历史 artifact 的 Known/OOS 计数、manifest
和代码合同也不同。因此即使数据集名称相同，也不能直接按百分数排序。

### 3.3 历史边界和调参合同更宽松

`fulltex.tex:329` 固定了历史的 `K_y=2`、数据集相关 λ（CLINC=.5，其余=1）；
`fulltex.tex:501` 还说明历史实验使用了部分 unknown/OOS validation 样本帮助学习 λ。当前主协议
不使用 test OOS 或 held-out OOS 选参数，并且已证明 StackOverflow 固定 K=2 会产生并集过覆盖。
历史参数不能直接迁移到当前协议。

### 3.4 KIR 越高，Known/OOS 几何重叠更严重

当前 sweep 清楚显示 KIR 增大时三个数据集的 OOS F1 普遍下降，尤其 Banking77 从 `.25` 的
`92.17` 降到 `.75` 的 `69.53`。这是已知意图密度上升带来的开放空间重叠，不是简单增加 epoch
就能消除的误差。

### 3.5 当前训练目标优化的是 K=1 表示，不是完整多中心边界

当前 Trainable checkpoint 的稳定收益集中在 K=1；同一 checkpoint 的 K=2−K=1 OOS F1 为
CLINC `-0.28pp`、Banking `+0.13pp`、StackOverflow `-19.06pp`，StackOverflow false acceptance
增加约 `34.11pp`。因此“训练 MiniLM”不能自动复现历史多中心 Cascade，更不能自动修复多个接受球
并集造成的误接收。

## 4. 与现有 baseline 的分层对照（StackOverflow，KIR=.50）

以下只列出可追溯的 Gate/兼容结果；监督条件、表示和 seed 不同的行不能混成无条件排名：

| 方法 | OOS F1 | Known Macro-F1 | Known Recall | Accuracy | 协议层 |
|---|---:|---:|---:|---:|---|
| Trainable K=1 | 86.71±0.96 | 85.55±0.33 | 83.92±0.39 | 85.82±0.78 | 当前协议、MiniLM、3 seeds |
| Trainable K=2 | 67.65±7.53 | 77.72±0.97 | 93.62±0.36 | 71.56±4.25 | 当前协议、固定 K、3 seeds |
| Frozen single centroid | 76.55 | 80.32 | 87.15 | 76.58 | 当前协议、Frozen MiniLM、5 seeds |
| Frozen fixed K=2 | 63.53 | 73.68 | 86.89 | 66.93 | 当前协议、Frozen MiniLM、5 seeds |
| MOGB-MiniLM | 72.92 | 40.34 | 27.09 | 63.06 | 公平组件适配、Frozen MiniLM、5 seeds |
| MOGB partition + s2c boundary | 79.25 | 61.75 | 50.39 | 74.07 | 公平组件适配、Frozen MiniLM、5 seeds |
| ADB | 89.47 | 87.44 | — | 88.53 | BERT/兼容单格，不是同协议主表 |
| DA-ADB | 90.90 | 89.06 | — | 90.07 | BERT/兼容单格，不是同协议主表 |
| MOGB official strict single-cell | 79.97 | 67.19 | 51.53 | 75.17 | BERT，官方逻辑现代兼容复现，未复现论文 |

来源：`results/analysis/kir50_method_comparison_v1/mean_std.csv`、
`docs/analysis/KIR50_METHOD_COMPARISON_V1.md`、`docs/对比实验/MOGB_DCLOOS_对比结果报告.md`。
ADB/DA-ADB 的数字不是严格公平胜负；MOGB 官方单格也不能写成论文结果已复现。

## 5. 最终判断

当前“可训练 MiniLM 低于 fulltex”的主要原因按优先级是：

1. **系统层级不同**：Gate-only 对比 full Cascade；
2. **数据和 split 合同不同**：历史快照/registry/测试池与当前协议不完全一致；
3. **调参监督不同**：历史使用了更宽松的 unknown/OOS validation，当前是 Known-only；
4. **边界结构不同**：历史固定 K=2 的收益不能在当前 StackOverflow 协议中重现；
5. **训练目标不同**：当前 Trainable 只证明 K=1 表示排序改善，不包含 Router/Expert 和端到端伪 OOS 监督。

因此目前最合理的表述不是“Trainable MiniLM 不如论文方法”，而是：

> 在严格的当前 Known-only Gate 合同下，Trainable MiniLM 能稳定改善部分数据集的单中心 OOS
> 排序；历史 `fulltex.tex` 的高分来自不同数据/调参合同和完整 Cascade，不能作为当前 Gate-only
> 训练结果的直接上限或失败证据。

## 6. 下一步

先完成当前结果的协议分层汇总和同合同统计核验，再决定是否做统一条件的强 baseline/端到端对照；
不要用本报告的 Trainable K=1 结果直接宣称 SOTA，也不要在 StackOverflow 上继续盲目扩 K=2--5。
