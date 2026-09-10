# Gate→Cascade 配对桥接分析 V2（中文）

## 1. 目的与边界

本阶段只分析已经完成的 `protocol_v2_textoir_v1` 结果，没有重新训练、调阈值、修改
E2/E3/R1 或覆盖历史 artifact。输入为
`results/analysis/archive/analysis/gate_cascade_bridge_v1/per_seed.csv` 的 45 行：3 个数据集 × 3 个
seed × 5 个 Gate/Cascade 条目。

需要特别区分两个评价层：`trainable_k1_gate` 是 **Gate-only** 行，而
`frozen_k1`、`frozen_selected_k`、`ce_recon_selected_k` 和
`best_controlled_baseline` 是 **Cascade** 行。二者的 Router/Expert、标签空间和错误
传播合同不同，因此本报告的 Trainable–Cascade 差值是“桥接诊断”，不是端到端统一排名。
Cascade 内部的配对比较才是同层级比较。

统计使用同一数据集和 seed 配对，固定 bootstrap seed `20260725`、10,000 次重采样；
3 个 seed 的样本量很小，置信区间只作稳定性描述，不替代完整五 seed 外部基线实验。

## 2. 输出

- `results/analysis/archive/analysis/gate_cascade_paired_bridge_v2/gate_vs_cascade_effects.csv`
- `results/analysis/archive/analysis/gate_cascade_paired_bridge_v2/cascade_pairwise_effects.csv`
- `results/analysis/archive/analysis/gate_cascade_paired_bridge_v2/MANIFEST.json`
- `figures/archive/analysis/gate_cascade_paired_bridge_v2/gate_cascade_oos_f1_bridge_forest.png`
- `figures/archive/analysis/gate_cascade_paired_bridge_v2/cascade_only_oos_accuracy_tradeoff.png`
- `figures/archive/analysis/gate_cascade_paired_bridge_v2/cascade_router_expert_error_decomposition.png`

## 3. Gate-only 到 Cascade 的桥接结果

相对于 Frozen K=1 Cascade，Trainable K=1 Gate 的 OOS F1 差值为：

| 数据集 | Trainable Gate − Frozen K=1 Cascade | 同方向 seed | 解释 |
|---|---:|---:|---|
| CLINC150 | +2.41 pp | 3/3 | Gate 层明显改善，但不能直接等同为 Cascade 提升 |
| Banking77 | −0.05 pp | 2/3 | OOS F1 基本持平，Gate false acceptance 更低 |
| StackOverflow | +7.69 pp | 3/3 | Gate 层改善最大，且 false acceptance 下降约 12.38 pp |

false acceptance 的桥接差值（Cascade − Trainable，正值表示 Trainable 更安全）为：

- CLINC150：+3.44 pp；
- Banking77：+10.20 pp；
- StackOverflow：+12.38 pp。

这说明 Trainable 表示在 Gate 层确实形成了更安全的 Known/OOS 分离；但 Gate 改善
进入完整 Cascade 后，还会经过 Router 和 Expert，不能把上述差值写成完整系统的
同监督胜出结论。

## 4. Cascade 内部比较

在同一 Cascade 合同下，CE-Recon selected-K 相对于 Frozen K=1 的 OOS F1 提升为：

- CLINC150：+1.97 pp，整体准确率 +2.73 pp，Known macro-F1 +1.18 pp；
- Banking77：+4.66 pp，整体准确率 +5.96 pp，但 Known macro-F1 差值为 −3.25 pp；
- StackOverflow：+8.60 pp，整体准确率 +8.12 pp，Known macro-F1 +2.70 pp。

该结果只能说明当前 Cascade 数据中 CE-Recon selected-K 是一个较强的受控候选；它
不能说明 CE-Recon 已经超过 MOGB、ADB、DA-ADB 或 DCLOOS，因为这些外部方法尚未在
相同监督、split、运行环境和 Cascade 接口下完成统一运行。

Frozen selected-K 在本 3-seed 输入中与 Frozen K=1 的行数值相同，故其配对差值为零；
这不是“证明 selected-K 没有作用”的一般结论，只说明当前桥接源文件的这组 Cascade
记录没有产生不同的 Frozen selected-K 结果。

## 5. 错误传播图的含义

`cascade_router_expert_error_decomposition.png` 将 Router error 和 Expert error 分开，
用于判断 Gate 层变化是否在 Cascade 中被后续模块放大或抵消。当前可见趋势是：

- CLINC150 的 CE-Recon Cascade 同时降低 Expert error，并提高整体准确率；
- Banking77 的 CE-Recon Cascade OOS F1 和整体准确率上升，但 Known macro-F1 下降，
  说明开放拒识收益伴随 Known 分类代价；
- StackOverflow 的 CE-Recon Cascade 在该桥接源中改善 OOS F1、整体准确率和 Known
  macro-F1，但这仍不是外部基线意义上的 SOTA 证据。

## 6. 当前能得出的研究结论

1. Trainable MiniLM 的优势首先在 Gate 层得到证据支持，尤其是 StackOverflow 的
   false acceptance 降低；
2. Gate 层优势不应直接宣传为完整 Gate–Router–Expert 优势，必须用同一 Cascade
   合同重算；
3. CE-Recon selected-K 是当前 Cascade 记录中的强候选，但不同数据集存在 Known
   macro-F1 与 OOS F1 的权衡；
4. 本阶段没有产生可与 MOGB、ADB、DA-ADB、DCLOOS 统一排名的结果，也不能支持
   “已经达到 SOTA”；
5. 该桥接分析的价值是把“表示层收益”和“级联传播收益”分开，避免把不同评价层
   的数字混在一张表中。

## 7. 下一步

先保持现有模型、数据和历史结果冻结。外部 baseline 只有在独立 runtime 通过
`import torch`、BERT forward 和最小 CPU/GPU smoke 后，才从 StackOverflow/KIR=.50/
seed=42 单格开始。随后应使用同一 Cascade 评价器和明确监督标签（Known-only、
pseudo-OOS、external-OOS）建立可比表；在此之前，不扩大新的训练矩阵，也不把本报告
的桥接结果写成外部 SOTA 排名。
