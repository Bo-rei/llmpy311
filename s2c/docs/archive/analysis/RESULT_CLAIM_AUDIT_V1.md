# 当前实验结论审计（V1）

更新时间：2026-08-10  
活动协议：`protocol_v2_textoir_v1`

本文件是实验阶段的结论口径审计。它不新增实验、不修改任何结果，只规定已有证据能支持什么、不能支持什么。

## 1. 当前“我的方法”是哪一行

| 名称 | 实际对象 | 证据 | 当前状态 |
|---|---|---|---|
| `S2C-Trainable-K1` | Known-only 训练 MiniLM 最后两层+projection，K=1 Gate | `cross_protocol_tradeoff_v1`，315 行、5 seeds | 当前 protocol_v2 最强自有 Gate |
| 历史 `Ours` | Frozen MiniLM Gate + Router/Expert 完整 Cascade | `fulltex.tex`、72 行历史表 | 旧合同结果，不是当前 Trainable-K1 |
| RC-AMBL / joint-adaptive | 训练或规则参与式多中心候选 | adaptive/joint artifact | 候选分裂均安全回退，不能作为当前最佳方法 |

后续没有特殊说明时，“当前自有方法”只指 `S2C-Trainable-K1`。

## 2. 可以直接写出的结论

| 结论 | 证据 | 允许的准确表述 |
|---|---|---|
| Trainable-K1 是当前 fair Gate 矩阵最强自有结果 | 3 数据集×3 KIR×5 seeds；OOS F1 8/9 第一、F1-All 9/9 第一 | “在当前 Known-only MiniLM Gate 合同下，Trainable-K1 最稳定、最平衡。” |
| Trainable-K1 优于 MOGB-MiniLM-Fair | 45 个相同 registry/sample_id 配对单元；错误预算、KIR 趋势、逐样本转移均已审计 | “在相同轻量表示和 TEXTOIR split 下，Trainable-K1 优于 MOGB-Fair 组件。” |
| 优势主要来自 Known 覆盖和分数排序 | 45 行转移分解零不一致；Known 恢复、F1-All、工作点图 | “MOGB-Fair 较保守，Trainable-K1 恢复大量 Known，同时付出有限 OOS 拒识代价。” |
| 固定多中心在 StackOverflow 有 union 过覆盖 | E2/E3、KIR/λ、逐样本 intent 风险和 K=2 error budget | “固定 K>1 的主要风险是多个接受区域并集增加 OOS false acceptance。” |
| Trainable-K1 的优势随 KIR 增大 | `TRAINABLE_MOGB_KIR_TREND_V1.md` | “当前差距并非单个 KIR 的偶然现象。” |

## 3. 不能直接写出的结论

| 禁止表述 | 原因 | 应替换为 |
|---|---|---|
| “S2C 已经超过完整 MOGB” | 当前比较的是 Frozen MiniLM MOGB-Fair；官方 BERT 单格也未严格复现论文 | “S2C 已超过同表示 MOGB-Fair 组件；完整论文 MOGB 尚无同合同结论。” |
| “S2C 已经超过 DCLOOS” | DCLOOS reduced 使用 pseudo-OOS+外部 SQuAD，KIR/seed/监督不同 | “DCLOOS reduced 结果仅作更强监督条件下的外部参照。” |
| “S2C 已超过 ADB/DA-ADB” | ADB 是 BERT/TextOIR 兼容单元；DA-ADB 目前只有一个有效隔离 CUDA 单格 | “ADB/DA-ADB 仅可作外部合同参照，不构成同条件胜负；DA-ADB 新单格 OOS F1=70.82%，旧兼容单格 90.90% 尚未完成合同对齐。” |
| “当前 Trainable-K1 复现了论文 Ours” | 历史 `fulltex.tex` Ours 是完整 Cascade | “当前 Trainable-K1 是新的 Gate-only 结果；历史 Ours 是旧 Cascade 合同。” |
| “MOGB 算法无效” | 作者数据、Known 列表、旧环境和完整随机状态未恢复 | “本地官方逻辑在可用材料下未复现论文工作点，差距来源仍有合同因素。” |

## 4. 已有数字的合同分层

| 层级 | 方法 | StackOverflow/KIR=.50 OOS F1 | 是否可进当前 Known-only 主表 |
|---|---|---:|---|
| 当前 fair Gate | Trainable-K1 | 87.67±1.66% | 是 |
| 当前 fair 组件 | MOGB-Fair | 72.92±0.62% | 是，作为组件对照 |
| 当前 fair 组件 | MOGB partition + S2C boundary | 79.25±1.49% | 是，作为组件对照 |
| BERT 外部参照 | ADB | 87.36±1.61% | 否，合同不同 |
| BERT 官方逻辑兼容单格 | MOGB official local | F1-All 68.35% | 否，未复现论文 |
| BERT+额外未知监督 | DCLOOS reduced | OOS F1 87.05% | 否，监督/KIR/seed不同 |
| BERT 外部方法 | DA-ADB | valid_external_cell_pending_replication | 否 |

## 5. 当前可视化证据链

1. 性能总览：`figures/archive/analysis/experiment_comparison_overview_v2/current_fair_matrix_mean.png`
2. 历史 Cascade 与当前 Gate：`figures/archive/analysis/cross_contract_gap_v1/current_vs_historical.svg`
3. Trainable 与 MOGB 错误预算：`figures/archive/analysis/trainable_mogb_error_budget_v1/error_budget_decomposition.svg`
4. KIR 趋势：`figures/archive/analysis/trainable_mogb_kir_trend_v1/kir_trend.svg`
5. StackOverflow 逐样本错误：`figures/archive/analysis/stackoverflow_error_attribution_v2/`
6. MOGB 论文差距：`figures/archive/analysis/mogb_reproduction_gap_analysis_v2/`

## 6. 当前实验缺口

- 外部 ADB/DA-ADB/DCLOOS 尚未在同一 runtime、同一 split、同一 seed 数下形成统一主表；
- 官方 BERT MOGB 的作者数据/完整环境未恢复；
- 当前 Trainable-K1 尚未与历史完整 Cascade 在同一 protocol_v2 下重建全三数据集主表。

因此当前阶段仍然是实验和机制分析阶段，不能把结果写成跨合同 SOTA。

## 7. 唯一下一步

先恢复可回收的独立 PyTorch runtime，完成 ADB/DA-ADB/DCLOOS 的最小有效单格；通过数据、监督、seed 和评价审计后，再扩展多 seed。不要重复 E2/E3、固定 K/KIR 或 MOGB-Fair 已完成矩阵。
