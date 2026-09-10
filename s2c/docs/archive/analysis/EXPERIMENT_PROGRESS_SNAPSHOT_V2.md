# s2c 实验进展快照 V2

更新时间：2026-08-10  
活动协议：`protocol_v2_textoir_v1`

本文是当前实验阶段的短入口。目标是说明已经做了什么、当前哪一行是自有主方法、哪些结果可以直接比较、哪些只能作为外部合同参照，以及下一步仍缺什么。本文不提出新方法，不重跑历史实验，也不把不同监督合同的数字合成一个 SOTA 排名。

## 一、当前到底在比较什么

当前自有主方法是 `S2C-Trainable-K1`：Known-only 训练 MiniLM 最后两层和 residual projection，随后使用 K=1、对角 Mahalanobis Gate。它不是固定 K>1，也不是 `fulltex.tex` 中旧协议的完整 Gate–Router–Expert Cascade。

同协议 fair 组件包括 Frozen K=1、Frozen/Random K=2、MOGB-MiniLM 及其分区/边界替换组合。MOGB 官方 BERT、ADB、DA-ADB、DCLOOS 属于外部兼容或不同监督合同，单独列出。

## 二、已完成实验规模

| 阶段 | 规模 | 状态 | 主要结论 |
|---|---:|---|---|
| E0 | 3 canonical、165 registry、165 views、990 exports | 完成 | 本地数据和协议独立 |
| E1 | 36/36 smoke | 完成 | 三数据集 Gate 链路可运行 |
| E2 | 1650/1650 | 完成 | 不存在跨数据集统一最优 K |
| E3 | 720 控制单元、180 诊断组 | 完成 | 稳定聚类不等于有效 OOS 边界 |
| MiniLM 表示 | Frozen、CE、SupCon、CE-Recon、Trainable | 完成 | K=1 表示适配最稳定 |
| MOGB-Fair | 3 数据集×3 KIR×5 seed | 完成 | 组件工作点保守，Known 覆盖不足 |
| ADB 外部参照 | 45 单元 | 完成 | BERT/TextOIR 合同下可运行 |
| DA-ADB 当前协议 | StackOverflow 3 seed | 完成 | 数值有效，但仍是外部 BERT 合同 |
| DCLOOS | reduced 参考、smoke、固定 Registry GPU 尝试 | 未闭合 | 当前 Registry 单格没有最终 metrics |

## 三、当前同协议最可靠结果

3 数据集×3 KIR×5 seed 的 9 个 fair 单元均值：

| 方法 | OOS F1 | F1-All | Known Recall | False Acceptance |
|---|---:|---:|---:|---:|
| S2C-Trainable-K1 | 85.75% | 83.36% | 80.22% | 8.73% |
| Frozen K=1 | 78.64% | 78.30% | 83.50% | 23.01% |
| Random K=2 | 78.39% | 78.51% | 84.25% | 23.86% |
| Frozen K=2 | 75.51% | 76.79% | 81.44% | 25.81% |
| MOGB partition + S2C boundary | 77.74% | 63.92% | 52.21% | 2.65% |
| MOGB-MiniLM | 73.39% | 46.26% | 31.21% | 0.90% |

Trainable-K1 在 OOS F1 上 9 个单元中 8 个第一，在 F1-All 上 9 个单元全部第一。它的优势不是单纯把更多样本拒为 OOS：相对 MOGB-Fair，跨 KIR 的错误预算显示主要收益来自恢复 MOGB 拒绝的 Known 样本；因此这是覆盖—拒识工作点和表示排序的优势。

## 四、StackOverflow 的关键机制证据

KIR=.50、五 seed：

| 方法 | OOS F1 | F1-All | Known Recall | False Acceptance |
|---|---:|---:|---:|---:|
| S2C-Trainable-K1 | 87.67% | 86.55% | 83.89% | 9.34% |
| Frozen K=1 | 76.55% | 79.98% | 87.15% | 29.71% |
| Frozen K=2 | 63.53% | 72.76% | 86.89% | 47.17% |
| MOGB-MiniLM | 72.92% | 43.30% | 27.09% | 0.79% |

固定 K=2 的主要失败不是 Known Recall 崩溃，而是新增多个接受区域后 false acceptance 急剧增加。MOGB-MiniLM 的低 false acceptance 也不能单独解释为更好，因为它同时拒绝大量 Known 样本。训练参与式 adaptive-K pilot 已真正执行，但候选分裂均被 Known calibration 安全门拒绝，最终 `K_y=1`。

## 五、外部方法和 MOGB 的真实状态

| 方法 | 当前结果层 | 结论 |
|---|---|---|
| MOGB-MiniLM-Fair | 同协议、冻结 MiniLM、5 seed | 可用于组件归因；S2C 在 45 个配对中 OOS F1 胜 44、F1-All 胜 45 |
| MOGB 官方 BERT | StackOverflow/Banking 单格现代兼容运行 | 未达到论文公开数字；不能称严格论文复现 |
| ADB | BERT/TextOIR、StackOverflow 3 seed | OOS F1 87.36±1.61%，与 S2C 接近但不是同骨干合同 |
| DA-ADB | 当前协议 StackOverflow 3 seed | OOS F1 72.48±6.24%，仍是外部 BERT 合同 |
| DCLOOS reduced | BERT+pseudo-OOS+外部 SQuAD，KIR=.75/seed=888 | OOS F1 87.05%，只能作不同监督参考 |
| DCLOOS 固定 Registry GPU | StackOverflow/KIR=.50/seed=42 | 实际使用 RTX 5070，但超出 30 分钟预算，无最终指标 |

MOGB 论文差距已拆为 loss 距离归一化、平均半径造成 Known 过拒、selected-ball 缺类、数据/Known-list、旧运行环境和随机状态等因素。当前证据支持“现有材料下未复现论文”，不支持“算法已经被证明无效”。

## 六、现有可视化证据链

- 性能热力图和 KIR 曲线：`figures/archive/analysis/experiment_analysis_master_v1/`
- S2C–MOGB 配对、Pareto 和工作点：`figures/s2c_vs_mogb_mechanism_dashboard_v1/`、`figures/archive/analysis/cross_kir_pareto_frontier_v1/`
- Known/OOS 错误预算：`figures/archive/analysis/cross_kir_transition_attribution_v1/`、`figures/archive/analysis/trainable_mogb_error_budget_v1/`
- StackOverflow intent 风险和多中心过覆盖：`figures/archive/analysis/cross_dataset_intent_risk_visuals_v1/`、`figures/archive/analysis/stackoverflow_intent_diagnostic_v1/`
- 表示和 detector 机制：`figures/archive/analysis/representation_geometry_visuals_v1/`、`figures/archive/analysis/detector_mechanism_v1/`
- 外部合同分层：`figures/comparison_atlas_v2/`、`figures/archive/analysis/baseline_contract_visuals_v1/`

ADB 的完整跨数据集/KIR/五 seed 误差预算见 `ADB_TRAINABLE_KIR_ANALYSIS_V1.md`，图表位于
`figures/archive/analysis/trainable_vs_adb_kir_v1/`；该包共享 split/KIR/seed，但明确保留 BERT/TextOIR 与 MiniLM 合同差异。
StackOverflow/KIR=.50 的 ADB 三 seed 逐样本错误预算见 `TRAINABLE_VS_ADB_ERROR_BUDGET_V1.md`，图表位于
`figures/archive/analysis/trainable_vs_adb_error_budget_v1/`；18,000 条记录完成文本顺序和 sample_id 对齐，结果显示
Trainable 少 2.90pp Known 条件误拒、但多 0.66pp OOS 条件误接收，属于工作点交换而非无条件胜出。

跨数据集的同类错误预算已扩展到 45/45 个 `dataset×KIR×seed` 单元（共 221,700 条协议测试记录），
详见 `TRAINABLE_VS_ADB_CROSS_DATASET_ERROR_BUDGET_V1.md` 与
`figures/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/`。CLINC150/Banking77 的 Trainable 主要表现为
更低 OOS 误接收但更高 Known 误拒；StackOverflow 则是更少 Known 误拒、但中高 KIR 的 OOS 误接收略高。
因此当前最可靠的机制表述是“数据集相关的覆盖--拒识重分配”，不是 Trainable 在所有工作点无条件优于 ADB。

对应的统一机制表已生成：`docs/archive/analysis/TRAINABLE_VS_ADB_MECHANISM_SUMMARY_V1.md`。它显示 CLINC150/Banking77
更接近“保守拒识”（Known 误拒增加、OOS 误接收下降），StackOverflow 则是“覆盖恢复，但中高 KIR
伴随轻微误接收权衡”。

OOS precision/recall 的独立分解见 `docs/archive/analysis/TRAINABLE_VS_ADB_OOS_DECOMPOSITION_V1.md`：CLINC150/Banking77
的 OOS F1 增益主要来自 recall 增加而不是 precision 增加；StackOverflow 在 KIR=.25 同时改善 precision
和 recall，在中高 KIR 则以 precision 增益抵消轻微 recall 下降。
意图级归因见 `docs/archive/analysis/TRAINABLE_VS_ADB_INTENT_ERROR_V1.md`：它把 Known 误拒和 OOS 误接收拆成真实/预测 intent 两种视角，显示差异是分散在多个 intent，而不是由单一类别完全造成。

## 七、当前结论与下一步

当前能严谨说的是：`S2C-Trainable-K1` 是当前统一 Known-only MiniLM Gate 矩阵中最强且最平衡的自有候选；它的优势主要来自表示适配和较好的 Known/OOS 工作点，而不是已经证明固定多中心或自适应多中心普遍有效。

当前不能说的是：已经超过完整 MOGB、ADB、DA-ADB、DCLOOS 或 `fulltex.tex` 历史 Cascade，也不能把不同 BERT、伪 OOS、外部 OOS 和旧数据合同混成一张 SOTA 排名表。

下一步只保留一个主方向：继续补齐同合同外部 baseline 的可验证单格，并用已有 fair 矩阵、错误预算和图表解释“为什么 Trainable-K1 在当前合同下更好”；不再重复 E2/E3、扩大 K 或盲目增加新的 loss。
