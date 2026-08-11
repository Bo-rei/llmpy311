# 方法对比地图 V1

更新时间：2026-08-09
活动协议：`protocol_v2_textoir_v1`

## 先给结论

目前图表中的“我的方法”不是一个单一模型，而是三类不同实验对象：

1. **当前最强、最稳定的自有候选：`S2C-Trainable-K1`**
2. **历史固定多中心组件：`S2C-Frozen-K2`、`S2C-Random-K2`**
3. **尚未成功的自适应多中心探索：`RC-AMBL`、`Joint-Adaptive`**

因此，当前可以说的是：

> `S2C-Trainable-K1` 在同一 Known-only、五 seed、三数据集的 Gate 矩阵中优于冻结/固定多中心和 MOGB 的 MiniLM 组件；不能说“自适应多中心已经优于 MOGB”，因为当前自适应多中心 pilot 没有安全地产生 `K_y>1`，完整 MOGB/ADB/DA-ADB/DCLOOS 也没有形成同协议主表。

## TextOIR baseline inventory 校正（2026-08-11）

独立 `/home/bo/bo01/llmpy311/textoir/open_intent_detection/` 中实际登记的检测路线包括
MSP、SEG、OpenMax、LOF、DOC、DeepUnk、`(K+1)-way`、MDF、ARPL、KNNCL-last/all、ADB、DA-ADB、
DA-ADB-llama 和 EliDecide。StackOverflow、Banking77 与当前 OOS 数据源和 TextOIR 对应文件同源；
因此这些方法的差别应标记为 backbone/训练/阈值/评估合同差异，而不是数据集差异。

当前状态分层为：ADB 已完成 `45/45` 个外部 BERT 单元；DA-ADB 已完成 StackOverflow/KIR=.50 的
`3` 个外部单元并待复现；KNNCL、OpenMax、DOC、DeepUnk、SEG、MDF、ARPL、`(K+1)-way` 和
DA-ADB-llama 尚未全部形成当前统一 final-metrics 合同，但已有方法代码或 published/legacy reference。
它们不能从主表中消失，也不能用中间 prediction 填补当前公平排名。

## A. 当前正式公平矩阵：真正可以直接比较的七行

来源：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`。
覆盖：3 数据集 × 3 KIR × 5 seeds，共 63 个 summary rows。
所有行共享当前 `protocol_v2_textoir_v1` 的 split、Known 列表、评价器和 Gate threshold 合同。

| 统一名称 | 代码/结果 method | 表示 | 分区/中心 | 边界 | 它回答的问题 |
|---|---|---|---|---|---|
| **S2C-Trainable-K1** | `trainable_k1` | Known-only 训练 MiniLM（最后两层 + projection） | 每个 intent 一个中心 | S2C 当前边界 | 训练表示能否改善安全的单中心工作点 |
| **S2C-Frozen-K1** | `single_centroid` | 冻结 MiniLM | 每个 intent 一个中心 | S2C 当前边界 | 冻结表示基线 |
| **S2C-Frozen-K2** | `fixed_k2` | 冻结 MiniLM | 每个 intent 固定 KMeans-2 | S2C 当前边界 | 固定多中心是否比单中心好 |
| **S2C-Random-K2** | `random_partition` | 冻结 MiniLM | 每个 intent 随机平衡二分 | S2C 当前边界 | KMeans 收益是否只是“增加中心” |
| **MOGB-Fair** | `mogb_minilm` | 冻结 MiniLM | MOGB 自适应粒球 | MOGB 欧氏距离 + 平均半径 | 在相同 MiniLM 下，MOGB 组件是否有效 |
| **MOGB-Partition/S2C-Boundary** | `mogb_partition_ours_boundary` | 冻结 MiniLM | MOGB 粒球分区 | S2C 对角 Mahalanobis + `μ+λσ` | 动态分区与边界规则分别贡献多少 |
| **S2C-Partition/MOGB-Boundary** | `ours_partition_mogb_boundary` | 冻结 MiniLM | S2C 固定分区 | MOGB 欧氏距离 + 平均半径 | 只替换 MOGB 边界是否有效 |

这里的 `MOGB-Fair` 不是作者 BERT 端到端 MOGB，而是“冻结 MiniLM 条件下的公平组件适配”。

## B. 当前自有方法到底哪一个最好

在 9 个 dataset × KIR 工作点的五 seed 均值中：

| 方法 | OOS F1 | F1-All | F1-K | Known Recall | False acceptance | False rejection |
|---|---:|---:|---:|---:|---:|---:|
| **S2C-Trainable-K1** | **85.75%** | **83.36%** | **83.14%** | 80.22% | 8.73% | 19.78% |
| S2C-Frozen-K1 | 78.64% | 78.30% | 78.17% | 83.50% | 23.01% | 16.50% |
| S2C-Random-K2 | 78.39% | 78.51% | 78.39% | 84.25% | 23.86% | 15.75% |
| S2C-Frozen-K2 | 75.51% | 76.79% | 76.79% | 81.44% | 25.81% | 18.56% |
| MOGB-Partition/S2C-Boundary | 77.74% | 63.92% | 63.04% | 52.21% | **2.65%** | 47.79% |
| S2C-Partition/MOGB-Boundary | 75.90% | 62.30% | 61.48% | 49.98% | 7.35% | 50.02% |
| MOGB-Fair | 73.39% | 46.26% | 44.57% | 31.21% | **0.90%** | 68.79% |

这张表的正确解释不是“Trainable 每项都最好”：MOGB 组件的 false acceptance 更低，但它拒绝了大量 Known 样本。`S2C-Trainable-K1` 的优势是覆盖—拒识更平衡，并且 F1-All/ F1-K 更高。

## C. StackOverflow/KIR=0.50 的直观对照

| 方法 | OOS F1 | F1-All | Known Recall | FA | FR |
|---|---:|---:|---:|---:|---:|
| **S2C-Trainable-K1** | **87.67%** | **86.55%** | 83.89% | 9.34% | 16.11% |
| S2C-Frozen-K1 | 76.55% | 79.98% | 87.15% | 29.71% | 12.85% |
| S2C-Frozen-K2 | 63.53% | 72.76% | 86.89% | 47.17% | 13.11% |
| S2C-Random-K2 | 75.88% | 79.80% | 87.64% | 30.98% | 12.36% |
| MOGB-Partition/S2C-Boundary | 79.25% | 63.34% | 50.39% | 1.86% | 49.61% |
| MOGB-Fair | 72.92% | 43.30% | 27.09% | 0.79% | 72.91% |

所以 StackOverflow 的结论是：

> 固定 K=2 的主要损失来自新增 OOS false acceptance；MOGB-Fair 的低 FA 则主要通过 Known false rejection 获得。当前最可靠的自有结果是单中心 Trainable MiniLM，而不是固定多中心。

## D. 自适应多中心 pilot 不是当前主结果

以下实验是真正尝试过的自适应多中心，但不能和上表的七行混称：

| 实验 | 是否训练参与 | 当前结果 | 状态 |
|---|---|---|---|
| RC-AMBL | 冻结表示 + Known-only 风险门 | 候选 split 被稳定性/安全门拒绝，最终 `K_y=1` | 负诊断 |
| Joint-Adaptive | MiniLM/projection/prototype 共同训练 | 候选 split 全部被 calibration 拒绝，最终 `K_y=1` | 负诊断 |
| Contract-repair Joint-Adaptive | 训练参与 + 父边界/负载/分离约束 | 仍全部回退 `K_y=1` | 负诊断 |

这说明“当前的自适应多中心方法”还没有形成一个有效的 `K_y>1` 方法；它们只能用于解释为什么 StackOverflow 不适合当前多中心合同。

## E. MOGB、ADB、DA-ADB、DCLOOS 属于另一层

| 方法 | 当前证据 | 是否进入七行公平矩阵 |
|---|---|---|
| MOGB 官方 BERT | StackOverflow/KIR=.50/seed=0 与 Banking/KIR=.75/seed=0 的官方逻辑兼容尝试未达到论文数字 | 否 |
| ADB | 当前已完成 45/45 个同源数据外部 BERT 单元；仍是不同 backbone/训练合同 | 否 |
| DA-ADB | 当前已完成 StackOverflow/KIR=.50/3 seed 外部单元；状态为 pending replication | 否 |
| DCLOOS | reduced 结果使用 pseudo-OOS + 外部 OOS，监督合同不同 | 否 |

这些结果只能作为“合同参照”或复现状态，不能放进当前七行主排名。特别是 ADB/DA-ADB 的旧单格数字高于 Trainable，并不等于已经完成公平比较；DCLOOS 也不是 Known-only 方法。

## F. 以后所有图表如何命名

- 说“**自有方法当前最好**”：指 `S2C-Trainable-K1`。
- 说“**固定多中心**”：指 `S2C-Frozen-K2` 或 `S2C-Random-K2`，不是 Trainable-K1。
- 说“**MOGB 公平组件**”：指 `MOGB-Fair` 或两个组件混合行，不是官方 BERT 复现。
- 说“**自适应多中心**”：只能指 RC-AMBL/Joint-Adaptive pilot，并且当前结论是未产生安全 `K_y>1`。
- 说“**SOTA/外部基线**”：必须单独标记 ADB、DA-ADB、DCLOOS 的监督条件和运行合同；当前不能从七行矩阵推出 SOTA。

## 权威证据入口

- 五 seed 主表：`results/analysis/experiment_analysis_master_v1/audited_summary.csv`
- 当前综合分析：`docs/analysis/EXPERIMENT_ANALYSIS_MASTER_V1.md`
- 机制可视化：`figures/mechanism_evidence_v2/`
- 旧兼容基线状态：`docs/analysis/BASELINE_EXECUTION_STATUS_V1.md`
- MOGB/外部基线中文分层说明：`docs/对比实验/MOGB_DCLOOS_对比结果报告.md`
