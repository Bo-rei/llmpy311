# 跨方法 OOS 比较与机制汇报

更新时间：2026-09-09  
核心问题：在 OOS F1 优先的前提下，S2C 相对 TextOIR 方法、MOGB、ADB/DA-ADB 的差异来自哪里？  
图形语言：沿用 `RECENT_MECHANISM_ANALYSIS_PRESENTATION_V1.md` Fig.1 的蓝色圆点、红色叉号、黄色星标、灰色引导线和低复杂度图例。

> 本汇报把不同 backbone、监督条件和系统层级分开。`banking77` 表示标准 Banking77；历史论文主线的实际数据键 `banking77_oos` 只作为历史参考，不与标准 Banking77 合并。

## 一页结论

1. **在同一 `protocol_v2_textoir_v1`、MiniLM、Known-only、5-seed fair 矩阵中，S2C Trainable K=1 是最平衡的 OOS 工作点。** 它不是所有指标都最高，但同时保持较高 Known coverage 和较高 OOS F1。
2. **相对 MOGB-Fair component，S2C 的主要优势是 coverage–rejection 平衡。** MOGB-MiniLM 的 false acceptance 很低，但 Known Recall 只有约 27–33%；S2C 恢复大量 Known，同时保持更高 OOS F1/F1-All。
3. **TextOIR native 方法的差异明显依赖 KIR。** 在 BERT/TextOIR compatibility 线上，ADB 通常优于 DOC/MSP，但这条线不能直接与 MiniLM fair 行做无标注排名。
4. **ADB 是最接近的外部参照，但不是同 backbone。** 当前 StackOverflow/KIR=.50 下，S2C 与 ADB 的 OOS F1 接近；S2C 的 Known Recall 更高，ADB 略保守，差异不能归因到单一算法组件。
5. **DA-ADB 当前外部单元明显落在较差工作点。** OOS F1 为 `72.48±6.24%`，false acceptance 为 `29.07±11.19%`；它不是“只拒绝更多”造成的简单差异。
6. **DCLOOS 目前没有同协议可排名结果。** reduced 单元的 `87.05%` 只能作为 BERT+pseudo-OOS+外部 SQuAD 监督条件的参考点，不能写成超过或落后 S2C 的公平结论。

## 1. 比较合同先分层

| 图层 | 方法 | 数据 / backbone | 监督 | 证据用途 |
|---|---|---|---|---|
| same-protocol fair | S2C Trainable/Frozen、fixed K、MOGB-Fair component | `protocol_v2_textoir_v1`、MiniLM | Known-only | 主比较：表示、中心、边界和错误预算 |
| TextOIR native compatibility | MSP、DOC、ADB | TextOIR 原生 `oos/banking/stackoverflow`、BERT | Known-only | TextOIR 方法的独立 KIR 曲线 |
| external backbone | ADB、DA-ADB | 当前 StackOverflow compatibility、BERT/TextOIR | Known-only | 同数据参照；不进入 MiniLM fair 排名 |
| different supervision | DCLOOS reduced / smoke | BERT、pseudo-OOS、外部 SQuAD | 含外部 OOS 监督 | 监督条件与执行状态参考 |
| incomplete | KNNCL、DCLOOS official default | 无 final metrics | — | 只记录缺口，不补中间指标 |

## 2. Same-protocol fair：S2C 与 MOGB 的 OOS 工作点

以下为标准 `banking77`、CLINC150、StackOverflow，KIR=.50，5 个 seed 的 fair 均值。MOGB 行是 MiniLM fair component，不是完整官方 BERT MOGB。

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False acceptance |
|---|---|---:|---:|---:|---:|
| CLINC150 | **S2C Trainable K=1** | **90.44** | **81.82** | 74.44 | 3.69 |
| CLINC150 | Frozen single | 88.94 | 80.27 | 78.76 | 8.82 |
| CLINC150 | MOGB-MiniLM | 81.32 | 44.95 | 31.57 | **0.90** |
| StackOverflow | **S2C Trainable K=1** | **87.67** | **86.55** | 83.89 | 9.34 |
| StackOverflow | Frozen single | 76.55 | 79.98 | 87.15 | 29.71 |
| StackOverflow | MOGB-MiniLM | 72.92 | 43.30 | 27.09 | **0.79** |
| Banking77 | **S2C Trainable K=1** | **83.56** | **81.67** | 82.21 | 15.74 |
| Banking77 | Frozen single | 72.43 | 74.51 | 85.01 | 34.85 |
| Banking77 | MOGB-MiniLM | 74.99 | 48.60 | 33.41 | **1.09** |

![Fig. 1：fair OOS F1–coverage 工作点](../../figures/cross_method_oos_mechanism/fair_oos_boundary_tradeoff.png)

### Fig. 1 的强解释

横轴是 Known Recall，纵轴是 OOS F1；黄色星标是 S2C Trainable，红色叉号是 MOGB-MiniLM。

- MOGB-MiniLM 的红叉位于低 coverage 区域：它以拒绝大量 Known 为代价换来低 false acceptance。
- S2C 星标位于更高 OOS F1 和更高 coverage 区域，说明优势不是“全部拒绝”带来的。
- CLINC/StackOverflow 的 S2C 星标同时高于 Frozen single；Banking77 的提升也存在，但绝对 OOS F1 受数据任务难度影响。

![Fig. 2：OOS precision–recall 分解](../../figures/cross_method_oos_mechanism/fair_oos_precision_recall.png)

### Fig. 2 的强解释

这张图把 OOS F1 拆成 OOS precision 和 OOS recall。等 F1 虚线用于判断点位是“拒绝太少”还是“拒绝过度”：

- MOGB 位于高 OOS recall、低 OOS precision 或低 Known coverage 的保守区域；
- S2C 位于更平衡的 iso-F1 区域；
- 这说明比较方法时不能只看 false acceptance，必须同时检查 Known coverage 和 OOS F1。

## 2.1 Official MOGB：直接证据与可比性边界

为回应“最接近的 MOGB 没有进入实验”的意见，现将已有官方代码审计/兼容运行单独列出。下表保留 MOGB 原生结果字段，不强行映射为当前统一 evaluator 的 OOS F1、F1-All 或 Known F1。`official-fixed local` 两行来自 normalized `mode_manifest`（Known=Known Recall、Open=F1-U、F1-score=F1-All）；五 seed compatibility 和 smoke 行保留原始 `results.csv` 字段。

| 运行 | 数据 / KIR | seed | reported Known | reported Open | reported F1-score | Accuracy | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| official-fixed local | StackOverflow / .50 | 0 | 51.53 | 79.97 | 68.35 | 75.17 | 本地 exact-compatible，非严格官方复现 |
| corrected-loss diagnostic | StackOverflow / .50 | 0 | 59.20 | 80.96 | 72.64 | 77.25 | 改变公开 loss 公式的诊断，不是官方结果 |
| official-fixed local | 标准 Banking77 / .75 | 0 | 43.71 | 53.10 | 59.16 | 57.08 | 本地 exact-compatible，非严格官方复现 |
| official-logic compatibility | StackOverflow / .50 | 13/42/87/100/123 | 37.59±1.73 | 72.05±0.39 | 40.72±1.61 | 61.49±0.69 | 五 seed 非严格兼容运行 |
| official-logic compatibility | 标准 Banking77 / .50 | 13/42/87/100/123 | 17.97±2.96 | 69.28±0.56 | 19.28±2.89 | 55.49±0.96 | 五 seed 非严格兼容运行 |

机器可读源表是 [`official_mogb_status.csv`](../../results/analysis/cross_method_oos_mechanism/official_mogb_status.csv)。这些行不能直接与上面的 MiniLM fair 行合并排名，原因是：MOGB 使用 BERT 和原生数据/类别采样合同；exact-compatible 运行使用现代兼容层；corrected-loss 行改变了官方 loss；五 seed compatibility 行也没有恢复作者旧环境。因此，这里已经提供了 MOGB 的直接实验状态和数值参照，但仍不声称“严格复现完整 MOGB”或“在统一合同下超过 MOGB”。

## 3. TextOIR native 方法：MSP / DOC / ADB

![Fig. 3：TextOIR native compatibility 的 KIR 稳定性](../../figures/cross_method_oos_mechanism/textoir_native_oos_kir_stability.png)

该图使用 TextOIR 原生 `oos/banking/stackoverflow` 数据键、BERT、3 seeds。KIR=.50 的 OOS F1 为：

| TextOIR 数据键 | MSP | DOC | ADB |
|---|---:|---:|---:|
| `oos` / CLINC150 | 61.68±1.88 | 88.36±0.27 | **89.22±0.93** |
| `banking` | 44.67±1.74 | 72.57±2.03 | **78.65±0.14** |
| `stackoverflow` | 33.49±4.38 | 71.67±4.14 | **86.98±1.91** |

这里的 `banking` 是 TextOIR native source key，和历史论文主线实际使用的 `banking77_oos` 不是同一个报告键。该图的作用是说明 TextOIR 方法自身的 KIR 变化，不是把 BERT 结果冒充成 MiniLM fair 结果。

## 4. S2C 与 ADB：最接近的外部工作点参照

![Fig. 4：S2C 与 ADB 的 OOS frontier](../../figures/cross_method_oos_mechanism/s2c_vs_adb_oos_frontier.png)

当前协议、BERT/TextOIR ADB 与 MiniLM S2C 的同 dataset/KIR 配对差值如下：

| 数据集 | KIR=.25 | KIR=.50 | KIR=.75 | 主要机制 |
|---|---:|---:|---:|---|
| CLINC150 | +3.57 pp | +1.05 pp | −1.91 pp | S2C OOS recall 更高，但高 KIR precision 下降 |
| Banking77 | +8.85 pp | +8.60 pp | +2.39 pp | S2C 减少 OOS 误接收，同时 Known rejection 增加 |
| StackOverflow | +2.34 pp | +0.46 pp | +2.60 pp | S2C coverage 更高，差异较依赖 KIR |

StackOverflow/KIR=.50 的三 seed 外部错误预算为：

| 方法 | OOS F1 | Known Recall | OOS false acceptance | 合同 |
|---|---:|---:|---:|---|
| S2C Trainable K=1 | 88.21±1.72 | 83.68±0.28 | 8.18±3.36 | MiniLM fair |
| ADB | 约 87.21±1.17 | 81.37±1.33 | 8.27±1.80 | BERT/TextOIR external |
| DA-ADB | 72.48±6.24 | 75.97±4.10 | 29.07±11.19 | BERT/TextOIR external |

因此，当前可以说 S2C 在这条外部参照上取得了更高 coverage 和接近/略高的 OOS F1；不能把差异归因成“MiniLM 表示单独击败 BERT”，因为训练目标、边界和兼容运行也同时变化。

## 5. MOGB 机制：差距来自哪里？

![Fig. 5：MOGB component bridge](../../figures/cross_method_oos_mechanism/mogb_component_bridge.png)

箭头依次表示：`MOGB-MiniLM → MOGB partition + S2C boundary → S2C Trainable`。这是同协议 component swap 的描述性桥接，不是因果干预。

- 只替换边界/分区后，OOS F1 有改善，但仍明显低于 S2C Trainable；
- 这说明 MOGB 的差距不是单一边界阈值造成，表示适配和边界形式都贡献了结果；
- 不能把该图表述为“完整官方 MOGB 被 S2C 超过”，它比较的是 MiniLM fair component。

![Fig. 6：MOGB 与 S2C 的错误预算](../../figures/cross_method_oos_mechanism/oos_error_budget_frontier.png)

横轴是 OOS false acceptance，纵轴是 Known false rejection，左下角更好。MOGB 的红叉普遍位于高 Known rejection 区域；S2C 星标向左下移动，代表在可接受的 OOS 误接收代价下恢复 Known coverage。

![Fig. 7：MOGB 粒球局部风险](../../figures/cross_method_oos_mechanism/mogb_ball_risk_surface.png)

每个点是一个 MOGB ball，点大小表示 radius。风险不是均匀分布的：少数 OOS-contaminated balls 贡献了主要误接收，而大量相邻 ball 没有观测到 OOS 污染。这支持“局部 ball 风险”解释，不支持用一个全局半径参数概括全部问题。

![Fig. 8：同一样本的 MOGB→S2C 状态转移](../../figures/cross_method_oos_mechanism/same_sample_mogb_s2c_transitions.png)

该图只使用 StackOverflow/KIR=.50/seed=42 的同样本状态：

- Known rescue：MOGB rejected、S2C correct；
- OOS fixed：MOGB accepted、S2C rejected；
- 同时保留 Known coverage cost 和 OOS regression，避免把 S2C 说成无代价改进。

在该单元中，S2C 从 MOGB 状态中恢复约 `43.5%–54.3%` 的 Known 样本，并修正一部分 MOGB false acceptance；不同数据集的转移比例不同，因此机制结论要写成条件性结论。

## 6. DCLOOS：必须单独看监督条件和证据状态

![Fig. 9：跨方法证据状态图](../../figures/cross_method_oos_mechanism/cross_method_evidence_status.png)

当前唯一有数值的 DCLOOS reduced 单元为：

```text
OOS F1       87.05%
F1-All       90.26%
Known Recall 92.14%
Accuracy     88.68%
KIR=.75, seed=888
BERT + pseudo-OOS + external SQuAD OOS
```

它不是同一 Known-only 测试合同，不能放进 Fig.1 的 fair ranking。官方 default budget、fixed-registry 和完整 StackOverflow 单元仍没有可排名的 final metrics；中间 `predictions.npz` 不用于补表。

## 7. OOS 低于论文：参数确实有影响，但不是唯一原因

### 7.1 参数工作点已经能带来可测提升

![Fig. 10：参数工作点与 Known/Accuracy guard](../../figures/historical_archive_tuned_full_pipeline/pipeline_workpoint_search_frontier.png)

该图使用标准 archive `banking77`、CLINC150、StackOverflow 的 Partial MiniLM Gate。候选只保留验证集上 Known F1 和 Accuracy 不低于基线 1 个百分点的工作点；黄色星标是验证集选择结果，蓝色圆点是未调 `K=1, λ=1, threshold=1` 基线。

测试集的 full pipeline 确认结果为：

| 数据集 | 未调 Partial OOS F1 | tuned OOS F1 | tuned Known F1 | tuned Accuracy | tuned 配置 |
|---|---:|---:|---:|---:|---|
| CLINC150 | 89.56 | 91.36 | 85.94 | 87.35 | K=3, λ=2, normalized union |
| StackOverflow | 91.49 | **91.85** | 86.36 | 88.73 | K=1, λ=2, threshold=.85 |
| 标准 Banking77 | 82.99 | 85.62 | 81.12 | 82.44 | K=2, λ=1, threshold=.95 |

这说明参数/边界工作点确实是差距的一部分，尤其标准 Banking77 的 OOS F1 从 `82.99%` 提高到 `85.62%`。但标准 Banking77 与论文主表实际历史 `banking77_oos` 不是同一任务，不能用 `85.62%` 直接判定论文复现失败。

### 7.2 论文比较必须按任务分开

| 比较对象 | OOS F1 | 解释 |
|---|---:|---|
| 论文 `fulltex.tex` CLINC150 Ours | 91.96 | 历史完整 Cascade reference |
| 当前 archive tuned CLINC150 | 91.36 | 标准 archive、seed42、tuned full pipeline，差 −0.60 pp |
| 论文 `fulltex.tex` StackOverflow Ours | 89.71 | 历史完整 Cascade reference |
| 当前 archive tuned StackOverflow | 91.85 | 标准 archive、seed42、tuned full pipeline，高 +2.14 pp |
| 论文表头 Banking77 Ours | 88.23 | 实际历史主线数据键为 `banking77_oos` |
| 当前 archive tuned Banking77 | 85.62 | 标准 `banking77`，不能严格配对 |

历史 H1 `banking77_oos` 的 Trainable K=1 Gate 三 seed 均值为 `88.47%`，本身已经略高于论文 `88.23%` reference。当前真正未解决的是严格 H0 完整 Cascade 的逐字恢复，而不是 Trainable Gate 在 `banking77_oos` 上必然低于论文。

## 8. 最终回答

### 已经有证据支持的结论

- S2C Trainable K=1 在同协议 MiniLM fair 矩阵中是当前最平衡的 OOS 工作点；
- MOGB-Fair 的主要可见代价是 Known under-coverage，固定多中心的风险是 OOS acceptance-region expansion；
- TextOIR MSP/DOC/ADB 的 native compatibility 结果已闭合，但它们是 BERT/TextOIR 独立合同；
- 参数工作点能在 Known F1/Accuracy guard 下提升 archive full-pipeline OOS F1；
- Fig.1 旧二维图的“很多 OOS 在边界内”主要是 PCA 投影误读，3D 图已用真实 384 维 decision marker 修正。

### 仍然不能声称的结论

- 不能说 S2C 已经在同一合同下超过完整 MOGB 或 DCLOOS；
- 不能用 DCLOOS reduced 的 87.05% 做公平排名；
- 不能把标准 `banking77` 的 archive tuned 结果回填为论文历史 `banking77_oos`；
- 不能只凭 PCA 三维球面判断样本是否被 Gate 接受，最终判决仍以原始高维 score 为准。

## 9. 机器可复核入口

- 图与派生表：[cross_method_oos_mechanism](../../results/analysis/cross_method_oos_mechanism/)
- 图清单：[MANIFEST.json](../../results/analysis/cross_method_oos_mechanism/MANIFEST.json)
- 图源：[build_cross_method_oos_mechanism.py](../../tools/analysis/build_cross_method_oos_mechanism.py)
- 3D Fig.1 源：[build_historical_oos_visual_explanation.py](../../tools/analysis/build_historical_oos_visual_explanation.py)
- 3D Fig.1 真实判决摘要：[local_boundary_geometry_3d_summary.csv](../../results/analysis/historical_oos_visual_explanation/local_boundary_geometry_3d_summary.csv)
- 参数工作点源：[candidate_comparison.csv](../../results/analysis/historical_archive_tuned_full_pipeline/candidate_comparison.csv)

所有图均由已有结果和已落盘的 OOS 状态/几何摘要生成；没有新增训练、测试集调参、原始文本导出或 checkpoint 覆盖。
