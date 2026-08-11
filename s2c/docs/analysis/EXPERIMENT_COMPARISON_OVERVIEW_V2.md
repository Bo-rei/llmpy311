# 当前实验对比总览 V2（中文、合同分层）

更新时间：2026-08-10
活动协议：`protocol_v2_textoir_v1`

本文是当前实验阶段的单一入口。它只整理已经落盘的实验和后验诊断，**不把不同数据、骨干、监督条件或运行合同的数字混成一个 SOTA 排名**。

逐格的历史 Cascade—当前 Trainable-K1 差距另见 `docs/analysis/CROSS_CONTRACT_GAP_V1.md`；本文件保留为方法地图和合同分层总览。

统一的逐样本合同、对齐审计、配对统计和机制图入口已收口到 [`UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md`](UNIFIED_COMPARISON_AND_MECHANISM_REPORT_V1.md)。

## 一句话结论

当前最强的自有结果是 `S2C-Trainable-K1`（Known-only 训练 MiniLM，单中心 Gate）；它在当前统一 Gate 矩阵中优于冻结/固定多中心和冻结 MiniLM 的 MOGB 组件，但还不能据此声称超过完整 MOGB、ADB、DA-ADB、DCLOOS 或历史完整 Cascade。

## 1. 目前到底比较哪一个“我的方法”

| 层级 | 当前对象 | 代码/结果标识 | 监督与骨干 | 是否可直接排名 |
|---|---|---|---|---|
| 历史论文主结果 | 完整 `Ours` Cascade | `fulltex.tex` / historical SOTA | 冻结 MiniLM Gate + SmolLM Router/Expert | 只在历史旧合同内 |
| 当前自有最佳 | **S2C-Trainable-K1** | `trainable_k1` | Known-only 训练 MiniLM 最后两层 + projection，K=1 | 可以与当前 fair Gate 行比较 |
| 当前固定多中心 | Frozen K=2、Random K=2 | `fixed_k2`、`random_partition` | 冻结 MiniLM，后处理分中心 | 可以与当前 fair Gate 行比较 |
| 当前 MOGB 组件 | MOGB-MiniLM 与组件桥 | `mogb_minilm` 等 | 冻结 MiniLM，动态粒球/边界替换 | 可以做组件比较，不能称完整 MOGB |
| 外部兼容基线 | ADB、DA-ADB | TextOIR BERT adapter | BERT/不同训练实现；ADB 已完成 45/45，DA-ADB 当前协议 3 seed | 只作同数据外部参照 |
| 端到端外部方法 | DCLOOS | official/reduced compatibility | 使用 pseudo-OOS 和外部 OOS | 监督合同不同，单独报告 |

所以本文后面的“当前自有方法”默认指 `S2C-Trainable-K1`，而不是 RC-AMBL、Joint-Adaptive 或历史 Cascade。

## 2. 当前统一 fair 矩阵结果

来源：`results/analysis/cross_protocol_tradeoff_v1/summary_mean_std.csv`。覆盖 3 数据集 × 3 KIR × 5 seeds。

| 方法 | OOS F1（9 格均值） | F1-All（9 格均值） | 解释 |
|---|---:|---:|---|
| **S2C Trainable K=1** | **85.75%** | **83.36%** | 当前自有矩阵最平衡；OOS F1 在 9 格中 8 格第一，F1-All 9/9 第一 |
| Frozen K=1 | 78.64% | 78.30% | 冻结表示基线 |
| Random K=2 | 78.39% | 78.51% | 增加中心但不学习语义子结构 |
| Frozen K=2 | 75.51% | 76.79% | StackOverflow 的 union 过覆盖最明显 |
| MOGB partition + S2C boundary | 77.74% | 63.92% | 低误接收伴随严重 Known 拒绝 |
| S2C partition + MOGB boundary | 75.90% | 62.30% | MOGB 平均半径/欧氏边界工作点偏保守 |
| MOGB-MiniLM | 73.39% | 46.26% | false acceptance 很低，但 Known Recall 约 31.21% |

这张表回答的是：**在同一 protocol_v2、同一 Known 划分、同一 Gate evaluator 下，当前哪个组件最稳定。**它没有回答“谁超过论文 SOTA”。

## 2.1 当前协议下游 Cascade bridge（不与历史 Cascade 混排）

此前 Trainable Gate 不能直接接旧 v19 Router/Expert；该合同断点已经通过独立
`cascade_bridge_v1` 修复。当前实验在 StackOverflow/KIR=.50、seeds=13/42/87 上重新训练一个
Known-only SmolLM Expert，并用 `calibration_known` 选 checkpoint；同一 Expert 分别接入 Frozen K=1
和 Trainable K=1 Gate，完成 6/6 评价行。

| Gate | OOS F1 | F1-All | Known Recall | False acceptance | 层级 |
|---|---:|---:|---:|---:|---|
| Frozen K=1 Cascade | 77.29±5.10 | 76.55±2.54 | 83.71 | 26.54 | 当前 protocol_v2 Cascade |
| **Trainable K=1 Cascade** | **86.71±0.96** | **83.25±0.75** | **83.92** | **11.14** | 当前 protocol_v2 Cascade |

Trainable 相对 Frozen 的配对差值为 OOS F1 `+9.42pp`、F1-All `+6.70pp`、Known Recall `+0.21pp`、
false acceptance `-15.40pp`。这只说明当前 Trainable Gate 的收益能传递到同协议下游，不能替代
`fulltex.tex` 的历史 Cascade，也不能直接证明超过官方 BERT MOGB 或 DCLOOS。
证据：`docs/archive/analysis/CASCADE_BRIDGE_V1.md`、`results/analysis/archive/analysis/cascade_bridge_v1/`、
`figures/archive/analysis/cascade_bridge_v1/`。

## 3. StackOverflow/KIR=.50 的外部参照

来源：`results/analysis/archive/analysis/comparison_atlas_v1/STACKOVERFLOW_EXTERNAL_COMPARISON_V1.md`。

| 方法 | OOS F1 | F1-All | Known Recall | FA | 证据状态 |
|---|---:|---:|---:|---:|---|
| **S2C Trainable K=1** | **87.67±1.66%** | **86.55±1.30%** | 83.89% | 9.34% | 当前 MiniLM fair，5 seeds |
| ADB | 87.36±1.61% | 85.66±1.59% | 80.78% | 7.52% | BERT/TextOIR，StackOverflow/KIR=.50 五 seed外部参照 |
| MOGB partition + S2C boundary | 79.25±1.49% | 63.34±5.50% | 50.39% | 1.86% | 当前 fair 组件 |
| MOGB-MiniLM | 72.92±0.62% | 43.30±3.40% | 27.09% | 0.79% | 当前 fair 组件 |
| DA-ADB | 72.48±6.24% | 74.02±3.13% | 75.97% | 29.07% | BERT/TextOIR，StackOverflow/KIR=.50 三 seed外部参照；不与 MiniLM fair 行合并 |

ADB 的 OOS F1 与 Trainable 接近，但两者仍同时改变了 BERT/MiniLM 表示、训练过程和边界，因此这不是“边界组件胜负”的证明。最新相同 seed 配对差值（Trainable−ADB）为 OOS F1 `+0.85 pp`、F1-All `+1.03 pp`、Known Recall `+2.90 pp`，但 Trainable 的 false acceptance 高约 `0.66 pp`。

### 3.1 DA-ADB 当前协议三 seed 合同审计

DA-ADB 已从“无效/全类预测”推进为可审计的当前协议三 seed 外部参照：StackOverflow/KIR=.50、seed=`42,87,100`，均使用当前 protocol_v2 split 根和 seed-specific Known 列表，但仍是 BERT/TextOIR 训练合同。

| 方法 | OOS F1 | F1-All | F1-Known | Known Recall | FA | seed |
|---|---:|---:|---:|---:|---:|---|
| DA-ADB | 72.48±6.24 | 74.02±3.13 | 74.18±2.86 | 75.97±4.10 | 29.07±11.19 | 42/87/100 |
| S2C Trainable K=1 | 88.21±1.72 | 86.69±1.45 | 86.54±1.42 | 83.68±0.28 | 8.18±3.36 | 42/87/100 |

同 seed 的描述性配对差值（S2C−DA-ADB）为 OOS F1 `+15.73pp`、F1-All `+12.67pp`、Known Recall `+7.71pp`，FA `−20.89pp`。这不是同骨干公平排名，但已经证明旧兼容单格 `90.90%` 不能代表当前 protocol_v2 DA-ADB。三 seed 运行、bootstrap 区间和图见 `docs/analysis/DA_ADB_CURRENT_PROTOCOL_SUMMARY_V1.md`、`results/analysis/da_adb_current_protocol_summary_v1/`、`figures/da_adb_current_protocol_summary_v1/`。

### 3.2 ADB 跨数据集外部合同参照

已补齐 CLINC150、Banking77、StackOverflow 各 3 个 seed（KIR=.50），共 `9/9` 个可审计单元。该表仍是
BERT/TextOIR 外部参照，不是与 MiniLM 的同骨干排名：

| 数据集 | ADB OOS F1 | ADB F1-All | ADB Known Recall | Trainable−ADB OOS F1 | Trainable−ADB F1-All |
|---|---:|---:|---:|---:|---:|
| CLINC150 | 89.39±0.91% | 85.49±0.77% | 90.84±0.61% | +1.05 pp | −3.67 pp |
| Banking77 | 74.97±1.92% | 78.79±1.53% | 89.14±0.67% | +8.60 pp | +2.89 pp |
| StackOverflow | 87.21±1.17% | 85.81±1.15% | 81.37±1.33% | +0.46 pp | +0.75 pp |

这组结果修正了“Trainable 在所有外部基线上都更好”的过度概括：它在 OOS F1 上三数据集均接近或高于
ADB，但在 CLINC150 的综合 F1-All 和 Known Recall 上明显落后；差异同时包含 BERT/MiniLM backbone、训练目标、
边界学习和监督合同。逐 seed 结果、配对差值和重算来源见
`results/analysis/archive/analysis/adb_cross_dataset_v1/ADB_CROSS_DATASET_REPORT.md`。

### 3.3 ADB 跨 KIR 机制结果

进一步补跑 KIR=.25/.75 后，ADB 已完成三数据集×三 KIR×五 seed 的 `45/45` 单元。ADB 的 OOS F1 随 KIR
增加而下降：

| 数据集 | KIR=.25 | KIR=.50 | KIR=.75 |
|---|---:|---:|---:|
| CLINC150 | 91.53±1.08 | 90.01±0.45 | 85.09±1.16 |
| Banking77 | 83.83±2.46 | 74.30±1.50 | 65.64±3.06 |
| StackOverflow | 93.36±0.32 | 87.36±1.61 | 73.08±0.74 |

Trainable-K1 的 OOS F1 在 9 个 dataset×KIR 组中 8 个高于 ADB，但这不等于综合指标全面领先：
CLINC150 的 F1-All 和 Known Recall 在三个 KIR 都低于 ADB，Banking77 的 KIR=.75 F1-All 也低于 ADB。
因此这组实验主要证明外部基线差距具有数据集/KIR 依赖，不能只用一个 StackOverflow 工作点作结论。
曲线、热力图和逐 seed 数据见 `results/analysis/adb_kir_sensitivity_v2/` 与
`figures/adb_kir_sensitivity_v2/`；三 seed 历史汇总仍保留在 v1，不覆盖旧 artifact。

## 4. DCLOOS：已有 reduced 兼容结果，但不是公平主表

DCLOOS 的官方数据来源已经定位：作者 README 指向的 Drive `squad.tsv` 被原样复制为运行时所需的
`squad_placeh.tsv`，文件 SHA256 为
`f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`。因此“缺外部负样本”已经不是当前事实阻断。

默认预算的 `DCLOOS-official` 单元在三小时上限内超时，没有最终指标；但独立登记的 reduced-budget 单元
`KIR=.75/seed=888` 已从官方预测文件恢复出：

| 方法 | OOS F1 | F1-All | F1-K | Known Recall | Accuracy | 合同 |
|---|---:|---:|---:|---:|---:|---|
| DCLOOS reduced | 87.05 | 90.26 | 90.29 | 92.14 | 88.68 | BERT + pseudo-OOS + 外部 SQuAD OOS |

该结果高于当前部分 Frozen Gate，但它使用更强的 OOS 监督、不同 KIR 和不同 seed，且不是默认完整
预算的论文复现。因此目前可写成“DCLOOS 在更强监督条件下的 reduced 兼容结果较高”，不能写成
“当前 S2C 已超过 DCLOOS”或“同协议 SOTA”。证据见
`docs/archive/external_baselines/dcloos/DCLOOS_REPRODUCTION_REPORT.md` 和
`../artifacts/s2c/external/dcloos_official_oos_kir75_seed888_reduced_v2/`。

### 4.1 当前 registry 固定 Known-list 尝试

为排除 Known 类别重新抽样造成的混淆，已新增运行时 `--known-labels-file` 适配，并在当前
StackOverflow/KIR=.50/seed=42 registry 上启动 DCLOOS。该单格保留 BERT、pseudo-OOS 和外部
SQuAD 监督，运行约 3530 秒后未生成最终 `metrics.json`，只有中间预测，因此不进入任何数值表或图。
本次状态和资源阻塞证据见 `docs/archive/analysis/DCLOOS_CURRENT_PROTOCOL_BLOCKER_V1.md`；它不能被解释为
DCLOOS 的算法成绩，也不能用来替换已有 reduced 结果。

随后又进行了一个 `max_epochs=10, patient=3` 的 fixed-registry reduced 尝试；该运行约 1,246 秒后
仍未输出最终 metrics，因 CPU 资源路径而中止。它仅确认缩短预算仍不足以在当前环境完成一条可审计
DCLOOS 单格，不产生新的性能数字；中间 prediction 仍被排除。

## 5. 历史 fulltex 的“超过 baseline”到底是什么

历史 `fulltex.tex` 的 `Ours` 是完整 Gate→Router→Expert Cascade，不是当前 Trainable-K1。历史表比较 MSP、OpenMax、DOC、DeepUnk、KNNCL、ADB、DA-ADB；旧合同下 OOS F1 在 9/9 个 dataset×KIR 格子第一，优势为：

| 数据集 | KIR=.25 | KIR=.50 | KIR=.75 |
|---|---:|---:|---:|
| CLINC150 | +1.45 pp | +1.86 pp | +1.10 pp |
| Banking77 | +7.42 pp | +8.30 pp | +17.12 pp |
| StackOverflow | +1.82 pp | +0.85 pp | +1.02 pp |

这只能说明：**历史完整 Cascade 在旧协议的 OOS F1 上领先其表内基线**。Known F1 和 Accuracy 并非每一格都全面领先；也不能把历史 Cascade 的数字直接替换成当前 Gate-only 结果。

## 6. MOGB 为什么本地结果和论文差很多

MOGB 当前已经完成的是“作者公开逻辑的现代兼容单格”，不是作者原始环境逐字节复现：

- 使用了 BERT、最近子中心训练、递归粒球、平均半径和最近粒球推理；
- StackOverflow/KIR=.50/seed=0：本地 `Acc=75.17`、`F1-All=68.35`、`F1-U=79.97`、`F1-K=67.19`；
- 论文公开参考：`Acc=88.67`、`F1-All=87.49`、`F1-U=89.71`、`F1-K=87.27`。

现有代码级证据把差距拆成三部分：

1. 官方子中心损失先对非负距离做 L1 归一化，类别数增加后 logit 动态范围极窄，训练信号接近均匀分类；
2. 最终平均半径工作点过窄，本地 StackOverflow 的 OOS Recall 达 98.80%，但 Known Recall 只有 51.53%；
3. 作者原始数据快照、Known 类列表、旧依赖环境以及最终球的随机状态没有完整恢复，因此不能把本地低分解释成论文算法本身无效。

修正损失后，F1-All 只恢复约 4.29 pp；在 corrected checkpoint 上用 Known-only dev 扩大半径可以提高 Known Recall，但会快速增加 OOS 误接受，仍不能到达论文工作点。这说明 MOGB 差距不是单一 `lambda` 或单一半径常数造成的。

## 7. 已生成的可视化入口

本轮新增的四张总览图位于 `figures/archive/analysis/experiment_comparison_overview_v2/`：

1. `current_fair_matrix_mean.png`：当前 7 行 fair Gate 的 OOS F1/F1-All 均值；
2. `stackoverflow_kir50_contract_layers.png`：StackOverflow/KIR=.50 的 Trainable、MOGB 组件和 ADB 外部参照；DA-ADB 最新有效单格单独标注为外部合同；
3. `mogb_paper_vs_local_exact.png`：MOGB 论文公开工作点与本地兼容单格；
4. `historical_fulltex_margin.png`：历史 Cascade 相对表内最佳基线的 OOS F1 优势。
5. `external_supervision_reference.png`：S2C、ADB 与 DCLOOS reduced 的监督、KIR、骨干和预算差异；不作为统一排名。

机器可读结果和源哈希在 `results/analysis/archive/analysis/experiment_comparison_overview_v2/MANIFEST.json`。

## 8. 当前仍未完成的证据

- 完整官方合同下的 MOGB 多 seed 复现；
- 通过数值审计的 DA-ADB 当前协议三 seed（已完成）；旧兼容单格与当前合同差异已单独归因；
- DCLOOS 默认预算完整单元及同协议多 seed 对比；本轮固定 registry 单格仍因资源/进程中断没有最终指标，详见 `DCLOOS_CURRENT_PROTOCOL_BLOCKER_V1.md`；
- 当前 protocol_v2 Cascade 的其他数据集扩展（本轮只完成 StackOverflow/KIR=.50）。

因此目前不能写“当前方法已经超过 MOGB/DCLOOS 并达到新 SOTA”。可以写的严格结论是：

> 在当前统一 Known-only MiniLM Gate 合同下，Trainable-K1 是最平衡的自有候选；固定 K>1 在 StackOverflow 的主要风险是 OOS acceptance-union 过覆盖；MOGB-MiniLM 的低误接收主要以严重 Known 拒绝为代价；外部 ADB 参照与 Trainable 接近，但训练合同不同。

## 9. 下一步（仍然是实验，不是论文写作）

1. DA-ADB 当前 protocol_v2 三 seed 已完成；不再扩大这一 StackOverflow 工作点，继续整理已有方法的逐样本可视化，不重复已完成 K/KIR 矩阵。
2. DCLOOS 外部负样本来源已闭合，但固定 registry 单格没有最终指标；保留 blocker 证据与 reduced 结果的合同标签，不用中间预测或测试 OOS 替代。
3. 在外部默认单格未闭合前，把实验重点放在当前可验证的同协议证据：Trainable/Frozen/MOGB 组件的逐样本错误、Known/OOS 工作点和跨数据集图，而不是重复旧 K/KIR 矩阵。

## 证据入口

- 当前方法地图：`docs/archive/analysis/METHOD_COMPARISON_MAP_V1.md`
- 当前综合分析：`docs/archive/analysis/EXPERIMENT_ANALYSIS_MASTER_V1.md`
- MOGB 复现差距：`docs/archive/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`
- 外部单格审计：`results/analysis/archive/analysis/comparison_atlas_v1/STACKOVERFLOW_EXTERNAL_COMPARISON_V1.md`
- 可视化索引：`docs/analysis/VISUAL_ANALYSIS_INDEX_V1.md`
