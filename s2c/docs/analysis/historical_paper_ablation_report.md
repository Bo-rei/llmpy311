# 完整主实验结果与论文设置消融实验

## 结论先行

本报告补齐两类结果，但不混淆证据层级：

1. **当前 H1 Trainable full pipeline**：使用当前工作区已经完成的 CUDA 组件/推理结果，按数据集和 KIR 汇总，并与 `fulltex.tex` 的 Ours 逐格比较。
2. **论文设置消融**：读取已有 `paper_results` 下实际 materialized `eval_results.json`，完整覆盖 `3 数据集 × 3 KIR × 4 变体 = 36/36` 个单元。它是历史 CPU/旧环境 eval artifact，不是本轮新训练的 CUDA H1 消融，不能与第一部分直接合并排名。

因此，“消融实验完成”在本报告中表示：36 个实际 eval JSON 均存在并已重新读取指标；但它仍不是本轮重新训练的 CUDA 消融。当前 H1 的 Trainable 结果则单独报告其真实 CUDA provenance。

如果问题是“当前方法唯一把 Gate 从 Frozen MiniLM 换成 Trainable MiniLM 的收益”，请使用[Matched Frozen-vs-Trainable Gate 消融报告](historical_gate_ablation_report.md)。该报告固定 Router/Expert 和边界合同，完成了 54 个 CUDA full-pipeline 单元；本页的 36 个论文四变体 artifact 不替代这组直接对照。

## 1. 当前 H1 Trainable full pipeline

当前结果保持 Gate→Router→Expert 结构，OOS 训练和测试选择均为 false；不同 KIR 的最优配置不强行视为同一个模型：CLINC KIR=.50 使用 Known-only recipe search，StackOverflow/Banking77-OOS KIR=.50 使用 adaptive-centers overall 结果，KIR=.25 与 KIR=.75 使用直接 Trainable K=1 结果。

| 数据集 | KIR | 当前配置 | OOS F1 | full macro F1 | Accuracy | Known Recall | False Acceptance | 相对论文 Ours |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | 0.25 | trainable_k1 | 95.10±0.30 | 75.41±1.65 | 91.19±0.27 | 73.13±1.98 | 2.97±1.01 | +0.09 pp |
| CLINC150 | 0.50 | recipe:last2_no_inter;recipe:last2_temp10 | 91.93±0.48 | 82.57±0.54 | 88.02±0.46 | 86.99±1.79 | 7.27±0.17 | -0.03 pp |
| CLINC150 | 0.75 | trainable_k1_direct | 80.80±0.66 | 78.05±0.38 | 79.56±0.51 | 73.83±0.15 | 4.36±1.28 | -6.30 pp |
| StackOverflow | 0.25 | trainable_k1 | 95.38±1.02 | 82.62±0.13 | 91.58±0.92 | 83.56±0.19 | 3.83±2.00 | +0.91 pp |
| StackOverflow | 0.50 | adaptive_centers_overall | 89.96±1.50 | 84.04±1.45 | 86.94±1.27 | 82.69±2.03 | 4.12±1.33 | +0.25 pp |
| StackOverflow | 0.75 | trainable_k1_direct | 75.16±2.36 | 84.24±0.91 | 81.92±0.96 | 83.11±0.22 | 9.21±4.80 | -0.41 pp |
| Banking77-OOS | 0.25 | trainable_k1 | 92.58±0.89 | 62.89±2.41 | 86.91±1.36 | 84.38±3.12 | 12.01±1.63 | -1.41 pp |
| Banking77-OOS | 0.50 | adaptive_centers_overall | 91.89±0.19 | 66.28±0.84 | 85.96±0.22 | 65.03±2.80 | 5.36±1.03 | +3.66 pp |
| Banking77-OOS | 0.75 | trainable_k1_direct | 86.60±0.42 | 70.10±0.25 | 79.44±0.38 | 86.58±0.76 | 17.54±0.49 | +0.11 pp |

说明：均值和标准差使用 3 个 seed，std 为总体标准差（ddof=0）。当前 H1 表中的 OOS F1 与 Gate 的 OOS 二分类一致；Router/Expert 主要影响 Known 分类、full macro F1 和 Accuracy。
论文表头把该列显示为 `Banking77`，但当前历史 artifact 的实际数据键是 `banking77_oos`；`banking77` archive/protocol_v2 线是另一条数据线，不能替代它。

### 93.99 的来源与 Banking 协议拆分

`fulltex.tex` 主表的列名是 **Banking77**，但 artifacts 中的实际 Ours eval 位于 `paper_results/banking77_oos/kir25_seed42/full_anchor/eval_results.json`，其 OOS F1 为 `93.9852%`、Accuracy 为 `89.0686%`，即论文表中的 `93.99/89.07`。该文件的 `data_root` 也是 `data/multidataset/v19/banking77_oos/kir25_seed42`。因此，`93.99` 确实存在于 artifacts，不是当前 `banking77_oos` 中不存在的隐藏数字。

| 线 | Known | OOS | Test | 结果/含义 |
|---|---:|---:|---:|---|
| 历史 Ours artifact `banking77_oos/kir25_seed42` | 12 Known intents | 66 OOS intents | 4080 | 实际保存了 `93.9852/89.0686`，与论文 `93.99/89.07` 对应；这是历史 Ours，不是当前 Trainable K=1 结果 |
| archive/protocol_v2 `banking77/kir25_seed42` | 19 Known intents | 58 OOS intents | 3080 | 独立的标准 Banking77 数据线，不是上述 `93.99` artifact 的数据根 |

因此，当前 H1 Trainable 的 Banking77-OOS 结果可以与历史 Ours artifact 做同数据键下的历史系统参照，但仍必须标记方法、Gate、语义校准和下游模型合同不同；标准 `banking77` 结果不能混入这条比较。

## 2. 论文消融设置与覆盖情况

论文消融的四个变体为：

- **Ours**：论文原始 Gate–Router–Expert 配置；
- **Without Gate**：移除几何 Gate，使用论文规定的下游置信度拒识替代；Banking77-OOS 的历史 artifact 使用其对应的 expert-confidence 变体名称；
- **Cascade-MiniLM**：级联各阶段使用 MiniLM；
- **Cascade-SmolLM**：级联各阶段使用 SmolLM。

每个数据集均覆盖 KIR=.25/.50/.75，历史 anchor key 为 seed42；36 个实际 eval JSON 均存在。`banking77_oos` 与 archive `banking77` 仍是两条不同数据线。

### CLINC150

| KIR | Historical full anchor OOS F1 / Acc | Without Gate OOS F1 / Acc | Cascade-MiniLM OOS F1 / Acc | Cascade-SmolLM OOS F1 / Acc |
|---:|---:|---:|---:|---:|
| 0.25 | 95.30 / 91.04 | 76.88 / 67.49 | 94.27 / 90.49 | 90.98 / 84.76 |
| 0.50 | 91.96 / 86.78 | 70.24 / 67.90 | 90.67 / 85.89 | 78.69 / 73.58 |
| 0.75 | 82.14 / 80.75 | 66.50 / 73.40 | 81.00 / 80.10 | 71.79 / 74.31 |

### StackOverflow

| KIR | Historical full anchor OOS F1 / Acc | Without Gate OOS F1 / Acc | Cascade-MiniLM OOS F1 / Acc | Cascade-SmolLM OOS F1 / Acc |
|---:|---:|---:|---:|---:|
| 0.25 | 91.70 / 85.83 | 88.30 / 81.80 | 86.85 / 79.73 | 45.00 / 40.28 |
| 0.50 | 89.71 / 85.54 | 82.79 / 80.28 | 79.41 / 77.46 | 55.68 / 50.10 |
| 0.75 | 75.57 / 81.32 | 65.83 / 77.36 | 62.50 / 75.91 | 28.72 / 58.91 |

### Banking77-OOS

| KIR | Historical full anchor OOS F1 / Acc | Without Gate OOS F1 / Acc | Cascade-MiniLM OOS F1 / Acc | Cascade-SmolLM OOS F1 / Acc |
|---:|---:|---:|---:|---:|
| 0.25 | 93.99 / 89.07 | 91.27 / 84.90 | 91.41 / 85.10 | 64.43 / 51.20 |
| 0.50 | 88.23 / 78.98 | 82.88 / 77.15 | 84.63 / 77.11 | 77.61 / 65.83 |
| 0.75 | 85.28 / 77.84 | 84.20 / 76.50 | 83.69 / 77.23 | 55.41 / 52.48 |

### Historical anchor 与 `fulltex.tex` 的一致性

36 个消融组合均有结果，但历史 `full_anchor` 在 OOS F1 与 Accuracy 两个展示指标上只有 **5/9** 个数据集×KIR 单元同时与 `fulltex.tex` 一致；因此上面的第一列明确标为历史 anchor，而不把它自动改写成论文 Ours。

| 数据集 | KIR | 历史 full anchor OOS F1 / Acc | `fulltex.tex` Ours OOS F1 / Acc | 差值 |
|---|---:|---:|---:|---:|
| CLINC150 | 0.25 | 95.30 / 91.04 | 95.01 / 90.45 | OOS 0.29 pp; Acc 0.59 pp |
| CLINC150 | 0.50 | 91.96 / 86.78 | 91.96 / 86.78 | OOS 0.00 pp; Acc 0.00 pp |
| CLINC150 | 0.75 | 82.14 / 80.75 | 87.10 / 79.83 | OOS -4.96 pp; Acc 0.92 pp |
| StackOverflow | 0.25 | 91.70 / 85.83 | 94.47 / 91.04 | OOS -2.77 pp; Acc -5.21 pp |
| StackOverflow | 0.50 | 89.71 / 85.54 | 89.71 / 85.54 | OOS 0.00 pp; Acc 0.00 pp |
| StackOverflow | 0.75 | 75.57 / 81.32 | 75.57 / 81.32 | OOS 0.00 pp; Acc 0.00 pp |
| Banking77-OOS | 0.25 | 93.99 / 89.07 | 93.99 / 89.07 | OOS -0.00 pp; Acc -0.00 pp |
| Banking77-OOS | 0.50 | 88.23 / 78.98 | 88.23 / 78.98 | OOS 0.00 pp; Acc -0.00 pp |
| Banking77-OOS | 0.75 | 85.28 / 77.84 | 86.49 / 77.84 | OOS -1.21 pp; Acc 0.00 pp |

需要特别保留的 mismatch：

- CLINC150/KIR=0.25：历史 anchor 与论文差值为 OOS F1 0.29 pp、Accuracy 0.59 pp。
- CLINC150/KIR=0.75：历史 anchor 与论文差值为 OOS F1 -4.96 pp、Accuracy 0.92 pp。
- StackOverflow/KIR=0.25：历史 anchor 与论文差值为 OOS F1 -2.77 pp、Accuracy -5.21 pp。
- Banking77-OOS/KIR=0.75：历史 anchor 与论文差值为 OOS F1 -1.21 pp、Accuracy 0.00 pp。

## 3. 消融结果的直接解释

- **Gate 是必要的结构组件**：Without Gate 在三个数据集和多数 KIR 下 OOS F1 明显低于 Ours，说明只依赖下游置信度不能替代几何 OOS Gate。
- **MiniLM 级联优于 SmolLM 级联**：Cascade-MiniLM 通常保留更高的 OOS F1 和 Accuracy；SmolLM 在 StackOverflow 和 Banking77-OOS 上的退化尤其明显。
- **消融不是当前 Trainable MiniLM 的新结论**：这些表格回答的是论文原始系统组件是否必要；Trainable MiniLM 的表示适配属于当前 H1 follow-up，应看第一节和既有 Frozen/Trainable 配对结果。
- **不能用单个 OOS F1 排名替代整体判断**：例如更保守的变体可能降低 OOS false acceptance，却同时牺牲 Known coverage，因此同时保留 Accuracy、macro F1 和 Known 指标。

## 4. 证据边界

- 历史消融实际源文件：`../artifacts/s2c/outputs/paper_results/<dataset>/<kir>/<variant>/eval_results.json`；`ablation_summary.csv` 只作为覆盖/派生关系 metadata，不能覆盖实际 eval 指标。
- 历史 ledger 记录的 CUDA 不可用，因此 36 个消融单元应标记为 **historical paper-eval evidence**；不能写成“本轮 GPU 重跑完成”。
- 当前 H1 结果属于 `historical_v19_paper_main` 的 controlled evidence，严格 H0 的原始目录和完整旧 Cascade 尚未完全逐字恢复。
- `fulltex.tex` 未修改；论文 Ours 数字仅作为 reported reference。

## 5. 机器可读入口

- [当前 H1 full-pipeline seed 表](../../results/analysis/historical_paper_ablation/current_h1_full_pipeline_per_seed.csv)
- [当前 H1 full-pipeline 汇总](../../results/analysis/historical_paper_ablation/current_h1_full_pipeline_summary.csv)
- [36 个论文消融单元](../../results/analysis/historical_paper_ablation/paper_ablation_summary.csv)
- [bundle manifest](../../results/analysis/historical_paper_ablation/MANIFEST.json)
