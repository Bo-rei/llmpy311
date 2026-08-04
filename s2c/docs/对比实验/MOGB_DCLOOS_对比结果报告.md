# s2c 与 MOGB、DCLOOS 对比结果报告

更新时间：2026-08-04
活动协议：`protocol_v2_textoir_v1`

## 先说结论

当前确实已经做过 s2c 与 MOGB、DCLOOS 的相关对比，但证据分为两种，不能混成一张“SOTA 排名表”：

1. **同协议公平组件对比已经完成。** 在相同 TEXTOIR-compatible 划分、相同 Frozen `all-MiniLM-L6-v2` 表示和相同评价器下，比较了 s2c 单中心、固定 K=2、MOGB 风格自适应粒球和边界组件。
2. **MOGB 官方 BERT 复现没有达到论文结果。** 已经运行官方模块的现代兼容版本，也完成 StackOverflow 和 Banking77 单格审计，但只能标记为 `not_reproduced_strict` 或 `complete_non_strict_reproduction`。
3. **DCLOOS 的完整官方单元没有成功收敛。** 默认运行超时，没有可用的正式最终指标；另有一个使用官方外部 SQuAD 负样本的 reduced-budget 恢复结果，但它不是同协议结果。

因此，目前不能宣称“s2c 已经超过 MOGB 或 DCLOOS”，也不能宣称“s2c 达到 SOTA”。目前最可靠的结论是：**s2c 的单中心在统一 MiniLM 协议下综合质量最稳；MOGB 风格粒球有时提高 OOS F1，但明显牺牲 Known Recall 和 F1-All；DCLOOS 的恢复结果数值更高，但使用了额外的伪 OOS 和外部 OOS 监督，尚未完成公平比较。**

## 1. 方法和证据边界

### 1.1 本报告中的“s2c 方法”

| 名称                     | 表示          | 中心/边界                                         | 监督条件                       |
| ------------------------ | ------------- | ------------------------------------------------- | ------------------------------ |
| s2c 单中心               | Frozen MiniLM | 每个意图一个中心                                  | Known-only                     |
| s2c 固定多中心           | Frozen MiniLM | 每个意图固定 K=2 的 KMeans 中心                   | Known-only                     |
| MOGB 风格粒球 + s2c 边界 | Frozen MiniLM | 自适应粒球划分；使用 s2c 的距离和`μ+λσ` 半径 | Known-only                     |
| MOGB-MiniLM              | Frozen MiniLM | 自适应粒球划分；MOGB 欧氏距离和平均半径           | Known-only                     |
| MOGB 官方兼容            | BERT          | 官方粒球、最近子中心训练和最近粒球边界            | Known-only，但使用官方数据契约 |
| DCLOOS                   | BERT 端到端   | 直接分类器                                        | 伪 OOS + 外部开放域 OOS        |

这里必须特别说明：`MOGB 风格粒球 + s2c 边界` 和 `MOGB-MiniLM` 是本项目的组件对照，不是作者论文中的完整 MOGB。完整 MOGB 还包含 BERT 表示训练、最近子中心损失和交替聚类。

## 2. 同协议公平对比：KIR=0.50、5 个 seed

下表来自 `results/final_baselines/summary.csv` 和 `results/mogb/fair_matrix.csv`。每一行是同一数据集、KIR=0.50 下 5 个正式 seed 的均值。指标单位为百分比。

### 2.1 OOS F1

| 数据集        | s2c 单中心 | s2c 固定 K=2 | MOGB 风格粒球 + s2c 边界 | MOGB-MiniLM |
| ------------- | ---------: | -----------: | -----------------------: | ----------: |
| CLINC150      |      88.94 |        89.20 |                    85.56 |       81.32 |
| Banking77     |      72.43 |        75.46 |                    79.40 |       74.99 |
| StackOverflow |      76.55 |        63.53 |                    79.25 |       72.92 |

### 2.2 F1-All

| 数据集        | s2c 单中心 | s2c 固定 K=2 | MOGB 风格粒球 + s2c 边界 | MOGB-MiniLM |
| ------------- | ---------: | -----------: | -----------------------: | ----------: |
| CLINC150      |      80.27 |        80.09 |                    64.60 |       44.95 |
| Banking77     |      74.51 |        76.16 |                    64.81 |       48.60 |
| StackOverflow |      79.98 |        72.76 |                    63.34 |       43.30 |

### 2.3 Known Recall

| 数据集        | s2c 单中心 | s2c 固定 K=2 | MOGB 风格粒球 + s2c 边界 | MOGB-MiniLM |
| ------------- | ---------: | -----------: | -----------------------: | ----------: |
| CLINC150      |      78.76 |        75.14 |                    53.34 |       31.57 |
| Banking77     |      85.01 |        82.55 |                    52.18 |       33.41 |
| StackOverflow |      87.15 |        86.89 |                    50.39 |       27.09 |

### 2.4 直接解释

- 在 CLINC150，MOGB 风格粒球的 OOS F1 比固定 K=2 低约 3.65 个百分点，比单中心低约 3.39 个百分点。
- 在 Banking77，MOGB 风格粒球的 OOS F1 比固定 K=2 高约 3.94 个百分点，但 Known Recall 低约 30.37 个百分点，F1-All 低约 11.35 个百分点。
- 在 StackOverflow，MOGB 风格粒球的 OOS F1 比固定 K=2 高约 15.72 个百分点，但 Known Recall 低约 36.50 个百分点，F1-All 低约 9.41 个百分点。
- 与单中心相比，MOGB 风格粒球的 OOS F1 在 Banking77 和 StackOverflow 分别高约 6.97 和 2.70 个百分点，但 F1-All 分别低约 9.71 和 16.64 个百分点。

这说明 **MOGB 风格粒球的 OOS F1 提升主要伴随更强的 Known 拒绝**，而不是综合分类能力全面提升。若同时考虑 Known 和 OOS，当前同协议结果中 s2c 单中心最稳。

## 3. MOGB 官方 BERT 复现结果

### 3.1 StackOverflow：KIR=0.50、seed=0

| 指标     | MOGB 论文参考值 | 本地官方逻辑兼容运行 |
| -------- | --------------: | -------------------: |
| Accuracy |           88.67 |                75.17 |
| F1-All   |           87.49 |                68.35 |
| F1-U     |           89.71 |                79.97 |
| F1-K     |           87.27 |                67.19 |

本地结果比论文参考值低 13.50、19.14、9.74 和 20.08 个百分点，不能称为成功复现。

### 3.2 Banking77：KIR=0.75、seed=0

| 指标     | 论文表格参考值 | 本地运行 |
| -------- | -------------: | -------: |
| Accuracy |          80.58 |    57.08 |
| F1-All   |          81.52 |    59.16 |
| F1-U     |          81.04 |    53.10 |
| F1-K     |          81.53 |    59.27 |

Banking77 的本地单元也明显低于论文数字，而且本地 KIR 与论文表格的协议并不完全一致，因此只能作为负复现证据，不能作为公平的论文数值比较。

### 3.3 MOGB 复现结论

当前正确标签是：

```text
official_code_not_reproduced_under_available_materials
```

已经完成的是官方模块逻辑的现代兼容运行，不是原作者旧环境下的逐字节复现。详细证据见：

- [MOGB 严格单格复现报告](../archive/mogb_reproduction/mogb_integration/MOGB_EXACT_REPRODUCTION_REPORT.md)
- [MOGB 官方逻辑收口报告](../archive/mogb_reproduction/mogb_integration/MOGB_OFFICIAL_CLOSEOUT.md)
- [MOGB 官方代码审计](../archive/mogb_reproduction/mogb_integration/mogb_official_audit.md)

## 4. DCLOOS 端到端对比结果

### 4.1 官方运行状态

DCLOOS 的正式方法使用：

- 特征级伪 OOS；
- 外部开放域 OOS（官方 SQuAD 文件）；
- 端到端 BERT 分类训练；
- 测试阶段不依赖 s2c 的距离后处理。

默认官方单元因 BERT 训练超过预设时间上限而 `timeout_incomplete`，没有正式最终指标。因此，**当前没有一个可以与 s2c 做严格同协议比较的 DCLOOS 结果。**

### 4.2 reduced-budget 恢复结果

另一个独立登记的 reduced-budget 单元恢复出了预测结果：

| 指标         | DCLOOS reduced |
| ------------ | -------------: |
| OOS F1       |          87.05 |
| F1-All       |          90.26 |
| F1-K         |          90.29 |
| Known Recall |          92.14 |
| Accuracy     |          88.68 |

但该结果使用 KIR=0.75、seed=888、伪 OOS 和外部 SQuAD OOS，且不是默认完整预算的官方复现。因此它只能说明：**在更强监督条件下，DCLOOS 的兼容运行指标高于当前 Frozen MiniLM Gate；不能据此宣布 DCLOOS 在同协议下战胜 s2c。**

详细证据见 [DCLOOS 复现审计](../archive/external_baselines/dcloos/DCLOOS_REPRODUCTION_REPORT.md)。

## 5. 当前能否回答“谁更好”

### 5.1 s2c 与 MOGB

- **综合指标：**在同一 Frozen MiniLM 协议下，s2c 单中心的 F1-All 和 Known Recall 通常优于 MOGB 风格粒球。
- **单独 OOS F1：**MOGB 风格粒球在 Banking77 和 StackOverflow 更高，但主要代价是大量 Known 样本被拒绝。
- **官方 MOGB：**本地 BERT 兼容运行没有达到论文结果，因此不能用本地 MOGB 数字证明 s2c 超过或低于论文 MOGB。

### 5.2 s2c 与 DCLOOS

- **严格公平结论：**目前不能下结论，因为 DCLOOS 官方完整运行没有产生最终指标。
- **描述性结果：**reduced DCLOOS 的 F1-All=90.26、OOS F1=87.05，高于当前 Frozen MiniLM s2c 单中心的相应结果，但它使用了伪 OOS 和外部 OOS，监督条件更强。
- **论文表述：**只能写成“兼容性恢复结果显示 DCLOOS 在额外 OOS 监督下具有更高指标”，不能写成统一协议下的 SOTA 排名。

## 6. 现有文档入口

本报告是当前面向研究汇报的中文入口。原始机器结果和审计证据仍保留在：

| 用途            | 文件                                                                                  |
| --------------- | ------------------------------------------------------------------------------------- |
| 总表            | `results/final_baselines/summary.csv`                                               |
| MOGB 同协议矩阵 | `results/mogb/fair_matrix.csv`                                                      |
| MOGB 组件报告   | `docs/archive/mogb_reproduction/mogb_integration/MOGB_REPRODUCTION_REPORT.md`       |
| MOGB 官方复现   | `docs/archive/mogb_reproduction/mogb_integration/MOGB_EXACT_REPRODUCTION_REPORT.md` |
| MOGB 收口       | `docs/archive/mogb_reproduction/mogb_integration/MOGB_OFFICIAL_CLOSEOUT.md`         |
| DCLOOS 审计     | `docs/archive/external_baselines/dcloos/DCLOOS_REPRODUCTION_REPORT.md`              |
| 当前项目状态    | `docs/CURRENT_STATUS.md`                                                            |
| 实验总账        | `docs/EXPERIMENT_LEDGER.csv`                                                        |

归档英文报告是历史原始审计材料；后续新增的主动研究报告应以中文为主，并在需要时引用这些机器可读文件。

## 7. 审稿人意见对应状态

| 审稿要求                 | 当前状态                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------- |
| 与 MOGB 直接相关方法比较 | 已完成 Frozen MiniLM 组件级同协议比较；官方 BERT 只完成负复现，不可作为公平排名        |
| 与端到端 DCLOOS 比较     | 已完成来源、监督条件和运行审计；完整官方单元超时，只有 reduced-budget 兼容结果         |
| 多 seed 统计             | s2c/MOGB Frozen 组件矩阵已覆盖 5 个 seed；ADB、DA-ADB、DCLOOS 尚未形成同协议多 seed 表 |
| K 消融                   | 已完成 E2 和固定 K/MOGB 组件消融                                                       |
| λ 敏感性和泄漏审计      | 已完成 Known-only 选择边界与样本不重叠审计                                             |

因此，当前报告可以支持“已经做过相关对比并发现协议差异”，但还不能支持“已经完成公平 SOTA 排名”。
