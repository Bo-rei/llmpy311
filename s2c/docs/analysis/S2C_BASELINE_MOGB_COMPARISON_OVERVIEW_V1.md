# S2C 历史基线优势与 MOGB 复现差距总览（V1）

> 这是当前问题的统一入口。历史论文、当前 Gate 和 MOGB 诊断属于不同实验合同，禁止把数字直接混成一张 SOTA 排名表。

## 0. 先回答“现在到底在比哪个方法”

| 问题 | 自有方法 | 对手 | 当前能否公平下结论 |
|---|---|---|---|
| 历史论文为什么写成 SOTA | `fulltex.tex` 的完整 `Ours` Cascade：冻结 MiniLM 固定 K=2 Gate + SmolLM Router/Expert | MSP、OpenMax、DOC、DeepUnk、KNNCL、ADB、DA-ADB | 只能复述历史合同下的 OOS F1 结果 |
| 当前 protocol_v2 哪个自有方法最好 | `S2C-Trainable-K1`：Known-only 训练 MiniLM 最后两层和 projection + 单中心 Gate | Frozen K1/K2、Random K2、MOGB-Fair 与两个组件桥 | 可以，已有 3 数据集×3 KIR×5 seed 配对 |
| 当前是否超过论文完整 MOGB | 不能用当前 Gate 行回答 | 作者 BERT+交替粒球训练的完整 MOGB | 不能；本地 official-logic 单格未复现论文工作点 |
| 当前是否超过 DCLOOS/ADB/DA-ADB | 不能用分散兼容单格回答 | 不同 backbone、监督与运行合同的外部方法 | 不能；尚缺同 split、多 seed、统一 evaluator 主表 |

因此，后文所有“当前 S2C 优于 MOGB”的表述均专指 `S2C-Trainable-K1` 对
`MOGB-MiniLM-Fair` 的同协议组件比较；它不是历史 Cascade 对论文 BERT-MOGB 的直接排名。

## 1. `fulltex.tex` 中哪个方法超过了基线

历史表中的 `Ours` 是完整的 Gate–Router–Expert Cascade：冻结 all-MiniLM-L6-v2 Gate、每意图固定 K=2 KMeans、多局部对角马氏边界，以及 SmolLM-135M LoRA Router/Experts。它不是当前的 Trainable-K1，也不是 RC-AMBL。

历史表比较 MSP、OpenMax、DOC、DeepUnk、KNNCL、ADB 和 DA-ADB。`Ours` 在九个 dataset×KIR 格子的 OOS F1 都高于表中最强基线；逐格优势为：

- banking77 / KIR=0.25：+7.42 pp
- banking77 / KIR=0.50：+8.30 pp
- banking77 / KIR=0.75：+17.12 pp
- clinc150 / KIR=0.25：+1.45 pp
- clinc150 / KIR=0.50：+1.86 pp
- clinc150 / KIR=0.75：+1.10 pp
- stackoverflow / KIR=0.25：+1.82 pp
- stackoverflow / KIR=0.50：+0.85 pp
- stackoverflow / KIR=0.75：+1.02 pp

这只支持‘历史合同下完整 Cascade 的 OOS F1 领先’，不支持所有 Known F1/Accuracy 都为 SOTA，也不能直接证明超过后来发表的 MOGB。

## 2. 相关对比和可视化是否存在

已经生成，但此前分散。主要入口：

- `figures/archive/analysis/historical_sota_comparison_v1/`：历史 Ours 与七个基线的热力图和逐格优势；
- `figures/archive/analysis/experiment_analysis_master_v1/`：当前 protocol_v2 的方法总览；
- `figures/archive/analysis/mogb_reproduction_gap_analysis_v2/`：MOGB 训练、论文差距和粒球分布；
- `figures/archive/analysis/mogb_operating_point_visuals_v1/`：MOGB Known/OOS 工作点；
- `figures/s2c_baseline_mogb_overview_v1/`：本报告新增的统一对照图。
- `figures/s2c_vs_mogb_mechanism_dashboard_v1/`：当前 S2C-Trainable-K1 与
  MOGB-MiniLM-Fair 的 45 单元严格配对、误差预算和组件桥；对应中文报告为
  `docs/analysis/S2C_VS_MOGB_MECHANISM_DASHBOARD_V1.md`。
- `figures/archive/analysis/s2c_mogb_operating_curve_attribution_v1/`：阈值无关排序、事后最优阈值和相同Known覆盖前沿；
  对应中文报告为 `docs/archive/analysis/S2C_MOGB_OPERATING_CURVE_ATTRIBUTION_V1.md`。
- `figures/archive/analysis/s2c_mogb_intent_structure_bridge_v1/`：逐 intent Known 恢复、粒球数、恢复集中度和 OOS
  代价；对应中文报告为 `docs/archive/analysis/S2C_MOGB_INTENT_STRUCTURE_BRIDGE_V1.md`。

## 3. MOGB 当前到底复现了什么

本地 exact 单格确实加载了作者公开仓库的 BERT、CE、递归粒球、最近子中心损失、平均半径和最近粒球推理；第三方源码保持 pinned，现代 PyTorch 兼容修复位于外部适配层。它属于‘官方逻辑的现代兼容复现’，不是作者原环境的逐字节复现。

StackOverflow/KIR=0.50/seed=0 本地结果为 Acc=75.17、F1-All=68.35、F1-U=79.97、F1-K=67.19；论文公开参考为 88.67/87.49/89.71/87.27。

## 4. 为什么本地结果与论文差很多

已经排除‘没有训练’和‘没有生成动态粒球’：旧单格 Known dev accuracy 达到 91.60%，最终生成 28 个粒球。主要可验证问题有两层：

1. 官方 `myloss.py` 将非负类别距离先 L1 归一化，再 `softmax(-distance)`；十个 Known 类时 true-class probability 的理论上限只有约 0.1105，最近子中心监督非常弱。
2. 最终平均距离半径边界过于保守：旧单格 OOS Recall=98.80%，但 Known Recall 只有 51.53%，Known→OOS=1449，主要差距来自过度拒绝 Known。
3. 作者原始 sample ID、Known 列表、旧依赖环境和完整数据生成链未恢复，因此即使算法逻辑相同，也不能证明当前 split 与论文逐样本一致。

## 5. 只修正子中心损失后的受控结果

本次只把 L1 归一化距离改成 `raw_distance / temperature=1.0`；BERT、数据、粒球、early stopping、平均半径和推理规则全部保持。它是 diagnostic ablation，不是 MOGB-official。

修正后 Acc=77.25、F1-All=72.64、F1-U=80.96、F1-K=71.81、Known Recall=59.20。相对旧本地运行，F1-All +4.29 pp，F1-K +4.62 pp，Known Recall +7.67 pp。

结论：**单独修正损失不足以解释或关闭论文差距**。即使修正后更好，也只能说明公开损失合同是复现差距的一项来源；若仍明显低于论文，剩余差距主要需要从边界 calibration、数据合同和旧运行时语义继续定位。

### corrected-loss checkpoint 上的半径归因

现已继续完成 Known-only 半径覆盖实验，但必须先说明复现边界：原 corrected 运行没有保存最终
`cluster3` 的 Python RNG 状态、中心向量和成员索引，因此只能在固定 checkpoint 上做确定性 33 球重建；
ball identity 与源 artifact 不一致，默认指标最大绝对差为1.87pp。以下结果是结构相近的诊断，不是原球结构
的严格重放。

| 工作点 | Known Recall | F1-U | F1-All | OOS误接受 |
|---|---:|---:|---:|---:|
| default mean radius | 58.63 | 79.76 | 71.42 | 197 |
| dev-known coverage 80% | 77.00 | 73.45 | 73.97 | 911 |
| dev-known coverage 95% | 87.17 | 35.68 | 63.80 | 2311 |

这进一步缩小了原因范围：默认平均半径确实过窄，但将 Known 覆盖恢复到高水平会快速吞入 OOS，无法恢复
论文的 `F1-U=89.71/F1-All=87.49` 工作点。因此当前差距至少同时包含损失动态范围、边界排序/校准、
粒球随机状态不可重放和作者数据/split未恢复，而不是一个半径常数的问题。完整证据见
`docs/archive/analysis/MOGB_CORRECTED_RADIUS_COVERAGE_V1.md`。

## 6. 当前能否说 S2C 超过 MOGB

不能把 fulltex 的历史 Cascade 数字与 MOGB 论文数字直接做公平排名，因为 Known 列表、split、backbone、训练监督和指标合同未完全对齐。当前可以严谨地说：

- 历史完整 S2C 在其旧主表中超过了当时列出的七个基线；
- 当前 `S2C-Trainable-K1` 在 protocol_v2 的统一 Gate/组件矩阵中优于冻结 K1/K2 与 MOGB-MiniLM-Fair；
- 本地官方逻辑 MOGB 未复现论文公开结果，且差距集中在 Known coverage；
- 还缺同一 split、同一 seeds、统一评估器下的完整 S2C Cascade、MOGB、ADB/DA-ADB 与 DCLOOS 主表，才能作新的 SOTA 结论。

当前 protocol_v2 的五 seed 组件矩阵中，`S2C-Trainable-K1` 跨九个 dataset×KIR 单元的平均 OOS F1/F1-All 为 85.75/83.36，`MOGB-MiniLM-Fair` 为 73.39/46.26；StackOverflow/KIR=.50 分别为 87.67/86.55 与 72.92/43.30。这证明当前 S2C Gate 优于冻结 MiniLM 的 MOGB 组件，不等于超过论文中的完整 BERT MOGB。

进一步的逐样本阈值曲线审计显示，S2C 的 AUROC 在45/45配对单元上更高，平均高`7.93pp`；即使两方法
分别使用测试事后最优阈值，S2C的OOS F1上限仍平均高`6.43pp`。因此当前公平组件优势不能只归因于
MOGB默认mean-radius阈值太保守；表示排序质量同样是主要贡献。事后阈值只作诊断，不进入正式结果。

历史 fulltex 的 StackOverflow/KIR=.50 `Ours` OOS F1 恰为89.71，MOGB论文同格公开 F1-U也为89.71；数值相同只是描述性巧合，因为样本、Known列表、训练骨干和完整评价合同尚未证明一致。

### 6.1 ADB 三数据集外部参照

已在相同 `protocol_v2_textoir_v1` 导出 split、KIR=.50、seed=`42,87,100` 上完成 9 个 ADB
BERT/TextOIR 兼容单元。该结果用于外部合同与工作点分析，不并入 MiniLM fair 排名：

| 数据集 | ADB OOS F1 | ADB F1-All | ADB Known Recall | Trainable−ADB OOS F1 | Trainable−ADB F1-All |
|---|---:|---:|---:|---:|---:|
| CLINC150 | 90.01±0.45% | 86.00±0.44% | 90.64±0.72% | +0.35 pp | −4.14 pp |
| Banking77 | 74.30±1.50% | 78.18±1.67% | 89.25±0.61% | +9.79 pp | +3.68 pp |
| StackOverflow | 87.36±1.61% | 85.66±1.59% | 80.78±1.44% | +0.85 pp | +1.03 pp |

因此不能把当前 Trainable-K1 概括为“综合上全面超过 ADB”：CLINC150 的 F1-All 和 Known Recall 仍落后。
逐 seed 重算、配对差值和 manifest 见 `results/analysis/archive/analysis/adb_cross_dataset_v1/ADB_CROSS_DATASET_REPORT.md`。

逐 intent 结构桥接进一步排除了“差距只是少数异常类别或缺球造成”的解释：九个 dataset×KIR 组中，
正向 Known 恢复 intent 的比例均为 100%，平均逐 intent Known 拒绝率降低 `46.70pp`；MOGB 完全缺少
selected ball 的 intent×seed 行仅占 `1.41%`。ball count 与恢复幅度的 Spearman 相关在
`-0.382` 到 `+0.445` 之间，没有跨数据集一致方向。也就是说，当前差距是广泛的表示排序与边界覆盖
问题，不是简单“少数类漏球”或“粒球越多越好”。

## 6.2 当前协议 Cascade 桥接的补充

为了避免把 Gate-only 与完整 Cascade 混为一谈，已在相同 `protocol_v2_textoir_v1` views 下为
CLINC150、Banking77 训练了 Known-only SmolLM Expert，并将其分别接入 Frozen K=1 和 Trainable K=1
Gate；StackOverflow 使用已完成的同协议桥接。三数据集的 18 个 dataset×seed×Gate 单元已经完成。

| 数据集 | Trainable K=1 OOS F1 | Frozen K=1 OOS F1 | 配对差值 | FA 降幅 |
|---|---:|---:|---:|---:|
| CLINC150 | 90.43%±0.62% | 89.31%±0.79% | +1.12pp | 2.86pp |
| Banking77 | 84.77%±0.71% | 79.58%±2.04% | +5.18pp | 10.00pp |
| StackOverflow | 86.71%±0.96% | 77.29%±5.10% | +9.42pp | 15.40pp |

这说明 Trainable K=1 的下游优势并非 StackOverflow 单一现象；但它仍然不是对 MOGB 论文 BERT
结果或 DCLOOS 外部 OOS 监督结果的公平排名。完整三数据集表、逐样本 Gate→Cascade 错误预算和图见
`docs/archive/analysis/CASCADE_BRIDGE_CROSS_DATASET_V1.md`。

## 7. 本次证据文件

- `results/analysis/archive/analysis/s2c_baseline_mogb_overview_v1/method_comparison.csv`
- `loss_contract_deltas.csv`
- `ball_level_comparison.csv`
- `figures/s2c_baseline_mogb_overview_v1/mogb_published_local_corrected_metrics.png`
- `mogb_error_budget.png`
- `mogb_training_contract_curves.png`
- `mogb_ball_distribution_comparison.png`
- `docs/archive/analysis/S2C_MOGB_INTENT_STRUCTURE_BRIDGE_V1.md`
- `figures/archive/analysis/s2c_mogb_intent_structure_bridge_v1/`

## 8. 当前下一步

冻结 MiniLM 的45单元与 corrected-loss BERT单格半径归因均已完成：扩大 mean radius 虽可恢复 Known
coverage，却会大幅增加 OOS false acceptance，不能恢复论文工作点。当前 Cascade 桥接也已覆盖三数据集，
因此下一步不再扫描半径或重复 MOGB MiniLM 矩阵；应收口作者数据、Known列表、旧依赖和最终
selected-ball RNG/中心状态，并将外部方法主表明确分为“同协议组件证据”和“论文公开参考”。缺少这些材料
时，当前运行只能标记为 `official_code_not_reproduced_under_available_materials`，不能用本地低分证明
MOGB 算法无效。

原始 checkpoint：`e43553d9d52a75277726a5d8256b9cca93df43dc1048cd694b9e8e26679137f0`；修正 checkpoint：`127bd347fe9912f81b771630ae8f5d70bd91348567fb8bfd3b2c418da68dfc22`。
