# MOGB 粒球筛选缺类安全回补归因 V1

## 结论

本实验完成 45 个 dataset×KIR×seed 粒球重拟合单元和 180 个正式评分单元。原始结构在论文默认阈值下与 `mogb_known_calibration_attribution_v1` 的指标最大绝对差为 1e-12 以内；没有缺类的单元在加入回补逻辑后最大指标变化为 `0`。

实验只改一个变量：若 MOGB 叶球筛选导致某个注册 Known 类没有任何 selected ball，就用该类 `train_known` 样本建立一个单中心、欧氏平均半径回补球。它不改变 encoder、已有粒球、距离和最近球推理；cal-80 阈值只由 `calibration_known` 选择，test Known/OOS 从未参与结构或工作点选择。

结果表明：**缺类筛选确实直接损伤被遗漏类的 Known 分类，但它不是 S2C 与 MOGB 综合差距的主要来源。** 回补能恢复一部分被遗漏 Known，同时也为 OOS 新增接受通道；即使回补后，当前 S2C Trainable K=1 在 KIR=.50 的 Known/OOS 平衡仍明显更好。

StackOverflow 最能暴露这种结构风险：默认窄半径下回补使 Known Recall 平均提高 `8.40` pp，但 false acceptance 同时增加 `6.23` pp，AUROC 下降 `-13.87` pp；在两个方法分别用 Known calibration 对齐到 80% 工作点后，回补反而使 OOS F1 下降 `15.66` pp、false acceptance 增加 `18.61` pp。这说明问题不是简单“少了几个球”，而是新增局部接受区域改变了全局排序和开放空间风险。

## 受影响单元：论文默认阈值下的配对变化

| 数据集 | 受影响单元 | OOS F1 (pp) | F1-All (pp) | Known Recall (pp) | False acceptance (pp) |
|---|---:|---:|---:|---:|---:|
| clinc150 | 8 | +0.23 | +0.73 | +0.96 | +0.09 |
| banking77 | 3 | +0.60 | +1.82 | +1.80 | +0.09 |
| stackoverflow | 5 | +0.16 | +6.59 | +8.40 | +6.23 |

这里只汇总确实缺类的单元；未缺类的 29 个单元是严格零变化对照。cal-80 的完整变化位于 `affected_summary.csv`。

## 样本转移：恢复 Known 的代价

| 数据集 | 恢复 Known | 新增误收 OOS | 缺类 Known 正确（前） | 缺类 Known 正确（后） | 被回补球直接误收 OOS |
|---|---:|---:|---:|---:|---:|
| banking77 | 125 | 2 | 0 | 104 | 2 |
| clinc150 | 230 | 18 | 0 | 158 | 18 |
| stackoverflow | 1726 | 580 | 0 | 1276 | 580 |

这些计数按受影响 dataset×KIR×seed 单元求和，不是独立语料规模。它们用于解释机制，不用于调参。

## KIR=.50 同协议工作点

| 数据集 | 方法 | OOS F1 | F1-All | Known Recall | False acceptance |
|---|---|---:|---:|---:|---:|
| clinc150 | S2C Trainable K=1 | 90.44 | 81.82 | 74.44 | 3.69 |
| clinc150 | MOGB cal-80 | 83.70 | 76.38 | 80.69 | 18.92 |
| clinc150 | MOGB cal-80 + 缺类回补 | 84.15 | 76.87 | 81.01 | 18.31 |
| banking77 | S2C Trainable K=1 | 83.56 | 81.67 | 82.21 | 15.74 |
| banking77 | MOGB cal-80 | 70.39 | 72.10 | 82.03 | 36.10 |
| banking77 | MOGB cal-80 + 缺类回补 | 70.39 | 72.10 | 82.03 | 36.10 |
| stackoverflow | S2C Trainable K=1 | 87.67 | 86.55 | 83.89 | 9.34 |
| stackoverflow | MOGB cal-80 | 74.64 | 73.15 | 80.28 | 27.74 |
| stackoverflow | MOGB cal-80 + 缺类回补 | 70.30 | 73.84 | 80.01 | 32.02 |

在 45 个 KIR/seed 配对中，S2C Trainable K=1 相对回补后的 MOGB cal-80 仍取得 OOS F1 胜出 `45/45`。因此“部分 Known 类没有 selected ball”是可验证缺陷，但不能单独解释 MOGB-Fair 的弱排序和工作点；剩余差距仍主要指向子中心表示信号、粒球几何和平均半径边界的联合失配。

## 图表

- `figures/archive/analysis/mogb_selected_class_rescue_v1/affected_cell_metric_delta.png`
- `figures/archive/analysis/mogb_selected_class_rescue_v1/known_recovery_vs_oos_cost.png`
- `figures/archive/analysis/mogb_selected_class_rescue_v1/missing_class_correct_recovery.png`
- `figures/archive/analysis/mogb_selected_class_rescue_v1/rescue_vs_trainable_k1.png`

## 证据边界

- 这是 MOGB-Fair 组件归因，不是作者 BERT 完整 MOGB 的修正版复现。
- 回补规则由 train Known 标签和 embedding 确定；没有根据 test 指标决定是否回补。
- 每个遗漏类只补一个球，目的是隔离“类别无支持”缺陷，不宣称这是新的模型或最佳修复。
- Manifest SHA256：`0968ada93bcbdb04b19bc5b733c4451267feba357cdaf1820f1369152717a05e`。
