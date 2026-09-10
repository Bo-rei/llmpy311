# MOGB 与 Trainable MiniLM 的同工作点可视化诊断（v1）

## 1. 目的

这是一份**实验阶段的机制诊断报告**，不是论文主结果，也不是新的模型训练。目标是把已有的 Trainable MiniLM、Frozen 单中心/固定双中心、随机划分和 MOGB 组件结果放到同一张“Known Recall 工作点”坐标系中，回答：

1. 在相同 Known Recall 下，谁的 OOS 拒识更好；
2. OOS F1 的提升是否伴随 false acceptance 增加；
3. MOGB 的动态粒球和边界规则分别造成了什么变化；
4. StackOverflow 的固定多中心失败是否仍然表现为 acceptance-region 过覆盖。

## 2. 重要协议边界

本诊断从各方法的**测试集 Known 分数**事后计算达到 0.75、0.85、0.90、0.95 Known Recall 所需的分数阈值，再在同一测试集上计算 OOS F1 和 false acceptance。因此它是阈值无关 ROC/PR 之外的可读性诊断，属于 `test_known_scores_posthoc`，不得用于：

- 选择模型、表示、中心数、半径或正式阈值；
- 改写 protocol_v2 的主结果；
- 声称消除了测试集调参；
- 与 calibration-only 的正式工作点混列。

原始 E2/E3、MiniLM 训练和 MOGB 组件 artifacts 未修改。输入为已有的 `cross_protocol_tradeoff_v1/per_seed.csv` 以及每个 run 的逐样本预测文件；每个源文件 SHA256 写入本阶段 manifest。

## 3. 输入覆盖

| 项目 | 覆盖 |
|---|---|
| protocol | `protocol_v2_textoir_v1` |
| 数据集 | CLINC150、Banking77、StackOverflow |
| KIR | 0.25、0.50、0.75 |
| seeds | 13、42、87、100、123 |
| 方法 | Trainable MiniLM K=1、Frozen single centroid、Frozen fixed K=2、random partition、MOGB-MiniLM、MOGB partition + ours boundary、ours partition + MOGB boundary |
| 逐方法逐 seed 记录 | 315 |
| 工作点 | Known Recall 0.75、0.85、0.90、0.95 |

输出：

- [逐 seed 工作点数据](../../results/analysis/archive/analysis/mogb_operating_point_visuals_v1/per_seed_targets.csv)
- [均值/标准差汇总](../../results/analysis/archive/analysis/mogb_operating_point_visuals_v1/summary.csv)
- [来源与哈希 manifest](../../results/analysis/archive/analysis/mogb_operating_point_visuals_v1/MANIFEST.json)

## 4. 关键结果：KIR=0.50、Known Recall≈0.85

下面的数字是五个 seed 的均值；它们是**事后工作点诊断值**，不是正式调参结果。

| 方法 | CLINC150 OOS F1 / FA | Banking77 OOS F1 / FA | StackOverflow OOS F1 / FA |
|---|---:|---:|---:|
| Trainable MiniLM K=1 | **0.9164 / 0.0715** | **0.8241 / 0.1964** | **0.8705 / 0.1132** |
| Frozen single centroid | 0.8876 / 0.1241 | 0.7271 / 0.3450 | 0.7934 / 0.2424 |
| Frozen fixed K=2 | 0.8883 / 0.1228 | 0.7274 / 0.3445 | 0.6692 / 0.4216 |
| Random partition | 0.8880 / 0.1235 | 0.7279 / 0.3438 | 0.7930 / 0.2431 |
| MOGB-MiniLM | 0.7935 / 0.2774 | 0.6622 / 0.4319 | 0.6860 / 0.3913 |
| MOGB partition + ours boundary | 0.8491 / 0.1896 | 0.7057 / 0.3746 | 0.7209 / 0.3413 |
| Ours partition + MOGB boundary | 0.8278 / 0.2246 | 0.6959 / 0.3873 | 0.4169 / 0.6928 |

### 直接观察

1. **Trainable MiniLM K=1 在三个数据集都处于最优的左上区域**：OOS F1 最高，同时 false acceptance 最低。它的优势不是通过把 Known 样本大量拒绝来换取 OOS F1；工作点已经固定在约 0.85 Known Recall。
2. **StackOverflow 上固定 K=2 的退化仍然存在**。在同一 Known Recall 下，Frozen single centroid 的 OOS F1 为 0.7934，固定 K=2 降到 0.6692，false acceptance 从 0.2424 升到 0.4216。
3. **MOGB-MiniLM 并没有在当前统一 MiniLM 协议下恢复 StackOverflow**；其 OOS F1 0.6860、FA 0.3913，仍比 Frozen single centroid 差。
4. **MOGB partition + ours boundary 比 MOGB-MiniLM 稳定，但仍不及 Trainable K=1**。这说明动态划分本身不是全部问题，MOGB 的平均半径/边界规则与表示之间也存在明显影响。
5. **ours partition + MOGB boundary 是最危险的组合**，StackOverflow 在同一工作点只有 0.4169 OOS F1，FA 达 0.6928；这支持“边界规则和多球接受区域的并集风险”解释。

## 5. 图形证据

- [StackOverflow Known Recall≈0.85 的 OOS F1/FA 对照](../../figures/archive/analysis/mogb_operating_point_visuals_v1/stackoverflow_kir050_target_085.png)
- [StackOverflow Known Recall≈0.90 的 OOS F1/FA 对照](../../figures/archive/analysis/mogb_operating_point_visuals_v1/stackoverflow_kir050_target_09.png)
- [StackOverflow 工作点曲线](../../figures/archive/analysis/mogb_operating_point_visuals_v1/stackoverflow_operating_curves.png)
- [三个数据集同 Known Recall≈0.85 的 OOS utility–open-space risk 图](../../figures/archive/analysis/mogb_operating_point_visuals_v1/cross_dataset_target085_tradeoff.png)

工作点曲线显示：Trainable K=1 在 Known Recall 从约 0.75 提升到 0.95 的整个区间都保持最高 OOS F1；MOGB 组件和固定 K=2 的曲线明显更低；ours partition + MOGB boundary 还出现随 Known Recall 提升而快速恶化的风险曲线。

## 6. 当前实验结论

这组诊断不能证明 Trainable MiniLM 是所有协议下的 SOTA，也不能代替 DCLOOS、ADB、DA-ADB 的统一同协议结果。它能确认的是：

- 在已有 `protocol_v2_textoir_v1` 的多 seed、三数据集组件结果中，Trainable MiniLM K=1 是当前最稳定的自有 Gate 候选；
- 当前 MOGB-MiniLM 和 MOGB/ours 边界混合结果没有超过它；
- StackOverflow 的主要失效信号不是单纯 Known Recall 过低，而是在同一 Known Recall 下仍然出现过高 OOS false acceptance；
- 因此下一步应优先做**工作点受控的基线扩展与错误归因**，而不是继续增加固定中心数或继续堆复杂多球规则；
- 外部 ADB/DA-ADB/DCLOOS 仍须保持协议隔离：ADB 已有 StackOverflow/KIR=.50 的三个有效 BERT/TextOIR
  兼容单元，DA-ADB 仍为 NaN/全类预测无效，DCLOOS 已找到外部 SQuAD 来源但默认预算超时、仅有
  reduced 单元。它们都不能直接与当前 Known-only MiniLM fair 行合并排名。

## 7. 复现命令

```bash
python tools/analysis/build_mogb_operating_point_visuals_v1.py
```

验证：

```bash
python -m py_compile tools/analysis/build_mogb_operating_point_visuals_v1.py
ruff check tools/analysis/build_mogb_operating_point_visuals_v1.py
```

## 8. 唯一下一步

先保留这组分析，不扩展新的模型结构。逐样本错误集合已经在
`stackoverflow_error_attribution_v2`、`trainable_mogb_open_intent_transitions_v1` 和
`s2c_mogb_operating_curve_attribution_v1` 中按相同 `sample_id` 对齐。下一步应继续维护这些图和
工作点证据；外部 ADB/DA-ADB/DCLOOS 仍按合同分层，ADB 三个兼容单元可作参照，DA-ADB 无效，
DCLOOS reduced 不进入 Known-only 主表。
