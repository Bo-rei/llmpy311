# StackOverflow raw Gate error visualization v1

## 范围与证据边界

本报告只读取 `protocol_v2_textoir_v1` 已完成的逐样本 predictions，不训练、不调参、不改变阈值，也不覆盖历史 artifact。范围为 StackOverflow、KIR=0.50、seed=13/42/87。所有结果均为 Gate-only；不能替代完整 Gate–Router–Expert Cascade 结果。

分数方向统一为：数值越大越像 OOS。raw 文件中的 `sample_id` 在四种方法内逐 seed 完全对齐；原始文本没有复制到结果目录。

## 均值结果

| 方法 | OOS F1 | F1-All | Known Recall | FA rate | FR rate | AUROC | AUPR-OOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trainable K=1 | 0.8671 | 0.8565 | 0.8392 | 0.1114 | 0.1608 | 0.9175 | 0.8878 |
| Frozen K=1 | 0.7729 | 0.7860 | 0.8371 | 0.2654 | 0.1629 | 0.8819 | 0.8468 |
| Fixed K=2 | 0.6765 | 0.7681 | 0.9362 | 0.4526 | 0.0638 | 0.8945 | 0.8754 |
| MOGB fair component | 0.7319 | 0.4515 | 0.2834 | 0.0092 | 0.7166 | 0.8570 | 0.7983 |

## 逐样本机制解释

1. **Trainable K=1 的优势首先体现在分数排序和覆盖—拒识平衡。** 直接从 raw score 计算的 ROC/PR、Known/OOS 分布和指标可检查该优势是否只是阈值变化。若 Known 分数整体左移、OOS 分数右移且 FA 同时下降，说明表示训练改变了可分性，而不是单纯拒绝更多 Known。
2. **Fixed K=2 的风险是新增 OOS 误接收。** K=2 可能恢复更多 Known，但 union-style 多球接受区会让更多 OOS 通过至少一个球。`pairwise_transitions.csv` 和 `boundary_expansion_transitions.csv` 将“Trainable/K=1 正确、K=2 错误”以及“被 K=2 新接收的 OOS”分别计数，避免只看总体 F1。
3. **MOGB fair component 是冻结 MiniLM 下的动态粒球/平均半径组件，不是官方 BERT MOGB。** 它的低 FA 若伴随很高 FR/较低 Known Recall，应解释为更保守的工作点，而不是无条件优于当前 Trainable。
4. **intent 热图只用于机制诊断。** OOS 误接收按预测 intent 归因，Known 误拒绝按 gold intent 归因；这些表不能反过来选择 K、半径或阈值。

## 图表

- 分数分布：`figures/archive/analysis/raw_gate_error_visualization_v1/score_distributions.png`
- ROC/PR：`figures/archive/analysis/raw_gate_error_visualization_v1/roc_pr_curves.png`
- 指标对比：`figures/archive/analysis/raw_gate_error_visualization_v1/metric_bars.png`
- OOS 逐样本转移：`figures/archive/analysis/raw_gate_error_visualization_v1/oos_transitions.png`
- 接受区域扩张：`figures/archive/analysis/raw_gate_error_visualization_v1/boundary_expansion_waterfall.png`
- intent 风险热图：`figures/archive/analysis/raw_gate_error_visualization_v1/intent_error_heatmap.png`

## 外部 MOGB 结果的隔离

`external_official_mogb_summary.csv` 若存在，记录的是独立的官方 BERT 兼容复现；由于数据、表示、划分/评价合同与本节不同，不能与上述四种 raw predictions 合并排名。它只能用于复现失败/协议差异说明。

StackOverflow KIR=0.50 的官方 BERT 兼容复现（5 seed，非公平同协议）为：Known F1 `37.59±1.73`、Open F1 `72.05±0.39`、F1-score `40.72±1.61`、Accuracy `61.49±0.69`。这些数字仅说明当前现代兼容环境下的外部复现状态，不能与本报告的 Frozen MiniLM raw Gate 表直接排名。

## 可复现性

- 输出目录：`results/analysis/archive/analysis/raw_gate_error_visualization_v1`
- source hashes 保存在 `MANIFEST.json`；原始 predictions 保持在本地 artifacts，未复制到 Git 跟踪目录。
- 重算的 OOS F1、F1-All、F1-K、Accuracy、AUROC 和 AUPR-OOS 与四类运行目录中的 `metrics.json` 逐 seed 一致（浮点容差内）。
- 本报告未使用 test OOS 选择任何参数；使用 test labels 只进行事后可视化与错误归因。

## 不应作出的结论

不要据此声称 Trainable 已达到 SOTA、MOGB 已被公平击败、或 StackOverflow 的所有多中心方法均失败。这里证明的是同一 StackOverflow/KIR/seed 下的逐样本错误结构和协议内 Gate trade-off；外部端到端方法仍需按其监督条件单独标注。
