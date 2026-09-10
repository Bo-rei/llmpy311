# 当前协议三数据集 Cascade 桥接分析（V1）

> 这是当前 `protocol_v2_textoir_v1` 的下游桥接，不是 `fulltex.tex` 历史 Cascade 的逐字节复现；Gate、Expert、数据和选择协议均单独记录。

## 1. 实验范围与合同

- 数据集：CLINC150、Banking77、StackOverflow；KIR=0.50；seeds=13/42/87。
- Frozen K=1 使用已完成 E2 的 `test.jsonl`；Trainable K=1 使用当前协议 Known-only MiniLM control 的预测。
- 两个 Gate 变体共享同一数据 views、同一 SmolLM Expert 训练代码和同一 Expert checkpoint（同 dataset×seed）。
- Expert 只读 `train_known`，用 `calibration_known` 选 epoch；测试 OOS 不参与训练或选择。
- StackOverflow 的已有 `cascade_bridge_v1` 不覆盖；CLINC150/Banking77 写入独立 `cascade_bridge_multidataset_v1`。

## 2. 均值±标准差

| 数据集 | Gate | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | FA | FR | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| banking77 | frozen_k1 | 79.58%±2.04% | 74.34%±1.38% | 80.42%±0.13% | 76.32%±1.73% | 83.86%±0.90% | 23.46%±3.80% | 16.14%±0.90% | 87.71%±0.90% |
| banking77 | trainable_k1 | 84.77%±0.71% | 77.48%±0.63% | 80.26%±0.16% | 80.92%±0.66% | 81.91%±0.17% | 13.46%±1.22% | 18.09%±0.17% | 90.80%±0.64% |
| clinc150 | frozen_k1 | 89.31%±0.79% | 77.07%±1.54% | 79.59%±1.38% | 84.76%±1.03% | 75.60%±2.22% | 6.47%±0.41% | 24.40%±2.22% | 92.51%±0.13% |
| clinc150 | trainable_k1 | 90.43%±0.62% | 78.59%±1.05% | 79.54%±0.55% | 86.24%±0.85% | 74.27%±0.32% | 3.61%±1.06% | 25.73%±0.32% | 95.13%±0.61% |
| stackoverflow | frozen_k1 | 77.29%±5.10% | 76.55%±2.54% | 77.93%±1.17% | 76.11%±3.83% | 83.71%±0.15% | 26.54%±7.74% | 16.29%±0.15% | 88.19%±2.29% |
| stackoverflow | trainable_k1 | 86.71%±0.96% | 83.25%±0.75% | 79.55%±0.82% | 84.67%±0.77% | 83.92%±0.39% | 11.14%±2.02% | 16.08%±0.39% | 91.75%±1.07% |

## 3. Trainable 相对 Frozen 的配对差值

| 数据集 | OOS F1 | F1-All | Accuracy | Known Recall | FA | FR | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 5.18% | 3.13% | 4.60% | -1.95% | -10.00% | 1.95% | 3.09% |
| clinc150 | 1.12% | 1.52% | 1.48% | -1.33% | -2.86% | 1.33% | 2.62% |
| stackoverflow | 9.42% | 6.70% | 8.57% | 0.21% | -15.40% | -0.21% | 3.56% |

## 4. Gate→Cascade 错误预算

这里 FA 的分母是 OOS 数量，FR 的分母是 Known 数量。Trainable 的主要变化仍应先看 Gate 的 OOS false acceptance，不能把 Expert 的闭集分类变化误读成 OOS 检测来源。

| 数据集 | Gate | FA均值 | FR均值 | Known正确均值 | Known错误均值 |
|---|---|---:|---:|---:|---:|
| banking77 | frozen_k1 | 23.46% | 16.14% | 1209.3 | 65.3 |
| banking77 | trainable_k1 | 13.46% | 18.09% | 1212.0 | 33.0 |
| clinc150 | frozen_k1 | 6.47% | 24.40% | 1642.0 | 59.0 |
| clinc150 | trainable_k1 | 3.61% | 25.73% | 1656.3 | 14.7 |
| stackoverflow | frozen_k1 | 26.54% | 16.29% | 2314.3 | 197.0 |
| stackoverflow | trainable_k1 | 11.14% | 16.08% | 2483.7 | 34.0 |

## 5. 当前可以下的结论

1. `S2C-Trainable-K1` 的当前 Cascade 证据已从 StackOverflow 扩展到三个数据集，但样本量仍为每个数据集 3 seeds，不替代已有 Gate 5-seed 主矩阵。
2. 若 Trainable 在三个数据集均降低 FA 且 OOS F1 上升，说明当前最可重复的收益来源是 Trainable Gate 的 score separation，而不是 Expert 单独变强。
3. 这仍不是对论文 MOGB/DCLOOS 的公平 SOTA 排名：MOGB 论文使用 BERT、交替粒球训练和作者原始数据合同；DCLOOS 使用 pseudo-OOS 与外部 OOS。它们必须保留为不同监督合同。
4. `fulltex.tex` 的历史 `Ours` 是旧版完整 Cascade；当前 Trainable K=1 是当前协议候选，二者不能直接混排。

## 6. 证据路径

- 逐 seed：`results/analysis/archive/analysis/cascade_bridge_cross_dataset_v1/per_seed.csv`
- 配对差值：`results/analysis/archive/analysis/cascade_bridge_cross_dataset_v1/paired_effects.csv`
- 错误预算：`results/analysis/archive/analysis/cascade_bridge_cross_dataset_v1/error_budget.csv`
- 图：`figures/archive/analysis/cascade_bridge_cross_dataset_v1/`
- StackOverflow 单独报告：`docs/archive/analysis/CASCADE_BRIDGE_V1.md`
- MOGB 复现差距：`docs/archive/analysis/MOGB_REPRODUCTION_GAP_ANALYSIS_V2.md`
