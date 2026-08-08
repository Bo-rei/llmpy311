# KIR=0.50 方法对比（协议分层）

本报告只整理已有结果，不重新训练。比较先按数据、表示、监督条件和 seed 数分层；不同合同的结果不合并为单一 SOTA 排名。

## StackOverflow 当前协议结果

| 方法 | OOS F1 | Known Macro-F1 | Known Recall | F1-All | Accuracy | 结果层 |
|---|---:|---:|---:|---:|---:|---|
| Trainable K=1 | 86.71% | 85.55% | 83.92% | 85.65% | 85.82% | same protocol / Trainable MiniLM / 3 seeds |
| Trainable K=2 | 67.65% | 77.72% | 93.62% | 76.81% | 71.56% | same protocol / Trainable MiniLM / 3 seeds |
| Single centroid | 76.55% | 80.32% | 87.15% | 79.98% | 76.58% | same protocol / frozen MiniLM / 5 seeds |
| Fixed K=2 | 63.53% | 73.68% | 86.89% | 72.76% | 66.93% | same protocol / frozen MiniLM / 5 seeds |
| Random partition | 75.88% | 80.20% | 87.64% | 79.80% | 76.10% | same protocol / frozen MiniLM / 5 seeds |
| MOGB-MiniLM | 72.92% | 40.34% | 27.09% | 43.30% | 63.06% | same protocol / frozen MiniLM / 5 seeds |
| MOGB partition + s2c boundary | 79.25% | 61.75% | 50.39% | 63.34% | 74.07% | same protocol / frozen MiniLM / 5 seeds |
| ADB | 89.47% | 87.44% | — | 87.63% | 88.53% | external compatibility / BERT or pseudo-OOS |
| DA-ADB | 90.90% | 89.06% | — | 89.23% | 90.07% | external compatibility / BERT or pseudo-OOS |
| BRAK | 81.34% | 79.97% | 83.33% | 80.10% | 78.90% | external compatibility / BERT or pseudo-OOS |

## 解释

- Trainable K=1 是当前协议下的 3-seed Known-only 表示训练结果；它相对当前 Frozen K=1 有稳定收益，但与 5-seed frozen fair rows 的 seed 数不同。
- MOGB-MiniLM、固定 K、随机分簇和 MOGB 分区/边界替换属于同一 frozen MiniLM 组件层，可用于分析分区和半径贡献。
- ADB/DA-ADB 是 BERT/兼容单格结果，BRAK 也使用不同实验合同；它们显示性能差距，但不能直接证明 Trainable MiniLM 方法优于或劣于这些方法。
- DCLOOS/MOGB 官方行若缺少可比 OOS F1，则保留为复现/兼容性证据，不伪造排名。

## 当前可支持的结论

1. 在当前统一协议内部，Trainable MiniLM K=1 是比 Frozen K=1 更强的表示基线。
2. Trainable K=2 在 StackOverflow 明显退化，不能把 K=1 表示收益归因于多中心。
3. MOGB fair 结果能回答组件机制问题，但不能替代作者 BERT 原始协议复现。
4. 要宣称超过 ADB/DA-ADB/DCLOOS，必须先统一表示、监督条件、known list、seed 和评价器；当前证据不足。

## 文件

- `results/analysis/kir50_method_comparison_v1/rows.csv`：逐方法分层表。
- `results/analysis/kir50_method_comparison_v1/mean_std.csv`：按数据集/方法的轻量汇总。
- `figures/active_experiment_dashboard_v1/kir50_method_layers.png`：协议分层柱状图。
- `figures/active_experiment_dashboard_v1/kir50_method_tradeoff.png`：Known/OOS 权衡图。
