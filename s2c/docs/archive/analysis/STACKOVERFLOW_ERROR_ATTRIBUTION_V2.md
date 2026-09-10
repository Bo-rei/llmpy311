# StackOverflow 逐样本错误归因（v2）

## 1. 目的

本阶段继续围绕实验解释，不新增模型、不重跑 E2/E3/R1，也不调整阈值。使用同一 `sample_id` 对齐 StackOverflow、KIR=0.50、5 个 seed 下的 7 组已有预测，回答：

- 哪些方法把同一批 OOS 错误地接收成 Known；
- 哪些方法主要牺牲 Known Recall；
- 固定 K=2 的错误是否集中在少数 intent；
- MOGB 的保守性来自什么代价；
- Trainable MiniLM 相对其它方法到底“多做对了什么”。

## 2. 数据和检查

| 项目 | 值 |
|---|---|
| protocol | `protocol_v2_textoir_v1` |
| dataset / KIR | `stackoverflow / 0.50` |
| seeds | 13、42、87、100、123 |
| 方法数 | 7 |
| 每个预测文件 | 6,000 条（3,000 Known + 3,000 OOS） |
| 对齐行数 | 210,000 条逐样本方法记录 |
| 正式判定 | score `<=1` 为 Known，`>1` 为 OOS |

逐个 seed 检查了：

1. 所有方法的 `sample_id` 顺序一致；
2. `gold_intent` 一致；
3. Known/OOS 数量均为 3,000/3,000；
4. 根据 score 重算的 OOS F1、Known Recall 与源 metrics 完全一致（浮点误差内）。

输出：

- [方法级指标](../../results/analysis/archive/analysis/stackoverflow_error_attribution_v2/method_metrics.csv)
- [逐 seed 错误转移](../../results/analysis/archive/analysis/stackoverflow_error_attribution_v2/pairwise_transitions_per_seed.csv)
- [错误转移汇总](../../results/analysis/archive/analysis/stackoverflow_error_attribution_v2/pairwise_transition_summary.csv)
- [intent 级误接收/误拒绝汇总](../../results/analysis/archive/analysis/stackoverflow_error_attribution_v2/intent_error_attribution_summary.csv)
- [provenance manifest](../../results/analysis/archive/analysis/stackoverflow_error_attribution_v2/MANIFEST.json)

## 3. 正式阈值下的总体错误来源

五个 seed 均值如下：

| 方法 | OOS F1 | Known Recall | OOS false acceptance | Known false rejection |
|---|---:|---:|---:|---:|
| Trainable K=1 | **0.8767** | 0.8389 | **0.0934** | 0.1611 |
| Frozen K=1 | 0.7655 | 0.8715 | 0.2971 | 0.1285 |
| Frozen K=2 | 0.6353 | 0.8689 | 0.4717 | 0.1311 |
| Random K=2 | 0.7588 | 0.8764 | 0.3098 | 0.1236 |
| MOGB-MiniLM | 0.7292 | **0.2709** | **0.0079** | **0.7291** |
| MOGB partition + ours boundary | 0.7925 | 0.5039 | 0.0186 | 0.4961 |
| Ours partition + MOGB boundary | 0.7335 | 0.5449 | 0.1569 | 0.4551 |

### 解释

- Trainable K=1 是当前最平衡的结果：它相对于 Frozen K=1 将 OOS false acceptance 从 29.71% 降到 9.34%，同时 Known false rejection 只从 12.85% 增加到 16.11%。
- Frozen K=2 的主要问题是 OOS false acceptance 达 47.17%，不是 Known Recall 不足；它在 Known 上甚至比 Trainable K=1 更宽松。
- MOGB-MiniLM 的 OOS false acceptance 只有 0.79%，但 Known false rejection 高达 72.91%。因此它不是“同工作点下更好”，而是明显偏向极保守拒绝。
- MOGB partition + ours boundary 仍保留较强保守性，说明动态粒球划分本身不能恢复正常 Known 覆盖。

## 4. 与 Trainable K=1 的 OOS 错误转移

在 OOS 测试样本上，比较“Trainable K=1 是否正确”与另一方法是否正确：

| 对比方法 | Trainable-only correct | Comparison-only correct | Both incorrect |
|---|---:|---:|---:|
| Frozen K=1 | 22.36% | 1.99% | 7.35% |
| Frozen K=2 | **39.04%** | 1.21% | 8.13% |
| Random K=2 | 23.44% | 1.80% | 7.54% |
| MOGB-MiniLM | 0.02% | **8.57%** | 0.77% |
| MOGB partition + ours | 0.03% | 7.51% | 1.83% |
| Ours partition + MOGB | 10.85% | 4.49% | 4.85% |

固定 K=2 相比 Trainable K=1 的差异几乎全部体现为：大量 OOS 样本只有 Trainable 能正确拒绝，而固定 K=2 将它们接收为 Known。MOGB 则相反：它很少误接收 OOS，但代价是把大量 Known 也拒绝掉。

## 5. Intent-level 错误来源

固定 K=2 吸收 OOS 最严重的 intent 是：

| intent | 占全部 OOS 的平均误接收比例 |
|---|---:|
| cocoa | 17.87% |
| sharepoint | 15.67% |
| osx | 12.13% |
| spring | 11.42% |
| scala | 11.23% |

这五个 intent 合计贡献超过一半的固定 K=2 OOS false acceptance。它们不是所有 intent 都均匀退化，而是少数 Known 边界成为 OOS 的主要“吸收器”。

Trainable K=1 下，同一批高风险 intent 的误接收明显下降；例如 `sharepoint` 从 Frozen K=1 的约 19.91% 降到约 1.38%，`cocoa` 从约 5.10% 降到约 3.23%。这支持“Trainable MiniLM 改善了 score separation”而不是简单收紧一个全局阈值的解释。

## 6. 可视化证据

- [OOS 错误转移热力图](../../figures/archive/analysis/stackoverflow_error_attribution_v2/stackoverflow_oos_transition_heatmap.png)
- [OOS false acceptance / Known false rejection 对照](../../figures/archive/analysis/stackoverflow_error_attribution_v2/stackoverflow_error_source_waterfall.png)
- [intent 级 OOS 吸收热力图](../../figures/archive/analysis/stackoverflow_error_attribution_v2/stackoverflow_intent_oos_acceptor_heatmap.png)
- [intent 级 Known 误拒绝热力图](../../figures/archive/analysis/stackoverflow_error_attribution_v2/stackoverflow_intent_known_reject_heatmap.png)

图形证据呈现了三个不同机制：

1. 固定多中心：Known 覆盖相对宽，但接受区域把 OOS 吸收进来；
2. MOGB-MiniLM：边界极窄，OOS 很少被接受，但 Known 大量被拒；
3. Trainable K=1：在这两个风险之间取得更好的平衡，并且高风险 intent 的 OOS 吸收更少。

## 7. 当前结论边界

本报告支持的结论是：在 StackOverflow/KIR=0.50 的已有同协议 fair 组件结果中，Trainable MiniLM K=1 的错误结构最平衡；它相对于固定 K=2 的主要优势是减少 OOS false acceptance，相对于 MOGB-MiniLM 的主要优势是恢复 Known 覆盖。

本报告不支持：

- 已达到跨数据集 SOTA；
- 已公平击败 ADB、DA-ADB 或 DCLOOS；
- MOGB 官方论文结果已被完整复现；
- 仅凭 StackOverflow 机制图就能证明某个新边界规则普适。

外部基线仍需在独立运行时恢复后，使用相同 split、seed 和评价脚本重新进入对比表。

## 8. 复现命令

```bash
python tools/analysis/build_stackoverflow_error_attribution_v2.py
```

该脚本只读取已有预测并输出 analysis-only 结果，不改写任何训练 artifact。
