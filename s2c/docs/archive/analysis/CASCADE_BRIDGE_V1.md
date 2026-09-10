# 当前协议 Cascade 桥接实验（V1）

更新时间：2026-08-10
实验阶段：`cascade_bridge_v1`
范围：`protocol_v2_textoir_v1`、StackOverflow、KIR=0.50、seeds={13,42,87}。

## 这次补的是什么

这是一次**当前数据协议内的下游桥接实验**：同一份 `train_known` 训练 SmolLM Expert，使用
`calibration_known` 选择 checkpoint，然后分别接入已有的 Frozen K=1 和 Trainable K=1 Gate。
StackOverflow 在当前快照中只有一个 domain，因此 Router 是显式的常量路由，不伪造多领域 Router 结果。

它解决了此前的合同断点：Trainable Gate 不再直接拼接旧 v19 Router/Expert。它不重训 Gate，不修改
E2/E3/R1/MOGB artifacts，也不是 fulltex 历史 Cascade 的复现。

## 结果（均值 ± seed 标准差）

| Gate | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | FA | FR | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen K=1 | 77.29% ± 5.10pp | 76.55% ± 2.54pp | 77.93% ± 1.17pp | 76.11% ± 3.83pp | 83.71% ± 0.15pp | 26.54% ± 7.74pp | 16.29% ± 0.15pp | 88.19% ± 2.29pp |
| Trainable K=1 | 86.71% ± 0.96pp | 83.25% ± 0.75pp | 79.55% ± 0.82pp | 84.67% ± 0.77pp | 83.92% ± 0.39pp | 11.14% ± 2.02pp | 16.08% ± 0.39pp | 91.75% ± 1.07pp |

Trainable K=1 相对 Frozen K=1 的配对均值差值：OOS F1 `+9.42pp`、
F1-All `+6.70pp`、Known Recall `+0.21pp`、
FA `-15.40pp`、FR `-0.21pp`、
Accuracy `+8.57pp`。

## 与旧结果和外部基线的关系

| 结果层 | 能否直接和本次表格混排 | 原因 |
|---|---|---|
| 当前 Trainable/Frozen K=1 Cascade | 可以，在本报告内部 | 同一 TEXTOIR protocol、同一 views、同一 Expert checkpoint/seed |
| 当前 fair Gate matrix | 只能作 Gate-only 参照 | 没有 Router/Expert；本次只改变下游层 |
| fulltex 历史 `Ours` | 不可以直接排名 | 是旧合同的完整 Gate–Router–Expert Cascade |
| MOGB-MiniLM-Fair | 只能作同协议 Gate 组件参照 | 没有同一当前协议下的 Cascade Expert |
| MOGB 官方 BERT 兼容复现 | 不可以混成同协议主表 | 数据快照、BERT/训练损失、旧环境和论文配置未完全可恢复 |
| ADB/DCLOOS | 不能直接作为本表 SOTA 排名 | 训练监督、backbone 或外部 OOS 条件不同 |

## 机制解读

1. Trainable K=1 的收益仍主要来自 Gate 的分数排序和较低的 OOS 误接收；在当前 Cascade 中，
   同一 Expert 只在 Gate 接受的 Known 样本上工作，因此下游训练没有把旧 v19 数据差异混入结果。
2. Frozen K=1 的主要损失仍是 Gate false acceptance 较高，而不是 Expert 本身完全失效。
3. 该实验没有证明 Trainable K=1 已经超过 MOGB 论文或 DCLOOS。它只证明：在同一当前协议、
   同一 Expert 和相同已知/未知测试集合下，Trainable Gate 的下游桥接优于 Frozen Gate。

## 可复现性与限制

- `metrics.json` 中 `test_used_for_selection=false`、`oos_used_for_training=false`。
- 每个 seed 记录 train/calibration/test sample-id hash 和 Expert checkpoint SHA256。
- 完整模型 checkpoint 保留在 artifacts，轻量汇总和图在 `results/analysis/archive/analysis/cascade_bridge_v1/`、
  `figures/archive/analysis/cascade_bridge_v1/`。
- 只有三个 StackOverflow seed，尚未覆盖 CLINC150/Banking77，也没有 Router 多 domain 训练。
- 旧 fulltex Cascade、MOGB 官方论文数字和 DCLOOS 外部 OOS 结果仍需要单独的统一协议桥接，不能从本实验外推。

## 证据文件

- `results/analysis/archive/analysis/cascade_bridge_v1/per_seed.csv`
- `results/analysis/archive/analysis/cascade_bridge_v1/summary_mean_std.csv`
- `results/analysis/archive/analysis/cascade_bridge_v1/paired_effects.csv`
- `figures/archive/analysis/cascade_bridge_v1/frozen_vs_trainable_cascade_metrics.png`
- `figures/archive/analysis/cascade_bridge_v1/trainable_minus_frozen_cascade_effect.png`
- `../artifacts/s2c/runs/protocol_v2_textoir_v1/cascade_bridge_v1/PROVENANCE.json`
