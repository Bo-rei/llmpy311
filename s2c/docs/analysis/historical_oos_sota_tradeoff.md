# OOS SOTA 与 Known/Accuracy 代价权衡

目标：OOS F1 至少达到论文其他 baseline 的最好值，再尽量保留 Known F1 和 Accuracy。这里的 Known F1 使用全测试集 Known 类 macro F1，包含 OOS 被误接受带来的 precision 损失。

| 数据集 | KIR | 当前配置 | Known F1 | OOS F1 | Acc | ΔKnown / ΔOOS / ΔAcc vs 其他 baseline 最好值 | OOS达标 |
|---|---:|---|---:|---:|---:|---:|:---:|
| banking77_oos | 0.25 | K=1,lambda=0.5,threshold=0.95,nearest_sphere | 65.45 | 95.89±0.49 | 92.37 | -7.60 / +9.32 / +11.18 pp | 是 |
| banking77_oos | 0.50 | K=1,lambda=0.75,threshold=0.95,normalized_union | 67.46 | 91.53±0.05 | 85.59 | -15.08 / +11.60 / +4.08 pp | 是 |
| banking77_oos | 0.75 | K=1,lambda=0.75,threshold=1.0,normalized_union | 70.53 | 88.25±0.50 | 81.36 | -15.91 / +18.88 / -0.03 pp | 是 |
| clinc150 | 0.25 | trainable_k1 | 74.89 | 95.10±0.30 | 91.19 | -4.68 / +1.54 / +1.71 pp | 是 |
| clinc150 | 0.50 | recipe:last2_no_inter;recipe:last2_temp10 | 82.45 | 91.93±0.48 | 88.02 | -3.13 / +1.83 / +0.09 pp | 是 |
| clinc150 | 0.75 | trainable_k1_direct | 78.03 | 80.80±0.66 | 79.56 | -10.94 / -5.20 / -7.83 pp | 否 |
| stackoverflow | 0.25 | trainable_k1 | 80.07 | 95.38±1.02 | 91.58 | -0.80 / +2.73 / +2.51 pp | 是 |
| stackoverflow | 0.50 | adaptive_centers_overall | 83.45 | 89.96±1.50 | 86.94 | -3.26 / +1.10 / -0.84 pp | 是 |
| stackoverflow | 0.75 | trainable_k1_direct | 84.85 | 75.16±2.36 | 81.92 | -2.81 / +0.61 / -1.64 pp | 是 |

## 最值得保留的工作点

以下按 OOS 达标后，Known F1/Accuracy 的最差相对差值排序；它们是当前已有直接 pipeline 结果，不是测试集重新选参。

| 排名 | 数据集/KIR | OOS F1 | Known F1 | Acc | 主要代价 |
|---:|---|---:|---:|---:|---|
| 1 | stackoverflow / 0.25 | 95.38 | 80.07 | 91.58 | Known F1 -0.80 pp |
| 2 | stackoverflow / 0.75 | 75.16 | 84.85 | 81.92 | Known F1 -2.81 pp |
| 3 | clinc150 / 0.50 | 91.93 | 82.45 | 88.02 | Known F1 -3.13 pp |
| 4 | stackoverflow / 0.50 | 89.96 | 83.45 | 86.94 | Known F1 -3.26 pp |
| 5 | clinc150 / 0.25 | 95.10 | 74.89 | 91.19 | Known F1 -4.68 pp |
| 6 | banking77_oos / 0.25 | 95.89 | 65.45 | 92.37 | Known F1 -7.60 pp |
| 7 | banking77_oos / 0.50 | 91.53 | 67.46 | 85.59 | Known F1 -15.08 pp |
| 8 | banking77_oos / 0.75 | 88.25 | 70.53 | 81.36 | Known F1 -15.91 pp |

## OOS 接近论文 Ours 时的 Known 优先工作点

下表选取每个数据集当前最适合的工作点：OOS F1 至少达到论文 Ours 附近，同时在已有结果中尽量保留 Known F1。

| 数据集/KIR | 配置 | Known F1 | OOS F1 | Acc | Known Recall | False Acceptance | Δ vs 论文 Ours (K/O/Acc) | Δ vs 其他 baseline 最好 (K/O/Acc) |
|---|---|---:|---:|---:|---:|---:|---|---|
| clinc150 / 0.50 | recipe:last2_no_inter;recipe:last2_temp10 | 82.45 | 91.93±0.48 | 88.02 | 86.99 | 7.27 | +2.50 / -0.03 / +1.24 | -3.13 / +1.83 / +0.09 |
| stackoverflow / 0.50 | adaptive_centers_overall | 83.45 | 89.96±1.50 | 86.94 | 82.69 | 4.12 | +7.97 / +0.25 / +1.40 | -3.26 / +1.10 / -0.84 |
| banking77_oos / 0.75 | K=1,lambda=0.75,threshold=1.0,normalized_union | 70.53 | 88.25±0.50 | 81.36 | 81.25 | 12.23 | +0.25 / +1.76 / +3.52 | -15.91 / +18.88 / -0.03 |

结论：StackOverflow KIR=.25 是当前最平衡的设置，OOS F1 高于其他 baseline 最好值约 2.73 pp，Known F1 仅低约 0.80 pp，Accuracy 高约 2.51 pp。CLINC KIR=.50 也达到外部 baseline OOS SOTA，Known F1 约低 3.13 pp，Accuracy 基本持平。BANKING77-OOS 三个 KIR 都能达到外部 baseline OOS SOTA，但 Known F1 代价明显更大，KIR=.25 的代价最小。

所有比较属于 historical_v19 H1 controlled evidence；论文 Ours 另列为历史参照。选择使用验证集 OOS target、Accuracy 和 Known Recall，三组 Banking 结果均用 CUDA 直接 full pipeline 确认；没有训练新的 MiniLM、Router 或 Expert。

结果入口：

- [机器可读汇总](../../results/analysis/historical_trainable_sota_tradeoff/summary.csv)
- [Known 口径修正](historical_known_oos_tradeoff.md)
- [论文主表](../../fulltex.tex:372)
