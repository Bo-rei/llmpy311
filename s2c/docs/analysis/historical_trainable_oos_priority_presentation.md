# Banking77-OOS OOS-first Trainable Gate 搜索

2026-09-08 更正：本页旧 Known F1 是真实 Known 子集指标，未包含 OOS false positives；请使用[全测试集口径修正及折中配置](historical_known_oos_tradeoff.md)进行论文对比。OOS F1 和 Accuracy 不变。

本轮只用 validation OOS F1 选择边界，Known F1、Accuracy、Known Recall 和 False Acceptance 记录为代价。

| KIR | OOS-first 配置 | 论文 Known F1 / OOS F1 / Acc | 当前默认 OOS F1 | OOS-first Known F1 / OOS F1 / Acc | 相对默认 OOS F1 | 相对论文 OOS F1 |
|---:|---|---:|---:|---:|---:|---:|
| 0.25 | K=1,lambda=1.5,threshold=0.8,normalized_union | 75.83 / 93.99 / 89.07 | 92.58±0.89 | 71.86 / 95.95 / 92.47 | +3.37 pp | +1.96 pp |
| 0.50 | K=1,lambda=0.5,threshold=0.95,normalized_union | 74.90 / 88.23 / 78.98 | 91.89±0.19 | 71.30 / 91.73 / 85.84 | -0.16 pp | +3.50 pp |
| 0.75 | K=1,lambda=0.25,threshold=1.05,nearest_sphere | 70.28 / 86.49 / 77.84 | 86.60±0.42 | 75.38 / 88.56 / 81.79 | +1.96 pp | +2.07 pp |

## 之前的‘均衡结果’与当前 OOS-first 结果

之前的均衡搜索只在 KIR=.50 进行：validation 仍优化 OOS F1，但要求 Known F1 和 Accuracy 相对 K=1 基线最多下降 1 pp。它不是三种 KIR 的统一最优，也不是简单平均三个指标。

| 结果线 | 选择目标 | 配置 | Known F1 | OOS F1 | Acc | Known Recall | False Acceptance | 相对均衡 OOS F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| paper_ours | historical reference | Historical Ours | 74.90 | 88.23 | 78.98 | — | — | -3.30 pp |
| balanced_parameter_search | validation OOS F1 with Known F1 and Accuracy <=1 pp guard | K=1,lambda=0.75,threshold=0.95,normalized_union | 75.84 | 91.53±0.05 | 85.59 | 71.63 | 7.85 | +0.00 pp |
| current_adaptive_default | current H1 adaptive-centers selected workpoint | adaptive_centers_overall | 71.19 | 91.89±0.19 | 85.96 | 65.03 | 5.36 | +0.36 pp |
| oos_priority | validation OOS F1 only; no Known guard | K=1,lambda=0.5,threshold=0.95,normalized_union | 71.30 | 91.73±0.08 | 85.84 | 63.97 | 5.37 | +0.20 pp |

关键结论：在 KIR=.50，OOS-first 相比均衡结果只增加约 +0.20 pp OOS F1，但 Known F1 下降约 4.54 pp、Known Recall 下降约 7.67 pp；False Acceptance 下降约 2.48 pp，Accuracy 仅增加约 0.25 pp。因此这次搜索确实沿着‘牺牲 Known coverage 换 OOS 拒识’方向移动，但 OOS F1 的增益很小，并没有出现同等幅度的跃升。当前 H1 adaptive default 的 OOS F1 为 91.89，仍略高于 KIR=.50 OOS-first 的 91.73。

KIR=.25 和 KIR=.75 的 OOS-first 结果分别是 95.95±0.48 和 88.56±0.48；它们相对各自论文 Ours 高约 +1.96 和 +2.07 pp，但不能与 KIR=.50 的均衡结果直接当作同一个 operating point。

KIR=.25 达到 95.95±0.48 OOS F1，相对历史 Ours 93.99 提升 +1.96 pp；Known F1 约 71.86，Known Recall 约 61.53%。
KIR=.50 的本轮 OOS-first 配置为 91.73±0.08，低于此前 adaptive-centers 的 91.89；KIR=.75 达到 88.56±0.48，相对历史 Ours 提升约 2.07 pp。

搜索范围：K={1,2,3,5}，lambda={.25,.5,.75,1,1.25,1.5,2,2.5}，threshold=.20–1.20，两种 acceptance mode。测试集只在 validation 选择锁定后确认。

结果入口：

- [三 KIR 汇总](../../results/analysis/historical_trainable_oos_priority_search/summary.csv)
- [KIR=.50 均衡 vs OOS-first 汇总](../../results/analysis/historical_trainable_oos_priority_search/balanced_vs_oos_priority_kir50.csv)
- [原始均衡参数搜索汇报](historical_trainable_parameter_full_pipeline_presentation.md)
- [逐 seed 结果](../../results/analysis/historical_trainable_oos_priority_search/per_seed.csv)
- [KIR=.25 manifest](../../results/analysis/historical_trainable_oos_priority_search/kir25/MANIFEST.json)
- [KIR=.50 manifest](../../results/analysis/historical_trainable_oos_priority_search/kir50/MANIFEST.json)
- [KIR=.75 manifest](../../results/analysis/historical_trainable_oos_priority_search/kir75/MANIFEST.json)
