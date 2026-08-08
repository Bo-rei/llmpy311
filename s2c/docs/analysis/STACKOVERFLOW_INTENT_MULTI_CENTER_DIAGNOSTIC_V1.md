# StackOverflow 固定 K=2 按意图机制诊断

更新时间：2026-08-06  
协议：`protocol_v2_textoir_v1`  
来源：`RACAL_V1_STAGE2_INTENT_DIAGNOSTICS.csv`；本报告只做已有诊断汇总，不重新运行实验。

## 结论

在当前诊断样本中，K=2 平均每个意图恢复约 `29.73` 个 Known 样本，却新增接受约 `102.53` 个 OOS 样本，净收益为 `-72.80`。因此，StackOverflow 的固定多中心问题不是聚类不稳定：平均 bootstrap ARI 为 `0.91`，而是稳定的子簇划分仍然把更多 OOS 带入接受区域。

这与“增加中心可以覆盖更多 Known”同时发生：平均 Known Recall 变化为 `0.097`，但新增 OOS 的代价更大。稳定性指标不能作为接受 K>1 的充分条件。

## 最差的意图

| intent | 新增 OOS | 恢复 Known | 净收益 | ARI |
|---|---:|---:|---:|---:|
| drupal | 539.0 | 32.0 | -507.0 | 0.99 |
| osx | 336.5 | 30.5 | -306.0 | 0.91 |
| cocoa | 269.0 | 36.0 | -233.0 | 0.98 |
| scala | 237.0 | 38.5 | -198.5 | 0.99 |
| spring | 195.0 | 32.0 | -163.0 | 0.96 |

## 相对安全的意图（仅限本诊断样本）

| intent | 新增 OOS | 恢复 Known | 净收益 | ARI |
|---|---:|---:|---:|---:|
| linq | 12.3 | 39.7 | 27.3 | 0.95 |
| excel | 0.0 | 20.3 | 20.3 | 0.55 |
| oracle | 25.7 | 43.0 | 17.3 | 0.99 |
| svn | 0.0 | 16.5 | 16.5 | 0.84 |
| apache | 0.3 | 13.7 | 13.3 | 0.92 |

## 解释边界

- 诊断文件只有当前 RACAL Stage-2 选取的 30 个 intent/seed 记录，不代表 StackOverflow 全部 20 个 intent；不能把该表写成全数据集的逐意图定律。
- `newly_accepted_oos_count` 是 K=2 相对于 K=1 的新增 OOS 接受量，直接用于解释 false-accept 代价；它不是 OOS F1 本身。
- ARI、silhouette、簇规模和半径只描述 Known 训练结构，没有使用 test OOS 选择参数。
- 这些结果支持“固定 K=2 在 StackOverflow 存在接受并集风险”，不支持“所有多中心方法都必然失败”。

## 原始证据与图

- [`RACAL_V1_STAGE2_INTENT_DIAGNOSTICS.csv`](../../results/diagnostics/racal_v1/stage2_fixed_k2/RACAL_V1_STAGE2_INTENT_DIAGNOSTICS.csv)
- [`intent_diagnostic_summary.csv`](../../results/analysis/stackoverflow_intent_diagnostic_v1/intent_diagnostic_summary.csv)
- [`intent_net_benefit.png`](../../figures/stackoverflow_intent_diagnostic_v1/intent_net_benefit.png)
- [`ari_vs_new_oos.png`](../../figures/stackoverflow_intent_diagnostic_v1/ari_vs_new_oos.png)
- [`intent_recovered_vs_oos.png`](../../figures/stackoverflow_intent_diagnostic_v1/intent_recovered_vs_oos.png)
- [`intent_diagnostic_heatmap.png`](../../figures/stackoverflow_intent_diagnostic_v1/intent_diagnostic_heatmap.png)

## Spearman 相关

相关矩阵已保存为 [`intent_diagnostic_correlations.csv`](../../results/analysis/stackoverflow_intent_diagnostic_v1/intent_diagnostic_correlations.csv)，仅作探索性机制分析，未进行多重比较校正。
