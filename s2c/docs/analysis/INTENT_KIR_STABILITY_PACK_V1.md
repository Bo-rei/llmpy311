# 意图级 KIR 稳定性与多中心收益分析

更新时间：2026-08-06  
协议：`protocol_v2_textoir_v1`  
性质：analysis-only；来源是已完成的 `adaptive_k/intent_level.csv` oracle/test-sensitivity 审计，不用于正式选择 K。

## 结论

数据集级别的 KIR/K 曲线会掩盖意图异质性：同一数据集内只有部分意图在 oracle 口径下从 K>1 获益，且这种选择通常不能跨 seed 稳定重复。该结果支持“每个意图的几何结构不同”，但不等于已经构造出一个无泄漏的 adaptive-K 规则。

| 数据集 | 诊断行数 | oracle 中 K>1 比例 | 同时 OOS 增益且 Known Recall 下降≤1pp 的比例 | 平均 OOS F1 变化 | 平均 Known Recall 变化 |
|---|---:|---:|---:|---:|---:|
| banking77 | 4230 | 65.3% | 39.8% | 5.80pp | 0.82pp |
| clinc150 | 8250 | 45.2% | 35.7% | 3.08pp | 2.25pp |
| stackoverflow | 1100 | 34.2% | 24.7% | 0.99pp | 0.27pp |

## 如何解读

- `best_k` 是利用测试敏感性审计得到的 oracle 诊断量；不能用于训练、阈值、结构选择或正式主表。
- `safe_gain_oracle` 只是一个描述性筛选：OOS F1 增加且 Known Recall 降幅不超过 1pp；它不是已经验证的 calibration 规则。
- 真正的 adaptive-K 需要在 Known train/calibration 上预注册规则，并在冻结后只评估一次 test。
- 如果某意图的 `best_k` 在不同 seed 间反复变化，说明固定 K 或简单 oracle 选择不够稳定。

## 证据文件

- [`intent_level.csv`](../../results/diagnostics/adaptive_k/intent_level.csv)
- [`intent_kir_summary.csv`](../../results/analysis/intent_kir_stability_pack_v1/intent_kir_summary.csv)
- [`intent_kir_seed_stability.csv`](../../results/analysis/intent_kir_stability_pack_v1/intent_kir_seed_stability.csv)
- [`oracle_multicenter_rate_heatmap.png`](../../figures/intent_kir_stability_pack_v1/oracle_multicenter_rate_heatmap.png)
- [`oracle_gain_by_kir.png`](../../figures/intent_kir_stability_pack_v1/oracle_gain_by_kir.png)
- [`oracle_oos_known_tradeoff.png`](../../figures/intent_kir_stability_pack_v1/oracle_oos_known_tradeoff.png)
- [`oracle_multicenter_seed_stability.png`](../../figures/intent_kir_stability_pack_v1/oracle_multicenter_seed_stability.png)
