# Known Recall 对齐的阈值工作点诊断 V1

> 这是事后诊断，不是正式调参结果。阈值使用 test 标签仅用于把不同 score 标度放到同一 Known Recall 横轴，不能用于论文主结果或后续模型选择。

## 为什么需要这项诊断

原生 MSP/Energy/kNN/LOF 的 Known-only conformal 阈值默认保留约 95% Known 样本；当前 Trainable 固定 threshold=1 的 Known Recall 明显更低。因此直接比较 OOS F1 会把不同拒识工作点混在一起。

## KIR=0.50、目标 Known Recall≈85%

| 数据集 | 方法 | 实际 Known Recall | OOS F1 | False Accept | False Reject |
|---|---|---:|---:|---:|---:|
| banking77 | energy | 85.00% | 75.01% | 31.09% | 15.00% |
| banking77 | knn | 85.00% | 78.89% | 25.28% | 15.00% |
| banking77 | lof | 85.00% | 73.52% | 33.29% | 15.00% |
| banking77 | msp | 85.00% | 80.61% | 22.60% | 15.00% |
| banking77 | trainable_k1 | 85.00% | 82.40% | 19.65% | 15.00% |
| clinc150 | energy | 84.98% | 91.08% | 8.19% | 15.02% |
| clinc150 | knn | 84.98% | 85.37% | 18.23% | 15.02% |
| clinc150 | lof | 84.98% | 86.00% | 17.15% | 15.02% |
| clinc150 | msp | 84.98% | 91.03% | 8.28% | 15.02% |
| clinc150 | trainable_k1 | 84.98% | 91.64% | 7.14% | 15.02% |
| stackoverflow | energy | 85.00% | 80.71% | 22.15% | 15.00% |
| stackoverflow | knn | 85.00% | 78.96% | 24.87% | 15.00% |
| stackoverflow | lof | 85.00% | 64.35% | 45.37% | 15.00% |
| stackoverflow | msp | 85.00% | 80.04% | 23.15% | 15.00% |
| stackoverflow | trainable_k1 | 85.00% | 87.05% | 11.32% | 15.00% |

## 结论边界

1. Trainable 的 OOS F1 优势不能只解释为阈值造成；需要看同一 Known Recall 工作点的曲线。
2. 若对齐后 Trainable 仍保持更高 OOS F1，说明表示/score 排序本身更有利；若优势消失，说明主要是当前 threshold=1 导致拒识工作点更激进。
3. 该诊断不改变正式 protocol_v2_textoir_v1 的 Known-only 选择规则。

- 数据：`results/analysis/operating_point_diagnostic_v1/per_seed_targets.csv`
- 图：`figures/operating_point_diagnostic_v1/`
