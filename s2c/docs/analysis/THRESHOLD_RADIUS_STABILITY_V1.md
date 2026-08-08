# Threshold 与半径稳定性诊断 V1

> 本报告只做已完成 K=1 预测的事后诊断。阈值曲线使用测试 score 计算，但不用于选择正式阈值、不修改任何 run。

## 1. 主要发现

1. Trainable 与 Frozen 的 score 标度不同，因此固定 threshold=1 对两种表示并非完全等价的工作点。
2. 这可以解释部分 Trainable 与历史 fulltex 的差距，但不能解释 StackOverflow 固定 K=2 的 union false acceptance；后者是独立的多中心问题。
3. 半径 CV 只描述估计稳定性；稳定半径不等于 OOS 方向正确，必须和 Known Recall、false acceptance 一起看。

## 2. KIR=0.50 诊断性工作点

| 数据集 | 表示 | threshold=1 OOS F1 | threshold 网格最佳 OOS F1（oracle diagnostic） | 对应 threshold |
|---|---|---:|---:|---:|
| clinc150 | frozen | 89.32 | 89.32 | 1.00 |
| clinc150 | trainable | 90.44 | 91.53 | 1.05 |
| banking77 | frozen | 78.85 | 81.26 | 0.90 |
| banking77 | trainable | 83.56 | 84.07 | 0.95 |
| stackoverflow | frozen | 78.12 | 83.13 | 0.95 |
| stackoverflow | trainable | 87.67 | 88.19 | 0.95 |

## 3. 证据文件

- `results/analysis/threshold_radius_stability_v1/threshold_sensitivity_mean_std.csv`
- `results/analysis/threshold_radius_stability_v1/radius_stability_mean_std.csv`
- `figures/threshold_radius_stability_v1/`

## 4. 结论边界

- 网格最佳 threshold 只作为 score 标度敏感性诊断，不能写成正式调参结果；正式协议仍是 threshold=1。
- 若要提高当前 Trainable 与历史结果的可比性，下一步应预注册 Known-only 的 threshold/半径校准规则，再在新的独立验证池上运行，而不是读取 test oracle。
