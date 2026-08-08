# MiniLM Trainable K=1 跨数据集控制实验

> 这是 `minilm_trainable_control_v1` 的受控实验，不扩展 K、不使用 OOS 训练、不用测试集选 checkpoint。CLINC150 和 Banking77 为本轮新运行，StackOverflow 复用已完成且同协议的 RACAL-v1 K=1 结果。

## 1. 配置

- protocol：`protocol_v2_textoir_v1`；KIR=0.50；K=1；对角 Mahalanobis；`mean_std` 半径。
- 表示：`all-MiniLM-L6-v2`，只训练最后两层和残差 projection。
- checkpoint：只用 Known calibration 的 `f1_k + 0.05 × Known Recall` 选择。
- 新运行：CLINC150/Banking77 × seeds 13, 42, 87，共 6 个单元，全部成功。

## 2. Frozen 与 Trainable

| 数据集 | Frozen OOS F1 | Trainable OOS F1 | 差值 | Frozen Known Recall | Trainable Known Recall | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| clinc150 | 89.31% | 90.43% | +1.12 pp | 75.60% | 74.27% | -1.33 pp |
| banking77 | 79.58% | 84.77% | +5.18 pp | 83.86% | 81.91% | -1.95 pp |
| stackoverflow | 77.29% | 86.71% | +9.42 pp | 83.71% | 83.92% | +0.21 pp |

## 3. 结果解释

- CLINC150：Trainable K=1 的 OOS F1 提升约 1.12 个百分点，但 Known Recall 下降约 1.33 个百分点；表示排序变好，但已知样本覆盖略有损失。
- Banking77：OOS F1 提升约 5.18 个百分点，Known Recall 下降约 1.95 个百分点；这是正向但伴随代价的改进，仍需校准和更多 seed 验证。
- StackOverflow：Trainable K=1 提升约 9.42 个百分点且 Known Recall略升；但此前 K=2 仍崩溃，表示收益局限于单中心。

## 4. 当前阶段结论

Trainable MiniLM 不是跨数据集无条件优于 Frozen：三个数据集的 OOS F1 均不低于 Frozen，但 CLINC150 和 Banking77 的 Known Recall 分别下降约 1.33 和 1.95 个百分点。当前证据支持“Known-only 表示适配有潜力”，不支持“无条件替代 Frozen”。下一批不应直接扩大 K，而应先分析训练表示对类内方差、半径和 Known Recall 的影响，再决定是否需要校准或表示训练范围消融。

## 5. 证据文件

- `results/diagnostics/minilm_trainable_control_v1/summary.csv`
- `results/diagnostics/minilm_trainable_control_v1/paired_deltas.csv`
- `figures/active_experiment_dashboard_v1/trainable_cross_dataset.png`
- `artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_control_v1/`
