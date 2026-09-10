# Trainable MiniLM Known-only λ/K 受控实验

本阶段复用已经完成的 Trainable MiniLM checkpoint，只改变半径系数 λ，并在 Known calibration 上选择每个 dataset/seed/K 的最小可行 λ（calibration false-reject rate ≤ 0.05）。测试 OOS 仅用于冻结选择规则后的评价，不参与训练、选 λ 或选 K。

## 完整性

- 数据集：clinc150, banking77, stackoverflow；KIR=0.50；seed=[13, 42, 87]；K=[1, 2]；λ=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]。
- 计划与实际结果：108/108 个 λ 评价单元；每个 dataset×seed 12 行。
- test_used_for_selection=False；oos_used_for_training=False。
- 该阶段不重新训练 encoder，因此不会把半径修正误报为表示学习收益。

## Known-only 选择出的 λ

| 数据集 | K | 选择的 λ（seed 逐项） | 约束在三个 seed 是否都满足 | OOS F1 | Known Recall | False Acceptance | F1-All |
|---|---:|---|---|---:|---:|---:|---:|
| banking77 | 1 | 2.00,2.00,2.00 | False | 0.6971 ± 0.0238 | 0.9384 ± 0.0034 | 0.4325 ± 0.0288 | 0.7877 ± 0.0086 |
| banking77 | 2 | 2.00,2.00,2.00 | False | 0.7179 ± 0.0043 | 0.9263 ± 0.0078 | 0.3998 ± 0.0074 | 0.7884 ± 0.0042 |
| clinc150 | 1 | 2.00,2.00,2.00 | False | 0.9082 ± 0.0092 | 0.9081 ± 0.0103 | 0.1184 ± 0.0131 | 0.8682 ± 0.0137 |
| clinc150 | 2 | 2.00,2.00,2.00 | False | 0.9160 ± 0.0046 | 0.8893 ± 0.0065 | 0.0939 ± 0.0076 | 0.8689 ± 0.0093 |
| stackoverflow | 1 | 2.00,2.00,2.00 | False | 0.6868 ± 0.0533 | 0.9343 ± 0.0072 | 0.4410 ± 0.0645 | 0.7882 ± 0.0121 |
| stackoverflow | 2 | 1.25,1.50,1.25 | True | 0.5917 ± 0.0744 | 0.9570 ± 0.0078 | 0.5593 ± 0.0771 | 0.7421 ± 0.0126 |

## K=2−K=1 配对差值

| 数据集 | 选择方式 | OOS F1 Δ | F1-All Δ | Known Recall Δ | False Acceptance Δ | AUROC Δ |
|---|---|---:|---:|---:|---:|---:|
| banking77 | fixed_lambda_1 | +0.13 pp | -0.55 pp | -1.95 pp | -1.62 pp | -0.10 pp |
| banking77 | known_only_selected | +2.08 pp | +0.07 pp | -1.21 pp | -3.27 pp | +0.07 pp |
| clinc150 | fixed_lambda_1 | -0.28 pp | -1.54 pp | -3.26 pp | -1.20 pp | +0.35 pp |
| clinc150 | known_only_selected | +0.79 pp | +0.07 pp | -1.88 pp | -2.44 pp | +0.25 pp |
| stackoverflow | fixed_lambda_1 | -19.06 pp | -8.85 pp | +9.70 pp | +34.11 pp | -2.30 pp |
| stackoverflow | known_only_selected | -9.51 pp | -4.61 pp | +2.27 pp | +11.83 pp | -2.38 pp |

## 结论边界

- 若 λ=2 仍无法满足 Known calibration false-reject ≤5%，说明当前训练表示与半径统计之间存在契约失配，单纯扩大半径不能同时保留 Known 覆盖和 OOS 拒识。
- StackOverflow 即使使用 Known-only 选择的 K=2 λ（约 1.25–1.50），仍保持显著的 OOS F1 损失和较高 false acceptance；因此 K=2 退化不是仅由 λ=1 的偶然设置造成。
- 本阶段只能支持“Trainable MiniLM 改善 K=1，但没有普遍修复固定 K=2”的结论；不能据此宣称自适应多中心已解决。

## 文件

- `per_seed.csv`：全部 108 个逐 seed×K×λ 结果。
- `mean_std.csv`：λ 曲线均值和标准差。
- `selected.csv`：Known-only 选择结果。
- `k_delta_by_lambda.csv`：固定 λ=1 与 Known-only 选择下的 K=2−K=1 配对差值。
