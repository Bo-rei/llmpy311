# MiniLM Trainable 五 seed 公平对比与可视化 V1

> 本报告把 Trainable K=1 补齐到与 MOGB 公平矩阵相同的五个 seed（13/42/87/100/123），用于方法和机制对比。它不把历史 `fulltex.tex` 数字伪装成同协议结果，也不把 MOGB 组件适配称为官方复现。

## 1. 完成状态

- Trainable K=1：45/45；新增扩展：18/18；失败、缺失、重复、无效指标：0。
- 范围：CLINC150、Banking77、StackOverflow；KIR=0.25/0.50/0.75；五个 seed；仅 K=1。
- 训练：MiniLM 最后两层 + residual projection；Known train 训练，Known calibration 选 checkpoint。
- 测试 OOS 未用于训练、checkpoint、阈值或边界选择。
- 新增 artifact：`../artifacts/s2c/runs/protocol_v2_textoir_v1/minilm_trainable_kir_sweep_extension_v1/`。

## 2. Trainable K=1 五 seed 结果

| 数据集 | KIR | OOS F1 | F1-All | Known Recall | False Accept | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| banking77 | 0.25 | 91.65±1.80 | 78.51±2.56 | 81.84±0.78 | 10.33±3.36 | 92.30±1.40 |
| banking77 | 0.50 | 83.56±1.83 | 81.67±1.05 | 82.21±0.48 | 15.74±3.43 | 90.14±1.11 |
| banking77 | 0.75 | 68.65±2.71 | 84.13±0.77 | 82.41±0.41 | 19.61±4.58 | 88.90±1.70 |
| clinc150 | 0.25 | 95.16±0.65 | 79.69±1.95 | 73.63±0.84 | 3.24±1.10 | 95.89±1.21 |
| clinc150 | 0.50 | 90.44±0.44 | 81.82±0.92 | 74.44±0.34 | 3.69±0.76 | 95.33±0.55 |
| clinc150 | 0.75 | 82.85±0.51 | 82.91±0.77 | 75.12±0.87 | 4.02±0.40 | 95.00±0.42 |
| stackoverflow | 0.25 | 95.48±1.21 | 87.64±2.37 | 84.29±0.59 | 3.85±2.14 | 95.30±1.61 |
| stackoverflow | 0.50 | 87.67±1.66 | 86.55±1.30 | 83.89±0.36 | 9.34±3.26 | 92.35±1.49 |
| stackoverflow | 0.75 | 76.30±2.91 | 87.28±0.68 | 84.12±0.67 | 8.80±5.97 | 92.90±1.01 |

## 3. 与同一 E2 Frozen K=1 的配对差值

| 数据集 | KIR | Trainable−Frozen OOS F1 | Trainable−Frozen Known Recall | Trainable−Frozen False Accept |
|---|---:|---:|---:|---:|
| banking77 | 0.25 | +2.47 pp | -1.21 pp | -4.68 pp |
| banking77 | 0.50 | +4.72 pp | -2.05 pp | -9.14 pp |
| banking77 | 0.75 | +6.94 pp | -2.90 pp | -15.71 pp |
| clinc150 | 0.25 | +0.45 pp | -1.33 pp | -1.17 pp |
| clinc150 | 0.50 | +1.12 pp | -1.37 pp | -2.88 pp |
| clinc150 | 0.75 | +1.38 pp | -0.70 pp | -3.38 pp |
| stackoverflow | 0.25 | +5.06 pp | +0.77 pp | -8.99 pp |
| stackoverflow | 0.50 | +9.55 pp | +0.43 pp | -15.69 pp |
| stackoverflow | 0.75 | +10.50 pp | +0.13 pp | -18.33 pp |

## 4. 与 Frozen/MOGB 组件的同协议上下文

MOGB 公平矩阵固定 Frozen MiniLM，但其组件使用欧氏距离/平均半径或 MOGB 分区；Trainable 使用最后两层适配 + 对角马氏距离/mean+std。因此下面是协议分层比较，不是无条件排名。

| 数据集 | KIR | 方法 | OOS F1 | F1-All | Known Recall | False Accept |
|---|---:|---|---:|---:|---:|---:|
| banking77 | 0.25 | Frozen fixed K=2 | 85.78 | 71.10 | 81.89 | 20.33 |
| banking77 | 0.25 | MOGB partition | 90.30 | 55.88 | 39.97 | 1.50 |
| banking77 | 0.25 | MOGB partition + s2c boundary | 92.06 | 71.15 | 61.68 | 4.02 |
| banking77 | 0.25 | s2c partition + MOGB boundary | 90.68 | 65.49 | 53.87 | 4.51 |
| banking77 | 0.25 | Random partition | 85.00 | 69.31 | 84.74 | 22.31 |
| banking77 | 0.25 | Frozen single | 85.17 | 69.29 | 83.89 | 21.84 |
| banking77 | 0.25 | Trainable K=1 | 91.65 | 78.51 | 81.84 | 10.33 |
| banking77 | 0.50 | Frozen fixed K=2 | 75.46 | 76.16 | 82.55 | 29.05 |
| banking77 | 0.50 | MOGB partition | 74.99 | 48.60 | 33.41 | 1.09 |
| banking77 | 0.50 | MOGB partition + s2c boundary | 79.40 | 64.81 | 52.18 | 3.50 |
| banking77 | 0.50 | s2c partition + MOGB boundary | 78.86 | 65.68 | 53.25 | 5.24 |
| banking77 | 0.50 | Random partition | 71.62 | 74.72 | 85.96 | 36.50 |
| banking77 | 0.50 | Frozen single | 72.43 | 74.51 | 85.01 | 34.85 |
| banking77 | 0.50 | Trainable K=1 | 83.56 | 81.67 | 82.21 | 15.74 |
| banking77 | 0.75 | Frozen fixed K=2 | 59.31 | 80.26 | 83.14 | 36.08 |
| banking77 | 0.75 | MOGB partition | 47.63 | 42.39 | 28.77 | 0.76 |
| banking77 | 0.75 | MOGB partition + s2c boundary | 53.13 | 58.61 | 45.86 | 4.05 |
| banking77 | 0.75 | s2c partition + MOGB boundary | 55.07 | 65.67 | 53.56 | 8.13 |
| banking77 | 0.75 | Random partition | 53.38 | 78.89 | 87.28 | 49.32 |
| banking77 | 0.75 | Frozen single | 53.66 | 78.21 | 86.41 | 48.00 |
| banking77 | 0.75 | Trainable K=1 | 68.65 | 84.13 | 82.41 | 19.61 |
| clinc150 | 0.25 | Frozen fixed K=2 | 94.47 | 78.01 | 74.35 | 4.73 |
| clinc150 | 0.25 | MOGB partition | 92.38 | 52.24 | 37.11 | 0.66 |
| clinc150 | 0.25 | MOGB partition + s2c boundary | 94.13 | 70.16 | 60.18 | 2.24 |
| clinc150 | 0.25 | s2c partition + MOGB boundary | 92.64 | 56.67 | 41.89 | 1.18 |
| clinc150 | 0.25 | Random partition | 94.24 | 78.78 | 78.84 | 6.17 |
| clinc150 | 0.25 | Frozen single | 94.21 | 78.44 | 77.95 | 6.04 |
| clinc150 | 0.25 | Trainable K=1 | 95.16 | 79.69 | 73.63 | 3.24 |
| clinc150 | 0.50 | Frozen fixed K=2 | 89.20 | 80.09 | 75.14 | 6.44 |
| clinc150 | 0.50 | MOGB partition | 81.32 | 44.95 | 31.57 | 0.90 |
| clinc150 | 0.50 | MOGB partition + s2c boundary | 85.56 | 64.60 | 53.34 | 2.50 |
| clinc150 | 0.50 | s2c partition + MOGB boundary | 83.32 | 56.41 | 42.24 | 1.69 |
| clinc150 | 0.50 | Random partition | 89.07 | 80.75 | 79.48 | 8.96 |
| clinc150 | 0.50 | Frozen single | 88.94 | 80.27 | 78.76 | 8.82 |
| clinc150 | 0.50 | Trainable K=1 | 90.44 | 81.82 | 74.44 | 3.69 |
| clinc150 | 0.75 | Frozen fixed K=2 | 81.50 | 81.43 | 75.80 | 7.32 |
| clinc150 | 0.75 | MOGB partition | 65.93 | 41.80 | 29.20 | 0.83 |
| clinc150 | 0.75 | MOGB partition + s2c boundary | 71.97 | 59.60 | 48.09 | 1.89 |
| clinc150 | 0.75 | s2c partition + MOGB boundary | 70.04 | 56.93 | 42.85 | 1.88 |
| clinc150 | 0.75 | Random partition | 82.21 | 82.35 | 80.05 | 10.22 |
| clinc150 | 0.75 | Frozen single | 81.75 | 81.74 | 79.07 | 10.10 |
| clinc150 | 0.75 | Trainable K=1 | 82.85 | 82.91 | 75.12 | 4.02 |
| stackoverflow | 0.25 | Frozen fixed K=2 | 86.66 | 76.80 | 85.21 | 19.47 |
| stackoverflow | 0.25 | MOGB partition | 89.44 | 54.07 | 32.61 | 0.94 |
| stackoverflow | 0.25 | MOGB partition + s2c boundary | 92.60 | 73.45 | 59.11 | 2.03 |
| stackoverflow | 0.25 | s2c partition + MOGB boundary | 90.64 | 69.05 | 54.15 | 4.44 |
| stackoverflow | 0.25 | Random partition | 89.11 | 80.86 | 86.09 | 15.68 |
| stackoverflow | 0.25 | Frozen single | 89.48 | 81.01 | 85.61 | 14.96 |
| stackoverflow | 0.25 | Trainable K=1 | 95.48 | 87.64 | 84.29 | 3.85 |
| stackoverflow | 0.50 | Frozen fixed K=2 | 63.53 | 72.76 | 86.89 | 47.17 |
| stackoverflow | 0.50 | MOGB partition | 72.92 | 43.30 | 27.09 | 0.79 |
| stackoverflow | 0.50 | MOGB partition + s2c boundary | 79.25 | 63.34 | 50.39 | 1.86 |
| stackoverflow | 0.50 | s2c partition + MOGB boundary | 73.35 | 63.59 | 54.49 | 15.69 |
| stackoverflow | 0.50 | Random partition | 75.88 | 79.80 | 87.64 | 30.98 |
| stackoverflow | 0.50 | Frozen single | 76.55 | 79.98 | 87.15 | 29.71 |
| stackoverflow | 0.50 | Trainable K=1 | 87.67 | 86.55 | 83.89 | 9.34 |
| stackoverflow | 0.75 | Frozen fixed K=2 | 43.63 | 74.47 | 88.02 | 61.67 |
| stackoverflow | 0.75 | MOGB partition | 45.59 | 33.09 | 21.17 | 0.67 |
| stackoverflow | 0.75 | MOGB partition + s2c boundary | 51.57 | 49.58 | 39.07 | 1.80 |
| stackoverflow | 0.75 | s2c partition + MOGB boundary | 48.48 | 61.22 | 53.50 | 23.37 |
| stackoverflow | 0.75 | Random partition | 65.01 | 81.08 | 88.20 | 34.56 |
| stackoverflow | 0.75 | Frozen single | 65.61 | 81.24 | 87.62 | 32.83 |
| stackoverflow | 0.75 | Trainable K=1 | 76.30 | 87.28 | 84.12 | 8.80 |

## 5. 与 fulltex.tex 的历史数字

这些数字来自 `s2c/fulltex.tex` 的 `tab:main_results_all`，历史 `Ours` 使用冻结 MiniLM、固定 K_y=2、数据集相关 λ，并且正文说明 unknown/OOS validation 参与 λ 学习；当前 Trainable 是 Known-only、Gate-only、K=1。因此这里只做参考，不作公平排名。

| 数据集 | KIR | 当前 Trainable OOS F1 | fulltex Ours OOS F1 | 差值（仅描述） |
|---|---:|---:|---:|---:|
| banking77 | 0.25 | 91.65 | 93.99 | -2.34 pp |
| banking77 | 0.50 | 83.56 | 88.23 | -4.67 pp |
| banking77 | 0.75 | 68.65 | 86.49 | -17.84 pp |
| clinc150 | 0.25 | 95.16 | 95.01 | +0.15 pp |
| clinc150 | 0.50 | 90.44 | 91.96 | -1.52 pp |
| clinc150 | 0.75 | 82.85 | 87.10 | -4.25 pp |
| stackoverflow | 0.25 | 95.48 | 94.47 | +1.01 pp |
| stackoverflow | 0.50 | 87.67 | 89.71 | -2.04 pp |
| stackoverflow | 0.75 | 76.30 | 75.57 | +0.73 pp |

## 6. 当前可以确认的机制结论

1. Trainable K=1 的收益最稳定地来自单中心分数排序，而不是自动恢复多中心：在此前同一 checkpoint 的 K=2 控制中，StackOverflow OOS F1 下降约 19pp，false acceptance 上升约 34pp。
2. KIR 增大后 Banking77 和 StackOverflow 的 OOS F1 下降，说明 Known/OOS 几何重叠是主要瓶颈；这不是简单增加训练 epoch 能解决的。
3. Trainable 在同协议下通常优于 Frozen K=1，但 Banking77 高 KIR 退化，说明微调改变了类内方差和边界校准，收益具有数据集依赖性。
4. MOGB 组件的部分 OOS F1 提升常伴随 Known Recall/F1-All 下降，必须同时看拒识收益和已知类覆盖，不能只按 OOS F1 排名。
5. `fulltex.tex` 的高分不能直接作为当前 Trainable 的失败证据；先前历史方法和当前协议在 K、λ、调参监督、数据快照及系统层级上都不同。

## 7. 图表与数据文件

- `results/analysis/minilm_trainable_5seed_fair_v1/`：五 seed 逐单元、均值、配对差值和历史参照 CSV。
- `figures/minilm_trainable_5seed_fair_v1/trainable_vs_frozen_5seed_kir.png`：Trainable/Frozen KIR 曲线。
- `figures/minilm_trainable_5seed_fair_v1/all_methods_5seed_kir.png`：Trainable、固定 K、随机分区和 MOGB 组件曲线。
- `figures/minilm_trainable_5seed_fair_v1/all_methods_known_oos_tradeoff_5seed.png`：Known Recall–OOS F1 权衡。
- `figures/minilm_trainable_5seed_fair_v1/trainable_vs_fulltex_reference.png`：历史 fulltex 参照图。

## 8. 仍然不能声称什么

- 不能声称 Trainable 已达到 SOTA；ADB/DA-ADB/DCLOOS 仍不是同协议五 seed 主表。
- 不能声称 K=2 或 MOGB 组件是普遍更优；StackOverflow 的固定多中心退化仍然存在。
- 不能把历史 `fulltex.tex` 的 OOS validation 调参结果与当前严格 Known-only 结果直接做绝对排名。
