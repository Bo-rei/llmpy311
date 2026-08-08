# MiniLM Trainable 训练动态诊断 V1

> 只读取已完成 Trainable K=1 的 Known-only training history 和最终 metrics；不重新训练、不改变 checkpoint、不使用测试 OOS 选择任何参数。

## 1. 结果

- 读取 45 个训练 run、179 条 epoch 记录；三数据集、KIR={.25,.50,.75}、五 seed。
- 可训练参数量：3,746,944；epoch 记录均来自 Known calibration 选择。
- `selection_score = calibration F1-K + 0.05 × Known Recall`，不是 OOS F1；因此它只能保证 Known-only 目标，不保证历史 fulltex 的 OOS 目标。

## 2. 机制解释

1. 训练动态图展示的是 Known-only 选择目标；测试 OOS 图仅用于事后检查二者是否错位。
2. 当 KIR 增大时，Known intent universe 变小、OOS 组成变难，单一 Known-only 选择目标不能自动适配所有开放程度。
3. 这解释了为什么 Trainable 在当前 K=1 Gate 上通常优于 Frozen，但仍可能低于 fulltex：历史 fulltex 使用了不同的 λ/unknown validation、固定 K=2 和完整 Cascade。
4. 即使 Known-only 选择曲线稳定，也不能说明固定 K>1 的 union boundary 安全；该问题已经由 StackOverflow K=2 配对实验单独证明。

## 3. Known coverage transfer

- 选模后的 calibration Known Recall 与 test Known Recall 的平均差异并不大；例如 KIR=.50 为 CLINC150 `-0.64pp`、Banking77 `-0.21pp`、StackOverflow `+0.33pp`。
- 因此当前高 KIR 的 OOS F1 下降不能简单解释为 Known 覆盖崩溃，更可能来自 OOS score 分布和边界校准错位。

## 4. 文件

- `results/analysis/minilm_training_dynamics_v1/run_summary.csv`
- `results/analysis/minilm_training_dynamics_v1/history.csv`
- `figures/minilm_training_dynamics_v1/`（含 calibration/test Known Recall、selection dynamics 和 OOS 对齐图）

## 5. 结论边界

- 本阶段是训练选择机制诊断，不是新方法结果，也不是 SOTA 排名。
- 后续如要提高历史协议可比性，应先做同一表示、同一 K、同一阈值监督条件的 bridge baseline，而不是盲目增加训练 epoch。
