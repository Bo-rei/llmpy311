# H1 Trainable MiniLM：为什么低于论文 Ours，以及第二轮实验结论

## 结论先行

- 当前 CLINC150 低于论文 Ours，核心原因不是 Router/Expert 把 OOS F1 拉低：Gate OOS F1 与真实 full pipeline 逐 seed 完全一致；差距位于 Trainable 表示的 OOS score ranking / acceptance geometry，以及 H1 与论文 H0 的协议差异。
- 逐 seed validation 选参是有效改进：StackOverflow 真实 full pipeline 为 **89.91±1.54（+0.20 pp）**，Banking77-OOS 为 **91.65±0.14（+3.42 pp）**，三个 seed 都执行了直接 pipeline。​
- CLINC150 当前仍为 **90.99±0.68（−0.97 pp）**；扩大 K/λ/threshold 后，Gate-only 测试候选最高仍约 91.04%，说明仅继续扩大标量边界搜索不能解决剩余差距。

## 1. 为什么当前结果低于 Ours

1. **比较对象不是同一系统合同。** 论文 `fulltex.tex` 的 Ours 是历史 H0 完整 Cascade：多中心 Gate、历史 Router/Expert 和对应语义/数据链；当前实验是 H1 controlled 数据快照、已有 Trainable MiniLM checkpoint、固定 H1 Router/Expert。论文值应作为 historical reported reference，而不是严格 H0 replay。
2. **OOS F1 差距发生在 Gate。** 在当前 full pipeline 中，Gate 已拒绝的 OOS 不会进入 Router/Expert；逐 seed 的 Gate OOS F1 与 direct pipeline OOS F1 完全一致。因此 CLINC/SO 的剩余差距来自表示和 Gate score ordering，不是下游分类器吞掉了 OOS 结果。
3. **统一参数掩盖了 seed-specific operating point。** 同一数据集三个 seed 的 OOS score 分布不同；aggregate-selected 配置对 StackOverflow 为 89.15，而 per-seed validation selection 提升到 89.91。
4. **局部半径不是唯一瓶颈。** per-sphere Known-validation calibration 在三个数据集最终都没有被 validation 选中；扩大的 K=1..5、λ=.25..4、threshold=.70..1.75 网格也没有让 CLINC/SO 超过论文均值。

## 2. 实验结果

| 实验线 | CLINC150 | StackOverflow | Banking77-OOS | 证据层级 |
|---|---:|---:|---:|---|
| aggregate validation selected | 91.04 (-0.92 pp) | 89.15 (-0.56 pp) | 91.53 (+3.30 pp) | direct pipeline |
| per-seed validation selected | 90.99 (-0.97 pp) | 89.91 (+0.20 pp) | 91.65 (+3.42 pp) | direct pipeline |
| fine threshold Gate-only | 91.07 (-0.89 pp) | 89.86 (+0.15 pp) | 91.64 (+3.41 pp) | Gate-only confirmation |
| extended boundary Gate-only | 91.04 (-0.92 pp) | 89.25 (-0.46 pp) | 91.76 (+3.53 pp) | Gate-only confirmation |
| per-sphere calibration selected | 90.92 (-1.04 pp) | 89.91 (+0.20 pp) | 91.65 (+3.42 pp) | direct pipeline |

论文参考 OOS F1：CLINC150 91.96、StackOverflow 89.71、Banking77-OOS 88.23。括号为相对论文的百分点差值。

## 3. 保留的有效结果

- StackOverflow：逐 seed validation 选中的真实 full pipeline 结果超过论文 **+0.20 pp**；seed42 单元达到 **91.85**，其余 seed 的波动说明应报告均值和方差，不能只报告单个 seed。
- Banking77-OOS：逐 seed validation 选中的真实 full pipeline 结果超过论文 **+3.42 pp**，三个 seed 均超过。
- CLINC150：当前所有已完成边界搜索仍低于论文均值；这条负结果说明下一步应改变表示训练/目标或恢复 H0 链，而不是继续堆 global threshold。

## 4. 没有保留为正结果的实验

per-sphere calibration 使用 Known validation 为每个局部球估计 score 分位数，再用 OOS validation 选择 global multiplier；三个数据集最终 selected configuration 都退回 `calibration_mode=none`。这说明该校准没有在当前 Trainable 表示上提供可验证收益。

早期 OOS-validation-assisted pilot 曾在 CUDA 检查阶段停止。2026-09-06 后续独立 checkpoint 实验已实际完成 CUDA last2_long/seed13 训练，随后因 projection-only 工厂入口失败中止；不能再把早期 GPU 不可用描述作为当前状态。当前以 `docs/CURRENT_STATUS.md` 和 `results/analysis/historical_trainable_checkpoint_selection/` 为准，部分 checkpoint 不构成 test-confirmed 性能结论。

后续 [自适应中心实验](historical_trainable_adaptive_centers_presentation.md)已完成九个真实 CUDA pipeline 单元，overall OOS F1 为 CLINC `90.99±0.78`、SO `89.96±1.50`、BANKING77-OOS `91.89±0.19`；超过论文的 seed 数分别为 `0/2/3`（各三个）。该报告同时补充 Gate/pipeline macro F1 和 Accuracy 的损失对账，不能只凭 OOS 一致性说全链路无损。

2026-09-06 收口更新：[checkpoint follow-up](historical_trainable_checkpoint_selection_presentation.md) 已修复恢复入口并完成九个 CUDA 训练单元、三个锁定配置的真实 pipeline。CLINC 为 `91.64±.27%`，仅 seed87 `92.01%` 超过论文；test 固定 score oracle 均值 `91.78%`，因此不继续扩阈值。上面的失败描述保留为运行历史，不是当前阻断状态。

## 5. 结果入口

- [逐 seed 真实 full pipeline](../../results/analysis/historical_trainable_per_seed_full_pipeline/summary.csv)
- [细 threshold Gate 搜索](../../results/analysis/historical_trainable_fine_threshold_search/selected_test_summary.csv)
- [扩展 K/λ/threshold 搜索](../../results/analysis/historical_trainable_extended_boundary_search/best_test_candidate_by_dataset.csv)
- [per-sphere calibration](../../results/analysis/historical_trainable_per_sphere_calibration_search/selected_test_summary.csv)
- [原始参数化 full pipeline 汇报](historical_trainable_parameter_full_pipeline_presentation.md)

所有搜索均保持 validation-only selection，test 只用于 confirmation；没有修改 `fulltex.tex` 或覆盖历史 artifact。
