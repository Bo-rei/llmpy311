# MiniLM 表示—边界诊断 V1

> 本报告只读取已经完成的 Frozen E2 K=1 与 Trainable K=1 预测和 metrics，做 test-score 机制诊断；不重新训练、不选择阈值、不选择 checkpoint，也不把诊断结果当作调参依据。

## 1. 范围与数据来源

- 三个数据集、KIR={0.25,0.50,0.75}、seeds={13,42,87,100,123}、K=1、diagonal Mahalanobis、mean+std、threshold=1。
- Trainable 读取 `minilm_trainable_kir_sweep_v1` 与其五 seed 扩展；Frozen 读取相同 E2 单元。
- 逐样本 score 只在内存中用于分组统计，导出的 CSV 不包含文本；test OOS 仅用于事后解释。

## 2. 诊断指标

- `score_gap_median_oos_minus_known`：OOS 与 Known 的中位 normalized score 差；越大通常表示排序分离更强。
- `radius_gap_oos_minus_known`：被最近中心选中的半径差；只描述半径与样本分配关系，不表示训练时使用了 OOS。
- false acceptance/rejection、OOS F1 和 Known Recall 直接来自各 run 的正式 metrics。

## 3. 当前可见的机制解释

1. Trainable K=1 的收益首先体现在 score 排序：它改变了 Known/OOS score 分布，而不是恢复多中心边界。
2. 当前训练目标只优化 Known 分类、类内紧致和类间 margin；它没有直接约束 OOS 误接受，因此不能保证 K>1 的并集安全。
3. checkpoint 只按 Known calibration 的 `F1-K + 0.05×Known Recall` 选择，历史 fulltex 的 λ 则使用了不同的 OOS/unknown validation 合同。
4. KIR 增大后，Known intent 覆盖减少而 OOS 组成变难；表示适配收益因数据集而异，不能用一个全局 epoch 或阈值解释。

## 4. KIR=0.50 的数值证据

- clinc150：median score gap（OOS−Known）由 Frozen 0.258 变为 Trainable 0.438；OOS median score 1.163→1.317；false acceptance 6.57%→3.69%。
- banking77：median score gap（OOS−Known）由 Frozen 0.259 变为 Trainable 0.334；OOS median score 1.086→1.160；false acceptance 24.88%→15.74%。
- stackoverflow：median score gap（OOS−Known）由 Frozen 0.115 变为 Trainable 0.434；OOS median score 1.024→1.194；false acceptance 25.03%→9.34%。

## 4. 证据文件

- `results/analysis/minilm_boundary_diagnostics_v1/run_summary.csv`：45×2 个逐 seed 诊断摘要。
- `results/analysis/minilm_boundary_diagnostics_v1/summary_mean_std.csv`：dataset×KIR×representation 汇总。
- `results/analysis/minilm_boundary_diagnostics_v1/score_quantiles.csv`：Known/OOS score 分位数。
- `figures/minilm_boundary_diagnostics_v1/`：score 分布、几何差值和 OOS F1—false acceptance 图。

## 5. 当前结论边界

- 这些图解释为什么 Trainable 能改善当前 K=1，但不能证明它已经达到 fulltex 历史结果或 SOTA。
- 这些图也不能把 K=1 的表示收益外推为固定 K=2/多中心收益；StackOverflow 的多球 false acceptance 仍需单独处理。
- 下一步应优先做 calibration coverage、半径稳定性和历史 fulltex/当前协议的逐组件桥接，而不是继续盲目扩展 K。
