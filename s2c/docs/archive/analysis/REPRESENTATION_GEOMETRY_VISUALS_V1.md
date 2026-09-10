# 表示几何与 Near-OOS 可视化证据 V1

## 范围

本阶段只读取已完成的 Frozen/CE/SupCon 表示几何、K=1/K=2 结果、Trainable/Frozen score 诊断和 StackOverflow raw transition。没有训练、调参、重建中心或修改历史 artifact。测试标签只用于事后统计和画图。

## 主要发现

1. 表示训练确实改善 K=1 的类内/类间几何和 score separation，但几何指标不能单独预测固定 K=2 是否安全。
2. StackOverflow 的 K=2 near-OOS 下降与 acceptance-union 风险一致；即使聚类稳定或类内对齐更强，也可能新增大量 OOS 误接收。
3. Trainable K=1 的优势更接近分数排序和覆盖—拒识平衡，而不是简单提高拒绝率。
4. Trainable 与 MOGB fair 的错误集合存在互补：MOGB 更保守，Trainable 保留更多 Known 覆盖；两者不能仅按单一 OOS F1 排名。

## 输入与输出

- 几何汇总：`results/analysis/archive/analysis/representation_geometry_visuals_v1/geometry_summary.csv`
- K=2 变化：`results/analysis/archive/analysis/representation_geometry_visuals_v1/k_effects.csv`
- score gap：`results/analysis/archive/analysis/representation_geometry_visuals_v1/score_gap_false_accept.csv`
- 错误四象限：`results/analysis/archive/analysis/representation_geometry_visuals_v1/stackoverflow_error_quadrants.csv`
- 图目录：`figures/archive/analysis/representation_geometry_visuals_v1/`

## 图表

1. `geometry_tradeoff.png`：effective rank、relative separation 与 K=2−K=1 OOS F1 的关系。
2. `near_oos_delta.png`：各数据集/表示的近邻 OOS 变化。
3. `score_gap_false_accept.png`：score separation 与 false acceptance 的关系。
4. `stackoverflow_error_quadrants.png`：Trainable K=1 与 MOGB fair 的逐样本错误重叠。

## 解释边界

这些图是机制证据，不是新的正式模型选择，也不是官方 MOGB 或 DCLOOS 的公平 SOTA 排名。near/medium/far 仍属于已有诊断合同；完整 Cascade 与外部端到端监督条件保持隔离。

## 几何均值（KIR=0.50）

| 数据集 | 表示 | Relative separation | Effective rank | Purity@10 | Same-intent alignment |
|---|---|---:|---:|---:|---:|
| Banking77 | CE | 0.914 | 27.31 | 0.902 | 0.836 |
| Banking77 | Frozen | 0.547 | 70.99 | 0.868 | 0.547 |
| Banking77 | SupCon | 0.777 | 40.15 | 0.890 | 0.742 |
| CLINC150 | CE | 0.969 | 63.07 | 0.968 | 0.939 |
| CLINC150 | Frozen | 0.661 | 149.77 | 0.903 | 0.522 |
| CLINC150 | SupCon | 0.832 | 86.56 | 0.960 | 0.711 |
| StackOverflow | CE | 0.948 | 12.34 | 0.928 | 0.897 |
| StackOverflow | Frozen | 0.425 | 148.01 | 0.831 | 0.238 |
| StackOverflow | SupCon | 0.909 | 15.55 | 0.924 | 0.897 |
