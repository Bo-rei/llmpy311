# 跨数据集可视化证据链 V1

## 证据范围

本报告是 analysis-only：读取已完成的 `experimental_mechanism_pack_v3` 五 seed fair-component 汇总、固定 K=1…5 的 3-seed light sweep 和 StackOverflow intent 诊断；不训练、不重新选择超参数、不覆盖历史 artifact。相同协议 fair matrix 与外部/兼容 baseline 分开。

## 当前同协议结论

- Trainable K=1 是当前三个数据集、三个 KIR 下最稳定的自有 Gate 候选；它的主要优势是 OOS 分数排序和覆盖—拒识平衡，而不是单纯把 Known 拒成 OOS。
- MOGB MiniLM / MOGB partition 组件通常降低 false acceptance，但同时牺牲大量 Known Recall/F1-All；因此不能仅按 OOS F1 或 FA 单指标排名。
- 固定 K=2 在 Banking77 某些 KIR 可能提高 OOS F1，但 Known Recall 与 F1-All 的代价必须同时报告；在 StackOverflow 上固定多中心的风险最明显。
- StackOverflow intent 诊断显示，K=2 的聚类 ARI 可以很高，但新增 OOS 误接收仍远多于恢复 Known，说明“聚类稳定”不是开放边界有效性的充分条件。

## Trainable 的跨数据集均值（跨 KIR，五 seed）

| 数据集 | OOS F1 | F1-All | Known Recall | False Acceptance |
|---|---:|---:|---:|---:|
| CLINC150 | 89.48 | 81.47 | 74.40 | 3.65 |
| Banking77 | 81.29 | 81.44 | 82.16 | 15.23 |
| StackOverflow | 86.48 | 87.16 | 84.10 | 7.33 |

## 图表解释

1. `performance_heatmap.png` 显示方法优势集中在哪个数据集/KIR，而不是把所有结果压成一个平均数。
2. `pareto_tradeoff.png` 用 F1-All/OOS F1 展示 Known 分类与 OOS 拒识的联合工作点；`relative_trainable_heatmap.png` 直接显示 Trainable 相对同协议组件的差值。
3. `kir_curves.png` 用 KIR 曲线检查收益是否随开放程度变化；曲线不能被解释成超出当前三 KIR 的外推。
4. `k_sweep_tradeoff.png` 显示增加固定中心数后 OOS F1 与 Known Recall 的不同方向，使用 3-seed light sweep，只作机制消融，不构成 adaptive-K 选择规则。
5. `intent_risk_scatter.png` 把 StackOverflow 的 intent-level 恢复 Known 与新增误接收 OOS 放在同一坐标；点在对角线之上表示边界扩张的风险大于覆盖收益。

## 与外部 baseline 的边界

ADB、DA-ADB、官方 BERT MOGB 和 DCLOOS（如有记录）保存在 `external_baselines_isolated.csv`，不与同协议 MiniLM fair matrix 合并。它们的表示、训练监督、数据/划分或评价合同不同，只能作为复现/兼容性参照，不能从本报告推导无条件 SOTA 排名。

## 当前最重要的机制判断

Trainable 的正结果来自表示适配后更好的 K=1 score separation；固定多中心的负结果来自 acceptance union 扩张。下一步若继续做实验，应优先补同一监督条件下的强 baseline 工作点与表示几何图，而不是盲目增加 K 或重新搜索 test-oracle 最优点。

## 可复现与限制

- 输入 SHA256 记录在 `MANIFEST.json`；没有输出原始文本、embedding、checkpoint 或逐样本预测。
- 所有 test OOS 使用仅限事后可视化/误差分析；没有用于选择 K、阈值、半径或 checkpoint。
- 本报告仍是 Gate-only/组件级分析，不是完整 Cascade 论文主表。
