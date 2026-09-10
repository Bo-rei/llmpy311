# Trainable-K1 与 ADB 跨数据集逐样本错误预算 V1

更新时间：2026-08-10

本报告只读取已完成的 45 个 Trainable-K1 与 45 个 ADB artifact，覆盖三个数据集、三个 KIR 和五个 seed。每个单元先验证 ADB 测试快照与运行 manifest 的 SHA256，再验证文本顺序和 protocol view 完全一致，最后按同一测试样本统计五种互斥错误状态。

## 合同边界

Trainable-K1 是 Known-only MiniLM 表示适配；ADB 是 BERT/TextOIR 外部兼容实现。样本级对齐保证错误预算比较可信，但不能消除 backbone、训练目标和边界实现差异，因此本报告不是跨骨干 SOTA 排名。

## 关键结果

| 数据集 | KIR | Trainable Known 拒绝 | ADB Known 拒绝 | Trainable OOS 误接收 | ADB OOS 误接收 |
|---|---:|---:|---:|---:|---:|
| clinc150 | 0.25 | 26.37% | 9.02% | 3.24% | 13.58% |
| clinc150 | 0.50 | 25.56% | 9.16% | 3.69% | 14.35% |
| clinc150 | 0.75 | 24.88% | 8.46% | 4.02% | 17.49% |
| banking77 | 0.25 | 18.16% | 12.53% | 10.33% | 26.41% |
| banking77 | 0.50 | 17.79% | 10.86% | 15.74% | 33.67% |
| banking77 | 0.75 | 17.59% | 9.65% | 19.61% | 35.82% |
| stackoverflow | 0.25 | 15.71% | 16.48% | 3.85% | 8.03% |
| stackoverflow | 0.50 | 16.11% | 18.63% | 9.34% | 8.27% |
| stackoverflow | 0.75 | 15.88% | 19.16% | 8.80% | 8.12% |

## 解释

- Trainable 的优势并非跨所有工作点都表现为更低 OOS 误接收；其主要表现是减少部分 Known false rejection，同时在部分数据集/KIR 增加一定 OOS false acceptance。
- 需要同时阅读 OOS F1、F1-All、Known Recall 和错误预算；只看单个 OOS F1 会掩盖工作点交换。
- 数据集差异明显：StackOverflow 的 Trainable/ADB 差异主要是覆盖—拒识平衡；Banking77 和 CLINC150 的方向随 KIR 改变，不能用一个全局机制解释。
- 本分析不使用测试 OOS 调参，不改变任何 checkpoint、registry、阈值或历史 artifact；逐样本结果仅在本地分析脚本中读取，输出只保留聚合计数。

## 产物

- 结果：`results/analysis/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/`
- 图表：`figures/archive/analysis/trainable_vs_adb_cross_dataset_error_budget_v1/`
- 对齐单元：45/45；各单元使用其 protocol test count，共 221,700 条记录。

## 下一步

继续围绕已完成的 fair matrix 做表示几何、边界和错误预算可视化；若要宣称超过 ADB/MOGB/DCLOOS，仍需统一 backbone、训练监督和评估合同，不能由本报告单独推出。
