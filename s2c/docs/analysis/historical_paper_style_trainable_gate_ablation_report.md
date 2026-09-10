# 论文设置下的 Trainable Gate 扩展消融

论文原始消融包含 Ours、Without Gate、Cascade-MiniLM 和 Cascade-SmolLM。本报告在同样的 KIR、K=2、lambda、threshold=1 几何设定下，新增一个 Trainable-Gate 变体。

由于论文原始 Router/Expert 的部分历史 checkpoint 路径无法在当前工作区完整恢复，本变体使用当前 H1 对应 KIR 的固定 Router/Expert；因此它是 paper-style H1 extension，不是论文 H0 的严格复现。

| 数据集 | KIR | 变体 | Known F1 | OOS F1 | Accuracy |
|---|---:|---|---:|---:|---:|
| banking77_oos | 0.25 | Trainable Gate + paper geometry | 56.58 | 92.32 | 86.30 |
| banking77_oos | 0.50 | Trainable Gate + paper geometry | 61.62 | 86.40 | 78.63 |
| banking77_oos | 0.75 | Trainable Gate + paper geometry | 67.49 | 84.96 | 77.50 |
| clinc150 | 0.25 | Trainable Gate + paper geometry | 62.66 | 93.97 | 89.00 |
| clinc150 | 0.50 | Trainable Gate + paper geometry | 67.61 | 86.16 | 80.55 |
| clinc150 | 0.75 | Trainable Gate + paper geometry | 66.39 | 73.48 | 70.60 |
| stackoverflow | 0.25 | Trainable Gate + paper geometry | 60.33 | 80.24 | 71.97 |
| stackoverflow | 0.50 | Trainable Gate + paper geometry | 75.27 | 77.17 | 75.63 |
| stackoverflow | 0.75 | Trainable Gate + paper geometry | 82.06 | 72.89 | 79.80 |

## 论文历史四变体的对应表

以下数值直接来自历史 paper_results artifact，表中保留论文原始消融的 Acc/OOS F1 口径。

| 数据集 | KIR | Ours | Without Gate | Cascade-MiniLM | Cascade-SmolLM |
|---|---:|---|---|---|---|
| banking77_oos | 0.25 | 89.07 / 93.99 | 84.90 / 91.27 | 85.10 / 91.41 | 51.20 / 64.43 |
| banking77_oos | 0.50 | 78.98 / 88.23 | 77.15 / 82.88 | 77.11 / 84.63 | 65.83 / 77.61 |
| banking77_oos | 0.75 | 77.84 / 85.28 | 76.50 / 84.20 | 77.23 / 83.69 | 52.48 / 55.41 |
| clinc150 | 0.25 | 91.04 / 95.30 | 67.49 / 76.88 | 90.49 / 94.27 | 84.76 / 90.98 |
| clinc150 | 0.50 | 86.78 / 91.96 | 67.90 / 70.24 | 85.89 / 90.67 | 73.58 / 78.69 |
| clinc150 | 0.75 | 80.75 / 82.14 | 73.40 / 66.50 | 80.10 / 81.00 | 74.31 / 71.79 |
| stackoverflow | 0.25 | 85.83 / 91.70 | 81.80 / 88.30 | 79.73 / 86.85 | 40.28 / 45.00 |
| stackoverflow | 0.50 | 85.54 / 89.71 | 80.28 / 82.79 | 77.46 / 79.41 | 50.10 / 55.68 |
| stackoverflow | 0.75 | 81.32 / 75.57 | 77.36 / 65.83 | 75.91 / 62.50 | 58.91 / 28.72 |

## 关键限制

- 论文四变体是历史 seed42 artifact；Trainable-Gate extension 也是 seed42，但下游来源是当前 H1 fixed Router/Expert。
- Trainable-Gate extension 使用 Trainable embedding、每 intent K=2、对角 Mahalanobis、CLINC lambda=.5、其余 lambda=1、normalized boundary threshold=1。
- Known F1 使用全测试集 Known 类 macro F1；论文历史表主要展示 Acc/OOS F1，因此两类 Known F1 不能直接拿旧 artifact 的 legacy 字段替代。
- 没有修改 fulltex.tex，也没有使用测试集选择配置。

结果入口：

- [Trainable Gate 9 个单元](../../results/analysis/historical_paper_style_trainable_gate_ablation/per_cell.csv)
- [实验 manifest](../../results/analysis/historical_paper_style_trainable_gate_ablation/MANIFEST.json)
- [论文四变体历史报告](historical_paper_ablation_report.md)
