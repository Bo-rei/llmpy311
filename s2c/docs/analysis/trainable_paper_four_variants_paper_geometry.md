# Trainable Gate 完整系统：论文四变体消融

新评估完成 36/36 个单元。三个数据集 × KIR=.25/.50/.75 × 四个变体，seed=42。主表使用论文布局：每个数据集两列 Acc、OOS F1；下表所有已填数字均来自本轮运行。

| KIR | Variant | CLINC Acc | CLINC OOS F1 | StackOverflow Acc | StackOverflow OOS F1 | BANKING77-OOS Acc | BANKING77-OOS OOS F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.25 | Ours | 89.49 | 94.21 | 92.15 | 95.97 | 88.36 | 93.59 |
| 0.25 | Without Gate | 73.00 | 81.75 | 66.84 | 74.95 | 82.72 | 89.81 |
| 0.25 | Cascade-MiniLM | 90.36 | 94.21 | 92.72 | 95.97 | 88.50 | 93.59 |
| 0.25 | Cascade-SmolLM | 80.85 | 88.09 | 40.33 | 45.10 | 51.15 | 64.41 |
| 0.50 | Ours | 81.69 | 87.01 | 88.43 | 91.51 | 80.20 | 87.60 |
| 0.50 | Without Gate | 73.00 | 76.03 | 78.36 | 80.16 | 82.43 | 88.75 |
| 0.50 | Cascade-MiniLM | 82.51 | 87.01 | 89.40 | 91.51 | 80.96 | 87.60 |
| 0.50 | Cascade-SmolLM | 67.65 | 69.71 | 44.41 | 6.38 | 47.89 | 55.38 |
| 0.75 | Ours | 73.36 | 75.54 | 83.07 | 78.49 | 79.34 | 86.60 |
| 0.75 | Without Gate | 71.65 | 61.76 | 77.40 | 65.94 | 76.00 | 82.32 |
| 0.75 | Cascade-MiniLM | 74.58 | 75.54 | 84.79 | 78.49 | 80.69 | 86.60 |
| 0.75 | Cascade-SmolLM | 68.44 | 55.02 | 64.86 | 0.13 | 45.88 | 42.07 |

## 四行实际含义

- Ours：Trainable MiniLM Gate + 固定 SmolLM Router/Expert，使用论文几何 K=2、CLINC lambda=.5、其余 lambda=1、threshold=1、normalized boundary。
- Without Gate：移除几何 Gate，所有样本进入原 Router/Expert；用下游 Expert 意图置信度拒识。这遵循原论文执行代码的 intent_confidence 分支；论文文字的 Router rejection 在单域数据中没有可用置信信息。
- Cascade-MiniLM：保留当前 Trainable MiniLM Gate，复用论文 MiniLM cascade 的域分类头和域内意图分类头（Known train 上 logistic regression），替换 SmolLM 下游。表示取当前 Trainable MiniLM 的输出；三阶段均为 MiniLM。
- Cascade-SmolLM：Gate改为 SmolLM mean-pooled/L2-normalized 意图原型余弦分数，复用论文原型 Gate 实现；下游仍是相同 SmolLM Router/Expert。CLINC使用router backbone；单域常量router没有模型参数，使用预训练SmolLM backbone。

Without Gate和Cascade-SmolLM的拒识分数阈值在验证集上按完整macro F1选择，搜索分数阈值为0.20至0.95，步长0.05。所有阈值与验证集分数单独保存。Ours/Cascade-MiniLM共用已确定的Gate及边界，所以OOS F1相同是设计预期，Known分类和Acc可以不同。

本表属于当前H1同数据、同下游条件下的结构/模型替换实验。旧论文四行历史数字未参与本表填充或参数选择。

## 补充完整指标

| Dataset | KIR | Variant | Known F1 | OOS F1 | Acc | Known Recall | False Acceptance |
|---|---:|---|---:|---:|---:|---:|---:|
| banking77_oos | 0.25 | Ours | 60.77 | 93.59 | 88.36 | 81.88 | 9.92 |
| banking77_oos | 0.25 | Without Gate | 51.59 | 89.81 | 82.72 | 70.83 | 15.33 |
| banking77_oos | 0.25 | Cascade-MiniLM | 62.94 | 93.59 | 88.50 | 81.88 | 9.92 |
| banking77_oos | 0.25 | Cascade-SmolLM | 26.83 | 64.41 | 51.15 | 75.42 | 50.94 |
| banking77_oos | 0.50 | Ours | 63.53 | 87.60 | 80.20 | 84.00 | 18.02 |
| banking77_oos | 0.50 | Without Gate | 65.24 | 88.75 | 82.43 | 72.80 | 13.18 |
| banking77_oos | 0.50 | Cascade-MiniLM | 67.23 | 87.60 | 80.96 | 84.00 | 18.02 |
| banking77_oos | 0.50 | Cascade-SmolLM | 40.44 | 55.38 | 47.89 | 83.40 | 59.64 |
| banking77_oos | 0.75 | Ours | 69.31 | 86.60 | 79.34 | 86.38 | 17.46 |
| banking77_oos | 0.75 | Without Gate | 67.47 | 82.32 | 76.00 | 79.47 | 21.52 |
| banking77_oos | 0.75 | Cascade-MiniLM | 73.23 | 86.60 | 80.69 | 86.38 | 17.46 |
| banking77_oos | 0.75 | Cascade-SmolLM | 51.31 | 42.07 | 45.88 | 91.84 | 72.07 |
| clinc150 | 0.25 | Ours | 65.71 | 94.21 | 89.49 | 58.25 | 1.22 |
| clinc150 | 0.25 | Without Gate | 53.27 | 81.75 | 73.00 | 67.81 | 25.05 |
| clinc150 | 0.25 | Cascade-MiniLM | 70.84 | 94.21 | 90.36 | 58.25 | 1.22 |
| clinc150 | 0.25 | Cascade-SmolLM | 61.19 | 88.09 | 80.85 | 76.23 | 16.40 |
| clinc150 | 0.50 | Ours | 69.93 | 87.01 | 81.69 | 59.87 | 1.60 |
| clinc150 | 0.50 | Without Gate | 70.95 | 76.03 | 73.00 | 80.67 | 30.46 |
| clinc150 | 0.50 | Cascade-MiniLM | 72.41 | 87.01 | 82.51 | 59.87 | 1.60 |
| clinc150 | 0.50 | Cascade-SmolLM | 67.40 | 69.71 | 67.65 | 85.02 | 40.95 |
| clinc150 | 0.75 | Ours | 70.26 | 75.54 | 73.36 | 60.27 | 1.45 |
| clinc150 | 0.75 | Without Gate | 77.49 | 61.76 | 71.65 | 89.08 | 47.66 |
| clinc150 | 0.75 | Cascade-MiniLM | 72.77 | 75.54 | 74.58 | 60.27 | 1.45 |
| clinc150 | 0.75 | Cascade-SmolLM | 75.28 | 55.02 | 68.44 | 94.29 | 58.64 |
| stackoverflow | 0.25 | Ours | 80.08 | 95.97 | 92.15 | 83.37 | 2.63 |
| stackoverflow | 0.25 | Without Gate | 60.79 | 74.95 | 66.84 | 82.03 | 36.48 |
| stackoverflow | 0.25 | Cascade-MiniLM | 82.38 | 95.97 | 92.72 | 83.37 | 2.63 |
| stackoverflow | 0.25 | Cascade-SmolLM | 42.02 | 45.10 | 40.33 | 75.95 | 68.55 |
| stackoverflow | 0.50 | Ours | 84.86 | 91.51 | 88.43 | 84.03 | 2.20 |
| stackoverflow | 0.50 | Without Gate | 76.91 | 80.16 | 78.36 | 78.72 | 18.89 |
| stackoverflow | 0.50 | Cascade-MiniLM | 86.98 | 91.51 | 89.40 | 84.03 | 2.20 |
| stackoverflow | 0.50 | Cascade-SmolLM | 60.66 | 6.38 | 44.41 | 97.66 | 96.63 |
| stackoverflow | 0.75 | Ours | 85.07 | 78.49 | 83.07 | 83.05 | 2.60 |
| stackoverflow | 0.75 | Without Gate | 82.09 | 65.94 | 77.40 | 81.23 | 23.15 |
| stackoverflow | 0.75 | Cascade-MiniLM | 87.57 | 78.49 | 84.79 | 83.05 | 2.60 |
| stackoverflow | 0.75 | Cascade-SmolLM | 74.97 | 0.13 | 64.86 | 99.89 | 99.93 |

Known F1直接在全测试集上对Known类求macro F1，保留真实OOS引起的Known false positives。全部四行同一数据/KIR/seed；这是单seed结果，不是三个seed平均。

[源表](../../results/analysis/trainable_paper_four_variants_paper_geometry/summary.csv)
